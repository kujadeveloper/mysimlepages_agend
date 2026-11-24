"""HTTP servisi. Çalıştırma: uvicorn agent.service:app --host 0.0.0.0 --port 8100

Akış: Django backend kullanıcı adına POST /v1/designs -> iş kimliği; GET /v1/designs/{id} ile durum/ilerleme;
bitince html. Kullanıcının puanı POST /v1/designs/{id}/feedback ile eğitim verisine dönüşür.
Servis yalnızca iç ağda çalışmalı; X-API-Key ile korunur.
"""
import hmac
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent import __version__
from agent.config import get_settings
from agent.datasets import Datasets
from agent.intent import classify
from agent.jobs import JobStore, WorkerPool
from agent.llm import build_llm
from agent.pipeline import DesignPipeline
from agent.registry import Registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("agent")

settings = get_settings()
registry = Registry(settings.registry_dir)
datasets = Datasets(settings.datasets_dir)
store = JobStore(settings.db_path)


class Engine:
    """Etkin model süreç başına bir kez yüklenir; sürüm değişince bir sonraki işte yeniden yüklenir."""

    def __init__(self):
        self._lock = threading.Lock()
        self._key = None
        self._pipeline: DesignPipeline | None = None

    def pipeline(self) -> tuple[DesignPipeline, str]:
        mv = registry.active()
        key = mv.version if mv else "default"
        with self._lock:
            if self._key != key:
                self._pipeline = None  # eski modeli bellekten bırak
                llm = build_llm(settings, mv)
                self._pipeline = DesignPipeline(llm, settings, mv.prompt_version if mv else settings.prompt_version)
                self._key = key
                log.info("Model yüklendi: %s (%s)", key, llm.name)
            return self._pipeline, key


engine = Engine()


def handle(job: dict):
    pipeline, version = engine.pipeline()
    result = pipeline.run(job["brief"], lambda stage, p: store.progress(job["id"], stage, p))
    store.finish(job["id"], html=result.html, plan=result.plan.model_dump(), metrics=result.metrics,
                 trace=result.trace, model_version=version)


workers = WorkerPool(store, handle, settings.workers)


@asynccontextmanager
async def lifespan(_app):
    workers.start()
    yield
    workers.stop()


app = FastAPI(title="MySimplePage AI Agent", version=__version__, lifespan=lifespan)


def require_key(x_api_key: str = Header(default="")):
    if settings.api_key and not hmac.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(401, "Geçersiz API anahtarı")


# ---- Tasarım üretimi ----

class DesignIn(BaseModel):
    brief: str = Field(min_length=10, max_length=3000)
    user_ref: str | None = Field(default=None, max_length=64)   # backend'deki kullanıcı kimliği (kişisel veri değil)


class FeedbackIn(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)
    # Kullanıcı/ekip bir bölümü düzelttiyse: {"hero": "<section ...>"} — en değerli eğitim verisi budur
    corrections: dict[str, str] = Field(default_factory=dict)


@app.get("/health")
def health():
    mv = registry.active()
    return {"status": "ok", "version": __version__, "model": mv.version if mv else "default"}


@app.post("/v1/designs", dependencies=[Depends(require_key)], status_code=202)
def create_design(body: DesignIn):
    job_id = store.create(body.brief, body.user_ref)
    workers.notify()
    return {"id": job_id, "status": "queued", "position": store.queue_position(job_id)}


class IntentIn(BaseModel):
    message: str = Field(min_length=1, max_length=3000)


@app.post("/v1/intent", dependencies=[Depends(require_key)])
def intent(body: IntentIn):
    """Mesaj tasarım isteği mi, domain satın alma mı? Model çağırmaz, anında döner."""
    return vars(classify(body.message))


@app.get("/v1/designs", dependencies=[Depends(require_key)])
def list_designs(user_ref: str, limit: int = 20, since: str | None = None):
    """Kullanıcının son tasarımları; `since` (ISO zaman) verilirse o andan beri oluşturulan sayı da döner (kota)."""
    out = {"results": store.list_for(user_ref, max(1, min(limit, 100)))}
    if since:
        out["count_since"] = store.count_since(user_ref, since)
    return out


def _job_or_404(job_id: str) -> dict:
    job = store.get(job_id) if len(job_id) == 32 and job_id.isalnum() else None
    if job is None:
        raise HTTPException(404, "Bulunamadı")
    return job


@app.get("/v1/designs/{job_id}", dependencies=[Depends(require_key)])
def get_design(job_id: str, include_html: bool = True):
    job = _job_or_404(job_id)
    out = {k: job[k] for k in ("id", "status", "user_ref", "brief", "stage", "progress", "plan", "metrics",
                               "model_version", "error", "rating", "created_at", "finished_at")}
    if job["status"] == "queued":
        out["position"] = store.queue_position(job_id)
    if include_html and job["status"] == "done":
        out["html"] = job["html"]
    return out


@app.get("/v1/designs/{job_id}/preview", dependencies=[Depends(require_key)], response_class=HTMLResponse)
def preview(job_id: str):
    job = _job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "Henüz hazır değil")
    # Önizleme ayrı kökenden (sandbox iframe) gösterilmeli; CSP yalnızca gerekli CDN'lere izin verir
    csp = ("default-src 'none'; img-src https: data:; style-src 'unsafe-inline' https://fonts.googleapis.com; "
           "font-src https://fonts.gstatic.com; script-src https://cdn.tailwindcss.com 'unsafe-inline'")
    return HTMLResponse(job["html"], headers={"Content-Security-Policy": csp, "X-Content-Type-Options": "nosniff"})


@app.post("/v1/designs/{job_id}/feedback", dependencies=[Depends(require_key)])
def feedback(job_id: str, body: FeedbackIn):
    job = _job_or_404(job_id)
    if job["status"] != "done":
        raise HTTPException(409, "Tamamlanmamış tasarım puanlanamaz")
    store.rate(job_id, body.rating, body.comment)
    added = 0
    if settings.collect_feedback and job["trace"]:
        examples = []
        for call in job["trace"]:
            if call.get("invalid"):
                continue
            ex = {"task": call["task"], "system": call["system"], "user": call["user"],
                  "assistant": call["assistant"], "rating": body.rating, "job": job_id,
                  "model_version": job["model_version"]}
            sid = _section_id(call)
            if sid and sid in body.corrections:
                ex.update(assistant=body.corrections[sid], rating=5, corrected=True)
            examples.append(ex)
        datasets.add_examples(examples)
        added = len(examples)
    return {"ok": True, "examples": added}


def _section_id(call: dict) -> str | None:
    if call["task"] != "section":
        return None
    marker = 'Write the section id="'
    i = call["user"].find(marker)
    return call["user"][i + len(marker):].split('"', 1)[0] if i != -1 else None


# ---- Model ve veri seti yönetimi (yalnızca ekip) ----

class ModelIn(BaseModel):
    backend: str = Field(pattern="^(llamacpp|openai|mock)$")
    model_ref: str = ""
    adapter_path: str | None = None
    base_model: str = ""
    dataset_version: str | None = None
    prompt_version: str = Field(default_factory=lambda: settings.prompt_version)
    notes: str = ""


@app.get("/v1/models", dependencies=[Depends(require_key)])
def list_models():
    active = registry.active()
    return {"active": active.version if active else None, "models": [vars(m) for m in registry.list()]}


@app.post("/v1/models", dependencies=[Depends(require_key)], status_code=201)
def register_model(body: ModelIn):
    return vars(registry.register(**body.model_dump()))


@app.post("/v1/models/{version}/activate", dependencies=[Depends(require_key)])
def activate_model(version: str):
    try:
        return vars(registry.activate(version))
    except (KeyError, ValueError):
        raise HTTPException(404, "Sürüm bulunamadı")


@app.get("/v1/datasets", dependencies=[Depends(require_key)])
def list_datasets():
    return datasets.list()


@app.post("/v1/datasets/freeze", dependencies=[Depends(require_key)], status_code=201)
def freeze_dataset(min_rating: int = 4):
    try:
        return datasets.freeze(min_rating=min_rating)
    except ValueError as e:
        raise HTTPException(400, str(e))
