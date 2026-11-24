"""Dosya tabanlı model kayıt defteri.

data/registry/
    v1/manifest.json      # hangi temel model, hangi adaptör, hangi veri seti/prompt sürümü, değerlendirme metrikleri
    v2/manifest.json
    active                # etkin sürümün adı (ör. "v2")

Dosya tabanlı olduğu için git/DVC/S3 ile yedeklenebilir; geri almak = `active` dosyasını eski sürüme çevirmek.
"""
import json
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_VERSION = re.compile(r"^v(\d+)$")
_lock = threading.Lock()


@dataclass
class ModelVersion:
    version: str
    backend: str                         # llamacpp | openai | mock
    model_ref: str = ""                  # GGUF yolu ya da sunucudaki model adı
    adapter_path: str | None = None      # LoRA adaptörü (GGUF), birleştirilmemişse
    base_model: str = ""                 # eğitimin temel aldığı HF modeli
    dataset_version: str | None = None
    prompt_version: str = "v1"           # model hangi prompt şablonlarıyla eğitildiyse onunla çalışmalı
    parent: str | None = None            # hangi sürümden türetildi
    metrics: dict = field(default_factory=dict)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


class Registry:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _manifest(self, version: str) -> Path:
        if not _VERSION.match(version):
            raise ValueError(f"Geçersiz sürüm adı: {version}")
        return self.root / version / "manifest.json"

    def list(self) -> list[ModelVersion]:
        out = []
        for p in self.root.glob("v*/manifest.json"):
            if _VERSION.match(p.parent.name):
                out.append(ModelVersion(**json.loads(p.read_text("utf-8"))))
        return sorted(out, key=lambda v: int(v.version[1:]))

    def get(self, version: str) -> ModelVersion | None:
        p = self._manifest(version)
        return ModelVersion(**json.loads(p.read_text("utf-8"))) if p.exists() else None

    def next_version(self) -> str:
        nums = [int(v.version[1:]) for v in self.list()]
        return f"v{max(nums, default=0) + 1}"

    def register(self, **fields) -> ModelVersion:
        with _lock:
            mv = ModelVersion(version=self.next_version(), **fields)
            p = self._manifest(mv.version)
            p.parent.mkdir(parents=True)
            _atomic_write(p, json.dumps(asdict(mv), ensure_ascii=False, indent=2))
            return mv

    def update_metrics(self, version: str, metrics: dict) -> ModelVersion:
        with _lock:
            mv = self.get(version)
            if mv is None:
                raise KeyError(version)
            mv.metrics.update(metrics)
            _atomic_write(self._manifest(version), json.dumps(asdict(mv), ensure_ascii=False, indent=2))
            return mv

    def active(self) -> ModelVersion | None:
        p = self.root / "active"
        return self.get(p.read_text().strip()) if p.exists() else None

    def activate(self, version: str) -> ModelVersion:
        mv = self.get(version)
        if mv is None:
            raise KeyError(version)
        _atomic_write(self.root / "active", version)
        return mv


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, "utf-8")
    os.replace(tmp, path)
