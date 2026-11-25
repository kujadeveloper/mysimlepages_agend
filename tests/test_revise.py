import json
import re
import time

from agent.config import Settings
from agent.pipeline import DesignPipeline, Plan, Section, assemble, split_sections

PLAN = Plan(title="Kahve Durağı", palette={"primary": "#7c4a1e", "secondary": "#faf7f2", "accent": "#c8a27a",
                                           "text": "#2a211a"},
            sections=[Section(id=i, type=t, goal=g) for i, t, g in [
                ("navbar", "navbar", "Logo ve bağlantılar"), ("hero", "hero", "Taze kahve"),
                ("pricing", "pricing", "Kahve fiyatları"), ("contact", "contact", "İletişim formu"),
                ("footer", "footer", "Telif")]])
SECTIONS = {s.id: f'<section id="{s.id}" class="py-16"><p>eski {s.id}</p></section>' for s in PLAN.sections}


class ScriptedLLM:
    """Düzeltme planı yanıtı verilir; bölüm çağrıları hangi bölüm için yapıldıysa onu kaydeder."""
    name = "scripted"

    def __init__(self, revise_reply):
        self.revise_reply = revise_reply
        self.section_calls = []

    def complete(self, system, user, *, max_tokens, temperature, json_mode=False, stop=None):
        if json_mode:
            return self.revise_reply if isinstance(self.revise_reply, str) else json.dumps(self.revise_reply)
        sid = re.search(r'Write the section id="([^"]+)"', user).group(1)
        mode = "revise" if "Current HTML of this section" in user else "new"
        self.section_calls.append((mode, sid))
        return f'<section id="{sid}" class="py-16"><p>yeni {sid}</p></section>'


def _pipe(llm, tmp_path):
    return DesignPipeline(llm, Settings(data_dir=tmp_path, prompt_version="v5"))


def test_only_requested_section_is_rewritten(tmp_path):
    llm = ScriptedLLM({"plan": PLAN.model_dump(), "change": ["pricing"], "summary": "Fiyatlar güncellendi"})
    r = _pipe(llm, tmp_path).revise("Kafe", PLAN, SECTIONS, "Latte fiyatını 150 TL yap")
    assert llm.section_calls == [("revise", "pricing")]
    assert r.sections["pricing"].endswith("yeni pricing</p></section>")
    assert all(r.sections[i] == SECTIONS[i] for i in ("navbar", "hero", "contact", "footer"))
    assert r.metrics["revision"]["summary"] == "Fiyatlar güncellendi"


def test_color_change_applies_without_regenerating(tmp_path):
    new = PLAN.model_dump()
    new["palette"] = {**new["palette"], "primary": "#1e3a8a"}
    llm = ScriptedLLM({"plan": new, "change": [], "summary": "Ana renk lacivert yapıldı"})
    r = _pipe(llm, tmp_path).revise("Kafe", PLAN, SECTIONS, "Ana rengi lacivert yap")
    assert llm.section_calls == []
    assert "--primary: #1e3a8a" in r.html


def test_added_section_is_generated_and_menu_updated(tmp_path):
    new = PLAN.model_dump()
    new["sections"].insert(3, {"id": "faq", "type": "faq", "goal": "Sık sorulan sorular"})
    llm = ScriptedLLM({"plan": new, "change": [], "summary": "SSS eklendi"})
    r = _pipe(llm, tmp_path).revise("Kafe", PLAN, SECTIONS, "Bir SSS bölümü ekle")
    assert ("new", "faq") in llm.section_calls
    assert {("revise", "navbar"), ("revise", "footer")} <= set(llm.section_calls)
    assert list(r.sections) == ["navbar", "hero", "pricing", "faq", "contact", "footer"]
    assert r.metrics["revision"]["added"] == ["faq"]


def test_removed_section_is_dropped(tmp_path):
    new = PLAN.model_dump()
    new["sections"] = [s for s in new["sections"] if s["id"] != "pricing"]
    r = _pipe(ScriptedLLM({"plan": new, "change": []}), tmp_path).revise("Kafe", PLAN, SECTIONS, "Fiyatları kaldır")
    assert "pricing" not in r.sections and 'id="pricing"' not in r.html
    assert r.metrics["revision"]["removed"] == ["pricing"]


def test_invalid_plan_falls_back_to_mentioned_sections(tmp_path):
    llm = ScriptedLLM("bu json değil")
    r = _pipe(llm, tmp_path).revise("Kafe", PLAN, SECTIONS, "Fiyat listesine filtre kahve ekle")
    assert llm.section_calls == [("revise", "pricing")]
    assert r.plan.palette == PLAN.palette


def test_requested_new_section_is_guaranteed(tmp_path):
    # Model planı değiştirmeden döndürse bile "kariyer bölümü ekle" talimatı kariyer bölümünü ekler
    llm = ScriptedLLM({"plan": PLAN.model_dump(), "change": []})
    r = _pipe(llm, tmp_path).revise("Kafe", PLAN, SECTIONS, "Bir kariyer bölümü ekle, iş başvuru formu olsun")
    assert "careers" in r.sections and ("new", "careers") in llm.section_calls


def test_split_sections_recovers_old_designs():
    html = assemble(PLAN, list(SECTIONS.values()))
    assert split_sections(html, PLAN) == SECTIONS


def test_service_revise_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MSP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MSP_API_KEY", "secret")
    monkeypatch.setenv("MSP_BACKEND", "mock")
    from agent.config import get_settings
    get_settings.cache_clear()
    import importlib
    import agent.service as service
    importlib.reload(service)
    from fastapi.testclient import TestClient

    def wait(c, h, job_id):
        for _ in range(200):
            d = c.get(f"/v1/designs/{job_id}", headers=h).json()
            if d["status"] in ("done", "failed"):
                return d
            time.sleep(0.05)
        raise AssertionError("bitmedi")

    with TestClient(service.app) as c:
        h = {"X-API-Key": "secret"}
        first = c.post("/v1/designs", json={"brief": "Kurumsal firma sitesi", "user_ref": "7"}, headers=h).json()
        assert c.post(f"/v1/designs/{first['id']}/revise", json={"instruction": "Hero metnini değiştir"},
                      headers=h).status_code in (202, 409)  # henüz bitmediyse 409
        wait(c, h, first["id"])
        rev = c.post(f"/v1/designs/{first['id']}/revise", json={"instruction": "Hero metnini değiştir"}, headers=h)
        assert rev.status_code == 202 and rev.json()["parent_id"] == first["id"]
        d = wait(c, h, rev.json()["id"])
        assert d["status"] == "done", d
        assert d["parent_id"] == first["id"] and d["instruction"] == "Hero metnini değiştir"
        assert d["revision"] is not None and "<!doctype html>" in d["html"]
        listed = c.get("/v1/designs", params={"user_ref": "7"}, headers=h).json()["results"]
        assert {r["id"]: r["parent_id"] for r in listed}[d["id"]] == first["id"]
        assert c.post("/v1/designs/" + "0" * 32 + "/revise", json={"instruction": "x" * 5}, headers=h).status_code == 404
