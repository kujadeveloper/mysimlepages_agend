import json
import time

import pytest

from agent.config import Settings
from agent.datasets import Datasets
from agent.llm.mock import MockLLM
from agent.pipeline import DesignPipeline
from agent.registry import Registry
from agent.sanitize import extract_html, sanitize


def test_sanitize_removes_dangerous_markup():
    html, st = sanitize('<section id="a"><a href="javascript:x()" onclick="x()">hi</a>'
                        '<script>alert(1)</script><img src="x.png"><div><p>open</section>')
    assert "script" not in html and "onclick" not in html and "javascript" not in html
    assert '<img src="x.png">' in html and html.endswith("</section>")
    assert st["removed"] == 3 and st["unbalanced"] >= 1


def test_extract_html_from_fenced_reply():
    assert extract_html("Sure!\n```html\n<nav id=\"n\">x</nav>\n```\nDone") == '<nav id="n">x</nav>'


def test_pipeline_with_mock(tmp_path):
    r = DesignPipeline(MockLLM(), Settings(data_dir=tmp_path)).run("Kafe için tek sayfalık site")
    assert r.html.startswith("<!doctype html>") and 'id="hero"' in r.html and "<script>alert" not in r.html
    assert [c["task"] for c in r.trace] == ["plan", "section", "section", "section"]
    assert r.metrics["fallback_sections"] == 0


class BadPlanLLM(MockLLM):
    def complete(self, system, user, **kw):
        return "not json" if kw.get("json_mode") else super().complete(system, user, **kw)


def test_invalid_plan_falls_back(tmp_path):
    r = DesignPipeline(BadPlanLLM(), Settings(data_dir=tmp_path)).run("Blog sitesi lütfen")
    assert len(r.plan.sections) == 6 and all(c.get("invalid") for c in r.trace if c["task"] == "plan")


def test_registry_versions(tmp_path):
    reg = Registry(tmp_path)
    v1 = reg.register(backend="mock")
    v2 = reg.register(backend="mock", parent="v1")
    assert (v1.version, v2.version) == ("v1", "v2") and reg.active() is None
    reg.activate("v2")
    assert reg.active().version == "v2"
    reg.update_metrics("v2", {"bench": {"plan_ok": 1}})
    assert reg.get("v2").metrics["bench"]["plan_ok"] == 1
    with pytest.raises(ValueError):
        reg.get("../x")


def test_dataset_freeze_dedup_and_filter(tmp_path):
    ds = Datasets(tmp_path)
    ex = {"task": "section", "system": "s", "user": "u", "assistant": "a", "rating": 5}
    ds.add_examples([ex, ex, {**ex, "assistant": "b", "rating": 2}])
    m = ds.freeze(min_rating=4)
    assert m["version"] == "d1" and m["counts"] == {"eval": 0, "train": 1}
    assert ds.freeze()["version"] == "d2"


def test_service_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("MSP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MSP_API_KEY", "secret")
    monkeypatch.setenv("MSP_BACKEND", "mock")
    from agent.config import get_settings
    get_settings.cache_clear()
    import importlib
    import agent.service as service
    importlib.reload(service)
    from fastapi.testclient import TestClient

    with TestClient(service.app) as c:
        assert c.post("/v1/designs", json={"brief": "Kurumsal firma sitesi"}).status_code == 401
        h = {"X-API-Key": "secret"}
        job = c.post("/v1/designs", json={"brief": "Kurumsal firma sitesi"}, headers=h).json()
        for _ in range(100):
            d = c.get(f"/v1/designs/{job['id']}", headers=h).json()
            if d["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        assert d["status"] == "done", d
        assert "<!doctype html>" in d["html"]
        assert c.get(f"/v1/designs/{job['id']}/preview", headers=h).headers["content-security-policy"]
        fb = c.post(f"/v1/designs/{job['id']}/feedback", headers=h,
                    json={"rating": 5, "corrections": {"hero": '<section id="hero">Düzeltilmiş</section>'}}).json()
        assert fb["examples"] == 4
        rows = [json.loads(l) for l in (tmp_path / "datasets/raw/feedback.jsonl").read_text("utf-8").splitlines()]
        assert any(r.get("corrected") and "Düzeltilmiş" in r["assistant"] for r in rows)
        assert c.post("/v1/datasets/freeze", headers=h).status_code == 201
