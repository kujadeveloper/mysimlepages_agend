"""Bir model sürümünü sabit açıklama setinde çalıştırıp otomatik metrikleri kayıt defterine yazar.

Metrikler (yüksek iyi): plan_ok (model geçerli plan üretti mi), section_ok (bölüm yedeğe düşmeden üretildi mi), clean
(temizleyicinin bir şey silmesi gerekmedi mi), balanced (etiketler düzgün mü); düşük iyi: seconds.
Yeni sürüm ancak bir önceki sürümden kötü değilse etkinleştirilmeli (cli: eval -> activate).
"""
import json
import statistics
from pathlib import Path

from agent.config import Settings
from agent.llm import build_llm
from agent.pipeline import DesignPipeline
from agent.registry import ModelVersion

BENCH = Path(__file__).with_name("bench_briefs.jsonl")


def evaluate(settings: Settings, version: ModelVersion | None, limit: int | None = None) -> dict:
    briefs = [json.loads(l)["brief"] for l in BENCH.read_text("utf-8").splitlines() if l.strip()][:limit]
    pipe = DesignPipeline(build_llm(settings, version), settings, version.prompt_version if version else "v1")
    rows = []
    for brief in briefs:
        r = pipe.run(brief)
        plan_calls = [c for c in r.trace if c["task"] == "plan"]
        m = r.metrics
        rows.append({
            "plan_ok": float(not plan_calls[-1].get("invalid")),
            "section_ok": 1 - m["fallback_sections"] / max(1, m["sections"]),
            "clean": float(m["removed"] == 0),
            "balanced": float(m["unbalanced"] == 0),
            "seconds": m["seconds"],
        })
    return {k: round(statistics.mean(r[k] for r in rows), 3) for k in rows[0]} | {"n": len(rows)}
