"""msp-agent komutları. Örnekler:

    msp-agent serve                                  # HTTP servisi
    msp-agent generate "Kafe sitesi..." -o site.html # tek seferlik deneme (etkin model)
    msp-agent distill briefs.jsonl --model qwen2.5-coder:14b   # güçlü modelle eğitim verisi üret
    msp-agent freeze --min-rating 4                  # ham veriden d1, d2... dondur
    msp-agent register --backend llamacpp --model-ref models/base.gguf --adapter-path data/adapters/run1/lora.gguf \
                       --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct --dataset d1
    msp-agent eval v2 && msp-agent activate v2       # ölç, iyiyse etkinleştir
    msp-agent models                                 # sürümler ve metrikleri
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agent.config import get_settings
from agent.fsutil import read_text, write_text


def _utf8_console():
    """Windows konsolu (cp1252/cp857) Türkçe karakterlerde UnicodeEncodeError vermesin."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # yönlendirilmiş / değiştirilmiş akış
            pass


def main(argv=None):
    _utf8_console()
    ap = argparse.ArgumentParser(prog="msp-agent")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)

    p = sub.add_parser("generate")
    p.add_argument("brief")
    p.add_argument("-o", "--out", default="site.html")
    p.add_argument("--version", help="Etkin sürüm yerine bu sürümü kullan")

    p = sub.add_parser("distill", help="Açıklamaları (güçlü) bir modelle çalıştırıp ham eğitim verisine ekler")
    p.add_argument("briefs", help='JSONL: {"brief": "..."} satırları')
    p.add_argument("--model", required=True, help="OpenAI uyumlu sunucudaki model adı")
    p.add_argument("--base-url")
    p.add_argument("--rating", type=int, default=4, help="Bu örneklere verilecek puan (gözden geçirdikten sonra)")

    p = sub.add_parser("freeze")
    p.add_argument("--min-rating", type=int, default=4)
    p.add_argument("--eval-ratio", type=float, default=0.1)

    p = sub.add_parser("register")
    p.add_argument("--backend", required=True, choices=["llamacpp", "openai", "mock"])
    p.add_argument("--model-ref", default="")
    p.add_argument("--adapter-path")
    p.add_argument("--base-model", default="")
    p.add_argument("--dataset")
    p.add_argument("--prompt-version", default=None)
    p.add_argument("--parent")
    p.add_argument("--notes", default="")

    p = sub.add_parser("eval")
    p.add_argument("version", nargs="?", help="Boşsa varsayılan model")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("activate")
    p.add_argument("version")

    sub.add_parser("models")
    sub.add_parser("datasets")

    a = ap.parse_args(argv)
    s = get_settings()
    from agent.registry import Registry
    reg = Registry(s.registry_dir)

    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("agent.service:app", host=a.host, port=a.port, workers=1)  # model bellekte tek kopya

    elif a.cmd == "generate":
        from agent.llm import build_llm
        from agent.pipeline import DesignPipeline
        mv = reg.get(a.version) if a.version else reg.active()
        pipe = DesignPipeline(build_llm(s, mv), s, mv.prompt_version if mv else s.prompt_version)
        r = pipe.run(a.brief, lambda stage, prog: print(f"[{prog:4.0%}] {stage}", file=sys.stderr))
        write_text(Path(a.out), r.html)
        print(json.dumps(r.metrics, ensure_ascii=False))

    elif a.cmd == "distill":
        from agent.datasets import Datasets
        from agent.llm.openai_compat import OpenAICompatLLM
        from agent.pipeline import DesignPipeline
        llm = OpenAICompatLLM(a.base_url or s.openai_base_url, a.model, s.openai_api_key)
        pipe, ds = DesignPipeline(llm, s), Datasets(s.datasets_dir)
        for i, line in enumerate(read_text(Path(a.briefs)).splitlines()):
            if not line.strip():
                continue
            r = pipe.run(json.loads(line)["brief"])
            ds.add_examples([{**c, "rating": a.rating, "teacher": a.model} for c in r.trace if not c.get("invalid")],
                            source="distill")
            print(f"{i + 1}: {r.metrics}", file=sys.stderr)

    elif a.cmd == "freeze":
        from agent.datasets import Datasets
        print(json.dumps(Datasets(s.datasets_dir).freeze(min_rating=a.min_rating, eval_ratio=a.eval_ratio), indent=2))

    elif a.cmd == "register":
        mv = reg.register(backend=a.backend, model_ref=a.model_ref, adapter_path=a.adapter_path,
                          base_model=a.base_model, dataset_version=a.dataset, prompt_version=a.prompt_version or s.prompt_version,
                          parent=a.parent or (reg.active().version if reg.active() else None), notes=a.notes)
        print(json.dumps(asdict(mv), indent=2, ensure_ascii=False))

    elif a.cmd == "eval":
        from training.evaluate import evaluate
        mv = reg.get(a.version) if a.version else None
        if a.version and mv is None:
            sys.exit(f"Sürüm yok: {a.version}")
        metrics = evaluate(s, mv, a.limit)
        if mv:
            reg.update_metrics(mv.version, {"bench": metrics})
        print(json.dumps(metrics, indent=2))

    elif a.cmd == "activate":
        print(f"Etkin: {reg.activate(a.version).version}")

    elif a.cmd == "models":
        active = reg.active()
        for mv in reg.list():
            mark = "*" if active and mv.version == active.version else " "
            print(f"{mark} {mv.version:5} {mv.backend:9} {mv.base_model or mv.model_ref:40} "
                  f"data={mv.dataset_version} bench={mv.metrics.get('bench')}")

    elif a.cmd == "datasets":
        from agent.datasets import Datasets
        for m in Datasets(s.datasets_dir).list():
            print(m["version"], m["counts"], m["created_at"], m["sha256"][:12])


if __name__ == "__main__":
    main()
