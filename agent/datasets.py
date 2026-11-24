"""Eğitim verisi: geri bildirimlerden ve elle eklenen örneklerden sürümlü, değişmez veri setleri.

data/datasets/
    raw/feedback.jsonl         # servisin topladığı ham örnekler (beğenilen / düzeltilen çıktılar)
    raw/*.jsonl                # elle eklenen örnekler (aynı biçim)
    d1/train.jsonl, d1/eval.jsonl, d1/manifest.json   # dondurulmuş sürüm: içerik özeti (sha256) ile

Her satır bir sohbet örneği: {"task": "plan"|"section", "system": ..., "user": ..., "assistant": ..., "rating": 1..5}
Eğitim sadece dondurulmuş sürümden yapılır; böylece her model sürümü hangi veriyle eğitildiğini bilir.
"""
import hashlib
import json
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

_lock = threading.Lock()


class Datasets:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)

    def add_examples(self, examples: list[dict], source: str = "feedback"):
        path = self.raw / f"{source}.jsonl"
        with _lock, path.open("a", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def _raw_examples(self, min_rating: int):
        seen = set()
        for p in sorted(self.raw.glob("*.jsonl")):
            for line in p.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                ex = json.loads(line)
                if ex.get("rating", 5) < min_rating or not ex.get("assistant"):
                    continue
                key = hashlib.sha256(f'{ex["system"]}\0{ex["user"]}\0{ex["assistant"]}'.encode()).hexdigest()
                if key not in seen:  # yinelenen örnekleri at
                    seen.add(key)
                    yield ex

    def freeze(self, *, min_rating: int = 4, eval_ratio: float = 0.1, seed: int = 42) -> dict:
        """Ham veriden yeni bir değişmez sürüm (d1, d2, ...) üretir."""
        examples = list(self._raw_examples(min_rating))
        if not examples:
            raise ValueError("Dondurulacak örnek yok")
        random.Random(seed).shuffle(examples)
        n_eval = max(1, int(len(examples) * eval_ratio)) if len(examples) > 10 else 0
        splits = {"eval": examples[:n_eval], "train": examples[n_eval:]}
        with _lock:
            nums = [int(p.name[1:]) for p in self.root.glob("d*") if p.name[1:].isdigit()]
            name = f"d{max(nums, default=0) + 1}"
            out = self.root / name
            out.mkdir()
            digest = hashlib.sha256()
            for split, rows in splits.items():
                text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
                (out / f"{split}.jsonl").write_text(text, "utf-8")
                digest.update(text.encode())
            manifest = {"version": name, "sha256": digest.hexdigest(), "min_rating": min_rating, "seed": seed,
                        "counts": {k: len(v) for k, v in splits.items()},
                        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
        return manifest

    def list(self) -> list[dict]:
        return [json.loads(p.read_text("utf-8")) for p in sorted(self.root.glob("d*/manifest.json"))]

    def path(self, version: str) -> Path:
        p = self.root / version
        if not (p / "manifest.json").exists():
            raise KeyError(version)
        return p
