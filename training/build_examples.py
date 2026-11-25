"""Elle yazılan örnekleri (training/examples/) eğitim satırlarına çevirir -> data/datasets/raw/manual.jsonl

Siz yalnızca müşteri açıklamasını ve beklenen çıktıyı yazarsınız; system/user metinleri servisin kullandığı
şablonlarla (agent/pipeline.py) birebir üretilir. Böylece model eğitimde ne gördüyse serviste de onu görür.

Önerilen biçim, bir işletme = bir klasör:
  training/examples/berber/site.json   {"brief": "...", "plan": {plan JSON nesnesi}}
  training/examples/berber/navbar.html, hero.html, ...   (dosya adı = plandaki bölüm id'si)
  -> 1 plan örneği + her .html için 1 bölüm örneği. Planda olup .html'i olmayan bölüm atlanır.

Tekil dosyalar (training/examples/*.json) da desteklenir:
  plan:    {"task": "plan", "brief": "...", "output": {plan JSON nesnesi}}
  section: {"task": "section", "brief": "...", "plan_from": "01_plan.json" | "plan": {...},
            "section_id": "hero", "output": "<section id=\"hero\" ...>...</section>"}
  İsteğe bağlı: "rating" (varsayılan 5).

Çalıştırma:
    python -m training.build_examples              # doğrula + yaz
    python -m training.build_examples --check      # yalnızca doğrula
    python -m training.build_examples --preview    # her işletmenin sayfasını data/tmp/preview/ altına birleştir
"""
import argparse
import json
import sys
from pathlib import Path

from agent import prompts
from agent.config import get_settings
from agent.fsutil import read_text, write_text
from agent.pipeline import Plan, assemble, plan_system, plan_user, section_user
from agent.sanitize import sanitize

SRC = Path(__file__).parent / "examples"


class Builder:
    def __init__(self, prompt_version: str, max_sections: int):
        self.pv, self.max_sections = prompt_version, max_sections
        self.rows: list[dict] = []
        self.errors: list[str] = []
        self.pages: dict[str, str] = {}   # önizleme: ad -> birleştirilmiş HTML

    def plan_row(self, brief: str, out: dict, rating: int, source: str) -> Plan:
        plan = Plan.model_validate(out)
        if not 5 <= len(plan.sections) <= self.max_sections:
            raise ValueError(f"bölüm sayısı 5-{self.max_sections} olmalı ({len(plan.sections)})")
        if [s.id for s in plan.sections] != [s["id"] for s in out["sections"]]:
            raise ValueError("bölüm id'leri kebab-case ve benzersiz olmalı")
        given = {k: str(v).lower() for k, v in out.get("palette", {}).items()}
        fixed = {k: v.lower() for k, v in plan.palette.items()}
        if given != fixed:  # servis paleti düzeltirdi: modele okunaksız renk öğretmeyelim
            diff = ", ".join(f"{k}: {given.get(k)} -> {v}" for k, v in fixed.items() if given.get(k) != v)
            raise ValueError(f"palet kontrast kuralına uymuyor ({diff})")
        self.rows.append({"task": "plan", "system": plan_system(self.pv, self.max_sections),
                          "user": plan_user(brief), "assistant": json.dumps(out, ensure_ascii=False),
                          "rating": rating, "source": source})
        return plan

    def section_row(self, brief: str, plan: Plan, section_id: str, html: str, rating: int, source: str) -> str:
        sec = next((s for s in plan.sections if s.id == section_id), None)
        if sec is None:
            raise ValueError(f'planda "{section_id}" bölümü yok')
        html = html.strip()
        _, st = sanitize(html)
        if st["removed"] or st["unbalanced"]:
            raise ValueError(f"HTML temiz değil (silinecek öğe: {st['removed']}, kapanmamış etiket: {st['unbalanced']})")
        if f'id="{sec.id}"' not in html[:300]:
            raise ValueError(f'kök öğede id="{sec.id}" yok')
        self.rows.append({"task": "section", "system": prompts.load(self.pv, "section_system"),
                          "user": section_user(brief, plan, sec), "assistant": html,
                          "rating": rating, "source": source})
        return html

    def site(self, folder: Path):
        ex = json.loads(read_text(folder / "site.json"))
        brief, rating = ex["brief"].strip(), ex.get("rating", 5)
        plan = self.plan_row(brief, ex["plan"], rating, f"{folder.name}/site.json")
        known = {s.id for s in plan.sections}
        for f in folder.glob("*.html"):
            if f.stem not in known:
                self.errors.append(f"{folder.name}/{f.name}: planda bu id'de bölüm yok")
        parts = []
        for sec in plan.sections:
            f = folder / f"{sec.id}.html"
            if not f.exists():
                continue
            try:
                parts.append(self.section_row(brief, plan, sec.id, read_text(f), rating, f"{folder.name}/{f.name}"))
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{folder.name}/{f.name}: {e}")
        if parts:
            self.pages[folder.name] = assemble(plan, parts)

    def single(self, src: Path, f: Path):
        ex = json.loads(read_text(f))
        task, rating = ex["task"], ex.get("rating", 5)
        if task == "plan":
            self.plan_row(ex["brief"].strip(), ex["output"], rating, f.name)
        elif task == "section":
            ref = json.loads(read_text(src / ex["plan_from"])) if "plan_from" in ex else None
            plan = Plan.model_validate(ex.get("plan") or ref["output"])
            brief = (ex.get("brief") or ref["brief"]).strip()
            self.section_row(brief, plan, ex["section_id"], ex["output"], rating, f.name)
        else:
            raise ValueError(f"bilinmeyen task: {task}")

    def build(self, src: Path):
        for folder in sorted(p.parent for p in src.glob("*/site.json")):
            try:
                self.site(folder)
            except Exception as e:  # noqa: BLE001 - tüm hataları dosya adıyla raporla
                self.errors.append(f"{folder.name}/site.json: {e}")
        for f in sorted(src.glob("*.json")):
            try:
                self.single(src, f)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{f.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--check", action="store_true", help="Yalnızca doğrula, yazma")
    ap.add_argument("--preview", action="store_true", help="İşletme klasörlerinden tam sayfa önizlemesi üret")
    args = ap.parse_args()
    s = get_settings()
    b = Builder(s.prompt_version, s.max_sections)
    b.build(args.src)
    for e in b.errors:
        print("HATA", e, file=sys.stderr)
    counts = {t: sum(r["task"] == t for r in b.rows) for t in ("plan", "section")}
    print(f"{len(b.rows)} geçerli örnek {counts}, {len(b.errors)} hatalı (prompt {s.prompt_version})")
    if args.preview:
        out_dir = s.data_dir / "tmp" / "preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, html in b.pages.items():
            write_text(out_dir / f"{name}.html", html)
        print(f"Önizleme: {out_dir} ({len(b.pages)} sayfa)")
    if b.errors:
        sys.exit(1)
    if not args.check:
        out = s.datasets_dir / "raw" / "manual.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_text(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in b.rows))
        print(f"Yazıldı: {out}")


if __name__ == "__main__":
    main()
