"""Tasarım hattı: açıklama -> plan (JSON) -> bölüm bölüm HTML -> temizlenmiş tek sayfa.

Neden bölüm bölüm? Küçük modeller uzun çıktıda tutarlılığı kaybeder ve tek seferde tüm sayfa büyük bağlam (KV bellek)
ister. Her bölüm kısa ve bağımsız üretildiği için 4K bağlamlı 1.5-3B model düşük VRAM'de yeterli olur; bir bölüm bozulursa
yalnızca o yeniden denenir. Her LLM çağrısı `trace` olarak saklanır: geri bildirimle doğrudan eğitim örneğine dönüşür.
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from html import escape
from typing import Callable

from pydantic import BaseModel, Field, ValidationError, field_validator

from agent import prompts
from agent.colors import fix_palette
from agent.config import Settings
from agent.llm import LLM
from agent.sanitize import extract_html, sanitize

log = logging.getLogger(__name__)

_HEX = re.compile(r"^#[0-9a-fA-F]{3,8}$")
_ID = re.compile(r"[^a-z0-9-]+")


class Section(BaseModel):
    id: str
    type: str = "custom"
    goal: str = ""

    @field_validator("id")
    @classmethod
    def _slug(cls, v):
        return _ID.sub("-", v.lower()).strip("-")[:40] or "section"


class Plan(BaseModel):
    title: str = Field(max_length=120)
    language: str = "tr"
    style: str = "modern"
    palette: dict[str, str] = Field(default_factory=dict, validate_default=True)  # boşsa varsayılan renkler
    font: str = "Inter"
    sections: list[Section] = Field(min_length=1)

    @field_validator("palette")
    @classmethod
    def _colors(cls, v):
        base = {"primary": "#2563eb", "secondary": "#f8fafc", "accent": "#f59e0b", "text": "#0f172a"}
        return fix_palette({k: (v.get(k) if _HEX.match(str(v.get(k, ""))) else d) for k, d in base.items()})

    @field_validator("font")
    @classmethod
    def _font(cls, v):
        return re.sub(r"[^A-Za-z0-9 ]", "", v)[:40] or "Inter"


@dataclass
class Result:
    html: str
    plan: Plan
    trace: list[dict] = field(default_factory=list)   # {"task","system","user","assistant"}
    metrics: dict = field(default_factory=dict)


class DesignPipeline:
    def __init__(self, llm: LLM, settings: Settings, prompt_version: str | None = None):
        self.llm = llm
        self.s = settings
        self.pv = prompt_version or settings.prompt_version

    def run(self, brief: str, progress: Callable[[str, float], None] = lambda *_: None) -> Result:
        t0 = time.monotonic()
        trace: list[dict] = []
        progress("plan", 0.05)
        plan = self._plan(brief, trace)
        sections, stats = [], {"removed": 0, "unbalanced": 0, "fallback_sections": 0, "retries": 0}
        for i, sec in enumerate(plan.sections):
            progress(f"section:{sec.id}", 0.1 + 0.85 * i / len(plan.sections))
            html, st = self._section(brief, plan, sec, trace)
            for k in stats:
                stats[k] += st.get(k, 0)
            sections.append(html)
        progress("assemble", 0.97)
        stats.update(seconds=round(time.monotonic() - t0, 2), sections=len(sections), llm=self.llm.name)
        return Result(html=assemble(plan, sections), plan=plan, trace=trace, metrics=stats)

    def _call(self, task: str, system: str, user: str, trace: list, **kw) -> str:
        out = self.llm.complete(system, user, **kw)
        trace.append({"task": task, "system": system, "user": user, "assistant": out})
        return out

    def _plan(self, brief: str, trace: list) -> Plan:
        system = prompts.load(self.pv, "plan_system").replace("{max_sections}", str(self.s.max_sections))
        user = f"Customer description:\n{brief.strip()}"
        for attempt in range(2):
            raw = self._call("plan", system, user, trace, max_tokens=700, json_mode=True,
                             temperature=self.s.temperature if attempt == 0 else 0.1)
            try:
                plan = Plan.model_validate(json.loads(_json_block(raw)))
                plan.sections = _unique(plan.sections)[: self.s.max_sections]
                return plan
            except (ValueError, ValidationError) as e:
                log.warning("Geçersiz plan (deneme %s): %s", attempt + 1, e)
                trace[-1]["invalid"] = True
        # Model plan üretemezse makul bir varsayılan: hat yine de sonuç verir
        return Plan(title=brief.strip()[:60] or "Website", sections=[
            Section(id=t, type=t) for t in ("navbar", "hero", "features", "about", "contact", "footer")])

    def _section(self, brief: str, plan: Plan, sec: Section, trace: list) -> tuple[str, dict]:
        system = prompts.load(self.pv, "section_system")
        outline = ", ".join(s.id for s in plan.sections)
        p = plan.palette
        user = (f"Site: {plan.title} | style: {plan.style} | language: {plan.language} | font: {plan.font} "
                f"| year: {time.localtime().tm_year}\n"
                f"Palette: primary {p['primary']}, background {p['secondary']}, accent {p['accent']}, text {p['text']}\n"
                f"Page sections in order: {outline}\n"
                f"Customer description: {brief.strip()[:1500]}\n\n"
                f'Write the section id="{sec.id}" (type: {sec.type}). Goal: {sec.goal}')
        stats = {"retries": 0}
        for attempt in range(2):
            raw = self._call("section", system, user, trace, max_tokens=self.s.max_section_tokens,
                             temperature=self.s.temperature if attempt == 0 else 0.2)
            html, st = sanitize(extract_html(raw))
            if html and f'id="{sec.id}"' not in html[:300]:
                html = f'<section id="{sec.id}">{html}</section>'
            if len(html) > 40:
                stats.update(st)
                return html, stats
            trace[-1]["invalid"] = True
            stats["retries"] += 1
        stats["fallback_sections"] = 1
        return (f'<section id="{sec.id}" class="py-16 px-6"><div class="max-w-5xl mx-auto">'
                f'<h2 class="text-3xl font-bold">{escape(sec.goal or sec.type)}</h2></div></section>'), stats


def _json_block(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("JSON yok")
    return text[start:end + 1]


def _unique(sections: list[Section]) -> list[Section]:
    seen, out = set(), []
    for s in sections:
        sid, n = s.id, 2
        while sid in seen:
            sid, n = f"{s.id}-{n}", n + 1
        seen.add(sid)
        out.append(s.model_copy(update={"id": sid}))
    return out


def assemble(plan: Plan, sections: list[str]) -> str:
    p = plan.palette
    font_q = plan.font.replace(" ", "+")
    lang = "en" if plan.language.lower().startswith("en") else "tr"
    return f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(plan.title)}</title>
<meta name="generator" content="MySimplePage AI">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family={font_q}:wght@400;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<style>
:root {{ --primary: {p["primary"]}; --secondary: {p["secondary"]}; --accent: {p["accent"]}; --text: {p["text"]}; }}
html {{ scroll-behavior: smooth; }}
body {{ font-family: '{plan.font}', ui-sans-serif, system-ui, sans-serif; color: var(--text); background: var(--secondary); }}
</style>
</head>
<body>
{chr(10).join(sections)}
</body>
</html>
"""
