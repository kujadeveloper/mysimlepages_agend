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
from agent.requested import ensure_frame, ensure_requested
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
    sections: dict[str, str] = field(default_factory=dict)   # bölüm id -> html (düzeltmelerde korunur)


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
        return Result(html=assemble(plan, sections), plan=plan, trace=trace, metrics=stats,
                      sections={s.id: h for s, h in zip(plan.sections, sections)})

    # ---- Düzeltme: mevcut tasarımdan yeni sürüm ----

    def revise(self, brief: str, plan: Plan, sections: dict[str, str], instruction: str,
               progress: Callable[[str, float], None] = lambda *_: None) -> Result:
        """Kullanıcının talimatını uygular. Yalnızca değişmesi gereken bölümler yeniden yazılır; renk/font/başlık
        plandan uygulanır; eklenen bölüm sıfırdan üretilir, çıkarılan bölüm atılır."""
        t0 = time.monotonic()
        trace: list[dict] = []
        progress("revise_plan", 0.05)
        new_plan, change, summary = self._revise_plan(plan, instruction, trace)
        old_ids = [s.id for s in plan.sections]
        new_ids = [s.id for s in new_plan.sections]
        added = [i for i in new_ids if i not in sections]
        # Bölüm eklendi/çıkarıldı ya da marka adı değişti: menü ve footer linkleri/adı güncellenmeli
        if new_ids != old_ids or new_plan.title != plan.title:
            change |= {s.id for s in new_plan.sections if s.type in ("navbar", "footer")}
        change -= set(added)
        todo = [s for s in new_plan.sections if s.id in change or s.id in added]
        out, stats = {}, {"removed": 0, "unbalanced": 0, "fallback_sections": 0, "retries": 0}
        done = 0
        for sec in new_plan.sections:
            if sec.id in added:
                progress(f"section:{sec.id}", 0.1 + 0.85 * done / max(1, len(todo)))
                html, st = self._section(f"{brief}\nChange request: {instruction}", new_plan, sec, trace)
                done += 1
            elif sec.id in change:
                progress(f"revise:{sec.id}", 0.1 + 0.85 * done / max(1, len(todo)))
                html, st = self._revise_section(brief, new_plan, sec, sections[sec.id], instruction, trace)
                done += 1
            else:
                html, st = sections[sec.id], {}
            for k in stats:
                stats[k] += st.get(k, 0)
            out[sec.id] = html
        progress("assemble", 0.97)
        stats.update(seconds=round(time.monotonic() - t0, 2), sections=len(out), llm=self.llm.name,
                     revision={"summary": summary, "changed": sorted(change), "added": added,
                               "removed": [i for i in old_ids if i not in new_ids]})
        return Result(html=assemble(new_plan, list(out.values())), plan=new_plan, trace=trace, metrics=stats,
                      sections=out)

    def _revise_plan(self, plan: Plan, instruction: str, trace: list) -> tuple[Plan, set[str], str]:
        system = prompts.load_latest(self.pv, "revise_plan_system")
        user = (f"Current plan:\n{json.dumps(plan.model_dump(), ensure_ascii=False)}\n\n"
                f"Change request:\n{instruction.strip()}")
        for attempt in range(2):
            raw = self._call("revise_plan", system, user, trace, max_tokens=1200, json_mode=True,
                             temperature=0.2 if attempt == 0 else 0.0)
            try:
                data = json.loads(_json_block(raw))
                new = Plan.model_validate(data.get("plan") or data)
                new.sections = _unique(new.sections)[: self.s.max_sections + 2]
                ensure_frame(new, Section)
                ensure_requested(instruction, new, Section, limit=self.s.max_sections + 2)
                ids = {s.id for s in new.sections}
                change = {str(i) for i in data.get("change") or [] if str(i) in ids}
                if not change and not _plan_changed(plan, new):
                    change = _mentioned(instruction, new) or ids   # model hiçbir şeyi değiştirmediyse
                return new, change, str(data.get("summary") or "")[:300]
            except (ValueError, ValidationError, TypeError, AttributeError) as e:
                log.warning("Geçersiz düzeltme planı (deneme %s): %s", attempt + 1, e)
                trace[-1]["invalid"] = True
        # Yedek: plan aynen kalır, talimatta adı geçen bölümler (yoksa tümü) yeniden yazılır
        new = plan.model_copy(deep=True)
        ensure_requested(instruction, new, Section, limit=self.s.max_sections + 2)
        return new, _mentioned(instruction, new) or {s.id for s in new.sections}, ""

    def _revise_section(self, brief: str, plan: Plan, sec: Section, current: str, instruction: str,
                        trace: list) -> tuple[str, dict]:
        system = (prompts.load(self.pv, "section_system") + "\n\n"
                  + prompts.load_latest(self.pv, "revise_section_rules"))
        user = (f"{section_user(brief, plan, sec)}\n\nCurrent HTML of this section:\n{current}\n\n"
                f"Change request: {instruction.strip()}")
        # Mevcut HTML uzunsa çıktı da uzun olur: kırpılmasın
        budget = max(self.s.max_section_tokens, len(current) // 3 + 400)
        stats = {"retries": 0}
        for attempt in range(2):
            raw = self._call("revise_section", system, user, trace, max_tokens=budget,
                             temperature=0.2 if attempt == 0 else 0.0)
            html, st = sanitize(extract_html(raw))
            if html and f'id="{sec.id}"' not in html[:300]:
                html = f'<section id="{sec.id}">{html}</section>'
            if len(html) > 40:
                stats.update(st)
                return html, stats
            trace[-1]["invalid"] = True
            stats["retries"] += 1
        stats["fallback_sections"] = 1
        return current, stats   # düzeltilemezse eski hali korunur

    def _call(self, task: str, system: str, user: str, trace: list, **kw) -> str:
        out = self.llm.complete(system, user, **kw)
        trace.append({"task": task, "system": system, "user": user, "assistant": out})
        return out

    def _plan(self, brief: str, trace: list) -> Plan:
        system = plan_system(self.pv, self.s.max_sections)
        user = plan_user(brief)
        for attempt in range(2):
            raw = self._call("plan", system, user, trace, max_tokens=1000, json_mode=True,
                             temperature=self.s.temperature if attempt == 0 else 0.1)
            try:
                plan = Plan.model_validate(json.loads(_json_block(raw)))
                plan.sections = _unique(plan.sections)[: self.s.max_sections]
                added = ensure_frame(plan, Section)
                added += ensure_requested(brief, plan, Section, limit=self.s.max_sections + 2)
                if added:
                    log.info("Açıklamada istenip planda olmayan bölümler eklendi: %s", added)
                    trace[-1]["added_sections"] = added
                return plan
            except (ValueError, ValidationError) as e:
                log.warning("Geçersiz plan (deneme %s): %s", attempt + 1, e)
                trace[-1]["invalid"] = True
        # Model plan üretemezse makul bir varsayılan: hat yine de sonuç verir
        return Plan(title=brief.strip()[:60] or "Website", sections=[
            Section(id=t, type=t) for t in ("navbar", "hero", "features", "about", "contact", "footer")])

    def _section(self, brief: str, plan: Plan, sec: Section, trace: list) -> tuple[str, dict]:
        system = prompts.load(self.pv, "section_system")
        user = section_user(brief, plan, sec)
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


# Prompt metinleri: servis ve eğitim verisi (training/build_examples.py) aynı fonksiyonları kullanır
def plan_system(prompt_version: str, max_sections: int) -> str:
    return prompts.load(prompt_version, "plan_system").replace("{max_sections}", str(max_sections))


def plan_user(brief: str) -> str:
    return f"Customer description:\n{brief.strip()}"


def section_user(brief: str, plan: Plan, sec: Section) -> str:
    outline = ", ".join(s.id for s in plan.sections)
    p = plan.palette
    return (f"Site: {plan.title} | style: {plan.style} | language: {plan.language} | font: {plan.font} "
            f"| year: {time.localtime().tm_year}\n"
            f"Palette: primary {p['primary']}, background {p['secondary']}, accent {p['accent']}, text {p['text']}\n"
            f"Page sections in order: {outline}\n"
            f"Customer description: {brief.strip()[:1500]}\n\n"
            f'Write the section id="{sec.id}" (type: {sec.type}). Goal: {sec.goal}')


def _plan_changed(old: Plan, new: Plan) -> bool:
    return (old.palette != new.palette or old.font != new.font or old.title != new.title
            or [(s.id, s.goal) for s in old.sections] != [(s.id, s.goal) for s in new.sections])


# Talimatta bir bölümün adı geçiyor mu? (Düzeltme planı geçersizse hangi bölümlerin yeniden yazılacağını bulur)
_ALIASES = {
    "navbar": r"men[üu]|navbar|navigasyon|[üu]st\s*bar|nav\b",
    "hero": r"hero|giri[şs]\s*(b[öo]l[üu]m|k[ıi]sm)|kar[şs][ıi]lama|banner|man[şs]et|ana\s*ba[şs]l[ıi]k|[üu]st\s*k[ıi]s[ıi]m",
    "footer": r"footer|alt\s*(k[ıi]s[ıi]m|bilgi|b[öo]l[üu]m)|altbilgi",
    "pricing": r"fiyat|[üu]cret|price|pricing|paket",
    "contact": r"ileti[şs]im|contact|telefon|adres|e-?posta|phone|address",
    "testimonials": r"yorum|testimonial|review",
    "gallery": r"galeri|gallery|portfolyo|portfolio",
    "team": r"ekip|team|kadro|doktorlar",
    "faq": r"\bsss\b|s[ıi]k\s*sorulan|faq",
    "menu": r"\bmen[üu]s[üu]\b|yemek\s*listesi",
}


def _mentioned(instruction: str, plan: Plan) -> set[str]:
    from agent.requested import CONCEPTS
    text = instruction.lower()
    found = set()
    patterns = {c.type: c.covered_re for c in CONCEPTS} | _ALIASES
    for s in plan.sections:
        if re.search(rf"\b{re.escape(s.id)}\b", text) or re.search(rf"\b{re.escape(s.type)}\b", text):
            found.add(s.id)
        elif s.type in patterns and re.search(patterns[s.type], text, re.I):
            found.add(s.id)
    return found


_ROOT_TAG = re.compile(r'<(section|nav|footer|header|main|div)\b[^>]*\bid="([^"]+)"[^>]*>', re.I)


def split_sections(html: str, plan: Plan) -> dict[str, str]:
    """Birleştirilmiş sayfadan bölüm HTML'lerini geri çıkarır (bölümleri ayrı saklanmayan eski tasarımlar için)."""
    out = {}
    wanted = {s.id for s in plan.sections}
    for m in _ROOT_TAG.finditer(html):
        tag, sid = m.group(1).lower(), m.group(2)
        if sid not in wanted or sid in out:
            continue
        depth, end = 0, None
        for t in re.finditer(rf"<(/?){tag}\b[^>]*>", html[m.start():], re.I):
            depth += -1 if t.group(1) else 1
            if depth == 0:
                end = m.start() + t.end()
                break
        if end:
            out[sid] = html[m.start():end]
    return out


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
