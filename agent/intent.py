"""Kullanıcı mesajının niyeti: site tasarımı mı, domain satın alma mı?

Kurallarla çalışır (model çağrısı yok: anında ve ücretsiz). Belirsiz durumlar `other` döner ve backend kullanıcıya
sorar. Bu kararlar ileride eğitim verisi olarak toplanıp model tabanlı sınıflandırıcıya geçilebilir.
"""
import re
from dataclasses import dataclass, field

# Yaygın ve ikinci düzey uzantılar (com.tr gibi) dahil
_DOMAIN = re.compile(
    r"(?<![\w@.-])((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com\.tr|net\.tr|org\.tr|gen\.tr|web\.tr|biz\.tr|"
    r"info\.tr|av\.tr|co\.uk|[a-z]{2,24}))(?![\w-])", re.I)
_DOMAIN_KW = re.compile(r"\b(domain|domainler|alan ?ad[ıi]|alan ?adlar[ıi]|uzant[ıi])", re.I)
_BUY_KW = re.compile(r"(sat[ıi]n al|almak|alabilir|al[ıi]r m[ıi]y|kay[ıi]t|tescil|buy|register|purchase|sorgula|"
                     r"bo[şs] mu|müsait|uygun mu|available|check|kontrol|bakar m[ıi]s[ıi]n|ister|istiyorum)", re.I)
_DESIGN_KW = re.compile(r"(tasarla|tasar[ıi]m|web ?sitesi|site(si)?\b|sayfa|landing|portf[öo]y|blog|e-?ticaret|"
                        r"kurumsal|website|design|menü|b[öo]l[üu]m|renk|tema)", re.I)
_DESIGN_VERB = re.compile(r"(tasarla|tasar[ıi]m|olu[şs]tur|\byap\b|\byap[ıi]n|haz[ıi]rla|design|create|build|make)", re.I)
# Dosya adı gibi görünen ama domain olmayan sonlar
_NOT_TLD = {"html", "css", "js", "png", "jpg", "jpeg", "gif", "svg", "pdf", "txt", "php", "json"}


@dataclass
class Intent:
    intent: str                       # design | domain | other
    domains: list[str] = field(default_factory=list)
    reason: str = ""


def find_domains(text: str) -> list[str]:
    out = []
    for m in _DOMAIN.finditer(text.lower()):
        name = m.group(1).removeprefix("www.")
        if name.rsplit(".", 1)[-1] in _NOT_TLD or name in out:
            continue
        out.append(name)
    return out[:5]


def classify(message: str) -> Intent:
    text = message.strip()
    domains = find_domains(text)
    domain_kw = bool(_DOMAIN_KW.search(text))
    buy_kw = bool(_BUY_KW.search(text))
    design_kw = bool(_DESIGN_KW.search(text))

    # "…sitesi yap, alan adımız x.com olacak" bir tasarım isteğidir; geçen domain'ler ayrıca bildirilir
    if design_kw and _DESIGN_VERB.search(text):
        return Intent("design", domains, "tasarım isteği")
    if domain_kw and (buy_kw or domains):
        return Intent("domain", domains, "domain anahtar kelimesi")
    if domains and not design_kw:
        return Intent("domain", domains, "yalnızca domain adı")
    if design_kw or len(text) >= 40:
        return Intent("design", domains, "site açıklaması")   # açıklamada geçen domain'ler de bildirilir
    return Intent("other", domains, "belirsiz")
