"""Model çıktısını güvenli HTML'e çevirir ve kalitesini ölçer.

Üretilen sayfa müşterinin alan adında yayınlanacağı için betik, olay öznitelikleri, javascript: bağlantıları vb. kaldırılır.
Ölçümler (kaldırılan öğe, kapanmamış etiket) değerlendirme ve eğitim verisi filtrelemede kullanılır.
"""
import re
from html import escape
from html.parser import HTMLParser

DROP_WITH_CONTENT = {"script", "style", "iframe", "object", "embed", "noscript", "template"}
DROP_TAG = {"html", "head", "body", "link", "meta", "base", "frame", "frameset", "applet"}
VOID = {"area", "br", "col", "hr", "img", "input", "source", "track", "wbr"}
URL_ATTRS = {"href", "src", "action", "formaction", "poster", "srcset", "xlink:href"}
_BAD_URL = re.compile(r"^\s*(javascript|vbscript|data:text/html)", re.I)
_FENCE = re.compile(r"```(?:html)?\s*(.*?)```", re.S | re.I)


def extract_html(text: str) -> str:
    """Model markdown çiti ya da açıklama eklediyse yalnızca HTML'i al."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("<")
    end = text.rfind(">")
    return text[start:end + 1] if start != -1 and end > start else ""


class _Sanitizer(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.out: list[str] = []
        self.stack: list[str] = []
        self.skip_depth = 0
        self.removed = 0
        self.unbalanced = 0

    def handle_starttag(self, tag, attrs):
        if self.skip_depth or tag in DROP_WITH_CONTENT:
            if tag in DROP_WITH_CONTENT:
                self.skip_depth += 1
                self.removed += 1
            return
        if tag in DROP_TAG:
            self.removed += 1
            return
        clean = []
        for name, value in attrs:
            bad_style = name == "style" and value and re.search(r"expression\(|javascript:", value, re.I)
            if name.startswith("on") or bad_style:
                self.removed += 1
                continue
            if name in URL_ATTRS and value and _BAD_URL.match(value):
                self.removed += 1
                continue
            clean.append(f' {name}="{escape(value, quote=True)}"' if value is not None else f" {name}")
        if tag == "a" and any(a[0] == "target" for a in attrs):
            clean.append(' rel="noopener noreferrer"')
        self.out.append(f"<{tag}{''.join(clean)}>")
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack and self.stack[-1] == tag:
            self.stack.pop()
            self.out[-1] = self.out[-1][:-1] + f"></{tag}>"

    def handle_endtag(self, tag):
        if tag in DROP_WITH_CONTENT:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth or tag in DROP_TAG or tag in VOID:
            return
        if tag not in self.stack:
            self.unbalanced += 1
            return
        while self.stack:  # araya giren kapanmamış etiketleri kapat
            t = self.stack.pop()
            self.out.append(f"</{t}>")
            if t == tag:
                break
            self.unbalanced += 1

    def handle_data(self, data):
        if not self.skip_depth:
            self.out.append(data)

    def handle_entityref(self, name):
        if not self.skip_depth:
            self.out.append(f"&{name};")

    def handle_charref(self, name):
        if not self.skip_depth:
            self.out.append(f"&#{name};")

    def handle_comment(self, data):
        pass


def sanitize(fragment: str) -> tuple[str, dict]:
    p = _Sanitizer()
    p.feed(fragment)
    p.close()
    unclosed = len(p.stack)
    for t in reversed(p.stack):
        p.out.append(f"</{t}>")
    return "".join(p.out).strip(), {"removed": p.removed, "unbalanced": p.unbalanced + unclosed}
