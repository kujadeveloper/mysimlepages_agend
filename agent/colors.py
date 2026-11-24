"""Paletteki okunabilirlik hatalarını kodla düzeltir (WCAG kontrast oranı).

Küçük modeller renk uyumunu sık bozar (koyu zemin üstüne koyu yazı gibi). Bunu eğitmeyi beklemeden deterministik
olarak düzeltiyoruz: zemin/metin en az 4.5:1, ana renk üstündeki beyaz düğme metni en az 4.5:1 olur.
"""


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h[:3])
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def _rgb_to_hex(rgb) -> str:
    return "#" + "".join(f"{round(max(0, min(1, c)) * 255):02x}" for c in rgb)


def luminance(h: str) -> float:
    def ch(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = map(ch, _hex_to_rgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _shade(h: str, factor: float) -> str:
    """factor < 1 koyulaştırır, > 1 açar (beyaza doğru)."""
    rgb = _hex_to_rgb(h)
    if factor <= 1:
        return _rgb_to_hex(c * factor for c in rgb)
    t = factor - 1
    return _rgb_to_hex(c + (1 - c) * t for c in rgb)


def fix_palette(p: dict[str, str]) -> dict[str, str]:
    p = dict(p)
    bg = p["secondary"]
    # Metin zeminde okunmalı; değilse en iyi koyu/açık nötr renge geç
    if contrast(p["text"], bg) < 4.5:
        p["text"] = max(("#111827", "#ffffff"), key=lambda c: contrast(c, bg))
    # Ana renk düğme zemini: üstündeki beyaz metin okunana dek koyulaştır
    for _ in range(12):
        if contrast("#ffffff", p["primary"]) >= 4.5:
            break
        p["primary"] = _shade(p["primary"], 0.85)
    # Açık zeminde ana renk başlık/bağlantı olarak da görünmeli. Yalnızca koyulaştırırız: düğme kontrastı bozulmaz.
    # (Koyu zeminde düğmeler beyaz metinle zaten ayrışır; açmak düğme metnini okunmaz yapardı.)
    for _ in range(12):
        if luminance(bg) <= 0.5 or contrast(p["primary"], bg) >= 3:
            break
        p["primary"] = _shade(p["primary"], 0.85)
    return p
