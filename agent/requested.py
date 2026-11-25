"""Müşterinin açıkça istediği bölümleri garanti eder.

Küçük modeller "hakkımızda, kariyer, blog, iletişim olsun" gibi listelerde bazı bölümleri atlıyor ya da iki formu
birleştiriyor (ör. iş başvurusunu iletişim formuna gömüyor). Açıklamada geçen bir bölüm planda yoksa burada
deterministik olarak, mantıklı sıraya eklenir; bölümün içeriğini yine model yazar.
"""
import re
from dataclasses import dataclass

# Sayfadaki doğal sıra: eksik bölüm, sırası kendisinden sonra gelen ilk mevcut bölümün önüne eklenir
ORDER = ["hero", "about", "services", "categories", "products", "gallery", "membership", "pricing", "appointment",
         "careers", "blog", "faq", "register", "order-tracking", "contact"]


@dataclass(frozen=True)
class Concept:
    type: str
    brief_re: str      # açıklamada istenmiş mi
    covered_re: str    # plandaki bir bölüm (id/type/goal) bunu zaten karşılıyor mu
    goal_tr: str
    goal_en: str


CONCEPTS = [
    Concept("about", r"hakk[ıi]m[ıi]zda|about\s*us",
            r"\babout\b|hakk[ıi]m[ıi]zda|hikaye|hikâye",
            "Firmanın hikayesi, kuruluş yılı, değerleri ve ekibi",
            "Company story, founding year, values and team"),
    Concept("services", r"hizmetlerimiz|hizmetler\b|our\s*services",
            r"\bservices?\b|hizmet",
            "Sunulan hizmetlerin açıklamalı kartları",
            "Cards describing each service offered"),
    Concept("products", r"[üu]r[üu]nlerim|[üu]r[üu]nler\b|our\s*products|\bshop\b",
            r"\bproducts?\b|[üu]r[üu]n|\bshop\b|koleksiyon",
            "Fiyatlı ürün kartları ve kategoriler",
            "Product cards with prices and categories"),
    Concept("membership", r"[üu]yelik(?!\s*form)|membership",
            r"\bmembership\b|[üu]yelik|\bplans?\b",
            "Aylık ve yıllık üyelik planları, fiyatları, her plana dahil olanlar ve Üye Ol butonları",
            "Monthly and yearly membership plans with prices, what is included and Join buttons"),
    Concept("appointment", r"randevu|appointment|book(ing)?\s+(an?\s+)?(appointment|session)",
            r"\bappointment\b|randevu|\bbooking\b|rezervasyon",
            "Hizmet, tarih, saat, ad, telefon ve not alanlı randevu formu ve çalışma saatleri",
            "Appointment form with service, date, time slot, name, phone and note, next to opening hours"),
    Concept("careers", r"kariyer|i[şs]\s*ba[şs]vuru|insan\s*kaynaklar|a[çc][ıi]k\s*pozisyon|career|job\s*application|hiring",
            r"\bcareers?\b|kariyer|ba[şs]vuru|\bjobs?\b",
            "Açık pozisyonlar listesi ve ad, e-posta, telefon, pozisyon, CV bağlantısı, ön yazı alanlı iş başvuru formu",
            "Open positions and a job application form with name, email, phone, position, CV link and cover letter"),
    Concept("blog", r"\bblog",
            r"\bblog\b|yaz[ıi]lar|\barticles?\b",
            "Başlık, tarih ve kısa özetli üç blog yazısı kartı",
            "Three blog post cards with title, date and short summary"),
    Concept("faq", r"\bsss\b|s[ıi]k\s*sorulan|\bfaq\b",
            r"\bfaq\b|\bsss\b|s[ıi]k\s*sorulan",
            "Sık sorulan sorular ve cevapları",
            "Frequently asked questions with answers"),
    Concept("register", r"[üu]ye\s*ol|[üu]ye\s*kay[ıi]t|kay[ıi]t\s*ol|[üu]yelik\s*form|sign\s*-?\s*up|\bregist(er|ration)\b",
            r"\bregister\b|\bsign-?up\b|[üu]ye\s*ol|kay[ıi]t",
            "Ad, e-posta, telefon, şifre, şifre tekrar ve onay kutusu alanlı üye kayıt formu",
            "Sign-up form with name, email, phone, password, password again and consent checkbox"),
    Concept("order-tracking", r"sipari[şs]\s*takip|kargo\s*takip|order\s*track|track\s*(your\s*)?order",
            r"track|takip",
            "Sipariş numarası ve e-posta alanlı sipariş takip formu",
            "Order tracking form with order number and email"),
    Concept("contact", r"ileti[şs]im|contact",
            r"\bcontact\b|ileti[şs]im",
            "Adres, telefon, e-posta ve ad, e-posta, konu, mesaj alanlı iletişim formu",
            "Address, phone, email and a contact form with name, email, subject and message"),
]


def _covered(concept: Concept, sections) -> bool:
    for s in sections:
        if s.type == concept.type:
            return True
        # Hedef metni başka bir formu tarif ediyorsa (ör. "iletişim ... iş başvurusu formu") o bölüm bu kavramı
        # karşılamış sayılmaz: iki ayrı form isteniyorsa ikisi de ayrı bölüm olmalı.
        if concept.type in ("careers", "register", "appointment", "order-tracking", "contact"):
            if re.search(concept.covered_re, s.id, re.I):
                return True
            continue
        if re.search(concept.covered_re, f"{s.id} {s.goal}", re.I):
            return True
    return False


FRAME_GOALS = {
    "navbar": ("Marka adı ve bölüm bağlantıları, mobilde ☰ menü", "Brand name and section links, mobile ☰ menu"),
    "hero": ("Büyük başlık, tek cümlelik değer önerisi ve ana çağrı butonu", "Big headline, one-sentence value proposition and main call to action"),
    "footer": ("Marka adı, kısa açıklama, bölüm bağlantıları ve telif yılı", "Brand name, short line, section links and copyright year"),
}


def ensure_frame(plan, section_cls) -> list[str]:
    """Sayfa iskeleti: navbar en başta, hemen ardından hero, footer en sonda. Eksikse ekler, yanlış yerdeyse taşır."""
    en = plan.language.lower().startswith("en")
    fixed = []

    def take(t):
        for i, s in enumerate(plan.sections):
            if s.type == t or s.id == t:
                return plan.sections.pop(i)
        fixed.append(t)
        goal = FRAME_GOALS[t][1 if en else 0]
        ids = {s.id for s in plan.sections}
        return section_cls(id=t if t not in ids else f"{t}-main", type=t, goal=goal)

    nav, hero, foot = take("navbar"), take("hero"), take("footer")
    plan.sections = [nav, hero, *plan.sections, foot]
    return fixed


def ensure_requested(brief: str, plan, section_cls, limit: int = 12) -> list[str]:
    """Açıklamada istenip planda olmayan bölümleri ekler; eklenen tiplerin listesini döndürür."""
    text = brief.lower()
    added = []
    for c in CONCEPTS:
        if not re.search(c.brief_re, text, re.I) or _covered(c, plan.sections):
            continue
        if len(plan.sections) >= limit:
            break
        ids = {s.id for s in plan.sections}
        sid, n = c.type, 2
        while sid in ids:
            sid, n = f"{c.type}-{n}", n + 1
        goal = c.goal_en if plan.language.lower().startswith("en") else c.goal_tr
        new = section_cls(id=sid, type=c.type, goal=goal)
        rank = ORDER.index(c.type)
        pos = next((i for i, s in enumerate(plan.sections)
                    if s.type == "footer" or (s.type in ORDER and ORDER.index(s.type) > rank)), len(plan.sections))
        plan.sections.insert(pos, new)
        added.append(c.type)
    return added
