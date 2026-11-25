from agent.pipeline import Plan, Section
from agent.requested import ensure_frame, ensure_requested


def test_frame_adds_missing_footer_and_reorders():
    # Gerçek hata (run5, kuaför): plan footer'sız bitti; navbar ikinci sıradaydı
    plan = Plan(title="X", sections=[Section(id="hero", type="hero"), Section(id="navbar", type="navbar"),
                                     Section(id="appointment", type="appointment"), Section(id="pricing", type="pricing")])
    assert ensure_frame(plan, Section) == ["footer"]
    assert [s.id for s in plan.sections] == ["navbar", "hero", "appointment", "pricing", "footer"]


def test_frame_keeps_complete_plan_untouched():
    plan = Plan(title="X", sections=[Section(id=t, type=t) for t in ("navbar", "hero", "about", "footer")])
    assert ensure_frame(plan, Section) == []
    assert [s.id for s in plan.sections] == ["navbar", "hero", "about", "footer"]


def _plan(*sections, lang="tr"):
    return Plan(title="X", language=lang, sections=[Section(id=i, type=t, goal=g) for i, t, g in sections])


def test_ttihub_case_adds_about_and_separate_careers():
    # Gerçek hata: model kariyeri iletişim formuna gömdü, hakkımızda'yı atladı
    plan = _plan(("navbar", "navbar", ""), ("hero", "hero", ""), ("services", "services", "Hizmetler"),
                 ("blog", "blog", "Yazılar"), ("contact", "contact", "Ad, telefon ve proje alanlı iş başvurusu formu"),
                 ("footer", "footer", ""))
    brief = ("Bana ttihub.com isimli siteme web tasarımı yap anasayfa hakkımızda iletişim hizmetlerimiz kariyerim blog "
             "vs olsun ve kariyerim sayfasında iş başvurusu yapabilecekleri bir form olsun")
    added = ensure_requested(brief, plan, Section)
    assert added == ["about", "careers"]
    assert [s.id for s in plan.sections] == ["navbar", "hero", "about", "services", "careers", "blog", "contact", "footer"]
    assert "başvuru formu" in next(s for s in plan.sections if s.id == "careers").goal


def test_nothing_added_when_plan_already_covers():
    plan = _plan(("navbar", "navbar", ""), ("hero", "hero", ""), ("prices", "pricing", ""),
                 ("contact", "contact", "Randevu formu"), ("footer", "footer", ""))
    # Açıklama yalnızca fiyat listesi istiyor; listede olmayan kavramlar için bölüm eklenmez
    assert ensure_requested("Berber sitesi, fiyat listesi olsun", plan, Section) == []


def test_register_and_membership_in_english():
    plan = _plan(("navbar", "navbar", ""), ("hero", "hero", ""), ("footer", "footer", ""), lang="en")
    added = ensure_requested("Tennis club site with membership plans and a sign up form", plan, Section)
    assert added == ["membership", "register"]
    assert [s.type for s in plan.sections] == ["navbar", "hero", "membership", "register", "footer"]
    assert plan.sections[3].goal.startswith("Sign-up form")


def test_order_tracking_added_for_shops():
    plan = _plan(("navbar", "navbar", ""), ("hero", "hero", ""), ("products", "products", "Ürünler"),
                 ("contact", "contact", "İletişim formu ve sipariş takibi"), ("footer", "footer", ""))
    added = ensure_requested("Online kozmetik mağazası, ürünlerimiz ve sipariş takip olsun", plan, Section)
    assert added == ["order-tracking"]
    assert [s.id for s in plan.sections] == ["navbar", "hero", "products", "order-tracking", "contact", "footer"]


def test_limit_is_respected():
    plan = _plan(*[(f"s{i}", "custom", "") for i in range(12)])
    assert ensure_requested("hakkımızda kariyer blog iletişim", plan, Section, limit=12) == []
