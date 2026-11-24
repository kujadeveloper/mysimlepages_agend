"""GPU/model olmadan uçtan uca test ve geliştirme için deterministik arka uç."""
import json
import re


class MockLLM:
    name = "mock"

    def complete(self, system, user, *, max_tokens, temperature, json_mode=False, stop=None):
        if json_mode:
            return json.dumps({
                "title": "Demo Site", "language": "tr", "style": "modern",
                "palette": {"primary": "#2563eb", "secondary": "#f8fafc", "accent": "#f59e0b", "text": "#0f172a"},
                "font": "Inter",
                "sections": [
                    {"id": "hero", "type": "hero", "goal": "Ziyaretçiyi karşıla ve ana mesajı ver"},
                    {"id": "features", "type": "features", "goal": "Öne çıkan üç özellik"},
                    {"id": "contact", "type": "contact", "goal": "İletişim formu"},
                ],
            })
        m = re.search(r'id="?([a-z0-9-]+)', user)
        sid = m.group(1) if m else "section"
        return (f'<section id="{sid}" class="py-16 px-6"><div class="max-w-5xl mx-auto">'
                f'<h2 class="text-3xl font-bold">{sid.title()}</h2><p class="mt-4">Örnek içerik.</p>'
                f'<a href="#" onclick="alert(1)">Bağlantı</a><script>alert(1)</script></div></section>')
