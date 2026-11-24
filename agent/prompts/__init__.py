from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).parent


@lru_cache
def load(version: str, name: str) -> str:
    """Sürümlü prompt şablonu: prompts/<version>/<name>.txt. Eğitilen model, eğitildiği sürümle çalıştırılır."""
    path = _ROOT / version / f"{name}.txt"
    if not path.is_file() or path.parent.parent != _ROOT:
        raise FileNotFoundError(f"Prompt bulunamadı: {version}/{name}")
    # Windows'ta git CRLF ile çekmiş olabilir: model her platformda aynı prompt'u görsün
    return path.read_text("utf-8").replace("\r\n", "\n").strip()
