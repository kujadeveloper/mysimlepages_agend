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


@lru_cache
def load_latest(version: str, name: str) -> str:
    """Şablon bu sürümde yoksa en yakın yeni sürümden alır (ör. düzeltme prompt'ları v5'te eklendi; eski bir sürümle
    eğitilmiş model de düzeltme yapabilsin)."""
    try:
        return load(version, name)
    except FileNotFoundError:
        versions = sorted((p.name for p in _ROOT.iterdir() if p.is_dir() and p.name[1:].isdigit()),
                          key=lambda v: int(v[1:]), reverse=True)
        for v in versions:
            if (_ROOT / v / f"{name}.txt").is_file():
                return load(v, name)
        raise
