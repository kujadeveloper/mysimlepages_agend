from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Tüm ayarlar ortam değişkeninden (MSP_ öneki) ya da .env dosyasından okunur."""
    model_config = SettingsConfigDict(env_prefix="MSP_", env_file=".env", extra="ignore")

    # Kalıcı veri: model kayıt defteri, veri setleri, iş kuyruğu (sqlite)
    data_dir: Path = Path("data")

    # Servisler arası paylaşılan anahtar (Django backend -> ajan). Boşsa doğrulama kapalı (yalnızca geliştirme).
    api_key: str = ""

    # LLM arka ucu: "llamacpp" (yerel GGUF), "openai" (Ollama / vLLM / llama.cpp server), "mock" (test)
    backend: str = "mock"
    # Kayıt defterinde etkin sürüm yoksa kullanılacak model
    default_model_path: str = ""          # llamacpp: .gguf dosyası
    openai_base_url: str = "http://localhost:11434/v1"
    openai_model: str = "qwen2.5-coder:3b"
    openai_api_key: str = "none"

    # Düşük donanım ayarları
    n_ctx: int = 4096                    # küçük bağlam = az VRAM (KV cache)
    n_gpu_layers: int = -1               # -1: hepsi GPU'da; VRAM yetmezse 10-20 gibi kısmi değer, 0: yalnızca CPU
    n_threads: int = 0                   # 0: otomatik
    max_section_tokens: int = 1200       # bölüm başına üretim sınırı
    temperature: float = 0.4
    max_sections: int = 8
    workers: int = 1                     # tek GPU'da aynı anda 1 iş: bellek taşmasın

    # Kullanıcı geri bildirimini eğitim verisi olarak sakla
    collect_feedback: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.sqlite3"

    @property
    def registry_dir(self) -> Path:
        return self.data_dir / "registry"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
