from agent.config import Settings
from agent.registry import ModelVersion

from .base import LLMError


def build_llm(settings: Settings, version: ModelVersion | None):
    """Etkin model sürümüne göre arka ucu kurar. Sürüm yoksa ayarlardaki varsayılan model kullanılır."""
    backend = version.backend if version else settings.backend
    if backend == "mock":
        from .mock import MockLLM
        return MockLLM()
    if backend == "openai":
        from .openai_compat import OpenAICompatLLM
        model = (version.model_ref if version else None) or settings.openai_model
        return OpenAICompatLLM(settings.openai_base_url, model, settings.openai_api_key)
    if backend == "llamacpp":
        from .llamacpp import LlamaCppLLM
        model_path = (version.model_ref if version else None) or settings.default_model_path
        if not model_path:
            raise LLMError("GGUF model yolu yok: MSP_DEFAULT_MODEL_PATH ayarlayın ya da bir model sürümü etkinleştirin")
        return LlamaCppLLM(model_path, lora_path=version.adapter_path if version else None, n_ctx=settings.n_ctx,
                           n_gpu_layers=settings.n_gpu_layers, n_threads=settings.n_threads)
    raise LLMError(f"Bilinmeyen arka uç: {backend}")
