"""Yerel GGUF modeli (llama-cpp-python).

Düşük kartlar için: 4-bit (Q4_K_M) kuantize küçük bir kod modeli (ör. Qwen2.5-Coder-1.5B/3B-Instruct) ~1.5-3 GB VRAM ile çalışır.
VRAM yetmezse MSP_N_GPU_LAYERS ile katmanların bir kısmı CPU'ya bırakılır. Model süreç başına bir kez yüklenir.
"""
import threading

from .base import LLMError


class LlamaCppLLM:
    def __init__(self, model_path: str, *, lora_path: str | None, n_ctx: int, n_gpu_layers: int, n_threads: int):
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover
            raise LLMError("llama-cpp-python kurulu değil: pip install -e .[llamacpp]") from e
        kwargs = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False,
                      flash_attn=True)  # flash attention: daha az KV bellek
        if n_threads:
            kwargs["n_threads"] = n_threads
        if lora_path:
            kwargs["lora_path"] = lora_path  # eğitilmiş adaptör, birleştirmeden kullanılabilir
        self._llm = Llama(**kwargs)
        self._lock = threading.Lock()  # llama.cpp örneği iş parçacığı güvenli değil
        self.name = f"llamacpp:{model_path}" + (f"+{lora_path}" if lora_path else "")

    def complete(self, system, user, *, max_tokens, temperature, json_mode=False, stop=None):
        kwargs = dict(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                      max_tokens=max_tokens, temperature=temperature, stop=stop)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}  # gramerle kısıtlı: geçersiz JSON üretemez
        with self._lock:
            out = self._llm.create_chat_completion(**kwargs)
        return out["choices"][0]["message"]["content"] or ""
