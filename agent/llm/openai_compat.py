"""OpenAI uyumlu sunucu: Ollama, vLLM, llama.cpp server, LM Studio... Modeli ayrı bir süreçte/cihazda çalıştırmak için."""
import httpx

from .base import LLMError


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout: float = 300):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    headers={"Authorization": f"Bearer {api_key}"})
        self._model = model
        self.name = f"openai:{model}"

    def complete(self, system, user, *, max_tokens, temperature, json_mode=False, stop=None):
        body = {"model": self._model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if stop:
            body["stop"] = stop
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = self._client.post("/chat/completions", json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError) as e:
            raise LLMError(f"LLM sunucusu yanıt vermedi: {e}") from e
