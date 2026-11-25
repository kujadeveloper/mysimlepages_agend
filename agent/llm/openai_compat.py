"""OpenAI uyumlu sunucu: Ollama, vLLM, llama.cpp server, LM Studio... Modeli ayrı bir süreçte/cihazda çalıştırmak için."""
import logging
import time

import httpx

from .base import LLMError

log = logging.getLogger(__name__)

# Sunucu açılırken/yeniden başlarken (vLLM model yüklemesi birkaç dakika sürer) işi hemen düşürmek yerine bekleriz.
_RETRY_STATUS = {502, 503, 504}


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "none", timeout: float = 300,
                 wait_ready: float = 300, retry_interval: float = 5):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                    headers={"Authorization": f"Bearer {api_key}"})
        self._model = model
        self._wait_ready = wait_ready
        self._retry_interval = retry_interval
        self.name = f"openai:{model}"

    def complete(self, system, user, *, max_tokens, temperature, json_mode=False, stop=None):
        body = {"model": self._model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if stop:
            body["stop"] = stop
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        deadline = time.monotonic() + self._wait_ready
        while True:
            try:
                r = self._client.post("/chat/completions", json=body)
                if r.status_code in _RETRY_STATUS and time.monotonic() < deadline:
                    raise _NotReady(f"HTTP {r.status_code}")
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except (httpx.ConnectError, httpx.RemoteProtocolError, _NotReady) as e:
                if time.monotonic() >= deadline:
                    raise LLMError(f"LLM sunucusu {self._wait_ready:.0f} sn içinde hazır olmadı: {e}") from e
                log.warning("LLM sunucusu hazır değil (%s), %.0f sn sonra yeniden denenecek", e, self._retry_interval)
                time.sleep(self._retry_interval)
            except (httpx.HTTPError, KeyError, IndexError) as e:
                raise LLMError(f"LLM sunucusu yanıt vermedi: {e}") from e


class _NotReady(Exception):
    pass
