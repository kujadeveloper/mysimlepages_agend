from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    """Tüm arka uçların ortak arayüzü. Sohbet şablonu arka uca aittir; ajan yalnızca system/user metni verir."""
    name: str

    def complete(self, system: str, user: str, *, max_tokens: int, temperature: float,
                 json_mode: bool = False, stop: list[str] | None = None) -> str: ...
