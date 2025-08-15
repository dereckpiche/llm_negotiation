from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class PolicyOutput:
    content: str
    reasoning_content: str | None = None

class LLMInferenceBackend(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def prepare_adapter(
        self, adapter_id: str, weights_got_updated: bool = False
    ) -> None:
        """Ensure adapter is ready/loaded for next generation call."""

    @abstractmethod
    async def generate(self, prompt: list[dict], regex: Optional[str] = None) -> PolicyOutput:
        ...

    @abstractmethod
    def toggle_training_mode(self) -> None:
        ...

    @abstractmethod
    def toggle_eval_mode(self) -> None:
        ...

    @abstractmethod
    def shutdown(self) -> None:
        ...
