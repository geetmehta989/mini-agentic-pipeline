from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any

import os

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class LLMResponse:
    content: str
    tokens_in: int
    tokens_out: int


class LLMClient:
    def __init__(self, model: str, temperature: float = 0.2, echo: bool = False):
        self.model = model
        self.temperature = temperature
        self.echo = echo
        self._client = None
        if not echo and OpenAI is not None:
            self._client = OpenAI()

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        if self.echo or self._client is None:
            # Deterministic echo for offline/dev
            joined = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            return LLMResponse(content=f"[echo]\n{joined}", tokens_in=len(joined)//4, tokens_out=len(joined)//5)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        return LLMResponse(content=content, tokens_in=tokens_in, tokens_out=tokens_out)
