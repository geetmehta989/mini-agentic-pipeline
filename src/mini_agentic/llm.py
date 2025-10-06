from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import os
import httpx

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
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        echo: bool = False,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.echo = echo
        # Resolve proxy configuration with environment and safe defaults
        self.base_url = base_url or os.getenv("PROXY_LLM_BASE_URL", "https://proxyllm.ximplify.id")
        self.api_key = api_key or os.getenv("PROXY_LLM_API_KEY", "sk-1ZD9lcxMClJfD9AZC6_Kxg")

        self._client = None
        if not echo and OpenAI is not None:
            # Configure OpenAI client with a custom httpx client that ignores env proxies
            self._http_client = httpx.Client(
                timeout=60.0,
                follow_redirects=True,
                trust_env=False,
            )
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                http_client=self._http_client,
            )
        elif not echo and OpenAI is None:
            raise RuntimeError(
                "openai package is not available. Install it or run with --echo."
            )

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
