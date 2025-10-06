from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import httpx


@dataclass
class HTTPToolResult:
    status: int
    json: Any | None
    text: str | None


class HTTPTool:
    def __init__(self, timeout_sec: float = 10.0):
        self._client = httpx.Client(timeout=timeout_sec, follow_redirects=True)

    def get(self, url: str, headers: Dict[str, str] | None = None) -> HTTPToolResult:
        resp = self._client.get(url, headers=headers)
        try:
            data = resp.json()
        except Exception:
            data = None
        text = None if data is not None else resp.text[:4000]
        return HTTPToolResult(status=resp.status_code, json=data, text=text)

    def post(self, url: str, json_body: Dict[str, Any] | None = None, headers: Dict[str, str] | None = None) -> HTTPToolResult:
        resp = self._client.post(url, json=json_body, headers=headers)
        try:
            data = resp.json()
        except Exception:
            data = None
        text = None if data is not None else resp.text[:4000]
        return HTTPToolResult(status=resp.status_code, json=data, text=text)
