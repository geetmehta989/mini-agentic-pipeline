from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path


@dataclass
class KBChunk:
    path: str
    content: str


class KnowledgeBase:
    def __init__(self, root: str):
        self.root = Path(root)

    def search(self, query: str, limit: int = 5) -> List[KBChunk]:
        """Very simple keyword search over text files in the KB directory."""
        if not self.root.exists():
            return []
        hits: List[Tuple[int, KBChunk]] = []
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".txt", ".md", ".mdx"}:
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                score = self._score(text, query)
                if score > 0:
                    hits.append((score, KBChunk(str(path), text[:4000])))
        hits.sort(key=lambda x: x[0], reverse=True)
        return [h[1] for h in hits[:limit]]

    def _score(self, text: str, query: str) -> int:
        q_terms = [t.lower() for t in query.split() if len(t) > 2]
        text_l = text.lower()
        score = 0
        for t in q_terms:
            score += text_l.count(t)
        return score
