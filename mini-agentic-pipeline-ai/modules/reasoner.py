import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> None:  # type: ignore
        return None
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

PROMPTS = {
    "v1": "prompts/reasoner_v1.txt",
    "v2": "prompts/reasoner_v2.txt",
}

@dataclass
class ReasonerOutput:
    thought: str
    should_call_tool: bool
    tool_action: Optional[Dict[str, Any]]
    answer_plan: List[str] | str

class Reasoner:
    def __init__(self, prompts_dir: str, model: str = "gpt-4o-mini"):
        load_dotenv()
        self.model = model
        self.prompts_dir = prompts_dir
        self.offline_mode = False
        self.client = None
        if os.getenv("OPENAI_API_KEY") and OpenAI is not None:
            try:
                self.client = OpenAI()
            except Exception:
                self.offline_mode = True
        else:
            self.offline_mode = True

    def _load_prompt(self, version: str) -> str:
        path = PROMPTS.get(version)
        if not path:
            raise ValueError(f"Unknown prompt version: {version}")
        full = os.path.join(self.prompts_dir, os.path.basename(path))
        with open(full, "r", encoding="utf-8") as f:
            return f.read()

    def decide(self, query: str, retrieved: List[Dict[str, Any]], version: str = "v2") -> ReasonerOutput:
        prompt = self._load_prompt(version)
        context = "\n\n".join([f"[Doc {i+1} | score={d['score']:.3f} | {d['metadata'].get('source','')} ]\n{d['content']}" for i, d in enumerate(retrieved)])
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"User question: {query}\n\nRetrieved context:\n{context}"},
        ]
        if self.offline_mode or self.client is None:
            # Minimal deterministic fallback: never call tool, provide a simple plan
            data = {
                "thought": "No API key detected; using fallback reasoning.",
                "should_call_tool": any(word in query.lower() for word in ["price", "sku", "cost"]),
                "tool_action": {"type": "csv_lookup", "query": query} if any(word in query.lower() for word in ["price", "sku", "cost"]) else None,
                "answer_plan": ["summarize retrieved context", "use tool if relevant", "compose final answer"],
            }
            return ReasonerOutput(
                thought=data["thought"],
                should_call_tool=data["should_call_tool"],
                tool_action=data["tool_action"],
                answer_plan=data["answer_plan"],
            )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        text = resp.choices[0].message["content"]  # type: ignore[index]
        # Ensure strict JSON parse by stripping code fences if any
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.splitlines()[1:-1])
        data = json.loads(cleaned)
        return ReasonerOutput(
            thought=data.get("thought", ""),
            should_call_tool=bool(data.get("should_call_tool", False)),
            tool_action=data.get("tool_action"),
            answer_plan=data.get("answer_plan", []),
        )

