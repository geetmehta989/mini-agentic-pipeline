from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from .kb import KnowledgeBase
from .llm import LLMClient
from .tools import CSVTool, HTTPTool


@dataclass
class PipelineConfig:
    kb_path: str
    data_path: str
    model: str = "azure/gpt-5-mini"
    echo: bool = False
    temperature: float = 0.2
    max_steps: int = 6
    # Proxy configuration (optional; will read env defaults in LLMClient)
    proxy_base_url: str | None = None
    proxy_api_key: str | None = None


class AgentPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.kb = KnowledgeBase(config.kb_path)
        self.llm = LLMClient(
            model=config.model,
            temperature=config.temperature,
            echo=config.echo,
            base_url=config.proxy_base_url,
            api_key=config.proxy_api_key,
        )
        self.csv_tool = CSVTool(config.data_path)
        self.http_tool = HTTPTool()

    def run(self, question: str) -> tuple[str, List[Dict[str, Any]]]:
        trace: List[Dict[str, Any]] = []
        # Step 1: retrieve
        chunks = self.kb.search(question, limit=4)
        trace.append({
            "step": "retrieve",
            "num_chunks": len(chunks),
            "chunks": [{"path": c.path, "content": c.content[:500]} for c in chunks],
        })

        # Step 2: think & tool plan
        system = (
            "You are a helpful assistant with access to a local knowledge base and simple tools.\n"
            "If needed, propose a plan using tools (CSV or HTTP). Otherwise answer directly."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\nKB snippets: \n" + "\n\n".join(c.content[:500] for c in chunks)},
        ]
        plan_resp = self.llm.complete(messages)
        trace.append({"step": "plan", "model": self.config.model, "response": plan_resp.content})

        # Step 3: execute at most N tool steps (very simple heuristic)
        # For this minimal pipeline, we won't parse tool calls; we only demonstrate tool availability.
        # Future enhancement could parse JSON tool plans.

        # Step 4: final answer
        final_messages = messages + [
            {"role": "assistant", "content": plan_resp.content},
            {"role": "user", "content": "Provide the final concise answer now."},
        ]
        final_resp = self.llm.complete(final_messages)
        trace.append({"step": "finalize", "response": final_resp.content})
        return final_resp.content, trace
