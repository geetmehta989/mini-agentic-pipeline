import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .retriever import Retriever
from .reasoner import Reasoner
from .actor import Actor, ToolResult

@dataclass
class StepLog:
    step: str
    data: Dict[str, Any]
    started_at: float
    finished_at: float
    elapsed_ms: float

@dataclass
class PipelineState:
    query: str
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None
    final_answer: str = ""
    logs: List[StepLog] = field(default_factory=list)

class Controller:
    def __init__(self, data_dir: str, tools_dir: str, prompts_dir: str):
        self.retriever = Retriever(data_dir=data_dir)
        self.reasoner = Reasoner(prompts_dir=prompts_dir)
        self.actor = Actor(tools_dir=tools_dir)

    def _log(self, state: PipelineState, step: str, payload: Dict[str, Any], started: float, finished: float) -> None:
        state.logs.append(
            StepLog(
                step=step,
                data=payload,
                started_at=started,
                finished_at=finished,
                elapsed_ms=(finished - started) * 1000.0,
            )
        )

    def run(self, query: str, prompt_version: str = "v2") -> PipelineState:
        state = PipelineState(query=query)
        # Step 1: Retrieve
        t0 = time.time()
        retrieved_docs = self.retriever.search(query, k=5)
        t1 = time.time()
        state.retrieved = [
            {"content": d.content, "metadata": d.metadata, "score": d.score}
            for d in retrieved_docs
        ]
        self._log(state, "retrieve", {"hits": state.retrieved}, t0, t1)

        # Step 2: Reason
        t2 = time.time()
        reasoning = self.reasoner.decide(query, state.retrieved, version=prompt_version)
        t3 = time.time()
        state.reasoning = {
            "thought": reasoning.thought,
            "should_call_tool": reasoning.should_call_tool,
            "tool_action": reasoning.tool_action,
            "answer_plan": reasoning.answer_plan,
        }
        self._log(state, "reason", state.reasoning, t2, t3)

        # Step 3: Act (optional)
        tool_res: Optional[ToolResult] = None
        if reasoning.should_call_tool:
            t4 = time.time()
            tool_res = self.actor.run(reasoning.tool_action)
            t5 = time.time()
            state.tool_result = {
                "tool": tool_res.tool if tool_res else None,
                "input": tool_res.input if tool_res else None,
                "output": tool_res.output if tool_res else None,
            }
            self._log(state, "act", state.tool_result or {}, t4, t5)

        # Step 4: Finalize answer
        final_parts: List[str] = []
        final_parts.append(f"Question: {query}")
        if state.retrieved:
            final_parts.append("Top retrieval hits:")
            for i, h in enumerate(state.retrieved[:3]):
                src = h["metadata"].get("source", "")
                final_parts.append(f"- Hit {i+1} (score={h['score']:.3f}): {src}")
        if tool_res and tool_res.output:
            final_parts.append("Tool results:")
            if isinstance(tool_res.output, list):
                for row in tool_res.output[:5]:
                    final_parts.append(f"- {row}")
            else:
                final_parts.append(str(tool_res.output))
        final_parts.append("Answer plan: " + ("; ".join(state.reasoning.get("answer_plan", [])) if isinstance(state.reasoning.get("answer_plan"), list) else str(state.reasoning.get("answer_plan"))))
        state.final_answer = "\n".join(final_parts)
        return state

