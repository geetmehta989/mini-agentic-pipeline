from typing import Optional
import json
import sys
from rich import print as rprint
import typer

from .pipeline import AgentPipeline, PipelineConfig

app = typer.Typer(add_completion=False, help="Mini agentic pipeline CLI")


@app.command()
def run(
    question: str = typer.Argument(..., help="User question to answer"),
    kb_path: str = typer.Option("examples/kb", help="Path to knowledge base directory"),
    data_path: str = typer.Option("examples/data", help="Path to data directory for CSVs"),
    model: str = typer.Option("gpt-4o-mini", help="LLM model name"),
    echo: bool = typer.Option(False, help="Use echo LLM (no API calls)"),
    temperature: float = typer.Option(0.2, help="LLM temperature"),
    max_steps: int = typer.Option(6, help="Max tool/think steps"),
    log: bool = typer.Option(True, help="Print full step-by-step reasoning log"),
):
    """Run the mini agentic pipeline over a small KB and optional tools."""
    config = PipelineConfig(
        kb_path=kb_path,
        data_path=data_path,
        model=model,
        echo=echo,
        temperature=temperature,
        max_steps=max_steps,
    )

    pipeline = AgentPipeline(config)
    answer, trace = pipeline.run(question)

    if log:
        rprint({"trace": trace})
    rprint({"answer": answer})


if __name__ == "__main__":
    app()
