from rich import print as rprint
import typer

from .pipeline import AgentPipeline, PipelineConfig

app = typer.Typer(add_completion=False, help="Mini agentic pipeline CLI")


@app.command()
def run(
    question: str = typer.Argument(..., help="User question to answer"),
    kb_path: str = typer.Option("examples/kb", help="Path to knowledge base directory"),
    data_path: str = typer.Option("examples/data", help="Path to data directory for CSVs"),
    model: str = typer.Option("azure/gpt-5-mini", help="LLM model name (proxy)"),
    echo: bool = typer.Option(False, help="Use echo LLM (no API calls)"),
    temperature: float = typer.Option(0.2, help="LLM temperature"),
    max_steps: int = typer.Option(6, help="Max tool/think steps"),
    log: bool = typer.Option(True, help="Print full step-by-step reasoning log"),
    proxy_base_url: str = typer.Option(
        None,
        help="Proxy base URL (defaults to env PROXY_LLM_BASE_URL or https://proxyllm.ximplify.id)",
    ),
    proxy_api_key: str = typer.Option(
        None,
        help="Proxy API key (defaults to env PROXY_LLM_API_KEY)",
    ),
):
    """Run the mini agentic pipeline over a small KB and optional tools."""
    config = PipelineConfig(
        kb_path=kb_path,
        data_path=data_path,
        model=model,
        echo=echo,
        temperature=temperature,
        max_steps=max_steps,
        proxy_base_url=proxy_base_url,
        proxy_api_key=proxy_api_key,
    )

    pipeline = AgentPipeline(config)
    answer, trace = pipeline.run(question)

    if log:
        rprint({"trace": trace})
    rprint({"answer": answer})


if __name__ == "__main__":
    app()
