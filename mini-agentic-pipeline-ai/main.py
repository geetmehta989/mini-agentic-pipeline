import argparse
import json
import os
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv() -> None:  # type: ignore
        return None

from modules.controller import Controller

load_dotenv()

def run_cli():
    parser = argparse.ArgumentParser(description="Mini Agentic Pipeline AI")
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("--version", default="v2", help="Prompt version: v1 or v2")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    app = Controller(
        data_dir=os.path.join(base, "data"),
        tools_dir=os.path.join(base, "tools"),
        prompts_dir=os.path.join(base, "prompts"),
    )
    state = app.run(args.query, prompt_version=args.version)

    print("=== Final Answer ===")
    print(state.final_answer)
    print("\n=== Reasoning Trace ===")
    for log in state.logs:
        print(json.dumps({
            "step": log.step,
            "elapsed_ms": round(log.elapsed_ms, 2),
            "data": log.data,
        }, ensure_ascii=False))

if __name__ == "__main__":
    run_cli()
