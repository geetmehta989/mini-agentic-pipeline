# mini-agentic-pipeline
An AI-driven mini agentic pipeline that retrieves knowledge from a small knowledge base, reasons using an LLM, executes actions via API or CSV tools, and generates final answers with a complete step-by-step reasoning log.

## Quickstart

- Install: `pip install -e .`
- Run with echo LLM (no API needed):
  
  ```bash
  mini-agent --echo "What does this project do?"
  ```

- Run with the proxy (default):
  - Defaults baked into the client:
    - Base URL: `https://proxyllm.ximplify.id`
    - Model: `azure/gpt-5-mini`
    - API key can come from env `PROXY_LLM_API_KEY` or CLI flag

  ```bash
  export PROXY_LLM_API_KEY="sk-1ZD9lcxMClJfD9AZC6_Kxg"
  mini-agent "Summarize the demo KB"
  ```

## CLI options

```bash
mini-agent --help
```

## Examples

- CSV query example:
  
  ```bash
  mini-agent --echo "How many rows in sales.csv where region == 'East'?"
  ```

- HTTP example (planning only in this minimal version):
  
  ```bash
  mini-agent --echo "Fetch JSON from https://api.github.com and summarize"
  ```

## Test snippet: run against the proxy

```bash
export PROXY_LLM_API_KEY="sk-1ZD9lcxMClJfD9AZC6_Kxg"
mini-agent "Explain AI agent architecture"
```

You can override defaults:

```bash
mini-agent \
  --model "azure/gpt-5-mini" \
  --proxy-base-url "https://proxyllm.ximplify.id" \
  --proxy-api-key "$PROXY_LLM_API_KEY" \
  "Explain AI agent architecture"
```
