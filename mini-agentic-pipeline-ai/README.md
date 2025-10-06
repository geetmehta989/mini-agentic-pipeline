# Mini Agentic Pipeline AI

A minimal end-to-end agentic pipeline that retrieves context from a small knowledge base, reasons with an LLM, optionally uses a CSV lookup tool, and produces a final answer with a transparent step-by-step trace.

## Objective
- Retrieve relevant context from a small KB (8–20 docs)
- Use an LLM (OpenAI API) for decision-making
- Execute an action (CSV price lookup tool)
- Output answer with a clear reasoning trace/log

## Architecture
- **Retriever**: FAISS vector index using OpenAI `text-embedding-3-small`
- **Reasoner**: GPT model (default `gpt-4o-mini`) guided by prompts in `prompts/`
- **Actor**: Executes one tool – CSV lookup over `tools/prices.csv`
- **Controller**: Orchestrates Retriever → Reasoner → Actor and aggregates logs

```
User Query → Retriever (FAISS) → Reasoner (GPT) → Actor (CSV) → Final Answer
                                    ↘─── logs (JSON) ───↗
```

## Folder Structure
```
mini-agentic-pipeline-ai/
├── data/
├── tools/
├── modules/
│   ├── retriever.py
│   ├── reasoner.py
│   ├── actor.py
│   └── controller.py
├── prompts/
│   ├── reasoner_v1.txt
│   └── reasoner_v2.txt
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup
1. Python 3.10+
2. Create a virtualenv (recommended)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```

## Run
```bash
python main.py "What is the price of the Pro plan?" --version v2
```

## Knowledge Base
Place 8–20 small `.txt` or `.md` files in `data/`. Example topics can include product docs, pricing explanations, feature matrices, and FAQs.

## Tool: CSV Lookup
Add a `tools/prices.csv` with headers: `product,sku,price,currency`. The Reasoner will decide to call this tool for pricing-related questions.

## Prompt Versions
- `reasoner_v1.txt`: Basic JSON plan + optional tool call
- `reasoner_v2.txt`: More structured steps and conservative tool usage

## Example Queries
- What is the Free vs Pro feature difference?
- How do I enable API access?
- What is the price of the Pro plan?
- Do you have a discount for students?
- Which SKU corresponds to the Business plan?
- Summarize the onboarding steps.
- What context is relevant for data retention?
- Compare rate limits between tiers.
- How can I contact support?
- What does the CSV tool return for "Pro"?
- What are the available currencies?
- Which plans include SSO?

## Evaluation Metrics
- Latency per step (ms)
- Retrieval hit quality (scores)
- Tool precision/recall on price queries
- End-to-end success rate on a test set of queries

## Demo Video
Placeholder: https://example.com/demo

## Known Limitations
- Requires OpenAI API key
- Small KB and simple tool logic
- No persistence for vector index (rebuilt each run)

## Author
Geetesh Mehta
