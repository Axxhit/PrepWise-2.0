# PrepWise 2.0 — Agentic Interview Coach

An AI-powered mock interview platform with RAG, multi-tool agents, and multi-model support.

---

## Architecture

```
User → FastAPI Backend
           ├── /upload        → saves PDF to disk
           ├── /process       → extracts text, chunks it
           ├── /embed         → embeds chunks → ChromaDB
           ├── /retrieve      → RAG similarity search
           └── /generate-questions → RAG + LLM question gen

Agent (LangGraph)
    ├── fetch_resume_context  → calls /retrieve
    ├── search_web            → Tavily search
    └── evaluate_answer       → LLM-based scoring
```

---

## RAG Pipeline

Resumes and job descriptions are chunked into ~300 token segments with 50-token overlap
and embedded using `sentence-transformers/all-MiniLM-L6-v2`. Vectors are stored in
ChromaDB with per-session isolation. At query time, the top 3 chunks by cosine similarity
are retrieved and injected into the LLM prompt as context.

**Why RAG over fine-tuning?**

| | RAG | Fine-tuning |
|---|---|---|
| Setup time | Hours | Days–weeks |
| Handles new resumes | Yes (runtime) | No (retrain needed) |
| Cost | Low | High |
| Hallucination risk | Lower (grounded) | Higher |
| Best for | Dynamic personal data | Fixed domain knowledge |

---

## Agent Loop

Built with LangGraph. Two nodes — `call_llm` and `call_tools` — connected by a
conditional edge that routes back to `call_tools` if the LLM requests a tool call,
or exits to `END` on a final text response. Max 5 iterations per run.

**Tools:**
- `fetch_resume_context(topic)` — RAG retrieval over uploaded resume
- `search_web(query)` — Tavily web search for current interview trends
- `evaluate_answer(question, answer)` — LLM-based answer scoring (1–5)

**Context window management:**
Sliding window trimmer keeps the original task message + most recent messages
within a 6000 token budget. Token counts logged via `tiktoken` on every iteration.

---

## Model Comparison

Same 20 interview question generation prompts, temperature=0.7, across 3 models.

| Provider | Model | Avg Quality (1-5) | Avg Latency | Avg Output Tokens |
|---|---|---|---|---|
| Gemini | gemini-2.0-flash | X.XX | X.XXs | XXX |
| Llama | llama-3.1-70b-versatile | X.XX | X.XXs | XXX |
| Mixtral | mixtral-8x7b-32768 | X.XX | X.XXs | XXX |

> Fill in X.XX values from your `benchmark_summary.csv`

**Key findings:**
- [Your finding 1 from benchmark_notes.md]
- [Your finding 2]
- [Your finding 3]

**Production choice:** [Your chosen model] — [one sentence reason]

---

## Temperature Experiments

| Temperature | Avg Quality (1-5) | Avg Latency | Avg Output Length |
|---|---|---|---|
| 0.3 | X.XX | X.XXs | XXX chars |
| 0.7 | X.XX | X.XXs | XXX chars |
| 1.0 | X.XX | X.XXs | XXX chars |

> Fill in from `temperature_results.csv`

**Observation:** [Your finding from temperature_notes.md — 2 sentences]

---

## Stack

| Layer | Tool |
|---|---|
| LLM (primary) | Gemini 2.0 Flash |
| LLM (OSS) | Llama 3.1 70B, Mixtral 8x7B via Groq |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB |
| Agent framework | LangGraph |
| Web search | Tavily |
| Backend | FastAPI |
| Token counting | tiktoken |

---

## Setup

```bash
git clone https://github.com/yourusername/prepwise-2
cd prepwise-backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in keys:
```
GEMINI_API_KEY=
GROQ_API_KEY=
TAVILY_API_KEY=
LLM_PROVIDER=gemini
LLM_TEMPERATURE=0.7
```

Start backend:
```bash
uvicorn main:app --reload
```

---

## Project Structure

```
prepwise-backend/
├── main.py                  # FastAPI endpoints
├── agent_graph.py           # LangGraph agent
├── agent_loop.py            # Raw agent loop (learning artifact)
├── llm_client.py            # Abstracted LLM clients
├── context_manager.py       # Token counting + sliding window
├── benchmark.py             # Model comparison pipeline
├── temperature_experiment.py
├── benchmark_results.csv
├── benchmark_summary.csv
├── temperature_results.csv
├── benchmark_notes.md
├── temperature_notes.md
└── requirements.txt
```

---

## Eval Suite

> Phase 4 — coming next