# TrustRAG — Governed, Observable RAG with Hallucination Detection

```
Streamlit UI ──► FastAPI ──► LangGraph pipeline
                              guard (governance) ─► retrieve (Pinecone + MiniLM)
                              ─► context budget (tiktoken) ─► generate (Groq Llama 3.3)
                              ─► verify (NLI DeBERTa) ──► retry (strict prompt) or finalize
```

| Concern | Implementation |
|---|---|
| RAG | LlamaIndex loader + SentenceSplitter, Pinecone serverless, sentence-transformers |
| Orchestration | LangGraph state machine with conditional edges and a retry loop |
| LLM | Groq (`llama-3.3-70b-versatile`) via `langchain-groq` |
| Hallucination | Sentence-level NLI entailment (transformers) vs retrieved chunks; answers below threshold are withheld |
| Context window | Greedy token-budget packing by relevance, dedupe, truncation, history trimming |
| Governance | PII redaction (in + out), prompt-injection screen, topic policy, hashed append-only audit log |
| Observability | Per-node spans, JSONL traces, `/metrics` (latency, tokens, groundedness, block rates) |

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY and PINECONE_API_KEY
uvicorn app.main:app --reload            # terminal 1
streamlit run frontend/streamlit_app.py  # terminal 2
```
Upload `data/sample/policy.txt` in the sidebar, then ask: "How long do I have to get a refund?"

Or: `docker compose up --build` (UI on :8501, API docs on :8000/docs).

## Endpoints
`POST /ingest` · `POST /chat` · `GET /metrics` · `GET /audit` · `GET /health`

## Evaluate
`python -m eval.evaluate`

## Tuning
- `GROUNDEDNESS_THRESHOLD` (default 0.60): raise for stricter factuality.
- `CONTEXT_TOKEN_BUDGET`: keep it well under the model's window minus answer and history.
- Never commit `.env`; `logs/audit.jsonl` stores hashes only, not raw questions.
