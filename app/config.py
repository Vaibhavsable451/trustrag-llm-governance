import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "trustrag")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_dim: int = 384  # all-MiniLM-L6-v2
    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    context_budget: int = int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000"))
    grounded_threshold: float = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.60"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "1"))
    top_k: int = 8
    min_score: float = 0.25
    blocked_topics: tuple = field(default_factory=lambda: tuple(
        t.strip().lower() for t in os.getenv("BLOCKED_TOPICS", "").split(",") if t.strip()))
    trace_file: str = "logs/traces.jsonl"
    audit_file: str = "logs/audit.jsonl"


settings = Settings()
