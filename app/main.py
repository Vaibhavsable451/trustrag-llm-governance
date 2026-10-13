import json
import os
import shutil
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from . import graph, ingestion
from .config import settings
from .observability import metrics

app = FastAPI(title="TrustRAG", version="1.0.0")
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("logs", exist_ok=True)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok", "model": settings.groq_model, "index": settings.pinecone_index}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    paths = []
    for f in files:
        dest = os.path.join("data/uploads", os.path.basename(f.filename))
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
        paths.append(dest)
    try:
        return ingestion.ingest_paths(paths)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        return graph.run(req.question, req.history)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")


@app.get("/metrics")
def get_metrics():
    return metrics()


@app.get("/audit")
def get_audit(limit: int = 50):
    if not os.path.exists(settings.audit_file):
        return []
    with open(settings.audit_file) as f:
        return [json.loads(l) for l in f.readlines()[-limit:]]
