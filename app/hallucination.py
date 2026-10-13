"""Hallucination detection: sentence-level NLI entailment against retrieved context.

Each answer sentence is the hypothesis; each retrieved chunk is a premise.
A sentence is supported if any chunk entails it. Groundedness = mean support score.
"""
import re
from functools import lru_cache
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .config import settings


@lru_cache(maxsize=1)
def _nli():
    tok = AutoTokenizer.from_pretrained(settings.nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.nli_model).eval()
    ent = next(i for i, l in model.config.id2label.items() if l.lower().startswith("entail"))
    return tok, model, ent


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", "", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*") for p in parts if len(p.strip()) > 20]


@torch.no_grad()
def check(answer: str, chunks: list[dict]) -> dict:
    sents = _sentences(answer)
    if not sents or not chunks:
        return {"groundedness": 0.0, "sentences": [], "unsupported": []}
    tok, model, ent = _nli()
    pairs = [(c["text"], s) for s in sents for c in chunks]
    probs = []
    for i in range(0, len(pairs), 16):
        b = pairs[i:i + 16]
        enc = tok([p for p, _ in b], [h for _, h in b], truncation="only_first",
                  max_length=512, padding=True, return_tensors="pt")
        probs += torch.softmax(model(**enc).logits, dim=-1)[:, ent].tolist()
    n, per = len(chunks), []
    for i, s in enumerate(sents):
        per.append({"sentence": s, "support": round(max(probs[i * n:(i + 1) * n]), 3)})
    score = sum(p["support"] for p in per) / len(per)
    return {"groundedness": round(score, 3), "sentences": per,
            "unsupported": [p["sentence"] for p in per if p["support"] < 0.5]}
