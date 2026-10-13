"""Lightweight observability: spans, JSONL traces, and aggregate metrics."""
import json
import threading
import time
import uuid
from contextlib import contextmanager
from .config import settings

_lock = threading.Lock()
_m = {"requests": 0, "blocked": 0, "ungrounded": 0, "no_context": 0, "retries": 0,
      "latency_ms_sum": 0.0, "tokens_in": 0, "tokens_out": 0,
      "groundedness_sum": 0.0, "groundedness_n": 0}


@contextmanager
def span(trace: list, name: str):
    rec, t0 = {"span": name}, time.perf_counter()
    try:
        yield rec
    finally:
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        trace.append(rec)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def record(result: dict) -> None:
    total = sum(s["ms"] for s in result["trace"])
    with _lock:
        _m["requests"] += 1
        _m["latency_ms_sum"] += total
        _m["tokens_in"] += result.get("usage", {}).get("in", 0)
        _m["tokens_out"] += result.get("usage", {}).get("out", 0)
        _m["retries"] += max(result.get("attempts", 1) - 1, 0)
        v = result.get("verdict")
        _m["blocked"] += v == "blocked"
        _m["ungrounded"] += v == "ungrounded"
        _m["no_context"] += v == "no_context"
        if result.get("groundedness") is not None:
            _m["groundedness_sum"] += result["groundedness"]
            _m["groundedness_n"] += 1
    with open(settings.trace_file, "a") as f:
        f.write(json.dumps({"request_id": result["request_id"], "verdict": result.get("verdict"),
                            "groundedness": result.get("groundedness"), "total_ms": round(total, 1),
                            "usage": result.get("usage"), "trace": result["trace"]}) + "\n")


def metrics() -> dict:
    with _lock:
        n = max(_m["requests"], 1)
        g = max(_m["groundedness_n"], 1)
        return {**_m, "avg_latency_ms": round(_m["latency_ms_sum"] / n, 1),
                "avg_groundedness": round(_m["groundedness_sum"] / g, 3),
                "hallucination_block_rate": round(_m["ungrounded"] / n, 3)}
