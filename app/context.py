"""Context-window management: token budgeting for retrieved chunks and chat history."""
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")  # close approximation for Llama tokenizers


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def truncate(text: str, max_tokens: int) -> str:
    ids = _enc.encode(text)
    return text if len(ids) <= max_tokens else _enc.decode(ids[:max_tokens]) + " …"


def build_context(chunks: list[dict], budget: int) -> tuple[list[dict], str, int]:
    """Greedy pack by relevance score; truncate the last chunk if it partly fits."""
    selected, parts, used, seen = [], [], 0, set()
    for c in sorted(chunks, key=lambda x: -x["score"]):
        if c["text"] in seen:
            continue
        seen.add(c["text"])
        header = f"[{len(selected) + 1}] (source: {c['source']})\n"
        room = budget - used - count_tokens(header)
        if room < 50:
            break
        body = truncate(c["text"], room)
        parts.append(header + body)
        selected.append({**c, "text": body})
        used += count_tokens(header + body)
    return selected, "\n\n".join(parts), used


def trim_history(history: list[dict], budget: int = 800) -> list[dict]:
    """Keep the most recent turns that fit the history budget."""
    kept, used = [], 0
    for m in reversed(history):
        t = count_tokens(m["content"])
        if used + t > budget:
            break
        kept.append(m)
        used += t
    return list(reversed(kept))
