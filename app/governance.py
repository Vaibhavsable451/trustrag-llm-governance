"""AI governance: PII redaction, prompt-injection screening, topic policy, audit trail."""
import hashlib
import json
import re
import time
from .config import settings

PII = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}(?!\d)"),
}
INJECTION = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the )?(system|previous) prompt",
    r"reveal (your )?(system )?prompt",
    r"you are now (dan|developer mode)",
    r"jailbreak",
]


def redact(text: str) -> tuple[str, list[str]]:
    found = []
    for label, rx in PII.items():
        if rx.search(text):
            found.append(label)
            text = rx.sub(f"[REDACTED_{label}]", text)
    return text, found


def check_input(question: str) -> dict:
    flags, blocked, reason = [], False, ""
    low = question.lower()
    if any(re.search(p, low) for p in INJECTION):
        blocked, reason = True, "Possible prompt-injection attempt."
        flags.append("PROMPT_INJECTION")
    for topic in settings.blocked_topics:
        if topic in low:
            blocked, reason = True, f"Topic blocked by policy: {topic}"
            flags.append("BLOCKED_TOPIC")
    clean, pii = redact(question)
    flags += [f"PII_{p}" for p in pii]
    if len(question) > 4000:
        blocked, reason = True, "Input too long."
        flags.append("TOO_LONG")
    return {"clean": clean, "flags": flags, "blocked": blocked, "reason": reason}


def audit(record: dict) -> None:
    """Append-only audit log. Raw questions are hashed, never stored."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question_sha256": hashlib.sha256(record["question"].encode()).hexdigest(),
        "model": settings.groq_model,
        "flags": record.get("flags", []),
        "verdict": record.get("verdict"),
        "groundedness": record.get("groundedness"),
        "sources": record.get("sources", []),
    }
    with open(settings.audit_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
