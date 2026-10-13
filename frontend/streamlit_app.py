import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="TrustRAG", page_icon="🛡️", layout="wide")
st.title("🛡️ TrustRAG — governed, observable RAG")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Knowledge base")
    files = st.file_uploader("Upload PDF / TXT / MD / DOCX", accept_multiple_files=True)
    if st.button("Ingest", disabled=not files):
        with st.spinner("Chunking and indexing into Pinecone…"):
            r = requests.post(f"{API}/ingest", files=[("files", (f.name, f.getvalue())) for f in files])
        if r.ok:
            st.success(str(r.json()))
        else:
            st.error(r.text)
    st.divider()
    st.header("Live metrics")
    try:
        m = requests.get(f"{API}/metrics", timeout=3).json()
        c1, c2 = st.columns(2)
        c1.metric("Requests", m["requests"])
        c2.metric("Avg latency", f"{m['avg_latency_ms']:.0f} ms")
        c1.metric("Avg groundedness", m["avg_groundedness"])
        c2.metric("Blocked", m["blocked"])
        c1.metric("Ungrounded", m["ungrounded"])
        c2.metric("Tokens out", m["tokens_out"])
    except Exception:
        st.caption("API offline")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if q := st.chat_input("Ask about your documents"):
    st.chat_message("user").markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            r = requests.post(f"{API}/chat", json={"question": q, "history": st.session_state.history[-6:]})
        if not r.ok:
            st.error(r.text)
        else:
            d = r.json()
            st.markdown(d["answer"])
            badge = {"grounded": "🟢", "ungrounded": "🔴", "blocked": "⛔", "no_context": "🟡"}[d["verdict"]]
            st.caption(f"{badge} {d['verdict']} · groundedness {d['groundedness']} · "
                       f"context {d['context_tokens']}/{d['context_budget']} tokens · request {d['request_id']}")
            if d["flags"]:
                st.warning("Governance flags: " + ", ".join(d["flags"]))
            with st.expander("Sources"):
                for i, s in enumerate(d["sources"], 1):
                    st.markdown(f"**[{i}] {s['source']}** (similarity {s['score']})")
                    st.caption(s["text"])
            with st.expander("Trace"):
                st.dataframe(d["trace"])
            if d["unsupported"]:
                with st.expander("Unsupported claims"):
                    st.write(d["unsupported"])
            st.session_state.history += [{"role": "user", "content": q},
                                         {"role": "assistant", "content": d["answer"]}]
