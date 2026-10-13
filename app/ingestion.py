"""LlamaIndex loading + sentence-aware chunking -> Pinecone."""
import hashlib
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from . import vectorstore


def ingest_paths(paths: list[str], chunk_size: int = 512, overlap: int = 64) -> dict:
    docs = SimpleDirectoryReader(input_files=paths).load_data()
    nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap).get_nodes_from_documents(docs)
    chunks = []
    for n in nodes:
        text = n.get_content().strip()
        if len(text) < 40:
            continue
        src = n.metadata.get("file_name", "unknown")
        cid = hashlib.md5(f"{src}:{text}".encode()).hexdigest()
        chunks.append({"id": cid, "text": text, "source": src})
    return {"documents": len(docs), "chunks": vectorstore.upsert(chunks) if chunks else 0}
