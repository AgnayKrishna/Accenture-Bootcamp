"""Retrieval: embed the query locally, search ChromaDB, return chunks plus metadata."""
import time

import chromadb

import config
from ingest import embed_texts

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        _collection = client.get_collection(config.COLLECTION_NAME)
    return _collection


def retrieve_chunks(query: str, top_k: int | None = None) -> dict:
    top_k = top_k or config.TOP_K
    start = time.time()
    query_vector = embed_texts([query], config.EMBED_QUERY_PREFIX)[0]
    result = _get_collection().query(
        query_embeddings=[query_vector],  # same embedding model as ingest
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        chunks.append(
            {
                "content": doc,
                "source": meta["source"],
                "page": meta["page"],
                "relevance_score": round(1 - dist, 4),  # cosine similarity
            }
        )
    return {
        "query": query,
        "retrieved_chunks": chunks,
        "retrieval_time_ms": round((time.time() - start) * 1000),
        "embedding_model": config.EMBEDDING_MODEL,
    }


if __name__ == "__main__":
    import json
    import sys

    q = " ".join(sys.argv[1:]) or "How many days of remote work are allowed per week?"
    print(json.dumps(retrieve_chunks(q), indent=2, ensure_ascii=False))
