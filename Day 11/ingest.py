"""Ingestion: PDFs -> text -> chunks -> local embeddings -> ChromaDB.

Run:  uv run python ingest.py
Rebuilds the collection from scratch every time, so chunk counts never double.
"""
import os
import re
import sys
import time

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

import config

EMBED_BATCH = 32


def embed_texts(texts: list[str], prefix: str) -> list[list[float]]:
    """Embed with the local embedding model (OpenAI-compatible /v1/embeddings)."""
    client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key=config.EMBEDDING_API_KEY)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = [prefix + t for t in texts[i : i + EMBED_BATCH]]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return vectors


def clean_page_text(text: str) -> str:
    """Tidy layout-mode text: collapse runs of spaces (table columns become ' | ')."""
    lines = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        # two or more spaces inside a line mark a table column gap
        line = re.sub(r"(?<=\S) {2,}(?=\S)", " | ", line.strip())
        lines.append(line)
    return "\n".join(lines)


def load_pdf_pages(path: str) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        if config.PDF_EXTRACTION_MODE == "layout":
            text = clean_page_text(page.extract_text(extraction_mode="layout") or "")
        else:
            text = (page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def build_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    pdfs = sorted(f for f in os.listdir(config.DOCUMENTS_DIR) if f.lower().endswith(".pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {config.DOCUMENTS_DIR}/")
    for name in pdfs:
        pages = load_pdf_pages(os.path.join(config.DOCUMENTS_DIR, name))
        count_before = len(chunks)
        for page_number, text in pages:
            for idx, piece in enumerate(splitter.split_text(text)):
                chunks.append(
                    {
                        # stable id: re-ingesting never creates duplicates
                        "id": f"{name}-p{page_number}-c{idx}",
                        "text": piece,
                        "source": name,
                        "page": page_number,
                    }
                )
        print(f"  {name}: {len(pages)} pages -> {len(chunks) - count_before} chunks")
    return chunks


def main() -> None:
    print(f"Chunking (unit: characters): size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}, "
          f"extraction={config.PDF_EXTRACTION_MODE}")
    chunks = build_chunks()
    print(f"Total chunks: {len(chunks)}. Embedding with {config.EMBEDDING_MODEL} ...")

    start = time.time()
    vectors = embed_texts([c["text"] for c in chunks], config.EMBED_DOC_PREFIX)
    print(f"Embedded {len(vectors)} chunks, dimension {len(vectors[0])}, {time.time() - start:.1f}s")

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:  # first run: collection does not exist yet
        pass
    # cosine space, so relevance = 1 - distance is a real similarity
    collection = client.create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
        embeddings=vectors,  # always pass embeddings: Chroma must never embed on its own
    )
    print(f"Stored. collection.count() = {collection.count()}")


if __name__ == "__main__":
    main()
