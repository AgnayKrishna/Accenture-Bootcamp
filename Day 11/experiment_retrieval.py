"""Retrieval experiment: which chunk size / overlap / top-K / extraction mode finds the evidence?

This measures RETRIEVAL only (no LLM), so it is fast. For every answerable question in
test_questions.json, a configuration scores a hit when the retrieved chunks contain the
"evidence_phrases" of that question.

Run:    uv run python experiment_retrieval.py
Check the phrases against the PDF text without Ollama:   uv run python experiment_retrieval.py --check-evidence
Output: retrieval_experiments.json  (cite this file for any retrieval number in the decision log)
Uses its own temporary ChromaDB in .experiment_db/, so it never touches chroma_db/.
"""
import json
import os
import re
import shutil
import sys

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from ingest import embed_texts, load_pdf_pages

CONFIGS = [
    # (extraction_mode, chunk_size, chunk_overlap)
    ("plain", 512, 50),     # the brief's defaults
    ("layout", 512, 50),
    ("layout", 1000, 150),
    ("layout", 1500, 200),
]
TOP_KS = [3, 5, 8]
OUTPUT = "retrieval_experiments.json"
TMP_DB = ".experiment_db"


def normalize(text: str) -> str:
    """Lowercase, drop table separators, collapse whitespace, so phrase matching is robust."""
    text = text.replace("|", " ").lower()
    return re.sub(r"\s+", " ", text).strip()


def load_questions() -> list[dict]:
    with open(config.TEST_QUESTIONS_FILE, encoding="utf-8") as f:
        return [q for q in json.load(f) if q.get("evidence_phrases")]


def pages_for(mode: str) -> list[tuple[str, int, str]]:
    config.PDF_EXTRACTION_MODE = mode
    out = []
    for name in sorted(f for f in os.listdir(config.DOCUMENTS_DIR) if f.lower().endswith(".pdf")):
        for page, text in load_pdf_pages(os.path.join(config.DOCUMENTS_DIR, name)):
            out.append((name, page, text))
    return out


def check_evidence() -> None:
    """Confirm every evidence phrase really occurs in the extracted text (no Ollama needed)."""
    bad = 0
    for mode in ("plain", "layout"):
        full = normalize(" ".join(t for _, _, t in pages_for(mode)))
        for q in load_questions():
            for phrase in q["evidence_phrases"]:
                if normalize(phrase) not in full:
                    bad += 1
                    print(f"[{mode}] {q['id']}: phrase NOT found: {phrase!r}")
    print("All evidence phrases found." if not bad else f"{bad} phrase(s) missing.")


def main() -> None:
    shutil.rmtree(TMP_DB, ignore_errors=True)  # leftovers from an interrupted run
    questions = load_questions()
    results = []
    for mode, size, overlap in CONFIGS:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
        chunks = []
        for name, page, text in pages_for(mode):
            for i, piece in enumerate(splitter.split_text(text)):
                chunks.append({"id": f"{name}-p{page}-c{i}", "text": piece, "source": name, "page": page})
        print(f"\n{mode}, size={size}, overlap={overlap} (characters): {len(chunks)} chunks")

        client = chromadb.PersistentClient(path=TMP_DB)
        col_name = f"exp_{mode}_{size}_{overlap}"  # unique per config; Chroma caches clients per path
        col = client.create_collection(col_name, metadata={"hnsw:space": "cosine"})
        col.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
                embeddings=embed_texts([c["text"] for c in chunks], config.EMBED_DOC_PREFIX))
        qvecs = embed_texts([q["question"] for q in questions], config.EMBED_QUERY_PREFIX)

        for k in TOP_KS:
            per_question = []
            for q, vec in zip(questions, qvecs):
                res = col.query(query_embeddings=[vec], n_results=k, include=["documents"])
                joined = normalize(" ".join(res["documents"][0]))
                found = [normalize(p) in joined for p in q["evidence_phrases"]]
                per_question.append({"id": q["id"], "found": found, "all_found": all(found)})
            hits = sum(r["all_found"] for r in per_question)
            partial = sum(sum(r["found"]) for r in per_question)
            total_phrases = sum(len(r["found"]) for r in per_question)
            print(f"  top_k={k}: all evidence found for {hits}/{len(questions)} questions "
                  f"({partial}/{total_phrases} phrases)")
            results.append({
                "extraction_mode": mode, "chunk_size_chars": size, "chunk_overlap_chars": overlap,
                "top_k": k, "num_chunks": len(chunks), "embedding_model": config.EMBEDDING_MODEL,
                "questions_with_all_evidence": hits, "questions_total": len(questions),
                "phrases_found": partial, "phrases_total": total_phrases,
                "per_question": per_question,
            })
    del client
    shutil.rmtree(TMP_DB, ignore_errors=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} configurations to {OUTPUT}")


if __name__ == "__main__":
    if "--check-evidence" in sys.argv:
        check_evidence()
    else:
        main()
