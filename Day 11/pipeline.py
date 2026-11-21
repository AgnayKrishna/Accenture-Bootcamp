"""End-to-end: query -> retrieve -> generate -> answer.

Run one question:   uv run python pipeline.py "What is the PTO carryover limit?"
Run the test set:   uv run python pipeline.py --all     (writes pipeline_outputs.json)
"""
import json
import sys

import config
from generate import generate_answer
from retrieve import retrieve_chunks


def answer_question(query: str) -> dict:
    retrieval_result = retrieve_chunks(query)
    generation_result = generate_answer(query=query, chunks=retrieval_result["retrieved_chunks"])
    return {**generation_result, "retrieval": retrieval_result}


def run_all() -> None:
    with open(config.TEST_QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for n, q in enumerate(questions, start=1):
        print(f"[{n}/{len(questions)}] {q['id']}: {q['question']}")
        r = answer_question(q["question"])
        chunks = r["retrieval"]["retrieved_chunks"]
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "actual_answer": r["answer"],
                "retrieved_context": [c["content"] for c in chunks],
                "context_sources": [
                    {"source": c["source"], "page": c["page"], "score": c["relevance_score"]}
                    for c in chunks
                ],
                "model": r["model"],
                "thinking": r["thinking"],
                "finish_reason": r["finish_reason"],
                "embedding_model": r["retrieval"]["embedding_model"],
                "prompt_tokens": r["prompt_tokens"],
                "tokens_generated": r["tokens_generated"],
                "retrieval_time_ms": r["retrieval"]["retrieval_time_ms"],
                "generation_time_ms": r["generation_time_ms"],
                "passed": None,  # you fill these two in by hand after the final run
                "rating_note": "",
            }
        )
        # save after every question so a crash never loses the finished ones
        with open(config.PIPELINE_OUTPUTS_FILE, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"    -> {r['finish_reason']}, {r['generation_time_ms']} ms: {r['answer'][:90]!r}")
    print(f"Saved {len(rows)} rows to {config.PIPELINE_OUTPUTS_FILE}")


if __name__ == "__main__":
    if "--all" in sys.argv:
        run_all()
    else:
        question = " ".join(sys.argv[1:]) or "What is the maximum number of remote work days per week?"
        out = answer_question(question)
        print(out["answer"])
        print(f"\n[{out['model']} | thinking={out['thinking']} | {out['finish_reason']} | "
              f"prompt_tokens={out['prompt_tokens']} | {out['generation_time_ms']} ms]")
        for c in out["retrieval"]["retrieved_chunks"]:
            print(f"  - {c['source']} p{c['page']} score={c['relevance_score']}")
