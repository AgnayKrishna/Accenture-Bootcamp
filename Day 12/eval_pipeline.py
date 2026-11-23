"""Runs five DeepEval RAG metrics over pipeline_outputs_<label>.json and writes eval_results_<label>.json.

    uv run python eval_pipeline.py local
    uv run python eval_pipeline.py cloud

Needs JUDGE_BASE_URL (your Foundry URL ending in /openai/v1/) and AZURE_OPENAI_API_KEY in the shell.
Judge gpt-5.4-mini; JUDGE_MODEL and EVAL_CONCURRENCY override the defaults.
Metrics that fail with a connection error or timeout are re-run up to three times.
"""

import json
import os
import sys

# The whole class shares one judge deployment (250 requests per minute): wait and retry on 429
# instead of recording an error. DeepEval reads these when it is imported.
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "6")
os.environ.setdefault("DEEPEVAL_RETRY_CAP_SECONDS", "30")
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate import AsyncConfig, DisplayConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric,
    ContextualRelevancyMetric, FaithfulnessMetric,
)
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")


def make_judge():
    # gpt-4.1 models reject reasoning_effort; for gpt-5.x DeepEval already leaves out temperature
    extra = {} if JUDGE_MODEL.startswith("gpt-4") else {"reasoning_effort": "low"}
    return OpenAIModel(
        model=JUDGE_MODEL,
        base_url=os.environ["JUDGE_BASE_URL"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        generation_kwargs=extra,
    )


def main(label: str):
    with open(f"pipeline_outputs_{label}.json", encoding="utf-8") as f:
        rows = json.load(f)
    test_cases = [
        LLMTestCase(name=r["id"], input=r["question"], actual_output=r["actual_answer"],
                    expected_output=r["expected_answer"], retrieval_context=r["retrieved_context"])
        for r in rows
    ]
    judge = make_judge()
    # By default Faithfulness only fails claims that contradict the context; a claim the context
    # never mentions passes. The flag counts those against the score too.
    factories = {
        "Faithfulness": lambda: FaithfulnessMetric(threshold=0.7, model=judge, penalize_ambiguous_claims=True),
        "Answer Relevancy": lambda: AnswerRelevancyMetric(threshold=0.7, model=judge),
        "Contextual Relevancy": lambda: ContextualRelevancyMetric(threshold=0.7, model=judge),
        "Contextual Recall": lambda: ContextualRecallMetric(threshold=0.7, model=judge),
        "Contextual Precision": lambda: ContextualPrecisionMetric(threshold=0.7, model=judge),
    }
    by_id = {r["id"]: r for r in rows}
    cases = {tc.name: tc for tc in test_cases}

    def run(cases_to_run, metric_names, concurrency):
        result = evaluate(cases_to_run, [factories[n]() for n in metric_names],
                          async_config=AsyncConfig(max_concurrent=concurrency),
                          error_config=ErrorConfig(ignore_errors=True),
                          display_config=DisplayConfig(print_results=False))
        return {tr.name: {m.name: {"score": m.score, "success": m.success, "reason": m.reason, "error": m.error}
                          for m in tr.metrics_data or []} for tr in result.test_results}

    scores = run(test_cases, list(factories), int(os.environ.get("EVAL_CONCURRENCY", "2")))

    # Connection errors and timeouts come from the judge endpoint, not from the answers.
    # Re-run only the metrics that errored (max 3 more rounds) and keep everything else as is.
    for attempt in range(1, 4):
        failed = {(q, n) for q, ms in scores.items() for n, m in ms.items() if m["error"]}
        if not failed:
            break
        print(f"Retry round {attempt}: {len(failed)} metric(s) errored, running them again")
        for name in sorted({n for _, n in failed}):
            ids = sorted({q for q, n in failed if n == name})
            again = run([cases[q] for q in ids], [name], 1)
            for q in ids:
                if q in again and name in again[q]:
                    scores[q][name] = again[q][name]

    out = [{
        "id": r["id"],
        "question": r["question"],
        "model": r["model"],
        "generation_time_ms": r["generation_time_ms"],
        "judge": judge.get_model_name(),
        "metrics": scores.get(r["id"], {}),
    } for r in rows]
    with open(f"eval_results_{label}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    errors = sum(1 for r in out for m in r["metrics"].values() if m["error"])
    print(f"Saved {len(out)} rows to eval_results_{label}.json ({errors} metric errors)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "local")