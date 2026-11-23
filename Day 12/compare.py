"""Local vs cloud comparison. Reads the saved result files and prints markdown to paste into the report.

    uv run python compare.py

Reads: eval_results_local.json, eval_results_cloud.json, pipeline_outputs_local.json,
       pipeline_outputs_cloud.json, test_questions.json
Writes nothing. Redirect to a file if you want to keep it:  uv run python compare.py > compare_output.md
"""

import json
import statistics
import sys

REFUSAL = "cannot find this information"
JUDGE_NEEDS = ("faithfulness", "answer relevancy")  # metrics used for "the judge passes this answer"


def load(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Missing file: {path}. Run the earlier steps first.")


def mean_2se(values: list[float]) -> tuple[float, float]:
    """Mean and 2 x standard error (standard deviation divided by the square root of n)."""
    if not values:
        return float("nan"), float("nan")
    if len(values) < 2:
        return values[0], float("nan")
    return statistics.mean(values), 2 * statistics.stdev(values) / len(values) ** 0.5


def by_id(rows: list[dict]) -> dict:
    return {r["id"]: r for r in rows}


def metric_table(local_eval: dict, cloud_eval: dict) -> list[str]:
    names = sorted({m for r in local_eval.values() for m in r["metrics"]}
                   & {m for r in cloud_eval.values() for m in r["metrics"]})
    lines = ["| Metric | n | Local mean ± 2×SE | Cloud mean ± 2×SE | Cloud − local (same question) | Verdict |",
             "|---|---|---|---|---|---|"]
    for m in names:
        ids = [i for i in local_eval if i in cloud_eval
               and local_eval[i]["metrics"].get(m, {}).get("score") is not None
               and cloud_eval[i]["metrics"].get(m, {}).get("score") is not None]
        a = [local_eval[i]["metrics"][m]["score"] for i in ids]
        b = [cloud_eval[i]["metrics"][m]["score"] for i in ids]
        (ma, ea), (mb, eb) = mean_2se(a), mean_2se(b)
        md, ed = mean_2se([y - x for x, y in zip(a, b)])
        if len(ids) < 2:
            verdict = "not enough questions"
        elif md - ed <= 0 <= md + ed:
            verdict = "no clear difference"
        else:
            verdict = "cloud higher" if md > 0 else "local higher"
        lines.append(f"| {m} | {len(ids)} | {ma:.2f} ± {ea:.2f} | {mb:.2f} ± {eb:.2f} | "
                     f"{md:+.2f} ± {ed:.2f} | {verdict} |")
    return lines


def run_stats(rows: list[dict]) -> dict:
    times = [r["generation_time_ms"] for r in rows if r.get("generation_time_ms") is not None]
    tokens = [r["tokens_generated"] for r in rows if r.get("tokens_generated") is not None]
    empty = [r["id"] for r in rows if not (r.get("actual_answer") or "").strip()]
    cut = [r["id"] for r in rows if r.get("finish_reason") not in (None, "stop")]
    return {
        "n": len(rows),
        "median_s": statistics.median(times) / 1000 if times else float("nan"),
        "slowest_s": max(times) / 1000 if times else float("nan"),
        "slowest_id": max(rows, key=lambda r: r.get("generation_time_ms") or 0)["id"] if rows else "-",
        "avg_tokens": statistics.mean(tokens) if tokens else float("nan"),
        "empty": empty,
        "cut": cut,
        "model": rows[0]["model"] if rows else "-",
    }


def judge_passes(row: dict) -> bool | None:
    scored = [v for k, v in row["metrics"].items() if k.lower() in JUDGE_NEEDS]
    if not scored or any(v.get("success") is None for v in scored):
        return None
    return all(v["success"] for v in scored)


def main() -> None:
    le, ce = by_id(load("eval_results_local.json")), by_id(load("eval_results_cloud.json"))
    lp, cp = load("pipeline_outputs_local.json"), load("pipeline_outputs_cloud.json")
    questions = by_id(load("test_questions.json"))

    print("## Scores by metric (source: eval_results_local.json, eval_results_cloud.json)\n")
    print("\n".join(metric_table(le, ce)))
    print("\nn = questions with a score on both sides. ± is 2 × standard error. The verdict says "
          "\"no clear difference\" when the range of the same-question difference includes zero.")

    print("\n## Time, tokens and failures (source: pipeline_outputs_local.json, pipeline_outputs_cloud.json)\n")
    print("| | Local | Cloud |\n|---|---|---|")
    ls, cs = run_stats(lp), run_stats(cp)
    print(f"| Model | {ls['model']} | {cs['model']} |")
    print(f"| Questions | {ls['n']} | {cs['n']} |")
    print(f"| Median generation time | {ls['median_s']:.1f} s | {cs['median_s']:.1f} s |")
    print(f"| Slowest generation | {ls['slowest_s']:.1f} s ({ls['slowest_id']}) | {cs['slowest_s']:.1f} s ({cs['slowest_id']}) |")
    print(f"| Average completion tokens | {ls['avg_tokens']:.0f} | {cs['avg_tokens']:.0f} |")
    print(f"| Empty answers | {len(ls['empty'])} {ls['empty'] or ''} | {len(cs['empty'])} {cs['empty'] or ''} |")
    print(f"| finish_reason not \"stop\" | {len(ls['cut'])} {ls['cut'] or ''} | {len(cs['cut'])} {cs['cut'] or ''} |")
    print("\nCompletion tokens are `tokens_generated` from the API usage block. Reasoning tokens are not "
          "saved separately by the pipeline; the cloud runs with thinking off, so none are expected.")

    unanswerable = [i for i, q in questions.items() if q.get("type") == "unanswerable"]
    print(f"\n## Unanswerable questions: refusal sentence present (string check, {len(unanswerable)} questions)\n")
    for label, rows in (("Local", lp), ("Cloud", cp)):
        rid = by_id(rows)
        hits = [i for i in unanswerable if i in rid and REFUSAL in rid[i]["actual_answer"].lower()]
        print(f"- {label}: {len(hits)} of {len(unanswerable)} refused ({', '.join(hits) or 'none'})")

    print("\n## Judge vs my own `passed` ratings (local run)\n")
    print("Judge passes an answer when Faithfulness and Answer Relevancy both succeed (threshold 0.7).\n")
    hand = {r["id"]: r.get("passed") for r in lp}
    cases = {"both pass": [], "both fail": [], "judge passes, I failed": [], "judge fails, I passed": []}
    unrated = []
    for i, row in le.items():
        j, h = judge_passes(row), hand.get(i)
        if j is None or h is None:
            unrated.append(i)
        elif j and h:
            cases["both pass"].append(i)
        elif not j and not h:
            cases["both fail"].append(i)
        elif j and not h:
            cases["judge passes, I failed"].append(i)
        else:
            cases["judge fails, I passed"].append(i)
    for k, ids in cases.items():
        print(f"- {k}: {len(ids)} {ids or ''}")
    if unrated:
        print(f"- not comparable (missing score or rating): {unrated}")


if __name__ == "__main__":
    # Write compare_output.md ourselves in UTF-8 (the Windows console and PowerShell's ">" would
    # mangle the ± and − characters), then show the same text on screen.
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    text = buf.getvalue()
    with open("compare_output.md", "w", encoding="utf-8") as f:
        f.write(text)
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    print("\nSaved compare_output.md")