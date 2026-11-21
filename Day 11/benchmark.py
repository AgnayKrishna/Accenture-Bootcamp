"""Exercise 1: compare local models on speed and quality.

Run:  uv run python benchmark.py
Then open benchmark_results.json and fill in quality_rating (1-5) and rating_note BY HAND.
Re-running overwrites the file, so rate last.
"""
import json
import time

from openai import OpenAI

MODELS = ["qwen3.5:4b-q4_K_M", "qwen3.5:2b-q4_K_M"]
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"
REASONING_EFFORT = "none"  # same for every model, so the comparison is fair
OUTPUT_FILE = "benchmark_results.json"

PARAGRAPH = (
    "Remote work at the company is allowed up to three days a week once an employee has finished "
    "the six-month probationary period. Before that, employees may work remotely one day a week. "
    "Every request goes through the HR portal and the manager answers within five business days. "
    "Remote staff must be reachable during the core hours of 10:00 to 15:00 CET and use the "
    "company VPN when they open internal systems."
)

PROMPTS = [
    {"category": "factual",
     "prompt": "What causes tides on Earth? Answer in 2-3 sentences."},
    {"category": "reasoning",
     "prompt": "If a train travels 120km in 1.5 hours, and then 80km in 1 hour, what is the average speed for the entire journey?"},
    {"category": "summarization",
     "prompt": "Summarize this paragraph in one sentence:\n\n" + PARAGRAPH},
    {"category": "structured",
     "prompt": 'List 3 European capitals. Respond ONLY with valid JSON: [{"city": ..., "country": ...}]'},
    {"category": "code",
     "prompt": "Write a Python function is_palindrome(s) that ignores case and spaces. Return only the code."},
    {"category": "instruction_following",
     "prompt": "Write exactly three bullet points about why backups matter. Each bullet must start with the word 'Because' and be under 12 words. No other text."},
]


def run_benchmark(model: str, prompt_item: dict) -> dict:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_item["prompt"]}],
        temperature=0.7,
        reasoning_effort=REASONING_EFFORT,
        max_tokens=1024,
    )
    elapsed = time.time() - start
    choice = response.choices[0]
    usage = response.usage
    text = (choice.message.content or "").strip()
    if not text:
        print(f"  WARNING: EMPTY RESPONSE from {model} on '{prompt_item['category']}' "
              f"(finish_reason={choice.finish_reason})")
    return {
        "model": model,
        "category": prompt_item["category"],
        "prompt": prompt_item["prompt"],
        "response": text,
        "thinking": REASONING_EFFORT != "none",
        "finish_reason": choice.finish_reason,
        "time_seconds": round(elapsed, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "tokens_per_second": round(usage.completion_tokens / elapsed, 1)
        if usage and usage.completion_tokens else None,
        "quality_rating": None,
        "rating_note": "",
    }


def main() -> None:
    results = []
    for model in MODELS:
        # a short warm-up call so model loading time does not pollute the first prompt
        print(f"Warming up {model} ...")
        OpenAI(base_url=BASE_URL, api_key=API_KEY).chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Hi"}],
            reasoning_effort=REASONING_EFFORT, max_tokens=5)
        for item in PROMPTS:
            print(f"{model} | {item['category']}")
            row = run_benchmark(model, item)
            print(f"  {row['finish_reason']}, {row['time_seconds']}s, {row['tokens_per_second']} tok/s")
            results.append(row)
        print(f"Tip: run `ollama stop {model}` before the next model if RAM is tight.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} rows to {OUTPUT_FILE}. Now rate each response by hand.")


if __name__ == "__main__":
    main()
