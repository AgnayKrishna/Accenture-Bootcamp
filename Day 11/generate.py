"""Generation: retrieved chunks + question -> answer from an OpenAI-compatible LLM."""
import time

from openai import OpenAI

import config


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c['source']}, page {c['page']})\n{c['content']}")
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    context = format_context(chunks)
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
    start = time.time()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
        temperature=config.TEMPERATURE,
        reasoning_effort=config.REASONING_EFFORT,
        **{config.TOKEN_LIMIT_PARAM: config.MAX_TOKENS},
    )
    elapsed_ms = round((time.time() - start) * 1000)
    choice = response.choices[0]
    usage = response.usage
    answer = (choice.message.content or "").strip()

    if not answer:
        # an empty answer saved silently is the most expensive bug of this capstone
        print(f"WARNING: EMPTY ANSWER for {query!r} (finish_reason={choice.finish_reason}). "
              "Thinking may have used the whole token cap.")

    return {
        "query": query,
        "answer": answer,
        "context_used": context,  # exactly what the LLM saw
        "model": config.LLM_MODEL,
        "thinking": config.REASONING_EFFORT != "none",
        "finish_reason": choice.finish_reason,
        "generation_time_ms": elapsed_ms,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "tokens_generated": usage.completion_tokens if usage else None,
    }
