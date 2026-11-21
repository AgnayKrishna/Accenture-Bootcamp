# AGENTS.md

Instructions for any AI coding assistant working in this repo.

## Rules of this project (from the Day 11 brief)

- **Local models only.** The model inside the pipeline runs on this machine through Ollama.
  No cloud-tagged Ollama models (no `cloud` in the tag). Check with `ollama ps`.
- **Numbers come from files.** Every number in `decision_log.md` and `README.md` must come from a
  file the code wrote: `benchmark_results.json`, `pipeline_outputs.json`,
  `retrieval_experiments.json`. The assistant may write the script that measures. It must not
  write the number.
- **`decision_log.md` is written by hand** by the student. The assistant does not write or edit it.
- **Keep `.env` away from the assistant.** Do not read, print or commit it.
- **One step at a time, one commit per step:** ingest, retrieve, generate, benchmark.
- **Every chat call** passes `reasoning_effort=config.REASONING_EFFORT` (default `"none"`) and the
  token cap as `**{config.TOKEN_LIMIT_PARAM: config.MAX_TOKENS}`.
- **Always pass embeddings to Chroma** (`embeddings=` on add, `query_embeddings=` on query).
- **Chunk size is in characters** (RecursiveCharacterTextSplitter). State the unit whenever it is reported.

## How to run (from the project folder)

```bash
uv sync                                          # install dependencies
ollama pull qwen3.5:4b-q4_K_M && ollama pull qwen3.5:2b-q4_K_M && ollama pull nomic-embed-text
uv run python ingest.py                          # rebuilds chroma_db/ from documents/
uv run python retrieve.py "your question"        # check retrieval alone
uv run python pipeline.py "your question"        # one question end to end
uv run python pipeline.py --all                  # all test questions -> pipeline_outputs.json
uv run python benchmark.py                       # Exercise 1 -> benchmark_results.json
uv run python experiment_retrieval.py            # chunk size / top-K comparison -> retrieval_experiments.json
```

## Checks to run after each step

1. `ollama ps` shows the model and CONTEXT 8192.
2. `ingest.py` prints a chunk count and `collection.count()` equal to it (no doubling on re-run).
3. `retrieve.py` returns chunks from the expected document and page.
4. `pipeline.py` returns a non-empty answer with `finish_reason` = `stop`.
5. After `pipeline.py --all`, open `pipeline_outputs.json`, read each answer against
   `expected_answer`, and fill in `passed` and `rating_note` by hand. Re-running overwrites the file, so rate last.
