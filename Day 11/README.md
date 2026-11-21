# Local Document Q&A (RAG) with Ollama

A small question-answering system over five internal PDFs that runs completely on the local
machine: local LLM and local embeddings through Ollama, local ChromaDB for vector search.
No document text is sent to a cloud service.

**AI assistant used while writing the code:** Claude (Sonnet 5.5, claude.ai). The model inside
the pipeline is local (`qwen3.5:4b-q4_K_M` through Ollama, no cloud tags). The decision log was written by hand.

## Setup

1. Install Ollama from https://ollama.com/download (native Windows installer) and pull the models:

   ```bash
   ollama pull qwen3.5:4b-q4_K_M
   ollama pull qwen3.5:2b-q4_K_M
   ollama pull nomic-embed-text
   ```

2. Set the Ollama context window to 8192 (user environment variable `OLLAMA_CONTEXT_LENGTH=8192`,
   then restart Ollama) and check with `ollama ps` that CONTEXT reads 8192.
3. Install dependencies (Python 3.12, [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync
   ```

## Run

```bash
uv run python ingest.py            # rebuilds chroma_db/ (the one command that recreates the vector DB)
uv run python pipeline.py "What is the PTO carryover limit?"   # one question
uv run python pipeline.py --all    # all questions in test_questions.json -> pipeline_outputs.json
uv run python benchmark.py         # model comparison -> benchmark_results.json
uv run python experiment_retrieval.py   # retrieval-only comparison of chunk sizes and top-K -> retrieval_experiments.json
```

## Project layout

| File | Purpose |
|---|---|
| `config.py` | All settings. Provider swap = environment variables only. |
| `ingest.py` | PDF text extraction, chunking, local embeddings, ChromaDB storage. |
| `retrieve.py` | Embed the query, search ChromaDB, return chunks with source, page and score. |
| `generate.py` | Build the prompt, call the LLM, return answer plus the exact context it saw. |
| `pipeline.py` | Query, retrieve, generate. `--all` runs the whole test set. |
| `benchmark.py` | Exercise 1, model comparison on 6 prompts. |
| `experiment_retrieval.py` | Retrieval-only experiment (no LLM), used to pick chunking and top-K. |
| `test_questions.json` | 19 questions with expected answers and evidence phrases. |
| `benchmark_results.json`, `pipeline_outputs.json`, `retrieval_experiments.json` | Result files written by the code. |
| `decision_log.md` | Decisions and reasoning. |

## Swapping to a cloud model (Day 12)

Only environment variables change, no code (PowerShell):

```powershell
$env:LLM_BASE_URL = "https://<resource>.openai.azure.com/openai/v1/"
$env:LLM_API_KEY = $env:AZURE_OPENAI_API_KEY
$env:LLM_MODEL = "gpt-5.6-luna"
$env:TOKEN_LIMIT_PARAM = "max_completion_tokens"
```

Embeddings stay local, so the same `chroma_db/` serves both runs. To write the cloud results to a
separate file, also set `$env:PIPELINE_OUTPUTS_FILE = "pipeline_outputs_cloud.json"`.

## Results

See `decision_log.md`. Every number there comes from `benchmark_results.json`,
`pipeline_outputs.json` or `retrieval_experiments.json`.
