# Decision Log: Local RAG Pipeline

<!--
This file is yours to write, by hand, from what you saw (rule 5 of the brief).
Replace every [bracket] with your own words and numbers. Delete these comments when done.
Every number needs a file name next to it, for example "(benchmark_results.json, 4B, factual)".
-->

## Machine
NAME                  ID              SIZE     PROCESSOR    CONTEXT
qwen3.5:4b-q4_K_M     d8b0f5e9760c    3.4 GB   100% CPU     8192


## Model Selection
- **Chosen LLM:qwen3.5:4b-q4_K_M
- **Thinking:off (the thinking column is False on every row in my result files).
- **Why:I manually assessed each response after running the identical six questions on the 4B and the 2B (benchmark_results.json). The 2B received a 2 on every six, while the 4B received a 4. Although the 2B is faster—roughly 7.0 tokens/s compared to 4.2 and 11.8 seconds per prompt compared to 19.6 seconds—I would prefer to wait an extra 8 seconds rather than receive an inferior response.
- **What I considered but rejected:qwen3.5:2b-q4_K_M, because of the lower ratings.

## Embedding Model
nomic-embed-text, 768 dimensions (printed by ingest.py). Everything stays local because it flows in Ollama adjacent to the LLM. Other embedding models were not compared by me.

## Parsing
Column gaps in pypdf in layout mode are changed to "|."

The plain mode on page 14 (SLA credits) of the product handbook runs a table row into a single line ("99.0% - 99.49% 5% — —"), making it impossible to determine which column a number belongs to. Under a header row, layout mode maintains it as "99.0% - 99.49% | 5% | — | —".

Layout did not win on that measure in my retrieval trial at 512 characters; plain performed about as well as layout (14 vs. 13 questions at top-5). I saved it for the tables that could be read.

## Chunking Strategy
- **Chunk size:200 overlap and 1500 characters (the splitter counts characters, not tokens). Ingest.py creates 142 pieces.

- **Why:I tested it (retrieval_experiments.json). The number is how many of the 17 answerable questions had all their evidence in the retrieved chunks:
My first default, 1000 / 150, was the weakest at top-5, so I changed it. 1500 / 200 only misses q17.

## Retrieval Configuration
- **Top-K:Going from top-3 to top-5 increased the score from 13 to 16 out of 17, and top-8 provided no further information, which would have just slowed down responses.
- **Context window:8192 tokens (ollama ps). They fit readily because prompts at K=5 ranged from 1030 to 2147 tokens, with an average of 1503 (pipeline_outputs.json).
- **Why:retrieval took about 2.2 s per question and generation about 104 s (retrieval_time_ms, generation_time_ms).
## Observations
- **What worked well:16 of 19 passed (`pipeline_outputs.json`), including both unanswerable questions (q18, q19) and the conflicting-source ones (q13, q14, q16), where it named both sources.
- **What failed:
q17 (retrieval):** it gave the 30-day rule but missed the 72-hour figure. All five chunks came from the security policy, and even top-8 doesn't find it.
  - q08 (generation):** the right table was retrieved, but the model invented a €79 price and a €33 saving instead of €20.
  - q09 (generation):** right accounts and €420K, but it opens by saying none were flagged.
- **Traps I hit:the context window had to be raised to 8192 first. Chunk size counts characters, so 512 splits tables. The model guesses numbers from partial tables (q08), so I checked the retrieved context for every failure.
- **Local vs cloud expectations:I expect cloud to be much faster and to make fewer slips like q08 and q09, but still fail q17, because the missing chunk never reaches the model.

## If I Had More Time / Better Hardware
Retrieve from each document separately and merge, to fix q17. A bigger K won't do it.
Do the price calculations in code instead of in the model (q08).
Compare another embedding model and a bigger LLM, and rate more than six prompts.
