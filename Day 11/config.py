"""All configuration in one place.

Swapping from local to cloud = change only the environment variables
(LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, TOKEN_LIMIT_PARAM). Nothing else in
the code needs to change. Embeddings stay local, so chroma_db/ serves both.
"""
import os

# ---- LLM (OpenAI-compatible endpoint; Ollama by default) ----
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:4b-q4_K_M")

# ---- Embeddings (local) ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "ollama")
# nomic-embed-text is trained with task prefixes; using them is the documented way.
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "search_document: ")
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: ")

# ---- Documents and vector store ----
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "bootcamp_docs")

# ---- Chunking (RecursiveCharacterTextSplitter: the unit is CHARACTERS) ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))      # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # characters
# "layout" keeps table rows on one line; "plain" is pypdf's default extraction.
PDF_EXTRACTION_MODE = os.getenv("PDF_EXTRACTION_MODE", "layout")

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "5"))

# ---- Generation ----
SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided context.
Use ONLY the context below to answer. If the answer is not in the context, say "I cannot find this information in the provided documents."
When the context contains figures from different documents that disagree, say so and name the documents.

Context:
{context}
"""
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
# Qwen3.5 models think by default; thinking and answer share the token cap.
# "none" goes into EVERY chat call.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "none")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
# Ollama reads only max_tokens; the Day 12 cloud model wants max_completion_tokens.
TOKEN_LIMIT_PARAM = os.getenv("TOKEN_LIMIT_PARAM", "max_tokens")

# ---- Files ----
TEST_QUESTIONS_FILE = "test_questions.json"
PIPELINE_OUTPUTS_FILE = os.getenv("PIPELINE_OUTPUTS_FILE", "pipeline_outputs.json")
BENCHMARK_RESULTS_FILE = "benchmark_results.json"
