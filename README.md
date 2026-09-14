# Document Q&A (RAG)

A small Retrieval-Augmented Generation project: upload a PDF or text file,
ask questions about it, get answers grounded in the actual document content.

## Pipeline

```
Document (PDF/txt)
      ↓  ingest.py
Extracted text → overlapping chunks
      ↓  vector_store.py
Local embeddings (sentence-transformers) → FAISS index
      ↓  generate.py
Question → embed → retrieve top-k chunks → LLM (Gemini or local Ollama) → answer
      ↓  app.py
Flask API + simple web UI wrapping all of the above
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill it in:
   ```bash
   cp .env.example .env
   ```
   - If using **Gemini** (default): get a free API key at
     https://aistudio.google.com/apikey and paste it into `.env`.
   - If using **Ollama** (fully local, zero API key): install Ollama from
     https://ollama.com, run `ollama pull llama3.2`, and set
     `LLM_BACKEND=ollama` in `.env`.

## Run it — CLI first

Get the core pipeline working before touching Flask at all:

```bash
python rag.py path/to/your_document.pdf
```

This ingests the file, builds the index, and drops you into a question loop
in the terminal. First run downloads the local embedding model (~90MB) —
that's a one-time cost, cached afterward.

## Run it — as a web app

```bash
python app.py
```

Then open http://localhost:5000, upload a document, and ask questions.

## Run it — in Docker

```bash
docker build -t rag-app .
docker run -p 5000:5000 --env-file .env rag-app
```

## Design choices worth being able to explain

- **Chunking is word-count based (500 words, 50-word overlap)**, not
  token- or sentence-based. Simple to reason about; a production system
  would usually chunk on tokens or sentence boundaries instead.
- **Embeddings run locally** (`sentence-transformers`, `all-MiniLM-L6-v2`)
  rather than through an API — zero cost, zero rate limits, and it proves
  you understand what an embedding actually is rather than just calling
  an endpoint.
- **FAISS `IndexFlatIP`** with normalized vectors gives cosine similarity
  search. Flat index means brute-force search — fine at this scale (fast
  up to tens of thousands of chunks), but not what you'd reach for at
  millions of vectors (that's where an approximate index like IVF or HNSW
  would come in — a good "what would you change at scale" answer).
- **The index lives in memory, one document at a time** — a deliberate
  scope cut for a demo, not an oversight. Be ready to say what you'd add
  for multi-document, persistent use: a persisted FAISS index or a
  managed vector DB (e.g. Postgres + pgvector), plus per-document
  metadata so retrieval can be scoped to the right file.

## Future enhancements

- Persist the vector index (Postgres + `pgvector`) instead of in-memory
- Support multiple documents with metadata filtering
- Swap word-count chunking for token-aware chunking
- Deploy to Cloud Run / App Runner (same pattern as the other projects)
