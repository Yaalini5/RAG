# Document Q&A (RAG)

A simple Retrieval-Augmented Generation app that lets you upload a PDF or text file, retrieve the most relevant chunks, and ask questions about the document with AI-grounded answers.

## Features

- Upload a PDF or `.txt` file
- Extract and chunk document text
- Embed chunks locally with a sentence-transformers model
- Search using FAISS similarity
- Generate answers using either Gemini or Ollama
- Serve a Flask web interface for interactive use
- Ready for deployment with Gunicorn and Docker

## Architecture

```text
Document (PDF/txt)
  ↓
ingest.py
  ↓
chunk_text() → overlapping word chunks
  ↓
vector_store.py
  ↓
SentenceTransformer embeddings → FAISS index
  ↓
generate.py
  ↓
question + retrieved chunks → Gemini or Ollama answer
  ↓
Flask app served via Gunicorn
```

## Local setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on `.env.example`:

```bash
copy .env.example .env
```

Then set values such as:

```env
LLM_BACKEND=gemini
GEMINI_API_KEY=your_key_here
PORT=5000
UPLOAD_FOLDER=uploads
```

If you want a fully local model instead, use:

```env
LLM_BACKEND=ollama
```

Make sure Ollama is installed and the model is downloaded:

```bash
ollama pull llama3.2
```

## Run it locally

### Web app

```bash
python app.py
```

Open `http://localhost:5000`

### CLI check

```bash
python rag.py path/to/your_document.pdf
```

This builds the vector index and lets you ask questions from the terminal.

## Production-style deployment

This project is configured to run behind Gunicorn and to read the runtime port from the environment, which makes it compatible with Google Cloud Run and Render.

### Google Cloud Run deployment

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/rag-app

gcloud run deploy rag-app \
  --image gcr.io/YOUR_PROJECT_ID/rag-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 5000 \
  --memory 2Gi \
  --min-instances 1 \
  --max-instances 1 \
  --set-env-vars LLM_BACKEND=gemini,GEMINI_API_KEY=your_key_here,PORT=5000,UPLOAD_FOLDER=uploads
```

The one-instance cap is intentional here because the FAISS index is kept in memory and is not persisted across cold starts or new instances. For this demo/interview setup, a single always-on instance is the most reliable approach.

### Render deployment

1. Push this repo to GitHub.
2. Create a new Web Service on Render.
3. Connect the repo and use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Add environment variables in the Render dashboard:
   - `LLM_BACKEND=gemini`
   - `GEMINI_API_KEY=your_key_here`
   - `PORT=10000` (Render sets this automatically; keep it as provided by the platform)

A `render.yaml` file is included at the repo root for one-click Render config.

### Docker deployment

```bash
docker build -t rag-app .
docker run -p 5000:5000 --env-file .env rag-app
```

## Project files

- `app.py` — Flask web app
- `rag.py` — CLI entry point
- `ingest.py` — PDF/text extraction and chunking
- `vector_store.py` — embeddings and FAISS retrieval
- `generate.py` — Gemini/Ollama answer generation
- `Dockerfile` — container image for deployment
- `render.yaml` — Render deployment config
- `.env.example` — example environment variables

## Notes

- Chunking is intentionally word-based for simplicity.
- The FAISS index is in-memory, which is fine for a demo but not for large multi-document production workloads.
- A production version would usually add persistent storage, metadata filtering, and a managed vector database.

## Future improvements

- Persist the vector index in Postgres with pgvector
- Support multiple uploaded documents
- Improve chunking using token-aware boundaries
- Add user authentication and document history
- Deploy with a managed AI service and persistent database
