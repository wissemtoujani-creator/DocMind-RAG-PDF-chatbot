# DocMind — RAG PDF Chatbot

> Upload any PDF and query it in natural language. Answers are grounded in the document and carry a citation for every claim.

## What is DocMind?

DocMind is a PDF question-answering system built on the RAG (Retrieval-Augmented
Generation) architecture. You upload a PDF, the system extracts, chunks, and
embeds it, and you then ask questions in natural language through a clean web
interface. The system answers strictly from the document's content and always
tells you which page a claim came from.

## Features

- **PDF upload** — drag and drop, up to 25 MB, with magic-byte and page-text validation
- **Semantic search** — queries matched by meaning, not just keywords
- **Conversational memory** — the last 5 turns per conversation, so follow-up questions resolve pronouns correctly
- **Source citations** — every answer includes numbered citations with the page, a relevance score, and a passage preview
- **Honest refusal** — if the answer is not in the document, it says so instead of hallucinating
- **Multiple documents** — index several PDFs at once and query them independently
- **Operational baseline** — structured JSON logs, per-request correlation ids, liveness/readiness probes, security headers

## Tech stack

| Layer            | Technology                                                        |
| ---------------- | ----------------------------------------------------------------- |
| Backend          | FastAPI + Uvicorn, versioned `/api/v1`                            |
| Configuration    | pydantic-settings, `.env`                                          |
| RAG orchestration | Custom pipeline (LangChain splitters + provider clients)          |
| Embeddings       | HuggingFace `all-MiniLM-L6-v2`, 384-dim, CPU                      |
| Vector store     | ChromaDB (HNSW, local disk); pgvector planned                    |
| LLM              | Groq `qwen/qwen3.8-27b`                                           |
| PDF loading      | `pypdf`                                                           |
| Observability    | structlog, JSON in production                                      |
| Testing          | pytest, 87% coverage, no network required                         |
| Frontend         | Pure HTML / CSS / vanilla JS                                       |

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/wissemtoujani-creator/DocMind-RAG-PDF-chatbot.git
cd DocMind-RAG-PDF-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements-dev.txt    # or requirements.txt for runtime only
```

### 4. Add your Groq API key

```bash
cp .env.example .env
```

Then set `GROQ_API_KEY` in `.env`. Get a free key at
[console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
uvicorn app.main:app --reload --port 8000
```

Or `make dev` on Windows/macOS. Then open **http://localhost:8000**, or
**http://localhost:8000/docs** for the interactive OpenAPI reference.

> **Note:** the first launch downloads the `all-MiniLM-L6-v2` embedding model
> (~90 MB). This happens once and is cached automatically.

## How it works

```
INGESTION (once per PDF)
Upload → validate (size, %PDF- magic, extractable text) → per-page extraction
→ recursive chunking (512 chars / 64 overlap, page provenance + offsets)
→ MiniLM-L6-v2 embeddings → ChromaDB HNSW index → document marked ready

QUERY (every question)
Question + last 5 turns of this conversation → cosine search over that
document's chunks only (top-5) → numbered context block → prompt
→ Groq LLM → answer + numbered citations
```

The prompt instructs the model to cite passage numbers and to reply with a
fixed refusal string when the passages do not contain the answer. Retrieved
passages are numbered `[1]`, `[2]`, … in the prompt and the returned citations
carry the same markers, so a citation in the text maps to a passage you can
inspect.

## API

All endpoints live under `/api/v1`.

| Method     | Path                | Purpose                                          |
| ---------- | ------------------- | ------------------------------------------------ |
| `GET`      | `/health`           | Liveness. Answers while the process is up.       |
| `GET`      | `/health/ready`     | Readiness. Touches the vector store.              |
| `POST`     | `/documents`        | Upload a PDF; it is extracted, chunked, indexed. |
| `GET`      | `/documents`        | List indexed documents.                           |
| `GET`      | `/documents/{id}`   | Fetch one document.                               |
| `DELETE`   | `/documents/{id}`   | Delete a document and its embeddings.             |
| `POST`     | `/chat`             | Ask a grounded question.                          |
| `POST`     | `/chat/reset`       | Clear a conversation's history.                   |
| `GET`      | `/stats`            | Corpus statistics.                                |

Every response carries an `X-Request-ID` header, echoed in every log line
produced while handling that request — supply your own to trace across services.

Errors use a single envelope:

```json
{ "error": { "code": "unsupported_file_type", "message": "notes.txt is not a PDF.", "details": {} } }
```

## Configuration

Everything is environment-driven; `.env.example` lists all options with
defaults. The values you are most likely to change:

| Variable               | Default            | Notes                                                 |
| ---------------------- | ------------------ | ----------------------------------------------------- |
| `GROQ_API_KEY`         | _unset_            | Required for `/chat`.                                 |
| `LLM_MODEL`            | `qwen/qwen3.8-27b` | `openai/gpt-oss-120b` is slower to answer.            |
| `EMBEDDING_MODEL`      | `all-MiniLM-L6-v2` | 384-dim, runs on CPU.                                 |
| `CHUNK_SIZE`           | `512`              | Characters per chunk.                                 |
| `CHUNK_OVERLAP`        | `64`               | Must be `< CHUNK_SIZE`.                               |
| `TOP_K`                | `5`                | Passages retrieved per question.                      |
| `MAX_UPLOAD_BYTES`     | `26214400`         | 25 MB.                                                |
| `LOG_FORMAT`           | `console`          | `json` in production.                                 |
| `VECTOR_STORE_BACKEND` | `chroma`           | `chroma` is the only working backend; `pgvector` is validated but **not implemented yet**. |

## Architecture

```
app/
  main.py         application factory, lifespan, error handlers
  core/           config, logging, errors, context, middleware
  api/v1/routes/  health, documents, chat, stats
  repositories/   document + conversation persistence
  services/
    rag/          pipeline, chunking, loader, prompts, providers
    vectordb/     VectorStore protocol + backend adapters
  static/         frontend
```

Dependencies point strictly inward: `api` → `services` → `repositories` and the
`VectorStore` protocol. Nothing in `services` imports from `api`, which is what
lets the pipeline be tested directly with fakes and lets the vector store, LLM,
and embedder be swapped without touching the layers above.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the request lifecycle, the
reasoning behind each seam, and the data model. A diagram of the layout and
dependency direction is in [docs/STRUCTURE.md](docs/STRUCTURE.md). Planned work
is tracked in [docs/ROADMAP.md](docs/ROADMAP.md).

## Development

```bash
make test       # unit + HTTP contract tests, no network needed
make test-cov   # with a coverage report
make lint       # ruff check + format check
make smoke      # end-to-end against the live providers
```

The test suite never touches the network — the embedder, generator, and vector
store are faked in `tests/conftest.py`. `make smoke` is the exception and needs
a real `GROQ_API_KEY`.

## Known limitations

Tracked deliberately rather than hidden:

- **Single-process state.** Documents and conversations live in memory, so state
  is lost on restart and not shared across workers. Postgres addresses both.
- **No authentication.** Every endpoint is public.
- **Pure vector retrieval.** No hybrid keyword search or reranking, so rare
  exact terms (part numbers, error codes) retrieve poorly.
- **No score threshold.** On small documents, weak passages are still returned
  and shown as citations; the model usually ignores them, but they are noise.
- **Synchronous ingestion.** A large PDF is parsed and embedded inside the
  request. This should become a background job with a status poll.

## Authors

- **Wissem Toujani**
- **Hedy Ben Hamadou**

Final Year Project (PFA) — National Institute of Applied Sciences, Tunis — 2026
