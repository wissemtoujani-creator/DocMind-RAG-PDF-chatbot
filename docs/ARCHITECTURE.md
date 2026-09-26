# Architecture

## Why the seams are where they are

The original version of this service was two files: `app.py` and
`rag_pipeline.py`, with a single `RAGPipeline` instance created at import time
as a module global. That design has a specific failure mode — the pipeline
holds the current document, the current chunks, the current vector store, and
the current chat history, so **the whole server can serve exactly one document
to exactly one user at a time**. Uploading a second PDF destroyed the first
user's context, and every user's turns appeared in every other user's history.

The refactor that produced the current layout removes state from the pipeline
rather than merely tidying it. State now lives in two places:

- **Vectors and passages** belong to the vector store, keyed by `document_id`.
- **Document metadata and conversation history** belong to repositories.

`RAGPipeline` is now stateless: it takes a `document_id` on every call. That is
what makes concurrent, multi-document operation possible, and it is a
prerequisite for authentication, where documents must additionally be scoped to
an owner.

## Layering

```
            ┌──────────────────────────────────────┐
  HTTP ───▶ │ api/            routing, schemas       │  knows about FastAPI
            ├──────────────────────────────────────┤
            │ services/rag/   pipeline, chunking,    │  knows about RAG
            │                loader, prompts         │  knows nothing about HTTP
            ├──────────────────────────────────────┤
            │ repositories/   documents, chat history│
            │ services/vectordb/   VectorStore protocol
            └──────────────────────────────────────┘
```

Imports point inward only. `services/` never imports from `api/`, and the
pipeline has no idea FastAPI exists. This is what lets the test suite exercise
the pipeline directly with fakes, and what will let the evaluation harness
later run the same code path without an HTTP server.

## Ports and adapters

Three boundaries exist specifically so the expensive or network-dependent parts
can be replaced:

| Protocol                | Production adapter          | Test / future adapter        |
| ----------------------- | --------------------------- | ---------------------------- |
| `VectorStore`           | `ChromaVectorStore`, later `PgVectorStore` | `InMemoryVectorStore` |
| `Embedder`              | `SentenceTransformerEmbedder` | `FakeEmbedder` (deterministic hashing) |
| `AnswerGenerator`       | `GroqGenerator`             | `FakeGenerator`              |
| `DocumentRepository`    | `InMemoryDocumentRepository` | `PostgresDocumentRepository` |
| `Chunker`               | `RecursiveTextChunker`      | `SectionChunker`, later      |

`build_vector_store` selects an implementation from `VECTOR_STORE_BACKEND`, so
the pgvector migration is a configuration change plus a new adapter, not a
rewrite.

## Request lifecycle

`POST /api/v1/documents`

1. `BodySizeLimitMiddleware` — bounds JSON bodies. Uploads are bounded
   separately in the route, where the file is already streaming to disk.
2. `SecurityHeadersMiddleware` — adds `nosniff`, `DENY`, `no-referrer`.
3. `CORSMiddleware`
4. `RequestContextMiddleware` — assigns or validates `X-Request-ID`, binds it to
   a `contextvar`, times the request, and logs the outcome.
5. Route: validate name and size → stream to a staging directory → verify the
   `%PDF-` magic bytes → hash the content → create a `DocumentRecord` in
   `indexing` state.
6. `pipeline.ingest` runs in a worker thread (`asyncio.to_thread`) because
   parsing and embedding are blocking. **Inside it:** extract pages → chunk →
   embed → write vectors.
7. The record is updated to `ready`; the staging directory is removed in a
   `finally` so a failure cannot leak a file.
8. Any exception deletes the half-created record, so a failed upload never
   appears in listings.

`POST /api/v1/chat`

1. Middleware chain as above.
2. Resolve the document: the requested `document_id`, or the most recent `ready`
   one. This is what keeps the old single-document client working unchanged.
3. Load the conversation's last N turns.
4. `pipeline.answer` in a worker thread: retrieve top-k **scoped to that
   document** → render a numbered context block → generate → build citations
   numbered to match the prompt.
5. Append the exchange to the conversation and return it.

## Why pure ASGI middleware

`RequestContextMiddleware` is written as raw ASGI rather than Starlette's
`BaseHTTPMiddleware`. `BaseHTTPMiddleware` runs the downstream application in a
separate task, which breaks `contextvars` propagation — and `contextvars` is
exactly how the `request_id` reaches the log formatter. The symptom would be
`request_id` on the entry log line and missing from every line logged during
handler execution, which is worse than having no correlation id at all.

## Error handling

Services raise subclasses of `AppError` carrying `status_code`, a stable
machine-readable `code`, and optional `details`. Handlers in `main.py` render
one envelope. A catch-all `Exception` handler logs the traceback against the
request id and returns a generic message — internals are never sent to a
client, and the user gets an id they can quote.

Provider failures are translated at the boundary in `generator._translate`, so
an upstream 429 becomes a `429` with `Retry-After` rather than a `500`.

## Logging

One structlog pipeline serves application logs *and* third-party libraries
(uvicorn, chromadb), via `structlog.stdlib.ProcessorFormatter` with a
`foreign_pre_chain`. That is why a uvicorn access line and a
`pipeline.ready` line have identical structure.

Log level selects the renderer: `console` for humans, `json` for machines. The
call sites are identical either way.

## Data model

```
documents
  id            text primary key
  filename      text
  status        pending | indexing | ready | failed
  pages         int
  chunks        int
  size_bytes    bigint
  content_hash  text        -- deduplicates re-uploads
  created_at    timestamptz
  indexed_at    timestamptz

chunks  (the vector store, not the relational schema)
  id            text primary key
  document_id   text        -- foreign key, the isolation boundary
  page          int
  chunk_index   int
  char_start    int         -- offset into the page text
  char_end      int
  text          text
  embedding     vector(384)
```

`char_start`/`char_end` are recorded at chunking time so a citation can later
highlight the exact span in the source page rather than only naming the page.

## Known gaps on the roadmap

- In-memory repositories must become Postgres; state is per-process today.
- `Embedding` is computed inside the request; large PDFs need a job queue.
- Retrieval is pure vector, so exact-token queries (error codes, part numbers)
  do poorly. Hybrid BM25 + dense retrieval with a reranker is the fix, and the
  evaluation harness is what will tell us whether it helped.
