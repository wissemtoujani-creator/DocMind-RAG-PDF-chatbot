# Roadmap

Each milestone is independently shippable and demoable. The app runs at every
checkpoint.

## 1. Foundation — done

- [x] `app/` package layout with inward-only dependencies
- [x] `core/config.py` — all tunables in one env-driven place
- [x] Structured logging (structlog), JSON in production, one pipeline for app
      and third-party records
- [x] Per-request `X-Request-ID` correlation via `contextvars`
- [x] Error taxonomy + single response envelope
- [x] Middleware: body-size limit, security headers, CORS (wildcard refused in
      production), access logging
- [x] Stateless pipeline; per-document and per-conversation state
- [x] Versioned `/api/v1` router with typed request/response models
- [x] `VectorStore` / `Embedder` / `AnswerGenerator` / repository protocols
- [x] Chunking with page provenance and character offsets
- [x] Numbered citations
- [x] 80 tests, 87% coverage, ruff clean, no network required
- [x] Fixed: duplicate `/` route, `index.html` outside `static/`, off-by-one
      page numbers in the UI, global singleton, global chat history

## 2. PostgreSQL + pgvector

- [ ] async SQLAlchemy engine, session factory, `pool_pre_ping`
- [ ] Alembic migrations as the only way to change the schema
- [ ] `PgVectorStore` implementing `VectorStore`; HNSW index on `embedding`
- [ ] `PostgresDocumentRepository`, `PostgresConversationRepository`
- [ ] `asyncpg` driver; the sync pipeline runs in a thread as it does now
- [ ] Partial index on `(document_id)` for scoped deletes and filters
- [ ] Retention: cascade from `documents` to `chunks`
- [ ] Remove the Chroma adapter once nothing depends on it
- [ ] Decision: keep source PDFs in object storage so re-indexing is possible

## 3. Authentication

- [ ] `users` table; Argon2id hashing
- [ ] OAuth2 password flow; short-lived access JWT + rotating refresh token
- [ ] Refresh-token table with reuse detection (revoke the family on replay)
- [ ] Ownership: `documents.owner_id`, all queries filtered by it
- [ ] `require_user` dependency; public routes reduced to health and login
- [ ] Login screen in the UI, token refresh on 401
- [ ] Per-user quotas

## 4. Citations and streaming

- [ ] Inline `[1]` markers already render; make them clickable to the page span
- [ ] `/documents/{id}/pages/{n}` to return the page text for highlighting
- [ ] Highlight `char_start`..`char_end` in the returned page
- [ ] Verify a cited passage actually supports the claim; drop bad citations
      instead of showing them
- [ ] SSE streaming for token-by-token answers
- [ ] Retrieval scores and timings surfaced in the response for the eval harness

## 5. Retrieval quality

- [ ] Section-aware chunking using PDF headings; keep a recursive fallback
- [ ] Hybrid retrieval: BM25 + dense, fused (RRF)
- [ ] Cross-encoder reranking of the fused candidates
- [ ] Score threshold so weak passages are dropped rather than cited
- [ ] Query rewriting for multi-turn questions that reference earlier turns
- [ ] Measure every one of these with the harness below before and after

## 6. Evaluation

- [ ] Labelled question set with gold source pages
- [ ] Retrieval: recall@k, MRR, nDCG
- [ ] Generation: faithfulness, answer relevancy, context precision/recall
- [ ] Citation accuracy: does each cited page contain the claim?
- [ ] Refusal accuracy: unanswerable questions must produce the refusal
- [ ] JSON results committed per run so regressions are visible in diffs
- [ ] CI gate: fail on a metric regression beyond a threshold

## 7. Rate limiting and resilience

- [ ] Redis-backed token bucket, per user and per IP
- [ ] `/chat` gets a tighter bucket than `/health`
- [ ] Uploads additionally bounded by user quota
- [ ] Response headers: `X-RateLimit-Limit`, `-Remaining`, `-Reset`
- [ ] Circuit breaker on the LLM provider; serve cached answers when open
- [ ] Background ingestion queue for large PDFs

## 8. Packaging and delivery

- [ ] Dockerfile: multi-stage, non-root user, pinned base image
- [ ] `docker-compose.yml` — app, Postgres+pgvector, Redis
- [ ] Healthchecks wired to `/health/ready`
- [ ] GitHub Actions: lint, typecheck, tests, coverage upload
- [ ] Build and push the image on tag
- [ ] Migrations run as a release step, not at app start

## 9. Observability

- [ ] OpenTelemetry traces across request → retrieve → generate
- [ ] Prometheus metrics: latency histograms, token counts, cache hit rate
- [ ] Slowest-stage breakdown in the trace, so regressions are diagnosable
- [ ] Alerting on error rate and p95 latency

## 10. Documentation

- [ ] `docs/ARCHITECTURE.md` (written)
- [ ] API reference generated from OpenAPI
- [ ] A written walkthrough of the eval results and what each change bought
