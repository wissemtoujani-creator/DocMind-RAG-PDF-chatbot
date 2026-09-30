# Project structure

Two views: the directory layout, and the dependency direction that makes it
testable. The second is the one that matters when deciding where new code goes.

## Directory layout

```text
.
├── app/
│   ├── main.py                  application factory, lifespan, error handlers
│   │
│   ├── api/                     HTTP layer — no business logic
│   │   ├── deps.py              resolves singletons off app.state
│   │   ├── schemas.py           Pydantic request/response models
│   │   └── v1/
│   │       ├── router.py        aggregates route modules under /api/v1
│   │       └── routes/
│   │           ├── chat.py        POST /chat, POST /chat/reset
│   │           ├── documents.py   POST/GET/DELETE /documents
│   │           ├── health.py      GET /health, /health/ready
│   │           └── stats.py       GET /stats
│   │
│   ├── core/                    cross-cutting concerns
│   │   ├── config.py            pydantic-settings, validated env
│   │   ├── logging.py           structlog, JSON in production
│   │   ├── errors.py            AppError hierarchy + error codes
│   │   ├── context.py           request_id binding
│   │   └── middleware.py        request context, body size, security headers
│   │
│   ├── repositories/            persistence boundary
│   │   ├── documents.py         DocumentRepository (Protocol + in-memory)
│   │   └── conversations.py     ConversationRepository (Protocol + in-memory)
│   │
│   └── services/                all real logic lives here
│       ├── rag/
│       │   ├── pipeline.py      orchestrates ingest + query
│       │   ├── chunking.py      RecursiveCharacterTextSplitter wrapper
│       │   ├── loader.py        pypdf extraction + upload validation
│       │   ├── embedding.py     MiniLM-L6-v2 embedder
│       │   ├── generator.py     Groq LLM client
│       │   ├── prompts.py       answer + refusal prompts
│       │   └── schemas.py       Chunk / Passage / Answer types
│       │
│       └── vectordb/            swappable store
│           ├── base.py          VectorStore Protocol + backend factory
│           └── chroma_store.py  ChromaDB (HNSW, local disk)
│                            └─ pgvector_store.py  NOT YET WRITTEN
│
│   └── static/index.html        frontend (vanilla JS)
│
├── tests/                       80 tests, 87% coverage, no network
│   ├── conftest.py              fakes: embedder, generator, vector store
│   ├── test_api.py              HTTP contract tests
│   ├── test_chunking.py         chunker + overlap invariants
│   ├── test_core.py             config, errors, middleware
│   └── test_pipeline.py         ingest/query flows with fakes
│
├── docs/
│   ├── ARCHITECTURE.md          request lifecycle, design rationale
│   └── ROADMAP.md               planned work
│
├── scripts/smoke.py             end-to-end against live providers
├── Makefile                     test / lint / coverage / dev targets
├── pyproject.toml               ruff + pytest config
├── requirements.txt             runtime
├── requirements-dev.txt         runtime + test/lint
└── .env.example                 every option, documented
```

## Dependency direction

Dependencies point strictly inward. Nothing in `services/` imports from `api/`,
which is exactly what lets the pipeline be tested without HTTP and the store be
swapped without touching callers.

```mermaid
graph TD
    Client([Browser / API client])

    subgraph Entry
        Main["main.py<br/>create_app, lifespan"]
        MW["core/middleware.py<br/>context, body size, headers"]
        Handlers["error handlers<br/>single JSON envelope"]
    end

    subgraph API["api/ — HTTP only"]
        Router["v1/router.py"]
        Routes["routes/<br/>chat · documents<br/>health · stats"]
        Deps["deps.py<br/>resolve off app.state"]
        Schemas["schemas.py<br/>Pydantic models"]
    end

    subgraph Services["services/ — all logic"]
        Pipeline["rag/pipeline.py<br/>orchestrates"]
        Loader["rag/loader.py<br/>pypdf + validation"]
        Chunker["rag/chunking.py<br/>512 / 64 overlap"]
        Embedder["rag/embedding.py<br/>MiniLM-L6-v2"]
        Prompts["rag/prompts.py"]
        Generator["rag/generator.py<br/>Groq"]
    end

    subgraph Ports["ports & adapters"]
        VectorStore["vectordb/base.py<br/>VectorStore Protocol"]
        Chroma["chroma_store.py<br/>HNSW, local disk"]
        Pgvector["pgvector_store.py<br/>MISSING"]
        DocRepo["repositories/documents.py<br/>Protocol"]
        ConvRepo["repositories/conversations.py<br/>Protocol"]
        InMemory["in-memory impls<br/>lost on restart"]
    end

    Ext[["External<br/>Groq API · HuggingFace · ChromaDB"]]

    Client --> Main
    Main --> MW
    MW --> Router
    Router --> Routes
    Routes --> Deps
    Routes --> Schemas
    Routes --> Pipeline
    Routes --> DocRepo
    Routes --> ConvRepo
    Main --> Handlers

    Pipeline --> Loader
    Pipeline --> Chunker
    Pipeline --> Embedder
    Pipeline --> Prompts
    Pipeline --> Generator
    Pipeline --> VectorStore
    Pipeline --> DocRepo
    Pipeline --> ConvRepo

    VectorStore --> Chroma
    VectorStore -. "factory branch" .-> Pgvector
    DocRepo --> InMemory
    ConvRepo --> InMemory

    Chroma --> Ext
    Embedder --> Ext
    Generator --> Ext

    classDef missing fill:#3d1d1d,stroke:#a0522d,stroke-dasharray:5 5,color:#e8e8e8
    classDef ok fill:#1d2d1d,stroke:#2e6b3a,color:#e8e8e8
    class Pgvector missing
    class Chroma,InMemory ok
```

## Ingest vs. query

The two flows are asymmetric: ingest writes to every store, query reads from
most of them and calls out to the LLM.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant D as POST /documents
    participant P as RAGPipeline
    participant L as loader
    participant Ck as chunker
    participant E as embedder
    participant V as VectorStore
    participant R as DocumentRepo

    C->>D: PDF (≤ 25 MB)
    D->>L: validate + extract
    L-->>D: pages
    D->>P: ingest()
    P->>Ck: split(pages)
    Ck-->>P: chunks (page + offsets)
    P->>E: embed(texts)
    E-->>P: vectors[384]
    P->>V: upsert(document_id, vectors, chunks)
    P->>R: mark status = ready
    P-->>C: 201 {document_id, pages, chunks}
```

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant D as POST /chat
    participant P as RAGPipeline
    participant CR as ConversationRepo
    participant E as embedder
    participant V as VectorStore
    participant G as generator

    C->>D: {question, document_id}
    D->>P: query()
    P->>CR: last 5 turns
    CR-->>P: history
    P->>E: embed(question)
    E-->>P: vector
    P->>V: search(document_id, top_k=5)
    V-->>P: passages + scores
    P->>G: prompt(passages numbered [1]..[n])
    G-->>P: answer citing [n] or REFUSAL
    P->>CR: append turn
    P-->>C: 200 {answer, citations[]}
```

## Two things the diagram makes obvious

**The pgvector branch is a promise, not a feature.** `base.py:43` dispatches on
`VECTOR_STORE_BACKEND` and `config.py:87` validates the value, but
`pgvector_store.py` does not exist — setting that variable raises `ImportError`
at startup. The dashed node is deliberate.

**Persistence is the same story.** Both repositories are Protocols with
in-memory implementations, so state is lost on restart and cannot be shared
across workers. The Postgres work replaces the two in-memory nodes and fills in
the dashed one; nothing above that line changes, because the Protocols are
already the seam.
