# DocMind — RAG PDF Chatbot

> Upload any PDF and query it in natural language. Powered by Retrieval-Augmented Generation.


# What is DocMind?

DocMind is a local PDF question-answering system built on the RAG (Retrieval-Augmented Generation) architecture. You upload a PDF, the system indexes it semantically, and you can then ask questions about it in natural language through a clean web interface. The system answers strictly from the document content and always provides the source page number.



# Features

- **PDF Upload**  drag and drop or click to upload any PDF up to 100MB
- **Semantic Search** queries are matched by meaning, not just keywords
- **Conversational Memory**  remembers the last 5 turns so follow-up questions work correctly
- **Source Citations**  every answer includes the exact page number with a text preview tooltip
- **Honest Refusal**  if the answer is not in the document, the system says so instead of hallucinating
- **Fast**  average query latency under 3 seconds powered by Groq inference



# Tech Stack

| Layer | Technology |
| Backend | FastAPI + Uvicorn |
| RAG Orchestration | LangChain v0.2 (RetrievalQA) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (HNSW index, local disk) |
| LLM | Groq `llama-3.1-8b-instant` |
| Frontend | Pure HTML / CSS / Vanilla JS |
| PDF Loading | PyPDF |



# Getting Started

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
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file in the root folder:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
uvicorn app:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

> **Note:** The first launch will download the `all-MiniLM-L6-v2` embedding model (~90MB). This happens once and is cached automatically.

---

## How It Works

```
INGESTION (once per PDF)
PDF Upload → PyPDF extraction → Text chunking (512 tok / 64 overlap)
→ MiniLM-L6-v2 embeddings → ChromaDB HNSW index → RetrievalQA chain ready

QUERY (every question)
User question + last 5 turns → ChromaDB cosine search (top-5 chunks)
→ RAG prompt → Groq LLaMA 3.1 → Answer + page citations
```


## Authors

- **Wissem Toujani**
- **Hedy Ben Hamadou**

Final Year Project (PFA) — National Institute of Applied Sciences, Tunis — 2026
