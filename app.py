import os
import shutil
import time
import tempfile
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

load_dotenv()

app = FastAPI(title="DocMind — PDF RAG System")
@app.get("/")
def home():
    return FileResponse("index.html")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

pipeline = RAGPipeline()


class QuestionRequest(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = static_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static/")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pipeline.reset()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        docs   = pipeline.load_pdf(tmp_path)
        chunks = pipeline.chunk_documents(docs)
        pipeline.create_vector_store(chunks)
        pipeline.setup_qa_chain()
        elapsed = round(time.time() - t0, 1)
        stats   = pipeline.get_stats()

        return JSONResponse({
            "success":   True,
            "filename":  file.filename,
            "pages":     stats["pages"],
            "chunks":    stats["chunks"],
            "elapsed_s": elapsed,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/ask")
async def ask_question(body: QuestionRequest):
    t0     = time.time()
    result = pipeline.answer_question(body.question)
    result["latency_s"] = round(time.time() - t0, 2)
    return JSONResponse(result)


@app.get("/stats")
async def get_stats():
    return JSONResponse(pipeline.get_stats())


@app.post("/reset")
async def reset_pipeline():
    pipeline.reset()
    return JSONResponse({"success": True})