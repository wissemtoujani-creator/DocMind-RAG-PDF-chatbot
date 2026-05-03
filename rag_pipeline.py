import os
import logging
from typing import List, Dict, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        k_retrieved: int = 5,
        temperature: float = 0.3,
        model_name: str = "llama-3.1-8b-instant",
        memory_window: int = 5,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieved = k_retrieved
        self.temperature = temperature
        self.model_name = model_name
        self.memory_window = memory_window

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store: Optional[Chroma] = None
        self.qa_chain = None

        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.pdf_metadata: Dict = {}

        # Manual conversation memory
        self.chat_history: List[Dict] = []

        logger.info("RAG Pipeline initialized")

    # ------------------------------------------------------------------ #
    #  INGESTION                                                           #
    # ------------------------------------------------------------------ #

    def load_pdf(self, pdf_path: str) -> List[Document]:
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        pdf_name = os.path.basename(pdf_path)
        for doc in documents:
            doc.metadata["source_pdf"] = pdf_name

        self.documents = documents
        self.pdf_metadata[pdf_name] = {"pages": len(documents), "loaded": True}
        logger.info(f"Loaded {len(documents)} pages from {pdf_name}")
        return documents

    def chunk_documents(self, documents: List[Document] = None) -> List[Document]:
        if documents is None:
            documents = self.documents
        if not documents:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        self.chunks = chunks
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def create_vector_store(
        self,
        chunks: List[Document] = None,
        persist_directory: str = "chroma_db",
    ) -> Chroma:
        if chunks is None:
            chunks = self.chunks
        if not chunks:
            return None

        logger.info("Building ChromaDB vector store...")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name="pdf_documents",
        )
        logger.info(f"Vector store ready ({len(chunks)} chunks)")
        return self.vector_store

    # ------------------------------------------------------------------ #
    #  QA CHAIN                                                            #
    # ------------------------------------------------------------------ #

    def setup_qa_chain(self) -> None:
        if self.vector_store is None:
            logger.error("Vector store not initialised — run create_vector_store first")
            return

        llm = ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        # History is prepended to the query string — RetrievalQA only accepts {context} and {question}
        prompt_template = """You are a precise document assistant.
Use ONLY the retrieved context below to answer the question.
If the answer is not in the context, say exactly:
"I don't have this information in the document."
Always mention the page number when you can.

Retrieved context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"],
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k_retrieved},
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        logger.info("QA chain ready")

    # ------------------------------------------------------------------ #
    #  QUERY                                                               #
    # ------------------------------------------------------------------ #

    def _format_history(self) -> str:
        """Return the last memory_window turns as a plain-text string."""
        window = self.chat_history[-(self.memory_window * 2):]
        if not window:
            return "None"
        lines = []
        for msg in window:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def answer_question(self, question: str) -> Dict:
        if self.qa_chain is None:
            return {
                "answer": "Please upload and process a PDF first.",
                "sources": [],
                "chat_history": [],
                "error": True,
            }
        if not question.strip():
            return {
                "answer": "Please enter a valid question.",
                "sources": [],
                "chat_history": [],
                "error": True,
            }

        try:
            # Prepend conversation history into the query itself
            history = self._format_history()
            if history and history != "None":
                full_query = f"Previous conversation:\n{history}\n\nCurrent question: {question}"
            else:
                full_query = question

            result = self.qa_chain.invoke({"query": full_query})

            answer = result.get("result", "No answer generated")

            # Deduplicated source list
            sources = []
            seen = set()
            for doc in result.get("source_documents", []):
                page = doc.metadata.get("page", "?")
                src  = doc.metadata.get("source_pdf", "document")
                key  = (page, src)
                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "page": page,
                        "source": src,
                        "preview": doc.page_content[:200] + "…",
                    })

            # Append to manual memory
            self.chat_history.append({"role": "user",      "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})

            return {
                "answer": answer,
                "sources": sources,
                "chat_history": self.chat_history[-(self.memory_window * 2):],
                "error": False,
            }

        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "chat_history": self.chat_history,
                "error": True,
            }

    # ------------------------------------------------------------------ #
    #  UTILS                                                               #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict:
        return {
            "pages":        sum(m.get("pages", 0) for m in self.pdf_metadata.values()),
            "chunks":       len(self.chunks),
            "pdfs":         list(self.pdf_metadata.keys()),
            "memory_turns": len(self.chat_history) // 2,
        }

    def reset(self) -> None:
        self.documents    = []
        self.chunks       = []
        self.vector_store = None
        self.qa_chain     = None
        self.chat_history = []
        self.pdf_metadata = {}
        logger.info("Pipeline reset")