import os
import json
import torch
import asyncio
import traceback
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import CrossEncoder

class Config:
    PDF_DOC_PATH = "sample_docs"
    VECTOR_STORE_DIR = "vector_stores"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gemma3:4b"
    QUERY_GENERATOR_MODEL = "llama3:8b-instruct-q5_K_M"
    RERANKER_MODEL = 'BAAI/bge-reranker-base'
    RETRIEVAL_K = 10
    RERANK_TOP_N = 8

    def __init__(self):
        self.DEVICE = self._get_device()
        print(f"✅ Configuration Initialized. Using device: {self.DEVICE.upper()}")

    def _get_device(self):
        if torch.cuda.is_available(): return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'

class ConversationMemory:
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={'device': config.DEVICE})
        self.vector_store = None
        self.memory_docs = []

    def add_interaction(self, query: str, answer_json: Dict[str, Any]):
        memory_text = f"User Question: {query}\nAI Answer: {answer_json.get('finalAnswer', 'N/A')}"
        new_doc = Document(page_content=memory_text, metadata={"source": "conversation_memory"})
        self.memory_docs.append(new_doc)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(self.memory_docs, self.embeddings)
        else:
            self.vector_store.add_documents([new_doc])
        print("🧠 New interaction saved to conversation memory.")

    async def a_retrieve_relevant_memories(self, query: str) -> List[Document]:
        if self.vector_store is None: return []
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})
        return await retriever.ainvoke(query)

class DocumentProcessor:
    def load_and_chunk_pdf(self, file_path: str) -> List[Document]:
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        for i, doc in enumerate(pages):
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["page"] = i + 1
        print(f"📄 Processed {os.path.basename(file_path)} into {len(pages)} pages/chunks.")
        return pages

class VectorStoreManager:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={'device': config.DEVICE})

    def create_or_load_store(self, chunks: List[Document], vector_store_path: str, force_recreate: bool = False) -> FAISS:
        if not force_recreate and os.path.exists(vector_store_path):
            print(f"✅ Loading existing vector store from '{vector_store_path}'...")
            return FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        
        print(f"✨ Creating new vector store at '{vector_store_path}'...")
        if not chunks: raise ValueError("No document chunks provided to create a vector store.")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(vector_store_path)
        print("✅ Vector store created and saved.")
        return vector_store

class RAGPipeline:
    def __init__(self, config: Config, doc_vector_store: FAISS, memory: ConversationMemory):
        self.config = config
        self.memory = memory
        self.llm = Ollama(model=config.LLM_MODEL, temperature=0.0)
        self.query_generator = Ollama(model=config.QUERY_GENERATOR_MODEL, format="json", temperature=0.2)
        self.reranker = CrossEncoder(config.RERANKER_MODEL, device=config.DEVICE)
        self.doc_retriever = doc_vector_store.as_retriever(search_kwargs={'k': config.RETRIEVAL_K})

    async def _generate_multiple_queries(self, original_query: str) -> List[str]:
        prompt = f'You are an AI assistant. Generate 3 diverse, alternative queries for the user query: "{original_query}". Output a JSON object with a "queries" key containing a list of strings.'
        try:
            response = await self.query_generator.ainvoke(prompt)
            data = json.loads(response)
            return [original_query] + data.get("queries", [])
        except Exception as e:
            print(f"⚠️ Query generation failed: {e}. Falling back to original query.")
            return [original_query]

    async def _retrieve(self, all_queries: List[str], original_query: str) -> List[Document]:
        tasks = [self.doc_retriever.ainvoke(q) for q in all_queries]
        tasks.append(self.memory.a_retrieve_relevant_memories(original_query))
        results = await asyncio.gather(*tasks)
        all_retrieved_docs = [doc for res in results for doc in res]
        unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}
        return list(unique_docs.values())

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs: return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:self.config.RERANK_TOP_N]]

    async def _generate(self, query: str, reranked_docs: List[Document]) -> str:
        prompt = f"""
        You are a hyper-analytical AI expert. Your sole task is to generate a structured JSON response based ONLY on the provided context. Follow these steps meticulously:
        1.  **Analyze the Query:** Deconstruct the user's query to understand its core intent and key components.
        2.  **Review Context:** Scan all provided context snippets (from documents and memory) to identify relevant facts and evidence.
        3.  **Synthesize Evidence:** Connect the different pieces of evidence to build a coherent chain of reasoning that directly addresses the query.
        4.  **Assess Confidence:** Rate your confidence in the answer based on the quality and directness of the evidence.
        5.  **Formulate Final Answer:** Construct a clear, concise final answer.
        6.  **Generate JSON:** Compile all steps into the specified JSON format. Do not add any text before or after the JSON object.

        **JSON OUTPUT STRUCTURE:**
        {{
          "thinkingProcess": [
            {{ "step": 1, "action": "Query Analysis", "summary": "Briefly describe your understanding of what the user is asking." }},
            {{ "step": 2, "action": "Context Review", "summary": "Summarize the key pieces of information found in the context." }},
            {{ "step": 3, "action": "Evidence Synthesis", "summary": "Explain how the pieces of evidence connect to form a complete answer." }}
          ],
          "finalAnswer": "The detailed, multi-sentence answer to the user's query.",
          "confidence": "A score from 0.0 to 1.0.",
          "evidence": [
            {{ "text": "The exact verbatim sentence from the context.", "document": "The source document name.", "page": "The page number or 'memory'." }}
          ]
        }}

        **CONTEXT:**
        ---
        {self._format_docs(reranked_docs)}
        ---
        **USER QUERY:** {query}
        **FINAL JSON RESPONSE:**
        """
        return await self.llm.ainvoke(prompt)

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'memory')}\nContent: {doc.page_content}" for doc in docs)

    async def query(self, query_text: str) -> str:
        print("\n--- 1. Generating & Retrieving Queries (in parallel) ---")
        all_queries = await self._generate_multiple_queries(query_text)
        print(f"🔍 Searching with {len(all_queries)} query variations...")
        retrieved_docs = await self._retrieve(all_queries, query_text)
        print(f"📚 Retrieved {len(retrieved_docs)} unique documents and memories.")
        
        print("\n--- 2. Re-ranking for Final Accuracy ---")
        reranked_docs = self._rerank(query_text, retrieved_docs)
        print(f"✨ Filtered to top {len(reranked_docs)} most relevant items.")
        
        print("\n--- 3. Generating Final Answer with Visible Thinking ---")
        response = await self._generate(query_text, reranked_docs)
        print("✅ Generation complete.")
        return response

app = FastAPI(
    title="Document RAG API",
    description="An API to chat with PDF documents using a RAG pipeline.",
    version="1.0.0"
)

app_state: Dict[str, Any] = {
    "config": None,
    "pipeline": None,
    "memory": None,
    "loaded_document": None
}

class LoadDocumentRequest(BaseModel):
    filename: str
    force_recreate: bool = False

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    config = Config()
    app_state["config"] = config
    
    if not os.path.exists(config.PDF_DOC_PATH):
        os.makedirs(config.PDF_DOC_PATH)
        print(f"📂 Created document directory: {config.PDF_DOC_PATH}")
    if not os.path.exists(config.VECTOR_STORE_DIR):
        os.makedirs(config.VECTOR_STORE_DIR)
        print(f"📂 Created vector store directory: {config.VECTOR_STORE_DIR}")

@app.get("/documents", summary="List Available PDFs")
async def list_documents() -> Dict[str, List[str]]:
    config: Config = app_state["config"]
    try:
        pdf_files = [f for f in os.listdir(config.PDF_DOC_PATH) if f.lower().endswith('.pdf')]
        return {"documents": pdf_files}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Document directory not found: {config.PDF_DOC_PATH}")

@app.post("/load-document", summary="Load and Process a PDF")
async def load_document(request: LoadDocumentRequest = Body(...)) -> Dict[str, str]:
    config: Config = app_state["config"]
    pdf_path = os.path.join(config.PDF_DOC_PATH, request.filename)
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {request.filename}")

    try:
        print(f"\n--- Loading Document: {request.filename} ---")
        app_state["memory"] = ConversationMemory(config)

        processor = DocumentProcessor()
        chunks = processor.load_and_chunk_pdf(pdf_path)

        vector_store_path = os.path.join(config.VECTOR_STORE_DIR, f"vs_{os.path.basename(pdf_path)}.faiss")
        vs_manager = VectorStoreManager(config)
        doc_vector_store = vs_manager.create_or_load_store(chunks, vector_store_path, force_recreate=request.force_recreate)

        app_state["pipeline"] = RAGPipeline(config, doc_vector_store, app_state["memory"])
        app_state["loaded_document"] = request.filename
        
        message = "re-created and loaded" if request.force_recreate else "loaded"
        print(f"✅ Pipeline ready for '{request.filename}'.")
        return {"message": f"Document '{request.filename}' has been successfully {message}."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred while loading the document: {e}")


@app.post("/query", summary="Ask a Question")
async def ask_query(request: QueryRequest = Body(...)) -> Dict[str, Any]:
    if not app_state.get("pipeline"):
        raise HTTPException(status_code=400, detail="No document has been loaded. Please call the /load-document endpoint first.")

    pipeline: RAGPipeline = app_state["pipeline"]
    memory: ConversationMemory = app_state["memory"]
    
    try:
        print(f"\n🚀 Received Query: {request.query}")
        final_response_str = await pipeline.query(request.query)
        
        try:
            response_json = json.loads(final_response_str)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse LLM response as JSON. Returning as raw string.")
            response_json = {"finalAnswer": final_response_str, "error": "Failed to parse LLM output as JSON."}

        memory.add_interaction(request.query, response_json)
        
        print("\n--- FINAL API RESPONSE ---")
        print(json.dumps(response_json, indent=2))
        return response_json

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during the query process: {e}")

@app.get("/status", summary="Check API Status")
async def get_status() -> Dict[str, Optional[str]]:
    return {
        "status": "online",
        "loaded_document": app_state.get("loaded_document"),
        "device": app_state["config"].DEVICE.upper() if app_state.get("config") else None
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)