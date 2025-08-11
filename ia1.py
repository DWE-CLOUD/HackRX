import os
import json
import torch
import asyncio
import traceback
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import CrossEncoder
from scipy.spatial.distance import cosine
import numpy as np

class Config:
    PDF_DOC_PATH = "sample_docs"
    VECTOR_STORE_DIR = "vector_stores"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "phi3:latest"
    QUERY_GENERATOR_MODEL = "llama3:8b-instruct-q5_K_M"
    RERANKER_MODEL = 'BAAI/bge-reranker-base'
    RETRIEVAL_K = 10
    RERANK_TOP_N = 6

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

    def add_interaction(self, query, answer_json):
        memory_text = f"User Question: {query}\nAI Answer: {answer_json.get('answer', 'N/A')}"
        new_doc = Document(page_content=memory_text, metadata={"source": "conversation_memory"})
        self.memory_docs.append(new_doc)
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(self.memory_docs, self.embeddings)
        else:
            self.vector_store.add_documents([new_doc])
        print("🧠 New interaction saved to conversation memory.")

    async def a_retrieve_relevant_memories(self, query):
        if self.vector_store is None: return []
        retriever = self.vector_store.as_retriever(search_kwargs={'k': 3})
        return await retriever.ainvoke(query)

class DocumentProcessor:
    def load_and_chunk_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        for i, doc in enumerate(pages):
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["page"] = i + 1
        print(f"📄 Processed {os.path.basename(file_path)} into {len(pages)} pages/chunks.")
        return pages

class VectorStoreManager:
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={'device': config.DEVICE})

    def create_or_load_store(self, chunks, vector_store_path, force_recreate=False):
        if not force_recreate and os.path.exists(vector_store_path):
            return FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"✨ Creating new vector store at '{vector_store_path}'...")
        if not chunks: raise ValueError("No document chunks provided.")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(vector_store_path)
        return vector_store

class RAGPipeline:
    def __init__(self, config, doc_vector_store, memory):
        self.config = config
        self.memory = memory
        self.llm = Ollama(model=config.LLM_MODEL, temperature=0.1)
        self.query_generator = Ollama(model=config.QUERY_GENERATOR_MODEL, format="json", temperature=0.2)
        self.reranker = CrossEncoder(config.RERANKER_MODEL, device=config.DEVICE)
        self.doc_retriever = doc_vector_store.as_retriever(search_kwargs={'k': config.RETRIEVAL_K})

    async def _generate_multiple_queries(self, original_query):
        prompt = f'You are an AI assistant. Generate 3 diverse, alternative queries for the user query: "{original_query}". Output a JSON object with a "queries" key containing a list of strings.'
        try:
            response = await self.query_generator.ainvoke(prompt); data = json.loads(response)
            return [original_query] + data.get("queries", [])
        except Exception: return [original_query]

    async def _retrieve(self, all_queries, original_query):
        tasks = [self.doc_retriever.ainvoke(q) for q in all_queries]
        tasks.append(self.memory.a_retrieve_relevant_memories(original_query))
        results = await asyncio.gather(*tasks)
        all_retrieved_docs = [doc for res in results for doc in res]
        unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}
        return list(unique_docs.values())

    def _rerank(self, query, docs):
        if not docs: return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:self.config.RERANK_TOP_N]]

    async def _generate(self, query, reranked_docs):
        prompt = f"""
        <|user|>
        You are an expert AI analyst. Your task is to provide a clear and factual answer based ONLY on the context provided.
        Format your entire response as a single, valid JSON object. Do not add any text before or after the JSON.

        **JSON Output Structure:**
        {{
          "reasoning": "A brief, step-by-step explanation of how you derived the answer from the evidence.",
          "answer": "A detailed, multi-sentence answer to the user's query.",
          "evidence": [
            {{
              "text": "The exact verbatim sentence from the context.",
              "document": "The source document name.",
              "page": "The page number or 'memory'."
            }}
          ]
        }}

        **Context:**
        ---
        {self._format_docs(reranked_docs)}
        ---

        **User Query:** {query}
        <|end|>
        <|assistant|>
        """
        return await self.llm.ainvoke(prompt)

    def _format_docs(self, docs):
        return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'memory')}\nContent: {doc.page_content}" for doc in docs)

    async def query(self, query_text):
        print("\n--- 1. Generating & Retrieving Queries (in parallel) ---")
        all_queries = await self._generate_multiple_queries(query_text)
        print(f"🔍 Searching with {len(all_queries)} query variations...")
        retrieved_docs = await self._retrieve(all_queries, query_text)
        print(f"📚 Retrieved {len(retrieved_docs)} unique documents and memories.")

        print("\n--- 2. Re-ranking for Final Accuracy ---")
        reranked_docs = self._rerank(query_text, retrieved_docs)
        print(f"✨ Filtered to top {len(reranked_docs)} most relevant items.")

        print("\n--- 3. Generating Final Answer ---")
        response = await self._generate(query_text, reranked_docs)
        print("✅ Generation complete.")
        return response

def select_pdf(doc_path):
    pdf_files = [f for f in os.listdir(doc_path) if f.lower().endswith('.pdf')]
    if not pdf_files: print(f"Error: No PDF files found in '{doc_path}'."); return None
    print("\nPlease select a document to analyze:")
    for i, filename in enumerate(pdf_files): print(f"  [{i+1}] {filename}")
    while True:
        try:
            choice = int(input(f"Enter a number (1-{len(pdf_files)}): "))
            if 1 <= choice <= len(pdf_files): return os.path.join(doc_path, pdf_files[choice-1])
            else: print("Invalid number.")
        except (ValueError, IndexError): print("Invalid input.")

async def main():
    try:
        config = Config()
        memory = ConversationMemory(config)
        if not os.path.exists(config.PDF_DOC_PATH): os.makedirs(config.PDF_DOC_PATH)
        if not os.path.exists(config.VECTOR_STORE_DIR): os.makedirs(config.VECTOR_STORE_DIR)
        
        selected_pdf_path = select_pdf(config.PDF_DOC_PATH)
        if not selected_pdf_path: return

        vector_store_path = os.path.join(config.VECTOR_STORE_DIR, f"vs_{os.path.basename(selected_pdf_path)}.faiss")
        processor = DocumentProcessor()
        chunks = processor.load_and_chunk_pdf(selected_pdf_path)
        vs_manager = VectorStoreManager(config)
        doc_vector_store = vs_manager.create_or_load_store(chunks, vector_store_path, force_recreate=True)
        pipeline = RAGPipeline(config, doc_vector_store, memory)
        
        while True:
            user_query = input("\nEnter your query (or type 'exit' to quit): ")
            if user_query.lower() == 'exit': break
            if not user_query.strip(): continue

            final_response_str = await pipeline.query(user_query)
            
            try:
                response_json = json.loads(final_response_str)
                print("\n--- FINAL ANSWER ---")
                print(json.dumps(response_json, indent=2))
                memory.add_interaction(user_query, response_json)
            except json.JSONDecodeError:
                print("\n--- WARNING: LLM did not return valid JSON ---")
                print("Raw LLM Output:\n", final_response_str)

    except Exception as e:
        print(f"\n--- A CRITICAL ERROR OCCURRED: {e} ---"); traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())