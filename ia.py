import os
import json
import torch
import fitz
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
# FIX: Updated Ollama import
from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

class Config:
    PDF_DOC_PATH = "sample_docs"
    VECTOR_STORE_DIR = "vector_stores"
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    LLM_MODEL = "llama3:8b"
    RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    CHAR_CHUNK_LIMIT = 2000
    RETRIEVAL_K = 15
    RERANK_TOP_N = 5

    def __init__(self):
        self.DEVICE = self._get_device()
        print(f"✅ Configuration Initialized. Using device: {self.DEVICE.upper()}")

    def _get_device(self):
        if torch.cuda.is_available(): return 'cuda'
        if torch.backends.mps.is_available(): return 'mps'
        return 'cpu'

class DocumentProcessor:
    def __init__(self, chunk_limit):
        self.chunk_limit = chunk_limit

    def load_and_chunk_pdf(self, file_path):
        all_chunks = []
        filename = os.path.basename(file_path)
        doc = fitz.open(file_path)
        print(f"📄 Processing PDF: {filename}")
        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            current_chunk_text, current_chunk_bboxes = "", []
            for b in blocks:
                block_text = b[4]
                if not block_text.strip(): continue
                if len(current_chunk_text) + len(block_text) > self.chunk_limit:
                    if current_chunk_text:
                        metadata = {"source": filename, "page": page_num + 1, "bboxes": current_chunk_bboxes}
                        all_chunks.append(Document(page_content=current_chunk_text.strip(), metadata=metadata))
                    current_chunk_text, current_chunk_bboxes = "", []
                current_chunk_text += block_text
                current_chunk_bboxes.append(list(b[:4]))
            if current_chunk_text:
                metadata = {"source": filename, "page": page_num + 1, "bboxes": current_chunk_bboxes}
                all_chunks.append(Document(page_content=current_chunk_text.strip(), metadata=metadata))
        print(f"📄 Created {len(all_chunks)} semantic chunks from {filename}.")
        return all_chunks

class VectorStoreManager:
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={'device': config.DEVICE})

    def create_or_load_store(self, chunks, vector_store_path, force_recreate=False):
        if not force_recreate and os.path.exists(vector_store_path):
            print(f"✅ Vector store found at '{vector_store_path}'. Loading from disk.")
            return FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"✨ Creating new vector store at '{vector_store_path}'...")
        if not chunks: raise ValueError("No document chunks provided to create the vector store.")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(vector_store_path)
        print("✅ Vector store created and saved.")
        return vector_store

class RAGPipeline:
    def __init__(self, config, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.llm = Ollama(model=config.LLM_MODEL, format="json", temperature=0)
        self.reranker = CrossEncoder(config.RERANKER_MODEL, device=config.DEVICE)
        self.retriever = vector_store.as_retriever(search_kwargs={'k': config.RETRIEVAL_K})

    def _retrieve(self, query):
        return self.retriever.invoke(query)

    def _rerank(self, query, docs):
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:self.config.RERANK_TOP_N]]

    def _generate(self, query, reranked_docs):
        prompt_template = """
        You are a meticulous analyst AI. Your task is to provide a detailed, factual answer based exclusively on the provided document excerpts.
        Return a single, valid JSON object with the following structure. Do not add any text before or after the JSON.
        {{
          "contextAssessment": "Your evaluation of the context's quality and relevance.",
          "reasoning": "Your detailed step-by-step thought process, citing context.",
          "finalAnswer": "The direct, synthesized answer to the query.",
          "evidence": [
            {{
              "clauseText": "The exact, verbatim text used as evidence.",
              "document": "The source document name.",
              "page": "The page number.",
              "bboxes": [[x0, y0, x1, y1], ...]
            }}
          ]
        }}

        CONTEXT:
        {context}

        QUERY:
        {query}
        """
        
        def format_docs(docs):
            return "\n\n".join(
                f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}\n"
                f"Content: {doc.page_content}\n"
                f"METADATA_JSON: {json.dumps(doc.metadata)}"
                for doc in docs
            )
        
        context_str = format_docs(reranked_docs)
        prompt = prompt_template.format(context=context_str, query=query)
        
        # --- START: Added Debug Statements ---
        print("\n" + "="*50)
        print("--- [DEBUG] FINAL PROMPT SENT TO LLM ---")
        print(prompt)
        print("="*50 + "\n")
        # --- END: Added Debug Statements ---

        response_str = self.llm.invoke(prompt)

        # --- START: Added Debug Statements ---
        print("\n" + "="*50)
        print("--- [DEBUG] RAW RESPONSE FROM LLM ---")
        print(response_str)
        print("="*50 + "\n")
        # --- END: Added Debug Statements ---

        return response_str

    def query(self, query_text):
        print("\n--- 1. Initial Retrieval ---")
        retrieved_docs = self._retrieve(query_text)
        print(f"🔍 Retrieved {len(retrieved_docs)} initial candidates.")

        print("\n--- 2. Re-ranking for Relevance ---")
        reranked_docs = self._rerank(query_text, retrieved_docs)
        print(f"✨ Filtered to top {len(reranked_docs)} most relevant documents.")

        print("\n--- 3. Generating Detailed Response ---")
        response = self._generate(query_text, reranked_docs)
        print("✅ Generation complete.")
        return response
    
def highlight_evidence_in_pdf(evidence, doc_path):
    if not evidence: return
    doc_highlights = {}
    for clause in evidence:
        doc_name = clause['document']
        if doc_name not in doc_highlights: doc_highlights[doc_name] = {}
        page_num = clause['page']
        if page_num not in doc_highlights[doc_name]: doc_highlights[doc_name][page_num] = []
        doc_highlights[doc_name][page_num].extend(clause['bboxes'])
    for doc_name, pages in doc_highlights.items():
        filepath = os.path.join(doc_path, doc_name)
        if not os.path.exists(filepath): continue
        doc = fitz.open(filepath)
        for page_num, bboxes in pages.items():
            page = doc.load_page(page_num - 1)
            for bbox in bboxes: page.add_highlight_annot(fitz.Rect(bbox))
            output_filename = f"EVIDENCE_{doc_name}_page_{page_num}.png"
            pix = page.get_pixmap(dpi=200); pix.save(output_filename)
            print(f"🖼️  Saved visual evidence to: {output_filename}")
        doc.close()

def select_pdf(doc_path):
    pdf_files = [f for f in os.listdir(doc_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No PDF files found in the '{doc_path}' directory.")
        return None
    print("\nPlease select a document to analyze:")
    for i, filename in enumerate(pdf_files):
        print(f"  [{i+1}] {filename}")
    while True:
        try:
            choice_str = input(f"Enter a number (1-{len(pdf_files)}): ")
            choice = int(choice_str)
            if 1 <= choice <= len(pdf_files):
                return os.path.join(doc_path, pdf_files[choice-1])
            else:
                print("Invalid number. Please try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid number from the list.")

def main():
    try:
        config = Config()
        if not os.path.exists(config.PDF_DOC_PATH): os.makedirs(config.PDF_DOC_PATH)
        if not os.path.exists(config.VECTOR_STORE_DIR): os.makedirs(config.VECTOR_STORE_DIR)
        
        selected_pdf_path = select_pdf(config.PDF_DOC_PATH)
        if not selected_pdf_path: return

        vector_store_path = os.path.join(config.VECTOR_STORE_DIR, f"vs_{os.path.basename(selected_pdf_path)}.faiss")

        processor = DocumentProcessor(chunk_limit=config.CHAR_CHUNK_LIMIT)
        chunks = processor.load_and_chunk_pdf(selected_pdf_path)

        vs_manager = VectorStoreManager(config)
        vector_store = vs_manager.create_or_load_store(chunks, vector_store_path, force_recreate=True)

        pipeline = RAGPipeline(config, vector_store)
        
        while True:
            user_query = input("\nEnter your query (or type 'exit' to quit): ")
            if user_query.lower() == 'exit': break
            if not user_query.strip(): continue

            print(f"\n--- EXECUTING QUERY: '{user_query}' ---")
            final_response_str = pipeline.query(user_query)
            
            response_json = json.loads(final_response_str)
            print("\n--- FINAL ANSWER ---")
            print(json.dumps(response_json, indent=2))
            
            highlight_evidence_in_pdf(response_json.get('evidence'), config.PDF_DOC_PATH)

    except Exception as e:
        import traceback
        print(f"\n--- A CRITICAL ERROR OCCURRED: {e} ---")
        traceback.print_exc()

if __name__ == "__main__":
    main()