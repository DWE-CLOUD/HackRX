import os
import json
import torch
import fitz
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import CrossEncoder

def get_device():
    if torch.cuda.is_available():
        print("✅ Using CUDA (NVIDIA GPU) for processing.")
        return 'cuda'
    if torch.backends.mps.is_available():
        print("✅ Using MPS (Apple Silicon GPU) for processing.")
        return 'mps'
    print("⚠️ No GPU detected. Processing will be on the CPU.")
    return 'cpu'

DEVICE = get_device()

def load_documents_with_visual_info(doc_path):
    all_docs = []
    for filename in os.listdir(doc_path):
        filepath = os.path.join(doc_path, filename)
        if filename.lower().endswith('.pdf'):
            try:
                doc = fitz.open(filepath)
                print(f"📄 Processing PDF: {filename}")
                for page_num, page in enumerate(doc):
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                for span in line["spans"]:
                                    text = span["text"]
                                    bbox = span["bbox"]
                                    metadata = {
                                        "source": filename,
                                        "page": page_num + 1,
                                        "bbox": list(bbox)
                                    }
                                    doc_obj = Document(page_content=text, metadata=metadata)
                                    all_docs.append(doc_obj)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return all_docs

def create_vector_store(doc_path="sample_docs", force_recreate=False):
    vector_store_path = "faiss_index_visual"
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", model_kwargs={'device': DEVICE}
    )

    if not force_recreate and os.path.exists(vector_store_path):
        print("✅ Vector store found. Loading from disk.")
        return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

    print("✨ Creating new vector store with visual info...")
    documents = load_documents_with_visual_info(doc_path)
    if not documents:
        raise ValueError("No documents were loaded. Check your 'sample_docs' directory.")
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_store_path)
    print(f"✅ Visual vector store created and saved.")
    return vector_store

def setup_visual_rag_chain(vector_store):
    llm = Ollama(model="gemma3:4b", format="json", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={'k': 15})
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device=DEVICE)

    prompt_template = """
    You are an expert AI. Based ONLY on the provided context, answer the user's query.
    Return a single, valid JSON object. For each source clause, you MUST include the 'bbox' and 'page' from its metadata.

    CONTEXT:
    {context}

    QUERY:
    {query}

    JSON OUTPUT FORMAT:
    {{
      "answer": "A direct answer.",
      "sourceClauses": [
        {{
          "clauseText": "The exact text from the context.",
          "document": "The source document name.",
          "page": "The page number of the source.",
          "bbox": [x0, y0, x1, y1]
        }}
      ]
    }}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}, BBox: {doc.metadata['bbox']}\nContent: {doc.page_content}"
            for doc in docs
        )
    
    def re_ranker(inputs):
        query = inputs['query']
        docs = inputs['retrieved_docs']
        pairs = [[query, doc.page_content] for doc in docs]
        scores = cross_encoder.predict(pairs)
        doc_scores = list(zip(docs, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_scores[:4]]

    rag_chain = (
        {
            "retrieved_docs": retriever,
            "query": RunnablePassthrough()
        }
        | RunnableLambda(re_ranker).with_config(run_name="ReRanker")
        | {
            "context": format_docs,
            "query": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def highlight_source_in_pdf(source_clauses, doc_path="sample_docs"):
    if not source_clauses:
        print("No source clauses found to highlight.")
        return

    doc_name = source_clauses[0]['document']
    filepath = os.path.join(doc_path, doc_name)
    
    doc = fitz.open(filepath)
    
    page_bboxes = {}
    for clause in source_clauses:
        page_num = clause['page']
        if page_num not in page_bboxes:
            page_bboxes[page_num] = []
        page_bboxes[page_num].append(clause['bbox'])
        
    output_files = []
    for page_num, bboxes in page_bboxes.items():
        page = doc.load_page(page_num - 1)
        for bbox in bboxes:
            highlight = fitz.Rect(bbox)
            page.draw_rect(highlight, color=(1, 1, 0), fill=(1, 1, 0), fill_opacity=0.4, overlay=True)
            
        output_filename = f"highlighted_{doc_name}_page_{page_num}.png"
        pix = page.get_pixmap(dpi=200)
        pix.save(output_filename)
        output_files.append(output_filename)
        print(f"✅ Highlighted visualization saved to: {output_filename}")

    doc.close()
    return output_files


if __name__ == "__main__":
    pdf_path = "sample_docs"
    if not os.path.exists(pdf_path): os.makedirs(pdf_path)
    sample_pdf_file = os.path.join(pdf_path, "Financial_Contract.pdf")

    if not os.path.exists(sample_pdf_file):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Financial Contract Agreement 2025", fontsize=20)
        page.insert_text((72, 120), "Clause 4.1: Standard car rental reimbursement is limited to $35 per day for a maximum of 30 days.")
        page.insert_text((72, 150), "Clause 4.2: For theft of a vehicle, rental reimbursement is upgraded to $55 per day, for up to 60 days.")
        page.insert_text((72, 180), "Clause 5.1: Paternity leave is granted for 15 working days for all full-time staff.")
        doc.save(sample_pdf_file)
        print(f"✅ Created sample PDF: {sample_pdf_file}")

    vector_store = create_vector_store(force_recreate=True)
    rag_chain = setup_visual_rag_chain(vector_store)

    print("\n--- Querying the Visual System --- 🚀")
    query = "What is the reimbursement for a stolen car?"
    print(f"❓ Query: {query}\n")

    response_str = rag_chain.invoke(query)
    
    try:
        response_json = json.loads(response_str)
        print("✅ LLM Response Received:\n")
        print(json.dumps(response_json, indent=2))
        
        print("\n--- Generating Visual Evidence ---")
        highlight_source_in_pdf(response_json.get('sourceClauses'))

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error processing LLM response: {e}\nRaw output:\n{response_str}")