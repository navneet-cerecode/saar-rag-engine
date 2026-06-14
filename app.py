import streamlit as st
import os
import shutil
import fitz
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models as rest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# Dynamically locate the Tesseract executable path for the current OS
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print("Warning: Tesseract executable not found in system path.")

# Configuration
COLLECTION_NAME = "saar_documents"
DB_PATH = "qdrant_db"

@st.cache_resource
def initialize_models():
    """Cache models and database connections for stable CPU execution."""
    
    # 1. Initialize Database with Self-Healing
    try:
        client = QdrantClient(path=DB_PATH)
    except RuntimeError:
        # If the server crashes trying to read an incompatible/corrupted DB, wipe it clean
        print("Corrupted database detected. Executing self-healing wipe...")
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        client = QdrantClient(path=DB_PATH) # Recreate fresh
        
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME, 
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        
    # 2. Load Lightweight English Models explicitly onto the CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    
    # 3. Initialize Groq Llama 3.1 
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.1-8b-instant",  
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )
    
    return client, embeddings, reranker, llm

def extract_layout_aware_pdf(file_path, progress_bar=None, status_text=None):
    doc = fitz.open(file_path)
    processed_documents = []
    total_pages = len(doc)
    
    for page_num, page in enumerate(doc):
        if progress_bar and status_text:
            progress_bar.progress((page_num + 1) / total_pages)
            status_text.text(f"Parsing Page {page_num + 1} of {total_pages}...")

        # 1. Extract structural tables first
        tabs = page.find_tables()
        table_rects = [t.bbox for t in tabs]
        
        for t in tabs:
            try:
                df = t.to_pandas()
                table_md = f"\n### Table on Page {page_num + 1}\n" + df.to_markdown(index=False) + "\n"
                processed_documents.append(Document(page_content=table_md, metadata={"page": page_num + 1, "type": "table"}))
            except Exception:
                continue

        # 2. Extract standard digital text
        blocks = page.get_text("blocks", sort=True)
        page_text_segments = []
        for b in blocks:
            x0, y0, x1, y1, text, _, _ = b
            in_table = any(x0 >= r[0] and y0 >= r[1] and x1 <= r[2] and y1 <= r[3] for r in table_rects)
            if not in_table and text.strip():
                page_text_segments.append(text.strip())

        raw_text = "\n\n".join(page_text_segments)

        # 3. OCR FALLBACK CRITERIA: If digital text is blank but the page has pixels/images
        if not raw_text.strip():
            if status_text:
                status_text.text(f"Scanning image/scan on Page {page_num + 1} via OCR...")
            
            # Render the PDF page directly to a high-resolution image in-memory
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR accuracy
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Run OCR
            raw_text = pytesseract.image_to_string(img)
            
            if raw_text.strip():
                raw_text = f"[OCR Scanned Content]\n" + raw_text

        # 4. Chunk and save what we found
        if raw_text.strip():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])
            for chunk in text_splitter.split_text(raw_text):
                processed_documents.append(Document(page_content=chunk, metadata={"page": page_num + 1, "type": "text"}))
                
    return processed_documents

# === APPLICATION UI ===
st.set_page_config(page_title="S.A.A.R. Engine", layout="wide")
st.title("S.A.A.R. | Semantic Analysis & Automated Retrieval")

# Load models
with st.spinner("Waking up AI models..."):
    qdrant_client, embeddings_model, reranker_model, llm_engine = initialize_models()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Document Processing")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if st.button("Index Document"):
            with st.spinner("Extracting text and tables..."):
                documents = extract_layout_aware_pdf(temp_path, progress_bar, status_text)
                
            if documents:
                with st.spinner("Wiping old memory and generating new embeddings..."):
                    # 1. WIPE THE OLD MEMORY
                    if qdrant_client.collection_exists(COLLECTION_NAME):
                        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
                        
                    # 2. CREATE A FRESH, EMPTY DATABASE
                    qdrant_client.create_collection(
                        collection_name=COLLECTION_NAME, 
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                    
                    # 3. ADD THE NEW PDF
                    vector_store = QdrantVectorStore(
                        client=qdrant_client, collection_name=COLLECTION_NAME, embedding=embeddings_model
                    )
                    vector_store.add_documents(documents)
                    
                st.success("Document successfully indexed! Old memory wiped.")
            else:
                st.error("Could not extract any text from the document.")
                
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    st.divider()
    st.header("Page Summarizer")
    target_page = st.number_input("Enter Page Number to Summarize:", min_value=1, step=1)
    
    if st.button("Summarize Specific Page"):
        with st.spinner(f"Fetching and summarizing page {target_page}..."):
            if qdrant_client.collection_exists(COLLECTION_NAME):
                page_filter = rest.Filter(
                    must=[rest.FieldCondition(key="metadata.page", match=rest.MatchValue(value=target_page))]
                )
                records, _ = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME, scroll_filter=page_filter, limit=50, with_payload=True
                )
                
                if not records:
                    st.error(f"No text found on Page {target_page}.")
                else:
                    page_text = "\n\n".join([record.payload.get("page_content", "") for record in records])
                    summary_prompt = f"""You are an expert technical synthesizer. Summarize the following text from Page {target_page}.

--- PAGE TEXT ---
{page_text}
--- END TEXT ---

Summary:"""
                    summary_result = llm_engine.invoke(summary_prompt)
                    
                    # Robust extraction of the clean text content from the LangChain response object
                    if hasattr(summary_result, 'content'):
                        clean_summary = summary_result.content
                    elif isinstance(summary_result, dict) and 'content' in summary_result:
                        clean_summary = summary_result['content']
                    else:
                        clean_summary = str(summary_result)
                        
                    clean_summary = clean_summary.strip()
                    
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    
                    st.session_state.messages.append({"role": "user", "content": f"Summarize page {target_page}."})
                    st.session_state.messages.append({"role": "assistant", "content": clean_summary})
                    st.rerun() 
            else:
                st.error("Please index a document first.")

# --- MAIN CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a specific question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context..."):
            if qdrant_client.collection_exists(COLLECTION_NAME):
                vector_store = QdrantVectorStore(
                    client=qdrant_client, collection_name=COLLECTION_NAME, embedding=embeddings_model
                )
                
                initial_docs = vector_store.similarity_search(query, k=20)
                
                if initial_docs:
                    pairs = [[query, doc.page_content] for doc in initial_docs]
                    scores = reranker_model.predict(pairs)
                    reranked_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)[:4]
                    relevant_chunks = [doc for doc, score in reranked_docs]
                    
                    formatted_context = ""
                    for i, doc in enumerate(relevant_chunks):
                        formatted_context += f"\n--- [Block {i+1} | Page: {doc.metadata.get('page')}] ---\n{doc.page_content}\n"
                    
                    prompt_template = """You are an expert technical intelligence and synthesis engine. Your goal is to provide a helpful, comprehensive, and accurate answer to the user's query using the provided context blocks.

--- CONTEXT ---
{context}
--- END CONTEXT ---

INSTRUCTION RULES:
1. Context Synthesis: Connect relevant facts, data points, and concepts distributed across different context blocks to form a complete answer.
2. Logical Deduction: If the context does not state the answer verbatim but contains clear, undeniable premises that logically imply the answer, use safe technical deduction to bridge the gap.
3. Acknowledge Limitations: If the context provides partial information, answer using what is available, clearly stating what details are missing.
4. Hard Fallback: If there is absolute zero relation to the text, output EXACTLY: "[NO_DATA]"

User Query: {query}
Answer:"""
                    
                    final_prompt = PromptTemplate.from_template(prompt_template).format(context=formatted_context, query=query)
                    raw_response = llm_engine.invoke(final_prompt)
                    
                    # Extract text safely whether Groq returns an AIMessage object or a raw string
                    if hasattr(raw_response, 'content'):
                        response_text = raw_response.content
                    elif isinstance(raw_response, dict) and 'content' in raw_response:
                        response_text = raw_response['content']
                    else:
                        response_text = str(raw_response)
                        
                    clean_answer = response_text.strip()
                    
                    if "[NO_DATA]" in clean_answer:
                        clean_answer = "The document mentions topics related to your query, but lacks specific details to give a definitive answer. Could you rephrase or specify a page number?"
                else:
                    clean_answer = "No relevant text found. Make sure the document is indexed."
            else:
                clean_answer = "Please upload and index a PDF document first."
                
        st.markdown(clean_answer)
        st.session_state.messages.append({"role": "assistant", "content": clean_answer})