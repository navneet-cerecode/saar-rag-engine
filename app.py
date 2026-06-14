import streamlit as st
import os
import fitz  # PyMuPDF
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http import models as rest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import re
from PIL import Image
import pytesseract

# Tell Python where your Windows Tesseract engine is installed:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- CONFIGURATION & SESSION STATE INITIALIZATION ---
st.set_page_config(page_title="Production RAG Engine", layout="wide")
st.title("PDF RAG Engine")

DB_PATH = "./qdrant_db"
COLLECTION_NAME = "pdf_knowledge_base"

@st.cache_resource
def initialize_models():
    """Cache models and database connections for stable CPU execution."""
    # 1. Initialize Database
    client = QdrantClient(path=DB_PATH)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME, 
            # Note: 768 is the correct size for the base-en model
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        
    # 2. Load Lightweight English Models explicitly onto the CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    reranker = CrossEncoder("BAAI/bge-reranker-base", device='cpu')
    llm = ChatGroq(
        temperature=0, 
        model_name="llama3-8b-8192", 
        groq_api_key=st.secrets["GROQ_API_KEY"] # In production, hide this in st.secrets!
    )
    
    return client, embeddings, reranker, llm

qdrant_client, embeddings_model, reranker_model, llm_engine = initialize_models()

# --- PHASE 1: LAYOUT-AWARE INGESTION ENGINE ---
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
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])
            for chunk in text_splitter.split_text(raw_text):
                processed_documents.append(Document(page_content=chunk, metadata={"page": page_num + 1, "type": "text"}))
                
    return processed_documents

# --- UI SIDEBAR: INGESTION & SUMMARIZER ---
with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload Target PDF", type=["pdf"])
    process_btn = st.button("Index Document", type="primary")

    if uploaded_file and process_btn:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.sidebar.markdown("---")
        status_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0.0)
        
        documents = extract_layout_aware_pdf(temp_path, progress_bar, status_text)
        
        status_text.text("Generating vectors (CPU Mode)...")
        progress_bar.progress(1.0)
        
        vector_store = QdrantVectorStore(
            client=qdrant_client, 
            collection_name=COLLECTION_NAME, 
            embedding=embeddings_model
        )
        vector_store.add_documents(documents)
        
        os.remove(temp_path)
        status_text.text("Indexing Complete!")
        st.sidebar.success(f"Indexed {len(documents)} blocks successfully!")

    st.markdown("---")
    st.header("2. Page Summarizer")
    target_page = st.number_input("Target Page Number", min_value=1, step=1)
    
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
                    <text>
                    {page_text}
                    </text>
                    Summary:"""
                    summary_result = llm_engine.invoke(summary_prompt)
                    
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    
                    st.session_state.messages.append({"role": "user", "content": f"Summarize page {target_page}."})
                    st.session_state.messages.append({"role": "assistant", "content": summary_result})
                    st.rerun() 
            else:
                st.error("Please index a document first.")

# --- UI MAIN INTERFACE: CHAT PIPELINE ---
st.markdown("### Ask specific questions about the document data")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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

                    <context>
                    {context}
                    </context>

                    <instruction_rules>
                    1. **Context Synthesis**: You are encouraged to connect relevant facts, data points, and concepts distributed across different context blocks to form a complete answer.
                    2. **Logical Deduction**: If the context does not state the answer verbatim but contains clear, undeniable premises that logically imply the answer, use safe technical deduction to bridge the gap.
                    3. **Acknowledge Limitations**: If the context provides partial information, answer using what is available, and clearly state what specific details are missing.
                    4. **Hard Fallback**: Only trigger the fallback if the context blocks are completely irrelevant to the topic of the query. If there is absolute zero relation, output EXACTLY: "[NO_DATA]: Insufficient context."
                    </instruction_rules>

                    User Query: {query}
                    <scratchpad>"""
                    
                    final_prompt = PromptTemplate.from_template(prompt_template).format(context=formatted_context, query=query)
                    raw_response = llm_engine.invoke(final_prompt)
                    
                    clean_answer = raw_response.split("</scratchpad>")[-1].strip() if "</scratchpad>" in raw_response else re.sub(r'<scratchpad>.*?</scratchpad>', '', raw_response, flags=re.DOTALL).strip()
                    if "[NO_DATA]" in clean_answer:
                        clean_answer = "Insufficient data in the provided document to answer this query."
                else:
                    clean_answer = "No relevant text found. Make sure the document is indexed."
            else:
                clean_answer = "Please upload and index a PDF document first."
                
        st.markdown(clean_answer)
        st.session_state.messages.append({"role": "assistant", "content": clean_answer})