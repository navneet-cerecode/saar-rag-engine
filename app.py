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

def inject_custom_css():
    """Inject custom CSS to hide default Streamlit elements and create a minimalist dark UI."""
    st.markdown("""
        <style>
        /* 1. Snipe the right-side toolbar (Menu, Deploy) completely out of the DOM */
        [data-testid="stToolbar"] { 
            display: none !important; 
        }
        
        /* 2. Remove the footer */
        footer { 
            display: none !important; 
        }
        
        /* 3. Make the header transparent but leave its structural behavior alone */
        header { 
            background: transparent !important; 
        }
        
        /* Widen the main container and remove top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
            max-width: 850px;
        }
        
        /* Style the chat input container to float cleanly */
        .stChatInputContainer {
            padding-bottom: 20px;
            background-color: transparent !important;
        }
        
        /* Centered Hero Section */
        .hero-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #ffffff;
        }
        .hero-subtitle {
            text-align: center;
            color: #a0aec0;
            font-size: 1.1rem;
            margin-bottom: 3rem;
        }
        
        /* Custom Pill Buttons for suggestions */
        div[data-testid="stButton"] button {
            border-radius: 20px;
            border: 1px solid #4a5568;
            background-color: transparent;
            color: #e2e8f0;
            padding: 0.5rem 1rem;
            transition: all 0.2s;
        }
        div[data-testid="stButton"] button:hover {
            border-color: #cbd5e0;
            color: #ffffff;
            background-color: rgba(255, 255, 255, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_models():
    """Cache models and database connections for stable CPU execution."""
    try:
        client = QdrantClient(path=DB_PATH)
    except RuntimeError:
        print("Corrupted database detected. Executing self-healing wipe...")
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        client = QdrantClient(path=DB_PATH) 
        
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME, 
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    
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

        tabs = page.find_tables()
        table_rects = [t.bbox for t in tabs]
        
        for t in tabs:
            try:
                df = t.to_pandas()
                table_md = f"\n### Table on Page {page_num + 1}\n" + df.to_markdown(index=False) + "\n"
                processed_documents.append(Document(page_content=table_md, metadata={"page": page_num + 1, "type": "table"}))
            except Exception:
                continue

        blocks = page.get_text("blocks", sort=True)
        page_text_segments = []
        for b in blocks:
            x0, y0, x1, y1, text, _, _ = b
            in_table = any(x0 >= r[0] and y0 >= r[1] and x1 <= r[2] and y1 <= r[3] for r in table_rects)
            if not in_table and text.strip():
                page_text_segments.append(text.strip())

        raw_text = "\n\n".join(page_text_segments)

        if not raw_text.strip():
            if status_text:
                status_text.text(f"Scanning image/scan on Page {page_num + 1} via OCR...")
            
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            raw_text = pytesseract.image_to_string(img)
            
            if raw_text.strip():
                raw_text = f"[OCR Scanned Content]\n" + raw_text

        if raw_text.strip():
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300, separators=["\n\n", "\n", " ", ""])
            for chunk in text_splitter.split_text(raw_text):
                processed_documents.append(Document(page_content=chunk, metadata={"page": page_num + 1, "type": "text"}))
                
    return processed_documents

# === APPLICATION UI ===
st.set_page_config(page_title="S.A.A.R. Engine", page_icon="✨", layout="wide")
inject_custom_css()

with st.spinner("Waking up AI models..."):
    qdrant_client, embeddings_model, reranker_model, llm_engine = initialize_models()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; padding-top: 10px;'>S.A.A.R. Data Hub</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; font-size: 0.9rem;'>Upload documents to the knowledge base.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf", label_visibility="collapsed")
    
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if st.button("Index Document", type="primary", use_container_width=True):
            with st.spinner("Extracting text and tables..."):
                documents = extract_layout_aware_pdf(temp_path, progress_bar, status_text)
                
            if documents:
                with st.spinner("Wiping old memory & generating embeddings..."):
                    if qdrant_client.collection_exists(COLLECTION_NAME):
                        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
                        
                    qdrant_client.create_collection(
                        collection_name=COLLECTION_NAME, 
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                    
                    vector_store = QdrantVectorStore(
                        client=qdrant_client, collection_name=COLLECTION_NAME, embedding=embeddings_model
                    )
                    vector_store.add_documents(documents)
                    
                st.toast("Document successfully indexed!", icon="✅")
                status_text.empty()
                progress_bar.empty()
            else:
                st.toast("Could not extract text from document.", icon="❌")
                
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    st.divider()
    
    st.header("Page Summarizer")
    target_page = st.number_input("Enter Page Number:", min_value=1, step=1)
    
    if st.button("Summarize Page", use_container_width=True):
        with st.spinner(f"Summarizing page {target_page}..."):
            if qdrant_client.collection_exists(COLLECTION_NAME):
                page_filter = rest.Filter(
                    must=[rest.FieldCondition(key="metadata.page", match=rest.MatchValue(value=target_page))]
                )
                records, _ = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME, scroll_filter=page_filter, limit=50, with_payload=True
                )
                
                if not records:
                    st.toast(f"No text found on Page {target_page}.", icon="⚠️")
                else:
                    page_text = "\n\n".join([record.payload.get("page_content", "") for record in records])
                    summary_prompt = f"""You are an expert technical synthesizer. Summarize the following text from Page {target_page}.

--- PAGE TEXT ---
{page_text}
--- END TEXT ---

Summary:"""
                    summary_result = llm_engine.invoke(summary_prompt)
                    
                    if hasattr(summary_result, 'content'):
                        clean_summary = summary_result.content
                    elif isinstance(summary_result, dict) and 'content' in summary_result:
                        clean_summary = summary_result['content']
                    else:
                        clean_summary = str(summary_result)
                        
                    clean_summary = clean_summary.strip()
                    
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    
                    st.session_state.messages.append({"role": "user", "content": f"Summarize page {target_page}.", "avatar": "🧑‍💻"})
                    st.session_state.messages.append({"role": "assistant", "content": f"**Page {target_page} Summary:**\n\n{clean_summary}", "avatar": "✨"})
                    st.rerun() 
            else:
                st.toast("Please index a document first.", icon="⚠️")

# --- MAIN CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Logic to handle prompt suggestion clicks
def set_query(query_text):
    st.session_state.suggestion_query = query_text

# 1. Capture User Input (from text box or suggestion pill)
user_query = st.chat_input("Ask a question about your document...")
if "suggestion_query" in st.session_state:
    user_query = st.session_state.suggestion_query
    del st.session_state.suggestion_query

# 2. Append user query to history immediately if it exists
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query, "avatar": "🧑‍💻"})

# 3. Render the UI
if not st.session_state.messages:
    # Empty State Welcome Screen
    st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>S.A.A.R. Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Upload a document in the sidebar to begin analysis.</div>", unsafe_allow_html=True)
    
    # Suggested Prompt Pills
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        st.button("📄 What is this document about?", on_click=set_query, args=("What is the overall summary of this document?",), use_container_width=True)
    with col2:
        st.button("🔍 Find key methodologies", on_click=set_query, args=("Extract the key methodologies and techniques mentioned.",), use_container_width=True)
    with col3:
        st.button("📊 Extract numerical data", on_click=set_query, args=("List the important metrics, numbers, and data points found.",), use_container_width=True)
    with col4:
        st.button("📝 Summarize conclusions", on_click=set_query, args=("What are the final conclusions or results?",), use_container_width=True)

else:
    # Display Chat History with Custom Avatars
    for message in st.session_state.messages:
        avatar = message.get("avatar", "🧑‍💻" if message["role"] == "user" else "✨")
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# 4. Process the Assistant's Response
if user_query:
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Retrieving context..."):
            if qdrant_client.collection_exists(COLLECTION_NAME):
                vector_store = QdrantVectorStore(
                    client=qdrant_client, collection_name=COLLECTION_NAME, embedding=embeddings_model
                )
                
                initial_docs = vector_store.similarity_search(user_query, k=20)
                
                if initial_docs:
                    pairs = [[user_query, doc.page_content] for doc in initial_docs]
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
                    
                    final_prompt = PromptTemplate.from_template(prompt_template).format(context=formatted_context, query=user_query)
                    raw_response = llm_engine.invoke(final_prompt)
                    
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
        st.session_state.messages.append({"role": "assistant", "content": clean_answer, "avatar": "✨"})