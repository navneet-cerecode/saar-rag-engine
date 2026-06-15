# S.A.A.R. Engine 
**Semantic Analysis & Automated Retrieval**

S.A.A.R. is a production-ready, layout-aware Retrieval-Augmented Generation (RAG) system. Built to process complex PDF documents, it features intelligent OCR fallbacks, local vector search, semantic reranking, and hardware-accelerated LLM inference via Groq to provide instantaneous, highly accurate document intelligence.

## Enterprise-Grade Features
* **Layout-Aware Ingestion:** Utilizes `PyMuPDF` to intelligently separate structural tables (converted to Markdown) from standard digital text to preserve contextual boundaries.
* **Automated OCR Fallback:** Integrates `Tesseract OCR` to dynamically scan and extract text from image-based PDF pages when digital text is absent.
* **Self-Healing Vector Database:** Employs a local, file-based `Qdrant` instance with automated corruption detection and wipe-and-rebuild memory mechanics.
* **Semantic Reranking:** Upgrades standard vector search by passing the top 20 chunks through a `Cross-Encoder` (`ms-marco-MiniLM-L-6-v2`) to logically score and extract the top 4 most contextually relevant blocks.
* **Hardware-Accelerated Inference:** Synthesizes context using Meta's `Llama-3.1-8b-instant` hosted on `Groq` LPUs for near-zero latency generation.

## Technology Stack
* **Frontend:** Streamlit (Custom Dark Mode UI)
* **Ingestion:** PyMuPDF (fitz), Tesseract OCR, Pillow
* **Orchestration:** LangChain
* **Embeddings & Reranking:** HuggingFace (`all-mpnet-base-v2`, Cross-Encoder)
* **Vector Store:** Qdrant (Local)
* **LLM Engine:** Groq (Llama 3.1)

## Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/saar-engine.git](https://github.com/yourusername/saar-engine.git)
cd saar-engine
```

**2. Install System Dependencies (Linux/Debian)**
*Required for the OCR fallback engine.*
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils
```

**3. Set up the Python Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**4. Configure Environment Variables**
Create a `.streamlit/secrets.toml` file in the root directory and add your Groq API key:
```toml
GROQ_API_KEY = "your_api_key_here"
```

**5. Launch the Engine**
```bash
streamlit run app.py
```

## Usage
1. Open the sidebar and upload any PDF document.
2. Click **Index Document** to trigger the parsing, chunking, and embedding pipeline.
3. Use the **Page Summarizer** to get instant synopses of specific pages.
4. Use the **Main Chat** or suggested prompt pills to query the entire knowledge base.

---

## The Architecture Diagram Flow
To make this project look incredibly professional on your GitHub or resume, create a visual flowchart. Here is the exact logical flow you should map out in a diagram tool (like Draw.io or Excalidraw):

**Phase 1: Ingestion & Processing (The Eyes)**
* `[Box 1]` **User** -> Uploads PDF Document.
* `[Box 2]` **PyMuPDF Engine** -> Scans page layout.
    * *Branch A:* Finds Tables -> Converts to Markdown.
    * *Branch B:* Finds Digital Text -> Extracts string.
    * *Branch C (Fallback):* Finds Scanned Images -> Routes to **Tesseract OCR**.
* `[Box 3]` **LangChain Splitter** -> Chunks combined text into 1,500-character overlapping blocks.

**Phase 2: Embedding & Storage (The Memory)**
* `[Box 4]` **HuggingFace (`mpnet-v2`)** -> Converts text chunks into 768-dimensional vectors.
* `[Box 5]` **Local Qdrant DB** -> Stores vectors securely on disk (with Self-Healing mechanism).

**Phase 3: Retrieval & Reranking (The Search)**
* `[Box 6]` **User Query** -> Passed into the chat UI.
* `[Box 7]` **Vector Search** -> Qdrant retrieves the Top 20 mathematically similar chunks.
* `[Box 8]` **Cross-Encoder Reranker** -> Evaluates the Top 20 against the query logically, filtering down to the absolute Top 4 chunks.

**Phase 4: Synthesis (The Brain)**
* `[Box 9]` **LangChain Prompt Template** -> Merges the User Query with the Top 4 Context Blocks.
* `[Box 10]` **Groq API (Llama 3.1)** -> Reads the prompt, executes inference, and generates the final markdown answer.
* `[Box 11]` **Streamlit UI** -> Displays clean text to the user.
