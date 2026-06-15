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
