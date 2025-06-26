# Multimodal RAG Chat Application

This project is a sophisticated, full-stack multimodal chat application that leverages a local-first AI stack to provide a powerful and private Retrieval-Augmented Generation (RAG) experience.

It features a modular, configurable RAG pipeline, a robust API-driven architecture, and an intuitive user interface built with Next.js and Tailwind CSS.

## 🌟 Key Features

-   **🖥️ Full-Stack Architecture**: A complete solution with a Next.js frontend, a Python backend, and a separate, advanced RAG API server.
-   **🤖 Advanced RAG Pipeline**: A modular and configurable RAG system that can be adapted for different retrieval strategies.
-   **🧩 Modular Retrieval**: Easily enable or disable different retrieval techniques like **Graph-based RAG** and **Reranking** through simple configuration changes.
-   **📤 Upload, Index, then Chat**: A robust and intuitive workflow. Users upload documents, explicitly trigger an indexing job, and only then can they chat with the newly ingested knowledge.
-   **🧠 Agentic Triage**: The system intelligently routes user queries. General questions are answered directly by an LLM, while specific ones trigger the RAG pipeline.
-   **🔒 100% Local & Private**: The entire stack, including LLMs and embedding models, runs locally using [Ollama](https://ollama.com/), ensuring your data never leaves your machine.
-   **📝 Session-Based Chat**: Persistent, session-based conversations managed by a SQLite database.

## 🛠️ Tech Stack

-   **Frontend**: Next.js, React, Tailwind CSS, Shadcn/ui
-   **Backend**: Python (standard library `http.server`)
-   **Advanced RAG System**: Python, LanceDB, PyMuPDF, `transformers`
-   **Local AI**: Ollama (for running LLMs like Llama 3, Qwen, etc.)
-   **Database**: SQLite

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   Node.js and npm/yarn
-   [Ollama](https://ollama.com/) installed and running.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd multimodal-rag
    ```

2.  **Set up the Backend & RAG System:**
    -   Install Python dependencies for the main backend:
      ```bash
      pip install -r backend/requirements.txt
      ```
    -   Install Python dependencies for the RAG system:
      ```bash
      pip install -r rag_system/requirements.txt
      ```

3.  **Set up the Frontend:**
    ```bash
    npm install
    ```

4.  **Pull Required Ollama Models:**
    The system is configured to use specific models. Pull them using Ollama:
    ```bash
    ollama pull qwen2.5vl:7b  # Or your model of choice for generation/VLM
    ollama pull qwen3-embedding-0.6b # For embeddings
    ```
    *Note: You can change the models used in `rag_system/main.py`.*

### Running the Application

The application consists of three main components that need to be running simultaneously: the **Frontend**, the **Backend**, and the **RAG API Server**.

1.  **Start the Advanced RAG API Server:**
    This server handles all the heavy lifting of indexing and retrieval.
    ```bash
    python -m rag_system.main api
    ```
    You should see output indicating it's running on port 8001.

2.  **Start the Main Backend Server:**
    This server handles sessions, database interactions, and communication with the frontend.
    ```bash
    python backend/server.py
    ```
    This will run on port 8000.

3.  **Start the Frontend Development Server:**
    ```bash
    npm run dev
    ```
    The application will be available at [http://localhost:3002](http://localhost:3002).

## 📄 Workflow: Upload, Index, Chat

1.  Open the application and start a "New Chat".
2.  Use the attachment icon to select the PDF files you want to work with.
3.  Upon selection, the files are automatically uploaded.
4.  The UI will then prompt you to **"Index Documents"**. The chat input will be disabled.
5.  Click the "Index Documents" button. The RAG server will process your files, extract text, and create vector embeddings.
6.  Once indexing is complete, the chat input will be enabled, and you can start asking questions about your documents.

## 🔧 Configuration & Modularity

The RAG pipeline is highly configurable via the `PIPELINE_CONFIGS` dictionary in `rag_system/main.py`.

### Enabling/Disabling Retrieval Modules

You can easily switch retrieval strategies on or off:

-   **Graph RAG**: Set `graph_rag["enabled"]` to `true` or `false`.
-   **Reranker**: Set `reranker["enabled"]` to `true` or `false`.

```python
# In rag_system/main.py
...
"retrieval": {
    "graph_rag": {
        "enabled": False, # <-- Toggle this
        "graph_path": "./index_store/graph/knowledge_graph.gml"
    },
    "reranker": {
        "enabled": False, # <-- Toggle this
        "model_name": "Qwen/Qwen3-Reranker-0.6B",
    },
...
```

This modularity allows you to experiment with different RAG techniques to see what works best for your use case.

## ⚡ Local Run Guide (updated 2025-06-17)

The stack has three separate processes that need to be running:

1. **Advanced RAG API** – heavy lifting (indexing / retrieval)
   * Port `8001`
   * Start from repository root
     ```bash
     python -m rag_system.api_server   # Ctrl-C to stop
     ```
   * First launch downloads / loads the embedding model defined in
     `rag_system/main.py`  (`BAAI/bge-small-en-v1.5` by default).  The first
     request can therefore take ~1 min.  Keep the server running so the model
     stays in memory.

2. **Legacy Backend (`server.py`)** – sessions, uploads, REST layer that the
   Next.js UI talks to
   * Port `8000`
     ```bash
     cd backend
     python server.py
     ```
   * Needs to be running **in addition to** the RAG API; otherwise the UI
     shows "Backend offline".

3. **Frontend (Next.js)**
   * Port `3000` (or whichever you choose via `PORT`)
     ```bash
     npm install
     npm run dev       # open http://localhost:3000
     ```

💡  *Alternative:*  run both Python servers in one process:
```bash
python combined_server.py        # starts :8000 and :8001
```

### One-time setup
```bash
# Python deps
pip install -r backend/requirements.txt -r rag_system/requirements.txt

# Node deps (UI)
npm install

# Ollama models (generation + embedding)
ollama pull qwen3:8b                 # generation
ollama pull qwen3-embedding-0.6b     # embeddings
ollama serve                         # in a separate terminal
```

### Typical workflow
1. "Create new index" → give it a name.
2. Upload PDFs (stored under `backend/shared_uploads/`).
3. Click **Build Index** – this triggers `IndexingPipeline.run(...)` on
   the RAG server.  If you see `TypeError: unexpected keyword argument 'documents'`
   make sure the RAG server has been restarted after pulling the latest
   code (the patched `documents` alias is required).
4. When the build completes, open a new chat session, attach the newly
   built index, and start chatting.
5. Toggle **"Stream phases"** in the chat footer to watch step-by-step
   reasoning events (analyze → retrieval → rerank → … → final answer).

### Troubleshooting
* **Port already in use** – kill stray processes:
  ```bash
  lsof -i :8000 ; kill <pid>
  lsof -i :8001 ; kill <pid>
  ```
* **Broken pipe / 500** from `/chat` or `/index` – normally means the RAG
  server crashed.  Check its terminal for the real traceback.  Common
  causes: missing LanceDB table (build the index first), model still
  loading, out-of-memory on MPS.
* **Embedding model** – change `embedding_model_name` in
  `rag_system/main.py` (or `rag_system/config.py`) if you want to switch
  away from the default BAAI BGE model.

Happy hacking! 🎉
