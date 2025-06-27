# Multimodal RAG Project Overview

This document provides a comprehensive overview of the Multimodal RAG (Retrieval-Augmented Generation) project. It is intended for engineers who need to understand the system's architecture, components, and workflows.

## 1. System Architecture

The project is a sophisticated full-stack application designed to answer questions based on a collection of documents (PDFs, etc.). It follows a distributed architecture composed of two main services:

1.  **Backend Server (`localGPT`):** A Python-based server responsible for session management, file uploads, and orchestrating the overall workflow. It acts as the primary interface for the frontend.
2.  **Advanced RAG API Server:** A dedicated FastAPI server that houses the complex RAG pipeline. It handles the heavy lifting of document processing, indexing, retrieval, and answer synthesis.

The two servers communicate via REST API calls. This separation of concerns allows the resource-intensive RAG processes to be scaled and managed independently of the main application logic.

### Architectural Diagram

```mermaid
graph TD
    subgraph Frontend
        A[React UI]
    end

    subgraph Backend Server (localGPT)
        B[Python/Flask]
    end

    subgraph Advanced RAG API (FastAPI)
        C[Indexing Pipeline]
        D[Retrieval Pipeline]
    end

    subgraph Data Stores
        E[Vector Store - LanceDB]
        F[Keyword Store - BM25 Index]
    end

    A -- HTTP API --> B
    B -- /index --> C
    B -- /chat --> D

    C -- Writes --> E
    C -- Writes --> F

    D -- Reads --> E
    D -- Reads --> F

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style D fill:#ccf,stroke:#333,stroke-width:2px
```

--- 

## 2. Data Ingestion & Indexing Pipeline

The indexing pipeline is responsible for converting raw documents into searchable data stores (vector and keyword). This process is triggered by the `/index` endpoint of the Advanced RAG API.

The pipeline executes the following steps in sequence:

1.  **Document Conversion:**
    *   Input documents (currently PDFs) are processed by the `PDFConverter`, which leverages the `docling` library.
    *   Forcing OCR on macOS is enabled to ensure text is extracted even from image-based PDFs, producing clean Markdown output for each page.

2.  **Chunking:**
    *   The generated Markdown text is passed to a `MarkdownRecursiveChunker`.
    *   This component intelligently splits the text into smaller, semantically coherent chunks based on Markdown headers, ensuring that related content stays together. Each chunk retains metadata, including its source document and page number.

3.  **BM25 Indexing:**
    *   The **original, unprocessed text** from the chunks is used to build a BM25 index.
    *   This is a crucial step for keyword-based search. An instance of `BM25Indexer` handles the creation and saving of the index to a `.pkl` file. This ensures that keyword searches are performed on the original text, not the enriched version.

4.  **Contextual Enrichment (for Vector Search):**
    *   The `ContextualEnricher` processes each chunk to improve its representation for vector-based retrieval.
    *   For each chunk, it looks at the preceding and succeeding chunks (a configurable `window_size`) to create a summary.
    *   This summary is then prepended to the chunk's text. The original text is preserved in the chunk's metadata.
    *   This technique provides the embedding model with richer context, leading to more accurate vector representations.

5.  **Vector Embedding:**
    *   The `EmbeddingGenerator`, equipped with a `Qwen/Qwen3-Embedding-0.6B` model, converts the **enriched** text of each chunk into a high-dimensional vector.

6.  **Vector Indexing:**
    *   The generated embeddings and their corresponding chunks (with enriched text and metadata) are stored in a `LanceDB` table using the `VectorIndexer`. LanceDB is an embedded, serverless vector database that is highly efficient for this purpose.

7.  **Knowledge Graph Extraction (Optional):**
    *   If enabled in the configuration, a `GraphExtractor` uses a powerful LLM to identify entities and relationships within the chunks.
    *   This information is used to construct a `networkx` knowledge graph, which is then saved as a GML file for potential future graph-based retrieval methods.

--- 

## 3. Retrieval & Synthesis Pipeline

This pipeline is activated by a user query via the `/chat` endpoint. Its goal is to find the most relevant information in the indexed data and use it to generate a human-like answer.

1.  **Hybrid Retrieval:**
    *   The system uses a hybrid approach to fetch candidate documents, combining keyword-based and semantic search for robust results.
    *   **Dense Retrieval:** The `MultiVectorRetriever` takes the user's query, generates a `Qwen` embedding for it, and performs a vector similarity search in the `LanceDB` table.
    *   **BM25 Retrieval:** Simultaneously, the `BM25Retriever` uses the keyword index to find chunks that are textually similar to the query.
    *   The results from both retrievers are collected, and duplicates are removed.

2.  **Reranking:**
    *   The combined list of candidate chunks is passed to the ColBERT reranker via the `rerankers` library.
    *   This specialized model re-evaluates the relevance of each chunk against the original query, assigning a new score. This step is crucial for filtering out "near miss" documents that were retrieved but aren't truly relevant.
    *   Only the top `k` chunks (a configurable number) with the highest reranking scores are kept for the final step.

3.  **Answer Synthesis:**
    *   The text from the final, reranked chunks is concatenated into a single body of "facts".
    *   This context, along with the original query, is inserted into a prompt that instructs a powerful generative LLM (e.g., `qwen2.5vl:7b`) to synthesize a final, comprehensive answer.
    *   The `_synthesize_final_answer` method handles this interaction, ensuring that if no relevant facts are found, the model responds appropriately.
    *   The final answer and the source documents are then sent back to the user.

---

## 4. How to Run the Project

To run the full application, you need to start both the backend and the frontend servers.

### Prerequisites

- Python 3.9+ with `pip`
- Node.js and `npm` (or `yarn`)
- An Ollama instance running with the required models (see `config.yml`).

### 1. Backend & RAG Server

The Python environment contains both the `localGPT` backend and the Advanced RAG API server.

**Installation:**
```bash
# Navigate to the RAG system directory
cd rag_system

# Install Python dependencies
pip install -r requirements.txt
```

**Running the Servers:**
The `localGPT` backend and the RAG API server must be run separately in two different terminals.

*   **Terminal 1: Start the Backend Server**
    ```bash
    # From the project root
    python backend/server.py
    ```
    This will start the session management server, typically on port `8000`.

*   **Terminal 2: Start the Advanced RAG API Server**
    ```bash
    # From the project root
    python -m rag_system.main api
    ```
    This will start the RAG pipeline server, typically on port `8001`.

### 2. Frontend

The frontend is a React application built with Next.js.

**Installation:**
```bash
# From the project root
npm install
```

**Running the Development Server:**
```bash
# From the project root
npm run dev
```
This will start the frontend server, typically on port `3000`. You can now access the application in your browser.

--- 

## 5. Configuration

The project's configuration is managed through a Python dictionary named `PIPELINE_CONFIGS` located in `rag_system/main.py`. This centralized approach allows for easy tuning of the various components.

The main configuration objects are:

-   **`OLLAMA_CONFIG`**: Specifies the connection details for the Ollama server and defines which models to use for generation and vision-language tasks.

-   **`PIPELINE_CONFIGS`**: A nested dictionary containing the settings for different operational modes (`indexing`, `retrieval`, `default`).

Key configuration options within `PIPELINE_CONFIGS`:

-   **`storage`**: Defines all file paths and database names, including the LanceDB URI, the BM25 index path, and the table names for text and images.
-   **`retrievers`**: A dictionary to enable or disable different retrieval methods (`dense`, `bm25`, `graph`) and set their specific parameters (e.g., index names).
-   **`contextual_enricher`**: Contains settings for the context-enrichment step, such as `enabled` and `window_size`.
-   **`embedding_model_name`**: The Hugging Face identifier for the model used to generate vector embeddings (e.g., `Qwen/Qwen3-Embedding-0.6B`).
-   **`reranker`**: Settings for the reranking step, including the model name (`Qwen/Qwen3-Reranker-0.6B`) and the number of documents to keep (`top_k`).
-   **`retrieval_k`**: The number of initial documents to retrieve from the database before reranking.

--- 