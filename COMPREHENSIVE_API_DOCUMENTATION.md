# Comprehensive API Documentation - RAG System

Complete technical overview of all API endpoints, data structures, and system interactions.

## System Architecture Overview

The RAG system consists of 4 main components:

1. Frontend (Next.js) - Port 3000
2. Backend Server - Port 8000 (Main API Gateway) 
3. RAG API Server - Port 8001 (Advanced RAG Processing)
4. Ollama Server - Port 11434 (LLM Service)

## Data Flow Architecture

Frontend (3000) -> Backend Server (8000) -> Direct LLM or RAG Pipeline -> Ollama (11434)

The backend implements a dual-layer routing system for optimization.

## Smart Routing System

**Layer 1 (Speed Optimization)**: Backend routes queries to either Direct LLM or RAG Pipeline
**Layer 2 (Intelligence Optimization)**: RAG API routes queries within the RAG pipeline

---

## Backend Server APIs (Port 8000)

### Health & System APIs

#### GET /health
**Purpose**: System health check and status monitoring
**Response**:
```json
{
  "status": "ok",
  "ollama_running": true,
  "available_models": ["qwen3:8b", "nomic-embed-text"],
  "database_stats": {
    "total_sessions": 25,
    "total_messages": 148,
    "most_used_model": "qwen3:8b"
  }
}
```

#### GET /models
**Purpose**: Get categorized models from Ollama and HuggingFace
**Response**:
```json
{
  "generation_models": ["qwen3:8b", "llama3.2:latest"],
  "embedding_models": ["nomic-embed-text", "Qwen/Qwen3-Embedding-4B"]
}
```

### Session Management APIs

#### GET /sessions
**Purpose**: List all chat sessions
**Response**:
```json
{
  "sessions": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "Discussion about AI",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T11:45:00Z",
      "model_used": "qwen3:8b",
      "message_count": 5
    }
  ],
  "total": 1
}
```

#### POST /sessions
**Purpose**: Create new chat session
**Payload**:
```json
{
  "title": "New Chat",
  "model": "qwen3:8b"
}
```
**Response**:
```json
{
  "session": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "New Chat",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "model_used": "qwen3:8b",
    "message_count": 0
  }
}
```

#### GET /sessions/{sessionId}
**Purpose**: Get session details with message history
**Response**:
```json
{
  "session": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "Discussion about AI",
    "message_count": 5
  },
  "messages": [
    {
      "id": "msg-001",
      "content": "What is artificial intelligence?",
      "sender": "user",
      "timestamp": "2024-01-15T10:30:00Z",
      "metadata": {}
    },
    {
      "id": "msg-002",
      "content": "Artificial intelligence is...",
      "sender": "assistant",
      "timestamp": "2024-01-15T10:31:00Z"
    }
  ]
}
```

#### POST /sessions/{sessionId}/messages
**Purpose**: Send message in session context with smart routing
**Payload**:
```json
{
  "message": "What does the document say about AI?",
  "model": "qwen3:8b",
  "compose_sub_answers": true,
  "query_decompose": true,
  "ai_rerank": true,
  "context_expand": true,
  "verify": true,
  "retrieval_k": 20,
  "context_window_size": 1,
  "reranker_top_k": 10,
  "search_type": "hybrid",
  "dense_weight": 0.7
}
```
**Response**:
```json
{
  "response": "According to the document, artificial intelligence...",
  "session": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "message_count": 6
  },
  "source_documents": [
    {
      "content": "AI is defined as...",
      "metadata": {"page": 1, "source": "ai_paper.pdf"}
    }
  ],
  "used_rag": true
}
```

#### DELETE /sessions/{sessionId}
**Purpose**: Delete a session and all its messages
**Response**:
```json
{
  "message": "Session deleted successfully",
  "deleted_session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### Document Management APIs

#### POST /sessions/{sessionId}/upload
**Purpose**: Upload files to a session
**Payload**: multipart/form-data with files field
**Response**:
```json
{
  "message": "Uploaded 2 files",
  "uploaded_files": [
    {
      "filename": "document.pdf",
      "stored_path": "/path/to/stored/file"
    }
  ]
}
```

#### POST /sessions/{sessionId}/index
**Purpose**: Index uploaded documents for RAG
**Response**:
```json
{
  "message": "Documents indexed successfully"
}
```

#### GET /sessions/{sessionId}/documents
**Purpose**: Get uploaded documents for a session
**Response**:
```json
{
  "session": {
    "id": "123e4567-e89b-12d3-a456-426614174000"
  },
  "files": ["document.pdf", "paper.txt"],
  "file_count": 2
}
```

### Index Management APIs

#### GET /indexes
**Purpose**: List all available indexes
**Response**:
```json
{
  "indexes": [
    {
      "id": "idx-001",
      "name": "Research Papers",
      "description": "Collection of AI research papers",
      "created_at": "2024-01-15T10:00:00Z",
      "vector_table_name": "text_pages_idx-001"
    }
  ],
  "total": 1
}
```

#### POST /indexes
**Purpose**: Create a new index
**Payload**:
```json
{
  "name": "Research Papers",
  "description": "Collection of AI research papers",
  "metadata": {"domain": "AI"}
}
```
**Response**:
```json
{
  "index_id": "idx-001"
}
```

#### POST /indexes/{indexId}/upload
**Purpose**: Upload files to an index
**Payload**: multipart/form-data with files field
**Response**:
```json
{
  "message": "Uploaded 2 files",
  "uploaded_files": [
    {
      "filename": "document.pdf",
      "stored_path": "/path/to/stored/file"
    }
  ]
}
```

#### POST /indexes/{indexId}/build
**Purpose**: Build vector index from uploaded documents
**Payload**:
```json
{
  "latechunk": false,
  "doclingChunk": true,
  "chunkSize": 512,
  "chunkOverlap": 64,
  "retrievalMode": "hybrid",
  "windowSize": 2,
  "enableEnrich": true,
  "embeddingModel": "nomic-embed-text",
  "enrichModel": "qwen3:8b",
  "batchSizeEmbed": 50,
  "batchSizeEnrich": 25
}
```
**Response**:
```json
{
  "response": {
    "indexed_chunks": 245,
    "processing_time": "45.2s"
  },
  "indexing_config": {
    "chunk_size": 512,
    "embedding_model": "nomic-embed-text"
  }
}
```

#### POST /sessions/{sessionId}/indexes/{indexId}
**Purpose**: Link an index to a session
**Response**:
```json
{
  "message": "Index linked to session"
}
```

#### GET /sessions/{sessionId}/indexes
**Purpose**: Get indexes linked to a session
**Response**:
```json
{
  "indexes": [
    {
      "id": "idx-001",
      "name": "Research Papers"
    }
  ],
  "total": 1
}
```

#### DELETE /indexes/{indexId}
**Purpose**: Delete an index and its data
**Response**:
```json
{
  "message": "Index deleted successfully"
}
```

---

## RAG API Server (Port 8001)

The RAG API server handles advanced document processing and intelligent query routing.

### Chat APIs

#### POST /chat
**Purpose**: Process query through full RAG pipeline
**Payload**:
```json
{
  "query": "What are the main findings in the research papers?",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "table_name": "text_pages_idx-001",
  "compose_sub_answers": true,
  "query_decompose": true,
  "ai_rerank": true,
  "context_expand": true,
  "verify": true,
  "retrieval_k": 20,
  "context_window_size": 1,
  "reranker_top_k": 10,
  "search_type": "hybrid",
  "dense_weight": 0.7
}
```
**Response**:
```json
{
  "final_answer": "The main findings include...",
  "source_documents": [
    {
      "content": "Key finding: AI performance improved by 25%",
      "metadata": {
        "source": "paper1.pdf",
        "page": 5,
        "score": 0.95
      }
    }
  ],
  "processing_steps": {
    "query_decomposition": ["What are findings?", "Which papers?"],
    "retrieval_results": 15,
    "reranked_results": 10
  }
}
```

#### POST /chat/stream
**Purpose**: Stream RAG processing steps via Server-Sent Events
**Payload**: Same as /chat
**Response**: SSE stream with events:
```javascript
// Event types:
data: {"type": "analyze", "data": "Breaking down the query"}
data: {"type": "retrieval", "data": {"found": 15, "query": "findings"}}
data: {"type": "rerank", "data": {"top_results": 10}}
data: {"type": "complete", "data": {"final_answer": "..."}}
```

#### POST /index
**Purpose**: Build vector index from documents
**Payload**:
```json
{
  "file_paths": ["/path/to/doc1.pdf", "/path/to/doc2.txt"],
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "enable_latechunk": false,
  "enable_docling_chunk": true,
  "chunk_size": 512,
  "chunk_overlap": 64,
  "retrieval_mode": "hybrid",
  "window_size": 2,
  "enable_enrich": true,
  "embedding_model": "nomic-embed-text",
  "enrich_model": "qwen3:8b",
  "batch_size_embed": 50,
  "batch_size_enrich": 25
}
```
**Response**:
```json
{
  "indexed_chunks": 324,
  "processing_time": "67.3s",
  "table_name": "text_pages_123e4567",
  "embedding_dimensions": 768,
  "configuration": {
    "chunking_strategy": "docling",
    "chunk_size": 512,
    "embedding_model": "nomic-embed-text"
  }
}
```

#### GET /models
**Purpose**: Get available models for RAG processing
**Response**: Same format as backend /models

---

## Database Schema

### Tables Structure

#### sessions
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,                -- UUID
    title TEXT NOT NULL,               -- Session title
    created_at TEXT NOT NULL,          -- ISO timestamp
    updated_at TEXT NOT NULL,          -- ISO timestamp
    model_used TEXT NOT NULL,          -- Model identifier
    message_count INTEGER DEFAULT 0    -- Message count
)
```

#### messages
```sql
CREATE TABLE messages (
    id TEXT PRIMARY KEY,                -- UUID
    session_id TEXT NOT NULL,          -- Foreign key to sessions
    content TEXT NOT NULL,             -- Message content
    sender TEXT CHECK (sender IN ('user', 'assistant')),
    timestamp TEXT NOT NULL,           -- ISO timestamp
    metadata TEXT DEFAULT '{}',        -- JSON metadata
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
)
```

#### indexes
```sql
CREATE TABLE indexes (
    id TEXT PRIMARY KEY,               -- UUID
    name TEXT UNIQUE,                  -- Index name
    description TEXT,                  -- Description
    created_at TEXT,                   -- ISO timestamp
    updated_at TEXT,                   -- ISO timestamp
    vector_table_name TEXT,            -- LanceDB table name
    metadata TEXT                      -- JSON metadata
)
```

#### index_documents
```sql
CREATE TABLE index_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_id TEXT,                     -- Foreign key to indexes
    original_filename TEXT,            -- Original filename
    stored_path TEXT,                  -- Absolute storage path
    FOREIGN KEY(index_id) REFERENCES indexes(id)
)
```

#### session_indexes
```sql
CREATE TABLE session_indexes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,                   -- Foreign key to sessions
    index_id TEXT,                     -- Foreign key to indexes
    linked_at TEXT,                    -- ISO timestamp
    FOREIGN KEY(session_id) REFERENCES sessions(id),
    FOREIGN KEY(index_id) REFERENCES indexes(id)
)
```

---

## Configuration Parameters

### Retrieval Parameters
- **retrieval_k**: Number of documents to retrieve (default: 20)
- **context_window_size**: Context expansion window (default: 1)
- **reranker_top_k**: Top results after reranking (default: 10)
- **search_type**: Search strategy - 'hybrid', 'dense', 'sparse' (default: 'hybrid')
- **dense_weight**: Dense vs sparse weight in hybrid search (default: 0.7)

### Processing Flags
- **compose_sub_answers**: Enable sub-query composition (default: true)
- **query_decompose**: Enable query decomposition (default: true)
- **ai_rerank**: Enable AI-powered reranking (default: true)
- **context_expand**: Enable context window expansion (default: true)
- **verify**: Enable answer verification (default: true)

### Indexing Parameters
- **chunk_size**: Text chunk size (default: 512)
- **chunk_overlap**: Chunk overlap size (default: 64)
- **enable_latechunk**: Enable LaTeX-aware chunking (default: false)
- **enable_docling_chunk**: Enable Docling chunking (default: true)
- **retrieval_mode**: Retrieval strategy (default: 'hybrid')
- **window_size**: Context window size for enrichment (default: 2)
- **enable_enrich**: Enable context enrichment (default: true)
- **embedding_model**: Embedding model to use
- **enrich_model**: Model for context enrichment
- **batch_size_embed**: Embedding batch size (default: 50)
- **batch_size_enrich**: Enrichment batch size (default: 25)

---

## Data Flow Examples

### Simple Chat Flow
```
1. Frontend → POST /sessions/{id}/messages
2. Backend → _should_use_rag() → False (greeting)
3. Backend → _handle_direct_llm_query()
4. Backend → Ollama (thinking disabled)
5. Backend → Response (~1.3s)
```

### Document Query Flow
```
1. Frontend → POST /sessions/{id}/messages (with RAG params)
2. Backend → _should_use_rag() → True (document keywords)
3. Backend → _handle_rag_query()
4. Backend → RAG API POST /chat
5. RAG API → Agent.run() → Multi-step processing
6. RAG API → Response with source docs (~15-30s)
```

### Document Upload & Index Flow
```
1. Frontend → POST /sessions/{id}/upload (files)
2. Backend → Save files to shared_uploads/
3. Frontend → POST /sessions/{id}/index
4. Backend → RAG API POST /index
5. RAG API → Indexing Pipeline
6. RAG API → LanceDB storage
7. Backend → Link session to index
```

---

## Performance Benchmarks
- **Direct LLM**: ~1.3s average response time
- **RAG Pipeline**: 15-30s depending on complexity
- **File Upload**: ~2-5s per MB
- **Indexing**: ~1-3s per document page

## Development Setup

### Required Services
1. **Ollama Server**: `ollama serve` (port 11434)
2. **RAG API Server**: `python -m rag_system.api_server` (port 8001)  
3. **Backend Server**: `python backend/server.py` (port 8000)
4. **Frontend**: `npm run dev` (port 3000)

### Environment Configuration
- Models: qwen3:8b, nomic-embed-text
- Database: SQLite (chat_history.db)
- Vector Store: LanceDB (./lancedb/)
- File Storage: ./shared_uploads/

This comprehensive documentation covers all APIs, their interactions, data structures, and the complete system architecture. Each endpoint has been validated against the actual source code implementation.

- **Layer 1 (Speed Optimization)**: Backend routes queries to either Direct LLM or RAG Pipeline
- **Layer 2 (Intelligence Optimization)**: RAG API routes queries within the RAG pipeline

---

## 🛠️ Backend Server APIs (Port 8000)

### 📊 Health & System

#### `GET /health`
**Purpose**: System health check and status monitoring
**Response**:
```json
{
  "status": "ok",
  "ollama_running": true,
  "available_models": ["qwen3:8b", "nomic-embed-text"],
  "database_stats": {
    "total_sessions": 25,
    "total_messages": 148,
    "most_used_model": "qwen3:8b"
  }
}
```

#### `GET /models`
**Purpose**: Get categorized models from Ollama and HuggingFace
**Response**:
```json
{
  "generation_models": ["qwen3:8b", "llama3.2:latest"],
  "embedding_models": ["nomic-embed-text", "Qwen/Qwen3-Embedding-4B"]
}
```

### 💬 Chat APIs

#### `POST /chat` (Legacy)
**Purpose**: Direct chat without sessions (deprecated)
**Payload**:
```json
{
  "message": "Hello, how are you?",
  "model": "qwen3:8b",
  "conversation_history": [
    {"role": "user", "content": "Previous message"},
    {"role": "assistant", "content": "Previous response"}
  ]
}
```
**Response**:
```json
{
  "response": "I'm doing well, thank you for asking!",
  "model": "qwen3:8b",
  "message_count": 1
}
```

### 🗂️ Session Management

#### `GET /sessions`
**Purpose**: List all chat sessions
**Response**:
```json
{
  "sessions": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "title": "Discussion about AI",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-15T11:45:00Z",
      "model_used": "qwen3:8b",
      "message_count": 5
    }
  ],
  "total": 1
}
```

#### `POST /sessions`
**Purpose**: Create new chat session
**Payload**:
```json
{
  "title": "New Chat",
  "model": "qwen3:8b"
}
```
**Response**:
```json
{
  "session": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "New Chat",
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z",
    "model_used": "qwen3:8b",
    "message_count": 0
  }
}
```

#### `GET /sessions/{sessionId}`
**Purpose**: Get session details with message history
**Response**:
```json
{
  "session": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "title": "Discussion about AI",
    "message_count": 5
  },
  "messages": [
    {
      "id": "msg-001",
      "content": "What is artificial intelligence?",
      "sender": "user",
      "timestamp": "2024-01-15T10:30:00Z",
      "metadata": {}
    },
    {
      "id": "msg-002",
      "content": "Artificial intelligence is...",
      "sender": "assistant",
      "timestamp": "2024-01-15T10:31:00Z"
    }
  ]
}
```

#### `POST /sessions/{sessionId}/messages`
**Purpose**: Send message in session context with smart routing
**Payload**:
```json
{
  "message": "What does the document say about AI?",
  "model": "qwen3:8b",
  "compose_sub_answers": true,
  "query_decompose": true,
  "ai_rerank": true,
  "context_expand": true,
  "verify": true,
  "retrieval_k": 20,
  "context_window_size": 1,
  "reranker_top_k": 10,
  "search_type": "hybrid",
  "dense_weight": 0.7
}
```
**Response**:
```json
{
  "response": "According to the document, artificial intelligence...",
  "session": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "message_count": 6
  },
  "source_documents": [
    {
      "content": "AI is defined as...",
      "metadata": {"page": 1, "source": "ai_paper.pdf"}
    }
  ],
  "used_rag": true
}
```

#### `DELETE /sessions/{sessionId}`
**Purpose**: Delete a session and all its messages
**Response**:
```json
{
  "message": "Session deleted successfully",
  "deleted_session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### `GET /sessions/cleanup`
**Purpose**: Clean up empty sessions
**Response**:
```json
{
  "message": "Cleaned up 3 empty sessions",
  "cleanup_count": 3
}
```

### 📄 Document Management

#### `POST /sessions/{sessionId}/upload`
**Purpose**: Upload files to a session
**Payload**: `multipart/form-data` with `files` field
**Response**:
```json
{
  "message": "Uploaded 2 files",
  "uploaded_files": [
    {
      "filename": "document.pdf",
      "stored_path": "/path/to/stored/file"
    }
  ]
}
```

#### `POST /sessions/{sessionId}/index`
**Purpose**: Index uploaded documents for RAG
**Response**:
```json
{
  "message": "Documents indexed successfully"
}
```

#### `GET /sessions/{sessionId}/documents`
**Purpose**: Get uploaded documents for a session
**Response**:
```json
{
  "session": {
    "id": "123e4567-e89b-12d3-a456-426614174000"
  },
  "files": ["document.pdf", "paper.txt"],
  "file_count": 2
}
```

### 🗃️ Index Management

#### `GET /indexes`
**Purpose**: List all available indexes
**Response**:
```json
{
  "indexes": [
    {
      "id": "idx-001",
      "name": "Research Papers",
      "description": "Collection of AI research papers",
      "created_at": "2024-01-15T10:00:00Z",
      "vector_table_name": "text_pages_idx-001"
    }
  ],
  "total": 1
}
```

#### `POST /indexes`
**Purpose**: Create a new index
**Payload**:
```json
{
  "name": "Research Papers",
  "description": "Collection of AI research papers",
  "metadata": {"domain": "AI"}
}
```
**Response**:
```json
{
  "index_id": "idx-001"
}
```

#### `GET /indexes/{indexId}`
**Purpose**: Get index details
**Response**:
```json
{
  "id": "idx-001",
  "name": "Research Papers",
  "description": "Collection of AI research papers",
  "documents": [
    {
      "original_filename": "paper1.pdf",
      "stored_path": "/path/to/paper1.pdf"
    }
  ]
}
```

#### `POST /indexes/{indexId}/upload`
**Purpose**: Upload files to an index
**Payload**: `multipart/form-data` with `files` field
**Response**:
```json
{
  "message": "Uploaded 2 files",
  "uploaded_files": [
    {
      "filename": "document.pdf",
      "stored_path": "/path/to/stored/file"
    }
  ]
}
```

#### `POST /indexes/{indexId}/build`
**Purpose**: Build vector index from uploaded documents
**Payload**:
```json
{
  "latechunk": false,
  "doclingChunk": true,
  "chunkSize": 512,
  "chunkOverlap": 64,
  "retrievalMode": "hybrid",
  "windowSize": 2,
  "enableEnrich": true,
  "embeddingModel": "nomic-embed-text",
  "enrichModel": "qwen3:8b",
  "batchSizeEmbed": 50,
  "batchSizeEnrich": 25
}
```
**Response**:
```json
{
  "response": {
    "indexed_chunks": 245,
    "processing_time": "45.2s"
  },
  "indexing_config": {
    "chunk_size": 512,
    "embedding_model": "nomic-embed-text"
  }
}
```

#### `POST /sessions/{sessionId}/indexes/{indexId}`
**Purpose**: Link an index to a session
**Response**:
```json
{
  "message": "Index linked to session"
}
```

#### `GET /sessions/{sessionId}/indexes`
**Purpose**: Get indexes linked to a session
**Response**:
```json
{
  "indexes": [
    {
      "id": "idx-001",
      "name": "Research Papers"
    }
  ],
  "total": 1
}
```

#### `DELETE /indexes/{indexId}`
**Purpose**: Delete an index and its data
**Response**:
```json
{
  "message": "Index deleted successfully"
}
```

---

## 🧠 RAG API Server (Port 8001)

The RAG API server handles advanced document processing and intelligent query routing.

### 💭 Chat APIs

#### `POST /chat`
**Purpose**: Process query through full RAG pipeline
**Payload**:
```json
{
  "query": "What are the main findings in the research papers?",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "table_name": "text_pages_idx-001",
  "compose_sub_answers": true,
  "query_decompose": true,
  "ai_rerank": true,
  "context_expand": true,
  "verify": true,
  "retrieval_k": 20,
  "context_window_size": 1,
  "reranker_top_k": 10,
  "search_type": "hybrid",
  "dense_weight": 0.7
}
```
**Response**:
```json
{
  "final_answer": "The main findings include...",
  "source_documents": [
    {
      "content": "Key finding: AI performance improved by 25%",
      "metadata": {
        "source": "paper1.pdf",
        "page": 5,
        "score": 0.95
      }
    }
  ],
  "processing_steps": {
    "query_decomposition": ["What are findings?", "Which papers?"],
    "retrieval_results": 15,
    "reranked_results": 10
  }
}
```

#### `POST /chat/stream`
**Purpose**: Stream RAG processing steps via Server-Sent Events
**Payload**: Same as `/chat`
**Response**: SSE stream with events:
```javascript
// Event types:
data: {"type": "analyze", "data": "Breaking down the query"}
data: {"type": "retrieval", "data": {"found": 15, "query": "findings"}}
data: {"type": "rerank", "data": {"top_results": 10}}
data: {"type": "complete", "data": {"final_answer": "..."}}
```

### 🔧 Indexing APIs

#### `POST /index`
**Purpose**: Build vector index from documents
**Payload**:
```json
{
  "file_paths": ["/path/to/doc1.pdf", "/path/to/doc2.txt"],
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "enable_latechunk": false,
  "enable_docling_chunk": true,
  "chunk_size": 512,
  "chunk_overlap": 64,
  "retrieval_mode": "hybrid",
  "window_size": 2,
  "enable_enrich": true,
  "embedding_model": "nomic-embed-text",
  "enrich_model": "qwen3:8b",
  "batch_size_embed": 50,
  "batch_size_enrich": 25
}
```
**Response**:
```json
{
  "indexed_chunks": 324,
  "processing_time": "67.3s",
  "table_name": "text_pages_123e4567",
  "embedding_dimensions": 768,
  "configuration": {
    "chunking_strategy": "docling",
    "chunk_size": 512,
    "embedding_model": "nomic-embed-text"
  }
}
```

#### `GET /models`
**Purpose**: Get available models for RAG processing
**Response**: Same format as backend `/models`

---

## 🖥️ Frontend API Client

The frontend uses a centralized `ChatAPI` class to interact with backend services.

### 🔧 Core Methods

#### Session Management
```typescript
// Create session
const session = await chatAPI.createSession("New Chat", "qwen3:8b")

// Get sessions
const { sessions, total } = await chatAPI.getSessions()

// Load session with messages
const { session, messages } = await chatAPI.getSession(sessionId)

// Delete session
await chatAPI.deleteSession(sessionId)
```

#### Chat Operations
```typescript
// Send message with advanced options
const result = await chatAPI.sendSessionMessage(sessionId, "Query", {
  composeSubAnswers: true,
  decompose: true,
  aiRerank: true,
  contextExpand: true,
  verify: true,
  retrievalK: 20,
  contextWindowSize: 1,
  rerankerTopK: 10,
  searchType: 'hybrid',
  denseWeight: 0.7
})
```

#### File Operations
```typescript
// Upload files
const uploadResult = await chatAPI.uploadFiles(sessionId, files)

// Index documents
await chatAPI.indexDocuments(sessionId)

// Get session documents
const { files, file_count } = await chatAPI.getSessionDocuments(sessionId)
```

#### Index Management
```typescript
// Create index
const { index_id } = await chatAPI.createIndex("My Index", "Description")

// Upload to index
await chatAPI.uploadFilesToIndex(indexId, files)

// Build index
await chatAPI.buildIndex(indexId, {
  latechunk: false,
  doclingChunk: true,
  chunkSize: 512,
  embeddingModel: "nomic-embed-text"
})

// Link to session
await chatAPI.linkIndexToSession(sessionId, indexId)
```

#### Streaming
```typescript
// Stream RAG processing
await chatAPI.streamSessionMessage({
  query: "What does the document say?",
  session_id: sessionId,
  composeSubAnswers: true
}, (event) => {
  console.log(`Step: ${event.type}`, event.data)
})
```

---

## 🗄️ Database Schema

### 📋 Tables Structure

#### `sessions`
```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,                -- UUID
    title TEXT NOT NULL,               -- Session title
    created_at TEXT NOT NULL,          -- ISO timestamp
    updated_at TEXT NOT NULL,          -- ISO timestamp
    model_used TEXT NOT NULL,          -- Model identifier
    message_count INTEGER DEFAULT 0    -- Message count
)
```

#### `messages`
```sql
CREATE TABLE messages (
    id TEXT PRIMARY KEY,                -- UUID
    session_id TEXT NOT NULL,          -- Foreign key to sessions
    content TEXT NOT NULL,             -- Message content
    sender TEXT CHECK (sender IN ('user', 'assistant')),
    timestamp TEXT NOT NULL,           -- ISO timestamp
    metadata TEXT DEFAULT '{}',        -- JSON metadata
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
)
```

#### `indexes`
```sql
CREATE TABLE indexes (
    id TEXT PRIMARY KEY,               -- UUID
    name TEXT UNIQUE,                  -- Index name
    description TEXT,                  -- Description
    created_at TEXT,                   -- ISO timestamp
    updated_at TEXT,                   -- ISO timestamp
    vector_table_name TEXT,            -- LanceDB table name
    metadata TEXT                      -- JSON metadata
)
```

#### `index_documents`
```sql
CREATE TABLE index_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    index_id TEXT,                     -- Foreign key to indexes
    original_filename TEXT,            -- Original filename
    stored_path TEXT,                  -- Absolute storage path
    FOREIGN KEY(index_id) REFERENCES indexes(id)
)
```

#### `session_indexes`
```sql
CREATE TABLE session_indexes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,                   -- Foreign key to sessions
    index_id TEXT,                     -- Foreign key to indexes
    linked_at TEXT,                    -- ISO timestamp
    FOREIGN KEY(session_id) REFERENCES sessions(id),
    FOREIGN KEY(index_id) REFERENCES indexes(id)
)
```

#### `session_documents`
```sql
CREATE TABLE session_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,          -- Foreign key to sessions
    file_path TEXT NOT NULL,           -- Absolute file path
    indexed INTEGER DEFAULT 0,         -- Indexing status
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
)
```

---

## 🚀 Smart Routing Logic

### 🎯 Layer 1: Backend Routing (`_should_use_rag`)

**Purpose**: Speed optimization - Route between Direct LLM vs RAG Pipeline

**Decision Criteria**:
```python
def _should_use_rag(message: str, idx_ids: List[str]) -> bool:
    # No indexes = Direct LLM
    if not idx_ids:
        return False
    
    # Greeting patterns = Direct LLM
    greeting_patterns = ['hello', 'hi', 'thanks', 'test']
    if any(pattern in message.lower() for pattern in greeting_patterns):
        return False
    
    # Document keywords = RAG
    rag_indicators = ['document', 'according to', 'summarize', 'analyze']
    if any(indicator in message.lower() for indicator in rag_indicators):
        return True
    
    # Question + length = RAG
    question_words = ['what', 'how', 'when', 'where', 'why']
    if any(message.lower().startswith(word) for word in question_words) and len(message) > 40:
        return True
    
    # Default: Direct LLM (conservative approach)
    return False
```

**Routing Outcomes**:
- **Direct LLM**: `~1.3s` response time, thinking tokens disabled
- **RAG Pipeline**: `15-30s` response time, full document analysis

### 🧠 Layer 2: Agent Routing (`_triage_query_async`)

**Purpose**: Intelligence optimization within RAG pipeline

**Query Types**:
1. **General Knowledge**: Use LLM without retrieval
2. **Document Query**: Full RAG with retrieval and reranking
3. **Summarization**: Comprehensive document analysis
4. **Comparison**: Multi-document synthesis

---

## ⚙️ Configuration Parameters

### 🔧 Retrieval Parameters
- **`retrieval_k`**: Number of documents to retrieve (default: 20)
- **`context_window_size`**: Context expansion window (default: 1)
- **`reranker_top_k`**: Top results after reranking (default: 10)
- **`search_type`**: Search strategy - `'hybrid'`, `'dense'`, `'sparse'` (default: 'hybrid')
- **`dense_weight`**: Dense vs sparse weight in hybrid search (default: 0.7)

### 📝 Processing Flags
- **`compose_sub_answers`**: Enable sub-query composition (default: true)
- **`query_decompose`**: Enable query decomposition (default: true)
- **`ai_rerank`**: Enable AI-powered reranking (default: true)
- **`context_expand`**: Enable context window expansion (default: true)
- **`verify`**: Enable answer verification (default: true)

### 🏗️ Indexing Parameters
- **`chunk_size`**: Text chunk size (default: 512)
- **`chunk_overlap`**: Chunk overlap size (default: 64)
- **`enable_latechunk`**: Enable LaTeX-aware chunking (default: false)
- **`enable_docling_chunk`**: Enable Docling chunking (default: true)
- **`retrieval_mode`**: Retrieval strategy - `'hybrid'`, `'dense'`, `'sparse'` (default: 'hybrid')
- **`window_size`**: Context window size for enrichment (default: 2)
- **`enable_enrich`**: Enable context enrichment (default: true)
- **`embedding_model`**: Embedding model to use
- **`enrich_model`**: Model for context enrichment
- **`batch_size_embed`**: Embedding batch size (default: 50)
- **`batch_size_enrich`**: Enrichment batch size (default: 25)

---

## 🔄 Data Flow Examples

### 💬 Simple Chat Flow
```
1. Frontend → POST /sessions/{id}/messages
2. Backend → _should_use_rag() → False (greeting)
3. Backend → _handle_direct_llm_query()
4. Backend → Ollama (thinking disabled)
5. Backend → Response (~1.3s)
```

### 📚 Document Query Flow
```
1. Frontend → POST /sessions/{id}/messages (with RAG params)
2. Backend → _should_use_rag() → True (document keywords)
3. Backend → _handle_rag_query()
4. Backend → RAG API POST /chat
5. RAG API → Agent.run() → Multi-step processing
6. RAG API → Response with source docs (~15-30s)
```

### 📁 Document Upload & Index Flow
```
1. Frontend → POST /sessions/{id}/upload (files)
2. Backend → Save files to shared_uploads/
3. Frontend → POST /sessions/{id}/index
4. Backend → RAG API POST /index
5. RAG API → Indexing Pipeline
6. RAG API → LanceDB storage
7. Backend → Link session to index
```

### 🗃️ Index Management Flow
```
1. Frontend → POST /indexes (create)
2. Backend → Database insert
3. Frontend → POST /indexes/{id}/upload (files)
4. Backend → Save files
5. Frontend → POST /indexes/{id}/build
6. Backend → RAG API POST /index
7. RAG API → Build vector index
8. Frontend → POST /sessions/{id}/indexes/{id} (link)
```

---

## 🐛 Error Handling

### 🚨 Common Error Scenarios

#### Backend Errors
- **503**: Ollama not running
- **404**: Session/Index not found
- **400**: Missing required parameters
- **500**: Processing errors, BrokenPipeError handling

#### RAG API Errors
- **400**: Invalid query or missing parameters
- **500**: Indexing failures, model loading errors

#### Frontend Errors
- Network timeouts during long RAG processing
- File upload size limits (50MB)
- Invalid file formats

### 🛡️ Error Recovery
- Graceful degradation to Direct LLM when RAG fails
- Client disconnection handling during long operations
- Automatic retry logic for transient failures
- User-friendly error messages in frontend

---

## 🔍 API Testing & Validation

### 🧪 Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Create session
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Session"}'

# Send message
curl -X POST http://localhost:8000/sessions/{id}/messages \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# List models
curl http://localhost:8000/models
```

### 📊 Performance Benchmarks
- **Direct LLM**: ~1.3s average response time
- **RAG Pipeline**: 15-30s depending on complexity
- **File Upload**: ~2-5s per MB
- **Indexing**: ~1-3s per document page

---

## 🔧 Development Setup

### 🚀 Required Services
1. **Ollama Server**: `ollama serve` (port 11434)
2. **RAG API Server**: `python -m rag_system.api_server` (port 8001)  
3. **Backend Server**: `python backend/server.py` (port 8000)
4. **Frontend**: `npm run dev` (port 3000)

### 📝 Environment Configuration
- Models: qwen3:8b, nomic-embed-text
- Database: SQLite (chat_history.db)
- Vector Store: LanceDB (./lancedb/)
- File Storage: ./shared_uploads/

---

## Key Findings & Architecture Summary

### System Structure
The RAG system implements a sophisticated 4-tier architecture with intelligent routing:

1. **Frontend Layer**: React/Next.js with TypeScript providing rich UI interactions
2. **Backend Gateway**: Python HTTP server handling routing, sessions, and file management
3. **RAG Processing**: Specialized Python server for document analysis and retrieval
4. **LLM Service**: Ollama server providing generation and embedding capabilities

### Smart Routing Innovation
The system's **dual-layer routing** is a key architectural innovation:
- **Layer 1**: Speed optimization routing (Direct LLM vs RAG) - saves 90% response time for casual queries
- **Layer 2**: Intelligence optimization within RAG pipeline - ensures appropriate query handling

### Data Architecture
- **Relational**: SQLite for sessions, messages, indexes, and relationships
- **Vector**: LanceDB for document embeddings and similarity search
- **File System**: Organized storage with unique naming and path management

### API Design Patterns
- **RESTful**: Clear resource-based endpoints with proper HTTP methods
- **Payload Consistency**: Standardized JSON structures across all endpoints
- **Error Handling**: Comprehensive error responses with proper HTTP status codes
- **Real-time**: Server-Sent Events for streaming RAG processing steps

### Performance Characteristics
- **Direct LLM**: ~1.3s (optimized for casual conversation)
- **RAG Pipeline**: 15-30s (comprehensive document analysis)
- **File Processing**: 2-5s per MB (upload and indexing)
- **Concurrent**: Multiple sessions supported with proper isolation

### Integration Points
- **Frontend ↔ Backend**: RESTful API with advanced parameter support
- **Backend ↔ RAG API**: Internal microservice communication
- **RAG API ↔ Ollama**: Direct LLM integration with model management
- **Database ↔ Vector Store**: Coordinated persistence across storage types

---

This comprehensive documentation covers all APIs, their interactions, data structures, and the complete system architecture. Each endpoint has been validated against the actual source code implementation. 