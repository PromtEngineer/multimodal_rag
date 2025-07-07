# LocalGPT API Reference

**Version**: 1.0  
**Base URL**: `http://localhost:8000`  
**Content-Type**: `application/json`  
**Last Updated**: 2025-07-06

---

## Table of Contents

1. [Authentication](#authentication)
2. [Session Management API](#session-management-api)
3. [Index Management API](#index-management-api)
4. [Chat API](#chat-api)
5. [Document Upload API](#document-upload-api)
6. [Health & Status API](#health--status-api)
7. [RAG API (Advanced)](#rag-api-advanced)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [SDK Examples](#sdk-examples)

---

## Authentication

**Current Status**: No authentication required (local deployment)  
**Future Enhancement**: JWT-based authentication planned

### Headers

All API requests should include:

```http
Content-Type: application/json
Accept: application/json
```

---

## Session Management API

### Create Session

Create a new chat session with specified configuration.

```http
POST /sessions
```

#### Request Body

```json
{
  "title": "string",
  "model": "string",
  "embedding_model": "string (optional)"
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `title` | string | Yes | Human-readable session title |
| `model` | string | Yes | LLM model name (e.g., "qwen3:0.6b") |
| `embedding_model` | string | No | Embedding model override |

#### Response

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "My Research Session",
  "model": "qwen3:0.6b",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "created_at": "2025-07-06T12:00:00Z"
}
```

#### Example

```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Research Session",
    "model": "qwen3:0.6b"
  }'
```

### List Sessions

Retrieve all chat sessions with metadata.

```http
GET /sessions
```

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Maximum sessions to return |
| `offset` | integer | 0 | Number of sessions to skip |
| `sort` | string | "created_at" | Sort field |
| `order` | string | "desc" | Sort order (asc/desc) |

#### Response

```json
{
  "sessions": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "Research Session",
      "model": "qwen3:0.6b",
      "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
      "created_at": "2025-07-06T12:00:00Z",
      "updated_at": "2025-07-06T12:30:00Z",
      "message_count": 15,
      "linked_indexes": ["index-1", "index-2"]
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### Get Session Details

Retrieve detailed information about a specific session.

```http
GET /sessions/{session_id}
```

#### Response

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "title": "Research Session",
  "model": "qwen3:0.6b",
  "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
  "created_at": "2025-07-06T12:00:00Z",
  "updated_at": "2025-07-06T12:30:00Z",
  "message_count": 15,
  "linked_indexes": [
    {
      "id": "index-1",
      "name": "Research Papers",
      "document_count": 25
    }
  ],
  "recent_messages": [
    {
      "id": 1,
      "role": "user",
      "content": "What are the key findings?",
      "created_at": "2025-07-06T12:30:00Z"
    }
  ]
}
```

### Update Session

Update session configuration or metadata.

```http
PUT /sessions/{session_id}
```

#### Request Body

```json
{
  "title": "string (optional)",
  "model": "string (optional)",
  "embedding_model": "string (optional)"
}
```

### Delete Session

Delete a session and all associated messages.

```http
DELETE /sessions/{session_id}
```

#### Response

```json
{
  "message": "Session deleted successfully",
  "deleted_session_id": "550e8400-e29b-41d4-a716-446655440000",
  "deleted_messages": 15
}
```

### Get Session Messages

Retrieve chat history for a session.

```http
GET /sessions/{session_id}/messages
```

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Maximum messages to return |
| `offset` | integer | 0 | Number of messages to skip |
| `since` | string | null | ISO timestamp for recent messages |

#### Response

```json
{
  "messages": [
    {
      "id": 1,
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "role": "user",
      "content": "What are the main topics in the documents?",
      "metadata": {},
      "created_at": "2025-07-06T12:25:00Z"
    },
    {
      "id": 2,
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "role": "assistant",
      "content": "Based on the documents, the main topics include...",
      "metadata": {
        "sources": ["doc1.pdf", "doc2.pdf"],
        "processing_time": 2.3,
        "confidence": 0.89
      },
      "created_at": "2025-07-06T12:25:02Z"
    }
  ],
  "total": 2,
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Index Management API

### Create Index

Create a new document index for organizing and searching documents.

```http
POST /indexes
```

#### Request Body

```json
{
  "name": "string",
  "description": "string (optional)",
  "config": {
    "chunk_size": "integer (optional)",
    "chunk_overlap": "integer (optional)",
    "retrieval_mode": "string (optional)",
    "enable_enrich": "boolean (optional)"
  }
}
```

#### Response

```json
{
  "index_id": "b6d85f40-f54d-49e6-bfef-1aa426baf825",
  "name": "Research Papers",
  "description": "Collection of AI research papers",
  "status": "created",
  "created_at": "2025-07-06T12:00:00Z",
  "metadata": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "retrieval_mode": "hybrid",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "enable_enrich": true,
    "latechunk": true,
    "docling_chunk": true
  }
}
```

### List Indexes

Retrieve all available document indexes.

```http
GET /indexes
```

#### Response

```json
{
  "indexes": [
    {
      "id": "b6d85f40-f54d-49e6-bfef-1aa426baf825",
      "name": "Research Papers",
      "description": "AI research collection",
      "status": "functional",
      "document_count": 25,
      "total_chunks": 1250,
      "created_at": "2025-07-06T12:00:00Z",
      "updated_at": "2025-07-06T12:15:00Z"
    }
  ],
  "total": 1
}
```

### Get Index Details

Retrieve detailed information about a specific index.

```http
GET /indexes/{index_id}
```

#### Response

```json
{
  "id": "b6d85f40-f54d-49e6-bfef-1aa426baf825",
  "name": "Research Papers",
  "description": "AI research collection",
  "status": "functional",
  "vector_table_name": "text_pages_b6d85f40-f54d-49e6-bfef-1aa426baf825",
  "created_at": "2025-07-06T12:00:00Z",
  "updated_at": "2025-07-06T12:15:00Z",
  "metadata": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "retrieval_mode": "hybrid",
    "window_size": 5,
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "enrich_model": "qwen3:0.6b",
    "overview_model": "qwen3:0.6b",
    "enable_enrich": true,
    "latechunk": true,
    "docling_chunk": true,
    "total_chunks": 1250,
    "total_documents": 25,
    "vector_dimensions": 1024,
    "last_updated": "2025-07-06T12:15:00Z"
  },
  "documents": [
    {
      "id": "doc-1",
      "filename": "research_paper_1.pdf",
      "file_size": 2048576,
      "chunk_count": 45,
      "uploaded_at": "2025-07-06T12:05:00Z"
    }
  ]
}
```

### Update Index

Update index configuration or metadata.

```http
PUT /indexes/{index_id}
```

#### Request Body

```json
{
  "name": "string (optional)",
  "description": "string (optional)",
  "config": {
    "retrieval_mode": "string (optional)",
    "enable_enrich": "boolean (optional)"
  }
}
```

### Delete Index

Delete an index and all associated data.

```http
DELETE /indexes/{index_id}
```

#### Response

```json
{
  "message": "Index deleted successfully",
  "deleted_index_id": "b6d85f40-f54d-49e6-bfef-1aa426baf825",
  "deleted_documents": 25,
  "deleted_chunks": 1250
}
```

### Link Index to Session

Associate an index with a chat session for querying.

```http
POST /sessions/{session_id}/indexes/{index_id}
```

#### Response

```json
{
  "message": "Index linked to session successfully",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "index_id": "b6d85f40-f54d-49e6-bfef-1aa426baf825",
  "linked_at": "2025-07-06T12:30:00Z"
}
```

### Get Session Indexes

Retrieve all indexes linked to a session.

```http
GET /sessions/{session_id}/indexes
```

#### Response

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "indexes": [
    {
      "id": "b6d85f40-f54d-49e6-bfef-1aa426baf825",
      "name": "Research Papers",
      "status": "functional",
      "document_count": 25,
      "linked_at": "2025-07-06T12:30:00Z",
      "metadata": {
        "chunk_size": 512,
        "chunk_overlap": 64,
        "retrieval_mode": "hybrid",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B"
      }
    }
  ],
  "total": 1
}
```

---

## Chat API

### Send Chat Message

Send a message to the AI assistant with optional retrieval parameters.

```http
POST /chat
```

#### Request Body

```json
{
  "query": "string",
  "session_id": "string",
  "table_name": "string (optional)",
  "search_type": "hybrid|vector|bm25",
  "retrieval_k": "integer (default: 20)",
  "reranker_top_k": "integer (default: 10)",
  "context_window_size": "integer (default: 1)",
  "dense_weight": "float (default: 0.7)",
  "enable_verification": "boolean (default: true)",
  "stream": "boolean (default: false)"
}
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | - | User's question or message |
| `session_id` | string | - | Session identifier |
| `table_name` | string | auto | Vector table override |
| `search_type` | string | "hybrid" | Search strategy |
| `retrieval_k` | integer | 20 | Number of chunks to retrieve |
| `reranker_top_k` | integer | 10 | Top chunks after reranking |
| `context_window_size` | integer | 1 | Context window for chunks |
| `dense_weight` | float | 0.7 | Weight for vector search in hybrid mode |
| `enable_verification` | boolean | true | Enable answer verification |
| `stream` | boolean | false | Enable streaming response |

#### Response (Non-streaming)

```json
{
  "response": "Based on the research papers, the key findings include...",
  "sources": [
    {
      "chunk_id": "chunk-123",
      "document_id": "doc-1",
      "filename": "research_paper_1.pdf",
      "text": "The study found that artificial intelligence...",
      "metadata": {
        "page": 5,
        "section": "Results"
      },
      "score": 0.89,
      "rank": 1
    }
  ],
  "metadata": {
    "query_type": "rag_pipeline",
    "processing_time": 2.34,
    "retrieval_count": 20,
    "reranked_count": 10,
    "context_chunks": 5,
    "model_used": "qwen3:0.6b",
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "confidence_score": 0.87,
    "verification_passed": true
  },
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message_id": 15,
  "created_at": "2025-07-06T12:35:00Z"
}
```

#### Response (Streaming)

When `stream: true`, the response uses Server-Sent Events:

```
event: start
data: {"message_id": 15, "session_id": "550e8400-e29b-41d4-a716-446655440000"}

event: progress
data: {"step": "retrieval", "progress": 25, "message": "Searching documents..."}

event: progress
data: {"step": "reranking", "progress": 50, "message": "Reranking results..."}

event: progress
data: {"step": "generation", "progress": 75, "message": "Generating response..."}

event: chunk
data: {"text": "Based on the research papers, "}

event: chunk
data: {"text": "the key findings include..."}

event: sources
data: {"sources": [...], "metadata": {...}}

event: complete
data: {"message_id": 15, "total_time": 2.34}
```

### Quick Chat

Send a message without creating a persistent session.

```http
POST /chat/quick
```

#### Request Body

```json
{
  "query": "string",
  "model": "string (optional)",
  "search_type": "string (optional)",
  "retrieval_k": "integer (optional)"
}
```

#### Response

```json
{
  "response": "AI generated response",
  "metadata": {
    "query_type": "direct_llm",
    "processing_time": 1.2,
    "model_used": "qwen3:0.6b"
  },
  "temporary_session": true
}
```

---

## Document Upload API

### Upload Documents

Upload one or more documents to an index.

```http
POST /indexes/{index_id}/upload
```

#### Request

```http
Content-Type: multipart/form-data

files: File[]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `files` | File[] | Array of files to upload |

#### Supported File Types

- PDF (.pdf)
- Microsoft Word (.docx)
- Plain Text (.txt)
- Markdown (.md)
- Rich Text Format (.rtf)

#### Response

```json
{
  "message": "Uploaded 3 files successfully",
  "uploaded_files": [
    {
      "filename": "research_paper_1.pdf",
      "original_name": "AI Research - Deep Learning.pdf",
      "file_size": 2048576,
      "content_type": "application/pdf",
      "stored_path": "/uploads/550e8400-e29b-41d4-a716-446655440000_research_paper_1.pdf",
      "document_id": "doc-1",
      "uploaded_at": "2025-07-06T12:05:00Z"
    }
  ],
  "failed_files": [],
  "total_size": 6145728,
  "index_id": "b6d85f40-f54d-49e6-bfef-1aa426baf825"
}
```

#### Error Response

```json
{
  "error": "Upload failed",
  "message": "Some files could not be uploaded",
  "uploaded_files": [...],
  "failed_files": [
    {
      "filename": "corrupted_file.pdf",
      "error": "File is corrupted or unreadable",
      "error_code": "INVALID_FILE_FORMAT"
    }
  ]
}
```

### Build Index

Process uploaded documents and create searchable index.

```http
POST /indexes/{index_id}/build
```

#### Request Body (Optional)

```json
{
  "config": {
    "chunk_size": "integer (optional)",
    "chunk_overlap": "integer (optional)",
    "enable_enrich": "boolean (optional)",
    "batch_size": "integer (optional)"
  },
  "force_rebuild": "boolean (default: false)"
}
```

#### Response

```json
{
  "message": "Index built successfully",
  "index_id": "b6d85f40-f54d-49e6-bfef-1aa426baf825",
  "processing_stats": {
    "total_documents": 3,
    "total_chunks": 150,
    "processing_time": 45.6,
    "embedding_time": 23.4,
    "indexing_time": 12.1
  },
  "config_used": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "retrieval_mode": "hybrid",
    "enable_enrich": true,
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B"
  },
  "table_info": {
    "vector_table_name": "text_pages_b6d85f40-f54d-49e6-bfef-1aa426baf825",
    "vector_dimensions": 1024,
    "bm25_index_created": true,
    "fts_index_created": true
  }
}
```

### Get Build Status

Check the status of an ongoing index build operation.

```http
GET /indexes/{index_id}/build/status
```

#### Response

```json
{
  "index_id": "b6d85f40-f54d-49e6-bfef-1aa426baf825",
  "status": "building",
  "progress": {
    "current_step": "embedding_generation",
    "step_progress": 65,
    "overall_progress": 45,
    "estimated_completion": "2025-07-06T12:08:00Z"
  },
  "stats": {
    "documents_processed": 2,
    "total_documents": 3,
    "chunks_created": 98,
    "embeddings_generated": 65
  },
  "started_at": "2025-07-06T12:05:00Z"
}
```

---

## Health & Status API

### Health Check

Check the overall health of the system.

```http
GET /health
```

#### Response

```json
{
  "status": "ok",
  "timestamp": "2025-07-06T12:00:00Z",
  "version": "1.0.0",
  "services": {
    "database": {
      "status": "healthy",
      "response_time": 2
    },
    "rag_api": {
      "status": "healthy",
      "response_time": 15
    },
    "ollama": {
      "status": "healthy",
      "response_time": 8,
      "models_loaded": 3
    },
    "vector_db": {
      "status": "healthy",
      "response_time": 5,
      "total_tables": 12
    }
  },
  "system_info": {
    "cpu_usage": 45.2,
    "memory_usage": 68.7,
    "disk_usage": 23.1,
    "uptime": 86400
  }
}
```

### System Status

Get detailed system information and statistics.

```http
GET /status
```

#### Response

```json
{
  "system": {
    "uptime": 86400,
    "version": "1.0.0",
    "environment": "production",
    "started_at": "2025-07-05T12:00:00Z"
  },
  "database": {
    "total_sessions": 125,
    "total_messages": 3450,
    "total_indexes": 15,
    "total_documents": 450,
    "database_size": "245MB"
  },
  "ai_services": {
    "ollama_models": [
      {
        "name": "qwen3:0.6b",
        "size": "367MB",
        "status": "loaded"
      },
      {
        "name": "qwen3:8b",
        "size": "4.7GB",
        "status": "available"
      }
    ],
    "embedding_models": [
      {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "dimensions": 1024,
        "status": "loaded"
      }
    ]
  },
  "performance": {
    "avg_response_time": 2.3,
    "requests_per_minute": 45,
    "cache_hit_rate": 78.5,
    "error_rate": 0.2
  }
}
```

### Available Models

List all available AI models.

```http
GET /models
```

#### Response

```json
{
  "generation_models": [
    {
      "name": "qwen3:0.6b",
      "type": "ollama",
      "size": "367MB",
      "status": "loaded",
      "capabilities": ["chat", "completion"],
      "max_context": 2048
    },
    {
      "name": "qwen3:8b",
      "type": "ollama",
      "size": "4.7GB",
      "status": "available",
      "capabilities": ["chat", "completion"],
      "max_context": 4096
    }
  ],
  "embedding_models": [
    {
      "name": "Qwen/Qwen3-Embedding-0.6B",
      "type": "huggingface",
      "dimensions": 1024,
      "status": "loaded",
      "max_input_length": 512
    },
    {
      "name": "Qwen/Qwen3-Embedding-4B",
      "type": "huggingface",
      "dimensions": 2048,
      "status": "available",
      "max_input_length": 512
    }
  ],
  "reranker_models": [
    {
      "name": "BAAI/bge-reranker-base",
      "type": "huggingface",
      "status": "available",
      "max_pairs": 1000
    }
  ]
}
```

---

## RAG API (Advanced)

The RAG API provides advanced document processing and retrieval capabilities.

**Base URL**: `http://localhost:8001`

### Advanced Chat

Enhanced chat with fine-grained control over the RAG pipeline.

```http
POST /chat
```

#### Request Body

```json
{
  "query": "string",
  "session_id": "string",
  "compose_sub_answers": "boolean (default: false)",
  "query_decompose": "boolean (default: false)",
  "ai_rerank": "boolean (default: true)",
  "context_expand": "boolean (default: false)",
  "verify": "boolean (default: true)",
  "force_rag": "boolean (default: false)",
  "provence_prune": "boolean (default: false)",
  "provence_threshold": "float (default: 0.5)",
  "retrieval_k": "integer (default: 20)",
  "context_window_size": "integer (default: 1)",
  "reranker_top_k": "integer (default: 10)",
  "search_type": "hybrid|vector|bm25",
  "dense_weight": "float (default: 0.7)"
}
```

#### Advanced Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `compose_sub_answers` | boolean | Compose answers from sub-queries |
| `query_decompose` | boolean | Break complex queries into parts |
| `ai_rerank` | boolean | Use AI-powered reranking |
| `context_expand` | boolean | Expand context with related chunks |
| `verify` | boolean | Verify answers against sources |
| `force_rag` | boolean | Force RAG pipeline (skip triage) |
| `provence_prune` | boolean | Prune irrelevant sentences |
| `provence_threshold` | float | Threshold for sentence pruning |

### Streaming Chat

Real-time streaming chat with progress updates.

```http
POST /chat/stream
```

The request body is identical to the advanced chat endpoint, but responses are streamed as Server-Sent Events.

#### Stream Events

| Event | Description | Data Format |
|-------|-------------|-------------|
| `start` | Processing started | `{"session_id": "...", "query_id": "..."}` |
| `progress` | Processing step update | `{"step": "...", "progress": 0-100}` |
| `chunk` | Partial response text | `{"text": "..."}` |
| `sources` | Retrieved sources | `{"sources": [...]}` |
| `complete` | Processing finished | `{"total_time": 2.34}` |
| `error` | Processing error | `{"error": "...", "code": "..."}` |

### Document Indexing

Process documents with advanced configuration options.

```http
POST /index
```

#### Request Body

```json
{
  "file_paths": ["string"],
  "session_id": "string",
  "chunk_size": "integer (default: 512)",
  "chunk_overlap": "integer (default: 64)",
  "retrieval_mode": "hybrid|vector|bm25",
  "window_size": "integer (default: 2)",
  "enable_enrich": "boolean (default: true)",
  "enable_latechunk": "boolean (default: false)",
  "enable_docling_chunk": "boolean (default: false)",
  "embedding_model": "string (optional)",
  "enrich_model": "string (optional)",
  "overview_model": "string (optional)",
  "batch_size_embed": "integer (default: 50)",
  "batch_size_enrich": "integer (default: 25)"
}
```

#### Response

```json
{
  "message": "Indexing process for 3 file(s) completed successfully.",
  "table_name": "text_pages_b6d85f40-f54d-49e6-bfef-1aa426baf825",
  "latechunk": false,
  "docling_chunk": false,
  "indexing_config": {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "retrieval_mode": "hybrid",
    "window_size": 2,
    "enable_enrich": true,
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "enrich_model": "qwen3:0.6b",
    "batch_size_embed": 50,
    "batch_size_enrich": 25
  },
  "processing_stats": {
    "total_files": 3,
    "total_chunks": 245,
    "processing_time": 67.8,
    "embedding_time": 34.2,
    "enrichment_time": 18.9
  }
}
```

---

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```json
{
  "error": "string",
  "message": "string",
  "code": "string",
  "details": "object (optional)",
  "timestamp": "string",
  "path": "string",
  "request_id": "string"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Access denied |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 502 | Bad Gateway | Upstream service error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

#### Validation Errors (400)

```json
{
  "error": "Validation Error",
  "message": "Invalid request parameters",
  "code": "VALIDATION_ERROR",
  "details": {
    "field": "query",
    "reason": "Query cannot be empty",
    "value": ""
  }
}
```

#### Resource Not Found (404)

```json
{
  "error": "Not Found",
  "message": "Session not found",
  "code": "SESSION_NOT_FOUND",
  "details": {
    "session_id": "invalid-session-id"
  }
}
```

#### Rate Limiting (429)

```json
{
  "error": "Rate Limit Exceeded",
  "message": "Too many requests",
  "code": "RATE_LIMIT_EXCEEDED",
  "details": {
    "limit": 100,
    "window": "1h",
    "retry_after": 3600
  }
}
```

#### Server Error (500)

```json
{
  "error": "Internal Server Error",
  "message": "An unexpected error occurred",
  "code": "INTERNAL_ERROR",
  "details": {
    "error_id": "error-123456",
    "support_contact": "support@localgpt.com"
  }
}
```

---

## Rate Limiting

### Current Limits

| Endpoint Category | Requests per Minute | Requests per Hour |
|-------------------|---------------------|-------------------|
| Chat API | 60 | 1000 |
| Upload API | 10 | 100 |
| Index Management | 30 | 500 |
| General API | 100 | 2000 |

### Rate Limit Headers

All responses include rate limiting headers:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1641024000
X-RateLimit-Window: 60
```

---

## SDK Examples

### Python SDK

```python
import requests
from typing import Dict, List, Optional

class LocalGPTClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_session(self, title: str, model: str) -> Dict:
        """Create a new chat session."""
        response = self.session.post(
            f"{self.base_url}/sessions",
            json={"title": title, "model": model}
        )
        response.raise_for_status()
        return response.json()
    
    def chat(self, query: str, session_id: str, **kwargs) -> Dict:
        """Send a chat message."""
        data = {"query": query, "session_id": session_id, **kwargs}
        response = self.session.post(f"{self.base_url}/chat", json=data)
        response.raise_for_status()
        return response.json()
    
    def upload_documents(self, index_id: str, file_paths: List[str]) -> Dict:
        """Upload documents to an index."""
        files = [('files', open(path, 'rb')) for path in file_paths]
        try:
            response = self.session.post(
                f"{self.base_url}/indexes/{index_id}/upload",
                files=files
            )
            response.raise_for_status()
            return response.json()
        finally:
            for _, file in files:
                file.close()

# Usage example
client = LocalGPTClient()

# Create session
session = client.create_session("Research Session", "qwen3:0.6b")
session_id = session["session_id"]

# Upload and index documents
index_response = client.session.post(
    f"{client.base_url}/indexes",
    json={"name": "Research Papers"}
)
index_id = index_response.json()["index_id"]

client.upload_documents(index_id, ["paper1.pdf", "paper2.pdf"])

# Chat with documents
response = client.chat(
    "What are the main findings?",
    session_id,
    search_type="hybrid",
    retrieval_k=20
)
print(response["response"])
```

### JavaScript SDK

```javascript
class LocalGPTClient {
    constructor(baseURL = 'http://localhost:8000') {
        this.baseURL = baseURL;
    }

    async createSession(title, model) {
        const response = await fetch(`${this.baseURL}/sessions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, model })
        });
        return response.json();
    }

    async chat(query, sessionId, options = {}) {
        const response = await fetch(`${this.baseURL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, session_id: sessionId, ...options })
        });
        return response.json();
    }

    async uploadDocuments(indexId, files) {
        const formData = new FormData();
        files.forEach(file => formData.append('files', file));
        
        const response = await fetch(`${this.baseURL}/indexes/${indexId}/upload`, {
            method: 'POST',
            body: formData
        });
        return response.json();
    }

    async streamChat(query, sessionId, options = {}) {
        const response = await fetch(`${this.baseURL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                query, 
                session_id: sessionId, 
                stream: true, 
                ...options 
            })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        return {
            async *[Symbol.asyncIterator]() {
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (data !== '[DONE]') {
                                yield JSON.parse(data);
                            }
                        }
                    }
                }
            }
        };
    }
}

// Usage example
const client = new LocalGPTClient();

async function example() {
    // Create session
    const session = await client.createSession('Research Session', 'qwen3:0.6b');
    
    // Chat with streaming
    const stream = await client.streamChat('What are the key findings?', session.session_id);
    
    for await (const chunk of stream) {
        if (chunk.event === 'chunk') {
            console.log(chunk.data.text);
        }
    }
}
```

### cURL Examples

#### Create Session and Chat

```bash
#!/bin/bash

# Create session
SESSION_RESPONSE=$(curl -s -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Research Session", "model": "qwen3:0.6b"}')

SESSION_ID=$(echo $SESSION_RESPONSE | jq -r '.session_id')
echo "Created session: $SESSION_ID"

# Create index
INDEX_RESPONSE=$(curl -s -X POST http://localhost:8000/indexes \
  -H "Content-Type: application/json" \
  -d '{"name": "Research Papers"}')

INDEX_ID=$(echo $INDEX_RESPONSE | jq -r '.index_id')
echo "Created index: $INDEX_ID"

# Upload document
curl -X POST http://localhost:8000/indexes/$INDEX_ID/upload \
  -F "files=@research_paper.pdf"

# Build index
curl -X POST http://localhost:8000/indexes/$INDEX_ID/build

# Link index to session
curl -X POST http://localhost:8000/sessions/$SESSION_ID/indexes/$INDEX_ID

# Chat with documents
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What are the main findings in the research?\",
    \"session_id\": \"$SESSION_ID\",
    \"search_type\": \"hybrid\",
    \"retrieval_k\": 20
  }" | jq '.response'
```

---

This comprehensive API reference provides complete documentation for all LocalGPT endpoints, including request/response schemas, error handling, and practical examples for integration. 