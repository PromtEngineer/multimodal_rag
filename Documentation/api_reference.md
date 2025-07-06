# 📚 API Reference (Backend & Frontend)

_Last updated: 2025-07-06_

---

## Backend HTTP API (Python `backend/server.py`)
| Endpoint | Method | Description | Request Body | Success Response |
|----------|--------|-------------|--------------|------------------|
| `/health` | GET | Health probe incl. Ollama status & DB stats | – | 200 JSON `{ status, ollama_running, available_models, database_stats }` |
| `/chat` | POST | Stateless chat (no session) | `{ message:str, model?:str, conversation_history?:[{role,content}]}` | 200 `{ response:str, model:str, message_count:int }` |
| `/sessions` | GET | List all sessions | – | `{ sessions:ChatSession[], total:int }` |
| `/sessions` | POST | Create session | `{ title?:str, model?:str }` | 201 `{ session:ChatSession, session_id }` |
| `/sessions/<id>` | GET | Get session + msgs | – | `{ session, messages }` |
| `/sessions/<id>` | DELETE | Delete session | – | `{ message, deleted_session_id }` |
| `/sessions/<id>/rename` | POST | Rename session | `{ title:str }` | `{ message, session }` |
| `/sessions/<id>/messages` | POST | Session chat (builds history) | See `ChatAPI.sendSessionMessage` payload ▼ | `{ response, session, user_message_id, ai_message_id }` |
| `/sessions/<id>/documents` | GET | List uploaded docs | – | `{ files:string[], file_count:int, session }` |
| `/sessions/<id>/upload` | POST multipart | Upload docs to session | field `files[]` | `{ message, uploaded_files, processing_results?, session_documents?, total_session_documents? }` |
| `/sessions/<id>/index` | POST | Trigger RAG indexing for session | `{ latechunk?, doclingChunk?, chunkSize?, ... }` | `{ message }` |
| `/sessions/<id>/indexes` | GET | List indexes linked to session | – | `{ indexes, total }` |
| `/sessions/<sid>/indexes/<idxid>` | POST | Link index to session | – | `{ message }` |
| `/sessions/cleanup` | GET | Remove empty sessions | – | `{ message, cleanup_count }` |
| `/models` | GET | List generation / embedding models | – | `{ generation_models:str[], embedding_models:str[] }` |
| `/indexes` | GET | List all indexes | – | `{ indexes, total }` |
| `/indexes` | POST | Create index | `{ name:str, description?:str, metadata?:dict }` | `{ index_id }` |
| `/indexes/<id>` | GET | Get single index | – | `{ index }` |
| `/indexes/<id>` | DELETE | Delete index | – | `{ message, index_id }` |
| `/indexes/<id>/upload` | POST multipart | Upload docs to index | field `files[]` | `{ message, uploaded_files }` |
| `/indexes/<id>/build` | POST | Build / rebuild index (RAG) | `{ latechunk?, doclingChunk?, ...}` | 200 `{ response?, message?}` (idempotent) |

> **Note on CORS** – All endpoints include single `Access-Control-Allow-Origin: *` header after fix _af99b38_.

### Chat-streaming Experimental API (`rag_system/api_server.py`)
(`/chat/stream`, SSE) – See `ChatAPI.streamSessionMessage` payload and client-side handling in `src/lib/api.ts`.

---

## Frontend Wrapper (`src/lib/api.ts`)
The React/Next.js frontend calls the backend via a typed wrapper. Important methods & payloads:

| Method | Backend Endpoint | Payload Shape |
|--------|------------------|---------------|
| `checkHealth()` | `/health` | – |
| `sendMessage({ message, model?, conversation_history? })` | `/chat` | ChatRequest |
| `getSessions()` | `/sessions` | – |
| `createSession(title?, model?)` | `/sessions` | – |
| `getSession(sessionId)` | `/sessions/<id>` | – |
| `sendSessionMessage(sessionId, message, opts)` | `/sessions/<id>/messages` | `ChatRequest + retrieval opts (composeSubAnswers, decompose, aiRerank, ...)` |
| `uploadFiles(sessionId, files[])` | `/sessions/<id>/upload` | multipart |
| `indexDocuments(sessionId)` | `/sessions/<id>/index` | opts similar to buildIndex |
| `buildIndex(indexId, opts)` | `/indexes/<id>/build` | `{ latechunk?, doclingChunk?, chunkSize?, ... }` |
| `linkIndexToSession` | `/sessions/<sid>/indexes/<idx>` | – |

_TypeScript type aliases_ live at the bottom of `api.ts` (`ChatMessage`, `Step`, etc.).

---

## Payload Definitions (Canonical)
### ChatRequest (frontend ⇄ backend)
```jsonc
{
  "message": "string",            // Required – raw user text
  "model": "string",              // Optional – generation model id
  "conversation_history": [        // Optional – prior turn list
    { "role": "user|assistant", "content": "string" }
  ]
}
```

### Session Chat Extended Options (subset)
```jsonc
{
  "composeSubAnswers": true,
  "decompose": true,
  "aiRerank": false,
  "contextExpand": false,
  "verify": true,
  "retrievalK": 10,
  "contextWindowSize": 5,
  "rerankerTopK": 20,
  "searchType": "fts|hybrid|dense",
  "denseWeight": 0.75
}
```

### Index Build Options
```jsonc
{
  "latechunk": true,
  "doclingChunk": false,
  "chunkSize": 512,
  "chunkOverlap": 64,
  "retrievalMode": "hybrid|dense|fts",
  "windowSize": 2,
  "enableEnrich": true,
  "embeddingModel": "qwen3:0.6b",
  "enrichModel": "qwen3:8b",
  "overviewModel": "qwen3:0.6b",
  "batchSizeEmbed": 64,
  "batchSizeEnrich": 32
}
```

---

_This reference is derived from static code analysis of `backend/server.py`, `rag_system/api_server.py`, and `src/lib/api.ts`. Keep it in sync with route or type changes._ 