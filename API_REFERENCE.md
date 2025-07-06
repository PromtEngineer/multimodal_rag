# Source-of-Truth API Reference

*(generated from commit `bf6219a` – July 3 2025)*

This file lists every public call surface between the React/Next-JS front-end (`src/lib/api.ts`) and the two Python back-ends:

* **Session / Upload server** – `backend/server.py` (port **8000**)
* **Advanced RAG server**  – `rag_system/api_server.py` (port **8001**)

---
## 1  Front-end TypeScript wrapper (`ChatAPI`)
Located in `src/lib/api.ts`.

| Method | Backend Route | Request Params | Return Shape |
|---|---|---|---|
| `checkHealth()` | `GET /health` | — | `{ status, ollama_running, available_models, database_stats? }` |
| `sendMessage()` | `POST /chat` | `message*`, `model`, `conversation_history[]` | `{ response, model, message_count }` |
| `getSessions()` | `GET /sessions` | — | `{ sessions[], total }` |
| `createSession(title?, model?)` | `POST /sessions` | Defaults: `"New Chat"`, `llama3.2:latest` | `ChatSession` |
| `getSession(id*)` | `GET /sessions/{id}` | — | `{ session, messages[] }` |
| `sendSessionMessage(id*, message*, opts)` | `POST /sessions/{id}/messages` | see *Tuning opts* below | `{ response, session, user_message_id, ai_message_id, source_documents[] }` |
| `streamSessionMessage(params, cb)` | `POST /chat/stream` (8001) | same keys as above + `table_name?` | SSE events until `type:"complete"` |
| `deleteSession(id*)` | `DELETE /sessions/{id}` | — | `{ message, deleted_session_id }` |
| `cleanupEmptySessions()` | `GET /sessions/cleanup` | — | `{ message, cleanup_count }` |
| `uploadFiles(id*, files[])` | `POST /sessions/{id}/upload` | multipart `files` | `{ message, uploaded_files[] }` |
| `indexDocuments(id*)` | `POST /sessions/{id}/index` | — | `{ message }` |
| Index helpers (`createIndex`, `buildIndex`, …) map 1-to-1 to `/indexes/*` routes (see §2.2).

### 1.1 Tuning `opts`

| UI control | camelCase key | Sent as snake_case | Default |
|---|---|---|---|
| Query decomposition | `decompose` | `query_decompose` | pipeline config |
| Compose sub-answers | `composeSubAnswers` | `compose_sub_answers` | cfg |
| AI reranker | `aiRerank` | `ai_rerank` | cfg |
| Reranker top chunks | `rerankerTopK` | `reranker_top_k` | 10 |
| Retrieval chunks | `retrievalK` | `retrieval_k` | 20 |
| Dense search weight | `denseWeight` | `dense_weight` | 0.7 |
| Search type | `searchType` | `search_type` | "hybrid" |
| Expand context window | `contextExpand` | `context_expand` | cfg |
| Context window size | `contextWindowSize` | `context_window_size` | 1 |
| Always search documents | `forceRag` | `force_rag` | false |
| LLM model | `model` | `model` | server default |
| Prune irrelevant sentences | `provencePrune` | `provence_prune` | false |
| Prune threshold | `provenceThreshold` | `provence_threshold` | 0.1 |

---
## 2  Session / Upload Server (`backend/server.py`, :8000)

### 2.1 REST endpoints

| Path | Method | Purpose |
|---|---|---|
| `/health` | GET | Ollama + DB status |
| `/models` | GET | List generation & embedding models |
| `/sessions` | GET&nbsp;/&nbsp;POST | List or create sessions |
| `/sessions/cleanup` | GET | Delete empty sessions |
| `/sessions/{id}` | GET&nbsp;/&nbsp;DELETE | Retrieve or delete session |
| `/sessions/{id}/messages` | POST | Chat with smart routing / RAG |
| `/sessions/{id}/documents` | GET | List uploaded docs |
| `/sessions/{id}/upload` | POST (multipart) | Upload PDFs |
| `/sessions/{id}/index` | POST | Index all session docs |
| `/sessions/{id}/indexes` | GET | List linked indexes |
| `/sessions/{id}/indexes/{indexId}` | POST | Link existing index |
| `/chat` | POST | Legacy direct Ollama chat |

### 2.2 Index routes

| Path | Method | Purpose |
|---|---|---|
| `/indexes` | GET&nbsp;/&nbsp;POST | List or create index |
| `/indexes/{id}` | GET&nbsp;/&nbsp;DELETE | Describe / delete index |
| `/indexes/{id}/upload` | POST | Upload files to index |
| `/indexes/{id}/build` | POST | Build index (chunk-&-embed) |

---
## 3  Advanced RAG API (`rag_system/api_server.py`, :8001)

### 3.1 Endpoints

| Path | Method | Purpose |
|---|---|---|
| `/chat` | POST | One-shot RAG answer |
| `/chat/stream` | POST | SSE stream with phases & tokens |
| `/index` | POST | Index arbitrary file paths |
| `/models` | GET | Model list helper |

### 3.2 Streaming event types (`chat/stream`)

| type | data payload |
|---|---|
| `analyze` | `{}` |
| `decomposition` | `{ sub_queries[] }` |
| `retrieval_started` / `retrieval_done` | `{ count }` |
| `rerank_started` / `rerank_done` | `{ count }` |
| `context_expand_started` / `context_expand_done` | `{ count }` |
| `prune_started` / `prune_done` | `{ count }` |
| `sub_query_token` | `{ index, text, question }` |
| `sub_query_result` | `{ index, query, answer, source_documents[] }` |
| `token`