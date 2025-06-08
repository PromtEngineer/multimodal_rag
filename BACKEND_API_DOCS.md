# Backend API Documentation

This document specifies the RESTful API endpoints provided by the Python backend server (`server.py`).

---

## General Information
- **Base URL**: `http://localhost:8000`
- **Content-Type**: All requests and responses are `application/json`, except for file uploads.
- **CORS**: The server is configured to allow cross-origin requests from the frontend.

---

## 1. Health Check

### `GET /health`
- **Description**: Provides a health check of the backend services.
- **Request Body**: None.
- **Success Response (200 OK)**:
  ```json
  {
    "status": "ok",
    "ollama_running": true,
    "available_models": ["llama3.2:latest", "..."],
    "database_stats": {
      "total_sessions": 10,
      "total_messages": 152,
      "most_used_model": "llama3.2:latest"
    }
  }
  ```

---

## 2. Session Management

### `GET /sessions`
- **Description**: Retrieves a list of all chat sessions.
- **Request Body**: None.
- **Success Response (200 OK)**:
  ```json
  {
    "sessions": [
      {
        "id": "uuid-string-1",
        "title": "My First Chat",
        "created_at": "2023-10-27T10:00:00Z",
        "updated_at": "2023-10-27T10:05:00Z",
        "model_used": "llama3.2:latest",
        "message_count": 5
      }
    ],
    "total": 1
  }
  ```

### `POST /sessions`
- **Description**: Creates a new chat session.
- **Request Body**:
  ```json
  {
    "title": "A title for the chat",
    "model": "llama3.2:latest"
  }
  ```
- **Success Response (201 Created)**:
  ```json
  {
    "session": { ... a full ChatSession object ... },
    "session_id": "new-uuid-string"
  }
  ```

### `GET /sessions/{id}`
- **Description**: Retrieves a specific session and all its messages.
- **URL Parameters**:
    - `id` (string, required): The UUID of the session.
- **Request Body**: None.
- **Success Response (200 OK)**:
  ```json
  {
    "session": { ... ChatSession object ... },
    "messages": [ { ... ChatMessage object ... } ]
  }
  ```

### `DELETE /sessions/{id}`
- **Description**: Deletes a session and all its related content (messages, documents).
- **URL Parameters**:
    - `id` (string, required): The UUID of the session to delete.
- **Request Body**: None.
- **Success Response (200 OK)**:
  ```json
  {
    "message": "Session deleted successfully",
    "deleted_session_id": "deleted-uuid-string"
  }
  ```

### `POST /sessions/cleanup`
- **Description**: Triggers a cleanup process to remove any empty sessions from the database.
- **Request Body**: None.
- **Success Response (200 OK)**:
  ```json
  {
    "message": "Cleaned up 2 empty sessions",
    "cleanup_count": 2
  }
  ```
---

## 3. Chat Interaction

### `POST /sessions/{id}/messages`
- **Description**: Sends a message within a session and gets an AI response. The backend automatically handles conversation history and RAG context injection.
- **URL Parameters**:
    - `id` (string, required): The UUID of the session.
- **Request Body**:
  ```json
  {
    "message": "What is the capital of France?",
    "model": "optional-model-override"
  }
  ```
- **Success Response (200 OK)**:
  ```json
  {
    "response": "The capital of France is Paris.",
    "session": { ... updated ChatSession object ... },
    "user_message_id": "user-message-uuid",
    "ai_message_id": "ai-message-uuid"
  }
  ```

### `POST /chat` (Legacy)
- **Description**: Handles a legacy, single-turn chat request without session management. It accepts a message and an optional conversation history and returns a single AI response.
- **Request Body**:
  ```json
  {
    "message": "What is the capital of France?",
    "model": "llama3.2:latest",
    "conversation_history": [
        { "role": "user", "content": "Hello" },
        { "role": "assistant", "content": "Hi there!" }
    ]
  }
  ```
- **Success Response (200 OK)**:
  ```json
  {
    "response": "The capital of France is Paris.",
    "model": "llama3.2:latest",
    "message_count": 3
  }
  ```

---

## 4. Document Management

### `POST /sessions/{id}/upload`
- **Description**: Uploads one or more PDF files to a specific session.
- **Content-Type**: `multipart/form-data`
- **URL Parameters**:
    - `id` (string, required): The UUID of the session.
- **Request Body**: A `FormData` object containing file(s). Each file should have a unique key (e.g., `file_0`, `file_1`).
- **Success Response (200 OK)**:
  ```json
  {
    "message": "Processed 1 PDF files",
    "uploaded_files": [
      {
        "filename": "document1.pdf",
        "file_id": "uuid-for-doc-1",
        "text_length": 15023
      }
    ],
    "processing_results": [
        {
            "success": true,
            "filename": "document1.pdf",
            "file_id": "uuid-for-doc-1",
            "text_length": 15023
        }
    ],
    "session_documents": [ ... list of all documents in session ... ],
    "total_session_documents": 1
  }
  ``` 