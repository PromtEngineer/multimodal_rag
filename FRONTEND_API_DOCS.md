# Frontend API Documentation (`src/lib/api.ts`)

This document outlines the methods available in the `ChatAPI` service class, which acts as the primary interface between the Next.js frontend and the Python backend.

---

## `ChatAPI` Class

An instance of this class is exported as `chatAPI` and is used throughout the frontend components.

### 1. Health Check

#### `checkHealth()`
- **Purpose**: Checks the health of the backend server.
- **Inputs**: None.
- **Returns**: `Promise<HealthResponse>`
- **Example `HealthResponse` object**:
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

### 2. Session Management

#### `getSessions()`
- **Purpose**: Fetches a list of all chat sessions.
- **Inputs**: None.
- **Returns**: `Promise<SessionResponse>`
- **Example `SessionResponse` object**:
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

#### `createSession(title?: string, model?: string)`
- **Purpose**: Creates a new chat session.
- **Inputs**:
    - `title` (optional, string): The initial title for the session. Defaults to "New Chat".
    - `model` (optional, string): The AI model to use. Defaults to "llama3.2:latest".
- **Returns**: `Promise<ChatSession>` (the newly created session object).

#### `getSession(sessionId: string)`
- **Purpose**: Retrieves a single session and all its associated messages.
- **Inputs**:
    - `sessionId` (string, required): The unique ID of the session.
- **Returns**: `Promise<{ session: ChatSession; messages: ChatMessage[] }>`

#### `sendSessionMessage(sessionId: string, message: string, model?: string)`
- **Purpose**: Sends a message to a specific session and gets the AI response.
- **Inputs**:
    - `sessionId` (string, required): The ID of the session to send the message to.
    - `message` (string, required): The user's message content.
    - `model` (optional, string): The model to use for the response.
- **Returns**: `Promise<SessionChatResponse>`
- **Example `SessionChatResponse` object**:
  ```json
  {
    "response": "This is the AI's answer.",
    "session": { ...updated ChatSession object... },
    "user_message_id": "uuid-string-user-msg",
    "ai_message_id": "uuid-string-ai-msg"
  }
  ```

#### `deleteSession(sessionId: string)`
- **Purpose**: Deletes a session and all its related data.
- **Inputs**:
    - `sessionId` (string, required): The ID of the session to delete.
- **Returns**: `Promise<{ message: string; deleted_session_id: string }>`

#### `cleanupEmptySessions()`
- **Purpose**: Triggers a cleanup process on the backend to remove empty sessions.
- **Inputs**: None.
- **Returns**: `Promise<{ message:string; cleanup_count: number }>`

---

### 3. Document Handling

#### `uploadPDFs(sessionId: string, files: File[])`
- **Purpose**: Uploads an array of PDF files to a specific session.
- **Inputs**:
    - `sessionId` (string, required): The session to associate the files with.
    - `files` (`File[]`, required): An array of `File` objects to upload.
- **Returns**: `Promise<UploadResponse>`
- **Example `UploadResponse` object**:
  ```json
  {
    "message": "Processed 2 PDF files",
    "uploaded_files": [
      {
        "filename": "document1.pdf",
        "file_id": "uuid-for-doc-1",
        "text_length": 15023
      }
    ],
    "processing_results": [ ... ],
    "session_documents": [ ... ],
    "total_session_documents": 3
  }
  ```

---

### 4. Utility Methods

#### `createMessage(content: string, sender: 'user' | 'assistant', isLoading?: boolean)`
- **Purpose**: A client-side utility to create a `ChatMessage` object with a unique ID. Used for optimistic UI updates.
- **Inputs**:
    - `content` (string): The message text.
    - `sender` (`'user' | 'assistant'`): The sender of the message.
    - `isLoading` (optional, boolean): A flag to indicate a pending AI response.
- **Returns**: `ChatMessage` (a complete message object ready for the UI).

#### `convertDbMessage(dbMessage: Record<string, unknown>)`
- **Purpose**: A client-side utility to convert a message object from the database format into the frontend's `ChatMessage` format.
- **Inputs**:
    - `dbMessage` (object): A message object as received from the backend.
- **Returns**: `ChatMessage`. 