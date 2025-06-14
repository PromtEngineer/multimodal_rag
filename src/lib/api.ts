const API_BASE_URL = 'http://localhost:8000';

// 🆕 Simple UUID generator for client-side message IDs
export const generateUUID = () => {
  if (typeof window !== 'undefined' && window.crypto && window.crypto.randomUUID) {
    return window.crypto.randomUUID();
  }
  // Fallback for older browsers or non-secure contexts
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

export interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  isLoading?: boolean;
  metadata?: Record<string, unknown>;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  model_used: string;
  message_count: number;
}

export interface ChatRequest {
  message: string;
  model?: string;
  conversation_history?: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

export interface ChatResponse {
  response: string;
  model: string;
  message_count: number;
}

export interface HealthResponse {
  status: string;
  ollama_running: boolean;
  available_models: string[];
  database_stats?: {
    total_sessions: number;
    total_messages: number;
    most_used_model: string | null;
  };
}

export interface ModelsResponse {
  generation_models: string[];
  embedding_models: string[];
}

export interface SessionResponse {
  sessions: ChatSession[];
  total: number;
}

export interface SessionChatResponse {
  response: string;
  session: ChatSession;
  user_message_id: string;
  ai_message_id: string;
}

class ChatAPI {
  async checkHealth(): Promise<HealthResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: request.message,
          model: request.model || 'llama3.2:latest',
          conversation_history: request.conversation_history || [],
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Chat API error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Chat API failed:', error);
      throw error;
    }
  }

  // Convert ChatMessage array to conversation history format
  messagesToHistory(messages: ChatMessage[]): Array<{ role: 'user' | 'assistant'; content: string }> {
    return messages
      .filter(msg => !msg.isLoading && msg.content.trim())
      .map(msg => ({
        role: msg.sender,
        content: msg.content,
      }));
  }

  // Session Management
  async getSessions(): Promise<SessionResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`);
      if (!response.ok) {
        throw new Error(`Failed to get sessions: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get sessions failed:', error);
      throw error;
    }
  }

  async createSession(title: string = 'New Chat', model: string = 'llama3.2:latest'): Promise<ChatSession> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ title, model }),
      });

      if (!response.ok) {
        throw new Error(`Failed to create session: ${response.status}`);
      }

      const data = await response.json();
      return data.session;
    } catch (error) {
      console.error('Create session failed:', error);
      throw error;
    }
  }

  async getSession(sessionId: string): Promise<{ session: ChatSession; messages: ChatMessage[] }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`);
      if (!response.ok) {
        throw new Error(`Failed to get session: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Get session failed:', error);
      throw error;
    }
  }

  async sendSessionMessage(sessionId: string, message: string, model?: string): Promise<SessionChatResponse> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          ...(model && { model }),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Session chat error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Session chat failed:', error);
      throw error;
    }
  }

  async deleteSession(sessionId: string): Promise<{ message: string; deleted_session_id: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Delete session error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Delete session failed:', error);
      throw error;
    }
  }

  async cleanupEmptySessions(): Promise<{ message: string; cleanup_count: number }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/cleanup`);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Cleanup sessions error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Cleanup sessions failed:', error);
      throw error;
    }
  }

  async uploadFiles(sessionId: string, files: File[]): Promise<{ 
    message: string; 
    uploaded_files: {filename: string, stored_path: string}[]; 
  }> {
    try {
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('files', file, file.name);
      });

      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Upload failed' }));
        throw new Error(`Upload error: ${errorData.error || response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('File upload failed:', error);
      throw error;
    }
  }

  async indexDocuments(sessionId: string): Promise<{ message: string }> {
    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Indexing failed' }));
        throw new Error(`Indexing error: ${errorData.error || response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Indexing failed:', error);
      throw error;
    }
  }

  // Legacy upload function - can be removed if no longer needed
  async uploadPDFs(sessionId: string, files: File[]): Promise<{ 
    message: string; 
    uploaded_files: any[]; 
    processing_results: any[];
    session_documents: any[];
    total_session_documents: number;
  }> {
    try {
      // Test if files have content and show size info
      let totalSize = 0;
      for (const file of files) {
        if (file.size === 0) {
          throw new Error(`File ${file.name} is empty (0 bytes)`);
        }
        totalSize += file.size;
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        console.log(`📄 File ${file.name}: ${sizeMB}MB (${file.size} bytes), type: ${file.type}`);
      }
      
      const totalSizeMB = (totalSize / (1024 * 1024)).toFixed(2);
      console.log(`📄 Total upload size: ${totalSizeMB}MB`);
      
      if (totalSize > 50 * 1024 * 1024) { // 50MB limit
        throw new Error(`Total file size ${totalSizeMB}MB exceeds 50MB limit`);
      }
      
      const formData = new FormData();
      
      // Use a generic field name 'file' that the backend expects
      let i = 0;
      for (const file of files) {
        formData.append(`file_${i}`, file, file.name);
        i++;
      }
      
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`Upload error: ${errorData.error || response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('PDF upload failed:', error);
      throw error;
    }
  }

  // Convert database message format to ChatMessage format
  convertDbMessage(dbMessage: Record<string, unknown>): ChatMessage {
    return {
      id: dbMessage.id as string,
      content: dbMessage.content as string,
      sender: dbMessage.sender as 'user' | 'assistant',
      timestamp: dbMessage.timestamp as string,
      metadata: dbMessage.metadata as Record<string, unknown> | undefined,
    };
  }

  // Create a new ChatMessage with UUID (for loading states)
  createMessage(
    content: string, 
    sender: 'user' | 'assistant', 
    isLoading = false
  ): ChatMessage {
    return {
      id: generateUUID(),
      content,
      sender,
      timestamp: new Date().toISOString(),
      isLoading,
    };
  }

  // ---------------- Models ----------------
  async getModels(): Promise<ModelsResponse> {
    const resp = await fetch(`${API_BASE_URL}/models`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch models list: ${resp.status}`);
    }
    return resp.json();
  }

  async getSessionDocuments(sessionId: string): Promise<{ files: string[]; file_count: number; session: ChatSession }> {
    const resp = await fetch(`${API_BASE_URL}/sessions/${sessionId}/documents`);
    if (!resp.ok) {
      throw new Error(`Failed to fetch session documents: ${resp.status}`);
    }
    return resp.json();
  }
}

export const chatAPI = new ChatAPI(); 