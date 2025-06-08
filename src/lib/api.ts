const API_BASE_URL = 'http://localhost:8000';

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
      
      files.forEach((file, index) => {
        formData.append(`file_${index}`, file);
      });

      console.log('🔧 Frontend: FormData created, sending request...');
      
      // Create abort controller for timeout (increased for large files)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout for large PDFs
      
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/upload`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
        headers: {
          // Don't set Content-Type - let browser set it with boundary for multipart
        }
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(`PDF upload error: ${errorData.error || response.statusText}`);
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
      id: crypto.randomUUID(),
      content,
      sender,
      timestamp: new Date().toISOString(),
      isLoading,
    };
  }
}

export const chatAPI = new ChatAPI(); 