"use client"

import * as React from "react"
import { ConversationPage } from "./conversation-page"
import { ChatInput } from "./chat-input"
import { EmptyChatState } from "./empty-chat-state"
import { ChatMessage, ChatSession, chatAPI, generateUUID } from "@/lib/api"
import { AttachedFile } from "@/lib/types"
import { useEffect, useState, forwardRef, useImperativeHandle } from "react"
import { Button } from "./button"

interface SessionChatProps {
  sessionId?: string
  onSessionChange?: (session: ChatSession) => void
  onNewMessage?: (message: ChatMessage) => void
  className?: string
}

// Export sendMessage function for parent components
export interface SessionChatRef {
  sendMessage: (content: string, attachedFiles?: AttachedFile[]) => Promise<void>
  currentSession: ChatSession | null
}

export const SessionChat = forwardRef<SessionChatRef, SessionChatProps>(({ 
  sessionId,
  onSessionChange,
  onNewMessage,
  className = ""
}, ref) => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [uploadedFiles, setUploadedFiles] = useState<{filename: string, stored_path: string}[]>([])
  const [isIndexed, setIsIndexed] = useState(false)
  
  const apiService = chatAPI

  // Expose functions to parent component (moved after sendMessage definition)

  // Load session when sessionId changes
  useEffect(() => {
    if (sessionId) {
      // Only load session if we don't already have the current session
      // This prevents overriding messages when a new session is created
      if (!currentSession || currentSession.id !== sessionId) {
        loadSession(sessionId)
      }
    } else {
      // Clear messages if no session
      setMessages([])
      setCurrentSession(null)
    }
  }, [sessionId, currentSession]) // eslint-disable-line react-hooks/exhaustive-deps

  const loadSession = async (id: string) => {
    try {
      setError(null)
      const { session, messages: sessionMessages } = await apiService.getSession(id)
      
      const convertedMessages = sessionMessages.map((msg: unknown) => apiService.convertDbMessage(msg as Record<string, unknown>))
      setMessages(convertedMessages)
      setCurrentSession(session)
      
      if (onSessionChange) {
        onSessionChange(session)
      }
    } catch (error) {
      console.error('Failed to load session:', error)
      setError('Failed to load session')
    }
  }

  const sendMessage = async (content: string, attachedFiles?: AttachedFile[]) => {
    // --- Guard Clauses ---
    // If files are being indexed, do nothing.
    if (uploadedFiles.length > 0 && !isIndexed) {
      console.warn("sendMessage called while waiting for indexing. Action blocked.");
      return;
    }
    // If no content and no files, do nothing.
    if (!content.trim() && (!attachedFiles || attachedFiles.length === 0)) return;

    try {
      setError(null)
      
      let activeSessionId = sessionId
      if (!activeSessionId) {
        try {
          const newSession = await apiService.createSession()
          activeSessionId = newSession.id
          setCurrentSession(newSession)
          if (onSessionChange) onSessionChange(newSession)
        } catch (error) {
          console.error('Failed to create session:', error)
          setError('Failed to create session')
          return
        }
      }

      // --- Action Router: Decide if this is an upload or a chat message ---
      
      // A) UPLOAD ACTION: If files are attached, this action's priority is to upload. Ignore any text content.
      if (attachedFiles && attachedFiles.length > 0) {
        setIsLoading(true)
        try {
          const files = attachedFiles.map(af => af.file)
          const uploadResult = await apiService.uploadFiles(activeSessionId, files)
          console.log('✅ Files uploaded successfully:', uploadResult)
          
          setUploadedFiles(uploadResult.uploaded_files)
          setIsIndexed(false)

          const uploadMessage = apiService.createMessage(
            `📎 Uploaded ${uploadResult.uploaded_files.length} file(s): ${uploadResult.uploaded_files.map(f => f.filename).join(', ')}. Please click 'Index Documents' to chat with them.`,
            'assistant'
          )
          setMessages(prev => [...prev, uploadMessage])
        } catch (error) {
          console.error('❌ Failed to upload files:', error)
          const errorMessage = apiService.createMessage('❌ Failed to upload files. Please try again.', 'assistant')
          setMessages(prev => [...prev, errorMessage])
        } finally {
          setIsLoading(false)
        }
        return; // End the function here.
      }

      // B) CHAT ACTION: If no files, it's a standard chat message.
      if (!content.trim()) return;

      const userMessage = apiService.createMessage(content, 'user')
      setMessages(prev => [...prev, userMessage])
      if (onNewMessage) onNewMessage(userMessage)

      setIsLoading(true)

      const response = await apiService.sendSessionMessage(activeSessionId, content)
      
      const aiMessage: ChatMessage = {
        id: response.ai_message_id || generateUUID(),
        content: response.response,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      }
      setMessages(prev => [...prev, aiMessage])
      
      if (response.session) {
        setCurrentSession(response.session)
        if (onSessionChange) onSessionChange(response.session)
      }
      if (onNewMessage) onNewMessage(aiMessage)

    } catch (error) {
      console.error('Failed to send message:', error)
      setError('Failed to send message')
    } finally {
      setIsLoading(false)
    }
  }

  const handleIndexDocuments = async () => {
    if (!currentSession) return;

    setIsLoading(true);
    setError(null);
    try {
      const result = await apiService.indexDocuments(currentSession.id);
      console.log('✅ Indexing complete:', result);

      const indexMessage = apiService.createMessage(
        `✅ ${result.message}`,
        'assistant'
      );
      setMessages(prev => [...prev, indexMessage]);
      setIsIndexed(true);
      setUploadedFiles([]); // Clear uploaded files after indexing

    } catch (error) {
      console.error('❌ Failed to index documents:', error);
      const errorMessage = apiService.createMessage(
        '❌ Failed to index documents. Please try again.',
        'assistant'
      );
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  // Expose functions to parent component
  useImperativeHandle(ref, () => ({
    sendMessage,
    currentSession
  }))

  const handleAction = async (action: string, messageId: string, messageContent: string) => {
    console.log(`Action ${action} on message ${messageId}`)
    
    switch (action) {
      case 'copy':
        await navigator.clipboard.writeText(messageContent)
        break
      case 'regenerate':
        // Find the user message before this AI message and resend it
        const messageIndex = messages.findIndex(m => m.id === messageId)
        if (messageIndex > 0 && messages[messageIndex].sender === 'assistant') {
          const userMessage = messages[messageIndex - 1]
          if (userMessage.sender === 'user') {
            // Remove the AI message and resend the user message
            setMessages(prev => prev.filter(m => m.id !== messageId))
            await sendMessage(userMessage.content)
          }
        }
        break
      default:
        // Handle other actions
        break
    }
  }

  const showEmptyState = (!sessionId || messages.length === 0) && !isLoading

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {error && (
        <div className="bg-red-900 text-red-200 px-4 py-2 text-sm flex-shrink-0">
          {error}
        </div>
      )}

      {/* Conversation area (may be empty) */}
      {showEmptyState ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-gray-500 text-lg select-none">What can I help you find?</div>
        </div>
      ) : (
        <ConversationPage 
          messages={messages}
          isLoading={isLoading}
          onAction={handleAction}
          className="flex-1 min-h-0 overflow-hidden"
        />
      )}

      {/* Input section always present */}
      <div className="flex-shrink-0 sticky bottom-0 z-10 bg-black/90 backdrop-blur-md">
        {uploadedFiles.length > 0 && !isIndexed && (
          <div className="p-2 text-center bg-yellow-100 dark:bg-yellow-900 border-t border-b border-gray-200 dark:border-gray-700">
            <Button onClick={handleIndexDocuments} disabled={isLoading}>
              {isLoading ? 'Indexing...' : 'Index Documents to Enable Chat'}
            </Button>
          </div>
        )}
        <ChatInput
          onSendMessage={sendMessage}
          disabled={isLoading || (uploadedFiles.length > 0 && !isIndexed)}
          placeholder="Message localGPT..."
        />
      </div>
    </div>
  )
})

SessionChat.displayName = "SessionChat" 