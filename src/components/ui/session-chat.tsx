"use client"

import * as React from "react"
import { ConversationPage } from "./conversation-page"
import { ChatInput } from "./chat-input"
import { ChatMessage, ChatSession, chatAPI } from "@/lib/api"
import { useEffect, useState, forwardRef, useImperativeHandle } from "react"

interface SessionChatProps {
  sessionId?: string
  onSessionChange?: (session: ChatSession) => void
  onNewMessage?: (message: ChatMessage) => void
  className?: string
}

// Export sendMessage function for parent components
export interface SessionChatRef {
  sendMessage: (content: string) => Promise<void>
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
  
  const apiService = chatAPI

  // Expose functions to parent component
  useImperativeHandle(ref, () => ({
    sendMessage,
    currentSession
  }))

  // Load session when sessionId changes
  useEffect(() => {
    if (sessionId) {
      loadSession(sessionId)
    } else {
      // Clear messages if no session
      setMessages([])
      setCurrentSession(null)
    }
  }, [sessionId])

  const loadSession = async (id: string) => {
    try {
      setError(null)
      const { session, messages: sessionMessages } = await apiService.getSession(id)
      
      const convertedMessages = sessionMessages.map((msg: any) => apiService.convertDbMessage(msg))
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

  const sendMessage = async (content: string) => {
    if (!sessionId || !content.trim()) return

    try {
      setError(null)
      
      // Add user message immediately
      const userMessage = apiService.createMessage(content, 'user')
      setMessages(prev => [...prev, userMessage])
      
      if (onNewMessage) {
        onNewMessage(userMessage)
      }

      // Start loading
      setIsLoading(true)

      // Send to API
      const response = await apiService.sendSessionMessage(sessionId, content)
      
      // Add AI response
      const aiMessage: ChatMessage = {
        id: response.ai_message_id,
        content: response.response,
        sender: 'assistant',
        timestamp: new Date().toISOString(),
      }
      
      setMessages(prev => [...prev, aiMessage])
      
      // Update session info
      if (response.session) {
        setCurrentSession(response.session)
        if (onSessionChange) {
          onSessionChange(response.session)
        }
      }

      if (onNewMessage) {
        onNewMessage(aiMessage)
      }

    } catch (error) {
      console.error('Failed to send message:', error)
      setError('Failed to send message')
    } finally {
      setIsLoading(false)
    }
  }

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

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {error && (
        <div className="bg-red-900 text-red-200 px-4 py-2 text-sm">
          {error}
        </div>
      )}
      
      <ConversationPage 
        messages={messages}
        isLoading={isLoading}
        onAction={handleAction}
        className="flex-1"
      />
      
      <ChatInput
        onSendMessage={sendMessage}
        disabled={!sessionId || isLoading}
        placeholder={sessionId ? "Message localGPT..." : "Select or create a session to start chatting"}
      />
    </div>
  )
}) 