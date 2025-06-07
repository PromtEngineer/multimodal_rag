"use client"

import * as React from "react"
import { useState, useEffect } from "react"
import { Plus, MessageSquare, Calendar, Hash } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChatSession, chatAPI } from "@/lib/api"

interface SessionSidebarProps {
  currentSessionId?: string
  onSessionSelect: (sessionId: string) => void
  onNewSession: () => void
  className?: string
}

export function SessionSidebar({
  currentSessionId,
  onSessionSelect,
  onNewSession,
  className = ""
}: SessionSidebarProps) {
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Load sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  const loadSessions = async () => {
    try {
      setError(null)
      const response = await chatAPI.getSessions()
      setSessions(response.sessions)
    } catch (error) {
      console.error('Failed to load sessions:', error)
      setError('Failed to load sessions')
    } finally {
      setIsLoading(false)
    }
  }

  const handleNewSession = async () => {
    try {
      const newSession = await chatAPI.createSession()
      setSessions(prev => [newSession, ...prev])
      onSessionSelect(newSession.id)
      onNewSession()
    } catch (error) {
      console.error('Failed to create session:', error)
      setError('Failed to create session')
    }
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffInHours = (now.getTime() - date.getTime()) / (1000 * 60 * 60)
    
    if (diffInHours < 24) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } else if (diffInHours < 24 * 7) {
      return date.toLocaleDateString([], { weekday: 'short' })
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
    }
  }

  const truncateTitle = (title: string, maxLength: number = 25) => {
    return title.length > maxLength ? title.substring(0, maxLength) + '...' : title
  }

  return (
    <div className={`w-64 bg-gray-900 border-r border-gray-800 flex flex-col ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-white">Chats</h2>
          <Button
            onClick={handleNewSession}
            size="sm"
            className="w-8 h-8 p-0 bg-gray-800 hover:bg-gray-700 text-gray-300"
          >
            <Plus className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Sessions List */}
      <ScrollArea className="flex-1">
        <div className="p-2">
          {error && (
            <div className="mb-4 p-3 bg-red-900 text-red-200 text-sm rounded-lg">
              {error}
              <Button
                onClick={loadSessions}
                size="sm"
                className="ml-2 h-6 px-2 text-xs bg-red-800 hover:bg-red-700"
              >
                Retry
              </Button>
            </div>
          )}

          {isLoading ? (
            <div className="space-y-2">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="h-12 bg-gray-800 rounded-lg animate-pulse" />
              ))}
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No conversations yet</p>
              <p className="text-xs mt-1">Start a new chat to begin</p>
            </div>
          ) : (
            <div className="space-y-1">
              {sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => onSessionSelect(session.id)}
                  className={`w-full p-3 rounded-lg text-left transition-colors ${
                    currentSessionId === session.id
                      ? 'bg-blue-600 text-white'
                      : 'hover:bg-gray-800 text-gray-300'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <MessageSquare className="w-3 h-3 flex-shrink-0" />
                        <p className="font-medium text-sm truncate">
                          {truncateTitle(session.title)}
                        </p>
                      </div>
                      
                      <div className="flex items-center gap-3 text-xs opacity-70">
                        <div className="flex items-center gap-1">
                          <Hash className="w-3 h-3" />
                          <span>{session.message_count}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          <span>{formatDate(session.updated_at)}</span>
                        </div>
                      </div>
                      
                      <div className="text-xs mt-1 opacity-50">
                        {session.model_used}
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Footer with stats */}
      {sessions.length > 0 && (
        <div className="p-4 border-t border-gray-800 text-xs text-gray-400">
          <div className="flex justify-between">
            <span>{sessions.length} conversations</span>
            <span>
              {sessions.reduce((sum, s) => sum + s.message_count, 0)} messages
            </span>
          </div>
        </div>
      )}
    </div>
  )
} 