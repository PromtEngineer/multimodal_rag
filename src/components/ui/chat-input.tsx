"use client"

import * as React from "react"
import { useState, useRef } from "react"
import { ArrowUp, Mic } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ChatInputProps {
  onSendMessage: (message: string) => Promise<void>
  disabled?: boolean
  placeholder?: string
  className?: string
}

export function ChatInput({ 
  onSendMessage, 
  disabled = false,
  placeholder = "Message localGPT...",
  className = ""
}: ChatInputProps) {
  const [message, setMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!message.trim() || disabled || isLoading) return

    const messageToSend = message.trim()
    setMessage("")
    setIsLoading(true)

    try {
      await onSendMessage(messageToSend)
    } catch (error) {
      console.error("Failed to send message:", error)
      // Restore message on error
      setMessage(messageToSend)
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e as any)
    }
  }

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value)
    
    // Auto-resize textarea
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px'
    }
  }

  return (
    <div className={`border-t border-gray-800 bg-black p-4 ${className}`}>
      <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
        <div className="relative flex items-end gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={handleInput}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={disabled || isLoading}
              rows={1}
              className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 pr-12 text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
              style={{ maxHeight: '120px' }}
            />
            
            {/* Send button positioned inside textarea */}
            <div className="absolute right-2 bottom-2">
              <Button
                type="submit"
                size="sm"
                disabled={!message.trim() || disabled || isLoading}
                className="w-8 h-8 p-0 rounded-full bg-white hover:bg-gray-100 text-black disabled:bg-gray-600 disabled:text-gray-400"
              >
                {isLoading ? (
                  <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <ArrowUp className="w-4 h-4" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </form>
    </div>
  )
} 