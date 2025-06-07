"use client"

import * as React from "react"
import {
  ChatBubble,
  ChatBubbleAvatar,
  ChatBubbleMessage,
  ChatBubbleAction,
  ChatBubbleActionWrapper
} from "@/components/ui/chat-bubble"
import { Copy, RefreshCcw, ThumbsUp, ThumbsDown, Volume2, MoreHorizontal, ChevronDown } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChatMessage } from "@/lib/api"

interface ConversationPageProps {
  messages: ChatMessage[]
  isLoading?: boolean
  className?: string
  onAction?: (action: string, messageId: string, messageContent: string) => void
}

const actionIcons = [
  { icon: Copy, type: "Copy", action: "copy" },
  { icon: ThumbsUp, type: "Like", action: "like" },
  { icon: ThumbsDown, type: "Dislike", action: "dislike" },
  { icon: Volume2, type: "Speak", action: "speak" },
  { icon: RefreshCcw, type: "Regenerate", action: "regenerate" },
  { icon: MoreHorizontal, type: "More", action: "more" },
]

export function ConversationPage({ 
  messages, 
  isLoading = false,
  className = "",
  onAction
}: ConversationPageProps) {
  const handleAction = (action: string, messageId: string, messageContent: string) => {
    if (onAction) {
      onAction(action, messageId, messageContent)
      return
    }
    
    console.log(`Action ${action} clicked for message ${messageId}`)
    // Handle different actions here
    switch (action) {
      case 'copy':
        navigator.clipboard.writeText(messageContent)
        break
      case 'regenerate':
        // Regenerate AI response
        break
      case 'like':
        // Add like reaction
        break
      case 'dislike':
        // Add dislike reaction
        break
      case 'speak':
        // Text to speech
        break
      case 'more':
        // Show more options
        break
    }
  }

  return (
    <div className={`flex flex-col h-full bg-black relative ${className}`}>
      <ScrollArea className="flex-1 px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message) => {
            const variant = message.sender === "user" ? "sent" : "received"
            const isUser = message.sender === "user"
            
            return (
              <div key={message.id} className="w-full group">
                <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
                  {!isUser && (
                    <ChatBubbleAvatar 
                      fallback="L" 
                      className="mt-1 flex-shrink-0"
                    />
                  )}
                  
                  <div className={`flex flex-col space-y-2 ${isUser ? 'items-end' : 'items-start'} max-w-[80%]`}>
                    <div
                      className={`rounded-2xl px-4 py-3 ${
                        isUser 
                          ? "bg-white text-black" 
                          : "bg-gray-800 text-gray-100"
                      }`}
                    >
                      {message.isLoading ? (
                        <div className="flex items-center space-x-2">
                          <div className="flex space-x-1">
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                          </div>
                        </div>
                      ) : (
                        <div className="whitespace-pre-wrap text-sm leading-relaxed">
                          {message.content}
                        </div>
                      )}
                    </div>
                    
                    {!isUser && !message.isLoading && (
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        {actionIcons.map(({ icon: Icon, type, action }) => (
                          <button
                            key={action}
                            onClick={() => handleAction(action, message.id, message.content)}
                            className="p-1.5 hover:bg-gray-700 rounded-md transition-colors text-gray-400 hover:text-gray-200"
                            title={type}
                          >
                            <Icon className="w-3.5 h-3.5" />
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  {isUser && (
                    <ChatBubbleAvatar 
                      src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=64&h=64&q=80&crop=faces&fit=crop"
                      fallback="U" 
                      className="mt-1 flex-shrink-0"
                    />
                  )}
                </div>
              </div>
            )
          })}
          
          {/* Loading indicator for new message */}
          {isLoading && (
            <div className="w-full group">
              <div className="flex gap-3 justify-start">
                <ChatBubbleAvatar fallback="L" className="mt-1 flex-shrink-0" />
                <div className="flex flex-col space-y-2 items-start max-w-[80%]">
                  <div className="rounded-2xl px-4 py-3 bg-gray-800 text-gray-100">
                    <div className="flex items-center space-x-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
                      )}
        </div>
      </ScrollArea>
      
      {/* Scroll to bottom button */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
        <button
          onClick={() => {
            const scrollArea = document.querySelector('[data-radix-scroll-area-viewport]')
            if (scrollArea) {
              scrollArea.scrollTop = scrollArea.scrollHeight
            }
          }}
          className="p-2 bg-gray-800 border border-gray-700 rounded-full hover:bg-gray-700 transition-colors shadow-lg"
        >
          <ChevronDown className="w-4 h-4 text-gray-400" />
        </button>
      </div>
    </div>
  )
} 