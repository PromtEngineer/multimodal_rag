"use client"

import * as React from "react"
import {
  ChatBubble,
  ChatBubbleAvatar,
  ChatBubbleMessage,
  ChatBubbleAction,
  ChatBubbleActionWrapper
} from "@/components/ui/chat-bubble"
import { Copy, RefreshCcw, ThumbsUp, ThumbsDown } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"

interface Message {
  id: number
  content: string
  sender: "user" | "assistant"
  timestamp: Date
  isLoading?: boolean
}

interface ConversationPageProps {
  messages: Message[]
  isLoading?: boolean
  className?: string
}

const actionIcons = [
  { icon: Copy, type: "Copy", action: "copy" },
  { icon: RefreshCcw, type: "Regenerate", action: "regenerate" },
  { icon: ThumbsUp, type: "Like", action: "like" },
  { icon: ThumbsDown, type: "Dislike", action: "dislike" },
]

export function ConversationPage({ 
  messages, 
  isLoading = false,
  className = ""
}: ConversationPageProps) {
  const handleAction = (action: string, messageId: number) => {
    console.log(`Action ${action} clicked for message ${messageId}`)
    // Handle different actions here
    switch (action) {
      case 'copy':
        // Copy message to clipboard
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
    }
  }

  return (
    <div className={`flex flex-col h-full bg-black ${className}`}>
      <ScrollArea className="flex-1 px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message) => {
            const variant = message.sender === "user" ? "sent" : "received"
            const isUser = message.sender === "user"
            
            return (
              <div key={message.id} className="w-full group">
                <ChatBubble variant={variant} className="max-w-none">
                  <ChatBubbleAvatar 
                    src={isUser 
                      ? "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=64&h=64&q=80&crop=faces&fit=crop"
                      : undefined
                    }
                    fallback={isUser ? "U" : "L"} 
                    className="mt-1"
                  />
                  
                  <div className="flex-1 space-y-2">
                    <ChatBubbleMessage 
                      variant={variant}
                      isLoading={message.isLoading}
                      className={`max-w-none ${
                        isUser 
                          ? "bg-blue-600 text-white" 
                          : "bg-gray-800 text-gray-100 border border-gray-700"
                      }`}
                    >
                      {!message.isLoading && (
                        <div className="whitespace-pre-wrap">
                          {message.content}
                        </div>
                      )}
                    </ChatBubbleMessage>
                    
                    {!isUser && !message.isLoading && (
                      <ChatBubbleActionWrapper className="opacity-0 group-hover:opacity-100 transition-opacity">
                        {actionIcons.map(({ icon: Icon, type, action }) => (
                          <ChatBubbleAction
                            key={action}
                            icon={<Icon className="w-3 h-3" />}
                            onClick={() => handleAction(action, message.id)}
                            className="text-gray-400 hover:text-gray-200 hover:bg-gray-700"
                          />
                        ))}
                      </ChatBubbleActionWrapper>
                    )}
                  </div>
                </ChatBubble>
              </div>
            )
          })}
          
          {/* Loading indicator for new message */}
          {isLoading && (
            <div className="w-full">
              <ChatBubble variant="received" className="max-w-none">
                <ChatBubbleAvatar fallback="L" className="mt-1" />
                <div className="flex-1">
                  <ChatBubbleMessage 
                    isLoading={true}
                    className="bg-gray-800 text-gray-100 border border-gray-700"
                  />
                </div>
              </ChatBubble>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
} 