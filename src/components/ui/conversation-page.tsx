"use client"

import * as React from "react"
import { useRef, useEffect, useState } from "react"
import {
  ChatBubbleAvatar,
} from "@/components/ui/chat-bubble"
import { Copy, RefreshCcw, ThumbsUp, ThumbsDown, Volume2, MoreHorizontal, ChevronDown } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChatMessage } from "@/lib/api"
import { cn } from "@/lib/utils"

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

// Citation block toggle component
function Citation({doc, idx}: {doc:any, idx:number}){
  const [open,setOpen]=React.useState(false);
  const preview = (doc.text||'').replace(/\s+/g,' ').trim().slice(0,160) + ((doc.text||'').length>160?'…':'');
  return (
    <div onClick={()=>setOpen(!open)} className="text-xs text-gray-300 bg-gray-900/60 rounded p-2 cursor-pointer hover:bg-gray-800 transition">
      <span className="font-semibold mr-1">[{idx+1}]</span>{open?doc.text:preview}
    </div>
  );
}

// NEW: Expandable list of citations per assistant message
function CitationsBlock({docs}:{docs:any[]}){
  const scored = docs.filter(d => d.rerank_score || d.score || d._distance)
  scored.sort((a, b) => (b.rerank_score ?? b.score ?? 1/b._distance) - (a.rerank_score ?? a.score ?? 1/a._distance))
  const [expanded, setExpanded] = useState(false);

  if (scored.length === 0) return null;

  const visibleDocs = expanded ? scored : scored.slice(0, 5);

  return (
    <div className="mt-2 text-xs text-gray-400">
      <p className="font-semibold mb-1">Sources:</p>
      <div className="grid grid-cols-1 gap-2">
        {visibleDocs.map((doc, i) => <Citation key={doc.chunk_id || i} doc={doc} idx={i} />)}
      </div>
      {scored.length > 5 && (
        <button 
          onClick={() => setExpanded(!expanded)} 
          className="text-blue-400 hover:text-blue-300 mt-2 text-xs"
        >
          {expanded ? 'Show less' : `Show ${scored.length-5} more`}
        </button>
      )}
    </div>
  );
}

function StructuredMessageBlock({ content }: { content: Array<Record<string, any>> | { steps: any[] } }) {
  const steps: any[] = Array.isArray(content) ? content : (content as any).steps;
  return (
    <div className="flex flex-col">
      {steps.map((step: any, index: number) => {
        if (step.key && step.label) {
          // Stepwise progress rendering
          let statusIcon = step.status === 'done' ? '✅' : step.status === 'active' ? <span className="animate-pulse">⏳</span> : '•';
          let statusClass = step.status === 'active' ? 'bg-gray-800/60 border-l-4 border-blue-400 p-2 my-1 rounded' :
                          step.status === 'done' ? 'bg-gray-900/40 p-2 my-1 rounded' :
                          'p-2 my-1 text-gray-500';
          
          return (
            <div key={step.key} className={statusClass}>
              <div className="flex items-center gap-2 mb-1">
                <span>{statusIcon}</span>
                <span className={step.status === 'active' ? 'font-bold text-blue-200' : ''}>{step.label}</span>
              </div>
              {/* Details for each step */}
              {step.key === 'final' && step.details && typeof step.details === 'object' && !Array.isArray(step.details) ? (
                <div className="space-y-3">
                  <div className="whitespace-pre-wrap text-gray-100">
                    {step.details.answer}
                  </div>
                  {step.details.source_documents && step.details.source_documents.length > 0 && (
                    <CitationsBlock docs={step.details.source_documents} />
                  )}
                </div>
              ) : step.key === 'final' && step.details && typeof step.details === 'string' ? (
                <div className="whitespace-pre-wrap text-gray-100">
                  {step.details}
                </div>
              ) : Array.isArray(step.details) ? (
                // Handle array of sub-answers
                <div className="space-y-2">
                  {step.details.map((detail: any, idx: number) => (
                    <div key={idx} className="border-l-2 border-blue-400 pl-2">
                      <div className="font-semibold">{detail.question}</div>
                      <div>{detail.answer}</div>
                      {detail.source_documents && detail.source_documents.length > 0 && (
                        <CitationsBlock docs={detail.source_documents} />
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                // Handle string details
                step.details
              )}
            </div>
          );
        }
        return null;
      })}
    </div>
  );
}

export function ConversationPage({ 
  messages, 
  isLoading = false,
  className = "",
  onAction
}: ConversationPageProps) {
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showScrollButton, setShowScrollButton] = useState(false)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  // Monitor scroll position to show/hide scroll button
  useEffect(() => {
    const scrollContainer = scrollAreaRef.current?.querySelector('[data-radix-scroll-area-viewport]')
    if (!scrollContainer) return

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = scrollContainer
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100
      setShowScrollButton(!isNearBottom)
    }

    scrollContainer.addEventListener('scroll', handleScroll)
    handleScroll() // Check initial state

    return () => scrollContainer.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollToBottom = () => {
    // Try multiple methods to ensure scrolling works
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
    
    // Fallback: scroll the container directly
    setTimeout(() => {
      if (scrollAreaRef.current) {
        const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]') || scrollAreaRef.current
        if (scrollContainer) {
          scrollContainer.scrollTop = scrollContainer.scrollHeight
        }
      }
    }, 100)
  }

  const handleAction = (action: string, messageId: string, messageContent: string) => {
    if (onAction) {
      // For structured messages, we'll just join the text parts for copy/paste
      let contentToPass: string;
      if (typeof messageContent === 'string') {
        contentToPass = messageContent;
      } else if (Array.isArray(messageContent)) {
        contentToPass = (messageContent as any[]).map((s: any) => s.text || s.answer || '').join('\n');
      } else if (messageContent && typeof messageContent === 'object' && Array.isArray((messageContent as any).steps)) {
        // For {steps: Step[]} structure
        contentToPass = (messageContent as any).steps.map((s: any) => s.label + (s.details ? (typeof s.details === 'string' ? (': ' + s.details) : '') : '')).join('\n');
      } else {
        contentToPass = '';
      }
      onAction(action, messageId, contentToPass)
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
    <div className={`flex flex-col h-full bg-black relative overflow-hidden ${className}`}>
      <ScrollArea ref={scrollAreaRef} className="flex-1 h-full px-4 py-6 min-h-0">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.map((message) => {
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
                          {typeof message.content === 'string' 
                              ? message.content
                              : <StructuredMessageBlock content={message.content} />
                          }
                        </div>
                      )}
                    </div>
                    
                    {!isUser && !message.isLoading && (
                      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        {actionIcons.map(({ icon: Icon, type, action }) => (
                          <button
                            key={action}
                            onClick={() => {
                              const content = typeof message.content === 'string' ? message.content : (message.content as any[]).map(s => s.text || s.answer).join('\\n');
                              handleAction(action, message.id, content)
                            }}
                            className="p-1.5 hover:bg-gray-700 rounded-md transition-colors text-gray-400 hover:text-gray-200"
                            title={type}
                          >
                            <Icon className="w-3.5 h-3.5" />
                          </button>
                        ))}
                      </div>
                    )}

                    {/* Global citations only for plain-string messages */}
                    {(!isUser &&
                      !message.isLoading &&
                      typeof message.content === 'string' &&
                      Array.isArray((message as any).metadata?.source_documents) &&
                      (message as any).metadata.source_documents.length > 0) && (
                        <CitationsBlock docs={(message as any).metadata.source_documents} />
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
          
          {/* Invisible element to scroll to */}
          <div ref={messagesEndRef} />
        </div>
      </ScrollArea>
      
      {/* Scroll to bottom button - only show when not at bottom */}
      {showScrollButton && (
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 z-10">
          <button
            onClick={scrollToBottom}
            className="p-2 bg-gray-800 border border-gray-700 rounded-full hover:bg-gray-700 transition-all duration-200 shadow-lg group animate-in fade-in slide-in-from-bottom-2"
            title="Scroll to bottom"
          >
            <ChevronDown className="w-4 h-4 text-gray-400 group-hover:text-gray-200 transition-colors" />
          </button>
        </div>
      )}
    </div>
  )
} 