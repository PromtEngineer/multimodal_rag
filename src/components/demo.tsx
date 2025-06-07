"use client";

import { useState } from "react"
import { LocalGPTChat } from "@/components/ui/localgpt-chat"
import { SessionNavBar } from "@/components/ui/sidebar"
import { ConversationPage } from "@/components/ui/conversation-page"
import { ArrowUp } from "lucide-react"

// Sample conversation data
const sampleMessages = [
  {
    id: 1,
    content: "Hey, can you tell me a joke.",
    sender: "user" as const,
    timestamp: new Date(),
  },
  {
    id: 2,
    content: "Sure thing! Here's one for you:\n\nWhy did the computer keep sneezing?\n\nBecause it had a bad case of CAPS LOCK! 😄\n\nWant to hear another?",
    sender: "assistant" as const,
    timestamp: new Date(),
  },
  {
    id: 3,
    content: "tell me another one",
    sender: "user" as const,
    timestamp: new Date(),
  },
  {
    id: 4,
    content: "Alright, here's another:\n\nWhy don't programmers like nature?\n\nIt has too many bugs! 🐛😄\n\nWould you like one more?",
    sender: "assistant" as const,
    timestamp: new Date(),
  }
]

export function Demo() {
    const [showConversation, setShowConversation] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [inputValue, setInputValue] = useState("")

    const handleStartConversation = () => {
        setShowConversation(true)
    }

    const handleBackToChat = () => {
        setShowConversation(false)
    }

    const handleSendMessage = () => {
        if (inputValue.trim()) {
            setIsLoading(true)
            // Simulate AI response delay
            setTimeout(() => {
                setIsLoading(false)
            }, 3000)
            setInputValue("")
        }
    }

    const toggleLoading = () => {
        setIsLoading(!isLoading)
    }

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSendMessage()
        }
    }

    return (
        <div className="flex h-screen w-screen flex-row">
            <SessionNavBar />
            <main className="flex h-screen grow flex-col overflow-auto ml-12 transition-all duration-200">
                {!showConversation ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="space-y-4">
                            <LocalGPTChat />
                                                                <div className="flex justify-center gap-3">
                                        <button
                                            onClick={handleStartConversation}
                                            className="px-4 py-2 bg-white text-black rounded-lg hover:bg-gray-200 transition-colors text-sm"
                                        >
                                            View Sample Conversation
                                        </button>
                                    </div>
                        </div>
                    </div>
                ) : (
                    <div className="h-full flex flex-col">
                        <div className="p-4 border-b border-gray-800 bg-black">
                            <div className="flex items-center justify-between">
                                <h1 className="text-xl font-medium text-white">
                                    Conversation with localGPT
                                </h1>
                                <div className="flex gap-2">
                                    <button
                                        onClick={toggleLoading}
                                        className="px-3 py-1.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-xs"
                                    >
                                        {isLoading ? 'Stop Loading' : 'Test Loading'}
                                    </button>
                                    <button
                                        onClick={handleBackToChat}
                                        className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm"
                                    >
                                        Back to Chat
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <div className="flex-1 flex flex-col">
                            <ConversationPage 
                                messages={sampleMessages}
                                isLoading={isLoading}
                            />
                            
                            {/* Bottom input area */}
                            <div className="p-4 border-t border-gray-800 bg-black">
                                <div className="max-w-4xl mx-auto">
                                    <div className="flex items-center gap-3">
                                        <div className="flex-1 relative">
                                            <textarea
                                                value={inputValue}
                                                onChange={(e) => setInputValue(e.target.value)}
                                                onKeyPress={handleKeyPress}
                                                placeholder="Ask anything"
                                                className="w-full bg-gray-900 border border-gray-700 rounded-xl px-4 py-3 pr-12 text-white placeholder-gray-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent min-h-[48px] max-h-32"
                                                rows={1}
                                                style={{
                                                    scrollbarWidth: 'none',
                                                    msOverflowStyle: 'none'
                                                }}
                                            />
                                            <button
                                                onClick={handleSendMessage}
                                                disabled={!inputValue.trim()}
                                                className="absolute right-2 top-1/2 transform -translate-y-1/2 w-8 h-8 flex items-center justify-center bg-white text-black rounded-full hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:bg-gray-600 disabled:text-gray-400"
                                            >
                                                <ArrowUp className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
} 