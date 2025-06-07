"use client";

import { useState, useEffect } from "react"
import { LocalGPTChat } from "@/components/ui/localgpt-chat"
import { SessionNavBar } from "@/components/ui/sidebar"
import { ConversationPage } from "@/components/ui/conversation-page"
import { ArrowUp } from "lucide-react"
import { chatAPI, ChatMessage } from "@/lib/api"

export function Demo() {
    const [showConversation, setShowConversation] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [inputValue, setInputValue] = useState("")
    const [messages, setMessages] = useState<ChatMessage[]>([])
    const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking')
    const [nextMessageId, setNextMessageId] = useState(1)

    // Check backend health on component mount
    useEffect(() => {
        checkBackendHealth()
    }, [])

    const checkBackendHealth = async () => {
        try {
            await chatAPI.checkHealth()
            setBackendStatus('connected')
        } catch (error) {
            console.error('Backend health check failed:', error)
            setBackendStatus('error')
        }
    }

    const handleStartConversation = () => {
        setShowConversation(true)
        // Start with a sample conversation or empty
        if (messages.length === 0) {
            setMessages([
                chatAPI.createMessage(1, "Hey, can you tell me a joke?", "user"),
            ])
            setNextMessageId(2)
            // Send the first message automatically
            sendMessageToAPI("Hey, can you tell me a joke?", [])
        }
    }

    const handleBackToChat = () => {
        setShowConversation(false)
    }

    const sendMessageToAPI = async (message: string, currentMessages: ChatMessage[]) => {
        try {
            setIsLoading(true)
            
            // Convert messages to conversation history
            const conversationHistory = chatAPI.messagesToHistory(currentMessages)
            
            const response = await chatAPI.sendMessage({
                message,
                conversation_history: conversationHistory
            })

            // Add AI response
            const aiMessage = chatAPI.createMessage(
                nextMessageId + 1,
                response.response,
                "assistant"
            )

            setMessages(prev => [...prev, aiMessage])
            setNextMessageId(prev => prev + 2)
            
        } catch (error) {
            console.error('Failed to send message:', error)
            
            // Add error message
            const errorMessage = chatAPI.createMessage(
                nextMessageId + 1,
                `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please make sure the backend server is running.`,
                "assistant"
            )
            
            setMessages(prev => [...prev, errorMessage])
            setNextMessageId(prev => prev + 2)
        } finally {
            setIsLoading(false)
        }
    }

    const handleSendMessage = async () => {
        if (!inputValue.trim()) return

        // Add user message immediately
        const userMessage = chatAPI.createMessage(nextMessageId, inputValue.trim(), "user")
        const newMessages = [...messages, userMessage]
        setMessages(newMessages)
        
        const messageToSend = inputValue.trim()
        setInputValue("")
        setNextMessageId(prev => prev + 1)

        // Send to API
        await sendMessageToAPI(messageToSend, newMessages)
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
                                                                <div className="flex flex-col items-center gap-3">
                                        <div className="flex items-center gap-2 text-sm">
                                            {backendStatus === 'checking' && (
                                                <div className="flex items-center gap-2 text-gray-400">
                                                    <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
                                                    Connecting to backend...
                                                </div>
                                            )}
                                            {backendStatus === 'connected' && (
                                                <div className="flex items-center gap-2 text-green-400">
                                                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                                    Backend connected
                                                </div>
                                            )}
                                            {backendStatus === 'error' && (
                                                <div className="flex items-center gap-2 text-red-400">
                                                    <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                                                    Backend offline (using mock data)
                                                </div>
                                            )}
                                        </div>
                                        
                                        <button
                                            onClick={handleStartConversation}
                                            disabled={backendStatus === 'checking'}
                                            className="px-4 py-2 bg-white text-black rounded-lg hover:bg-gray-200 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                                        >
                                            {backendStatus === 'connected' ? 'Start Real Chat' : 'View Sample Conversation'}
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
                                messages={messages}
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