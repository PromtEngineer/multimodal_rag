"use client";

import { useState, useEffect } from "react"
import { LocalGPTChat } from "@/components/ui/localgpt-chat"
import { SessionSidebar } from "@/components/ui/session-sidebar"
import { SessionChat } from "@/components/ui/session-chat"
import { chatAPI, ChatSession } from "@/lib/api"

export function Demo() {
    const [currentSessionId, setCurrentSessionId] = useState<string | undefined>()
    const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
    const [showConversation, setShowConversation] = useState(false)
    const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking')

    console.log('Demo component rendering...')

    useEffect(() => {
        console.log('Demo component mounted')
        checkBackendHealth()
    }, [])

    const checkBackendHealth = async () => {
        try {
            const health = await chatAPI.checkHealth()
            setBackendStatus('connected')
            console.log('Backend connected:', health)
        } catch (error) {
            console.error('Backend health check failed:', error)
            setBackendStatus('error')
        }
    }

    const handleSessionSelect = (sessionId: string) => {
        setCurrentSessionId(sessionId)
        setShowConversation(true)
    }

    const handleNewSession = async () => {
        try {
            const newSession = await chatAPI.createSession()
            setCurrentSessionId(newSession.id)
            setCurrentSession(newSession)
            setShowConversation(true)
        } catch (error) {
            console.error('Failed to create session:', error)
        }
    }

    const handleSessionChange = (session: ChatSession) => {
        setCurrentSession(session)
    }

    const handleSessionDelete = (deletedSessionId: string) => {
        if (currentSessionId === deletedSessionId) {
            setCurrentSessionId(undefined)
            setCurrentSession(null)
        }
    }

    const handleStartConversation = () => {
        if (backendStatus === 'connected') {
            handleNewSession()
        } else {
            setShowConversation(true)
        }
    }

    return (
        <div className="flex h-screen w-screen flex-row">
            {/* Session Sidebar */}
            {showConversation && (
                <SessionSidebar
                    currentSessionId={currentSessionId}
                    onSessionSelect={handleSessionSelect}
                    onNewSession={() => handleNewSession()}
                    onSessionDelete={handleSessionDelete}
                />
            )}
            
            <main className={`flex h-screen grow flex-col transition-all duration-200 ${
                showConversation ? 'overflow-hidden' : 'overflow-auto ml-12'
            }`}>
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
                                            Backend connected • Session-based chat ready
                                        </div>
                                    )}
                                    {backendStatus === 'error' && (
                                        <div className="flex items-center gap-2 text-red-400">
                                            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                                            Backend offline • Start backend server to enable chat
                                        </div>
                                    )}
                                </div>
                                
                                <button
                                    onClick={handleStartConversation}
                                    disabled={backendStatus === 'checking'}
                                    className="px-6 py-3 bg-white text-black rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {backendStatus === 'connected' ? 'Start New Chat Session' : 'Backend Required'}
                                </button>
                                
                                {backendStatus === 'error' && (
                                    <div className="text-center text-xs text-gray-400 max-w-md">
                                        <p>To enable chat functionality:</p>
                                        <p className="mt-1 font-mono bg-gray-900 px-2 py-1 rounded">
                                            cd backend && python server.py
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="h-full flex flex-col">
                        {/* Header */}
                        <div className="p-4 border-b border-gray-800 bg-black">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <h1 className="text-xl font-medium text-white">
                                        {currentSession?.title || 'New Chat'}
                                    </h1>
                                    {currentSession && (
                                        <div className="flex items-center gap-4 text-sm text-gray-400">
                                            <span>{currentSession.message_count} messages</span>
                                            <span>{currentSession.model_used}</span>
                                        </div>
                                    )}
                                </div>
                                <div className="flex gap-2">
                                    <div className="flex items-center gap-2 text-sm">
                                        {backendStatus === 'connected' && (
                                            <div className="flex items-center gap-2 text-green-400">
                                                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                                Connected
                                            </div>
                                        )}
                                        {backendStatus === 'error' && (
                                            <div className="flex items-center gap-2 text-red-400">
                                                <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                                                Offline
                                            </div>
                                        )}
                                    </div>
                                    <button
                                        onClick={() => handleNewSession()}
                                        className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition-colors text-sm"
                                    >
                                        Back to Home
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        {/* Session Chat */}
                        <SessionChat
                            sessionId={currentSessionId}
                            onSessionChange={handleSessionChange}
                            className="flex-1"
                        />
                    </div>
                )}
            </main>
        </div>
    );
} 