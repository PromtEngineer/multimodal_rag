"use client";

import { useState, useEffect } from "react"
import { LocalGPTChat } from "@/components/ui/localgpt-chat"
import { SessionSidebar } from "@/components/ui/session-sidebar"
import { SessionChat } from '@/components/ui/session-chat'
import { chatAPI, ChatSession } from "@/lib/api"
import { LandingMenu } from "@/components/LandingMenu";
import { IndexForm } from "@/components/IndexForm";
import SessionIndexInfo from "@/components/SessionIndexInfo";
import IndexPicker from "@/components/IndexPicker";
import { QuickChat } from '@/components/ui/quick-chat'

export function Demo() {
    const [currentSessionId, setCurrentSessionId] = useState<string | undefined>()
    const [currentSession, setCurrentSession] = useState<ChatSession | null>(null)
    const [showConversation, setShowConversation] = useState(false)
    const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'error'>('checking')
    const [sidebarRef, setSidebarRef] = useState<{ refreshSessions: () => Promise<void> } | null>(null)
    const [homeMode, setHomeMode] = useState<'HOME' | 'INDEX' | 'CHAT_EXISTING' | 'QUICK_CHAT'>('HOME')
    const [showIndexInfo, setShowIndexInfo] = useState(false)
    const [showIndexPicker, setShowIndexPicker] = useState(false)

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

    const handleNewSession = () => {
        // Don't create session immediately - just show empty state
        setCurrentSessionId(undefined)
        setCurrentSession(null)
        setShowConversation(true)
    }

    const handleSessionChange = async (session: ChatSession) => {
        setCurrentSession(session)
        // Update the current session ID when a new session is created
        if (session.id !== currentSessionId) {
            setCurrentSessionId(session.id)
            // Refresh the sidebar to show the new session
            if (sidebarRef) {
                await sidebarRef.refreshSessions()
            }
        }
    }

    const handleSessionDelete = (deletedSessionId: string) => {
        if (currentSessionId === deletedSessionId) {
            // Stay in conversation mode but show empty state
            setCurrentSessionId(undefined)
            setCurrentSession(null)
        }
    }

    const handleStartConversation = () => {
        if (backendStatus === 'connected') {
            // Just show empty state, don't create session yet
            handleNewSession()
        } else {
            setShowConversation(true)
        }
    }

    return (
        <div className="flex h-full w-full flex-row bg-black">
            {/* Session Sidebar */}
            {showConversation && homeMode !== 'QUICK_CHAT' && (
                <SessionSidebar
                    currentSessionId={currentSessionId}
                    onSessionSelect={handleSessionSelect}
                    onNewSession={handleNewSession}
                    onSessionDelete={handleSessionDelete}
                    onSessionCreated={setSidebarRef}
                />
            )}
            
            <main className="flex flex-1 flex-col transition-all duration-200 bg-black min-h-0 overflow-hidden">
                {homeMode === 'HOME' ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="space-y-4">
                            <LandingMenu onSelect={(m)=>{
                                if(m==='CHAT_EXISTING'){ setShowIndexPicker(true); return; }
                                if(m==='QUICK_CHAT'){
                                    setHomeMode('QUICK_CHAT');
                                    return;
                                }
                                setHomeMode('INDEX');
                            }} />
                            <div className="flex flex-col items-center gap-3 mt-12">
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
                ) : homeMode==='CHAT_EXISTING' ? (
                    <SessionChat
                        sessionId={currentSessionId}
                        onSessionChange={handleSessionChange}
                        className="flex-1"
                    />
                ) : homeMode==='QUICK_CHAT' ? (
                    <QuickChat sessionId={currentSessionId} onSessionChange={handleSessionChange} className="flex-1" />
                ) : null}
            </main>

            {homeMode==='INDEX' && (
              <div className="fixed inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-50">
                <IndexForm onClose={()=>setHomeMode('HOME')} onIndexed={(s)=>{setHomeMode('CHAT_EXISTING'); handleSessionSelect(s.id);}} />
              </div>
            )}

            {showIndexInfo && currentSessionId && (
              <SessionIndexInfo sessionId={currentSessionId} onClose={()=>setShowIndexInfo(false)} />
            )}

            {showIndexPicker && (
              <IndexPicker onClose={()=>setShowIndexPicker(false)} onSelect={async (idxId)=>{
                // create session and link index then open chat
                const session = await chatAPI.createSession()
                await chatAPI.linkIndexToSession(session.id, idxId)
                setShowIndexPicker(false)
                setHomeMode('CHAT_EXISTING')
                handleSessionSelect(session.id)
              }} />
            )}
        </div>
    );
} 