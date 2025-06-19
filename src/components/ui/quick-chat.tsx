"use client";

import React, { useState } from 'react';
import { ChatInput } from '@/components/ui/chat-input';
import { chatAPI, ChatMessage } from '@/lib/api';
import { ConversationPage } from '@/components/ui/conversation-page';

export function QuickChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | undefined>(undefined);
  const api = chatAPI;

  const sendMessage = async (content: string, _files?: any) => {
    if (!content.trim()) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      content,
      sender: 'user',
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);

    setIsLoading(true);

    // Ensure we have a backend session to preserve history on the agent side
    let activeSessionId = sessionId;
    if (!activeSessionId) {
      try {
        const newSess = await api.createSession('Quick Chat');
        activeSessionId = newSess.id;
        setSessionId(activeSessionId);
      } catch (err) {
        console.error('Failed to create quick-chat session', err);
      }
    }

    // Placeholder assistant message for streaming
    const assistantMsg: ChatMessage = {
      id: crypto.randomUUID(),
      content: '',
      sender: 'assistant',
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, assistantMsg]);

    try {
      await api.streamSessionMessage(
        { query: content, session_id: activeSessionId, composeSubAnswers: false, decompose: false, aiRerank: false, contextExpand: false },
        (evt) => {
          if (evt.type === 'token') {
            const tok: string = evt.data.text || '';
            setMessages((prev) => prev.map((m) => (m.id === assistantMsg.id ? { ...m, content: (m.content as string) + tok } : m)));
          }
          if (evt.type === 'complete') {
            setIsLoading(false);
          }
        }
      );
    } catch (err) {
      console.error('Quick chat stream failed', err);
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <ConversationPage messages={messages} isLoading={isLoading} className="flex-1" />
      <div className="flex-shrink-0 bg-black/90 backdrop-blur-md">
        <ChatInput onSendMessage={sendMessage} disabled={isLoading} placeholder="Ask anything…" />
      </div>
    </div>
  );
} 