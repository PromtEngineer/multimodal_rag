"use client";

import { useState } from "react"
import { LocalGPTChat } from "@/components/ui/localgpt-chat"
import { SessionNavBar } from "@/components/ui/sidebar"
import { ConversationPage } from "@/components/ui/conversation-page"

// Sample conversation data
const sampleMessages = [
  {
    id: 1,
    content: "What is the importance of renewable energy in today's world?",
    sender: "user" as const,
    timestamp: new Date(),
  },
  {
    id: 2,
    content: "Renewable energy is crucial for several reasons:\n\n1. **Environmental Impact**: It significantly reduces greenhouse gas emissions and helps combat climate change.\n\n2. **Energy Security**: It reduces dependence on fossil fuel imports and provides energy independence.\n\n3. **Economic Benefits**: Creates jobs in growing green industries and reduces long-term energy costs.\n\n4. **Sustainability**: Unlike fossil fuels, renewable sources like solar and wind are inexhaustible.\n\n5. **Health Benefits**: Reduces air pollution, leading to better public health outcomes.\n\nThe transition to renewable energy is essential for a sustainable future.",
    sender: "assistant" as const,
    timestamp: new Date(),
  },
  {
    id: 3,
    content: "Can you explain how solar panels work?",
    sender: "user" as const,
    timestamp: new Date(),
  },
  {
    id: 4,
    content: "Solar panels work through the photovoltaic effect:\n\n**Basic Process:**\n1. **Light Absorption**: Solar cells (made of silicon) absorb photons from sunlight\n2. **Electron Excitation**: Photons knock electrons loose from silicon atoms\n3. **Electric Field**: Built-in electric field in the cell directs these free electrons\n4. **Current Generation**: Moving electrons create direct current (DC) electricity\n5. **Conversion**: An inverter converts DC to alternating current (AC) for home use\n\n**Key Components:**\n- **Photovoltaic cells**: Convert light to electricity\n- **Glass covering**: Protects cells while allowing light through\n- **Metal frame**: Provides structural support\n- **Wiring**: Connects cells and carries electricity\n\nEfficiency typically ranges from 15-22% for residential panels.",
    sender: "assistant" as const,
    timestamp: new Date(),
  }
]

export function Demo() {
    const [showConversation, setShowConversation] = useState(false)
    const [isLoading, setIsLoading] = useState(false)

    const handleStartConversation = () => {
        setShowConversation(true)
    }

    const handleBackToChat = () => {
        setShowConversation(false)
    }

    return (
        <div className="flex h-screen w-screen flex-row">
            <SessionNavBar />
            <main className="flex h-screen grow flex-col overflow-auto ml-12 transition-all duration-200">
                {!showConversation ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="space-y-4">
                            <LocalGPTChat />
                            <div className="flex justify-center">
                                <button
                                    onClick={handleStartConversation}
                                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                                >
                                    View Sample Conversation
                                </button>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="h-full flex flex-col">
                        <div className="p-4 border-b border-gray-700 bg-black">
                            <div className="flex items-center justify-between">
                                <h1 className="text-xl font-semibold text-white">
                                    Conversation with localGPT
                                </h1>
                                <button
                                    onClick={handleBackToChat}
                                    className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
                                >
                                    Back to Chat
                                </button>
                            </div>
                        </div>
                        <ConversationPage 
                            messages={sampleMessages}
                            isLoading={isLoading}
                        />
                    </div>
                )}
            </main>
        </div>
    );
} 