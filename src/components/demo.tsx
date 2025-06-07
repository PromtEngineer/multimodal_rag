"use client";

import { LocalGPTChat } from "@/components/ui/localgpt-chat"
import { SessionNavBar } from "@/components/ui/sidebar"

export function Demo() {
    return (
        <div className="flex h-screen w-screen flex-row">
            <SessionNavBar />
            <main className="flex h-screen grow flex-col overflow-auto ml-12 transition-all duration-200">
                <div className="flex items-center justify-center h-full">
                    <LocalGPTChat />
                </div>
            </main>
        </div>
    );
} 