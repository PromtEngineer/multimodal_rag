import { useCallback } from 'react'

export interface ActionHandlerOptions {
  onAction?: (action: string, messageId: string, messageContent: string) => void
  onRegenerate?: (messageId: string, userMessage: string) => Promise<void>
}

export interface Step {
  key: string
  label: string
  status: 'pending' | 'active' | 'done' | 'error'
  details?: string | { answer: string; source_documents: any[] }
}

/**
 * Shared hook for handling message actions (copy, regenerate, etc.)
 * Consolidates duplicate logic from conversation-page and session-chat components
 */
export function useActionHandler(options: ActionHandlerOptions = {}) {
  const { onAction, onRegenerate } = options

  const processMessageContent = useCallback((messageContent: string | Record<string, any>[] | { steps: Step[] }): string => {
    if (typeof messageContent === 'string') {
      return messageContent
    } else if (Array.isArray(messageContent)) {
      return (messageContent as any[]).map((s: any) => s.text || s.answer || '').join('\n')
    } else if (messageContent && typeof messageContent === 'object' && Array.isArray((messageContent as any).steps)) {
      // For {steps: Step[]} structure
      return (messageContent as any).steps.map((s: any) => 
        s.label + (s.details ? (typeof s.details === 'string' ? (': ' + s.details) : '') : '')
      ).join('\n')
    } else {
      return ''
    }
  }, [])

  const handleAction = useCallback(async (
    action: string, 
    messageId: string, 
    messageContent: string | Record<string, any>[] | { steps: Step[] },
    messages?: any[]
  ) => {
    const contentToPass = processMessageContent(messageContent)

    // If parent component provided onAction, use it
    if (onAction) {
      onAction(action, messageId, contentToPass)
      return
    }

    // Default action handling
    switch (action) {
      case 'copy':
        try {
          await navigator.clipboard.writeText(contentToPass)
          console.log('Content copied to clipboard')
        } catch (error) {
          console.error('Failed to copy to clipboard:', error)
        }
        break
        
      case 'regenerate':
        if (onRegenerate && messages) {
          // Find the user message before this AI message and resend it
          const messageIndex = messages.findIndex(m => m.id === messageId)
          if (messageIndex > 0 && messages[messageIndex].sender === 'assistant') {
            const userMessage = messages[messageIndex - 1]
            if (userMessage.sender === 'user') {
              await onRegenerate(messageId, userMessage.content as string)
            }
          }
        }
        break
        
      case 'like':
        console.log(`Liked message ${messageId}`)
        // TODO: Implement like functionality
        break
        
      case 'dislike':
        console.log(`Disliked message ${messageId}`)
        // TODO: Implement dislike functionality
        break
        
      case 'speak':
        console.log(`Text-to-speech for message ${messageId}`)
        // TODO: Implement text-to-speech
        break
        
      case 'more':
        console.log(`Show more options for message ${messageId}`)
        // TODO: Implement more options
        break
        
      default:
        console.log(`Unhandled action: ${action} for message ${messageId}`)
    }
  }, [onAction, onRegenerate, processMessageContent])

  return { handleAction, processMessageContent }
} 