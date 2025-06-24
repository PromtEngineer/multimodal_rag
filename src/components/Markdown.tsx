// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-nocheck
import dynamic from 'next/dynamic'
import React from 'react'

// Dynamically import react-markdown to avoid SSR issues
const ReactMarkdownDynamic: any = dynamic(() => import('react-markdown') as any, { ssr: false })
const ReactMarkdown: any = ReactMarkdownDynamic as any
// Import GFM plugin (types optional)
const remarkGfm = (await import('remark-gfm')).default || (await import('remark-gfm'))

interface MarkdownProps {
  text: string
  className?: string
}

export default function Markdown({ text, className = '' }: MarkdownProps) {
  return (
    <ReactMarkdown
      className={`prose prose-invert max-w-none ${className}`}
      // @ts-ignore — react-markdown typings don't yet include remarkPlugins array type
      remarkPlugins={[remarkGfm]}
      linkTarget="_blank"
    >
      {text}
    </ReactMarkdown>
  )
} 