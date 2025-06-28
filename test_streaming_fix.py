#!/usr/bin/env python3
"""
Test streaming functionality after embedding fix
"""

import sys
import asyncio
sys.path.append('.')

from rag_system.agent.loop import Agent
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG, PIPELINE_CONFIGS

async def test_streaming_fix():
    """Test that streaming works without embedding errors"""
    print("🌊 Testing Streaming After Embedding Fix")
    print("=" * 50)
    
    # Check if Ollama is running
    ollama_client = OllamaClient()
    if not ollama_client.is_ollama_running():
        print("❌ Ollama is not running. Please start Ollama first.")
        return False
    
    # Initialize Agent
    agent = Agent(
        PIPELINE_CONFIGS['default'], 
        ollama_client, 
        OLLAMA_CONFIG, 
        "test-agent-af4396d8"
    )
    
    test_query = "What is the total amount on the invoice?"
    table_name = "text_pages_test-agent-af4396d8"
    
    print(f"🔍 Streaming query: '{test_query}'")
    
    try:
        # Collect streaming events
        events = []
        token_count = 0
        
        def event_callback(event_type: str, data):
            events.append({'type': event_type})
            if event_type == "token":
                nonlocal token_count
                token_count += 1
                print(data.get('text', ''), end='', flush=True)
            elif event_type in ['analyze', 'complete']:
                print(f"\n[{event_type}]", end='')
        
        # Test async streaming
        result = await agent.stream_agent_response_async(
            query=test_query,
            table_name=table_name,
            session_id="streaming-fix-test",
            event_callback=event_callback
        )
        
        print(f"\n\n✅ Streaming completed successfully!")
        print(f"Events: {len(events)}, Tokens: {token_count}")
        print(f"Answer length: {len(result.get('answer', ''))}")
        print("✅ No embedding errors in streaming!")
        return True
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_streaming_fix())
    if success:
        print("\n🎉 Streaming fix verified successfully!")
    else:
        print("\n❌ Streaming fix test failed")