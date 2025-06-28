#!/usr/bin/env python3
"""
Test DeepSeek query decomposition and answer generation
"""

import sys
import asyncio
sys.path.append('.')

from rag_system.agent.loop import Agent
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG, PIPELINE_CONFIGS

async def test_deepseek_query():
    """Test complex multi-part query about DeepSeek"""
    print("🧠 Testing DeepSeek Query Decomposition")
    print("=" * 60)
    
    # Check if Ollama is running
    ollama_client = OllamaClient()
    if not ollama_client.is_ollama_running():
        print("❌ Ollama is not running. Please start Ollama first.")
        return False
    
    # Initialize Agent with index
    agent = Agent(
        PIPELINE_CONFIGS['default'], 
        ollama_client, 
        OLLAMA_CONFIG, 
        "test-agent-af4396d8"
    )
    
    # Complex multi-part query about DeepSeek
    query = "What is the training cost of deepseek and what were the total number of parameters in v3?"
    table_name = f"text_pages_test-agent-af4396d8"
    
    print(f"🔍 Query: '{query}'")
    print("-" * 60)
    
    # Track events to see decomposition
    events = []
    
    def event_callback(event_type: str, data):
        events.append({'type': event_type, 'data': data})
        
        if event_type == "decomposition":
            sub_queries = data.get('sub_queries', [])
            print(f"\n📋 Query Decomposition:")
            for i, sub_q in enumerate(sub_queries, 1):
                print(f"   {i}. {sub_q}")
        
        elif event_type == "sub_query_result":
            index = data.get('index', 0)
            sub_query = data.get('query', '')
            answer = data.get('answer', '')
            sources = len(data.get('source_documents', []))
            print(f"\n📝 Sub-Query {index + 1} Result:")
            print(f"   Q: {sub_query}")
            print(f"   A: {answer[:150]}{'...' if len(answer) > 150 else ''}")
            print(f"   Sources: {sources} documents")
            
        elif event_type == "token":
            print(data.get('text', ''), end='', flush=True)
            
        elif event_type in ['analyze', 'composition_start']:
            print(f"\n[{event_type}]")
    
    try:
        print("🚀 Starting query processing...")
        
        # Test using streaming async method to see all events
        result = await agent.stream_agent_response_async(
            query=query,
            table_name=table_name,
            session_id="deepseek-test",
            event_callback=event_callback
        )
        
        print(f"\n\n✅ Query completed!")
        print("=" * 60)
        print("📊 FINAL RESULT")
        print("=" * 60)
        
        answer = result.get('answer', 'No answer')
        source_docs = result.get('source_documents', [])
        
        print(f"Answer: {answer}")
        print(f"\nSources: {len(source_docs)} documents")
        
        if source_docs:
            print("\nSource Documents:")
            for i, doc in enumerate(source_docs[:3], 1):  # Show first 3
                chunk_id = doc.get('chunk_id', 'Unknown')
                print(f"   {i}. {chunk_id}")
        
        # Event summary
        print(f"\nEvents captured: {len(events)}")
        event_types = {}
        for event in events:
            event_type = event['type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print("Event breakdown:")
        for event_type, count in event_types.items():
            print(f"   {event_type}: {count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_deepseek_query())
    if success:
        print("\n🎉 DeepSeek query test completed!")
    else:
        print("\n❌ DeepSeek query test failed")