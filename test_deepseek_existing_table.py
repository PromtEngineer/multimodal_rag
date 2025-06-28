#!/usr/bin/env python3
"""
Test DeepSeek query against an existing table
"""

import sys
import asyncio
sys.path.append('.')

from rag_system.agent.loop import Agent
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG, PIPELINE_CONFIGS

async def test_deepseek_existing_table():
    """Test DeepSeek query against one of the existing tables"""
    print("🧠 Testing DeepSeek Query with Existing Table")
    print("=" * 60)
    
    # Check if Ollama is running
    ollama_client = OllamaClient()
    if not ollama_client.is_ollama_running():
        print("❌ Ollama is not running. Please start Ollama first.")
        return False
    
    # Use react config with decomposition enabled
    config = PIPELINE_CONFIGS['react'].copy()
    
    # Initialize Agent
    agent = Agent(config, ollama_client, OLLAMA_CONFIG)
    
    # Try one of the existing tables
    table_name = "text_pages_e29629e9-c1c3-4ce7-9ce6-c89aadc87502"  # Most recent table
    query = "What is the training cost of deepseek and what were the total number of parameters in v3?"
    
    print(f"🔍 Query: '{query}'")
    print(f"📊 Table: {table_name}")
    print("-" * 60)
    
    events = []
    
    def event_callback(event_type: str, data):
        events.append({'type': event_type, 'data': data})
        
        if event_type == "decomposition":
            sub_queries = data.get('sub_queries', [])
            print(f"\n📋 Query Decomposed into {len(sub_queries)} parts:")
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
        
        result = await agent.stream_agent_response_async(
            query=query,
            table_name=table_name,
            session_id="deepseek-existing-test",
            query_decompose=True,
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
            for i, doc in enumerate(source_docs[:5], 1):
                chunk_id = doc.get('chunk_id', 'Unknown')
                print(f"   {i}. {chunk_id}")
        
        # Check if decomposition happened
        had_decomposition = any(e['type'] == 'decomposition' for e in events)
        print(f"\n🔄 Query decomposition occurred: {'✅ Yes' if had_decomposition else '❌ No'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_deepseek_existing_table())
    if success:
        print("\n🎉 DeepSeek existing table test completed!")
    else:
        print("\n❌ DeepSeek existing table test failed")