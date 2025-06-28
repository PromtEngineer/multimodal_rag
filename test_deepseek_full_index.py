#!/usr/bin/env python3
"""
Test DeepSeek query against the full index with query decomposition enabled
"""

import sys
import asyncio
sys.path.append('.')

from rag_system.agent.loop import Agent
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG, PIPELINE_CONFIGS

async def test_deepseek_full_index():
    """Test complex multi-part query about DeepSeek with full index and decomposition"""
    print("🧠 Testing DeepSeek Query with Full Index + Decomposition")
    print("=" * 70)
    
    # Check if Ollama is running
    ollama_client = OllamaClient()
    if not ollama_client.is_ollama_running():
        print("❌ Ollama is not running. Please start Ollama first.")
        return False
    
    # Use 'react' config which has query decomposition enabled
    config = PIPELINE_CONFIGS['react'].copy()
    print(f"📋 Using '{config.get('description', 'react')}' pipeline config")
    print(f"🔄 Query decomposition enabled: {config.get('query_decomposition', {}).get('enabled', False)}")
    
    # Initialize Agent WITHOUT specific index (use global index with all documents)
    agent = Agent(
        config, 
        ollama_client, 
        OLLAMA_CONFIG
        # No specific index - use global index
    )
    
    # Complex multi-part query about DeepSeek
    query = "What is the training cost of deepseek and what were the total number of parameters in v3?"
    table_name = "text_pages_default"  # Use default table with all documents
    
    print(f"🔍 Query: '{query}'")
    print(f"📊 Table: {table_name}")
    print("-" * 70)
    
    # Track events to see decomposition
    events = []
    
    def event_callback(event_type: str, data):
        events.append({'type': event_type, 'data': data})
        
        if event_type == "decomposition_start":
            print(f"\n🔧 Starting query decomposition...")
            
        elif event_type == "decomposition":
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
            print(f"   A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            print(f"   Sources: {sources} documents")
            
        elif event_type == "composition_start":
            print(f"\n🔗 Composing final answer from sub-queries...")
            
        elif event_type == "token":
            print(data.get('text', ''), end='', flush=True)
            
        elif event_type in ['analyze', 'parallel_start']:
            print(f"\n[{event_type}]")
    
    try:
        print("🚀 Starting query processing...")
        
        # Test using streaming async method to see all events
        result = await agent.stream_agent_response_async(
            query=query,
            table_name=table_name,
            session_id="deepseek-full-test",
            query_decompose=True,  # Force enable decomposition
            event_callback=event_callback
        )
        
        print(f"\n\n✅ Query completed!")
        print("=" * 70)
        print("📊 FINAL RESULT")
        print("=" * 70)
        
        answer = result.get('answer', 'No answer')
        source_docs = result.get('source_documents', [])
        
        print(f"Answer: {answer}")
        print(f"\nSources: {len(source_docs)} documents")
        
        if source_docs:
            print("\nSource Documents:")
            for i, doc in enumerate(source_docs[:5], 1):  # Show first 5
                chunk_id = doc.get('chunk_id', 'Unknown')
                rerank_score = doc.get('rerank_score', 'N/A')
                print(f"   {i}. {chunk_id} (score: {rerank_score})")
        
        # Event summary
        print(f"\nEvents captured: {len(events)}")
        event_types = {}
        for event in events:
            event_type = event['type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print("Event breakdown:")
        for event_type, count in sorted(event_types.items()):
            print(f"   {event_type}: {count}")
        
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
    success = asyncio.run(test_deepseek_full_index())
    if success:
        print("\n🎉 DeepSeek full index query test completed!")
    else:
        print("\n❌ DeepSeek full index query test failed")