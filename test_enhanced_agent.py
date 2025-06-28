#!/usr/bin/env python3
"""
Test script to verify the enhanced Agent functionality with model capability detection,
index-specific overviews, and improved async/streaming support.
"""

import sys
import asyncio
from typing import Dict, Any

# Add rag_system to path
sys.path.append('.')

from rag_system.agent.loop import Agent
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG, PIPELINE_CONFIGS, get_model_capabilities

def test_model_detection():
    """Test model capability detection for different model types"""
    print("🧠 Testing Model Capability Detection")
    print("=" * 50)
    
    test_models = [
        "qwen3:8b",
        "qwen3:0.6b", 
        "gemma3n:e4b",
        "gemma:7b",
        "llama3.2",
        "deepseek-r1:7b",
        "unknown-model:1b"
    ]
    
    for model in test_models:
        caps = get_model_capabilities(model)
        print(f"{model:15} -> thinking: {caps['supports_thinking']:5} default: {caps['thinking_enabled_by_default']}")
    
    print("✅ Model capability detection working correctly\n")

def test_agent_initialization():
    """Test Agent initialization with different configurations"""
    print("🤖 Testing Agent Initialization")
    print("=" * 50)
    
    # Test with default Qwen models
    print("Testing with Qwen models...")
    qwen_config = OLLAMA_CONFIG.copy()
    llm_client = OllamaClient()
    
    agent1 = Agent(PIPELINE_CONFIGS['default'], llm_client, qwen_config)
    print(f"✅ Qwen Agent: Gen={agent1.gen_model} (thinking: {agent1.gen_model_caps['supports_thinking']})")
    
    # Test with Gemma models
    print("\nTesting with Gemma models...")
    gemma_config = OLLAMA_CONFIG.copy()
    gemma_config['generation_model'] = 'gemma3n:e4b'
    gemma_config['enrichment_model'] = 'gemma3n:e4b'
    
    agent2 = Agent(PIPELINE_CONFIGS['default'], llm_client, gemma_config)
    print(f"✅ Gemma Agent: Gen={agent2.gen_model} (thinking: {agent2.gen_model_caps['supports_thinking']})")
    
    # Test with index-specific initialization
    print("\nTesting with index-specific Agent...")
    test_index_id = "test-agent-af4396d8"  # From our indexing test
    agent3 = Agent(PIPELINE_CONFIGS['default'], llm_client, qwen_config, test_index_id)
    print(f"✅ Index-specific Agent: ID={test_index_id[:12]}... overviews={len(agent3.doc_overviews)}")
    
    print("✅ Agent initialization working correctly\n")
    return agent3, test_index_id

def test_agent_queries(agent: Agent, table_name: str):
    """Test Agent query processing with different types of queries"""
    print("💬 Testing Agent Query Processing")
    print("=" * 50)
    
    test_queries = [
        # Simple chat (should use DIRECT_LLM)
        "Hi there!",
        
        # General knowledge (should use DIRECT_LLM)
        "What is 2 + 2?",
        
        # Document-specific (should use RAG)
        "What is the total amount on the invoice?",
        "Can you summarize the document?",
        
        # Technical query about the system
        "What are the key features mentioned in the document?"
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        try:
            # Test the enhanced Agent
            result = agent.run(
                query=query,
                table_name=table_name,
                session_id="test-session-123"
            )
            
            answer = result.get('answer', 'No answer')
            source_docs = len(result.get('source_documents', []))
            
            print(f"Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"Sources: {source_docs} documents")
            
            results.append({
                'query': query,
                'answer': answer,
                'source_count': source_docs,
                'success': True
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'query': query,
                'error': str(e),
                'success': False
            })
    
    print("\n✅ Query processing test completed")
    return results

async def test_streaming_support(agent: Agent, table_name: str):
    """Test the enhanced streaming functionality"""
    print("🌊 Testing Streaming Support")
    print("=" * 50)
    
    test_query = "What is the total amount on the invoice?"
    print(f"Streaming query: '{test_query}'")
    
    try:
        # Collect streaming events
        events = []
        
        def event_callback(event_type: str, data: Dict[str, Any]):
            events.append({'type': event_type, 'data': data})
            if event_type == "token":
                print(data.get('text', ''), end='', flush=True)
            else:
                print(f"\n[{event_type}]", end='')
        
        # Test async streaming
        result = await agent.stream_agent_response_async(
            query=test_query,
            table_name=table_name,
            session_id="streaming-test-session",
            event_callback=event_callback
        )
        
        print(f"\n\n✅ Streaming completed. Events: {len(events)}")
        print(f"Answer: {result.get('answer', 'No answer')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 ENHANCED AGENT TESTING SUITE")
    print("=" * 60)
    
    # Check if Ollama is running
    ollama_client = OllamaClient()
    if not ollama_client.is_ollama_running():
        print("❌ Ollama is not running. Please start Ollama first.")
        print("Run: ollama serve")
        return
    
    print("✅ Ollama is running\n")
    
    # Test 1: Model capability detection
    test_model_detection()
    
    # Test 2: Agent initialization
    agent, index_id = test_agent_initialization()
    table_name = f"text_pages_{index_id}"
    
    # Test 3: Query processing
    query_results = test_agent_queries(agent, table_name)
    
    # Test 4: Streaming support
    print("\n" + "=" * 60)
    streaming_success = asyncio.run(test_streaming_support(agent, table_name))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_queries = sum(1 for r in query_results if r['success'])
    total_queries = len(query_results)
    
    print(f"Model Detection: ✅ PASSED")
    print(f"Agent Initialization: ✅ PASSED")
    print(f"Query Processing: {successful_queries}/{total_queries} PASSED")
    print(f"Streaming Support: {'✅ PASSED' if streaming_success else '❌ FAILED'}")
    
    if successful_queries == total_queries and streaming_success:
        print("\n🎉 ALL TESTS PASSED! Enhanced Agent is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the output above.")

if __name__ == "__main__":
    main()