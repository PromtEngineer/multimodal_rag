#!/usr/bin/env python3
"""
Quick test to verify the embedding fix
"""

import sys
sys.path.append('.')

from rag_system.agent.loop import Agent
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG, PIPELINE_CONFIGS

def test_embedding_fix():
    """Test that the embedding issue is fixed"""
    print("🔧 Testing Embedding Fix")
    print("=" * 40)
    
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
        "test-agent-af4396d8"  # Use the test index we created
    )
    
    # Test query that should trigger RAG (and embedding generation)
    test_query = "What is the total amount on the invoice?"
    table_name = "text_pages_test-agent-af4396d8"
    
    print(f"🔍 Testing query: '{test_query}'")
    
    try:
        result = agent.run(
            query=test_query,
            table_name=table_name,
            session_id="embedding-fix-test"
        )
        
        answer = result.get('answer', 'No answer')
        source_count = len(result.get('source_documents', []))
        
        print(f"✅ Success! Answer: {answer[:100]}...")
        print(f"Sources: {source_count} documents")
        print("✅ No embedding errors detected!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedding_fix()
    if success:
        print("\n🎉 Embedding fix verified successfully!")
    else:
        print("\n❌ Embedding fix test failed")