#!/usr/bin/env python3
"""
Test the updated triage behavior to ensure it properly routes to RAG
"""

import sys
import asyncio
sys.path.append('.')

from rag_system.agent.loop import Agent
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG, PIPELINE_CONFIGS

async def test_triage_behavior():
    """Test triage decisions with various query types"""
    print("🎯 Testing Updated Triage Behavior")
    print("=" * 50)
    
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
    
    # Test queries that should now route to RAG
    test_queries = [
        # These should now go to RAG (previously went to DIRECT_LLM)
        ("What are the key features mentioned in the document?", "USE_RAG"),
        ("Can you summarize the document?", "USE_RAG"),
        ("What is the total amount?", "USE_RAG"),
        ("Who is mentioned in the invoice?", "USE_RAG"),
        ("What companies are involved?", "USE_RAG"),
        ("What is the date?", "USE_RAG"),
        ("Tell me about the invoice", "USE_RAG"),
        ("What information is available?", "USE_RAG"),
        
        # These should still go to DIRECT_LLM  
        ("Hi there", "DIRECT_LLM"),
        ("What is 2 + 2?", "DIRECT_LLM"),
        ("Who is the president of France?", "DIRECT_LLM"),
        ("What's the weather like?", "DIRECT_LLM"),
    ]
    
    results = []
    
    print("Testing triage decisions:")
    print("-" * 60)
    
    for query, expected in test_queries:
        try:
            # Test the triage decision directly
            decision = await agent._triage_query_async(query)
            
            status = "✅" if decision == expected else "❌"
            results.append((query, expected, decision, decision == expected))
            
            print(f"{status} '{query}'")
            print(f"   Expected: {expected}, Got: {decision}")
            print()
            
        except Exception as e:
            print(f"❌ Error testing '{query}': {e}")
            results.append((query, expected, "ERROR", False))
    
    # Summary
    print("=" * 60)
    print("📊 TRIAGE TEST SUMMARY")
    print("=" * 60)
    
    correct = sum(1 for _, _, _, is_correct in results if is_correct)
    total = len(results)
    
    print(f"Correct decisions: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # Show failures
    failures = [r for r in results if not r[3]]
    if failures:
        print(f"\n❌ Failed cases ({len(failures)}):")
        for query, expected, actual, _ in failures:
            print(f"   '{query}' -> Expected {expected}, Got {actual}")
    
    print(f"\n{'🎉 All tests passed!' if correct == total else '⚠️ Some tests failed.'}")
    return correct == total

if __name__ == "__main__":
    success = asyncio.run(test_triage_behavior())
    if success:
        print("\n✅ Triage behavior test completed successfully!")
    else:
        print("\n❌ Triage behavior needs further adjustment")