#!/usr/bin/env python3
"""
Test Multi-Hop Retrieval with Real Documents
"""

import asyncio
import time
from rag_system.factory import get_agent

def test_multihop_with_documents():
    """Test multi-hop retrieval with actual indexed documents."""
    
    print("🧪 Testing Multi-Hop Retrieval with Real Documents")
    print("=" * 60)
    
    # Initialize the agent
    print("🔧 Initializing RAG agent...")
    agent = get_agent()
    
    # Use one of the available document tables
    table_name = "text_pages_a636401c-ed4b-400f-8be1-8e78f5a306c2"
    
    # Test queries that should benefit from multi-hop retrieval
    test_queries = [
        {
            "query": "What are the main topics discussed in the documents, and how do they relate to each other?",
            "description": "Should analyze topics first, then find relationships"
        },
        {
            "query": "What companies or entities are mentioned, and what are their key characteristics or activities?",
            "description": "Should identify entities first, then analyze their properties"
        },
        {
            "query": "What are the key technical concepts explained, and what are their practical applications?",
            "description": "Should understand concepts first, then find applications"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n📝 Test {i}: {description}")
        print(f"Query: {query}")
        print("-" * 60)
        
        # Test 1: Regular RAG (for comparison)
        print("🔍 Testing with Regular RAG...")
        start_time = time.time()
        
        regular_result = agent.run(
            query=query,
            table_name=table_name,
            multihop=False,
            max_retries=1
        )
        
        regular_time = time.time() - start_time
        regular_answer = regular_result.get('answer', 'No answer')
        regular_docs = len(regular_result.get('source_documents', []))
        
        print(f"   ⏱️  Time: {regular_time:.2f}s")
        print(f"   📄 Documents: {regular_docs}")
        print(f"   📝 Answer length: {len(regular_answer)} chars")
        print(f"   💬 Preview: {regular_answer[:200]}...")
        
        # Test 2: Multi-Hop RAG
        print("\n🔗 Testing with Multi-Hop RAG...")
        start_time = time.time()
        
        multihop_result = agent.run(
            query=query,
            table_name=table_name,
            multihop=True,
            max_retries=1
        )
        
        multihop_time = time.time() - start_time
        multihop_answer = multihop_result.get('answer', 'No answer')
        multihop_docs = len(multihop_result.get('source_documents', []))
        
        print(f"   ⏱️  Time: {multihop_time:.2f}s")
        print(f"   📄 Documents: {multihop_docs}")
        print(f"   📝 Answer length: {len(multihop_answer)} chars")
        print(f"   🔗 Multi-hop executed: {'Yes' if 'multi-hop' in multihop_answer.lower() or multihop_time > regular_time * 1.5 else 'No'}")
        print(f"   💬 Preview: {multihop_answer[:200]}...")
        
        # Comparison
        print(f"\n📊 Comparison:")
        print(f"   Time difference: {multihop_time - regular_time:.2f}s ({((multihop_time/regular_time - 1) * 100):+.1f}%)")
        print(f"   Document difference: {multihop_docs - regular_docs:+d}")
        print(f"   Answer length difference: {len(multihop_answer) - len(regular_answer):+d} chars")
        
        if i < len(test_queries):
            print(f"\n{'='*60}")

def test_frontend_integration():
    """Test that multi-hop works through the full frontend API stack."""
    
    print("\n🌐 Testing Frontend Integration")
    print("=" * 60)
    
    print("To test the full frontend integration:")
    print("1. Start the system: python -m rag_system.main api")
    print("2. Open the web UI and create/load a session with documents")
    print("3. Go to Settings (⚙️) and enable 'Multi-hop retrieval'")
    print("4. Ask a complex question like:")
    print("   - 'What are the main findings and how do they impact the conclusions?'")
    print("   - 'What are the key entities mentioned and what are their relationships?'")
    print("   - 'What problems are identified and what solutions are proposed?'")
    print("5. You should see longer processing time and more comprehensive answers")

if __name__ == "__main__":
    test_multihop_with_documents()
    test_frontend_integration() 