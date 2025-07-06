#!/usr/bin/env python3
"""
Test script for the enhanced overview-based routing system
"""

import requests
import json
import time

def test_routing_decision(query, expected_route=None):
    """Test a query and show the routing decision"""
    
    print(f"\n🧪 Testing Query: '{query}'")
    print(f"Expected Route: {expected_route or 'Unknown'}")
    print("-" * 60)
    
    # Create a test session first
    session_response = requests.post("http://localhost:8000/sessions", 
                                   json={"title": "Test Session"})
    
    if session_response.status_code != 201:
        print(f"❌ Failed to create session: {session_response.text}")
        return
    
    session_id = session_response.json()["session_id"]
    
    # Link an index to the session so routing can work
    # Use the first available index
    indexes_response = requests.get("http://localhost:8000/indexes")
    if indexes_response.status_code == 200:
        indexes = indexes_response.json().get("indexes", [])
        if indexes:
            index_id = indexes[0]["id"]
            requests.post(f"http://localhost:8000/sessions/{session_id}/indexes/{index_id}")
            print(f"📎 Linked index {index_id[:8]}... to session")
        else:
            print("⚠️ No indexes available - routing may default to Direct LLM")
    
    # Send the test message
    start_time = time.time()
    
    message_payload = {
        "message": query,
        "model": "qwen3:8b"
    }
    
    try:
        response = requests.post(
            f"http://localhost:8000/sessions/{session_id}/messages",
            json=message_payload,
            timeout=60
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})
            
            # Check if we got source documents (indicates RAG was used)
            source_docs = message.get("source_documents", [])
            has_sources = len(source_docs) > 0
            
            actual_route = "RAG" if has_sources else "Direct LLM"
            route_emoji = "🎯" if has_sources else "⚡"
            
            print(f"{route_emoji} Actual Route: {actual_route}")
            print(f"⏱️ Response Time: {elapsed:.2f}s")
            print(f"📝 Response: {message.get('content', 'No content')[:100]}...")
            
            if has_sources:
                print(f"📚 Source Documents: {len(source_docs)}")
            
            # Check if routing matched expectation
            if expected_route:
                match = actual_route.upper() == expected_route.upper()
                status = "✅" if match else "❌"
                print(f"{status} Routing {'matched' if match else 'did not match'} expectation")
        else:
            print(f"❌ Request failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Clean up test session
    requests.delete(f"http://localhost:8000/sessions/{session_id}")
    print("🧹 Cleaned up test session")

def main():
    """Run routing tests"""
    
    print("🚀 Enhanced Overview-Based Routing Test")
    print("=" * 50)
    
    # Test cases with expected routing outcomes
    test_cases = [
        # Document-specific queries (should use RAG)
        ("What invoice amounts are mentioned?", "RAG"),
        ("Who is PromptX AI LLC?", "RAG"),
        ("What is the DeepSeek model about?", "RAG"),
        ("Summarize the research paper", "RAG"),
        ("What payment methods are available?", "RAG"),
        
        # Greetings (should use Direct LLM)
        ("Hello!", "Direct LLM"),
        ("Hi there", "Direct LLM"),
        ("Thanks", "Direct LLM"),
        
        # General knowledge (should use Direct LLM)
        ("What is 2+2?", "Direct LLM"),
        ("What is the capital of France?", "Direct LLM"),
        
        # Ambiguous cases (could go either way)
        ("What is artificial intelligence?", None),
        ("How does machine learning work?", None),
    ]
    
    for query, expected in test_cases:
        test_routing_decision(query, expected)
        time.sleep(1)  # Brief pause between tests
    
    print("\n🎉 Enhanced Routing Test Complete!")
    print("\nKey Improvements:")
    print("• 📖 Reads actual document overviews")
    print("• 🧠 Uses LLM for intelligent decision making")
    print("• 🎯 Context-aware routing based on document content")
    print("• ⚡ Fallback to pattern matching if needed")

if __name__ == "__main__":
    main() 