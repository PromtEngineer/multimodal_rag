#!/usr/bin/env python3
"""
Simple Multi-Hop Test with Enhanced Logging
"""

import time
from rag_system.factory import get_agent

def test_multihop_with_logging():
    """Test multi-hop with detailed event logging."""
    
    print("🚀 Testing Enhanced Multi-Hop Retrieval")
    print("=" * 50)
    
    # Get the agent
    agent = get_agent()
    
    # Use available table
    table_name = "text_pages_a636401c-ed4b-400f-8be1-8e78f5a306c2"
    
    # Test query that should trigger multi-hop
    query = "What are the main concepts discussed in the documents, and how do they relate to practical applications?"
    
    print(f"📝 Query: {query}")
    print(f"📊 Table: {table_name}")
    print("🔗 Multi-hop: ENABLED")
    print("-" * 50)
    
    # Event callback for logging
    events_log = []
    
    def log_event(event_type, data):
        timestamp = time.time()
        events_log.append({
            'timestamp': timestamp,
            'type': event_type,
            'data': data
        })
        print(f"📢 EVENT: {event_type}")
        
        if event_type == "multihop_started":
            print(f"   🔗 Total steps: {data.get('total_steps', 0)}")
            for step in data.get('steps', []):
                print(f"   📝 Step {step['id']}: {step['query']}")
        
        elif event_type == "multihop_step_started":
            print(f"   🚀 Step {data.get('step_id')}: {data.get('query', '')[:80]}...")
        
        elif event_type == "multihop_step_retrieved":
            print(f"   📄 Step {data.get('step_id')}: {data.get('documents_found', 0)} documents found")
        
        elif event_type == "multihop_step_reranking":
            print(f"   🔄 Step {data.get('step_id')}: Reranking {data.get('documents_count', 0)} documents...")
        
        elif event_type == "multihop_step_reranked":
            print(f"   ✅ Step {data.get('step_id')}: Reranked to {data.get('reranked_count', 0)} documents")
        
        elif event_type == "multihop_step_pruning":
            print(f"   ✂️ Step {data.get('step_id')}: Pruning {data.get('documents_count', 0)} documents...")
        
        elif event_type == "multihop_step_pruned":
            print(f"   🎯 Step {data.get('step_id')}: Pruned to {data.get('pruned_count', 0)} documents")
        
        elif event_type == "multihop_step_processed":
            print(f"   🎯 Step {data.get('step_id')}: Final {data.get('final_documents', 0)} documents ready")
        
        elif event_type == "multihop_step_completed":
            print(f"   ✅ Step {data.get('step_id')} completed")
            print(f"   📄 Documents used: {data.get('documents_used', 0)}")
            print(f"   💬 Answer: {data.get('answer', '')[:100]}...")
        
        elif event_type == "multihop_synthesis_started":
            print(f"   🔄 Synthesizing from {data.get('total_steps_completed', 0)} steps")
        
        elif event_type == "multihop_completed":
            print(f"   🎉 Multi-hop completed!")
        
        print()
    
    # Run with multi-hop enabled
    start_time = time.time()
    
    result = agent.run(
        query=query,
        table_name=table_name,
        multihop=True,
        event_callback=log_event
    )
    
    execution_time = time.time() - start_time
    
    print("=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    print(f"⏱️  Execution time: {execution_time:.2f}s")
    print(f"📄 Source documents: {len(result.get('source_documents', []))}")
    print(f"📝 Answer length: {len(result.get('answer', ''))} characters")
    print(f"🔗 Multi-hop context: {'Yes' if result.get('multihop_context') else 'No'}")
    
    if result.get('multihop_context'):
        steps_executed = result['multihop_context'].get('steps_executed', 0)
        print(f"🎯 Steps executed: {steps_executed}")
    
    print(f"\n📢 Total events logged: {len(events_log)}")
    print(f"💬 Answer preview:")
    print("-" * 30)
    print(result.get('answer', 'No answer')[:500] + ("..." if len(result.get('answer', '')) > 500 else ""))

if __name__ == "__main__":
    test_multihop_with_logging() 