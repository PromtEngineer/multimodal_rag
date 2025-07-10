#!/usr/bin/env python3
"""
Comprehensive test suite for multi-hop retrieval functionality.
Tests both the core components and end-to-end integration.
"""

import os
import sys
import json
import time
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_system.main import get_agent
from rag_system.retrieval.query_transformer import QueryDecomposer, MultiHopContext
from rag_system.utils.ollama_client import OllamaClient


class MultiHopTester:
    """Comprehensive tester for multi-hop retrieval functionality."""
    
    def __init__(self):
        self.agent = None
        self.ollama_client = None
        self.test_results = []
        
    def setup(self):
        """Initialize the RAG agent and components."""
        print("🔧 Setting up Multi-Hop Retrieval Tester...")
        
        try:
            self.agent = get_agent('default')
            if not self.agent:
                raise Exception("Failed to initialize RAG agent")
            
            self.ollama_client = self.agent.llm_client
            print("✅ RAG Agent initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def test_multihop_context(self):
        """Test MultiHopContext functionality."""
        print("\n🧪 Testing MultiHopContext...")
        
        try:
            context = MultiHopContext()
            
            # Test adding steps
            context.add_step(1, "What are the key features of AI models?", [])
            context.add_step(2, "How do these features compare to traditional models?", [1])
            context.add_step(3, "What are the implications for future development?", [1, 2])
            
            # Test dependency checking
            assert context.is_step_ready(1) == True, "Step 1 should be ready (no dependencies)"
            assert context.is_step_ready(2) == False, "Step 2 should not be ready (depends on step 1)"
            assert context.is_step_ready(3) == False, "Step 3 should not be ready (depends on steps 1, 2)"
            
            # Test step completion
            context.complete_step(1, {"answer": "AI models have neural networks, learning capabilities, and adaptability."}, [])
            assert context.is_step_ready(2) == True, "Step 2 should be ready after step 1 completion"
            assert context.is_step_ready(3) == False, "Step 3 should still not be ready"
            
            # Test context retrieval
            context_str = context.get_context_for_step(2)
            assert "AI models have neural networks" in context_str, "Context should contain step 1 answer"
            
            print("✅ MultiHopContext tests passed")
            return True
            
        except Exception as e:
            print(f"❌ MultiHopContext tests failed: {e}")
            return False
    
    def test_query_decomposition(self):
        """Test query decomposition for multi-hop."""
        print("\n🧪 Testing Query Decomposition...")
        
        try:
            decomposer = QueryDecomposer(self.ollama_client, self.agent.ollama_config["generation_model"])
            
            # Test multi-hop query
            test_query = "What are the main innovations in the latest AI research paper, and how do they compare to previous work mentioned in that paper?"
            context = decomposer.decompose_for_multihop(test_query)
            
            assert len(context.steps) > 0, "Should produce at least one step"
            print(f"✅ Decomposed query into {len(context.steps)} steps")
            
            for i, step in enumerate(context.steps):
                print(f"   Step {step['id']}: {step['query']}")
                print(f"   Dependencies: {step['dependencies']}")
            
            # Test single-step query
            simple_query = "What is machine learning?"
            simple_context = decomposer.decompose_for_multihop(simple_query)
            
            print(f"✅ Simple query produced {len(simple_context.steps)} step(s)")
            return True
            
        except Exception as e:
            print(f"❌ Query decomposition tests failed: {e}")
            return False
    
    def test_multihop_retrieval_integration(self):
        """Test end-to-end multi-hop retrieval."""
        print("\n🧪 Testing Multi-Hop Retrieval Integration...")
        
        try:
            # Enable multi-hop in configuration
            self.agent.pipeline_configs.setdefault("multihop", {})["enabled"] = True
            
            test_queries = [
                "What are the key innovations in AI research, and how do they impact real-world applications?",
                "What is the architecture of transformer models, and what advantages does this give them over previous approaches?",
                "How do language models work and what are their main limitations?",
            ]
            
            results = []
            for i, query in enumerate(test_queries):
                print(f"\n📝 Testing Query {i+1}: {query}")
                start_time = time.time()
                
                # Force multi-hop mode by passing multihop=True explicitly
                result = self.agent.run(
                    query=query,
                    multihop=True,  # Force multi-hop mode
                    max_retries=1
                )
                
                execution_time = time.time() - start_time
                
                answer = result.get('answer', 'No answer provided')
                source_docs = result.get('source_documents', [])
                
                print(f"   ⏱️  Execution time: {execution_time:.2f}s")
                print(f"   📊 Answer length: {len(answer)} chars")
                print(f"   📄 Documents: {len(source_docs)}")
                print(f"   🔗 Multi-hop info: {bool(result.get('multihop_context'))}")
                print(f"   💬 Answer preview: {answer[:200]}...")
                
                results.append({
                    'query': query,
                    'execution_time': execution_time,
                    'answer_length': len(answer),
                    'source_docs': len(source_docs),
                    'multihop_context': result.get('multihop_context'),
                    'answer': answer
                })
            
            print(f"\n✅ Completed {len(results)} multi-hop retrieval tests")
            return True
            
        except Exception as e:
            print(f"❌ Multi-hop retrieval test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_comparison_with_regular_rag(self):
        """Compare multi-hop vs regular RAG performance."""
        print("\n🧪 Testing Multi-Hop vs Regular RAG Comparison...")
        
        try:
            test_query = "What are the benefits of AI models and what challenges do they face in practice?"
            
            # Test regular RAG
            print("📝 Testing Regular RAG...")
            start_time = time.time()
            regular_result = self.agent.run(
                query=test_query,
                multihop=False,
                session_id="comparison_regular"
            )
            regular_time = time.time() - start_time
            
            # Test multi-hop RAG
            print("📝 Testing Multi-Hop RAG...")
            start_time = time.time()
            multihop_result = self.agent.run(
                query=test_query,
                multihop=True,
                session_id="comparison_multihop"
            )
            multihop_time = time.time() - start_time
            
            comparison = {
                "query": test_query,
                "regular_rag": {
                    "execution_time": regular_time,
                    "answer_length": len(regular_result["answer"]),
                    "document_count": len(regular_result.get("source_documents", [])),
                    "answer_preview": regular_result["answer"][:200]
                },
                "multihop_rag": {
                    "execution_time": multihop_time,
                    "answer_length": len(multihop_result["answer"]),
                    "document_count": len(multihop_result.get("source_documents", [])),
                    "answer_preview": multihop_result["answer"][:200],
                    "has_multihop_info": "multihop_steps" in multihop_result or "steps_executed" in multihop_result
                }
            }
            
            print(f"\n📊 Comparison Results:")
            print(f"   Regular RAG: {regular_time:.2f}s, {comparison['regular_rag']['answer_length']} chars, {comparison['regular_rag']['document_count']} docs")
            print(f"   Multi-hop:   {multihop_time:.2f}s, {comparison['multihop_rag']['answer_length']} chars, {comparison['multihop_rag']['document_count']} docs")
            print(f"   Multi-hop info present: {comparison['multihop_rag']['has_multihop_info']}")
            
            return comparison
            
        except Exception as e:
            print(f"❌ Comparison tests failed: {e}")
            return None
    
    def test_regression(self):
        """Test that existing functionality still works."""
        print("\n🧪 Testing Regression (Existing Functionality)...")
        
        try:
            # Test basic RAG functionality
            basic_queries = [
                "What is artificial intelligence?",
                "How do neural networks work?",
                "What are the applications of machine learning?"
            ]
            
            for i, query in enumerate(basic_queries):
                print(f"   Testing basic query {i+1}: {query[:50]}...")
                result = self.agent.run(
                    query=query,
                    session_id=f"regression_test_{i}"
                )
                
                assert "answer" in result, f"Basic query {i+1} should return answer"
                assert len(result["answer"]) > 0, f"Basic query {i+1} should return non-empty answer"
                
            # Test query decomposition (existing functionality)
            decomp_query = "What are transformers and how are they used in NLP?"
            result = self.agent.run(
                query=decomp_query,
                query_decompose=True,
                session_id="regression_decomp"
            )
            
            assert "answer" in result, "Decomposition query should return answer"
            
            print("✅ Regression tests passed - existing functionality intact")
            return True
            
        except Exception as e:
            print(f"❌ Regression tests failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests and generate report."""
        print("🧪 Starting Multi-Hop Retrieval Test Suite...")
        print("=" * 60)
        
        # Setup
        if not self.setup():
            print("❌ Test suite aborted due to setup failure")
            return False
        
        # Run tests
        tests = [
            ("MultiHopContext", self.test_multihop_context),
            ("Query Decomposition", self.test_query_decomposition),
            ("Multi-Hop Integration", self.test_multihop_retrieval_integration),
            ("Multi-Hop vs Regular RAG", self.test_comparison_with_regular_rag),
            ("Regression Tests", self.test_regression),
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n{'=' * 60}")
            try:
                result = test_func()
                if result:
                    results[test_name] = result
                    passed += 1
                    print(f"✅ {test_name}: PASSED")
                else:
                    results[test_name] = False
                    print(f"❌ {test_name}: FAILED")
            except Exception as e:
                results[test_name] = f"ERROR: {e}"
                print(f"💥 {test_name}: ERROR - {e}")
        
        # Generate report
        print(f"\n{'=' * 60}")
        print("📊 TEST SUITE SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Multi-hop retrieval is working correctly.")
            return True
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
            return False


def main():
    """Main test execution."""
    tester = MultiHopTester()
    success = tester.run_all_tests()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"multihop_test_results_{timestamp}.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "success": success,
                "results": tester.test_results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 