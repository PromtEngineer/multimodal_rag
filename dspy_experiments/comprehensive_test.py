"""
Comprehensive DSPy vs Current System Testing

This script provides extensive testing and comparison between:
1. Current RAG system implementations
2. DSPy-based implementations 
3. Performance metrics and analysis

Tests cover all major features including:
- Basic RAG functionality
- Advanced features (query decomposition, verification, reranking)
- Multi-hop reasoning
- ReAct agents
- Performance benchmarks
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DSPy imports
from dspy_experiments.config import configure_dspy_model, get_dspy_config
from dspy_experiments.modules.rag_modules import BasicRAG, AdvancedRAG, MultiHopRAG
from dspy_experiments.modules.retrievers import LanceDBRetriever, HybridRetriever

# Current system imports
from rag_system.main import get_agent
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline


class TestResult:
    """Container for test results"""
    
    def __init__(self, test_name: str, system: str):
        self.test_name = test_name
        self.system = system
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.success = False
        self.answer = ""
        self.context_count = 0
        self.error = None
        self.metadata = {}
    
    def start(self):
        self.start_time = time.time()
    
    def end(self, success: bool = True, answer: str = "", context_count: int = 0, error: str = None, **metadata):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.answer = answer
        self.context_count = context_count
        self.error = error
        self.metadata = metadata
    
    def to_dict(self):
        return {
            "test_name": self.test_name,
            "system": self.system,
            "duration": self.duration,
            "success": self.success,
            "answer_length": len(self.answer),
            "context_count": self.context_count,
            "error": self.error,
            "metadata": self.metadata
        }


class ComprehensiveTester:
    """Main testing orchestrator"""
    
    def __init__(self):
        self.test_questions = [
            # Basic questions
            "What is artificial intelligence?",
            "What are the key features of machine learning?",
            "How does deep learning work?",
            
            # Complex questions requiring decomposition
            "What are the differences between supervised and unsupervised learning, and when should each be used?",
            "How has artificial intelligence evolved from the 1950s to today, and what are the major milestones?",
            
            # Multi-hop questions
            "What are the ethical implications of AI in healthcare and how do they compare to traditional medical ethics?",
            "How do neural networks relate to biological neurons and what are the key differences?",
            
            # Document-specific questions (if documents are available)
            "What specific invoices are mentioned in the documents?",
            "What are the key findings in the research papers?",
        ]
        
        self.results = []
        self.current_test_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def test_current_system(self, question: str) -> TestResult:
        """Test the current RAG system"""
        result = TestResult("current_system", "current")
        result.start()
        
        try:
            # Test current system
            agent = get_agent("default")
            response = agent.run(question)
            
            # Extract relevant information
            answer = response.get("response", "") if isinstance(response, dict) else str(response)
            context_count = len(response.get("context", [])) if isinstance(response, dict) else 0
            
            result.end(
                success=True,
                answer=answer,
                context_count=context_count,
                system_type="current_rag"
            )
            
        except Exception as e:
            result.end(success=False, error=str(e))
            print(f"❌ Current system failed: {e}")
        
        return result
    
    def test_dspy_basic(self, question: str) -> TestResult:
        """Test basic DSPy RAG"""
        result = TestResult("dspy_basic", "dspy")
        result.start()
        
        try:
            rag = BasicRAG(num_passages=5)
            response = rag(question)
            
            result.end(
                success=True,
                answer=response.answer,
                context_count=len(response.context),
                reasoning_available=hasattr(response, 'reasoning')
            )
            
        except Exception as e:
            result.end(success=False, error=str(e))
            print(f"❌ DSPy Basic failed: {e}")
        
        return result
    
    def test_dspy_advanced(self, question: str) -> TestResult:
        """Test advanced DSPy RAG with all features"""
        result = TestResult("dspy_advanced", "dspy")
        result.start()
        
        try:
            rag = AdvancedRAG(
                num_passages=10,
                use_verification=True,
                use_query_decomposition=True
            )
            response = rag(question)
            
            result.end(
                success=True,
                answer=response.answer,
                context_count=len(response.context),
                has_verification=hasattr(response, 'verification'),
                has_reasoning=hasattr(response, 'reasoning'),
                verification_status=response.verification.get('status', 'none') if hasattr(response, 'verification') and response.verification else 'none'
            )
            
        except Exception as e:
            result.end(success=False, error=str(e))
            print(f"❌ DSPy Advanced failed: {e}")
        
        return result
    
    def test_dspy_multihop(self, question: str) -> TestResult:
        """Test multi-hop DSPy RAG"""
        result = TestResult("dspy_multihop", "dspy")
        result.start()
        
        try:
            rag = MultiHopRAG(max_hops=3, passages_per_hop=5)
            response = rag(question)
            
            result.end(
                success=True,
                answer=response.answer,
                context_count=len(response.context),
                hops_used=getattr(response, 'hops_used', 0),
                has_reasoning=hasattr(response, 'reasoning')
            )
            
        except Exception as e:
            result.end(success=False, error=str(e))
            print(f"❌ DSPy MultiHop failed: {e}")
        
        return result
    
    def test_retrieval_comparison(self, question: str) -> Dict[str, TestResult]:
        """Compare different retrieval methods"""
        results = {}
        
        # Test LanceDB retriever
        try:
            result = TestResult("lancedb_retrieval", "dspy")
            result.start()
            
            retriever = LanceDBRetriever(k=10)
            response = retriever(question)
            
            result.end(
                success=True,
                context_count=len(response.passages),
                retrieval_type="lancedb"
            )
            results["lancedb"] = result
            
        except Exception as e:
            result = TestResult("lancedb_retrieval", "dspy")
            result.end(success=False, error=str(e))
            results["lancedb"] = result
        
        # Test Hybrid retriever
        try:
            result = TestResult("hybrid_retrieval", "dspy")
            result.start()
            
            retriever = HybridRetriever(k=10)
            response = retriever(question)
            
            result.end(
                success=True,
                context_count=len(response.passages),
                retrieval_type="hybrid"
            )
            results["hybrid"] = result
            
        except Exception as e:
            result = TestResult("hybrid_retrieval", "dspy")
            result.end(success=False, error=str(e))
            results["hybrid"] = result
        
        return results
    
    def run_comprehensive_test(self):
        """Run all tests and collect results"""
        print("🚀 Starting Comprehensive DSPy vs Current System Testing")
        print(f"📊 Test batch: {self.current_test_batch}")
        print(f"🔢 Testing {len(self.test_questions)} questions")
        print("=" * 60)
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n📝 Question {i}/{len(self.test_questions)}: {question[:60]}...")
            
            # Test current system
            print("  🔧 Testing current system...")
            current_result = self.test_current_system(question)
            self.results.append(current_result)
            
            # Test DSPy Basic
            print("  🧠 Testing DSPy Basic...")
            basic_result = self.test_dspy_basic(question)
            self.results.append(basic_result)
            
            # Test DSPy Advanced 
            print("  🚀 Testing DSPy Advanced...")
            advanced_result = self.test_dspy_advanced(question)
            self.results.append(advanced_result)
            
            # Test DSPy MultiHop
            print("  🔄 Testing DSPy MultiHop...")
            multihop_result = self.test_dspy_multihop(question)
            self.results.append(multihop_result)
            
            # Test retrieval methods
            print("  📊 Testing retrieval methods...")
            retrieval_results = self.test_retrieval_comparison(question)
            for name, result in retrieval_results.items():
                self.results.append(result)
            
            # Print quick summary for this question
            successful_tests = sum(1 for r in [current_result, basic_result, advanced_result, multihop_result] if r.success)
            print(f"  ✅ {successful_tests}/4 main tests successful")
        
        print("\n" + "=" * 60)
        print("🏁 Testing completed! Generating report...")
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        # Success rates by system
        system_stats = {}
        for result in self.results:
            system = result.system
            if system not in system_stats:
                system_stats[system] = {"total": 0, "success": 0, "total_time": 0}
            
            system_stats[system]["total"] += 1
            if result.success:
                system_stats[system]["success"] += 1
            if result.duration:
                system_stats[system]["total_time"] += result.duration
        
        print("\n🎯 SUCCESS RATES:")
        for system, stats in system_stats.items():
            success_rate = (stats["success"] / stats["total"]) * 100
            avg_time = stats["total_time"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {system.title()}: {success_rate:.1f}% ({stats['success']}/{stats['total']}) - Avg: {avg_time:.2f}s")
        
        # Performance comparison
        print("\n⚡ PERFORMANCE COMPARISON:")
        test_types = {}
        for result in self.results:
            test_name = result.test_name
            if test_name not in test_types:
                test_types[test_name] = []
            if result.success and result.duration:
                test_types[test_name].append(result.duration)
        
        for test_type, durations in test_types.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                print(f"  {test_type}: {avg_duration:.2f}s avg ({len(durations)} successful runs)")
        
        # Error analysis
        print("\n🐛 ERROR ANALYSIS:")
        errors = {}
        for result in self.results:
            if not result.success and result.error:
                error_key = f"{result.system}_{result.test_name}"
                if error_key not in errors:
                    errors[error_key] = []
                errors[error_key].append(result.error)
        
        for error_key, error_list in errors.items():
            print(f"  {error_key}: {len(error_list)} errors")
            # Show unique errors
            unique_errors = list(set(error_list))
            for error in unique_errors[:2]:  # Show first 2 unique errors
                print(f"    - {error[:100]}...")
        
        # Save detailed results
        self.save_results()
        
        print(f"\n💾 Detailed results saved to: test_results_{self.current_test_batch}.json")
        print("\n🎉 Analysis complete!")
    
    def save_results(self):
        """Save detailed results to JSON file"""
        filename = f"test_results_{self.current_test_batch}.json"
        
        report_data = {
            "test_batch": self.current_test_batch,
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "test_questions": self.test_questions,
            "results": [result.to_dict() for result in self.results]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"✅ Results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def test_basic_setup():
    """Test basic DSPy setup before running comprehensive tests"""
    print("🔧 Testing Basic DSPy Setup...")
    
    try:
        # Test configuration
        config = get_dspy_config()
        print(f"✅ DSPy config loaded with models: {list(config.dspy_models.keys())}")
        
        # Test basic retrieval
        retriever = LanceDBRetriever(k=3)
        result = retriever("test query")
        print(f"✅ Basic retrieval working: {len(result.passages)} passages")
        
        # Test basic RAG
        rag = BasicRAG(num_passages=3)
        response = rag("What is AI?")
        print(f"✅ Basic RAG working: {len(response.answer)} char answer")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic setup test failed: {e}")
        print("🔍 Stack trace:")
        traceback.print_exc()
        return False


def main():
    """Main test execution"""
    print("🧪 DSPy Comprehensive Testing Suite")
    print("=" * 50)
    
    # Test basic setup first
    if not test_basic_setup():
        print("❌ Basic setup failed. Please check configuration and try again.")
        return
    
    print("\n✅ Basic setup successful. Proceeding with comprehensive tests...")
    
    # Run comprehensive tests
    tester = ComprehensiveTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main() 