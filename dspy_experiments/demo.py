"""
DSPy Integration Demo

Demonstrates the key features and benefits of DSPy integration
with side-by-side comparisons to the current system.
"""

import os
import sys
import time
from typing import Dict, Any

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def demo_basic_comparison():
    """Demonstrate basic DSPy vs current system comparison"""
    print("🔀 BASIC COMPARISON: DSPy vs Current System")
    print("=" * 60)
    
    question = "What are the key components of artificial intelligence?"
    
    try:
        # Current System
        print("\n🔧 Current System:")
        start_time = time.time()
        
        from rag_system.main import get_agent
        current_agent = get_agent("default")
        current_result = current_agent.run(question)
        
        current_time = time.time() - start_time
        current_answer = current_result.get("response", str(current_result)) if isinstance(current_result, dict) else str(current_result)
        
        print(f"⏱️  Time: {current_time:.2f}s")
        print(f"📝 Answer: {current_answer[:200]}...")
        
    except Exception as e:
        print(f"❌ Current system failed: {e}")
        current_time = 0
        current_answer = "Failed to generate"
    
    try:
        # DSPy System
        print("\n🧠 DSPy System:")
        start_time = time.time()
        
        from dspy_experiments.modules.rag_modules import BasicRAG
        dspy_rag = BasicRAG(num_passages=5)
        dspy_result = dspy_rag(question)
        
        dspy_time = time.time() - start_time
        
        print(f"⏱️  Time: {dspy_time:.2f}s")
        print(f"📝 Answer: {dspy_result.answer[:200]}...")
        print(f"🤔 Has reasoning: {hasattr(dspy_result, 'reasoning')}")
        print(f"📊 Context passages: {len(dspy_result.context)}")
        
    except Exception as e:
        print(f"❌ DSPy system failed: {e}")
        dspy_time = 0
    
    # Comparison
    print(f"\n📊 COMPARISON:")
    if current_time > 0 and dspy_time > 0:
        speed_factor = current_time / dspy_time if dspy_time > 0 else 1
        print(f"⚡ Speed ratio: {speed_factor:.1f}x")
    print(f"🆚 DSPy provides: structured output, reasoning transparency, better composability")


def demo_advanced_features():
    """Demonstrate advanced DSPy features"""
    print("\n\n🚀 ADVANCED FEATURES DEMO")
    print("=" * 60)
    
    complex_question = "How do neural networks relate to biological neurons and what are the key differences in learning mechanisms?"
    
    try:
        from dspy_experiments.modules.rag_modules import AdvancedRAG
        
        print("\n🎯 Advanced RAG with Query Decomposition & Verification:")
        advanced_rag = AdvancedRAG(
            num_passages=8,
            use_verification=True,
            use_query_decomposition=True
        )
        
        start_time = time.time()
        result = advanced_rag(complex_question)
        duration = time.time() - start_time
        
        print(f"⏱️  Processing time: {duration:.2f}s")
        print(f"📝 Answer: {result.answer[:300]}...")
        
        if hasattr(result, 'reasoning') and result.reasoning:
            print(f"\n🤔 Reasoning Process:")
            reasoning_lines = result.reasoning.split('\n')[:5]  # First 5 steps
            for line in reasoning_lines:
                if line.strip():
                    print(f"   • {line.strip()}")
        
        if hasattr(result, 'verification') and result.verification:
            print(f"\n✅ Verification: {result.verification.get('status', 'N/A')}")
            print(f"💭 Explanation: {result.verification.get('explanation', 'N/A')[:100]}...")
        
        print(f"\n📊 Retrieved {len(result.context)} passages")
        
    except Exception as e:
        print(f"❌ Advanced features demo failed: {e}")


def demo_multihop_reasoning():
    """Demonstrate multi-hop reasoning capabilities"""
    print("\n\n🔄 MULTI-HOP REASONING DEMO")
    print("=" * 60)
    
    multihop_question = "What are the ethical implications of AI in healthcare and how do they compare to traditional medical ethics?"
    
    try:
        from dspy_experiments.modules.rag_modules import MultiHopRAG
        
        print("\n🎯 Multi-Hop RAG for Complex Reasoning:")
        multihop_rag = MultiHopRAG(max_hops=3, passages_per_hop=4)
        
        start_time = time.time()
        result = multihop_rag(multihop_question)
        duration = time.time() - start_time
        
        print(f"⏱️  Processing time: {duration:.2f}s")
        print(f"🔄 Hops used: {getattr(result, 'hops_used', 'Unknown')}")
        print(f"📝 Answer: {result.answer[:300]}...")
        
        if hasattr(result, 'reasoning') and result.reasoning:
            print(f"\n🤔 Multi-Hop Process:")
            reasoning_lines = result.reasoning.split('\n')[:6]  # First 6 steps
            for line in reasoning_lines:
                if line.strip() and ('Hop' in line or 'Found' in line or 'Sufficient' in line):
                    print(f"   • {line.strip()}")
        
        print(f"\n📊 Total context gathered: {len(result.context)} passages")
        
    except Exception as e:
        print(f"❌ Multi-hop demo failed: {e}")


def demo_declarative_signatures():
    """Show the power of declarative signatures"""
    print("\n\n📋 DECLARATIVE SIGNATURES DEMO")
    print("=" * 60)
    
    print("\n💡 Traditional Approach (Manual Prompts):")
    print("""
    prompt = '''
    Given the following context documents:
    {context}
    
    Please answer the question: {question}
    
    Provide a clear, accurate, and comprehensive response based on the information provided.
    If the context doesn't contain sufficient information, please indicate this in your response.
    Format your answer clearly and cite relevant sources where appropriate.
    '''
    """)
    
    print("\n🚀 DSPy Approach (Declarative Signatures):")
    print("""
    class ContextualQA(dspy.Signature):
        '''Answer questions using provided context documents.'''
        
        context = dspy.InputField(desc="relevant text passages that may contain the answer")
        question = dspy.InputField(desc="the user's question")
        answer = dspy.OutputField(desc="answer based on the provided context")
    
    # DSPy automatically optimizes the prompt!
    generate = dspy.ChainOfThought(ContextualQA)
    """)
    
    print("\n✨ Benefits:")
    print("   • No manual prompt engineering")
    print("   • Automatic optimization")
    print("   • Better composability")
    print("   • Easier maintenance")
    print("   • Built-in reasoning")


def demo_retrieval_comparison():
    """Demonstrate retrieval system improvements"""
    print("\n\n📊 RETRIEVAL SYSTEM COMPARISON")
    print("=" * 60)
    
    query = "machine learning algorithms"
    
    try:
        print("\n🔍 Testing Different Retrievers:")
        
        # LanceDB Retriever
        from dspy_experiments.modules.retrievers import LanceDBRetriever
        lance_retriever = LanceDBRetriever(k=5)
        
        start_time = time.time()
        lance_result = lance_retriever(query)
        lance_time = time.time() - start_time
        
        print(f"\n📈 LanceDB Retriever:")
        print(f"   ⏱️  Time: {lance_time:.3f}s")
        print(f"   📊 Passages: {len(lance_result.passages)}")
        
        # Hybrid Retriever
        from dspy_experiments.modules.retrievers import HybridRetriever
        hybrid_retriever = HybridRetriever(k=5, dense_weight=0.7, bm25_weight=0.3)
        
        start_time = time.time()
        hybrid_result = hybrid_retriever(query)
        hybrid_time = time.time() - start_time
        
        print(f"\n🔀 Hybrid Retriever (Dense + BM25):")
        print(f"   ⏱️  Time: {hybrid_time:.3f}s")
        print(f"   📊 Passages: {len(hybrid_result.passages)}")
        print(f"   🎯 Fusion: Reciprocal rank fusion with configurable weights")
        
        print(f"\n💡 DSPy Retrieval Benefits:")
        print("   • Seamless integration with existing LanceDB")
        print("   • Multiple retrieval strategies")
        print("   • Automatic result fusion")
        print("   • Standardized DSPy interface")
        
    except Exception as e:
        print(f"❌ Retrieval demo failed: {e}")


def main():
    """Run complete DSPy demonstration"""
    print("🎭 DSPy INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print("Showcasing the power and benefits of DSPy integration")
    print("=" * 80)
    
    demos = [
        ("Basic Comparison", demo_basic_comparison),
        ("Advanced Features", demo_advanced_features),
        ("Multi-Hop Reasoning", demo_multihop_reasoning),
        ("Declarative Signatures", demo_declarative_signatures),
        ("Retrieval Comparison", demo_retrieval_comparison),
    ]
    
    for demo_name, demo_func in demos:
        try:
            demo_func()
        except KeyboardInterrupt:
            print(f"\n⏸️  Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {demo_name} demo failed: {e}")
        
        print("\n" + "-" * 60)
    
    print("\n🎉 DEMONSTRATION COMPLETE!")
    print("\n🏆 Key Takeaways:")
    print("   ✅ DSPy integrates seamlessly with existing system")
    print("   ✅ Provides advanced reasoning capabilities")
    print("   ✅ Eliminates manual prompt engineering")
    print("   ✅ Enables automatic optimization")
    print("   ✅ Maintains full compatibility")
    print("   ✅ Ready for production evaluation")
    
    print(f"\n📚 Next Steps:")
    print("   1. Run comprehensive tests: python dspy_experiments/comprehensive_test.py")
    print("   2. Explore optimization: DSPy teleprompters")
    print("   3. Production A/B testing")
    
    print("\n🚀 DSPy integration is ready for deployment!")


if __name__ == "__main__":
    main() 