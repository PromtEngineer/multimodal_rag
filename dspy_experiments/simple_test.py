"""
Simple DSPy Test

Basic validation of DSPy components before running comprehensive tests.
"""

import os
import sys
import traceback

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all imports work correctly"""
    print("🔧 Testing imports...")
    
    try:
        from dspy_experiments.config import get_dspy_config
        print("✅ Config import successful")
        
        from dspy_experiments.modules.signatures import BasicQA, ContextualQA
        print("✅ Signatures import successful")
        
        from dspy_experiments.modules.retrievers import LanceDBRetriever
        print("✅ Retrievers import successful")
        
        from dspy_experiments.modules.rag_modules import BasicRAG
        print("✅ RAG modules import successful")
        
        from dspy_experiments.modules.react_modules import DSPyReActAgent
        print("✅ ReAct modules import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_rag():
    """Test basic RAG functionality"""
    print("\n🧠 Testing Basic RAG...")
    
    try:
        from dspy_experiments.modules.rag_modules import BasicRAG
        
        # Try to create BasicRAG instance
        rag = BasicRAG(num_passages=3)
        print("✅ BasicRAG instantiated successfully")
        
        # Try simple query
        result = rag("What is machine learning?")
        print(f"✅ BasicRAG query successful: {len(result.answer)} characters")
        print(f"📝 Answer preview: {result.answer[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ BasicRAG test failed: {e}")
        traceback.print_exc()
        return False

def test_retrievers():
    """Test retriever functionality"""
    print("\n📊 Testing Retrievers...")
    
    try:
        from dspy_experiments.modules.retrievers import LanceDBRetriever
        
        # Try to create retriever
        retriever = LanceDBRetriever(k=3)
        print("✅ LanceDBRetriever instantiated successfully")
        
        # Try retrieval
        result = retriever("artificial intelligence")
        print(f"✅ Retrieval successful: {len(result.passages)} passages")
        
        if result.passages:
            print(f"📝 First passage: {result.passages[0].text[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all simple tests"""
    print("🧪 DSPy Simple Test Suite")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Retrievers", test_retrievers),
        ("Basic RAG", test_basic_rag),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} test PASSED")
        else:
            print(f"❌ {test_name} test FAILED")
    
    print(f"\n{'='*40}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for comprehensive testing.")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before comprehensive testing.")
        return False

if __name__ == "__main__":
    main() 