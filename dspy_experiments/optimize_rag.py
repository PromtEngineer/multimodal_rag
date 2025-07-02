"""
DSPy RAG Optimization using Teleprompters

This script demonstrates how to create an optimized version of the RAG system
using DSPy's automatic optimization capabilities (teleprompters).
"""

import dspy
import json
from typing import List, Dict, Any
from config import get_dspy_config
from modules.rag_modules import BasicRAG
from modules.signatures import BasicQA

def create_training_examples() -> List[dspy.Example]:
    """
    Create training examples for DSPy optimization.
    In a real scenario, you'd have human-labeled examples.
    """
    examples = [
        dspy.Example(
            question="What invoice numbers are mentioned in the documents?", 
            answer="Invoice 1039 and Invoice 1041 are mentioned in the documents."
        ).with_inputs('question'),
        
        dspy.Example(
            question="What company issued the invoices?",
            answer="PromptX AI LLC issued the invoices."
        ).with_inputs('question'),
        
        dspy.Example(
            question="What is the total amount due?",
            answer="The total amount due is $9,000.00 as mentioned in invoice 1039."
        ).with_inputs('question'),
        
        dspy.Example(
            question="What are the payment methods mentioned?",
            answer="Bank Transfer (ACH) is mentioned as a payment method with account number 102103174 and routing number 211370150."
        ).with_inputs('question'),
        
        dspy.Example(
            question="Who is the bill to company?",
            answer="DeepDyve (ssmith@deepdyve.com) at 2221 Broadway Street Redwood, CA 94063 is the bill to company."
        ).with_inputs('question'),
    ]
    
    return examples

def optimize_basic_rag():
    """
    Optimize BasicRAG using DSPy teleprompters
    """
    print("🚀 Starting DSPy RAG Optimization...")
    
    # Configure DSPy
    config = get_dspy_config()
    dspy.configure(lm=config.dspy_models['generation'])
    
    # Create training examples
    training_examples = create_training_examples()
    
    # Split into train/validation
    train_examples = training_examples[:3]
    val_examples = training_examples[3:]
    
    print(f"📊 Training with {len(train_examples)} examples, validating with {len(val_examples)}")
    
    # Create unoptimized RAG
    unoptimized_rag = BasicRAG(num_passages=3)
    
    # Define evaluation metric
    def accuracy_metric(example, pred, trace=None):
        """Simple accuracy metric - checks if key terms are present"""
        if hasattr(pred, 'answer'):
            answer = pred.answer.lower()
            expected = example.answer.lower()
            
            # Extract key terms from expected answer
            key_terms = []
            if 'invoice' in expected:
                key_terms.extend(['invoice', '1039', '1041'])
            if 'promptx' in expected or 'ai llc' in expected:
                key_terms.extend(['promptx', 'ai'])
            if '$9,000' in expected or '9000' in expected:
                key_terms.extend(['9000', '$'])
            if 'bank' in expected and 'transfer' in expected:
                key_terms.extend(['bank', 'transfer', 'ach'])
            if 'deepdyve' in expected:
                key_terms.extend(['deepdyve'])
            
            # Check if key terms are present
            score = sum(1 for term in key_terms if term in answer) / max(len(key_terms), 1)
            return score >= 0.5  # At least 50% of key terms present
        
        return False
    
    print("🔧 Setting up BootstrapFewShot optimizer...")
    
    # Use BootstrapFewShot teleprompter for optimization
    optimizer = dspy.BootstrapFewShot(
        metric=accuracy_metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        max_rounds=1  # Keep it simple for demo
    )
    
    print("⚡ Optimizing RAG system...")
    
    # Optimize the RAG
    try:
        optimized_rag = optimizer.compile(
            student=unoptimized_rag,
            trainset=train_examples
        )
        
        print("✅ Optimization completed!")
        
        # Test both versions
        test_question = "What invoice amounts are mentioned in the documents?"
        
        print(f"\n🧪 Testing question: {test_question}")
        
        # Test unoptimized
        print("\n📊 Unoptimized RAG:")
        unopt_result = unoptimized_rag(test_question)
        print(f"Answer: {unopt_result.answer}")
        
        # Test optimized  
        print("\n🚀 Optimized RAG:")
        opt_result = optimized_rag(test_question)
        print(f"Answer: {opt_result.answer}")
        
        # Save optimized model
        optimized_rag.save("optimized_rag_model.json")
        print("\n💾 Optimized model saved to 'optimized_rag_model.json'")
        
        return optimized_rag
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print("💡 This is normal - optimization requires good examples and may need tuning")
        return unoptimized_rag

def load_optimized_rag(model_path: str = "optimized_rag_model.json"):
    """
    Load a previously optimized RAG model
    """
    try:
        config = get_dspy_config()
        dspy.configure(lm=config.dspy_models['generation'])
        
        # Create base RAG and load optimized version
        rag = BasicRAG(num_passages=3)
        rag.load(model_path)
        
        print(f"✅ Loaded optimized RAG from {model_path}")
        return rag
        
    except Exception as e:
        print(f"❌ Failed to load optimized model: {e}")
        print("💡 Run optimize_basic_rag() first to create the optimized model")
        return None

def quick_optimization_demo():
    """
    Quick demo of DSPy optimization capabilities
    """
    print("🎭 DSPy Optimization Demo")
    print("=" * 40)
    
    # Simple signature optimization
    config = get_dspy_config()
    dspy.configure(lm=config.dspy_models['generation'])
    
    # Create a simple QA module
    class SimpleQA(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_answer = dspy.ChainOfThought(BasicQA)
        
        def forward(self, question):
            return self.generate_answer(question=question)
    
    # Create training data
    examples = [
        dspy.Example(question="What is 2+2?", answer="4").with_inputs('question'),
        dspy.Example(question="What color is the sky?", answer="blue").with_inputs('question'),
    ]
    
    # Simple metric
    def simple_metric(example, pred, trace=None):
        return example.answer.lower() in pred.answer.lower()
    
    # Create and optimize
    qa = SimpleQA()
    
    print("🔧 Testing simple optimization...")
    try:
        optimizer = dspy.BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1)
        optimized_qa = optimizer.compile(qa, trainset=examples)
        
        result = optimized_qa("What is the capital of France?")
        print(f"✅ Optimized answer: {result.answer}")
        
    except Exception as e:
        print(f"⚠️ Simple optimization demo failed: {e}")

if __name__ == "__main__":
    print("🎯 Choose optimization approach:")
    print("1. quick_optimization_demo() - Simple demo")
    print("2. optimize_basic_rag() - Full RAG optimization")
    print("3. load_optimized_rag() - Load existing optimized model")
    
    # Run quick demo by default
    quick_optimization_demo() 