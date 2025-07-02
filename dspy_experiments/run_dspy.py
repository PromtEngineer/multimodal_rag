#!/usr/bin/env python3
"""
DSPy Runner Script - Easy way to test DSPy systems

Usage:
    python run_dspy.py                    # Interactive mode
    python run_dspy.py "your question"    # Direct question
    python run_dspy.py --demo             # Run demo questions
    python run_dspy.py --compare          # Compare systems
"""

import sys
import time
import dspy
from config import get_dspy_config
from modules.rag_modules import BasicRAG, AdvancedRAG, MultiHopRAG

def setup_dspy():
    """Initialize DSPy configuration"""
    config = get_dspy_config()
    dspy.configure(lm=config.dspy_models['generation'])
    return config

def run_basic_rag(question: str):
    """Run BasicRAG with a question"""
    print("🔵 Running DSPy Basic RAG...")
    rag = BasicRAG(num_passages=3)
    start_time = time.time()
    result = rag(question)
    elapsed = time.time() - start_time
    
    print(f"📝 Question: {question}")
    print(f"🎯 Answer: {result.answer}")
    print(f"⏱️  Time: {elapsed:.2f}s")
    return result

def run_advanced_rag(question: str):
    """Run AdvancedRAG with a question"""
    print("🟡 Running DSPy Advanced RAG...")
    rag = AdvancedRAG(num_passages=3)
    start_time = time.time()
    result = rag(question)
    elapsed = time.time() - start_time
    
    print(f"📝 Question: {question}")
    print(f"🎯 Answer: {result.answer}")
    print(f"⏱️  Time: {elapsed:.2f}s")
    return result

def run_multihop_rag(question: str):
    """Run MultiHopRAG with a question"""
    print("🔴 Running DSPy MultiHop RAG...")
    rag = MultiHopRAG(num_passages=3)
    start_time = time.time()
    result = rag(question)
    elapsed = time.time() - start_time
    
    print(f"📝 Question: {question}")
    print(f"🎯 Answer: {result.answer}")
    if hasattr(result, 'reasoning'):
        print(f"🧠 Reasoning: {result.reasoning}")
    print(f"⏱️  Time: {elapsed:.2f}s")
    return result

def run_demo():
    """Run demonstration with sample questions"""
    setup_dspy()
    
    demo_questions = [
        "What invoice amounts are mentioned?",
        "Who issued the invoices?",
        "What are the payment methods?",
    ]
    
    print("🎭 DSPy Demo - Testing Multiple Questions")
    print("=" * 50)
    
    for question in demo_questions:
        print(f"\n{'='*20} Question {demo_questions.index(question)+1} {'='*20}")
        run_basic_rag(question)
        print()

def compare_systems(question: str):
    """Compare all DSPy systems with the same question"""
    setup_dspy()
    
    print("⚖️  DSPy Systems Comparison")
    print("=" * 40)
    print(f"📝 Question: {question}\n")
    
    # Test each system
    systems = [
        ("Basic RAG", run_basic_rag),
        ("Advanced RAG", run_advanced_rag),
        ("MultiHop RAG", run_multihop_rag),
    ]
    
    for name, func in systems:
        print(f"\n{'-'*15} {name} {'-'*15}")
        try:
            func(question)
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
        print()

def interactive_mode():
    """Interactive question-answering mode"""
    setup_dspy()
    
    print("🤖 DSPy Interactive Mode")
    print("Type your questions or 'quit' to exit")
    print("Commands: 'basic', 'advanced', 'multihop', 'compare'")
    print("=" * 50)
    
    mode = "basic"  # Default mode
    
    while True:
        user_input = input(f"\n[{mode}] Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'basic':
            mode = "basic"
            print("🔵 Switched to Basic RAG mode")
            continue
        elif user_input.lower() == 'advanced':
            mode = "advanced"
            print("🟡 Switched to Advanced RAG mode")
            continue
        elif user_input.lower() == 'multihop':
            mode = "multihop"
            print("🔴 Switched to MultiHop RAG mode")
            continue
        elif user_input.lower() == 'compare':
            if not user_input:
                continue
            question = input("Question to compare: ").strip()
            if question:
                compare_systems(question)
            continue
        
        if not user_input:
            continue
            
        try:
            if mode == "basic":
                run_basic_rag(user_input)
            elif mode == "advanced":
                run_advanced_rag(user_input)
            elif mode == "multihop":
                run_multihop_rag(user_input)
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    if len(sys.argv) == 1:
        # No arguments - interactive mode
        interactive_mode()
    elif "--demo" in sys.argv:
        # Demo mode
        run_demo()
    elif "--compare" in sys.argv:
        # Compare mode
        question = input("Enter question to compare across systems: ").strip()
        if question:
            compare_systems(question)
    else:
        # Direct question mode
        question = " ".join(sys.argv[1:])
        setup_dspy()
        run_basic_rag(question)

if __name__ == "__main__":
    main() 