#!/usr/bin/env python3
"""
All Systems Runner - Compare Original vs DSPy RAG Systems

Usage:
    python run_all_systems.py                    # Interactive mode
    python run_all_systems.py "your question"    # Direct question
    python run_all_systems.py --demo             # Demo with sample questions
"""

import sys
import time
import os
import subprocess

def run_original_system(question: str):
    """Run the original RAG system"""
    print("🟢 Running Original RAG System...")
    start_time = time.time()
    
    try:
        from rag_system.main import get_agent
        agent = get_agent('default')
        result = agent.run(question)
        elapsed = time.time() - start_time
        
        print(f"📝 Question: {question}")
        print(f"🎯 Answer: {result}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error in Original System: {e}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        return None

def run_dspy_system(question: str, system_type: str = "basic"):
    """Run DSPy system using subprocess"""
    print(f"🔵 Running DSPy {system_type.title()} RAG...")
    start_time = time.time()
    
    try:
        # Change to dspy_experiments directory and run
        cmd = [
            sys.executable, '-c', f'''
import dspy
from config import get_dspy_config
from modules.rag_modules import BasicRAG, AdvancedRAG, MultiHopRAG

config = get_dspy_config()
dspy.configure(lm=config.dspy_models['generation'])

if "{system_type}" == "basic":
    rag = BasicRAG(num_passages=3)
elif "{system_type}" == "advanced":
    rag = AdvancedRAG(num_passages=3)
elif "{system_type}" == "multihop":
    rag = MultiHopRAG(num_passages=3)
else:
    rag = BasicRAG(num_passages=3)

result = rag("{question}")
print(f"ANSWER: {{result.answer}}")
'''
        ]
        
        # Run from dspy_experiments directory
        result = subprocess.run(
            cmd, 
            cwd='dspy_experiments',
            capture_output=True, 
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # Extract the answer from output
            lines = result.stdout.strip().split('\n')
            answer_line = [line for line in lines if line.startswith('ANSWER:')]
            if answer_line:
                answer = answer_line[0].replace('ANSWER:', '').strip()
                print(f"📝 Question: {question}")
                print(f"🎯 Answer: {answer}")
                print(f"⏱️  Time: {elapsed:.2f}s")
                return answer
            else:
                print(f"📝 Question: {question}")
                print(f"🎯 Answer: [Could not extract answer]")
                print(f"⏱️  Time: {elapsed:.2f}s")
                print(f"📋 Full output: {result.stdout}")
                return result.stdout
        else:
            print(f"❌ Error in DSPy System: {result.stderr}")
            print(f"⏱️  Time: {elapsed:.2f}s")
            return None
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏰ DSPy System timed out after {elapsed:.2f}s")
        return None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Error running DSPy System: {e}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        return None

def compare_systems(question: str):
    """Compare all systems with the same question"""
    print("⚖️  All Systems Comparison")
    print("=" * 50)
    print(f"📝 Question: {question}\n")
    
    systems = [
        ("Original RAG", lambda q: run_original_system(q)),
        ("DSPy Basic", lambda q: run_dspy_system(q, "basic")),
        ("DSPy Advanced", lambda q: run_dspy_system(q, "advanced")),
    ]
    
    results = {}
    
    for name, func in systems:
        print(f"\n{'-'*15} {name} {'-'*15}")
        try:
            result = func(question)
            results[name] = result
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results[name] = None
        print()
    
    # Summary
    print("📊 Summary")
    print("-" * 20)
    for name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {name}: {'Success' if result else 'Failed'}")

def run_demo():
    """Run demonstration with sample questions"""
    demo_questions = [
        "What invoice amounts are mentioned?",
        "Who issued the invoices?", 
        "What are the payment methods?",
    ]
    
    print("🎭 All Systems Demo")
    print("=" * 30)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'='*10} Question {i} {'='*10}")
        compare_systems(question)

def interactive_mode():
    """Interactive question-answering mode"""
    print("🤖 All Systems Interactive Mode")
    print("Type questions or commands:")
    print("- 'compare' - Compare all systems")
    print("- 'original' - Use original system only")
    print("- 'dspy' - Use DSPy basic only")
    print("- 'quit' - Exit")
    print("=" * 40)
    
    mode = "compare"  # Default mode
    
    while True:
        user_input = input(f"\n[{mode}] Your question: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'compare':
            mode = "compare"
            print("⚖️  Switched to comparison mode")
            continue
        elif user_input.lower() == 'original':
            mode = "original"
            print("🟢 Switched to original system only")
            continue
        elif user_input.lower() == 'dspy':
            mode = "dspy"
            print("🔵 Switched to DSPy basic only")
            continue
        
        if not user_input:
            continue
            
        try:
            if mode == "compare":
                compare_systems(user_input)
            elif mode == "original":
                run_original_system(user_input)
            elif mode == "dspy":
                run_dspy_system(user_input, "basic")
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
    else:
        # Direct question mode
        question = " ".join(sys.argv[1:])
        compare_systems(question)

if __name__ == "__main__":
    main() 