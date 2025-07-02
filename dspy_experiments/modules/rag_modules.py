"""
DSPy RAG Modules

Implements various RAG architectures using DSPy modules and signatures.
Provides direct comparison points with the existing RAG system.
"""

import dspy
from typing import List, Dict, Any, Optional, Union
from .signatures import *
from .retrievers import LanceDBRetriever, HybridRetriever, GraphEnhancedRetriever


class BasicRAG(dspy.Module):
    """
    Simple RAG module for baseline comparison
    """
    
    def __init__(self, num_passages: int = 5, config_mode: str = "default"):
        super().__init__()
        
        self.retrieve = LanceDBRetriever(config_mode=config_mode, k=num_passages)
        self.generate_answer = dspy.ChainOfThought(ContextualQA)
        
    def forward(self, question: str) -> dspy.Prediction:
        # Retrieve relevant passages
        context_pred = self.retrieve(question)
        
        # Format context for the generator
        context = "\n---\n".join([
            f"Source {i+1}: {passage.text}" 
            for i, passage in enumerate(context_pred.passages)
        ])
        
        # Generate answer
        answer_pred = self.generate_answer(context=context, question=question)
        
        return dspy.Prediction(
            answer=answer_pred.answer,
            context=context_pred.passages,
            reasoning=getattr(answer_pred, 'rationale', '')
        )


class AdvancedRAG(dspy.Module):
    """
    Advanced RAG with query decomposition, verification, and hybrid retrieval
    """
    
    def __init__(
        self, 
        num_passages: int = 10,
        config_mode: str = "default",
        use_verification: bool = True,
        use_query_decomposition: bool = True
    ):
        super().__init__()
        
        self.num_passages = num_passages
        self.use_verification = use_verification
        self.use_query_decomposition = use_query_decomposition
        
        # Initialize components
        self.retrieve = HybridRetriever(config_mode=config_mode, k=num_passages)
        
        # Optional query decomposition
        if use_query_decomposition:
            self.decompose_query = dspy.ChainOfThought(QueryDecomposition)
        
        # Core generation
        self.generate_answer = dspy.ChainOfThought(ContextualQA)
        
        # Optional verification
        if use_verification:
            self.verify_answer = dspy.ChainOfThought(AnswerVerification)
        
        # Answer synthesis for complex queries
        self.synthesize_answer = dspy.ChainOfThought(AnswerSynthesis)
    
    def forward(self, question: str) -> dspy.Prediction:
        all_passages = []
        reasoning_steps = []
        
        # Step 1: Query decomposition (if enabled)
        if self.use_query_decomposition:
            try:
                decomp_pred = self.decompose_query(question=question)
                sub_questions = decomp_pred.sub_questions
                reasoning_steps.append(f"Decomposed into: {sub_questions}")
                
                # Retrieve for each sub-question
                for sub_q in sub_questions.split('\n')[:3]:  # Limit to 3 sub-questions
                    if sub_q.strip():
                        sub_context = self.retrieve(sub_q.strip())
                        all_passages.extend(sub_context.passages)
                        
            except Exception as e:
                reasoning_steps.append(f"Query decomposition failed: {e}")
                # Fallback to direct retrieval
                context_pred = self.retrieve(question)
                all_passages = context_pred.passages
        else:
            # Direct retrieval
            context_pred = self.retrieve(question)
            all_passages = context_pred.passages
        
        # Step 2: Deduplicate and format context
        seen_pids = set()
        unique_passages = []
        for passage in all_passages:
            if passage.pid not in seen_pids:
                seen_pids.add(passage.pid)
                unique_passages.append(passage)
                if len(unique_passages) >= self.num_passages:
                    break
        
        context = "\n---\n".join([
            f"Source {i+1}: {passage.text}" 
            for i, passage in enumerate(unique_passages)
        ])
        
        # Step 3: Generate initial answer
        if len(unique_passages) > 5:  # Use synthesis for complex cases
            sources = context
            answer_pred = self.synthesize_answer(question=question, sources=sources)
        else:
            answer_pred = self.generate_answer(context=context, question=question)
        
        reasoning_steps.append(f"Generated answer using {len(unique_passages)} sources")
        
        # Step 4: Verification (if enabled)
        verification_result = None
        if self.use_verification and unique_passages:
            try:
                verify_pred = self.verify_answer(
                    question=question,
                    answer=answer_pred.answer,
                    context=context
                )
                verification_result = {
                    "status": verify_pred.verification,
                    "explanation": verify_pred.explanation
                }
                reasoning_steps.append(f"Verification: {verify_pred.verification}")
            except Exception as e:
                reasoning_steps.append(f"Verification failed: {e}")
        
        return dspy.Prediction(
            answer=answer_pred.answer,
            context=unique_passages,
            reasoning="\n".join(reasoning_steps),
            verification=verification_result
        )


class MultiHopRAG(dspy.Module):
    """
    Multi-hop reasoning RAG for complex questions requiring iterative search
    """
    
    def __init__(
        self, 
        max_hops: int = 3,
        passages_per_hop: int = 5,
        config_mode: str = "default"
    ):
        super().__init__()
        
        self.max_hops = max_hops
        self.passages_per_hop = passages_per_hop
        
        # Initialize retriever
        self.retrieve = HybridRetriever(config_mode=config_mode, k=passages_per_hop)
        
        # Multi-hop components
        self.generate_hop_query = dspy.ChainOfThought(QueryGeneration)
        self.generate_answer = dspy.ChainOfThought(ContextualQA)
        self.check_completion = dspy.ChainOfThought(ContextExpansion)
        
    def forward(self, question: str) -> dspy.Prediction:
        all_context = []
        reasoning_steps = []
        
        current_question = question
        
        for hop in range(self.max_hops):
            reasoning_steps.append(f"Hop {hop + 1}: Searching for '{current_question}'")
            
            # Generate search query for this hop
            if hop > 0:  # For subsequent hops, use context to generate better queries
                context_summary = "\n".join([p.text[:200] for p in all_context[-3:]])
                query_pred = self.generate_hop_query(
                    question=question,
                    context=context_summary
                )
                search_query = query_pred.search_query
            else:
                search_query = current_question
            
            # Retrieve documents
            hop_context = self.retrieve(search_query)
            all_context.extend(hop_context.passages)
            
            reasoning_steps.append(f"Found {len(hop_context.passages)} documents")
            
            # Check if we have enough information
            current_context = "\n---\n".join([
                f"Source {i+1}: {passage.text}" 
                for i, passage in enumerate(all_context[-10:])  # Use recent context
            ])
            
            try:
                completion_check = self.check_completion(
                    question=question,
                    current_context=current_context
                )
                
                if "sufficient" in completion_check.needed_info.lower() or hop == self.max_hops - 1:
                    reasoning_steps.append("Sufficient information gathered")
                    break
                else:
                    # Generate next hop question
                    next_query = completion_check.search_query
                    current_question = next_query
                    reasoning_steps.append(f"Need more info: {completion_check.needed_info}")
                    
            except Exception as e:
                reasoning_steps.append(f"Completion check failed: {e}")
                break
        
        # Generate final answer
        final_context = "\n---\n".join([
            f"Source {i+1}: {passage.text}" 
            for i, passage in enumerate(all_context)
        ])
        
        answer_pred = self.generate_answer(context=final_context, question=question)
        
        return dspy.Prediction(
            answer=answer_pred.answer,
            context=all_context,
            reasoning="\n".join(reasoning_steps),
            hops_used=hop + 1
        )


class ConversationalRAG(dspy.Module):
    """
    RAG module that maintains conversation context
    """
    
    def __init__(self, num_passages: int = 8, config_mode: str = "default"):
        super().__init__()
        
        self.retrieve = HybridRetriever(config_mode=config_mode, k=num_passages)
        self.generate_answer = dspy.ChainOfThought(ConversationContextual)
        
        # Store conversation history
        self.conversation_history = []
    
    def forward(
        self, 
        question: str, 
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> dspy.Prediction:
        
        # Use provided history or internal history
        if conversation_history is not None:
            self.conversation_history = conversation_history
        
        # Retrieve relevant context
        context_pred = self.retrieve(question)
        
        # Format context
        context = "\n---\n".join([
            f"Source {i+1}: {passage.text}" 
            for i, passage in enumerate(context_pred.passages)
        ])
        
        # Format conversation history
        history_text = "\n".join([
            f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}"
            for turn in self.conversation_history[-3:]  # Keep last 3 turns
        ])
        
        # Generate contextual answer
        answer_pred = self.generate_answer(
            conversation_history=history_text,
            current_question=question,
            context=context
        )
        
        # Update conversation history
        self.conversation_history.append({
            "user": question,
            "assistant": answer_pred.answer
        })
        
        return dspy.Prediction(
            answer=answer_pred.answer,
            context=context_pred.passages,
            conversation_turns=len(self.conversation_history)
        )


class SelfCorrectingRAG(dspy.Module):
    """
    RAG module that can self-correct its answers through iterative refinement
    """
    
    def __init__(
        self, 
        num_passages: int = 10,
        max_corrections: int = 2,
        config_mode: str = "default"
    ):
        super().__init__()
        
        self.max_corrections = max_corrections
        
        self.retrieve = HybridRetriever(config_mode=config_mode, k=num_passages)
        self.generate_answer = dspy.ChainOfThought(ContextualQA)
        self.verify_answer = dspy.ChainOfThought(AnswerVerification)
        self.diagnose_error = dspy.ChainOfThought(ErrorDiagnosis)
    
    def forward(self, question: str) -> dspy.Prediction:
        context_pred = self.retrieve(question)
        context = "\n---\n".join([
            f"Source {i+1}: {passage.text}" 
            for i, passage in enumerate(context_pred.passages)
        ])
        
        corrections_made = []
        
        # Initial answer
        answer_pred = self.generate_answer(context=context, question=question)
        current_answer = answer_pred.answer
        
        for correction_round in range(self.max_corrections):
            # Verify current answer
            verify_pred = self.verify_answer(
                question=question,
                answer=current_answer,
                context=context
            )
            
            if verify_pred.verification.lower() == "accurate":
                corrections_made.append(f"Round {correction_round + 1}: Answer verified as accurate")
                break
            else:
                # Diagnose issues
                diagnosis_pred = self.diagnose_error(
                    question=question,
                    retrieved_context=context,
                    generated_answer=current_answer
                )
                
                corrections_made.append(
                    f"Round {correction_round + 1}: {diagnosis_pred.diagnosis}"
                )
                
                # Try to improve (simplified - could involve new retrieval)
                improvement_context = f"{context}\n\nPrevious answer: {current_answer}\nIssues identified: {diagnosis_pred.diagnosis}\nSuggestions: {diagnosis_pred.suggestions}"
                
                corrected_pred = self.generate_answer(
                    context=improvement_context,
                    question=question
                )
                current_answer = corrected_pred.answer
        
        return dspy.Prediction(
            answer=current_answer,
            context=context_pred.passages,
            corrections=corrections_made,
            final_verification=verify_pred.verification if 'verify_pred' in locals() else None
        )


# Test function
def test_rag_modules():
    """Test all RAG modules"""
    print("🧪 Testing DSPy RAG Modules...")
    
    test_question = "What are the key features of artificial intelligence?"
    
    modules_to_test = [
        ("BasicRAG", BasicRAG()),
        ("AdvancedRAG", AdvancedRAG()),
        ("MultiHopRAG", MultiHopRAG()),
    ]
    
    for name, module in modules_to_test:
        try:
            print(f"\n📊 Testing {name}...")
            result = module(test_question)
            print(f"✅ {name} completed successfully")
            print(f"📝 Answer: {result.answer[:100]}...")
            if hasattr(result, 'reasoning'):
                print(f"🤔 Reasoning: {result.reasoning[:100]}...")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    print("✅ RAG module tests completed")


if __name__ == "__main__":
    test_rag_modules() 