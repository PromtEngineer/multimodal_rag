"""
DSPy Signatures for RAG System

Defines all the declarative signatures used in the DSPy-based RAG implementation.
These signatures replace manual prompt engineering with declarative task descriptions.
"""

import dspy
from typing import List, Optional


class BasicQA(dspy.Signature):
    """Answer questions with accurate, concise responses based on context."""
    
    question = dspy.InputField(desc="the user's question")
    answer = dspy.OutputField(desc="a clear, accurate answer to the question")


class ContextualQA(dspy.Signature):
    """Answer questions using provided context documents."""
    
    context = dspy.InputField(desc="relevant text passages that may contain the answer")
    question = dspy.InputField(desc="the user's question")
    answer = dspy.OutputField(desc="answer based on the provided context")


class QueryDecomposition(dspy.Signature):
    """Break down complex questions into simpler sub-questions for better retrieval."""
    
    question = dspy.InputField(desc="the original complex question")
    sub_questions = dspy.OutputField(desc="list of 2-3 simpler questions that together answer the original")


class QueryGeneration(dspy.Signature):
    """Generate effective search queries for document retrieval."""
    
    question = dspy.InputField(desc="the user's question")
    context = dspy.InputField(desc="any existing context or previous search results")
    search_query = dspy.OutputField(desc="optimized query for document search")


class AnswerSynthesis(dspy.Signature):
    """Synthesize information from multiple sources into a coherent answer."""
    
    question = dspy.InputField(desc="the original question")
    sources = dspy.InputField(desc="relevant information from multiple documents")
    answer = dspy.OutputField(desc="comprehensive answer synthesized from all sources")


class AnswerVerification(dspy.Signature):
    """Verify if an answer is accurate and complete based on the source context."""
    
    question = dspy.InputField(desc="the original question")
    answer = dspy.InputField(desc="the proposed answer")
    context = dspy.InputField(desc="source documents used to generate the answer")
    verification = dspy.OutputField(desc="assessment of answer accuracy: 'accurate', 'partially accurate', or 'inaccurate'")
    explanation = dspy.OutputField(desc="brief explanation of the verification decision")


class DocumentRelevance(dspy.Signature):
    """Assess the relevance of a document to a given question."""
    
    question = dspy.InputField(desc="the question being asked")
    document = dspy.InputField(desc="the document text to assess")
    relevance = dspy.OutputField(desc="relevance score: 'high', 'medium', 'low', or 'none'")
    reason = dspy.OutputField(desc="brief explanation of relevance assessment")


class ContextReranking(dspy.Signature):
    """Rerank documents based on their relevance to the question."""
    
    question = dspy.InputField(desc="the user's question")
    documents = dspy.InputField(desc="list of retrieved documents")
    ranked_documents = dspy.OutputField(desc="documents reordered by relevance to the question")


class FactExtraction(dspy.Signature):
    """Extract key facts from a document that are relevant to a question."""
    
    question = dspy.InputField(desc="the question for context")
    document = dspy.InputField(desc="the source document")
    facts = dspy.OutputField(desc="relevant facts extracted from the document")


class HopQuestionGeneration(dspy.Signature):
    """Generate follow-up questions for multi-hop reasoning."""
    
    original_question = dspy.InputField(desc="the original question")
    current_context = dspy.InputField(desc="information found so far")
    next_question = dspy.OutputField(desc="follow-up question to find missing information")


class ContextExpansion(dspy.Signature):
    """Expand context by identifying what additional information is needed."""
    
    question = dspy.InputField(desc="the user's question")
    current_context = dspy.InputField(desc="information currently available")
    needed_info = dspy.OutputField(desc="description of additional information needed")
    search_query = dspy.OutputField(desc="query to find the needed information")


class ToolSelection(dspy.Signature):
    """Select the appropriate tool for answering a question."""
    
    question = dspy.InputField(desc="the user's question")
    available_tools = dspy.InputField(desc="list of available tools and their capabilities")
    selected_tool = dspy.OutputField(desc="name of the most appropriate tool")
    reasoning = dspy.OutputField(desc="explanation for tool selection")


class TaskPlanning(dspy.Signature):
    """Create a plan to answer complex questions requiring multiple steps."""
    
    question = dspy.InputField(desc="the complex question to answer")
    plan = dspy.OutputField(desc="step-by-step plan to gather information and answer the question")


class ErrorDiagnosis(dspy.Signature):
    """Diagnose why a RAG system might have failed to answer a question properly."""
    
    question = dspy.InputField(desc="the question that was asked")
    retrieved_context = dspy.InputField(desc="the context that was retrieved")
    generated_answer = dspy.InputField(desc="the answer that was generated")
    diagnosis = dspy.OutputField(desc="explanation of potential issues with retrieval or generation")
    suggestions = dspy.OutputField(desc="suggestions for improving the answer")


class ConversationContextual(dspy.Signature):
    """Answer questions taking into account conversation history."""
    
    conversation_history = dspy.InputField(desc="previous messages in the conversation")
    current_question = dspy.InputField(desc="the current question")
    context = dspy.InputField(desc="retrieved relevant documents")
    answer = dspy.OutputField(desc="contextual answer considering conversation history")


class MultiModalQA(dspy.Signature):
    """Answer questions that may involve both text and images."""
    
    question = dspy.InputField(desc="the user's question")
    text_context = dspy.InputField(desc="relevant text passages")
    image_context = dspy.InputField(desc="descriptions of relevant images", required=False)
    answer = dspy.OutputField(desc="answer incorporating both text and visual information")


class SourceCitation(dspy.Signature):
    """Generate proper citations for sources used in an answer."""
    
    answer = dspy.InputField(desc="the generated answer")
    sources = dspy.InputField(desc="the source documents used")
    cited_answer = dspy.OutputField(desc="answer with proper citations to sources")


class ConfidenceAssessment(dspy.Signature):
    """Assess confidence level in a generated answer."""
    
    question = dspy.InputField(desc="the original question")
    answer = dspy.InputField(desc="the generated answer")
    context = dspy.InputField(desc="the context used to generate the answer")
    confidence = dspy.OutputField(desc="confidence level: 'high', 'medium', 'low'")
    reasoning = dspy.OutputField(desc="explanation for the confidence assessment")


# Signature collections for different use cases
RAG_SIGNATURES = {
    "basic_qa": BasicQA,
    "contextual_qa": ContextualQA,
    "query_decomposition": QueryDecomposition,
    "answer_synthesis": AnswerSynthesis,
    "answer_verification": AnswerVerification,
    "document_relevance": DocumentRelevance,
    "fact_extraction": FactExtraction
}

MULTIHOP_SIGNATURES = {
    "hop_question": HopQuestionGeneration,
    "context_expansion": ContextExpansion,
    "query_generation": QueryGeneration,
    "answer_synthesis": AnswerSynthesis
}

REACT_SIGNATURES = {
    "tool_selection": ToolSelection,
    "task_planning": TaskPlanning,
    "contextual_qa": ContextualQA
}

ADVANCED_SIGNATURES = {
    "reranking": ContextReranking,
    "verification": AnswerVerification,
    "multimodal": MultiModalQA,
    "citation": SourceCitation,
    "confidence": ConfidenceAssessment,
    "conversation": ConversationContextual
} 