"""
DSPy Modules Package

Contains custom DSPy modules that integrate with the existing RAG system:
- LanceDBRetriever: Custom retriever for LanceDB integration
- HybridRetriever: Multi-modal retrieval combining dense + BM25
- RAGModules: Core RAG signatures and modules
- ReActModules: DSPy-based ReAct agent implementations
"""

from .retrievers import LanceDBRetriever, HybridRetriever
from .rag_modules import BasicRAG, AdvancedRAG, MultiHopRAG
from .react_modules import DSPyReActAgent, DSPyTool
from .signatures import *

__all__ = [
    "LanceDBRetriever",
    "HybridRetriever", 
    "BasicRAG",
    "AdvancedRAG",
    "MultiHopRAG",
    "DSPyReActAgent",
    "DSPyTool"
] 