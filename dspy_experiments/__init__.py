"""
DSPy Experiments Package

This package contains comprehensive DSPy implementations and tests for the RAG system.
It includes:
- Custom retrievers integrating with LanceDB
- DSPy-based RAG pipelines
- Multi-hop reasoning modules
- ReAct agents using DSPy
- Optimization and evaluation frameworks

All experiments are designed to compare DSPy performance against the current system.
"""

__version__ = "1.0.0"
__author__ = "RAG System DSPy Integration Team"

from . import modules
from . import pipelines  
from . import optimizers
from . import evaluations

__all__ = [
    "modules",
    "pipelines", 
    "optimizers",
    "evaluations"
] 