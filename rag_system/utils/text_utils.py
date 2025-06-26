import re
from typing import List

def tokenize_text(text: str) -> List[str]:
    """
    Standard tokenization function used across the RAG system.
    Extracts word tokens from text using regex pattern.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of lowercase word tokens
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens 