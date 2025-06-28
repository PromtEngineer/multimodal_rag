import os

# Define pipeline configurations
PIPELINE_CONFIGS = {
    "default": {
        "description": "A comprehensive pipeline using a multi-vector retriever with hybrid search, reranking, and verification.",
        "storage": {
            "db_path": "lancedb",
            "text_table_name": "text_pages_default",
            "image_table_name": "image_pages"
        },
        "reranker": {
            "enabled": True,
            "strategy": "rerankers-lib",
            "model_name": "answerdotai/answerai-colbert-small-v1",
            "top_percent": 0.4
        },
        "retrieval": {
            "retriever": "multivector",
            "embeddings": "qwen",
            "search_type": "hybrid",
            "reranker": "qwen", 
            "context_expansion": True,
        },
        "graph_rag": {
            "enabled": False, 
        },
        "verification": {"enabled": True},
        "caching": {"enabled": True},
        "contextual_enricher": {
            "enabled": True,
            "window_size": 1
        },
        "indexing": {
            "embedding_batch_size": 50,
            "enrichment_batch_size": 10
        }
    },
    "fast": {
        "description": "A pipeline optimized for speed, with caching and vector search but no reranking or verification.",
        "storage": {
            "db_path": "lancedb",
            "text_table_name": "text_pages_fast",
            "image_table_name": "image_pages"
        },
        "retrieval": {
            "retriever": "multivector",
            "embeddings": "qwen",
            "search_type": "hybrid",
            "reranker": None,
            "context_expansion": False,
        },
        "verification": {"enabled": False},
        "caching": {"enabled": True},
        "indexing": {
            "embedding_batch_size": 50,
            "enrichment_batch_size": 10
        }
    },
    "react": {
        "description": "A ReAct-style agent that uses tools to answer queries.",
        "retrieval": {
            "retriever": "multivector",
            "embeddings": "qwen",
            "search_type": "hybrid",
            "reranker": "qwen", 
            "context_expansion": True,
        },
        "storage": {
            "db_path": "lancedb",
            "text_table_name": "text_pages",
            "image_table_name": "image_pages"
        },
        "react": {
            "max_iterations": 5
        },
        "contextual_enricher": {
            "enabled": True,
            "window_size": 2
        },
        "query_decomposition": {
            "enabled": True,
            "compose_from_sub_answers": True
        },
        "indexing": {
            "embedding_batch_size": 50,
            "enrichment_batch_size": 25
        }
    }
}

OLLAMA_CONFIG = {
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "embedding_model": "qwen3-embedding-0.6b",
    "generation_model": "gemma3n:e4b",
    "rerank_model": "answerdotai/answerai-colbert-small-v1",
    "enrichment_model": "qwen3:0.6b",
    "qwen_vl_model": "qwen-vl-chat"
}

# Model capability detection for thinking token support
MODEL_CAPABILITIES = {
    # Models WITH thinking token support
    "qwen": {"supports_thinking": True, "thinking_enabled_by_default": True},
    "qwen3": {"supports_thinking": True, "thinking_enabled_by_default": True},
    "qwen2.5": {"supports_thinking": True, "thinking_enabled_by_default": True},
    "deepseek": {"supports_thinking": True, "thinking_enabled_by_default": True},
    "deepthink": {"supports_thinking": True, "thinking_enabled_by_default": True},
    
    # Models WITHOUT thinking token support  
    "gemma": {"supports_thinking": False, "thinking_enabled_by_default": False},
    "gemma3n": {"supports_thinking": False, "thinking_enabled_by_default": False},
    "llama": {"supports_thinking": False, "thinking_enabled_by_default": False},
    "mistral": {"supports_thinking": False, "thinking_enabled_by_default": False},
    "phi": {"supports_thinking": False, "thinking_enabled_by_default": False},
    "codellama": {"supports_thinking": False, "thinking_enabled_by_default": False},
    "llava": {"supports_thinking": False, "thinking_enabled_by_default": False},
}

def get_model_capabilities(model_name: str) -> dict:
    """
    Get capabilities for a model based on its name prefix.
    
    Args:
        model_name: Full model name (e.g., "qwen3:8b", "gemma:7b")
        
    Returns:
        Dict with 'supports_thinking' and 'thinking_enabled_by_default' keys
    """
    if not model_name:
        return {"supports_thinking": False, "thinking_enabled_by_default": False}
    
    # Extract the base model name (before colon)
    base_model = model_name.lower().split(":")[0]
    
    # Check for exact matches first
    if base_model in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[base_model]
    
    # Check for prefix matches (for models like "qwen3-chat", "gemma2-instruct")
    for prefix, caps in MODEL_CAPABILITIES.items():
        if base_model.startswith(prefix):
            return caps
    
    # Default for unknown models - assume no thinking support for safety
    return {"supports_thinking": False, "thinking_enabled_by_default": False}

def get_thinking_setting(model_name: str, operation_type: str = "default") -> bool:
    """
    Get the appropriate thinking setting for a model and operation type.
    
    Args:
        model_name: Full model name
        operation_type: Type of operation - "fast" for quick operations like triage,
                       "reasoning" for complex tasks, "default" for auto-detection
                       
    Returns:
        bool: Whether to enable thinking tokens for this operation
    """
    caps = get_model_capabilities(model_name)
    
    if not caps["supports_thinking"]:
        return False
    
    if operation_type == "fast":
        return False  # Never use thinking for fast operations
    elif operation_type == "reasoning":
        return True   # Always use thinking for complex reasoning
    else:
        return caps["thinking_enabled_by_default"]  # Use model default
