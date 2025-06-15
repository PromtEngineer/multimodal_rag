import os

# Define pipeline configurations
PIPELINE_CONFIGS = {
    "default": {
        "description": "A comprehensive pipeline using a multi-vector retriever with hybrid search, reranking, and verification.",
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
    },
    "fast": {
        "description": "A pipeline optimized for speed, with caching and vector search but no reranking or verification.",
        "retrieval": {
            "retriever": "multivector",
            "embeddings": "qwen",
            "search_type": "hybrid",
            "reranker": None,
            "context_expansion": False,
        },
        "verification": {"enabled": False},
        "caching": {"enabled": True},
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
    "embedding_model": "nomic-embed-text",
    "generation_model": "qwen3:8b",
    "rerank_model": "answerdotai/answerai-colbert-small-v1",
    "enrichment_model": "qwen3:0.6b",
    "qwen_vl_model": "qwen-vl-chat"
}
