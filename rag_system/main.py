import os
import json
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# The sys.path manipulation has been removed to prevent import conflicts.
# This script should be run as a module from the project root, e.g.:
# python -m rag_system.main api

from rag_system.agent.loop import Agent
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.multimodal import MultimodalProcessor, LocalVisionModel
from rag_system.pipelines.indexing_pipeline import IndexingPipeline
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.utils.ollama_client import OllamaClient

# --- Configuration ---
# This defines the models and storage locations for our advanced RAG system.
# The embedding and reranker models are now loaded locally via Hugging Face.
# Ensure you have run:
# ollama pull llama3            (for text generation)
# ollama pull qwen2.5vl:7b       (for vision-language tasks)
OLLAMA_CONFIG = {
    "host": "http://localhost:11434",
    "generation_model": "qwen2.5vl:7b",
    "vlm_model": "qwen2.5vl:7b" # Vision-Language Model for synthesis
}

PIPELINE_CONFIGS = {
    "indexing": {
        "storage": {
            "lancedb_path": "./index_store/lancedb",
            "text_table_name": "local_text_pages_v2",
            "image_table_name": None,
        },
        "text_embedding": {
            "provider": "huggingface",
            "lancedb_table_name": "local_text_pages_v2"
        },
        "embedding_model_name": "Qwen/Qwen3-Embedding-0.6B",
        "vision_model_name": None,
        "graph_rag": {
            "enabled": False,
            "graph_path": "./index_store/graph/knowledge_graph.gml"
        }
    },
    "retrieval": {
        "storage": {
            "lancedb_path": "./index_store/lancedb",
            "doc_path": "rag_system/documents", # Add path to docs for image retrieval
            "text_table_name": "local_text_pages_v2",
            "image_table_name": None,
        },
        "embedding_model_name": "Qwen/Qwen3-Embedding-0.6B",
        "vision_model_name": None,
        "graph_rag": {
            "enabled": False,
            "graph_path": "./index_store/graph/knowledge_graph.gml"
        },
        "reranker": {
            "enabled": False, 
            "model_name": "Qwen/Qwen3-Reranker-0.6B",
            "top_k": 3
        },
        "retrieval_k": 10
    },
    "default": {
        "storage": {
            "lancedb_path": "./index_store/lancedb",
            "doc_path": "rag_system/documents", # Add path to docs for image retrieval
            "text_table_name": "local_text_pages_v2",
            "image_table_name": None,
        },
        "embedding_model_name": "Qwen/Qwen3-Embedding-0.6B",
        "vision_model_name": None,
        "graph_rag": {
            "enabled": False,
            "graph_path": "./index_store/graph/knowledge_graph.gml"
        },
        "reranker": {
            "enabled": False, 
            "model_name": "Qwen/Qwen3-Reranker-0.6B",
            "top_k": 3
        },
        "retrieval_k": 10
    }
}

def run_indexing(file_paths: list = None):
    print("\n--- Running Multimodal Indexing Pipeline ---")
    config = PIPELINE_CONFIGS["indexing"]
    
    try:
        ollama_client = OllamaClient(OLLAMA_CONFIG["host"])
    except ConnectionError as e:
        print(e)
        return

    pipeline = IndexingPipeline(config, ollama_client, OLLAMA_CONFIG)
    
    pdf_files = []
    if file_paths:
        pdf_files = [f for f in file_paths if f.endswith('.pdf')]
        if not pdf_files:
            print("Warning: No PDF files found in the provided file paths.")
            return
    else:
        # Default behavior: Get all PDF files from the documents directory
        doc_dir = "rag_system/documents"
        if os.path.exists(doc_dir):
            pdf_files = [os.path.join(doc_dir, f) for f in os.listdir(doc_dir) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"Warning: No PDF files found in the default directory '{doc_dir}'.")
            return

    pipeline.run(pdf_files)
    print("\n--- Indexing Complete ---")

def run_chat(query: str):
    """
    Runs the agentic RAG pipeline for a given query.
    Returns the result as a JSON string.
    """
    try:
        ollama_client = OllamaClient(OLLAMA_CONFIG["host"])
    except ConnectionError as e:
        print(e)
        return json.dumps({"error": str(e)}, indent=2)

    agent = Agent(PIPELINE_CONFIGS, ollama_client, OLLAMA_CONFIG)
    result = agent.run(query)
    return json.dumps(result, indent=2, ensure_ascii=False)

def show_graph():
    """
    Loads and displays the knowledge graph.
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    graph_path = PIPELINE_CONFIGS["indexing"]["graph_rag"]["graph_path"]
    if not os.path.exists(graph_path):
        print("Knowledge graph not found. Please run the 'index' command first.")
        return

    G = nx.read_gml(graph_path)
    print("--- Knowledge Graph ---")
    print("Nodes:", G.nodes(data=True))
    print("Edges:", G.edges(data=True))
    print("---------------------")

    # Optional: Visualize the graph
    try:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold")
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Knowledge Graph Visualization")
        plt.show()
    except Exception as e:
        print(f"\nCould not visualize the graph. Matplotlib might not be installed or configured for your environment.")
        print(f"Error: {e}")

def run_api_server():
    """Starts the advanced RAG API server."""
    from rag_system.api_server import start_server
    start_server()

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [index|chat|show_graph|api] [query]")
        return

    command = sys.argv[1]
    if command == "index":
        # Allow passing file paths from the command line
        files = sys.argv[2:] if len(sys.argv) > 2 else None
        run_indexing(files)
    elif command == "chat":
        if len(sys.argv) < 3:
            print("Usage: python main.py chat <query>")
            return
        query = " ".join(sys.argv[2:])
        # 🆕 Print the result for command-line usage
        print(run_chat(query))
    elif command == "show_graph":
        show_graph()
    elif command == "api":
        run_api_server()
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
