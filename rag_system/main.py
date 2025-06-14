import os
import json
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# The sys.path manipulation has been removed to prevent import conflicts.
# This script should be run as a module from the project root, e.g.:
# python -m rag_system.main api

from rag_system.agent.loop import Agent
from rag_system.agent.react_agent import ReActAgent
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.multimodal import MultimodalProcessor, LocalVisionModel
from rag_system.pipelines.indexing_pipeline import IndexingPipeline
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.utils.ollama_client import OllamaClient
from rag_system.factory import get_agent
from rag_system.config import PIPELINE_CONFIGS

# Define pipeline configurations
PIPELINE_CONFIGS = {
    "indexing": {
        "storage": {
            "lancedb_uri": "./index_store/lancedb",
            "doc_path": "rag_system/documents",
            "text_table_name": "local_text_pages_v3",
            "image_table_name": None,
            "bm25_path": "./index_store/bm25"
        },
        "retrievers": {
            "dense": { 
                "enabled": True,
                "lancedb_table_name": "local_text_pages_v3"
            },
            "bm25": { 
                "enabled": True,
                "index_name": "rag_bm25_index"
            },
            "graph": { 
                "enabled": False,
                "graph_path": "./index_store/graph/knowledge_graph.gml"
            }
        },
        "contextual_enricher": {
            "enabled": True,
            "window_size": 1
        },
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "vision_model_name": None
    },
    "retrieval": {
        "storage": {
            "lancedb_uri": "./index_store/lancedb",
            "doc_path": "rag_system/documents", # Add path to docs for image retrieval
            "text_table_name": "local_text_pages_v3",
            "image_table_name": None,
        },
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "vision_model_name": None,
        "graph_rag": {
            "enabled": False,
            "graph_path": "./index_store/graph/knowledge_graph.gml"
        },
        "reranker": {
            "enabled": False, 
            "model_name": "BAAI/bge-reranker-base",
            "top_k": 3
        },
        "retrieval_k": 20
    },
    "fast": {
        "storage": {
            "lancedb_uri": "./index_store/lancedb",
            "bm25_path": "./index_store/bm25",
            "text_table_name": "local_text_pages_v3",
            "image_table_name": None,
            "graph_path": "./index_store/graph/knowledge_graph.gml"
        },
        "retrievers": {
            "dense": { 
                "enabled": True,
                "lancedb_table_name": "local_text_pages_v3"
            },
            "bm25": { 
                "enabled": False,
                "index_name": "rag_bm25_index"
            },
            "graph": { 
                "enabled": False,
                "graph_path": "./index_store/graph/knowledge_graph.gml"
            }
        },
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "vision_model_name": "Qwen/Qwen-VL-Chat",
        "reranker": {
            "enabled": False, 
            "model_name": "BAAI/bge-reranker-base",
            "top_k": 5
        },
        "query_decomposition": {
            "enabled": False,
            "max_sub_queries": 1
        },
        "retrieval_k": 20,
        "context_window_size": 0,
        "verification": {
            "enabled": False
        },
        "fusion": {
            "method": "linear",
            "bm25_weight": 0.5,
            "vec_weight": 0.5
        }
    },
    "default": {
        "storage": {
            "lancedb_uri": "./index_store/lancedb",
            "bm25_path": "./index_store/bm25",
            "text_table_name": "local_text_pages_v3",
            "image_table_name": None,
            "graph_path": "./index_store/graph/knowledge_graph.gml"
        },
        "retrievers": {
            "dense": { 
                "enabled": True,
                "lancedb_table_name": "local_text_pages_v3"
            },
            "bm25": { 
                "enabled": True,
                "index_name": "rag_bm25_index"
            },
            "graph": { 
                "enabled": False,
                "graph_path": "./index_store/graph/knowledge_graph.gml"
            }
        },
        "embedding_model_name": "BAAI/bge-small-en-v1.5",
        "vision_model_name": "Qwen/Qwen-VL-Chat",
        "reranker": {
            "enabled": True, 
            "type": "ai",
            "model_name": "BAAI/bge-reranker-base",
            "top_k": 10
        },
        "query_decomposition": {
            "enabled": True,
            "max_sub_queries": 3,
            "compose_from_sub_answers": True
        },
        "retrieval_k": 20,
        "context_window_size": 0
    },
    "bm25": {
        "enabled": True,
        "index_name": "rag_bm25_index"
    },
    "graph_rag": {
        "enabled": False, # Keep disabled for now unless specified
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
        "react": {
            "max_iterations": 5
        }
    }
}

OLLAMA_CONFIG = {
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "embedding_model": "nomic-embed-text",
    "generation_model": "qwen:7b",
    "rerank_model": "qwen:7b",
    "qwen_vl_model": "qwen-vl-chat"
}

def get_agent(mode: str = "default") -> Agent | ReActAgent:
    """
    Factory function to get an instance of the RAG agent based on the specified mode.
    """
    load_dotenv()
    
    # Initialize the Ollama client with the host from config
    llm_client = OllamaClient(host=OLLAMA_CONFIG["host"])
    
    # Get the configuration for the specified mode
    config = PIPELINE_CONFIGS.get(mode, PIPELINE_CONFIGS['default'])
    
    # Determine which agent class to instantiate
    if mode == "react":
        agent_class = ReActAgent
    else:
        agent_class = Agent
        
    agent = agent_class(
        pipeline_configs=config, 
        llm_client=llm_client, 
        ollama_config=OLLAMA_CONFIG
    )
    return agent

def run_indexing(docs_path: str, config_mode: str = "default"):
    """Runs the indexing pipeline for the specified documents."""
    print(f"📚 Starting indexing for documents in: {docs_path}")
    
    # Get the appropriate indexing pipeline from the factory
    indexing_pipeline = IndexingPipeline(PIPELINE_CONFIGS[config_mode])
    
    # Find all PDF files in the directory
    pdf_files = [os.path.join(docs_path, f) for f in os.listdir(docs_path) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF files found to index.")
        return

    # Process all documents through the pipeline
    indexing_pipeline.process_documents(pdf_files)
    print("✅ Indexing complete.")

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

    agent = Agent(PIPELINE_CONFIGS['default'], ollama_client, OLLAMA_CONFIG)
    result = agent.run(query)
    return json.dumps(result, indent=2, ensure_ascii=False)

def show_graph():
    """
    Loads and displays the knowledge graph.
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    graph_path = PIPELINE_CONFIGS["indexing"]["graph_path"]
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
    # This allows running the script from the command line to index documents.
    parser = argparse.ArgumentParser(description="Main entry point for the RAG system.")
    parser.add_argument(
        '--index',
        type=str,
        help='Path to the directory containing documents to index.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='The configuration profile to use (e.g., "default", "fast").'
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    if args.index:
        run_indexing(args.index, args.config)
    else:
        # This is where you might start a server or interactive session
        print("No action specified. Use --index to process documents.")
        # Example of how to get an agent instance
        # agent = get_agent(args.config)
        # print(f"Agent loaded with '{args.config}' config.")
