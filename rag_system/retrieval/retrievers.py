import lancedb
import pickle
import json
from typing import List, Dict, Any
import numpy as np
import networkx as nx
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.representations import OllamaEmbedder
from rag_system.indexing.multimodal import LocalVisionModel

# (BM25Retriever and GraphRetriever remain the same)
class BM25Retriever:
    def __init__(self, index_path: str, index_name: str):
        self.index_path = os.path.join(index_path, f"{index_name}.pkl")
        self.bm25 = None
        self.chunks = None
        self._load_index()

    def _load_index(self):
        """Loads the BM25 index and chunks from a pickle file."""
        if not os.path.exists(self.index_path):
            print(f"Warning: BM25 index file not found at {self.index_path}. BM25 retrieval will be skipped.")
            return
        
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["index"]
                self.chunks = data["chunks"]
            print(f"✅ BM25 index loaded successfully from {self.index_path}")
        except Exception as e:
            print(f"Error loading BM25 index: {e}")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        print(f"\n--- Performing BM25 Retrieval for query: '{query}' ---")
        import re
        def tokenize_text(text):
            # Use regex to split on whitespace and punctuation, then lowercase
            tokens = re.findall(r'\b\w+\b', text.lower())
            return tokens
        
        tokenized_query = tokenize_text(query)
        
        # 🚀 OPTIMIZED: Get scores for all documents once, then get top-k efficiently
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Create (score, index) pairs and get top-k
        scored_indices = [(score, idx) for idx, score in enumerate(doc_scores)]
        # Sort by score descending and take top k
        top_scored_indices = sorted(scored_indices, key=lambda x: x[0], reverse=True)[:k]
        
        results = []
        for score, original_index in top_scored_indices:
            chunk = self.chunks[original_index]
            results.append({
                'chunk_id': chunk.get('chunk_id'),
                'text': chunk.get('text'),
                'score': float(score),
                'metadata': chunk.get('metadata', {})
            })

        print(f"Retrieved {len(results)} documents using BM25 (optimized).")
        return results

from fuzzywuzzy import process

class GraphRetriever:
    def __init__(self, graph_path: str):
        self.graph = nx.read_gml(graph_path)

    def retrieve(self, query: str, k: int = 5, score_cutoff: int = 80) -> List[Dict[str, Any]]:
        print(f"\n--- Performing Graph Retrieval for query: '{query}' ---")
        
        query_parts = query.split()
        entities = []
        for part in query_parts:
            # Find the best match for each part of the query
            match = process.extractOne(part, self.graph.nodes(), score_cutoff=score_cutoff)
            if match and isinstance(match[0], str):
                entities.append(match[0])
        
        retrieved_docs = []
        for entity in set(entities): # Use set to avoid duplicate entities
            for neighbor in self.graph.neighbors(entity):
                retrieved_docs.append({
                    'chunk_id': f"graph_{entity}_{neighbor}",
                    'text': f"Entity: {entity}, Neighbor: {neighbor}",
                    'score': 1.0, # Placeholder score
                    'metadata': {'source': 'graph'}
                })
        
        print(f"Retrieved {len(retrieved_docs)} documents from the graph.")
        return retrieved_docs[:k]

class MultiVectorRetriever:
    """
    Performs hybrid retrieval across separate text and image vector indexes.
    """
    def __init__(self, db_manager: LanceDBManager, text_embedder: OllamaEmbedder, vision_model: LocalVisionModel = None):
        self.db_manager = db_manager
        self.text_embedder = text_embedder
        self.vision_model = vision_model

    def _search_table(self, table_name: str, query_embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        """Helper to search a single LanceDB table."""
        try:
            tbl = self.db_manager.get_table(table_name)
            results = tbl.search(query_embedding).limit(k).to_df()
            
            retrieved_docs = []
            for _, row in results.iterrows():
                metadata = json.loads(row.get('metadata', '{}'))
                retrieved_docs.append({
                    'chunk_id': row.get('chunk_id'),
                    'text': metadata.get('original_text', row.get('text')),
                    'score': row.get('_distance'),
                    'metadata': metadata
                })
            return retrieved_docs
        except Exception as e:
            print(f"Could not search table '{table_name}': {e}")
            return []

    def retrieve(self, text_query: str, text_table: str, image_table: str, k: int = 10) -> List[Dict[str, Any]]:
        print(f"\n--- Performing Text-Based Retrieval for query: '{text_query}' ---")
        
        # 1. Create Text Embedding for the Query
        text_query_embedding = self.text_embedder.create_embeddings([text_query])[0]
        
        # 2. Search the text table with the text embedding
        results = self._search_table(text_table, text_query_embedding, k)
        
        print(f"Retrieved {len(results)} documents.")
        return results

if __name__ == '__main__':
    # This test requires models to be downloaded and indexes to exist.
    # It's best tested as part of the full retrieval pipeline.
    print("retrievers.py updated with MultiVectorRetriever.")
    print("This component will be tested in the final retrieval pipeline.")
