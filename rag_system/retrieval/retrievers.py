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
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.multimodal import LocalVisionModel

# BM25Retriever is no longer needed.
# class BM25Retriever: ...

from fuzzywuzzy import process

class GraphRetriever:
    def __init__(self, graph_path: str):
        self.graph = nx.read_gml(graph_path)

    def retrieve(self, query: str, k: int = 5, score_cutoff: int = 80) -> List[Dict[str, Any]]:
        print(f"\n--- Performing Graph Retrieval for query: '{query}' ---")
        
        query_parts = query.split()
        entities = []
        for part in query_parts:
            match = process.extractOne(part, self.graph.nodes(), score_cutoff=score_cutoff)
            if match and isinstance(match[0], str):
                entities.append(match[0])
        
        retrieved_docs = []
        for entity in set(entities):
            for neighbor in self.graph.neighbors(entity):
                retrieved_docs.append({
                    'chunk_id': f"graph_{entity}_{neighbor}",
                    'text': f"Entity: {entity}, Neighbor: {neighbor}",
                    'score': 1.0,
                    'metadata': {'source': 'graph'}
                })
        
        print(f"Retrieved {len(retrieved_docs)} documents from the graph.")
        return retrieved_docs[:k]

class MultiVectorRetriever:
    """
    Performs hybrid (vector + FTS) or vector-only retrieval.
    """
    def __init__(self, db_manager: LanceDBManager, text_embedder: QwenEmbedder, vision_model: LocalVisionModel = None):
        self.db_manager = db_manager
        self.text_embedder = text_embedder
        self.vision_model = vision_model

    def retrieve(self, text_query: str, table_name: str, k: int, reranker=None) -> List[Dict[str, Any]]:
        """
        Performs a search on a single LanceDB table.
        If a reranker is provided, it performs a hybrid search.
        Otherwise, it performs a standard vector search.
        """
        print(f"\n--- Performing Retrieval for query: '{text_query}' on table '{table_name}' ---")
        
        try:
            tbl = self.db_manager.get_table(table_name)
            
            # Create text embedding for the query
            text_query_embedding = self.text_embedder.create_embeddings([text_query])[0]
            
            # Choose search method based on whether reranker is provided
            if reranker:
                print("Performing hybrid search with reranking...")
                # For hybrid search, use the query parameter with query_type="hybrid"
                search_query = tbl.search(query=text_query, query_type="hybrid")
                search_query = search_query.rerank(reranker=reranker)
            else:
                print("Performing vector-only search...")
                # For vector-only search, use the vector parameter
                search_query = tbl.search(query=text_query_embedding)

            results_df = search_query.limit(k).to_df()
            
            retrieved_docs = []
            for _, row in results_df.iterrows():
                metadata = json.loads(row.get('metadata', '{}'))
                # Add top-level fields back into metadata for consistency if they don't exist
                metadata.setdefault('document_id', row.get('document_id'))
                metadata.setdefault('chunk_index', row.get('chunk_index'))
                
                retrieved_docs.append({
                    'chunk_id': row.get('chunk_id'),
                    'text': metadata.get('original_text', row.get('text')),
                    'score': row.get('_distance') or row.get('score'), # Reranker might produce 'score'
                    'document_id': row.get('document_id'),
                    'chunk_index': row.get('chunk_index'),
                    'metadata': metadata
                })

            print(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs
        
        except Exception as e:
            print(f"Could not search table '{table_name}': {e}")
            return []

if __name__ == '__main__':
    print("retrievers.py updated for LanceDB FTS Hybrid Search.")
