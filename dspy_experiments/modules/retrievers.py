"""
Custom DSPy Retrievers

Integrates DSPy retrieval capabilities with the existing RAG system's
LanceDB storage, hybrid search, and multi-modal capabilities.
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
import dspy

# Simple dotdict implementation since dsp.utils.dotdict may not be available
class dotdict(dict):
    """Dict that supports dot notation access"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'dotdict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'dotdict' object has no attribute '{key}'")

# Import existing RAG system components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.representations import QwenEmbedder
from rag_system.retrieval.retrievers import MultiVectorRetriever
from rag_system.main import PIPELINE_CONFIGS, EXTERNAL_MODELS


class LanceDBRetriever(dspy.Retrieve):
    """
    Custom DSPy retriever that integrates with the existing LanceDB infrastructure
    """
    
    def __init__(
        self, 
        config_mode: str = "default",
        k: int = 10,
        embedding_model: Optional[str] = None
    ):
        super().__init__(k=k)
        
        # Get configuration from existing system with auto-detection
        self.config = PIPELINE_CONFIGS[config_mode]
        from config import get_dspy_config
        dspy_config = get_dspy_config()
        self.storage_config = dspy_config.get_storage_config(config_mode)
        
        # Initialize components using existing infrastructure  
        # Use the root lancedb path where the actual tables exist
        db_path = self.storage_config["lancedb_uri"]
        # Adjust path if we're in dspy_experiments directory
        if db_path == "./lancedb":
            import os
            if os.path.basename(os.getcwd()) == "dspy_experiments":
                db_path = "../lancedb"
        
        self.db_manager = LanceDBManager(db_path=db_path)
        
        # Use configured embedding model
        embedding_model_name = embedding_model or self.config["embedding_model_name"]
        self.embedder = QwenEmbedder(model_name=embedding_model_name)
        
        # Table names
        self.text_table = self.storage_config["text_table_name"]
        self.image_table = self.storage_config.get("image_table_name")
        
        print(f"✅ LanceDBRetriever initialized with {config_mode} mode")
        print(f"📊 Text table: {self.text_table}")
        print(f"🔧 Embedding model: {embedding_model_name}")
    
    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        """
        Retrieve relevant passages for the given query/queries
        """
        if k is None:
            k = self.k
            
        # Handle both single query and list of queries
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        all_passages = []
        
        for query in queries:
            try:
                # Generate embeddings for the query
                embeddings = self.embedder.create_embeddings([query])
                if embeddings is not None and len(embeddings) > 0:
                    query_embedding = embeddings[0]
                    # Ensure it's a proper numpy array for LanceDB
                    if hasattr(query_embedding, 'tolist'):
                        query_embedding = query_embedding.tolist()
                else:
                    query_embedding = None
                
                if query_embedding is None:
                    print(f"⚠️ Failed to generate embedding for query: {query}")
                    continue
                
                # Retrieve from LanceDB
                table = self.db_manager.get_table(self.text_table)
                
                # Ensure embedding is in the right format for LanceDB search
                if isinstance(query_embedding, list):
                    query_embedding = np.array(query_embedding)
                
                results = table.search(query_embedding).limit(k).to_list()
                
                # Convert to DSPy format
                passages = []
                for result in results:
                    # Extract text content
                    text = result.get("text", "")
                    
                    # Parse metadata if it's a JSON string
                    metadata = result.get("metadata", {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    
                    # Create passage in DSPy format
                    passage = dotdict({
                        "long_text": text,
                        "pid": result.get("chunk_id", "unknown"),
                        "score": result.get("_distance", 0.0),
                        "title": metadata.get("title", ""),
                        "text": text  # For compatibility
                    })
                    passages.append(passage)
                
                all_passages.extend(passages)
                
            except Exception as e:
                print(f"❌ Error retrieving for query '{query}': {e}")
                continue
        
        # Sort by score and limit to k
        all_passages.sort(key=lambda x: x.get("score", float('inf')))
        all_passages = all_passages[:k]
        
        return dspy.Prediction(passages=all_passages)


class HybridRetriever(dspy.Retrieve):
    """
    Advanced hybrid retriever that combines dense search, BM25, and optionally graph search
    """
    
    def __init__(
        self,
        config_mode: str = "default", 
        k: int = 10,
        dense_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        super().__init__(k=k)
        
        self.config = PIPELINE_CONFIGS[config_mode]
        self.retrieval_config = self.config["retrieval"]
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        
        # Initialize dense retriever
        self.dense_retriever = LanceDBRetriever(config_mode=config_mode, k=k*2)  # Get more for fusion
        
        # Initialize BM25 if enabled (currently disabled as BM25Retriever is not available)
        self.bm25_retriever = None
        if False:  # Temporarily disabled until BM25Retriever is restored
            if self.retrieval_config.get("bm25", {}).get("enabled"):
                try:
                    from rag_system.retrieval.retrievers import BM25Retriever
                    storage_config = self.config["storage"]
                    bm25_config = self.retrieval_config["bm25"]
                    
                    self.bm25_retriever = BM25Retriever(
                        index_path=storage_config["bm25_path"],
                        index_name=bm25_config["index_name"]
                    )
                    print("✅ BM25 retriever initialized")
                except Exception as e:
                    print(f"⚠️ BM25 retriever failed to initialize: {e}")
        
        print(f"✅ HybridRetriever initialized with dense_weight={dense_weight}, bm25_weight={bm25_weight}")
    
    def _fuse_results(self, dense_results: List, bm25_results: List, k: int) -> List[dotdict]:
        """
        Fuse dense and BM25 results using reciprocal rank fusion
        """
        # Create score dictionaries
        dense_scores = {p.pid: 1.0 / (i + 1) for i, p in enumerate(dense_results)}
        bm25_scores = {p.pid: 1.0 / (i + 1) for i, p in enumerate(bm25_results)} if bm25_results else {}
        
        # Combine all passages
        all_passages = {p.pid: p for p in dense_results}
        if bm25_results:
            for p in bm25_results:
                if p.pid not in all_passages:
                    all_passages[p.pid] = p
        
        # Calculate fusion scores
        fusion_scores = []
        for pid, passage in all_passages.items():
            dense_score = dense_scores.get(pid, 0.0)
            bm25_score = bm25_scores.get(pid, 0.0)
            
            fusion_score = (self.dense_weight * dense_score) + (self.bm25_weight * bm25_score)
            fusion_scores.append((fusion_score, passage))
        
        # Sort by fusion score and return top k
        fusion_scores.sort(key=lambda x: x[0], reverse=True)
        return [passage for _, passage in fusion_scores[:k]]
    
    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        """
        Perform hybrid retrieval combining dense and BM25 search
        """
        if k is None:
            k = self.k
            
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        
        all_fused_passages = []
        
        for query in queries:
            try:
                # Get dense results
                dense_pred = self.dense_retriever.forward(query, k=k*2)
                dense_results = dense_pred.passages
                
                # Get BM25 results if available
                bm25_results = []
                if self.bm25_retriever:
                    try:
                        bm25_raw = self.bm25_retriever.retrieve(query, k=k*2)
                        # Convert to DSPy format
                        bm25_results = [
                            dotdict({
                                "long_text": doc["text"],
                                "pid": doc.get("chunk_id", f"bm25_{i}"),
                                "score": doc.get("score", 0.0),
                                "text": doc["text"]
                            })
                            for i, doc in enumerate(bm25_raw)
                        ]
                    except Exception as e:
                        print(f"⚠️ BM25 retrieval failed: {e}")
                
                # Fuse results
                fused_passages = self._fuse_results(dense_results, bm25_results, k)
                all_fused_passages.extend(fused_passages)
                
            except Exception as e:
                print(f"❌ Error in hybrid retrieval for query '{query}': {e}")
                continue
        
        # Final deduplication and limiting
        seen_pids = set()
        final_passages = []
        for passage in all_fused_passages:
            if passage.pid not in seen_pids:
                seen_pids.add(passage.pid)
                final_passages.append(passage)
                if len(final_passages) >= k:
                    break
        
        return dspy.Prediction(passages=final_passages)


class GraphEnhancedRetriever(dspy.Retrieve):
    """
    Retriever that combines traditional search with knowledge graph information
    """
    
    def __init__(self, config_mode: str = "default", k: int = 10):
        super().__init__(k=k)
        
        self.config = PIPELINE_CONFIGS[config_mode]
        self.hybrid_retriever = HybridRetriever(config_mode=config_mode, k=k)
        
        # Initialize graph retriever if enabled
        self.graph_retriever = None
        if self.config["retrieval"].get("graph", {}).get("enabled"):
            try:
                from rag_system.retrieval.retrievers import GraphRetriever
                storage_config = self.config["storage"]
                self.graph_retriever = GraphRetriever(
                    graph_path=storage_config["graph_path"]
                )
                print("✅ Graph retriever initialized")
            except Exception as e:
                print(f"⚠️ Graph retriever failed to initialize: {e}")
    
    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        """
        Retrieve using hybrid search and optionally enhance with graph information
        """
        # Get base results from hybrid retriever
        hybrid_pred = self.hybrid_retriever.forward(query_or_queries, k=k)
        
        # If graph retriever is available, potentially enhance results
        if self.graph_retriever:
            try:
                queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
                
                # For each query, try to get graph context
                for query in queries:
                    graph_results = self.graph_retriever.search(query)
                    # TODO: Integrate graph results with hybrid results
                    # This would require more sophisticated fusion logic
                    
            except Exception as e:
                print(f"⚠️ Graph enhancement failed: {e}")
        
        return hybrid_pred


# Test function
def test_retrievers():
    """Test the custom retrievers"""
    print("🧪 Testing Custom DSPy Retrievers...")
    
    try:
        # Test LanceDB retriever
        print("\n📊 Testing LanceDBRetriever...")
        lance_retriever = LanceDBRetriever(k=5)
        result = lance_retriever("What is artificial intelligence?")
        print(f"✅ Retrieved {len(result.passages)} passages")
        if result.passages:
            print(f"📝 First passage: {result.passages[0].text[:100]}...")
        
        # Test Hybrid retriever
        print("\n🔀 Testing HybridRetriever...")
        hybrid_retriever = HybridRetriever(k=5)
        result = hybrid_retriever("machine learning algorithms")
        print(f"✅ Retrieved {len(result.passages)} passages with hybrid search")
        
        print("✅ Retriever tests completed successfully")
        
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")


if __name__ == "__main__":
    test_retrievers() 