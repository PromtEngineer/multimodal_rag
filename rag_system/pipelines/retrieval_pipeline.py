import pymupdf
from typing import List, Dict, Any, Tuple
from PIL import Image
import concurrent.futures
import time

from rag_system.utils.ollama_client import OllamaClient
from rag_system.retrieval.retrievers import MultiVectorRetriever, GraphRetriever, BM25Retriever
from rag_system.retrieval.reranker import QwenReranker
from rag_system.indexing.multimodal import LocalVisionModel
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.chunk_store import ChunkStore

import os
from PIL import Image

class RetrievalPipeline:
    """
    Orchestrates the state-of-the-art multimodal RAG pipeline.
    """
    def __init__(self, config: Dict[str, Any], ollama_client: OllamaClient, ollama_config: Dict[str, Any]):
        self.config = config
        self.ollama_config = ollama_config
        self.ollama_client = ollama_client
        
        retriever_configs = self.config.get("retrievers", {})
        storage_config = self.config["storage"]
        
        if storage_config.get("chunk_store_path"):
            self.chunk_store = ChunkStore(store_path=storage_config["chunk_store_path"])
        else:
            self.chunk_store = None
        
        if retriever_configs.get("dense", {}).get("enabled"):
            db_manager = LanceDBManager(db_path=storage_config["lancedb_uri"])
            text_embedder = QwenEmbedder(
                model_name=self.config.get("embedding_model_name", "Qwen/Qwen2-7B-instruct")
            )
            # Vision model is not needed for text-only dense retrieval.
            self.dense_retriever = MultiVectorRetriever(db_manager, text_embedder, vision_model=None)
        else:
            self.dense_retriever = None

        if retriever_configs.get("bm25", {}).get("enabled"):
            print(f"🔧 Initializing BM25 retriever with path: {storage_config['bm25_path']}, index: {retriever_configs['bm25']['index_name']}")
            try:
                self.bm25_retriever = BM25Retriever(
                    index_path=storage_config["bm25_path"],
                    index_name=retriever_configs["bm25"]["index_name"]
                )
                print("✅ BM25 retriever initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize BM25 retriever: {e}")
                self.bm25_retriever = None
        else:
            print(f"❌ BM25 retriever disabled in config: {retriever_configs.get('bm25', {})}")
            self.bm25_retriever = None
        
        if retriever_configs.get("graph", {}).get("enabled"):
            self.graph_retriever = GraphRetriever(
                graph_path=storage_config["graph_path"]
            )
        
        reranker_config = self.config.get("reranker", {})
        if reranker_config.get("enabled"):
            self.reranker = QwenReranker(
                model_name=reranker_config.get("model_name", "Qwen/Qwen-reranker")
            )
        else:
            self.reranker = None

    def _synthesize_final_answer(self, query: str, facts: str) -> str:
        """Uses a text LLM to synthesize a final answer from extracted facts."""
        prompt = f"""
You are a helpful assistant. Synthesize a final, comprehensive answer from the following verified facts.
If the facts are empty, state that you could not find an answer in the documents.

Verified Facts:
---
{facts}
---

Query: "{query}"

Final Answer:
"""
        response = self.ollama_client.generate_completion(
            model=self.ollama_config["generation_model"],
            prompt=prompt
        )
        return response.get('response', 'Failed to generate a final answer.')

    def run(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        retrieved_docs = []
        retrieval_k = self.config.get("retrieval_k", 10)

        # 🚀 OPTIMIZED: Parallel retrieval execution
        print(f"\n--- Running parallel retrieval for query: '{query}' ---")
        
        retrieval_futures = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all retrieval tasks in parallel
            if hasattr(self, 'dense_retriever'):
                future = executor.submit(
                    self.dense_retriever.retrieve,
                    text_query=query,
                    text_table=self.config["storage"]["text_table_name"],
                    image_table=self.config["storage"]["image_table_name"],
                    k=retrieval_k
                )
                retrieval_futures['dense'] = future

            if hasattr(self, 'bm25_retriever') and self.bm25_retriever is not None:
                future = executor.submit(self.bm25_retriever.retrieve, query, retrieval_k)
                retrieval_futures['bm25'] = future

            if hasattr(self, 'graph_retriever'):
                future = executor.submit(self.graph_retriever.retrieve, query, retrieval_k)
                retrieval_futures['graph'] = future
            
            # Collect results as they complete
            seen_chunk_ids = set()
            for retrieval_type, future in retrieval_futures.items():
                try:
                    docs = future.result()
                    print(f"✅ {retrieval_type.capitalize()} retrieval completed: {len(docs)} docs")
                    
                    # Add unique documents
                    for doc in docs:
                        if doc['chunk_id'] not in seen_chunk_ids:
                            retrieved_docs.append(doc)
                            seen_chunk_ids.add(doc['chunk_id'])
                except Exception as e:
                    print(f"❌ {retrieval_type.capitalize()} retrieval failed: {e}")
        
        retrieval_time = time.time() - start_time
        print(f"🚀 Parallel retrieval completed in {retrieval_time:.2f}s - {len(retrieved_docs)} total docs")

        # 2. Parent-Child Context Expansion
        if not self.chunk_store:
            print("Chunk store not initialized, skipping context expansion.")
            return retrieved_docs
        
        # Reload the chunk store to ensure it's up-to-date with the latest index
        self.chunk_store.reload()

        unique_chunks_by_id = {chunk['chunk_id']: chunk for chunk in retrieved_docs}
        expanded_chunks = {}
        window_size = self.config.get("context_window_size", 1)
        if window_size > 0:
            print(f"\n--- Expanding context with window size: {window_size}... ---")
            for chunk in unique_chunks_by_id.values():
                surrounding_chunks = self.chunk_store.get_surrounding_chunks(chunk['chunk_id'], window_size=window_size)
                for surrounding_chunk in surrounding_chunks:
                    # Use a dictionary to automatically handle duplicates
                    expanded_chunks[surrounding_chunk['chunk_id']] = surrounding_chunk
            
            # The new set of documents is the unique, expanded set
            retrieved_docs = list(expanded_chunks.values())
            print(f"Expanded to {len(retrieved_docs)} unique chunks.")

        # 3. Reranking
        if hasattr(self, 'reranker') and retrieved_docs:
            print(f"\n--- Reranking {len(retrieved_docs)} documents... ---")
            final_docs = self.reranker.rerank(query, retrieved_docs, top_k=self.config.get("reranker", {}).get("top_k", 3))
        else:
            final_docs = retrieved_docs

        print("\n--- Final Documents for Synthesis ---")
        if not final_docs:
            print("No documents to synthesize.")
        else:
            for i, doc in enumerate(final_docs):
                print(f"  [{i+1}] Chunk ID: {doc.get('chunk_id')}")
                print(f"      Score: {doc.get('score', 'N/A')}")
                print(f"      Rerank Score: {doc.get('rerank_score', 'N/A')}")
                print(f"      Text: \"{doc.get('text', '').strip()}\"")
        print("------------------------------------")

        # 4. Final Answer Synthesis
        if not final_docs:
            return {
                "answer": "I could not find an answer in the documents.",
                "source_documents": []
            }
        
        context = "\n\n".join([doc['text'] for doc in final_docs])
        final_answer = self._synthesize_final_answer(query, context)
        
        return {
            "answer": final_answer,
            "source_documents": final_docs
        }
