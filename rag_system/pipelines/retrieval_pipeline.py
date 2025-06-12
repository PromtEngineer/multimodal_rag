import pymupdf
from typing import List, Dict, Any, Tuple
from PIL import Image
import concurrent.futures
import time
import json
import lancedb

from rag_system.utils.ollama_client import OllamaClient
from rag_system.retrieval.retrievers import MultiVectorRetriever, GraphRetriever
from rag_system.indexing.multimodal import LocalVisionModel
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.embedders import LanceDBManager
from rag_system.rerankers.reranker import QwenReranker
# from rag_system.indexing.chunk_store import ChunkStore

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
        
        self.retriever_configs = self.config.get("retrievers", {})
        self.storage_config = self.config["storage"]
        
        # Defer initialization to just-in-time methods
        self.db_manager = None
        self.text_embedder = None
        self.dense_retriever = None
        self.bm25_retriever = None
        self.graph_retriever = None
        self.reranker = None
        self.ai_reranker = None

    def _get_db_manager(self):
        if self.db_manager is None:
            self.db_manager = LanceDBManager(db_path=self.storage_config["lancedb_uri"])
        return self.db_manager

    def _get_text_embedder(self):
        if self.text_embedder is None:
            self.text_embedder = QwenEmbedder(
                model_name=self.config.get("embedding_model_name", "Qwen/Qwen2-7B-instruct")
            )
        return self.text_embedder

    def _get_dense_retriever(self):
        if self.dense_retriever is None and self.retriever_configs.get("dense", {}).get("enabled"):
            db_manager = self._get_db_manager()
            text_embedder = self._get_text_embedder()
            self.dense_retriever = MultiVectorRetriever(db_manager, text_embedder, vision_model=None)
        return self.dense_retriever

    def _get_bm25_retriever(self):
        if self.bm25_retriever is None and self.retriever_configs.get("bm25", {}).get("enabled"):
            try:
                print(f"🔧 Lazily initializing BM25 retriever...")
                self.bm25_retriever = BM25Retriever(
                    index_path=self.storage_config["bm25_path"],
                    index_name=self.retriever_configs["bm25"]["index_name"]
                )
                print("✅ BM25 retriever initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize BM25 retriever on demand: {e}")
                # Keep it None so we don't try again
        return self.bm25_retriever

    def _get_graph_retriever(self):
        if self.graph_retriever is None and self.retriever_configs.get("graph", {}).get("enabled"):
            self.graph_retriever = GraphRetriever(graph_path=self.storage_config["graph_path"])
        return self.graph_retriever

    def _get_reranker(self):
        """Initializes the reranker for hybrid search score fusion."""
        reranker_config = self.config.get("reranker", {})
        # This is for the LanceDB internal reranker, not the AI one.
        if self.reranker is None and reranker_config.get("type") == "linear_combination":
            rerank_weight = reranker_config.get("weight", 0.5) 
            self.reranker = lancedb.rerankers.LinearCombinationReranker(weight=rerank_weight)
            print(f"✅ Initialized LinearCombinationReranker with weight {rerank_weight}")
        return self.reranker

    def _get_ai_reranker(self):
        """Initializes a dedicated AI-based reranker."""
        reranker_config = self.config.get("reranker", {})
        if self.ai_reranker is None and reranker_config.get("enabled") and reranker_config.get("type") == "ai":
            try:
                print(f"🔧 Lazily initializing AI reranker ({reranker_config.get('model_name')})...")
                self.ai_reranker = QwenReranker(
                    model_name=reranker_config.get("model_name")
                )
                print("✅ AI reranker initialized successfully.")
            except Exception as e:
                print(f"❌ Failed to initialize AI reranker: {e}")
        return self.ai_reranker

    def _get_surrounding_chunks_lancedb(self, chunk: Dict[str, Any], window_size: int) -> List[Dict[str, Any]]:
        """
        Retrieves a window of chunks around a central chunk using LanceDB.
        """
        db_manager = self._get_db_manager()
        if not db_manager:
            return [chunk]

        # Extract identifiers needed for the query
        document_id = chunk.get("document_id")
        chunk_index = chunk.get("chunk_index")

        # If essential identifiers are missing, return the chunk itself
        if document_id is None or chunk_index is None or chunk_index == -1:
            return [chunk]

        table_name = self.config["storage"]["text_table_name"]
        try:
            tbl = db_manager.get_table(table_name)
        except Exception:
            # If the table can't be opened, we can't get surrounding chunks
            return [chunk]

        # Define the window for the search
        start_index = max(0, chunk_index - window_size)
        end_index = chunk_index + window_size
        
        # Construct the SQL filter for an efficient metadata-based search
        sql_filter = f"document_id = '{document_id}' AND chunk_index >= {start_index} AND chunk_index <= {end_index}"
        
        try:
            # Execute a filter-only search, which is very fast on indexed metadata
            results = tbl.search().where(sql_filter).to_list()
            
            # The results must be sorted by chunk_index to maintain logical order
            results.sort(key=lambda c: c['chunk_index'])

            # The 'metadata' field is a JSON string and needs to be parsed
            for res in results:
                if isinstance(res.get('metadata'), str):
                    try:
                        res['metadata'] = json.loads(res['metadata'])
                    except json.JSONDecodeError:
                        res['metadata'] = {} # Handle corrupted metadata gracefully
            return results
        except Exception:
            # If the query fails for any reason, fall back to the single chunk
            return [chunk]

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
        retrieval_k = self.config.get("retrieval_k", 10)

        print(f"\n--- Running Hybrid Search for query: '{query}' ---")
        
        # Unified retrieval using the refactored MultiVectorRetriever
        dense_retriever = self._get_dense_retriever()
        # Get the LanceDB reranker for initial score fusion
        lancedb_reranker = self._get_reranker()
        
        retrieved_docs = []
        if dense_retriever:
            retrieved_docs = dense_retriever.retrieve(
                text_query=query,
                table_name=self.storage_config["text_table_name"],
                k=retrieval_k,
                reranker=lancedb_reranker # Pass the reranker to enable hybrid search
            )
        
        retrieval_time = time.time() - start_time
        print(f"🚀 Initial retrieval completed in {retrieval_time:.2f}s - {len(retrieved_docs)} total docs")

        # --- AI Reranking Step ---
        ai_reranker = self._get_ai_reranker()
        if ai_reranker and retrieved_docs:
            print(f"\n--- Reranking top {len(retrieved_docs)} docs with AI model... ---")
            start_rerank_time = time.time()
            top_k = self.config.get("reranker", {}).get("top_k", 5)
            reranked_docs = ai_reranker.rerank(query, retrieved_docs, top_k=top_k)
            rerank_time = time.time() - start_rerank_time
            print(f"✅ Reranking completed in {rerank_time:.2f}s. Refined to {len(reranked_docs)} docs.")
        else:
            # If no AI reranker, proceed with the initially retrieved docs
            reranked_docs = retrieved_docs

        window_size = self.config.get("context_window_size", 1)
        if window_size > 0 and reranked_docs:
            print(f"\n--- Expanding context for {len(reranked_docs)} top documents (window size: {window_size})... ---")
            expanded_chunks = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_chunk = {executor.submit(self._get_surrounding_chunks_lancedb, chunk, window_size): chunk for chunk in reranked_docs}
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        surrounding_chunks = future.result()
                        for surrounding_chunk in surrounding_chunks:
                            if surrounding_chunk['chunk_id'] not in expanded_chunks:
                                expanded_chunks[surrounding_chunk['chunk_id']] = surrounding_chunk
                    except Exception as e:
                        print(f"Error expanding context for a chunk: {e}")

            final_docs = list(expanded_chunks.values())
            final_docs.sort(key=lambda c: (c.get('document_id', ''), c.get('chunk_index', 0)))
            print(f"Expanded to {len(final_docs)} unique chunks for synthesis.")
        else:
            final_docs = reranked_docs

        print("\n--- Final Documents for Synthesis ---")
        if not final_docs:
            print("No documents to synthesize.")
        else:
            for i, doc in enumerate(final_docs):
                print(f"  [{i+1}] Chunk ID: {doc.get('chunk_id')}")
                print(f"      Score: {doc.get('score', 'N/A')}")
                print(f"      Text: \"{doc.get('text', '').strip()}\"")
        print("------------------------------------")

        if not final_docs:
            return {"answer": "I could not find an answer in the documents.", "source_documents": []}
        
        context = "\n\n".join([doc['text'] for doc in final_docs])
        final_answer = self._synthesize_final_answer(query, context)
        
        return {"answer": final_answer, "source_documents": final_docs}
