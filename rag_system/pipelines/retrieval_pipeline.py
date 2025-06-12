import pymupdf
from typing import List, Dict, Any, Tuple
from PIL import Image
import concurrent.futures
import time
import json

from rag_system.utils.ollama_client import OllamaClient
from rag_system.retrieval.retrievers import MultiVectorRetriever, GraphRetriever, BM25Retriever
from rag_system.retrieval.reranker import QwenReranker
from rag_system.indexing.multimodal import LocalVisionModel
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.embedders import LanceDBManager
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
        reranker_config = self.config.get("reranker", {})
        if self.reranker is None and reranker_config.get("enabled"):
            self.reranker = QwenReranker(
                model_name=reranker_config.get("model_name", "Qwen/Qwen-reranker")
            )
        return self.reranker

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
        retrieved_docs = []
        retrieval_k = self.config.get("retrieval_k", 10)

        # 🚀 OPTIMIZED: Parallel retrieval execution
        print(f"\n--- Running parallel retrieval for query: '{query}' ---")
        
        retrieval_futures = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            dense_retriever = self._get_dense_retriever()
            if dense_retriever:
                future = executor.submit(
                    dense_retriever.retrieve,
                    text_query=query,
                    text_table=self.storage_config["text_table_name"],
                    image_table=self.storage_config.get("image_table_name"),
                    k=retrieval_k
                )
                retrieval_futures['dense'] = future

            bm25_retriever = self._get_bm25_retriever()
            if bm25_retriever:
                future = executor.submit(bm25_retriever.retrieve, query, retrieval_k)
                retrieval_futures['bm25'] = future

            graph_retriever = self._get_graph_retriever()
            if graph_retriever:
                future = executor.submit(graph_retriever.retrieve, query, retrieval_k)
                retrieval_futures['graph'] = future
            
            seen_chunk_ids = set()
            for retrieval_type, future in retrieval_futures.items():
                try:
                    docs = future.result()
                    if docs:
                        print(f"✅ {retrieval_type.capitalize()} retrieval completed: {len(docs)} docs")
                        for doc in docs:
                            if doc['chunk_id'] not in seen_chunk_ids:
                                retrieved_docs.append(doc)
                                seen_chunk_ids.add(doc['chunk_id'])
                except Exception as e:
                    print(f"❌ {retrieval_type.capitalize()} retrieval failed: {e}")
        
        retrieval_time = time.time() - start_time
        print(f"🚀 Parallel retrieval completed in {retrieval_time:.2f}s - {len(retrieved_docs)} total docs")

        reranker = self._get_reranker()
        if reranker and retrieved_docs:
            print(f"\n--- Reranking {len(retrieved_docs)} documents before expansion... ---")
            reranked_docs = reranker.rerank(query, retrieved_docs, top_k=self.config.get("reranker", {}).get("top_k", 10))
        else:
            reranked_docs = retrieved_docs

        # 4. Parent-Child Context Expansion using LanceDB
        window_size = self.config.get("context_window_size", 1)
        if window_size > 0 and reranked_docs:
            print(f"\n--- Expanding context for {len(reranked_docs)} top documents (window size: {window_size})... ---")
            expanded_chunks = {}
            
            # Use a thread pool to expand context in parallel for efficiency
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
                print(f"      Rerank Score: {doc.get('rerank_score', 'N/A')}")
                print(f"      Text: \"{doc.get('text', '').strip()}\"")
        print("------------------------------------")

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
