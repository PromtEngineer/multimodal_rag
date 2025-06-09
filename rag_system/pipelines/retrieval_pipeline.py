import pymupdf
from typing import List, Dict, Any
from PIL import Image

from rag_system.utils.ollama_client import OllamaClient
from rag_system.retrieval.retrievers import MultiVectorRetriever, GraphRetriever, BM25Retriever
from rag_system.retrieval.reranker import QwenReranker
from rag_system.indexing.multimodal import LocalVisionModel
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.embedders import LanceDBManager

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
            self.bm25_retriever = BM25Retriever(
                index_path=storage_config["bm25_path"],
                index_name=retriever_configs["bm25"]["index_name"]
            )
        
        if retriever_configs.get("graph", {}).get("enabled"):
            self.graph_retriever = GraphRetriever(
                graph_path=storage_config["graph_path"]
            )
        
        if config.get("reranker", {}).get("enabled"):
            self.reranker = QwenReranker(
                model_name=config.get("reranker", {}).get("model_name", "Qwen/Qwen-reranker")
            )

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
        retrieved_docs = []
        retrieval_k = self.config.get("retrieval_k", 10)

        # 1. Retrieval (now fully modular)
        if hasattr(self, 'dense_retriever'):
            dense_docs = self.dense_retriever.retrieve(
                text_query=query,
                text_table=self.config["storage"]["text_table_name"],
                image_table=self.config["storage"]["image_table_name"],
                k=retrieval_k
            )
            retrieved_docs.extend(dense_docs)

        if hasattr(self, 'bm25_retriever'):
            bm25_docs = self.bm25_retriever.retrieve(query, k=retrieval_k)
            existing_ids = {doc['chunk_id'] for doc in retrieved_docs}
            for doc in bm25_docs:
                if doc['chunk_id'] not in existing_ids:
                    retrieved_docs.append(doc)

        if hasattr(self, 'graph_retriever'):
            graph_docs = self.graph_retriever.retrieve(query, k=retrieval_k)
            existing_ids = {doc['chunk_id'] for doc in retrieved_docs}
            for doc in graph_docs:
                if doc['chunk_id'] not in existing_ids:
                    retrieved_docs.append(doc)

        # 2. Reranking
        if hasattr(self, 'reranker') and retrieved_docs:
            final_docs = self.reranker.rerank(query, retrieved_docs, top_k=self.config["reranker"].get("top_k", 3))
        else:
            final_docs = retrieved_docs

        # 3. Final Answer Synthesis
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
