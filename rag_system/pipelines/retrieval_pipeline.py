import pymupdf
from typing import List, Dict, Any
from PIL import Image

from rag_system.utils.ollama_client import OllamaClient
from rag_system.retrieval.retrievers import MultiVectorRetriever, GraphRetriever
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
    def __init__(self, config: Dict[str, Any], ollama_client: OllamaClient, ollama_config: Dict[str, str]):
        self.config = config
        self.ollama_client = ollama_client
        self.ollama_config = ollama_config

        # --- Initialize all components ---
        db_manager = LanceDBManager(config["storage"]["lancedb_path"])
        
        text_embedder = QwenEmbedder(
            model_name=config.get("embedding_model_name", "Qwen/Qwen2-7B-instruct")
        )
        
        if config.get("vision_model_name"):
            self.vision_model = LocalVisionModel(model_name=config.get("vision_model_name"))
        else:
            self.vision_model = None

        self.retriever = MultiVectorRetriever(db_manager, text_embedder, self.vision_model)
        
        if config.get("reranker", {}).get("enabled"):
            self.reranker = QwenReranker(model_name=config.get("reranker", {}).get("model_name", "Qwen/Qwen-reranker"))

        if config.get("graph_rag", {}).get("enabled"):
            self.graph_retriever = GraphRetriever(config["graph_rag"]["graph_path"])

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
        # 1. Retrieval
        retrieved_docs = self.retriever.retrieve(
            text_query=query,
            text_table=self.config["storage"]["text_table_name"],
            image_table=self.config["storage"]["image_table_name"],
            k=self.config.get("retrieval_k", 10)
        )

        if hasattr(self, 'graph_retriever'):
            graph_docs = self.graph_retriever.retrieve(query)
            retrieved_docs.extend(graph_docs)

        # 2. Reranking
        if not hasattr(self, 'reranker') or not retrieved_docs:
            final_docs = retrieved_docs
        else:
            final_docs = self.reranker.rerank(query, retrieved_docs, top_k=self.config["reranker"].get("top_k", 3))

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
