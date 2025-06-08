
from typing import List, Dict, Any, Optional
import os
import networkx as nx

from rag_system.ingestion.pdf_converter import PDFConverter
from rag_system.ingestion.chunking import MarkdownRecursiveChunker
from rag_system.indexing.contextualizer import ContextualEnricher
from rag_system.indexing.graph_extractor import GraphExtractor
from rag_system.indexing.representations import EmbeddingGenerator, OllamaEmbedder, QwenEmbedder
from rag_system.indexing.embedders import LanceDBManager, VectorIndexer, BM25Indexer
from rag_system.utils.ollama_client import OllamaClient

class IndexingPipeline:
    """
    Orchestrates the indexing process using a live Ollama client.
    """
    def __init__(self, config: Dict[str, Any], ollama_client: OllamaClient, ollama_config: Dict[str, str]):
        self.config = config
        self.ollama_client = ollama_client
        self.ollama_config = ollama_config
        self.pdf_converter = PDFConverter()
        self.chunker = MarkdownRecursiveChunker()
        
        db_path = config["storage"]["lancedb_path"]
        self.db_manager = LanceDBManager(db_path=db_path)

        if config.get("contextual_enrichment", {}).get("enabled"):
            self.contextual_enricher = ContextualEnricher(
                llm_client=self.ollama_client, 
                llm_model=self.ollama_config["generation_model"]
            )

        if config.get("text_embedding"):
            embedding_config = config["text_embedding"]
            if embedding_config["provider"] == "huggingface":
                model = QwenEmbedder(
                    model_name=config["embedding_model_name"]
                )
            elif embedding_config["provider"] == "ollama":
                model = OllamaEmbedder(
                    client=self.ollama_client, 
                    model_name=self.ollama_config["embedding_model"]
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {embedding_config['provider']}")
            
            self.embedding_generator = EmbeddingGenerator(embedding_model=model)
            self.vector_indexer = VectorIndexer(db_manager=self.db_manager)

        if config.get("bm25"):
            bm25_path = config["storage"]["bm25_path"]
            self.bm25_indexer = BM25Indexer(index_path=bm25_path)

        if config.get("graph_rag", {}).get("enabled"):
            self.graph_extractor = GraphExtractor(
                llm_client=self.ollama_client,
                llm_model=self.ollama_config["generation_model"]
            )

    def run(self, pdf_paths: List[str], user_metadata: Optional[Dict[str, Dict[str, Any]]] = None):
        print("Starting indexing pipeline with live Ollama models...")
        all_chunks = []
        
        for pdf_path in pdf_paths:
            document_id = os.path.basename(pdf_path)
            pages_data = self.pdf_converter.convert_to_markdown(pdf_path)
            
            for markdown_text, extracted_metadata in pages_data:
                combined_metadata = {**extracted_metadata, **(user_metadata or {}).get(document_id, {})}
                
                chunks = self.chunker.chunk(markdown_text, document_id, combined_metadata)
                
                if hasattr(self, 'contextual_enricher'):
                    chunks = self.contextual_enricher.enrich_chunks(chunks)
                
                all_chunks.extend(chunks)

        if hasattr(self, 'embedding_generator') and all_chunks:
            embeddings = self.embedding_generator.generate(all_chunks)
            table_name = self.config["text_embedding"]["lancedb_table_name"]
            self.vector_indexer.index(table_name, all_chunks, embeddings)

        if hasattr(self, 'bm25_indexer') and all_chunks:
            self.bm25_indexer.index(self.config["bm25"]["index_name"], all_chunks)

        if hasattr(self, 'graph_extractor') and all_chunks:
            print("\n--- Building Knowledge Graph with Ollama ---")
            graph_data = self.graph_extractor.extract(all_chunks)
            G = nx.DiGraph()
            for entity in graph_data['entities']:
                G.add_node(entity['id'], type=entity['type'], properties=entity.get('properties', {}))
            for rel in graph_data['relationships']:
                G.add_edge(rel['source'], rel['target'], label=rel['label'])
            
            graph_path = self.config["graph_rag"]["graph_path"]
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            nx.write_gml(G, graph_path)
            print(f"Knowledge graph saved to {graph_path}")
            
        print("\nIndexing pipeline finished successfully.")
