from typing import List, Dict, Any
import os
import networkx as nx
from rag_system.ingestion.pdf_converter import PDFConverter
from rag_system.ingestion.chunking import MarkdownRecursiveChunker
from rag_system.indexing.representations import QwenEmbedder, EmbeddingGenerator
from rag_system.indexing.embedders import LanceDBManager, VectorIndexer, BM25Indexer
from rag_system.indexing.graph_extractor import GraphExtractor
from rag_system.utils.ollama_client import OllamaClient
from rag_system.indexing.contextualizer import ContextualEnricher
from rag_system.indexing.chunk_store import ChunkStore

class IndexingPipeline:
    def __init__(self, config: Dict[str, Any], ollama_client: OllamaClient, ollama_config: Dict[str, str]):
        self.config = config
        self.ollama_client = ollama_client
        self.ollama_config = ollama_config
        self.pdf_converter = PDFConverter()
        self.chunker = MarkdownRecursiveChunker()

        retriever_configs = self.config.get("retrievers", {})
        storage_config = self.config["storage"]

        if storage_config.get("chunk_store_path"):
            self.chunk_store = ChunkStore(store_path=storage_config["chunk_store_path"])

        if retriever_configs.get("dense", {}).get("enabled"):
            self.lancedb_manager = LanceDBManager(db_path=storage_config["lancedb_uri"])
            self.vector_indexer = VectorIndexer(self.lancedb_manager)
            embedding_model = QwenEmbedder(
                model_name=self.config.get("embedding_model_name", "Qwen/Qwen2-7B-instruct")
            )
            self.embedding_generator = EmbeddingGenerator(embedding_model=embedding_model)

        if retriever_configs.get("bm25", {}).get("enabled"):
            self.bm25_indexer = BM25Indexer(index_path=storage_config["bm25_path"])
        
        if retriever_configs.get("graph", {}).get("enabled"):
            self.graph_extractor = GraphExtractor(
                llm_client=self.ollama_client,
                llm_model=self.ollama_config["generation_model"]
            )

        if self.config.get("contextual_enricher", {}).get("enabled"):
            self.contextual_enricher = ContextualEnricher(
                llm_client=self.ollama_client,
                llm_model=self.ollama_config["generation_model"]
            )

    def run(self, file_paths: List[str]):
        """
        Processes and indexes documents based on the pipeline's configuration.
        """
        print(f"--- Starting indexing process for {len(file_paths)} files. ---")
        
        all_chunks = []
        for file_path in file_paths:
            document_id = os.path.basename(file_path)
            pages_data = self.pdf_converter.convert_to_markdown(file_path)
            for markdown_text, metadata in pages_data:
                chunks = self.chunker.chunk(markdown_text, document_id, metadata)
                all_chunks.extend(chunks)

        if not all_chunks:
            print("No text chunks were generated. Skipping indexing.")
            return

        print(f"\n✅ Generated {len(all_chunks)} text chunks.")

        retriever_configs = self.config.get("retrievers", {})

        # --- Save original chunks to the chunk store ---
        if hasattr(self, 'chunk_store'):
            self.chunk_store.save(all_chunks)

        # --- Create BM25 Index from original chunks FIRST ---
        if hasattr(self, 'bm25_indexer'):
            index_name = retriever_configs.get("bm25", {}).get("index_name", "default_bm25_index")
            print(f"\n--- Creating BM25 index from original chunk text: {index_name} ---")
            self.bm25_indexer.index(index_name, all_chunks)

        # --- Optional: Contextual Enrichment for vector-based retrieval---
        if hasattr(self, 'contextual_enricher'):
            enricher_config = self.config.get("contextual_enricher", {})
            window_size = enricher_config.get("window_size", 1)
            # This modifies the 'text' field in each chunk dictionary
            all_chunks = self.contextual_enricher.enrich_chunks(all_chunks, window_size=window_size)
            print(f"✅ Enriched {len(all_chunks)} chunks with context for vectorization.")

        if hasattr(self, 'vector_indexer') and hasattr(self, 'embedding_generator'):
            table_name = retriever_configs.get("dense", {}).get("lancedb_table_name", "default_text_table")
            print(f"\n--- Generating embeddings with {self.config.get('embedding_model_name')} ---")
            embeddings = self.embedding_generator.generate(all_chunks)
            print(f"\n--- Indexing {len(embeddings)} vectors into LanceDB table: {table_name} ---")
            self.vector_indexer.index(table_name, all_chunks, embeddings)
        
        # BM25 indexing is now done before enrichment.
            
        if hasattr(self, 'graph_extractor'):
            graph_path = retriever_configs.get("graph", {}).get("graph_path", "./index_store/graph/default_graph.gml")
            print(f"\n--- Building and saving knowledge graph to: {graph_path} ---")
            
            graph_data = self.graph_extractor.extract(all_chunks)
            G = nx.DiGraph()
            for entity in graph_data['entities']:
                G.add_node(entity['id'], type=entity.get('type', 'Unknown'), properties=entity.get('properties', {}))
            for rel in graph_data['relationships']:
                G.add_edge(rel['source'], rel['target'], label=rel['label'])
            
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            nx.write_gml(G, graph_path)
            print(f"✅ Knowledge graph saved successfully.")
            
        print("\n--- ✅ Indexing Complete ---")
