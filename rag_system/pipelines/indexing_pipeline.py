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
        
        # Get batch processing configuration
        indexing_config = self.config.get("indexing", {})
        self.embedding_batch_size = indexing_config.get("embedding_batch_size", 50)
        self.enrichment_batch_size = indexing_config.get("enrichment_batch_size", 10)
        self.enable_progress_tracking = indexing_config.get("enable_progress_tracking", True)

        if storage_config.get("chunk_store_path"):
            self.chunk_store = ChunkStore(store_path=storage_config["chunk_store_path"])

        if retriever_configs.get("dense", {}).get("enabled"):
            self.lancedb_manager = LanceDBManager(db_path=storage_config["lancedb_uri"])
            self.vector_indexer = VectorIndexer(self.lancedb_manager)
            embedding_model = QwenEmbedder(
                model_name=self.config.get("embedding_model_name", "Qwen/Qwen2-7B-instruct")
            )
            self.embedding_generator = EmbeddingGenerator(
                embedding_model=embedding_model, 
                batch_size=self.embedding_batch_size
            )

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
                llm_model=self.ollama_config["generation_model"],
                batch_size=self.enrichment_batch_size
            )

    def run(self, file_paths: List[str]):
        """
        Processes and indexes documents based on the pipeline's configuration.
        """
        print(f"--- Starting indexing process for {len(file_paths)} files. ---")
        
        # Import progress tracking utilities
        from rag_system.utils.batch_processor import timer, ProgressTracker, estimate_memory_usage
        
        with timer("Complete Indexing Pipeline"):
            # Step 1: Document Processing and Chunking
            all_chunks = []
            with timer("Document Processing & Chunking"):
                file_tracker = ProgressTracker(len(file_paths), "Document Processing")
                
                for file_path in file_paths:
                    try:
                        document_id = os.path.basename(file_path)
                        print(f"Processing: {document_id}")
                        
                        pages_data = self.pdf_converter.convert_to_markdown(file_path)
                        file_chunks = []
                        
                        for markdown_text, metadata in pages_data:
                            chunks = self.chunker.chunk(markdown_text, document_id, metadata)
                            file_chunks.extend(chunks)
                        
                        all_chunks.extend(file_chunks)
                        print(f"  Generated {len(file_chunks)} chunks from {document_id}")
                        file_tracker.update(1)
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {file_path}: {e}")
                        file_tracker.update(1, errors=1)
                        continue
                
                file_tracker.finish()

            if not all_chunks:
                print("No text chunks were generated. Skipping indexing.")
                return

            print(f"\n✅ Generated {len(all_chunks)} text chunks total.")
            memory_mb = estimate_memory_usage(all_chunks)
            print(f"📊 Estimated memory usage: {memory_mb:.1f}MB")

            retriever_configs = self.config.get("retrievers", {})

            # Step 2: Save original chunks to the chunk store
            if hasattr(self, 'chunk_store'):
                with timer("Chunk Store Save"):
                    self.chunk_store.save(all_chunks)
                    print("✅ Saved chunks to chunk store")

            # Step 3: Optional Contextual Enrichment (before indexing for consistency)
            if hasattr(self, 'contextual_enricher'):
                with timer("Contextual Enrichment"):
                    enricher_config = self.config.get("contextual_enricher", {})
                    window_size = enricher_config.get("window_size", 1)
                    print(f"\n--- Starting contextual enrichment (window_size={window_size}) ---")
                    
                    # This modifies the 'text' field in each chunk dictionary
                    all_chunks = self.contextual_enricher.enrich_chunks(all_chunks, window_size=window_size)
                    print(f"✅ Enriched {len(all_chunks)} chunks with context for indexing.")

            # Step 4: Create BM25 Index from enriched chunks (for consistency with vector index)
            if hasattr(self, 'bm25_indexer'):
                with timer("BM25 Index Creation"):
                    index_name = retriever_configs.get("bm25", {}).get("index_name", "default_bm25_index")
                    print(f"\n--- Creating BM25 index from enriched chunk text: {index_name} ---")
                    self.bm25_indexer.index(index_name, all_chunks)
                    print("✅ BM25 index created successfully")

            # Step 5: Vector Embedding Generation and Indexing
            if hasattr(self, 'vector_indexer') and hasattr(self, 'embedding_generator'):
                with timer("Vector Embedding & Indexing"):
                    table_name = retriever_configs.get("dense", {}).get("lancedb_table_name", "default_text_table")
                    print(f"\n--- Generating embeddings with {self.config.get('embedding_model_name')} ---")
                    
                    embeddings = self.embedding_generator.generate(all_chunks)
                    
                    print(f"\n--- Indexing {len(embeddings)} vectors into LanceDB table: {table_name} ---")
                    self.vector_indexer.index(table_name, all_chunks, embeddings)
                    print("✅ Vector embeddings indexed successfully")
                
            # Step 6: Knowledge Graph Extraction (Optional)
            if hasattr(self, 'graph_extractor'):
                with timer("Knowledge Graph Extraction"):
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
        self._print_final_statistics(len(file_paths), len(all_chunks))
    
    def _print_final_statistics(self, num_files: int, num_chunks: int):
        """Print final indexing statistics"""
        print(f"\n📈 Final Statistics:")
        print(f"  Files processed: {num_files}")
        print(f"  Chunks generated: {num_chunks}")
        print(f"  Average chunks per file: {num_chunks/num_files:.1f}")
        
        # Component status
        components = []
        if hasattr(self, 'chunk_store'):
            components.append("✅ Chunk Store")
        if hasattr(self, 'bm25_indexer'):
            components.append("✅ BM25 Index")
        if hasattr(self, 'contextual_enricher'):
            components.append("✅ Contextual Enrichment")
        if hasattr(self, 'vector_indexer'):
            components.append("✅ Vector Index")
        if hasattr(self, 'graph_extractor'):
            components.append("✅ Knowledge Graph")
            
        print(f"  Components: {', '.join(components)}")
        print(f"  Batch sizes: Embeddings={self.embedding_batch_size}, Enrichment={self.enrichment_batch_size}")
