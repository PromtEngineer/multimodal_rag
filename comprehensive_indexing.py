#!/usr/bin/env python3
"""
Comprehensive PDF Indexing Script
=================================

This script provides complete control over the indexing pipeline with extensive
parameter configuration and detailed logging.

Usage:
    python comprehensive_indexing.py --folder /path/to/pdfs --index-name my-index

Features:
    - Full parameter control (embedding models, LLMs, chunk sizes, etc.)
    - Late chunking vs standard chunking
    - Multiple embedding model options
    - Comprehensive logging with timing
    - Progress tracking
    - Error handling and recovery
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
sys.path.append('.')

from rag_system.pipelines.indexing_pipeline import IndexingPipeline
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG

# Set up comprehensive logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging with both console and file output."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate default log file name if not provided
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"indexing_{timestamp}.log"
    
    # Configure logging format
    log_format = "%(asctime)s | %(levelname)8s | %(name)20s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always DEBUG for file
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    
    return logger

def validate_pdf_folder(folder_path: str) -> List[str]:
    """Validate folder and return list of PDF files."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all PDF files
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in folder: {folder_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
    for pdf_file in pdf_files:
        logger.debug(f"  - {pdf_file}")
    
    return pdf_files

def check_ollama_models(required_models: List[str]) -> Dict[str, bool]:
    """Check if required Ollama models are available."""
    logger = logging.getLogger(__name__)
    
    ollama_client = OllamaClient()
    
    if not ollama_client.is_ollama_running():
        logger.error("Ollama is not running!")
        logger.info("Please start Ollama with: ollama serve")
        sys.exit(1)
    
    available_models = ollama_client.list_models()
    logger.info(f"Ollama is running with {len(available_models)} models")
    
    model_status = {}
    for model in required_models:
        is_available = model in available_models
        model_status[model] = is_available
        status = "✅" if is_available else "❌"
        logger.info(f"  {status} {model}")
        
        if not is_available:
            logger.warning(f"Model '{model}' not found. Install with: ollama pull {model}")
    
    return model_status

def create_indexing_config(args) -> Dict[str, Any]:
    """Create comprehensive indexing configuration from arguments."""
    logger = logging.getLogger(__name__)
    
    # Generate unique index ID
    index_id = args.index_name if args.index_name else f"idx-{uuid.uuid4().hex[:8]}"
    
    config = {
        "storage": {
            "lancedb_uri": args.lancedb_path,
            "doc_path": "indexed_documents",  # Where to store processed docs
            "text_table_name": f"text_pages_{index_id}",
            "image_table_name": f"image_pages_{index_id}" if args.enable_multimodal else None,
            "bm25_path": args.bm25_path
        },
        "indexing": {
            "embedding_batch_size": args.embedding_batch_size,
            "enrichment_batch_size": args.enrichment_batch_size,
            "enable_multimodal": args.enable_multimodal,
            "late_chunking": args.late_chunking,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap
        },
        "contextual_enricher": {
            "enabled": args.enable_contextual_enricher,
            "window_size": args.contextual_window_size,
            "model_name": args.enrichment_model
        },
        "embedding_model_name": args.embedding_model,
        "chunker_type": args.chunker_type,
        "reranker": {
            "enabled": args.enable_reranker,
            "model_name": args.reranker_model,
            "top_k": args.reranker_top_k
        }
    }
    
    logger.info("Indexing Configuration:")
    logger.info(f"  Index ID: {index_id}")
    logger.info(f"  Text Table: {config['storage']['text_table_name']}")
    logger.info(f"  Embedding Model: {args.embedding_model}")
    logger.info(f"  LLM Model: {args.llm_model}")
    logger.info(f"  Chunker Type: {args.chunker_type}")
    logger.info(f"  Late Chunking: {args.late_chunking}")
    logger.info(f"  Chunk Size: {args.chunk_size}")
    logger.info(f"  Multimodal: {args.enable_multimodal}")
    logger.info(f"  Contextual Enricher: {args.enable_contextual_enricher}")
    logger.info(f"  Reranker: {args.enable_reranker}")
    
    return config, index_id

def create_ollama_config(args) -> Dict[str, str]:
    """Create Ollama configuration from arguments."""
    return {
        "host": args.ollama_host,
        "embedding_model": args.embedding_model,
        "generation_model": args.llm_model,
        "enrichment_model": args.enrichment_model,
        "rerank_model": args.reranker_model,
        "qwen_vl_model": args.vision_model
    }

def run_indexing_pipeline(config: Dict, ollama_config: Dict, pdf_files: List[str], index_id: str) -> Dict[str, Any]:
    """Run the indexing pipeline with comprehensive logging."""
    logger = logging.getLogger(__name__)
    
    total_start_time = time.time()
    
    try:
        # Initialize Ollama client
        logger.info("Initializing Ollama client...")
        init_start = time.time()
        ollama_client = OllamaClient(host=ollama_config["host"])
        logger.info(f"Ollama client initialized in {time.time() - init_start:.2f}s")
        
        # Initialize indexing pipeline
        logger.info("Initializing indexing pipeline...")
        pipeline_start = time.time()
        indexing_pipeline = IndexingPipeline(
            config=config,
            ollama_client=ollama_client,
            ollama_config=ollama_config
        )
        logger.info(f"Indexing pipeline initialized in {time.time() - pipeline_start:.2f}s")
        
        # Run indexing
        logger.info(f"Starting indexing of {len(pdf_files)} files...")
        indexing_start = time.time()
        
        result = indexing_pipeline.run(file_paths=pdf_files)
        
        indexing_duration = time.time() - indexing_start
        total_duration = time.time() - total_start_time
        
        # Log results
        if result and result.get("success"):
            logger.info("✅ Indexing completed successfully!")
            logger.info(f"  Total Duration: {total_duration:.2f}s")
            logger.info(f"  Indexing Duration: {indexing_duration:.2f}s")
            logger.info(f"  Documents Processed: {result.get('documents_processed', 'unknown')}")
            logger.info(f"  Chunks Created: {result.get('chunks_created', 'unknown')}")
            logger.info(f"  Index ID: {index_id}")
            logger.info(f"  Table Name: {config['storage']['text_table_name']}")
            
            # Save index metadata
            metadata = {
                "success": True,
                "index_id": index_id,
                "table_name": config["storage"]["text_table_name"],
                "config": config,
                "ollama_config": ollama_config,
                "files_processed": pdf_files,
                "result": result,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": total_duration
            }
            
            metadata_file = f"index_metadata_{index_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Index metadata saved to: {metadata_file}")
            
            return metadata
            
        else:
            logger.error("❌ Indexing failed!")
            logger.error(f"Result: {result}")
            return {"success": False, "error": "Indexing pipeline failed", "result": result}
            
    except Exception as e:
        logger.error(f"❌ Error during indexing: {e}")
        logger.exception("Full error traceback:")
        return {"success": False, "error": str(e)}

def main():
    """Main function with comprehensive argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive PDF Indexing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic indexing
  python comprehensive_indexing.py --folder /path/to/pdfs --index-name my-docs
  
  # Advanced configuration
  python comprehensive_indexing.py \\
    --folder /path/to/pdfs \\
    --index-name research-papers \\
    --embedding-model "Qwen/Qwen3-Embedding-0.6B" \\
    --llm-model "gemma3n:e4b" \\
    --chunker-type "docling" \\
    --late-chunking \\
    --chunk-size 1024 \\
    --enable-contextual-enricher \\
    --enable-reranker \\
    --log-level DEBUG
        """
    )
    
    # Required arguments
    parser.add_argument("--folder", required=True, help="Path to folder containing PDF files")
    parser.add_argument("--index-name", required=True, help="Name for the index (will be used in table names)")
    
    # Model configuration
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B", 
                       help="Embedding model name (default: Qwen/Qwen3-Embedding-0.6B)")
    parser.add_argument("--llm-model", default="gemma3n:e4b", 
                       help="LLM model for enrichment (default: gemma3n:e4b)")
    parser.add_argument("--enrichment-model", default="qwen3:0.6b", 
                       help="Model for contextual enrichment (default: qwen3:0.6b)")
    parser.add_argument("--vision-model", default="qwen2.5vl:7b", 
                       help="Vision model for multimodal processing (default: qwen2.5vl:7b)")
    parser.add_argument("--reranker-model", default="answerdotai/answerai-colbert-small-v1", 
                       help="Reranker model (default: answerdotai/answerai-colbert-small-v1)")
    
    # Chunking configuration
    parser.add_argument("--chunker-type", choices=["docling", "simple", "late"], default="docling",
                       help="Type of chunker to use (default: docling)")
    parser.add_argument("--late-chunking", action="store_true", 
                       help="Enable late chunking (embed first, then chunk)")
    parser.add_argument("--chunk-size", type=int, default=512, 
                       help="Chunk size in tokens (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=50, 
                       help="Chunk overlap in tokens (default: 50)")
    
    # Processing configuration
    parser.add_argument("--embedding-batch-size", type=int, default=20, 
                       help="Batch size for embedding generation (default: 10)")
    parser.add_argument("--enrichment-batch-size", type=int, default=20, 
                       help="Batch size for contextual enrichment (default: 5)")
    
    # Feature toggles
    parser.add_argument("--enable-multimodal", action="store_true", 
                       help="Enable multimodal processing (images, tables)")
    parser.add_argument("--enable-contextual-enricher", action="store_true", 
                       help="Enable contextual enrichment")
    parser.add_argument("--contextual-window-size", type=int, default=5, 
                       help="Window size for contextual enrichment (default: 1)")
    parser.add_argument("--enable-reranker", action="store_true", 
                       help="Enable reranking during indexing")
    parser.add_argument("--reranker-top-k", type=int, default=10, 
                       help="Top-K for reranker (default: 10)")
    
    # Storage configuration
    parser.add_argument("--lancedb-path", default="./lancedb", 
                       help="Path to LanceDB storage (default: ./lancedb)")
    parser.add_argument("--bm25-path", default="./index_store/bm25", 
                       help="Path to BM25 index storage (default: ./index_store/bm25)")
    
    # System configuration
    parser.add_argument("--ollama-host", default="http://localhost:11434", 
                       help="Ollama host URL (default: http://localhost:11434)")
    
    # Logging configuration
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                       help="Logging level (default: INFO)")
    parser.add_argument("--log-file", help="Custom log file path (default: auto-generated)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("🚀 Starting Comprehensive PDF Indexing")
    logger.info("=" * 60)
    
    try:
        # Validate input folder and find PDFs
        logger.info("Step 1: Validating input folder...")
        pdf_files = validate_pdf_folder(args.folder)
        
        # Check required Ollama models
        logger.info("Step 2: Checking Ollama models...")
        required_models = [args.llm_model, args.enrichment_model]
        if args.enable_multimodal:
            required_models.append(args.vision_model)
        
        model_status = check_ollama_models(required_models)
        missing_models = [model for model, available in model_status.items() if not available]
        
        if missing_models:
            logger.error(f"Missing required models: {missing_models}")
            logger.info("Please install missing models and retry.")
            sys.exit(1)
        
        # Create configuration
        logger.info("Step 3: Creating indexing configuration...")
        config, index_id = create_indexing_config(args)
        ollama_config = create_ollama_config(args)
        
        # Run indexing pipeline
        logger.info("Step 4: Running indexing pipeline...")
        result = run_indexing_pipeline(config, ollama_config, pdf_files, index_id)
        
        # Final summary
        logger.info("=" * 60)
        if result.get("success", False):
            logger.info("🎉 Indexing completed successfully!")
            logger.info(f"Index ID: {index_id}")
            logger.info(f"Table Name: {config['storage']['text_table_name']}")
            logger.info("You can now use the retrieval script to test queries.")
        else:
            logger.error("❌ Indexing failed!")
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\n🛑 Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()