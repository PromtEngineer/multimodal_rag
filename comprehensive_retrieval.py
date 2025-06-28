#!/usr/bin/env python3
"""
Comprehensive Retrieval Testing Script
=====================================

This script provides complete control over the retrieval pipeline with extensive
parameter configuration and detailed logging.

Usage:
    python comprehensive_retrieval.py --index-name my-index --query "What is DeepSeek?"

Features:
    - Full parameter control (models, retrieval strategies, reranking, etc.)
    - Multiple retrieval modes (simple, RAG, Agent with decomposition)
    - Comprehensive logging with timing for each step
    - Performance metrics and analysis
    - Interactive mode for multiple queries
    - Export results to JSON/CSV
"""

import argparse
import json
import logging
import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import csv

# Add project root to path
sys.path.append('.')

from rag_system.agent.loop import Agent
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG, PIPELINE_CONFIGS

# Set up comprehensive logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup comprehensive logging with both console and file output."""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate default log file name if not provided
    if log_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"retrieval_{timestamp}.log"
    
    # Configure logging format
    log_format = "%(asctime)s | %(levelname)8s | %(name)20s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
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

def load_index_metadata(index_name: str) -> Optional[Dict[str, Any]]:
    """Load index metadata if available."""
    logger = logging.getLogger(__name__)
    
    metadata_file = f"index_metadata_{index_name}.json"
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded index metadata from {metadata_file}")
            logger.info(f"  Created: {metadata.get('created_at', 'unknown')}")
            logger.info(f"  Files: {len(metadata.get('files_processed', []))}")
            logger.info(f"  Table: {metadata.get('table_name', 'unknown')}")
            return metadata
        except Exception as e:
            logger.warning(f"Could not load index metadata: {e}")
    else:
        logger.warning(f"No metadata file found: {metadata_file}")
    
    return None

def check_index_exists(table_name: str, lancedb_path: str = "./lancedb") -> bool:
    """Check if the specified index table exists."""
    logger = logging.getLogger(__name__)
    
    table_path = Path(lancedb_path) / f"{table_name}.lance"
    exists = table_path.exists()
    
    logger.info(f"Index table '{table_name}': {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    if exists:
        logger.info(f"  Path: {table_path}")
    else:
        logger.warning(f"  Expected path: {table_path}")
        
        # List available tables
        if Path(lancedb_path).exists():
            available_tables = [f.name.replace('.lance', '') for f in Path(lancedb_path).iterdir() 
                              if f.is_dir() and f.name.endswith('.lance')]
            if available_tables:
                logger.info(f"Available tables: {', '.join(available_tables[:5])}")
                if len(available_tables) > 5:
                    logger.info(f"  ... and {len(available_tables) - 5} more")
            else:
                logger.warning("No LanceDB tables found")
    
    return exists

def create_retrieval_config(args, table_name: str) -> Dict[str, Any]:
    """Create retrieval pipeline configuration."""
    logger = logging.getLogger(__name__)
    
    config = {
        "storage": {
            "db_path": args.lancedb_path,
            "text_table_name": table_name,
            "image_table_name": f"image_pages_{args.index_name}" if args.enable_multimodal else None
        },
        "reranker": {
            "enabled": args.enable_reranker,
            "strategy": "rerankers-lib",
            "model_name": args.reranker_model,
            "top_percent": args.reranker_top_percent
        },
        "retrieval": {
            "retriever": "multivector",
            "embeddings": "qwen",
            "search_type": args.search_type,
            "reranker": "qwen" if args.enable_reranker else None,
            "context_expansion": args.enable_context_expansion,
            "top_k": args.top_k
        },
        "verification": {
            "enabled": args.enable_verification
        },
        "caching": {
            "enabled": args.enable_caching
        },
        "contextual_enricher": {
            "enabled": args.enable_contextual_enricher,
            "window_size": args.contextual_window_size
        },
        "query_decomposition": {
            "enabled": args.enable_query_decomposition,
            "compose_from_sub_answers": True
        },
        "embedding_model_name": args.embedding_model
    }
    
    logger.info("Retrieval Configuration:")
    logger.info(f"  Search Type: {args.search_type}")
    logger.info(f"  Top-K: {args.top_k}")
    logger.info(f"  Reranker: {args.enable_reranker}")
    logger.info(f"  Context Expansion: {args.enable_context_expansion}")
    logger.info(f"  Query Decomposition: {args.enable_query_decomposition}")
    logger.info(f"  Verification: {args.enable_verification}")
    
    return config

def create_ollama_config(args) -> Dict[str, str]:
    """Create Ollama configuration."""
    return {
        "host": args.ollama_host,
        "embedding_model": args.embedding_model,
        "generation_model": args.llm_model,
        "enrichment_model": args.enrichment_model,
        "rerank_model": args.reranker_model,
        "qwen_vl_model": args.vision_model
    }

class RetrievalTimer:
    """Context manager for timing retrieval operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"⏱️  Starting: {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.info(f"✅ Completed: {self.operation_name} ({duration:.3f}s)")
        else:
            self.logger.error(f"❌ Failed: {self.operation_name} ({duration:.3f}s)")
        
    def elapsed(self) -> float:
        return time.time() - self.start_time if self.start_time else 0.0

def test_simple_retrieval(config: Dict, ollama_config: Dict, query: str, table_name: str) -> Dict[str, Any]:
    """Test simple retrieval pipeline."""
    logger = logging.getLogger(__name__)
    
    with RetrievalTimer("Simple Retrieval Pipeline") as timer:
        try:
            # Initialize components
            with RetrievalTimer("Component Initialization"):
                ollama_client = OllamaClient(host=ollama_config["host"])
                retrieval_pipeline = RetrievalPipeline(config, ollama_client, ollama_config)
            
            # Run retrieval
            with RetrievalTimer("Document Retrieval"):
                result = retrieval_pipeline.run(
                    query=query,
                    table_name=table_name,
                    window_size=1 if config["retrieval"]["context_expansion"] else 0
                )
            
            # Process results
            answer = result.get("answer", "No answer generated")
            source_docs = result.get("source_documents", [])
            
            logger.info(f"Retrieved {len(source_docs)} source documents")
            logger.info(f"Answer length: {len(answer)} characters")
            
            return {
                "mode": "simple_retrieval",
                "success": True,
                "answer": answer,
                "source_documents": source_docs,
                "duration": timer.elapsed(),
                "metadata": {
                    "query": query,
                    "table_name": table_name,
                    "config": config
                }
            }
            
        except Exception as e:
            logger.error(f"Simple retrieval failed: {e}")
            logger.exception("Full error traceback:")
            return {
                "mode": "simple_retrieval",
                "success": False,
                "error": str(e),
                "duration": timer.elapsed()
            }

def test_agent_retrieval(config: Dict, ollama_config: Dict, query: str, table_name: str, 
                        index_name: str, enable_streaming: bool = False, 
                        enable_context_expansion: bool = True, enable_query_decomposition: bool = True,
                        enable_reranker: bool = False) -> Dict[str, Any]:
    """Test Agent-based retrieval with advanced features."""
    logger = logging.getLogger(__name__)
    
    mode_name = "Agent Retrieval (Streaming)" if enable_streaming else "Agent Retrieval"
    
    with RetrievalTimer(mode_name) as timer:
        try:
            # Initialize Agent
            with RetrievalTimer("Agent Initialization"):
                ollama_client = OllamaClient(host=ollama_config["host"])
                agent = Agent(config, ollama_client, ollama_config, index_name)
            
            # Track events for detailed logging
            events = []
            step_timings = {}
            streaming_answer = []  # Accumulate streaming tokens
            
            def event_callback(event_type: str, data: Any):
                timestamp = time.time()
                events.append({
                    "type": event_type,
                    "timestamp": timestamp,
                    "data": data
                })
                
                if event_type == "decomposition":
                    sub_queries = data.get('sub_queries', [])
                    logger.info(f"🔄 Query decomposed into {len(sub_queries)} parts:")
                    for i, sub_q in enumerate(sub_queries, 1):
                        logger.info(f"    {i}. {sub_q}")
                
                elif event_type == "sub_query_result":
                    index = data.get('index', 0)
                    sub_query = data.get('query', '')
                    answer = data.get('answer', '')
                    sources = len(data.get('source_documents', []))
                    logger.info(f"📝 Sub-query {index + 1} completed:")
                    logger.info(f"    Q: {sub_query}")
                    logger.info(f"    A: {answer[:100]}{'...' if len(answer) > 100 else ''}")
                    logger.info(f"    Sources: {sources}")
                
                elif event_type == "token":
                    # Handle streaming tokens
                    token_text = data.get('text', '')
                    streaming_answer.append(token_text)
                    print(token_text, end='', flush=True)  # Real-time token streaming
                
                elif event_type in ["analyze", "composition_start", "verification_start"]:
                    logger.info(f"🔧 {event_type.replace('_', ' ').title()}")
            
            # Run Agent
            if enable_streaming:
                # Async streaming version
                result = asyncio.run(agent.stream_agent_response_async(
                    query=query,
                    table_name=table_name,
                    session_id=f"test-session-{int(time.time())}",
                    event_callback=event_callback,
                    context_expand=enable_context_expansion,
                    query_decompose=enable_query_decomposition,
                    ai_rerank=enable_reranker
                ))
            else:
                # Synchronous version
                result = agent.run(
                    query=query,
                    table_name=table_name,
                    session_id=f"test-session-{int(time.time())}",
                    event_callback=event_callback,
                    context_expand=enable_context_expansion,
                    query_decompose=enable_query_decomposition,
                    ai_rerank=enable_reranker
                )
            
            # Process results
            if enable_streaming and streaming_answer:
                # Use accumulated streaming tokens for answer
                answer = "".join(streaming_answer)
            else:
                answer = result.get("answer", "No answer generated")
            source_docs = result.get("source_documents", [])
            
            # Analyze events
            event_summary = {}
            for event in events:
                event_type = event["type"]
                event_summary[event_type] = event_summary.get(event_type, 0) + 1
            
            logger.info(f"Retrieved {len(source_docs)} source documents")
            logger.info(f"Answer length: {len(answer)} characters")
            logger.info(f"Events captured: {len(events)}")
            logger.info(f"Event breakdown: {dict(event_summary)}")
            
            return {
                "mode": mode_name,
                "success": True,
                "answer": answer,
                "source_documents": source_docs,
                "duration": timer.elapsed(),
                "events": events,
                "event_summary": event_summary,
                "metadata": {
                    "query": query,
                    "table_name": table_name,
                    "config": config,
                    "index_name": index_name
                }
            }
            
        except Exception as e:
            logger.error(f"Agent retrieval failed: {e}")
            logger.exception("Full error traceback:")
            return {
                "mode": mode_name,
                "success": False,
                "error": str(e),
                "duration": timer.elapsed()
            }

def analyze_retrieval_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze and compare retrieval results."""
    logger = logging.getLogger(__name__)
    
    logger.info("📊 Analyzing Retrieval Results")
    logger.info("=" * 50)
    
    analysis = {
        "total_tests": len(results),
        "successful_tests": sum(1 for r in results if r.get("success", False)),
        "failed_tests": sum(1 for r in results if not r.get("success", False)),
        "modes_tested": list(set(r.get("mode", "unknown") for r in results)),
        "performance": {},
        "quality_metrics": {}
    }
    
    # Performance analysis
    for result in results:
        if result.get("success"):
            mode = result.get("mode", "unknown")
            duration = result.get("duration", 0)
            answer_length = len(result.get("answer", ""))
            source_count = len(result.get("source_documents", []))
            
            if mode not in analysis["performance"]:
                analysis["performance"][mode] = {
                    "durations": [],
                    "answer_lengths": [],
                    "source_counts": []
                }
            
            analysis["performance"][mode]["durations"].append(duration)
            analysis["performance"][mode]["answer_lengths"].append(answer_length)
            analysis["performance"][mode]["source_counts"].append(source_count)
    
    # Calculate averages
    for mode, metrics in analysis["performance"].items():
        durations = metrics["durations"]
        answer_lengths = metrics["answer_lengths"]
        source_counts = metrics["source_counts"]
        
        analysis["performance"][mode] = {
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            "avg_source_count": sum(source_counts) / len(source_counts) if source_counts else 0,
            "test_count": len(durations)
        }
    
    # Log analysis
    logger.info(f"Total Tests: {analysis['total_tests']}")
    logger.info(f"Successful: {analysis['successful_tests']}")
    logger.info(f"Failed: {analysis['failed_tests']}")
    logger.info(f"Success Rate: {(analysis['successful_tests']/analysis['total_tests']*100):.1f}%")
    
    logger.info("\nPerformance by Mode:")
    for mode, perf in analysis["performance"].items():
        logger.info(f"  {mode}:")
        logger.info(f"    Avg Duration: {perf['avg_duration']:.3f}s")
        logger.info(f"    Duration Range: {perf['min_duration']:.3f}s - {perf['max_duration']:.3f}s")
        logger.info(f"    Avg Answer Length: {perf['avg_answer_length']:.0f} chars")
        logger.info(f"    Avg Sources: {perf['avg_source_count']:.1f}")
        logger.info(f"    Tests: {perf['test_count']}")
    
    return analysis

def export_results(results: List[Dict[str, Any]], analysis: Dict[str, Any], output_format: str, 
                  output_file: Optional[str] = None) -> str:
    """Export results to JSON or CSV format."""
    logger = logging.getLogger(__name__)
    
    if output_file is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        extension = "json" if output_format == "json" else "csv"
        output_file = f"retrieval_results_{timestamp}.{extension}"
    
    try:
        if output_format == "json":
            export_data = {
                "results": results,
                "analysis": analysis,
                "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif output_format == "csv":
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    "Mode", "Success", "Duration", "Answer_Length", "Source_Count", 
                    "Query", "Error", "Timestamp"
                ])
                
                # Data rows
                for result in results:
                    writer.writerow([
                        result.get("mode", ""),
                        result.get("success", False),
                        result.get("duration", 0),
                        len(result.get("answer", "")),
                        len(result.get("source_documents", [])),
                        result.get("metadata", {}).get("query", ""),
                        result.get("error", ""),
                        time.strftime("%Y-%m-%d %H:%M:%S")
                    ])
        
        logger.info(f"Results exported to: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        return ""

def interactive_mode(config: Dict, ollama_config: Dict, table_name: str, index_name: str, args):
    """Interactive mode for multiple queries."""
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Entering Interactive Mode")
    logger.info("Type 'quit' or 'exit' to stop, 'help' for commands")
    logger.info("=" * 50)
    
    results = []
    
    try:
        while True:
            try:
                query = input("\n💬 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    print("\nCommands:")
                    print("  quit/exit/q - Exit interactive mode")
                    print("  help - Show this help")
                    print("  Any other text - Query the index")
                    continue
                elif not query:
                    print("Please enter a query or 'quit' to exit")
                    continue
                
                print(f"\n🔍 Processing query: '{query}'")
                
                # Test the configured mode
                if args.mode == "simple":
                    result = test_simple_retrieval(config, ollama_config, query, table_name)
                elif args.mode == "agent":
                    result = test_agent_retrieval(config, ollama_config, query, table_name, index_name, False,
                                                 args.enable_context_expansion, args.enable_query_decomposition, args.enable_reranker)
                elif args.mode == "streaming":
                    result = test_agent_retrieval(config, ollama_config, query, table_name, index_name, True,
                                                 args.enable_context_expansion, args.enable_query_decomposition, args.enable_reranker)
                else:  # all
                    # Run simple retrieval
                    result = test_simple_retrieval(config, ollama_config, query, table_name)
                
                results.append(result)
                
                # Display results
                if result.get("success"):
                    answer = result.get("answer", "No answer")
                    sources = len(result.get("source_documents", []))
                    duration = result.get("duration", 0)
                    
                    print(f"\n📝 Answer ({duration:.2f}s, {sources} sources):")
                    print(f"{answer}")
                    
                    if sources > 0:
                        print(f"\n📚 Source Documents:")
                        for i, doc in enumerate(result.get("source_documents", [])[:3], 1):
                            chunk_id = doc.get("chunk_id", "Unknown")
                            print(f"  {i}. {chunk_id}")
                        if sources > 3:
                            print(f"  ... and {sources - 3} more")
                else:
                    print(f"\n❌ Query failed: {result.get('error', 'Unknown error')}")
                
            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"Error: {e}")
    
    finally:
        if results:
            logger.info(f"\n🏁 Interactive session completed with {len(results)} queries")
            
            # Analyze results
            analysis = analyze_retrieval_results(results)
            
            # Export if requested
            if args.export_format:
                export_results(results, analysis, args.export_format, args.export_file)

def main():
    """Main function with comprehensive argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Comprehensive Retrieval Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query test
  python comprehensive_retrieval.py --index-name my-docs --query "What is DeepSeek?"
  
  # Advanced configuration with agent mode
  python comprehensive_retrieval.py \\
    --index-name research-papers \\
    --query "What is the training cost and parameter count?" \\
    --mode agent \\
    --enable-query-decomposition \\
    --enable-reranker \\
    --search-type hybrid \\
    --log-level DEBUG
  
  # Interactive mode
  python comprehensive_retrieval.py --index-name my-docs --interactive
  
  # Test all modes and export results
  python comprehensive_retrieval.py \\
    --index-name my-docs \\
    --query "Test query" \\
    --mode all \\
    --export-format json \\
    --export-file results.json
        """
    )
    
    # Required arguments
    parser.add_argument("--index-name", required=True, 
                       help="Name of the index to query (must match indexing script)")
    
    # Query arguments
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", help="Query string to test")
    query_group.add_argument("--interactive", action="store_true", 
                           help="Interactive mode for multiple queries")
    
    # Testing mode
    parser.add_argument("--mode", choices=["simple", "agent", "streaming", "all"], default="agent",
                       help="Retrieval mode to test (default: agent)")
    
    # Model configuration
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B",
                       help="Embedding model name")
    parser.add_argument("--llm-model", choices=["gemma3n:e4b", "qwen3:0.6b", "qwen3:8b"], default="gemma3n:e4b", 
                       help="LLM model for generation")
    parser.add_argument("--enrichment-model", default="qwen3:0.6b",
                       help="Model for enrichment")
    parser.add_argument("--vision-model", default="qwen2.5vl:7b",
                       help="Vision model for multimodal")
    parser.add_argument("--reranker-model", default="answerdotai/answerai-colbert-small-v1",
                       help="Reranker model")
    
    # Retrieval configuration  
    parser.add_argument("--search-type", choices=["vector", "bm25", "hybrid"], default="hybrid",
                       help="Search strategy (default: hybrid)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Number of documents to retrieve (default: 10)")
    
    # Feature toggles
    parser.add_argument("--enable-reranker", action="store_true",
                       help="Enable reranking")
    parser.add_argument("--reranker-top-percent", type=float, default=0.4,
                       help="Reranker top percent (default: 0.4)")
    parser.add_argument("--enable-context-expansion", action="store_true",
                       help="Enable context expansion")
    parser.add_argument("--enable-query-decomposition", action="store_true",
                       help="Enable query decomposition")
    parser.add_argument("--enable-verification", action="store_true",
                       help="Enable answer verification")
    parser.add_argument("--enable-caching", action="store_true",
                       help="Enable semantic caching")
    parser.add_argument("--enable-contextual-enricher", action="store_true",
                       help="Enable contextual enrichment")
    parser.add_argument("--contextual-window-size", type=int, default=1,
                       help="Contextual enrichment window size")
    parser.add_argument("--enable-multimodal", action="store_true",
                       help="Enable multimodal processing")
    
    # Storage configuration
    parser.add_argument("--lancedb-path", default="./lancedb",
                       help="Path to LanceDB storage")
    
    # System configuration
    parser.add_argument("--ollama-host", default="http://localhost:11434",
                       help="Ollama host URL")
    
    # Export options
    parser.add_argument("--export-format", choices=["json", "csv"],
                       help="Export results format")
    parser.add_argument("--export-file", help="Export file path")
    
    # Logging configuration
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                       help="Logging level")
    parser.add_argument("--log-file", help="Custom log file path")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    logger.info("🔍 Starting Comprehensive Retrieval Testing")
    logger.info("=" * 60)
    
    try:
        # Load index metadata
        logger.info("Step 1: Loading index metadata...")
        metadata = load_index_metadata(args.index_name)
        
        # Determine table name
        if metadata:
            table_name = metadata.get("table_name", f"text_pages_{args.index_name}")
        else:
            table_name = f"text_pages_{args.index_name}"
        
        logger.info(f"Using table name: {table_name}")
        
        # Check if index exists
        logger.info("Step 2: Checking index existence...")
        if not check_index_exists(table_name, args.lancedb_path):
            logger.error("Index not found! Please run the indexing script first.")
            sys.exit(1)
        
        # Create configuration
        logger.info("Step 3: Creating retrieval configuration...")
        config = create_retrieval_config(args, table_name)
        ollama_config = create_ollama_config(args)
        
        # Check Ollama
        logger.info("Step 4: Checking Ollama connection...")
        ollama_client = OllamaClient(host=args.ollama_host)
        if not ollama_client.is_ollama_running():
            logger.error("Ollama is not running! Please start with: ollama serve")
            sys.exit(1)
        logger.info("✅ Ollama is running")
        
        # Run tests
        if args.interactive:
            # Interactive mode
            interactive_mode(config, ollama_config, table_name, args.index_name, args)
        else:
            # Single query mode
            logger.info("Step 5: Running retrieval tests...")
            results = []
            
            if args.mode in ["simple", "all"]:
                logger.info("Testing Simple Retrieval...")
                result = test_simple_retrieval(config, ollama_config, args.query, table_name)
                results.append(result)
            
            if args.mode in ["agent", "all"]:
                logger.info("Testing Agent Retrieval...")
                result = test_agent_retrieval(config, ollama_config, args.query, table_name, args.index_name, False,
                                             args.enable_context_expansion, args.enable_query_decomposition, args.enable_reranker)
                results.append(result)
            
            if args.mode in ["streaming", "all"]:
                logger.info("Testing Streaming Agent Retrieval...")
                result = test_agent_retrieval(config, ollama_config, args.query, table_name, args.index_name, True,
                                             args.enable_context_expansion, args.enable_query_decomposition, args.enable_reranker)
                results.append(result)
            
            # Analyze results
            logger.info("Step 6: Analyzing results...")
            analysis = analyze_retrieval_results(results)
            
            # Export if requested
            if args.export_format:
                logger.info("Step 7: Exporting results...")
                export_results(results, analysis, args.export_format, args.export_file)
            
            # Display final results
            logger.info("=" * 60)
            logger.info("🎉 Testing completed!")
            
            successful_results = [r for r in results if r.get("success")]
            if successful_results:
                logger.info("📝 Sample Result:")
                sample = successful_results[0]
                answer = sample.get("answer", "No answer")
                sources = len(sample.get("source_documents", []))
                duration = sample.get("duration", 0)
                
                logger.info(f"Mode: {sample.get('mode', 'Unknown')}")
                logger.info(f"Duration: {duration:.3f}s")
                logger.info(f"Sources: {sources}")
                logger.info(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            
    except KeyboardInterrupt:
        logger.warning("\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()