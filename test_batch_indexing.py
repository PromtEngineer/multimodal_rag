#!/usr/bin/env python3
"""
Test script to demonstrate the new batch processing and progress tracking features
"""

import json
import os
import sys
import time
from typing import List, Dict, Any

# Add the rag_system to the path
sys.path.append('.')

from rag_system.pipelines.indexing_pipeline import IndexingPipeline
from rag_system.utils.ollama_client import OllamaClient
from rag_system.utils.batch_processor import BatchProcessor, ProgressTracker, timer, estimate_memory_usage

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def create_sample_chunks(num_chunks: int = 100) -> List[Dict[str, Any]]:
    """Create sample chunks for testing batch processing"""
    chunks = []
    for i in range(num_chunks):
        chunk = {
            'chunk_id': f'test_chunk_{i}',
            'text': f'This is test chunk number {i}. ' * 10,  # Make it reasonably sized
            'metadata': {
                'document_id': f'test_doc_{i // 20}',
                'chunk_number': i % 20,
                'source': f'test_document_{i // 20}.pdf'
            }
        }
        chunks.append(chunk)
    return chunks

def test_batch_processor():
    """Test the BatchProcessor utility"""
    print("=" * 60)
    print("Testing BatchProcessor Utility")
    print("=" * 60)
    
    def dummy_process_func(batch):
        """Simulate processing time"""
        time.sleep(0.1)  # Simulate processing
        return [f"processed_{item}" for item in batch]
    
    test_items = list(range(50))
    processor = BatchProcessor(batch_size=10)
    
    results = processor.process_in_batches(
        test_items, 
        dummy_process_func, 
        "Batch Processing Test"
    )
    
    print(f"\n✅ Successfully processed {len(results)} items")
    print(f"Sample results: {results[:5]}")

def test_memory_estimation():
    """Test memory estimation utility"""
    print("\n" + "=" * 60)
    print("Testing Memory Estimation")
    print("=" * 60)
    
    # Test with different chunk sizes
    chunk_sizes = [10, 50, 100, 500]
    
    for size in chunk_sizes:
        chunks = create_sample_chunks(size)
        memory_mb = estimate_memory_usage(chunks)
        avg_chunk_length = sum(len(c['text']) for c in chunks[:10]) / min(10, len(chunks))
        
        print(f"Chunks: {size:3d} | Memory: {memory_mb:6.2f}MB | Avg length: {avg_chunk_length:.0f} chars")

def test_progress_tracker():
    """Test progress tracking functionality"""
    print("\n" + "=" * 60)
    print("Testing Progress Tracker")
    print("=" * 60)
    
    total_items = 100
    tracker = ProgressTracker(total_items, "Progress Test")
    
    # Simulate processing with progress updates
    batch_size = 15
    for i in range(0, total_items, batch_size):
        # Simulate processing time
        time.sleep(0.2)
        
        current_batch_size = min(batch_size, total_items - i)
        tracker.update(current_batch_size)
        
        # Simulate occasional errors
        if i == 45:  # Simulate error in middle
            tracker.update(0, errors=3)
    
    tracker.finish()

def test_indexing_pipeline_with_batch_config():
    """Test the indexing pipeline with batch configuration"""
    print("\n" + "=" * 60)
    print("Testing Indexing Pipeline with Batch Config")
    print("=" * 60)
    
    try:
        # Load the batch configuration
        config = load_config('batch_indexing_config.json')
        print(f"✅ Loaded batch configuration")
        print(f"   Embedding batch size: {config['indexing']['embedding_batch_size']}")
        print(f"   Enrichment batch size: {config['indexing']['enrichment_batch_size']}")
        
        # Initialize Ollama client (but we won't actually use it for this test)
        ollama_client = OllamaClient()
        ollama_config = {
            "generation_model": "llama3.2:1b",
            "embedding_model": "mxbai-embed-large"
        }
        
        # Create pipeline with batch configuration
        pipeline = IndexingPipeline(config, ollama_client, ollama_config)
        
        print(f"✅ Pipeline initialized with batch processing")
        print(f"   Embedding batch size: {pipeline.embedding_batch_size}")
        print(f"   Enrichment batch size: {pipeline.enrichment_batch_size}")
        
        # Note: We don't actually run the pipeline here to avoid requiring real documents
        print("✅ Configuration validation passed")
        
    except Exception as e:
        print(f"❌ Error testing pipeline configuration: {e}")

def test_timer_utility():
    """Test the timer context manager"""
    print("\n" + "=" * 60)
    print("Testing Timer Utility")
    print("=" * 60)
    
    with timer("Sample Operation 1"):
        time.sleep(0.5)
        print("Completed some work...")
    
    with timer("Sample Operation 2"):
        time.sleep(0.2)
        for i in range(5):
            time.sleep(0.05)
            if i % 2 == 0:
                print(f"  Step {i+1}/5 completed")

def main():
    """Run all batch processing tests"""
    print("🚀 Starting Batch Processing and Progress Tracking Tests")
    print("=" * 80)
    
    try:
        # Test individual components
        test_batch_processor()
        test_memory_estimation()
        test_progress_tracker()
        test_timer_utility()
        test_indexing_pipeline_with_batch_config()
        
        print("\n" + "=" * 80)
        print("🎉 All batch processing tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 