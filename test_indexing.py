#!/usr/bin/env python3
"""
Test script to create an index with the enhanced Agent functionality
"""

import os
import sys
import uuid
import tempfile
import shutil
from pathlib import Path

# Add rag_system to path
sys.path.append('.')

from rag_system.pipelines.indexing_pipeline import IndexingPipeline
from rag_system.utils.ollama_client import OllamaClient
from rag_system.config import OLLAMA_CONFIG

def test_indexing():
    """Create a test index with our test document"""
    print("🚀 Testing Enhanced Agent Indexing")
    print("=" * 50)
    
    # Generate unique index ID for this test
    test_index_id = f"test-agent-{uuid.uuid4().hex[:8]}"
    print(f"Creating test index: {test_index_id}")
    
    # Use existing document for testing
    test_doc_path = "rag_system/documents/invoice_1039.pdf"
    
    if not os.path.exists(test_doc_path):
        print(f"❌ Test document not found: {test_doc_path}")
        return None
    
    print(f"✅ Using existing test document: {test_doc_path}")
    
    # Configuration for indexing
    indexing_config = {
        "storage": {
            "lancedb_uri": "./lancedb",
            "doc_path": "rag_system/documents",
            "text_table_name": f"text_pages_{test_index_id}",
            "image_table_name": None,
            "bm25_path": "./index_store/bm25"
        },
        "indexing": {
            "embedding_batch_size": 10,
            "enrichment_batch_size": 5
        },
        "contextual_enricher": {
            "enabled": True,
            "window_size": 1
        }
    }
    
    try:
        # Initialize Ollama client
        ollama_client = OllamaClient()
        
        # Check if Ollama is running
        if not ollama_client.is_ollama_running():
            print("❌ Ollama is not running. Please start Ollama first.")
            print("Run: ollama serve")
            return None
            
        print("✅ Ollama is running")
        
        # Initialize indexing pipeline
        indexing_pipeline = IndexingPipeline(
            config=indexing_config,
            ollama_client=ollama_client,
            ollama_config=OLLAMA_CONFIG
        )
        
        print("📊 Starting indexing process...")
        
        # Get list of files to index
        file_paths = [test_doc_path]
        
        # Run the indexing
        result = indexing_pipeline.run(file_paths=file_paths)
        
        if result and result.get("success"):
            print("✅ Indexing completed successfully!")
            print(f"Documents processed: {result.get('documents_processed', 'unknown')}")
            print(f"Index ID: {test_index_id}")
            print(f"Table name: text_pages_{test_index_id}")
            return test_index_id
        else:
            print("❌ Indexing failed")
            print(f"Result: {result}")
            return None
            
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # No cleanup needed since we're using existing documents
        pass

if __name__ == "__main__":
    index_id = test_indexing()
    if index_id:
        print(f"\n🎉 Test index created successfully: {index_id}")
        print("You can now test the enhanced Agent with this index!")
    else:
        print("\n❌ Failed to create test index")