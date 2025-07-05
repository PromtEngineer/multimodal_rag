#!/usr/bin/env python3
"""
Independent BM25 Debug Tests
This script tests BM25 functionality in isolation to debug the retrieval issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
from rank_bm25 import BM25Okapi
from rag_system.retrieval.retrievers import BM25Retriever
from rag_system.indexing.representations import BM25Generator

def test_bm25_file_loading():
    """Test if we can load the BM25 index file directly"""
    print("=" * 60)
    print("TEST 1: BM25 File Loading")
    print("=" * 60)
    
    bm25_path = "./index_store/bm25/rag_bm25_index.pkl"
    
    if not os.path.exists(bm25_path):
        print(f"❌ BM25 file does not exist: {bm25_path}")
        return None, None
    
    try:
        with open(bm25_path, "rb") as f:
            bm25_index, chunks = pickle.load(f)
        
        print(f"✅ Successfully loaded BM25 index")
        print(f"   - BM25 index type: {type(bm25_index)}")
        print(f"   - Number of chunks: {len(chunks)}")
        print(f"   - BM25 corpus size: {len(bm25_index.corpus_size) if hasattr(bm25_index, 'corpus_size') else 'N/A'}")
        
        # Show first chunk
        if chunks:
            print(f"   - First chunk preview: {chunks[0].get('text', '')[:100]}...")
            print(f"   - First chunk keys: {list(chunks[0].keys())}")
        
        return bm25_index, chunks
        
    except Exception as e:
        print(f"❌ Error loading BM25 file: {e}")
        return None, None

def test_bm25_tokenization(chunks):
    """Test the tokenization used during indexing vs querying"""
    print("\n" + "=" * 60)
    print("TEST 2: Tokenization Analysis")
    print("=" * 60)
    
    if not chunks:
        print("❌ No chunks to test tokenization")
        return
    
    # Test original tokenization (what was used before fix)
    print("🔍 Original tokenization (before fix):")
    original_tokenized = [chunk['text'].split(" ") for chunk in chunks]
    for i, tokens in enumerate(original_tokenized[:2]):
        print(f"   Chunk {i}: {tokens[:10]}...")
    
    # Test new tokenization (after fix)
    print("\n🔍 New tokenization (after fix):")
    new_tokenized = [chunk['text'].lower().split() for chunk in chunks]
    for i, tokens in enumerate(new_tokenized[:2]):
        print(f"   Chunk {i}: {tokens[:10]}...")
    
    # Test query tokenization
    test_queries = [
        "What is the relationship between PromptX and DeepDyve?",
        "Who paid the amount?",
        "What was the amount?"
    ]
    
    print("\n🔍 Query tokenization:")
    for query in test_queries:
        tokenized = query.lower().split()
        print(f"   '{query}' → {tokenized}")

def test_bm25_retriever_class():
    """Test the BM25Retriever class directly"""
    print("\n" + "=" * 60)
    print("TEST 3: BM25Retriever Class Test")
    print("=" * 60)
    
    try:
        retriever = BM25Retriever(
            index_path="./index_store/bm25",
            index_name="rag_bm25_index"
        )
        
        print(f"✅ BM25Retriever initialized successfully")
        print(f"   - Has BM25 index: {retriever.bm25 is not None}")
        print(f"   - Has chunks: {retriever.chunks is not None}")
        print(f"   - Number of chunks: {len(retriever.chunks) if retriever.chunks else 0}")
        
        return retriever
        
    except Exception as e:
        print(f"❌ Error initializing BM25Retriever: {e}")
        return None

def test_bm25_queries(retriever):
    """Test specific queries that are failing"""
    print("\n" + "=" * 60)
    print("TEST 4: Query Testing")
    print("=" * 60)
    
    if not retriever:
        print("❌ No retriever to test")
        return
    
    # Test queries
    test_queries = [
        # Simple working queries (from earlier logs)
        "what is the invoice amount?",
        "promptx",
        "deepdyve",
        
        # Failing sub-queries from decomposition
        "What is the relationship between PromptX and DeepDyve?",
        "Who paid the amount?",
        "What was the amount?",
        
        # Variations
        "PromptX",
        "DeepDyve",
        "amount",
        "relationship"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        try:
            results = retriever.retrieve(query, k=5)
            print(f"   ✅ Retrieved {len(results)} documents")
            
            # Show first result if available
            if results:
                first_result = results[0]
                print(f"   📄 First result preview: {first_result.get('text', '')[:100]}...")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_raw_bm25_search(bm25_index, chunks):
    """Test BM25 search directly without the retriever wrapper"""
    print("\n" + "=" * 60)
    print("TEST 5: Raw BM25 Search")
    print("=" * 60)
    
    if not bm25_index or not chunks:
        print("❌ No BM25 index or chunks to test")
        return
    
    test_queries = [
        "What is the relationship between PromptX and DeepDyve?",
        "promptx deepdyve",
        "promptx",
        "amount"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Raw BM25 search: '{query}'")
        
        # Tokenize query the same way as in retriever
        tokenized_query = query.lower().split()
        print(f"   📝 Tokenized: {tokenized_query}")
        
        try:
            # Get BM25 scores
            scores = bm25_index.get_scores(tokenized_query)
            print(f"   📊 Score range: {min(scores):.4f} to {max(scores):.4f}")
            print(f"   📊 Non-zero scores: {sum(1 for s in scores if s > 0)}")
            
            # Get top documents
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
            print(f"   🏆 Top 5 indices: {top_indices}")
            print(f"   🏆 Top 5 scores: {[scores[i] for i in top_indices]}")
            
            # Show top document content
            if top_indices and scores[top_indices[0]] > 0:
                top_chunk = chunks[top_indices[0]]
                print(f"   📄 Top document: {top_chunk.get('text', '')[:150]}...")
            
        except Exception as e:
            print(f"   ❌ Error in raw search: {e}")

def test_rebuild_bm25():
    """Test rebuilding BM25 index from scratch with debug info"""
    print("\n" + "=" * 60)
    print("TEST 6: Rebuild BM25 Index")
    print("=" * 60)
    
    # Load chunks from chunk store
    chunk_store_path = "./index_store/chunk_store/chunks.pkl"
    
    if not os.path.exists(chunk_store_path):
        print(f"❌ Chunk store does not exist: {chunk_store_path}")
        return
    
    try:
        with open(chunk_store_path, "rb") as f:
            chunks = pickle.load(f)
        
        print(f"✅ Loaded {len(chunks)} chunks from chunk store")
        
        # Test BM25Generator
        generator = BM25Generator()
        
        print(f"🔨 Rebuilding BM25 index...")
        bm25_index = generator.generate(chunks)
        
        if bm25_index:
            print(f"✅ Successfully rebuilt BM25 index")
            
            # Test a simple query
            test_query = "promptx deepdyve"
            tokenized = test_query.lower().split()
            scores = bm25_index.get_scores(tokenized)
            
            print(f"🔍 Test search for '{test_query}':")
            print(f"   📊 Non-zero scores: {sum(1 for s in scores if s > 0)}")
            print(f"   📊 Max score: {max(scores)}")
        else:
            print(f"❌ Failed to rebuild BM25 index")
            
    except Exception as e:
        print(f"❌ Error rebuilding BM25: {e}")

def main():
    print("🔍 BM25 Debug Test Suite")
    print("This will test BM25 functionality step by step\n")
    
    # Test 1: Load BM25 file
    bm25_index, chunks = test_bm25_file_loading()
    
    # Test 2: Analyze tokenization
    test_bm25_tokenization(chunks)
    
    # Test 3: Test BM25Retriever class
    retriever = test_bm25_retriever_class()
    
    # Test 4: Test specific queries
    test_bm25_queries(retriever)
    
    # Test 5: Raw BM25 search
    test_raw_bm25_search(bm25_index, chunks)
    
    # Test 6: Rebuild index
    test_rebuild_bm25()
    
    print("\n" + "=" * 60)
    print("🏁 Debug tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 