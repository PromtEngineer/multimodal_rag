#!/usr/bin/env python3
"""
Detailed BM25 Investigation
Based on the debug test results, investigating specific issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
from rank_bm25 import BM25Okapi

def investigate_bm25_file():
    """Investigate the corrupted BM25 file"""
    print("🔍 INVESTIGATING BM25 FILE CORRUPTION")
    print("=" * 60)
    
    bm25_path = "./index_store/bm25/rag_bm25_index.pkl"
    
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"📦 Loaded data type: {type(data)}")
    print(f"📦 Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    
    if isinstance(data, tuple) or isinstance(data, list):
        print(f"📦 Data contents:")
        for i, item in enumerate(data):
            print(f"   [{i}] Type: {type(item)}, Length: {len(item) if hasattr(item, '__len__') else 'N/A'}")
            if hasattr(item, '__dict__'):
                print(f"       Attributes: {list(item.__dict__.keys())}")
    else:
        print(f"📦 Data content preview: {str(data)[:200]}...")

def investigate_chunk_content():
    """Investigate the actual chunk content"""
    print("\n🔍 INVESTIGATING CHUNK CONTENT")
    print("=" * 60)
    
    chunk_store_path = "./index_store/chunk_store/chunks.pkl"
    
    with open(chunk_store_path, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"📄 Number of chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n📄 Chunk {i}:")
        print(f"   Type: {type(chunk)}")
        print(f"   Keys: {list(chunk.keys()) if isinstance(chunk, dict) else 'Not a dict'}")
        
        if isinstance(chunk, dict) and 'text' in chunk:
            text = chunk['text']
            print(f"   Text length: {len(text)}")
            print(f"   Text preview: {text[:200]}...")
            
            # Test tokenization on this chunk
            tokens_old = text.split(" ")
            tokens_new = text.lower().split()
            
            print(f"   Tokens (old): {len(tokens_old)} → {tokens_old[:10]}...")
            print(f"   Tokens (new): {len(tokens_new)} → {tokens_new[:10]}...")
            
            # Check for key terms
            key_terms = ['promptx', 'deepdyve', 'amount', 'invoice', 'PromptX', 'DeepDyve']
            for term in key_terms:
                in_original = term in text
                in_lower = term.lower() in text.lower()
                print(f"   Contains '{term}': original={in_original}, lower={in_lower}")

def test_manual_bm25_creation():
    """Manually create BM25 index and test"""
    print("\n🔍 MANUAL BM25 CREATION TEST")
    print("=" * 60)
    
    # Load chunks
    chunk_store_path = "./index_store/chunk_store/chunks.pkl"
    with open(chunk_store_path, "rb") as f:
        chunks = pickle.load(f)
    
    print(f"📄 Creating BM25 index from {len(chunks)} chunks...")
    
    # Create tokenized corpus with new method
    tokenized_corpus = []
    for chunk in chunks:
        if isinstance(chunk, dict) and 'text' in chunk:
            tokens = chunk['text'].lower().split()
            tokenized_corpus.append(tokens)
            print(f"   Added {len(tokens)} tokens: {tokens[:5]}...")
    
    print(f"📊 Tokenized corpus size: {len(tokenized_corpus)}")
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"✅ BM25 index created successfully")
    print(f"   BM25 type: {type(bm25)}")
    print(f"   BM25 attributes: {dir(bm25)}")
    
    # Test queries
    test_queries = [
        "promptx",
        "deepdyve", 
        "promptx deepdyve",
        "amount",
        "invoice"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing manual BM25 with query: '{query}'")
        tokenized_query = query.lower().split()
        print(f"   Tokenized query: {tokenized_query}")
        
        scores = bm25.get_scores(tokenized_query)
        print(f"   Scores: {scores}")
        print(f"   Max score: {max(scores) if len(scores) > 0 else 'No scores'}")
        print(f"   Non-zero scores: {sum(1 for s in scores if s > 0)}")
        
        if len(scores) > 0 and max(scores) > 0:
            best_idx = scores.index(max(scores))
            best_chunk = chunks[best_idx]
            print(f"   Best match: {best_chunk.get('text', '')[:100]}...")

def main():
    print("🔍 Detailed BM25 Investigation")
    print("Investigating specific issues found in debug test\n")
    
    investigate_bm25_file()
    investigate_chunk_content()
    test_manual_bm25_creation()
    
    print("\n" + "=" * 60)
    print("🏁 Investigation completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 