#!/usr/bin/env python3
"""
Manual BM25 Test - Verify rank-bm25 library works correctly
"""

from rank_bm25 import BM25Okapi
import re

def tokenize_text(text):
    """Same tokenization as our fixed version"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def main():
    print("🔍 MANUAL BM25 LIBRARY TEST")
    print("=" * 50)
    
    # Simple test documents
    docs = [
        "PromptX AI LLC is a consulting company",
        "DeepDyve is a research platform for scientific literature",
        "The invoice amount is $9000"
    ]
    
    print("📄 Test documents:")
    for i, doc in enumerate(docs):
        print(f"   {i}: {doc}")
    
    # Tokenize documents
    tokenized_docs = [tokenize_text(doc) for doc in docs]
    print("\n🔧 Tokenized documents:")
    for i, tokens in enumerate(tokenized_docs):
        print(f"   {i}: {tokens}")
    
    # Create BM25 index
    print("\n🏗️ Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_docs)
    
    print(f"✅ BM25 created successfully")
    print(f"   Type: {type(bm25)}")
    print(f"   Corpus size: {bm25.corpus_size}")
    print(f"   Average doc length: {bm25.avgdl}")
    print(f"   Doc freqs type: {type(bm25.doc_freqs)}")
    
    # Check doc_freqs structure
    if hasattr(bm25.doc_freqs, 'keys'):
        print(f"   Doc freqs keys: {list(bm25.doc_freqs.keys())}")
        print(f"   Total unique tokens: {len(bm25.doc_freqs)}")
    else:
        print(f"   Doc freqs structure: {bm25.doc_freqs}")
    
    # Test queries
    test_queries = [
        "promptx",
        "deepdyve", 
        "amount",
        "PromptX DeepDyve",
        "invoice amount"
    ]
    
    print("\n🔍 Testing queries:")
    for query in test_queries:
        tokenized_query = tokenize_text(query)
        print(f"\n   Query: '{query}' → {tokenized_query}")
        
        scores = bm25.get_scores(tokenized_query)
        print(f"   Scores: {scores}")
        print(f"   Max score: {max(scores) if len(scores) > 0 else 'No scores'}")
        
        if len(scores) > 0 and max(scores) > 0:
            best_idx = scores.tolist().index(max(scores))
            print(f"   Best match: Doc {best_idx}: '{docs[best_idx]}'")
        else:
            print(f"   ❌ No matches found")

if __name__ == "__main__":
    main() 