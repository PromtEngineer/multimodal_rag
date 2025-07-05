#!/usr/bin/env python3
"""
Final BM25 tokenization diagnosis
"""

import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def diagnose_bm25_tokenization():
    print("🔍 FINAL BM25 TOKENIZATION DIAGNOSIS")
    print("=" * 60)
    
    # Load the BM25 index
    bm25_path = "./index_store/bm25/rag_bm25_index.pkl"
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    
    bm25_index = data['index']
    chunks = data['chunks']
    
    print(f"📊 Loaded BM25 index with {len(chunks)} chunks")
    
    # Inspect the internal BM25 tokenization
    print(f"\n🔍 BM25 Index Internal Analysis:")
    print(f"   Corpus size: {bm25_index.corpus_size}")
    print(f"   Average document length: {bm25_index.avgdl}")
    print(f"   Document frequencies type: {type(bm25_index.doc_freqs)}")
    
    if hasattr(bm25_index.doc_freqs, 'keys'):
        doc_freq_keys = list(bm25_index.doc_freqs.keys())[:10]
        print(f"   Document frequencies keys (first 10): {doc_freq_keys}")
        all_tokens = set(bm25_index.doc_freqs.keys())
    else:
        print(f"   Document frequencies content: {bm25_index.doc_freqs}")
        all_tokens = set()
    
    print(f"   Total unique tokens in index: {len(all_tokens)}")
    
    # Look for specific terms
    search_terms = ['promptx', 'deepdyve', 'amount', 'invoice', 'PromptX', 'DeepDyve']
    print(f"\n🔍 Token presence analysis:")
    for term in search_terms:
        in_index = term in all_tokens
        in_lower = term.lower() in all_tokens
        print(f"   '{term}': in_index={in_index}, lower_in_index={in_lower}")
    
    # Show some actual tokens from the index
    print(f"\n📝 Sample tokens from index (first 20):")
    sample_tokens = list(all_tokens)[:20]
    print(f"   {sample_tokens}")
    
    # Test the chunk tokenization process
    print(f"\n🔍 Chunk tokenization analysis:")
    for i, chunk in enumerate(chunks):
        text = chunk['text']
        tokens = text.lower().split()
        print(f"\n   Chunk {i}:")
        print(f"      Length: {len(tokens)} tokens")
        print(f"      First 10 tokens: {tokens[:10]}")
        print(f"      Contains 'promptx': {'promptx' in tokens}")
        print(f"      Contains 'deepdyve': {'deepdyve' in tokens}")
        print(f"      Contains 'amount': {'amount' in tokens}")
        
        # Check if these tokens are in the BM25 index
        for term in ['promptx', 'deepdyve', 'amount']:
            if term in tokens:
                in_bm25 = term in all_tokens
                print(f"      '{term}' in chunk → in BM25 index: {in_bm25}")
    
    # Test a manual query
    print(f"\n🔍 Manual query test:")
    test_query = "promptx"
    tokenized_query = test_query.lower().split()
    print(f"   Query: '{test_query}'")
    print(f"   Tokenized: {tokenized_query}")
    
    scores = bm25_index.get_scores(tokenized_query)
    print(f"   Scores: {scores}")
    print(f"   Max score: {max(scores)}")
    
    # Debug the scoring process
    for token in tokenized_query:
        if all_tokens and hasattr(bm25_index, 'doc_freqs') and hasattr(bm25_index.doc_freqs, 'get'):
            if token in bm25_index.doc_freqs:
                print(f"   Token '{token}' found in {bm25_index.doc_freqs[token]} documents")
            else:
                print(f"   Token '{token}' NOT FOUND in index")
        else:
            print(f"   Cannot check token '{token}' - doc_freqs structure unknown")

if __name__ == "__main__":
    diagnose_bm25_tokenization() 