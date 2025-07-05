#!/usr/bin/env python3
"""
BM25 Test with Actual System Data
"""

from rank_bm25 import BM25Okapi
import re
import pickle

def tokenize_text(text):
    """Same tokenization as our system"""
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens

def main():
    print("🔍 BM25 TEST WITH ACTUAL SYSTEM DATA")
    print("=" * 50)
    
    # These are the actual chunks from our system (based on debug output)
    actual_chunks = [
        {
            'text': """## PromptX Al LLC
engineerprompt@gmail.com 1401 21ST STREET SUITE R SACRAMENTO,, CA 95811 +1 (205) 765-3769
## Bill to:
DeepDyve ssmith@deepdyve.com
2221 Broadway Street
Redwood, CA 94063
+ 1 (650) 562-7221
| Item Name                                         | Quantity   | Price     | Amount    |
|---------------------------------------------------|------------|-----------|-----------|
| AI Retainer                                       | 1          | $3,000.00 | $3,000.00 |
| AI Consulting services for scientific literature. |            |           |           |
|                                                   | Subtotal   |           | $3,000.00 |
|                                                   | Total      |           | $3,000.00 |
## Payment Options
## Send a Bank Transfer (ACH)
Account Number
102103174
Routing Number
211370150
Account Type
Checking
<!-- image -->
Invoice Number 1041
Invoice Date
Dec 03, 2024
Due Date
Dec 10, 2024
Amount Due
$3,000.00
$3,000.00 due Dec 10, 2024"""
        },
        {
            'text': """## PromptX Al LLC
engineerprompt@gmail.com 1401 21ST STREET SUITE R SACRAMENTO,, CA 95811 +1 (205) 765-3769
## Bill to:
DeepDyve ssmith@deepdyve.com 2221 Broadway Street Redwood, CA 94063 +1 (650) 562-7221
## Note
Sending this earlier because of my travels.
| Item Name                                         | Quantity   | Price     | Amount    |
|---------------------------------------------------|------------|-----------|-----------|
| AI Retainer                                       | 1          | $9,000.00 | $9,000.00 |
| AI Consulting services for scientific literature. |            |           |           |
|                                                   | Subtotal   |           | $9,000.00 |
|                                                   | Total      |           | $9,000.00 |
## Payment Options
## Send a Bank Transfer (ACH)
Account Number
102103174
Routing Number
211370150
Account Type
Checking
<!-- image -->
Invoice Number 1039
Invoice Date
Nov 20, 2024
Due Date
Nov 30, 2024
Amount Due
$9,000.00
$9,000.00 due Nov 30, 2024"""
        }
    ]
    
    print("📄 Testing with actual invoice data...")
    print(f"   Number of chunks: {len(actual_chunks)}")
    print(f"   Chunk 0 length: {len(actual_chunks[0]['text'])} chars")
    print(f"   Chunk 1 length: {len(actual_chunks[1]['text'])} chars")
    
    # Tokenize chunks exactly like our system
    tokenized_corpus = [tokenize_text(chunk['text']) for chunk in actual_chunks]
    
    print(f"\n🔧 Tokenized corpus:")
    for i, tokens in enumerate(tokenized_corpus):
        print(f"   Chunk {i}: {len(tokens)} tokens")
        print(f"   First 10 tokens: {tokens[:10]}")
        print(f"   Contains 'promptx': {'promptx' in tokens}")
        print(f"   Contains 'deepdyve': {'deepdyve' in tokens}")
        print(f"   Contains 'amount': {'amount' in tokens}")
    
    # Create BM25 index
    print(f"\n🏗️ Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"✅ BM25 created successfully")
    print(f"   Type: {type(bm25)}")
    print(f"   Corpus size: {bm25.corpus_size}")
    print(f"   Average doc length: {bm25.avgdl}")
    print(f"   Doc freqs type: {type(bm25.doc_freqs)}")
    print(f"   Num unique tokens: {len(bm25.doc_freqs) if hasattr(bm25.doc_freqs, '__len__') else 'Unknown'}")
    
    # Test the same queries our system is failing on
    test_queries = [
        "What is the relationship between PromptX and DeepDyve",
        "Who paid the amount",
        "What was the amount",
        "promptx",
        "deepdyve",
        "amount"
    ]
    
    print(f"\n🔍 Testing queries:")
    for query in test_queries:
        tokenized_query = tokenize_text(query)
        print(f"\n   Query: '{query}' → {tokenized_query}")
        
        scores = bm25.get_scores(tokenized_query)
        print(f"   Scores: {scores}")
        print(f"   Max score: {max(scores) if len(scores) > 0 else 'No scores'}")
        
        if len(scores) > 0 and max(scores) > 0:
            best_idx = scores.tolist().index(max(scores))
            print(f"   ✅ Best match: Chunk {best_idx} (score: {max(scores):.4f})")
        else:
            print(f"   ❌ No matches found")
    
    # Test saving and loading the index (like our system does)
    print(f"\n💾 Testing save/load functionality...")
    
    # Save index and chunks
    bm25_data = {
        'bm25': bm25,
        'chunks': actual_chunks
    }
    
    with open('test_bm25_index.pkl', 'wb') as f:
        pickle.dump(bm25_data, f)
    
    print(f"   ✅ Index saved to test_bm25_index.pkl")
    
    # Load index and test again
    with open('test_bm25_index.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    
    loaded_bm25 = loaded_data['bm25']
    loaded_chunks = loaded_data['chunks']
    
    print(f"   ✅ Index loaded successfully")
    print(f"   Loaded BM25 type: {type(loaded_bm25)}")
    print(f"   Loaded chunks: {len(loaded_chunks)}")
    
    # Test query with loaded index
    test_query = "promptx"
    tokenized_query = tokenize_text(test_query)
    loaded_scores = loaded_bm25.get_scores(tokenized_query)
    
    print(f"\n🔍 Testing with loaded index:")
    print(f"   Query: '{test_query}' → {tokenized_query}")
    print(f"   Scores: {loaded_scores}")
    print(f"   Max score: {max(loaded_scores) if len(loaded_scores) > 0 else 'No scores'}")

if __name__ == "__main__":
    main() 