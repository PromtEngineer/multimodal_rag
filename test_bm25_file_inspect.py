#!/usr/bin/env python3
"""
Simple BM25 file inspection
"""

import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def inspect_bm25_file():
    bm25_path = "./index_store/bm25/rag_bm25_index.pkl"
    
    print(f"🔍 Inspecting: {bm25_path}")
    print("=" * 60)
    
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"📦 Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"📦 Dictionary keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"   '{key}': {type(value)}")
            if key == 'chunks' and isinstance(value, list):
                print(f"      Number of chunks: {len(value)}")
                if value:
                    print(f"      First chunk type: {type(value[0])}")
                    if isinstance(value[0], dict):
                        print(f"      First chunk keys: {list(value[0].keys())}")
            elif key == 'index':
                print(f"      Index type: {type(value)}")
                if hasattr(value, '__dict__'):
                    print(f"      Index attributes: {list(value.__dict__.keys())}")
                    
    elif isinstance(data, tuple) or isinstance(data, list):
        print(f"📦 Sequence length: {len(data)}")
        for i, item in enumerate(data):
            print(f"   [{i}]: {type(item)}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    inspect_bm25_file() 