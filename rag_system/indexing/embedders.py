from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_system.indexing.representations import QwenEmbedder
# from rag_system.indexing.representations import BM25Generator
import lancedb
import pandas as pd
import pyarrow as pa
from typing import List, Dict, Any
import numpy as np
import os
import pickle
import json

class LanceDBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        print(f"LanceDB connection established at: {db_path}")

    def get_table(self, table_name: str):
        return self.db.open_table(table_name)

    def create_table(self, table_name: str, schema: pa.Schema, mode: str = "overwrite"):
        print(f"Creating table '{table_name}' with mode '{mode}'...")
        return self.db.create_table(table_name, schema=schema, mode=mode)

class VectorIndexer:
    """
    Handles the indexing of vector embeddings and rich metadata into LanceDB.
    The 'text' field is the content that gets embedded (which can be enriched).
    The original, clean text is stored in the metadata.
    """
    def __init__(self, db_manager: LanceDBManager):
        self.db_manager = db_manager

    def index(self, table_name: str, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must be the same.")
        if not chunks:
            print("No chunks to index.")
            return

        vector_dim = embeddings[0].shape[0]
        
        # The schema stores the text that was used for the embedding (potentially enriched)
        # and the full metadata object as a JSON string.
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("text", pa.string(), nullable=False),
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("metadata", pa.string())
        ])

        data = []
        for chunk, vector in zip(chunks, embeddings):
            # Ensure original_text is in metadata if not already present
            if 'original_text' not in chunk['metadata']:
                chunk['metadata']['original_text'] = chunk['text']

            # Extract document_id and chunk_index for top-level storage
            doc_id = chunk.get("metadata", {}).get("document_id", "unknown")
            chunk_idx = chunk.get("metadata", {}).get("chunk_index", -1)

            # Defensive check for text content to ensure it's a non-empty string
            text_content = chunk.get('text', '')
            if not text_content or not isinstance(text_content, str):
                text_content = ""

            data.append({
                "vector": vector.tolist(),
                "text": text_content,
                "chunk_id": chunk['chunk_id'],
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "metadata": json.dumps(chunk)
            })

        # Always overwrite the table to ensure the schema is up-to-date.
        # This is simpler and more robust for this project's workflow.
        print(f"Creating/Overwriting table '{table_name}'...")
        tbl = self.db_manager.create_table(table_name, schema=schema, mode="overwrite")
        tbl.add(data)
        print(f"Indexed {len(data)} vectors into table '{table_name}'.")

# BM25Indexer is no longer needed as we are moving to LanceDB's native FTS.
# class BM25Indexer:
#     ...

if __name__ == '__main__':
    print("embedders.py updated for contextual enrichment.")
    
    # This chunk has been "enriched". The 'text' field contains the context.
    enriched_chunk = {
        'chunk_id': 'doc1_0', 
        'text': 'Context: Discusses animals.\n\n---\n\nOriginal: The cat sat on the mat.', 
        'metadata': {
            'original_text': 'The cat sat on the mat.',
            'contextual_summary': 'Discusses animals.',
            'document_id': 'doc1', 
            'title': 'Pet Stories'
        }
    }
    sample_embeddings = np.random.rand(1, 128).astype('float32')

    DB_PATH = "./rag_system/index_store/lancedb"
    db_manager = LanceDBManager(db_path=DB_PATH)
    vector_indexer = VectorIndexer(db_manager=db_manager)

    vector_indexer.index(
        table_name="enriched_text_embeddings", 
        chunks=[enriched_chunk], 
        embeddings=sample_embeddings
    )
    
    try:
        tbl = db_manager.get_table("enriched_text_embeddings")
        df = tbl.limit(1).to_pandas()
        df['metadata'] = df['metadata'].apply(json.loads)
        print("\n--- Verification ---")
        print("Embedded Text:", df['text'].iloc[0])
        print("Original Text from Metadata:", df['metadata'].iloc[0]['original_text'])
    except Exception as e:
        print(f"Could not verify LanceDB table. Error: {e}")
