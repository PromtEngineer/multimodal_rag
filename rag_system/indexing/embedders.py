
import lancedb
import pyarrow as pa
from typing import List, Dict, Any
import numpy as np
import os
import pickle
import json

from rag_system.indexing.representations import BM25Generator

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
            pa.field("text", pa.string()), # This is the text that was embedded
            pa.field("chunk_id", pa.string()),
            pa.field("metadata", pa.string()) # Contains original_text, summary, etc.
        ])

        data = []
        for chunk, vector in zip(chunks, embeddings):
            # Ensure original_text is in metadata if not already present
            if 'original_text' not in chunk['metadata']:
                chunk['metadata']['original_text'] = chunk['text']

            data.append({
                "vector": vector.tolist(),
                "text": chunk["text"], # This is the enriched text
                "chunk_id": chunk["chunk_id"],
                "metadata": json.dumps(chunk.get("metadata", {}))
            })

        table_names = self.db_manager.db.table_names()
        if table_name in table_names:
            tbl = self.db_manager.get_table(table_name)
            tbl.add(data, mode="overwrite") 
            print(f"Overwrote {len(data)} vectors in table '{table_name}'.")
        else:
            tbl = self.db_manager.create_table(table_name, schema=schema, mode="create")
            tbl.add(data)
            print(f"Created table '{table_name}' and indexed {len(data)} vectors.")


class BM25Indexer:
    def __init__(self, index_path: str):
        self.index_path = index_path
        os.makedirs(self.index_path, exist_ok=True)

    def index(self, index_name: str, chunks: List[Dict[str, Any]]):
        # BM25 should also operate on the enriched text
        bm25_generator = BM25Generator()
        bm25_index = bm25_generator.generate(chunks)
        
        if bm25_index:
            data_to_save = {"index": bm25_index, "chunks": chunks}
            file_path = os.path.join(self.index_path, index_name)
            with open(file_path, "wb") as f:
                pickle.dump(data_to_save, f)
            print(f"BM25 index and chunk data saved to {file_path}")

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
