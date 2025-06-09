import pickle
import os
from typing import List, Dict, Any, Optional

class ChunkStore:
    """
    A simple file-based store to save, load, and access all text chunks.
    This is necessary for the parent-child retrieval strategy, allowing us
    to retrieve the context window around an initially retrieved chunk.
    """
    def __init__(self, store_path: str):
        self.store_path = store_path
        self.chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self.chunks_by_doc: Dict[str, List[Dict[str, Any]]] = {}
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        self._load()

    def reload(self):
        """Forces a reload of the chunk store from disk."""
        self._load()

    def save(self, chunks: List[Dict[str, Any]]):
        """Saves a list of chunks to the store and rebuilds indices."""
        self._index_chunks(chunks)
        with open(self.store_path, "wb") as f:
            pickle.dump(chunks, f)
        print(f"Saved {len(chunks)} chunks to {self.store_path}")

    def _load(self):
        """Loads chunks from the store file if it exists."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "rb") as f:
                    chunks = pickle.load(f)
                self._index_chunks(chunks)
                print(f"Loaded and indexed {len(chunks)} chunks from {self.store_path}")
            except (pickle.UnpicklingError, EOFError) as e:
                print(f"Warning: Could not load chunk store from {self.store_path}. It may be empty or corrupted. Error: {e}")
                self.chunks_by_id = {}
                self.chunks_by_doc = {}
        else:
            print(f"Chunk store not found at {self.store_path}. A new one will be created upon saving.")

    def _index_chunks(self, chunks: List[Dict[str, Any]]):
        """Creates in-memory indexes for fast lookup."""
        self.chunks_by_id = {chunk['chunk_id']: chunk for chunk in chunks}
        
        self.chunks_by_doc = {}
        for chunk in chunks:
            doc_id = chunk.get('metadata', {}).get('document_id')
            if doc_id:
                if doc_id not in self.chunks_by_doc:
                    self.chunks_by_doc[doc_id] = []
                self.chunks_by_doc[doc_id].append(chunk)

        # Sort chunks within each document to ensure correct order
        for doc_id in self.chunks_by_doc:
            # Assumes chunk_id is in the format 'doc_id_page_chunk' or similar sortable format
            self.chunks_by_doc[doc_id].sort(key=lambda c: c['chunk_id'])
            
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single chunk by its ID."""
        return self.chunks_by_id.get(chunk_id)

    def get_surrounding_chunks(self, chunk_id: str, window_size: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieves a window of chunks around a central chunk.
        """
        target_chunk = self.get_chunk_by_id(chunk_id)
        if not target_chunk:
            return []

        doc_id = target_chunk.get('metadata', {}).get('document_id')
        if not doc_id or doc_id not in self.chunks_by_doc:
            return [target_chunk]
            
        doc_chunks = self.chunks_by_doc[doc_id]
        
        try:
            current_index = doc_chunks.index(target_chunk)
        except ValueError:
            # Fallback if chunk isn't in the list for some reason
            return [target_chunk]
            
        start_index = max(0, current_index - window_size)
        end_index = min(len(doc_chunks), current_index + window_size + 1)
        
        return doc_chunks[start_index:end_index] 