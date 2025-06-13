from typing import List, Dict, Any, Protocol
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

# We keep the protocol to ensure a consistent interface
class EmbeddingModel(Protocol):
    def create_embeddings(self, texts: List[str]) -> np.ndarray: ...

# --- New Ollama Embedder ---
class QwenEmbedder(EmbeddingModel):
    """
    An embedding model that uses a local Hugging Face transformer model.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        # Auto-select the best available device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Global cache so the HF model loads only once per process
        global _TOKENIZER, _MODEL
        if _TOKENIZER is None or _MODEL is None:
            print(f"Initializing HF Embedder with model '{model_name}' on device '{self.device}'. (first load)")
            _TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
            _MODEL = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device != "cpu" else None,
            ).to(self.device).eval()
            print("QwenEmbedder weights loaded and cached.")
        else:
            print("Reusing cached QwenEmbedder weights.")
        self.tokenizer = _TOKENIZER
        self.model = _MODEL

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"Generating {len(texts)} embeddings with Qwen model...")
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # [B, seq, dim]
            # Pool via last valid token per sequence (recommended for Qwen3)
            seq_len = inputs["attention_mask"].sum(dim=1) - 1  # index of last token
            batch_indices = torch.arange(last_hidden.size(0), device=self.device)
            embeddings = last_hidden[batch_indices, seq_len]
        return embeddings.cpu().numpy()

class EmbeddingGenerator:
    def __init__(self, embedding_model: EmbeddingModel, batch_size: int = 50):
        self.model = embedding_model
        self.batch_size = batch_size

    def generate(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for all chunks using batch processing"""
        texts_to_embed = [chunk['text'] for chunk in chunks]
        if not texts_to_embed: 
            return []
        
        from rag_system.utils.batch_processor import BatchProcessor, estimate_memory_usage
        
        memory_mb = estimate_memory_usage(chunks)
        print(f"Estimated memory usage for {len(chunks)} chunks: {memory_mb:.1f}MB")
        
        batch_processor = BatchProcessor(batch_size=self.batch_size)
        
        def process_text_batch(text_batch):
            if not text_batch:
                return []
            batch_embeddings = self.model.create_embeddings(text_batch)
            return [embedding for embedding in batch_embeddings]
        
        all_embeddings = batch_processor.process_in_batches(
            texts_to_embed,
            process_text_batch,
            "Embedding Generation"
        )
        
        return all_embeddings

_TOKENIZER = None  # Will hold the shared HF tokenizer instance
_MODEL = None      # Will hold the shared HF model instance

if __name__ == '__main__':
    print("representations.py cleaned up.")
    try:
        qwen_embedder = QwenEmbedder()
        emb_gen = EmbeddingGenerator(embedding_model=qwen_embedder)
        
        sample_chunks = [{'text': 'Hello world'}, {'text': 'This is a test'}]
        embeddings = emb_gen.generate(sample_chunks)
        
        print(f"\nSuccessfully generated {len(embeddings)} embeddings.")
        print(f"Shape of first embedding: {embeddings[0].shape}")

    except Exception as e:
        print(f"\nAn error occurred during the QwenEmbedder test: {e}")
        print("Please ensure you have an internet connection for model downloads.")