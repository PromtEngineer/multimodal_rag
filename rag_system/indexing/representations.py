from typing import List, Dict, Any, Protocol
import numpy as np

# We keep the protocol to ensure a consistent interface
class EmbeddingModel(Protocol):
    def create_embeddings(self, texts: List[str]) -> np.ndarray: ...

# --- New Ollama Embedder ---
from transformers import AutoModel, AutoTokenizer
import torch

class QwenEmbedder(EmbeddingModel):
    """
    An embedding model that uses a local Hugging Face transformer model.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing QwenEmbedder with model '{model_name}' on device '{self.device}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        print("QwenEmbedder loaded successfully.")

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"Generating {len(texts)} embeddings with Qwen model...")
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()

# --- Other Representation Generators (BM25, HyDE) ---
# These remain largely the same but will be updated later to use the Ollama client.
from rank_bm25 import BM25Okapi
from rag_system.utils.ollama_client import OllamaClient

class OllamaEmbedder(EmbeddingModel):
    """
    An embedding model that uses a local Ollama model.
    """
    def __init__(self, llm_client: OllamaClient, model_name: str):
        self.llm_client = llm_client
        self.model_name = model_name
        print(f"Initializing OllamaEmbedder with model '{model_name}'.")

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        print(f"Generating {len(texts)} embeddings with Ollama model...")
        embeddings = []
        for text in texts:
            response = self.llm_client.generate_embedding(self.model_name, text)
            embeddings.append(response['embedding'])
        return np.array(embeddings)

class BM25Generator:
    def generate(self, chunks: List[Dict[str, Any]]):
        import re
        def tokenize_text(text):
            # Use regex to split on whitespace and punctuation, then lowercase
            tokens = re.findall(r'\b\w+\b', text.lower())
            return tokens
        
        tokenized_corpus = [tokenize_text(chunk['text']) for chunk in chunks]
        if not tokenized_corpus: return None
        return BM25Okapi(tokenized_corpus)

class HyDEGenerator:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def generate(self, query: str) -> str:
        print(f"Generating hypothetical document for query: '{query}'...")
        prompt = f"Generate a short, hypothetical document that answers the following question: {query}"
        response = self.llm_client.generate_completion(self.llm_model, prompt)
        return response.get('response', '')

class EmbeddingGenerator:
    def __init__(self, embedding_model: EmbeddingModel):
        self.model = embedding_model

    def generate(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        texts_to_embed = [chunk['text'] for chunk in chunks]
        if not texts_to_embed: return []
        embeddings = self.model.create_embeddings(texts_to_embed)
        return [embedding for embedding in embeddings]

if __name__ == '__main__':
    print("representations.py updated with a functional QwenEmbedder.")
    try:
        # --- Test QwenEmbedder ---
        qwen_embedder = QwenEmbedder()
        emb_gen = EmbeddingGenerator(embedding_model=qwen_embedder)
        
        sample_chunks = [{'text': 'Hello world'}, {'text': 'This is a test'}]
        embeddings = emb_gen.generate(sample_chunks)
        
        print(f"\nSuccessfully generated {len(embeddings)} embeddings.")
        print(f"Shape of first embedding: {embeddings[0].shape}")

    except Exception as e:
        print(f"\nAn error occurred during the QwenEmbedder test: {e}")
        print("Please ensure you have an internet connection for model downloads.")