from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Any

class QwenReranker:
    """
    A reranker that uses a local Hugging Face transformer model.
    """
    def __init__(self, model_name: str = "Qwen/Qwen-reranker"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing QwenReranker with model '{model_name}' on device '{self.device}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        print("QwenReranker loaded successfully.")

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.
        """
        if not documents:
            return []
            
        pairs = [[query, doc['text']] for doc in documents]
        with torch.no_grad():
            scores = self.model.compute_score(pairs)
        
        # Combine scores with original documents and sort
        scored_docs = zip(scores, documents)
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        # Return the top_k documents
        return [doc for score, doc in sorted_docs][:top_k]

if __name__ == '__main__':
    # This test requires an internet connection to download the models.
    try:
        # 1. Setup the reranker
        reranker = QwenReranker()
        
        # 2. Rerank some documents
        query = "What is the capital of France?"
        documents = [
            {'text': "Paris is the capital of France.", 'metadata': {'doc_id': 'a'}},
            {'text': "The Eiffel Tower is in Paris.", 'metadata': {'doc_id': 'b'}},
            {'text': "France is a country in Europe.", 'metadata': {'doc_id': 'c'}},
        ]
        
        reranked_documents = reranker.rerank(query, documents)
        
        # 3. Verify
        print("\n--- Verification ---")
        print(f"Query: {query}")
        print(f"Original documents: {[doc['text'] for doc in documents]}")
        print(f"Reranked documents: {[doc['text'] for doc in reranked_documents]}")

    except Exception as e:
        print(f"\nAn error occurred during the QwenReranker test: {e}")
        print("Please ensure you have an internet connection for model downloads.")
