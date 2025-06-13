from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Any

class QwenReranker:
    """
    A reranker that uses a local Hugging Face transformer model.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        # Auto-select the best available device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Initializing BGE Reranker with model '{model_name}' on device '{self.device}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else None,
        ).to(self.device).eval()
        
        print("BGE Reranker loaded successfully.")

    def _format_instruction(self, query: str, doc: str):
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.
        """
        if not documents:
            return []
            
        # Prepare query-document pairs as required by cross-encoder
        pairs = [[query, doc['text']] for doc in documents]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            # Model outputs a single relevance logit per pair
            logits = self.model(**inputs).logits.view(-1)
            scores = logits.float().cpu().tolist()

        scored_docs = zip(scores, documents)
        sorted_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        
        reranked_docs = []
        for score, doc in sorted_docs:
            doc_with_score = doc.copy()
            doc_with_score['rerank_score'] = score
            reranked_docs.append(doc_with_score)
            
        return reranked_docs[:top_k]

if __name__ == '__main__':
    # This test requires an internet connection to download the models.
    try:
        reranker = QwenReranker(model_name="BAAI/bge-reranker-base")
        
        query = "What is the capital of France?"
        documents = [
            {'text': "Paris is the capital of France.", 'metadata': {'doc_id': 'a'}},
            {'text': "The Eiffel Tower is in Paris.", 'metadata': {'doc_id': 'b'}},
            {'text': "France is a country in Europe.", 'metadata': {'doc_id': 'c'}},
        ]
        
        reranked_documents = reranker.rerank(query, documents)
        
        print("\n--- Verification ---")
        print(f"Query: {query}")
        print("Reranked documents:")
        for doc in reranked_documents:
            print(f"  - Score: {doc['rerank_score']:.4f}, Text: {doc['text']}")

    except Exception as e:
        print(f"\nAn error occurred during the QwenReranker test: {e}")
        print("Please ensure you have an internet connection for model downloads.")
