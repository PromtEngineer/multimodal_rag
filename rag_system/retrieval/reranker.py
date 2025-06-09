from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any

class QwenReranker:
    """
    A reranker that uses a local Hugging Face transformer model.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing QwenReranker with model '{model_name}' on device '{self.device}'.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device).eval()
        
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        print("QwenReranker loaded successfully.")

    def _format_instruction(self, query: str, doc: str):
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on their relevance to a query.
        """
        if not documents:
            return []
            
        pairs = [self._format_instruction(query, doc['text']) for doc in documents]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=8192
            ).to(self.device)
            
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            stacked_scores = torch.stack([false_vector, true_vector], dim=1)
            softmax_scores = torch.nn.functional.log_softmax(stacked_scores, dim=1)
            scores = softmax_scores[:, 1].exp().tolist()

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
        reranker = QwenReranker(model_name="Qwen/Qwen3-Reranker-0.6B")
        
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
