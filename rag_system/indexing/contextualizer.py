from typing import List, Dict, Any
from rag_system.utils.ollama_client import OllamaClient

class ContextualEnricher:
    """
    Enriches chunks with a prepended summary of their surrounding context using Ollama.
    """
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model
        print(f"Initialized ContextualEnricher with Ollama model '{self.llm_model}'.")

    def _generate_summary(self, text_window: str) -> str:
        prompt = (
            "You are an expert at creating concise summaries for context. "
            "Summarize the key topic of the following text in one short sentence. "
            "Do not add any preamble or explanation.\n\n"
            f"Text: \"{text_window}\"\n\n"
            "Summary:"
        )
        response = self.llm_client.generate_completion(self.llm_model, prompt)
        return response.get('response', '').strip()

    def enrich_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        print(f"Enriching {len(chunks)} chunks with contextual summaries using Ollama...")
        enriched_chunks = []
        
        # To prevent circular import, we define the window creator function locally
        # if it's not passed in. A better design might be to move this to a shared utils file.
        def create_contextual_window(all_chunks, chunk_index, window_size=1):
            start = max(0, chunk_index - window_size)
            end = min(len(all_chunks), chunk_index + window_size + 1)
            return " ".join([c['text'] for c in all_chunks[start:end]])

        for i, chunk in enumerate(chunks):
            context_window = create_contextual_window(chunks, chunk_index=i, window_size=1)
            summary = self._generate_summary(context_window)
            
            original_text = chunk['text']
            enriched_text = f"Context: {summary}\n\n---\n\n{original_text}"
            
            new_chunk = chunk.copy()
            new_chunk['text'] = enriched_text
            new_chunk['metadata']['original_text'] = original_text
            new_chunk['metadata']['contextual_summary'] = summary
            
            enriched_chunks.append(new_chunk)
            
        return enriched_chunks