from typing import List, Any, Dict
import json
from rag_system.utils.ollama_client import OllamaClient
from rag_system.prompts import fmt, get

class QueryDecomposer:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def decompose(self, query: str) -> List[str]:
        prompt = fmt("query_decompose", query=query)
        response = self.llm_client.generate_completion(self.llm_model, prompt, format="json")
        response_text = response.get('response', '{}')
        try:
            # Handle potential markdown code blocks in the response
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-4]

            data = json.loads(response_text)
            sub_queries = data.get('sub_queries', [query])
            reasoning = data.get('reasoning', 'No reasoning provided.')
            
            print(f"Query Decomposition Reasoning: {reasoning}")

            # Ensure we always have at least the original query
            if not sub_queries or len(sub_queries) == 0:
                return [query]
            
            # Deduplicate while preserving order
            sub_queries = list(dict.fromkeys(sub_queries))

            # Limit to maximum 3 sub-queries to avoid excessive API calls
            return sub_queries[:3]
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from query decomposer: {response_text}")
            return [query]

class HyDEGenerator:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def generate(self, query: str) -> str:
        prompt = fmt("hyde_generator", query=query)
        response = self.llm_client.generate_completion(self.llm_model, prompt)
        return response.get('response', '')

class GraphQueryTranslator:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def _generate_translation_prompt(self, query: str) -> str:
        return fmt("graph_translation", query=query)

    def translate(self, query: str) -> Dict[str, Any]:
        prompt = self._generate_translation_prompt(query)
        response = self.llm_client.generate_completion(self.llm_model, prompt, format="json")
        try:
            return json.loads(response.get('response', '{}'))
        except json.JSONDecodeError:
            return {}