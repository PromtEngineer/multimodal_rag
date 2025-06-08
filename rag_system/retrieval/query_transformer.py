from typing import List, Any, Dict
import json
from rag_system.utils.ollama_client import OllamaClient

class QueryDecomposer:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def decompose(self, query: str) -> List[str]:
        prompt = f"""
You are a query decomposition expert. Break the following user query into 1 to 3 simple, self-contained questions that can be answered independently.
Return a JSON object with a single key "sub_queries" which is a list of strings.

Query: "{query}"

JSON Output:
"""
        response = self.llm_client.generate_completion(self.llm_model, prompt, format="json")
        try:
            data = json.loads(response.get('response', '{}'))
            return data.get('sub_queries', [query])
        except json.JSONDecodeError:
            return [query]

class HyDEGenerator:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def generate(self, query: str) -> str:
        prompt = f"Generate a short, hypothetical document that answers the following question. The document should be dense with keywords and concepts related to the query.\n\nQuery: {query}\n\nHypothetical Document:"
        response = self.llm_client.generate_completion(self.llm_model, prompt)
        return response.get('response', '')

class GraphQueryTranslator:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def _generate_translation_prompt(self, query: str) -> str:
        return f"""
You are an expert query planner. Convert the user's question into a structured JSON query for a knowledge graph.
The JSON should contain a 'start_node' (the known entity in the query) and an 'edge_label' (the relationship being asked about).
The graph has nodes (entities) and directed edges (relationships). For example, (Tim Cook) -[IS_CEO_OF]-> (Apple).
Return ONLY the JSON object.

User Question: "{query}"

JSON Output:
"""

    def translate(self, query: str) -> Dict[str, Any]:
        prompt = self._generate_translation_prompt(query)
        response = self.llm_client.generate_completion(self.llm_model, prompt, format="json")
        try:
            return json.loads(response.get('response', '{}'))
        except json.JSONDecodeError:
            return {}