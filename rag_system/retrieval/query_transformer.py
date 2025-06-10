from typing import List, Any, Dict
import json
from rag_system.utils.ollama_client import OllamaClient

class QueryDecomposer:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def decompose(self, query: str) -> List[str]:
        prompt = f"""
You are a query decomposition expert. Analyze the user query and decide if it should be broken down into sub-questions.

WHEN TO DECOMPOSE:
- Complex queries with multiple distinct aspects (e.g., "What are the costs and benefits of DeepSeek?")
- Queries asking for comparisons (e.g., "Compare the pricing between invoice 1039 and 1041")
- Multi-part questions (e.g., "Who is the CEO and what is the company's revenue?")
- Questions with conjunctions like "and", "or", "also" (e.g., "What is the address and phone number?")

WHEN NOT TO DECOMPOSE:
- Simple, direct questions (e.g., "What is the invoice amount?")
- Single-concept queries (e.g., "What is DeepSeek?")
- Questions asking for a specific piece of information (e.g., "What is the due date?")

If the query should be decomposed, break it into 2-3 simple, independent sub-questions that together cover the original query.
If not, return the original query as the only item.

Each sub-question should:
- Be answerable independently
- Cover a specific aspect of the original query
- Be clear and unambiguous

Query: "{query}"

Return a JSON object with a single key "sub_queries" which is a list of strings.

JSON Output:
"""
        response = self.llm_client.generate_completion(self.llm_model, prompt, format="json")
        try:
            data = json.loads(response.get('response', '{}'))
            sub_queries = data.get('sub_queries', [query])
            
            # Ensure we always have at least the original query
            if not sub_queries or len(sub_queries) == 0:
                return [query]
            
            # Limit to maximum 3 sub-queries to avoid excessive API calls
            return sub_queries[:3]
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