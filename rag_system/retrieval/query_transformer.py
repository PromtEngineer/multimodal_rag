from typing import List, Any, Dict
import json
from rag_system.utils.ollama_client import OllamaClient

class QueryDecomposer:
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model

    def decompose(self, query: str) -> List[str]:
        prompt = f"""
You are an expert at query decomposition for a Retrieval-Augmented Generation (RAG) system. Your goal is to break down a complex user query into a series of simpler, standalone sub-queries. Each sub-query will be used to retrieve relevant documents independently.

First, analyze the user's query to determine if it requires decomposition.

**Decomposition is necessary for:**
- **Multi-part questions:** Queries containing "and", "or", "also" that join distinct informational needs. (e.g., "What are the performance and training costs of DeepSeek-V3?")
- **Comparative questions:** Queries that ask to compare or contrast two or more items. For these, break the query into separate fact-finding questions for each item. The final synthesis step will handle the comparison.
- **Temporal or sequential questions:** Queries asking about a sequence of events or changes over time.

**Decomposition is NOT necessary for:**
- **Simple, factual questions:** Queries asking for a single piece of information. (e.g., "What is the architecture of DeepSeek-V3?")
- **Ambiguous queries that need clarification, not decomposition.** (e.g., "Tell me about the model.")

**Instructions:**
1.  **Analyze the query step-by-step.** First, decide if decomposition is needed and explain why in your reasoning.
2.  If decomposition is needed, generate self-contained sub-queries. Each must be answerable on its own without context from the others.
3.  If decomposition is not needed, the `sub_queries` list should contain only the original, unmodified query.
4.  The sub-queries should be phrased as clear, simple questions.

**Query:** "{query}"

---
**EXAMPLES**

**Example 1: Multi-Part Query**
Query: "What were the main findings of the aiconfig report and how do they compare to the results from the RAG paper?"
JSON Output:
{{
  "reasoning": "The query asks for two distinct pieces of information: the findings from one report and a comparison to another. This requires two separate retrieval steps.",
  "sub_queries": [
    "What were the main findings of the aiconfig report?",
    "How do the findings of the aiconfig report compare to the results from the RAG paper?"
  ]
}}

**Example 2: Simple Query**
Query: "Summarize the contributions of the DeepSeek-V3 paper."
JSON Output:
{{
  "reasoning": "This is a direct request for a summary of a single document and does not contain multiple parts.",
  "sub_queries": [
    "Summarize the contributions of the DeepSeek-V3 paper."
  ]
}}

**Example 3: Comparative Query**
Query: "Did Microsoft or Google make more money last year?"
JSON Output:
{{
  "reasoning": "This is a comparative query that requires fetching the profit for each company before a comparison can be made.",
  "sub_queries": [
    "How much profit did Microsoft make last year?",
    "How much profit did Google make last year?"
  ]
}}

**Example 4: Comparative Query with different phrasing**
Query: "Who has more siblings, Jamie or Sansa?"
JSON Output:
{{
  "reasoning": "This comparative query needs the sibling count for both individuals to be answered.",
  "sub_queries": [
    "How many siblings does Jamie have?",
    "How many siblings does Sansa have?"
  ]
}}
---

Now, provide your response for the query above. Return a JSON object with "reasoning" and "sub_queries" keys.

JSON Output:
"""
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