from typing import Dict, Any
import json
from rag_system.utils.ollama_client import OllamaClient
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.agent.verifier import Verifier
from rag_system.retrieval.query_transformer import QueryDecomposer, GraphQueryTranslator
from rag_system.retrieval.retrievers import GraphRetriever

class Agent:
    """
    The main agent, now fully wired to use a live Ollama client.
    """
    def __init__(self, pipeline_configs: Dict[str, Dict], llm_client: OllamaClient, ollama_config: Dict[str, str]):
        self.pipeline_configs = pipeline_configs
        self.llm_client = llm_client
        self.ollama_config = ollama_config
        
        gen_model = self.ollama_config["generation_model"]
        
        self.verifier = Verifier(llm_client, gen_model)
        self.query_decomposer = QueryDecomposer(llm_client, gen_model)
        
        graph_config = self.pipeline_configs.get("graph_strategy", {})
        if graph_config.get("enabled"):
            self.graph_query_translator = GraphQueryTranslator(llm_client, gen_model)
            self.graph_retriever = GraphRetriever(graph_config["graph_path"])
            print("Agent initialized with live GraphRAG capabilities.")
        else:
            print("Agent initialized (GraphRAG disabled).")

    def _triage_query(self, query: str) -> str:
        prompt = f"""
You are a query routing expert. Analyze the user's query and classify it into one of three categories:
1. "graph_query": If the query is asking for a specific factual relationship that would likely be found in a knowledge graph (e.g., "Who is the CEO of X?", "What did Company Y announce?").
2. "rag_query": If the query is asking a question that requires searching specific uploaded documents for an answer (e.g., "What is the total of the invoice?", "Summarize the report on Q3 earnings.").
3. "direct_answer": If the query is a general conversational question, a philosophical question, or anything that does not require knowledge of specific uploaded documents (e.g., "Hello", "What is the meaning of life?", "What is the capital of France?").

Respond with a single JSON object with one key, "category".

Query: "{query}"

JSON Output:
"""
        response = self.llm_client.generate_completion(self.ollama_config["generation_model"], prompt, format="json")
        try:
            data = json.loads(response.get('response', '{}'))
            return data.get("category", "rag_query")
        except json.JSONDecodeError:
            return "rag_query"

    def _run_pipeline(self, config_name: str, query: str) -> Dict[str, Any]:
        config = self.pipeline_configs[config_name]
        pipeline = RetrievalPipeline(config, self.llm_client, self.ollama_config)
        return pipeline.run(query)

    def _run_graph_query(self, query: str) -> Dict[str, Any]:
        structured_query = self.graph_query_translator.translate(query)
        if not structured_query.get("start_node"):
            return self._run_pipeline("default", query)
        results = self.graph_retriever.retrieve(structured_query)
        if not results:
            return self._run_pipeline("default", query)
        answer = ", ".join([res['details']['node_id'] for res in results])
        return {"answer": f"From the knowledge graph: {answer}", "source_documents": results}

    def run(self, query: str, max_retries: int = 1):
        triage_result = self._triage_query(query)
        print(f"Agent Triage Decision: '{triage_result}'")

        if triage_result == "direct_answer":
            prompt = f"You are a helpful assistant. Answer the user's question directly.\n\nUser: {query}\n\nAssistant:"
            response = self.llm_client.generate_completion(self.ollama_config["generation_model"], prompt)
            return {"answer": response.get('response'), "source_documents": []}
        
        if triage_result == "graph_query" and hasattr(self, 'graph_retriever'):
            return self._run_graph_query(query)

        # --- Standard RAG is required ---
        for attempt in range(max_retries + 1):
            result = self._run_pipeline("default", query)
            context_str = "\n".join([doc['text'] for doc in result['source_documents']])
            verification = self.verifier.verify(query, context_str, result['answer'])
            
            if verification.is_grounded:
                return result
            if attempt == max_retries:
                result['answer'] += " [Warning: This answer could not be fully verified.]"
                return result
            print("Answer not grounded. Retrying is not yet implemented, returning best effort.")
            return result
