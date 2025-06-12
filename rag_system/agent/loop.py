from typing import Dict, Any
import json
import concurrent.futures
import time
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
        
        # Initialize the single, persistent retrieval pipeline for this agent
        self.retrieval_pipeline = RetrievalPipeline(pipeline_configs, self.llm_client, self.ollama_config)
        
        self.verifier = Verifier(llm_client, gen_model)
        self.query_decomposer = QueryDecomposer(llm_client, gen_model)
        
        # 🚀 OPTIMIZED: Simple query cache for repeated queries
        self._query_cache = {}
        self._cache_max_size = 100  # Limit cache size to prevent memory bloat
        
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

    def _run_graph_query(self, query: str) -> Dict[str, Any]:
        structured_query = self.graph_query_translator.translate(query)
        if not structured_query.get("start_node"):
            return self.retrieval_pipeline.run(query)
        results = self.graph_retriever.retrieve(structured_query)
        if not results:
            return self.retrieval_pipeline.run(query)
        answer = ", ".join([res['details']['node_id'] for res in results])
        return {"answer": f"From the knowledge graph: {answer}", "source_documents": results}

    def _get_cache_key(self, query: str, query_type: str) -> str:
        """Generate a cache key for the query"""
        # Simple cache key based on query and type
        return f"{query_type}:{query.strip().lower()}"
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache a result with size limit"""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

    def run(self, query: str, max_retries: int = 1) -> Dict[str, Any]:
        start_time = time.time()
        
        query_type = self._triage_query(query)
        print(f"Agent Triage Decision: '{query_type}'")
        
        # 🚀 OPTIMIZED: Check cache first for non-direct answers
        if query_type != "direct_answer":
            cache_key = self._get_cache_key(query, query_type)
            if cache_key in self._query_cache:
                cached_entry = self._query_cache[cache_key]
                cache_age = time.time() - cached_entry['timestamp']
                
                # Use cache if less than 5 minutes old
                if cache_age < 300:
                    print(f"🚀 Cache hit! Returning cached result (age: {cache_age:.1f}s)")
                    return cached_entry['result']
                else:
                    # Remove stale cache entry
                    del self._query_cache[cache_key]

        if query_type == "direct_answer":
            prompt = f"You are a helpful assistant. Answer the user's question directly.\n\nUser: {query}\n\nAssistant:"
            response = self.llm_client.generate_completion(self.ollama_config["generation_model"], prompt)
            return {"answer": response.get('response'), "source_documents": []}
        
        if query_type == "graph_query" and hasattr(self, 'graph_retriever'):
            return self._run_graph_query(query)

        # --- RAG Query Processing with Optional Query Decomposition ---
        query_decomp_config = self.pipeline_configs.get("query_decomposition", {})
        if query_decomp_config.get("enabled", False):
            print(f"\n--- Query Decomposition Enabled ---")
            sub_queries = self.query_decomposer.decompose(query)
            print(f"Original query: '{query}'")
            print(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            
            if len(sub_queries) > 1:
                # 🚀 OPTIMIZED: Parallel multi-query retrieval
                print(f"\n--- Processing {len(sub_queries)} sub-queries in parallel ---")
                start_time = time.time()
                
                all_source_docs = []
                seen_chunk_ids = set()
                
                # Process sub-queries in parallel using ThreadPoolExecutor
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as executor:
                    # Submit all sub-queries for parallel processing
                    future_to_query = {
                        executor.submit(self.retrieval_pipeline.run, sub_query): (i, sub_query)
                        for i, sub_query in enumerate(sub_queries)
                    }
                    
                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(future_to_query):
                        i, sub_query = future_to_query[future]
                        try:
                            sub_result = future.result()
                            print(f"✅ Sub-Query {i+1} completed: '{sub_query}'")
                            
                            # Collect unique documents from this sub-query
                            for doc in sub_result['source_documents']:
                                if doc['chunk_id'] not in seen_chunk_ids:
                                    all_source_docs.append(doc)
                                    seen_chunk_ids.add(doc['chunk_id'])
                        except Exception as e:
                            print(f"❌ Sub-Query {i+1} failed: '{sub_query}' - {e}")
                
                parallel_time = time.time() - start_time
                print(f"🚀 Parallel processing completed in {parallel_time:.2f}s")
                print(f"\n--- Aggregated {len(all_source_docs)} unique documents from all sub-queries ---")
                
                # Synthesize final answer using original query and aggregated context
                if all_source_docs:
                    aggregated_context = "\n\n".join([doc['text'] for doc in all_source_docs])
                    final_answer = self.retrieval_pipeline._synthesize_final_answer(query, aggregated_context)
                    result = {
                        "answer": final_answer,
                        "source_documents": all_source_docs
                    }
                else:
                    result = {
                        "answer": "I could not find relevant information to answer your question.",
                        "source_documents": []
                    }
            else:
                # Single query - standard flow
                print("Query does not need decomposition, proceeding with standard retrieval.")
                result = self.retrieval_pipeline.run(query)
        else:
            # Standard RAG without decomposition
            result = self.retrieval_pipeline.run(query)
        
        # Verification step (simplified for now) - Skip in fast mode
        if self.pipeline_configs.get("verification", {}).get("enabled", True):
            context_str = "\n".join([doc['text'] for doc in result['source_documents']])
            verification = self.verifier.verify(query, context_str, result['answer'])
            
            if not verification.is_grounded:
                result['answer'] += " [Warning: This answer could not be fully verified.]"
        else:
            print("🚀 Skipping verification for speed")
        
        # 🚀 OPTIMIZED: Cache the result for future queries
        if query_type != "direct_answer":
            cache_key = self._get_cache_key(query, query_type)
            self._cache_result(cache_key, result)
        
        total_time = time.time() - start_time
        print(f"🚀 Total query processing time: {total_time:.2f}s")
        
        return result
