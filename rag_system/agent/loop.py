import asyncio
import concurrent.futures
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from cachetools import LRUCache, TTLCache

# Assume these imports work and provide SYNC clients/functions
from rag_system.utils.ollama_client import OllamaClient
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.agent.verifier import Verifier
from rag_system.retrieval.query_transformer import QueryDecomposer
from rag_system.config import get_model_capabilities, get_thinking_setting

# Set up basic logging (You might want to configure this more robustly)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class Agent:
    """
    Refactored Agent: Implements Overview-based Triage, runs RAG or
    Direct LLM paths, and handles async operations safely by running
    assumed-sync components in threads. GraphRAG removed.
    """
    def __init__(self, pipeline_configs: Dict[str, Dict], llm_client: OllamaClient, ollama_config: Dict[str, str], current_index_id: str = None):
        self.pipeline_configs = pipeline_configs
        self.llm_client = llm_client
        self.ollama_config = ollama_config
        self.current_index_id = current_index_id
        
        self.gen_model = self.ollama_config["generation_model"]
        self.fast_model = self.ollama_config.get("enrichment_model", self.gen_model)
        
        # Auto-detect model capabilities for thinking token support
        self.gen_model_caps = get_model_capabilities(self.gen_model)
        self.fast_model_caps = get_model_capabilities(self.fast_model)
        
        log.info(f"Agent Models -> Gen: {self.gen_model} (thinking: {self.gen_model_caps['supports_thinking']}), "
                f"Fast: {self.fast_model} (thinking: {self.fast_model_caps['supports_thinking']})")
        
        self.retrieval_pipeline = RetrievalPipeline(pipeline_configs, self.llm_client, self.ollama_config)
        self.verifier = Verifier(llm_client, self.gen_model)
        self.query_decomposer = QueryDecomposer(llm_client, self.gen_model) # Use reasoning model
        
        # Caching
        cache_max_size = int(pipeline_configs.get("cache_max_size", 100))
        self._query_cache: TTLCache = TTLCache(maxsize=cache_max_size, ttl=300)
        self.semantic_cache_threshold = float(pipeline_configs.get("semantic_cache_threshold", 0.98))
        self.cache_scope = pipeline_configs.get("cache_scope", "global")

        # History
        self.chat_histories: LRUCache = LRUCache(maxsize=100)

        # Load Document Overviews (index-specific or global)
        self.doc_overviews: List[str] = self._load_doc_overviews()
        self.overview_text: str = "\n\n".join(
            f"- {o}" for o in self.doc_overviews
        ) if self.doc_overviews else "No document overviews are available."

        index_info = f" for index {current_index_id[:8]}..." if current_index_id else " (global)"
        log.info(f"Agent initialized{index_info}. Overviews Loaded: {len(self.doc_overviews)}")

    def _load_doc_overviews(self) -> List[str]:
        """Loads document overview strings from index-specific or global jsonl file."""
        # Determine overview path - index-specific first, then global fallback
        if self.current_index_id:
            overview_path = f"index_store/overviews/overviews_{self.current_index_id}.jsonl"
            fallback_path = os.path.join("index_store", "overviews", "overviews.jsonl")
        else:
            overview_path = self.pipeline_configs.get("overview_path", 
                                                 os.path.join("index_store", "overviews", "overviews.jsonl"))
            fallback_path = None
            
        overviews = []
        
        # Try index-specific file first
        if os.path.exists(overview_path):
            try:
                with open(overview_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line.strip())
                            if isinstance(rec, dict) and rec.get("overview"):
                                overviews.append(rec["overview"].strip())
                        except json.JSONDecodeError as json_err:
                            log.warning(f"Skipping malformed line in overviews: {json_err} - Line: '{line.strip()}'")
                log.info(f"Loaded {len(overviews)} document overviews from {overview_path}")
                return overviews
            except Exception as e:
                log.error(f"Failed to load document overviews from {overview_path}: {e}", exc_info=True)
        else:
            log.warning(f"Index-specific overview file not found: {overview_path}")
            
        # Fallback to global overviews if index-specific not found
        if fallback_path and os.path.exists(fallback_path):
            try:
                with open(fallback_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line.strip())
                            if isinstance(rec, dict) and rec.get("overview"):
                                overviews.append(rec["overview"].strip())
                        except json.JSONDecodeError as json_err:
                            log.warning(f"Skipping malformed line in overviews: {json_err} - Line: '{line.strip()}'")
                log.info(f"Loaded {len(overviews)} document overviews from fallback {fallback_path}")
                return overviews
            except Exception as e:
                log.error(f"Failed to load document overviews from fallback {fallback_path}: {e}", exc_info=True)
        
        if self.current_index_id:
            log.warning(f"No overview files found for index {self.current_index_id} or global fallback")
        else:
            log.warning(f"No overview files found")
            
        return []

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        v1 = np.array(v1); v2 = np.array(v2)
        if v1.shape != v2.shape:
            log.warning(f"Cosine shape mismatch: {v1.shape} vs {v2.shape}")
            return 0.0
        norm_v1 = np.linalg.norm(v1); norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

    def _find_in_semantic_cache(self, query_embedding: np.ndarray, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Finds a semantically similar query in the cache."""
        if query_embedding is None: return None
        for key, cached_item in list(self._query_cache.items()): 
            cached_embedding = cached_item.get('embedding')
            if cached_embedding is None: continue
            if self.cache_scope == "session" and session_id != cached_item.get("session_id"): continue
            similarity = self._cosine_similarity(query_embedding, np.array(cached_embedding))
            if similarity >= self.semantic_cache_threshold:
                log.info(f"Semantic cache hit! Similarity: {similarity:.3f} with '{key}'")
                return cached_item.get('result') 
        return None

    def _get_cache_key(self, query: str) -> str:
        """Generate a simple cache key."""
        return query.strip().lower()
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any], query_embedding: np.ndarray, session_id: Optional[str] = None):
        """Cache a result. TTLCache handles size/TTL."""
        if query_embedding is None: return
        self._query_cache[cache_key] = {
            'result': result,
            'embedding': query_embedding,
            'timestamp': time.time(),
            'session_id': session_id
        }

    def _format_query_with_history(self, query: str, history: list) -> str:
        """Formats the user query with recent conversation history for context."""
        if not history: return query
        history_snippet = history[-3:] 
        formatted_history = "\n".join([f"User: {turn['query']}\nAssistant: {turn['answer']}" for turn in history_snippet])
        return f"--- Conversation History ---\n{formatted_history}\n---\nLatest User Query: \"{query}\""

    async def _triage_query_async(self, query: str) -> str:
        """Uses Doc Overviews for RAG vs LLM routing. Returns 'USE_RAG' or 'DIRECT_LLM'."""
        
        # Simple pre-filter for basic chat. Consider making this list configurable.
        simple_chat_starters = ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay"]
        if query.strip().lower() in simple_chat_starters:
            log.info("Triage Decision: Simple Chat (Pre-Filter) -> DIRECT_LLM")
            return "DIRECT_LLM"

        prompt = f"""Your task is to decide if a query should use our Knowledge Base (KB) or answer directly.

CRITICAL PRINCIPLE: **When documents exist in the KB, strongly prefer USE_RAG unless the query is purely conversational or completely unrelated to any possible document content.**

You MUST choose ONE decision:
1.  **"USE_RAG"**: Choose this for ANY query that could potentially be answered by document content, including:
    - Questions about documents, invoices, reports, data, amounts, dates, names, companies
    - Requests to summarize, explain, or analyze anything
    - Questions about "the document", "this file", or any specific information
    - Any factual question that might relate to document content
    - When in doubt, choose this option

2.  **"DIRECT_LLM"**: ONLY use this for:
    - Simple greetings: "Hi", "Hello", "Thanks"
    - Basic math: "What is 2+2?"
    - General world knowledge clearly unrelated to documents: "Who is the president of France?"
    - Weather, current events, or topics obviously not in documents

--- Knowledge Base (KB) Content ---
Our KB contains these types of documents:
{self.overview_text}

--- Decision Examples ---
*   Query: "What is the total amount?" -> {{"decision": "USE_RAG"}} (document-specific)
*   Query: "Can you summarize?" -> {{"decision": "USE_RAG"}} (document operation)
*   Query: "What are the key features?" -> {{"decision": "USE_RAG"}} (could be about documents)
*   Query: "Who is mentioned in the invoice?" -> {{"decision": "USE_RAG"}} (document content)
*   Query: "What is the date?" -> {{"decision": "USE_RAG"}} (likely document date)
*   Query: "Hi there" -> {{"decision": "DIRECT_LLM"}} (greeting only)
*   Query: "What is 2+2?" -> {{"decision": "DIRECT_LLM"}} (pure math)
*   Query: "Who is the US president?" -> {{"decision": "DIRECT_LLM"}} (world knowledge)

**Remember: If ANY document might contain relevant information, use USE_RAG. Only use DIRECT_LLM for clearly unrelated queries.**

User Query: "{query}"

Respond ONLY with a valid JSON object: {{"decision": "<Your Choice Here>"}}"""

        raw_response_text = "" # Keep track for logging
        try:
            # Auto-determine thinking setting for fast triage operations
            thinking_enabled = get_thinking_setting(self.fast_model, "fast")
            resp = await asyncio.to_thread(
                self.llm_client.generate_completion,
                model=self.fast_model, prompt=prompt, format="json", enable_thinking=thinking_enabled
            )
            raw_response_text = resp.get("response", "{}").strip()
            log.debug(f"Triage Raw LLM Response: '{raw_response_text}'")

            # Handle markdown code blocks if the LLM adds them.
            if raw_response_text.startswith("```json"):
                 raw_response_text = raw_response_text.split("```json")[1].split("```")[0].strip()
            elif raw_response_text.startswith("`"):
                raw_response_text = raw_response_text.strip("`").strip()

            data = json.loads(raw_response_text)
            decision = data.get("decision")
            
            if decision in ["USE_RAG", "DIRECT_LLM"]:
                log.info(f"Triage Decision: {decision} for query '{query}'")
                return decision
            else:
                 # Default to RAG as 'safe' if JSON is bad, but log a warning.
                 log.warning(f"Triage: LLM returned invalid JSON or decision '{decision}'. Defaulting to USE_RAG.")
                 return "USE_RAG" 
        except Exception as e:
            log.error(f"Triage call or JSON parsing failed: {e}. Raw Response: '{raw_response_text}'. Defaulting to USE_RAG.", exc_info=True)
            return "USE_RAG" # Safe default on any error.

    async def _answer_direct_llm_async(self, query_with_history: str) -> str:
        """Generates a direct answer using LLM (Fast Model), no RAG."""
        prompt = f"You are a helpful AI Assistant. Provide a direct, concise answer to the latest user query, using history for context: {query_with_history}"
        try:
            # Use fast operation thinking setting for direct answers
            thinking_enabled = get_thinking_setting(self.fast_model, "fast")
            resp = await asyncio.to_thread(
                self.llm_client.generate_completion, 
                model=self.fast_model, 
                prompt=prompt,
                enable_thinking=thinking_enabled
            )
            return resp.get("response", "I am sorry, I could not formulate an answer.")
        except Exception as e:
            log.error(f"Direct LLM call failed: {e}", exc_info=True)
            return "I encountered an issue while generating an answer."

    def _run_rag_sub_query_sync(self, sub_query: str, table_name: Optional[str], window_size: Optional[int], max_retries: int) -> Dict[str, Any]:
        """Sync wrapper to run one RAG call with basic retries. For ThreadPool use."""
        for attempt in range(max_retries):
            try:
                # Assuming retrieval_pipeline.run is SYNC
                return self.retrieval_pipeline.run(sub_query, table_name, window_size, None)
            except Exception as e:
                log.warning(f"Sub-query '{sub_query}' attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt + 1 >= max_retries:
                    log.error(f"Sub-query '{sub_query}' FAILED after {max_retries} attempts.")
                    return {"answer": f"Failed to process: '{sub_query}'.", "source_documents": []}
                time.sleep(0.5) # Simple backoff before retry
        return {"answer": "Error: Max retries loop finished unexpectedly.", "source_documents": []}

    async def _run_rag_pipeline_async(self, raw_query: str, contextual_query: str, table_name: Optional[str], query_decompose_override: Optional[bool], context_expand: Optional[bool], max_retries: int, event_callback: Optional[Callable]) -> Dict[str, Any]:
        """Runs RAG, handles decomposition, and uses async/threads."""
        query_decomp_config = self.pipeline_configs.get("query_decomposition", {})
        decomp_enabled = query_decomp_config.get("enabled", False)
        if query_decompose_override is not None:
            decomp_enabled = query_decompose_override
            
        window_size = 0 if context_expand is False else None
        
        # Use contextual query if not decomposing, otherwise raw query for sub-queries
        query_to_run = contextual_query if not decomp_enabled else raw_query

        if not decomp_enabled:
            return await asyncio.to_thread(
                self._run_rag_sub_query_sync, query_to_run, table_name, window_size, max_retries
            )

        log.info("RAG: Query Decomposition Enabled.")
        sub_queries = await asyncio.to_thread(self.query_decomposer.decompose, raw_query)
        if not sub_queries: sub_queries = [raw_query]
        if event_callback: event_callback("decomposition", {"sub_queries": sub_queries})
        log.info(f"Decomposed into {len(sub_queries)} queries: {sub_queries}")

        if len(sub_queries) == 1:
            log.info("Only one sub-query; using direct RAG path.")
            return await asyncio.to_thread(
                self._run_rag_sub_query_sync, sub_queries[0], table_name, window_size, max_retries
            )

        log.info(f"Processing {len(sub_queries)} sub-queries in parallel...")
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as executor:
            futures = [
                loop.run_in_executor(executor, self._run_rag_sub_query_sync, sq, table_name, window_size, max_retries) 
                for sq in sub_queries
            ]
            sub_results = await asyncio.gather(*futures, return_exceptions=True)
        
        sub_answers, all_source_docs, citations_seen = [], [], set()
        for i, res in enumerate(sub_results):
            sub_query = sub_queries[i]
            if isinstance(res, Exception):
                log.error(f"Sub-Query {i+1} ('{sub_query}') execution failed: {res}", exc_info=True)
                sub_answers.append({"question": sub_query, "answer": f"Error Processing Sub-Query."})
            else:
                log.info(f"Sub-Query {i+1} completed: '{sub_query}'")
                sub_answers.append({"question": sub_query, "answer": res.get("answer", "")})
                for doc in res.get("source_documents", []):
                    doc_id = doc.get('chunk_id')
                    if doc_id and doc_id not in citations_seen:
                        all_source_docs.append(doc); citations_seen.add(doc_id)
                if event_callback: event_callback("sub_query_result", {"index": i, "query": sub_query, "answer": res.get("answer", ""), "source_documents": res.get("source_documents", [])})

        log.info("Composing final answer from sub-answers...")
        compose_prompt = f"""You are an expert answer composer. Synthesize a single, cohesive answer from sub-answers. Use ONLY information from them.
Original Question: "{raw_query}"
Sub-Answers: {json.dumps(sub_answers, indent=2)}
Final Answer:"""
        final_answer_resp = await asyncio.to_thread(
             self.llm_client.generate_completion, model=self.gen_model, prompt=compose_prompt, enable_thinking=None
        )
        final_answer = final_answer_resp.get("response", "Unable to generate a final answer.")
        
        result = {"answer": final_answer, "source_documents": all_source_docs}
        if event_callback: event_callback("final_answer", result)
        return result

    def run(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, max_retries: int = 1, event_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Synchronous entry point. Runs the main async logic."""
        log.info(f"Agent Run Start (Sync Wrapper): Q='{query[:50]}...' SID={session_id}")
        try:
            return asyncio.run(self._run_async(query, table_name, session_id, compose_sub_answers, query_decompose, ai_rerank, context_expand, max_retries, event_callback))
        except RuntimeError as e:
             if "Cannot run the event loop while another loop is running" in str(e):
                 log.error("ASYNC ERROR: Cannot use Agent.run() if an asyncio loop is already running. Call Agent._run_async() directly with `await`.")
                 return {"answer": "System Error: Event loop conflict.", "source_documents": []}
             else:
                 log.critical(f"Agent.run encountered critical error: {e}", exc_info=True)
                 return {"answer": f"A system error occurred: {e}", "source_documents": []}
        except Exception as e:
            log.critical(f"Agent.run encountered critical error: {e}", exc_info=True)
            return {"answer": f"A system error occurred: {e}", "source_documents": []}

    async def _run_async(self, query: str, table_name: Optional[str], session_id: Optional[str], compose_sub_answers: Optional[bool], query_decompose: Optional[bool], ai_rerank: Optional[bool], context_expand: Optional[bool], max_retries: int, event_callback: Optional[Callable]) -> Dict[str, Any]:
        start_time = time.time()
        raw_query = query.strip()
        query_embedding = None
        final_result = {"answer": "An error occurred during processing.", "source_documents": []}
        cached_result = None # Track if we hit cache

        if not raw_query: return {"answer": "Please provide a query.", "source_documents": []}
        if event_callback: event_callback("analyze", {"query": raw_query})
        
        history = self.chat_histories.get(session_id, []) if session_id else []
        route_decision = await self._triage_query_async(raw_query)
        contextual_query = self._format_query_with_history(raw_query, history)
        is_rag_flow = (route_decision == "USE_RAG")
        cache_key = self._get_cache_key(raw_query)
        
        if is_rag_flow:
            if hasattr(self.retrieval_pipeline, '_get_text_embedder'):
                text_embedder = self.retrieval_pipeline._get_text_embedder()
                if text_embedder:
                    try:
                        embeddings = await asyncio.to_thread(text_embedder.create_embeddings, [raw_query])
                        if embeddings is not None and len(embeddings) > 0:
                            query_embedding = np.array(embeddings[0])
                        else:
                            query_embedding = None
                    except Exception as e: 
                        log.error(f"Embedding failed: {e}", exc_info=True)
                        query_embedding = None
            
            cached_result = self._find_in_semantic_cache(query_embedding, session_id)
            if cached_result:
                final_result = cached_result # Use cached result
            else:
                 # RAG logic, only if NOT cached
                 if ai_rerank is not None and hasattr(self.retrieval_pipeline, 'config'):
                     log.info(f"Applying AI Rerank Override: {ai_rerank}")
                     rr_cfg = self.retrieval_pipeline.config.setdefault("reranker", {})
                     rr_cfg["enabled"] = bool(ai_rerank)

                 final_result = await self._run_rag_pipeline_async(
                    raw_query, contextual_query, table_name, query_decompose, 
                    context_expand, max_retries, event_callback
                 )
        else: # DIRECT_LLM Flow
            final_answer = await self._answer_direct_llm_async(contextual_query)
            final_result = {"answer": final_answer, "source_documents": []}

        # Verification (Only for RAG results NOT from cache)
        verify_enabled = self.pipeline_configs.get("verification", {}).get("enabled", True)
        if is_rag_flow and final_result.get("source_documents") and verify_enabled and not cached_result:
            context_str = "\n".join([doc.get('text', '') for doc in final_result['source_documents']]).strip()
            if context_str:
                log.info("Verifying final answer against sources...")
                try:
                    verification = await self.verifier.verify_async(raw_query, context_str, final_result['answer'])
                    score = verification.confidence_score
                    final_result['answer'] += f" [Confidence: {score:.1f}%]"
                    if not verification.is_grounded or score < 50: final_result['answer'] += " [Groundedness: Low]"
                    log.info(f"Verification complete. Grounded: {verification.is_grounded}, Score: {score:.1f}%")
                except Exception as e:
                    log.error(f"Verification call failed: {e}", exc_info=True)
                    final_result['answer'] += " [Verification Failed]"
            else: log.warning("Skipping verification because RAG context text is empty.")

        # Cache only if RAG, has embedding, and wasn't a cache hit.
        if is_rag_flow and query_embedding is not None and not cached_result:
            self._cache_result(cache_key, final_result, query_embedding, session_id)
            
        # Update History
        if session_id:
            history_list = self.chat_histories.setdefault(session_id, [])
            history_list.append({"query": raw_query, "answer": final_result.get("answer", "")})
            self.chat_histories[session_id] = history_list

        end_time = time.time()
        log.info(f"Agent loop finished in {end_time - start_time:.2f}s. Answer: '{final_result.get('answer', '')[:50]}...'")
        return final_result

    # Kept for signature, but streaming is not supported.
    def stream_agent_response(self, query: str, session_id: str):
        log.warning("Agent.stream_agent_response is NOT supported in this version.")
        raise NotImplementedError("Live token streaming requires native async components.")

    async def stream_agent_response_async(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, max_retries: int = 1, event_callback: Optional[Callable] = None):
        """
        Streaming version of the agent response that preserves all existing logic.
        Yields events for sub-queries, final answer tokens, and completion.
        """
        start_time = time.time()
        raw_query = query.strip()
        query_embedding = None
        final_result = {"answer": "An error occurred during processing.", "source_documents": []}
        cached_result = None

        if not raw_query:
            if event_callback: event_callback("error", {"message": "Please provide a query."})
            return {"answer": "Please provide a query.", "source_documents": []}
        
        if event_callback: event_callback("analyze", {"query": raw_query})
        
        history = self.chat_histories.get(session_id, []) if session_id else []
        route_decision = await self._triage_query_async(raw_query)
        contextual_query = self._format_query_with_history(raw_query, history)
        is_rag_flow = (route_decision == "USE_RAG")
        cache_key = self._get_cache_key(raw_query)
        
        if is_rag_flow:
            if hasattr(self.retrieval_pipeline, '_get_text_embedder'):
                text_embedder = self.retrieval_pipeline._get_text_embedder()
                if text_embedder:
                    try:
                        embeddings = await asyncio.to_thread(text_embedder.create_embeddings, [raw_query])
                        if embeddings is not None and len(embeddings) > 0:
                            query_embedding = np.array(embeddings[0])
                        else:
                            query_embedding = None
                    except Exception as e: 
                        log.error(f"Embedding failed: {e}", exc_info=True)
                        query_embedding = None
            
            cached_result = self._find_in_semantic_cache(query_embedding, session_id)
            if cached_result:
                final_result = cached_result
                if event_callback: event_callback("cache_hit", {"result": final_result})
            else:
                # RAG logic with streaming support
                if ai_rerank is not None and hasattr(self.retrieval_pipeline, 'config'):
                    log.info(f"Applying AI Rerank Override: {ai_rerank}")
                    rr_cfg = self.retrieval_pipeline.config.setdefault("reranker", {})
                    rr_cfg["enabled"] = bool(ai_rerank)

                final_result = await self._run_rag_pipeline_streaming_async(
                   raw_query, contextual_query, table_name, query_decompose, 
                   context_expand, max_retries, event_callback
                )
        else: # DIRECT_LLM Flow with streaming
            if event_callback: event_callback("direct_llm_start", {"query": contextual_query})
            final_answer = await self._answer_direct_llm_streaming_async(contextual_query, event_callback)
            final_result = {"answer": final_answer, "source_documents": []}

        # Verification (Only for RAG results NOT from cache)
        verify_enabled = self.pipeline_configs.get("verification", {}).get("enabled", True)
        if is_rag_flow and final_result.get("source_documents") and verify_enabled and not cached_result:
            context_str = "\n".join([doc.get('text', '') for doc in final_result['source_documents']]).strip()
            if context_str:
                log.info("Verifying final answer against sources...")
                if event_callback: event_callback("verification_start", {})
                try:
                    verification = await self.verifier.verify_async(raw_query, context_str, final_result['answer'])
                    score = verification.confidence_score
                    final_result['answer'] += f" [Confidence: {score:.1f}%]"
                    if not verification.is_grounded or score < 50: final_result['answer'] += " [Groundedness: Low]"
                    log.info(f"Verification complete. Grounded: {verification.is_grounded}, Score: {score:.1f}%")
                    if event_callback: event_callback("verification_done", {"score": score, "grounded": verification.is_grounded})
                except Exception as e:
                    log.error(f"Verification call failed: {e}", exc_info=True)
                    final_result['answer'] += " [Verification Failed]"
                    if event_callback: event_callback("verification_error", {"error": str(e)})
            else: 
                log.warning("Skipping verification because RAG context text is empty.")

        # Cache only if RAG, has embedding, and wasn't a cache hit.
        if is_rag_flow and query_embedding is not None and not cached_result:
            self._cache_result(cache_key, final_result, query_embedding, session_id)
            
        # Update History
        if session_id:
            history_list = self.chat_histories.setdefault(session_id, [])
            history_list.append({"query": raw_query, "answer": final_result.get("answer", "")})
            self.chat_histories[session_id] = history_list

        end_time = time.time()
        log.info(f"Agent loop finished in {end_time - start_time:.2f}s. Answer: '{final_result.get('answer', '')[:50]}...'")
        
        if event_callback: event_callback("complete", final_result)
        return final_result

    async def _answer_direct_llm_streaming_async(self, query_with_history: str, event_callback: Optional[Callable] = None) -> str:
        """Generates a direct answer using LLM with streaming support."""
        prompt = f"You are a helpful AI Assistant. Provide a direct, concise answer to the latest user query, using history for context: {query_with_history}"
        try:
            # Use fast operation thinking setting for streaming direct answers
            thinking_enabled = get_thinking_setting(self.fast_model, "fast")
            answer_parts = []
            async for token in self.llm_client.stream_completion_async(
                model=self.fast_model, 
                prompt=prompt,
                enable_thinking=thinking_enabled
            ):
                answer_parts.append(token)
                if event_callback: event_callback("token", {"text": token})
            
            return "".join(answer_parts)
        except Exception as e:
            log.error(f"Direct LLM streaming call failed: {e}", exc_info=True)
            return "I encountered an issue while generating an answer."

    async def _run_rag_pipeline_streaming_async(self, raw_query: str, contextual_query: str, table_name: Optional[str], query_decompose_override: Optional[bool], context_expand: Optional[bool], max_retries: int, event_callback: Optional[Callable]) -> Dict[str, Any]:
        """Streaming version of RAG pipeline that preserves all existing logic."""
        query_decomp_config = self.pipeline_configs.get("query_decomposition", {})
        decomp_enabled = query_decomp_config.get("enabled", False)
        if query_decompose_override is not None:
            decomp_enabled = query_decompose_override
            
        window_size = 0 if context_expand is False else None
        
        # Use contextual query if not decomposing, otherwise raw query for sub-queries
        query_to_run = contextual_query if not decomp_enabled else raw_query

        if not decomp_enabled:
            if event_callback: event_callback("single_query_start", {"query": query_to_run})
            return await self._run_rag_sub_query_streaming_async(query_to_run, table_name, window_size, max_retries, event_callback)

        log.info("RAG: Query Decomposition Enabled.")
        if event_callback: event_callback("decomposition_start", {"query": raw_query})
        
        sub_queries = await asyncio.to_thread(self.query_decomposer.decompose, raw_query)
        if not sub_queries: sub_queries = [raw_query]
        if event_callback: event_callback("decomposition", {"sub_queries": sub_queries})
        log.info(f"Decomposed into {len(sub_queries)} queries: {sub_queries}")

        if len(sub_queries) == 1:
            log.info("Only one sub-query; using direct RAG path.")
            if event_callback: event_callback("single_subquery", {"query": sub_queries[0]})
            return await self._run_rag_sub_query_streaming_async(sub_queries[0], table_name, window_size, max_retries, event_callback)

        log.info(f"Processing {len(sub_queries)} sub-queries in parallel...")
        if event_callback: event_callback("parallel_start", {"count": len(sub_queries)})
        
        # Process sub-queries in parallel but collect streaming events
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as executor:
            futures = [
                loop.run_in_executor(executor, self._run_rag_sub_query_sync, sq, table_name, window_size, max_retries) 
                for sq in sub_queries
            ]
            sub_results = await asyncio.gather(*futures, return_exceptions=True)
        
        sub_answers, all_source_docs, citations_seen = [], [], set()
        for i, res in enumerate(sub_results):
            sub_query = sub_queries[i]
            if isinstance(res, Exception):
                log.error(f"Sub-Query {i+1} ('{sub_query}') execution failed: {res}", exc_info=True)
                sub_answers.append({"question": sub_query, "answer": f"Error Processing Sub-Query."})
                if event_callback: event_callback("sub_query_error", {"index": i, "query": sub_query, "error": str(res)})
            else:
                log.info(f"Sub-Query {i+1} completed: '{sub_query}'")
                sub_answers.append({"question": sub_query, "answer": res.get("answer", "")})
                for doc in res.get("source_documents", []):
                    doc_id = doc.get('chunk_id')
                    if doc_id and doc_id not in citations_seen:
                        all_source_docs.append(doc); citations_seen.add(doc_id)
                if event_callback: event_callback("sub_query_result", {"index": i, "query": sub_query, "answer": res.get("answer", ""), "source_documents": res.get("source_documents", [])})

        log.info("Composing final answer from sub-answers...")
        if event_callback: event_callback("composition_start", {"sub_answers": sub_answers})
        
        compose_prompt = f"""You are an expert answer composer. Synthesize a single, cohesive answer from sub-answers. Use ONLY information from them.
Original Question: "{raw_query}"
Sub-Answers: {json.dumps(sub_answers, indent=2)}
Final Answer:"""
        
        # Stream the final answer composition - use reasoning mode for complex synthesis
        thinking_enabled = get_thinking_setting(self.gen_model, "reasoning")
        answer_parts = []
        async for token in self.llm_client.stream_completion_async(
            model=self.gen_model, 
            prompt=compose_prompt,
            enable_thinking=thinking_enabled
        ):
            answer_parts.append(token)
            if event_callback: event_callback("token", {"text": token})
        
        final_answer = "".join(answer_parts)
        
        result = {"answer": final_answer, "source_documents": all_source_docs}
        if event_callback: event_callback("final_answer", result)
        return result

    async def _run_rag_sub_query_streaming_async(self, sub_query: str, table_name: Optional[str], window_size: Optional[int], max_retries: int, event_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Streaming wrapper for individual sub-query processing."""
        for attempt in range(max_retries):
            try:
                # Create a streaming event callback wrapper for the retrieval pipeline
                def streaming_callback(event_type: str, data: Dict[str, Any]):
                    if event_callback:
                        event_callback(event_type, data)
                
                # Use the existing retrieval pipeline but with streaming callback
                return await asyncio.to_thread(
                    self.retrieval_pipeline.run, 
                    sub_query, 
                    table_name, 
                    window_size, 
                    streaming_callback
                )
            except Exception as e:
                log.warning(f"Sub-query '{sub_query}' attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt + 1 >= max_retries:
                    log.error(f"Sub-query '{sub_query}' FAILED after {max_retries} attempts.")
                    return {"answer": f"Failed to process: '{sub_query}'.", "source_documents": []}
                time.sleep(0.5)
        return {"answer": "Error: Max retries loop finished unexpectedly.", "source_documents": []}