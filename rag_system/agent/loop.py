from typing import Dict, Any, Optional
import json
import concurrent.futures
import time
import asyncio
from cachetools import TTLCache, LRUCache
import numpy as np
from rag_system.utils.ollama_client import OllamaClient
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.agent.verifier import Verifier
from rag_system.retrieval.query_transformer import QueryDecomposer, GraphQueryTranslator
from rag_system.retrieval.retrievers import GraphRetriever
import os
import json as _json

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
        
        # 🚀 OPTIMIZED: TTL cache now stores embeddings for semantic matching
        self._cache_max_size = 100  # fallback size limit for manual eviction helper
        self._query_cache: TTLCache = TTLCache(maxsize=self._cache_max_size, ttl=300)
        self.semantic_cache_threshold = self.pipeline_configs.get("semantic_cache_threshold", 0.98)
        # If set to "session", semantic-cache hits will be restricted to the same chat session.
        # Otherwise (default "global") answers can be reused across sessions.
        self.cache_scope = self.pipeline_configs.get("cache_scope", "global")  # 'global' or 'session'
        
        # 🚀 NEW: In-memory store for conversational history per session
        self.chat_histories: LRUCache = LRUCache(maxsize=100) # Stores history for 100 recent sessions

        graph_config = self.pipeline_configs.get("graph_strategy", {})
        if graph_config.get("enabled"):
            self.graph_query_translator = GraphQueryTranslator(llm_client, gen_model)
            self.graph_retriever = GraphRetriever(graph_config["graph_path"])
            print("Agent initialized with live GraphRAG capabilities.")
        else:
            print("Agent initialized (GraphRAG disabled).")

        # ---- Load document overviews for fast routing ----
        overview_path = os.path.join("index_store", "overviews", "overviews.jsonl")
        self.doc_overviews: list[str] = []
        if os.path.exists(overview_path):
            try:
                with open(overview_path, encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = _json.loads(line)
                            if isinstance(rec, dict) and rec.get("overview"):
                                self.doc_overviews.append(rec["overview"].strip())
                        except Exception:
                            continue
            except Exception as e:
                print(f"⚠️  Failed to load document overviews: {e}")

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        if not isinstance(v1, np.ndarray): v1 = np.array(v1)
        if not isinstance(v2, np.ndarray): v2 = np.array(v2)
        
        if v1.shape != v2.shape:
            raise ValueError("Vectors must have the same shape for cosine similarity.")

        if np.all(v1 == 0) or np.all(v2 == 0):
            return 0.0
            
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)

    def _find_in_semantic_cache(self, query_embedding: np.ndarray, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Finds a semantically similar query in the cache."""
        if not self._query_cache or query_embedding is None:
            return None

        for key, cached_item in self._query_cache.items():
            cached_embedding = cached_item.get('embedding')
            if cached_embedding is None:
                continue

            # Respect cache scoping: if scope is session-level, skip results from other sessions
            if self.cache_scope == "session" and session_id is not None:
                if cached_item.get("session_id") != session_id:
                    continue

            try:
                similarity = self._cosine_similarity(query_embedding, cached_embedding)

                if similarity >= self.semantic_cache_threshold:
                    print(f"🚀 Semantic cache hit! Similarity: {similarity:.3f} with cached query '{key}'")
                    return cached_item.get('result')
            except ValueError:
                # In case of shape mismatch, just skip
                continue

        return None

    def _format_query_with_history(self, query: str, history: list) -> str:
        """Formats the user query with conversation history for context."""
        if not history:
            return query
        
        formatted_history = "\n".join([f"User: {turn['query']}\nAssistant: {turn['answer']}" for turn in history])
        
        prompt = f"""
Given the following conversation history, answer the user's latest query. The history provides context for resolving pronouns or follow-up questions.

--- Conversation History ---
{formatted_history}
---

Latest User Query: "{query}"
"""
        return prompt

    # ---------------- Asynchronous triage using Ollama ----------------
    async def _triage_query_async(self, query: str, history: list) -> str:
        """
        The main triage entrypoint. Defers to the overview-based router first,
        then falls back to a simpler LLM-based triage if no overviews are available.
        """
        
        # 1️⃣ Fast, context-aware routing using precomputed overviews (if available)
        if self.doc_overviews:
            # Add history context to the query before sending to the router
            contextual_query = self._format_query_with_history(query, history)
            return self._route_via_overviews(contextual_query)

        # 2️⃣ Fallback to a simpler, general-purpose router if no overviews exist
        if history:
            # If there's history, the query is likely a follow-up, so we default to RAG.
            # A more advanced implementation could use an LLM to see if the new query
            # changes the topic entirely.
            return "rag_query"

        prompt = f"""
You are a query routing expert. Analyse the user's question and decide which backend should handle it. Choose **exactly one** of the following categories:

1. "graph_query" – The user asks for a specific factual relation best served by a knowledge-graph lookup (e.g. "Who is the CEO of Apple?", "Which company acquired DeepMind?").
2. "rag_query" – The answer is most likely inside the user's uploaded documents (reports, PDFs, slide decks, invoices, research papers, etc.). Examples: "What is the total on invoice 1041?", "Summarise the Q3 earnings report".
3. "direct_answer" – General chit-chat or open-domain knowledge that does **not** rely on the user's private documents or the knowledge graph (e.g. "Hello", "What is the capital of France?", "Explain quantum entanglement").

Respond with JSON of the form: {{"category": "<your_choice>"}}

User query: "{query}"
"""
        resp = self.llm_client.generate_completion(
            model=self.ollama_config["generation_model"], prompt=prompt, format="json"
        )
        try:
            data = json.loads(resp.get("response", "{}"))
            return data.get("category", "rag_query")
        except json.JSONDecodeError:
            return "rag_query"

    def _run_graph_query(self, query: str, history: list) -> Dict[str, Any]:
        contextual_query = self._format_query_with_history(query, history)
        structured_query = self.graph_query_translator.translate(contextual_query)
        if not structured_query.get("start_node"):
            return self.retrieval_pipeline.run(contextual_query, window_size_override=0)
        results = self.graph_retriever.retrieve(structured_query)
        if not results:
            return self.retrieval_pipeline.run(contextual_query, window_size_override=0)
        answer = ", ".join([res['details']['node_id'] for res in results])
        return {"answer": f"From the knowledge graph: {answer}", "source_documents": results}

    def _get_cache_key(self, query: str, query_type: str) -> str:
        """Generate a cache key for the query"""
        # Simple cache key based on query and type
        return f"{query_type}:{query.strip().lower()}"
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any], session_id: Optional[str] = None):
        """Cache a result with size limit"""
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time(),
            'session_id': session_id
        }

    # ---------------- Public sync API (kept for backwards compatibility) --------------
    def run(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, max_retries: int = 1, event_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Synchronous helper. If *event_callback* is supplied, important
        milestones will be forwarded to that callable as

            event_callback(phase:str, payload:Any)
        """
        return asyncio.run(self._run_async(query, table_name, session_id, compose_sub_answers, query_decompose, ai_rerank, context_expand, max_retries, event_callback))

    # ---------------- Main async implementation --------------------------------------
    async def _run_async(self, query: str, table_name: str = None, session_id: str = None, compose_sub_answers: Optional[bool] = None, query_decompose: Optional[bool] = None, ai_rerank: Optional[bool] = None, context_expand: Optional[bool] = None, max_retries: int = 1, event_callback: Optional[callable] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        # Emit analyze event at the start
        if event_callback:
            event_callback("analyze", {"query": query})
        
        # 🚀 NEW: Get conversation history
        history = self.chat_histories.get(session_id, []) if session_id else []
        
        # The triage function now receives the raw query and the history separately
        query_type = await self._triage_query_async(query, history)
        print(f"Agent Triage Decision: '{query_type}'")
        
        # NEW: Decide early whether this turn needs any retrieval work
        needs_rag = query_type in ("rag_query", "graph_query")
        
        # Create a contextual query that includes history for most operations
        contextual_query = self._format_query_with_history(query, history)
        raw_query = query.strip()
        
        # --- Apply runtime AI reranker override (must happen before any retrieval calls) ---
        if ai_rerank is not None and needs_rag:
            rr_cfg = self.retrieval_pipeline.config.setdefault("reranker", {})
            rr_cfg["enabled"] = bool(ai_rerank)
            if ai_rerank:
                rr_cfg.setdefault("type", "ai")
                rr_cfg.setdefault("strategy", "rerankers-lib")
                rr_cfg.setdefault(
                    "model_name",
                    self.ollama_config.get("rerank_model", "answerai-colbert-small-v1"),
                )

        query_embedding = None
        # 🚀 OPTIMIZED: Semantic Cache Check – only if we plan to run retrieval
        if needs_rag:
            text_embedder = self.retrieval_pipeline._get_text_embedder()
            if text_embedder:
                query_embedding_list = text_embedder.create_embeddings([raw_query])
                if isinstance(query_embedding_list, np.ndarray):
                    query_embedding = query_embedding_list[0]
                else:
                    query_embedding = np.array(query_embedding_list[0])

                cached_result = self._find_in_semantic_cache(query_embedding, session_id)
                if cached_result:
                    self.chat_histories.setdefault(session_id, []).append({"query": raw_query, "answer": cached_result["answer"]})
                    return cached_result

        # --- Route based on triage decision ---
        # A bit of a hack: if the triage returns something other than a known category,
        # it's the direct answer itself, synthesized from the overviews.
        if query_type == "rag_query":
            final_result = await self._run_rag_pipeline(
                contextual_query, table_name, compose_sub_answers, 
                query_decompose, ai_rerank, context_expand, 
                max_retries, event_callback
            )
        elif query_type == "graph_query":
            final_result = self._run_graph_query(query, history)
        elif query_type == "clarification":
            final_result = {"answer": "I'm sorry, your query is a bit ambiguous. Could you please clarify?", "source_documents": []}
        else:  # The triage router already produced the final answer (general chat or direct answer)
            direct_ans = query_type  # this string holds the answer text itself
            # Stream it token-by-token if the caller provided an event callback
            if event_callback:
                for token in direct_ans.split():
                    event_callback("token", {"text": token + " "})
            final_result = {"answer": direct_ans, "source_documents": []}

        # --- Verification Step ---
        # Only run verification if the result came from a RAG pipeline and has source documents.
        if self.pipeline_configs.get("verification", {}).get("enabled", True) and final_result.get("source_documents"):
            context_str = "\n".join([doc['text'] for doc in final_result['source_documents']])
            if context_str: # Ensure context is not empty
                print("\n--- Verifying final answer against sources ---")
                verification = await self.verifier.verify_async(contextual_query, context_str, final_result['answer'])
                
                # Append confidence score to the answer
                score = verification.confidence_score
                final_result['answer'] += f" [Confidence: {score}%]"
                
                if not verification.is_grounded or score < 50:
                     final_result['answer'] += f" [Warning: Low confidence. Groundedness: {verification.is_grounded}]"
                print(f"✅ Verification complete. Grounded: {verification.is_grounded}, Score: {score}%")
            else:
                print("⚠️ Skipping verification because context from source documents is empty.")
        else:
            print("- Skipping verification step (not applicable or disabled).")

        # Cache the final result (using raw query as key)
        if needs_rag and query_embedding is not None:
            cache_key = self._get_cache_key(raw_query, query_type)
            result_to_cache = final_result.copy()
            result_to_cache['embedding'] = query_embedding
            self._cache_result(cache_key, result_to_cache, session_id)
            
        # Add the final answer to the conversation history
        if session_id:
            self.chat_histories.setdefault(session_id, []).append({"query": raw_query, "answer": final_result["answer"]})

        end_time = time.time()
        print(f"Agent loop took {end_time - start_time:.2f} seconds.")
        return final_result

    async def _run_rag_pipeline(self, query: str, table_name: str, compose_sub_answers: bool, query_decompose: bool, ai_rerank: bool, context_expand: bool, max_retries: int, event_callback: callable) -> Dict[str, Any]:
        """Helper to run the RAG pipeline with all its options."""
        
        query_decomp_config = self.pipeline_configs.get("query_decomposition", {})
        decomp_enabled = query_decomp_config.get("enabled", False)
        if query_decompose is not None:
            decomp_enabled = query_decompose

        if not decomp_enabled:
            # If decomposition is disabled, run a standard single retrieval.
            window_size = 0 if context_expand is False else None
            return await asyncio.to_thread(
                self.retrieval_pipeline.run,
                query=query,
                table_name=table_name,
                window_size_override=window_size,
                event_callback=event_callback
            )

        # --- Query Decomposition is Enabled ---
        print(f"\n--- Query Decomposition Enabled ---")
        # Use the raw user query (without conversation history) for decomposition to avoid leakage of prior context
        # Note: the `query` param to this function is already the contextual_query from the main loop.
        # We need the original raw query for decomposition. This is a bit awkward. Let's assume for now
        # the contextual query is okay for decomposition, but this could be improved.
        sub_queries = self.query_decomposer.decompose(query)
        if event_callback:
            event_callback("decomposition", {"sub_queries": sub_queries})
        print(f"Decomposed query into {len(sub_queries)} sub-queries: {sub_queries}")

        # If decomposition produced only a single sub-query, skip parallel execution.
        if len(sub_queries) == 1:
            print("--- Only one sub-query; using direct retrieval path ---")
            window_size = 0 if context_expand is False else None
            result = await asyncio.to_thread(
                self.retrieval_pipeline.run,
                sub_queries[0],
                table_name,
                window_size_override=window_size,
                event_callback=event_callback
            )
            if event_callback:
                event_callback("single_query_result", result)
            return result

        # --- Parallel Sub-Query Execution ---
        compose_from_sub_answers = query_decomp_config.get("compose_from_sub_answers", True)
        if compose_sub_answers is not None:
            compose_from_sub_answers = compose_sub_answers

        print(f"\n--- Processing {len(sub_queries)} sub-queries in parallel ---")
        start_time_inner = time.time()

        sub_answers = []
        all_source_docs = []
        citations_seen = set()

        def make_cb(idx: int):
            def _cb(ev_type: str, payload):
                if event_callback:
                    if ev_type == "token":
                        event_callback("sub_query_token", {"index": idx, "text": payload.get("text", ""), "question": sub_queries[idx]})
                    else:
                        event_callback(ev_type, payload)
            return _cb

        # This must run in a thread pool because the underlying pipeline run is sync
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as executor:
            window_size = 0 if context_expand is False else None
            future_to_query = {
                executor.submit(
                    self.retrieval_pipeline.run,
                    sub_query,
                    table_name,
                    window_size,
                    make_cb(i),
                ): (i, sub_query)
                for i, sub_query in enumerate(sub_queries)
            }

            for future in concurrent.futures.as_completed(future_to_query):
                i, sub_query = future_to_query[future]
                try:
                    sub_result = future.result()
                    print(f"✅ Sub-Query {i+1} completed: '{sub_query}'")
                    if event_callback:
                        event_callback("sub_query_result", {"index": i, "query": sub_query, "answer": sub_result.get("answer", ""), "source_documents": sub_result.get("source_documents", [])})
                    
                    sub_answers.append({"question": sub_query, "answer": sub_result.get("answer", "")})
                    for doc in sub_result.get("source_documents", [])[:5]:
                        if doc['chunk_id'] not in citations_seen:
                            all_source_docs.append(doc)
                            citations_seen.add(doc['chunk_id'])
                except Exception as e:
                    print(f"❌ Sub-Query {i+1} failed: '{sub_query}' - {e}")
        
        parallel_time = time.time() - start_time_inner
        print(f"🚀 Parallel processing completed in {parallel_time:.2f}s")
        
        # --- Final Answer Composition ---
        print("\n--- Composing final answer from sub-answers ---")
        compose_prompt = f"""
You are an expert answer composer. Your task is to synthesize a single, cohesive answer from a set of question-answer pairs generated by a previous step.
- Use ONLY the information from the provided sub-answers.
- If the original question involves a comparison, clearly state the outcome.
- If a sub-answer is not relevant, ignore it.
- If the sub-answers are insufficient, state that the information could not be found.

Original Question: "{query}"
Sub-Answers:
{json.dumps(sub_answers, indent=2)}

Final Answer:
"""
        final_answer_parts = []
        for tok in self.llm_client.stream_completion(model=self.ollama_config["generation_model"], prompt=compose_prompt):
            final_answer_parts.append(tok)
            if event_callback:
                event_callback("token", {"text": tok})
        
        final_answer = "".join(final_answer_parts) or "Unable to generate a final answer."
        result = {"answer": final_answer, "source_documents": all_source_docs}
        if event_callback:
            event_callback("final_answer", result)
            
        return result

    def _route_via_overviews(self, query: str) -> str | None:
        """
        An intelligent router that uses LLM reasoning on document overviews
        to make a fast, high-quality decision.
        """
        if not self.doc_overviews: return None # Should not happen if called correctly

        # Present the overviews to the LLM
        overview_text = "\n\n".join(f"--- Overview {i+1} ---\n{o}" for i, o in enumerate(self.doc_overviews))

        prompt = f"""
You are an expert routing agent. Your job is to decide how to answer a user's query based on a set of available document summaries. You must choose one of four actions:

1.  **RAG Query**: The user's query requires retrieving specific information from the full text of one or more documents. This is the best choice if the query asks for details, quotes, or mentions "the document", "the paper", "the invoice", etc. It's the safest default when in doubt.

2.  **Direct Answer**: The provided overviews ALREADY contain a complete and sufficient answer to the user's query. This is a shortcut for simple factual questions where the summary is enough and no further reading is needed.

3.  **General Chat**: The user's query is conversational, a general knowledge question, or completely unrelated to the documents provided. Examples: "Hello", "What's the weather like?", "Who was the first person on the moon?".

4.  **Needs Clarification**: The user's query is ambiguous, or it's unclear which document it refers to when multiple documents seem relevant.

**Your thought process:**
1.  Analyze the user's query to understand their intent. Are they asking a factual question, a summarization, a specific detail, or just chatting?
2.  Read the provided overviews. Do they seem relevant to the query?
3.  Decide:
    *   If the query explicitly asks for information "according to the document" or for specific details not present in the overview, choose **RAG Query**.
    *   If the answer is clearly and fully present in the overviews, choose **Direct Answer**.
    *   If the query is unrelated to the documents, choose **General Chat**.
    *   If the overviews do **not** contain the requested fact—or contain it only partially—choose **RAG Query**. Do **NOT** return a "Direct Answer" that simply says "information not provided".
    *   If you are unsure or the query is vague, choose **Needs Clarification**.
    *   If the query is a general knowledge question, choose **General Chat**.
    *   If the overview has information about the query but not specifically answers the query, choose **RAG Query**.

Here are the available document overviews:
{overview_text}

---

Now, analyze the following user query and choose the best action.

**User Query**: "{query}"

Respond with a single JSON object with two keys: "action" (one of ["rag_query", "direct_answer", "general_chat", "needs_clarification"]) and "reasoning" (a brief explanation of your choice).
"""

        # Call the LLM
        resp = self.llm_client.generate_completion(
            model=self.ollama_config["generation_model"],
            prompt=prompt,
            format="json"
        )
        
        try:
            choice_data = _json.loads(resp.get("response", "{}"))
            action = choice_data.get("action")
            reasoning = choice_data.get("reasoning", "No reasoning provided.")
            print(f"Overview Router Decision: '{action}'. Reason: {reasoning}")

            if action == "rag_query":
                return "rag_query"
            elif action == "direct_answer":
                # For a direct answer, we need the LLM to synthesize it from the overviews.
                answer = self._answer_from_overviews(query, overview_text)
                # Heuristic fallback: if the answer basically says information is missing, reroute to rag_query
                neg_phrases = [
                    "not provided",
                    "not mentioned",
                    "not available",
                    "cannot find",
                    "could not find",
                    "no direct",
                ]
                ans_lc = answer.lower()
                if any(p in ans_lc for p in neg_phrases):
                    print("🔄 Overview answer indicates missing info – falling back to full RAG query…")
                    return "rag_query"
                return answer
            elif action == "general_chat":
                # Generate a free-form answer using general knowledge (no document constraints)
                return self._answer_general_chat(query)
            elif action == "needs_clarification":
                return "clarification"
        except Exception:
            # If JSON parsing or logic fails, default to the safest option
            return "rag_query"

        return "rag_query" # Fallback

    def _answer_from_overviews(self, query: str, overview_text: str) -> str:
        """
        If the router decides the answer is in the overviews, this function
        generates that answer.
        """
        prompt = f"""
Given the following document summaries, please provide a direct and concise answer to the user's query.
Use ONLY the information present in the summaries. Do not add any external knowledge.

Summaries:
{overview_text}

---
User Query: {query}

Answer:
"""
        resp = self.llm_client.generate_completion(
            model=self.ollama_config["generation_model"],
            prompt=prompt
        )
        # The triage router returns the answer *itself* as the category.
        # The main loop will then wrap this in the final response dict.
        return resp.get("response", "I am sorry, I could not formulate an answer.")

    def _answer_general_chat(self, query: str) -> str:
        """
        Generates a free-form answer using general knowledge (no document constraints)
        """
        prompt = f"""
You are a helpful general-purpose AI assistant. Answer the following user query using general knowledge.

User Query: "{query}"

Answer:
"""
        resp = self.llm_client.generate_completion(
            model=self.ollama_config["generation_model"],
            prompt=prompt
        )
        return resp.get("response", "I am sorry, I could not formulate an answer.")

    def stream_agent_response(self, query: str, session_id: str):
        """High-level streaming entry point."""
        # ... existing code ...
