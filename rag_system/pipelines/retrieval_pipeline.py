import pymupdf
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import concurrent.futures
import time
import json
import lancedb
import logging
import math
import numpy as np

from rag_system.utils.ollama_client import OllamaClient
from rag_system.retrieval.retrievers import MultiVectorRetriever, GraphRetriever
from rag_system.indexing.multimodal import LocalVisionModel
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.embedders import LanceDBManager
from rag_system.rerankers.reranker import QwenReranker
# from rag_system.indexing.chunk_store import ChunkStore

import os
from PIL import Image

class RetrievalPipeline:
    """
    Orchestrates the state-of-the-art multimodal RAG pipeline.
    """
    def __init__(self, config: Dict[str, Any], ollama_client: OllamaClient, ollama_config: Dict[str, Any]):
        self.config = config
        self.ollama_config = ollama_config
        self.ollama_client = ollama_client
        
        # Support both legacy "retrievers" key and newer "retrieval" key
        self.retriever_configs = self.config.get("retrievers") or self.config.get("retrieval", {})
        self.storage_config = self.config["storage"]
        
        # Defer initialization to just-in-time methods
        self.db_manager = None
        self.text_embedder = None
        self.dense_retriever = None
        self.bm25_retriever = None
        # Use a private attribute to avoid clashing with the public property
        self._graph_retriever = None
        self.reranker = None
        self.ai_reranker = None

    def _get_db_manager(self):
        if self.db_manager is None:
            # Accept either "db_path" (preferred) or legacy "lancedb_uri"
            db_path = self.storage_config.get("db_path") or self.storage_config.get("lancedb_uri")
            if not db_path:
                raise ValueError("Storage config must contain 'db_path' or 'lancedb_uri'.")
            self.db_manager = LanceDBManager(db_path=db_path)
        return self.db_manager

    def _get_text_embedder(self):
        if self.text_embedder is None:
            self.text_embedder = QwenEmbedder(
                model_name=self.config.get("embedding_model_name", "BAAI/bge-small-en-v1.5")
            )
        return self.text_embedder

    def _get_dense_retriever(self):
        """Ensure a dense MultiVectorRetriever is always available unless explicitly disabled."""
        if self.dense_retriever is None:
            # If the config explicitly sets dense.enabled to False, respect it
            if self.retriever_configs.get("dense", {}).get("enabled", True) is False:
                return None

            try:
                db_manager = self._get_db_manager()
                text_embedder = self._get_text_embedder()
                fusion_cfg = self.config.get("fusion", {})
                self.dense_retriever = MultiVectorRetriever(
                    db_manager,
                    text_embedder,
                    vision_model=None,
                    fusion_config=fusion_cfg,
                )
            except Exception as e:
                print(f"❌ Failed to initialise dense retriever: {e}")
                self.dense_retriever = None
        return self.dense_retriever

    def _get_bm25_retriever(self):
        if self.bm25_retriever is None and self.retriever_configs.get("bm25", {}).get("enabled"):
            try:
                print(f"🔧 Lazily initializing BM25 retriever...")
                self.bm25_retriever = BM25Retriever(
                    index_path=self.storage_config["bm25_path"],
                    index_name=self.retriever_configs["bm25"]["index_name"]
                )
                print("✅ BM25 retriever initialized successfully")
            except Exception as e:
                print(f"❌ Failed to initialize BM25 retriever on demand: {e}")
                # Keep it None so we don't try again
        return self.bm25_retriever

    def _get_graph_retriever(self):
        if self._graph_retriever is None and self.retriever_configs.get("graph", {}).get("enabled"):
            self._graph_retriever = GraphRetriever(graph_path=self.storage_config["graph_path"])
        return self._graph_retriever

    def _get_reranker(self):
        """Initializes the reranker for hybrid search score fusion."""
        reranker_config = self.config.get("reranker", {})
        # This is for the LanceDB internal reranker, not the AI one.
        if self.reranker is None and reranker_config.get("type") == "linear_combination":
            rerank_weight = reranker_config.get("weight", 0.5) 
            self.reranker = lancedb.rerankers.LinearCombinationReranker(weight=rerank_weight)
            print(f"✅ Initialized LinearCombinationReranker with weight {rerank_weight}")
        return self.reranker

    def _get_ai_reranker(self):
        """Initializes a dedicated AI-based reranker."""
        reranker_config = self.config.get("reranker", {})
        if self.ai_reranker is None and reranker_config.get("enabled"):
            try:
                model_name = reranker_config.get("model_name")
                strategy = reranker_config.get("strategy", "qwen")

                if strategy == "rerankers-lib":
                    print(f"🔧 Initialising Answer.AI ColBERT reranker ({model_name}) via rerankers lib…")
                    # The rerankers library now exposes a single generic `Reranker` class
                    # that auto-detects the appropriate backend (cross-encoder, ColBERT, etc.)
                    # We explicitly pass `model_type="colbert"` to guarantee late-interaction.
                    from rerankers import Reranker
                    self.ai_reranker = Reranker(model_name, model_type="colbert")
                else:
                    # Fallback to existing Qwen cross-encoder reranker
                    print(f"🔧 Lazily initializing Qwen reranker ({model_name})…")
                    self.ai_reranker = QwenReranker(model_name=model_name)

                print("✅ AI reranker initialized successfully.")
            except Exception as e:
                print(f"❌ Failed to initialize AI reranker: {e}")
        return self.ai_reranker

    def _get_surrounding_chunks_lancedb(self, chunk: Dict[str, Any], window_size: int) -> List[Dict[str, Any]]:
        """
        Retrieves a window of chunks around a central chunk using LanceDB.
        """
        db_manager = self._get_db_manager()
        if not db_manager:
            return [chunk]

        # Extract identifiers needed for the query
        document_id = chunk.get("document_id")
        chunk_index = chunk.get("chunk_index")

        # If essential identifiers are missing, return the chunk itself
        if document_id is None or chunk_index is None or chunk_index == -1:
            return [chunk]

        table_name = self.config["storage"]["text_table_name"]
        try:
            tbl = db_manager.get_table(table_name)
        except Exception:
            # If the table can't be opened, we can't get surrounding chunks
            return [chunk]

        # Define the window for the search
        start_index = max(0, chunk_index - window_size)
        end_index = chunk_index + window_size
        
        # Construct the SQL filter for an efficient metadata-based search
        sql_filter = f"document_id = '{document_id}' AND chunk_index >= {start_index} AND chunk_index <= {end_index}"
        
        try:
            # Execute a filter-only search, which is very fast on indexed metadata
            results = tbl.search().where(sql_filter).to_list()
            
            # The results must be sorted by chunk_index to maintain logical order
            results.sort(key=lambda c: c['chunk_index'])

            # The 'metadata' field is a JSON string and needs to be parsed
            for res in results:
                if isinstance(res.get('metadata'), str):
                    try:
                        res['metadata'] = json.loads(res['metadata'])
                    except json.JSONDecodeError:
                        res['metadata'] = {} # Handle corrupted metadata gracefully
            return results
        except Exception:
            # If the query fails for any reason, fall back to the single chunk
            return [chunk]

    def _synthesize_final_answer(self, query: str, facts: str) -> str:
        """Uses a text LLM to synthesize a final answer from extracted facts."""
        prompt = f"""
You are an AI assistant specialised in answering questions from retrieved context.

Context you receive
• VERIFIED FACTS – text snippets retrieved from the user's documents. Some may be irrelevant noise.  
• ORIGINAL QUESTION – the user's actual query.

Instructions
1. Evaluate each snippet for relevance to the ORIGINAL QUESTION; ignore those that do not help answer it.  
2. Synthesise an answer **using only information from the relevant snippets**.  
3. If snippets contradict one another, mention the contradiction explicitly.  
4. If the snippets do not contain the needed information, reply exactly with:  
   "I could not find that information in the provided documents."  
5. Provide a thorough, well-structured answer. Use paragraphs or bullet points where helpful, and include any relevant numbers/names exactly as they appear. There is **no strict sentence limit**, but aim for clarity over brevity.  
6. Do **not** introduce external knowledge unless step 4 applies; in that case you may add a clearly-labelled "General knowledge" sentence after the required statement.

Output format
Answer:
<your answer here>

–––––  Retrieved Snippets  –––––
{facts}
––––––––––––––––––––––––––––––

ORIGINAL QUESTION: "{query}"
"""
        response = self.ollama_client.generate_completion(
            model=self.ollama_config["generation_model"],
            prompt=prompt
        )
        return response.get('response', 'Failed to generate a final answer.')

    def run(self, query: str, table_name: str = None, window_size_override: Optional[int] = None) -> Dict[str, Any]:
        start_time = time.time()
        retrieval_k = self.config.get("retrieval_k", 10)

        logger = logging.getLogger(__name__)
        logger.debug("--- Running Hybrid Search for query '%s' (table=%s) ---", query, table_name or self.storage_config.get("text_table_name"))
        
        # If a custom table_name is provided, propagate it to storage config so helper methods use it
        if table_name:
            self.storage_config["text_table_name"] = table_name

        # Unified retrieval using the refactored MultiVectorRetriever
        dense_retriever = self._get_dense_retriever()
        # Get the LanceDB reranker for initial score fusion
        lancedb_reranker = self._get_reranker()
        
        retrieved_docs = []
        if dense_retriever:
            retrieved_docs = dense_retriever.retrieve(
                text_query=query,
                table_name=table_name or self.storage_config["text_table_name"],
                k=retrieval_k,
                reranker=lancedb_reranker # Pass the reranker to enable hybrid search
            )
        
        retrieval_time = time.time() - start_time
        logger.debug("Retrieved %s chunks in %.2fs", len(retrieved_docs), retrieval_time)

        # --- AI Reranking Step ---
        ai_reranker = self._get_ai_reranker()
        if ai_reranker and retrieved_docs:
            print(f"\n--- Reranking top {len(retrieved_docs)} docs with AI model... ---")
            start_rerank_time = time.time()

            rerank_cfg = self.config.get("reranker", {})
            top_k_cfg = rerank_cfg.get("top_k")
            top_percent = rerank_cfg.get("top_percent")  # value in range 0–1

            if top_percent is not None:
                try:
                    pct = float(top_percent)
                    assert 0 < pct <= 1
                    top_k = max(1, int(len(retrieved_docs) * pct))
                except Exception:
                    print("⚠️  Invalid top_percent value; falling back to top_k")
                    top_k = top_k_cfg or len(retrieved_docs)
            else:
                top_k = top_k_cfg or len(retrieved_docs)

            strategy = self.config.get("reranker", {}).get("strategy", "qwen")

            if strategy == "rerankers-lib":
                texts = [d['text'] for d in retrieved_docs]
                ranked = ai_reranker.rank(query=query, docs=texts)
                # ranked is RankedResults; convert to list of (score, idx)
                try:
                    pairs = [(r.score, r.document.doc_id) for r in ranked.results]
                    if any(p[1] is None for p in pairs):
                        pairs = [(r.score, i) for i, r in enumerate(ranked.results)]
                except Exception:
                    pairs = ranked
                # Keep only top_k results if requested
                if top_k is not None and len(pairs) > top_k:
                    pairs = pairs[:top_k]
                reranked_docs = [retrieved_docs[idx] | {"rerank_score": score} for score, idx in pairs]
            else:
                try:
                    reranked_docs = ai_reranker.rerank(query, retrieved_docs, top_k=top_k)
                except TypeError:
                    texts = [d['text'] for d in retrieved_docs]
                    pairs = ai_reranker.rank(query, texts, top_k=top_k)
                    reranked_docs = [retrieved_docs[idx] | {"rerank_score": score} for score, idx in pairs]

            rerank_time = time.time() - start_rerank_time
            print(f"✅ Reranking completed in {rerank_time:.2f}s. Refined to {len(reranked_docs)} docs.")
        else:
            # If no AI reranker, proceed with the initially retrieved docs
            reranked_docs = retrieved_docs

        window_size = self.config.get("context_window_size", 1)
        if window_size_override is not None:
            window_size = window_size_override
        if window_size > 0 and reranked_docs:
            print(f"\n--- Expanding context for {len(reranked_docs)} top documents (window size: {window_size})... ---")
            expanded_chunks = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_chunk = {executor.submit(self._get_surrounding_chunks_lancedb, chunk, window_size): chunk for chunk in reranked_docs}
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        surrounding_chunks = future.result()
                        for surrounding_chunk in surrounding_chunks:
                            if surrounding_chunk['chunk_id'] not in expanded_chunks:
                                expanded_chunks[surrounding_chunk['chunk_id']] = surrounding_chunk
                    except Exception as e:
                        print(f"Error expanding context for a chunk: {e}")

            final_docs = list(expanded_chunks.values())
            # Sort by reranker score if present, otherwise by raw score/distance
            if any('rerank_score' in d for d in final_docs):
                final_docs.sort(key=lambda c: c.get('rerank_score', -1), reverse=True)
            elif any('_distance' in d for d in final_docs):
                # For vector search smaller distance is better
                final_docs.sort(key=lambda c: c.get('_distance', 1e9))
            elif any('score' in d for d in final_docs):
                final_docs.sort(key=lambda c: c.get('score', 0), reverse=True)
            else:
                # Fallback to document order
                final_docs.sort(key=lambda c: (c.get('document_id', ''), c.get('chunk_index', 0)))

            print(f"Expanded to {len(final_docs)} unique chunks for synthesis.")
        else:
            final_docs = reranked_docs

        # Optionally hide non-reranked chunks: if any chunk carries a
        # `rerank_score`, we assume the caller wants to focus on those.
        if any('rerank_score' in d for d in final_docs):
            final_docs = [d for d in final_docs if 'rerank_score' in d]

        print("\n--- Final Documents for Synthesis ---")
        if not final_docs:
            print("No documents to synthesize.")
        else:
            for i, doc in enumerate(final_docs):
                print(f"  [{i+1}] Chunk ID: {doc.get('chunk_id')}")
                print(f"      Score: {doc.get('score', 'N/A')}")
                if 'rerank_score' in doc:
                    print(f"      Rerank Score: {doc.get('rerank_score'):.4f}")
                print(f"      Text: \"{doc.get('text', '').strip()}\"")
        print("------------------------------------")

        if not final_docs:
            return {"answer": "I could not find an answer in the documents.", "source_documents": []}
        
        # --- Sanitize docs for JSON serialization (no NaN/Inf types) ---
        def _clean_val(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            if isinstance(v, (np.floating,)):
                try:
                    f = float(v)
                    if math.isnan(f) or math.isinf(f):
                        return None
                    return f
                except Exception:
                    return None
            return v

        for doc in final_docs:
            for key in ['score', '_distance', 'rerank_score']:
                if key in doc:
                    doc[key] = _clean_val(doc[key])

        context = "\n\n".join([doc['text'] for doc in final_docs])
        final_answer = self._synthesize_final_answer(query, context)
        
        return {"answer": final_answer, "source_documents": final_docs}

    # ------------------------------------------------------------------
    # Public utility
    # ------------------------------------------------------------------
    def list_document_titles(self, max_items: int = 25) -> List[str]:
        """Return up to *max_items* distinct document titles (or IDs).

        This is used only for prompt-routing, so we favour robustness over
        perfect recall. If anything goes wrong we return an empty list so
        the caller can degrade gracefully.
        """
        try:
            tbl_name = self.storage_config.get("text_table_name")
            if not tbl_name:
                return []

            tbl = self._get_db_manager().get_table(tbl_name)

            field_name = "document_title" if "document_title" in tbl.schema.names else "document_id"

            # Use a cheap SQL filter to grab distinct values; fall back to a
            # simple scan if the driver lacks DISTINCT support.
            try:
                sql = f"SELECT DISTINCT {field_name} FROM tbl LIMIT {max_items}"
                rows = tbl.search().where("true").sql(sql).to_list()  # type: ignore
                titles = [r[field_name] for r in rows if r.get(field_name)]
            except Exception:
                # Fallback: scan first N rows
                rows = tbl.search().select(field_name).limit(max_items * 4).to_list()
                seen = set()
                titles = []
                for r in rows:
                    val = r.get(field_name)
                    if val and val not in seen:
                        titles.append(val)
                        seen.add(val)
                        if len(titles) >= max_items:
                            break

            # Ensure we don't exceed max_items
            return titles[:max_items]
        except Exception:
            # Any issues (missing table, bad schema, etc.) –> just return []
            return []

    # -------------------- Public helper properties --------------------
    @property
    def retriever(self):
        """Lazily exposes the main (dense) retriever so external components
        like the ReAct agent tools can call `.retrieve()` directly without
        reaching into private helpers. If the retriever has not yet been
        instantiated, it is created on first access via `_get_dense_retriever`."""
        return self._get_dense_retriever()

    @property
    def graph_retriever(self):
        """Lazily exposes the graph retriever (if configured)."""
        return self._get_graph_retriever()
