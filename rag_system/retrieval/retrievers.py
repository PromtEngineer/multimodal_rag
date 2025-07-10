import lancedb
import pickle
import json
from typing import List, Dict, Any
import numpy as np
import networkx as nx
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import logging
import pandas as pd
import math
import concurrent.futures
from functools import lru_cache

from rag_system.indexing.embedders import LanceDBManager
from rag_system.indexing.representations import QwenEmbedder
from rag_system.indexing.multimodal import LocalVisionModel
from rag_system.utils.logging_utils import log_retrieval_results
from rag_system.retrieval.query_transformer import MultiHopContext

# BM25Retriever is no longer needed.
# class BM25Retriever: ...

from fuzzywuzzy import process

class MultiHopRetriever:
    """
    Performs multi-hop retrieval with sequential dependency-aware steps.
    Now includes optional reranking and pruning for each step.
    """
    
    def __init__(self, base_retriever, text_embedder: QwenEmbedder, synthesis_llm_client, synthesis_model: str, 
                 reranker=None, pruner=None, pipeline_config=None):
        self.base_retriever = base_retriever
        self.text_embedder = text_embedder
        self.synthesis_llm_client = synthesis_llm_client
        self.synthesis_model = synthesis_model
        self.reranker = reranker
        self.pruner = pruner
        self.pipeline_config = pipeline_config or {}
    
    def retrieve_multihop(self, context: MultiHopContext, table_name: str, k: int = 10, event_callback=None) -> Dict[str, Any]:
        """
        Execute multi-hop retrieval following the dependency graph.
        """
        print(f"\n--- Multi-Hop Retrieval Started ---")
        print(f"Total steps: {len(context.steps)}")
        
        all_results = {}
        all_source_docs = []
        
        if event_callback:
            event_callback("multihop_started", {
                "total_steps": len(context.steps),
                "steps": [{"id": s['id'], "query": s['query'], "dependencies": s['dependencies']} for s in context.steps]
            })
        
        # Execute steps in dependency order
        step_number = 0
        while True:
            ready_steps = context.get_next_ready_steps()
            if not ready_steps:
                break
            
            print(f"Executing ready steps: {ready_steps}")
            
            # Execute ready steps (can be parallel if independent)
            for step_id in ready_steps:
                step = next(s for s in context.steps if s['id'] == step_id)
                step_query = step['query']
                step_number += 1
                
                print(f"\n--- Executing Step {step_id}: {step_query} ---")
                
                if event_callback:
                    event_callback("multihop_step_started", {
                        "step_id": step_id,
                        "step_number": step_number,
                        "total_steps": len(context.steps),
                        "query": step_query,
                        "dependencies": step['dependencies'],
                        "status": "retrieving"
                    })
                
                # Get context from previous steps
                accumulated_context = context.get_context_for_step(step_id)
                
                # Enhance query with accumulated context if available
                if accumulated_context:
                    enhanced_query = f"""
Context from previous steps:
{accumulated_context}

Current question: {step_query}

Please answer the current question using the provided context and any additional relevant information.
"""
                    print(f"Enhanced query with context: {enhanced_query[:200]}...")
                else:
                    enhanced_query = step_query
                    print(f"No context available, using original query: {step_query}")
                
                if event_callback:
                    event_callback("multihop_step_retrieving", {
                        "step_id": step_id,
                        "step_number": step_number,
                        "enhanced_query": enhanced_query,
                        "has_context": bool(accumulated_context)
                    })
                
                # Perform retrieval for this step
                print(f"\n--- Performing Retrieval for query: '{enhanced_query[:100]}...' on table '{table_name}' ---")
                try:
                    retrieved_docs = self.base_retriever.retrieve(
                        text_query=enhanced_query,
                        table_name=table_name,
                        k=k
                    )
                    
                    if event_callback:
                        event_callback("multihop_step_retrieved", {
                            "step_id": step_id,
                            "step_number": step_number,
                            "documents_found": len(retrieved_docs),
                            "status": "processing"
                        })
                    
                    # Apply reranking if enabled
                    if self.reranker and self.pipeline_config.get("reranker", {}).get("enabled", False):
                        print(f"🔄 Applying reranking to {len(retrieved_docs)} documents...")
                        if event_callback:
                            event_callback("multihop_step_reranking", {
                                "step_id": step_id,
                                "step_number": step_number,
                                "documents_count": len(retrieved_docs),
                                "status": "reranking"
                            })
                        
                        strategy = self.pipeline_config.get("reranker", {}).get("strategy", "qwen")
                        rerank_cfg = self.pipeline_config.get("reranker", {})
                        top_k = rerank_cfg.get("top_k", len(retrieved_docs))
                        
                        if strategy == "rerankers-lib":
                            # ColBERT from rerankers library
                            texts = [d['text'] for d in retrieved_docs]
                            # Use the same lock as the main pipeline for thread safety
                            from rag_system.pipelines.retrieval_pipeline import _rerank_lock
                            with _rerank_lock:
                                ranked = self.reranker.rank(query=enhanced_query, docs=texts)
                            # Convert RankedResults to reranked documents
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
                            # Local QwenReranker
                            try:
                                reranked_docs = self.reranker.rerank(enhanced_query, retrieved_docs, top_k=top_k)
                            except (TypeError, AttributeError):
                                # Fallback if rerank method signature is different
                                texts = [d['text'] for d in retrieved_docs]
                                pairs = self.reranker.rank(enhanced_query, texts, top_k=top_k)
                                reranked_docs = [retrieved_docs[idx] | {"rerank_score": score} for score, idx in pairs]
                        
                        retrieved_docs = reranked_docs
                        print(f"✅ Reranking completed: {len(retrieved_docs)} documents")
                        
                        if event_callback:
                            event_callback("multihop_step_reranked", {
                                "step_id": step_id,
                                "step_number": step_number,
                                "reranked_count": len(retrieved_docs),
                                "status": "reranked"
                            })
                    
                    # Apply pruning if enabled
                    if self.pruner and self.pipeline_config.get("provence", {}).get("enabled", False):
                        print(f"✂️ Applying pruning to {len(retrieved_docs)} documents...")
                        if event_callback:
                            event_callback("multihop_step_pruning", {
                                "step_id": step_id,
                                "step_number": step_number,
                                "documents_count": len(retrieved_docs),
                                "status": "pruning"
                            })
                        
                        pruning_threshold = self.pipeline_config.get("provence", {}).get("threshold", 0.1)
                        pruned_docs = self.pruner.prune_documents(enhanced_query, retrieved_docs, threshold=pruning_threshold)
                        # Remove any chunks that were fully pruned (empty text)
                        retrieved_docs = [d for d in pruned_docs if d.get('text', '').strip()]
                        print(f"✅ Pruning completed: {len(retrieved_docs)} documents remaining")
                        
                        if event_callback:
                            event_callback("multihop_step_pruned", {
                                "step_id": step_id,
                                "step_number": step_number,
                                "pruned_count": len(retrieved_docs),
                                "status": "pruned"
                            })
                    
                    if event_callback:
                        event_callback("multihop_step_processed", {
                            "step_id": step_id,
                            "step_number": step_number,
                            "final_documents": len(retrieved_docs),
                            "status": "synthesizing"
                        })
                    
                except Exception as e:
                    print(f"Could not search table '{table_name}': {e}")
                    retrieved_docs = []
                    
                    if event_callback:
                        event_callback("multihop_step_error", {
                            "step_id": step_id,
                            "step_number": step_number,
                            "error": str(e),
                            "status": "error"
                        })
                
                # Synthesize answer for this step
                step_answer = self._synthesize_step_answer(enhanced_query, retrieved_docs, accumulated_context)
                print(f"✅ Step {step_id} completed: {step_answer[:100]}...")
                
                if event_callback:
                    event_callback("multihop_step_completed", {
                        "step_id": step_id,
                        "step_number": step_number,
                        "query": step_query,
                        "answer": step_answer,
                        "documents_used": len(retrieved_docs),
                        "status": "completed"
                    })
                
                # Store results
                all_results[step_id] = {
                    'query': step_query,
                    'enhanced_query': enhanced_query,
                    'answer': step_answer,
                    'documents': retrieved_docs
                }
                all_source_docs.extend(retrieved_docs)
                
                # Mark step as completed
                context.complete_step(step_id, {
                    'answer': step_answer,
                    'documents': retrieved_docs,
                    'query': step_query
                }, [accumulated_context] if accumulated_context else [])
        
        print(f"--- Multi-Hop Retrieval Completed ---")
        print(f"Steps executed: {len(all_results)}")
        print(f"Total unique source documents: {len(set(doc.get('chunk_id', id(doc)) for doc in all_source_docs))}")
        
        if event_callback:
            event_callback("multihop_synthesis_started", {
                "total_steps_completed": len(all_results),
                "total_documents": len(all_source_docs),
                "step_results": [
                    {
                        "step_id": step_id,
                        "query": result['query'], 
                        "answer_preview": result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
                    } 
                    for step_id, result in all_results.items()
                ]
            })
        
        # Synthesize final answer from all steps
        final_answer = self._synthesize_final_multihop_answer(all_results, context)
        
        if event_callback:
            event_callback("multihop_completed", {
                "total_steps": len(all_results),
                "final_answer": final_answer,
                "source_documents": all_source_docs
            })
        
        return {
            "answer": final_answer,
            "source_documents": all_source_docs,
            "multihop_context": {
                "steps_executed": len(all_results),
                "step_results": all_results
            }
        }
    
    def _synthesize_step_answer(self, query: str, retrieved_docs: List[Dict], accumulated_context: str = "") -> str:
        """Synthesize answer for a single step using retrieved documents."""
        if not retrieved_docs:
            return f"No relevant information found for step."
        
        # Create context from retrieved documents
        context = "\n\n".join([doc.get('text', '') for doc in retrieved_docs])
        
        prompt = f"""Based on the provided context, answer the following question:

Question: {query}

Context:
{context}

{f"Additional context from previous steps: {accumulated_context}" if accumulated_context else ""}

Provide a clear, concise answer based on the available information:"""

        try:
            response = self.synthesis_llm_client.generate_completion(
                model=self.synthesis_model,
                prompt=prompt
            )
            return response.get('response', 'Unable to generate answer for this step.')
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def _synthesize_final_multihop_answer(self, all_results: Dict, context: MultiHopContext) -> str:
        """Synthesize final answer from all multi-hop steps."""
        if not all_results:
            return "No information was found to answer your question."
        
        # Prepare step summaries
        step_summaries = []
        for step_id in sorted(all_results.keys()):
            result = all_results[step_id]
            step_summaries.append(f"Step {step_id}: {result['query']}\nAnswer: {result['answer']}")
        
        combined_context = "\n\n".join(step_summaries)
        
        # Get the original query from context or reconstruct
        original_query = "Please provide a comprehensive answer based on the multi-hop analysis."
        if hasattr(context, 'original_query'):
            original_query = context.original_query
        
        prompt = f"""Based on the multi-hop analysis below, provide a comprehensive final answer:

Original Question: {original_query}

Multi-hop Analysis:
{combined_context}

Please synthesize a complete, coherent answer that integrates insights from all steps:"""

        try:
            print(f"🔄 Generating final synthesis with model: {self.synthesis_model}")
            response = self.synthesis_llm_client.generate_completion(
                model=self.synthesis_model,
                prompt=prompt
            )
            final_response = response.get('response', 'Unable to synthesize final answer.')
            print(f"✅ Final synthesis completed: {len(final_response)} characters")
            return final_response
        except Exception as e:
            error_msg = f"Error synthesizing final answer: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

class GraphRetriever:
    def __init__(self, graph_path: str):
        self.graph = nx.read_gml(graph_path)

    def retrieve(self, query: str, k: int = 5, score_cutoff: int = 80) -> List[Dict[str, Any]]:
        print(f"\n--- Performing Graph Retrieval for query: '{query}' ---")
        
        query_parts = query.split()
        entities = []
        for part in query_parts:
            match = process.extractOne(part, self.graph.nodes(), score_cutoff=score_cutoff)
            if match and isinstance(match[0], str):
                entities.append(match[0])
        
        retrieved_docs = []
        for entity in set(entities):
            for neighbor in self.graph.neighbors(entity):
                retrieved_docs.append({
                    'chunk_id': f"graph_{entity}_{neighbor}",
                    'text': f"Entity: {entity}, Neighbor: {neighbor}",
                    'score': 1.0,
                    'metadata': {'source': 'graph'}
                })
        
        print(f"Retrieved {len(retrieved_docs)} documents from the graph.")
        return retrieved_docs[:k]

# region === MultiVectorRetriever ===
class MultiVectorRetriever:
    """
    Performs hybrid (vector + FTS) or vector-only retrieval.
    """
    def __init__(self, db_manager: LanceDBManager, text_embedder: QwenEmbedder, vision_model: LocalVisionModel = None, *, fusion_config: Dict[str, Any] | None = None):
        self.db_manager = db_manager
        self.text_embedder = text_embedder
        self.vision_model = vision_model
        self.fusion_config = fusion_config or {"method": "linear", "bm25_weight": 0.5, "vec_weight": 0.5}

        # Lightweight in-memory LRU cache for single-query embeddings (256 entries)
        @lru_cache(maxsize=256)
        def _embed_single(q: str):
            return self.text_embedder.create_embeddings([q])[0]

        self._embed_single = _embed_single

    def retrieve(self, text_query: str, table_name: str, k: int, reranker=None) -> List[Dict[str, Any]]:
        """
        Performs a search on a single LanceDB table.
        If a reranker is provided, it performs a hybrid search.
        Otherwise, it performs a standard vector search.
        """
        print(f"\n--- Performing Retrieval for query: '{text_query}' on table '{table_name}' ---")
        
        try:
            if table_name is None:
                table_name = "default_text_table"
            tbl = self.db_manager.get_table(table_name)
            
            # Create / fetch cached text embedding for the query
            text_query_embedding = self._embed_single(text_query)
            
            logger = logging.getLogger(__name__)

            # Always perform hybrid lexical + vector search
            logger.debug(
                "Running hybrid search on table '%s' (k=%s, have_reranker=%s)",
                table_name,
                k,
                bool(reranker),
            )

            if reranker:
                logger.debug("Hybrid + reranker path not yet implemented with manual fusion; proceeding without extra reranker.")

            # Manual two-leg hybrid: take half from each modality
            fts_k = k // 2
            vec_k = k - fts_k

            # Run FTS and vector search in parallel to cut latency
            def _run_fts():
                # Very short queries often underperform → add fuzzy wildcard
                fts_query = text_query
                if len(text_query.split()) == 1:
                    fts_query = f"{text_query}* OR {text_query}~"
                return (
                     tbl.search(query=fts_query, query_type="fts")
                        .limit(fts_k)
                        .to_df()
                 )

            def _run_vec():
                if vec_k == 0:
                    return None
                return (
                    tbl.search(text_query_embedding)
                       .limit(vec_k * 2)  # fetch extra to allow for dedup
                       .to_df()
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                fts_future = executor.submit(_run_fts)
                vec_future = executor.submit(_run_vec)
                fts_df = fts_future.result()
                vec_df = vec_future.result()

            if vec_df is not None:
                combined = pd.concat([fts_df, vec_df])
            else:
                combined = fts_df

            # Remove duplicates preserving first occurrence, then trim to k
            dedup_subset = ["_rowid"] if "_rowid" in combined.columns else (["chunk_id"] if "chunk_id" in combined.columns else None)
            if dedup_subset:
                combined = combined.drop_duplicates(subset=dedup_subset, keep="first")
            combined = combined.head(k)

            results_df = combined
            logger.debug(
                "Hybrid (fts=%s, vec=%s) → %s unique chunks",
                len(fts_df),
                0 if vec_df is None else len(vec_df),
                len(results_df),
            )
            
            retrieved_docs = []
            for _, row in results_df.iterrows():
                metadata = json.loads(row.get('metadata', '{}'))
                # Add top-level fields back into metadata for consistency if they don't exist
                metadata.setdefault('document_id', row.get('document_id'))
                metadata.setdefault('chunk_index', row.get('chunk_index'))
                
                # Determine score (vector distance or FTS). Replace NaN with 0.0
                raw_score = row.get('_distance') if '_distance' in row else row.get('score')
                try:
                    if raw_score is None or (isinstance(raw_score, float) and math.isnan(raw_score)):
                        raw_score = 0.0
                except Exception:
                    raw_score = 0.0

                combined_score = raw_score
                # Optional linear-weight fusion if both FTS & vector scores exist
                if '_distance' in row and 'score' in row:
                    try:
                        bm25 = row.get('score', 0.0)
                        vec_sim = 1.0 / (1.0 + row.get('_distance', 1.0))  # convert distance to similarity
                        w_bm25 = float(self.fusion_config.get('bm25_weight', 0.5))
                        w_vec = float(self.fusion_config.get('vec_weight', 0.5))
                        combined_score = w_bm25 * bm25 + w_vec * vec_sim
                    except Exception:
                        pass

                retrieved_docs.append({
                    'chunk_id': row.get('chunk_id'),
                    'text': metadata.get('original_text', row.get('text')),
                    'score': combined_score,
                    'bm25': row.get('score'),
                    '_distance': row.get('_distance'),
                    'document_id': row.get('document_id'),
                    'chunk_index': row.get('chunk_index'),
                    'metadata': metadata
                })

            logger.debug("Hybrid search returned %s results", len(retrieved_docs))
            log_retrieval_results(retrieved_docs, k)
            print(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs
        
        except Exception as e:
            print(f"Could not search table '{table_name}': {e}")
            return []
# endregion

if __name__ == '__main__':
    print("retrievers.py updated for LanceDB FTS Hybrid Search.")
