# Multimodal RAG System – Architecture & Code Walk-Through

*Last updated: 13 Jun 2025*

---

## 1  High-level picture
```
                    ┌──────────────────────┐
User ↔ Frontend  ↔  │ localGPT backend 8000│ ← uploads / chat
                    └────────┬─────────────┘
                             │ REST /json
                             ▼
                    ┌──────────────────────┐
                    │ Advanced RAG API 8001│  (rag_system.api_server)
                    └────────┬─────────────┘
                             ▼
               ┌──────────────────────────────┐
               │  rag_system.agent.Agent      │
               │  • Triage / Decompose        │
               │  • RetrievalPipeline         │
               │  • Verification              │
               └────────┬─────────────────────┘
                        ▼
             ┌────────────────────────┐
             │  Storage & Indices     │
             │  • LanceDB vectors     │
             │  • Tantivy / native FTS│
             │  • NetworkX graph      │
             └────────────────────────┘
```
* Two execution modes during development:
  * **Dual-process**: run **backend/server.py** (8000) + **rag_system/api_server.py** (8001) as before.
  * **Single-process**: run **combined_server.py** which boots *both* back-ends in separate threads – handy on limited ports / Heroku-style dynos.
* All heavyweight processing lives inside the `rag_system` package so it can be reused offline or via CLI (`python rag_system/main.py`).

---

## 2  Core packages
| Path | Purpose |
|------|---------|
| `rag_system/indexing` | Converts raw docs → chunks → vectors / BM25 / graph.  Includes **Embedders**, **Representations**, **Contextualiser** (parent-child windows) |
| `rag_system/retrieval` | Runtime search. `MultiVectorRetriever` performs half-FTS / half-vector hybrid with dedup, supports BM25 & graph fallbacks |
| `rag_system/pipelines` | End-to-end flows:<br>• **IndexingPipeline** – orchestrates embedding + DB write<br>• **RetrievalPipeline** – orchestrates search → rerank → context expansion → LLM synth |
| `rag_system/agent` | Long-running stateful helper: triages query → decomposition → retrieval pipeline → verifier → caching |
| `rag_system/rerankers` | AI reranking via ColBERT (answerdotai/answerai-colbert-small-v1) |
| `rag_system/utils` | Thin wrapper around **Ollama** HTTP API, logging helpers |

---

## 3  Key runtime flow (default mode)
1. **Triage** (`Agent._triage_query`) – classifies query into `graph_query`, `rag_query`, `direct_answer` via a quick LLM call.
2. **(Optional) Decomposition** – if `PIPELINE_CONFIGS['query_decomposition']['enabled']` is true, the query is broken into ≤3 sub-queries (see `QueryDecomposer`).
3. **Parallel retrieval** – every sub-query is passed to `RetrievalPipeline.run` concurrently (ThreadPoolExecutor, max 3 workers).
4. **Inside `RetrievalPipeline`**
   * `MultiVectorRetriever` pulls **k/2 FTS** + **k/2 vector** rows, dedups on `_rowid` or `chunk_id`.
   * Optional LanceDB reranker (linear-combination) and **AI reranker** (cross-encoder) refine top-k.
   * Parent-child expansion adds surrounding chunks (window size configurable).
   * Prompt **_synthesize_final_answer** builds the *Verified Facts → Question* prompt and calls Ollama generation model.
5. **Two possible aggregation strategies** (toggle `compose_from_sub_answers`):
   * **Single-stage** (default): all unique chunks from all sub-queries merged, one final synthesis call.
   * **Two-stage**: each sub-query already returns an answer; a composer prompt merges the mini-answers into a final reply.
6. **Verification** – lightweight LLM check adds `[Warning: …]` tag if answer is not fully grounded.
7. **Cache** – last 100 non-direct queries cached for 5 min.

---

## 4  Prompt inventory
| Component | File | Purpose |
|-----------|------|---------|
| `QueryDecomposer` | `retrieval/query_transformer.py` | JSON-only list of sub-queries (max 3) |
| `RetrievalPipeline._synthesize_final_answer` | `pipelines/retrieval_pipeline.py` | *Verified Facts → Answer* with rules on discarding irrelevant snippets, contradiction handling, ≤5 sentences |
| Composer prompt (two-stage) | `agent/loop.py` | Combines sub-answers into final answer |
| Verifier prompt | `agent/verifier.py` | Returns `{verdict, is_grounded, reasoning}` |
| Graph translation prompt | `retrieval/query_transformer.py` | Creates `(start_node, edge_label)` JSON for GraphRAG |

All prompts live as f-strings for easy tweaking; no external template engine required.

---

## 5  Storage layout
```
index_store/
├─ lancedb/            ← vector store (.lance + parquet)
├─ bm25/               ← Tantivy index (optional)
└─ graph/knowledge_graph.gml
```
* LanceDB tables are session-scoped: `text_pages_<session_id>`.
* Indexing pipeline appends instead of recreating tables – safe for incremental updates.

---

## 6  Config knobs (`rag_system/main.py`)
* `PIPELINE_CONFIGS` contains **fast**, **default**, plus toggles for BM25 & graph.
* Important flags:
  * `retrieval_k` – number of chunks per sub-query (default 20).
  * `context_window_size` – parent-child expansion window (0 = off).
  * `reranker.enabled` / `type` – `ai` uses BAAI/bge-reranker.
  * `query_decomposition.enabled` & `compose_from_sub_answers` toggle decomposition and two-stage flow.
* Device auto-selection prioritises CUDA → MPS → CPU.

---

## 7  Running the system
```bash
# 1. create venv + install deps (torch, lancedb, pydantic-v2, numexpr…)

# 2. export environment
export HF_TOKEN=...           # HF access
export RAG_LOG_LEVEL=DEBUG    # fine-grained logs

# 3. Index documents
python rag_system/main.py index --mode default --path /docs/*.pdf

# 4. Start servers (two terminals)
cd backend && RAG_LOG_LEVEL=DEBUG python server.py | cat &
HF_TOKEN=$HF_TOKEN RAG_LOG_LEVEL=DEBUG python -m rag_system.api_server | cat &

# Single-process option
HF_TOKEN=$HF_TOKEN RAG_LOG_LEVEL=DEBUG python combined_server.py | cat &
```

Open `localhost:8000` for the localGPT UI; it will proxy chat to port 8001.

---

## 8  Adding new features – cheat-sheet
| Task | Where to start |
|------|----------------|
| New embed model | `indexing/embedders.py` (add class + update config) |
| Custom reranker | implement in `rag_system/rerankers/` and wire in `RetrievalPipeline._get_ai_reranker()` |
| Different hybrid fusion | tweak `MultiVectorRetriever.retrieve()` |
| Answer citation markers | modify `_synthesize_final_answer` prompt to print snippet indices |
| Async Ollama calls | replace `utils/ollama_client` with aiohttp version and switch ThreadPoolExecutor to asyncio.gather |

---

## 9  Known limitations / TODO
* LanceDB hybrid API lacks native "vector+fts" → manual fusion.
* Score sanitisation now drops NaN/Inf before JSON serialisation (fixed in `retrieval_pipeline.py`).
* `_TOKENIZER` / `_MODEL` globals initialised on module import (fixed QwenEmbedder crash).
* Memory leak: occasional semaphore warning from `multiprocessing` when API server exits.
* Frontend currently drops citation metadata – UI not showing sources.

---

Happy hacking!  Feel free to extend this doc as the system evolves. 

## 10  Optimisation roadmap (June 2025)

The following four milestones will be tackled sequentially; each is protected behind feature flags so the system stays deployable at every step.

| Milestone | Focus | Key changes |
|-----------|-------|-------------|
| **0 – Baseline** | Measurement | Benchmark harness, feature-flag skeleton |
| **1 – Retrieval latency** | Parallel FTS + vector search, query embedding LRU, bottleneck ≥ 1.3.7 | Expected ‑40 % mean latency |
| **2 – Candidate quality** | LanceDB native hybrid recall, learned fusion weights, short-query wildcard boost | +3–4 pp Recall@20 |
| **3 – Rerank & Context** | Batched cross-encoder with early-exit, single-shot SQL window for context, fp16 unit-norm embeddings | 3× reranker speed |
| **4 – Polish** | RRF option, 4-bit larger embeddings, prune dead BM25 path | Accuracy + misc cleanup |

Each milestone will land as an independent PR and will update this document upon merge. 