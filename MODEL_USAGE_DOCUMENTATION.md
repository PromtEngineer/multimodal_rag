# **Model Usage Documentation for Advanced RAG System**

**Generated:** 2025-06-28  
**Based on:** Code analysis of `rag_system/agent/loop.py`, `rag_system/pipelines/retrieval_pipeline.py`, and related components  
**Status:** ✅ All configurations validated and conflicts resolved

---

## **📋 Executive Summary**

This document provides a comprehensive mapping of **which models are used at which stage** throughout your advanced RAG system. All model configurations have been **consolidated and validated** to eliminate previous conflicts.

### **🚨 Issues Resolved:**
- ✅ **Embedding Model Conflict**: Unified to use `Qwen/Qwen3-Embedding-0.6B`
- ✅ **Generation Model Standardization**: Standardized to `qwen3:8b`
- ✅ **Reranker Model Clarity**: Consolidated to `answerdotai/answerai-colbert-small-v1`
- ✅ **Configuration Deduplication**: Single source of truth in `main.py`

---

## **🎯 Master Model Configuration**

### **Ollama Models (Local Inference)**
| Model Type | Model Name | Usage |
|------------|------------|-------|
| **Generation Model** | `qwen3:8b` | Primary text generation, answer synthesis, query decomposition |
| **Enrichment Model** | `qwen3:0.6b` | Lightweight routing decisions, document overview analysis |

### **External Models (HuggingFace/Direct)**
| Model Type | Model Name | Usage |
|------------|------------|-------|
| **Embedding Model** | `Qwen/Qwen3-Embedding-0.6B` | Text embeddings, vector search, semantic cache |
| **Reranker Model** | `answerdotai/answerai-colbert-small-v1` | AI-powered document reranking |
| **Vision Model** | `Qwen/Qwen-VL-Chat` | Multimodal processing (images, documents) |
| **Fallback Reranker** | `BAAI/bge-reranker-base` | Backup reranker when ColBERT unavailable |

---

## **🔄 Model Usage by System Stage**

### **Stage 1: Query Ingestion & Triage** 
**Location:** `rag_system/agent/loop.py`

| Substage | Model Used | Purpose | Code Location |
|----------|------------|---------|---------------|
| **Router Decision** | `qwen3:0.6b` (enrichment) | Decide: RAG vs Direct vs Graph query | `_route_via_overviews()` |
| **Query Embedding** | `Qwen/Qwen3-Embedding-0.6B` | Generate query embeddings for semantic cache | `_find_in_semantic_cache()` |
| **Semantic Cache Lookup** | `Qwen/Qwen3-Embedding-0.6B` | Compare query similarity with cached responses | `_cosine_similarity()` |

**Model Flow:**
```
User Query → [qwen3:0.6b] Router → {rag_query|direct_answer|graph_query}
           ↘ [Qwen-Embedding] → Semantic Cache Check
```

### **Stage 2: Query Processing & Decomposition**
**Location:** `rag_system/agent/loop.py`

| Substage | Model Used | Purpose | Code Location |
|----------|------------|---------|---------------|
| **Query Decomposition** | `qwen3:8b` (generation) | Split complex queries into sub-queries | `QueryDecomposer.decompose()` |
| **History Formatting** | `qwen3:8b` (generation) | Context-aware query formatting | `_format_query_with_history()` |
| **Sub-Answer Composition** | `qwen3:8b` (generation) | Synthesize final answer from sub-answers | `compose_prompt` streaming |

**Model Flow:**
```
Complex Query → [qwen3:8b] → Sub-queries → Parallel Retrieval → [qwen3:8b] → Final Answer
```

### **Stage 3: Document Retrieval**
**Location:** `rag_system/pipelines/retrieval_pipeline.py`

| Substage | Model Used | Purpose | Code Location |
|----------|------------|---------|---------------|
| **Text Embedding** | `Qwen/Qwen3-Embedding-0.6B` | Convert query to vector for search | `_get_text_embedder()` |
| **Dense Retrieval** | `Qwen/Qwen3-Embedding-0.6B` | Vector similarity search in LanceDB | `MultiVectorRetriever` |
| **Late Chunking** | `Qwen/Qwen3-Embedding-0.6B` | Context-aware chunk embeddings | `LateChunkEncoder` |
| **BM25 Search** | N/A (algorithmic) | Keyword-based retrieval | `BM25Retriever` |

**Model Flow:**
```
Query → [Qwen-Embedding] → Dense Search → Documents
      ↘ BM25 Search → Documents
                    ↘ Hybrid Fusion → Ranked Documents
```

### **Stage 4: Document Reranking**
**Location:** `rag_system/pipelines/retrieval_pipeline.py`, `rag_system/rerankers/reranker.py`

| Substage | Model Used | Purpose | Code Location |
|----------|------------|---------|---------------|
| **AI Reranking** | `answerdotai/answerai-colbert-small-v1` | Rerank docs using ColBERT | `_get_ai_reranker()` |
| **Fallback Reranking** | `BAAI/bge-reranker-base` | Backup reranker when primary fails | `QwenReranker` |
| **Linear Combination** | N/A (algorithmic) | Weighted score fusion | `LinearCombinationReranker` |

**Model Flow:**
```
Retrieved Docs → [ColBERT] → Reranked Docs → Top-K Selection
              ↘ [BGE Reranker] (fallback)
```

### **Stage 5: Answer Generation**
**Location:** `rag_system/agent/loop.py`, `rag_system/pipelines/retrieval_pipeline.py`

| Substage | Model Used | Purpose | Code Location |
|----------|------------|---------|---------------|
| **Answer Synthesis** | `qwen3:8b` (generation) | Generate final answer from context | `_synthesize_final_answer()` |
| **Streaming Response** | `qwen3:8b` (generation) | Stream answer tokens to UI | `stream_completion()` |
| **Direct Answers** | `qwen3:8b` (generation) | Answer general knowledge queries | Direct answer path |

**Model Flow:**
```
Retrieved Context → [qwen3:8b] → Streaming Tokens → Final Answer
```

### **Stage 6: Verification & Quality Control**
**Location:** `rag_system/agent/verifier.py`

| Substage | Model Used | Purpose | Code Location |
|----------|------------|---------|---------------|
| **Answer Verification** | `qwen3:8b` (generation) | Verify answer groundedness & confidence | `Verifier.verify_async()` |
| **Confidence Scoring** | `qwen3:8b` (generation) | Generate confidence percentages | Verification pipeline |

**Model Flow:**
```
Answer + Context → [qwen3:8b] → Confidence Score → [Answer + Score]%
```

---

## **🧩 Specialized Processing Paths**

### **Late Chunking Pipeline**
**Location:** `rag_system/indexing/latechunk.py`

| Stage | Model Used | Purpose |
|-------|------------|---------|
| **Document Processing** | `Qwen/Qwen3-Embedding-0.6B` | Full-document context embedding |
| **Chunk Boundary Pooling** | `Qwen/Qwen3-Embedding-0.6B` | Mean-pool vectors within chunks |

**Impact:** Enhanced retrieval quality through context-aware embeddings

### **Multimodal Processing** 
**Location:** `rag_system/indexing/multimodal.py`

| Stage | Model Used | Purpose |
|-------|------------|---------|
| **Vision Processing** | `Qwen/Qwen-VL-Chat` | Process images and visual documents |
| **Text Extraction** | `Qwen/Qwen-VL-Chat` | Extract text from images |

### **Graph RAG (Optional)**
**Location:** `rag_system/retrieval/retrievers.py`

| Stage | Model Used | Purpose |
|-------|------------|---------|
| **Graph Query Translation** | `qwen3:8b` (generation) | Convert natural language to graph queries |
| **Entity Extraction** | `qwen3:8b` (generation) | Extract entities for graph lookups |

---

## **⚡ Performance Optimizations**

### **Model Caching & Reuse**
| Component | Optimization | Impact |
|-----------|--------------|---------|
| **QwenEmbedder** | Global model cache (`_TOKENIZER`, `_MODEL`) | Avoids repeated model loading |
| **AI Reranker** | Lazy initialization with thread lock | Prevents memory crashes |
| **Semantic Cache** | TTL cache with embedding similarity | Reduces redundant processing |

### **Batch Processing**
| Stage | Batch Strategy | Model Used |
|-------|----------------|------------|
| **Embedding Generation** | 50-item batches | `Qwen/Qwen3-Embedding-0.6B` |
| **Reranking** | 8-document batches | `answerdotai/answerai-colbert-small-v1` |
| **Sub-query Processing** | Parallel execution (3 workers) | `qwen3:8b` |

---

## **🔧 Configuration Architecture**

### **Single Source of Truth**
All model configurations are now centralized in `rag_system/main.py`:

```python
# Ollama Models Configuration  
OLLAMA_CONFIG = {
    "generation_model": "qwen3:8b",
    "enrichment_model": "qwen3:0.6b"
}

# External Models Configuration
EXTERNAL_MODELS = {
    "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    "reranker_model": "answerdotai/answerai-colbert-small-v1",
    "vision_model": "Qwen/Qwen-VL-Chat"
}
```

### **Configuration Validation**
- ✅ **Automated validation** prevents model mismatches
- ✅ **Consistency checks** across all pipeline configurations  
- ✅ **Error detection** for missing or conflicting models

---

## **📊 Model Usage Statistics**

### **By Frequency of Use**
1. **`qwen3:8b`** (Most Used) - Primary text generation across all stages
2. **`Qwen/Qwen3-Embedding-0.6B`** (High Use) - All embedding operations
3. **`answerdotai/answerai-colbert-small-v1`** (Moderate) - Document reranking
4. **`qwen3:0.6b`** (Selective) - Routing and lightweight decisions
5. **`Qwen/Qwen-VL-Chat`** (Optional) - Multimodal processing when enabled

### **By System Load**
| Model | Memory Usage | Compute Load | Optimization |
|-------|--------------|--------------|--------------|
| `qwen3:8b` | High | High | Ollama server caching |
| `Qwen-Embedding` | Moderate | Moderate | Global model cache |
| `ColBERT Reranker` | High | Moderate | Lazy loading + batching |
| `qwen3:0.6b` | Low | Low | Fast routing decisions |

---

## **🚀 Validation Results**

✅ **All model configurations validated successfully**  
✅ **No configuration conflicts detected**  
✅ **All pipeline components can instantiate models**  
✅ **Consistent model naming across all files**

**Validation Command:**
```bash
python validate_model_config.py
```

---

## **🔍 Key Insights**

### **✅ Strengths**
1. **Comprehensive Model Usage** - Every stage properly utilizes appropriate models
2. **Smart Model Selection** - Lightweight models for routing, powerful models for generation
3. **Advanced Techniques** - Late chunking, hybrid search, AI reranking all implemented
4. **Performance Optimizations** - Caching, batching, lazy loading throughout

### **⚠️ Areas for Monitoring**
1. **Memory Usage** - Multiple large models can consume significant RAM
2. **Model Dependencies** - External model availability and updates
3. **Performance Scaling** - Batch sizes and parallel processing limits

### **🔧 Recommendations**
1. **Monitor model performance** regularly using built-in validation
2. **Consider model quantization** for memory-constrained environments  
3. **Implement model fallbacks** for critical production use cases
4. **Regular updates** of model configurations as new models become available

---

**🎉 Your RAG system demonstrates sophisticated model orchestration with proper configuration management!** 