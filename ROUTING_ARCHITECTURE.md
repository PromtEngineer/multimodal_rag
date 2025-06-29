# RAG System Routing Architecture

## Overview

The RAG system implements a **dual-layer routing architecture** with two distinct routing mechanisms that operate at different levels of the request processing pipeline.

## 🏗️ Architecture Diagram

```
User Message
     ↓
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: Backend Routing                │
│                    (backend/server.py)                     │
│                                                             │
│  _should_use_rag(message, idx_ids) → Boolean               │
│                                                             │
│  ┌─────────────────┐              ┌─────────────────────┐   │
│  │   Direct LLM    │              │    RAG Pipeline     │   │
│  │   (Fast ~1.3s)  │              │  (Comprehensive)    │   │
│  │                 │              │                     │   │
│  │ • Ollama Client │              │ • Port 8001 API     │   │
│  │ • No thinking   │              │ • Agent.run()       │   │
│  │ • No sources    │              │                     │   │
│  └─────────────────┘              └─────────────────────┘   │
│                                             ↓               │
└─────────────────────────────────────────────────────────────┘
                                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 2: Agent Routing                  │
│                  (rag_system/agent/loop.py)                │
│                                                             │
│  _triage_query_async(query, history) → String              │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ direct_answer│  │ graph_query │  │     rag_query       │ │
│  │             │  │             │  │                     │ │
│  │ • LLM Only  │  │ • Knowledge │  │ • Document Search   │ │
│  │ • General   │  │   Graph     │  │ • Retrieval         │ │
│  │   Knowledge │  │ • Relations │  │ • Synthesis         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📍 Layer 1: Backend Routing (Speed Optimization)

### **Location:** `backend/server.py`
### **Function:** `_should_use_rag(message: str, idx_ids: List[str]) -> bool`

**Purpose:** Optimize response speed by routing simple queries to Direct LLM and complex queries to RAG Pipeline.

### Routing Logic (Priority Order):

```python
def _should_use_rag(message: str, idx_ids: List[str]) -> bool:
    # 1. No indexes? → Direct LLM
    if not idx_ids:
        return False
    
    # 2. Greeting patterns? → Direct LLM  
    greeting_patterns = ['hello', 'hi', 'hey', 'test', 'thanks', ...]
    if any(pattern in message.lower() for pattern in greeting_patterns):
        return False
    
    # 3. Document keywords? → RAG
    rag_indicators = ['document', 'summarize', 'according to', ...]
    if any(indicator in message.lower() for indicator in rag_indicators):
        return True
    
    # 4. Question + substantial length? → RAG
    question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which']
    if message.lower().startswith(tuple(question_words)) and len(message) > 40:
        return True
    
    # 5. Very short messages? → Direct LLM
    if len(message.strip()) < 20:
        return False
    
    # 6. Default → Direct LLM
    return False
```

### Decision Outcomes:

| Decision | Route | Processing | Speed | Features |
|----------|-------|------------|-------|----------|
| `False` | **Direct LLM** | `_handle_direct_llm_query()` | ~1.3s | No thinking tokens, no sources |
| `True` | **RAG Pipeline** | `_handle_rag_query()` → Port 8001 | 15-30s | Thinking cleaned, source documents |

## 📍 Layer 2: Agent Routing (Intelligence Optimization)

### **Location:** `rag_system/agent/loop.py`
### **Function:** `_triage_query_async(query: str, history: list) -> str`

**Purpose:** Intelligently categorize queries within the RAG pipeline to use the most appropriate processing method.

### Routing Steps:

#### **Step 1: Document Overview Routing**
```python
def _route_via_overviews(query: str) -> str | None:
    # Uses actual document summaries loaded from index_store/overviews/overviews.jsonl
    if any_document_overview_relates_to_query:
        return "rag_query"
    if simple_factual_triple_about_public_entities:
        return "graph_query"  
    return "direct_answer"
```

#### **Step 2: Conversation History Analysis**
```python
if conversation_history_exists:
    return "rag_query"  # Follow-up questions likely need document context
```

#### **Step 3: LLM-Based Classification**
```python
# Sophisticated prompt-based routing using generation model
prompt = """
Choose exactly one category:
1. "graph_query" – Factual relations (e.g., "Who is the CEO of Apple?")
2. "rag_query" – Document-based queries (e.g., "What's in invoice 1041?")  
3. "direct_answer" – General knowledge (e.g., "What is quantum entanglement?")
"""
```

### Decision Outcomes:

| Decision | Processing Method | Use Case |
|----------|------------------|-----------|
| `"direct_answer"` | LLM generation only | General knowledge, greetings |
| `"graph_query"` | Knowledge graph lookup | Factual entity relationships |
| `"rag_query"` | Full document retrieval | Document analysis, synthesis |

## 🔄 Complete Request Flow

### **Example 1: Greeting**
```
"Hello!" 
→ Layer 1: greeting_pattern → Direct LLM 
→ Response: ~1.3s, no sources
```

### **Example 2: Document Query**
```
"What does the document say about pricing?"
→ Layer 1: rag_indicator → RAG Pipeline
→ Layer 2: _route_via_overviews() → "rag_query" 
→ Response: ~20s, with source documents
```

### **Example 3: General Knowledge (bypassed Layer 2)**
```
"What is the capital of France?"
→ Layer 1: no indicators → Direct LLM
→ Response: ~1.3s, no sources
```

## 🎯 Current Implementation Status

### ✅ **Active Components:**
- **Layer 1 Backend Routing**: Fully implemented and active
- **Layer 2 Agent Routing**: Implemented but only used within RAG pipeline

### 🔧 **Key Files:**

| File | Function | Purpose |
|------|----------|---------|
| `backend/server.py` | `_should_use_rag()` | Speed optimization routing |
| `backend/server.py` | `_handle_direct_llm_query()` | Direct Ollama processing |
| `backend/server.py` | `_handle_rag_query()` | RAG pipeline delegation |
| `rag_system/agent/loop.py` | `_triage_query_async()` | Intelligence routing |
| `rag_system/agent/loop.py` | `_route_via_overviews()` | Document-aware routing |

## 🎪 Routing Examples

### **Layer 1 Routing Patterns:**

```bash
# Direct LLM Routes
"Hello!"                           → False (greeting)
"Thanks!"                          → False (greeting) 
"What's 2+2?"                      → False (short + no indicators)
"Test"                             → False (greeting pattern)

# RAG Routes  
"What does the document say?"      → True (rag_indicator: "document")
"Summarize the report"             → True (rag_indicator: "summarize")
"What are the key findings?"       → True (question + length > 40)
"According to the text..."         → True (rag_indicator: "according to")
```

### **Layer 2 Routing Patterns:**

```bash
# Within RAG Pipeline:
"Who is the CEO of Apple?"         → "graph_query" (factual relation)
"What's in invoice 1041?"          → "rag_query" (document-specific)
"Hello, how are you?"              → "direct_answer" (even in RAG)
```

## 🚨 Important Notes

### **Thinking Token Cleanup:**
Both routing paths implement thinking token removal:
- **Direct LLM**: `enable_thinking=False` + regex cleanup
- **RAG Pipeline**: Regex cleanup of `<think>` tags in response

### **Performance Characteristics:**
- **Direct LLM**: ~1.3s average response time
- **RAG Pipeline**: 15-30s depending on query complexity and decomposition

### **Session Context:**
- **Layer 1**: Uses session indexes to determine RAG availability  
- **Layer 2**: Uses conversation history for context-aware routing

## 🔮 Potential Improvements

### **Option 1: Unified Routing**
Replace Layer 1 simple routing with Layer 2 intelligent routing for all queries.

### **Option 2: Enhanced Layer 1**
Improve backend routing with document overview awareness without full Agent complexity.

### **Option 3: Hybrid Approach**
Keep fast Layer 1 for obvious cases, delegate edge cases to Layer 2 for intelligence.

---

## 📋 Quick Reference

### **To Modify Backend Routing:**
Edit `backend/server.py` → `_should_use_rag()`

### **To Modify Agent Routing:**  
Edit `rag_system/agent/loop.py` → `_triage_query_async()` or `_route_via_overviews()`

### **To Test Routing:**
```bash
# Check routing decisions in logs:
tail -f logs/backend.log | grep "Using"
# ⚡ Using direct LLM for general query
# 🔍 Using RAG pipeline for document query
``` 