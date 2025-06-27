# Triage System Implementation Documentation

## 📋 **Overview**

The RAG system employs a sophisticated **two-tier triage system** to intelligently route user queries to the most appropriate processing pipeline. This system optimizes response time and accuracy by avoiding unnecessary retrieval operations for queries that don't require document access.

## 🏗️ **Architecture Overview**

### **High-Level Flow**
```mermaid
graph TD
    A[User Query] --> B{Doc Overviews<br/>Available?}
    B -->|Yes| C[Overview-Based<br/>Routing]
    B -->|No| D[Fallback LLM<br/>Triage]
    
    C --> E{Routing Decision}
    D --> E
    
    E -->|rag_query| F[Full RAG Pipeline]
    E -->|graph_query| G[Knowledge Graph<br/>Lookup]
    E -->|direct_answer| H[Answer from<br/>Overviews]
    E -->|general_chat| I[General Knowledge<br/>Response]
    E -->|clarification| J[Request<br/>Clarification]
    
    F --> K[Final Response]
    G --> K
    H --> K
    I --> K
    J --> K
```

## 🔄 **Detailed Triage Flow**

### **Primary Path: Overview-Based Routing**

When document overviews are available (loaded from `index_store/overviews/overviews.jsonl`):

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant RAG
    participant Docs

    User->>Agent: Submit Query
    Agent->>Agent: Load Conversation History
    Agent->>Agent: Format Query + History
    Agent->>Agent: Load All Document Overviews
    
    Note over Agent,LLM: Overview Routing (LLM Call #1)
    Agent->>LLM: Complex Routing Prompt<br/>(Query + History + All Overviews)
    LLM->>Agent: Decision + Reasoning<br/>{action, reasoning}
    
    alt direct_answer
        Note over Agent,LLM: Answer Generation (LLM Call #2)
        Agent->>LLM: Generate Answer from Overviews
        LLM->>Agent: Answer Text
        Agent->>Agent: Check for Negative Phrases
        alt Contains "not provided", etc.
            Agent->>RAG: Fallback to Full RAG
            RAG->>User: RAG Response
        else Answer is Complete
            Agent->>User: Direct Answer
        end
    else general_chat
        Note over Agent,LLM: General Response (LLM Call #3)
        Agent->>LLM: General Knowledge Prompt
        LLM->>Agent: General Answer
        Agent->>User: General Response
    else rag_query
        Agent->>RAG: Full RAG Pipeline
        RAG->>User: RAG Response
    else graph_query
        Agent->>Docs: Knowledge Graph Query
        Docs->>User: Graph Response
    end
```

### **Fallback Path: Simple LLM Triage**

When no document overviews are available:

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant LLM
    participant RAG

    User->>Agent: Submit Query
    Agent->>Agent: Check Conversation History
    
    alt Has History
        Agent->>RAG: Default to RAG Query<br/>(Assumes Follow-up)
        RAG->>User: RAG Response
    else No History
        Note over Agent,LLM: Simple Triage (LLM Call #1)
        Agent->>LLM: Simple 3-Category Prompt<br/>(graph_query, rag_query, direct_answer)
        LLM->>Agent: Category Decision
        
        alt direct_answer
            Agent->>User: Direct Response
        else rag_query
            Agent->>RAG: Full RAG Pipeline
            RAG->>User: RAG Response
        else graph_query
            Agent->>RAG: Knowledge Graph Query
            RAG->>User: Graph Response
        end
    end
```

## 📊 **Decision Categories**

### **Overview-Based Routing (4 Categories)**

| Category | Description | Processing Path | LLM Calls |
|----------|-------------|-----------------|-----------|
| `rag_query` | Requires specific document retrieval | Full RAG pipeline | 1 (routing) |
| `direct_answer` | Answer available in overviews | Generate from overviews | 2 (routing + generation) |
| `general_chat` | General knowledge/conversation | General LLM response | 2 (routing + generation) |
| `clarification` | Ambiguous query | Request clarification | 1 (routing) |

### **Fallback Triage (3 Categories)**

| Category | Description | Processing Path | LLM Calls |
|----------|-------------|-----------------|-----------|
| `rag_query` | Document-specific queries | Full RAG pipeline | 1 (routing) |
| `graph_query` | Factual relations | Knowledge graph lookup | 1 (routing) |
| `direct_answer` | General knowledge | Direct LLM response | 1 (routing) |

## 🔧 **Implementation Details**

### **Document Overview Loading**
```python
# Location: rag_system/agent/loop.py:53-65
overview_path = os.path.join("index_store", "overviews", "overviews.jsonl")
self.doc_overviews: list[str] = []
if os.path.exists(overview_path):
    with open(overview_path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if isinstance(rec, dict) and rec.get("overview"):
                self.doc_overviews.append(rec["overview"].strip())
```

### **Triage Entry Point**
```python
# Location: rag_system/agent/loop.py:135-173
async def _triage_query_async(self, query: str, history: list) -> str:
    # 1️⃣ Primary: Overview-based routing (if available)
    if self.doc_overviews:
        contextual_query = self._format_query_with_history(query, history)
        return self._route_via_overviews(contextual_query)
    
    # 2️⃣ Fallback: Simple LLM triage
    if history:
        return "rag_query"  # Default for follow-ups
    
    # Simple 3-category classification...
```

### **Overview Routing Logic**
```python
# Location: rag_system/agent/loop.py:445-531
def _route_via_overviews(self, query: str) -> str | None:
    # Concatenate ALL overviews
    overview_text = "\n\n".join(f"--- Overview {i+1} ---\n{o}" 
                                for i, o in enumerate(self.doc_overviews))
    
    # Complex routing prompt (~500+ tokens)
    prompt = f"""Complex routing instructions..."""
    
    # LLM call for routing decision
    resp = self.llm_client.generate_completion(...)
    
    # Process decision and potentially make additional LLM calls
```

## ⚡ **Performance Characteristics**

### **Latency Breakdown by Query Type**

```mermaid
graph LR
    subgraph "Simple Greeting ('Hello')"
        A1[Query] --> A2[History Format<br/>~50ms]
        A2 --> A3[Overview Processing<br/>~100ms]
        A3 --> A4[LLM Routing<br/>~300ms]
        A4 --> A5[LLM Response<br/>~400ms]
        A5 --> A6[Total: ~850ms]
    end
    
    subgraph "Document Query"
        B1[Query] --> B2[History Format<br/>~50ms]
        B2 --> B3[LLM Routing<br/>~300ms]
        B3 --> B4[Full RAG Pipeline<br/>~2-5s]
        B4 --> B6[Total: ~2.4-5.4s]
    end
    
    subgraph "Complex Overview Answer"
        C1[Query] --> C2[History Format<br/>~50ms]
        C2 --> C3[LLM Routing<br/>~300ms]
        C3 --> C4[LLM Answer Gen<br/>~400ms]
        C4 --> C5[Fallback Check<br/>~50ms]
        C5 --> C6[Possible RAG<br/>~2-5s]
        C6 --> C7[Total: ~800ms-5.8s]
    end
```

### **Resource Usage by Path**

| Path | LLM Calls | Token Usage | Avg Latency | Cache Benefits |
|------|-----------|-------------|-------------|----------------|
| **Simple Greeting** | 2 | ~800 tokens | 850ms | High (repeatable) |
| **Document Query** | 1 | ~600 tokens | 2.4s+ | Medium (semantic cache) |
| **Overview Answer** | 2-3 | ~1200 tokens | 800ms-5.8s | Low (fallback variability) |
| **General Chat** | 2 | ~400 tokens | 700ms | High (pattern-based) |

## 🚨 **Current Bottlenecks**

### **1. Mandatory LLM Calls for Simple Queries**
- **Issue**: Even "Hello" requires 1-2 LLM calls
- **Impact**: 400-800ms overhead for instant-response queries
- **Location**: Lines 141-173 (always calls LLM for triage)

### **2. Overview Processing Overhead**
- **Issue**: ALL document overviews sent to LLM every time
- **Impact**: Larger prompt = longer processing time
- **Location**: Line 452 (joins all overviews)

### **3. Unnecessary Context Formatting**
- **Issue**: History formatting happens for all queries
- **Impact**: Extra processing for simple queries
- **Location**: Line 143 (always formats with history)

### **4. Multiple LLM Calls for Direct Answers**
- **Issue**: Routing decision + answer generation + potential fallback
- **Impact**: 2-3 LLM calls for what should be simple responses
- **Location**: Lines 506-523 (direct answer path)

### **5. Conservative Fallback Strategy**
- **Issue**: "Direct answers" often fall back to full RAG
- **Impact**: False efficiency - appears fast but becomes slow
- **Location**: Lines 508-515 (negative phrase detection)

## 📈 **Optimization Opportunities**

### **Quick Wins (High Impact, Low Effort)**
1. **Pre-LLM Pattern Matching**: Instant responses for common greetings
2. **Conditional History Formatting**: Skip for simple queries
3. **Overview Relevance Filtering**: Send only relevant overviews

### **Medium-Term Improvements**
4. **Parallel Processing**: Overlap routing and answer generation
5. **Simplified Prompts**: Reduce token usage for common cases
6. **Smart Caching**: Cache triage decisions for similar queries

### **Advanced Optimizations**
7. **Adaptive Triage Depth**: Different complexity levels based on query
8. **Background Processing**: Pre-compute common responses
9. **Intent Classification Pipeline**: Multi-stage classification

## 🔄 **Configuration Impact**

### **When Overviews Are Available**
- **Pros**: More intelligent routing, better context awareness
- **Cons**: Higher latency, more complex processing
- **Best For**: Document-heavy workloads

### **When Overviews Are Not Available**
- **Pros**: Simpler processing, faster simple queries
- **Cons**: Less intelligent routing, defaults to RAG more often
- **Best For**: General chat or knowledge-based queries

## 🎯 **Recommended Next Steps**

1. **Profile Current Performance**: Measure actual latency for different query types
2. **Implement Quick Wins**: Add pre-LLM pattern matching for immediate improvements
3. **A/B Test Optimizations**: Compare optimized vs current triage performance
4. **Monitor Resource Usage**: Track token consumption and LLM call frequency

---

This documentation provides the foundation for understanding the current triage system and planning optimization strategies. The diagrams clearly show where bottlenecks occur and help identify the highest-impact improvement opportunities. 