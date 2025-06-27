# RAG Agent Architecture Overview

## 🎯 **System Purpose**

The multimodal RAG (Retrieval-Augmented Generation) agent provides intelligent document-based question answering with support for text, images, and knowledge graphs. The system intelligently routes queries through different processing pipelines based on query type and complexity.

## 🏗️ **Complete System Architecture**

```mermaid
graph TB
    subgraph "User Interface Layer"
        UI[Web Frontend<br/>React/TypeScript]
        API[Backend API<br/>Python FastAPI]
    end
    
    subgraph "Agent Core"
        AGENT[Agent Loop<br/>Main Orchestrator]
        TRIAGE[Triage System<br/>Query Routing]
        CACHE[Semantic Cache<br/>Response Caching]
        HISTORY[Conversation<br/>History]
    end
    
    subgraph "Processing Pipelines"
        RAG[RAG Pipeline<br/>Document Retrieval]
        GRAPH[Knowledge Graph<br/>Structured Queries]
        DIRECT[Direct Response<br/>General Knowledge]
        VERIFY[Verification<br/>Answer Validation]
    end
    
    subgraph "Retrieval Components"
        MULTI[MultiVector<br/>Retriever]
        BM25[BM25<br/>Text Search]
        VECTOR[Vector<br/>Similarity]
        RERANK[ColBERT<br/>Reranker]
    end
    
    subgraph "Data Storage"
        LANCE[LanceDB<br/>Vector Store]
        INDEX[Document<br/>Indexes]
        OVER[Document<br/>Overviews]
        KG[Knowledge<br/>Graph]
    end
    
    subgraph "Language Models"
        OLLAMA[Ollama Server<br/>LLM Gateway]
        EMBED[Embedding<br/>Models]
        GEN[Generation<br/>Models]
        VL[Vision-Language<br/>Models]
    end
    
    UI --> API
    API --> AGENT
    AGENT --> TRIAGE
    AGENT --> CACHE
    AGENT --> HISTORY
    
    TRIAGE --> RAG
    TRIAGE --> GRAPH
    TRIAGE --> DIRECT
    
    RAG --> MULTI
    RAG --> VERIFY
    MULTI --> BM25
    MULTI --> VECTOR
    MULTI --> RERANK
    
    GRAPH --> KG
    DIRECT --> OLLAMA
    VERIFY --> OLLAMA
    
    MULTI --> LANCE
    BM25 --> INDEX
    VECTOR --> EMBED
    RERANK --> OLLAMA
    
    TRIAGE --> OVER
    LANCE --> INDEX
    
    EMBED --> OLLAMA
    GEN --> OLLAMA
    VL --> OLLAMA
```

## 🎛️ **Agent Loop Execution Flow**

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Triage
    participant Cache
    participant Pipeline
    participant LLM
    participant Storage

    User->>Agent: Submit Query + Session ID
    Agent->>Agent: Load Conversation History
    
    Note over Agent,Triage: Query Routing Phase
    Agent->>Triage: Analyze Query Type
    Triage->>Storage: Load Document Overviews
    Triage->>LLM: Route Query (LLM Call)
    LLM->>Triage: Routing Decision
    Triage->>Agent: Route + Optional Direct Answer
    
    Note over Agent,Cache: Caching Phase
    alt needs_rag == true
        Agent->>Cache: Check Semantic Cache
        Cache->>Agent: Cache Hit/Miss
        
        alt Cache Miss
            Agent->>Pipeline: Execute Appropriate Pipeline
            Pipeline->>Storage: Retrieve Documents
            Pipeline->>LLM: Generate Response
            LLM->>Pipeline: Response
            Pipeline->>Agent: Final Result
            Agent->>Cache: Store Result
        else Cache Hit
            Cache->>Agent: Cached Result
        end
    else needs_rag == false
        Agent->>Agent: Use Direct Answer from Triage
    end
    
    Note over Agent,LLM: Verification Phase
    alt Has Source Documents
        Agent->>LLM: Verify Answer Against Sources
        LLM->>Agent: Verification Score
        Agent->>Agent: Add Confidence Score
    end
    
    Agent->>Agent: Update Conversation History
    Agent->>User: Final Response
```

## 🧠 **Triage System Integration**

The triage system is the **decision engine** that determines which processing pipeline should handle each query:

### **Triage Decision Impact**

```mermaid
graph TD
    QUERY[User Query] --> TRIAGE{Triage System}
    
    TRIAGE -->|rag_query| RAG_FLOW[RAG Pipeline Flow]
    TRIAGE -->|graph_query| GRAPH_FLOW[Graph Pipeline Flow]
    TRIAGE -->|direct_answer| DIRECT_FLOW[Direct Response Flow]
    TRIAGE -->|general_chat| CHAT_FLOW[Chat Response Flow]
    TRIAGE -->|clarification| CLARIFY_FLOW[Clarification Flow]
    
    RAG_FLOW --> RAG_STEPS[1. Retrieval<br/>2. Reranking<br/>3. Context Expansion<br/>4. Synthesis<br/>5. Verification]
    
    GRAPH_FLOW --> GRAPH_STEPS[1. Query Translation<br/>2. Graph Traversal<br/>3. Result Formatting]
    
    DIRECT_FLOW --> DIRECT_STEPS[1. Answer from Overviews<br/>2. Fallback Detection<br/>3. Optional RAG Fallback]
    
    CHAT_FLOW --> CHAT_STEPS[1. General LLM Call<br/>2. Direct Response]
    
    CLARIFY_FLOW --> CLARIFY_STEPS[1. Request Clarification<br/>2. Static Response]
```

## 🔄 **Processing Pipeline Details**

### **RAG Pipeline (Full Document Retrieval)**

```mermaid
flowchart TD
    START[Query] --> EMBED[Generate Query<br/>Embedding]
    EMBED --> RETRIEVE[Multi-Vector<br/>Retrieval]
    
    subgraph "Retrieval Phase"
        RETRIEVE --> DENSE[Dense Vector<br/>Search]
        RETRIEVE --> SPARSE[BM25 Sparse<br/>Search]
        RETRIEVE --> LATE[Late Chunk<br/>Retrieval]
        
        DENSE --> FUSION[Score Fusion<br/>LinearCombination]
        SPARSE --> FUSION
        LATE --> FUSION
    end
    
    FUSION --> RERANK[ColBERT<br/>Reranking]
    RERANK --> EXPAND[Context<br/>Expansion]
    EXPAND --> SYNTH[Answer<br/>Synthesis]
    SYNTH --> VERIFY[Answer<br/>Verification]
    VERIFY --> RESULT[Final Response]
```

### **Performance Characteristics by Pipeline**

| Pipeline | Avg Latency | Token Usage | Cache Benefits | Use Cases |
|----------|-------------|-------------|----------------|-----------|
| **Direct Answer** | 800ms-5.8s | 800-1200 | Low | Overview-answerable questions |
| **General Chat** | 700ms | 400 | High | Greetings, general knowledge |
| **RAG Pipeline** | 2.4s-5.4s | 600+ | Medium | Document-specific queries |
| **Graph Query** | 1.2s-2.0s | 300-500 | Medium | Factual relationships |
| **Clarification** | 50ms | 0 | N/A | Ambiguous queries |

## 🎚️ **Configuration Layers**

The system supports multiple configuration layers that affect triage behavior:

### **Pipeline Configurations**

```python
PIPELINE_CONFIGS = {
    "default": {
        "verification": {"enabled": True},
        "query_decomposition": {"enabled": True},
        "reranker": {"enabled": True, "strategy": "rerankers-lib"},
        "contextual_enricher": {"enabled": True, "window_size": 1}
    },
    "fast": {
        "verification": {"enabled": False},
        "reranker": {"enabled": False},
        "context_expansion": False
    }
}
```

### **Triage Behavior Modifiers**

| Configuration | Impact on Triage | Performance Trade-off |
|---------------|-------------------|----------------------|
| `verification.enabled` | Affects post-processing, not routing | +200ms for verified responses |
| `query_decomposition.enabled` | Changes RAG pipeline complexity | +500ms-2s for complex queries |
| `reranker.enabled` | Affects retrieval quality | +300-500ms for reranked results |
| Document overviews available | Enables advanced routing | +100-200ms triage overhead |

## 🎭 **Agent States and Context**

### **Session Management**

```mermaid
stateDiagram-v2
    [*] --> NewSession
    NewSession --> ActiveSession: First Query
    ActiveSession --> ContextualQuery: Subsequent Query
    ContextualQuery --> ActiveSession: Response Generated
    ActiveSession --> CachedResponse: Cache Hit
    CachedResponse --> ActiveSession: Return Cached
    ActiveSession --> [*]: Session Timeout
    
    note right of ContextualQuery
        Triage considers conversation
        history for routing decisions
    end note
    
    note right of CachedResponse
        Semantic cache bypasses
        triage for similar queries
    end note
```

### **Context Influence on Triage**

| Context State | Triage Behavior | Example |
|---------------|-----------------|---------|
| **New Session** | Full triage evaluation | "Hello" → general_chat |
| **Has History** | Biased toward RAG | "What about the total?" → rag_query |
| **Recent RAG** | Semantic cache check first | Similar question → cached |
| **Multiple Docs** | Overview-based routing | Complex routing logic |

## 🔧 **Key Integration Points**

### **1. Frontend → Agent**
- **Input**: Query + session ID + optional parameters
- **Output**: Streamed response with events
- **Triage Impact**: Session continuity affects routing

### **2. Agent → Triage**
- **Input**: Query + conversation history
- **Output**: Route decision + optional direct answer
- **Performance**: 300-800ms for routing decision

### **3. Triage → Pipelines**
- **RAG**: Full document retrieval and synthesis
- **Graph**: Structured query and graph traversal  
- **Direct**: Immediate response (various sources)

### **4. Cache Integration**
- **Semantic Cache**: Bypasses triage for similar queries
- **Response Cache**: Stores final responses with embeddings
- **Triage Cache**: Could cache routing decisions (not implemented)

## 📊 **Performance Monitoring Points**

### **Key Metrics to Track**

1. **Triage Latency**: Time from query to routing decision
2. **Pipeline Selection Accuracy**: How often routing is optimal
3. **Cache Hit Rates**: Effectiveness of semantic caching
4. **End-to-End Latency**: Total response time by query type
5. **LLM Token Usage**: Cost optimization opportunities

### **Optimization Tracking**

```mermaid
graph LR
    subgraph "Triage Metrics"
        T1[Routing Latency]
        T2[Decision Accuracy]
        T3[LLM Calls Count]
    end
    
    subgraph "Pipeline Metrics"
        P1[RAG Latency]
        P2[Graph Latency]
        P3[Cache Hit Rate]
    end
    
    subgraph "Overall Metrics"
        O1[End-to-End Latency]
        O2[User Satisfaction]
        O3[Resource Efficiency]
    end
    
    T1 --> O1
    T2 --> O2
    T3 --> O3
    P1 --> O1
    P2 --> O1
    P3 --> O1
```

This architecture overview provides the complete context for understanding how the triage system integrates with and affects the overall RAG agent performance. 