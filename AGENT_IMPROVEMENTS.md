# Proposed Improvements for the RAG Agent

This document outlines potential enhancements for the agent implementation in `rag_system/agent/`. The existing agent has a strong foundation with features like query triage, decomposition, and caching. The following ideas aim to build upon this foundation to create a more dynamic, intelligent, and robust reasoning engine.

---

### 1. Adaptive Retrieval & Tool Use

Move from a fixed, one-shot decision process to a more dynamic, tool-based architecture where the agent can adaptively choose the best tool for the job.

*   **Current State**: The agent performs an upfront triage to decide between `graph_query`, `rag_query`, and `direct_answer`.
*   **Proposed Improvement**:
    *   **Tool-based Architecture**: Reframe retrievers and other functionalities as "tools" that the agent can select from. This is highly extensible and could include:
        *   `rag_search`: The existing document retrieval pipeline.
        *   `graph_lookup`: The existing knowledge graph retriever.
        *   `web_search`: A new tool to query a search engine (e.g., via SerpAPI, Tavily) for up-to-date information.
        *   `code_interpreter`: A new tool to execute Python code for calculations, data analysis, or plotting.
        *   `database_query`: A new tool to query structured data from a SQL or NoSQL database.
    *   **Multi-hop Reasoning (ReAct Framework)**: Implement a "Reason + Act" (ReAct) loop, allowing the agent to chain tool calls to answer complex questions. The flow for each query would be:
        1.  **Reason**: The agent analyzes the query and its internal state to decide which tool to use next.
        2.  **Act**: The agent executes the chosen tool and gets a result.
        3.  **Observe**: The agent analyzes the tool's output. If the answer is complete, it synthesizes a final response. If not, it can loop back to the **Reason** step to select another tool, using the new information to inform its next choice.

---

### 2. More Sophisticated Query Decomposition

Enhance the query decomposition logic to be more accurate, efficient, and context-aware.

*   **Current State**: A single LLM call unconditionally decomposes the query into a list of sub-queries that are processed in parallel.
*   **Proposed Improvement**:
    *   **Conditional Decomposition**: Before decomposing, use a quick LLM call to determine *if* the query would benefit from it. Simple, atomic queries can skip this step, reducing latency.
    *   **Strategy-aware Decomposition**: Allow the decomposition prompt to be tailored to the query type. For a "compare and contrast" query, the prompt should instruct the LLM to generate specific sub-queries for each item, followed by a final comparison query.
    *   **Dependency-aware Decomposition**: For queries with inherent dependencies (e.g., "Who is the CEO of the company that created product X?"), the decomposition should produce a logical plan or a dependency graph, not just an independent list of sub-queries. The agent would execute this plan sequentially.

---

### 3. Smarter Caching

Upgrade the caching mechanism from simple string matching to a more intelligent, semantic-based approach.

*   **Current State**: A TTL cache stores results for 5 minutes, keyed by the exact query string and its type.
*   **Proposed Improvement**:
    *   **Semantic Caching**: Cache results based on the *semantic meaning* of the query, not just the raw text.
        1.  When a query is received, generate its vector embedding.
        2.  Compare this embedding against the embeddings of previously cached queries.
        3.  If a semantically similar query (e.g., cosine similarity > 0.95) is found in the cache, return the cached result. This would handle linguistic variations like "What were the Q3 earnings?" and "Tell me about the earnings in the third quarter."
    *   **Intelligent Cache Invalidation**: Implement a robust strategy for cache invalidation. When documents are updated, any cache entries that relied on those documents should be evicted. This could be achieved by tagging cache entries with the `document_id`s used to generate the answer.

---

### 4. Improved State Management and History

Evolve the agent from a stateless processor to a stateful, conversational partner.

*   **Current State**: Each query is handled independently, with no memory of past interactions (aside from the cache).
*   **Proposed Improvement**:
    *   **Conversational Memory**: Maintain a short-term memory of the recent conversation history. When a follow-up question is asked (e.g., "What about in Europe?"), the agent can use the context from the previous turn to resolve the ambiguity (e.g., understanding it means "What were the Q3 earnings in Europe?").
    *   **Persistent Task State**: For complex, multi-step tasks that might be interrupted, allow the agent to save its current reasoning state (e.g., its plan, intermediate findings) and resume the task later.

---

### 5. Enhanced Verification and Confidence Scoring

Make the verification step more granular, transparent, and useful for downstream applications.

*   **Current State**: The verifier attaches a generic warning tag to the entire answer if it's deemed not fully grounded.
*   **Proposed Improvement**:
    *   **Sentence-level Attribution**: Modify the verifier and synthesis prompt to attribute each sentence in the final answer to its source. The output should include explicit citations, like: "The system achieved 95% accuracy in testing [doc1, page 5]. This was an improvement over the previous year [doc2, page 12]."
    *   **Confidence Score**: Instead of a binary `is_grounded` flag, the verifier should produce a numerical confidence score (e.g., 0.0 to 1.0) for the answer. This allows the application to define its own trust threshold and handle low-confidence answers differently. 