import asyncio
import re
import json
import concurrent.futures
from typing import Dict, Any, List, Optional

from rag_system.agent.verifier import Verifier
from rag_system.retrieval.query_transformer import QueryDecomposer
from rag_system.utils.ollama_client import OllamaClient
from rag_system.pipelines.retrieval_pipeline import RetrievalPipeline
from rag_system.utils.logging_utils import log_query


class Tool:
    """A base class for tools that the ReAct agent can use."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def use(self, query: str, **kwargs) -> str:
        raise NotImplementedError("Each tool must implement the 'use' method.")

    async def use_async(self, query: str, **kwargs) -> str:
        return await asyncio.to_thread(self.use, query, **kwargs)


class DocumentSearchTool(Tool):
    """Tool to search for relevant documents using the retrieval pipeline."""
    def __init__(self, retrieval_pipeline: RetrievalPipeline):
        super().__init__(
            name="document_search",
            description="Searches and retrieves relevant document excerpts based on a user query. Use this for questions about specific facts, details, or summaries from the document collection."
        )
        self.pipeline = retrieval_pipeline

    def use(self, query: str, **kwargs) -> str:
        table_name = kwargs.get("table_name")
        # Use a small k to keep context concise for the LLM
        retrieved_docs = self.pipeline.retriever.retrieve(query, table_name=table_name, k=5)
        if not retrieved_docs:
            default_table = self.pipeline.storage_config.get("text_table_name", "default_text_table")
            if default_table != table_name:
                retrieved_docs = self.pipeline.retriever.retrieve(query, table_name=default_table, k=5)
        # Format the output for the LLM
        formatted_docs = [f"Source {i+1} (ID: {doc['chunk_id']}):\n{doc['text']}" for i, doc in enumerate(retrieved_docs)]
        return "\n---\n".join(formatted_docs) if formatted_docs else "No relevant documents found."


class GraphSearchTool(Tool):
    """Tool to query the knowledge graph."""
    def __init__(self, retrieval_pipeline: RetrievalPipeline):
        super().__init__(
            name="graph_search",
            description="Queries a knowledge graph to find relationships between entities. Use this for questions about how concepts are connected, e.g., 'What is the relationship between X and Y?'"
        )
        # Assuming the graph retriever is accessible via the main pipeline
        self.retriever = retrieval_pipeline.graph_retriever

    def use(self, query: str) -> str:
        if not self.retriever:
            return "Graph retriever is not configured."
        # The graph retriever might need a specific query format, we'll assume it takes a simple string for now
        result = self.retriever.search(query)
        return str(result)

class AnswerTool(Tool):
    """Tool to generate the final answer for the user."""
    def __init__(self, llm_client: OllamaClient, generation_model: str):
        super().__init__(
            name="final_answer",
            description="Provides the final answer to the user's query based on the gathered information. Use this when you have sufficient information to answer the original question."
        )
        self.llm_client = llm_client
        self.generation_model = generation_model

    def use(self, final_thought: str) -> str:
        # This tool doesn't need to call the LLM again, it just formats the final thought.
        # The LLM's thought process itself produces the answer.
        return final_thought


class ReActAgent:
    PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers user queries by using a set of tools. 
Your goal is to gather enough information to confidently answer the user's question.

You will work in a loop of Thought, Action, and Observation.

**Thought**: First, think about what you need to do. Analyze the user's query and the conversation history. Decide if you can answer immediately or if you need to use a tool. Your thought process should be a brief, clear plan.

**Action**: Based on your thought, choose a tool to use. You must output a JSON object with two keys: "tool_name" and "tool_input".

**Observation**: After you perform an action, you will receive an observation. This is the output from the tool.

Repeat this cycle until you have enough information to answer the query.

---

**TOOLS:**
You have access to the following tools. Only use these tools.

{tool_descriptions}

---

**ACTION FORMAT:**
You must output your action as a single JSON object. For example:
```json
{{
  "tool_name": "document_search",
  "tool_input": "What were the main findings of the study?"
}}
```

---

EXAMPLE (follow this pattern):

Thought: I should look up the invoice amount first.
Action:
```json
{{
  "tool_name": "document_search",
  "tool_input": "invoice amount"
}}
```
Observation: Source 1: ... "$3,000" ...
Thought: I have the amount, now I can answer.
Action:
```json
{{
  "tool_name": "final_answer",
  "tool_input": "The invoice amount is $3,000."
}}
```
---

**CONVERSATION HISTORY:**

{history}

---

Now, begin!

User Query: "{query}"
    """

    """
    A ReAct-style agent that uses a loop of Thought, Action, and Observation
    to answer user queries.
    """
    def __init__(self, pipeline_configs: Dict[str, Any], llm_client: OllamaClient, ollama_config: Dict[str, str]):
        self.llm_client = llm_client
        self.ollama_config = ollama_config
        # Give the model a bit more room before we step in with a failsafe
        self.max_iterations = pipeline_configs.get("react", {}).get("max_iterations", 8)

        # The retrieval pipeline is used by the tools
        self.retrieval_pipeline = RetrievalPipeline(pipeline_configs, llm_client, ollama_config)

        # Initialize tools (graph_search temporarily disabled)
        self.tools = {
            "document_search": DocumentSearchTool(self.retrieval_pipeline),
            "final_answer": AnswerTool(llm_client, ollama_config["generation_model"])
        }
        
        # Pre-generate the tool descriptions string
        self.tool_descriptions = "\n".join(
            [f"- **{name}**: {tool.description}" for name, tool in self.tools.items()]
        )

        # Enable simplified two-phase mode by default (can be toggled in config)
        self.two_phase = pipeline_configs.get("react", {}).get("two_phase", True)

        # --- Query Decomposition ---
        self.query_decomp_config = pipeline_configs.get("query_decomposition", {})
        self.query_decomposer = None
        if self.query_decomp_config.get("enabled", False):
            self.query_decomposer = QueryDecomposer(llm_client, ollama_config["generation_model"])

        # --- Simple in-memory conversation transcripts (per session) ---
        # Key: session_id (or "global"), Value: list[str] of history lines that
        # will be re-injected into the prompt on the next user message.
        self.session_transcripts: Dict[str, List[str]] = {}

    def _parse_action(self, llm_output: str) -> Optional[Dict[str, str]]:
        """Parses the LLM's output to find the action JSON."""
        try:
            # First try to find a fenced JSON block
            match = re.search(r'```json\n(.*?)\n```', llm_output, re.DOTALL)
            if not match:
                # Fallback: look for the first balanced curly-brace object
                match = re.search(r'\{[\s\S]*\}', llm_output)
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            return None
        except json.JSONDecodeError:
            return None  # Failed to parse any viable JSON object

    def run(self, query: str, session_id: Optional[str] = None, table_name: Optional[str] = None):
        session_key = session_id or "global"

        # ---- logging ----
        log_query(query)

        if self.two_phase:
            # -------- Phase A: retrieval (with optional decomposition) --------

            table = table_name or self.retrieval_pipeline.storage_config.get("text_table_name", "default_text_table")

            # --- Optional Query Decomposition ---
            if self.query_decomp_config.get("enabled", False) and self.query_decomposer:
                sub_queries = self.query_decomposer.decompose(query)

                # Log the decomposition result for observability
                log_query(query, sub_queries=sub_queries)

                if len(sub_queries) > 1:
                    compose_from_sub_answers = self.query_decomp_config.get("compose_from_sub_answers", True)

                    aggregated_docs = []
                    citations_seen = set()
                    sub_answers = []

                    # --- Parallel retrieval for sub-queries ---
                    sub_query_docs: Dict[str, List[Dict[str, Any]]] = {}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(sub_queries))) as executor:
                        future_to_query = {
                            executor.submit(
                                self.retrieval_pipeline.retriever.retrieve, q, table_name=table, k=5
                            ): q for q in sub_queries
                        }
                        for future in concurrent.futures.as_completed(future_to_query):
                            sub_q = future_to_query[future]
                            try:
                                docs_for_q = future.result()
                            except Exception as e:
                                docs_for_q = []
                            sub_query_docs[sub_q] = docs_for_q

                    # Aggregate docs and (optionally) create sub-answers
                    for sub_q, sub_docs in sub_query_docs.items():
                        for d in sub_docs:
                            if d["chunk_id"] not in citations_seen:
                                aggregated_docs.append(d)
                                citations_seen.add(d["chunk_id"])

                        if compose_from_sub_answers:
                            sub_evidence = "\n---\n".join([f"Source {i+1}:\n{d['text']}" for i, d in enumerate(sub_docs)])
                            sub_prompt = (
                                "You are a helpful assistant. Using only the evidence given, answer the sub-question in one or two concise sentences.\n\n"
                                f"Evidence:\n{sub_evidence}\n\n"
                                f"Sub-Question: {sub_q}\nAnswer:"
                            )
                            sub_resp = self.llm_client.generate_completion(
                                model=self.ollama_config["generation_model"],
                                prompt=sub_prompt,
                            )
                            sub_answer = sub_resp.get("response", "Not found").strip()
                            sub_answers.append({"question": sub_q, "answer": sub_answer})

                    # --- Compose final answer ---
                    if compose_from_sub_answers and sub_answers:
                        compose_prompt = f"""
You are an expert answer composer for a Retrieval-Augmented Generation (RAG) system.

Context:
• The ORIGINAL QUESTION from the user is shown below.
• That question was automatically decomposed into simpler SUB-QUESTIONS.
• Each sub-question has already been answered by an earlier step and the resulting Question→Answer pairs are provided to you in JSON.

Your task:
1. Read every sub-answer carefully.
2. Write a single, final answer to the ORIGINAL QUESTION **using only the information contained in the sub-answers**. Do NOT invent facts that are not present.
3. Keep the answer concise (≤ 5 sentences).

ORIGINAL QUESTION:
"{query}"

SUB-ANSWERS (JSON):
{json.dumps(sub_answers, indent=2)}

FINAL ANSWER:
"""
                        compose_resp = self.llm_client.generate_completion(
                            model=self.ollama_config["generation_model"],
                            prompt=compose_prompt,
                        )
                        answer_text = compose_resp.get("response", "Unable to compose answer.").strip()
                    else:
                        # Aggregate evidence from all sub-queries and answer once
                        aggregated_evidence = "\n---\n".join(
                            [f"Source {i+1}:\n{d['text']}" for i, d in enumerate(aggregated_docs)]
                        ) or "(No relevant snippets found.)"
                        answer_prompt = (
                            "You are a helpful assistant. Using only the evidence given, answer the user question in one or two concise sentences.\n\n"
                            f"Evidence:\n{aggregated_evidence}\n\n"
                            f"User Question: {query}\nAnswer:"
                        )
                        resp = self.llm_client.generate_completion(
                            model=self.ollama_config["generation_model"],
                            prompt=answer_prompt,
                        )
                        answer_text = resp.get("response", "I'm not sure.").strip()

                    # cache minimal transcript
                    self.session_transcripts[session_key] = [f"Q: {query}", f"A: {answer_text}"]

                    return {
                        "answer": answer_text,
                        "source_documents": [d["chunk_id"] for d in aggregated_docs],
                    }

            # --- No decomposition (single query) ---
            docs = self.retrieval_pipeline.retriever.retrieve(query, table_name=table, k=5)
            evidence = "\n---\n".join(
                [f"Source {i+1}:\n{d['text']}" for i, d in enumerate(docs)]
            ) or "(No relevant snippets found.)"

            answer_prompt = (
                "You are a helpful assistant. Using only the evidence given, answer the user question in one or two concise sentences.\n\n"
                f"Evidence:\n{evidence}\n\n"
                f"User Question: {query}\nAnswer:"
            )

            resp = self.llm_client.generate_completion(
                model=self.ollama_config["generation_model"],
                prompt=answer_prompt,
            )
            answer_text = resp.get("response", "I'm not sure.").strip()

            # cache minimal transcript
            self.session_transcripts[session_key] = [f"Q: {query}", f"A: {answer_text}"]

            return {
                "answer": answer_text,
                "source_documents": [d["chunk_id"] for d in docs],
            }

        # Start with prior transcript so the agent remembers earlier answers
        history_lines: List[str] = self.session_transcripts.get(session_key, []).copy()
        history = "\n".join(history_lines)

        last_observation = ""
        last_tool_name = None
        last_tool_input = None
        
        for i in range(self.max_iterations):
            print(f"--- ReAct Iteration {i+1} ---")

            # 1. THOUGHT
            prompt = self.PROMPT_TEMPLATE.format(
                tool_descriptions=self.tool_descriptions,
                history=history,
                query=query
            )
            
            response = self.llm_client.generate_completion(
                model=self.ollama_config["generation_model"],
                prompt=prompt
            )
            # `generate_completion` returns a dict with a 'response' key
            thought_and_action_str = response.get("response", "")
            history += f"Thought: {thought_and_action_str}\n"
            print(f"LLM Output (Thought & Action):\n{thought_and_action_str}")

            # 2. ACTION
            action = self._parse_action(thought_and_action_str)
            if not action or "tool_name" not in action or "tool_input" not in action:
                history += "Observation: Invalid action format. Please try again.\n"
                continue

            tool_name = action["tool_name"]
            tool_input = action["tool_input"]
            history += f"Action: {json.dumps(action)}\n"
            print(f"Action: {tool_name}('{tool_input}')")
            
            if tool_name == "final_answer":
                print("--- ReAct Finished ---")
                return {"answer": tool_input, "source_documents": []} # Simplified for now

            # 3. OBSERVATION
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                try:
                    observation = tool.use(tool_input, table_name=table_name)
                except Exception as e:
                    observation = f"Error using tool {tool_name}: {e}"
                
                last_observation = observation
                history += f"Observation: {observation}\n"
                print(f"Observation: {observation}")
            else:
                history += f"Observation: Tool '{tool_name}' not found.\n"

            # ----- Detect useless repetition (same tool & input) -----
            def _norm(txt: str) -> str:
                import re, json
                return re.sub(r"\s+", " ", txt.strip().lower())

            if (
                last_tool_name == tool_name
                and last_tool_input is not None
                and _norm(last_tool_input) == _norm(tool_input)
            ):
                history += "Observation: Repeated the same action without new information. Stopping early.\n"
                break
            last_tool_name = tool_name
            last_tool_input = tool_input

        # Persist the history for this session (truncate to last 40 lines to limit prompt size)
        self.session_transcripts[session_key] = (history.splitlines())[-40:]

        # --- Failsafe: ask the LLM to craft a direct answer with what it has learned ---
        if last_observation:
            fallback_prompt = (
                "You have reached the maximum number of tool uses. "
                "Based on the following information, answer the user's question in one or two sentences.\n\n"
                f"Information:\n{last_observation}\n\nUser Question: {query}\nAnswer:"
            )
            resp = self.llm_client.generate_completion(
                model=self.ollama_config["generation_model"],
                prompt=fallback_prompt,
            )
            answer_text = resp.get("response", "I'm sorry, I couldn't find the answer.")
            return {"answer": answer_text.strip(), "source_documents": []}

        return {"answer": "Agent stopped after reaching max iterations without relevant observations.", "source_documents": []}

    async def run_async(self, query: str, session_id: Optional[str] = None, table_name: Optional[str] = None):
        return await asyncio.to_thread(self.run, query, session_id=session_id, table_name=table_name) 