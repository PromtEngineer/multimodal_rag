from typing import Dict, Any
import json
from rag_system.utils.ollama_client import OllamaClient

class VerificationResult:
    def __init__(self, is_grounded: bool, reasoning: str, verdict: str, confidence_score: int):
        self.is_grounded = is_grounded
        self.reasoning = reasoning
        self.verdict = verdict
        self.confidence_score = confidence_score

class Verifier:
    """
    Verifies if a generated answer is grounded in the provided context using Ollama.
    """
    def __init__(self, llm_client: OllamaClient, llm_model: str):
        self.llm_client = llm_client
        self.llm_model = llm_model
        print(f"Initialized Verifier with Ollama model '{self.llm_model}'.")

    def verify(self, query: str, context: str, answer: str) -> VerificationResult:
        print("\nVerifying answer against context with Ollama...")
        prompt = f"""
You are a verification expert. Your task is to determine if the provided 'Answer' is fully supported by the 'Context'.
The answer must not contain information that is not present in the context.

Respond with a JSON object containing four keys:
1. "verdict": A single word, either "SUPPORTED", "NEEDS_CLARIFICATION", or "NOT_SUPPORTED".
2. "is_grounded": A boolean value (true or false).
3. "reasoning": A brief explanation for your verdict.
4. "confidence_score": An integer between 0 and 100, representing your confidence in the answer's groundedness.

Query: "{query}"

Context:
---
{context}
---

Answer:
---
{answer}
---

JSON Output:
"""
        response = self.llm_client.generate_completion(self.llm_model, prompt, format="json")
        try:
            data = json.loads(response.get('response', '{}'))
            return VerificationResult(
                is_grounded=data.get('is_grounded', False),
                reasoning=data.get('reasoning', 'Failed to parse verifier response.'),
                verdict=data.get('verdict', 'NOT_SUPPORTED'),
                confidence_score=data.get('confidence_score', 0)
            )
        except (json.JSONDecodeError, AttributeError):
            return VerificationResult(False, "Failed to get a valid JSON response from the verifier model.", "NOT_SUPPORTED", 0)

    # --- Async wrapper ------------------------------------------------
    async def verify_async(self, query: str, context: str, answer: str) -> VerificationResult:
        """Async variant that calls the Ollama client asynchronously."""
        prompt = f"""
You are a verification expert. Your task is to determine if the provided 'Answer' is fully supported by the 'Context'.
Respond with JSON {{"verdict":"SUPPORTED|NOT_SUPPORTED|NEEDS_CLARIFICATION","is_grounded":bool,"reasoning":str,"confidence_score":int}}

Query: "{query}"

Context:
{context}

Answer:
{answer}

JSON Output:
"""
        resp = await self.llm_client.generate_completion_async(self.llm_model, prompt, format="json")
        try:
            data = json.loads(resp.get("response", "{}"))
            return VerificationResult(
                is_grounded=data.get("is_grounded", False),
                reasoning=data.get("reasoning", "async parse error"),
                verdict=data.get("verdict", "NOT_SUPPORTED"),
                confidence_score=data.get('confidence_score', 0)
            )
        except (json.JSONDecodeError, AttributeError):
            return VerificationResult(False, "Failed async parse", "NOT_SUPPORTED", 0)