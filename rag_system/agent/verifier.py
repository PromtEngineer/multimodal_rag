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

    # Synchronous verify() method removed – async version is used everywhere.

    # --- Async wrapper ------------------------------------------------
    async def verify_async(self, query: str, context: str, answer: str) -> VerificationResult:
        """Async variant that calls the Ollama client asynchronously."""
        prompt = f"""
You are an expert fact checker.

TASK
  Determine whether the given ANSWER is fully supported by the CONTEXT.

OUTPUT FORMAT (very strict!)
  Reply **only** with a single line JSON object, no markdown, no extra keys:
  {{
    "verdict": "SUPPORTED" | "NOT_SUPPORTED" | "NEEDS_CLARIFICATION",
    "is_grounded": true | false,
    "reasoning": "< concise reasoning >",
    "confidence_score": <integer 0-100>
  }}

Guidelines
  • If any part of the answer is not in the context, verdict = "NOT_SUPPORTED".
  • If answer is partially supported but misses critical info, use "NEEDS_CLARIFICATION".
  • "is_grounded" mirrors whether verdict is SUPPORTED.
  • Keep reasoning under 30 words.

QUERY: "{query}"
CONTEXT:
"""
        prompt += context[:4000]  # Clamp to avoid huge prompts
        prompt += """
ANSWER:
"""
        prompt += answer
        prompt += """
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