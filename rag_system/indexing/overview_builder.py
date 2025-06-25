from __future__ import annotations

import os, json, logging, re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class OverviewBuilder:
    """Generates and stores a one-paragraph overview for each document.
    The overview is derived from the first *n* chunks of the document.
    """

    DEFAULT_PROMPT = (
        """You are an expert document analyst. Your task is to create a concise, one-sentence overview of a document based on its initial text. This overview will be used by a routing agent to decide if a user's query is related to this document.

The overview MUST be a single sentence and include the following key information, if available:
- Document Type (e.g., Invoice, Research Paper, Report, Agreement, Resume)
- Key Entities (e.g., companies, people, products)
- Specific Identifiers (e.g., invoice numbers, report IDs, dates)
- Main Topic

Here are some examples of high-quality overviews:

===
DOCUMENT_START: "Invoice #1234. To: ACME Corp. From: Stark Industries. Date: 2024-10-26. Amount: $50,000. For consulting services."
OVERVIEW: "An invoice (#1234) from Stark Industries to ACME Corp for $50,000, dated October 26, 2024, covering consulting services."
===
DOCUMENT_START: "A Study of Quantum Entanglement in Multi-Modal AI Systems. Authors: Dr. Eva Rostova, Dr. Alexi Romanov. Abstract: We explore the...'
OVERVIEW: "A research paper by Dr. Rostova and Dr. Romanov on quantum entanglement in multi-modal AI systems."
===
DOCUMENT_START: "Q4 2024 Financial Report. ACME Corp. Revenue: $1.2M. Net Profit: $250,000. This report details the financial performance of ACME Corp..."
OVERVIEW: "Q4 2024 financial report for ACME Corp, showing revenue of $1.2M and net profit of $250,000."
===

Now, generate the overview for the following document.

DOCUMENT_START:
{text}

OVERVIEW:"""
    )

    def __init__(self, llm_client, model: str = "qwen3:0.6b", first_n_chunks: int = 5,
                 out_path: str = "index_store/overviews/overviews.jsonl"):
        self.llm_client = llm_client
        self.model = model
        self.first_n = first_n_chunks
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def build_and_store(self, doc_id: str, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        head_text = "\n".join(c["text"] for c in chunks[: self.first_n] if c.get("text"))
        prompt = self.DEFAULT_PROMPT.format(text=head_text[:5000])  # safety cap
        try:
            resp = self.llm_client.generate_completion(model=self.model, prompt=prompt, enable_thinking=False)
            summary_raw = resp.get("response", "")
            # Remove any lingering <think>...</think> blocks just in case
            summary = re.sub(r'<think[^>]*>.*?</think>', '', summary_raw, flags=re.IGNORECASE | re.DOTALL).strip()
        except Exception as e:
            summary = f"Failed to generate overview: {e}"
        record = {"doc_id": doc_id, "overview": summary.strip()}
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"📄 Overview generated for {doc_id} (stored in {self.out_path})") 