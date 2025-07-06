# New file
"""Centralised prompt registry for the RAG system.

Usage
-----
from rag_system.prompts import fmt, get

prompt = fmt("synthesize_final_answer", query="...", facts="...")
"""
import pathlib
import functools
import string
from typing import Dict

import yaml

_PROMPT_PATH = pathlib.Path(__file__).with_name("registry.yaml")

@functools.lru_cache(maxsize=1)
def _load() -> Dict[str, str]:
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {_PROMPT_PATH}")
    with _PROMPT_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data

def get(prompt_id: str) -> str:
    """Return raw prompt template by id."""
    data = _load()
    if prompt_id not in data:
        raise KeyError(f"Prompt '{prompt_id}' not found in registry")
    return data[prompt_id]

def fmt(prompt_id: str, **kwargs) -> str:
    """Return prompt with ``str.format`` applied to **kwargs."""
    template = get(prompt_id)
    try:
        return template.format(**kwargs)
    except KeyError as e:
        missing = e.args[0]
        raise KeyError(
            f"Missing format key '{missing}' for prompt '{prompt_id}'. "
            "Provided keys: " + ", ".join(kwargs.keys())
        ) from None

def list_prompts():
    """Return list of available prompt IDs."""
    return list(_load().keys()) 