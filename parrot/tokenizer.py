"""Mistral tokenizer loaded as data only; no model weights or remote Python code."""
from tokenizers import Tokenizer

DEFAULT_TOKENIZER = "mistralai/Mistral-7B-v0.3"


def load_tokenizer(repo_id: str = DEFAULT_TOKENIZER) -> Tokenizer:
    """Download a Hugging Face tokenizer.json on first use, then use its cache."""
    return Tokenizer.from_pretrained(repo_id)
