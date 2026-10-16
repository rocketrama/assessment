"""Corpus token counting with tiktoken `cl100k_base`, whichever LLM is used (TASK.pdf §4.3).

The encoding is baked into the Docker image (scripts/prefetch_models.py), so
counting never needs the network at startup.
"""

from functools import lru_cache

import tiktoken

ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))
