"""Download the tiktoken encoding and the embedding model at Docker build time,
so startup never needs a network download (must-pass #1)."""

import os

import tiktoken
from fastembed import TextEmbedding

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

enc = tiktoken.get_encoding("cl100k_base")
assert enc.encode("hello world"), "tiktoken encoding failed to load"

model = TextEmbedding(EMBEDDING_MODEL, cache_dir=os.environ["FASTEMBED_CACHE_PATH"])
assert len(next(iter(model.embed(["hello world"])))) == 384, "unexpected embedding size"

print("prefetched cl100k_base and", EMBEDDING_MODEL)
