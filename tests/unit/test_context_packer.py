"""The answer context never exceeds 3,000 cl100k corpus tokens (must-pass #4)."""

import random

from app.llm.client import CORPUS_TOKEN_LIMIT
from app.retrieval.context_packer import pack_context
from app.retrieval.corpus import ChunkRecord
from app.tokens import count_tokens


def chunk(chunk_id, words):
    text = " ".join(f"w{i}" for i in range(words))
    return ChunkRecord(chunk_id, chunk_id.split("::")[0], "loc", text, count_tokens(text), False)


def test_priority_order_kept_and_duplicates_dropped():
    chunks = {c.chunk_id: c for c in (chunk("A::000", 10), chunk("B::000", 10), chunk("C::000", 10))}
    packed = pack_context(["B::000", "A::000", "B::000", "C::000"], chunks)
    assert packed.chunk_ids == ("B::000", "A::000", "C::000")
    assert packed.tokens == sum(count_tokens(c.text) for c in chunks.values())


def test_chunk_that_does_not_fit_is_skipped_but_smaller_ones_still_added():
    chunks = {c.chunk_id: c for c in (chunk("BIG::000", 400), chunk("MID::000", 400), chunk("SMALL::000", 20))}
    budget = count_tokens(chunks["BIG::000"].text) + count_tokens(chunks["SMALL::000"].text) + 5
    packed = pack_context(["BIG::000", "MID::000", "SMALL::000"], chunks, budget=budget)
    assert packed.chunk_ids == ("BIG::000", "SMALL::000")
    assert packed.omitted == ("MID::000",)


def test_never_over_the_limit_property():
    rng = random.Random(7)
    for trial in range(200):
        chunks = {f"D{trial}::{i:03d}": chunk(f"D{trial}::{i:03d}", rng.randint(1, 700)) for i in range(rng.randint(0, 25))}
        order = list(chunks)
        rng.shuffle(order)
        packed = pack_context(order, chunks)
        assert packed.tokens <= CORPUS_TOKEN_LIMIT
        assert packed.tokens == sum(count_tokens(s.text) for s in packed.segments)
        assert set(packed.chunk_ids) | set(packed.omitted) == set(chunks)
