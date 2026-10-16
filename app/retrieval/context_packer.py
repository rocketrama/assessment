"""Token-budgeted context for the answer call (TASK.pdf §4.3, must-pass #4).

Chunks arrive in priority order (winning passages, conflicting ones, upcoming
ones, then other relevant context) and are added while they fit in
CORPUS_TOKEN_LIMIT cl100k tokens. A chunk that doesn't fit is skipped and the
next, smaller one is still tried. llm/client.py recounts and refuses anything
over the limit, so the budget is enforced twice.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.llm.client import CORPUS_TOKEN_LIMIT, CorpusSegment
from app.retrieval.corpus import ChunkRecord
from app.tokens import count_tokens


@dataclass(frozen=True)
class PackedContext:
    segments: tuple[CorpusSegment, ...]
    tokens: int
    omitted: tuple[str, ...]  # chunk ids that did not fit

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(s.chunk_id for s in self.segments)


def pack_context(
    ordered_chunk_ids: Sequence[str],
    chunks: Mapping[str, ChunkRecord],
    budget: int = CORPUS_TOKEN_LIMIT,
) -> PackedContext:
    segments: list[CorpusSegment] = []
    omitted: list[str] = []
    used = 0
    seen: set[str] = set()
    for chunk_id in ordered_chunk_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        text = chunks[chunk_id].text
        tokens = count_tokens(text)  # recount: what is sent is what is counted
        if used + tokens > budget:
            omitted.append(chunk_id)
            continue
        segments.append(CorpusSegment(chunk_id, text))
        used += tokens
    return PackedContext(tuple(segments), used, tuple(omitted))
