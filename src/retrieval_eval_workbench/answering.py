from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import Retriever, SearchHit


STOPWORDS = {"a", "an", "and", "are", "does", "for", "how", "in", "is", "of", "on", "the", "to", "what", "when", "where", "which", "with"}


@dataclass(frozen=True)
class Citation:
    document_id: str
    title: str
    line_start: int
    line_end: int
    quote: str
    source_url: str


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[Citation]
    hits: list[SearchHit]
    answerable: bool


def tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS}


def answer_query(retriever: Retriever, query: str, limit: int = 3, minimum_score: float = 0.16) -> GroundedAnswer:
    hits = retriever.search(query, limit=limit)
    if not hits or hits[0].score < minimum_score:
        return GroundedAnswer(
            answer="I could not find supporting evidence for that question in this corpus.",
            citations=[],
            hits=hits,
            answerable=False,
        )
    query_terms = tokens(query)
    candidate: tuple[float, SearchHit, int, str] | None = None
    for hit in hits:
        for line_number, line in enumerate(hit.document.lines, start=1):
            overlap = len(query_terms & tokens(line))
            score = overlap + hit.score * 0.35
            if candidate is None or score > candidate[0]:
                candidate = (score, hit, line_number, line)
    assert candidate is not None
    _, hit, line_number, sentence = candidate
    citation = Citation(
        document_id=hit.document.id,
        title=hit.document.title,
        line_start=line_number,
        line_end=line_number,
        quote=sentence,
        source_url=hit.document.source_url,
    )
    return GroundedAnswer(answer=sentence, citations=[citation], hits=hits, answerable=True)
