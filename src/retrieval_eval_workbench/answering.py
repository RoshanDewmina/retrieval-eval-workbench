from __future__ import annotations

import re
from dataclasses import dataclass

from .data import Document
from .retrieval import Retriever, SearchHit


STOPWORDS = {"a", "an", "and", "are", "corpus", "does", "for", "how", "in", "is", "meridian", "of", "on", "service", "the", "to", "use", "what", "when", "where", "which", "with"}


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


def citation_matches_corpus(citation: Citation, documents: list[Document]) -> bool:
    """Check document identity, physical source range, quote, and immutable source URL."""
    document = next((item for item in documents if item.id == citation.document_id), None)
    if document is None or document.source_url != citation.source_url:
        return False
    if citation.line_start < 1 or citation.line_end != citation.line_start:
        return False
    lines = document.text.splitlines()
    if citation.line_start > len(lines):
        return False
    return lines[citation.line_start - 1].strip() == citation.quote


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
    candidate: tuple[float, int, SearchHit, int, str] | None = None
    for hit in hits:
        for line_number, line in hit.document.numbered_body_lines:
            overlap = len(query_terms & tokens(line))
            score = overlap + hit.score * 0.35
            if candidate is None or score > candidate[0]:
                candidate = (score, overlap, hit, line_number, line)
    assert candidate is not None
    _, overlap, hit, line_number, sentence = candidate
    if overlap == 0:
        return GroundedAnswer(
            answer="I could not find a corpus sentence that supports that question.",
            citations=[],
            hits=hits,
            answerable=False,
        )
    citation = Citation(
        document_id=hit.document.id,
        title=hit.document.title,
        line_start=line_number,
        line_end=line_number,
        quote=sentence,
        source_url=hit.document.source_url,
    )
    return GroundedAnswer(answer=sentence, citations=[citation], hits=hits, answerable=True)
