import numpy as np

from retrieval_eval_workbench.answering import answer_query
from retrieval_eval_workbench.data import load_documents
from retrieval_eval_workbench.retrieval import LexicalRetriever, SemanticRetriever


def test_manifest_hashes_and_document_count():
    documents = load_documents()
    assert len(documents) == 10
    assert {document.id for document in documents} == {"authentication", "availability", "exports", "imports", "incidents", "notifications", "quality", "retention", "roles", "routing"}


def test_lexical_retrieval_finds_routing_document():
    retriever = LexicalRetriever(load_documents())
    assert retriever.search("Where do unmatched routing cases go?", limit=1)[0].document.id == "routing"


def test_extractive_answer_has_line_level_citation():
    answer = answer_query(LexicalRetriever(load_documents()), "Where do unmatched routing cases go?")
    assert answer.answerable is True
    assert answer.citations[0].document_id == "routing"
    assert answer.citations[0].line_start >= 1


class _TinyEmbeddingModel:
    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            lower = sentence.lower()
            vectors.append([float("case routing rules" in lower or "which queue" in lower), float("retention" in lower or "deletion" in lower)])
        array = np.array(vectors, dtype=float)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.where(norms == 0, 1, norms)


def test_semantic_retriever_uses_embedding_similarity():
    retriever = SemanticRetriever(load_documents(), model=_TinyEmbeddingModel())
    assert retriever.search("Which queue gets a routing case?", limit=1)[0].document.id == "routing"
