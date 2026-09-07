from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

import retrieval_eval_workbench.app as workbench_app
from retrieval_eval_workbench.answering import answer_query
from retrieval_eval_workbench.data import load_documents, load_frozen_settings
from retrieval_eval_workbench.retrieval import LexicalRetriever


def test_health_contract():
    response = TestClient(workbench_app.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "retrieval-eval-workbench", "version": "0.2.0"}


def test_lexical_query_returns_citation():
    response = TestClient(workbench_app.app).post("/api/query", json={"query": "Where do unmatched routing cases go?", "strategy": "lexical", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["citations"][0]["document_id"] == "routing"


def test_query_validation_rejects_empty_input():
    response = TestClient(workbench_app.app).post("/api/query", json={"query": "x", "strategy": "lexical"})
    assert response.status_code == 422


def test_api_default_matches_frozen_evaluator_settings():
    question = "How many downstream retries does ingestion use?"
    api_response = TestClient(workbench_app.app).post("/api/query", json={"query": question, "strategy": "lexical"})
    settings = load_frozen_settings()
    expected = answer_query(LexicalRetriever(load_documents()), question, limit=settings["top_k"], minimum_score=settings["retrievers"]["lexical"]["minimum_score"])
    assert api_response.status_code == 200
    assert api_response.json()["answer"] == expected.answer
    assert api_response.json()["effective_settings"]["minimum_score"] == settings["retrievers"]["lexical"]["minimum_score"]


def test_semantic_initialization_is_single_flight(monkeypatch):
    calls = 0

    class FakeSemantic:
        pass

    def construct_once():
        nonlocal calls
        calls += 1
        return FakeSemantic()

    workbench_app._semantic_initialized.cache_clear()
    monkeypatch.setattr(workbench_app, "_semantic_initialized", __import__("functools").lru_cache(maxsize=1)(construct_once))
    with ThreadPoolExecutor(max_workers=8) as pool:
        returned = list(pool.map(lambda _: workbench_app._semantic(), range(8)))
    assert calls == 1
    assert all(item is returned[0] for item in returned)
