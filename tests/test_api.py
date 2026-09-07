from fastapi.testclient import TestClient

from retrieval_eval_workbench.app import app


def test_health_contract():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "retrieval-eval-workbench", "version": "0.1.0"}


def test_lexical_query_returns_citation():
    response = TestClient(app).post("/api/query", json={"query": "Where do unmatched routing cases go?", "strategy": "lexical", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["citations"][0]["document_id"] == "routing"


def test_query_validation_rejects_empty_input():
    response = TestClient(app).post("/api/query", json={"query": "x", "strategy": "lexical"})
    assert response.status_code == 422
