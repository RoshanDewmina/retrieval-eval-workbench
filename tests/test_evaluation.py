import json

from retrieval_eval_workbench.answering import Citation, GroundedAnswer
from retrieval_eval_workbench.cli import benchmark
from retrieval_eval_workbench.data import Question, load_documents, load_questions
from retrieval_eval_workbench.evaluation import _citation_label_matches, _critical_fact_matches, regression_check, write_receipt


def _response(citation: Citation, answer: str | None = None) -> GroundedAnswer:
    return GroundedAnswer(answer=answer or citation.quote, citations=[citation], hits=[], answerable=True)


def test_forged_line_and_source_do_not_pass_labeled_support():
    question = next(item for item in load_questions() if item.id == "test-01")
    forged = Citation("authentication", "Authentication and sessions", 999, 999, "invented", "https://invented.invalid")
    assert _citation_label_matches(_response(forged), question, load_documents()) is False


def test_relevant_document_wrong_sentence_and_altered_fact_do_not_pass():
    question = next(item for item in load_questions() if item.id == "test-01")
    wrong_sentence = Citation("authentication", "Authentication and sessions", 3, 3, load_documents()[0].text.splitlines()[2], "https://example.invalid/meridian-service-docs/authentication")
    assert _citation_label_matches(_response(wrong_sentence), question, load_documents()) is False
    expected = question.supporting_citations[0]
    exact = Citation(expected.document_id, "Authentication and sessions", expected.line_start, expected.line_end, expected.quote, expected.source_url)
    altered = _response(exact, answer=expected.quote.replace("eight hours", "eighty hours"))
    assert _citation_label_matches(altered, question, load_documents()) is False
    assert _critical_fact_matches(altered, question) is False


def test_regression_check_fails_closed_for_missing_and_nan_metrics(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"minimum_metrics": {"expected": {"quality": 0.8}}}))
    missing = regression_check({"results": {}}, baseline)
    assert missing["passed"] is False
    assert missing["failures"][0]["kind"] == "missing_retriever"
    nan = regression_check({"results": {"expected": {"metrics": {"quality": float("nan")}}}}, baseline)
    assert nan["passed"] is False
    assert nan["failures"][0]["metric"] == "quality"


def test_failed_receipt_records_nonzero_exit_status(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text("{}")
    output = tmp_path / "failed.json"
    write_receipt({"regression": {"passed": False}}, output, "probe", 2, {"input": input_file})
    assert json.loads(output.read_text())["exit_status"] == 2


def test_cli_failure_writes_failed_receipt_without_model_load(monkeypatch, tmp_path):
    import retrieval_eval_workbench.cli as cli

    class FakeRetriever:
        def __init__(self, name): self.name = name

    monkeypatch.setattr(cli, "load_documents", lambda: [])
    monkeypatch.setattr(cli, "load_questions", lambda: [])
    monkeypatch.setattr(cli, "load_model_manifest", lambda: {"model_id": "fake", "revision": "r", "fingerprint": "m"})
    monkeypatch.setattr(cli, "load_frozen_settings", lambda: {"top_k": 3, "fingerprint": "s", "question_set_version": "dev", "retrievers": {"lexical": {"minimum_score": 0}, "semantic": {"minimum_score": 0}}})
    monkeypatch.setattr(cli, "LexicalRetriever", lambda _: FakeRetriever("lexical-tfidf"))
    monkeypatch.setattr(cli, "SemanticRetriever", lambda _: FakeRetriever("semantic-all-MiniLM-L6-v2"))
    monkeypatch.setattr(cli, "evaluate", lambda retriever, *args, **kwargs: {"metrics": {"probe": 0.0}})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"minimum_metrics": {"lexical-tfidf": {"probe": 1.0}, "semantic-all-MiniLM-L6-v2": {"probe": 1.0}}}))
    receipt = tmp_path / "receipt.json"
    assert benchmark(receipt, baseline) == 2
    assert json.loads(receipt.read_text())["exit_status"] == 2


def test_fresh_reference_rejects_drop_and_api_exposes_actual_cases():
    import copy
    from fastapi.testclient import TestClient
    from retrieval_eval_workbench.app import app
    from retrieval_eval_workbench.data import ROOT
    receipt = TestClient(app).get('/api/results').json()
    assert receipt['source_revision']=='661852a370ab35d6747a8bedab2a6b16864e9ccb'
    assert len(receipt['results']['semantic']['examples'])==20
    result={'results':{v['retriever']:v for v in receipt['results'].values()}}
    baseline=ROOT/'evidence/independent-v2/reference-baseline.json'
    assert regression_check(result,baseline)['passed']
    broken=copy.deepcopy(result)
    broken['results']['semantic-all-MiniLM-L6-v2']['metrics']['critical_fact_rate']=0
    assert not regression_check(broken,baseline)['passed']


def test_independent_regression_cli_preserves_failure_and_uses_frozen_labels(monkeypatch,tmp_path):
    import retrieval_eval_workbench.cli as cli
    class Stub:
        def __init__(self,name):self.name=name
    monkeypatch.setattr(cli,'LexicalRetriever',lambda _:Stub('lexical-tfidf'))
    monkeypatch.setattr(cli,'SemanticRetriever',lambda _:Stub('semantic-all-MiniLM-L6-v2'))
    def failed(retriever,questions,*args,**kwargs):
        assert len(questions)==20 and questions[0].supporting_citations
        return {'metrics':{'critical_fact_rate':0}}
    monkeypatch.setattr(cli,'evaluate',failed)
    dest=tmp_path/'failed.json'
    assert cli.benchmark_independent(dest)==2
    result=json.loads(dest.read_text())
    assert result['exit_status']==2
    assert result['measured_results']['evaluation_status']=='exposed_independent_set_regression_rerun'
