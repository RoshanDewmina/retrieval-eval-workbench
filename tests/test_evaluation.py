import json

from retrieval_eval_workbench.evaluation import regression_check, select_unanswerable_threshold


def test_regression_check_reports_a_named_metric_drop(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"minimum_metrics": {"demo": {"recall_at_3": 0.8}}}))
    result = regression_check({"results": {"demo": {"metrics": {"recall_at_3": 0.5}}}}, baseline)
    assert result["passed"] is False
    assert result["failures"] == [{"retriever": "demo", "metric": "recall_at_3", "minimum": 0.8, "actual": 0.5}]


class _ScoredRetriever:
    name = "scored"

    def search(self, query, limit=1):
        if query.startswith("test"):
            raise AssertionError("held-out question entered threshold calibration")
        return [type("Hit", (), {"score": 0.8 if query == "answerable" else 0.1})()]


def test_threshold_calibration_uses_tuning_questions_only():
    Question = type("Question", (), {})
    answerable = Question()
    answerable.id, answerable.split, answerable.question, answerable.answerable = "tune-1", "tuning", "answerable", True
    unanswerable = Question()
    unanswerable.id, unanswerable.split, unanswerable.question, unanswerable.answerable = "tune-2", "tuning", "unanswerable", False
    held_out = Question()
    held_out.id, held_out.split, held_out.question, held_out.answerable = "test-1", "test", "test should not run", True
    calibration = select_unanswerable_threshold(_ScoredRetriever(), [answerable, unanswerable, held_out])
    assert calibration["tuning_question_count"] == 2
    assert calibration["threshold"] > 0.1
