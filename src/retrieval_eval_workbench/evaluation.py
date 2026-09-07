from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .answering import GroundedAnswer, answer_query, citation_matches_corpus, tokens
from .data import Document, Question, file_sha256
from .retrieval import Retriever


def _git(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def select_unanswerable_threshold(retriever: Retriever, questions: list[Question]) -> dict:
    """Developer-only helper; frozen settings, not evaluation results, drive serving."""
    tuning = [question for question in questions if question.split == "tuning"]
    observations = [(question, retriever.search(question.question, limit=1)[0].score) for question in tuning]
    scores = sorted({round(score, 6) for _, score in observations})
    candidates = [0.0, *[(left + right) / 2 for left, right in zip(scores, scores[1:])], min(1.0, scores[-1] + 0.01)]

    def score(threshold: float) -> float:
        return mean(float((value >= threshold) == question.answerable) for question, value in observations)

    selected = max(candidates, key=lambda threshold: (score(threshold), -threshold))
    return {"threshold": selected, "tuning_question_count": len(tuning), "tuning_accuracy": score(selected)}


def _citation_label_matches(response: GroundedAnswer, question: Question, documents: list[Document]) -> bool:
    if len(response.citations) != 1 or response.answer != response.citations[0].quote:
        return False
    citation = response.citations[0]
    if not citation_matches_corpus(citation, documents):
        return False
    return any(
        citation.document_id == expected.document_id
        and citation.line_start == expected.line_start
        and citation.line_end == expected.line_end
        and citation.quote == expected.quote
        and citation.source_url == expected.source_url
        for expected in question.supporting_citations
    )


def _critical_fact_matches(response: GroundedAnswer, question: Question) -> bool:
    answer = response.answer.lower()
    return bool(question.required_answer_phrases) and all(phrase.lower() in answer for phrase in question.required_answer_phrases)


def evaluate(retriever: Retriever, questions: list[Question], documents: list[Document], split: str, limit: int, threshold: float) -> dict:
    selected = [question for question in questions if question.split == split]
    answerable = [question for question in selected if question.answerable]
    rows = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_integrity: list[float] = []
    labeled_support: list[float] = []
    critical_fact: list[float] = []
    answer_overlap_proxy: list[float] = []
    unanswerable_correct: list[float] = []
    for question in selected:
        response = answer_query(retriever, question.question, limit=limit, minimum_score=threshold)
        ranked = [hit.document.id for hit in response.hits]
        first_rank = next((index + 1 for index, doc_id in enumerate(ranked) if doc_id in question.relevant_document_ids), None)
        if question.answerable:
            recalls.append(float(first_rank is not None))
            reciprocal_ranks.append(0.0 if first_rank is None else 1 / first_rank)
            local_valid = bool(response.citations) and citation_matches_corpus(response.citations[0], documents)
            support_valid = _citation_label_matches(response, question, documents)
            citation_integrity.append(float(local_valid))
            labeled_support.append(float(support_valid))
            critical_fact.append(float(support_valid and _critical_fact_matches(response, question)))
            expected_terms = tokens(question.expected_answer)
            answer_overlap_proxy.append(float(len(expected_terms & tokens(response.answer)) / max(1, len(expected_terms))))
        else:
            unanswerable_correct.append(float(not response.answerable and not response.citations))
        rows.append({
            "question_id": question.id, "question": question.question, "answerable_expected": question.answerable,
            "answerable_predicted": response.answerable, "ranked_document_ids": ranked, "answer": response.answer,
            "citations": [citation.__dict__ for citation in response.citations], "expected_document_ids": list(question.relevant_document_ids),
            "retrieval_rank": first_rank, "citation_integrity": None if not question.answerable else bool(citation_integrity[-1]),
            "labeled_support": None if not question.answerable else bool(labeled_support[-1]),
            "critical_fact": None if not question.answerable else bool(critical_fact[-1]),
        })
    return {
        "retriever": retriever.name, "split": split, "question_count": len(selected), "answerable_question_count": len(answerable),
        "metrics": {
            "recall_at_3": mean(recalls) if recalls else None, "mrr_at_3": mean(reciprocal_ranks) if reciprocal_ranks else None,
            "citation_integrity_rate": mean(citation_integrity) if citation_integrity else None,
            "labeled_support_rate": mean(labeled_support) if labeled_support else None,
            "critical_fact_rate": mean(critical_fact) if critical_fact else None,
            "answer_term_overlap_proxy": mean(answer_overlap_proxy) if answer_overlap_proxy else None,
            "unanswerable_correct_rate": mean(unanswerable_correct) if unanswerable_correct else None,
        },
        "examples": rows,
        "conditions": {"top_k": limit, "unanswerable_threshold": threshold, "answering_method": "deterministic extractive sentence selection"},
    }


def regression_check(results: dict, baseline_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text())
    expected = baseline.get("minimum_metrics")
    actual_results = results.get("results")
    failures = []
    if not isinstance(expected, dict) or not isinstance(actual_results, dict):
        return {"passed": False, "failures": [{"kind": "malformed_results_or_baseline"}], "baseline": str(baseline_path)}
    for retriever_name in sorted(set(expected) | set(actual_results)):
        if retriever_name not in expected:
            failures.append({"retriever": retriever_name, "kind": "unexpected_retriever"})
            continue
        if retriever_name not in actual_results:
            failures.append({"retriever": retriever_name, "kind": "missing_retriever"})
            continue
        current_metrics = actual_results[retriever_name].get("metrics", {})
        for metric, minimum in expected[retriever_name].items():
            actual = current_metrics.get(metric)
            if not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < minimum:
                failures.append({"retriever": retriever_name, "metric": metric, "minimum": minimum, "actual": actual})
    return {"passed": not failures, "failures": failures, "baseline": str(baseline_path)}


def write_receipt(results: dict, destination: Path, command: str, exit_status: int, input_files: dict[str, Path]) -> None:
    receipt = {
        "schema_version": 2, "source_revision": _git(["git", "rev-parse", "HEAD"]),
        "dirty_tree": bool(_git(["git", "status", "--porcelain"])), "command": command,
        "timestamp_utc": datetime.now(UTC).isoformat(), "exit_status": exit_status,
        "environment": {"os": platform.platform(), "architecture": platform.machine(), "python": sys.version.split()[0]},
        "input_fingerprints": {name: file_sha256(path) for name, path in input_files.items()}, "measured_results": results,
        "limitations": [
            "Original and evolved question sets are exposed development/regression history, not a fresh held-out evaluation.",
            "Fresh independently authored evaluation cases must be frozen before final evaluation and never used for correction or calibration.",
            "The deterministic extractive answerer is evaluated for labeled source support, not fluent synthesis or real-world factual correctness.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
