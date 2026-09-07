from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

from .answering import answer_query, tokens
from .data import Question
from .retrieval import Retriever


def _git(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def evaluate(retriever: Retriever, questions: list[Question], split: str, limit: int = 3, threshold: float = 0.16) -> dict:
    selected = [question for question in questions if question.split == split]
    answerable = [question for question in selected if question.answerable]
    rows = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_correct: list[float] = []
    grounded_correct: list[float] = []
    unanswerable_correct: list[float] = []
    for question in selected:
        response = answer_query(retriever, question.question, limit=limit, minimum_score=threshold)
        ranked = [hit.document.id for hit in response.hits]
        first_rank = next((index + 1 for index, doc_id in enumerate(ranked) if doc_id in question.relevant_document_ids), None)
        if question.answerable:
            recalls.append(float(first_rank is not None))
            reciprocal_ranks.append(0.0 if first_rank is None else 1 / first_rank)
            valid_citation = bool(response.citations) and response.citations[0].document_id in question.relevant_document_ids
            citation_correct.append(float(valid_citation))
            expected_terms = tokens(question.expected_answer)
            answer_overlap = len(expected_terms & tokens(response.answer)) / max(1, len(expected_terms))
            grounded_correct.append(float(valid_citation and answer_overlap >= 0.55))
        else:
            unanswerable_correct.append(float(not response.answerable and not response.citations))
        rows.append(
            {
                "question_id": question.id,
                "question": question.question,
                "answerable_expected": question.answerable,
                "answerable_predicted": response.answerable,
                "ranked_document_ids": ranked,
                "answer": response.answer,
                "citations": [citation.__dict__ for citation in response.citations],
                "expected_document_ids": list(question.relevant_document_ids),
                "retrieval_rank": first_rank,
            }
        )
    return {
        "retriever": retriever.name,
        "split": split,
        "question_count": len(selected),
        "answerable_question_count": len(answerable),
        "metrics": {
            "recall_at_3": mean(recalls) if recalls else None,
            "mrr_at_3": mean(reciprocal_ranks) if reciprocal_ranks else None,
            "citation_correct_rate": mean(citation_correct) if citation_correct else None,
            "grounded_answer_rate": mean(grounded_correct) if grounded_correct else None,
            "unanswerable_correct_rate": mean(unanswerable_correct) if unanswerable_correct else None,
        },
        "examples": rows,
        "conditions": {"top_k": limit, "unanswerable_threshold": threshold, "answering_method": "deterministic extractive sentence selection"},
    }


def regression_check(results: dict, baseline_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text())
    failures = []
    for retriever_name, current in results["results"].items():
        expected = baseline["minimum_metrics"][retriever_name]
        for metric, minimum in expected.items():
            actual = current["metrics"][metric]
            if actual is None or actual < minimum:
                failures.append({"retriever": retriever_name, "metric": metric, "minimum": minimum, "actual": actual})
    return {"passed": not failures, "failures": failures, "baseline": str(baseline_path)}


def write_receipt(results: dict, destination: Path, command: str) -> None:
    receipt = {
        "schema_version": 1,
        "source_revision": _git(["git", "rev-parse", "HEAD"]),
        "dirty_tree": bool(_git(["git", "status", "--porcelain"])),
        "command": command,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "exit_status": 0,
        "environment": {"os": platform.platform(), "architecture": platform.machine(), "python": sys.version.split()[0]},
        "inputs": {"dataset_id": "meridian-service-docs", "dataset_version": "1.0.0", "seed": 17},
        "measured_results": results,
        "limitations": [
            "The corpus and questions are original synthetic service documentation; these metrics do not establish performance on private or production corpora.",
            "The deterministic extractive answerer is evaluated for citation support, not answer fluency or broad factual correctness.",
            "A small held-out set is a regression signal, not statistically conclusive evidence.",
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
