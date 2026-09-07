from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DATA_DIR, ROOT, load_documents, load_frozen_settings, load_model_manifest, load_questions
from .evaluation import evaluate, regression_check, write_receipt
from .retrieval import LexicalRetriever, SemanticRetriever


def benchmark(receipt_path: Path, baseline_path: Path) -> int:
    documents = load_documents()
    questions = load_questions()
    settings = load_frozen_settings()
    model = load_model_manifest()
    retrievers = [("lexical", LexicalRetriever(documents)), ("semantic", SemanticRetriever(documents))]
    results = {"dataset": "meridian-service-docs@1.1.0-exposed-development", "evaluation_status": "development_regression_only", "results": {}}
    for key, retriever in retrievers:
        configured = settings["retrievers"][key]
        results["results"][retriever.name] = evaluate(
            retriever, questions, documents, split="test", limit=settings["top_k"], threshold=configured["minimum_score"]
        )
    results["frozen_settings"] = {"fingerprint": settings["fingerprint"], "question_set_version": settings["question_set_version"]}
    results["model"] = {"model_id": model["model_id"], "revision": model["revision"], "fingerprint": model["fingerprint"]}
    results["regression"] = regression_check(results, baseline_path)
    status = 0 if results["regression"]["passed"] else 2
    write_receipt(results, receipt_path, "make benchmark", status, {
        "dataset_manifest": DATA_DIR / "manifest.json", "questions": DATA_DIR / "questions.json",
        "settings": DATA_DIR / "calibration-v1.json", "model_manifest": DATA_DIR / "model-manifest-v1.json", "baseline": baseline_path,
    })
    print(json.dumps(results, indent=2))
    return status


def benchmark_independent(receipt_path: Path) -> int:
    """Re-run the now-exposed independent set; never label reruns fresh heldout."""
    from .data import Question, ExpectedCitation, file_sha256
    directory = ROOT / "evidence" / "independent-v2"
    manifest = json.loads((directory / "manifest.json").read_text())
    for filename, expected in manifest["files"].items():
        if file_sha256(directory / filename) != expected:
            raise ValueError(f"Independent input hash mismatch: {filename}")
    labels = json.loads((directory / "labels.json").read_text())["labels"]
    questions = [Question(**{**q, "relevant_document_ids": tuple(q["relevant_document_ids"]),
        "supporting_citations": tuple(ExpectedCitation(**c) for c in labels[q["id"]]["supporting_citations"]),
        "required_answer_phrases": tuple(labels[q["id"]]["required_answer_phrases"])})
        for q in json.loads((directory / "questions.json").read_text())["questions"]]
    docs, settings = load_documents(), load_frozen_settings()
    result = {"evaluation_status": "exposed_independent_set_regression_rerun", "results": {}}
    for key, cls in (("lexical", LexicalRetriever), ("semantic", SemanticRetriever)):
        retriever = cls(docs)
        result["results"][retriever.name] = evaluate(retriever, questions, docs, split="test",
            limit=settings["top_k"], threshold=settings["retrievers"][key]["minimum_score"])
    result["regression"] = regression_check(result, directory / "reference-baseline.json")
    status = 0 if result["regression"]["passed"] else 2
    write_receipt(result, receipt_path, "make benchmark", status, {
        "questions": directory / "questions.json", "labels": directory / "labels.json",
        "baseline": directory / "reference-baseline.json", "settings": DATA_DIR / "calibration-v1.json",
        "model_manifest": DATA_DIR / "model-manifest-v1.json", "corpus": DATA_DIR / "manifest.json"})
    print(json.dumps(result, indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["benchmark", "independent-regression"])
    parser.add_argument("--receipt", type=Path, default=ROOT / "evidence" / "benchmarks" / "frozen-run.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "evidence" / "baseline-v2-freeze.json")
    args = parser.parse_args()
    raise SystemExit(benchmark_independent(args.receipt) if args.command == "independent-regression" else benchmark(args.receipt, args.baseline))


if __name__ == "__main__":
    main()
