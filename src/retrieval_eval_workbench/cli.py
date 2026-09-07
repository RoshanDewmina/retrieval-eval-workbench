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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["benchmark"])
    parser.add_argument("--receipt", type=Path, default=ROOT / "evidence" / "benchmarks" / "frozen-run.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "evidence" / "baseline-v2-freeze.json")
    args = parser.parse_args()
    raise SystemExit(benchmark(args.receipt, args.baseline))


if __name__ == "__main__":
    main()
