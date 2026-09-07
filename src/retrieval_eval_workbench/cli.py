from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import ROOT, load_documents, load_questions
from .evaluation import evaluate, regression_check, select_unanswerable_threshold, write_receipt
from .retrieval import LexicalRetriever, SemanticRetriever


def benchmark() -> int:
    documents = load_documents()
    questions = load_questions()
    retrievers = [LexicalRetriever(documents), SemanticRetriever(documents)]
    results = {"dataset": "meridian-service-docs@1.0.0", "results": {}}
    results["calibration"] = {}
    for retriever in retrievers:
        calibration = select_unanswerable_threshold(retriever, questions)
        results["calibration"][retriever.name] = calibration
        results["results"][retriever.name] = evaluate(retriever, questions, split="test", threshold=calibration["threshold"])
    results["regression"] = regression_check(results, ROOT / "evidence" / "baseline.json")
    write_receipt(results, ROOT / "evidence" / "benchmarks" / "latest.json", "make benchmark")
    print(json.dumps(results, indent=2))
    return 0 if results["regression"]["passed"] else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["benchmark"])
    args = parser.parse_args()
    raise SystemExit(benchmark() if args.command == "benchmark" else 2)


if __name__ == "__main__":
    main()
