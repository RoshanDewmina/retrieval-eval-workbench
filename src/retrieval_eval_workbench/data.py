from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    license: str
    source_url: str

    @property
    def numbered_body_lines(self) -> list[tuple[int, str]]:
        """Physical source line numbers, excluding Markdown headings and blank lines."""
        return [
            (line_number, line.strip())
            for line_number, line in enumerate(self.text.splitlines(), start=1)
            if line.strip() and not line.startswith("#")
        ]


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    question: str
    answerable: bool
    relevant_document_ids: tuple[str, ...]
    expected_answer: str
    supporting_citations: tuple["ExpectedCitation", ...] = ()
    required_answer_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpectedCitation:
    document_id: str
    line_start: int
    line_end: int
    quote: str
    source_url: str


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    documents = []
    for entry in manifest["documents"]:
        path = data_dir / entry["path"]
        contents = path.read_text().strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"hash mismatch for {entry['id']}")
        documents.append(Document(entry["id"], entry["title"], contents, manifest["license"], entry["source_url"]))
    return documents


def load_questions(data_dir: Path = DATA_DIR) -> list[Question]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    path = data_dir / "questions.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["questions_sha256"]:
        raise ValueError("hash mismatch for questions.json")
    payload = json.loads(path.read_text())
    labels_path = data_dir / "development-labels-v1.json"
    labels: dict[str, Any] = json.loads(labels_path.read_text())["labels"] if labels_path.exists() else {}
    return [
        Question(
            id=item["id"],
            split=item["split"],
            question=item["question"],
            answerable=item["answerable"],
            relevant_document_ids=tuple(item["relevant_document_ids"]),
            expected_answer=item["expected_answer"],
            supporting_citations=tuple(ExpectedCitation(**citation) for citation in labels.get(item["id"], {}).get("supporting_citations", [])),
            required_answer_phrases=tuple(labels.get(item["id"], {}).get("required_answer_phrases", [])),
        )
        for item in payload["questions"]
    ]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_settings(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    path = data_dir / "calibration-v1.json"
    settings = load_json(path)
    settings["fingerprint"] = file_sha256(path)
    return settings


def load_model_manifest(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    path = data_dir / "model-manifest-v1.json"
    model = load_json(path)
    model["fingerprint"] = file_sha256(path)
    return model
