from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


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
    def lines(self) -> list[str]:
        return [line.strip() for line in self.text.splitlines() if line.strip()]


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    question: str
    answerable: bool
    relevant_document_ids: tuple[str, ...]
    expected_answer: str


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
    payload = json.loads((data_dir / "questions.json").read_text())
    return [
        Question(
            id=item["id"],
            split=item["split"],
            question=item["question"],
            answerable=item["answerable"],
            relevant_document_ids=tuple(item["relevant_document_ids"]),
            expected_answer=item["expected_answer"],
        )
        for item in payload["questions"]
    ]
