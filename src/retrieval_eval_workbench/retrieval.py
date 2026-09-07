from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import Document, file_sha256, load_model_manifest


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, limit: int = 3) -> list[SearchHit]: ...


class LexicalRetriever:
    name = "lexical-tfidf"

    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([f"{doc.title}\n{doc.text}" for doc in documents])

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = sorted(range(len(scores)), key=lambda index: (-scores[index], self.documents[index].id))[:limit]
        return [SearchHit(self.documents[index], float(scores[index])) for index in indices]


class EmbeddingModel(Protocol):
    def encode(self, sentences: list[str], **kwargs: object): ...


class SemanticRetriever:
    """Dense cosine retrieval backed by a real, locally downloaded pretrained encoder."""

    name = "semantic-all-MiniLM-L6-v2"

    def __init__(self, documents: list[Document], model: EmbeddingModel | None = None, model_manifest: dict | None = None) -> None:
        self.documents = documents
        self.model_manifest = model_manifest or load_model_manifest()
        if model is None:
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import snapshot_download

            snapshot = snapshot_download(
                repo_id=self.model_manifest["model_id"],
                revision=self.model_manifest["revision"],
            )
            self._verify_snapshot(Path(snapshot))
            model = SentenceTransformer(snapshot, device="cpu")
        self.model = model
        self.embeddings = self.model.encode(
            [f"{doc.title}. {doc.text}" for doc in documents], normalize_embeddings=True, show_progress_bar=False
        )

    def _verify_snapshot(self, snapshot: Path) -> None:
        """Fail closed when a pinned model file differs from the approved manifest."""
        for expected in self.model_manifest["files"]:
            path = snapshot / expected["path"]
            if not path.is_file() or file_sha256(path) != expected["sha256"]:
                raise ValueError(f"model manifest mismatch for {expected['path']}")

    def search(self, query: str, limit: int = 3) -> list[SearchHit]:
        query_embedding = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ query_embedding
        indices = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), self.documents[index].id))[:limit]
        return [SearchHit(self.documents[index], float(scores[index])) for index in indices]
