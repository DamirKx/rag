"""Общие фикстуры тестов: лёгкий детерминированный эмбеддер без загрузки нейросети."""

import zlib
from typing import List

import numpy as np
import pytest

from src.retriever import tokenize_legal_text


class FakeEmbedder:
    """Эмбеддер «мешок слов» с хешированием токенов: близкие по словам тексты — близкие векторы."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.documents_encoded = 0

    def _encode(self, texts: List[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize_legal_text(text):
                vectors[i, zlib.crc32(token.encode("utf-8")) % self.dim] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-9)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return self._encode(queries)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        self.documents_encoded += len(documents)
        return self._encode(documents)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
