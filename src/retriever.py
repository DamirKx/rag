import faiss
import numpy as np
from typing import List, Tuple

class VectorRetriever:
    def __init__(self, embedding_dim: int):
        """
        Используем FAISS для быстрого поиска ближайших соседей.
        Поскольку мы нормализуем эмбеддинги (normalize_embeddings=True в SentenceTransformer),
        можно использовать скалярное произведение (Inner Product), которое будет равно косинусному сходству.
        """
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents = []
        self.doc_ids = []

    def add_documents(self, embeddings: np.ndarray, documents: List[str], doc_ids: List[str]):
        """
        Добавление векторов в индекс FAISS.
        """
        assert len(embeddings) == len(documents) == len(doc_ids)
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск k наиболее релевантных документов.
        Возвращает:
        - distances: значения косинусного сходства (N_queries x k)
        - indices: индексы в FAISS индексе (N_queries x k)
        """
        distances, indices = self.index.search(query_embeddings, k)
        return distances, indices

    def get_doc_id(self, idx: int) -> str:
        return self.doc_ids[idx]
