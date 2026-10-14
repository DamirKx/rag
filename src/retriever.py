"""Модуль векторного индексирования и семантического поиска на базе FAISS.

Использует библиотеку Facebook AI Similarity Search (FAISS) для быстрого
поиска k ближайших соседей (k-NN) в пространстве высокой размерности.
"""

from typing import List, Tuple
import faiss
import numpy as np


class VectorRetriever:
    """Векторный ретривер на основе точного скалярного произведения (Inner Product).

    Особенности реализации:
    - Использует индекс `faiss.IndexFlatIP` (Exact Inner Product Search).
    - Так как векторы документов и запросов нормализованы по L2 (их норма равна 1),
      скалярное произведение <u, v> математически строго эквивалентно косинусному сходству:
          cos(u, v) = <u, v> / (||u|| * ||v||) = <u, v>
    - Индекс 'Flat' осуществляет полный перебор без квантования (exhaustive search),
      что гарантирует 100% точность поиска и делает его идеальным эталоном (Ground Truth)
      для сравнительной оценки качества в исследовательских задачах.
    """

    def __init__(self, embedding_dim: int):
        """Инициализирует пустой индекс FAISS заданной размерности.

        Args:
            embedding_dim (int): Размерность пространства эмбеддингов
                                 (например, 384 для MiniLM, 768 для e5-base/ruBERT, 1024 для bge-m3).
        """
        self.embedding_dim = embedding_dim

        # Создаем плоский индекс для скалярного произведения (Inner Product)
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Списки для сопоставления порядкового номера в индексе с метаданными
        self.documents: List[str] = []
        self.doc_ids: List[str] = []

    def add_documents(self, embeddings: np.ndarray, documents: List[str], doc_ids: List[str]) -> None:
        """Добавляет векторы документов и соответствующие метаданные в индекс.

        Args:
            embeddings (np.ndarray): Двумерный массив эмбеддингов формы (N_docs, embedding_dim).
            documents (List[str]): Список исходных текстов чанков.
            doc_ids (List[str]): Список уникальных строковых идентификаторов чанков.

        Raises:
            AssertionError: Если количество векторов, текстов и идентификаторов не совпадает.
        """
        assert len(embeddings) == len(documents) == len(doc_ids), (
            f"Несовпадение размерностей: эмбеддингов={len(embeddings)}, "
            f"документов={len(documents)}, идентификаторов={len(doc_ids)}"
        )

        # Гарантируем, что массив векторов непрерывен в памяти C++ и имеет тип float32
        embeddings_contiguous = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Добавляем векторы в C++ индекс FAISS
        self.index.add(embeddings_contiguous)

        # Сохраняем текстовое содержимое и идентификаторы в синхронизированные списки
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)

    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Осуществляет поиск k наиболее релевантных документов для каждого переданного запроса.

        Args:
            query_embeddings (np.ndarray): Массив эмбеддингов запросов формы (N_queries, embedding_dim).
            k (int): Количество возвращаемых ближайших соседей (top-k). По умолчанию 5.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - distances: Массив оценок сходства формы (N_queries, k).
                             Чем выше значение, тем ближе семантический смысл.
                - indices: Массив целочисленных индексов документов формы (N_queries, k).
        """
        # Проверяем формат массива запросов
        queries_contiguous = np.ascontiguousarray(query_embeddings, dtype=np.float32)

        # Ограничиваем k максимальным числом доступных документов в индексе
        actual_k = min(k, self.index.ntotal)

        # Запуск параллельного поиска в FAISS
        distances, indices = self.index.search(queries_contiguous, actual_k)
        return distances, indices

    def get_doc_id(self, idx: int) -> str:
        """Возвращает строковый идентификатор документа по его внутреннему порядковому номеру FAISS.

        Args:
            idx (int): Индекс документа в хранилище FAISS.

        Returns:
            str: Уникальный строковый идентификатор документа/чанка.
        """
        return self.doc_ids[idx]

    def get_document(self, idx: int) -> str:
        """Возвращает исходный текст документа по его внутреннему порядковому номеру FAISS.

        Args:
            idx (int): Индекс документа в хранилище FAISS.

        Returns:
            str: Исходный текстовый фрагмент (чанк).
        """
        return self.documents[idx]
