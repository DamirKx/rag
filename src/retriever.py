"""Модуль векторного индексирования и семантического поиска на базе FAISS.

Использует библиотеку Facebook AI Similarity Search (FAISS) для быстрого
поиска k ближайших соседей (k-NN) в пространстве высокой размерности.
"""

from typing import List, Tuple, Dict, Any, Optional
import re
import faiss
import numpy as np
from rank_bm25 import BM25Okapi


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
                                 (например, 384 для e5-small, 768 для e5-base, 1024 для bge-m3).
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
                  Если индекс пуст, возвращаются массивы формы (N_queries, 0).
        """
        # Проверяем формат массива запросов
        queries_contiguous = np.ascontiguousarray(query_embeddings, dtype=np.float32)

        # Ограничиваем k максимальным числом доступных документов в индексе
        actual_k = min(k, self.index.ntotal)
        if actual_k <= 0:
            n_queries = queries_contiguous.shape[0]
            return np.empty((n_queries, 0), dtype=np.float32), np.empty((n_queries, 0), dtype=np.int64)

        # Запуск параллельного поиска в FAISS
        distances, indices = self.index.search(queries_contiguous, actual_k)
        return distances, indices

    def get_doc_id(self, idx: int) -> str:
        """Возвращает строковый идентификатор документа по его внутреннему порядковому номеру FAISS.

        Args:
            idx (int): Индекс документа в хранилище FAISS.

        Returns:
            str: Уникальный строковый идентификатор документа/чанка.

        Raises:
            IndexError: Если idx вне диапазона (в т.ч. -1, которым FAISS помечает пустые позиции).
        """
        if not 0 <= idx < len(self.doc_ids):
            raise IndexError(f"Индекс документа вне диапазона: {idx}")
        return self.doc_ids[idx]

    def get_document(self, idx: int) -> str:
        """Возвращает исходный текст документа по его внутреннему порядковому номеру FAISS.

        Args:
            idx (int): Индекс документа в хранилище FAISS.

        Returns:
            str: Исходный текстовый фрагмент (чанк).

        Raises:
            IndexError: Если idx вне диапазона (в т.ч. -1, которым FAISS помечает пустые позиции).
        """
        if not 0 <= idx < len(self.documents):
            raise IndexError(f"Индекс документа вне диапазона: {idx}")
        return self.documents[idx]


def tokenize_legal_text(text: str) -> List[str]:
    """Разбивает юридический текст на лексические токены.

    Приводит текст к нижнему регистру и сохраняет составные номера законов,
    статей и пунктов с дефисами: например, '221-vii', '28-1', '122', 'мрп'.
    Знаки препинания отбрасываются ('ст.' -> 'ст').
    """
    return re.findall(r"(?u)\b[\w\-]+\b", text.lower())


class BM25Retriever:
    """Лексический ретривер на основе алгоритма BM25 (Okapi BM25).

    Специализируется на точном поиске ключевых слов:
    - Номера законов, статей и пунктов (например, '№ 221-VII', 'статья 122', 'пункт 8-1');
    - Даты и числовые нормативы ('05.04.2023', '311');
    - Специальные аббревиатуры и термины ('МРП', 'ЗҚАИ').
    """

    def __init__(self):
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    def add_documents(self, documents: List[str], doc_ids: List[str]) -> None:
        """Индексирует текстовые документы с предварительной токенизацией.

        Args:
            documents (List[str]): Список исходных текстов чанков.
            doc_ids (List[str]): Список уникальных строковых идентификаторов чанков.
        """
        assert len(documents) == len(doc_ids), (
            f"Несовпадение размерностей: документов={len(documents)}, идентификаторов={len(doc_ids)}"
        )
        self.documents = list(documents)
        self.doc_ids = list(doc_ids)
        tokenized_corpus = [tokenize_legal_text(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Осуществляет поиск k наиболее релевантных документов по текстовому запросу.

        Args:
            query (str): Текстовый поисковый запрос.
            k (int): Количество возвращаемых результатов. По умолчанию 5.

        Returns:
            List[Tuple[str, float]]: Список пар (doc_id, score), отсортированных по убыванию релевантности.
        """
        if not self.bm25 or not self.documents:
            return []

        tokens = tokenize_legal_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        actual_k = min(k, len(self.documents))
        top_indices = np.argsort(scores)[::-1][:actual_k]

        return [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def batch_search(self, queries: List[str], k: int = 5) -> List[List[str]]:
        """Пакетный поиск топ-k doc_id для списка текстовых запросов."""
        results = []
        for q in queries:
            top_pairs = self.search(q, k=k)
            results.append([doc_id for doc_id, _ in top_pairs])
        return results


class HybridRetriever:
    """Гибридный ретривер, объединяющий плотный (Dense/FAISS) и разреженный (Sparse/BM25) поиск.

    Использует алгоритм Reciprocal Rank Fusion (RRF) для устойчивого слияния рангов
    без необходимости ручной нормализации разнородных шкал оценок:

        RRF_Score(d) = w_dense / (rrf_k + rank_dense(d)) + w_sparse / (rrf_k + rank_sparse(d))

    где rrf_k (по умолчанию 60) сглаживает влияние первых позиций.
    """

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """Инициализирует гибридный ретривер.

        Args:
            vector_retriever (VectorRetriever): Инициализированный плотный индекс FAISS.
            bm25_retriever (BM25Retriever): Инициализированный лексический индекс BM25.
            dense_weight (float): Весовой коэффициент семантического поиска (по умолчанию 0.5).
            sparse_weight (float): Весовой коэффициент ключевого поиска BM25 (по умолчанию 0.5).
            rrf_k (int): Константа сглаживания рангов RRF (по умолчанию 60).
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k

    def search_with_scores(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        candidate_multiplier: int = 4
    ) -> List[Tuple[str, float]]:
        """Выполняет гибридный поиск и возвращает топ-k пар (doc_id, RRF-оценка).

        Args:
            query (str): Текстовый запрос для BM25.
            query_embedding (np.ndarray): Вектор запроса для FAISS формы (embedding_dim,).
            k (int): Количество возвращаемых результатов. По умолчанию 5.
            candidate_multiplier (int): Множитель пула кандидатов для первичной выборки.

        Returns:
            List[Tuple[str, float]]: Список пар (doc_id, score) в порядке убывания RRF-оценки.
        """
        pool_size = max(k * candidate_multiplier, 20)

        # 1. Поиск в плотном индексе (FAISS)
        q_emb = np.ascontiguousarray(query_embedding).reshape(1, -1)
        _, dense_indices = self.vector_retriever.search(q_emb, k=pool_size)
        dense_doc_ids = [self.vector_retriever.get_doc_id(idx) for idx in dense_indices[0] if idx >= 0]

        # 2. Поиск в лексическом индексе (BM25)
        bm25_results = self.bm25_retriever.search(query, k=pool_size)
        sparse_doc_ids = [doc_id for doc_id, _ in bm25_results]

        # 3. Слияние ранжированных списков по алгоритму Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}

        for rank, doc_id in enumerate(dense_doc_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.dense_weight / (self.rrf_k + rank)

        for rank, doc_id in enumerate(sparse_doc_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + self.sparse_weight / (self.rrf_k + rank)

        # Сортируем документы по убыванию итогового RRF балла
        ranked_docs = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)
        return [(doc_id, rrf_scores[doc_id]) for doc_id in ranked_docs[:k]]

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        candidate_multiplier: int = 4
    ) -> List[str]:
        """Выполняет гибридный поиск по запросу и возвращает топ-k строковых doc_id.

        Args:
            query (str): Текстовый запрос для BM25.
            query_embedding (np.ndarray): Вектор запроса для FAISS формы (embedding_dim,).
            k (int): Количество возвращаемых результатов. По умолчанию 5.
            candidate_multiplier (int): Множитель пула кандидатов для первичной выборки.

        Returns:
            List[str]: Список топ-k идентификаторов чанков doc_id в порядке убывания RRF-оценки.
        """
        pairs = self.search_with_scores(query, query_embedding, k=k, candidate_multiplier=candidate_multiplier)
        return [doc_id for doc_id, _ in pairs]

    def batch_search(
        self,
        queries: List[str],
        query_embeddings: np.ndarray,
        k: int = 5
    ) -> List[List[str]]:
        """Выполняет пакетный гибридный поиск для набора запросов.

        Args:
            queries (List[str]): Список текстовых запросов.
            query_embeddings (np.ndarray): Двумерный массив эмбеддингов запросов формы (N_queries, dim).
            k (int): Количество результатов топ-k.

        Returns:
            List[List[str]]: Список списков найденных идентификаторов doc_id.
        """
        results = []
        for i, q in enumerate(queries):
            res_ids = self.search(query=q, query_embedding=query_embeddings[i], k=k)
            results.append(res_ids)
        return results
