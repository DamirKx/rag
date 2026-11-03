"""Модуль векторного индексирования и семантического поиска на базе FAISS.

Использует библиотеку Facebook AI Similarity Search (FAISS) для быстрого
поиска k ближайших соседей (k-NN) в пространстве высокой размерности.
"""

from collections import Counter
from typing import List, Tuple, Dict, Any, Optional
import re
import faiss
import numpy as np
import scipy.sparse as sp

from src.text_norm import get_normalizer


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
    """Лексический ретривер Okapi BM25 на разреженных матрицах (scipy.sparse).

    Специализируется на точном поиске ключевых слов:
    - Номера законов, статей и пунктов (например, '№ 221-VII', 'статья 122', 'пункт 8-1');
    - Даты и числовые нормативы ('05.04.2023', '311');
    - Специальные аббревиатуры и термины ('МРП', 'ЗҚАИ').

    Оценка документа d по запросу q:

        score(d, q) = Σ_{t ∈ q} idf(t) · tf(t, d) · (k1 + 1) / (tf(t, d) + k1 · (1 − b + b · |d| / avgdl))

    Вес каждой пары «документ — слово» не зависит от запроса, поэтому при индексации один раз
    строится разреженная матрица весов W (слова × документы), а оценки пачки запросов — это одно
    произведение Q · W, где Q — разреженная матрица «запросы × слова» (сколько раз слово в запросе).
    Это в десятки раз быстрее, чем перебор документов в цикле Python.

    Параметры:
    - k1 — насыщение по частоте: при малом k1 важен сам факт вхождения слова, а не число повторов;
    - b — нормировка по длине документа (0 — нет, 1 — полная);
    - normalizer — приведение словоформ к основе (см. src/text_norm.py): 'none', 'prefixN', 'snowball', 'lemma';
    - idf — 'okapi' (как в библиотеке rank_bm25: у слов, которые есть больше чем в половине документов,
      idf заменяется на epsilon · средний idf) или 'lucene' (log(1 + (N − df + 0.5) / (df + 0.5)), всегда > 0).

    По умолчанию (k1=1.5, b=0.75, 'none', 'okapi') оценки совпадают с rank_bm25.BM25Okapi,
    которым проект пользовался раньше, — старые результаты воспроизводятся.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        normalizer: str = "none",
        idf: str = "okapi",
        epsilon: float = 0.25,
    ):
        if idf not in ("okapi", "lucene"):
            raise ValueError(f"idf должен быть 'okapi' или 'lucene', получено {idf!r}")
        self.k1 = k1
        self.b = b
        self.normalizer_name = normalizer
        self.normalize = get_normalizer(normalizer)
        self.idf_variant = idf
        self.epsilon = epsilon

        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        self.vocab: Dict[str, int] = {}
        self._tf: Optional[sp.csr_matrix] = None   # документы × слова, число вхождений
        self._weights: Optional[sp.csr_matrix] = None   # слова × документы, вес BM25

    @property
    def params(self) -> Dict[str, Any]:
        """Параметры индекса (для отчётов)."""
        return {"k1": self.k1, "b": self.b, "normalizer": self.normalizer_name, "idf": self.idf_variant}

    def tokenize(self, text: str) -> List[str]:
        """Токены текста после нормализации словоформ — одинаково для документов и запросов."""
        return self.normalize(tokenize_legal_text(text))

    def add_documents(self, documents: List[str], doc_ids: List[str]) -> None:
        """Индексирует текстовые документы (повторный вызов заменяет индекс).

        Args:
            documents (List[str]): Список исходных текстов чанков.
            doc_ids (List[str]): Список уникальных строковых идентификаторов чанков.
        """
        assert len(documents) == len(doc_ids), (
            f"Несовпадение размерностей: документов={len(documents)}, идентификаторов={len(doc_ids)}"
        )
        self.documents = list(documents)
        self.doc_ids = list(doc_ids)

        vocab: Dict[str, int] = {}
        rows, cols, counts = [], [], []
        for doc_idx, doc in enumerate(self.documents):
            for token, count in Counter(self.tokenize(doc)).items():
                rows.append(doc_idx)
                cols.append(vocab.setdefault(token, len(vocab)))
                counts.append(count)
        self.vocab = vocab
        self._tf = sp.csr_matrix(
            (np.asarray(counts, dtype=np.float64), (rows, cols)),
            shape=(len(self.documents), len(vocab)),
        )
        self._compute_weights()

    def set_params(self, k1: Optional[float] = None, b: Optional[float] = None, idf: Optional[str] = None) -> None:
        """Меняет k1 / b / idf и пересчитывает веса без повторной токенизации корпуса (для подбора на val)."""
        if idf is not None and idf not in ("okapi", "lucene"):
            raise ValueError(f"idf должен быть 'okapi' или 'lucene', получено {idf!r}")
        self.k1 = self.k1 if k1 is None else k1
        self.b = self.b if b is None else b
        self.idf_variant = self.idf_variant if idf is None else idf
        if self._tf is not None:
            self._compute_weights()

    def _idf(self, df: np.ndarray, n_docs: int) -> np.ndarray:
        if self.idf_variant == "lucene":
            return np.log1p((n_docs - df + 0.5) / (df + 0.5))
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        # Как в rank_bm25: отрицательный idf (слово в > половине документов) заменяется на epsilon · средний idf
        idf[idf < 0] = self.epsilon * idf.mean()
        return idf

    def _compute_weights(self) -> None:
        tf = self._tf
        n_docs, n_terms = tf.shape
        if n_docs == 0 or n_terms == 0:
            self._weights = sp.csr_matrix((n_terms, n_docs))
            return
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean() or 1.0
        df = np.bincount(tf.indices, minlength=n_terms)
        idf = self._idf(df, n_docs)

        coo = tf.tocoo()
        norm = self.k1 * (1 - self.b + self.b * doc_len[coo.row] / avgdl)
        values = idf[coo.col] * coo.data * (self.k1 + 1) / (coo.data + norm)
        self._weights = sp.csr_matrix((values, (coo.col, coo.row)), shape=(n_terms, n_docs))

    def _query_matrix(self, queries: List[str]) -> sp.csr_matrix:
        """Запросы × слова: сколько раз слово словаря встречается в запросе (как в rank_bm25)."""
        rows, cols, counts = [], [], []
        for q_idx, query in enumerate(queries):
            for token, count in Counter(self.tokenize(query)).items():
                term = self.vocab.get(token)
                if term is not None:
                    rows.append(q_idx)
                    cols.append(term)
                    counts.append(count)
        return sp.csr_matrix(
            (np.asarray(counts, dtype=np.float64), (rows, cols)), shape=(len(queries), len(self.vocab))
        )

    def get_scores(self, query: str) -> np.ndarray:
        """Оценки BM25 запроса по всем документам индекса (массив длины N_docs)."""
        if self._weights is None:
            return np.zeros(0)
        return (self._query_matrix([query]) @ self._weights).toarray().ravel()

    def batch_search_with_scores(
        self, queries: List[str], k: int = 5, batch_size: int = 256
    ) -> List[List[Tuple[str, float]]]:
        """Пакетный поиск: для каждого запроса — топ-k пар (doc_id, score) с положительной оценкой.

        Запросы обрабатываются пачками по batch_size, чтобы плотная матрица оценок
        (пачка × N_docs) занимала немного памяти. При равных оценках выше документ с меньшим номером.
        """
        if self._weights is None or not self.documents:
            return [[] for _ in queries]
        k = min(k, len(self.documents))
        results: List[List[Tuple[str, float]]] = []
        for start in range(0, len(queries), batch_size):
            scores = (self._query_matrix(queries[start:start + batch_size]) @ self._weights).toarray()
            if k < scores.shape[1]:
                top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            else:
                top = np.tile(np.arange(scores.shape[1]), (scores.shape[0], 1))
            for row, cand in zip(scores, top):
                order = cand[np.lexsort((cand, -row[cand]))]
                results.append([(self.doc_ids[i], float(row[i])) for i in order if row[i] > 0])
        return results

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Осуществляет поиск k наиболее релевантных документов по текстовому запросу.

        Args:
            query (str): Текстовый поисковый запрос.
            k (int): Количество возвращаемых результатов. По умолчанию 5.

        Returns:
            List[Tuple[str, float]]: Список пар (doc_id, score), отсортированных по убыванию релевантности.
        """
        return self.batch_search_with_scores([query], k=k)[0]

    def batch_search(self, queries: List[str], k: int = 5) -> List[List[str]]:
        """Пакетный поиск топ-k doc_id для списка текстовых запросов."""
        return [[doc_id for doc_id, _ in pairs] for pairs in self.batch_search_with_scores(queries, k=k)]


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
