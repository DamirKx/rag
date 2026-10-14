"""Модуль генерации векторных представлений (эмбеддингов) для текстов.

Преобразует текстовые запросы пользователей и фрагменты документов (чанки)
в плотные вещественные векторы фиксированной размерности (Dense Embeddings)
с использованием предобученных нейросетевых моделей на базе архитектуры Transformer.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Обертка над нейросетевыми моделями Sentence Transformers.

    Обеспечивает:
    - Инкапсуляцию логики инференса на CPU или GPU;
    - L2-нормализацию векторов (для быстрого косинусного сходства через скалярное произведение);
    - Автоматическую обработку префиксов для асимметричных моделей поиска (семейство E5);
    - Безопасное приведение векторов к типу np.float32 для совместимости с FAISS.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """Инициализирует модель эмбеддингов и определяет её тип.

        Рекомендуемые модели для русскоязычных нормативных документов:
        - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2':
            Легковесная (около 470 МБ), быстрая, симметричная модель. Идеальна для CPU.
        - 'intfloat/multilingual-e5-base' / 'intfloat/multilingual-e5-small':
            Асимметричная модель. Обучена с разделением запросов ('query: ')
            и пассажей документов ('passage: ').
        - 'BAAI/bge-m3':
            Тяжелая мультиязычная модель с поддержкой длинного контекста (до 8192 токенов).

        Args:
            model_name (str): Идентификатор модели на HuggingFace Hub или путь к локальным весам.
            device (str): Устройство для инференса ('cpu', 'cuda', 'mps'). По умолчанию 'cpu'.
        """
        self.model_name = model_name
        self.device = device

        # Загрузка предобученной модели через SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

        # Проверка: является ли модель представителем семейства E5.
        # Модели E5 обучались контрастным обучением с жестким требованием к префиксам:
        # запросы должны предваряться 'query: ', а индексируемые документы — 'passage: '.
        self.is_e5 = "e5" in model_name.lower()

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Векторизует список входящих пользовательских запросов.

        Args:
            queries (List[str]): Список текстовых поисковых запросов.

        Returns:
            np.ndarray: Двумерный массив эмбеддингов формы (N_queries, Embedding_dim)
                        в формате float32 с единичной L2-нормой.
        """
        # Если модель относится к семейству E5, добавляем обязательный префикс запроса
        if self.is_e5:
            processed_queries = [f"query: {q}" for q in queries]
        else:
            processed_queries = queries

        # Вычисление эмбеддингов:
        # normalize_embeddings=True гарантирует ||v||_2 = 1 (длина каждого вектора равна 1)
        embeddings = self.model.encode(
            processed_queries,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # FAISS строго требует непрерывный массив типа float32 в памяти
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Векторизует коллекцию документов / чанков для наполнения векторного индекса.

        Args:
            documents (List[str]): Список текстовых фрагментов (чанков).

        Returns:
            np.ndarray: Двумерный массив эмбеддингов формы (N_docs, Embedding_dim)
                        в формате float32 с единичной L2-нормой.
        """
        # Для E5 моделей документы должны маркироваться префиксом 'passage: '
        if self.is_e5:
            processed_docs = [f"passage: {d}" for d in documents]
        else:
            processed_docs = documents

        # Векторизация с отображением индикатора выполнения (progress bar)
        embeddings = self.model.encode(
            processed_docs,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Приведение к непрерывному массиву float32 для последующей передачи в индекс FAISS
        return np.ascontiguousarray(embeddings, dtype=np.float32)
