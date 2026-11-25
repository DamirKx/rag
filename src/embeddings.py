"""Модуль генерации векторных представлений (эмбеддингов) для текстов.

Преобразует текстовые запросы пользователей и фрагменты документов (чанки)
в плотные вещественные векторы фиксированной размерности (Dense Embeddings)
с использованием предобученных нейросетевых моделей на базе архитектуры Transformer.
"""

from typing import List, Tuple
import numpy as np

# Префиксы, с которыми модель обучалась: (к вопросу, к фрагменту документа). Ключ — часть имени модели
# в нижнем регистре, берётся первая подходящая строка. У моделей без строки (bge-m3, deepvk/USER-bge-m3)
# префиксов нет. Значения — из config_sentence_transformers.json моделей (поле prompts).
MODEL_PREFIXES = [
    ("frida", ("search_query: ", "search_document: ")),
    ("qwen3-embedding", ("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", "")),
    ("e5", ("query: ", "passage: ")),  # семейство E5, в том числе дообученная models/e5-small-adilet-rus
]


def model_prefixes(model_name: str) -> Tuple[str, str]:
    """Префиксы вопроса и фрагмента для модели (пустые, если модель обучалась без них)."""
    name = model_name.lower()
    return next((prefixes for key, prefixes in MODEL_PREFIXES if key in name), ("", ""))


class EmbeddingModel:
    """Обертка над нейросетевыми моделями Sentence Transformers.

    Обеспечивает:
    - Инкапсуляцию логики инференса на CPU или GPU;
    - L2-нормализацию векторов (для быстрого косинусного сходства через скалярное произведение);
    - Автоматическую обработку префиксов для асимметричных моделей поиска (семейство E5);
    - Безопасное приведение векторов к типу np.float32 для совместимости с FAISS.
    """

    def __init__(self, model_name: str, device: str = "cpu", fp16: bool = False):
        """Инициализирует модель эмбеддингов и определяет её тип.

        Рекомендуемые модели для русскоязычных нормативных документов:
        - 'intfloat/multilingual-e5-small' (основная) / 'intfloat/multilingual-e5-base':
            Асимметричные модели поиска с лимитом 512 токенов. Обучены с разделением
            запросов ('query: ') и пассажей документов ('passage: ').
        - 'BAAI/bge-m3':
            Тяжелая мультиязычная модель с поддержкой длинного контекста (до 8192 токенов).

        Args:
            model_name (str): Идентификатор модели на HuggingFace Hub или путь к локальным весам.
            device (str): Устройство для инференса ('cpu', 'cuda', 'mps'). По умолчанию 'cpu'.
            fp16 (bool): Половинная точность (только для GPU): большие модели (FRIDA, bge-m3) иначе
                         не помещаются в 4 ГБ видеопамяти вместе с вычислениями.
        """
        self.model_name = model_name
        self.device = device

        # Загрузка предобученной модели через SentenceTransformer. Библиотека импортируется здесь,
        # а не в начале модуля: её импорт занимает ~10 с, а веб-демо должно сразу показать страницу
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        if fp16:
            self.model.half()

        # Префиксы вопроса и документа, с которыми модель обучалась (MODEL_PREFIXES). Например, модели E5
        # обучались контрастным обучением с жестким требованием: запросы предваряются 'query: ', документы — 'passage: '.
        self.query_prefix, self.document_prefix = model_prefixes(model_name)
        self.is_e5 = "e5" in model_name.lower()

    def to(self, device: str) -> "EmbeddingModel":
        """Переносит модель на другое устройство (веб-демо векторизует корпус на GPU, а вопросы — на CPU)."""
        self.model.to(device)
        self.device = device
        return self

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Векторизует список входящих пользовательских запросов.

        Args:
            queries (List[str]): Список текстовых поисковых запросов.

        Returns:
            np.ndarray: Двумерный массив эмбеддингов формы (N_queries, Embedding_dim)
                        в формате float32 с единичной L2-нормой.
        """
        # Префикс запроса, с которым обучалась модель ('query: ' у E5; у bge-m3 префикса нет)
        processed_queries = [f"{self.query_prefix}{q}" for q in queries]

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
        # Префикс документа, с которым обучалась модель ('passage: ' у E5)
        processed_docs = [f"{self.document_prefix}{d}" for d in documents]

        # Векторизация с отображением индикатора выполнения (progress bar)
        embeddings = self.model.encode(
            processed_docs,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Приведение к непрерывному массиву float32 для последующей передачи в индекс FAISS
        return np.ascontiguousarray(embeddings, dtype=np.float32)
