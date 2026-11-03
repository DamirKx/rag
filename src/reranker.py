"""Модуль переранжирования (Reranking) кандидатов поиска нейросетевым cross-encoder.

Двухэтапный поиск: быстрый ретривер (Dense / BM25 / Hybrid) отбирает пул кандидатов,
а cross-encoder пересчитывает релевантность каждой пары «запрос — чанк» и
переупорядочивает пул. В отличие от bi-encoder (E5), который кодирует запрос и документ
независимо, cross-encoder читает их вместе через полное внимание (full attention),
поэтому точнее, но и намного медленнее — применять его ко всему корпусу нельзя.

Документы длиннее лимита модели (фрагменты законов целиком в ask.py) делятся на окна,
оценкой документа служит максимум по окнам (MaxP): иначе модель увидела бы только начало текста.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from sentence_transformers import CrossEncoder

from src.chunking import get_fixed_size_chunks

# Мультиязычный реранкер на базе bge-m3: поддерживает русский и казахский языки
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """Обёртка над `sentence_transformers.CrossEncoder` для переранжирования кандидатов."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
        window_chars: int = 1000,
        window_overlap: int = 100,
        fp16: Optional[bool] = None,
    ):
        """Загружает модель cross-encoder.

        Args:
            model_name (str): Идентификатор модели на HuggingFace Hub или путь к локальным весам.
            device (Optional[str]): Устройство инференса ('cpu', 'cuda'); None — выбор автоматически.
            max_length (int): Максимальная длина пары «запрос + документ» в токенах.
                              Чанк 500 символов занимает ~150 токенов, поэтому 512 хватает с запасом.
            batch_size (int): Размер батча пар при инференсе.
            window_chars (int): Длина окна (символов) для документов длиннее неё.
                                ~1000 символов русского текста укладываются в 512 токенов вместе с запросом.
            window_overlap (int): Перекрытие соседних окон в символах.
            fp16 (Optional[bool]): Инференс в половинной точности; None — включается на GPU.
                                   Примерно вдвое быстрее, на порядок кандидатов практически не влияет.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.window_chars = window_chars
        self.window_overlap = window_overlap
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        if fp16 is None:
            fp16 = device == "cuda" if device is not None else torch.cuda.is_available()
        self.fp16 = fp16
        if fp16:
            self.model.half()

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        """Возвращает оценки релевантности каждого документа запросу (чем выше, тем релевантнее)."""
        if not documents:
            return []
        # Все окна всех документов оцениваются одним батчем, owner[i] — номер документа окна i
        pairs, owner = [], []
        for doc_idx, doc in enumerate(documents):
            windows = [doc] if len(doc) <= self.window_chars else get_fixed_size_chunks(
                doc, chunk_size=self.window_chars, chunk_overlap=self.window_overlap
            )
            for window in windows:
                pairs.append((query, window))
                owner.append(doc_idx)

        window_scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        scores = [float("-inf")] * len(documents)
        for doc_idx, s in zip(owner, window_scores):
            scores[doc_idx] = max(scores[doc_idx], float(s))
        return scores

    def rerank(
        self,
        query: str,
        candidates: Sequence[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Переранжирует кандидатов первого этапа.

        Args:
            query (str): Текстовый запрос.
            candidates (Sequence[Tuple[str, str]]): Пары (doc_id, текст) от ретривера.
            top_k (Optional[int]): Сколько результатов вернуть; None — все.

        Returns:
            List[Tuple[str, float]]: Пары (doc_id, оценка cross-encoder) по убыванию оценки.
                При равных оценках сохраняется порядок первого этапа.
        """
        scores = self.score(query, [text for _, text in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [(candidates[i][0], scores[i]) for i in order[:top_k]]
