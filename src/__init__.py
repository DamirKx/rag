"""Пакет модулей для подсистемы семантического поиска (Retrieval) в RAG.

Содержит ключевые компоненты конвейера оценки поиска:
- chunking: алгоритмы разбиения текста на смысловые фрагменты (фиксированный и структурный чанкинг НПА);
- embeddings: обёртка над нейросетевыми моделями векторных представлений (Sentence Transformers, E5 и др.);
- retriever: построение векторного индекса и быстрый приближённый поиск ближайших соседей через FAISS;
- eval: вычисление классических метрик качества информационного поиска (MRR, HitRate@k, Recall@k).
"""

from src.chunking import get_fixed_size_chunks, get_structural_chunks
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from src.eval import calculate_mrr, calculate_hit_rate_at_k, calculate_recall_at_k

__all__ = [
    "get_fixed_size_chunks",
    "get_structural_chunks",
    "EmbeddingModel",
    "VectorRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "calculate_mrr",
    "calculate_hit_rate_at_k",
    "calculate_recall_at_k",
]
