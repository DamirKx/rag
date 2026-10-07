from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Инициализация модели. 
        Популярные двуязычные (RU/EN) модели:
        - 'intfloat/multilingual-e5-base' (требует префикса 'query: ' и 'passage: ')
        - 'BAAI/bge-m3' (поддерживает многоязычность и длинный контекст)
        - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (легкая, быстрая)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        
        # Для E5 моделей часто нужен префикс
        self.is_e5 = "e5" in model_name.lower()

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        if self.is_e5:
            queries = [f"query: {q}" for q in queries]
        embeddings = self.model.encode(queries, normalize_embeddings=True)
        return embeddings

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        if self.is_e5:
            documents = [f"passage: {d}" for d in documents]
        embeddings = self.model.encode(documents, normalize_embeddings=True, show_progress_bar=True)
        return embeddings
