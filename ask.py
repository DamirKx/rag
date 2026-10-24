"""Интерактивный консольный поиск по законодательству Республики Казахстан.

Позволяет в реальном времени задавать любые юридические вопросы на русском или казахском языках
и мгновенно получать релевантные статьи нормативно-правовых актов (НПА).

Поддерживает три режима поиска:
1. 'dense' — семантический векторный поиск на базе FAISS (нейросеть E5);
2. 'bm25' — лексический поиск по ключевым словам и номерам статей;
3. 'hybrid' — гибридное ранжирование Reciprocal Rank Fusion (RRF) Dense + BM25.
"""

import os
import sys
import re
import time
import argparse
from typing import List, Tuple
import pandas as pd

# Обеспечиваем корректный вывод UTF-8 в консоли Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from prepare_adilet_data import clean_adilet_context
from run_adilet_eval import get_or_compute_embeddings, DEFAULT_MODEL

SEARCH_MODES = ("dense", "bm25", "hybrid")


def load_legal_corpus(cache_parquet: str, lang: str) -> pd.DataFrame:
    """Загружает и очищает корпус законов."""
    if not os.path.exists(cache_parquet):
        raise FileNotFoundError(
            f"Файл {cache_parquet} не найден! Сначала запустите prepare_adilet_data.py"
        )
    df = pd.read_parquet(cache_parquet)
    sub_df = df[df["language"] == lang].copy().reset_index(drop=True)
    sub_df["cleaned_context"] = sub_df["context"].apply(clean_adilet_context)
    return sub_df


def format_preview(text: str, max_chars: int = 350) -> str:
    """Форматирует аккуратное превью текста статьи."""
    clean = re.sub(r'\s+', ' ', text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(' ', 1)[0] + "..."


class LegalSearchEngine:
    """Поисковый движок по корпусу НПА с тремя режимами: dense, bm25, hybrid.

    BM25-индекс строится лениво — при первом обращении к режиму 'bm25' или 'hybrid'.
    """

    def __init__(self, docs: List[str], doc_ids: List[str], embedder: EmbeddingModel, cache_dir: str, cache_key: str):
        self.embedder = embedder
        self.docs_by_id = dict(zip(doc_ids, docs))
        self._docs = docs
        self._doc_ids = doc_ids

        emb_matrix = get_or_compute_embeddings(
            embedder=embedder,
            texts=docs,
            cache_dir=cache_dir,
            cache_key=cache_key,
            use_cache=True
        )
        self.dense_retriever = VectorRetriever(embedding_dim=emb_matrix.shape[1])
        self.dense_retriever.add_documents(emb_matrix, docs, doc_ids)

        self._bm25_retriever = None
        self._hybrid_retriever = None

    @property
    def bm25_retriever(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever()
            self._bm25_retriever.add_documents(self._docs, self._doc_ids)
        return self._bm25_retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                vector_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
                dense_weight=0.5,
                sparse_weight=0.5,
                rrf_k=60
            )
        return self._hybrid_retriever

    def search(self, query: str, mode: str, top_k: int) -> List[Tuple[str, str, float]]:
        """Выполняет поиск и возвращает список троек (doc_id, текст, оценка)."""
        if mode == "hybrid":
            q_emb = self.embedder.encode_queries([query])[0]
            pairs = self.hybrid_retriever.search_with_scores(query, q_emb, k=top_k)
        elif mode == "bm25":
            pairs = self.bm25_retriever.search(query, k=top_k)
        elif mode == "dense":
            q_emb = self.embedder.encode_queries([query])
            scores, indices = self.dense_retriever.search(q_emb, k=top_k)
            pairs = [
                (self.dense_retriever.get_doc_id(idx), float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
        else:
            raise ValueError(f"Неизвестный режим поиска: {mode}")

        return [(doc_id, self.docs_by_id[doc_id], score) for doc_id, score in pairs]


def interactive_cli(
    lang: str = "rus",
    model_name: str = DEFAULT_MODEL,
    search_mode: str = "hybrid",
    top_k: int = 3,
    cache_dir: str = "data/adilet/cache",
    cache_parquet: str = "data/raw/adilet_legal_qa_kz.parquet"
):
    print("=" * 80)
    print("  ИНТЕРАКТИВНЫЙ ПОИСК ПО БАЗЕ ЗАКОНОДАТЕЛЬСТВА РК (Adilet Legal QA)")
    print("=" * 80)
    print(f" Язык документов: {lang.upper()}")
    print(f" Модель поиска:   {model_name}")
    print(f" Режим поиска:    {search_mode.upper()}")
    print(f" Количество результатов: {top_k}")
    print("-" * 80)

    # 1. Загрузка корпуса
    print("\n[1/3] Загрузка статей законодательства...")
    sub_df = load_legal_corpus(cache_parquet, lang)

    docs = []
    doc_ids = []
    for idx, row in sub_df.iterrows():
        doc_id = f"ctx_{idx:04d}"
        header = f"Закон: {row['source_title']}\n"
        docs.append(header + row["cleaned_context"])
        doc_ids.append(doc_id)

    print(f"  Загружено {len(docs)} статей нормативно-правовых актов.")

    # 2. Инициализация модели и загрузка кэша эмбеддингов
    print("\n[2/3] Инициализация поискового индекса...")
    embedder = EmbeddingModel(model_name=model_name, device="cpu")
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)
    engine = LegalSearchEngine(
        docs=docs,
        doc_ids=doc_ids,
        embedder=embedder,
        cache_dir=cache_dir,
        cache_key=f"{safe_model}_{lang}_structural"
    )

    print("\n[3/3] База знаний готова к работе!")
    print("=" * 80)
    print(" Введите ваш вопрос (например: 'С какого числа вступил в силу земельный кодекс?')")
    print(" Для переключения режима введите ':mode dense', ':mode bm25' или ':mode hybrid'")
    print(" Для выхода из программы введите 'exit' или 'q'")
    print("=" * 80)

    while True:
        try:
            prompt_str = f"\n[{search_mode.upper()}] Введите вопрос > "
            query = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход из программы.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q", "выход"):
            print("Завершение сеанса поиска. До свидания!")
            break

        # Команды смены режима
        if query.startswith(":mode"):
            parts = query.split()
            if len(parts) > 1 and parts[1].lower() in SEARCH_MODES:
                search_mode = parts[1].lower()
                print(f"Режим переключен на: {search_mode.upper()}")
            else:
                print("Неизвестный режим. Доступные: dense, bm25, hybrid")
            continue

        # Выполнение поиска
        t0 = time.time()
        results = engine.search(query, mode=search_mode, top_k=top_k)
        elapsed_ms = (time.time() - t0) * 1000.0

        if not results:
            print("\nНичего не найдено. Попробуйте переформулировать вопрос или сменить режим поиска.")
            continue

        # Вывод результатов
        print(f"\n🔎 Найдено {len(results)} релевантных статей за {elapsed_ms:.1f} мс:")
        print("-" * 80)

        for rank, (doc_id, text, score) in enumerate(results, 1):
            # Первая строка документа — заголовок «Закон: ...»
            source_line, _, body_text = text.partition("\n")
            preview = format_preview(body_text, max_chars=400)

            score_str = f"{score:.2f}" if search_mode == "bm25" else f"{score:.4f}"
            print(f" [{rank}] {source_line} | ID: {doc_id} | Релевантность: {score_str}")
            print(f"     \"{preview}\"")
            print()


def main():
    parser = argparse.ArgumentParser(description="Интерактивный консольный поиск по НПА РК.")
    parser.add_argument("--lang", choices=["rus", "kaz"], default="rus", help="Язык ('rus' или 'kaz').")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Модель эмбеддингов.")
    parser.add_argument("--mode", choices=list(SEARCH_MODES), default="hybrid", help="Режим поиска.")
    parser.add_argument("--top-k", type=int, default=3, help="Количество возвращаемых статей.")
    parser.add_argument("--cache-dir", type=str, default="data/adilet/cache", help="Папка с кэшем векторов.")
    args = parser.parse_args()

    interactive_cli(
        lang=args.lang,
        model_name=args.model,
        search_mode=args.mode,
        top_k=args.top_k,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
