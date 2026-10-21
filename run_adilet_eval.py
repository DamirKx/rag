"""Пайплайн оценки семантического поиска (Retrieval) на датасете Adilet Legal QA.

Выполняет объективное тестирование качества информационного поиска (Information Retrieval)
на реальных юридических документах Республики Казахстан без использования генеративных моделей (LLM).

Проводит сравнительный эксперимент двух стратегий чанкинга:
1. Фиксированный чанкинг (Fixed-size: окно 500 символов, перекрытие 50 символов);
2. Структурный чанкинг НПА (Article-level: сохранение статей/пунктов как единых смысловых блоков).

Рассчитываемые метрики:
- MRR (Mean Reciprocal Rank);
- HitRate@1 (доля случаев, когда правильный фрагмент находится на 1-м месте);
- HitRate@k (доля случаев, когда правильный фрагмент входит в топ-k);
- Recall@k (полнота поиска в топ-k).
"""

import os
import sys
import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# Обеспечиваем корректное отображение UTF-8 в консоли Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.chunking import get_fixed_size_chunks
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever
from src.eval import calculate_mrr, calculate_hit_rate_at_k, calculate_recall_at_k
from prepare_adilet_data import clean_adilet_context


def normalize_text(text: str) -> str:
    """Нормализует строку для устойчивого текстового сопоставления."""
    return re.sub(r"\s+", " ", text).strip().lower()


def evaluate_pipeline(
    retriever: VectorRetriever,
    embedder: EmbeddingModel,
    qa_data: List[Dict[str, Any]],
    ground_truth: List[List[str]],
    top_k: int = 3,
    verbose: bool = False
) -> Dict[str, float]:
    """Выполняет поиск и расчет метрик IR для заданного ретривера.

    Args:
        retriever (VectorRetriever): Построенный индекс FAISS.
        embedder (EmbeddingModel): Модель векторных представлений.
        qa_data (List[Dict[str, Any]]): Список вопросов.
        ground_truth (List[List[str]]): Список эталонных doc_id для каждого вопроса.
        top_k (int): Глубина выдачи (k).
        verbose (bool): Выводить ли детализацию по каждому вопросу.

    Returns:
        Dict[str, float]: Словарь метрик качества.
    """
    queries = [item["query"] for item in qa_data]

    # Векторизация поисковых запросов
    t0 = time.time()
    query_embeddings = embedder.encode_queries(queries)
    enc_time = time.time() - t0

    # Поиск в FAISS
    t0 = time.time()
    _, indices = retriever.search(query_embeddings, k=top_k)
    search_time = time.time() - t0

    predicted: List[List[str]] = []
    for q_idx in range(len(queries)):
        pred_ids = [retriever.get_doc_id(idx) for idx in indices[q_idx]]
        predicted.append(pred_ids)

    if verbose:
        print("\n" + "=" * 80)
        print(f"ПРИМЕРЫ ВЫДАЧИ ПОИСКА (ТОП-{top_k})")
        print("=" * 80)
        for i in range(min(5, len(qa_data))):
            item = qa_data[i]
            pred_ids = predicted[i]
            gt_ids = ground_truth[i]
            hit = any(p in gt_ids for p in pred_ids)
            status = " УСПЕХ" if hit else " ПРОМАХ"
            print(f"\n[{i + 1}] {status} | {item['source_title']}")
            print(f"Вопрос: {item['query']}")
            print(f"Ожидаемый ID: {gt_ids[:3]}")
            print(f"Найденные ID: {pred_ids}")

    # Расчет метрик
    mrr = calculate_mrr(ground_truth, predicted)
    hit_rate_1 = calculate_hit_rate_at_k(ground_truth, predicted, k=1)
    hit_rate_k = calculate_hit_rate_at_k(ground_truth, predicted, k=top_k)
    recall_k = calculate_recall_at_k(ground_truth, predicted, k=top_k)

    return {
        "MRR": mrr,
        "HitRate@1": hit_rate_1,
        f"HitRate@{top_k}": hit_rate_k,
        f"Recall@{top_k}": recall_k,
        "query_encode_sec": enc_time,
        "search_sec": search_time
    }


def get_or_compute_embeddings(
    embedder: EmbeddingModel,
    texts: List[str],
    cache_dir: str,
    cache_key: str,
    use_cache: bool = True
) -> np.ndarray:
    """Загружает готовые эмбеддинги из бинарного кэша (.npy) или вычисляет их и сохраняет."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.npy")

    if use_cache and os.path.exists(cache_file):
        print(f"⚡ Загрузка векторов из дискового кэша: {cache_file}...")
        t0 = time.time()
        try:
            emb = np.load(cache_file)
            if len(emb) == len(texts):
                print(f"  Успешно загружено {len(emb)} векторов за {time.time() - t0:.2f} сек. Форма: {emb.shape}")
                return emb
            else:
                print(f"  Размер кэша ({len(emb)}) не совпадает с числом текстов ({len(texts)}), пересчитываем...")
        except Exception as e:
            print(f"  Ошибка чтения кэша ({e}), пересчитываем...")

    print(f"Векторизация {len(texts)} текстов через нейросеть...")
    t0 = time.time()
    emb = embedder.encode_documents(texts)
    print(f"Векторизация завершена за {time.time() - t0:.2f} сек. Форма матрицы: {emb.shape}")

    if use_cache:
        try:
            np.save(cache_file, emb)
            size_mb = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"💾 Векторы сохранены в кэш: {cache_file} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"⚠️ Предупреждение: не удалось сохранить кэш: {e}")

    return emb


def run_evaluation(
    lang: str = "rus",
    qa_path: str = "data/adilet/qa_rus_test.json",
    cache_parquet: str = "data/raw/adilet_legal_qa_kz.parquet",
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    top_k: int = 3,
    max_test_queries: int = 100,
    output_report: str = None,
    use_cache: bool = True,
    cache_dir: str = "data/adilet/cache"
) -> Dict[str, Any]:
    """Основной пайплайн сравнительной оценки стратегий чанкинга."""
    print("=" * 80)
    print("      БЕНЧМАРК RETRIEVAL-ЧАСТИ RAG: ДАТАСЕТ ADILET LEGAL QA")
    print("=" * 80)
    print(f" Язык: {lang.upper()}")
    print(f" Модель эмбеддингов: {model_name}")
    print(f" Глубина поиска (top-k): {top_k}")
    print(f" Использование кэша эмбеддингов: {'ДА' if use_cache else 'НЕТ'}")

    # 1. Загрузка исходных данных
    if not os.path.exists(cache_parquet):
        raise FileNotFoundError(f"Файл датасета не найден: {cache_parquet}. Запустите prepare_adilet_data.py")

    print(f"\n Загрузка данных из: {cache_parquet}...")
    df = pd.read_parquet(cache_parquet)
    sub_df = df[df["language"] == lang].copy().reset_index(drop=True)
    sub_df["cleaned_context"] = sub_df["context"].apply(clean_adilet_context)
    print(f"Загружено статей/контекстов: {len(sub_df)}")

    # 2. Загрузка тестового набора вопросов
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    if max_test_queries and len(qa_data) > max_test_queries:
        qa_data = qa_data[:max_test_queries]

    print(f"Количество оцениваемых тестовых вопросов: {len(qa_data)}")

    # 3. Инициализация модели эмбеддингов
    print(f"\n Инициализация модели {model_name}...")
    t0 = time.time()
    embedder = EmbeddingModel(model_name=model_name, device="cpu")
    print(f"Модель загружена за {time.time() - t0:.2f} сек.")

    # Безопасный идентификатор модели для имени файла кэша
    safe_model = re.sub(r'[^a-zA-Z0-9_\-]', '_', model_name)

    # =========================================================================
    # Эксперимент 1: Структурный чанкинг НПА (Article-level / смысловые блоки)
    # =========================================================================
    print("\n" + "-" * 80)
    print("СТРАТЕГИЯ 1: Структурный чанкинг НПА (целые статьи/блоки с сохранением контекста)")
    print("-" * 80)

    struct_chunks = []
    struct_doc_ids = []
    for idx, row in sub_df.iterrows():
        doc_id = f"ctx_{idx:04d}"
        header = f"Закон: {row['source_title']}\n"
        full_chunk = header + row["cleaned_context"]
        struct_chunks.append(full_chunk)
        struct_doc_ids.append(doc_id)

    print(f"Создано структурных чанков: {len(struct_chunks)}")
    struct_cache_key = f"{safe_model}_{lang}_structural"
    struct_emb = get_or_compute_embeddings(
        embedder=embedder,
        texts=struct_chunks,
        cache_dir=cache_dir,
        cache_key=struct_cache_key,
        use_cache=use_cache
    )

    struct_retriever = VectorRetriever(embedding_dim=struct_emb.shape[1])
    struct_retriever.add_documents(struct_emb, struct_chunks, struct_doc_ids)

    # Формируем Ground Truth для структурного чанкинга
    struct_gt = [[item["context_id"]] for item in qa_data]

    print("Оценка структурного поиска...")
    res_struct = evaluate_pipeline(
        retriever=struct_retriever,
        embedder=embedder,
        qa_data=qa_data,
        ground_truth=struct_gt,
        top_k=top_k,
        verbose=True
    )

    # =========================================================================
    # Эксперимент 2: Фиксированный чанкинг (Baseline: окно 500, overlap 50)
    # =========================================================================
    print("\n" + "-" * 80)
    print("СТРАТЕГИЯ 2: Фиксированный чанкинг (Baseline: окно 500 символов, перекрытие 50)")
    print("-" * 80)

    fixed_chunks = []
    fixed_doc_ids = []
    for idx, row in sub_df.iterrows():
        ctx_id = f"ctx_{idx:04d}"
        raw_slices = get_fixed_size_chunks(row["cleaned_context"], chunk_size=500, chunk_overlap=50)
        for s_idx, slice_text in enumerate(raw_slices):
            chunk_id = f"{ctx_id}_fix_{s_idx:03d}"
            fixed_chunks.append(slice_text)
            fixed_doc_ids.append(chunk_id)

    print(f"Создано фиксированных чанков: {len(fixed_chunks)}")
    fixed_cache_key = f"{safe_model}_{lang}_fixed_500_50"
    fixed_emb = get_or_compute_embeddings(
        embedder=embedder,
        texts=fixed_chunks,
        cache_dir=cache_dir,
        cache_key=fixed_cache_key,
        use_cache=use_cache
    )

    fixed_retriever = VectorRetriever(embedding_dim=fixed_emb.shape[1])
    fixed_retriever.add_documents(fixed_emb, fixed_chunks, fixed_doc_ids)

    # Формируем Ground Truth для фиксированного чанкинга
    # Релевантным считается чанк, принадлежащий целевому контексту И содержащий фразу ответа
    fixed_gt = []
    for item in qa_data:
        target_ctx = item["context_id"]
        norm_rel = normalize_text(item.get("relevant_text", ""))
        short_rel = norm_rel[:45] if len(norm_rel) > 45 else norm_rel

        matching_fixed_ids = []
        for fid, ftext in zip(fixed_doc_ids, fixed_chunks):
            if fid.startswith(target_ctx):
                norm_chunk = normalize_text(ftext)
                if (short_rel and short_rel in norm_chunk) or (norm_rel and norm_rel in norm_chunk):
                    matching_fixed_ids.append(fid)

        # Если цитата не совпала из-за границ разреза, берем все чанки целевой статьи
        if not matching_fixed_ids:
            matching_fixed_ids = [fid for fid in fixed_doc_ids if fid.startswith(target_ctx)]

        fixed_gt.append(matching_fixed_ids)

    print("Оценка фиксированного поиска...")
    res_fixed = evaluate_pipeline(
        retriever=fixed_retriever,
        embedder=embedder,
        qa_data=qa_data,
        ground_truth=fixed_gt,
        top_k=top_k,
        verbose=False
    )

    # =========================================================================
    # Итоговый вывод результатов
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"ИТОГОВОЕ СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА НА ADILET QA (k={top_k}, N={len(qa_data)})")
    print("=" * 80)
    header = f"{'Метрика':<15} | {'Фиксированный (500)':<22} | {'Структурный (НПА)':<22} | {'Разница':<12}"
    print(header)
    print("-" * len(header))

    metrics = ["MRR", "HitRate@1", f"HitRate@{top_k}", f"Recall@{top_k}"]
    for m in metrics:
        v_fix = res_fixed[m]
        v_str = res_struct[m]
        diff = v_str - v_fix
        diff_str = f"{diff:+.4f}"
        print(f"{m:<15} | {v_fix:<22.4f} | {v_str:<22.4f} | {diff_str:<12}")

    print("=" * 80)
    print(f"Количество индексируемых чанков: Фиксированный = {len(fixed_chunks)}, Структурный = {len(struct_chunks)}")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": lang,
        "model_name": model_name,
        "top_k": top_k,
        "num_test_queries": len(qa_data),
        "fixed_chunks_count": len(fixed_chunks),
        "structural_chunks_count": len(struct_chunks),
        "results_fixed": res_fixed,
        "results_structural": res_struct
    }

    if output_report:
        out_p = Path(output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Полный отчет о тестировании сохранен в: {out_p.resolve()}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Оценка Retrieval на датасете Adilet Legal QA.")
    parser.add_argument("--lang", choices=["rus", "kaz"], default="rus", help="Язык ('rus' или 'kaz').")
    parser.add_argument("--qa-path", type=str, default=None, help="Путь к файлу с вопросами.")
    parser.add_argument("--test-size", type=int, default=100, help="Количество вопросов для оценки (по умолчанию 100).")
    parser.add_argument("--top-k", type=int, default=3, help="Глубина поисковой выдачи top-k (по умолчанию 3).")
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--save-report", type=str, default="data/adilet/eval_results.json")
    parser.add_argument("--cache-dir", type=str, default="data/adilet/cache", help="Папка для кэширования эмбеддингов.")
    parser.add_argument("--no-cache", action="store_true", help="Принудительно пересчитать эмбеддинги без использования кэша.")
    args = parser.parse_args()

    qa_file = args.qa_path or f"data/adilet/qa_{args.lang}_test.json"

    run_evaluation(
        lang=args.lang,
        qa_path=qa_file,
        model_name=args.model,
        top_k=args.top_k,
        max_test_queries=args.test_size,
        output_report=args.save_report,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
