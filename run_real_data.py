"""Пайплайн оценки Retrieval-части RAG на реальных данных законодательства.

Использует размеченный экспертный датасет (data/qa_dataset.json) для расчёта
объективных метрик семантического поиска (MRR, HitRate@k, Recall@k)
без использования генеративных языковых моделей (LLM).

Проводит сравнительный эксперимент двух стратегий чанкинга:
1. Фиксированный чанкинг (Baseline: окно 500 символов, перекрытие 50 символов);
2. Структурный чанкинг НПА (разбиение по статьям, пунктам и подпунктам с сохранением контекста).
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any

# Обеспечиваем корректное отображение кириллических символов UTF-8 в консоли Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.chunking import get_fixed_size_chunks, get_structural_chunks
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever
from src.eval import calculate_mrr, calculate_hit_rate_at_k, calculate_recall_at_k


def load_documents(data_dir: str) -> List[str]:
    """Считывает все текстовые файлы (*.txt) из указанной директории.

    Args:
        data_dir (str): Путь к директории с исходными документами.

    Returns:
        List[str]: Список строк, каждая из которых содержит полный текст одного файла.
    """
    texts = []
    # Перебираем все .txt файлы в алфавитном порядке
    for txt_path in sorted(Path(data_dir).glob("*.txt")):
        with open(txt_path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def load_qa_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Загружает контрольный датасет вопросов и эталонных ответов из JSON-файла.

    Структура каждого элемента датасета:
    - 'id': Уникальный номер вопроса.
    - 'query': Текст вопроса на естественном языке.
    - 'target_section': Название или номер целевой статьи НПА.
    - 'relevant_text': Точная цитата / фрагмент нормы права, содержащий ответ.
    - 'structural_id': Структурный идентификатор фрагмента (для структурного чанкинга).

    Args:
        dataset_path (str): Путь к файлу qa_dataset.json.

    Returns:
        List[Dict[str, Any]]: Список словарей с эталонной разметкой.

    Raises:
        FileNotFoundError: Если файл датасета отсутствует по указанному пути.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл эталонного датасета не найден: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """Нормализует строку для устойчивого текстового сопоставления.

    Удаляет повторяющиеся пробельные символы, символы табуляции и переносы строк,
    а также приводит весь текст к нижнему регистру (lowercase).

    Args:
        text (str): Исходный фрагмент текста.

    Returns:
        str: Нормализованная текстовая строка.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def find_ground_truth_chunks(
    chunks: List[str],
    doc_ids: List[str],
    target_phrase: str,
    structural_id: str = None
) -> List[str]:
    """Определяет, какие именно чанки содержат эталонный фрагмент ответа.

    Логика сопоставления:
    1. Ищет вхождение нормализованной целевой фразы (target_phrase) в текст чанка.
    2. Использует укороченный префикс (short_target, первые 45 символов) на случай,
       если длинная цитата была разделена границей скользящего окна чанкинга.
    3. Если передан structural_id (например, 'art1_p2'), сопоставляет его
       с идентификатором чанка doc_id.

    Args:
        chunks (List[str]): Список текстов всех сформированных чанков.
        doc_ids (List[str]): Список идентификаторов всех чанков.
        target_phrase (str): Эталонный фрагмент текста ответа из разметки.
        structural_id (str, optional): Идентификатор структурной единицы НПА.

    Returns:
        List[str]: Список идентификаторов doc_id чанков, признанных релевантными.
    """
    norm_target = normalize_text(target_phrase)
    # Короткий фрагмент для защиты от разрыва длинных предложений на границе чанков
    short_target = norm_target[:45] if len(norm_target) > 45 else norm_target

    relevant = []
    for doc_id, chunk in zip(doc_ids, chunks):
        norm_chunk = normalize_text(chunk)

        # Проверка текстового вхождения полной или усеченной цитаты
        if norm_target in norm_chunk or short_target in norm_chunk:
            relevant.append(doc_id)
        # Проверка структурного идентификатора (актуально для get_structural_chunks)
        elif structural_id and structural_id in doc_id:
            relevant.append(doc_id)

    return relevant


def evaluate_retrieval(
    retriever: VectorRetriever,
    embedder: EmbeddingModel,
    qa_data: List[Dict[str, Any]],
    chunks: List[str],
    doc_ids: List[str],
    top_k: int = 3,
    verbose: bool = True
) -> Dict[str, float]:
    """Выполняет пакетную оценку качества поиска по размеченному датасету.

    Этапы оценки:
    1. Поиск эталонных релевантных чанков (Ground Truth) для каждого вопроса.
    2. Векторизация всех поисковых запросов через модель эмбеддингов.
    3. Поиск k ближайших векторов в индексе FAISS.
    4. Детальный лог попаданий (УСПЕХ / ПРОМАХ) при verbose=True.
    5. Расчет численных метрик: MRR, HitRate@1, HitRate@k, Recall@k.

    Args:
        retriever (VectorRetriever): Инициализированный векторный индекс с чанками.
        embedder (EmbeddingModel): Модель эмбеддингов для векторизации запросов.
        qa_data (List[Dict[str, Any]]): Список вопросов с экспертной разметкой.
        chunks (List[str]): Тексты проиндексированных чанков.
        doc_ids (List[str]): Идентификаторы проиндексированных чанков.
        top_k (int): Глубина поисковой выдачи для оценки. По умолчанию 3.
        verbose (bool): Выводить ли подробную информацию по каждому запросу.

    Returns:
        Dict[str, float]: Словарь со значениями рассчитанных метрик.
    """
    queries = [item["query"] for item in qa_data]
    ground_truth: List[List[str]] = []

    # 1. Формируем эталонный список истинно релевантных чанков для каждого вопроса
    for item in qa_data:
        rel = find_ground_truth_chunks(
            chunks=chunks,
            doc_ids=doc_ids,
            target_phrase=item["relevant_text"],
            structural_id=item.get("structural_id")
        )
        ground_truth.append(rel)

    # 2. Векторизуем список поисковых запросов
    query_embeddings = embedder.encode_queries(queries)

    # 3. Выполняем векторный поиск в FAISS
    _, indices = retriever.search(query_embeddings, k=top_k)

    # 4. Преобразуем числовые индексы FAISS в строковые doc_id
    predicted: List[List[str]] = []
    for q_idx in range(len(queries)):
        pred_ids = [retriever.get_doc_id(idx) for idx in indices[q_idx]]
        predicted.append(pred_ids)

    # 5. Детальная печать результатов каждого вопроса при включенном режиме verbose
    if verbose:
        print("\n" + "=" * 80)
        print(f"ДЕТАЛИЗАЦИЯ ПОИСКА (ТОП-{top_k})")
        print("=" * 80)
        for i, item in enumerate(qa_data, 1):
            pred_ids = predicted[i - 1]
            gt_ids = ground_truth[i - 1]
            # Запрос успешен, если хотя бы один из найденных чанков входит в эталонные
            hit = any(p in gt_ids for p in pred_ids)
            status = " УСПЕХ" if hit else " ПРОМАХ"

            print(f"\n[{i}/{len(qa_data)}] {status}")
            print(f"Вопрос: {item['query']}")
            print(f"Целевая статья: {item['target_section']}")
            print(f"Ожидаемые чанки: {gt_ids}")
            print(f"Найденные чанки (топ-{top_k}): {pred_ids}")

    # 6. Вычисление агрегированных метрик качества поиска
    mrr = calculate_mrr(ground_truth, predicted)
    hit_rate = calculate_hit_rate_at_k(ground_truth, predicted, k=top_k)
    recall = calculate_recall_at_k(ground_truth, predicted, k=top_k)
    hit_rate_1 = calculate_hit_rate_at_k(ground_truth, predicted, k=1)

    return {
        "MRR": mrr,
        "HitRate@1": hit_rate_1,
        f"HitRate@{top_k}": hit_rate,
        f"Recall@{top_k}": recall
    }


def main() -> None:
    """Точка входа: запускает сравнительный эксперимент стратегий чанкинга на реальных данных."""
    print("=" * 80)
    print("   ОЦЕНКА RETRIEVAL-ЧАСТИ RAG НА РЕАЛЬНОМ ЗАКОНОДАТЕЛЬСТВЕ РК")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Загрузка исходных нормативно-правовых документов
    # -------------------------------------------------------------------------
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    documents = load_documents(data_dir)
    if not documents:
        print(f"Ошибка: текстовые документы не найдены в директории {data_dir}. Проверьте data.txt.")
        return

    full_text = "\n\n".join(documents)
    print(f"Загружено документов: {len(documents)}, общий объём текста: {len(full_text):,} символов.")

    # -------------------------------------------------------------------------
    # 2. Загрузка экспертного бенчмарка вопросов и ответов
    # -------------------------------------------------------------------------
    qa_path = os.path.join(data_dir, "qa_dataset.json")
    qa_data = load_qa_dataset(qa_path)
    print(f"Загружен контрольный датасет: {len(qa_data)} вопросов с эталонной разметкой.")

    # -------------------------------------------------------------------------
    # 3. Инициализация нейросетевой модели векторных представлений
    # -------------------------------------------------------------------------
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\nЗагрузка модели эмбеддингов: {model_name}...")
    embedder = EmbeddingModel(model_name=model_name, device="cpu")

    # =========================================================================
    # Эксперимент 1: Фиксированный чанкинг (Baseline: 500 символов, 50 overlap)
    # =========================================================================
    print("\n" + "-" * 80)
    print("СТРАТЕГИЯ 1: Фиксированный размер чанка (chunk_size=500, overlap=50)")
    print("-" * 80)
    fixed_chunks = get_fixed_size_chunks(full_text, chunk_size=500, chunk_overlap=50)
    fixed_doc_ids = [f"fixed_{i}" for i in range(len(fixed_chunks))]
    print(f"Создано чанков фиксированного размера: {len(fixed_chunks)}")

    print("Векторизация фиксированных чанков и построение FAISS индекса...")
    fixed_emb = embedder.encode_documents(fixed_chunks)
    fixed_retriever = VectorRetriever(embedding_dim=fixed_emb.shape[1])
    fixed_retriever.add_documents(fixed_emb, fixed_chunks, fixed_doc_ids)

    # Оценка качества поиска для фиксированного чанкинга
    results_fixed = evaluate_retrieval(
        retriever=fixed_retriever,
        embedder=embedder,
        qa_data=qa_data,
        chunks=fixed_chunks,
        doc_ids=fixed_doc_ids,
        top_k=3,
        verbose=True
    )

    # =========================================================================
    # Эксперимент 2: Структурный чанкинг НПА (по статьям и подпунктам закона)
    # =========================================================================
    print("\n" + "-" * 80)
    print("СТРАТЕГИЯ 2: Структурный чанкинг НПА (по статьям и подпунктам закона)")
    print("-" * 80)
    structural_items = get_structural_chunks(full_text)
    struct_chunks = [item["text"] for item in structural_items]
    struct_doc_ids = [item["id"] for item in structural_items]
    print(f"Создано структурных чанков НПА: {len(struct_chunks)}")

    print("Векторизация структурных чанков и построение FAISS индекса...")
    struct_emb = embedder.encode_documents(struct_chunks)
    struct_retriever = VectorRetriever(embedding_dim=struct_emb.shape[1])
    struct_retriever.add_documents(struct_emb, struct_chunks, struct_doc_ids)

    # Оценка качества поиска для структурного чанкинга (детализацию скрываем, чтобы не дублировать вывод)
    results_struct = evaluate_retrieval(
        retriever=struct_retriever,
        embedder=embedder,
        qa_data=qa_data,
        chunks=struct_chunks,
        doc_ids=struct_doc_ids,
        top_k=3,
        verbose=False
    )

    # =========================================================================
    # Итоговая сводная таблица результатов
    # =========================================================================
    print("\n" + "=" * 80)
    print("ИТОГОВОЕ СРАВНЕНИЕ СТРАТЕГИЙ ЧАНКИНГА (k=3)")
    print("=" * 80)
    header = f"{'Метрика':<15} | {'Фиксированный (500)':<22} | {'Структурный (НПА)':<22}"
    print(header)
    print("-" * len(header))
    print(f"{'MRR':<15} | {results_fixed['MRR']:<22.4f} | {results_struct['MRR']:<22.4f}")
    print(f"{'HitRate@1':<15} | {results_fixed['HitRate@1']:<22.4f} | {results_struct['HitRate@1']:<22.4f}")
    print(f"{'HitRate@3':<15} | {results_fixed['HitRate@3']:<22.4f} | {results_struct['HitRate@3']:<22.4f}")
    print(f"{'Recall@3':<15} | {results_fixed['Recall@3']:<22.4f} | {results_struct['Recall@3']:<22.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
