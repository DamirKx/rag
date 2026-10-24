"""Демонстрационный скрипт оценки Retrieval-компонента RAG на модельных данных.

Иллюстрирует базовый сквозной цикл семантического поиска:
1. Создание корпуса синтетических документов (правила внутреннего распорядка вуза);
2. Разбиение исходных текстов на перекрывающиеся чанки (chunking);
3. Векторизация чанков с использованием трансформерной модели (embeddings);
4. Построение плоского индекса быстрого векторного поиска (FAISS retriever);
5. Выполнение тестовых поисковых запросов и расчет метрик ранжирования (MRR, HitRate@k, Recall@k).
"""

import sys

# Обеспечиваем корректный вывод символов UTF-8 (русского текста) в консоли Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.chunking import get_fixed_size_chunks
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever
from src.eval import calculate_mrr, calculate_recall_at_k, calculate_hit_rate_at_k


def main() -> None:
    """Запускает демонстрационный конвейер оценки качества семантического поиска."""
    print("=" * 75)
    print("  ДЕМОНСТРАЦИОННЫЙ ПАЙПЛАЙН ОЦЕНКИ RETRIEVAL-ЧАСТИ RAG")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # Шаг 1: Подготовка синтетических документов (база знаний)
    # -------------------------------------------------------------------------
    print("\n[1/5] Подготовка синтетических нормативных документов...")
    # Документы представляют собой типичные выдержки из университетских регламентов
    doc1 = "Студенты обязаны посещать не менее 80% занятий. В случае пропусков требуется справка от врача."
    doc2 = "Students must attend at least 80% of classes. A medical certificate is required for absences."
    doc3 = "Курсовая работа сдается до 15 мая. Задержка сдачи влечет за собой снижение итоговой оценки."

    # -------------------------------------------------------------------------
    # Шаг 2: Разбиение документов на чанки фиксированного размера
    # -------------------------------------------------------------------------
    print("\n[2/5] Разбиение текста на перекрывающиеся фрагменты (чанги)...")
    # Используем окно в 60 символов и перекрытие в 15 символов для демонстрации
    chunks1 = get_fixed_size_chunks(doc1, chunk_size=60, chunk_overlap=15)
    chunks2 = get_fixed_size_chunks(doc2, chunk_size=60, chunk_overlap=15)
    chunks3 = get_fixed_size_chunks(doc3, chunk_size=60, chunk_overlap=15)

    all_chunks = chunks1 + chunks2 + chunks3

    # Формируем уникальные идентификаторы для каждого полученного чанка
    doc_ids = (
        [f"doc1_{i}" for i in range(len(chunks1))] +
        [f"doc2_{i}" for i in range(len(chunks2))] +
        [f"doc3_{i}" for i in range(len(chunks3))]
    )
    print(f"Всего создано {len(all_chunks)} чанков из {3} исходных документов.")

    # -------------------------------------------------------------------------
    # Шаг 3: Векторизация чанков с помощью нейросетевой модели
    # -------------------------------------------------------------------------
    print("\n[3/5] Загрузка модели эмбеддингов и векторизация чанков...")
    # Используем мультиязычную компактную модель поиска E5-small (быстро работает на CPU)
    model_name = "intfloat/multilingual-e5-small"
    embedder = EmbeddingModel(model_name=model_name, device="cpu")

    print("Генерация векторных представлений базы знаний...")
    doc_embeddings = embedder.encode_documents(all_chunks)

    # -------------------------------------------------------------------------
    # Шаг 4: Построение векторного индекса FAISS
    # -------------------------------------------------------------------------
    print("\n[4/5] Инициализация векторного индекса FAISS (IndexFlatIP)...")
    embedding_dim = doc_embeddings.shape[1]
    retriever = VectorRetriever(embedding_dim=embedding_dim)

    # Добавляем полученные эмбеддинги, тексты и идентификаторы в поисковый индекс
    retriever.add_documents(doc_embeddings, all_chunks, doc_ids)
    print(f"Индекс успешно построен. Число векторов в индексе: {embedding_dim}D x {len(all_chunks)}.")

    # -------------------------------------------------------------------------
    # Шаг 5: Оценка качества поиска по контрольным вопросам
    # -------------------------------------------------------------------------
    print("\n[5/5] Выполнение запросов и расчет метрик информационного поиска...")
    queries = [
        "What happens if I submit my coursework late?",  # Запрос на английском -> релевантен doc3
        "Сколько занятий можно пропустить?"              # Запрос на русском -> релевантен doc1 или doc2
    ]

    # Эталонные списки истинно релевантных идентификаторов (Ground Truth)
    ground_truth_relevant = [
        [doc_id for doc_id in doc_ids if "doc3" in doc_id],
        [doc_id for doc_id in doc_ids if "doc1" in doc_id or "doc2" in doc_id]
    ]

    # Векторизуем контрольные запросы
    query_embeddings = embedder.encode_queries(queries)

    # Ищем топ-2 наиболее семантически близких чанка
    top_k = 2
    _, indices = retriever.search(query_embeddings, k=top_k)

    # Собираем список предсказанных документов для каждого запроса
    predicted_docs = []
    for q_idx in range(len(queries)):
        pred = [retriever.get_doc_id(idx) for idx in indices[q_idx]]
        predicted_docs.append(pred)
        print(f"\nЗапрос [{q_idx + 1}]: \"{queries[q_idx]}\"")
        print(f"  Найденные чанки (топ-{top_k}): {pred}")
        print(f"  Ожидаемые чанки: {ground_truth_relevant[q_idx]}")

    # Расчет численных метрик качества
    mrr = calculate_mrr(ground_truth_relevant, predicted_docs)
    hit_rate = calculate_hit_rate_at_k(ground_truth_relevant, predicted_docs, k=top_k)
    recall = calculate_recall_at_k(ground_truth_relevant, predicted_docs, k=top_k)

    # Итоговый вывод результатов
    print("\n" + "=" * 50)
    print(f"ИТОГОВЫЕ МЕТРИКИ ПОИСКА (k={top_k}):")
    print("=" * 50)
    print(f"  MRR:         {mrr:.4f}  (качество ранжирования первой статьи)")
    print(f"  HitRate@{top_k}:  {hit_rate:.4f}  (доля запросов с хотя бы 1 верным чанком)")
    print(f"  Recall@{top_k}:   {recall:.4f}  (полнота покрытия релевантных чанков)")
    print("=" * 50)


if __name__ == "__main__":
    main()
