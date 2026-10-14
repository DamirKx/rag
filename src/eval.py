"""Модуль вычисления метрик качества информационного поиска (Information Retrieval).

Используется для объективной оценки этапа Retrieval в RAG-системах
без привлечения ресурсоемких генеративных языковых моделей (LLM).

Основные рассчитываемые метрики:
1. MRR (Mean Reciprocal Rank) — качество ранжирования первого релевантного ответа.
2. HitRate@k — вероятность попадания хотя бы одного правильного документа в топ-k.
3. Recall@k — доля всех найденных релевантных фрагментов от общего их числа.
"""

from typing import List


def calculate_mrr(relevant_docs: List[List[str]], retrieved_docs: List[List[str]]) -> float:
    """Вычисляет средний взаимный ранг (Mean Reciprocal Rank, MRR).

    Математическая формула:
        MRR = (1 / |Q|) * sum_{i=1}^{|Q|} (1 / rank_i)
    где:
        - |Q| — общее количество тестовых запросов;
        - rank_i — позиция первого релевантного документа в списке результатов для i-го запроса
          (нумерация с 1). Если релевантный документ не найден в выдаче, вклад равен 0.

    Пример шкалы оценки:
        - Релевантный документ на 1-м месте -> вклад = 1 / 1 = 1.0
        - Релевантный документ на 2-м месте -> вклад = 1 / 2 = 0.5
        - Релевантный документ на 3-м месте -> вклад = 1 / 3 ≈ 0.333
        - Релевантный документ не найден в топ-k -> вклад = 0.0

    Args:
        relevant_docs (List[List[str]]): Список эталонных (Ground Truth) ID документов
                                         для каждого поискового запроса.
        retrieved_docs (List[List[str]]): Список предсказанных ретривером ID документов
                                          в порядке убывания их релевантности.

    Returns:
        float: Значение метрики MRR в диапазоне [0.0, 1.0]. Чем ближе к 1.0, тем лучше ранжирование.
    """
    if not relevant_docs:
        return 0.0

    mrr_sum = 0.0

    # Проходим по каждой паре: эталонные документы и найденные ретривером
    for rel_d, ret_d in zip(relevant_docs, retrieved_docs):
        rr = 0.0
        # Преобразуем список истинных ID в множество для мгновенного поиска за O(1)
        rel_set = set(rel_d)

        # Перебираем найденные документы и фиксируем первое совпадение
        for rank, doc_id in enumerate(ret_d, start=1):
            if doc_id in rel_set:
                rr = 1.0 / rank
                break  # В MRR учитывается строго только ПЕРВОЕ релевантное попадание

        mrr_sum += rr

    # Усредняем взаимный ранг по всем запросам выборки
    return mrr_sum / len(relevant_docs)


def calculate_hit_rate_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> float:
    """Вычисляет метрику Hit Rate@k (также известную как Accuracy@k или Success@k).

    Определяет долю запросов, для которых ХОТЯ БЫ ОДИН истинно релевантный документ
    был успешно обнаружен в пределах первых k позиций поисковой выдачи.

    Практическое значение для RAG:
        Это критически важная метрика: если HitRate@k = 0.8, это означает, что в 80% случаев
        генеративная модель получит правильный правовой контекст и сможет дать верный ответ,
        а в 20% случаев будет либо галлюцинировать, либо признавать отсутствие информации.

    Args:
        relevant_docs (List[List[str]]): Список истинных релевантных ID документов для каждого запроса.
        retrieved_docs (List[List[str]]): Список ID предсказанных документов в порядке ранжирования.
        k (int): Глубина поисковой выдачи (топ-k документов).

    Returns:
        float: Доля успешных запросов в диапазоне [0.0, 1.0].
    """
    if not relevant_docs:
        return 0.0

    hits = 0

    for rel_d, ret_d in zip(relevant_docs, retrieved_docs):
        # Ограничиваемся первыми k кандидатами
        top_k_retrieved = set(ret_d[:k])
        rel_set = set(rel_d)

        # Проверяем, есть ли пересечение между топ-k выдачей и эталоном
        if any(doc in rel_set for doc in top_k_retrieved):
            hits += 1

    return hits / len(relevant_docs)


def calculate_recall_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> float:
    """Вычисляет классическую метрику полноты поиска (Recall@k).

    Математическая формула:
        Recall@k = (1 / |Q|) * sum_{i=1}^{|Q|} (|Retrieved_{i, k} ∩ Relevant_i| / |Relevant_i|)
    где:
        - Retrieved_{i, k} — множество первых k документов, найденных ретривером;
        - Relevant_i — полное множество всех документов, признанных релевантными экспертами.

    Различие между HitRate@k и Recall@k:
        - HitRate@k равен 1.0, даже если найден всего 1 чанк из 5 необходимых.
        - Recall@k оценивает долю покрытия: если для полного ответа требовалось 2 статьи,
          а в топ-k попала только 1, Recall@k для данного запроса составит 0.5 (50%).

    Args:
        relevant_docs (List[List[str]]): Список истинных релевантных ID документов.
        retrieved_docs (List[List[str]]): Список предсказанных документов.
        k (int): Глубина топа (число позиций для оценки).

    Returns:
        float: Средняя полнота по всей тестовой выборке в диапазоне [0.0, 1.0].
    """
    if not relevant_docs:
        return 0.0

    total_recall = 0.0

    for rel_d, ret_d in zip(relevant_docs, retrieved_docs):
        top_k_retrieved = set(ret_d[:k])
        rel_set = set(rel_d)

        # Если для данного запроса нет релевантных документов в разметке, пропускаем
        if not rel_set:
            continue

        # Количество найденных релевантных фрагментов в пределах топ-k
        relevant_hits = len(top_k_retrieved.intersection(rel_set))

        # Доля найденных от общего числа требуемых фрагментов
        total_recall += relevant_hits / len(rel_set)

    return total_recall / len(relevant_docs)
