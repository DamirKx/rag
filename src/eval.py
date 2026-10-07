from typing import List, Dict

def calculate_mrr(relevant_docs: List[List[str]], retrieved_docs: List[List[str]]) -> float:
    """
    Вычисляет Mean Reciprocal Rank (MRR).
    
    :param relevant_docs: Список истинных релевантных ID документов для каждого запроса.
    :param retrieved_docs: Список ID предсказанных документов (по порядку релевантности).
    """
    mrr = 0.0
    for rel_d, ret_d in zip(relevant_docs, retrieved_docs):
        rr = 0.0
        for rank, doc_id in enumerate(ret_d, 1):
            if doc_id in rel_d:
                rr = 1.0 / rank
                break # Учитываем только первое правильное попадание для MRR
        mrr += rr
    return mrr / len(relevant_docs) if relevant_docs else 0.0

def calculate_recall_at_k(relevant_docs: List[List[str]], retrieved_docs: List[List[str]], k: int) -> float:
    """
    Вычисляет Recall@k (Полноту на k).
    
    :param relevant_docs: Список истинных релевантных ID документов для каждого запроса.
    :param retrieved_docs: Список ID предсказанных документов (длины >= k).
    :param k: Ограничение топ-k результатов.
    """
    recall = 0.0
    for rel_d, ret_d in zip(relevant_docs, retrieved_docs):
        ret_k = set(ret_d[:k])
        rel_set = set(rel_d)
        
        # Находим пересечение
        hits = len(ret_k.intersection(rel_set))
        # Считаем полноту для конкретного запроса
        recall += hits / len(rel_set) if len(rel_set) > 0 else 0.0
        
    return recall / len(relevant_docs) if relevant_docs else 0.0
