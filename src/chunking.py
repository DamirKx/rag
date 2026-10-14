"""Модуль стратегий разбиения текста (чанкинга) для RAG-системы.

Содержит реализации двух подходов к фрагментации документов:
1. Фиксированный чанкинг (get_fixed_size_chunks) — разбиение текста на блоки
   заданного размера с перекрытием (sliding window). Используется как базовый (baseline) метод.
2. Структурный чанкинг (get_structural_chunks) — специализированный алгоритм для
   нормативно-правовых актов (НПА), разбивающий документ по статьям, пунктам и подпунктам
   с сохранением контекста закона (Context Injection / Header Prepending).
"""

import re
from typing import List, Dict, Any


def get_fixed_size_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Разбивает текст на перекрывающиеся фрагменты (чанки) фиксированного размера в символах.

    Принцип работы (Sliding Window):
    Окно размером `chunk_size` перемещается по тексту с шагом `chunk_size - chunk_overlap`.
    Перекрытие (`chunk_overlap`) необходимо для того, чтобы важная семантическая
    информация и связность предложений не терялись на границах разреза текста.

    Args:
        text (str): Исходный текстовый документ.
        chunk_size (int): Максимальный размер одного чанка в символах (по умолчанию 500).
        chunk_overlap (int): Размер перекрытия соседних чанков в символах (по умолчанию 50).

    Returns:
        List[str]: Список текстовых фрагментов (чанков).

    Raises:
        ValueError: Если chunk_size <= 0 или chunk_overlap некорректен (отрицательный/больше размера чанка).
    """
    # Проверка корректности переданных числовых параметров
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть строго положительным числом")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap должен быть неотрицательным и строго меньше chunk_size")

    chunks = []
    start = 0
    text_len = len(text)

    # Итеративное скольжение окна по всей длине текста
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        # Добавляем только непустые фрагменты
        if chunk:
            chunks.append(chunk)

        # Смещаем указатель начала следующего чанка с учетом размера перекрытия
        start += chunk_size - chunk_overlap

    return chunks


def get_structural_chunks(text: str) -> List[Dict[str, Any]]:
    """Разбивает нормативно-правовой акт на смысловые структурные единицы.

    Особенности алгоритма для юридических документов:
    - Классический фиксированный чанкинг часто разрезает правовые нормы посередине
      санкции или диспозиции статьи, что приводит к галлюцинациям или потере релевантности.
    - Данная функция иерархически разбирает структуру НПА:
        1. Выделяет разделы по ключевому маркеру «Статья N».
        2. Для объемных статей (например, «Статья 1» о внесении изменений в другие законы)
           проводит вторичное разбиение по отдельным законам и подпунктам «1. В Закон...», «1) ...».
        3. Применяет механизм внедрения заголовка (Header Prepending): к каждому подпункту
           добавляется преамбула статьи и название закона, что обеспечивает сохранность
           контекста при векторизации эмбеддинг-моделью.

    Args:
        text (str): Полный текст нормативно-правового документа.

    Returns:
        List[Dict[str, Any]]: Список структурных элементов, где каждый элемент — словарь:
            - 'id' (str): Уникальный семантический идентификатор чанка (например, 'art1_p2_sub1').
            - 'title' (str): Человекочитаемый заголовок или структурная привязка фрагмента.
            - 'text' (str): Обогащенный контекстом текст фрагмента для индексации.
    """
    # 1. Первичный уровень: разделение документа на статьи с помощью регулярного выражения.
    # Используем positive lookahead (?=...), чтобы не стирать само слово «Статья N».
    article_pattern = re.compile(r'(?=\bСтатья\s+\d+[\.\:])', re.IGNORECASE)
    article_splits = [p.strip() for p in article_pattern.split(text) if p.strip()]

    structural_chunks = []

    for art_idx, art_text in enumerate(article_splits, start=1):
        # Извлекаем строки фрагмента, первая непустая строка обычно содержит название статьи
        lines = [line.strip() for line in art_text.splitlines() if line.strip()]
        if not lines:
            continue
        first_line = lines[0]

        # 2. Вторичный уровень: специальная обработка вводных статей с изменениями
        # (в законодательстве «Статья 1» обычно содержит массив правок в разные законы)
        if "статья 1" in first_line.lower():
            # Разделяем по нумерованным пунктам внесения изменений: "\n 1. В Закон...", "\n 2. В Закон..."
            point_pattern = re.compile(r'(?=\n\s*\d+\.\s+В\s+Закон)', re.IGNORECASE)
            points = point_pattern.split(art_text)

            # Первый элемент (header) содержит вводную шапку: "Статья 1. Внести изменения..."
            header = points[0].strip()

            # Обрабатываем каждый блок изменений в конкретный закон
            for pt_idx, point_text in enumerate(points[1:], start=1):
                pt_text = point_text.strip()

                # Извлекаем краткое наименование изменяемого закона (до первого двоеточия)
                match = re.search(r'^\d+\.\s+В\s+Закон[^\:]+\:', pt_text)
                pt_title = match.group(0) if match else f"Статья 1, пункт {pt_idx}"

                # 3. Третичный уровень: разделение по подпунктам «1)», «2)», «3)»
                sub_pattern = re.compile(r'(?=\n\s*\d+\)\s+)', re.IGNORECASE)
                sub_points = sub_pattern.split(pt_text)

                if len(sub_points) > 1:
                    # Подпункт имеет вводный текст («пункт 1 изложить в следующей редакции:»)
                    sub_header = sub_points[0].strip()
                    for sub_idx, sub_text in enumerate(sub_points[1:], start=1):
                        sub_clean = sub_text.strip()
                        sub_first = sub_clean.splitlines()[0] if sub_clean else ""

                        # Формируем чанк с внедрением контекста (заголовок статьи + закон + подпункт)
                        structural_chunks.append({
                            "id": f"art1_p{pt_idx}_sub{sub_idx}",
                            "title": f"Статья 1, п.{pt_idx}: {sub_first[:60]}",
                            "text": f"{header}\n{sub_header}\n{sub_clean}"
                        })
                else:
                    # Если деления на мелкие подпункты нет, сохраняем пункт закона целиком
                    structural_chunks.append({
                        "id": f"art1_p{pt_idx}",
                        "title": f"Статья 1, пункт {pt_idx} ({pt_title})",
                        "text": f"{header}\n{pt_text}"
                    })
        else:
            # Другие самостоятельные статьи (например, «Статья 2. Порядок введения в действие...»)
            structural_chunks.append({
                "id": f"art{art_idx}",
                "title": first_line[:80],
                "text": art_text
            })

    return structural_chunks
