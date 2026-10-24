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


# Заголовок статьи НПА: «Статья 12.» / «Статья 5-1.» (рус.) или «12-бап.» / «5-1-бап.» (каз.).
# Заголовок должен начинаться с новой строки: цитаты вида «"Статья 5-1. ...» внутри
# текста поправок начинаются с кавычки и не считаются началом новой статьи.
ARTICLE_HEADING = re.compile(
    r'^[ \t]*(?:Статья\s+(?P<rus>\d+(?:-\d+)?)|(?P<kaz>\d+(?:-\d+)?)-бап)\.',
    re.IGNORECASE | re.MULTILINE,
)
# Пункт статьи о поправках: «1. В Закон ...:» / «2. В Кодекс ...:»
AMENDMENT_POINT = re.compile(r'(?=\n\s*\d+\.\s+В\s+(?:Закон|Кодекс))', re.IGNORECASE)
# Подпункт: «1) ...»
SUB_POINT = re.compile(r'(?=\n\s*\d+\)\s+)')


def get_structural_chunks(text: str) -> List[Dict[str, Any]]:
    """Разбивает нормативно-правовой акт на смысловые структурные единицы.

    Особенности алгоритма для юридических документов:
    - Классический фиксированный чанкинг часто разрезает правовые нормы посередине
      санкции или диспозиции статьи, что приводит к галлюцинациям или потере релевантности.
    - Данная функция иерархически разбирает структуру НПА:
        1. Выделяет статьи по заголовкам «Статья N.» (русский) и «N-бап.» (казахский),
           стоящим в начале строки. Текст до первой статьи сохраняется как 'preamble'.
        2. Статьи о внесении изменений (содержащие пункты «1. В Закон...», «2. В Кодекс...»)
           дополнительно разбиваются по изменяемым законам и подпунктам «1) ...».
           Остальные статьи сохраняются целиком.
        3. Применяет механизм внедрения заголовка (Header Prepending): к каждому подпункту
           добавляется преамбула статьи и название закона, что обеспечивает сохранность
           контекста при векторизации эмбеддинг-моделью.

    Args:
        text (str): Полный текст нормативно-правового документа.

    Returns:
        List[Dict[str, Any]]: Список структурных элементов, где каждый элемент — словарь:
            - 'id' (str): Уникальный семантический идентификатор чанка на основе номера статьи
              (например, 'art2', 'art1_p2_sub1', 'art5-1').
            - 'title' (str): Человекочитаемый заголовок или структурная привязка фрагмента.
            - 'text' (str): Обогащенный контекстом текст фрагмента для индексации.
    """
    structural_chunks: List[Dict[str, Any]] = []
    headings = list(ARTICLE_HEADING.finditer(text))

    # Текст до первой статьи (название и реквизиты закона)
    preamble = text[:headings[0].start()].strip() if headings else text.strip()
    if preamble:
        structural_chunks.append({
            "id": "preamble",
            "title": preamble.splitlines()[0][:80],
            "text": preamble,
        })

    seen_ids: Dict[str, int] = {}

    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        art_text = text[heading.start():end].strip()
        art_num = heading.group("rus") or heading.group("kaz")

        # Номер статьи может повториться (например, в нескольких документах одного файла)
        art_id = f"art{art_num}"
        seen_ids[art_id] = seen_ids.get(art_id, 0) + 1
        if seen_ids[art_id] > 1:
            art_id = f"{art_id}_{seen_ids[art_id]}"

        first_line = art_text.splitlines()[0]
        points = AMENDMENT_POINT.split(art_text)

        if len(points) == 1:
            # Обычная статья без пунктов о поправках — сохраняем целиком
            structural_chunks.append({
                "id": art_id,
                "title": first_line[:80],
                "text": art_text,
            })
            continue

        # Первый элемент (header) содержит вводную шапку: "Статья 1. Внести изменения..."
        header = points[0].strip()

        # Обрабатываем каждый блок изменений в конкретный закон
        for pt_idx, point_text in enumerate(points[1:], start=1):
            pt_text = point_text.strip()

            # Извлекаем краткое наименование изменяемого закона (до первого двоеточия)
            match = re.search(r'^\d+\.\s+В\s+(?:Закон|Кодекс)[^\:]+\:', pt_text, re.IGNORECASE)
            pt_title = match.group(0) if match else f"{first_line[:20]}, пункт {pt_idx}"

            # Третичный уровень: разделение по подпунктам «1)», «2)», «3)»
            sub_points = SUB_POINT.split(pt_text)

            if len(sub_points) > 1:
                # Подпункт имеет вводный текст («пункт 1 изложить в следующей редакции:»)
                sub_header = sub_points[0].strip()
                for sub_idx, sub_text in enumerate(sub_points[1:], start=1):
                    sub_clean = sub_text.strip()
                    sub_first = sub_clean.splitlines()[0] if sub_clean else ""

                    # Формируем чанк с внедрением контекста (заголовок статьи + закон + подпункт)
                    structural_chunks.append({
                        "id": f"{art_id}_p{pt_idx}_sub{sub_idx}",
                        "title": f"{art_id}, п.{pt_idx}: {sub_first[:60]}",
                        "text": f"{header}\n{sub_header}\n{sub_clean}",
                    })
            else:
                # Если деления на мелкие подпункты нет, сохраняем пункт закона целиком
                structural_chunks.append({
                    "id": f"{art_id}_p{pt_idx}",
                    "title": f"{art_id}, пункт {pt_idx} ({pt_title})",
                    "text": f"{header}\n{pt_text}",
                })

    return structural_chunks
