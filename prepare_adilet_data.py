"""Модуль подготовки и конвертации датасета Adilet Legal QA (KZ/RU).

Загружает датасет 'olzhasAl/adilet-legal-qa-kz' с Hugging Face Hub (или из локального кэша),
производит качественную очистку текстов нормативно-правовых актов от артефактов
веб-интерфейса системы «Әділет» (adilet.zan.kz) и формирует структурированные
файлы корпуса и размеченных QA-запросов для оценки RAG-поиска (Retrieval Evaluation).
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd

# Обеспечиваем корректный вывод UTF-8 в консоли Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


HF_PARQUET_URL = "hf://datasets/olzhasAl/adilet-legal-qa-kz/data/train-00000-of-00001.parquet"


def clean_adilet_context(text: str) -> str:
    """Удаляет артефакты веб-навигации интерфейса системы «Әділет».

    На первых страницах нормативных актов парсер захватывает элементы меню:
    «Мобильдік нұсқаға қайта оралу», «Полная версия сайта», кнопки PDF/DOCX и т.д.
    Данная функция срезает этот шум вплоть до начала содержательной части документа.

    Args:
        text (str): Исходный сырой текст контекста из датасета.

    Returns:
        str: Очищенный текст фрагмента нормативного акта.
    """
    if not text:
        return ""

    lines = text.splitlines()
    start_idx = 0

    # Маркеры веб-меню сайта adilet.zan.kz
    noise_markers = {
        "мобільдік нұсқаға қайта оралу",
        "вернуться к мобильной версии",
        "толықэкранды режим",
        "көру қабілеті төмен",
        "полная версия сайта",
        "құжаттан сілтемелер",
        "құжатқа сілтемелер",
        "эбб нұсқасы",
        "жарнама орналастыру",
        "демонстрациялық режим"
    }

    # Ищем, присутствует ли шум в первых 50 строках
    has_noise = any(
        any(m in line.lower() for m in noise_markers)
        for line in lines[:50]
    )

    if has_noise:
        # Прокручиваем строки, пока не найдем начало значимого текста:
        # «Ескерту», «Статья», «Раздел», «Глава», «бап», «1. », «Закон», «Кодекс»
        content_starters = [
            r'^\s*статья\s+\d+',
            r'^\s*\d+[\-\s]*бап',
            r'^\s*ескерту[\.\:]',
            r'^\s*примечание[\.\:]',
            r'^\s*бөлім\s+\d+',
            r'^\s*раздел\s+\d+',
            r'^\s*тарау\s+\d+',
            r'^\s*глава\s+\d+',
            r'^\s*1\.\s+[А-ЯӘІҢҒҮҰҚӨҺA-Z]',
            r'^\s*қолданушылар\s+назарына',
            r'^\s*закон\s+республики\s+казахстан',
            r'^\s*қазақстан\s+республикасының\s+заңы'
        ]
        start_pattern = re.compile("|".join(content_starters), re.IGNORECASE)

        for idx, line in enumerate(lines[:80]):
            line_str = line.strip()
            if start_pattern.search(line_str):
                start_idx = idx
                break

    cleaned_lines = lines[start_idx:]
    cleaned_text = "\n".join(cleaned_lines)

    # 2. Обрезка технического подвала (футера) веб-страницы сайта «Әділет»
    footer_patterns = [
        r'\n\s*Егер\s+Сіз\s+беттен\s+қате\s+тапсаңыз.*$',
        r'\n\s*Если\s+Вы\s+заметили\s+ошибку.*$',
        r'\n\s*Базаның\s+жай-күйі\s*\n.*$',
        r'\n\s*Состояние\s+базы\s*\n.*$',
        r'\n\s*Құқықтық\s+ақпараттық\s+қызметі.*$',
        r'\n\s*Служба\s+правовой\s+информации.*$',
        r'\n\s*Қолдау\s+қызметі\s*\n.*$',
        r'\n\s*Служба\s+поддержки\s*\n.*$',
        r'\n.*?support@zqai\.kz.*$',
        r'\n.*?©\s*\d{4}\.?\s*Қазақстан\s+Республикасы\s+Әділет\s+министрлігі.*$',
        r'\n.*?©\s*\d{4}\.?\s*Министерство\s+юстиции\s+Республики\s+Казахстан.*$'
    ]
    for fp in footer_patterns:
        cleaned_text = re.sub(fp, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)

    # Сжатие множественных пустых строк до двойного переноса
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


def to_text_list(value: Any) -> List[str]:
    """Приводит поле prompts/responses к списку строк.

    В parquet это обычно numpy-массив, но одиночная строка не должна
    разбиваться на символы, а отсутствующее значение даёт пустой список.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def download_or_load_dataset(cache_path: str) -> pd.DataFrame:
    """Загружает датасет с Hugging Face Hub либо из локального файла кэша.

    Args:
        cache_path (str): Путь для сохранения локальной копии parquet.

    Returns:
        pd.DataFrame: Загруженный датафрейм.
    """
    cache_file = Path(cache_path)
    if cache_file.exists():
        print(f"📦 Загрузка датасета из локального кэша: {cache_file.resolve()}")
        df = pd.read_parquet(cache_file)
    else:
        print(f"🌐 Загрузка датасета с Hugging Face Hub: {HF_PARQUET_URL}...")
        df = pd.read_parquet(HF_PARQUET_URL)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Сохранение локального кэша: {cache_file.resolve()}")
        df.to_parquet(cache_file, index=False)

    print(f" Всего строк загружено: {len(df)}")
    return df


def process_language_partition(
    df: pd.DataFrame,
    language: str,
    output_dir: Path,
    test_size: int = 300,
    random_seed: int = 42
) -> Dict[str, Any]:
    """Обрабатывает срез данных по одному языку (rus или kaz).

    Выполняет:
    1. Фильтрацию по языку и очистку контекстов.
    2. Формирование объединенного текстового корпуса законов для индексации.
    3. Разворачивание списков prompts/responses в плоский QA-список.
    4. Стратифицированное выделение контрольной тестовой выборки (test split).

    Args:
        df (pd.DataFrame): Исходный датафрейм.
        language (str): Код языка ('rus' или 'kaz').
        output_dir (Path): Директория для сохранения результатов.
        test_size (int): Желаемый размер тестовой выборки вопросов.
        random_seed (int): Сид псевдослучайного генератора для воспроизводимости.

    Returns:
        Dict[str, Any]: Сводная статистика обработки.
    """
    sub_df = df[df["language"] == language].copy().reset_index(drop=True)
    if sub_df.empty:
        print(f"⚠️ Нет данных для языка '{language}'!")
        return {}

    print(f"\n Обработка среза '{language}': {len(sub_df)} контекстов...")

    # 1. Очистка контекстов
    sub_df["cleaned_context"] = sub_df["context"].apply(clean_adilet_context)

    # 2. Формирование единого текстового корпуса для индексации
    corpus_file = output_dir / f"corpus_{language}.txt"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for idx, row in sub_df.iterrows():
            f.write(f"=== ДОКУМЕНТ {idx + 1}: {row['source_title']} (ID: {row['source_id']}) ===\n\n")
            f.write(row["cleaned_context"])
            f.write("\n\n" + "=" * 80 + "\n\n")

    print(f" Текстовый корпус сохранен в: {corpus_file}")

    # 3. Разворачивание списков вопросов (prompts) и ответов (responses)
    flat_qa_records: List[Dict[str, Any]] = []
    global_q_idx = 1

    for row_idx, row in sub_df.iterrows():
        prompts = row["prompts"]
        responses = row["responses"]

        # Защита от неоднородных типов (список, numpy-массив, строка или None)
        p_list = to_text_list(prompts)
        r_list = to_text_list(responses)

        for q_sub_idx, q_text in enumerate(p_list):
            q_text = str(q_text).strip()
            r_text = str(r_list[q_sub_idx]).strip() if q_sub_idx < len(r_list) else ""

            if not q_text:
                continue

            # Извлекаем краткую ключевую фразу из ответа для сопоставления
            # (первое предложение либо до 80 символов)
            rel_phrase = r_text.split(".")[0].strip() if r_text else ""
            if not rel_phrase or len(rel_phrase) < 10:
                rel_phrase = r_text[:80].strip()

            flat_qa_records.append({
                "id": f"{language}_q{global_q_idx:05d}",
                "query": q_text,
                "target_section": row["source_title"],
                "structural_id": f"ctx_{row_idx:04d}",
                "context_id": f"ctx_{row_idx:04d}",
                "source_id": row["source_id"],
                "source_title": row["source_title"],
                "doc_type": row["doc_type"],
                "relevant_text": rel_phrase,
                "full_response": r_text
            })
            global_q_idx += 1

    total_questions = len(flat_qa_records)
    print(f" Всего сформировано вопросов: {total_questions}")

    # 4. Сохранение полного набора вопросов
    full_qa_file = output_dir / f"qa_{language}_full.json"
    with open(full_qa_file, "w", encoding="utf-8") as f:
        json.dump(flat_qa_records, f, ensure_ascii=False, indent=2)
    print(f" Полный QA датасет: {full_qa_file} ({total_questions} вопросов)")

    # 5. Формирование сбалансированной контрольной тестовой выборки (test split)
    # Берем вопросы из разных документов пропорционально
    qa_df = pd.DataFrame(flat_qa_records)
    actual_test_size = min(test_size, total_questions)

    # Стратифицированная или равномерная случайная выборка
    test_df = qa_df.sample(n=actual_test_size, random_state=random_seed).sort_values("id")
    test_records = test_df.to_dict(orient="records")

    test_qa_file = output_dir / f"qa_{language}_test.json"
    with open(test_qa_file, "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=2)
    print(f" Тестовая выборка для оценки: {test_qa_file} ({actual_test_size} вопросов)")

    return {
        "language": language,
        "total_contexts": len(sub_df),
        "total_questions": total_questions,
        "test_questions": actual_test_size,
        "corpus_file": str(corpus_file),
        "test_qa_file": str(test_qa_file),
        "full_qa_file": str(full_qa_file)
    }


def main():
    parser = argparse.ArgumentParser(description="Подготовка датасета Adilet Legal QA для RAG.")
    parser.add_argument(
        "--lang",
        choices=["rus", "kaz", "all"],
        default="rus",
        help="Языковой срез: 'rus' (русский), 'kaz' (казахский) или 'all' (оба). По умолчанию 'rus'."
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="Количество вопросов в тестовой выборке для быстрой оценки. По умолчанию 200."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/adilet",
        help="Путь к директории для сохранения результатов. По умолчанию 'data/adilet'."
    )
    parser.add_argument(
        "--cache-parquet",
        type=str,
        default="data/raw/adilet_legal_qa_kz.parquet",
        help="Путь к файлу локального кэша Parquet."
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем датасет (с кэшированием)
    df = download_or_load_dataset(args.cache_parquet)

    langs_to_process = ["rus", "kaz"] if args.lang == "all" else [args.lang]
    all_stats = {}

    for lang in langs_to_process:
        stats = process_language_partition(
            df=df,
            language=lang,
            output_dir=out_dir,
            test_size=args.test_size
        )
        all_stats[lang] = stats

    # Сохраняем сводные метаданные
    metadata_file = out_dir / "dataset_summary.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(" Подготовка данных успешно завершена!")
    print(f"Все результаты сохранены в: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
