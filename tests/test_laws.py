"""Тесты разбора PDF законов на статьи (без чтения настоящих PDF)."""

import pytest

from src.corpus import build_chunk_corpus
from src.laws import display_title, join_lines, parse_act

import pandas as pd

WIDE = "x" * 80  # задаёт типичную ширину строки в join_lines


def test_page_numbers_removed_and_wrapped_lines_joined():
    pages = [f"{WIDE}\nНаниматель обязан своевременно вносить плату за пользование\n1 / 2",
             f"жилым помещением и коммунальные услуги.\nСледующий абзац.\n{WIDE}\n2 / 2"]
    result = join_lines(pages)
    assert "Наниматель обязан своевременно вносить плату за пользование жилым помещением и коммунальные услуги." in " ".join(result)
    assert not any("/ 2" in p for p in result)
    assert any(p.startswith("Следующий абзац.") for p in result)


def test_numbered_points_start_new_paragraphs():
    long_line = "Работник имеет право на расторжение трудового договора в порядке, предусмотренном"
    result = join_lines([f"{WIDE}.\n{long_line}\nнастоящим Кодексом;\n1. Первый пункт\n2) подпункт\nа) буква"])
    assert f"{long_line} настоящим Кодексом;" in result
    assert result[-3:] == ["1. Первый пункт", "2) подпункт", "а) буква"]


def test_hyphen_at_line_end_keeps_compound_word():
    result = join_lines([f"{WIDE}\nОпека устанавливается над детьми-\nсиротами, оставшимися без попечения родителей и другими"])
    assert "детьми-сиротами" in result[-1]


@pytest.mark.parametrize("raw,kind,expected", [
    ("Трудовой кодекс Республики Казахстан", "Кодекс", "Трудовой кодекс РК"),
    ("СОЦИАЛЬНЫЙ КОДЕКС РЕСПУБЛИКИ КАЗАХСТАН", "Кодекс", "Социальный кодекс РК"),
    ("О браке (супружестве) и семье", "Кодекс", "Кодекс РК «О браке (супружестве) и семье»"),
    ("О жилищных отношениях", "Закон", "Закон РК «О жилищных отношениях»"),
    ("Конституция Республики Казахстан", "Конституция", "Конституция РК"),
    ("Гражданский кодекс Республики Казахстан (Общая часть)", "Кодекс", "Гражданский кодекс РК (Общая часть)"),
])
def test_display_title(raw, kind, expected):
    assert display_title(raw, kind) == expected


ACT = [
    "О защите прав потребителей\n"
    "Закон Республики Казахстан от 4 мая 2010 года № 274-IV.\n"
    "Вниманию пользователей!\n"
    "Для удобства пользования РЦПИ создано Оглавление\n"
    "Сноска. Заголовок изменён Законом РК от 01.01.2020 № 1.\n"
    f"Настоящий Закон определяет правовые основы защиты прав потребителей. {'Текст преамбулы. ' * 16}\n"
    "Глава 1. ОБЩИЕ ПОЛОЖЕНИЯ\n"
    "Статья 1. Основные понятия\n"
    "1. Потребитель — физическое лицо.\n"
    "Сноска. Статья 1 с изменениями, внесенными Законом РК от 02.02.2021 № 2.\n"
    "Статья 2. Цель закона\n"
    "Цель закона — защита потребителя. Статья 12 настоящего Закона применяется особо.\n"
    "Глава 2. ПРАВА ПОТРЕБИТЕЛЯ\n"
    "Статья 3\n"
    "1. Потребитель вправе вернуть товар.\n1 / 1"
]


def test_parse_act_splits_articles_and_drops_service_text():
    act = parse_act(ACT)
    assert act["title"] == "Закон РК «О защите прав потребителей»" and act["kind"] == "Закон"
    preamble, first, second, third = act["fragments"]
    assert preamble.startswith("Настоящий Закон") and "Вниманию" not in preamble and "ОБЩИЕ" not in preamble
    assert first.startswith("Статья 1. Основные понятия") and "Сноска" not in first
    assert second.startswith("Статья 2.") and "ПРАВА ПОТРЕБИТЕЛЯ" not in second  # заголовок следующей главы отрезан
    assert third.startswith("Статья 3\n")  # заголовок без точки, как в Конституции


def test_reference_at_line_start_does_not_split_article():
    pages = [f"Трудовой кодекс Республики Казахстан\nКодекс Республики Казахстан от 23 ноября 2015 года № 414-V.\n"
             f"Статья 1. Понятия\n1. Работник вправе обратиться.\nСтатья 12 настоящего Кодекса применяется особо.\nСтатья 2. Цель\nТекст."]
    assert len(parse_act(pages)["fragments"]) == 2


HEAD = "Трудовой кодекс Республики Казахстан\nКодекс Республики Казахстан от 23 ноября 2015 года № 414-V.\n"


def test_heading_glued_to_article_is_split():
    act = parse_act([HEAD + "Статья 1. Понятия\n1. Первая норма.\nГлава 4. Сделки Статья 2. Понятие сделки\n1. Вторая норма."])
    assert [f.split("\n")[0] for f in act["fragments"]] == ["Статья 1. Понятия", "Статья 2. Понятие сделки"]
    assert "Глава" not in act["fragments"][0]


def test_excluded_article_and_footnote_variants_are_dropped():
    act = parse_act([HEAD + "Статья 1. Чек\nСноска. Статья 1 исключена Законом РК от 11.07.1997 N 154.\n"
                            "Статья 2. Вексель\nСноска . Статья 2 исключена Законом.\n"
                            "Статья 3. Акция\n1. Акция - ценная бумага.\nСноска Статья 3 - в редакции Закона."])
    assert act["fragments"] == ["Статья 3. Акция\n1. Акция - ценная бумага."]


def test_text_after_next_chapter_heading_is_cut_but_reference_stays():
    act = parse_act([HEAD + "Статья 1. Понятия\n1. Норма.\nГлава 5 настоящего Кодекса применяется особо.\n"
                            "Глава 2. Договор\n(примечание к главе)\nСтатья 2. Цель\n1. Текст."])
    assert act["fragments"][0] == "Статья 1. Понятия\n1. Норма.\nГлава 5 настоящего Кодекса применяется особо."
    assert len(act["fragments"]) == 2


def test_chunks_of_article_fragment_know_their_article():
    text = "Статья 7. Отпуск\n" + "Работник имеет право на ежегодный оплачиваемый отпуск. " * 40
    df = pd.DataFrame({"cleaned_context": [text], "source_id": [""], "source_title": ["Трудовой кодекс РК"]})
    chunks = build_chunk_corpus(df, {})
    assert len(chunks) > 1
    assert all(c["articles"] == "ст. 7" and c["law"] == "Трудовой кодекс РК" for c in chunks)
