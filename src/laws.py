"""Корпус актуальных законов для граждан: разбор PDF с adilet.zan.kz на статьи.

Отдельный от датасета Adilet Legal QA корпус для веб-демо. Размеченных вопросов к нему нет, поэтому
метрики отчёта считаются только на датасете. Модели и параметры поиска используются те же.

PDF содержит текст с жёсткими переносами строк и номерами страниц («12 / 179»). Разбор:
1. убирает номера страниц и склеивает разорванные строки в абзацы;
2. убирает сноски об изменениях («Сноска. Статья 12 с изменениями...»): для вопросов граждан они не
   нужны, а в датасете похожие сноски были одной из причин ошибок поиска (REPORT.md, раздел 7.10);
3. делит акт на статьи: одна статья — один фрагмент, поэтому чанк всегда знает свою статью, а текст
   вокруг чанка (expand_hits) берётся только из неё.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

LAWS_DIR = "data/laws"
LAWS_PARQUET = "data/laws/laws_rus.parquet"

# Адреса на adilet.zan.kz, подтверждённые по самому сайту; у остальных актов источник показывается без ссылки
SOURCE_IDS = {
    "Гражданский кодекс РК (особенная часть)": "K990000409_",
    "Трудовой кодекс РК": "K1500000414",
    "Кодекс РК «О браке (супружестве) и семье»": "K1100000518",
    "Социальный кодекс РК": "K2300000224",
}

PAGE_NUMBER = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
ARTICLE_START = re.compile(r"^Статья\s+\d+(?:-\d+)?(?:\.|\s*$)")
# Заголовок главы: после номера точка или конец строки, иначе это фраза вроде «Глава 5 настоящего Кодекса применяется...»
SECTION_HEADING = re.compile(r"^(?:Раздел|РАЗДЕЛ|Глава|ГЛАВА|Параграф|ПАРАГРАФ)\s+[\dIVXLC]+(?:-\d+)?(?:\.|\s*$)")
# Иногда заголовок главы и первая статья стоят в PDF одной строкой: «Глава 4. Сделки Статья 147. Понятие сделки»
HEADING_WITH_ARTICLE = re.compile(r"^((?:Раздел|РАЗДЕЛ|Глава|ГЛАВА|Параграф|ПАРАГРАФ)\s.*?)\s+(Статья\s+\d+(?:-\d+)?\..*)$")
NEW_PARAGRAPH = re.compile(
    r"^(?:\d+(?:-\d+)?[.)]\s|[а-яё]\)\s|Статья\s+\d|Глава\s+[\dIVXLC]|ГЛАВА\s+[\dIVXLC]|Раздел\s+[\dIVXLC]|РАЗДЕЛ\s+[\dIVXLC]"
    r"|Параграф\s+\d|Сноска\b|Примечание\b|Вниманию\b|[-–—]\s)"
)
SERVICE_PARAGRAPH = re.compile(r"^(?:Сноска\b|Вниманию пользователей|Для удобства пользования|Примечание (?:ИЗПИ|РЦПИ)!)|^(?:СОДЕРЖАНИЕ|ОГЛАВЛЕНИЕ|Оглавление|Содержание)\s*\.?$")
ACT_LINE = re.compile(r"^(?P<kind>Кодекс|Закон|Конституционный закон|Конституция)\b.*?\bот\s+\d")
# Строка отчётного абзаца короче этой доли типичной ширины — конец абзаца, а не перенос
SHORT_LINE = 0.7
# Преамбула короче этого — служебный текст, а не содержание акта
MIN_PREAMBLE = 200


def join_lines(pages: Sequence[str]) -> List[str]:
    """Абзацы текста: номера страниц удалены, строки, разорванные вёрсткой, склеены."""
    lines: List[str] = []
    for page in pages:
        for line in page.splitlines():
            if line.strip() and not PAGE_NUMBER.match(line):
                glued = HEADING_WITH_ARTICLE.match(line.strip())
                lines.extend(glued.groups() if glued else [line.strip()])
    if not lines:
        return []
    width = sorted(len(line) for line in lines)[int(len(lines) * 0.9)]
    paragraphs = [lines[0]]
    previous = lines[0]
    for line in lines[1:]:
        ended = len(previous) < SHORT_LINE * width or previous[-1] in ".;:!?"
        if NEW_PARAGRAPH.match(line) or (not line[0].islower() and ended):
            paragraphs.append(line)
        elif re.search(r"\w-$", paragraphs[-1]):  # перенос внутри слова с дефисом: «детьми-/сиротами»
            paragraphs[-1] += line
        else:
            paragraphs[-1] += " " + line
        previous = line
    return paragraphs


def display_title(raw: str, kind: str) -> str:
    """Подпись акта в стиле остального проекта: «Трудовой кодекс РК», «Закон РК «О языках»»."""
    title = raw.strip()
    if title.isupper():
        title = title.capitalize()
    title = re.sub(r"Республики Казахстан|республики казахстан", "РК", title)
    if kind == "Конституция" or re.match(r"[А-ЯЁ]\w+ кодекс", title):
        return title
    prefix = {"Кодекс": "Кодекс РК", "Закон": "Закон РК", "Конституционный закон": "Конституционный закон РК"}[kind]
    return f"{prefix} «{title}»"


def _is_section_heading(paragraph: str) -> bool:
    letters = [ch for ch in paragraph if ch.isalpha()]
    return bool(SECTION_HEADING.match(paragraph)) or (bool(letters) and paragraph.upper() == paragraph)


def parse_act(pages: Sequence[str]) -> Dict[str, Any]:
    """Разбирает текст PDF: {"title", "kind", "fragments": [текст статьи или преамбулы, ...]}."""
    paragraphs = join_lines(pages)
    act_index = next((i for i, p in enumerate(paragraphs[:6]) if ACT_LINE.match(p)), None)
    if act_index is None:
        raise ValueError(f"Не найдена строка с видом и датой акта: {paragraphs[:3]}")
    kind = ACT_LINE.match(paragraphs[act_index]).group("kind")
    title = display_title(" ".join(paragraphs[:act_index]), kind)

    body = [p for p in paragraphs[act_index + 1:] if not SERVICE_PARAGRAPH.match(p)]
    # Подписи к служебным примечаниям в преамбуле («Порядок введения в действие ... см. ст. 283») — не содержание акта
    fragments: List[List[str]] = [[]]
    for paragraph in body:
        if ARTICLE_START.match(paragraph):
            fragments.append([])
        fragments[-1].append(paragraph)

    articles = []
    for fragment in fragments[1:]:
        # Всё от заголовка следующей главы — уже не текст этой статьи
        end = next((i for i, p in enumerate(fragment[1:], 1) if SECTION_HEADING.match(p)), len(fragment))
        kept = fragment[:end]
        while len(kept) > 1 and _is_section_heading(kept[-1]):  # продолжение заголовка главы заглавными буквами
            kept.pop()
        # От статьи, исключённой законодателем, после удаления сноски остаётся один заголовок
        if len(kept) > 1:
            articles.append("\n".join(kept))

    preamble = [p for p in fragments[0] if not _is_section_heading(p) and not p.startswith("Порядок введения")]
    text = "\n".join(preamble)
    return {"title": title, "kind": kind, "fragments": ([text] if len(text) >= MIN_PREAMBLE else []) + articles}


def _as_of(reader) -> str:
    """Дата создания PDF (дата выгрузки редакции с сайта) в виде ДД.ММ.ГГГГ."""
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", str((reader.metadata or {}).get("/CreationDate", "")))
    return f"{match.group(3)}.{match.group(2)}.{match.group(1)}" if match else ""


def read_laws(folder: str = LAWS_DIR) -> pd.DataFrame:
    """Читает все PDF папки в таблицу статей (колонки как в загрузке датасета, плюс as_of)."""
    from pypdf import PdfReader  # нужен только при подготовке корпуса

    rows = []
    for path in sorted(Path(folder).glob("*.pdf")):
        reader = PdfReader(path)
        act = parse_act([page.extract_text() or "" for page in reader.pages])
        as_of = _as_of(reader)
        for text in act["fragments"]:
            rows.append({
                "context": text, "cleaned_context": text, "source_id": SOURCE_IDS.get(act["title"], ""),
                "source_title": act["title"], "doc_type": act["kind"], "language": "rus", "as_of": as_of,
            })
    if not rows:
        raise FileNotFoundError(f"В папке {folder} нет PDF-файлов")
    return pd.DataFrame(rows)


def load_laws(parquet: str = LAWS_PARQUET) -> pd.DataFrame:
    """Подготовленный корпус (prepare_laws.py): индекс 0..N-1 задаёт идентификаторы чанков."""
    return pd.read_parquet(parquet).reset_index(drop=True)
