"""Корпус чанков с метаданными: название закона, статьи, ссылка на источник.

Общий для веб-демо (app.py) и оценки генерации (eval_generation.py): чанки совпадают
с run_adilet_eval.py (окно 500 символов, перекрытие 50, id «ctx_NNNN_fix_SSS»), поэтому
подходит готовый кэш эмбеддингов, а к каждому чанку прилагаются закон и номера статей —
их показывает интерфейс и получает языковая модель как подпись фрагмента.
"""

import re
from typing import Any, Dict, List

import pandas as pd

from src.chunking import get_fixed_size_chunks

CHUNK_SIZE, CHUNK_OVERLAP = 500, 50

# Заголовок статьи. Мягче, чем src.chunking.ARTICLE_HEADING: в Конституции заголовки записаны без точки («Статья 41»).
ARTICLE_HEADING = re.compile(
    r"^[ \t]*(?:Статья\s+(?P<rus>\d+(?:-\d+)?)|(?P<kaz>\d+(?:-\d+)?)-бап)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Название документа в шапке страницы adilet.zan.kz: «О языках в Республике Казахстан - ИПС "Әділет" ...»
PAGE_TITLE = re.compile(r'^\s*(.+?)\s+-\s+ИПС\s+"Әділет"')
TITLE_PREFIX = {"law": "Закон РК", "const_law": "Конституционный закон РК"}


def law_titles(df: pd.DataFrame) -> Dict[str, str]:
    """Названия документов по source_id, взятые из шапки исходной страницы.

    Поле source_title датасета у 9 из 14 документов не соответствует тексту
    (например, Z950002444_ подписан как «О хозяйственных товариществах»,
    а это Закон «О банках и банковской деятельности»). Шапка страницы есть только
    в первом русском фрагменте каждого документа; казахские фрагменты получают
    название по тому же source_id.

    Args:
        df: исходный датасет (колонки context, source_id, doc_type, language), до очистки текста.
    """
    titles = {}
    for row in df[df["language"] == "rus"].itertuples():
        match = PAGE_TITLE.match(row.context)
        if match and row.source_id not in titles:
            title = match.group(1).replace("Республики Казахстан", "РК")
            prefix = TITLE_PREFIX.get(row.doc_type)
            titles[row.source_id] = f"{prefix} «{title}»" if prefix else title
    return titles


def article_label(match: re.Match) -> str:
    """Короткая подпись статьи: «ст. 10» для русского текста, «10-бап» для казахского."""
    return f"ст. {match.group('rus')}" if match.group("rus") else f"{match.group('kaz')}-бап"


def build_chunk_corpus(
    sub_df: pd.DataFrame,
    titles: Dict[str, str],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Разбивает фрагменты законов на чанки и определяет для каждого чанка статьи.

    Args:
        sub_df: фрагменты одного языка (колонки cleaned_context, source_id, source_title),
                индекс 0..N-1 задаёт context_id «ctx_NNNN» — как в ask.load_legal_corpus.
        titles: названия документов по source_id (law_titles).

    Returns:
        Словари с полями id, text, context (весь фрагмент), law, source_id,
        articles («ст. 12, ст. 13»), article_title (заголовок первой статьи чанка).
    """
    corpus: List[Dict[str, Any]] = []
    for idx, row in sub_df.iterrows():
        text = row["cleaned_context"]
        headings = list(ARTICLE_HEADING.finditer(text))
        search_from = 0
        for s_idx, chunk in enumerate(get_fixed_size_chunks(text, chunk_size, chunk_overlap)):
            start = text.find(chunk, search_from)
            if start < 0:
                start = search_from
            end = start + len(chunk)
            search_from = start + 1

            # Статья, внутри которой начинается чанк, и статьи, начинающиеся внутри него
            before = [h for h in headings if h.start() <= start]
            inside = [h for h in headings if start < h.start() < end]
            covered = before[-1:] + inside
            title = covered[0].string[covered[0].start():].split("\n", 1)[0].strip() if covered else ""

            corpus.append({
                "id": f"ctx_{idx:04d}_fix_{s_idx:03d}",
                "text": chunk,
                "context": text,
                "law": titles.get(row["source_id"], row["source_title"]),
                "source_id": row["source_id"],
                "articles": ", ".join(article_label(h) for h in covered),
                "article_title": title,
            })
    return corpus
