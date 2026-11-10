"""Корпус чанков с метаданными: название закона, статьи, ссылка на источник.

Общий для веб-демо (app.py) и оценки генерации (eval_generation.py): чанки совпадают
с run_adilet_eval.py (окно 500 символов, перекрытие 50, id «ctx_NNNN_fix_SSS»), поэтому
подходит готовый кэш эмбеддингов, а к каждому чанку прилагаются закон и номера статей —
их показывает интерфейс и получает языковая модель как подпись фрагмента.
"""

import re
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from src.chunking import get_fixed_size_chunks

CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
# Сколько символов вокруг найденного чанка получает языковая модель (expand_hits). Окно чанкинга режет
# текст посреди слова, а ответ бывает в соседнем предложении; поиск и реранкер это значение не затрагивает
EXPAND_WINDOW = 300

# Граница предложения или абзаца. Точка засчитывается только перед заглавной буквой:
# в «ст. 12», «п. 3» и «т. д.» текст не разрывается
TEXT_BOUNDARY = re.compile(r"\n+|(?<=[.;!?])[ \t]+(?=[А-ЯЁӘҒҚҢӨҰҮҺІA-Z«\"])")

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


def _covered_articles(headings: Sequence[re.Match], start: int, end: int) -> List[re.Match]:
    """Заголовок статьи, внутри которой начинается отрезок, и заголовки статей, начинающихся внутри него."""
    return [h for h in headings if h.start() <= start][-1:] + [h for h in headings if start < h.start() < end]


def expand_span(text: str, start: int, end: int, window: int = EXPAND_WINDOW) -> Tuple[int, int]:
    """Расширяет отрезок [start, end) до границ предложений примерно на window символов в каждую сторону.

    Берётся самая дальняя граница в пределах window; если её нет, поиск продолжается ещё на window
    символов, а без границы отрезок заканчивается на целом слове. Сам отрезок всегда остаётся внутри результата.
    """
    if window <= 0:
        return start, end
    size = len(text)

    left = 0
    if start > window:
        lowest = start - 2 * window
        starts = [m.end() for m in TEXT_BOUNDARY.finditer(text, max(lowest, 0), start) if m.end() <= start]
        within = [p for p in starts if p >= start - window]
        if within:
            left = min(within)
        elif starts:
            left = max(starts)
        else:
            left = start - window
            while left > 0 and not text[left - 1].isspace():
                left -= 1

    right = size
    if end + window < size:
        ends = [m.start() for m in TEXT_BOUNDARY.finditer(text, end, min(end + 2 * window, size)) if m.start() >= end]
        within = [p for p in ends if p <= end + window]
        if within:
            right = max(within)
        elif ends:
            right = min(ends)
        else:
            right = end + window
            while right < size and not text[right].isspace():
                right += 1
    return left, right


def expand_hits(hits: Sequence[Dict[str, Any]], window: int = EXPAND_WINDOW) -> List[Dict[str, Any]]:
    """Готовит найденные чанки для языковой модели: окружающий текст вместо обрезанного окна.

    Отрезки одного фрагмента закона, которые пересекаются после расширения, склеиваются, чтобы модель
    не читала один текст дважды, поэтому результат может быть короче hits. Порядок — по лучшему чанку.

    Args:
        hits: чанки build_chunk_corpus с полем score, лучшие первыми.
        window: сколько символов добавить с каждой стороны (0 — чанки как есть, без склейки).

    Returns:
        Копии словарей с полями text, start, end, articles для расширенного отрезка, наибольшим score
        и chunk_ids — чанками, из которых отрезок собран.
    """
    if window <= 0:
        return [{**hit, "chunk_ids": [hit["id"]]} for hit in hits]

    spans: List[Dict[str, Any]] = []
    for hit in hits:
        fragment = hit["id"].split("_fix_")[0]
        left, right = expand_span(hit["context"], hit["start"], hit["end"], window)
        touching = [s for s in spans if s["fragment"] == fragment and left <= s["right"] and s["left"] <= right]
        if not touching:
            spans.append({"hit": hit, "fragment": fragment, "left": left, "right": right,
                          "chunk_ids": [hit["id"]], "score": hit["score"]})
            continue
        target = touching[0]
        target["chunk_ids"].append(hit["id"])
        target["score"] = max(target["score"], hit["score"])
        target["left"], target["right"] = min(target["left"], left), max(target["right"], right)
        for other in touching[1:]:  # новый отрезок может соединить два уже найденных
            target["chunk_ids"].extend(other["chunk_ids"])
            target["score"] = max(target["score"], other["score"])
            target["left"], target["right"] = min(target["left"], other["left"]), max(target["right"], other["right"])
            spans.remove(other)

    expanded = []
    for span in spans:
        context = span["hit"]["context"]
        covered = _covered_articles(list(ARTICLE_HEADING.finditer(context)), span["left"], span["right"])
        expanded.append({
            **span["hit"], "text": context[span["left"]:span["right"]].strip(), "start": span["left"], "end": span["right"],
            "articles": ", ".join(article_label(h) for h in covered), "score": span["score"], "chunk_ids": span["chunk_ids"],
        })
    return expanded


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
        Словари с полями id, text, context (весь фрагмент), start и end (положение чанка в context),
        law, source_id, articles («ст. 12, ст. 13»), article_title (заголовок первой статьи чанка).
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

            covered = _covered_articles(headings, start, end)
            title = covered[0].string[covered[0].start():].split("\n", 1)[0].strip() if covered else ""

            corpus.append({
                "id": f"ctx_{idx:04d}_fix_{s_idx:03d}",
                "text": chunk,
                "context": text,
                "start": start,
                "end": end,
                "law": titles.get(row["source_id"], row["source_title"]),
                "source_id": row["source_id"],
                "articles": ", ".join(article_label(h) for h in covered),
                "article_title": title,
            })
    return corpus
