"""Нормализация словоформ для лексического поиска (BM25).

Без нормализации «налогоплательщика» и «налогоплательщиком» для BM25 — разные слова,
и вопрос не находит фрагмент, где термин стоит в другом падеже. Нормализатор приводит
токены к общей основе одинаково для документов и запросов.

Варианты (имя передаётся в BM25Retriever(normalizer=...)):
- "none"      — без нормализации (как раньше);
- "prefixN"   — первые N букв слова, например "prefix5". Не зависит от языка — подходит и для казахского;
- "snowball"  — стеммер Snowball для русского (библиотека PyStemmer);
- "lemma"     — лемматизация (начальная форма слова) для русского (библиотека pymorphy3).

Токены с цифрами (номера законов, статей, даты: '221-vii', '28-1', '2003') не изменяются:
иначе «статья 28» и «статья 28-1» слились бы.
"""

import re
from functools import lru_cache
from typing import Callable, List

Normalizer = Callable[[List[str]], List[str]]

HAS_DIGIT = re.compile(r"\d")
PREFIX = re.compile(r"^prefix(\d+)$")


def _keep_numbers(fn: Callable[[str], str]) -> Normalizer:
    """Применяет fn к каждому токену без цифр; результаты кэшируются (словарь корпуса ~ десятки тысяч слов)."""
    cached = lru_cache(maxsize=None)(fn)

    def normalize(tokens: List[str]) -> List[str]:
        return [t if HAS_DIGIT.search(t) else cached(t) for t in tokens]

    return normalize


def _snowball() -> Callable[[str], str]:
    try:
        import Stemmer
    except ImportError as e:
        raise ImportError("Для normalizer='snowball' нужен пакет PyStemmer: pip install PyStemmer") from e
    return Stemmer.Stemmer("russian").stemWord


def _lemma() -> Callable[[str], str]:
    try:
        import pymorphy3
    except ImportError as e:
        raise ImportError("Для normalizer='lemma' нужен пакет pymorphy3: pip install pymorphy3") from e
    morph = pymorphy3.MorphAnalyzer()
    return lambda word: morph.parse(word)[0].normal_form


def get_normalizer(name: str) -> Normalizer:
    """Возвращает функцию нормализации списка токенов по имени варианта."""
    if name == "none":
        return lambda tokens: list(tokens)
    match = PREFIX.match(name)
    if match:
        n = int(match.group(1))
        if n < 1:
            raise ValueError("Длина основы в prefixN должна быть положительной")
        return _keep_numbers(lambda word: word[:n])
    if name == "snowball":
        return _keep_numbers(_snowball())
    if name == "lemma":
        return _keep_numbers(_lemma())
    raise ValueError(f"Неизвестный нормализатор: {name!r} (варианты: none, prefixN, snowball, lemma)")
