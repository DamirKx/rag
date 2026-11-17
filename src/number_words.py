"""Числа, записанные в тексте закона словами: «двадцать четыре календарных дня» → 24.

Нужно проверке чисел ответа (src/generation.py): модель пишет «24 дня», а в Трудовом кодексе —
«двадцать четыре». Разбираются русские количественные числительные во всех падежах (до миллионов),
сложные прилагательные с числом («восемнадцатилетнего», «шестимесячный») и доли («одной четверти»
→ 1/4 и 25 %). Порядковые числительные («второго») не разбираются.

Одиночное «один / одного» числом не считается: «одного из агентств» пропустило бы ответ «1 [1]».

Используется только для пополнения множества чисел контекста и включается для корпуса законов
(GroundedAnswerer(number_words=True)). На датасете Adilet проверка чисел со словами пропустила 7 ответов
при найденном нужном фрагменте, но и 4 ответа при убранном — с числом из соседней нормы, которые прежняя
проверка отсекала лишь потому, что не понимала чисел словами; пороги Adilet подобраны без неё.
"""

import re
from typing import Dict, List, Set


def _forms_ending_soft_sign(word: str) -> List[str]:
    """«пять» → пять, пяти, пятью; «двадцать» → двадцать, двадцати, двадцатью."""
    stem = word[:-1]
    return [word, stem + "и", stem + "ью"]


def _build_forms() -> Dict[str, int]:
    forms: Dict[str, int] = {}

    def add(value: int, *words: str) -> None:
        for w in words:
            forms[w] = value

    add(1, "один", "одна", "одно", "одного", "одной", "одному", "одним", "одну", "одном")
    add(2, "два", "две", "двух", "двум", "двумя")
    add(3, "три", "трех", "трем", "тремя")
    add(4, "четыре", "четырех", "четырем", "четырьмя")
    for value, word in ((5, "пять"), (6, "шесть"), (9, "девять"), (10, "десять")):
        add(value, *_forms_ending_soft_sign(word))
    add(7, "семь", "семи")  # «семью» — чаще «семья», чем «семью (7)»
    add(8, "восемь", "восьми", "восемью")
    teens = ("одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
             "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать")
    for value, word in enumerate(teens, 11):
        add(value, *_forms_ending_soft_sign(word))
    add(20, *_forms_ending_soft_sign("двадцать"))
    add(30, *_forms_ending_soft_sign("тридцать"))
    add(40, "сорок", "сорока")
    add(50, "пятьдесят", "пятидесяти", "пятьюдесятью")
    add(60, "шестьдесят", "шестидесяти", "шестьюдесятью")
    add(70, "семьдесят", "семидесяти", "семьюдесятью")
    add(80, "восемьдесят", "восьмидесяти", "восемьюдесятью")
    add(90, "девяносто", "девяноста")
    add(100, "сто", "ста")
    add(200, "двести", "двухсот", "двумстам", "двумястами", "двухстах")
    add(300, "триста", "трехсот", "тремстам", "тремястами", "трехстах")
    add(400, "четыреста", "четырехсот", "четыремстам", "четырьмястами", "четырехстах")
    for value, gen, ins in ((500, "пят", "пятью"), (600, "шест", "шестью"), (700, "сем", "семью"),
                            (800, "восьм", "восемью"), (900, "девят", "девятью")):
        nom = {500: "пятьсот", 600: "шестьсот", 700: "семьсот", 800: "восемьсот", 900: "девятьсот"}[value]
        add(value, nom, gen + "исот", gen + "истам", ins + "стами", gen + "истах")
    add(1000, "тысяча", "тысячи", "тысяч", "тысячу", "тысячей", "тысячью", "тысяче", "тысячам", "тысячами", "тысячах")
    add(10 ** 6, "миллион", "миллиона", "миллионов", "миллиону", "миллионом", "миллионе", "миллионам", "миллионами",
        "миллионах")
    return forms


FORMS = _build_forms()
MULTIPLIERS = {1000, 10 ** 6}

# Доли: значения, которыми модель может записать ту же долю
FRACTIONS = {
    **{w: ("1/4", "25") for w in ("четверть", "четверти", "четвертью")},
    **{w: ("1/3",) for w in ("треть", "трети", "третью")},
    **{w: ("1/2", "50") for w in ("половина", "половины", "половине", "половину", "половиной")},
}

# Сложные прилагательные: «восемнадцатилетнего», «шестимесячный», «трехдневный», «двухнедельный»
_COMPOUND_PREFIXES = sorted(
    {"одно": 1, "двух": 2, "трех": 3, "четырех": 4, "сорока": 40, "девяносто": 90, "ста": 100,
     **{w: v for w, v in FORMS.items() if w.endswith("и") and v < 100}}.items(),
    key=lambda kv: -len(kv[0]),
)
_COMPOUND_SUFFIX = re.compile(r"(?:летн|летия|месячн|дневн|недельн|часов|кратн|процентн)")

TOKEN = re.compile(r"[а-яa-z]+|\d+|[^\sа-яa-z\d]")


def _lowest_place(value: int) -> int:
    """Младший ненулевой разряд: 120 → 10, 124 → 1, 100 → 100."""
    place = 1
    while value % (place * 10) == 0:
        place *= 10
    return place


def _compound_value(word: str):
    for prefix, value in _COMPOUND_PREFIXES:
        if word.startswith(prefix) and _COMPOUND_SUFFIX.match(word, len(prefix)):
            return value
    return None


def numbers_in_words(text: str) -> Set[str]:
    """Числа, записанные словами, в виде строк цифр: {"24", "120", "1/4", "25", ...}."""
    found: Set[str] = set()
    total = current = words = 0
    active = False

    def flush() -> None:
        nonlocal total, current, active, words
        # Одиночное «один / одного» — чаще не число («одного из агентств»): с ним ответ «1 [1]» прошёл бы проверку
        if active and not (words == 1 and total + current == 1):
            found.add(str(total + current))
        total = current = words = 0
        active = False

    for token in TOKEN.findall(text.lower().replace("ё", "е")):
        value = FORMS.get(token)
        if value is None:
            flush()
            if token in FRACTIONS:
                found.update(FRACTIONS[token])
            compound = _compound_value(token)
            if compound is not None:
                found.add(str(compound))
            continue
        if value in MULTIPLIERS:
            current = (current or 1) * value
            total += current
            current = 0
            active = True
            words += 1
            continue
        if active and current and value >= _lowest_place(current):
            flush()  # «двадцать двадцать» — два разных числа
        current += value
        active = True
        words += 1
    flush()
    return found
