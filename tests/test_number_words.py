"""Числа словами в тексте закона (src/number_words.py) и проверка чисел ответа с ними."""

from src.generation import unsupported_numbers
from src.number_words import numbers_in_words


def test_cardinals_in_all_cases():
    assert numbers_in_words("продолжительностью двадцать четыре календарных дня") == {"24"}
    assert numbers_in_words("не более двенадцати часов в месяц и ста двадцати часов в год") == {"12", "120"}
    assert numbers_in_words("в течение шести месяцев, четырнадцати дней") == {"6", "14"}
    assert numbers_in_words("Двести тысяч пятьсот тенге") == {"200500"}


def test_compound_adjectives_and_fractions():
    assert numbers_in_words("до достижения восемнадцатилетнего возраста") == {"18"}
    assert numbers_in_words("в трехдневный срок") == {"3"}
    assert {"1/4", "25", "1/3"} <= numbers_in_words("в размере одной четверти, на двух детей — одной трети")


def test_separate_numbers_are_not_merged_and_words_are_not_numbers():
    assert numbers_in_words("двадцать, двадцать пять") == {"20", "25"}
    assert numbers_in_words("члены семьи, с семьей, семью") == set()


def test_single_one_is_not_a_number():
    """«одного из агентств» — не число: иначе ответ «1 [1]» прошёл бы проверку чисел."""
    assert numbers_in_words("рейтинг одного из рейтинговых агентств") == set()
    assert numbers_in_words("двадцать один день, одна тысяча") == {"21", "1000"}


def test_answer_number_matches_number_in_words_only_when_enabled():
    context = "Основной оплачиваемый ежегодный трудовой отпуск предоставляется продолжительностью двадцать четыре календарных дня."
    assert unsupported_numbers("Отпуск — 24 календарных дня [1].", context, words=True) == []
    assert unsupported_numbers("Отпуск — 28 календарных дней [1].", context, words=True) == ["28"]
    # Датасет Adilet: пороги подобраны с прежней проверкой
    assert unsupported_numbers("Отпуск — 24 календарных дня [1].", context) == ["24"]
