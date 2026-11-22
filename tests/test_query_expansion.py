"""Словарь «разговорное → юридическое» (src/query_expansion.py)."""

import re

from src.query_expansion import RULES, expand_query, legal_terms


def test_colloquial_question_gets_legal_terms():
    expanded = expand_query("Как уволиться по собственному желанию?")
    assert expanded.startswith("Как уволиться по собственному желанию? (")
    assert "расторжение трудового договора" in expanded and "по инициативе работника" in expanded


def test_divorce_and_consumer_terms():
    assert "расторжение брака (супружества)" in legal_terms("Как развестись с мужем?")
    assert any("ненадлежащего качества" in t for t in legal_terms("Купил бракованный телефон"))


def test_letter_yo_and_case_do_not_matter():
    assert legal_terms("Можно ли взять отпуск за свой СЧЁТ?") == legal_terms("можно ли взять отпуск за свой счет?")


def test_unrelated_question_unchanged():
    question = "Какой штраф за превышение скорости?"
    assert expand_query(question) == question


def test_terms_are_not_repeated():
    terms = legal_terms("Уволили, уволили, зарплату задерживают и зарплату не платят")
    assert len(terms) == len(set(terms))


def test_all_rules_compile_and_have_terms():
    for pattern, terms in RULES:
        re.compile(pattern)
        assert terms.strip()


def test_short_stems_do_not_fire_inside_other_words():
    # «долг» в «долго», «муж» в «мужчина», «жена» в «положена» — не те слова
    assert "договор займа" not in legal_terms("Как долго длится испытательный срок?")
    assert "супруги" not in legal_terms("Со скольки лет мужчина может выйти на пенсию?")
    assert "супруги" not in legal_terms("Положена ли мне бесплатная медицинская помощь?")


def test_word_forms_still_match():
    assert "договор займа" in legal_terms("Дал другу в долг")
    assert "договор займа" in legal_terms("Занял денег у соседа, а он не отдаёт долги")
    for question in ("Муж не платит", "Новой жене", "Мы женаты пять лет", "Она замужем"):
        assert "супруги" in legal_terms(question), question
    assert "наследование" in legal_terms("Можно ли признать умершим пропавшего?")
