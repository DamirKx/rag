"""Интерфейс веб-демо (app.py) в Streamlit AppTest: обычный режим и режим исследователя.

Поиск, модели и ответ подменены готовыми данными, поэтому тест не загружает нейросети, Ollama и корпус.
"""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(autouse=True)
def restore_app_module():
    """AppTest выполняет скрипт в том же процессе: подмены функций app нужно вернуть, иначе они достанутся другим тестам."""
    import app

    saved = dict(vars(app))
    yield
    for name in [n for n in vars(app) if n not in saved]:
        delattr(app, name)
    for name, value in saved.items():
        setattr(app, name, value)


def run_app(status: str = "answered", answer_available: bool = True) -> None:
    """Скрипт для AppTest: app.main() с подменёнными поиском, моделями и ответом."""
    import types

    import app
    from src.generation import Answer, Passage

    passage = Passage(id="tk_88_fix_000", law="Трудовой кодекс РК", articles="ст. 88", score=0.95, source_id="K1500000414",
                      text="Основной оплачиваемый ежегодный трудовой отпуск работникам предоставляется продолжительностью "
                           "двадцать четыре календарных дня.")
    other = {"id": "tk_87_fix_000", "law": "Трудовой кодекс РК", "articles": "ст. 87", "article_title": "Виды отпусков",
             "source_id": "K1500000414", "text": "Работникам предоставляются отпуска.", "context": "Работникам предоставляются отпуска.",
             "score": 0.6}
    hit88 = {"id": passage.id, "law": passage.law, "articles": passage.articles, "source_id": passage.source_id,
             "article_title": "Продолжительность основного оплачиваемого ежегодного трудового отпуска",
             "text": passage.text, "context": passage.text, "score": 0.95}
    # Второй чанк той же ст. 87: в списках статей он не должен стать второй карточкой
    other_b = {**other, "id": "tk_87_fix_001", "text": "Отпуска предоставляются в порядке очерёдности.", "score": 0.5}
    # Модель получает оба фрагмента, а ответ ссылается (цитата — указывает) только на первый
    passage87 = Passage(id=other["id"], text=other["text"], law=other["law"], articles=other["articles"], score=0.6,
                        source_id=other["source_id"])
    both = [passage, passage87]
    checks = {"top_score": 0.95, "min_sentence_support": 1.0, "answer_relevance": 0.9, "problems": []}
    answers = {
        "answered": Answer("answered", "Основной отпуск — двадцать четыре календарных дня. [1]", both, cited=[1],
                           checks=checks, usage={"latency_s": 3.2}),
        "no_context": Answer("no_context", "В найденных фрагментах закона ответа нет.", both, checks={"top_score": 0.3}),
        "refused": Answer("refused", "В найденных фрагментах закона ответа нет.", both, raw="В найденных фрагментах закона ответа нет.",
                          checks={"top_score": 0.9, "problems": []},
                          quotes=[{"passage": 1, "text": passage.text, "score": 0.8},
                                  {"passage": 1, "text": "Очерёдность утверждает работодатель.", "score": 0.7}]),
    }
    answer = answers[status] if answer_available else None

    # Кэш результатов общий на процесс: без подмены тест получил бы ответ предыдущего теста на тот же вопрос
    app.result_cache = lambda: app.ResultCache()
    app.start_warm_up = lambda *a, **k: None
    app.start_answer_warm_up = lambda *a, **k: None
    app.available_corpora = lambda: ["laws", "adilet"]
    app.laws_as_of = lambda: "2026-11-09"
    app.build_corpus = lambda lang, corpus_name="adilet": [hit88, other, other_b]
    app.chunks_by_id = lambda lang, corpus_name: {h["id"]: h for h in (hit88, other, other_b)}
    app.corpus_settings = lambda lang, corpus_name: {"reranker": "BAAI/bge-reranker-v2-m3", "dictionary": True,
                                                    "focus_on_refusal": 4}
    app.answer_params = lambda lang, corpus_name: {"min_retrieval_score": 0.676, "min_sentence_support": 0.8,
                                                  "min_answer_relevance": 0.039, "min_entailment": None}
    app.load_answerer = lambda lang, corpus_name: types.SimpleNamespace(
        min_retrieval_score=0.676, min_sentence_support=0.8, min_answer_relevance=0.039)
    app.load_hybrid_params = lambda model, lang: {"dense_weight": 0.5, "sparse_weight": 0.5, "rrf_k": 20}
    app.available_models = lambda lang: ["models/e5-small-adilet-rus"]
    app.default_model = lambda lang: "models/e5-small-adilet-rus"
    app.rerank_pool = lambda lang: 20
    app.get_langfuse = lambda: None
    app.load_models = lambda *a, **k: None
    app.run_search = lambda *a, **k: [hit88, other, other_b]
    app.search_and_answer = lambda *a, **k: {
        "hits": [hit88, other, other_b], "search_ms": 12.0, "answer": answer, "passages": both, "trace_id": None,
        "error": None if answer_available else "Ollama недоступна по адресу http://localhost:11434",
    }
    app.main()


def html_text(at: AppTest) -> str:
    """Весь текст страницы: markdown (в том числе HTML нормы закона) и подписи."""
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def ask(at: AppTest, question: str = "Сколько дней отпуска положено?") -> AppTest:
    at.text_input(key="query").input(question)
    return at.run(timeout=30)


def test_citizen_mode_shows_answer_with_article_in_margin():
    at = AppTest.from_function(run_app).run(timeout=30)
    assert not at.sidebar.radio  # настроек для исследования нет
    at = ask(at)
    page = html_text(at)
    assert "двадцать четыре календарных дня" in page
    assert "Ответ проверен по тексту закона" in page
    assert "norm-article'>Статья 88<" in page
    assert "Открыть на adilet.zan.kz" in page
    assert "реранкер" not in page.lower() and "оценка" not in page.lower()
    # Фрагмент, который модель получила, но на который ответ не ссылается, не пропадает со страницы;
    # два чанка ст. 87 — одна карточка
    assert any(e.label == "Другие найденные статьи (1)" for e in at.expander)
    assert "Статья 87" in page
    assert "не юридическая консультация" in page


def test_research_mode_shows_settings_scores_and_comparison():
    at = AppTest.from_function(run_app).run(timeout=30)
    at.toggle(key="research").set_value(True).run(timeout=30)
    assert at.sidebar.header[0].value == "Настройки для исследования"
    at = ask(at)
    page = html_text(at)
    assert "Проверки: числа есть в тексте закона" in page
    assert "1-е место, cross-encoder 0.950" in page
    assert [t.label for t in at.tabs] == ["Результаты", "Сравнение методов"]


def test_uncertain_search_lists_laws_and_shows_found_articles_folded():
    at = ask(AppTest.from_function(run_app, kwargs={"status": "no_context"}).run(timeout=30))
    page = html_text(at)
    assert "Уверенного ответа нет" in page
    assert "Справочник ищет только в этих актах: Трудовой кодекс РК" in page
    # Тема может быть в справочнике, хотя поиск не уверен: найденные статьи свёрнуты, по одной карточке на статью
    assert any(e.label == "Статьи, которые нашёл поиск, — проверьте, подходят ли они (2)" for e in at.expander)
    assert "Статья 88" in page and "Статья 87" in page


def test_refusal_shows_law_quotes_with_caution():
    at = ask(AppTest.from_function(run_app, kwargs={"status": "refused"}).run(timeout=30))
    page = html_text(at)
    assert "Готового ответа нет" in page
    assert "проверьте, подходят ли" in page.lower()
    # Два пункта ст. 88 — одна карточка с обоими пунктами
    assert page.count("norm-article'>Статья 88<") == 1
    assert "Очерёдность утверждает работодатель." in page


def test_model_unavailable_shows_found_articles():
    at = ask(AppTest.from_function(run_app, kwargs={"answer_available": False}).run(timeout=30))
    page = html_text(at)
    assert "Ответ сейчас недоступен" in page
    assert "Статья 87" in page and "Статья 88" in page
    assert not at.error  # техническая ошибка показывается только в режиме исследователя


def test_example_fills_question():
    at = AppTest.from_function(run_app).run(timeout=30)
    at.button_group[0].set_value("Сколько дней отпуска положено в год?").run(timeout=30)
    assert at.text_input(key="query").value == "Сколько дней отпуска положено в год?"


def test_same_example_picked_again_fills_question_again():
    example = "Сколько дней отпуска положено в год?"
    at = AppTest.from_function(run_app).run(timeout=30)
    at.button_group[0].set_value(example).run(timeout=30)
    at = ask(at, "Сколько алиментов платят?")
    at.button_group[0].set_value(None).run(timeout=30)  # снять выбор примера
    at.button_group[0].set_value(example).run(timeout=30)
    assert at.text_input(key="query").value == example


def test_example_unselected_when_question_typed_by_hand():
    at = AppTest.from_function(run_app).run(timeout=30)
    at.button_group[0].set_value("Сколько дней отпуска положено в год?").run(timeout=30)
    at = ask(at, "Сколько алиментов платят?")
    assert at.button_group[0].value is None
    assert at.text_input(key="query").value == "Сколько алиментов платят?"


def test_refusal_keeps_unquoted_passages_in_other_articles():
    at = ask(AppTest.from_function(run_app, kwargs={"status": "refused"}).run(timeout=30))
    assert any(e.label == "Другие найденные статьи (1)" for e in at.expander)
