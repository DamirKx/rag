"""Тесты генерации ответа: проверки опоры на текст закона, отказы, порог поиска (без запуска Ollama)."""

import json

import pytest

from src.generation import (
    REFUSAL,
    STATUS_ANSWERED,
    STATUS_NO_CONTEXT,
    STATUS_REFUSED,
    STATUS_UNVERIFIED,
    GroundedAnswerer,
    OllamaClient,
    OllamaError,
    Passage,
    answer_sentences,
    build_messages,
    cited_passages,
    drop_invalid_citations,
    extract_citations,
    is_malformed,
    is_refusal,
    load_generation_params,
    sentence_support,
    strip_citations,
    unsupported_numbers,
)
from src.nli import entailment_scorer, load_entailment_scorer

PASSAGES = [
    Passage(
        id="ctx_0001_fix_000",
        text="Закон вступает в действие с 1 июля 2023 года. Срок аренды земельного участка — до 49 лет.",
        law="Земельный кодекс РК", articles="ст. 12", score=0.98,
    ),
    Passage(id="ctx_0002_fix_003", text="Хранитель обязан изменить способ и место хранения вещи.", law="ГК РК", score=0.4),
]


class FakeClient:
    """Возвращает заданный ответ и запоминает сообщения."""

    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return {"text": self.text, "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 0.1, "cached": False}


class SequenceClient:
    """Возвращает ответы по очереди и запоминает сообщения."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, messages):
        self.calls.append(list(messages))
        return {"text": self.texts[len(self.calls) - 1], "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 0.5, "cached": False}


class TestChecks:
    def test_citations(self):
        assert extract_citations("Срок — до 49 лет [1]. См. также [2, 1] и [ 3 ].") == [1, 2, 3]
        assert strip_citations("Срок — до 49 лет [1].") == "Срок — до 49 лет."

    def test_numbers_from_context_are_supported(self):
        context = PASSAGES[0].label + "\n" + PASSAGES[0].text
        assert unsupported_numbers("Аренда — до 49 лет [1], ст. 12.", context) == []

    def test_date_in_words_matches_numeric_date(self):
        context = PASSAGES[0].text
        assert unsupported_numbers("Закон действует с 01.07.2023 [1].", context) == []
        assert unsupported_numbers("Закон действует с 1.07.2023.", context) == []

    def test_invented_number_is_flagged(self):
        assert unsupported_numbers("Аренда — до 99 лет, штраф 50 МРП [1].", PASSAGES[0].text) == ["99", "50"]

    def test_list_numbering_is_not_a_fact(self):
        answer = "Документы:\n1) перечень продуктов [1];\n2. сведения о продуктах;\n- 3) порядок"
        assert unsupported_numbers(answer, "перечень продуктов, сведения") == []
        # Число внутри строки по-прежнему проверяется
        assert unsupported_numbers("Срок — 3) года", "срок") == ["3"]

    def test_nonbreaking_hyphen_and_space(self):
        # В датасете номера пишутся с неразрывным дефисом и узким пробелом: «пункт 1‑1», «№ 221‑VII»
        assert unsupported_numbers("Пункт 1-1, закон № 221-VII.", "пункт 1‑1, закон № 221‑VII") == []

    def test_sentence_support(self):
        context = PASSAGES[0].text
        supported, invented = sentence_support(
            "Срок аренды земельного участка составляет до 49 лет [1]. Продление возможно через нотариуса.", context
        )
        assert supported[1] == 1.0  # связка «составляет» не считается новым сведением
        assert invented[1] < 0.5

    def test_refusal_detection(self):
        assert is_refusal(REFUSAL["rus"])
        assert is_refusal("В найденных фрагментах ответа нет [1].")
        assert not is_refusal("Срок аренды — до 49 лет [1]; иных сроков, кроме этого, ответа нет во фрагменте [2], "
                              "однако статья 12 его устанавливает.")


class TestGroundedAnswerer:
    def test_supported_answer(self):
        client = FakeClient("Срок аренды земельного участка — до 49 лет [1].")
        answer = GroundedAnswerer(client, min_retrieval_score=0.5).answer("Каков срок аренды?", PASSAGES)
        assert answer.status == STATUS_ANSWERED
        assert answer.cited == [1]
        assert answer.text == client.text
        assert answer.usage["prompt_tokens"] == 10

    def test_low_retrieval_score_skips_model(self):
        client = FakeClient("не должен вызываться")
        weak = [Passage(id="x", text="Текст", score=0.1)]
        answer = GroundedAnswerer(client, min_retrieval_score=0.5).answer("Какой штраф за скорость?", weak)
        assert answer.status == STATUS_NO_CONTEXT
        assert answer.text == REFUSAL["rus"]
        assert client.calls == []

    def test_no_threshold_calls_model(self):
        client = FakeClient(REFUSAL["rus"])
        weak = [Passage(id="x", text="Текст", score=0.1)]
        answer = GroundedAnswerer(client, min_retrieval_score=None).answer("Вопрос?", weak)
        assert answer.status == STATUS_REFUSED
        assert len(client.calls) == 1

    def test_invented_number_is_not_shown(self):
        client = FakeClient("Срок аренды земельного участка — до 99 лет [1].")
        answer = GroundedAnswerer(client).answer("Каков срок аренды?", PASSAGES)
        assert answer.status == STATUS_UNVERIFIED
        assert answer.checks["unsupported_numbers"] == ["99"]
        assert "99" not in answer.text

    def test_invented_statement_is_not_shown(self):
        client = FakeClient("Срок аренды до 49 лет [1]. Продлить договор можно через нотариуса онлайн.")
        answer = GroundedAnswerer(client, min_sentence_support=0.6).answer("Каков срок аренды?", PASSAGES)
        assert answer.status == STATUS_UNVERIFIED
        assert "нотариуса" in answer.checks["weakest_sentence"]

    def test_off_topic_answer_is_not_shown(self):
        # Ответ переписан из фрагмента не по теме: все слова есть в законе, но вопрос другой
        client = FakeClient("Хранитель обязан изменить способ и место хранения вещи [2].")
        scorer_calls = []

        def scorer(question, answer):
            scorer_calls.append((question, answer))
            return 0.2

        answer = GroundedAnswerer(client, min_answer_relevance=0.5, relevance_scorer=scorer).answer("Каков срок аренды?", PASSAGES)
        assert answer.status == STATUS_UNVERIFIED
        assert answer.checks["answer_relevance"] == 0.2
        assert answer.checks["min_sentence_support"] == 1.0
        assert scorer_calls == [("Каков срок аренды?", client.text)]

    def test_relevance_check_off_or_given(self):
        raw = "Срок аренды земельного участка — до 49 лет [1]."
        off = GroundedAnswerer(None, min_answer_relevance=None, relevance_scorer=lambda q, a: 0.0)
        assert off.judge("Срок?", raw, PASSAGES).status == STATUS_ANSWERED
        # Готовая оценка из judge(relevance=...) используется без вызова scorer
        given = GroundedAnswerer(None, min_answer_relevance=0.5)
        assert given.judge("Срок?", raw, PASSAGES, relevance=0.9).status == STATUS_ANSWERED
        assert given.judge("Срок?", raw, PASSAGES, relevance=0.1).status == STATUS_UNVERIFIED
        # Без оценки и без scorer проверка пропускается
        assert given.judge("Срок?", raw, PASSAGES).checks["answer_relevance"] is None

    def test_invalid_citation_is_dropped_not_rejected(self):
        # «[7]» при двух фрагментах — номер пункта, а не фрагмента: ссылка убирается, ответ проверяется как обычно
        answer = GroundedAnswerer(FakeClient("Срок аренды до 49 лет [7].")).answer("Срок?", PASSAGES)
        assert answer.status == STATUS_ANSWERED
        assert answer.text == "Срок аренды до 49 лет."
        assert answer.checks["invalid_citations"] == [7]
        assert answer.cited == [1]

    def test_invalid_citation_does_not_hide_unsupported_text(self):
        judge = GroundedAnswerer(None, min_answer_relevance=None)
        assert judge.judge("Срок?", "[6] Срок аренды до 99 лет.", PASSAGES).status == STATUS_UNVERIFIED
        assert judge.judge("Срок?", "[7]", PASSAGES).status == STATUS_UNVERIFIED  # без текста

    @pytest.mark.parametrize("raw,expected", [
        ("[6] Срок аренды до 49 лет [1].", "Срок аренды до 49 лет [1]."),
        ("Срок аренды до 49 лет [1, 5].", "Срок аренды до 49 лет [1]."),
        ("Срок аренды до 49 лет [2; 9] и далее [4].", "Срок аренды до 49 лет [2] и далее."),
        ("Срок аренды до 49 лет [1].", "Срок аренды до 49 лет [1]."),
    ])
    def test_drop_invalid_citations(self, raw, expected):
        assert drop_invalid_citations(raw, 2) == expected

    def test_citation_only_answer_is_not_shown(self):
        # Ответ «[1]» без текста: в нём нет непроверенных предложений, но и ответа нет
        judge = GroundedAnswerer(None, min_answer_relevance=None)
        for raw in ("[1]", "[1] [2]", " [3]. "):
            answer = judge.judge("Срок?", raw, PASSAGES)
            assert answer.status == STATUS_UNVERIFIED, raw
            assert "без текста" in answer.checks["problems"][0]
        # Короткий ответ числом — это ответ
        assert judge.judge("Сколько лет?", "49 [1]", PASSAGES).status == STATUS_ANSWERED

    def test_verify_off_returns_raw(self):
        client = FakeClient("Срок аренды — до 99 лет [1].")
        answer = GroundedAnswerer(client, verify=False).answer("Срок?", PASSAGES)
        assert answer.status == STATUS_ANSWERED
        assert answer.checks["unsupported_numbers"] == ["99"]

    @pytest.mark.parametrize("bad", ["[1]", "Срок аренды до 49 лет [7]."])
    def test_malformed_answer_is_requested_again(self, bad):
        client = SequenceClient(bad, "Срок аренды земельного участка — до 49 лет [1].")
        answer = GroundedAnswerer(client).answer("Каков срок аренды?", PASSAGES)
        assert answer.status == STATUS_ANSWERED
        assert answer.usage["retried"] is True
        assert answer.usage["prompt_tokens"] == 20 and answer.usage["latency_s"] == 1.0
        assert len(client.calls) == 2
        assert client.calls[1][-2] == {"role": "assistant", "content": bad}
        assert "от 1 до 2" in client.calls[1][-1]["content"]

    def test_well_formed_answer_and_refusal_are_not_repeated(self):
        for text in ("Срок аренды земельного участка — до 49 лет [1].", REFUSAL["rus"]):
            client = SequenceClient(text)
            GroundedAnswerer(client).answer("Каков срок аренды?", PASSAGES)
            assert len(client.calls) == 1

    def test_retry_failure_is_not_shown(self):
        client = SequenceClient("[1]", "[9]")
        answer = GroundedAnswerer(client).answer("Каков срок аренды?", PASSAGES)
        assert answer.status == STATUS_UNVERIFIED
        assert len(client.calls) == 2

    def test_entailment_check(self):
        raw = "Срок аренды земельного участка — до 49 лет [1]."
        seen = []

        def scorer(question, answer, passages):
            seen.append([p.id for p in passages])
            return 0.2

        strict = GroundedAnswerer(None, min_answer_relevance=None, min_entailment=0.5, entailment_scorer=scorer)
        answer = strict.judge("Срок?", raw, PASSAGES)
        assert answer.status == STATUS_UNVERIFIED
        assert "не следует" in answer.checks["problems"][0]
        assert answer.checks["entailment"] == 0.2
        assert seen == [[PASSAGES[0].id]]  # оценивается только цитируемый фрагмент
        assert strict.judge("Срок?", raw, PASSAGES, entailment=0.9).status == STATUS_ANSWERED

    def test_entailment_is_off_without_threshold_and_skipped_after_rejection(self):
        calls = []

        def scorer(question, answer, passages):
            calls.append(answer)
            return 0.0

        raw = "Срок аренды земельного участка — до 49 лет [1]."
        off = GroundedAnswerer(None, min_answer_relevance=None, entailment_scorer=scorer)
        assert off.judge("Срок?", raw, PASSAGES).status == STATUS_ANSWERED
        rejected = GroundedAnswerer(None, min_answer_relevance=None, min_entailment=0.5, entailment_scorer=scorer)
        assert rejected.judge("Срок?", "Срок аренды — до 99 лет [1].", PASSAGES).status == STATUS_UNVERIFIED
        assert calls == []


def test_malformed_and_cited_helpers():
    assert is_malformed("[1]", 3) and is_malformed("Текст [4].", 3)
    assert not is_malformed("Текст [3].", 3) and not is_malformed(REFUSAL["rus"], 3)
    assert [p.id for p in cited_passages("Текст [2].", PASSAGES)] == [PASSAGES[1].id]
    assert cited_passages("Текст без ссылки.", PASSAGES) == PASSAGES
    assert answer_sentences("Первое [1].\n1)\nВторое предложение [2].") == ["Первое.", "Второе предложение."]


def test_prompt_states_citation_range():
    system, _ = build_messages("Вопрос?", PASSAGES)
    assert "от 1 до 2" in system["content"]
    assert "[1]" not in system["content"]  # пример «[1]» модель копировала вместо ответа
    kaz, _ = build_messages("Сұрақ?", PASSAGES, "kaz")
    assert "1-ден 2-ге" in kaz["content"]


class FakeNLI:
    def __init__(self, table):
        self.table = table

    def entailment(self, pairs):
        return [self.table[(premise, hypothesis)] for premise, hypothesis in pairs]


def test_entailment_score_is_min_over_sentences_of_max_over_passages():
    a, b = Passage(id="a", text="A"), Passage(id="b", text="B")
    score = entailment_scorer(FakeNLI({
        ("A", "Q? Первое."): 0.9, ("B", "Q? Первое."): 0.1,
        ("A", "Q? Второе."): 0.2, ("B", "Q? Второе."): 0.6,
    }))
    assert score("Q?", "Первое [1].\nВторое [2].", [a, b]) == 0.6
    assert score("Q?", "[1]", [a, b]) == 1.0  # без текста проверять нечего


def test_nli_model_is_not_loaded_without_threshold():
    assert load_entailment_scorer(None) is None


def test_messages_contain_numbered_passages_and_refusal():
    system, user = build_messages("Каков срок аренды?", PASSAGES)
    assert REFUSAL["rus"] in system["content"]
    assert "[1] Земельный кодекс РК, ст. 12\nЗакон вступает" in user["content"]
    assert "[2] ГК РК\n" in user["content"]
    assert user["content"].endswith("Вопрос: Каков срок аренды?")


def test_client_cache(tmp_path, monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return FakeResponse({"message": {"content": " Ответ [1]. "}, "prompt_eval_count": 7, "eval_count": 3})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    cache = tmp_path / "llm.jsonl"
    messages = build_messages("Вопрос?", PASSAGES)
    first = OllamaClient(cache_path=str(cache)).chat(messages)
    assert first["text"] == "Ответ [1]." and not first["cached"]
    assert calls[0]["options"]["temperature"] == 0.0
    # Новый клиент читает кэш с диска и не обращается к модели
    second = OllamaClient(cache_path=str(cache)).chat(messages)
    assert second["cached"] and second["text"] == "Ответ [1]."
    assert len(calls) == 1


def test_load_params(tmp_path):
    path = tmp_path / "generation_eval_rus.json"
    assert load_generation_params("rus", path=str(path))["min_retrieval_score"] == 0.5
    path.write_text(json.dumps({"params": {"min_retrieval_score": 0.3, "min_sentence_support": 0.7,
                                           "reranker": "models/r"}}), encoding="utf-8")
    assert load_generation_params("rus", "models/r", path=str(path)) == {
        "min_retrieval_score": 0.3, "min_sentence_support": 0.7, "min_answer_relevance": 0.5, "min_entailment": None}
    # Для другого реранкера шкала оценок другая — проверки в его шкале отключаются
    other = load_generation_params("rus", "other", path=str(path))
    assert other["min_retrieval_score"] is None and other["min_answer_relevance"] is None
    assert other["min_sentence_support"] == 0.7


def test_ollama_client_retries_on_timeout(monkeypatch):
    attempts = []

    class FakeResponse:
        def read(self):
            return json.dumps({"message": {"content": "Ответ [1]."}}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def flaky_urlopen(request, timeout):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise TimeoutError("timed out")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", flaky_urlopen)
    messages = build_messages("Вопрос?", PASSAGES)
    assert OllamaClient(retries=2).chat(messages)["text"] == "Ответ [1]."
    assert len(attempts) == 3

    attempts.clear()
    with pytest.raises(OllamaError):  # без повторов таймаут сразу превращается в понятную ошибку
        OllamaClient(retries=0).chat(messages)
    assert len(attempts) == 1
