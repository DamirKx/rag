"""Оценки эксперимента Langfuse (langfuse_experiment.py) — без модели и без сервера."""

from types import SimpleNamespace

from langfuse_experiment import averages, evaluate, field
from src.generation import STATUS_ANSWERED, STATUS_UNVERIFIED

PASSAGES = [{"label": "ГК РК, ст. 17", "chunk_ids": ["c1", "c2"]}, {"label": "ГК РК, ст. 18", "chunk_ids": ["c9"]}]


def scores(**output):
    base = {"answer": None, "status": STATUS_UNVERIFIED, "problems": [], "passages": PASSAGES, "cited": []}
    result = evaluate(input={"question": "?"}, output={**base, **output},
                      expected_output="Дееспособность возникает с наступлением совершеннолетия", metadata={"gold": ["c2"]})
    return {e.name: e.value for e in result}


def test_shown_answer_from_gold():
    s = scores(answer="Дееспособность возникает с наступлением совершеннолетия [1].", status=STATUS_ANSWERED, cited=[1])
    assert s["gold_in_context"] == 1.0 and s["answer_shown"] == 1.0
    # Сравнение по словам (основам), а не по буквам: тот же ответ с другой пунктуацией — F1 = 1
    assert s["token_f1"] == 1.0 and s["rouge_l"] == 1.0 and s["answered_from_gold"] == 1.0


def test_answer_from_other_passage():
    s = scores(answer="Дееспособность возникает с совершеннолетия [2].", status=STATUS_ANSWERED, cited=[2])
    assert s["answered_from_gold"] == 0.0 and 0 < s["token_f1"] < 1


def test_similarity_is_counted_by_words_not_letters():
    """Ответ из тех же букв, но других слов не должен получать высокий F1."""
    s = scores(answer="Совершеннолетие наступает с возникновением дееспособности [1].", status=STATUS_ANSWERED, cited=[1])
    assert s["token_f1"] < 0.8


def test_rejected_answer_gets_no_similarity_scores():
    s = scores(problems=["числа, которых нет в тексте закона: 15"])
    assert s == {"gold_in_context": 1.0, "answer_shown": 0.0}


def test_averages_count_only_present_scores():
    E = lambda name, value: SimpleNamespace(name=name, value=value)
    results = [SimpleNamespace(evaluations=[E("answer_shown", 1.0), E("token_f1", 0.5)]),
               SimpleNamespace(evaluations=[E("answer_shown", 0.0)])]
    assert averages(results) == {"answer_shown": {"mean": 0.5, "n": 2}, "token_f1": {"mean": 0.5, "n": 1}}


def test_field_reads_dict_and_dataset_item():
    assert field({"input": {"question": "q"}}, "input") == {"question": "q"}
    assert field(SimpleNamespace(input={"question": "q"}), "input") == {"question": "q"}
