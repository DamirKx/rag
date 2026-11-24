"""doc2query: фильтр пересказов проверочных вопросов, привязка вопросов к чанкам, сворачивание выдачи."""

from src import doc2query as d2q

LAW = "Закон РК «О жилищных отношениях»"


def rec(question, positive="ctx_0001_fix_002", article="ctx_0001"):
    return {"question": question, "positive": positive, "article": article}


class TestLeakFilter:
    def test_paraphrase_of_same_article_removed(self):
        records = [rec("отчислили из универа, должен ли съехать из общаги"),
                   rec("сколько процентов алименты на двоих детей")]
        eval_q = [("Меня отчислили из колледжа, через сколько я должен съехать из общаги?", {(LAW, "114")})]
        keys = [{(LAW, "114")}, {(LAW, "114")}]
        kept, removed = d2q.leak_filter(records, eval_q, keys, 0.3)
        assert [r["question"] for r in removed] == ["отчислили из универа, должен ли съехать из общаги"]
        assert removed[0]["leak_sim"] >= 0.3
        assert [r["question"] for r in kept] == ["сколько процентов алименты на двоих детей"]

    def test_similar_question_to_other_article_kept(self):
        """Похожая формулировка, но про другую статью — не подсказка ответа, остаётся."""
        records = [rec("отчислили из универа, должен ли съехать из общаги")]
        eval_q = [("Меня отчислили из колледжа, через сколько я должен съехать из общаги?", {(LAW, "114")})]
        kept, removed = d2q.leak_filter(records, eval_q, [{(LAW, "25")}], 0.2)
        assert len(kept) == 1 and not removed

    def test_out_of_corpus_question_removes_nothing(self):
        records = [rec("как открыть ИП")]
        kept, removed = d2q.leak_filter(records, [("Как открыть ИП?", set())], [{(LAW, "1")}], 0.1)
        assert len(kept) == 1 and not removed


def test_expansions_group_by_positive_chunk():
    by_chunk = d2q.expansions([rec("а", "c_fix_001"), rec("б ", "c_fix_002"), rec("в", "c_fix_001")])
    assert by_chunk == {"c_fix_001": ["а", "в"], "c_fix_002": ["б"]}


def test_append_questions():
    assert d2q.append_questions("Текст статьи.", ["вопрос 1", "вопрос 2"]) == "Текст статьи.\nвопрос 1\nвопрос 2"
    assert d2q.append_questions("Текст статьи.", []) == "Текст статьи."


def test_question_docs_and_chunk_of():
    texts, ids = d2q.question_docs({"ctx_0001_fix_002": ["а", "б"]})
    assert texts == ["а", "б"]
    assert ids == ["ctx_0001_fix_002#q1", "ctx_0001_fix_002#q2"]
    assert d2q.chunk_of(ids[1]) == "ctx_0001_fix_002"
    assert d2q.chunk_of("ctx_0001_fix_002") == "ctx_0001_fix_002"


def test_collapse_dedupes_in_order_and_cuts():
    hits = [("a#q1", "", 0.9), ("b", "", 0.8), ("a", "", 0.7), ("c#q2", "", 0.6), ("d", "", 0.5)]
    assert d2q.collapse(hits, 3) == [("a", 0.9), ("b", 0.8), ("c", 0.6)]
    assert d2q.collapse(hits, 10) == [("a", 0.9), ("b", 0.8), ("c", 0.6), ("d", 0.5)]
