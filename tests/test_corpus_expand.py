"""Тесты контекста для языковой модели: расширение чанка до границ предложений, склейка, трассировка."""

import pandas as pd

from src import tracing
from src.corpus import build_chunk_corpus, expand_hits, expand_span
from src.generation import Answer, GroundedAnswerer, Passage


def hit(context, start, end, chunk_id="ctx_0001_fix_000", score=0.9, **extra):
    return {"id": chunk_id, "text": context[start:end], "context": context, "start": start, "end": end, "score": score, **extra}


class TestExpandSpan:
    TEXT = "Первое предложение закона. Второе предложение закона. В четвертую очередь удовлетворяются требования. Последнее."

    def test_cut_word_is_restored(self):
        start = self.TEXT.index("вертую")
        left, right = expand_span(self.TEXT, start, start + 20, window=40)
        assert "В четвертую очередь удовлетворяются требования." in self.TEXT[left:right]
        assert left == 0 or self.TEXT[left - 2:left] == ". "  # начало — граница предложения, а не середина слова

    def test_span_stays_inside(self):
        for start, end in ((0, 5), (30, 60), (len(self.TEXT) - 5, len(self.TEXT))):
            left, right = expand_span(self.TEXT, start, end, window=25)
            assert left <= start and end <= right

    def test_zero_window_keeps_span(self):
        assert expand_span(self.TEXT, 10, 20, window=0) == (10, 20)

    def test_abbreviation_is_not_a_boundary(self):
        text = "Х" * 50 + " согласно ст. 12 настоящего закона действует правило " + "Я" * 50
        start = text.index("настоящего")
        left, _ = expand_span(text, start, start + 5, window=20)
        assert not text[left:].startswith("12")

    def test_no_boundary_ends_on_whole_word(self):
        text = " ".join(["слово"] * 60)
        left, right = expand_span(text, 100, 120, window=23)
        assert left == 0 or text[left - 1] == " "
        assert right == len(text) or text[right] == " "


class TestExpandHits:
    def test_overlapping_chunks_are_merged(self):
        context = "Статья 1. Первая норма закона.\nВторая норма закона.\nСтатья 2. Третья норма закона.\nЧетвертая норма."
        first, second = hit(context, 31, 50, "ctx_0001_fix_001", 0.5), hit(context, 52, 80, "ctx_0001_fix_002", 0.9)
        result = expand_hits([first, second], window=25)
        assert len(result) == 1
        assert result[0]["chunk_ids"] == ["ctx_0001_fix_001", "ctx_0001_fix_002"]
        assert result[0]["score"] == 0.9
        assert result[0]["articles"] == "ст. 1, ст. 2"

    def test_other_fragment_is_not_merged(self):
        context = "Одна норма закона. Другая норма закона."
        result = expand_hits([hit(context, 0, 10, "ctx_0001_fix_000"), hit(context, 0, 10, "ctx_0002_fix_000", 0.4)], window=10)
        assert [r["chunk_ids"] for r in result] == [["ctx_0001_fix_000"], ["ctx_0002_fix_000"]]

    def test_bridge_chunk_joins_two_spans(self):
        context = "\n".join(f"Норма номер {i} закона." for i in range(12))
        low, high = context.index("Норма номер 1 "), context.index("Норма номер 7 ")
        middle = context.index("Норма номер 4 ")
        hits = [hit(context, low, low + 10, "ctx_0001_fix_000"), hit(context, high, high + 10, "ctx_0001_fix_002"),
                hit(context, middle, middle + 10, "ctx_0001_fix_001")]
        separate = expand_hits(hits[:2], window=30)
        assert len(separate) == 2
        joined = expand_hits(hits, window=45)
        assert len(joined) == 1 and set(joined[0]["chunk_ids"]) == {"ctx_0001_fix_000", "ctx_0001_fix_001", "ctx_0001_fix_002"}

    def test_zero_window_returns_chunks(self):
        context = "Норма закона. Вторая норма."
        result = expand_hits([hit(context, 0, 5)], window=0)
        assert result[0]["text"] == context[:5] and result[0]["chunk_ids"] == ["ctx_0001_fix_000"]

    def test_input_is_not_modified(self):
        context = "Норма закона. Вторая норма."
        original = hit(context, 0, 5)
        expand_hits([original], window=30)
        assert original["text"] == context[:5] and "chunk_ids" not in original

    def test_passage_from_expanded_hit_lists_chunks(self):
        context = "Норма закона. Вторая норма."
        passage = Passage.from_hit(expand_hits([hit(context, 0, 5)], window=30)[0])
        assert passage.ids == ["ctx_0001_fix_000"] and passage.text == context
        assert Passage(id="x", text="t").ids == ["x"]


def test_corpus_chunks_know_their_position():
    text = "Статья 1. " + "Норма закона о земле. " * 60
    df = pd.DataFrame({"cleaned_context": [text], "source_id": ["K1"], "source_title": ["Кодекс"]})
    corpus = build_chunk_corpus(df, {})
    assert len(corpus) > 1
    for chunk in corpus:
        assert chunk["context"][chunk["start"]:chunk["end"]] == chunk["text"]


class TestTracing:
    def reset(self, monkeypatch, client=None):
        monkeypatch.setattr(tracing, "_client", client)
        monkeypatch.setattr(tracing, "_initialized", True)

    def test_no_keys_means_no_client(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setattr(tracing, "_client", None)
        monkeypatch.setattr(tracing, "_initialized", False)
        assert tracing.get_langfuse() is None

    def test_tracing_failure_does_not_break_answer(self, monkeypatch, capsys):
        class Broken:
            def start_as_current_observation(self, **kwargs):
                raise RuntimeError("сервер недоступен")

        self.reset(monkeypatch, Broken())

        class Client:
            model = "m"

            def chat(self, messages):
                return {"text": "Норма закона [1].", "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.1, "cached": False}

        passages = [Passage(id="a", text="Норма закона.", score=1.0)]
        answer = GroundedAnswerer(Client(), min_retrieval_score=None, min_answer_relevance=None).answer("Какая норма?", passages)
        assert isinstance(answer, Answer) and answer.answered
        assert "сервер недоступен" in capsys.readouterr().out
