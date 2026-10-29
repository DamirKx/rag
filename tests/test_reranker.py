import builtins

import pandas as pd
import pytest

import ask
import src.reranker as reranker_module
from ask import LegalSearchEngine
from src.reranker import CrossEncoderReranker
from src.retriever import tokenize_legal_text


class FakeCrossEncoder:
    """Оценка пары — число общих слов запроса и документа; запоминает оценённые пары."""

    def __init__(self, *args, **kwargs):
        self.pairs = []

    def predict(self, pairs, **kwargs):
        self.pairs.extend(pairs)
        return [len(set(tokenize_legal_text(q)) & set(tokenize_legal_text(d))) for q, d in pairs]


@pytest.fixture
def reranker(monkeypatch):
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    return CrossEncoderReranker("fake", window_chars=100, window_overlap=10)


class TestCrossEncoderReranker:
    def test_orders_by_score(self, reranker):
        candidates = [("d0", "порядок регистрации"), ("d1", "ставка земельного налога"), ("d2", "земельный участок")]
        result = reranker.rerank("ставка земельного налога", candidates)
        assert result == [("d1", 3.0), ("d0", 0.0), ("d2", 0.0)]

    def test_top_k_and_stable_ties(self, reranker):
        candidates = [("d0", "нет совпадений"), ("d1", "тоже нет"), ("d2", "грант")]
        assert reranker.rerank("грант", candidates, top_k=2) == [("d2", 1.0), ("d0", 0.0)]

    def test_long_document_scored_by_best_window(self, reranker):
        long_doc = "вводные положения закона. " * 20 + "образовательный грант"
        scores = reranker.score("образовательный грант", [long_doc, "грант"])
        assert scores == [2.0, 1.0]
        # Документ длиннее окна разбит на несколько пар, короткий — одна пара
        assert len(reranker.model.pairs) > 2

    def test_empty_candidates(self, reranker):
        assert reranker.rerank("запрос", []) == []


@pytest.fixture
def engine(fake_embedder, tmp_path, reranker):
    docs = ["земельный налог ставка", "образовательный грант военнослужащим", "порядок регистрации прав"]
    return LegalSearchEngine(
        docs=docs, doc_ids=["c0", "c1", "c2"], embedder=fake_embedder,
        cache_dir=str(tmp_path), cache_key="k", reranker=reranker, rerank_pool=3,
    )


class TestEngineRerank:
    def test_rerank_returns_cross_encoder_scores(self, engine):
        results = engine.search("образовательный грант", mode="hybrid", top_k=2, rerank=True)
        assert results[0][0] == "c1"
        assert results[0][2] == 2.0
        assert len(results) == 2

    def test_rerank_without_reranker_raises(self, engine):
        engine.reranker = None
        with pytest.raises(ValueError):
            engine.search("грант", mode="dense", top_k=1, rerank=True)


def test_ask_cli_rerank_toggle(fake_embedder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    corpus = pd.DataFrame({
        "source_title": ["Земельный кодекс", "Закон об образовании", "Закон о регистрации"],
        "cleaned_context": [
            "Статья 1. Земельный налог уплачивается ежегодно.",
            "Статья 2. Образовательный грант выдаётся военнослужащим.",
            "Статья 3. Порядок регистрации прав.",
        ],
    })
    monkeypatch.setattr(ask, "load_legal_corpus", lambda *_: corpus)
    monkeypatch.setattr(ask, "EmbeddingModel", lambda **_: fake_embedder)
    commands = iter([":rerank on", "образовательный грант", ":rerank maybe", ":rerank off", "q"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(commands))

    ask.interactive_cli(search_mode="hybrid", top_k=1, cache_dir=str(tmp_path))

    out = capsys.readouterr().out
    assert "Реранкер включен" in out and "Реранкер выключен" in out
    assert "Закон об образовании" in out
    assert "Используйте ':rerank on'" in out
