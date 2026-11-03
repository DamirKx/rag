"""Тесты BM25 на разреженных матрицах и нормализации словоформ."""

import json
import math
from collections import Counter

import numpy as np
import pytest

import ask
from src.retriever import BM25Retriever, tokenize_legal_text
from src.text_norm import get_normalizer

CORPUS = [
    "Налогоплательщик обязан уплатить налог в срок, установленный статьёй 28-1.",
    "Налогоплательщики уплачивают налог по месту нахождения земельного участка.",
    "Земельный участок предоставляется в аренду на срок до 49 лет.",
    "Закон № 221-VII вступает в силу с 1 января 2024 года.",
    "Статья 28. Порядок обжалования решений налогового органа.",
    "Порядок предоставления земельных участков в собственность граждан.",
]
IDS = [f"d{i}" for i in range(len(CORPUS))]
QUERIES = ["срок уплаты налога", "земельный участок в аренду", "статья 28-1", "закон 221-vii", "налог налог"]


def reference_scores(query, docs, k1, b, idf_variant, normalize=lambda t: t, epsilon=0.25):
    """BM25 «в лоб» по формуле — эталон для матричной реализации."""
    tokenized = [normalize(tokenize_legal_text(d)) for d in docs]
    n = len(docs)
    avgdl = sum(map(len, tokenized)) / n
    df = Counter(t for doc in tokenized for t in set(doc))
    if idf_variant == "okapi":
        raw = {t: math.log(n - f + 0.5) - math.log(f + 0.5) for t, f in df.items()}
        floor = epsilon * sum(raw.values()) / len(raw)
        idf = {t: (v if v >= 0 else floor) for t, v in raw.items()}
    else:
        idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    scores = []
    for doc in tokenized:
        tf = Counter(doc)
        s = 0.0
        for t in normalize(tokenize_legal_text(query)):   # повтор слова в запросе считается дважды
            if t in tf:
                s += idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(s)
    return np.array(scores)


class TestFormula:
    @pytest.mark.parametrize("idf", ["okapi", "lucene"])
    @pytest.mark.parametrize("k1,b", [(1.5, 0.75), (0.4, 0.0), (1.2, 1.0)])
    def test_matches_reference(self, idf, k1, b):
        bm25 = BM25Retriever(k1=k1, b=b, idf=idf)
        bm25.add_documents(CORPUS, IDS)
        for q in QUERIES:
            np.testing.assert_allclose(bm25.get_scores(q), reference_scores(q, CORPUS, k1, b, idf), rtol=1e-9, atol=1e-12)

    def test_matches_reference_with_normalizer(self):
        bm25 = BM25Retriever(k1=0.4, b=0.0, idf="lucene", normalizer="prefix5")
        bm25.add_documents(CORPUS, IDS)
        norm = get_normalizer("prefix5")
        for q in QUERIES:
            np.testing.assert_allclose(bm25.get_scores(q), reference_scores(q, CORPUS, 0.4, 0.0, "lucene", norm), rtol=1e-9)

    def test_default_matches_rank_bm25(self):
        rank_bm25 = pytest.importorskip("rank_bm25")
        ref = rank_bm25.BM25Okapi([tokenize_legal_text(d) for d in CORPUS])
        bm25 = BM25Retriever()
        bm25.add_documents(CORPUS, IDS)
        for q in QUERIES:
            np.testing.assert_allclose(bm25.get_scores(q), ref.get_scores(tokenize_legal_text(q)), rtol=1e-9, atol=1e-12)

    def test_set_params_equals_fresh_index(self):
        tuned = BM25Retriever()
        tuned.add_documents(CORPUS, IDS)
        tuned.set_params(k1=0.4, b=0.1, idf="lucene")
        fresh = BM25Retriever(k1=0.4, b=0.1, idf="lucene")
        fresh.add_documents(CORPUS, IDS)
        for q in QUERIES:
            np.testing.assert_allclose(tuned.get_scores(q), fresh.get_scores(q))


class TestSearch:
    @pytest.fixture
    def bm25(self):
        retriever = BM25Retriever(normalizer="prefix5", k1=0.4, b=0.0, idf="lucene")
        retriever.add_documents(CORPUS, IDS)
        return retriever

    def test_batch_equals_single(self, bm25):
        batch = bm25.batch_search_with_scores(QUERIES, k=3, batch_size=2)   # несколько пачек
        assert batch == [bm25.search(q, k=3) for q in QUERIES]

    def test_sorted_positive_scores(self, bm25):
        for pairs in bm25.batch_search_with_scores(QUERIES, k=len(CORPUS)):
            scores = [s for _, s in pairs]
            assert scores == sorted(scores, reverse=True)
            assert all(s > 0 for s in scores)

    def test_ties_broken_by_document_order(self):
        # idf='lucene': у okapi слово из половины документов получает idf = 0 и не находится вовсе
        bm25 = BM25Retriever(idf="lucene")
        bm25.add_documents(["налог", "налог", "земля"], ["a", "b", "c"])
        assert [d for d, _ in bm25.search("налог", k=3)] == ["a", "b"]

    def test_word_forms_match_after_normalization(self):
        raw = BM25Retriever(idf="lucene")
        raw.add_documents(CORPUS, IDS)
        stemmed = BM25Retriever(normalizer="prefix5", idf="lucene")
        stemmed.add_documents(CORPUS, IDS)
        # «налогоплательщика» нет в корпусе дословно; после обрезки до 5 букв совпадает с
        # «налогоплательщик», «налогоплательщики» и «налогового» (основа «налог»)
        assert raw.search("налогоплательщика", k=3) == []
        assert {d for d, _ in stemmed.search("налогоплательщика", k=3)} == {"d0", "d1", "d4"}

    def test_okapi_idf_zero_for_half_of_documents(self):
        # Особенность формулы rank_bm25: слово ровно в половине документов не влияет на оценку
        bm25 = BM25Retriever(idf="okapi")
        bm25.add_documents(["налог", "налог", "земля", "суд"], ["a", "b", "c", "d"])
        assert bm25.search("налог", k=3) == []

    def test_unknown_words_and_empty_index(self, bm25):
        assert bm25.search("квазиэлектрон", k=3) == []
        assert BM25Retriever().search("налог", k=3) == []
        assert BM25Retriever().batch_search(["налог", "срок"], k=3) == [[], []]

    def test_invalid_idf(self):
        with pytest.raises(ValueError):
            BM25Retriever(idf="bm11")
        with pytest.raises(ValueError):
            BM25Retriever().set_params(idf="bm11")


class TestNormalizers:
    def test_prefix_keeps_numbers(self):
        norm = get_normalizer("prefix5")
        assert norm(["налогоплательщика", "28-1", "221-vii", "2024", "мрп"]) == ["налог", "28-1", "221-vii", "2024", "мрп"]

    def test_none_is_identity(self):
        assert get_normalizer("none")(["земельного", "участка"]) == ["земельного", "участка"]

    @pytest.mark.parametrize("name", ["porter", "prefix0", "prefix"])
    def test_unknown_normalizer(self, name):
        with pytest.raises(ValueError):
            get_normalizer(name)

    def test_snowball_unifies_forms(self):
        pytest.importorskip("Stemmer")
        norm = get_normalizer("snowball")
        assert len(set(norm(["налогоплательщика", "налогоплательщиком", "налогоплательщику"]))) == 1
        assert norm(["28-1"]) == ["28-1"]

    def test_lemma_gives_normal_form(self):
        pytest.importorskip("pymorphy3")
        norm = get_normalizer("lemma")
        assert norm(["земельного", "участков"]) == ["земельный", "участок"]


def test_load_bm25_params(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert ask.load_bm25_params("rus") == {}
    (tmp_path / "data" / "adilet").mkdir(parents=True)
    best = {"normalizer": "prefix5", "idf": "lucene", "k1": 0.4, "b": 0.0}
    (tmp_path / "data" / "adilet" / "bm25_tuning_rus.json").write_text(json.dumps({"best": best}), encoding="utf-8")
    params = ask.load_bm25_params("rus")
    assert params == best
    BM25Retriever(**params)   # параметры подходят конструктору
