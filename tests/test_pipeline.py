"""Тесты кэша эмбеддингов, подготовки данных и дымовой прогон ask.py без загрузки нейросети."""

import builtins

import pandas as pd
import pytest

import ask
from prepare_adilet_data import to_text_list
from run_adilet_eval import evaluate_strategy, get_or_compute_embeddings


class TestEmbeddingCache:
    def test_reuses_cache_for_same_texts(self, fake_embedder, tmp_path):
        texts = ["первый текст", "второй текст"]
        get_or_compute_embeddings(fake_embedder, texts, str(tmp_path), "key")
        get_or_compute_embeddings(fake_embedder, texts, str(tmp_path), "key")
        assert fake_embedder.documents_encoded == 2

    def test_recomputes_when_texts_change_with_same_count(self, fake_embedder, tmp_path):
        get_or_compute_embeddings(fake_embedder, ["старый текст"], str(tmp_path), "key")
        emb = get_or_compute_embeddings(fake_embedder, ["новый текст"], str(tmp_path), "key")
        assert fake_embedder.documents_encoded == 2
        assert (emb == fake_embedder.encode_documents(["новый текст"])).all()


@pytest.mark.parametrize("value, expected", [
    (None, []),
    ("один вопрос", ["один вопрос"]),
    (["а", "б"], ["а", "б"]),
])
def test_to_text_list(value, expected):
    assert to_text_list(value) == expected


def test_evaluate_strategy_reports_all_methods(fake_embedder):
    # BM25 нужен корпус минимум из трёх документов: при N=2 IDF термина из одного документа равен 0
    chunks = ["земельный налог ставка", "образовательный грант военнослужащим", "порядок регистрации"]
    doc_ids = ["c0", "c1", "c2"]
    qa = [{"query": "какая ставка земельного налога", "source_title": "НК"}]
    results = evaluate_strategy(
        chunks=chunks,
        doc_ids=doc_ids,
        chunk_embeddings=fake_embedder.encode_documents(chunks),
        qa_data=qa,
        query_embeddings=fake_embedder.encode_queries([q["query"] for q in qa]),
        ground_truth=[["c0"]],
        top_k=1,
    )
    assert set(results) == {"dense", "bm25", "hybrid"}
    for metrics in results.values():
        assert metrics["HitRate@1"] == 1.0


def test_ask_cli_smoke(fake_embedder, tmp_path, monkeypatch, capsys):
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

    commands = iter([
        "земельный налог",
        ":mode bm25", "образовательный грант",
        ":mode dense", "земельный налог",
        ":mode unknown",
        ":mode bm25", "несуществующееслово",
        "q",
    ])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(commands))

    ask.interactive_cli(search_mode="hybrid", top_k=2, cache_dir=str(tmp_path))

    out = capsys.readouterr().out
    assert out.count("🔎 Найдено") == 3
    assert "Земельный кодекс" in out and "Закон об образовании" in out
    assert "Неизвестный режим" in out
    assert "Ничего не найдено" in out
    assert "До свидания" in out
