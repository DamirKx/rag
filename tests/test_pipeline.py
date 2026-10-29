"""Тесты кэша эмбеддингов, подготовки данных и дымовой прогон ask.py без загрузки нейросети."""

import builtins

import pandas as pd
import pytest

import ask
from prepare_adilet_data import clean_adilet_context, to_text_list
from run_adilet_eval import GoldChunkMatcher, evaluate_strategy, get_or_compute_embeddings


def test_clean_removes_site_menu_but_keeps_act_header():
    raw = (
        'Земельный кодекс РК - ИПС "Әділет" ' + "\n" * 60
        + "Вернуться в мобильную версию\n\nБиблиотека исследований \n\nПолная версия сайта \n\n"
        + "Земельный кодекс РК\n\nОбновленный \n\nКодекс РК от 20 июня 2003 года № 442.\n\nPDF \nDOCX \n\n"
        + "Статья 1. Земельный фонд"
    )
    cleaned = clean_adilet_context(raw)
    for menu in ("Вернуться в мобильную версию", "Библиотека исследований", "Полная версия сайта", "PDF", "DOCX"):
        assert menu not in cleaned
    for kept in ("Обновленный", "от 20 июня 2003 года № 442", "Статья 1. Земельный фонд"):
        assert kept in cleaned
    assert "\n\n\n" not in cleaned


class TestGoldChunkMatcher:
    CHUNKS = [
        "Статья 69. Частный сервитут устанавливается по договору",
        "по договору или решением суда. Размер платы за сервитут",
        "Статья 70. Публичный сервитут",
    ]
    CTX = ["c1", "c1", "c1"]

    def test_exact_match(self):
        item = {"context_id": "c1", "relevant_text": "Статья 70. Публичный", "full_response": ""}
        assert GoldChunkMatcher(self.CHUNKS, self.CTX).find(item) == ([2], "exact")

    def test_near_tie_marks_both_chunks(self):
        item = {"context_id": "c1", "relevant_text": "пересказ", "full_response": "Сервитут по договору"}
        assert GoldChunkMatcher(self.CHUNKS, self.CTX).find(item) == ([0, 1], "overlap")

    def test_falls_back_to_whole_context(self):
        item = {"context_id": "c1", "relevant_text": "пересказ", "full_response": "налог на имущество"}
        assert GoldChunkMatcher(self.CHUNKS, self.CTX).find(item) == ([0, 1, 2], "context")


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
