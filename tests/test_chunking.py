from pathlib import Path

import pytest

from src.chunking import get_fixed_size_chunks, get_structural_chunks


def ids(chunks):
    return [c["id"] for c in chunks]


class TestFixedSizeChunks:
    def test_sliding_window_with_overlap(self):
        chunks = get_fixed_size_chunks("abcdefghij", chunk_size=4, chunk_overlap=1)
        assert chunks == ["abcd", "defg", "ghij", "j"]

    def test_empty_text(self):
        assert get_fixed_size_chunks("", chunk_size=10, chunk_overlap=2) == []

    @pytest.mark.parametrize("size, overlap", [(0, 0), (10, -1), (10, 10)])
    def test_invalid_params(self, size, overlap):
        with pytest.raises(ValueError):
            get_fixed_size_chunks("text", chunk_size=size, chunk_overlap=overlap)


class TestStructuralChunks:
    def test_plain_article_1_and_10_to_19_are_kept(self):
        text = "Статья 1. Основные понятия\nТекст один.\nСтатья 2. Цели\nТекст два.\nСтатья 12. Прочее\nТекст двенадцать."
        chunks = get_structural_chunks(text)
        assert ids(chunks) == ["art1", "art2", "art12"]
        assert chunks[0]["text"].startswith("Статья 1. Основные понятия")
        assert "Текст двенадцать." in chunks[2]["text"]

    def test_kazakh_articles(self):
        text = "1-бап. Жер қоры\nмәтін бір\n2-бап. Санаттар\nмәтін екі\n5-1-бап. Қосымша\nмәтін"
        assert ids(get_structural_chunks(text)) == ["art1", "art2", "art5-1"]

    def test_preamble_is_kept(self):
        text = "Закон Республики Казахстан\nСтатья 1. Предмет\nТекст."
        chunks = get_structural_chunks(text)
        assert ids(chunks) == ["preamble", "art1"]
        assert chunks[0]["text"] == "Закон Республики Казахстан"

    def test_quoted_article_inside_amendment_is_not_split(self):
        text = (
            "Статья 1. Внести изменения:\n"
            "       1. В Закон \"Об образовании\":\n"
            "      дополнить статьей 5-1 следующего содержания:\n"
            "      \"Статья 5-1. Компетенция\n"
            "      текст новой статьи\".\n"
            "Статья 2. Вводится в действие."
        )
        chunks = get_structural_chunks(text)
        assert ids(chunks) == ["art1_p1", "art2"]
        assert "Статья 5-1. Компетенция" in chunks[0]["text"]

    def test_amendment_points_and_subpoints(self):
        text = (
            "Статья 1. Внести изменения:\n"
            " 1. В Закон \"О науке\":\n"
            " 1) в статье 1 слова заменить;\n"
            " 2) статью 2 исключить.\n"
            " 2. В Кодекс \"О земле\":\n"
            " пункт 3 исключить.\n"
        )
        chunks = get_structural_chunks(text)
        assert ids(chunks) == ["art1_p1_sub1", "art1_p1_sub2", "art1_p2"]
        # Заголовок статьи внедряется в каждый подпункт
        assert all(c["text"].startswith("Статья 1. Внести изменения:") for c in chunks)

    def test_repeated_article_numbers_get_unique_ids(self):
        text = "Статья 1. А\nтекст\nСтатья 1. Б\nтекст"
        assert ids(get_structural_chunks(text)) == ["art1", "art1_2"]

    def test_real_law_matches_qa_markup(self):
        text = (Path(__file__).parent.parent / "data" / "data.txt").read_text(encoding="utf-8")
        chunk_ids = set(ids(get_structural_chunks(text)))
        for expected in ["preamble", "art1_p1", "art1_p2_sub1", "art1_p3_sub6", "art1_p5_sub1", "art2"]:
            assert expected in chunk_ids
