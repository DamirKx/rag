import numpy as np
import pytest

from src.retriever import BM25Retriever, HybridRetriever, VectorRetriever, tokenize_legal_text

DOCS = [
    "Закон № 221-VII о земле и земельных участках",
    "Статья 28-1. Порядок уплаты налога в МРП",
    "Образовательный грант для военнослужащих",
]
IDS = ["d0", "d1", "d2"]


@pytest.fixture
def vector_retriever():
    retriever = VectorRetriever(embedding_dim=3)
    retriever.add_documents(np.eye(3, dtype=np.float32), DOCS, IDS)
    return retriever


@pytest.fixture
def bm25_retriever():
    retriever = BM25Retriever()
    retriever.add_documents(DOCS, IDS)
    return retriever


def test_tokenizer_keeps_compound_numbers():
    assert tokenize_legal_text("Закон № 221-VII, ст. 28-1") == ["закон", "221-vii", "ст", "28-1"]


class TestVectorRetriever:
    def test_nearest_neighbour(self, vector_retriever):
        scores, indices = vector_retriever.search(np.array([[0.1, 0.9, 0.0]]), k=2)
        assert vector_retriever.get_doc_id(indices[0][0]) == "d1"
        assert scores[0][0] > scores[0][1]

    def test_k_larger_than_index(self, vector_retriever):
        _, indices = vector_retriever.search(np.array([[1.0, 0.0, 0.0]]), k=10)
        assert indices.shape == (1, 3)
        assert (indices >= 0).all()

    def test_empty_index(self):
        scores, indices = VectorRetriever(embedding_dim=3).search(np.ones((2, 3)), k=5)
        assert scores.shape == indices.shape == (2, 0)

    @pytest.mark.parametrize("idx", [-1, 3])
    def test_out_of_range_index_raises(self, vector_retriever, idx):
        with pytest.raises(IndexError):
            vector_retriever.get_doc_id(idx)
        with pytest.raises(IndexError):
            vector_retriever.get_document(idx)

    def test_mismatched_lengths(self):
        with pytest.raises(AssertionError):
            VectorRetriever(embedding_dim=3).add_documents(np.eye(3, dtype=np.float32), DOCS[:2], IDS)


class TestBM25Retriever:
    def test_exact_number_match(self, bm25_retriever):
        results = bm25_retriever.search("статья 28-1", k=3)
        assert results[0][0] == "d1"

    def test_no_match_returns_empty(self, bm25_retriever):
        assert bm25_retriever.search("отсутствующееслово", k=3) == []

    def test_batch_search(self, bm25_retriever):
        assert bm25_retriever.batch_search(["221-vii", "грант"], k=1) == [["d0"], ["d2"]]


class TestHybridRetriever:
    def test_fuses_both_rankings(self, vector_retriever, bm25_retriever):
        hybrid = HybridRetriever(vector_retriever, bm25_retriever)
        # Dense отдаёт предпочтение d0, BM25 — d2; d2 в итоге выше за счёт двух совпадений
        query_emb = np.array([0.6, 0.0, 0.5], dtype=np.float32)
        pairs = hybrid.search_with_scores("образовательный грант", query_emb, k=3)
        doc_ids = [d for d, _ in pairs]
        assert doc_ids[0] == "d2"
        assert set(doc_ids) == set(IDS)
        assert [s for _, s in pairs] == sorted((s for _, s in pairs), reverse=True)
        assert hybrid.search("образовательный грант", query_emb, k=3) == doc_ids

    def test_batch_search(self, vector_retriever, bm25_retriever):
        hybrid = HybridRetriever(vector_retriever, bm25_retriever)
        results = hybrid.batch_search(["221-vii", "грант"], np.eye(3, dtype=np.float32)[[0, 2]], k=1)
        assert results == [["d0"], ["d2"]]
