import pytest

from src.eval import calculate_hit_rate_at_k, calculate_mrr, calculate_recall_at_k

RELEVANT = [["a"], ["b", "c"], ["z"]]
RETRIEVED = [["a", "x"], ["x", "c", "b"], ["x", "y"]]


def test_mrr():
    assert calculate_mrr(RELEVANT, RETRIEVED) == pytest.approx((1 + 1 / 2 + 0) / 3)


def test_hit_rate_at_k():
    assert calculate_hit_rate_at_k(RELEVANT, RETRIEVED, k=1) == pytest.approx(1 / 3)
    assert calculate_hit_rate_at_k(RELEVANT, RETRIEVED, k=2) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert calculate_recall_at_k(RELEVANT, RETRIEVED, k=2) == pytest.approx((1 + 0.5 + 0) / 3)
    assert calculate_recall_at_k(RELEVANT, RETRIEVED, k=3) == pytest.approx((1 + 1 + 0) / 3)


def test_recall_skips_queries_without_markup():
    assert calculate_recall_at_k([["a"], []], [["a"], ["b"]], k=1) == pytest.approx(1.0)


def test_empty_input():
    assert calculate_mrr([], []) == 0.0
    assert calculate_hit_rate_at_k([], [], k=3) == 0.0
    assert calculate_recall_at_k([], [], k=3) == 0.0


@pytest.mark.parametrize("metric", [
    calculate_mrr,
    lambda rel, ret: calculate_hit_rate_at_k(rel, ret, k=1),
    lambda rel, ret: calculate_recall_at_k(rel, ret, k=1),
])
def test_length_mismatch_raises(metric):
    with pytest.raises(ValueError):
        metric([["a"], ["b"]], [["a"]])


class TestAnswerMetrics:
    def test_tokens_drop_citations_and_normalize(self):
        from src.eval import answer_tokens
        from src.text_norm import get_normalizer

        assert answer_tokens("Срок — до 49 лет [1, 2].") == ["срок", "до", "49", "лет"]
        stem = get_normalizer("snowball")
        assert answer_tokens("земельного участка", stem) == answer_tokens("земельному участку", stem)

    def test_token_f1(self):
        from src.eval import answer_token_f1

        assert answer_token_f1(["a", "b"], ["a", "b"]) == 1.0
        assert answer_token_f1(["a", "b", "c", "d"], ["a", "b"]) == pytest.approx(2 * 0.5 * 1 / 1.5)
        assert answer_token_f1(["x"], ["a"]) == 0.0
        assert answer_token_f1([], []) == 1.0

    def test_rouge_l_respects_order(self):
        from src.eval import answer_rouge_l

        assert answer_rouge_l(["a", "b", "c"], ["a", "b", "c"]) == 1.0
        # LCS «a c» = 2 из 3
        assert answer_rouge_l(["a", "x", "c"], ["a", "b", "c"]) == pytest.approx(2 / 3)
        assert answer_rouge_l(["c", "b", "a"], ["a", "b", "c"]) == pytest.approx(1 / 3)
