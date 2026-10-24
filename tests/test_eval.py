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
