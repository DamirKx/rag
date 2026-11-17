"""Хранилище моделей и кэш результатов веб-демо (app.py): загрузка один раз, вытеснение старых результатов."""

import threading
import time

from app import ModelStore, ResultCache


def test_model_store_loads_once_for_concurrent_requests():
    """Фоновая загрузка и запрос пользователя ждут одну и ту же загрузку, а не запускают вторую."""
    store, calls = ModelStore(), []

    def factory():
        calls.append(1)
        time.sleep(0.1)
        return object()

    results = []
    threads = [threading.Thread(target=lambda: results.append(store.get("engine", factory))) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
    assert len({id(r) for r in results}) == 1
    assert store.ready("engine") and not store.ready("reranker")


def test_model_store_retries_after_failed_load():
    store, attempts = ModelStore(), []

    def factory():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("видеопамять занята")
        return "модель"

    try:
        store.get("reranker", factory)
    except RuntimeError:
        pass
    assert not store.ready("reranker")
    assert store.get("reranker", factory) == "модель"


def test_model_store_evict():
    store = ModelStore()
    store.get(("reranker", "a"), lambda: "реранкер a")
    store.get(("reranker", "b"), lambda: "реранкер b")
    store.get(("engine", "laws"), lambda: "индекс")
    dropped = store.evict(lambda k: k[0] == "reranker" and k != ("reranker", "b"))
    assert dropped == ["реранкер a"]
    assert not store.ready(("reranker", "a")) and store.ready(("reranker", "b")) and store.ready(("engine", "laws"))


def test_result_cache_evicts_least_recently_used():
    cache = ResultCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # «a» теперь свежее «b»
    cache.put("c", 3)
    assert cache.get("b") is None
    assert (cache.get("a"), cache.get("c")) == (1, 3)
