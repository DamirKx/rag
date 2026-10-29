"""Оценка переранжирования cross-encoder поверх гибридного поиска на validation / test.

Первый этап — Hybrid (RRF) с дообученной моделью эмбеддингов (если есть) и весами из
data/adilet/hybrid_tuning_{lang}.json. Каждый реранкер один раз оценивает топ-MAX_POOL
кандидатов первого этапа, затем для каждого размера пула N переранжируются топ-N, а
остальные кандидаты идут следом в исходном порядке. N выбирается по MRR@10 на val,
test считается для итогового сравнения. Результат сохраняется в
data/adilet/reranker_eval_{lang}.json и подхватывается веб-демо (app.py).

Запуск:
    python eval_reranker.py --lang rus
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ask import default_model, load_hybrid_params
from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever, BM25Retriever
from src.reranker import CrossEncoderReranker, DEFAULT_RERANKER
from run_adilet_eval import get_or_compute_embeddings, model_cache_name
from tune_hybrid import CHUNK_SIZE, CHUNK_OVERLAP, EVAL_K, ir_metrics, load_eval_data, rrf_fuse

RERANKERS = [DEFAULT_RERANKER, "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"]
POOL_SIZES = [10, 20, 30, 50]
MAX_POOL = max(POOL_SIZES)


def apply_rerank(candidates: List[List[str]], scores: List[Dict[str, float]], pool: int) -> List[List[str]]:
    """Переупорядочивает топ-pool каждого списка по оценкам реранкера, хвост оставляет как есть."""
    return [
        sorted(ids[:pool], key=lambda d: -s[d]) + ids[pool:]
        for ids, s in zip(candidates, scores)
    ]


def main():
    parser = argparse.ArgumentParser(description="Оценка реранкеров cross-encoder на validation / test.")
    parser.add_argument("--lang", choices=["rus", "kaz"], default="rus")
    parser.add_argument("--model", default=None, help="Модель эмбеддингов первого этапа (по умолчанию как в ask.py).")
    parser.add_argument("--rerankers", nargs="+", default=RERANKERS)
    parser.add_argument("--cache-dir", default="data/adilet/cache")
    args = parser.parse_args()

    model_name = args.model or default_model(args.lang)
    hybrid_params = load_hybrid_params(model_name, args.lang)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Первый этап: Hybrid, {model_name}, {hybrid_params} | устройство: {device}")

    chunks, chunk_ids, splits, gold = load_eval_data(args.lang)
    text_by_id = dict(zip(chunk_ids, chunks))
    queries = {name: [q["query"] for q in items] for name, items in splits.items()}

    # Первый этап: те же пулы dense и BM25, что для RRF в HybridRetriever при k = MAX_POOL
    first_pool = max(MAX_POOL * 4, 20)
    bm25 = BM25Retriever()
    bm25.add_documents(chunks, chunk_ids)
    embedder = EmbeddingModel(model_name, device=device)
    chunk_emb = get_or_compute_embeddings(
        embedder, chunks, args.cache_dir,
        cache_key=f"{model_cache_name(model_name)}_{args.lang}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
    )
    index = VectorRetriever(embedding_dim=chunk_emb.shape[1])
    index.add_documents(chunk_emb, chunks, chunk_ids)

    candidates = {}
    for name in splits:
        _, idx = index.search(embedder.encode_queries(queries[name]), k=first_pool)
        dense = [[chunk_ids[i] for i in row if i >= 0] for row in idx]
        sparse = bm25.batch_search(queries[name], k=first_pool)
        candidates[name] = rrf_fuse(dense, sparse, hybrid_params["dense_weight"], hybrid_params["rrf_k"], MAX_POOL)
    del embedder
    if device == "cuda":
        torch.cuda.empty_cache()

    report = {
        "language": args.lang,
        "first_stage": {"method": "hybrid", "model_name": Path(model_name).as_posix(), **hybrid_params},
        "selection_metric": "val MRR@10",
        "pool_sizes": POOL_SIZES,
        "hybrid": {name: ir_metrics(gold[name], [c[:EVAL_K] for c in candidates[name]]) for name in splits},
        "rerankers": {},
    }
    # Доля вопросов, у которых эталонный чанк вообще есть в пуле: потолок для реранкера
    ceiling = {
        name: {f"Hit@{n}": sum(any(d in set(g) for d in c[:n]) for c, g in zip(candidates[name], gold[name])) / len(gold[name])
               for n in POOL_SIZES}
        for name in splits
    }
    report["pool_recall"] = ceiling
    print(f"Эталон в пуле кандидатов: {ceiling}")

    for reranker_name in args.rerankers:
        print(f"\n=== {reranker_name}")
        reranker = CrossEncoderReranker(reranker_name, device=device)
        scores, sec_per_query = {}, {}
        for name in splits:
            t0 = time.time()
            scores[name] = [
                dict(reranker.rerank(q, [(d, text_by_id[d]) for d in cands]))
                for q, cands in zip(queries[name], candidates[name])
            ]
            sec_per_query[name] = (time.time() - t0) / len(queries[name])
            print(f"  {name}: {sec_per_query[name] * 1000:.0f} мс на вопрос (пул {MAX_POOL})")

        by_pool = {
            pool: {name: ir_metrics(gold[name], [p[:EVAL_K] for p in apply_rerank(candidates[name], scores[name], pool)])
                   for name in splits}
            for pool in POOL_SIZES
        }
        best_pool = max(POOL_SIZES, key=lambda pool: by_pool[pool]["val"]["MRR@10"])
        report["rerankers"][reranker_name] = {
            "best_pool": best_pool,
            "ms_per_query_at_max_pool": round(sec_per_query["test"] * 1000, 1),
            "by_pool": {str(pool): m for pool, m in by_pool.items()},
        }

        rows = {"Hybrid": report["hybrid"], **{f"+ rerank@{pool}": by_pool[pool] for pool in POOL_SIZES}}
        for name in splits:
            table = pd.DataFrame({label: r[name] for label, r in rows.items()}).T
            print(f"\n{name}:\n{table.round(4).to_string()}")
        print(f"Лучший пул по val MRR@10: {best_pool}")

        del reranker
        if device == "cuda":
            torch.cuda.empty_cache()

    out = Path(f"data/adilet/reranker_eval_{args.lang}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт сохранён: {out}")


if __name__ == "__main__":
    main()
