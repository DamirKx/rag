"""Подбор параметров гибридного поиска (RRF: Dense + BM25) на validation.

Использует разбиение train/val/test из notebooks/finetune_e5.ipynb (data/adilet/splits)
и те же чанки и эталоны, что и ноутбук. Параметры выбираются по MRR@10 на val,
test считается один раз для итогового сравнения. Результат сохраняется в
data/adilet/hybrid_tuning_{lang}.json и подхватывается веб-демо (app.py).

Запуск:
    python tune_hybrid.py --lang rus
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.embeddings import EmbeddingModel
from src.retriever import VectorRetriever, BM25Retriever
from src.eval import calculate_mrr, calculate_hit_rate_at_k, calculate_recall_at_k
from prepare_adilet_data import clean_adilet_context
from run_adilet_eval import DEFAULT_MODEL, GoldChunkMatcher, build_fixed_chunks, get_or_compute_embeddings, model_cache_name

CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
EVAL_K = 10
# Как в HybridRetriever: пул кандидатов каждого метода = max(k * 4, 20)
POOL = max(EVAL_K * 4, 20)
DENSE_WEIGHTS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RRF_KS = [5, 10, 20, 30, 60, 100]
DEFAULT_PARAMS = {"dense_weight": 0.5, "rrf_k": 60}


def ir_metrics(gold: List[List[str]], pred: List[List[str]]) -> Dict[str, float]:
    return {
        "MRR@10": calculate_mrr(gold, pred),
        "Hit@1": calculate_hit_rate_at_k(gold, pred, k=1),
        "Hit@3": calculate_hit_rate_at_k(gold, pred, k=3),
        "Hit@10": calculate_hit_rate_at_k(gold, pred, k=10),
        "Recall@10": calculate_recall_at_k(gold, pred, k=10),
    }


def rrf_fuse(dense: List[List[str]], sparse: List[List[str]], dense_weight: float, rrf_k: int, k: int) -> List[List[str]]:
    """То же слияние, что HybridRetriever.search_with_scores, по заранее найденным спискам."""
    fused = []
    for d_ids, s_ids in zip(dense, sparse):
        scores: Dict[str, float] = {}
        for rank, doc_id in enumerate(d_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + dense_weight / (rrf_k + rank)
        for rank, doc_id in enumerate(s_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1 - dense_weight) / (rrf_k + rank)
        fused.append(sorted(scores, key=scores.get, reverse=True)[:k])
    return fused


def main():
    parser = argparse.ArgumentParser(description="Подбор весов гибридного поиска на validation.")
    parser.add_argument("--lang", choices=["rus", "kaz"], default="rus")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Модели для подбора (по умолчанию базовая и models/e5-small-adilet-{lang}).")
    parser.add_argument("--cache-dir", default="data/adilet/cache")
    args = parser.parse_args()

    models = args.models or [DEFAULT_MODEL] + [p for p in [f"models/e5-small-adilet-{args.lang}"] if Path(p).exists()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet("data/raw/adilet_legal_qa_kz.parquet")
    sub_df = df[df["language"] == args.lang].copy().reset_index(drop=True)
    sub_df["cleaned_context"] = sub_df["context"].apply(clean_adilet_context)
    sub_df["context_id"] = [f"ctx_{i:04d}" for i in range(len(sub_df))]
    chunks, chunk_ids, chunk_ctx = build_fixed_chunks(sub_df, CHUNK_SIZE, CHUNK_OVERLAP)

    matcher = GoldChunkMatcher(chunks, chunk_ctx)
    splits, gold = {}, {}
    for name in ("val", "test"):
        path = Path(f"data/adilet/splits/qa_{args.lang}_{name}.json")
        if not path.exists():
            raise FileNotFoundError(f"{path} не найден — сначала выполните разбиение в notebooks/finetune_e5.ipynb")
        splits[name] = json.loads(path.read_text(encoding="utf-8"))
        gold[name] = [[chunk_ids[i] for i in matcher.find(q)[0]] for q in splits[name]]
    print(f"Чанков: {len(chunks)} | val: {len(splits['val'])} | test: {len(splits['test'])}")

    bm25 = BM25Retriever()
    bm25.add_documents(chunks, chunk_ids)
    t0 = time.time()
    sparse = {name: bm25.batch_search([q["query"] for q in items], k=POOL) for name, items in splits.items()}
    print(f"BM25: {time.time() - t0:.1f} сек.")

    report = {
        "language": args.lang,
        "selection_metric": "val MRR@10",
        "grid": {"dense_weight": DENSE_WEIGHTS, "rrf_k": RRF_KS},
        "pool": POOL,
        "bm25": {name: ir_metrics(gold[name], [p[:EVAL_K] for p in sparse[name]]) for name in splits},
        "models": {},
    }

    for model_name in models:
        print(f"\n=== {model_name}")
        embedder = EmbeddingModel(model_name, device=device)
        chunk_emb = get_or_compute_embeddings(
            embedder, chunks, args.cache_dir,
            cache_key=f"{model_cache_name(model_name)}_{args.lang}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
        )
        index = VectorRetriever(embedding_dim=chunk_emb.shape[1])
        index.add_documents(chunk_emb, chunks, chunk_ids)
        dense = {}
        for name, items in splits.items():
            _, idx = index.search(embedder.encode_queries([q["query"] for q in items]), k=POOL)
            dense[name] = [[chunk_ids[i] for i in row if i >= 0] for row in idx]

        grid = []
        for w in DENSE_WEIGHTS:
            for rrf_k in RRF_KS:
                pred = rrf_fuse(dense["val"], sparse["val"], w, rrf_k, EVAL_K)
                grid.append({"dense_weight": w, "rrf_k": rrf_k, "val_MRR@10": calculate_mrr(gold["val"], pred)})
        best = max(grid, key=lambda g: g["val_MRR@10"])
        best_params = {"dense_weight": best["dense_weight"], "rrf_k": best["rrf_k"]}

        def hybrid(params, name):
            return ir_metrics(gold[name], rrf_fuse(dense[name], sparse[name], params["dense_weight"], params["rrf_k"], EVAL_K))

        result = {
            **best_params,
            "sparse_weight": round(1 - best_params["dense_weight"], 2),
            "dense": {name: ir_metrics(gold[name], [p[:EVAL_K] for p in dense[name]]) for name in splits},
            "hybrid_default": {name: hybrid(DEFAULT_PARAMS, name) for name in splits},
            "hybrid_tuned": {name: hybrid(best_params, name) for name in splits},
            "grid": grid,
        }
        report["models"][Path(model_name).as_posix()] = result

        rows = {
            "BM25": report["bm25"],
            "Dense": result["dense"],
            "Hybrid (0.5 / 60)": result["hybrid_default"],
            f"Hybrid ({best_params['dense_weight']} / {best_params['rrf_k']})": result["hybrid_tuned"],
        }
        for name in ("val", "test"):
            table = pd.DataFrame({label: r[name] for label, r in rows.items()}).T
            print(f"\n{name}:\n{table.round(4).to_string()}")

    out = Path(f"data/adilet/hybrid_tuning_{args.lang}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт сохранён: {out}")


if __name__ == "__main__":
    main()
