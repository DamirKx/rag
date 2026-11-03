"""Подбор параметров BM25 (нормализация словоформ, вариант idf, k1, b) на validation.

Те же чанки (500 символов) и эталоны, что в notebooks/finetune_e5.ipynb и tune_hybrid.py.
Параметры выбираются по MRR@10 на val; test считается для итоговой таблицы и в выборе
не участвует. Результат сохраняется в data/adilet/bm25_tuning_{lang}.json и подхватывается
ask.py, веб-демо, tune_hybrid.py и eval_reranker.py (load_bm25_params в ask.py).

Запуск:
    python tune_bm25.py --lang rus
После него нужно заново подобрать веса гибрида: python tune_hybrid.py --lang rus
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.retriever import BM25Retriever
from tune_hybrid import EVAL_K, ir_metrics, load_eval_data

# Стеммер Snowball и лемматизатор pymorphy3 — только для русского
NORMALIZERS = {
    "rus": ["none", "prefix4", "prefix5", "prefix6", "prefix7", "snowball", "lemma"],
    "kaz": ["none", "prefix4", "prefix5", "prefix6", "prefix7"],
}
IDF_VARIANTS = ["okapi", "lucene"]
K1_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
B_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9]
DEFAULT_PARAMS = {"normalizer": "none", "idf": "okapi", "k1": 1.5, "b": 0.75}   # как rank_bm25 раньше


def main():
    parser = argparse.ArgumentParser(description="Подбор параметров BM25 на validation.")
    parser.add_argument("--lang", choices=["rus", "kaz"], default="rus")
    args = parser.parse_args()

    chunks, chunk_ids, splits, gold = load_eval_data(args.lang)
    queries = {name: [q["query"] for q in items] for name, items in splits.items()}

    def evaluate(bm25: BM25Retriever, split: str):
        return ir_metrics(gold[split], bm25.batch_search(queries[split], k=EVAL_K))

    report = {
        "language": args.lang,
        "selection_metric": "val MRR@10",
        "grid": {"normalizer": NORMALIZERS[args.lang], "idf": IDF_VARIANTS, "k1": K1_GRID, "b": B_GRID},
        "default": None,
        "by_normalizer": {},
        "val_mrr_grid": {},
    }

    t_start = time.time()
    best_overall = None
    for normalizer in NORMALIZERS[args.lang]:
        t0 = time.time()
        bm25 = BM25Retriever(normalizer=normalizer)
        bm25.add_documents(chunks, chunk_ids)   # токенизация один раз, дальше только пересчёт весов
        t_index = time.time() - t0

        if normalizer == DEFAULT_PARAMS["normalizer"]:
            bm25.set_params(k1=DEFAULT_PARAMS["k1"], b=DEFAULT_PARAMS["b"], idf=DEFAULT_PARAMS["idf"])
            report["default"] = {"params": DEFAULT_PARAMS, "val": evaluate(bm25, "val"), "test": evaluate(bm25, "test")}

        best = None
        for idf in IDF_VARIANTS:
            grid = np.zeros((len(K1_GRID), len(B_GRID)))
            for i, k1 in enumerate(K1_GRID):
                for j, b in enumerate(B_GRID):
                    bm25.set_params(k1=k1, b=b, idf=idf)
                    grid[i, j] = evaluate(bm25, "val")["MRR@10"]
                    if best is None or grid[i, j] > best[0]:
                        best = (grid[i, j], {"normalizer": normalizer, "idf": idf, "k1": k1, "b": b})
            report["val_mrr_grid"][f"{normalizer}/{idf}"] = grid.round(4).tolist()

        params = best[1]
        bm25.set_params(k1=params["k1"], b=params["b"], idf=params["idf"])
        row = {"params": params, "val": evaluate(bm25, "val"), "test": evaluate(bm25, "test"),
               "index_sec": round(t_index, 1)}
        report["by_normalizer"][normalizer] = row
        if best_overall is None or row["val"]["MRR@10"] > best_overall["val"]["MRR@10"]:
            best_overall = row
        print(f"{normalizer:<9} лучшее на val: idf={params['idf']}, k1={params['k1']}, b={params['b']} | "
              f"val MRR@10 {row['val']['MRR@10']:.4f} | индекс {t_index:.1f} с")

    report["best"] = best_overall["params"]
    report["best_metrics"] = {"val": best_overall["val"], "test": best_overall["test"]}

    rows = {"rank_bm25 по умолчанию (k1=1.5, b=0.75)": report["default"]}
    rows.update({f"{n} ({r['params']['idf']}, k1={r['params']['k1']}, b={r['params']['b']})": r
                 for n, r in report["by_normalizer"].items()})
    for split in ("val", "test"):
        table = pd.DataFrame({label: r[split] for label, r in rows.items()}).T
        print(f"\n{split}:\n{table.round(4).to_string()}")
    print(f"\nВыбрано по val: {report['best']} (перебор {time.time() - t_start:.0f} с)")

    out = Path(f"data/adilet/bm25_tuning_{args.lang}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт сохранён: {out}\nДальше: python tune_hybrid.py --lang {args.lang}")


if __name__ == "__main__":
    main()
