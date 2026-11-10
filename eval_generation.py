"""Оценка генерации ответа и защиты от галлюцинаций (src/generation.py).

Поиск — лучшая конфигурация проекта: гибрид дообученной e5 и BM25 (RRF) по чанкам 500 символов,
пул кандидатов переранжирует дообученный bge-reranker-v2-m3, модель получает top-k чанков.

Два условия для каждого вопроса:
- answerable — обычный поиск: ответ есть в корпусе, система должна ответить;
- no_gold   — из выдачи удалены все чанки фрагмента закона, по которому составлен вопрос:
              правильное поведение — отказ. Оговорка: часть норм продублирована в других законах
              корпуса (например, в законах о банках и о рынке ценных бумаг), поэтому ответ в no_gold —
              верхняя оценка доли галлюцинаций, а не точная.

Шаги:
1. Поиск для val и test в обоих условиях (кэшируется в data/adilet/cache/).
2. Модель отвечает на подвыборку val и на test (оба условия, без порогов — чтобы сравнить
   конфигурации на одних и тех же ответах; ответы кэшируются в data/adilet/cache/llm_{model}.jsonl).
3. Пороги подбираются на val по одному правилу: каждая проверка отбраковывает не больше
   MAX_FALSE_REJECT хороших случаев (answerable, эталонный чанк в top-k; для проверок ответа —
   и модель ответила). Порог реранкера — на всём val (модель не нужна).
4. На test сравниваются конфигурации, защита добавляется по слоям.

Результат: data/adilet/generation_eval_{lang}.json (пороги подхватывает веб-демо),
ответы по вопросам — data/adilet/cache/generation_records_{lang}.jsonl.

Запуск (нужна запущенная Ollama с моделью qwen2.5:3b):
    python eval_generation.py                     # val: 200 вопросов, test: все 737
    python eval_generation.py --test-limit 50     # быстрый прогон
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ask import LegalSearchEngine, default_model, load_bm25_params, load_hybrid_params, load_legal_corpus
from run_adilet_eval import GoldChunkMatcher, model_cache_name
from src.corpus import CHUNK_OVERLAP, CHUNK_SIZE, EXPAND_WINDOW, build_chunk_corpus, expand_hits, law_titles
from src.embeddings import EmbeddingModel
from src.eval import answer_rouge_l, answer_token_f1, answer_tokens
from src.generation import (
    DEFAULT_LLM, DEFAULT_TOP_K, STATUS_ANSWERED, STATUS_NO_CONTEXT, STATUS_REFUSED, STATUS_UNVERIFIED,
    GroundedAnswerer, OllamaClient, Passage, cited_passages, strip_citations,
)
from src.nli import DEFAULT_NLI, NLI_LANGS, NLIChecker, entailment_scorer
from src.reranker import CrossEncoderReranker, default_reranker
from src.text_norm import get_normalizer

CACHE_PARQUET = "data/raw/adilet_legal_qa_kz.parquet"
CACHE_DIR = "data/adilet/cache"
# Сколько кандидатов первого этапа берётся про запас в no_gold: после удаления чанков
# эталонного фрагмента (до ~13 чанков) в пуле должно остаться rerank_pool кандидатов
NO_GOLD_EXTRA = 100
# Доля хороших случаев, которую может отбраковать каждая проверка (шаг 3)
MAX_FALSE_REJECT = 0.05
SEED = 42

# Защита добавляется по слоям; каждая следующая конфигурация включает предыдущие
CONFIGS = {
    "llm": {"gate": False, "grounding": False, "relevance": False, "nli": False, "label": "только модель (промпт)"},
    "gate": {"gate": True, "grounding": False, "relevance": False, "nli": False, "label": "+ порог реранкера"},
    "grounding": {"gate": True, "grounding": True, "relevance": False, "nli": False, "label": "+ проверка по тексту закона"},
    "full": {"gate": True, "grounding": True, "relevance": True, "nli": False, "label": "+ проверка соответствия вопросу"},
    "nli": {"gate": True, "grounding": True, "relevance": True, "nli": True, "label": "+ следование из фрагмента (NLI)"},
}


def rerank_pool(lang: str, reranker: str) -> int:
    path = Path(f"data/adilet/reranker_eval_{lang}.json")
    if path.exists():
        tuned = json.loads(path.read_text(encoding="utf-8"))["rerankers"].get(Path(reranker).as_posix())
        if tuned:
            return tuned["best_pool"]
    return 30


def retrieve(engine: LegalSearchEngine, by_id: Dict[str, Dict], query: str, pool: int, top_k: int,
             exclude_ctx: Optional[str] = None, window: int = EXPAND_WINDOW) -> List[Passage]:
    """Гибрид → пул кандидатов (без чанков фрагмента exclude_ctx) → реранкер → top_k чанков с окружающим текстом.

    Текст берётся только из того же фрагмента закона, поэтому в no_gold эталонный фрагмент не возвращается.
    """
    hits = engine.search(query, mode="hybrid", top_k=pool + (NO_GOLD_EXTRA if exclude_ctx else 0))
    if exclude_ctx:
        hits = [h for h in hits if h[0].split("_fix_")[0] != exclude_ctx]
    reranked = engine.rerank(query, hits[:pool], top_k)
    return [Passage.from_hit(h) for h in expand_hits([{**by_id[doc_id], "score": score} for doc_id, _, score in reranked], window)]


def passage_dict(p: Passage) -> Dict[str, Any]:
    return {"id": p.id, "chunk_ids": p.ids, "text": p.text, "law": p.law, "articles": p.articles, "score": p.score}


def passages_from(rows: List[Dict[str, Any]]) -> List[Passage]:
    return [Passage(id=p["id"], chunk_ids=p["chunk_ids"], text=p["text"], law=p["law"], articles=p["articles"], score=p["score"]) for p in rows]


def threshold_for(good: List[float], max_false_reject: float = MAX_FALSE_REJECT) -> float:
    """Наибольший порог, ниже которого оказывается не больше max_false_reject хороших значений."""
    values = np.sort(np.asarray(good, dtype=float))
    # Порог = значение с индексом floor(n·α): строго ниже него не больше n·α значений
    idx = int(np.floor(len(values) * max_false_reject))
    return float(np.floor(values[min(idx, len(values) - 1)] * 1000) / 1000)


def reject_curve(good: List[float], bad: List[float], thresholds) -> List[Dict[str, float]]:
    good, bad = np.asarray(good, dtype=float), np.asarray(bad, dtype=float)
    return [{"threshold": round(float(t), 3),
             "reject_good": round(float((good < t).mean()), 4) if len(good) else None,
             "reject_bad": round(float((bad < t).mean()), 4) if len(bad) else None} for t in thresholds]


def make_answerer(cfg: Dict[str, Any], params: Dict[str, float], lang: str) -> GroundedAnswerer:
    return GroundedAnswerer(
        None, lang,
        min_retrieval_score=params["min_retrieval_score"] if cfg["gate"] else None,
        min_sentence_support=params["min_sentence_support"] if cfg["grounding"] else 0.0,
        min_answer_relevance=params["min_answer_relevance"] if cfg["relevance"] else None,
        min_entailment=params["min_entailment"] if cfg["nli"] else None,
        verify=cfg["grounding"] or cfg["relevance"] or cfg["nli"],
    )


def final_status(rec: Dict[str, Any], answerer: GroundedAnswerer) -> str:
    """Статус ответа при заданной конфигурации защиты (по сохранённому ответу модели)."""
    passages = passages_from(rec["passages"])
    if not answerer.retrieval_confident(passages):
        return STATUS_NO_CONTEXT
    checks = rec["checks"]
    return answerer.judge(rec["query"], rec["raw"], passages, relevance=checks.get("answer_relevance"), entailment=checks.get("entailment")).status


def ungrounded(rec: Dict[str, Any], params: Dict[str, float]) -> bool:
    """Ответ содержит числа или утверждения, которых нет во фрагментах (по порогам full)."""
    checks = rec["checks"]
    return bool(checks.get("unsupported_numbers")) or checks.get("min_sentence_support", 1.0) < params["min_sentence_support"]


def summarize(records: List[Dict[str, Any]], params: Dict[str, float], lang: str) -> Dict[str, Any]:
    """Метрики каждой конфигурации в обоих условиях (доли — от числа вопросов)."""
    normalize = get_normalizer("snowball" if lang == "rus" else "prefix5")
    out: Dict[str, Any] = {}
    for name, cfg in CONFIGS.items():
        answerer = make_answerer(cfg, params, lang)
        out[name] = {"label": cfg["label"]}
        for cond in ("answerable", "no_gold"):
            recs = [r for r in records if r["condition"] == cond]
            statuses = [final_status(r, answerer) for r in recs]
            answered = [r for r, s in zip(recs, statuses) if s == STATUS_ANSWERED]
            n = len(recs)
            m = {
                "n": n,
                "answered": len(answered) / n,
                "refused": sum(s in (STATUS_NO_CONTEXT, STATUS_REFUSED) for s in statuses) / n,
                "unverified": statuses.count(STATUS_UNVERIFIED) / n,
                # Выданные ответы с числами или утверждениями, которых нет во фрагментах
                "answered_ungrounded": sum(ungrounded(r, params) for r in answered) / n,
            }
            if cond == "answerable":
                f1, rl = [], []
                for r in answered:
                    pred, ref = answer_tokens(r["raw"], normalize), answer_tokens(r["reference"], normalize)
                    f1.append(answer_token_f1(pred, ref))
                    rl.append(answer_rouge_l(pred, ref))
                m["token_f1"] = float(np.mean(f1)) if f1 else 0.0
                m["rouge_l"] = float(np.mean(rl)) if rl else 0.0
                # Ответ опирается на эталонный чанк: среди цитируемых фрагментов есть эталонный
                m["answered_from_gold"] = sum(
                    any(i in r["gold"] for c in (r["cited"] or [1]) if 1 <= c <= len(r["passages"]) for i in r["passages"][c - 1]["chunk_ids"])
                    for r in answered
                ) / n
                # Нужный фрагмент был найден, но система не ответила
                m["missed_with_gold"] = sum(
                    r["gold_in_top"] and s != STATUS_ANSWERED for r, s in zip(recs, statuses)
                ) / n
            out[name][cond] = {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
    return out


def main():
    parser = argparse.ArgumentParser(description="Оценка генерации ответа и защиты от галлюцинаций.")
    parser.add_argument("--lang", choices=["rus", "kaz"], default="rus")
    parser.add_argument("--llm", default=DEFAULT_LLM, help="Модель Ollama.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Сколько чанков получает модель.")
    parser.add_argument("--val-limit", type=int, default=200, help="Вопросов val для подбора порогов проверки ответа (0 — все).")
    parser.add_argument("--test-limit", type=int, default=0, help="Вопросов test (0 — все).")
    parser.add_argument("--expand-window", type=int, default=EXPAND_WINDOW,
                        help="Сколько символов вокруг чанка получает модель (0 — только чанк).")
    parser.add_argument("--no-nli", action="store_true", help="Без проверки следования из фрагмента (NLI на CPU, самый долгий шаг).")
    args = parser.parse_args()
    lang = args.lang
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Корпус, поиск, эталонные чанки ---
    titles = law_titles(pd.read_parquet(CACHE_PARQUET, columns=["context", "source_id", "doc_type", "language"]))
    corpus = build_chunk_corpus(load_legal_corpus(CACHE_PARQUET, lang), titles)
    by_id = {c["id"]: c for c in corpus}
    model_name, reranker_name = default_model(lang), default_reranker(lang)
    engine = LegalSearchEngine(
        docs=[c["text"] for c in corpus], doc_ids=[c["id"] for c in corpus],
        embedder=EmbeddingModel(model_name=model_name, device=device), cache_dir=CACHE_DIR,
        cache_key=f"{model_cache_name(model_name)}_{lang}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
        hybrid_params=load_hybrid_params(model_name, lang), bm25_params=load_bm25_params(lang),
        reranker=CrossEncoderReranker(reranker_name, device=device),
    )
    pool = rerank_pool(lang, reranker_name)
    matcher = GoldChunkMatcher([c["text"] for c in corpus], [c["id"].split("_fix_")[0] for c in corpus])
    chunk_ids = [c["id"] for c in corpus]
    print(f"Корпус: {len(corpus)} чанков | эмбеддер {model_name} | реранкер {reranker_name}, пул {pool} | "
          f"top-k {args.top_k} | окно вокруг чанка {args.expand_window}")
    # NLI считается на CPU: видеопамять занята реранкером и языковой моделью
    nli = entailment_scorer(NLIChecker()) if lang in NLI_LANGS and not args.no_nli else None

    splits = {}
    for name in ("val", "test"):
        items = json.loads(Path(f"data/adilet/splits/qa_{lang}_{name}.json").read_text(encoding="utf-8"))
        for q in items:
            q["gold"] = [chunk_ids[i] for i in matcher.find(q)[0]]
        splits[name] = items

    # --- 1. Поиск в обоих условиях (с кэшем) ---
    key_src = json.dumps([model_cache_name(model_name), model_cache_name(reranker_name), pool, args.top_k,
                          NO_GOLD_EXTRA, args.expand_window, load_hybrid_params(model_name, lang), load_bm25_params(lang),
                          {n: [q["id"] for q in items] for n, items in splits.items()}], sort_keys=True)
    retrieval_cache = Path(f"{CACHE_DIR}/generation_retrieval_{lang}_{hashlib.sha1(key_src.encode()).hexdigest()[:12]}.json")
    if retrieval_cache.exists():
        retrieval = json.loads(retrieval_cache.read_text(encoding="utf-8"))
        print(f"Поиск: загружен из кэша {retrieval_cache}")
    else:
        retrieval = {}
        for name, items in splits.items():
            t0 = time.time()
            rows = []
            for q in items:
                for cond in ("answerable", "no_gold"):
                    passages = retrieve(engine, by_id, q["query"], pool, args.top_k,
                                        exclude_ctx=q["context_id"] if cond == "no_gold" else None, window=args.expand_window)
                    rows.append({"id": q["id"], "condition": cond, "passages": [passage_dict(p) for p in passages],
                                 "gold_in_top": any(i in q["gold"] for p in passages for i in p.ids)})
            retrieval[name] = rows
            print(f"Поиск {name}: {len(items)} вопросов × 2 условия за {time.time() - t0:.0f} с", flush=True)
        retrieval_cache.write_text(json.dumps(retrieval, ensure_ascii=False), encoding="utf-8")

    # --- 2. Ответы модели (без порогов) и их оценки соответствия вопросу ---
    client = OllamaClient(model=args.llm, retries=3, cache_path=f"{CACHE_DIR}/llm_{args.llm.replace(':', '_').replace('/', '_')}.jsonl")
    plain = GroundedAnswerer(client, lang, min_retrieval_score=None, min_answer_relevance=None, verify=False)
    rng = random.Random(SEED)
    limits = {"val": args.val_limit, "test": args.test_limit}
    records: Dict[str, List[Dict[str, Any]]] = {}
    for name, items in splits.items():
        chosen = items if not limits[name] or limits[name] >= len(items) else rng.sample(items, limits[name])
        by_q = {(r["id"], r["condition"]): r for r in retrieval[name]}
        recs, t0, calls = [], time.time(), 0
        for i, q in enumerate(chosen, 1):
            for cond in ("answerable", "no_gold"):
                row = by_q[(q["id"], cond)]
                passages = passages_from(row["passages"])
                response = plain.ask_model(q["query"], passages)
                calls += not response["cached"]
                judged = plain.judge(q["query"], response["text"], passages)
                if judged.status != STATUS_REFUSED:
                    judged.checks["answer_relevance"] = round(
                        float(engine.reranker.score(q["query"], [strip_citations(response["text"])])[0]), 4)
                recs.append({
                    "split": name, "id": q["id"], "condition": cond, "query": q["query"],
                    "reference": q["full_response"], "gold": q["gold"], "gold_in_top": row["gold_in_top"],
                    "passages": row["passages"], "raw": response["text"], "cited": judged.cited,
                    "checks": judged.checks, "refused": judged.status == STATUS_REFUSED,
                    "retried": response["retried"],
                    "latency_s": response["latency_s"], "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                })
            if i % 25 == 0 or i == len(chosen):
                print(f"  {name}: {i}/{len(chosen)} вопросов, новых вызовов модели {calls}, {time.time() - t0:.0f} с", flush=True)
        records[name] = recs

    # --- 3. Пороги на val: каждая проверка отбраковывает ≤ MAX_FALSE_REJECT хороших случаев ---
    val_rows = retrieval["val"]
    gate_good = [r["passages"][0]["score"] for r in val_rows if r["condition"] == "answerable" and r["gold_in_top"]]
    gate_bad = [r["passages"][0]["score"] for r in val_rows if r["condition"] == "no_gold"]
    good = [r for r in records["val"] if r["condition"] == "answerable" and r["gold_in_top"] and not r["refused"]]
    bad = [r for r in records["val"] if r["condition"] == "no_gold" and not r["refused"]]
    support_good = [r["checks"]["min_sentence_support"] for r in good]
    support_bad = [r["checks"]["min_sentence_support"] for r in bad]
    rel_good = [r["checks"]["answer_relevance"] for r in good]
    rel_bad = [r["checks"]["answer_relevance"] for r in bad]
    params = {
        "min_retrieval_score": threshold_for(gate_good),
        "min_sentence_support": threshold_for(support_good),
        "min_answer_relevance": threshold_for(rel_good),
    }

    # NLI идёт на CPU и медленный, поэтому считается только для ответов, прошедших все предыдущие слои:
    # остальные эта проверка не меняет. Порог подбирается на таких же ответах, какие она получит в работе
    if nli is not None:
        reach, t0, scored = make_answerer(CONFIGS["full"], params, lang), time.time(), 0
        for recs in records.values():
            for r in recs:
                if final_status(r, reach) == STATUS_ANSWERED:
                    r["checks"]["entailment"] = round(nli(r["query"], r["raw"], cited_passages(r["raw"], passages_from(r["passages"]))), 4)
                    scored += 1
                    if scored % 50 == 0:
                        print(f"  NLI: {scored} ответов, {time.time() - t0:.0f} с", flush=True)
        print(f"NLI: {scored} ответов за {time.time() - t0:.0f} с", flush=True)
    entail_good = [r["checks"]["entailment"] for r in good if r["checks"].get("entailment") is not None]
    entail_bad = [r["checks"]["entailment"] for r in bad if r["checks"].get("entailment") is not None]
    params["min_entailment"] = threshold_for(entail_good) if entail_good else None

    grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    tuning = {
        "rule": f"наибольший порог, при котором отбраковывается не больше {MAX_FALSE_REJECT:.0%} хороших случаев val",
        "min_retrieval_score": {"n_good": len(gate_good), "n_bad": len(gate_bad), "curve": reject_curve(gate_good, gate_bad, grid)},
        "min_sentence_support": {"n_good": len(good), "n_bad": len(bad), "curve": reject_curve(support_good, support_bad, grid)},
        "min_answer_relevance": {"n_good": len(good), "n_bad": len(bad), "curve": reject_curve(rel_good, rel_bad, grid)},
        "min_entailment": {"n_good": len(entail_good), "n_bad": len(entail_bad), "curve": reject_curve(entail_good, entail_bad, grid)},
    }
    for key, value in params.items():
        print(f"Порог {key}: {value}")

    # --- 4. Сравнение конфигураций ---
    report = {
        "params": {**params, "reranker": Path(reranker_name).as_posix(), "embedder": Path(model_name).as_posix(),
                   "llm": args.llm, "top_k": args.top_k, "rerank_pool": pool, "expand_window": args.expand_window,
                   "nli": DEFAULT_NLI if nli is not None else None},
        "tuning_val": tuning,
        "retrieval": {
            name: {"gold_in_top_answerable": round(float(np.mean([r["gold_in_top"] for r in rows if r["condition"] == "answerable"])), 4)}
            for name, rows in retrieval.items()
        },
    }
    for name in ("val", "test"):
        recs = records[name]
        report[name] = {
            "n_questions": len(recs) // 2,
            "raw_model": {
                cond: {
                    "refused": round(float(np.mean([r["refused"] for r in recs if r["condition"] == cond])), 4),
                    "with_unsupported_numbers": round(float(np.mean([
                        bool(r["checks"].get("unsupported_numbers")) for r in recs if r["condition"] == cond])), 4),
                } for cond in ("answerable", "no_gold")
            },
            "retried": round(float(np.mean([r["retried"] for r in recs])), 4),
            "latency_s_mean": round(float(np.mean([r["latency_s"] for r in recs])), 2),
            "completion_tokens_mean": round(float(np.mean([r["completion_tokens"] or 0 for r in recs])), 1),
            "configs": summarize(recs, params, lang),
        }

    out = Path(f"data/adilet/generation_eval_{lang}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    records_path = Path(f"{CACHE_DIR}/generation_records_{lang}.jsonl")
    with records_path.open("w", encoding="utf-8") as f:
        for name in ("val", "test"):
            for r in records[name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nTest ({report['test']['n_questions']} вопросов), доли от числа вопросов:")
    print(f"{'конфигурация':34s} | {'answerable: ответ':>17s} {'F1':>6s} {'ROUGE-L':>7s} {'из эталона':>10s} | {'no_gold: ответ':>14s} {'без опоры':>9s}")
    for m in report["test"]["configs"].values():
        a, n = m["answerable"], m["no_gold"]
        print(f"{m['label']:34s} | {a['answered']:17.1%} {a['token_f1']:6.3f} {a['rouge_l']:7.3f} {a['answered_from_gold']:10.1%} | "
              f"{n['answered']:14.1%} {n['answered_ungrounded']:9.1%}")
    print(f"\nСохранено: {out}, {records_path}")


if __name__ == "__main__":
    main()
