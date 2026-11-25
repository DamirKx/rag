"""Готовые модели эмбеддингов вместо дообученной e5-small на разговорных вопросах (корпус законов, настройки веб-демо).

Откуда вопрос. Разбор трасс Langfuse 25.11.2026 (100 независимых вопросов): у 12 вопросов из 80 правильную статью
не нашли в топ-20 ни FAISS, ни BM25 — разговорные слова далеки от языка закона; больший пул кандидатов и вес FAISS
не помогли (eval_laws_pool.py). Текущая e5-small дообучена на вопросах Adilet (язык закона) — возможно, большая
общая модель лучше понимает разговорные вопросы. Сравниваются (все — готовые модели, без дообучения):
    models/e5-small-adilet-rus (сейчас) · intfloat/multilingual-e5-small (исходная — контроль дообучения) ·
    deepvk/USER-bge-m3 · BAAI/bge-m3 · ai-forever/FRIDA · Qwen/Qwen3-Embedding-0.6B
Префиксы вопроса и фрагмента — у каждой модели свои (src/embeddings.MODEL_PREFIXES).

Одинаково для всех моделей: тексты индекса с заголовком статьи, BM25, параметры RRF текущей модели (rrf_k, веса),
пул 20 кандидатов, словарь, реранкер и порог уверенности из настроек веб-демо (data/laws/colloquial_eval.json).

Честная проверка — правило выбора задано до прогона:
- модель выбирается на ЧЁТНЫХ вопросах набора 100 (nq002, nq004, …), результат смотрится на НЕЧЁТНЫХ;
- выбор: (1) больше вопросов с правильной статьёй среди 3 лучших чанков после реранкера; (2) при равенстве — больше
  таких, что проходят порог уверенности; (3) больше статей в топ-20 самой модели (FAISS); (4) текущая модель;
- ограничение: ни один вопрос вне корпуса (20 из набора 100 и 10 старых, в том числе lq57) не проходит порог;
- старые 59 вопросов — проверка, что на них не стало хуже.

Векторы корпуса новых моделей считаются на видеокарте в половинной точности и кэшируются в data/adilet/cache
(ключ с «_fp16»); затем модель выгружается и на видеокарту загружается реранкер. Ollama и Langfuse не нужны.
Настройки веб-демо не меняются.
    python eval_laws_embedders.py        # → data/laws/embedders_eval.json
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ask import LegalSearchEngine, default_model, load_bm25_params, load_hybrid_params
from eval_laws_colloquial import CACHE_DIR, HEADING, LANG, gold_rank
from eval_laws_pool import NEW, OLD, SETTINGS, is_even
from run_adilet_eval import model_cache_name
from src.corpus import CHUNK_OVERLAP, CHUNK_SIZE, build_chunk_corpus, search_texts
from src.embeddings import EmbeddingModel
from src.laws import load_laws
from src.query_expansion import expand_query
from src.reranker import CrossEncoderReranker

CANDIDATE_MODELS = ["intfloat/multilingual-e5-small", "deepvk/USER-bge-m3", "BAAI/bge-m3", "ai-forever/FRIDA",
                    "Qwen/Qwen3-Embedding-0.6B"]
RESULT = Path("data/laws/embedders_eval.json")
POOL = 20      # кандидатов для реранкера, как в веб-демо
DEPTH = 20     # глубина выдачи самой модели (FAISS) в метриках
CONTEXT = 3    # сколько лучших чанков получает модель ответа


def summarize(rows: List[Dict[str, Any]], name: str, gate: float) -> Dict[str, Any]:
    inside = [r["models"][name] for r in rows if not r["outside"]]
    outside = [r["models"][name] for r in rows if r["outside"]]
    return {
        "questions": len(inside),
        "faiss@3": sum(c["faiss_top3"] for c in inside),
        "faiss@20": sum(c["faiss_rank"] is not None for c in inside),
        "in_candidates": sum(c["in_candidates"] for c in inside),
        "in_context": sum(c["in_context"] for c in inside),
        "in_context_passes_gate": sum(c["in_context"] and c["top_score"] >= gate for c in inside),
        "hit@1": sum(c["rank"] == 1 for c in inside),
        "outside": len(outside),
        "outside_max_score": max((c["top_score"] for c in outside), default=None),
        "outside_passes_gate": sum(c["top_score"] >= gate for c in outside),
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        sys.exit("Нужна видеокарта: векторы корпуса больших моделей на процессоре считались бы десятки минут.")
    params = json.loads(SETTINGS.read_text(encoding="utf-8"))["params"]
    gate = params["min_retrieval_score"]
    current = default_model(LANG)
    hybrid_params = load_hybrid_params(current, LANG)  # одинаковые для всех моделей
    corpus = build_chunk_corpus(load_laws(), {})
    by_id = {c["id"]: c for c in corpus}
    docs, doc_ids = search_texts(corpus, HEADING), [c["id"] for c in corpus]
    questions = {"new": json.loads(NEW.read_text(encoding="utf-8"))["questions"],
                 "old": json.loads(OLD.read_text(encoding="utf-8"))["questions"]}
    all_q = [(s, q) for s, qs in questions.items() for q in qs]
    queries = [expand_query(q["question"]) if params["dictionary"] else q["question"] for _, q in all_q]
    models = [current] + CANDIDATE_MODELS
    print(f"Корпус {len(corpus)} чанков | вопросов {len(all_q)} | RRF {hybrid_params} | пул {POOL} | "
          f"реранкер {params['reranker']} | порог {gate}")

    # 1. Для каждой модели — кандидаты гибрида и выдача FAISS (модель на видеокарте, потом выгружается)
    found: Dict[str, List[Dict[str, List[str]]]] = {}
    timing: Dict[str, Dict[str, float]] = {}
    for name in models:
        is_current = name == current
        t0 = time.perf_counter()
        # Текущая модель — векторы из кэша веб-демо (там они посчитаны в полной точности)
        embedder = EmbeddingModel(model_name=name, device="cuda", fp16=not is_current)
        engine = LegalSearchEngine(
            docs=docs, doc_ids=doc_ids, embedder=embedder, cache_dir=CACHE_DIR,
            cache_key=f"{model_cache_name(name)}_{LANG}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
                      f"{'_laws_heading' if HEADING else '_laws'}{'' if is_current else '_fp16'}",
            hybrid_params=hybrid_params, bm25_params=load_bm25_params(LANG), rerank_pool=POOL,
            rerank_docs=[c["text"] for c in corpus] if HEADING else None,
        )
        t_corpus = time.perf_counter() - t0
        t1 = time.perf_counter()
        q_emb = embedder.encode_queries(queries)
        t_queries = (time.perf_counter() - t1) / len(queries)
        found[name] = []
        for query, emb in zip(queries, q_emb):
            _, idx = engine.dense_retriever.search(emb.reshape(1, -1), k=DEPTH)
            faiss_ids = [engine.dense_retriever.get_doc_id(i) for i in idx[0] if i >= 0]
            hybrid_ids = [d for d, _ in engine.hybrid_retriever.search_with_scores(query, emb, k=POOL)]
            found[name].append({"faiss": faiss_ids, "hybrid": hybrid_ids})
        timing[name] = {"corpus_s": round(t_corpus, 1), "query_ms_gpu": round(1000 * t_queries, 1),
                        "dim": int(q_emb.shape[1])}
        print(f"  {name}: размерность {q_emb.shape[1]}, корпус {t_corpus:.0f} с, вопрос {1000 * t_queries:.1f} мс (GPU)", flush=True)
        del engine, embedder, q_emb
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Реранкер: каждый чанк оценивается один раз на вопрос, каждая модель сортирует свои кандидаты
    reranker = CrossEncoderReranker(params["reranker"], device="cuda")
    texts = {c["id"]: (c["text"] if HEADING else d) for c, d in zip(corpus, docs)}
    rows: Dict[str, List[Dict[str, Any]]] = {"new": [], "old": []}
    for i, ((set_name, q), query) in enumerate(zip(all_q, queries)):
        union = sorted({d for name in models for d in found[name][i]["hybrid"]})
        scores = dict(zip(union, reranker.score(query, [texts[d] for d in union])))
        row: Dict[str, Any] = {"id": q["id"], "outside": not q["gold"], "models": {}}
        for name in models:
            faiss_ids, cand = found[name][i]["faiss"], found[name][i]["hybrid"]
            ranked = sorted(cand, key=lambda d: -scores[d])
            has_gold = bool(q["gold"])
            row["models"][name] = {
                "faiss_rank": gold_rank(faiss_ids, q["gold"], by_id) if has_gold else None,
                "faiss_top3": has_gold and gold_rank(faiss_ids[:CONTEXT], q["gold"], by_id) is not None,
                "in_candidates": has_gold and gold_rank(cand, q["gold"], by_id) is not None,
                "rank": gold_rank(ranked, q["gold"], by_id) if has_gold else None,
                "in_context": has_gold and gold_rank(ranked[:CONTEXT], q["gold"], by_id) is not None,
                "top_score": round(scores[ranked[0]], 4),
            }
        rows[set_name].append(row)
        print(f"  {q['id']}: кандидатов для реранкера {len(union)}", flush=True)

    groups = {"new_even": [r for r in rows["new"] if is_even(r["id"])],
              "new_odd": [r for r in rows["new"] if not is_even(r["id"])],
              "new_all": rows["new"], "old": rows["old"]}
    summary = {g: {m: summarize(rs, m, gate) for m in models} for g, rs in groups.items()}
    outside_ok = {m: summary["new_all"][m]["outside_passes_gate"] == 0 and summary["old"][m]["outside_passes_gate"] == 0
                  for m in models}
    order = sorted([m for m in models if outside_ok[m]],
                   key=lambda m: (-summary["new_even"][m]["in_context"], -summary["new_even"][m]["in_context_passes_gate"],
                                  -summary["new_even"][m]["faiss@20"], models.index(m)))
    chosen = order[0] if order else None

    header = f"{'модель':34s} {'FAISS@3':>7s} {'FAISS@20':>8s} {'кандид.':>7s} {'топ-3':>5s} {'+порог':>6s} {'Hit@1':>5s}"
    for group, title in (("new_even", "Набор 100, ЧЁТНЫЕ (выбор)"), ("new_odd", "Набор 100, НЕЧЁТНЫЕ (проверка)"),
                         ("new_all", "Набор 100, все 80 вопросов с ответом"), ("old", "Старые 59 вопросов")):
        print(f"\n{title}\n{header}")
        for m in models:
            s = summary[group][m]
            mark = " ← сейчас" if m == current else (" ← выбрана" if m == chosen else "")
            print(f"{m:34s} {s['faiss@3']:>7} {s['faiss@20']:>8} {s['in_candidates']:>7} {s['in_context']:>5} "
                  f"{s['in_context_passes_gate']:>6} {s['hit@1']:>5}   вне корпуса макс {s['outside_max_score']:.3f}, "
                  f"≥ порога {s['outside_passes_gate']}{mark}{'' if outside_ok[m] else '  [вне корпуса проходит порог]'}")
    if chosen:
        a, b = summary["new_odd"][chosen], summary["new_odd"][current]
        print(f"\nВыбрана на чётных: {chosen}. Нечётные (проверка): статья среди 3 лучших {a['in_context']} против "
              f"{b['in_context']} сейчас; с проходом порога {a['in_context_passes_gate']} против {b['in_context_passes_gate']}; "
              f"FAISS@20 {a['faiss@20']} против {b['faiss@20']}. Старые 59: {summary['old'][chosen]['in_context']} против "
              f"{summary['old'][current]['in_context']}.")

    RESULT.write_text(json.dumps({
        "date": time.strftime("%Y-%m-%d %H:%M"), "current": current, "chosen_on_even": chosen,
        "reranker": params["reranker"], "gate": gate, "hybrid_params": hybrid_params, "pool": POOL,
        "rule": "чётные nq: max топ-3 после реранкера, затем топ-3 с порогом, затем FAISS@20, затем текущая; "
                "вне корпуса (оба набора) ни одного ≥ порога",
        "timing": timing, "summary": summary, "per_question": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nРезультат: {RESULT}. Настройки веб-демо не менялись.")


if __name__ == "__main__":
    main()
