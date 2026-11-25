"""Пул кандидатов реранкера и вес FAISS в RRF на разговорных вопросах (корпус законов, настройки веб-демо).

Откуда вопрос. Разбор трасс Langfuse 25.11.2026 (100 независимых вопросов, eval_laws_colloquial.py --langfuse):
правильная статья не попала в 20 кандидатов реранкера у 17 вопросов из 80; у 5 из них FAISS держал её в топ-20,
а слияние RRF выбросило из-за BM25. На этих вопросах FAISS нашёл статью в топ-20 у 65 вопросов, BM25 — у 47,
только BM25 — у 2. Проверяется, помогут ли больший пул кандидатов и больший вес FAISS:
    пул 20 / 40 / 80  ×  вес FAISS 0,5 / 0,65 / 0,8 / 1,0 (вес BM25 = 1 − вес FAISS; 1,0 — только FAISS).

Честная проверка — правило выбора задано до прогона:
- настройка выбирается на ЧЁТНЫХ вопросах набора 100 (nq002, nq004, …), результат смотрится на НЕЧЁТНЫХ;
- выбор: (1) больше вопросов с правильной статьёй среди 3 лучших чанков после реранкера (их получает модель);
  (2) при равенстве — больше таких, у которых оценка лучшего чанка проходит порог уверенности веб-демо;
  (3) меньший пул (быстрее); (4) вес FAISS ближе к текущему;
- ограничение: ни один вопрос вне корпуса (20 из набора 100 и 10 из старого набора, в том числе lq57 с оценкой
  0,557) не набирает оценку ≥ порога уверенности — иначе модель получила бы шанс ответить не по закону;
- старые 59 вопросов (data/laws/colloquial_questions.json) — проверка, что на них не стало хуже.

Словарь, реранкер и порог — из настроек веб-демо (data/laws/colloquial_eval.json, они не меняются). Оценка реранкера
пары «вопрос — чанк» от пула не зависит, поэтому каждый чанк оценивается один раз на вопрос, а каждая настройка
сортирует свои кандидаты по этим оценкам. Нужна только видеокарта для реранкера; Ollama и Langfuse не используются.

    python eval_laws_pool.py        # ~5–10 минут → data/laws/pool_eval.json
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ask import default_model
from eval_laws_colloquial import LANG, gold_rank, laws_engine
from src.corpus import build_chunk_corpus
from src.laws import load_laws
from src.query_expansion import expand_query
from src.reranker import CrossEncoderReranker

SETTINGS = Path("data/laws/colloquial_eval.json")
NEW = Path("data/laws/colloquial_questions_100.json")
OLD = Path("data/laws/colloquial_questions.json")
RESULT = Path("data/laws/pool_eval.json")
POOLS = [20, 40, 80]
DENSE_WEIGHTS = [0.5, 0.65, 0.8, 1.0]
CONTEXT = 3  # сколько лучших чанков получает модель


def config_name(pool: int, weight: float) -> str:
    return f"пул {pool} / FAISS {weight:g}"


def is_even(qid: str) -> bool:
    return int(re.sub(r"\D", "", qid)) % 2 == 0


def summarize(rows: List[Dict[str, Any]], name: str, gate: float) -> Dict[str, Any]:
    """Итоги одной настройки на списке вопросов: с ответом в корпусе — места статьи, вне корпуса — оценки."""
    inside = [r["configs"][name] for r in rows if not r["outside"]]
    outside = [r["configs"][name] for r in rows if r["outside"]]
    out: Dict[str, Any] = {"questions": len(inside)}
    if inside:
        out.update({
            "hit@1": sum(c["rank"] == 1 for c in inside),
            "in_context": sum(c["in_context"] for c in inside),
            "in_context_passes_gate": sum(c["in_context"] and c["top_score"] >= gate for c in inside),
            "in_candidates": sum(c["in_candidates"] for c in inside),
        })
    if outside:
        out.update({"outside": len(outside), "outside_max_score": max(c["top_score"] for c in outside),
                    "outside_passes_gate": sum(c["top_score"] >= gate for c in outside)})
    return out


def main() -> None:
    import torch

    params = json.loads(SETTINGS.read_text(encoding="utf-8"))["params"]
    gate = params["min_retrieval_score"]
    corpus = build_chunk_corpus(load_laws(), {})
    by_id = {c["id"]: c for c in corpus}
    engine = laws_engine(corpus, default_model(LANG))
    hybrid = engine.hybrid_retriever
    current = (engine.rerank_pool, hybrid.dense_weight)
    pools = sorted(set(POOLS) | {current[0]})
    weights = sorted(set(DENSE_WEIGHTS) | {current[1]})
    configs: List[Tuple[int, float]] = [(p, w) for p in pools for w in weights]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoderReranker(params["reranker"], device=device)
    print(f"Корпус {len(corpus)} чанков | реранкер {params['reranker']} ({device}) | порог {gate} | "
          f"сейчас: {config_name(*current)}, rrf_k {hybrid.rrf_k} | настроек {len(configs)}")

    rows: Dict[str, List[Dict[str, Any]]] = {}
    rerank_seconds, rerank_docs = 0.0, 0
    for set_name, path in (("new", NEW), ("old", OLD)):
        rows[set_name] = []
        for q in json.loads(path.read_text(encoding="utf-8"))["questions"]:
            query = expand_query(q["question"]) if params["dictionary"] else q["question"]
            q_emb = engine.embed_query(query)[0]
            candidates: Dict[Tuple[int, float], List[str]] = {}
            for pool, weight in configs:
                hybrid.dense_weight, hybrid.sparse_weight = weight, 1.0 - weight
                candidates[(pool, weight)] = [d for d, _ in hybrid.search_with_scores(query, q_emb, k=pool)]
            union = sorted(set().union(*candidates.values()))
            t0 = time.perf_counter()
            scores = dict(zip(union, reranker.score(query, [engine.rerank_texts.get(d, engine.docs_by_id[d]) for d in union])))
            rerank_seconds += time.perf_counter() - t0
            rerank_docs += len(union)
            row: Dict[str, Any] = {"id": q["id"], "outside": not q["gold"], "configs": {}}
            for (pool, weight), ids in candidates.items():
                ranked = sorted(ids, key=lambda d: -scores[d])
                row["configs"][config_name(pool, weight)] = {
                    "rank": gold_rank(ranked, q["gold"], by_id) if q["gold"] else None,
                    "in_context": bool(q["gold"]) and gold_rank(ranked[:CONTEXT], q["gold"], by_id) is not None,
                    "in_candidates": bool(q["gold"]) and gold_rank(ids, q["gold"], by_id) is not None,
                    "top_score": round(scores[ranked[0]], 4),
                }
            rows[set_name].append(row)
            print(f"  {q['id']}: кандидатов для реранкера {len(union)}", flush=True)
    hybrid.dense_weight, hybrid.sparse_weight = current[1], 1.0 - current[1]

    names = [config_name(p, w) for p, w in configs]
    groups = {
        "new_even": [r for r in rows["new"] if is_even(r["id"])],
        "new_odd": [r for r in rows["new"] if not is_even(r["id"])],
        "new_all": rows["new"],
        "old": rows["old"],
    }
    summary = {g: {n: summarize(rs, n, gate) for n in names} for g, rs in groups.items()}

    # Выбор по чётным вопросам; ограничение — вопросы вне корпуса обоих наборов не проходят порог
    outside_ok = {n: summary["new_all"][n]["outside_passes_gate"] == 0 and summary["old"][n]["outside_passes_gate"] == 0
                  for n in names}
    order = sorted(
        [(p, w) for p, w in configs if outside_ok[config_name(p, w)]],
        key=lambda c: (-summary["new_even"][config_name(*c)]["in_context"],
                       -summary["new_even"][config_name(*c)]["in_context_passes_gate"],
                       c[0], abs(c[1] - current[1])),
    )
    chosen = config_name(*order[0]) if order else None
    ms_per_doc = 1000 * rerank_seconds / max(rerank_docs, 1)

    def line(group: str, n: str) -> str:
        s = summary[group][n]
        return (f"{s['in_candidates']:>3}/{s['questions']:<3} {s['in_context']:>3} {s['in_context_passes_gate']:>4} "
                f"{s['hit@1']:>4}   вне корпуса макс {s['outside_max_score']:.3f}, ≥ порога {s['outside_passes_gate']}")

    header = f"{'настройка':24s} {'в кандид.':>8s} {'топ-3':>4s} {'+порог':>5s} {'Hit@1':>5s}"
    for group, title in (("new_even", "Набор 100, ЧЁТНЫЕ (выбор)"), ("new_odd", "Набор 100, НЕЧЁТНЫЕ (проверка)"),
                         ("old", "Старые 59 вопросов")):
        print(f"\n{title}\n{header}")
        for n in names:
            mark = " ← сейчас" if n == config_name(*current) else (" ← выбрано" if n == chosen else "")
            print(f"{n:24s} {line(group, n)}{mark}{'' if outside_ok[n] else '  [вне корпуса проходит порог]'}")
    print(f"\nРеранкер: {ms_per_doc:.1f} мс на чанк → пул 20 ≈ {20 * ms_per_doc:.0f} мс, 40 ≈ {40 * ms_per_doc:.0f} мс, "
          f"80 ≈ {80 * ms_per_doc:.0f} мс на вопрос")
    cur = config_name(*current)
    if chosen:
        print(f"\nВыбрано на чётных: «{chosen}». Нечётные (проверка), правильная статья среди 3 лучших: "
              f"{summary['new_odd'][chosen]['in_context']} против {summary['new_odd'][cur]['in_context']} сейчас; "
              f"с проходом порога: {summary['new_odd'][chosen]['in_context_passes_gate']} против "
              f"{summary['new_odd'][cur]['in_context_passes_gate']}. Старые 59: {summary['old'][chosen]['in_context']} "
              f"против {summary['old'][cur]['in_context']}.")

    RESULT.write_text(json.dumps({
        "date": time.strftime("%Y-%m-%d %H:%M"), "reranker": params["reranker"], "gate": gate,
        "current": cur, "chosen_on_even": chosen, "rule": "чётные nq: max топ-3, затем топ-3 с порогом, меньший пул, "
        "вес ближе к текущему; вне корпуса (оба набора) ни одного ≥ порога",
        "rerank_ms_per_chunk": round(ms_per_doc, 2), "summary": summary, "per_question": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nРезультат: {RESULT}. Настройки веб-демо не менялись.")


if __name__ == "__main__":
    main()
