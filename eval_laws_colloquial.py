"""Оценка поиска по корпусу актуальных законов на вопросах обычным языком (data/laws/colloquial_questions.json).

Вопросы датасета Adilet написаны языком закона, а люди спрашивают иначе («как уволиться», «как развестись»).
Здесь сравниваются на разговорных вопросах:
- методы поиска: BM25, Dense (e5), Hybrid (RRF) — без реранкера;
- Hybrid + реранкер: дообученный на Adilet и исходный bge-reranker-v2-m3;
- всё это без словаря и со словарём «разговорное → юридическое» (src/query_expansion.py).

Метрики — на уровне статей (одна статья = один фрагмент корпуса): Hit@1, Hit@3 (модель получает 3 лучших чанка),
MRR@10. Правильный ответ — любая из статей gold вопроса.

Порог уверенности поиска (оценка реранкера лучшего фрагмента): для вопросов вне корпуса правильное поведение — отказ,
для остальных — пропустить вопрос к модели. Для каждой конфигурации с реранкером печатается, какую долю тех и других
пропускает порог, и подбирается порог для корпуса законов.

Запуск (видеокарта нужна только реранкеру; e5 работает на процессоре, как в веб-демо):
    python eval_laws_colloquial.py
Результат: data/laws/colloquial_eval.json.
"""

import gc
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

from ask import LegalSearchEngine, default_model, load_bm25_params, load_hybrid_params, load_rerank_pool
from run_adilet_eval import model_cache_name
from src.corpus import CHUNK_OVERLAP, CHUNK_SIZE, build_chunk_corpus
from src.embeddings import EmbeddingModel
from src.generation import load_generation_params
from src.laws import load_laws
from src.query_expansion import expand_query
from src.reranker import DEFAULT_RERANKER, CrossEncoderReranker, default_reranker

QUESTIONS = Path("data/laws/colloquial_questions.json")
RESULT = Path("data/laws/colloquial_eval.json")
CACHE_DIR = "data/adilet/cache"
LANG = "rus"
DEPTH = 10  # сколько чанков выдачи учитывается в метриках


def article_numbers(chunk: Dict[str, Any]) -> List[str]:
    """Номера статей чанка: «ст. 159» → ['159'], «ст. 1072-2» → ['1072-2']."""
    return re.findall(r"\d+(?:-\d+)?", chunk.get("articles", ""))


def gold_rank(chunk_ids: Sequence[str], gold: List[Dict[str, str]], by_id: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """Место первой правильной статьи в выдаче (статьи без повторов, с 1); None — не найдена."""
    wanted = {(g["law"], g["article"]) for g in gold}
    seen: List[Tuple[str, str]] = []
    for doc_id in chunk_ids:
        chunk = by_id[doc_id]
        keys = [(chunk["law"], a) for a in article_numbers(chunk)] or [(chunk["law"], "")]
        for key in keys:
            if key not in seen:
                seen.append(key)
        if any(key in wanted for key in keys):
            return seen.index(next(k for k in keys if k in wanted)) + 1
    return None


def metrics(ranks: List[Optional[int]]) -> Dict[str, float]:
    n = len(ranks)
    return {
        "hit@1": round(sum(r == 1 for r in ranks) / n, 4),
        "hit@3": round(sum(r is not None and r <= 3 for r in ranks) / n, 4),
        "mrr@10": round(sum(1 / r for r in ranks if r is not None and r <= 10) / n, 4),
    }


def threshold_table(answerable: List[float], outside: List[float], thresholds: Sequence[float]) -> List[Dict[str, float]]:
    """Для каждого порога: доля вопросов с ответом в корпусе, которые он пропускает, и доля вопросов вне корпуса, которые отсекает."""
    return [
        {"threshold": round(float(t), 4),
         "answerable_passed": round(float(np.mean([s >= t for s in answerable])), 4),
         "outside_rejected": round(float(np.mean([s < t for s in outside])), 4)}
        for t in thresholds
    ]


def adilet_regression() -> Dict[str, Any]:
    """Не портит ли словарь вопросы языком закона: test датасета Adilet (737 вопросов), гибрид + дообученный реранкер."""
    import torch
    from tune_hybrid import load_eval_data

    chunks, chunk_ids, splits, gold = load_eval_data(LANG)
    model_name = default_model(LANG)
    engine = LegalSearchEngine(
        docs=chunks, doc_ids=chunk_ids, embedder=EmbeddingModel(model_name=model_name, device="cpu"), cache_dir=CACHE_DIR,
        cache_key=f"{model_cache_name(model_name)}_{LANG}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
        hybrid_params=load_hybrid_params(model_name, LANG), bm25_params=load_bm25_params(LANG),
        reranker=CrossEncoderReranker(default_reranker(LANG), device="cuda" if torch.cuda.is_available() else "cpu"),
        rerank_pool=load_rerank_pool(LANG),
    )
    questions = [q["query"] for q in splits["test"]]
    out: Dict[str, Any] = {"questions": len(questions), "changed_by_dictionary": sum(expand_query(q) != q for q in questions)}
    for variant, transform in (("без словаря", lambda q: q), ("со словарём", expand_query)):
        hit1, rr = [], []
        for q, g in zip(questions, gold["test"]):
            ids = [h[0] for h in engine.search(transform(q), mode="hybrid", top_k=DEPTH, rerank=True)]
            rank = next((i for i, d in enumerate(ids, 1) if d in g), None)
            hit1.append(rank == 1)
            rr.append(1 / rank if rank else 0.0)
        out[variant] = {"hit@1": round(float(np.mean(hit1)), 4), "mrr@10": round(float(np.mean(rr)), 4)}
        print(f"Adilet test, гибрид + дообученный реранкер, {variant}: Hit@1 {out[variant]['hit@1']:.3f}, MRR@10 {out[variant]['mrr@10']:.3f}")
    print(f"Словарь изменил {out['changed_by_dictionary']} вопросов из {len(questions)}")
    return out


def answers_stage(results: Dict[str, Any]) -> Dict[str, Any]:
    """Ответы Qwen на все вопросы: как сейчас в веб-демо и с новыми настройками для корпуса законов.

    Новые настройки: исходный реранкер, словарь (термины дописываются к вопросу для поиска и реранкера; модель получает
    исходный вопрос), порог уверенности поиска и порог соответствия ответа вопросу — по правилу eval_generation.py:
    каждый отбраковывает не больше 5% хороших случаев.
    """
    import torch
    from ask import build_passages
    from eval_generation import threshold_for
    from src.generation import DEFAULT_LLM, STATUS_ANSWERED, STATUS_UNVERIFIED, GroundedAnswerer, OllamaClient, reranker_relevance
    from src.nli import load_entailment_scorer

    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    corpus = build_chunk_corpus(load_laws(), {})
    by_id = {c["id"]: c for c in corpus}
    model_name = default_model(LANG)
    engine = LegalSearchEngine(
        docs=[c["text"] for c in corpus], doc_ids=[c["id"] for c in corpus],
        embedder=EmbeddingModel(model_name=model_name, device="cpu"), cache_dir=CACHE_DIR,
        cache_key=f"{model_cache_name(model_name)}_{LANG}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}_laws",
        hybrid_params=load_hybrid_params(model_name, LANG), bm25_params=load_bm25_params(LANG),
        rerank_pool=load_rerank_pool(LANG),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adilet = load_generation_params(LANG, default_reranker(LANG))
    new_gate = threshold_for(results["gates"]["hybrid + реранкер исходный / со словарём"]["answerable_top_scores"])
    client = OllamaClient(DEFAULT_LLM, retries=3, cache_path=f"{CACHE_DIR}/llm_{DEFAULT_LLM.replace(':', '_')}.jsonl")
    entailment = load_entailment_scorer(adilet["min_entailment"])
    configs = {
        "как сейчас": {"reranker": default_reranker(LANG), "dictionary": False,
                       "min_retrieval_score": adilet["min_retrieval_score"], "min_answer_relevance": adilet["min_answer_relevance"]},
        "новые настройки": {"reranker": DEFAULT_RERANKER, "dictionary": True,
                            "min_retrieval_score": new_gate, "min_answer_relevance": None},  # порог соответствия — ниже
        # То же, но и модель получает вопрос с терминами закона: связь «развестись» → «расторжение брака» ей подсказана
        "новые + термины модели": {"reranker": DEFAULT_RERANKER, "dictionary": True, "terms_for_model": True,
                                   "min_retrieval_score": new_gate, "min_answer_relevance": None},
    }
    out: Dict[str, Any] = {}
    for label, cfg in configs.items():
        reranker = CrossEncoderReranker(cfg["reranker"], device=device)
        engine.reranker = reranker
        answerer = GroundedAnswerer(
            client, LANG, min_retrieval_score=cfg["min_retrieval_score"], min_sentence_support=adilet["min_sentence_support"],
            min_answer_relevance=cfg["min_answer_relevance"], relevance_scorer=reranker_relevance(reranker),
            min_entailment=adilet["min_entailment"], entailment_scorer=entailment,
        )
        records = []
        for q in data:
            query = expand_query(q["question"]) if cfg["dictionary"] else q["question"]
            hits = engine.search(query, mode="hybrid", top_k=3, rerank=True)
            passages = build_passages([{**by_id[doc_id], "score": score} for doc_id, _, score in hits])
            answer = answerer.answer(query if cfg.get("terms_for_model") else q["question"], passages)
            gold = {(g["law"], g["article"]) for g in q["gold"]}
            cited = [passages[n - 1] for n in answer.cited if 1 <= n <= len(passages)] if answer.answered else []
            from_gold = any((by_id[i]["law"], a) in gold for p in cited for i in p.ids for a in article_numbers(by_id[i]))
            records.append({"id": q["id"], "status": answer.status, "from_gold": from_gold, "has_gold": bool(gold),
                            "relevance": answer.checks.get("answer_relevance"), "answer": answer.text})
            print(f"  [{label}] {q['id']} {answer.status:10s} {'по эталону' if from_gold else '':10s} {q['question']}", flush=True)
        if cfg["min_answer_relevance"] is None:
            # Порог соответствия ответа вопросу в шкале исходного реранкера: хорошие — показанные ответы по эталонной статье,
            # в которых есть слова (ответ «1 [1]» ссылается на нужную статью, но ответом не является)
            good = [r["relevance"] for r in records if r["status"] == STATUS_ANSWERED and r["from_gold"] and r["relevance"] is not None
                    and len(re.findall(r"[^\W\d_]{3,}", r["answer"])) >= 2]
            cfg["min_answer_relevance"] = threshold_for(good) if good else None
            for r in records:
                if r["status"] == STATUS_ANSWERED and r["relevance"] is not None and r["relevance"] < cfg["min_answer_relevance"]:
                    r["status"], r["from_gold"] = STATUS_UNVERIFIED, False
        inside = [r for r in records if r["has_gold"]]
        outside = [r for r in records if not r["has_gold"]]
        summary = {
            "answered": round(np.mean([r["status"] == STATUS_ANSWERED for r in inside]), 4),
            "answered_from_gold": round(np.mean([r["from_gold"] for r in inside]), 4),
            "outside_answered": round(np.mean([r["status"] == STATUS_ANSWERED for r in outside]), 4),
            "statuses": {s: sum(r["status"] == s for r in inside) for s in sorted({r["status"] for r in records})},
        }
        out[label] = {"params": {k: v for k, v in cfg.items()}, "summary": summary, "records": records}
        del reranker
        engine.reranker = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"\n{'вариант':18s} {'ответ показан':>14s} {'по эталону':>11s} {'ответ вне корпуса':>18s}  статусы (вопросы с ответом в корпусе)")
    for label, o in out.items():
        s = o["summary"]
        print(f"{label:18s} {s['answered']:14.1%} {s['answered_from_gold']:11.1%} {s['outside_answered']:18.1%}  {s['statuses']}")
    p = out[max(out, key=lambda k: out[k]["summary"]["answered_from_gold"])]["params"]
    print(f"\nПороги для корпуса законов: уверенность поиска {p['min_retrieval_score']}, соответствие вопросу {p['min_answer_relevance']}")
    return out


def main() -> None:
    if "--answers" in sys.argv:
        results = json.loads(RESULT.read_text(encoding="utf-8"))
        results["answers"] = answers_stage(results)
        best = max(results["answers"], key=lambda k: results["answers"][k]["summary"]["answered_from_gold"])
        new = results["answers"][best]["params"]
        # Настройки корпуса законов для веб-демо (app.py)
        results["params"] = {"reranker": new["reranker"], "dictionary": True, "terms_for_model": bool(new.get("terms_for_model")),
                             "min_retrieval_score": new["min_retrieval_score"], "min_answer_relevance": new["min_answer_relevance"]}
        RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    if "--adilet" in sys.argv:
        results = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
        results["adilet_regression"] = adilet_regression()
        RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    answerable = [q for q in data if q["gold"]]
    outside = [q for q in data if not q["gold"]]
    corpus = build_chunk_corpus(load_laws(), {})
    by_id = {c["id"]: c for c in corpus}
    model_name = default_model(LANG)
    engine = LegalSearchEngine(
        docs=[c["text"] for c in corpus], doc_ids=[c["id"] for c in corpus],
        embedder=EmbeddingModel(model_name=model_name, device="cpu"), cache_dir=CACHE_DIR,
        cache_key=f"{model_cache_name(model_name)}_{LANG}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}_laws",
        hybrid_params=load_hybrid_params(model_name, LANG), bm25_params=load_bm25_params(LANG),
    )
    pool = load_rerank_pool(LANG)
    print(f"Вопросов: {len(answerable)} с ответом в корпусе, {len(outside)} вне корпуса | корпус {len(corpus)} чанков | "
          f"e5 {model_name} (CPU) | пул реранкера {pool}")

    queries = {
        "без словаря": {q["id"]: q["question"] for q in data},
        "со словарём": {q["id"]: expand_query(q["question"]) for q in data},
    }
    covered = sum(queries["со словарём"][q["id"]] != q["question"] for q in data)
    print(f"Словарь дополнил {covered} вопросов из {len(data)}")

    results: Dict[str, Any] = {"questions": len(data), "answerable": len(answerable), "outside": len(outside),
                               "dictionary_covered": covered, "configs": {}, "per_question": {}}
    # Кандидаты гибрида для реранкера считаются один раз на вариант вопроса
    candidates: Dict[str, Dict[str, List[Tuple[str, str, float]]]] = {}
    for variant, texts in queries.items():
        candidates[variant] = {}
        for mode in ("bm25", "dense", "hybrid"):
            ranks = []
            for q in answerable:
                hits = engine.search(texts[q["id"]], mode=mode, top_k=max(DEPTH, pool))
                if mode == "hybrid":
                    candidates[variant][q["id"]] = hits[:pool]
                ranks.append(gold_rank([h[0] for h in hits[:DEPTH]], q["gold"], by_id))
            name = f"{mode} / {variant}"
            results["configs"][name] = metrics(ranks)
            results["per_question"][name] = dict(zip([q["id"] for q in answerable], ranks))
        for q in outside:
            candidates[variant][q["id"]] = engine.search(texts[q["id"]], mode="hybrid", top_k=pool)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gates = {}
    for label, reranker_name in (("дообученный", default_reranker(LANG)), ("исходный", DEFAULT_RERANKER)):
        reranker = CrossEncoderReranker(reranker_name, device=device)
        engine.reranker = reranker
        for variant, texts in queries.items():
            ranks, top_answerable, top_outside = [], [], []
            for q in data:
                reranked = engine.rerank(texts[q["id"]], candidates[variant][q["id"]], DEPTH)
                top = reranked[0][2] if reranked else float("-inf")
                if q["gold"]:
                    rank = gold_rank([h[0] for h in reranked], q["gold"], by_id)
                    ranks.append(rank)
                    # Хороший случай для порога: правильная статья среди 3 лучших — модели есть по чему ответить
                    if rank is not None and rank <= 3:
                        top_answerable.append(top)
                else:
                    top_outside.append(top)
            name = f"hybrid + реранкер {label} / {variant}"
            results["configs"][name] = metrics(ranks)
            results["per_question"][name] = dict(zip([q["id"] for q in answerable], ranks))
            scores = sorted(set(top_answerable + top_outside))
            gates[name] = {
                "reranker": reranker_name,
                "answerable_top_scores": [round(s, 4) for s in top_answerable],
                "outside_top_scores": [round(s, 4) for s in top_outside],
                "curve": threshold_table(top_answerable, top_outside, scores),
            }
        del reranker, engine.reranker
        engine.reranker = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Порог, подобранный на датасете Adilet (им пользуются веб-демо и ask.py)
    current = load_generation_params(LANG, default_reranker(LANG))["min_retrieval_score"]
    results["current_gate"] = current
    results["gates"] = gates
    if RESULT.exists():  # замер на Adilet (--adilet) считается отдельно и не пересчитывается здесь
        previous = json.loads(RESULT.read_text(encoding="utf-8"))
        results.update({k: previous[k] for k in ("adilet_regression",) if k in previous})
    RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n{'конфигурация':52s} {'Hit@1':>6s} {'Hit@3':>6s} {'MRR@10':>7s}")
    for name, m in results["configs"].items():
        print(f"{name:52s} {m['hit@1']:6.3f} {m['hit@3']:6.3f} {m['mrr@10']:7.3f}")
    print(f"\nПорог уверенности поиска, подобранный на Adilet: {current}")
    for name, g in gates.items():
        a, o = g["answerable_top_scores"], g["outside_top_scores"]
        passed = np.mean([s >= current for s in a]) if a and current is not None else float("nan")
        rejected = np.mean([s < current for s in o]) if o and current is not None else float("nan")
        print(f"  {name:50s} при пороге Adilet пропущено {passed:.0%} хороших, отсечено {rejected:.0%} вне корпуса; "
              f"оценки хороших: медиана {np.median(a):.3f}, мин {min(a):.3f}; вне корпуса: макс {max(o):.3f}")
    print(f"\nРезультат: {RESULT}")


if __name__ == "__main__":
    main()
