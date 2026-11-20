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
    python eval_laws_colloquial.py --answers                 # ответы Qwen2.5-3B, выбор настроек для веб-демо
    python eval_laws_colloquial.py --answers --llm qwen3:4b-instruct-2507-q4_K_M   # другая модель на тех же настройках
    python eval_laws_colloquial.py --nli                     # проверка по смыслу (NLI) показанных ответов всех моделей
    python eval_laws_colloquial.py --context                 # что получает модель: лучшие пункты, 2 фрагмента, промпт
Результат: data/laws/colloquial_eval.json.

Проверка на независимом наборе (100 вопросов, data/laws/colloquial_questions_100.json) — в отдельный файл, чтобы
не менять настройки веб-демо; --frozen берёт пороги из data/laws/colloquial_eval.json, --short — только «как сейчас»
и выбранные настройки, --no-gate — выбранные без порога уверенности поиска (порог потом считает eval_gate_cv.py):
    python eval_laws_colloquial.py --questions data/laws/colloquial_questions_100.json --result data/laws/colloquial_eval_100.json
    python eval_laws_colloquial.py --answers --short --frozen --questions ... --result ...
    python eval_laws_colloquial.py --answers --no-gate --frozen --questions ... --result ...
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

def _arg(name: str, default: str) -> str:
    """Значение ключа командной строки «--name значение» или значение по умолчанию."""
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


# Набор вопросов и файл результата. Веб-демо берёт настройки из data/laws/colloquial_eval.json (LAWS_SETTINGS в app.py),
# поэтому прогон на другом наборе пишется в отдельный файл: --questions data/laws/colloquial_questions_100.json
# --result data/laws/colloquial_eval_100.json
QUESTIONS = Path(_arg("--questions", "data/laws/colloquial_questions.json"))
RESULT = Path(_arg("--result", "data/laws/colloquial_eval.json"))
# --frozen: пороги уверенности поиска и соответствия вопросу не подбираются на этих вопросах, а берутся из настроек
# веб-демо (подобраны на прежнем наборе) — проверка на новых вопросах без подгонки
FROZEN_SETTINGS = Path("data/laws/colloquial_eval.json")
FROZEN = "--frozen" in sys.argv
MANUAL = Path("data/laws/colloquial_manual_check.json")  # ручная проверка показанных ответов (C / P / W)
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


# Настройки корпуса законов для веб-демо. Повтор после отказа добавил один правильный ответ из 49 (после просьбы
# перепроверить модель чаще отвечает одной ссылкой «[1]»), а каждый отказ делает дольше на запрос к модели — не включён
PREVIOUS = "+ числа словами, повтор коротких"  # выбор 7.15: на нём сравнивались модели ответа и NLI (7.16)
CHOSEN = "+ пункты статей после отказа"
CONTEXT_UNITS = "выбранные, 4 лучших пункта вместо фрагментов"
CONTEXT_TOP2 = "выбранные, 2 фрагмента"
CONTEXT_EXTRACTIVE = "выбранные, промпт «перепиши предложение»"
CONTEXT_UNITS_EXTRACTIVE = "выбранные, 4 пункта + промпт «перепиши предложение»"
# Варианты --answers: путь к выбранным настройкам (7.15 и 7.18); варианты контекста (--context) — отдельно
MAIN_VARIANTS = ["как сейчас", "новые настройки", "новые + термины модели", "+ числа словами", PREVIOUS,
                 "+ числа словами, повтор коротких и отказов", CHOSEN]
CONTEXT_VARIANTS = [PREVIOUS, CONTEXT_UNITS, CONTEXT_TOP2, CONTEXT_EXTRACTIVE, CONTEXT_UNITS_EXTRACTIVE, CHOSEN]


def gold_in(passages, gold, by_id: Dict[str, Dict[str, Any]]) -> bool:
    """Есть ли эталонная статья среди фрагментов, переданных модели."""
    return any((by_id[i]["law"], a) in gold for p in passages for i in p.ids for a in article_numbers(by_id[i]))


def quote_from_gold(quote: Dict[str, Any], passages, gold, by_id: Dict[str, Dict[str, Any]]) -> bool:
    """Взят ли пункт цитаты из эталонной статьи: статья чанка, в котором есть начало пункта."""
    head = quote["text"][:60]
    chunks = [by_id[i] for i in passages[quote["passage"] - 1].ids]
    owners = [c for c in chunks if head in c["text"]] or chunks
    return any((c["law"], a) in gold for c in owners for a in article_numbers(c))


def answers_stage(results: Dict[str, Any], llm: Optional[str] = None, only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Ответы Qwen на все вопросы: как сейчас в веб-демо и с новыми настройками для корпуса законов.

    Новые настройки: исходный реранкер, словарь (термины дописываются к вопросу для поиска и реранкера; модель получает
    исходный вопрос), порог уверенности поиска и порог соответствия ответа вопросу — по правилу eval_generation.py:
    каждый отбраковывает не больше 5% хороших случаев.

    llm — модель Ollama (по умолчанию DEFAULT_LLM), only — какие варианты настроек прогнать (по умолчанию все).
    """
    import torch
    from ask import build_passages
    from eval_generation import threshold_for
    from src.generation import (
        DEFAULT_LLM, STATUS_ANSWERED, STATUS_UNVERIFIED, GroundedAnswerer, OllamaClient, best_quotes, focus_passages,
        reranker_relevance,
    )
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
    frozen_relevance = None
    if FROZEN:
        frozen = json.loads(FROZEN_SETTINGS.read_text(encoding="utf-8"))["params"]
        new_gate, frozen_relevance = frozen["min_retrieval_score"], frozen["min_answer_relevance"]
        print(f"Пороги из {FROZEN_SETTINGS}: уверенность поиска {new_gate}, соответствие вопросу {frozen_relevance}")
    llm = llm or DEFAULT_LLM
    # Имя файла кэша из имени модели: «qwen2.5:3b» → llm_qwen2.5_3b.jsonl, «hf.co/автор/модель:Q4_K_M» — без «/»
    client = OllamaClient(llm, retries=3, cache_path=f"{CACHE_DIR}/llm_{re.sub(r'[^\w.-]', '_', llm)}.jsonl")
    entailment = load_entailment_scorer(adilet["min_entailment"])
    configs = {
        "как сейчас": {"reranker": default_reranker(LANG), "dictionary": False,
                       "min_retrieval_score": adilet["min_retrieval_score"], "min_answer_relevance": adilet["min_answer_relevance"]},
        "новые настройки": {"reranker": DEFAULT_RERANKER, "dictionary": True,
                            "min_retrieval_score": new_gate, "min_answer_relevance": None},  # порог соответствия — ниже
        # То же, но и модель получает вопрос с терминами закона: связь «развестись» → «расторжение брака» ей подсказана
        "новые + термины модели": {"reranker": DEFAULT_RERANKER, "dictionary": True, "terms_for_model": True,
                                   "min_retrieval_score": new_gate, "min_answer_relevance": None},
        # Проверка чисел понимает числа словами в законе («двадцать четыре календарных дня» = «24 дня»)
        "+ числа словами": {"reranker": DEFAULT_RERANKER, "dictionary": True, "number_words": True,
                            "min_retrieval_score": new_gate, "min_answer_relevance": None},
        # Повторный запрос, если ответ — одно число («24 [1]»), и после отказа при найденных фрагментах
        "+ числа словами, повтор коротких": {"reranker": DEFAULT_RERANKER, "dictionary": True, "number_words": True,
                                             "retry_short": True, "min_retrieval_score": new_gate,
                                             "min_answer_relevance": None},
        "+ числа словами, повтор коротких и отказов": {"reranker": DEFAULT_RERANKER, "dictionary": True,
                                                       "number_words": True, "retry_short": True, "retry_refusals": True,
                                                       "min_retrieval_score": new_gate, "min_answer_relevance": None},
    }
    # Что получает модель (--context): выбранные настройки, а вместо трёх фрагментов — лучшие пункты статей,
    # два фрагмента или промпт «сначала перепиши предложение закона с ответом»
    chosen = {**configs[PREVIOUS]}
    configs.update({
        CONTEXT_UNITS: {**chosen, "units": 4},
        CONTEXT_TOP2: {**chosen, "top_k": 2},
        CONTEXT_EXTRACTIVE: {**chosen, "prompt": "extractive"},
        CONTEXT_UNITS_EXTRACTIVE: {**chosen, "units": 4, "prompt": "extractive"},
        # После отказа — новый запрос по 4 лучшим пунктам статей: исправил 4 отказа, остальные ответы не изменились
        CHOSEN: {**chosen, "focus_on_refusal": 4},
    })
    if only:
        configs = {k: v for k, v in configs.items() if k in only}
    if FROZEN:
        for cfg in configs.values():
            if cfg["min_answer_relevance"] is None:
                cfg["min_answer_relevance"] = frozen_relevance
    if "--no-gate" in sys.argv:
        # Без порога уверенности поиска: все вопросы доходят до модели, порог потом считается по top_score записей
        for cfg in configs.values():
            cfg["min_retrieval_score"] = None
    out: Dict[str, Any] = {}
    for label, cfg in configs.items():
        reranker = CrossEncoderReranker(cfg["reranker"], device=device)
        engine.reranker = reranker
        answerer = GroundedAnswerer(
            client, LANG, min_retrieval_score=cfg["min_retrieval_score"], min_sentence_support=adilet["min_sentence_support"],
            min_answer_relevance=cfg["min_answer_relevance"], relevance_scorer=reranker_relevance(reranker),
            min_entailment=adilet["min_entailment"], entailment_scorer=entailment,
            retry_short=bool(cfg.get("retry_short")), retry_refusals=bool(cfg.get("retry_refusals")),
            number_words=bool(cfg.get("number_words")),
            # Цитата закона при отказе или отклонённом ответе (не меняет статусы и метрики ответа)
            quote_scorer=lambda query, units, r=reranker: r.score(query, units),
            prompt=cfg.get("prompt", "default"), focus_on_refusal=cfg.get("focus_on_refusal", 0),
        )
        records = []
        for q in data:
            query = expand_query(q["question"]) if cfg["dictionary"] else q["question"]
            hits = engine.search(query, mode="hybrid", top_k=cfg.get("top_k", 3), rerank=True)
            passages = build_passages([{**by_id[doc_id], "score": score} for doc_id, _, score in hits])
            if cfg.get("units"):
                # Пункты из найденных статей целиком (в корпусе законов фрагмент чанка — статья)
                passages = focus_passages(query, passages, answerer.quote_scorer, cfg["units"],
                                          texts=[by_id[p.id]["context"] for p in passages])
            answer = answerer.answer(query if cfg.get("terms_for_model") else q["question"], passages, search_query=query,
                                     source_texts=[by_id[p.id]["context"] for p in passages])
            gold = {(g["law"], g["article"]) for g in q["gold"]}
            cited = [passages[n - 1] for n in answer.cited if 1 <= n <= len(passages)] if answer.answered else []
            from_gold = any((by_id[i]["law"], a) in gold for p in cited for i in p.ids for a in article_numbers(by_id[i]))
            # raw — ответ модели до проверки (в answer при отклонении стоит запасной текст), problems — почему отклонён
            records.append({"id": q["id"], "status": answer.status, "from_gold": from_gold, "has_gold": bool(gold),
                            "top_score": round(float(max(p.score for p in passages)), 4) if passages else None,
                            "gold_in_context": gold_in(passages, gold, by_id),
                            "relevance": answer.checks.get("answer_relevance"), "answer": answer.text, "raw": answer.raw,
                            "problems": list(answer.checks.get("problems", [])),
                            "quotes": [{**x, "from_gold": quote_from_gold(x, passages, gold, by_id)} for x in answer.quotes]})
            if answer.status == STATUS_ANSWERED:
                # Цитата на случай, если ответ отклонит порог соответствия вопросу, подобранный ниже (в веб-демо он известен заранее)
                quotes = best_quotes(query, passages, answerer.quote_scorer, answerer.max_quotes)
                records[-1]["_quotes"] = [{**x, "from_gold": quote_from_gold(x, passages, gold, by_id)} for x in quotes]
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
                    r["problems"].append("ответ не относится к заданному вопросу")
                    r["quotes"] = r.get("_quotes", [])
        for r in records:
            r.pop("_quotes", None)
        inside = [r for r in records if r["has_gold"]]
        outside = [r for r in records if not r["has_gold"]]
        summary = {
            "answered": round(np.mean([r["status"] == STATUS_ANSWERED for r in inside]), 4),
            "answered_from_gold": round(np.mean([r["from_gold"] for r in inside]), 4),
            "outside_answered": round(np.mean([r["status"] == STATUS_ANSWERED for r in outside]), 4),
            "statuses": {s: sum(r["status"] == s for r in inside) for s in sorted({r["status"] for r in records})},
            # Цитаты закона вместо отказа или отклонённого ответа: из эталонной статьи первая / хотя бы одна
            "quotes": {"shown": sum(bool(r.get("quotes")) for r in inside),
                       "first_from_gold": sum(bool(r.get("quotes")) and r["quotes"][0]["from_gold"] for r in inside),
                       "any_from_gold": sum(any(x["from_gold"] for x in r.get("quotes", [])) for r in inside),
                       "outside": sum(bool(r.get("quotes")) for r in outside)},
        }
        out[label] = {"params": {k: v for k, v in cfg.items()}, "summary": summary, "records": records}
        del reranker
        engine.reranker = None
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"\n{'вариант':44s} {'ответ показан':>14s} {'по эталону':>11s} {'ответ вне корпуса':>18s}  статусы (вопросы с ответом в корпусе)")
    for label, o in out.items():
        s = o["summary"]
        print(f"{label:44s} {s['answered']:14.1%} {s['answered_from_gold']:11.1%} {s['outside_answered']:18.1%}  {s['statuses']}")
    p = out[CHOSEN]["params"] if CHOSEN in out else next(iter(out.values()))["params"]
    print(f"\nВыбран вариант «{CHOSEN}». Пороги для корпуса законов: уверенность поиска {p['min_retrieval_score']}, соответствие вопросу {p['min_answer_relevance']}")
    return out


def nli_stage(results: Dict[str, Any]) -> Dict[str, Any]:
    """Проверка по смыслу (NLI, src/nli.py) для показанных ответов каждой модели на выбранных настройках.

    Ответы берутся из сохранённых результатов, фрагменты восстанавливаются тем же поиском (он детерминирован).
    Рядом с оценкой — ручная метка ответа (MANUAL): ловит ли NLI неверные ответы, не отсекая верные. Порог —
    по правилу eval_generation.py (не больше 5% верных ответов отбраковывается), на тех же вопросах.
    """
    import torch
    from ask import build_passages
    from eval_generation import threshold_for
    from src.generation import DEFAULT_LLM, cited_passages
    from src.nli import NLIChecker, entailment_scorer

    data = {q["id"]: q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]}
    manual = json.loads(MANUAL.read_text(encoding="utf-8"))["models"]
    answers = {DEFAULT_LLM: results["answers"][PREVIOUS]["records"]}
    answers.update({llm: out[PREVIOUS]["records"] for llm, out in results.get("answers_by_llm", {}).items() if PREVIOUS in out})
    shown = {llm: [r for r in recs if r["status"] == "answered"] for llm, recs in answers.items()}

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
    engine.reranker = CrossEncoderReranker(results["params"]["reranker"], device=device)
    passages = {}
    for i in sorted({r["id"] for recs in shown.values() for r in recs}):
        hits = engine.search(expand_query(data[i]["question"]), mode="hybrid", top_k=3, rerank=True)
        passages[i] = build_passages([{**by_id[doc_id], "score": score} for doc_id, _, score in hits])
    engine.reranker = None
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    score = entailment_scorer(NLIChecker(device=device))
    out: Dict[str, Any] = {}
    for llm, recs in shown.items():
        labels = manual.get(llm, {})
        rows = [{"id": r["id"], "label": labels.get(r["id"], {}).get("label", "C"),
                 "nli": round(float(score(data[r["id"]]["question"], r["answer"], cited_passages(r["answer"], passages[r["id"]]))), 4)}
                for r in recs]
        threshold = threshold_for([x["nli"] for x in rows if x["label"] == "C"])
        kept = [x for x in rows if x["nli"] >= threshold]
        count = lambda xs, label: sum(x["label"] == label for x in xs)  # noqa: E731
        out[llm] = {"threshold": threshold, "rows": rows,
                    "without_nli": {k: count(rows, k) for k in "CPW"}, "with_nli": {k: count(kept, k) for k in "CPW"}}
        print(f"\n{llm}: порог NLI {threshold}; без NLI C/P/W = {out[llm]['without_nli']}, с NLI = {out[llm]['with_nli']}")
        for x in sorted(rows, key=lambda x: x["nli"]):
            print(f"  {x['nli']:.4f} {x['label']} {x['id']} {data[x['id']]['question']}")
    return out


def main() -> None:
    if "--nli" in sys.argv:
        results = json.loads(RESULT.read_text(encoding="utf-8"))
        results["nli"] = nli_stage(results)
        RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    if "--context" in sys.argv:
        # Что получает модель: варианты контекста и промпта на выбранных настройках; настройки веб-демо не меняются
        results = json.loads(RESULT.read_text(encoding="utf-8"))
        only = [v for v in CONTEXT_VARIANTS if "--fallback-only" not in sys.argv or v in (PREVIOUS, CHOSEN)]
        out = answers_stage(results, only=only)
        results["answers_context"] = {**results.get("answers_context", {}), **out}
        RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    if "--answers" in sys.argv and "--llm" in sys.argv:
        # Другая модель ответа — на выбранных настройках; настройки веб-демо (params) не меняются
        llm = sys.argv[sys.argv.index("--llm") + 1]
        results = json.loads(RESULT.read_text(encoding="utf-8"))
        out = answers_stage(results, llm, only=[PREVIOUS, "+ числа словами, повтор коротких и отказов"])
        results.setdefault("answers_by_llm", {})[llm] = out
        RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    if "--answers" in sys.argv and "--no-gate" in sys.argv:
        # Выбранные настройки без порога уверенности поиска — отдельный ключ, прежние ответы и настройки не меняются
        results = json.loads(RESULT.read_text(encoding="utf-8"))
        results["answers_no_gate"] = answers_stage(results, only=[CHOSEN])
        RESULT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    if "--answers" in sys.argv:
        results = json.loads(RESULT.read_text(encoding="utf-8"))
        # --short: только исходные настройки и выбранные (без промежуточных шагов 7.15–7.18)
        results["answers"] = answers_stage(results, only=["как сейчас", CHOSEN] if "--short" in sys.argv else MAIN_VARIANTS)
        new = results["answers"][CHOSEN]["params"]
        # Настройки корпуса законов для веб-демо (app.py)
        results["params"] = {"reranker": new["reranker"], "dictionary": True, "terms_for_model": bool(new.get("terms_for_model")),
                             "retry_short": bool(new.get("retry_short")), "retry_refusals": bool(new.get("retry_refusals")),
                             "number_words": bool(new.get("number_words")),
                             "focus_on_refusal": int(new.get("focus_on_refusal") or 0),
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
