"""Варианты поиска по корпусу законов на вопросах обычным языком — только поиск, без модели ответа.

Проверка 20.11 на 100 независимых вопросах показала, что ответ теряется прежде всего в поиске: в 25 из 80 нужной
статьи нет среди трёх найденных, ещё в 30 поиск не уверен (оценка реранкера ниже порога 0,6). Сравниваются варианты,
не подстроенные под конкретные вопросы и без языковой модели (правила курсовой разрешают её только для ответа):

- текст чанка для поиска: как есть или с названием закона и заголовком статьи (src/corpus.index_text) —
  заголовок есть лишь у 28% чанков, а тема лучше всего видна по нему;
- пул кандидатов реранкера: 20 (как сейчас) или 50;
- эмбеддер: дообученная на языке закона e5 или исходная multilingual-e5-small.

Во всех вариантах — гибрид (RRF) + исходный реранкер bge-reranker-v2-m3 + словарь, как в веб-демо.

Метрики (статья = фрагмент корпуса законов): Hit@1, Hit@3 по разным статьям; «в контексте» — нужная статья среди
трёх чанков, которые сейчас получает модель, и «в контексте, 3 статьи» — среди трёх разных статей (лучший чанк каждой);
«уверен и в контексте» — то же при оценке лучшего чанка не ниже порога 0,6; для вопросов вне корпуса — сколько
из них проходит порог (должно быть 0) и наибольшая оценка.

Подбор — на наборах, по которым система не настраивалась (colloquial_questions_independent.json, 50 вопросов)
и на прежних 59 (контроль); итоговая проверка — один раз на 100 независимых вопросах.

Запуск (нужна видеокарта; новые эмбеддинги корпуса считаются один раз и кэшируются):
    python eval_laws_search.py --questions data/laws/colloquial_questions_independent.json
    python eval_laws_search.py --questions data/laws/colloquial_questions_100.json --variants "..."
    python eval_laws_search.py --reranker models/reranker-bge-m3-laws-colloquial --pools 20
Результат дописывается в data/laws/search_variants.json (ключ — имя файла вопросов). С ключом --reranker к названию
варианта добавляется «/ реранкер <папка>», прежние результаты не перезаписываются. Порог 0,6 подобран для исходного
реранкера; для другого порог надо подбирать заново по per_question (top_score сохраняется у всех вопросов).

doc2query (src/doc2query.py) — сгенерированные разговорные вопросы к статьям как дополнительные ключи индекса;
реранкер по-прежнему видит только текст закона:
    python eval_laws_search.py --doc2query append --leak 0.2 --pools 20 --variants "без заголовка / e5 дообученная / пул 20"
--doc2query append — вопросы дописываются к чанку, separate — отдельные документы; --leak — порог сходства, с которого
сгенерированный вопрос к той же статье считается пересказом проверочного и убирается (0,2 строго, 0,3 умеренно;
none — без этого фильтра).
"""

import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ask import LegalSearchEngine, default_model, load_bm25_params, load_hybrid_params
from eval_laws_colloquial import DEPTH, article_numbers, gold_rank, metrics
from run_adilet_eval import DEFAULT_MODEL, model_cache_name
from src import doc2query as d2q
from src.corpus import CHUNK_OVERLAP, CHUNK_SIZE, build_chunk_corpus, index_text
from src.embeddings import EmbeddingModel
from src.laws import load_laws
from src.query_expansion import expand_query
from src.reranker import DEFAULT_RERANKER, CrossEncoderReranker

CACHE_DIR = "data/adilet/cache"
LANG = "rus"
GATE = 0.6  # порог уверенности поиска веб-демо (data/laws/colloquial_eval.json, params)
CONTEXT = 3  # сколько чанков получает модель (DEFAULT_TOP_K)
RESULT = Path("data/laws/search_variants.json")
EVAL_SETS = ["data/laws/colloquial_questions_100.json", "data/laws/colloquial_questions_independent.json",
             "data/laws/colloquial_questions.json"]
LEAKS = "data/laws/doc2query_leaks_{leak}.json"  # какие сгенерированные вопросы убраны как пересказы и почему


def _arg(name: str, default: Optional[str]) -> Optional[str]:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def article_keys(chunk: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [(chunk["law"], a) for a in article_numbers(chunk)] or [(chunk["law"], chunk["id"])]


def distinct_articles(chunk_ids: List[str], by_id: Dict[str, Dict[str, Any]], k: int) -> List[str]:
    """Лучший чанк каждой статьи, k первых статей."""
    seen, result = set(), []
    for doc_id in chunk_ids:
        key = article_keys(by_id[doc_id])[0]
        if key not in seen:
            seen.add(key)
            result.append(doc_id)
        if len(result) == k:
            break
    return result


def rerank_texts(corpus: List[Dict[str, Any]], heading: bool) -> Dict[str, str]:
    """Текст чанка, который видит реранкер: текст закона (с заголовком или без), без вопросов doc2query."""
    return {c["id"]: index_text(c) if heading else c["text"] for c in corpus}


def build_engine(corpus: List[Dict[str, Any]], model_name: str, heading: bool,
                 d2q_mode: Optional[str] = None, by_chunk: Optional[Dict[str, List[str]]] = None) -> LegalSearchEngine:
    """Индексы BM25 и FAISS по тексту чанков (с заголовком или без) и, если задан doc2query, по сгенерированным
    вопросам; эмбеддинги — на видеокарте, с кэшем (имя файла включает отпечаток текстов)."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = rerank_texts(corpus, heading)
    ids = [c["id"] for c in corpus]
    texts = [base[i] for i in ids]
    suffix = "_laws_heading" if heading else "_laws"
    if d2q_mode == "append":
        texts = [d2q.append_questions(base[i], (by_chunk or {}).get(i, [])) for i in ids]
        suffix += "_d2q_append"
    elif d2q_mode == "separate":
        q_texts, q_ids = d2q.question_docs(by_chunk or {})
        texts, ids = texts + q_texts, ids + q_ids
        suffix += "_d2q_separate"
    return LegalSearchEngine(
        docs=texts, doc_ids=ids,
        embedder=EmbeddingModel(model_name=model_name, device=device), cache_dir=CACHE_DIR,
        cache_key=f"{model_cache_name(model_name)}_{LANG}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}{suffix}",
        hybrid_params=load_hybrid_params(model_name, LANG), bm25_params=load_bm25_params(LANG),
    )


def candidates_for(engine: LegalSearchEngine, query: str, pool: int, texts: Dict[str, str],
                   separate: bool) -> List[Tuple[str, str, float]]:
    """pool чанков-кандидатов для реранкера с текстом закона. В режиме «separate» найденные вопросы заменяются
    своими чанками (повторы убираются), поэтому ищется с запасом."""
    hits = engine.search(query, mode="hybrid", top_k=pool * 4 if separate else pool)
    return [(cid, texts[cid], score) for cid, score in d2q.collapse(hits, pool)]


def evaluate(engine: LegalSearchEngine, data: List[Dict[str, Any]], by_id: Dict[str, Dict[str, Any]],
             pools: List[int], texts: Dict[str, str], separate: bool = False) -> Dict[int, Dict[str, Any]]:
    """Для каждого размера пула: метрики и по каждому вопросу место правильной статьи и оценка лучшего чанка."""
    out: Dict[int, Dict[str, Any]] = {}
    queries = {q["id"]: expand_query(q["question"]) for q in data}
    for pool in pools:
        per_question, ranks = {}, []
        for q in data:
            # Отдельный поиск для каждого пула: с пулом 20 — ровно те же кандидаты, что в веб-демо
            candidates = candidates_for(engine, queries[q["id"]], pool, texts, separate)
            reranked = engine.rerank(queries[q["id"]], candidates, DEPTH)
            ids = [h[0] for h in reranked]
            top = reranked[0][2] if reranked else float("-inf")
            record = {"top_score": round(float(top), 4)}
            if q["gold"]:
                wanted = {(g["law"], g["article"]) for g in q["gold"]}
                in_ctx = lambda chunk_ids: any(k in wanted for i in chunk_ids for k in article_keys(by_id[i]))  # noqa: E731
                rank = gold_rank(ids, q["gold"], by_id)
                ranks.append(rank)
                record.update({"rank": rank, "context": in_ctx(ids[:CONTEXT]),
                               "context_articles": in_ctx(distinct_articles(ids, by_id, CONTEXT))})
            per_question[q["id"]] = record
        inside = [r for r in per_question.values() if "rank" in r]
        outside = [r for r in per_question.values() if "rank" not in r]
        share = lambda xs: round(sum(xs) / len(inside), 4) if inside else None  # noqa: E731
        out[pool] = {
            **metrics(ranks),
            "context": share([r["context"] for r in inside]),
            "context_articles": share([r["context_articles"] for r in inside]),
            "confident_context": share([r["context"] and r["top_score"] >= GATE for r in inside]),
            "confident_context_articles": share([r["context_articles"] and r["top_score"] >= GATE for r in inside]),
            "outside_passed": sum(r["top_score"] >= GATE for r in outside),
            "outside_max": max((r["top_score"] for r in outside), default=None),
            "per_question": per_question,
        }
    return out


def load_doc2query(corpus: List[Dict[str, Any]], leak: Optional[float]) -> Dict[str, List[str]]:
    """Сгенерированные вопросы по чанкам без пересказов проверочных вопросов (всех трёх наборов сразу,
    чтобы индекс был один для отладки и итоговой проверки). leak=None — без этого фильтра (остаётся только
    фильтр prepare_reranker_colloquial.py: доля общих основ слов ≥ 0,5)."""
    if leak is None:
        records = d2q.load_generated()
        print(f"doc2query: {len(records)} вопросов, без дополнительного фильтра пересказов")
        return d2q.expansions(records)
    frag_keys: Dict[str, set] = {}
    for c in corpus:
        frag_keys.setdefault(c["id"].split("_fix_")[0], set()).update(article_keys(c))
    records = d2q.load_generated()
    eval_questions = [(q["question"], {(g["law"], g["article"]) for g in q["gold"]})
                      for f in EVAL_SETS for q in json.loads(Path(f).read_text(encoding="utf-8"))["questions"]]
    kept, removed = d2q.leak_filter(records, eval_questions, [frag_keys.get(r["article"], set()) for r in records], leak)
    print(f"doc2query: {len(records)} вопросов, убрано как пересказы проверочных {len(removed)} (сходство ≥ {leak}), "
          f"осталось {len(kept)}")
    Path(LEAKS.format(leak=leak)).write_text(json.dumps([{k: r[k] for k in ("question", "leak_eval", "leak_sim")} for r in removed],
                                ensure_ascii=False, indent=1), encoding="utf-8")
    return d2q.expansions(kept)


def main() -> None:
    questions = Path(_arg("--questions", "data/laws/colloquial_questions_independent.json"))
    data = json.loads(questions.read_text(encoding="utf-8"))["questions"]
    pools = [int(p) for p in (_arg("--pools", "20,50") or "").split(",")]
    models = {"e5 дообученная": default_model(LANG), "e5 исходная": DEFAULT_MODEL}
    wanted = _arg("--variants", None)
    reranker_name = _arg("--reranker", DEFAULT_RERANKER)
    reranker_suffix = "" if reranker_name == DEFAULT_RERANKER else f" / реранкер {Path(reranker_name).name}"
    corpus = build_chunk_corpus(load_laws(), {})
    by_id = {c["id"]: c for c in corpus}
    n_in = sum(bool(q["gold"]) for q in data)
    print(f"{questions.name}: {n_in} вопросов с ответом в корпусе, {len(data) - n_in} вне корпуса | пулы {pools}")

    d2q_mode = _arg("--doc2query", None)
    if d2q_mode not in (None, "append", "separate"):
        raise SystemExit("--doc2query: append или separate")
    leak_arg = _arg("--leak", str(d2q.LEAK_STRICT))
    leak = None if leak_arg == "none" else float(leak_arg)
    by_chunk: Dict[str, List[str]] = {}
    if d2q_mode:
        by_chunk = load_doc2query(corpus, leak)
        reranker_suffix += f" / doc2query {d2q_mode}, " + ("без фильтра пересказов" if leak is None else f"утечки ≥ {leak}")

    reranker = None
    results: Dict[str, Any] = {}
    for model_label, model_name in models.items():
        for heading in (False, True):
            labels = {pool: f"{'с заголовком' if heading else 'без заголовка'} / {model_label} / пул {pool}" for pool in pools}
            if wanted and not any(label in wanted for label in labels.values()):
                continue
            labels = {pool: label + reranker_suffix for pool, label in labels.items()}
            engine = build_engine(corpus, model_name, heading, d2q_mode, by_chunk)
            if reranker is None:
                import torch

                reranker = CrossEncoderReranker(reranker_name, device="cuda" if torch.cuda.is_available() else "cpu")
            engine.reranker = reranker
            texts = rerank_texts(corpus, heading)
            for pool, value in evaluate(engine, data, by_id, pools, texts, d2q_mode == "separate").items():
                results[labels[pool]] = value
            del engine
            gc.collect()

    print(f"\n{'вариант':48s} {'Hit@1':>6s} {'Hit@3':>6s} {'в конт.':>8s} {'3 стат.':>8s} {'увер.+к.':>9s} "
          f"{'увер.+3 ст.':>11s} {'вне: прошло / макс':>19s}")
    for label, m in results.items():
        print(f"{label:48s} {m['hit@1']:6.3f} {m['hit@3']:6.3f} {m['context']:8.3f} {m['context_articles']:8.3f} "
              f"{m['confident_context']:9.3f} {m['confident_context_articles']:11.3f} "
              f"{m['outside_passed']:>8d} / {m['outside_max'] if m['outside_max'] is not None else float('nan'):.3f}")

    saved = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
    saved.setdefault(questions.name, {}).update(results)
    RESULT.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nРезультат: {RESULT} [{questions.name}]")


if __name__ == "__main__":
    main()
