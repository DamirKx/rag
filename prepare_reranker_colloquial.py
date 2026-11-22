"""Обучающие данные для дообучения реранкера на разговорных вопросах к корпусу законов (Colab:
notebooks/finetune_reranker_colloquial.ipynb).

Проверка на 100 независимых вопросах показала: нужная статья часто есть среди кандидатов, но исходный реранкер
bge-reranker-v2-m3 плохо оценивает пары «вопрос обычными словами — текст закона». Вопросы для обучения написал
Claude по текстам статей корпуса (по 1–2 на статью, с дословной цитатой ответа); проверочные наборы вопросов
генераторам не показывались. Языковая модель в работающей системе не участвует — она лишь подготовила данные.

Что делает скрипт:
1. Собирает сгенерированные вопросы, оставляет те, чья цитата дословно есть в тексте статьи.
2. Убирает вопросы, похожие на проверочные (data/laws/colloquial_questions*.json): доля общих основ слов ≥ LEAK_JACCARD.
3. Положительный пример — чанк статьи, в котором начинается цитата (реранкер в работе оценивает чанки 500 символов).
4. Отрицательные примеры — чанки ДРУГИХ статей из топ-30 нынешнего поиска (гибрид + словарь), как в
   notebooks/finetune_reranker.ipynb; чанки, содержащие большую часть цитаты, отбрасываются как вероятные
   ложные отрицательные; недостающие — случайные чанки того же закона.
5. Делит сгенерированные вопросы на train / val по статьям (val — для контроля переобучения).
6. Для выбора лучшей точки обучения — настоящие разговорные вопросы, по которым система не проверяется итогово:
   colloquial_questions_independent.json (50) и colloquial_questions.json (59). 100 независимых вопросов в Colab
   не попадают — на них итоговая проверка.

Вопрос в парах — с терминами словаря (src/query_expansion.py), как его получает реранкер в веб-демо.

Запуск (видеокарта ускоряет поиск, но не обязательна):
    python prepare_reranker_colloquial.py --generated <папка с out_*.jsonl>
Результат: data/laws/reranker_colloquial/questions_generated.jsonl (в git) и
data/laws/reranker_colloquial/colab/*.jsonl + reranker_colloquial_data.zip (загрузить на Google Drive в MyDrive/rag/).
"""

import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ask import LegalSearchEngine, default_model, load_bm25_params, load_hybrid_params
from run_adilet_eval import model_cache_name
from src.corpus import CHUNK_OVERLAP, CHUNK_SIZE, build_chunk_corpus
from src.embeddings import EmbeddingModel
from src.laws import load_laws
from src.query_expansion import expand_query
from src.retriever import tokenize_legal_text
from src.text_norm import get_normalizer

OUT = Path("data/laws/reranker_colloquial")
COLAB = OUT / "colab"
EVAL_SETS = ["data/laws/colloquial_questions_100.json", "data/laws/colloquial_questions_independent.json",
             "data/laws/colloquial_questions.json"]
DEV_SETS = ["data/laws/colloquial_questions_independent.json", "data/laws/colloquial_questions.json"]
SEED = 42
VAL_SHARE = 0.1        # доля статей, вопросы к которым идут в val
LEAK_JACCARD = 0.5     # вопрос с такой долей общих основ слов с проверочным — убирается
NEG_DEPTH = 30         # отрицательные примеры — из топ-30 поиска
N_NEG = 7
NEG_MAX_QUOTE_COV = 0.6  # чанк другой статьи с такой долей основ слов цитаты — вероятный ложный отрицательный
EVAL_POOL = 20         # кандидатов для оценки — как пул реранкера в веб-демо
CACHE_DIR = "data/adilet/cache"
LANG = "rus"

stem = get_normalizer("snowball")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def stems(text: str) -> Set[str]:
    return set(stem([t for t in tokenize_legal_text(norm(text)) if len(t) >= 3]))


def jaccard(a: Set[str], b: Set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def main() -> None:
    generated_dir = Path(sys.argv[sys.argv.index("--generated") + 1])
    rng = random.Random(SEED)
    corpus = build_chunk_corpus(load_laws(), {})
    by_id = {c["id"]: c for c in corpus}
    chunks_of: Dict[str, List[str]] = defaultdict(list)  # статья (фрагмент) → её чанки по порядку
    for c in corpus:
        chunks_of[c["id"].split("_fix_")[0]].append(c["id"])
    article_text = {frag: by_id[ids[0]]["context"] for frag, ids in chunks_of.items()}

    # 1. Сгенерированные вопросы с проверенной цитатой
    stats = Counter()
    records = []
    # Только итоговые файлы out_NN.jsonl: генераторы пишут и промежуточные части (out_02_p1.jsonl)
    for path in sorted(generated_dir.glob("out_[0-9][0-9].jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["не JSON"] += 1
                continue
            frag, q, quote = rec.get("id"), str(rec.get("q", "")).strip(), str(rec.get("quote", "")).strip()
            if frag not in article_text or not q:
                stats["нет статьи или вопроса"] += 1
                continue
            if norm(quote.rstrip(" .…")) not in norm(article_text[frag]) or len(quote) < 15:
                stats["цитата не найдена в статье"] += 1
                continue
            records.append({"article": frag, "law": by_id[chunks_of[frag][0]]["law"],
                            "article_title": by_id[chunks_of[frag][0]]["article_title"],
                            "kind": rec.get("kind", ""), "question": q, "quote": quote})
    stats["вопросов с проверенной цитатой"] = len(records)

    # 2. Похожие на проверочные — убрать
    eval_questions = [(Path(f).name, q["question"]) for f in EVAL_SETS
                      for q in json.loads(Path(f).read_text(encoding="utf-8"))["questions"]]
    eval_stems = [(name, text, stems(text)) for name, text in eval_questions]
    kept, leaks = [], []
    for r in records:
        s = stems(r["question"])
        match = max(((jaccard(s, es), name, text) for name, text, es in eval_stems), default=(0, "", ""))
        if match[0] >= LEAK_JACCARD:
            leaks.append({"generated": r["question"], "eval_set": match[1], "eval": match[2], "jaccard": round(match[0], 3)})
        else:
            kept.append(r)
    records = kept
    stats["убрано как похожие на проверочные"] = len(leaks)

    # 3. Положительный чанк — где начинается цитата
    for r in records:
        head = norm(r["quote"])[:60]
        ids = chunks_of[r["article"]]
        owner = next((i for i in ids if head in norm(by_id[i]["text"])), None)
        if owner is None:  # цитата на границе чанков: чанк с наибольшей долей её основ
            qs = stems(r["quote"])
            owner = max(ids, key=lambda i: len(qs & stems(by_id[i]["text"])))
        r["positive"] = owner

    # 5. train / val по статьям
    articles = sorted({r["article"] for r in records})
    rng.shuffle(articles)
    val_articles = set(articles[:round(len(articles) * VAL_SHARE)])
    for r in records:
        r["split"] = "val" if r["article"] in val_articles else "train"

    # 4. Поиск: кандидаты для отрицательных примеров и оценки (гибрид + словарь, как в веб-демо)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = default_model(LANG)
    engine = LegalSearchEngine(
        docs=[c["text"] for c in corpus], doc_ids=[c["id"] for c in corpus],
        embedder=EmbeddingModel(model_name=model_name, device=device), cache_dir=CACHE_DIR,
        cache_key=f"{model_cache_name(model_name)}_{LANG}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}_laws",
        hybrid_params=load_hybrid_params(model_name, LANG), bm25_params=load_bm25_params(LANG),
    )
    chunks_of_law: Dict[str, List[str]] = defaultdict(list)
    for c in corpus:
        chunks_of_law[c["law"]].append(c["id"])
    chunk_stems = {c["id"]: stems(c["text"]) for c in corpus}

    pairs = {"train": [], "val": []}
    val_samples = []
    for n, r in enumerate(records, 1):
        query = expand_query(r["question"])
        r["query"] = query
        hits = [h[0] for h in engine.search(query, mode="hybrid", top_k=NEG_DEPTH)]
        own = set(chunks_of[r["article"]])
        quote_stems = stems(r["quote"])
        pool = []
        for i in hits:
            if i in own:
                continue
            cov = len(quote_stems & chunk_stems[i]) / len(quote_stems) if quote_stems else 0.0
            if cov >= NEG_MAX_QUOTE_COV:
                stats["отброшено как вероятный ложный отрицательный"] += 1
                continue
            pool.append(i)
        negs = rng.sample(pool, min(N_NEG, len(pool)))
        if len(negs) < N_NEG:
            stats["вопросов с добором случайными"] += 1
            other = [i for i in chunks_of_law[r["law"]] if i not in own and i not in negs]
            negs += rng.sample(other, min(N_NEG - len(negs), len(other)))
        rows = [{"query": query, "passage": by_id[r["positive"]]["text"], "label": 1}]
        rows += [{"query": query, "passage": by_id[i]["text"], "label": 0} for i in negs]
        pairs[r["split"]].extend(rows)
        if r["split"] == "val":
            val_samples.append({"query": query, "positive": [by_id[i]["text"] for i in own],
                                "documents": [by_id[i]["text"] for i in hits[:EVAL_POOL]]})
        if n % 500 == 0:
            print(f"  {n}/{len(records)}")

    # 6. Настоящие разговорные вопросы для выбора лучшей точки (без 100 итоговых)
    dev_samples = []
    for f in DEV_SETS:
        for q in json.loads(Path(f).read_text(encoding="utf-8"))["questions"]:
            if not q["gold"]:
                continue
            wanted = {(g["law"], g["article"]) for g in q["gold"]}
            gold_ids = [c["id"] for c in corpus
                        if any((c["law"], a) in wanted for a in re.findall(r"\d+(?:-\d+)?", c.get("articles", "")))]
            query = expand_query(q["question"])
            hits = [h[0] for h in engine.search(query, mode="hybrid", top_k=EVAL_POOL)]
            dev_samples.append({"set": Path(f).name, "id": q["id"], "query": query,
                                "positive": [by_id[i]["text"] for i in gold_ids],
                                "documents": [by_id[i]["text"] for i in hits]})

    OUT.mkdir(parents=True, exist_ok=True)
    COLAB.mkdir(parents=True, exist_ok=True)
    with (OUT / "questions_generated.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in ("article", "law", "article_title", "kind", "question", "quote",
                                                   "positive", "split")}, ensure_ascii=False) + "\n")
    (OUT / "leaks_removed.json").write_text(json.dumps(leaks, ensure_ascii=False, indent=1), encoding="utf-8")
    files = {
        "train_pairs.jsonl": pairs["train"],
        "val_pairs.jsonl": pairs["val"],
        "val_generated_samples.jsonl": val_samples,
        "dev_samples.jsonl": dev_samples,
    }
    for name, rows in files.items():
        with (COLAB / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"stats": dict(stats), "questions": len(records), "articles": len(articles),
               "train_questions": sum(r["split"] == "train" for r in records),
               "val_questions": sum(r["split"] == "val" for r in records),
               "train_pairs": len(pairs["train"]), "val_pairs": len(pairs["val"]),
               "dev_samples": len(dev_samples), "kinds": dict(Counter(r["kind"] for r in records)),
               "laws": dict(Counter(r["law"] for r in records)),
               "params": {"seed": SEED, "val_share": VAL_SHARE, "leak_jaccard": LEAK_JACCARD, "neg_depth": NEG_DEPTH,
                          "n_neg": N_NEG, "neg_max_quote_cov": NEG_MAX_QUOTE_COV, "eval_pool": EVAL_POOL}}
    (COLAB / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    archive = COLAB / "reranker_colloquial_data.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in [*files, "summary.json"]:
            z.write(COLAB / name, arcname=f"reranker_colloquial/{name}")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nАрхив для Google Drive: {archive} ({archive.stat().st_size / 2**20:.1f} МБ)")


if __name__ == "__main__":
    main()
