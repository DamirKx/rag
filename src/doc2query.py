"""doc2query: заранее написанные разговорные вопросы к статьям как дополнительные «ключи» поиска.

Разговорный вопрос («отчислили из колледжа, когда съезжать из общаги») часто не совпадает с языком закона
(«выселение в связи с прекращением обучения») ни по словам (BM25), ни по смыслу достаточно близко (e5). Если к
чанку статьи заранее приписаны вопросы, которые к ней обычно задают, поиск находит статью через них.

Вопросы — data/laws/reranker_colloquial/questions_generated.jsonl (написаны Claude по текстам статей, с дословной
цитатой ответа; см. prepare_reranker_colloquial.py). В работающей системе языковая модель не участвует: вопросы —
подготовленные данные индекса, как сам текст законов. Реранкер и модель ответа по-прежнему получают только текст
закона, поэтому вопросы помогают найти статью, но не влияют на оценку уверенности и на ответ.

Два способа:
- «append» — вопросы дописываются к тексту чанка в индексе (классический doc2query);
- «separate» — каждый вопрос отдельный короткий документ «<id чанка>#qN»; найденный вопрос заменяется своим чанком.

Утечки. Проверочные вопросы тоже черновики Claude, поэтому среди сгенерированных встречаются их пересказы
(«отчислили из универа, должен ли съехать из общаги» ↔ iq30). Такой вопрос подсказывал бы ответ, поэтому убирается
сгенерированный вопрос к той же статье, что и проверочный, если их сходство по символьным n-граммам ≥ порога.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

GENERATED = Path("data/laws/reranker_colloquial/questions_generated.jsonl")
QUESTION_SEP = "#q"
LEAK_STRICT = 0.2    # строгий фильтр: пессимистичная оценка
LEAK_MODERATE = 0.3  # умеренный: убираются явные пересказы

ArticleKey = Tuple[str, str]


def load_generated(path: Path = GENERATED) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def leak_filter(records: Sequence[Dict[str, Any]], eval_questions: Sequence[Tuple[str, Set[ArticleKey]]],
                record_keys: Sequence[Set[ArticleKey]], threshold: float
                ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Делит сгенерированные вопросы на оставленные и убранные как пересказы проверочных.

    eval_questions — (текст, статьи-эталоны); record_keys[i] — статьи, к которым относится records[i].
    Убирается вопрос, у которого есть проверочный с общей статьёй и косинусом TF-IDF по символьным
    3–5-граммам не ниже threshold. Вопросы вне корпуса (без эталона) ничего не убирают.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not records or not eval_questions:
        return list(records), []
    gen_texts = [r["question"] for r in records]
    eval_texts = [q for q, _ in eval_questions]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(gen_texts + eval_texts)
    sim = cosine_similarity(vec.transform(eval_texts), vec.transform(gen_texts))
    kept, removed = [], []
    for j, rec in enumerate(records):
        match = max(((float(sim[i, j]), eval_texts[i]) for i, (_, gold) in enumerate(eval_questions)
                     if gold & record_keys[j]), default=(0.0, ""))
        if match[0] >= threshold:
            removed.append({**rec, "leak_eval": match[1], "leak_sim": round(match[0], 3)})
        else:
            kept.append(rec)
    return kept, removed


def expansions(records: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Вопросы по чанкам: чанк, в котором начинается цитата-ответ (поле positive)."""
    out: Dict[str, List[str]] = {}
    for rec in records:
        out.setdefault(rec["positive"], []).append(rec["question"].strip())
    return out


def append_questions(text: str, questions: Sequence[str]) -> str:
    """Текст чанка для индекса с дописанными вопросами («append»)."""
    return text if not questions else text + "\n" + "\n".join(questions)


def question_docs(by_chunk: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """Отдельные документы-вопросы («separate»): тексты и id вида «<id чанка>#qN»."""
    texts, ids = [], []
    for chunk_id, questions in by_chunk.items():
        for n, q in enumerate(questions, 1):
            texts.append(q)
            ids.append(f"{chunk_id}{QUESTION_SEP}{n}")
    return texts, ids


def chunk_of(doc_id: str) -> str:
    """Id чанка для найденного документа: у вопроса — чанк, к которому он приписан."""
    return doc_id.split(QUESTION_SEP, 1)[0]


def collapse(hits: Sequence[Tuple[str, Any, float]], k: int) -> List[Tuple[str, float]]:
    """Найденные документы → чанки без повторов в порядке первого появления, k первых (с оценкой лучшего)."""
    seen: Set[str] = set()
    out: List[Tuple[str, float]] = []
    for doc_id, _, score in hits:
        cid = chunk_of(doc_id)
        if cid not in seen:
            seen.add(cid)
            out.append((cid, score))
            if len(out) == k:
                break
    return out
