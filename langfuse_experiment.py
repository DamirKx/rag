"""Эксперимент в Langfuse: ответы системы на вопросы тестовой выборки с оценками качества.

Набор данных Langfuse — первые N вопросов test (data/adilet/splits/qa_{lang}_test.json) с эталонным ответом
и эталонными чанками. Каждый вопрос проходит ту же цепочку, что ask.py и eval_generation.py: гибрид e5 + BM25
по чанкам 500 символов, реранкер, top-3 чанка с окружающим текстом, Qwen и проверки ответа. Каждый прогон —
отдельный запуск эксперимента: в Langfuse (Datasets → набор → Runs) запуски можно сравнить, например
до и после изменения промпта или порогов.

Оценки каждого вопроса (без LLM-судьи — облачные оценщики правилами курсовой не разрешены):
- gold_in_context — эталонный чанк среди фрагментов, переданных модели (Hit@3 поиска);
- answer_shown    — ответ прошёл проверки и показан пользователю;
- token_f1, rouge_l — совпадение показанного ответа с эталонным (только для показанных ответов);
- answered_from_gold — ответ ссылается на фрагмент с эталонным чанком (только для показанных ответов).
Средние по запуску пишутся оценками запуска и в data/adilet/langfuse_experiment_{lang}.json.

Запуск — в окне, где заданы ключи Langfuse; нужна Ollama с qwen2.5:3b и свободная видеокарта
(остановите веб-демо, иначе Qwen не поместится в видеопамять):
    python langfuse_experiment.py                  # 50 вопросов, ~5 минут
    python langfuse_experiment.py --limit 20 --run-name "новый промпт"
    python langfuse_experiment.py --dry-run --limit 3   # проверка без ключей: в Langfuse ничего не уходит
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ask import RAGAnswerer, default_model
from run_adilet_eval import GoldChunkMatcher
from src.embeddings import EmbeddingModel
from src.eval import answer_rouge_l, answer_token_f1
from src.generation import DEFAULT_LLM, STATUS_ANSWERED, strip_citations
from src.reranker import CrossEncoderReranker, default_reranker
import src.tracing as tracing

CACHE_PARQUET = "data/raw/adilet_legal_qa_kz.parquet"
CACHE_DIR = "data/adilet/cache"
METRICS = ("gold_in_context", "answer_shown", "token_f1", "rouge_l", "answered_from_gold")


def load_items(lang: str, limit: int, rag: RAGAnswerer) -> List[Dict[str, Any]]:
    """Первые limit вопросов test: вопрос, эталонный ответ и эталонные чанки (как в eval_generation.py)."""
    questions = json.loads(Path(f"data/adilet/splits/qa_{lang}_test.json").read_text(encoding="utf-8"))[:limit]
    chunks = list(rag.chunks_by_id.values())
    matcher = GoldChunkMatcher([c["text"] for c in chunks], [c["id"].split("_fix_")[0] for c in chunks])
    return [
        {
            "id": q["id"],
            "input": {"question": q["query"]},
            "expected_output": q["full_response"],
            "metadata": {"question_id": q["id"], "law": q.get("source_title"),
                         "gold": [chunks[i]["id"] for i in matcher.find(q)[0]]},
        }
        for q in questions
    ]


def field(item: Any, name: str) -> Any:
    """Поле элемента: словарь (локальные данные) или DatasetItem (набор данных Langfuse)."""
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def make_task(rag: RAGAnswerer, lang: str):
    def task(*, item: Any, **kwargs: Any) -> Dict[str, Any]:
        question = field(item, "input")["question"]
        # Своя трасса внутри элемента эксперимента: те же этапы и оценки, что у запроса веб-демо
        with tracing.trace("experiment-question", input=question, metadata={"lang": lang}, tags=["experiment"]) as root:
            answer = rag.ask(question)
            root.update(output=answer.text, metadata={"lang": lang, "status": answer.status})
        return {
            "answer": answer.text if answer.answered else None,
            "status": answer.status,
            "problems": answer.checks.get("problems", []),
            "passages": [{"label": p.label, "chunk_ids": p.ids} for p in answer.passages],
            "cited": answer.cited if answer.answered else [],
        }

    return task


def evaluate(*, input: Any, output: Dict[str, Any], expected_output: Any, metadata: Optional[Dict[str, Any]], **kwargs: Any):
    """Оценки одного вопроса; ответа нет — оценки совпадения с эталоном не ставятся (как в eval_generation.py)."""
    from langfuse import Evaluation

    gold = set((metadata or {}).get("gold", []))
    passages = output["passages"]
    shown = output["status"] == STATUS_ANSWERED
    result = [
        Evaluation(name="gold_in_context", value=float(any(i in gold for p in passages for i in p["chunk_ids"])),
                   comment="эталонный чанк среди фрагментов, переданных модели"),
        Evaluation(name="answer_shown", value=float(shown), comment="; ".join(output["problems"]) or output["status"]),
    ]
    if shown and expected_output:
        prediction = strip_citations(output["answer"])
        cited = [passages[n - 1] for n in output["cited"] if 1 <= n <= len(passages)]
        result += [
            Evaluation(name="token_f1", value=round(answer_token_f1(prediction, expected_output), 4)),
            Evaluation(name="rouge_l", value=round(answer_rouge_l(prediction, expected_output), 4)),
            Evaluation(name="answered_from_gold", value=float(any(i in gold for p in cited for i in p["chunk_ids"])),
                       comment="ответ ссылается на фрагмент с эталонным чанком"),
        ]
    return result


def averages(item_results: List[Any]) -> Dict[str, Dict[str, float]]:
    """Среднее каждой оценки и по скольким вопросам оно посчитано."""
    values: Dict[str, List[float]] = {name: [] for name in METRICS}
    for result in item_results:
        for evaluation in getattr(result, "evaluations", None) or []:
            if evaluation.name in values:
                values[evaluation.name].append(float(evaluation.value))
    return {name: {"mean": round(sum(v) / len(v), 4), "n": len(v)} for name, v in values.items() if v}


def run_averages(*, item_results: List[Any], **kwargs: Any):
    """Оценки запуска: средние по всем вопросам (видны в сравнении запусков в Langfuse)."""
    from langfuse import Evaluation

    return [Evaluation(name=f"avg_{name}", value=m["mean"], comment=f"по {m['n']} вопросам") for name, m in averages(item_results).items()]


def ensure_dataset(client: Any, name: str, items: List[Dict[str, Any]], lang: str) -> Any:
    """Набор данных Langfuse; у элементов постоянные id, поэтому повторный запуск их не дублирует."""
    try:
        client.get_dataset(name)
    except Exception:
        client.create_dataset(name=name, description=f"Первые {len(items)} вопросов test Adilet Legal QA ({lang}) с эталонным ответом",
                              metadata={"lang": lang, "source": f"data/adilet/splits/qa_{lang}_test.json"})
    for item in items:
        client.create_dataset_item(dataset_name=name, id=f"{name}-{item['id']}", input=item["input"],
                                   expected_output=item["expected_output"], metadata=item["metadata"])
    client.flush()
    return client.get_dataset(name)


def dry_run_client() -> Any:
    """Клиент Langfuse, который ничего не отправляет: трассы — в память, оценки — в никуда."""
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    client = Langfuse(public_key="pk-dry-run", secret_key="sk-dry-run", base_url="http://127.0.0.1:9",
                      span_exporter=InMemorySpanExporter())
    client.create_score = lambda **kwargs: None
    from langfuse._client.span import LangfuseObservationWrapper

    LangfuseObservationWrapper.score_trace = lambda self, **kwargs: None
    LangfuseObservationWrapper.score = lambda self, **kwargs: None
    tracing._client, tracing._initialized = client, True
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперимент Langfuse на вопросах тестовой выборки.")
    parser.add_argument("--lang", choices=["rus", "kaz"], default="rus")
    parser.add_argument("--limit", type=int, default=50, help="Сколько первых вопросов test взять.")
    parser.add_argument("--run-name", default=None, help="Название запуска (по умолчанию — модель и время).")
    parser.add_argument("--dry-run", action="store_true", help="Без ключей: в Langfuse ничего не отправляется.")
    args = parser.parse_args()

    client = dry_run_client() if args.dry_run else tracing.get_langfuse()
    if client is None:
        sys.exit("Langfuse не настроен: задайте LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY и LANGFUSE_BASE_URL "
                 "(или запустите с --dry-run для проверки без отправки).")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name, reranker_name = default_model(args.lang), default_reranker(args.lang)
    print(f"Загрузка поиска: {model_name} (CPU), реранкер {reranker_name} ({device}), модель {DEFAULT_LLM}...")
    # e5 на CPU, как в веб-демо: иначе Qwen не помещается в 4 ГБ видеопамяти рядом с реранкером
    rag = RAGAnswerer(args.lang, EmbeddingModel(model_name=model_name, device="cpu"), model_name,
                      CrossEncoderReranker(reranker_name, device=device), reranker_name, CACHE_DIR, CACHE_PARQUET)
    items = load_items(args.lang, args.limit, rag)
    run_name = args.run_name or f"{DEFAULT_LLM} {datetime.now():%Y-%m-%d %H:%M}"
    dataset_name = f"adilet-test-{args.lang}-{args.limit}"
    metadata = {"llm": DEFAULT_LLM, "embedder": model_name, "reranker": reranker_name, "lang": args.lang}

    t0 = time.time()
    # По одному вопросу за раз: видеокарта одна, параллельные запросы только мешали бы друг другу
    common = dict(name=f"adilet-test-{args.lang}", run_name=run_name, task=make_task(rag, args.lang),
                  evaluators=[evaluate], run_evaluators=[run_averages], max_concurrency=1, metadata=metadata)
    if args.dry_run:
        result = client.run_experiment(data=items, **common)
    else:
        dataset = ensure_dataset(client, dataset_name, items, args.lang)
        result = dataset.run_experiment(**common)
    client.flush()

    summary = averages(result.item_results)
    elapsed = time.time() - t0
    print(f"\nЗапуск «{run_name}»: {len(result.item_results)} вопросов за {elapsed:.0f} с ({elapsed / max(len(items), 1):.1f} с на вопрос)")
    labels = {"gold_in_context": "эталонный чанк в контексте модели", "answer_shown": "ответ показан",
              "token_f1": "token F1 с эталоном", "rouge_l": "ROUGE-L с эталоном", "answered_from_gold": "ответ по эталонному фрагменту"}
    for name, m in summary.items():
        print(f"  {labels[name]:34s} {m['mean']:.3f}  (по {m['n']} вопросам)")
    if not args.dry_run:
        out = Path(f"data/adilet/langfuse_experiment_{args.lang}.json")
        runs = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        runs.append({"run_name": run_name, "dataset": dataset_name, "questions": len(items), "seconds": round(elapsed),
                     "metrics": summary, **metadata, "date": f"{datetime.now():%Y-%m-%d %H:%M}"})
        out.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Итоги сохранены в {out}; в Langfuse: Datasets → {dataset_name} → Runs.")


if __name__ == "__main__":
    main()
