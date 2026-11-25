"""Показывает трассу Langfuse в консоли: как вопрос превращается в вектор и что находит поиск.

Читает уже записанные трассы (веб-демо app.py, консольный ask.py) через API Langfuse и печатает
этапы по порядку: embed-query (модель эмбеддингов, размерность вектора), dense-faiss (индекс и мера
близости, топ по косинусу), bm25, rrf (слияние), reranker, ответ модели и проверки.

Нужны те же ключи, что и для трассировки:
    $env:LANGFUSE_PUBLIC_KEY="pk-lf-..."; $env:LANGFUSE_SECRET_KEY="sk-lf-..."
    $env:LANGFUSE_BASE_URL="https://cloud.langfuse.com"

Примеры:
    python show_trace.py                 # список последних 10 трасс
    python show_trace.py --last          # подробно последняя трасса
    python show_trace.py --trace-id ID   # подробно трасса по id (id есть в списке и в адресе трассы в Langfuse)
    python show_trace.py --last --full   # без сокращения входов и выходов этапов
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

FIELDS = "core,basic,io,metadata,model,metrics,trace_context"
# Этапы, ради которых обычно смотрят трассу: путь вопроса до найденных фрагментов
NOISE_PREFIXES = ("scope.", "resourceAttributes.")
SEARCH_STEPS = {"hybrid", "embed-query", "dense-faiss", "faiss", "bm25", "rrf", "reranker", "search", "law-filter"}


def client() -> Any:
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        sys.exit("Не заданы LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY (см. README, раздел «Трассировка»).")
    from langfuse import Langfuse

    return Langfuse()


def parse(value: Any) -> Any:
    """Вход и выход этапа API отдаёт строкой JSON — разбираем, если получится."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def short(value: Any, limit: int) -> str:
    text = json.dumps(parse(value), ensure_ascii=False, default=str)
    return text if limit <= 0 or len(text) <= limit else text[:limit] + " …"


def list_traces(lf: Any, n: int) -> List[Any]:
    """Корневые этапы последних трасс (одна запись на трассу), новые сверху."""
    resp = lf.api.observations.get_many(is_root_observation=True, limit=n, fields=FIELDS)
    return sorted(resp.data, key=lambda o: o.start_time, reverse=True)


def observations(lf: Any, trace_id: str) -> List[Any]:
    items, cursor = [], None
    while True:
        resp = lf.api.observations.get_many(trace_id=trace_id, fields=FIELDS, limit=1000, cursor=cursor,
                                            expand_metadata="index,model,n_docs,dim,device")
        items.extend(resp.data)
        cursor = getattr(resp.meta, "cursor", None)
        if not cursor:
            return items


def print_list(roots: List[Any]) -> None:
    if not roots:
        print("Трасс нет. Задайте вопрос в веб-демо (streamlit run app.py) или в ask.py с ключами Langfuse.")
        return
    for o in roots:
        question = parse(o.input)
        if isinstance(question, dict):
            question = question.get("query") or question.get("question") or question
        print(f"{o.start_time:%d.%m %H:%M:%S}  {o.trace_id}  {o.trace_name or o.name}: {short(question, 90)}")


def print_tree(obs: List[Any], limit: int, only_search: bool) -> None:
    by_parent: Dict[Optional[str], List[Any]] = {}
    for o in obs:
        by_parent.setdefault(o.parent_observation_id, []).append(o)
    for children in by_parent.values():
        children.sort(key=lambda o: o.start_time)

    def walk(parent: Optional[str], depth: int) -> None:
        for o in by_parent.get(parent, []):
            show = not only_search or depth == 0 or o.name in SEARCH_STEPS
            if show:
                pad = "  " * depth
                ms = f"{o.latency * 1000:.0f} мс" if o.latency is not None else ""
                print(f"{pad}▸ {o.name} [{(o.type or '').lower()}] {ms}")
                if o.model:
                    print(f"{pad}    модель: {o.model}")
                meta = parse(o.metadata)
                if isinstance(meta, dict):
                    # Служебные поля OpenTelemetry (версия SDK, ключ проекта) одинаковы у всех этапов
                    meta = {k: v for k, v in meta.items() if not k.startswith(NOISE_PREFIXES)}
                if meta:
                    print(f"{pad}    metadata: {short(meta, limit)}")
                if o.input is not None:
                    print(f"{pad}    вход: {short(o.input, limit)}")
                if o.output is not None:
                    print(f"{pad}    выход: {short(o.output, limit)}")
            walk(o.id, depth + 1)

    walk(None, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace-id", help="id трассы")
    parser.add_argument("--last", action="store_true", help="подробно показать последнюю трассу")
    parser.add_argument("-n", type=int, default=10, help="сколько трасс в списке (по умолчанию 10)")
    parser.add_argument("--full", action="store_true", help="не сокращать вход и выход этапов")
    parser.add_argument("--search-only", action="store_true", help="только этапы поиска (вектор, FAISS, BM25, RRF, реранкер)")
    args = parser.parse_args()

    lf = client()
    trace_id = args.trace_id
    if not trace_id:
        roots = list_traces(lf, 1 if args.last else args.n)
        if not args.last or not roots:
            print_list(roots)
            return
        trace_id = roots[0].trace_id

    obs = observations(lf, trace_id)
    if not obs:
        sys.exit(f"Трасса {trace_id} не найдена (новая трасса появляется в API через несколько секунд).")
    print(f"Трасса {trace_id}\n")
    print_tree(obs, 0 if args.full else 400, args.search_only)


if __name__ == "__main__":
    main()
