"""Трассировка запросов в Langfuse: этапы поиска и ответа модели, у каждого своё время.

Включается, только если заданы переменные окружения LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY
(адрес сервера — LANGFUSE_BASE_URL). Без них данные никуда не отправляются. Ошибка трассировки
не должна мешать ответу, поэтому здесь она только печатается.

Устройство:
- trace(...) открывает трассу одного запроса (веб-демо app.py, консольный ask.py);
- span(...) — вложенный этап (BM25, FAISS, реранкер, модель, проверки). Начало и конец этапа
  Langfuse записывает сам, поэтому на временной шкале трассы видно, сколько занял каждый этап.

Этапы записываются, только если трасса открыта. Оценка (eval_generation.py, tune_*.py) вызывает
те же функции поиска и проверки без трассы, поэтому в Langfuse не уходят тысячи записей.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, List, Optional, Sequence, Tuple

_client: Any = None
_initialized = False
# Открыта ли трасса в текущем потоке (у каждого сеанса Streamlit свой поток)
_active: ContextVar[bool] = ContextVar("langfuse_trace_active", default=False)

# Сколько результатов поиска записывать в этап (пул после фильтра по законам бывает до 1200)
MAX_LOGGED_HITS = 20


def get_langfuse() -> Any:
    """Клиент Langfuse или None, если ключи не заданы либо клиент не удалось создать."""
    global _client, _initialized
    if _initialized:
        return _client
    _initialized = True
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        try:
            from langfuse import Langfuse

            _client = Langfuse()
        except Exception as e:
            print(f"Langfuse не подключён: {e}")
    return _client


def _warn(e: Exception) -> None:
    print(f"Не удалось записать трассировку Langfuse: {e}")


class Observation:
    """Этап трассы. Если трассировка выключена, методы ничего не делают; ошибки Langfuse печатаются."""

    def __init__(self, obs: Any = None):
        self._obs = obs

    @property
    def enabled(self) -> bool:
        return self._obs is not None

    def update(self, **kwargs: Any) -> None:
        """Дописывает output, metadata, level, usage_details и т. п. (как LangfuseSpan.update)."""
        if self._obs is None:
            return
        try:
            self._obs.update(**kwargs)
        except Exception as e:
            _warn(e)


@contextmanager
def _guarded(factory) -> Iterator[Any]:
    """Входит в контекст Langfuse; сбой трассировки печатается, а исключение кода внутри идёт дальше."""
    cm = obs = None
    try:
        cm = factory()
        obs = cm.__enter__()
    except Exception as e:
        _warn(e)
        cm = obs = None
    try:
        yield obs
    except BaseException as e:
        if cm is not None:
            try:
                cm.__exit__(type(e), e, e.__traceback__)
            except Exception as exit_error:
                _warn(exit_error)
        raise
    else:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception as e:
                _warn(e)


@contextmanager
def trace(
    name: str,
    input: Any = None,
    metadata: Optional[dict] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Iterator[Observation]:
    """Трасса одного запроса. session_id объединяет запросы одной вкладки веб-демо или одного запуска ask.py."""
    client = get_langfuse()
    if client is None or _active.get():
        # Без ключей — ничего; трасса внутри трассы — обычный этап
        with span(name, as_type="chain", input=input, metadata=metadata) as obs:
            yield obs
        return
    token = _active.set(True)
    try:
        with _guarded(lambda: client.start_as_current_observation(name=name, as_type="chain", input=input, metadata=metadata)) as root:
            attributes = {"session_id": session_id, "tags": tags, "trace_name": name}
            if metadata:
                # В атрибуты трассы идут только строки (так требует Langfuse) — для фильтров в списке трасс
                attributes["metadata"] = {k: str(v) for k, v in metadata.items() if v is not None}
            with _guarded(lambda: _propagate(**{k: v for k, v in attributes.items() if v})) if root is not None else _noop():
                yield Observation(root)
    finally:
        _active.reset(token)


def _propagate(**kwargs: Any) -> Any:
    from langfuse import propagate_attributes

    return propagate_attributes(**kwargs)


@contextmanager
def _noop() -> Iterator[None]:
    yield None


@contextmanager
def span(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Observation]:
    """Этап внутри открытой трассы (as_type: span, retriever, embedding, generation, guardrail, chain...).

    kwargs — поля Langfuse: input, output, metadata, model, model_parameters, usage_details, level.
    """
    client = get_langfuse() if _active.get() else None
    if client is None:
        yield Observation()
        return
    with _guarded(lambda: client.start_as_current_observation(name=name, as_type=as_type, **kwargs)) as obs:
        yield Observation(obs)


def tracing_active() -> bool:
    """Открыта ли сейчас трасса: тогда стоит готовить данные для этапов (иначе это лишняя работа)."""
    return _active.get() and get_langfuse() is not None


def brief_hits(pairs: Sequence[Tuple[str, float]], n: int = MAX_LOGGED_HITS) -> dict:
    """Первые n результатов поиска (doc_id, оценка) и их общее число — для output этапа."""
    return {
        "total": len(pairs),
        "top": [{"rank": i, "id": doc_id, "score": round(float(score), 4)} for i, (doc_id, score) in enumerate(pairs[:n], 1)],
    }
