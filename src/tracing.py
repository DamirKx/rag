"""Трассировка ответов в Langfuse: поиск, промпт, ответ модели и результаты проверок по каждому запросу.

Включается, только если заданы переменные окружения LANGFUSE_PUBLIC_KEY и LANGFUSE_SECRET_KEY
(адрес сервера — LANGFUSE_BASE_URL). Без них данные никуда не отправляются. Ошибка трассировки
не должна мешать ответу, поэтому здесь она только печатается.

Трассируются ответы веб-демо и ask.py (GroundedAnswerer.answer). Оценка eval_generation.py сюда
не попадает: она вызывает модель и проверки напрямую, иначе в Langfuse ушли бы тысячи записей.
"""

import os
from typing import Any, Dict, List, Optional

_client: Any = None
_initialized = False


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


def trace_answer(question: str, answer: Any, messages: List[Dict[str, str]], model: Optional[str], lang: str) -> None:
    """Записывает один запрос: корневая запись и вложенные поиск, модель (если вызывалась) и проверка."""
    client = get_langfuse()
    if client is None:
        return
    try:
        passages = [
            {"id": p.id, "chunk_ids": p.ids, "label": p.label, "score": p.score, "text": p.text}
            for p in answer.passages
        ]
        usage = answer.usage or {}
        with client.start_as_current_observation(
            name="rag-answer", as_type="chain", input=question, output=answer.text,
            metadata={"status": answer.status, "lang": lang, "cited": answer.cited},
        ):
            with client.start_as_current_observation(name="retrieval", as_type="retriever", input=question, output=passages):
                pass
            if answer.raw:
                details = {
                    name: usage[key]
                    for name, key in (("input", "prompt_tokens"), ("output", "completion_tokens"))
                    if isinstance(usage.get(key), int)
                }
                with client.start_as_current_observation(
                    name="llm", as_type="generation", model=model, input=messages, output=answer.raw,
                    usage_details=details,
                    metadata={"latency_s": usage.get("latency_s"), "cached": usage.get("cached"), "retried": usage.get("retried")},
                ):
                    pass
            with client.start_as_current_observation(name="verification", as_type="guardrail", input=answer.raw, output=answer.checks):
                pass
    except Exception as e:
        print(f"Не удалось записать трассировку Langfuse: {e}")
