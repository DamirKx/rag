"""Трассировка Langfuse: этапы поиска и ответа записываются внутри трассы и не меняют результаты.

Спаны пишутся в память (InMemorySpanExporter), в сеть ничего не уходит.
"""

import builtins
import json
import uuid

import pandas as pd
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import ask
import src.reranker as reranker_module
import src.tracing as tracing
from ask import LegalSearchEngine
from src.generation import STATUS_ANSWERED, GroundedAnswerer, Passage
from src.reranker import CrossEncoderReranker
from src.retriever import tokenize_legal_text

DOCS = [
    "Земельный налог уплачивается ежегодно до 1 апреля.",
    "Образовательный грант выдаётся военнослужащим.",
    "Порядок регистрации прав на недвижимость.",
    "Ставка земельного налога устанавливается местным органом.",
]
DOC_IDS = [f"c{i}" for i in range(len(DOCS))]


class FakeCrossEncoder:
    """Оценка пары — число общих слов запроса и документа."""

    def __init__(self, *args, **kwargs):
        pass

    def half(self):
        return self

    def to(self, device):
        return self

    def predict(self, pairs, **kwargs):
        return [len(set(tokenize_legal_text(q)) & set(tokenize_legal_text(d))) for q, d in pairs]


class FakeClient:
    """Модель Ollama: отдаёт ответы по очереди."""

    model = "fake-llm"
    options = {"temperature": 0.0}

    def __init__(self, *texts):
        self.texts = list(texts)

    def chat(self, messages):
        return {"text": self.texts.pop(0), "prompt_tokens": 10, "completion_tokens": 5, "latency_s": 0.2,
                "load_s": 0.01, "eval_s": 0.1, "cached": False}


@pytest.fixture
def exporter(monkeypatch):
    """Настоящий клиент Langfuse, который пишет спаны в память."""
    memory = InMemorySpanExporter()
    client = tracing_client(memory)
    monkeypatch.setattr(tracing, "_client", client)
    monkeypatch.setattr(tracing, "_initialized", True)
    memory.flush = client.flush
    # Оценки уходят в Langfuse по HTTP, а не спанами: в тестах они перехватываются, иначе клиент
    # пытался бы отправить их на несуществующий сервер
    from langfuse._client.span import LangfuseObservationWrapper

    memory.scores = {}

    def fake_score_trace(self, *, name, value, data_type=None, comment=None, **kwargs):
        memory.scores[name] = (value, data_type, comment)

    monkeypatch.setattr(LangfuseObservationWrapper, "score_trace", fake_score_trace)
    return memory


def tracing_client(memory):
    from langfuse import Langfuse

    # У каждого теста свой ключ: Langfuse держит по одному клиенту на ключ
    return Langfuse(public_key=f"pk-test-{uuid.uuid4().hex}", secret_key="sk-test", base_url="http://127.0.0.1:9",
                    span_exporter=memory)


def spans(memory):
    memory.flush()
    return {s.name: s for s in memory.get_finished_spans()}


def attr(span, key):
    """Значение поля этапа; Langfuse хранит input/output/metadata как JSON-строки."""
    value = span.attributes.get(f"langfuse.observation.{key}")
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


@pytest.fixture
def engine(fake_embedder, tmp_path, monkeypatch):
    monkeypatch.setattr(reranker_module, "CrossEncoder", FakeCrossEncoder)
    return LegalSearchEngine(
        docs=DOCS, doc_ids=DOC_IDS, embedder=fake_embedder, cache_dir=str(tmp_path), cache_key="t",
        reranker=CrossEncoderReranker("fake-reranker"), rerank_pool=4,
    )


def passages():
    return [Passage(id=i, text=t, law="Налоговый кодекс", articles="ст. 5", score=0.9) for i, t in zip(DOC_IDS[:2], DOCS[:2])]


def test_no_spans_without_trace(exporter, engine):
    """Оценка и подбор параметров вызывают те же функции без трассы — в Langfuse ничего не уходит."""
    engine.search("земельный налог", mode="hybrid", top_k=2, rerank=True)
    GroundedAnswerer(FakeClient("Налог уплачивается ежегодно [1]."), min_retrieval_score=None,
                     min_answer_relevance=None).answer("Когда уплачивается налог?", passages())
    assert spans(exporter) == {}


def test_search_and_answer_stages(exporter, engine):
    with tracing.trace("request", input="налог", session_id="s1", tags=["test"]):
        hits = engine.search("ставка земельного налога", mode="hybrid", top_k=2, rerank=True)
        answerer = GroundedAnswerer(
            FakeClient("Налог уплачивается ежегодно до 1 апреля [1]."), min_retrieval_score=0.5,
            min_answer_relevance=0.1, relevance_scorer=lambda q, a: 0.8,
        )
        answer = answerer.answer("Когда уплачивается земельный налог?", passages())
    assert answer.status == STATUS_ANSWERED and hits[0][0] in ("c0", "c3")

    found = spans(exporter)
    expected = {
        "request", "hybrid", "embed-query", "bm25-index-build", "dense-faiss", "bm25", "rrf", "reranker",
        "answer", "retrieval-gate", "llm", "verification", "citations", "numbers", "sentence-support", "relevance",
    }
    assert expected <= set(found)
    # Всё в одной трассе с общей сессией
    assert len({s.context.trace_id for s in found.values()}) == 1
    assert all(s.attributes.get("session.id") == "s1" for s in found.values())
    # Вложенность: этапы гибрида внутри hybrid, модель и проверки внутри answer
    parent = lambda name: next(s.name for s in found.values() if s.context.span_id == found[name].parent.span_id)
    assert parent("dense-faiss") == parent("bm25") == parent("rrf") == "hybrid"
    assert parent("hybrid") == parent("reranker") == "request"
    assert parent("llm") == parent("verification") == parent("retrieval-gate") == "answer"
    assert parent("numbers") == parent("sentence-support") == "verification"
    # У каждого этапа своё время
    assert all(s.end_time >= s.start_time for s in found.values())

    assert attr(found["llm"], "type") == "generation"
    assert attr(found["llm"], "usage_details") == {"input": 10, "output": 5}
    assert attr(found["llm"], "metadata.load_s") == 0.01
    reranked = attr(found["reranker"], "output")
    assert reranked[0]["rank"] == 1 and "rank_before" in reranked[0]
    assert attr(found["rrf"], "output")["top"][0].keys() >= {"dense_rank", "bm25_rank"}
    assert attr(found["retrieval-gate"], "output") == {"passed": True}
    assert attr(found["sentence-support"], "output")[0]["support"] == 1.0
    assert attr(found["verification"], "output") == {"status": STATUS_ANSWERED, "problems": []}


def test_llm_stage_has_speed_cost_and_placement(exporter, monkeypatch):
    import src.generation as generation

    client = FakeClient("Налог уплачивается ежегодно [1].")
    client.url = "http://ollama"
    monkeypatch.setattr(generation, "ollama_placement", lambda url, model: {"ollama_on_gpu": 0.26, "ollama_vram_mb": 630})
    with tracing.trace("request"):
        GroundedAnswerer(client, min_retrieval_score=None, min_answer_relevance=None).answer("Когда?", passages())
    llm = spans(exporter)["llm"]
    assert attr(llm, "metadata.output_tokens_per_s") == 50.0  # 5 токенов за 0,1 с
    assert attr(llm, "cost_details") == {"input": 0.0, "output": 0.0, "total": 0.0}
    assert attr(llm, "metadata.ollama_on_gpu") == 0.26
    # Часть модели на процессоре — ответ медленнее, этап помечен предупреждением
    assert attr(llm, "level") == "WARNING"


@pytest.fixture
def recorded_scores(exporter):
    return exporter.scores


def test_answer_scores(exporter, recorded_scores):
    answerer = GroundedAnswerer(FakeClient("Налог уплачивается ежегодно до 1 апреля [1]."), min_retrieval_score=0.5,
                                min_answer_relevance=0.1, relevance_scorer=lambda q, a: 0.8)
    with tracing.trace("request"):
        answerer.answer("Когда уплачивается налог?", passages())
    assert recorded_scores["retrieval_top_score"][:2] == (0.9, "NUMERIC")
    assert recorded_scores["sentence_support"][0] == 1.0
    assert recorded_scores["answer_relevance"][0] == 0.8
    assert recorded_scores["answer_status"][:2] == (STATUS_ANSWERED, "CATEGORICAL")
    assert recorded_scores["answer_shown"] == (1, "BOOLEAN", None)
    assert "entailment" not in recorded_scores  # проверка не включена — оценки нет


def test_rejected_answer_scores_explain_why(exporter, recorded_scores):
    answerer = GroundedAnswerer(FakeClient("Налог уплачивается до 15 мая [1]."), min_retrieval_score=None,
                                min_answer_relevance=None)
    with tracing.trace("request"):
        answerer.answer("Когда уплачивается налог?", passages())
    value, data_type, comment = recorded_scores["answer_shown"]
    assert (value, data_type) == (0, "BOOLEAN") and "15" in comment


def test_no_scores_without_trace(exporter, recorded_scores):
    GroundedAnswerer(FakeClient("Налог уплачивается ежегодно [1]."), min_retrieval_score=None,
                     min_answer_relevance=None).answer("Когда уплачивается налог?", passages())
    assert recorded_scores == {}


def test_user_feedback_score(exporter, monkeypatch):
    calls = []
    monkeypatch.setattr(tracing._client, "create_score", lambda **kwargs: calls.append(kwargs))
    with tracing.trace("request") as root:
        trace_id = root.trace_id
    assert trace_id
    assert tracing.score_trace(trace_id, "user_feedback", 1.0, data_type="BOOLEAN", score_id=f"{trace_id}-user-feedback")
    assert calls == [{"trace_id": trace_id, "name": "user_feedback", "value": 1.0, "data_type": "BOOLEAN",
                      "comment": None, "score_id": f"{trace_id}-user-feedback"}]
    assert not tracing.score_trace("", "user_feedback", 1.0)


def test_retry_stage_has_reason(exporter):
    answerer = GroundedAnswerer(FakeClient("[1]", "Налог уплачивается ежегодно [1]."), min_retrieval_score=None,
                                min_answer_relevance=None)
    with tracing.trace("request"):
        answerer.answer("Когда уплачивается налог?", passages())
    found = spans(exporter)
    assert attr(found["llm-retry"], "metadata.retry_reason") == "ответ без текста — только ссылка на фрагмент"


def test_rejected_answer_is_marked_as_warning(exporter):
    answerer = GroundedAnswerer(FakeClient("Налог уплачивается до 15 мая [1]."), min_retrieval_score=None,
                                min_answer_relevance=None)
    with tracing.trace("request"):
        answer = answerer.answer("Когда уплачивается налог?", passages())
    found = spans(exporter)
    assert answer.checks["unsupported_numbers"] == ["15"]
    assert attr(found["numbers"], "level") == "WARNING"
    assert attr(found["verification"], "level") == "WARNING"


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_tracing_does_not_change_results(exporter, engine, mode):
    plain = engine.search("ставка земельного налога", mode=mode, top_k=3, rerank=True)
    with tracing.trace("request"):
        traced = engine.search("ставка земельного налога", mode=mode, top_k=3, rerank=True)
    assert traced == plain
    assert mode in spans(exporter)


def test_judge_checks_unchanged_by_tracing(exporter):
    """Итог проверок в трассе и без неё совпадает (judge разбит на этапы)."""
    answerer = GroundedAnswerer(None, min_answer_relevance=None)
    raw = "Налог уплачивается ежегодно до 1 апреля [1] [7]."
    plain = answerer.judge("Когда?", raw, passages())
    with tracing.trace("request"):
        traced = answerer.judge("Когда?", raw, passages())
    assert traced.checks == plain.checks and list(traced.checks) == list(plain.checks)
    assert traced.checks["invalid_citations"] == [7]


def test_exception_inside_span_propagates(exporter):
    with pytest.raises(ValueError):
        with tracing.trace("request"):
            with tracing.span("stage"):
                raise ValueError("ошибка поиска")
    found = spans(exporter)
    # Langfuse показывает такой этап с уровнем ERROR
    assert found["stage"].status.status_code == StatusCode.ERROR
    assert [event.name for event in found["stage"].events] == ["exception"]
    assert not tracing.tracing_active()


def test_no_keys_means_no_client(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.setattr(tracing, "_initialized", False)
    assert tracing.get_langfuse() is None
    with tracing.trace("request") as root, tracing.span("stage") as stage:
        assert not root.enabled and not stage.enabled


def test_broken_langfuse_does_not_break_answer(monkeypatch, capsys):
    class Broken:
        def start_as_current_observation(self, **kwargs):
            raise RuntimeError("сервер недоступен")

    monkeypatch.setattr(tracing, "_client", Broken())
    monkeypatch.setattr(tracing, "_initialized", True)
    with tracing.trace("request") as root:
        root.update(output="x")
        answer = GroundedAnswerer(FakeClient("Налог уплачивается ежегодно [1]."), min_retrieval_score=None,
                                  min_answer_relevance=None).answer("Когда уплачивается налог?", passages())
    assert answer.answered
    assert "сервер недоступен" in capsys.readouterr().out


def test_ask_cli_one_trace_per_question(exporter, fake_embedder, tmp_path, monkeypatch):
    corpus = pd.DataFrame({"source_title": ["Земельный кодекс", "Закон об образовании"],
                           "cleaned_context": ["Статья 1. Земельный налог уплачивается ежегодно.",
                                               "Статья 2. Образовательный грант выдаётся военнослужащим."]})
    monkeypatch.setattr(ask, "load_legal_corpus", lambda *_: corpus)
    monkeypatch.setattr(ask, "EmbeddingModel", lambda **_: fake_embedder)
    commands = iter(["земельный налог", ":mode bm25", "образовательный грант", "q"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(commands))

    ask.interactive_cli(search_mode="hybrid", top_k=2, cache_dir=str(tmp_path))

    exporter.flush()
    roots = [s for s in exporter.get_finished_spans() if s.name == "ask-cli"]
    assert len(roots) == 2
    assert len({s.attributes["session.id"] for s in roots}) == 1
    assert [attr(s, "metadata.cold_start") for s in roots] == [True, False]
