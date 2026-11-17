"""Веб-демо поиска по законодательству Республики Казахстан (Streamlit).

Интерфейс к тем же компонентам, что и консольный ask.py: Dense (E5 + FAISS),
BM25 и Hybrid (RRF). В отличие от ask.py, поиск идёт по чанкам 500 символов
(лучшая стратегия в бенчмарке run_adilet_eval.py), а для каждого результата
показываются закон, номер статьи, ссылка на adilet.zan.kz и полный фрагмент.
Кандидатов любого метода можно переранжировать cross-encoder (src/reranker.py).

Запуск:
    streamlit run app.py

Скорость интерфейса:
- тяжёлые библиотеки (sentence-transformers, torch — ~13 с импорта) и модели загружаются в фоновом потоке
  после того, как страница показана, — пока пользователь читает её и набирает вопрос;
- результаты поиска и ответ модели запоминаются для пары «вопрос + настройки»: Streamlit перезапускает
  скрипт при любом клике, но повторно ничего не считается (и в Langfuse не уходят дубликаты трасс).
"""

import html
import json
import re
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Optional, Set

import pandas as pd
import streamlit as st

from ask import (
    LegalSearchEngine, build_passages, default_model, load_bm25_params, load_hybrid_params, load_legal_corpus,
    load_rerank_pool,
)
from run_adilet_eval import DEFAULT_MODEL, model_cache_name, texts_fingerprint
from src.corpus import CHUNK_OVERLAP, CHUNK_SIZE, build_chunk_corpus, law_titles
from src.nli import load_entailment_scorer
from src.embeddings import EmbeddingModel
from src.generation import (
    DEFAULT_LLM, DEFAULT_TOP_K, OLLAMA_URL, STATUS_ANSWERED, STATUS_NO_CONTEXT, STATUS_REFUSED,
    Answer, GroundedAnswerer, OllamaClient, OllamaError, Passage, load_generation_params, reranker_relevance,
)
from src.laws import LAWS_PARQUET, load_laws
from src.reranker import CrossEncoderReranker, default_reranker
from src.resources import ResourceMeter, gpu_memory, snapshot as resource_snapshot
from src.retriever import tokenize_legal_text
from src.tracing import get_langfuse, span, trace

CORPUS_LABELS = {"laws": "Законы для граждан (актуальные)", "adilet": "Датасет Adilet Legal QA (14 актов)"}
CACHE_PARQUET = "data/raw/adilet_legal_qa_kz.parquet"
CACHE_DIR = "data/adilet/cache"
MODELS_DIR = Path("models")
# Сколько кандидатов достаём перед фильтрацией по законам
FILTER_POOL = 300

MODE_LABELS = {
    "hybrid": "Hybrid (Dense + BM25, RRF)",
    "dense": "Dense (семантический, E5 + FAISS)",
    "bm25": "BM25 (по ключевым словам)",
}
LANG_LABELS = {"rus": "Русский", "kaz": "Қазақша"}

EXAMPLES = {
    "rus": [
        "В течение какого срока местные исполнительные органы уведомляют о резервировании земель?",
        "Кто может быть избран Президентом Республики Казахстан?",
        "Что такое дееспособность гражданина?",
        "Какие гарантии имеет обучающийся при получении образования?",
        "Каков срок полномочий депутатов Мажилиса?",
    ],
    "kaz": [
        "Қазақстан Республикасының Президенті болып кім сайлана алады?",
        "Азаматтың әрекет қабілеттілігі дегеніміз не?",
        "Жер учаскесін жалға беру мерзімі қандай?",
    ],
}
LAWS_EXAMPLES = [
    "Сколько дней ежегодного отпуска положено работнику?",
    "Можно ли вернуть товар надлежащего качества?",
    "Как расторгнуть брак?",
    "Как уволиться по собственному желанию?",
    "Кто имеет право на пособие по уходу за ребёнком?",
]

# Служебные слова не подсвечиваются в найденном тексте
STOPWORDS = {
    "какой", "какая", "какие", "каких", "каком", "какого", "кто", "что", "когда", "где", "как",
    "для", "при", "или", "если", "это", "его", "она", "они", "который", "которые", "течение",
    "может", "могут", "быть", "есть", "имеет",
    "қандай", "кім", "не", "қашан", "қалай", "және", "немесе", "үшін",
}


class ModelStore:
    """Корпуса, индексы и модели, общие для всех вкладок браузера: каждый объект загружается один раз за процесс.

    В отличие от st.cache_resource, загрузку можно начать в фоновом потоке. Если вопрос задан раньше,
    чем она закончилась, запрос дождётся той же загрузки, а не начнёт вторую.
    """

    def __init__(self):
        self._items: Dict[Hashable, Any] = {}
        self._locks: Dict[Hashable, threading.Lock] = {}
        self._guard = threading.Lock()

    def ready(self, key: Hashable) -> bool:
        return key in self._items

    def get(self, key: Hashable, factory: Callable[[], Any]) -> Any:
        if key in self._items:
            return self._items[key]
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            if key not in self._items:
                self._items[key] = factory()
        return self._items[key]


@st.cache_resource
def model_store() -> ModelStore:
    return ModelStore()


class ResultCache:
    """Последние результаты поиска и ответы модели по ключу «вопрос + настройки» (общие для всех вкладок)."""

    def __init__(self, max_entries: int = 200):
        self.max_entries = max_entries
        self._items: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any:
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
            return self._items.get(key)

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)


@st.cache_resource
def result_cache() -> ResultCache:
    return ResultCache()


def device() -> str:
    import torch  # импорт torch (~2 с) — только когда загружается модель

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_corpus(lang: str, corpus_name: str = "adilet") -> List[Dict[str, Any]]:
    """Чанки 500 символов с законом и статьями (src/corpus.py).

    Чанки датасета совпадают с run_adilet_eval.py, поэтому используется уже готовый кэш эмбеддингов.
    Возвращается общий для всех вкладок список: изменять его нельзя (результаты поиска — копии чанков).
    """
    def build() -> List[Dict[str, Any]]:
        if corpus_name == "laws":
            return build_chunk_corpus(load_laws(), {})
        titles = law_titles(pd.read_parquet(CACHE_PARQUET, columns=["context", "source_id", "doc_type", "language"]))
        return build_chunk_corpus(load_legal_corpus(CACHE_PARQUET, lang), titles)

    return model_store().get(("corpus", corpus_name, lang), build)


def chunks_by_id(lang: str, corpus_name: str) -> Dict[str, Dict[str, Any]]:
    return model_store().get(("chunks-by-id", corpus_name, lang), lambda: {c["id"]: c for c in build_corpus(lang, corpus_name)})


def laws_as_of() -> str:
    return model_store().get("laws-as-of", lambda: str(load_laws()["as_of"].max()))


def available_corpora() -> List[str]:
    """Актуальные законы первыми, если корпус подготовлен (prepare_laws.py)."""
    return (["laws"] if Path(LAWS_PARQUET).exists() else []) + ["adilet"]


def source_url(lang: str, source_id: str) -> str:
    """Адрес акта на adilet.zan.kz или пустая строка, если адрес не подтверждён."""
    return f"https://adilet.zan.kz/{lang}/docs/{source_id}" if source_id else ""


def engine_key(lang: str, model_name: str, corpus_name: str) -> tuple:
    return ("engine", corpus_name, lang, model_name)


def release_gpu_memory() -> None:
    """Возвращает видеокарте память, которую PyTorch держит про запас после реранкера: она нужна Qwen в Ollama."""
    torch = sys.modules.get("torch")  # torch уже загружен, если модели работали на видеокарте
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_embedder(model_name: str) -> EmbeddingModel:
    """Модель эмбеддингов вопросов — на процессоре.

    На видеокарте 4 ГБ не помещаются вместе e5, реранкер и Qwen: Ollama выносит часть слоёв в обычную память,
    и ответ модели замедляется в 1,5–2 раза. Вопрос из десятка слов e5-small векторизует на CPU за десятки
    миллисекунд — поиск от этого не замедляется (замер — в README).
    """
    return model_store().get(("embedder", model_name), lambda: EmbeddingModel(model_name=model_name, device="cpu"))


def load_engine(lang: str, model_name: str, corpus_name: str = "adilet") -> LegalSearchEngine:
    def build() -> LegalSearchEngine:
        corpus = build_corpus(lang, corpus_name)
        texts = [c["text"] for c in corpus]
        suffix = "_laws" if corpus_name == "laws" else ""
        cache_key = f"{model_cache_name(model_name)}_{lang}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}{suffix}"
        embedder = load_embedder(model_name)
        # Если кэша эмбеддингов корпуса нет, корпус векторизуется на видеокарте (на CPU это минуты), затем e5
        # возвращается на CPU. Блокировка — чтобы два корпуса не переносили общую модель одновременно
        cached = Path(CACHE_DIR, f"{cache_key}_{texts_fingerprint(texts)}.npy").exists()
        with model_store().get("embedder-device-lock", threading.Lock):
            gpu = not cached and device() == "cuda"
            if gpu:
                embedder.to("cuda")
            try:
                engine = LegalSearchEngine(
                    docs=texts,
                    doc_ids=[c["id"] for c in corpus],
                    embedder=embedder,
                    cache_dir=CACHE_DIR,
                    cache_key=cache_key,
                    hybrid_params=load_hybrid_params(model_name, lang),
                    bm25_params=load_bm25_params(lang),
                )
            finally:
                if gpu:
                    embedder.to("cpu")
                    release_gpu_memory()
        engine.rerank_pool = rerank_pool(lang)
        return engine

    return model_store().get(engine_key(lang, model_name, corpus_name), build)


def load_reranker(model_name: str) -> CrossEncoderReranker:
    return model_store().get(("reranker", model_name), lambda: CrossEncoderReranker(model_name, device=device()))


def warm_up(lang: str, model_name: str, corpus_name: str) -> None:
    """Фоновая загрузка поиска: e5 и индекс, BM25, первый проход моделей (на GPU — инициализация CUDA), реранкер."""
    try:
        engine = load_engine(lang, model_name, corpus_name)
        engine.bm25_retriever  # индекс BM25 строится при первом обращении
        engine.embed_query("прогрев модели")
        load_reranker(default_reranker(lang)).score("прогрев модели", ["прогрев модели"])
        release_gpu_memory()
    except Exception as e:  # запрос пользователя всё равно загрузит модели сам и покажет ошибку
        print(f"Фоновая загрузка моделей не удалась: {e}")


@st.cache_resource
def start_warm_up(lang: str, model_name: str, corpus_name: str) -> threading.Thread:
    """Запускает warm_up один раз на процесс для каждого сочетания корпуса, языка и модели."""
    thread = threading.Thread(target=warm_up, args=(lang, model_name, corpus_name), daemon=True, name="warm-up")
    thread.start()
    return thread


def warm_up_answer(lang: str) -> None:
    """Фоновая подготовка ответа модели: проверки ответа и загрузка Qwen в память Ollama (иначе +5–6 с к ответу)."""
    try:
        client = load_answerer(lang).client
        # Тот же размер контекста, что у запросов ответа: при другом num_ctx Ollama загрузила бы модель заново
        body = json.dumps({"model": client.model, "options": {"num_ctx": client.options["num_ctx"]}}).encode("utf-8")
        request = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300):
            pass
    except Exception as e:  # Ollama не запущена — об этом скажет сам запрос ответа
        print(f"Фоновая подготовка ответа модели не удалась: {e}")


def start_answer_warm_up(lang: str) -> None:
    """Не чаще раза в 4 минуты: Ollama по умолчанию держит модель в памяти 5 минут после запроса."""
    store = model_store()
    last = store.get("answer-warm-up", dict)
    if time.time() - last.get(lang, 0) > 240:
        last[lang] = time.time()
        threading.Thread(target=warm_up_answer, args=(lang,), daemon=True, name="warm-up-answer").start()


def rerank_pool(lang: str) -> int:
    """Размер пула кандидатов, выбранный eval_reranker.py на val (для чанков 500 символов)."""
    return load_rerank_pool(lang)


def is_embedding_model(path: Path) -> bool:
    """Папка с моделью эмбеддингов (не реранкер cross-encoder, который тоже лежит в models/)."""
    if not (path / "config.json").exists():
        return False
    st_config = path / "config_sentence_transformers.json"
    if st_config.exists():
        return json.loads(st_config.read_text(encoding="utf-8")).get("model_type") != "CrossEncoder"
    return True


def available_models(lang: str) -> List[str]:
    """Модель по умолчанию для языка (дообученная, если есть), затем остальные: базовая и models/*."""
    local = sorted(p.as_posix() for p in MODELS_DIR.glob("*") if is_embedding_model(p)) if MODELS_DIR.exists() else []
    first = default_model(lang)
    return [first] + [m for m in [DEFAULT_MODEL] + local if m != first]


def query_stems(query: str) -> Set[str]:
    """Основы слов запроса для подсветки: первые 5 букв значимых слов и числа целиком."""
    stems = set()
    for token in tokenize_legal_text(query):
        if token in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in token):
            stems.add(token)
        elif len(token) >= 4:
            stems.add(token[:5])
    return stems


def highlight(text: str, stems: Set[str]) -> str:
    """Экранирует текст для HTML и выделяет слова, совпадающие с основами запроса."""
    def repl(match: re.Match) -> str:
        word = match.group(0)
        low = word.lower()
        hit = low in stems or (len(low) >= 4 and low[:5] in stems)
        return f"<mark>{word}</mark>" if hit else word

    return re.sub(r"[\w\-]+", repl, html.escape(text)).replace("\n", "<br>")


def run_search(
    engine: LegalSearchEngine, lang: str, query: str, mode: str, top_k: int, laws: List[str], rerank: bool = False,
    corpus_name: str = "adilet",
) -> List[Dict[str, Any]]:
    """Поиск с фильтром по законам; при rerank фильтр применяется до переранжирования."""
    by_id = chunks_by_id(lang, corpus_name)
    wanted = max(top_k, engine.rerank_pool) if rerank else top_k
    pool = FILTER_POOL if laws else wanted
    with span("search", as_type="retriever", input=query, metadata={"mode": mode, "top_k": top_k, "rerank": rerank, "laws": laws}) as obs:
        found = engine.search(query, mode=mode, top_k=pool)
        with span("law-filter", input={"laws": laws, "pool": len(found), "wanted": wanted}) if laws else nullcontext() as filter_obs:
            hits = []
            for doc_id, _, score in found:
                chunk = by_id[doc_id]
                if laws and chunk["law"] not in laws:
                    continue
                hits.append({**chunk, "score": score})
                if len(hits) == wanted:
                    break
            if filter_obs is not None:
                filter_obs.update(output={"kept": len(hits)})
        if rerank:
            reranked = engine.rerank(query, [(h["id"], h["text"], h["score"]) for h in hits], top_k)
            hits = [{**by_id[doc_id], "score": score} for doc_id, _, score in reranked]
        obs.update(output=[
            {"rank": rank, "id": h["id"], "law": h["law"], "articles": h["articles"], "score": round(float(h["score"]), 4)}
            for rank, h in enumerate(hits, 1)
        ])
    return hits


def score_text(mode: str, score: float, rerank: bool = False) -> str:
    if rerank:
        return f"cross-encoder {score:.3f}"
    if mode == "dense":
        return f"косинусное сходство {score:.3f}"
    if mode == "bm25":
        return f"BM25 {score:.2f}"
    return f"RRF {score:.4f}"


def render_hit(
    rank: int, hit: Dict[str, Any], mode: str, lang: str, stems: Set[str], compact: bool = False, rerank: bool = False
) -> None:
    url = source_url(lang, hit["source_id"])
    with st.container(border=True):
        st.markdown(f"**{rank}. {hit['law']}**" + (f" · {hit['articles']}" if hit["articles"] else ""))
        if hit["article_title"] and not compact:
            st.caption(hit["article_title"])
        st.markdown(f"<div class='hit-text'>{highlight(hit['text'], stems)}</div>", unsafe_allow_html=True)
        st.caption(score_text(mode, hit["score"], rerank) + (f" · [открыть на adilet.zan.kz]({url})" if url else ""))
        if not compact:
            with st.expander("Весь фрагмент закона"):
                st.markdown(f"<div class='hit-text'>{highlight(hit['context'], stems)}</div>", unsafe_allow_html=True)


def load_answerer(lang: str) -> GroundedAnswerer:
    """Ответ модели с проверками; соответствие ответа вопросу оценивает тот же реранкер, что и поиск."""
    def build() -> GroundedAnswerer:
        params = load_generation_params(lang, default_reranker(lang))
        return GroundedAnswerer(
            OllamaClient(DEFAULT_LLM), lang,
            min_retrieval_score=params["min_retrieval_score"], min_sentence_support=params["min_sentence_support"],
            min_answer_relevance=params["min_answer_relevance"],
            relevance_scorer=reranker_relevance(load_reranker(default_reranker(lang))),
            min_entailment=params["min_entailment"],
            entailment_scorer=load_entailment_scorer(params["min_entailment"]),
        )

    return model_store().get(("answerer", lang), build)


def compute_answer(query: str, hits: List[Dict[str, Any]], lang: str) -> Dict[str, Any]:
    """Ответ модели по найденным фрагментам: {"answer", "passages", "error"} (error — Ollama недоступна)."""
    # Модель читает чанки с окружающим текстом; пересекающиеся склеиваются, поэтому номера [n] — номера passages
    passages = build_passages(hits)
    try:
        with st.spinner(f"{DEFAULT_LLM} формулирует ответ по найденным фрагментам..."):
            return {"answer": load_answerer(lang).answer(query, passages), "passages": passages, "error": None}
    except OllamaError as e:
        return {"answer": None, "passages": passages, "error": str(e)}


def render_answer(answer: Optional[Answer], passages: List[Passage], lang: str, error: Optional[str]) -> None:
    """Ответ модели с итогами проверки и источниками."""
    answerer = load_answerer(lang)
    with st.container(border=True):
        st.markdown("**Ответ**")
        if answer is None:
            st.error(error)
            return

        top = answer.checks.get("top_score")
        if answer.status == STATUS_ANSWERED:
            st.markdown(answer.text.replace("\n", "  \n"))
            sources = []
            for n in answer.cited:
                url = source_url(lang, passages[n - 1].source_id)
                sources.append(f"[{n}] [{passages[n - 1].label}]({url})" if url else f"[{n}] {passages[n - 1].label}")
            st.caption("Источники: " + " · ".join(sources))
            relevance = answer.checks.get("answer_relevance")
            entailment = answer.checks.get("entailment")
            st.caption(
                f"✓ Проверено: все числа есть в тексте закона; слова каждого предложения найдены во фрагментах "
                f"не меньше чем на {answerer.min_sentence_support:.0%} (минимум {answer.checks['min_sentence_support']:.0%})"
                + (f"; соответствие вопросу {relevance:.2f}" if relevance is not None else "")
                + (f"; следование из фрагмента {entailment:.2f}" if entailment is not None else "")
                + f" · {answer.usage.get('latency_s', 0):.1f} с"
            )
        elif answer.status == STATUS_NO_CONTEXT:
            st.info(answer.text)
            st.caption(
                f"Модель не вызывалась: лучший фрагмент получил оценку реранкера {top:.2f} — ниже порога "
                f"{answerer.min_retrieval_score}. Скорее всего, нужного закона нет в корпусе."
            )
        elif answer.status == STATUS_REFUSED:
            st.info(answer.text)
            st.caption("Модель прочитала найденные фрагменты и не нашла в них ответа.")
        else:
            st.warning(answer.text)
            st.markdown("> " + passages[0].text.strip().replace("\n", "\n> "))
            st.caption(f"[1] {passages[0].label}")
            with st.expander("Почему ответ модели не показан"):
                st.write("Проверка нашла: " + "; ".join(answer.checks["problems"]) + ".")
                st.caption("Отклонённый ответ модели (не является ответом системы):")
                st.text(answer.raw)


def load_models(lang: str, model_name: str, corpus_name: str, rerank: bool, generate: bool) -> LegalSearchEngine:
    """Загрузка индекса и моделей — отдельный этап трассы; если что-то ещё не загружено, это холодный старт.

    Обычно модели уже загрузил фоновый поток (warm_up); если он ещё работает, запрос ждёт его.
    """
    store = model_store()
    needed = [engine_key(lang, model_name, corpus_name)]
    if rerank:
        needed.append(("reranker", default_reranker(lang)))
    if generate:
        needed.append(("answerer", lang))
    cold = [":".join(key) for key in needed if not store.ready(key)]
    spinner = st.spinner("Загрузка моделей поиска — только при первом запросе после запуска...") if cold else nullcontext()
    with span("load-models", input=[":".join(key) for key in needed], metadata={"cold_start": bool(cold)}) as obs, spinner:
        engine = load_engine(lang, model_name, corpus_name)
        if rerank:
            engine.reranker = load_reranker(default_reranker(lang))
        if generate:
            load_answerer(lang)
        obs.update(output={"loaded_now": cold}, metadata={"cold_start": bool(cold), "device": device(), **gpu_memory()})
    return engine


def search_and_answer(
    query: str, corpus_name: str, lang: str, model_name: str, mode: str, top_k: int, laws: List[str], rerank: bool,
    generate: bool,
) -> Dict[str, Any]:
    """Поиск и ответ модели одним запросом — одна трасса Langfuse."""
    request = {
        "corpus": corpus_name, "lang": lang, "mode": mode, "top_k": top_k, "embedder": model_name,
        "rerank": rerank, "generate": generate, "laws": laws,
    }
    with trace("web-demo", input=query, metadata=request, session_id=streamlit_session_id(), tags=["app.py", mode]) as root:
        with ResourceMeter() as meter:
            engine = load_models(lang, model_name, corpus_name, rerank, generate)
            t0 = time.time()
            # Модели нужны DEFAULT_TOP_K лучших фрагментов, даже если показать нужно меньше
            hits = run_search(engine, lang, query, mode, max(top_k, DEFAULT_TOP_K) if generate else top_k, laws, rerank, corpus_name)
            result = {"hits": hits, "search_ms": (time.time() - t0) * 1000, "answer": None, "passages": [], "error": None}
            if hits and generate:
                result.update(compute_answer(query, hits[:DEFAULT_TOP_K], lang))
        answer = result["answer"]
        found = [f"{h['law']}, {h['articles']}" if h["articles"] else h["law"] for h in hits[:top_k]]
        # Процессор и память за весь запрос; видеопамять и размещение Qwen — на момент окончания запроса
        root.update(
            output=answer.text if answer is not None else found,
            metadata={
                **request, "search_ms": round(result["search_ms"]), "status": answer.status if answer is not None else None,
                **({"error": result["error"]} if result["error"] else {}),
                "resources": {**meter.result, **resource_snapshot(OLLAMA_URL, DEFAULT_LLM if generate else None)},
            },
        )
    return result


def streamlit_session_id() -> Optional[str]:
    """Id сеанса Streamlit (вкладки браузера) — объединяет её запросы в одну сессию Langfuse."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        return ctx.session_id if ctx else None
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="Поиск по законам РК", page_icon="⚖️", layout="wide")
    st.markdown(
        """
        <style>
        .hit-text { font-size: 0.93rem; line-height: 1.5; }
        .hit-text mark { background: #ffe58a; color: inherit; padding: 0 2px; border-radius: 3px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Настройки")
        corpus_name = st.radio("Корпус", available_corpora(), format_func=CORPUS_LABELS.get, key="corpus")
        if corpus_name == "laws":
            lang = "rus"
        else:
            lang = st.radio("Язык корпуса", list(LANG_LABELS), format_func=LANG_LABELS.get, horizontal=True, key="lang")
        mode = st.radio("Метод поиска", list(MODE_LABELS), format_func=MODE_LABELS.get)
        top_k = st.slider("Количество результатов", 1, 10, 5)
        model_name = st.selectbox("Модель эмбеддингов", available_models(lang))
        if mode == "hybrid":
            params = load_hybrid_params(model_name, lang)
            st.caption(f"RRF: вес Dense {params['dense_weight']}, вес BM25 {params['sparse_weight']}, k = {params['rrf_k']}")
        generate = st.toggle(
            "Ответ языковой модели",
            help=f"{DEFAULT_LLM} (локально через Ollama) формулирует ответ по {DEFAULT_TOP_K} лучшим фрагментам. "
                 "Ответ, не подтверждённый текстом закона, не показывается. Включает реранкер.",
        )
        rerank = st.toggle(
            "Реранкер (cross-encoder)", value=generate, disabled=generate,
            help="Переранжировать лучших кандидатов выбранного метода нейросетью, читающей вопрос и фрагмент вместе.",
        ) or generate
        if rerank:
            st.caption(f"{default_reranker(lang)}, топ-{rerank_pool(lang)} кандидатов")
        if generate:
            gen_params = load_generation_params(lang, default_reranker(lang))
            gate = gen_params["min_retrieval_score"]
            relevance = gen_params["min_answer_relevance"]
            st.caption(
                f"{DEFAULT_LLM}: порог реранкера {gate if gate is not None else 'не задан'}, "
                f"опора предложений на текст ≥ {gen_params['min_sentence_support']}, "
                f"соответствие ответа вопросу ≥ {relevance if relevance is not None else 'не задано'}"
            )
        st.caption(
            "Langfuse: трассировка включена" if get_langfuse() is not None
            else "Langfuse: выключен (не заданы LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)"
        )
        corpus = build_corpus(lang, corpus_name)
        all_laws = sorted({c["law"] for c in corpus})
        laws = st.multiselect("Искать только в законах", all_laws, placeholder="Все законы")
        st.divider()
        source_note = (
            "Тексты актов выгружены с adilet.zan.kz." if corpus_name == "laws"
            else "Данные: Adilet Legal QA (названия законов восстановлены по шапкам страниц adilet.zan.kz)."
        )
        st.caption(
            f"Корпус: {len(all_laws)} нормативных актов, "
            f"{len({c['id'].split('_fix_')[0] for c in corpus})} фрагментов, {len(corpus)} чанков "
            f"по {CHUNK_SIZE} символов. {source_note}"
        )

    st.title("⚖️ Поиск по законодательству Республики Казахстан")
    st.write(
        "Задайте вопрос обычным языком — система найдёт фрагменты нормативных актов, "
        "в которых содержится ответ. Каждый результат — текст закона со ссылкой на источник; по желанию языковая "
        "модель сформулирует ответ только по найденным фрагментам."
    )
    if corpus_name == "laws":
        st.caption(
            f"Редакции актов на {laws_as_of()}. Справочная информация, не является юридической "
            "консультацией: качество поиска по этому набору отдельно не измерялось, сверяйте ответ с текстом закона."
        )

    examples = LAWS_EXAMPLES if corpus_name == "laws" else EXAMPLES[lang]
    st.caption("Примеры вопросов:")
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, width="stretch"):
            st.session_state.query = example

    query = st.text_input("Ваш вопрос", key="query", placeholder="Например: кто может быть избран Президентом?")
    # Страница уже показана: пока пользователь набирает вопрос, модели загружаются в фоне
    start_warm_up(lang, model_name, corpus_name)
    if generate:
        start_answer_warm_up(lang)
    if not query.strip():
        return

    stems = query_stems(query)
    tab_search, tab_compare = st.tabs(["Результаты", "Сравнение методов"])

    cache = result_cache()
    with tab_search:
        # Streamlit перезапускает скрипт при любом клике: готовый результат берётся из памяти.
        # Трасса Langfuse пишется только при настоящем поиске; вкладка сравнения методов не трассируется
        key = ("search", corpus_name, lang, model_name, query.strip(), mode, top_k, tuple(laws), rerank, generate)
        result = cache.get(key)
        if result is None:
            result = search_and_answer(query, corpus_name, lang, model_name, mode, top_k, laws, rerank, generate)
            release_gpu_memory()
            if result["error"] is None:  # при недоступной Ollama следующий клик попробует снова
                cache.put(key, result)
        hits = result["hits"]
        if not hits:
            st.info("Ничего не найдено. Попробуйте переформулировать вопрос или снять фильтр по законам.")
        else:
            if generate:
                render_answer(result["answer"], result["passages"], lang, result["error"])
            label = MODE_LABELS[mode] + (" + реранкер" if rerank else "")
            st.caption(f"{label} · найдено {len(hits[:top_k])} за {result['search_ms']:.0f} мс")
            for rank, hit in enumerate(hits[:top_k], 1):
                render_hit(rank, hit, mode, lang, stems, rerank=rerank)

    with tab_compare:
        st.caption("Топ-3 каждого метода по одному и тому же вопросу.")
        engine = load_models(lang, model_name, corpus_name, rerank, False)
        # С включённым реранкером добавляется колонка Hybrid + реранкер
        columns = [(m, False) for m in MODE_LABELS] + ([("hybrid", True)] if rerank else [])
        for col, (m, m_rerank) in zip(st.columns(len(columns)), columns):
            with col:
                st.subheader(m.upper() + (" + RERANK" if m_rerank else ""))
                key = ("compare", corpus_name, lang, model_name, query.strip(), m, tuple(laws), m_rerank)
                hits = cache.get(key)
                if hits is None:
                    hits = run_search(engine, lang, query, m, 3, laws, m_rerank, corpus_name)
                    cache.put(key, hits)
                    release_gpu_memory()
                for rank, hit in enumerate(hits, 1):
                    render_hit(rank, hit, m, lang, stems, compact=True, rerank=m_rerank)

if __name__ == "__main__":
    main()
