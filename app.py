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
from src.query_expansion import expand_query
from src.reranker import CrossEncoderReranker, default_reranker
from src.resources import ResourceMeter, gpu_memory, snapshot as resource_snapshot
from src.retriever import tokenize_legal_text
from src.tracing import get_langfuse, score_trace, span, trace

CORPUS_LABELS = {"laws": "Законы для граждан (актуальные)", "adilet": "Датасет Adilet Legal QA (14 актов)"}
CACHE_PARQUET = "data/raw/adilet_legal_qa_kz.parquet"
CACHE_DIR = "data/adilet/cache"
# Настройки корпуса законов, подобранные eval_laws_colloquial.py на разговорных вопросах
LAWS_SETTINGS = Path("data/laws/colloquial_eval.json")
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
# Примеры для корпуса законов: короткие вопросы разных тем, на которые система отвечает по правильной статье
# (data/laws/colloquial_eval.json, REPORT 7.15–7.18)
LAWS_EXAMPLES = [
    "Сколько дней отпуска положено в год?",
    "Сколько алиментов платят на одного ребёнка?",
    "Кто получает наследство, если завещания нет?",
    "Сколько действует доверенность?",
]

# Шрифт показывает, откуда текст: антиква PT Serif — дословный текст закона, гротеск PT Sans — ответ и пояснения
# системы. Номер статьи стоит на полях, как в изданиях кодексов. Без интернета — запасные системные шрифты.
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=PT+Sans:ital,wght@0,400;0,700;1,400&family=PT+Serif:ital,wght@0,400;0,700&display=swap');
:root { --ink: #1B2A33; --muted: #56656E; --accent: #0F6A86; --rule: #C9D3D8; --mark: #F2E2A0;
        --ok: #2F6E4F; --caution: #8A5A00; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp h1, .stApp h2,
.stApp h3 { font-family: 'PT Sans', 'Segoe UI', Arial, sans-serif; }
.stApp h1 { font-size: 2.05rem; font-weight: 700; line-height: 1.15; letter-spacing: -0.01em; }
.lede { color: var(--muted); font-size: 1.08rem; line-height: 1.45; max-width: 36em; margin: 0.2rem 0 1.1rem; }
.answer { font-size: 1.22rem; line-height: 1.5; max-width: 36em; margin: 0.6rem 0 0.35rem; color: var(--ink); }
.answer sup { color: var(--accent); font-size: 0.68em; }
.note { font-size: 0.93rem; line-height: 1.4; color: var(--muted); max-width: 40em; margin: 0 0 1.3rem; }
.note.ok::before { content: "✓ "; color: var(--ok); font-weight: 700; }
.note.caution { color: var(--caution); }
.norm { display: grid; grid-template-columns: 8.5rem minmax(0, 1fr); gap: 0 1.25rem; margin: 0 0 1.5rem; }
.norm-margin { text-align: right; padding-top: 0.05rem; }
.norm-ref { font-size: 0.8rem; color: var(--accent); }
.norm-article { font-family: 'PT Serif', Georgia, serif; font-size: 1.2rem; font-weight: 700; line-height: 1.2;
                color: var(--accent); }
.norm-law { font-size: 0.82rem; line-height: 1.3; color: var(--muted); margin-top: 0.2rem; }
.norm-body { border-left: 2px solid var(--rule); padding-left: 1.1rem; min-width: 0; }
.norm-title { font-weight: 700; line-height: 1.35; margin-bottom: 0.3rem; }
.norm-text { font-family: 'PT Serif', Georgia, serif; font-size: 1.02rem; line-height: 1.65; max-width: 40em;
             overflow-wrap: anywhere; }
.norm-text mark { background: var(--mark); color: inherit; padding: 0 0.1em; border-radius: 2px; }
.norm-link, .norm-meta { font-size: 0.86rem; color: var(--muted); margin-top: 0.4rem; }
.norm-link a { color: var(--accent); }
.disclaimer { font-size: 0.86rem; line-height: 1.4; color: var(--muted); border-top: 1px solid var(--rule);
              padding-top: 0.75rem; margin-top: 2.2rem; max-width: 44em; }
.stApp a:focus-visible, .stApp button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (max-width: 640px) {
  .norm { grid-template-columns: minmax(0, 1fr); }
  .norm-margin { text-align: left; display: flex; gap: 0.6rem; align-items: baseline; margin-bottom: 0.35rem; }
  .norm-law { margin-top: 0; }
  .norm-body { padding-left: 0.8rem; }
  .answer { font-size: 1.1rem; }
}
</style>
"""

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

    def evict(self, predicate: Callable[[Hashable], bool]) -> List[Any]:
        """Убирает объекты, чьи ключи подходят под predicate; возвращает убранные."""
        with self._guard:
            keys = [key for key in self._items if predicate(key)]
            return [self._items.pop(key) for key in keys]

    def values(self) -> List[Any]:
        return list(self._items.values())

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


def corpus_settings(lang: str, corpus_name: str) -> Dict[str, Any]:
    """Реранкер, словарь и пороги в шкале реранкера для корпуса.

    Датасет Adilet — дообученный реранкер и пороги eval_generation.py. Законы для граждан — то, что выбрал
    eval_laws_colloquial.py на вопросах обычным языком: исходный реранкер (дообученный подстроился под язык закона),
    словарь «разговорное → юридическое» и свои пороги (пороги датасета пропускали к модели 12% вопросов).
    """
    if corpus_name == "laws":
        def read() -> Dict[str, Any]:
            if not LAWS_SETTINGS.exists():
                return {}
            return json.loads(LAWS_SETTINGS.read_text(encoding="utf-8")).get("params", {})

        params = model_store().get("laws-settings", read)
        if params:
            return {"reranker": params["reranker"], "dictionary": bool(params.get("dictionary")),
                    "retry_short": bool(params.get("retry_short")), "retry_refusals": bool(params.get("retry_refusals")),
                    "number_words": bool(params.get("number_words")),
                    "focus_on_refusal": int(params.get("focus_on_refusal") or 0),
                    "min_retrieval_score": params.get("min_retrieval_score"),
                    "min_answer_relevance": params.get("min_answer_relevance"), "source": LAWS_SETTINGS.as_posix()}
    return {"reranker": default_reranker(lang), "dictionary": False, "source": f"data/adilet/generation_eval_{lang}.json"}


def load_reranker(model_name: str) -> CrossEncoderReranker:
    """Реранкер; на видеокарте держится один: второй (~1,1 ГБ) не оставил бы места Qwen."""
    store = model_store()
    key = ("reranker", model_name)
    if not store.ready(key):
        dropped = store.evict(lambda k: isinstance(k, tuple) and k[0] == "reranker" and k != key)
        if dropped:
            # Ответы с проверкой через выгруженный реранкер и ссылки на него в индексах
            store.evict(lambda k: isinstance(k, tuple) and k[0] == "answerer" and k[-1] != model_name)
            for item in store.values():
                if isinstance(item, LegalSearchEngine) and item.reranker in dropped:
                    item.reranker = None
            del dropped
            release_gpu_memory()
    return store.get(key, lambda: CrossEncoderReranker(model_name, device=device()))


def warm_up(lang: str, model_name: str, corpus_name: str) -> None:
    """Фоновая загрузка поиска: e5 и индекс, BM25, первый проход моделей (на GPU — инициализация CUDA), реранкер."""
    try:
        engine = load_engine(lang, model_name, corpus_name)
        engine.bm25_retriever  # индекс BM25 строится при первом обращении
        engine.embed_query("прогрев модели")
        load_reranker(corpus_settings(lang, corpus_name)["reranker"]).score("прогрев модели", ["прогрев модели"])
        release_gpu_memory()
    except Exception as e:  # запрос пользователя всё равно загрузит модели сам и покажет ошибку
        print(f"Фоновая загрузка моделей не удалась: {e}")


@st.cache_resource
def start_warm_up(lang: str, model_name: str, corpus_name: str) -> threading.Thread:
    """Запускает warm_up один раз на процесс для каждого сочетания корпуса, языка и модели."""
    thread = threading.Thread(target=warm_up, args=(lang, model_name, corpus_name), daemon=True, name="warm-up")
    thread.start()
    return thread


def warm_up_answer(lang: str, corpus_name: str) -> None:
    """Фоновая подготовка ответа модели: проверки ответа и загрузка Qwen в память Ollama (иначе +5–6 с к ответу)."""
    try:
        client = load_answerer(lang, corpus_name).client
        # Тот же размер контекста, что у запросов ответа: при другом num_ctx Ollama загрузила бы модель заново
        body = json.dumps({"model": client.model, "options": {"num_ctx": client.options["num_ctx"]}}).encode("utf-8")
        request = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}/api/generate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300):
            pass
    except Exception as e:  # Ollama не запущена — об этом скажет сам запрос ответа
        print(f"Фоновая подготовка ответа модели не удалась: {e}")


def start_answer_warm_up(lang: str, corpus_name: str) -> None:
    """Не чаще раза в 4 минуты: Ollama по умолчанию держит модель в памяти 5 минут после запроса."""
    store = model_store()
    last = store.get("answer-warm-up", dict)
    if time.time() - last.get((lang, corpus_name), 0) > 240:
        last[(lang, corpus_name)] = time.time()
        threading.Thread(target=warm_up_answer, args=(lang, corpus_name), daemon=True, name="warm-up-answer").start()


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
    """Поиск с фильтром по законам; при rerank фильтр применяется до переранжирования.

    Для корпуса законов к вопросу дописываются термины закона (src/query_expansion.py): «как уволиться» →
    «расторжение трудового договора по инициативе работника». Поиск и реранкер получают дополненный вопрос.
    """
    by_id = chunks_by_id(lang, corpus_name)
    wanted = max(top_k, engine.rerank_pool) if rerank else top_k
    pool = FILTER_POOL if laws else wanted
    search_query = search_query_for(query, lang, corpus_name)
    metadata = {"mode": mode, "top_k": top_k, "rerank": rerank, "laws": laws,
                **({"search_query": search_query} if search_query != query else {})}
    with span("search", as_type="retriever", input=query, metadata=metadata) as obs:
        found = engine.search(search_query, mode=mode, top_k=pool)
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
            reranked = engine.rerank(search_query, [(h["id"], h["text"], h["score"]) for h in hits], top_k)
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


def article_heading(articles: str) -> str:
    """Подпись на полях: «ст. 88» → «Статья 88», «ст. 16, ст. 17» → «Статьи 16, 17»; без номера — пусто."""
    numbers = re.findall(r"\d+(?:-\d+)?", articles or "")
    if not numbers:
        return ""
    return f"Статья {numbers[0]}" if len(numbers) == 1 else "Статьи " + ", ".join(numbers)


def answer_html(text: str) -> str:
    """Ответ модели для показа: экранированный текст, ссылки [n] — верхним индексом."""
    escaped = html.escape(text.strip())
    escaped = re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", r"<sup>[\1]</sup>", escaped)
    return escaped.replace("\n", "<br>")


def norm_html(
    law: str, articles: str, text_html: str, *, title: str = "", url: str = "", ref: Optional[int] = None,
    meta: str = "",
) -> str:
    """Норма закона: номер статьи и закон на полях, справа — дословный текст (антиквой) и ссылка на источник.

    text_html — уже экранированный текст (highlight). ref — номер фрагмента, на который ссылается ответ ([n]);
    meta — служебная строка для режима исследователя (оценка поиска).
    """
    heading = article_heading(articles)
    margin = (
        (f"<div class='norm-ref'>[{ref}]</div>" if ref is not None else "")
        + (f"<div class='norm-article'>{html.escape(heading)}</div>" if heading else "")
        + f"<div class='norm-law'>{html.escape(law)}</div>"
    )
    body = (
        (f"<div class='norm-title'>{html.escape(title)}</div>" if title else "")
        + f"<div class='norm-text'>{text_html}</div>"
        + (f"<div class='norm-link'><a href='{html.escape(url, quote=True)}' target='_blank' rel='noopener'>"
           "Открыть на adilet.zan.kz</a></div>" if url else "")
        + (f"<div class='norm-meta'>{html.escape(meta)}</div>" if meta else "")
    )
    return f"<div class='norm'><div class='norm-margin'>{margin}</div><div class='norm-body'>{body}</div></div>"


def render_hit(
    rank: int, hit: Dict[str, Any], mode: str, lang: str, stems: Set[str], compact: bool = False, rerank: bool = False,
    research: bool = True,
) -> None:
    """Найденный фрагмент как норма закона; в режиме исследователя — с местом в выдаче и оценкой метода."""
    url = source_url(lang, hit["source_id"])
    meta = f"{rank}-е место, {score_text(mode, hit['score'], rerank)}" if research else ""
    st.markdown(
        norm_html(hit["law"], hit["articles"], highlight(hit["text"], stems),
                  title="" if compact else hit.get("article_title", ""), url=url, meta=meta),
        unsafe_allow_html=True,
    )
    if not compact and hit.get("context") and hit["context"].strip() != hit["text"].strip():
        with st.expander("Вся статья"):
            st.markdown(f"<div class='norm-text'>{highlight(hit['context'], stems)}</div>", unsafe_allow_html=True)


def answer_params(lang: str, corpus_name: str) -> Dict[str, Any]:
    """Пороги проверки ответа: общие — из eval_generation.py, в шкале реранкера — для реранкера корпуса."""
    settings = corpus_settings(lang, corpus_name)
    params = load_generation_params(lang, settings["reranker"])
    for name in ("min_retrieval_score", "min_answer_relevance"):
        if name in settings:
            params[name] = settings[name]
    return params


def load_answerer(lang: str, corpus_name: str) -> GroundedAnswerer:
    """Ответ модели с проверками; соответствие ответа вопросу оценивает тот же реранкер, что и поиск."""
    settings = corpus_settings(lang, corpus_name)
    reranker_name = settings["reranker"]

    def build() -> GroundedAnswerer:
        params = answer_params(lang, corpus_name)
        reranker = load_reranker(reranker_name)
        return GroundedAnswerer(
            OllamaClient(DEFAULT_LLM), lang,
            min_retrieval_score=params["min_retrieval_score"], min_sentence_support=params["min_sentence_support"],
            min_answer_relevance=params["min_answer_relevance"],
            relevance_scorer=reranker_relevance(reranker),
            # При отказе модели или отклонённом ответе — пункты закона, ближайшие к вопросу, дословно
            quote_scorer=lambda query, units: reranker.score(query, units),
            min_entailment=params["min_entailment"],
            entailment_scorer=load_entailment_scorer(params["min_entailment"]),
            # Для законов: ответ числом («24 [1]») запрашивается заново полным предложением, числа закона словами
            # («двадцать четыре») сверяются с цифрами ответа
            retry_short=settings.get("retry_short", False), retry_refusals=settings.get("retry_refusals", False),
            number_words=settings.get("number_words", False),
            # Для законов: после отказа модель спрашивают ещё раз по лучшим пунктам найденных статей
            focus_on_refusal=settings.get("focus_on_refusal", 0),
        )

    return model_store().get(("answerer", lang, corpus_name, reranker_name), build)


def search_query_for(query: str, lang: str, corpus_name: str) -> str:
    """Запрос поиска: для корпуса законов к вопросу дописываются термины закона (src/query_expansion.py)."""
    return expand_query(query) if corpus_settings(lang, corpus_name)["dictionary"] else query


def compute_answer(query: str, hits: List[Dict[str, Any]], lang: str, corpus_name: str) -> Dict[str, Any]:
    """Ответ модели по найденным фрагментам: {"answer", "passages", "error"} (error — Ollama недоступна)."""
    # Модель читает чанки с окружающим текстом; пересекающиеся склеиваются, поэтому номера [n] — номера passages
    passages = build_passages(hits)
    try:
        with st.spinner("Составляем ответ по тексту закона..."):
            contexts = {h["id"]: h.get("context", h["text"]) for h in hits}
            answer = load_answerer(lang, corpus_name).answer(
                query, passages, search_query=search_query_for(query, lang, corpus_name),
                source_texts=[contexts.get(p.id, p.text) for p in passages])
            return {"answer": answer, "passages": passages, "error": None}
    except OllamaError as e:
        return {"answer": None, "passages": passages, "error": str(e)}


def passage_title(passage: Passage, lang: str, corpus_name: str) -> str:
    """Заголовок статьи фрагмента (из метаданных чанка), если он есть."""
    return chunks_by_id(lang, corpus_name).get(passage.id, {}).get("article_title", "") or ""


def render_quotes(
    quotes: List[Dict[str, Any]], passages: List[Passage], lang: str, corpus_name: str, stems: Set[str],
    research: bool,
) -> None:
    """Пункты закона, ближайшие к вопросу по оценке реранкера, — дословно, без пересказа модели."""
    st.markdown("<p class='answer'>Готового ответа нет. Ближе всего к вашему вопросу эти нормы закона:</p>",
                unsafe_allow_html=True)
    for quote in quotes:
        passage = passages[quote["passage"] - 1]
        meta = f"оценка реранкера {quote['score']:.3f}" if research and quote.get("score") is not None else ""
        st.markdown(
            norm_html(passage.law, passage.articles, highlight(quote["text"].strip(), stems),
                      title=passage_title(passage, lang, corpus_name), url=source_url(lang, passage.source_id), meta=meta),
            unsafe_allow_html=True,
        )
    st.markdown("<p class='note caution'>Нормы выбраны автоматически и приведены дословно. Проверьте, подходят ли "
                "они к вашему случаю.</p>", unsafe_allow_html=True)


def render_answer(
    answer: Optional[Answer], passages: List[Passage], lang: str, corpus_name: str, error: Optional[str],
    trace_id: Optional[str] = None, stems: Optional[Set[str]] = None, research: bool = False,
) -> None:
    """Ответ и нормы закона, на которых он основан; в режиме исследователя — оценки и причины отказа.

    Под ответом — кнопки отзыва (если включён Langfuse).
    """
    stems = stems or set()
    if answer is None:
        if research:
            st.error(error)
        else:
            st.markdown("<p class='note caution'>Ответ сейчас недоступен: не запущена языковая модель. "
                        "Ниже — найденные статьи закона.</p>", unsafe_allow_html=True)
        return

    answerer = load_answerer(lang, corpus_name)
    top = answer.checks.get("top_score")
    if answer.status == STATUS_ANSWERED:
        st.markdown(f"<p class='answer'>{answer_html(answer.text)}</p>", unsafe_allow_html=True)
        note = "Ответ проверен по тексту закона: каждое число и каждое утверждение есть в статье ниже."
        if answer.checks.get("fallback"):
            note += " С первого раза модель ответа не нашла, поэтому он составлен по пунктам статей, ближайшим к вопросу."
        st.markdown(f"<p class='note ok'>{note}</p>", unsafe_allow_html=True)
        # Ссылки [n] — на фрагменты, по которым дан ответ (после повтора по пунктам статей — на пункты)
        for n in answer.cited:
            source = answer.passages[n - 1]
            st.markdown(
                norm_html(source.law, source.articles, highlight(source.text.strip(), stems),
                          title=passage_title(source, lang, corpus_name), url=source_url(lang, source.source_id), ref=n),
                unsafe_allow_html=True,
            )
        if research:
            relevance = answer.checks.get("answer_relevance")
            entailment = answer.checks.get("entailment")
            st.caption(
                f"Проверки: числа есть в тексте закона; слова каждого предложения найдены во фрагментах "
                f"не меньше чем на {answerer.min_sentence_support:.0%} (минимум {answer.checks['min_sentence_support']:.0%})"
                + (f"; соответствие вопросу {relevance:.2f} (порог {answerer.min_answer_relevance})" if relevance is not None else "")
                + (f"; следование из фрагмента {entailment:.2f}" if entailment is not None else "")
                + f"; уверенность поиска {top:.2f}; {answer.usage.get('latency_s', 0):.1f} с"
            )
    elif answer.status == STATUS_NO_CONTEXT:
        laws = sorted({c["law"] for c in build_corpus(lang, corpus_name)})
        st.markdown(
            "<p class='answer'>В законах справочника ответа на этот вопрос не нашлось.</p>"
            f"<p class='note'>Справочник ищет только в этих актах: {html.escape(', '.join(laws))}. Налоги, штрафы, "
            "уголовные дела, армия и многое другое в них не входят. Если тема подходит, попробуйте спросить иначе.</p>",
            unsafe_allow_html=True,
        )
        if research:
            st.caption(f"Модель не вызывалась: лучший фрагмент получил оценку реранкера {top:.2f}, "
                       f"порог {answerer.min_retrieval_score}.")
    else:
        if answer.quotes:
            render_quotes(answer.quotes, passages, lang, corpus_name, stems, research)
        else:
            st.markdown("<p class='answer'>Готового ответа нет. Ближе всего к вашему вопросу этот фрагмент закона:</p>",
                        unsafe_allow_html=True)
            first = passages[0]
            st.markdown(norm_html(first.law, first.articles, highlight(first.text.strip(), stems),
                                  title=passage_title(first, lang, corpus_name), url=source_url(lang, first.source_id)),
                        unsafe_allow_html=True)
        if research:
            reason = ("модель ответила, что во фрагментах ответа нет" if answer.status == STATUS_REFUSED
                      else "проверка нашла: " + "; ".join(answer.checks.get("problems", [])))
            with st.expander("Почему нет ответа модели"):
                st.write(reason[:1].upper() + reason[1:] + ".")
                if answer.raw and answer.status != STATUS_REFUSED:
                    st.caption("Отклонённый ответ модели (не является ответом системы):")
                    st.text(answer.raw)
                if answer.checks.get("fallback_raw"):
                    st.caption("Ответ после повтора по пунктам статей, тоже не принят:")
                    st.text(answer.checks["fallback_raw"])
    render_feedback(trace_id)


def render_feedback(trace_id: Optional[str]) -> None:
    """Кнопки 👍/👎: оценка user_feedback (1/0) ставится на трассу ответа в Langfuse.

    У оценки постоянный id, поэтому смена 👍 на 👎 перезаписывает её, а не добавляет вторую.
    """
    if not trace_id or get_langfuse() is None:
        return
    choice = st.feedback("thumbs", key=f"feedback-{trace_id}")
    sent_key = f"feedback-sent-{trace_id}"
    if choice is not None and st.session_state.get(sent_key) != choice:
        if score_trace(trace_id, "user_feedback", float(choice), data_type="BOOLEAN", score_id=f"{trace_id}-user-feedback"):
            st.session_state[sent_key] = choice
            st.toast("Спасибо, оценка сохранена.")


def load_models(lang: str, model_name: str, corpus_name: str, rerank: bool, generate: bool) -> LegalSearchEngine:
    """Загрузка индекса и моделей — отдельный этап трассы; если что-то ещё не загружено, это холодный старт.

    Обычно модели уже загрузил фоновый поток (warm_up); если он ещё работает, запрос ждёт его.
    """
    store = model_store()
    reranker_name = corpus_settings(lang, corpus_name)["reranker"]
    needed = [engine_key(lang, model_name, corpus_name)]
    if rerank:
        needed.append(("reranker", reranker_name))
    if generate:
        needed.append(("answerer", lang, corpus_name, reranker_name))
    cold = [":".join(key) for key in needed if not store.ready(key)]
    spinner = st.spinner("Загружаем справочник — это нужно только при первом вопросе...") if cold else nullcontext()
    with span("load-models", input=[":".join(key) for key in needed], metadata={"cold_start": bool(cold)}) as obs, spinner:
        engine = load_engine(lang, model_name, corpus_name)
        if rerank:
            engine.reranker = load_reranker(reranker_name)
        if generate:
            load_answerer(lang, corpus_name)
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
            result = {"hits": hits, "search_ms": (time.time() - t0) * 1000, "answer": None, "passages": [], "error": None,
                      "trace_id": root.trace_id}
            if hits and generate:
                result.update(compute_answer(query, hits[:DEFAULT_TOP_K], lang, corpus_name))
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


def research_settings() -> Dict[str, Any]:
    """Боковая панель режима исследователя: корпус, язык, метод, модель, реранкер, ответ модели, фильтр по законам."""
    with st.sidebar:
        st.header("Настройки для исследования")
        corpus_name = st.radio("Корпус", available_corpora(), format_func=CORPUS_LABELS.get, key="corpus")
        if corpus_name == "laws":
            lang = "rus"
        else:
            lang = st.radio("Язык корпуса", list(LANG_LABELS), format_func=LANG_LABELS.get, horizontal=True, key="lang")
        mode = st.radio("Метод поиска", list(MODE_LABELS), format_func=MODE_LABELS.get, key="mode")
        top_k = st.slider("Количество результатов", 1, 10, 5, key="top_k")
        model_name = st.selectbox("Модель эмбеддингов", available_models(lang), key="embedder")
        if mode == "hybrid":
            params = load_hybrid_params(model_name, lang)
            st.caption(f"RRF: вес Dense {params['dense_weight']}, вес BM25 {params['sparse_weight']}, k = {params['rrf_k']}")
        generate = st.toggle(
            "Ответ языковой модели", value=True, key="generate",
            help=f"{DEFAULT_LLM} (локально через Ollama) формулирует ответ по {DEFAULT_TOP_K} лучшим фрагментам. "
                 "Ответ, не подтверждённый текстом закона, не показывается. Включает реранкер.",
        )
        rerank = st.toggle(
            "Реранкер (cross-encoder)", value=True, disabled=generate, key="rerank",
            help="Переранжировать лучших кандидатов выбранного метода нейросетью, читающей вопрос и фрагмент вместе.",
        ) or generate
        settings = corpus_settings(lang, corpus_name)
        if rerank:
            st.caption(f"{settings['reranker']}, топ-{rerank_pool(lang)} кандидатов"
                       + ("; к вопросу дописываются термины закона" if settings["dictionary"] else ""))
        if generate:
            gen_params = answer_params(lang, corpus_name)
            gate = gen_params["min_retrieval_score"]
            relevance = gen_params["min_answer_relevance"]
            st.caption(
                f"{DEFAULT_LLM}: порог реранкера {gate if gate is not None else 'не задан'}, "
                f"опора предложений на текст ≥ {gen_params['min_sentence_support']}, "
                f"соответствие ответа вопросу ≥ {relevance if relevance is not None else 'не задано'}"
                + (f", повтор по {settings['focus_on_refusal']} пунктам после отказа"
                   if settings.get("focus_on_refusal") else "")
            )
        st.caption(
            "Langfuse: трассировка включена" if get_langfuse() is not None
            else "Langfuse: выключен (не заданы LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)"
        )
        corpus = build_corpus(lang, corpus_name)
        all_laws = sorted({c["law"] for c in corpus})
        laws = st.multiselect("Искать только в законах", all_laws, placeholder="Все законы", key="laws")
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
    return {"corpus_name": corpus_name, "lang": lang, "mode": mode, "top_k": top_k, "model_name": model_name,
            "generate": generate, "rerank": rerank, "laws": laws}


def citizen_settings() -> Dict[str, Any]:
    """Обычный режим: лучшая конфигурация (REPORT 7.15–7.18) — законы для граждан, гибрид, реранкер, ответ модели."""
    corpora = available_corpora()
    corpus_name = "laws" if "laws" in corpora else corpora[0]
    return {"corpus_name": corpus_name, "lang": "rus", "mode": "hybrid", "top_k": 5, "model_name": default_model("rus"),
            "generate": True, "rerank": True, "laws": []}


def render_disclaimer(corpus_name: str) -> None:
    """Оговорка внизу страницы: справочник, а не консультация; дата редакции законов."""
    if corpus_name == "laws":
        text = (f"Это справочник, а не юридическая консультация. Законы — в редакции на {laws_as_of()}. "
                "В спорной ситуации обратитесь к юристу.")
    else:
        text = ("Датасет используется для измерения качества поиска: часть его актов утратила силу, "
                "поэтому для реальных вопросов выберите корпус «Законы для граждан».")
    st.markdown(f"<p class='disclaimer'>{html.escape(text)}</p>", unsafe_allow_html=True)


def displayed_passages(answer: Answer, passages: List[Passage]) -> List[Passage]:
    """Фрагменты, которые render_answer показал под ответом: те, на которые ссылается ответ, процитированные пункты
    или первый фрагмент. Остальные фрагменты, полученные моделью, не показаны и попадают в «Другие найденные статьи»."""
    if answer.status == STATUS_ANSWERED:
        return [answer.passages[n - 1] for n in answer.cited]
    if answer.quotes:
        return [passages[quote["passage"] - 1] for quote in answer.quotes]
    return passages[:1]


def render_citizen_results(result: Dict[str, Any], lang: str, corpus_name: str, mode: str, top_k: int,
                           rerank: bool, stems: Set[str]) -> None:
    """Обычный режим: ответ с нормами закона, остальные найденные статьи свёрнуты."""
    answer = result["answer"]
    hits = result["hits"][:top_k]
    render_answer(answer, result["passages"], lang, corpus_name, result["error"], result.get("trace_id"), stems,
                  research=False)
    if answer is None:  # модель недоступна — найденные статьи и есть результат
        for rank, hit in enumerate(hits, 1):
            render_hit(rank, hit, mode, lang, stems, rerank=rerank, research=False)
        return
    if answer.status == STATUS_NO_CONTEXT:  # поиск не уверен: найденные статьи к вопросу, скорее всего, не относятся
        return
    shown = {chunk_id for p in displayed_passages(answer, result["passages"]) for chunk_id in p.ids}
    others = [h for h in hits if h["id"] not in shown]
    if others:
        with st.expander(f"Другие найденные статьи ({len(others)})"):
            for rank, hit in enumerate(others, 1):
                render_hit(rank, hit, mode, lang, stems, compact=True, rerank=rerank, research=False)


def render_compare(query: str, lang: str, model_name: str, corpus_name: str, laws: List[str], rerank: bool,
                   stems: Set[str]) -> None:
    """Вкладка режима исследователя: три лучших результата каждого метода по одному вопросу."""
    cache = result_cache()
    st.caption("Три лучших результата каждого метода по одному и тому же вопросу.")
    engine = load_models(lang, model_name, corpus_name, rerank, False)
    # С включённым реранкером добавляется колонка Hybrid + реранкер
    columns = [(m, False) for m in MODE_LABELS] + ([("hybrid", True)] if rerank else [])
    for col, (m, m_rerank) in zip(st.columns(len(columns)), columns):
        with col:
            st.subheader(m.upper() + (" + реранкер" if m_rerank else ""))
            key = ("compare", corpus_name, lang, model_name, query.strip(), m, tuple(laws), m_rerank)
            hits = cache.get(key)
            if hits is None:
                hits = run_search(engine, lang, query, m, 3, laws, m_rerank, corpus_name)
                cache.put(key, hits)
                release_gpu_memory()
            for rank, hit in enumerate(hits, 1):
                render_hit(rank, hit, m, lang, stems, compact=True, rerank=m_rerank)


def main() -> None:
    research = bool(st.session_state.get("research", False))
    st.set_page_config(page_title="Законы Казахстана — справочник", page_icon="⚖️",
                       layout="wide" if research else "centered")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    head, switch = st.columns([4, 1.4], vertical_alignment="bottom")
    with switch:
        research = st.toggle("Режим исследователя", key="research",
                             help="Настройки поиска и модели, оценки проверок и сравнение методов — для защиты и экспериментов.")
    with head:
        st.title("Справочник по законам Казахстана")
    s = research_settings() if research else citizen_settings()
    corpus_name, lang, mode, top_k = s["corpus_name"], s["lang"], s["mode"], s["top_k"]
    model_name, generate, rerank, laws = s["model_name"], s["generate"], s["rerank"], s["laws"]

    lede = ("Спросите своими словами. Ответ строится только по тексту закона, а рядом показана сама статья "
            "со ссылкой на источник." if corpus_name == "laws"
            else "Поиск по датасету Adilet Legal QA: фрагменты 14 нормативных актов, вопросы языком закона.")
    st.markdown(f"<p class='lede'>{lede}</p>", unsafe_allow_html=True)

    examples = LAWS_EXAMPLES if corpus_name == "laws" else EXAMPLES[lang]
    picked = st.pills("Можно начать с примера", examples, key=f"example-{corpus_name}-{lang}")
    # Снятый выбор тоже запоминается: иначе повторный выбор того же примера не подставил бы его в вопрос
    if st.session_state.get("example_used") != picked:
        if picked:
            st.session_state.query = picked
        st.session_state.example_used = picked

    query = st.text_input("Ваш вопрос", key="query", placeholder="Например: сколько дней отпуска положено в год?")
    # Страница уже показана: пока пользователь набирает вопрос, модели загружаются в фоне
    start_warm_up(lang, model_name, corpus_name)
    if generate:
        start_answer_warm_up(lang, corpus_name)
    if not query.strip():
        render_disclaimer(corpus_name)
        return

    stems = query_stems(query)
    cache = result_cache()
    tab_search, tab_compare = st.tabs(["Результаты", "Сравнение методов"]) if research else (st.container(), None)

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
            st.markdown("<p class='note'>Ничего не найдено. Попробуйте спросить иначе или назовите тему: работа, "
                        "семья, жильё, покупки, наследство.</p>", unsafe_allow_html=True)
        elif research:
            if generate:
                render_answer(result["answer"], result["passages"], lang, corpus_name, result["error"],
                              result.get("trace_id"), stems, research=True)
            label = MODE_LABELS[mode] + (" + реранкер" if rerank else "")
            st.subheader("Найденные фрагменты")
            st.caption(f"{label}: {len(hits[:top_k])} результатов за {result['search_ms']:.0f} мс")
            for rank, hit in enumerate(hits[:top_k], 1):
                render_hit(rank, hit, mode, lang, stems, rerank=rerank)
        else:
            render_citizen_results(result, lang, corpus_name, mode, top_k, rerank, stems)
        render_disclaimer(corpus_name)

    if tab_compare is not None:
        with tab_compare:
            render_compare(query, lang, model_name, corpus_name, laws, rerank, stems)


if __name__ == "__main__":
    main()
