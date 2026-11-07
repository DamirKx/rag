"""Веб-демо поиска по законодательству Республики Казахстан (Streamlit).

Интерфейс к тем же компонентам, что и консольный ask.py: Dense (E5 + FAISS),
BM25 и Hybrid (RRF). В отличие от ask.py, поиск идёт по чанкам 500 символов
(лучшая стратегия в бенчмарке run_adilet_eval.py), а для каждого результата
показываются закон, номер статьи, ссылка на adilet.zan.kz и полный фрагмент.
Кандидатов любого метода можно переранжировать cross-encoder (src/reranker.py).

Запуск:
    streamlit run app.py
"""

import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import streamlit as st
import torch

from ask import (
    LegalSearchEngine, default_model, load_bm25_params, load_hybrid_params, load_legal_corpus, load_rerank_pool,
)
from run_adilet_eval import DEFAULT_MODEL, model_cache_name
from src.corpus import CHUNK_OVERLAP, CHUNK_SIZE, build_chunk_corpus, law_titles
from src.embeddings import EmbeddingModel
from src.generation import (
    DEFAULT_LLM, DEFAULT_TOP_K, STATUS_ANSWERED, STATUS_NO_CONTEXT, STATUS_REFUSED,
    GroundedAnswerer, OllamaClient, OllamaError, Passage, load_generation_params, reranker_relevance,
)
from src.reranker import CrossEncoderReranker, default_reranker
from src.retriever import tokenize_legal_text

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

# Служебные слова не подсвечиваются в найденном тексте
STOPWORDS = {
    "какой", "какая", "какие", "каких", "каком", "какого", "кто", "что", "когда", "где", "как",
    "для", "при", "или", "если", "это", "его", "она", "они", "который", "которые", "течение",
    "может", "могут", "быть", "есть", "имеет",
    "қандай", "кім", "не", "қашан", "қалай", "және", "немесе", "үшін",
}


@st.cache_data(show_spinner="Загрузка и разбиение корпуса законов...")
def build_corpus(lang: str) -> List[Dict[str, Any]]:
    """Чанки 500 символов с законом и статьями (src/corpus.py).

    Чанки совпадают с run_adilet_eval.py, поэтому используется уже готовый кэш эмбеддингов.
    """
    titles = law_titles(pd.read_parquet(CACHE_PARQUET, columns=["context", "source_id", "doc_type", "language"]))
    return build_chunk_corpus(load_legal_corpus(CACHE_PARQUET, lang), titles)


@st.cache_resource(show_spinner="Загрузка модели эмбеддингов...")
def load_embedder(model_name: str) -> EmbeddingModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EmbeddingModel(model_name=model_name, device=device)


@st.cache_resource(show_spinner="Построение поискового индекса (при первом запуске векторизация занимает несколько минут)...")
def load_engine(lang: str, model_name: str) -> LegalSearchEngine:
    corpus = build_corpus(lang)
    return LegalSearchEngine(
        docs=[c["text"] for c in corpus],
        doc_ids=[c["id"] for c in corpus],
        embedder=load_embedder(model_name),
        cache_dir=CACHE_DIR,
        cache_key=f"{model_cache_name(model_name)}_{lang}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
        hybrid_params=load_hybrid_params(model_name, lang),
        bm25_params=load_bm25_params(lang),
    )


@st.cache_resource(show_spinner="Загрузка реранкера cross-encoder...")
def load_reranker(model_name: str) -> CrossEncoderReranker:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoderReranker(model_name, device=device)


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
    engine: LegalSearchEngine, lang: str, query: str, mode: str, top_k: int, laws: List[str], rerank: bool = False
) -> List[Dict[str, Any]]:
    """Поиск с фильтром по законам; при rerank фильтр применяется до переранжирования."""
    by_id = {c["id"]: c for c in build_corpus(lang)}
    wanted = max(top_k, engine.rerank_pool) if rerank else top_k
    pool = FILTER_POOL if laws else wanted
    hits = []
    for doc_id, _, score in engine.search(query, mode=mode, top_k=pool):
        chunk = by_id[doc_id]
        if laws and chunk["law"] not in laws:
            continue
        hits.append({**chunk, "score": score})
        if len(hits) == wanted:
            break
    if rerank:
        reranked = engine.rerank(query, [(h["id"], h["text"], h["score"]) for h in hits], top_k)
        hits = [{**by_id[doc_id], "score": score} for doc_id, _, score in reranked]
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
    url = f"https://adilet.zan.kz/{lang}/docs/{hit['source_id']}"
    with st.container(border=True):
        st.markdown(f"**{rank}. {hit['law']}**" + (f" · {hit['articles']}" if hit["articles"] else ""))
        if hit["article_title"] and not compact:
            st.caption(hit["article_title"])
        st.markdown(f"<div class='hit-text'>{highlight(hit['text'], stems)}</div>", unsafe_allow_html=True)
        st.caption(f"{score_text(mode, hit['score'], rerank)} · [открыть на adilet.zan.kz]({url})")
        if not compact:
            with st.expander("Весь фрагмент закона"):
                st.markdown(f"<div class='hit-text'>{highlight(hit['context'], stems)}</div>", unsafe_allow_html=True)


@st.cache_resource
def load_answerer(lang: str) -> GroundedAnswerer:
    """Ответ модели с проверками; соответствие ответа вопросу оценивает тот же реранкер, что и поиск."""
    params = load_generation_params(lang, default_reranker(lang))
    return GroundedAnswerer(
        OllamaClient(DEFAULT_LLM), lang,
        min_retrieval_score=params["min_retrieval_score"], min_sentence_support=params["min_sentence_support"],
        min_answer_relevance=params["min_answer_relevance"],
        relevance_scorer=reranker_relevance(load_reranker(default_reranker(lang))),
    )


def render_answer(query: str, hits: List[Dict[str, Any]], lang: str) -> None:
    """Ответ модели по найденным фрагментам с итогами проверки и источниками."""
    answerer = load_answerer(lang)
    passages = [Passage.from_hit(h) for h in hits]
    with st.container(border=True):
        st.markdown("**Ответ**")
        try:
            with st.spinner(f"{DEFAULT_LLM} формулирует ответ по найденным фрагментам..."):
                answer = answerer.answer(query, passages)
        except OllamaError as e:
            st.error(str(e))
            return

        top = answer.checks.get("top_score")
        if answer.status == STATUS_ANSWERED:
            st.markdown(answer.text.replace("\n", "  \n"))
            sources = []
            for n in answer.cited:
                hit = hits[n - 1]
                url = f"https://adilet.zan.kz/{lang}/docs/{hit['source_id']}"
                sources.append(f"[{n}] [{passages[n - 1].label}]({url})")
            st.caption("Источники: " + " · ".join(sources))
            relevance = answer.checks.get("answer_relevance")
            st.caption(
                f"✓ Проверено: все числа есть в тексте закона; слова каждого предложения найдены во фрагментах "
                f"не меньше чем на {answerer.min_sentence_support:.0%} (минимум {answer.checks['min_sentence_support']:.0%})"
                + (f"; соответствие вопросу {relevance:.2f}" if relevance is not None else "")
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
            st.markdown("> " + hits[0]["text"].strip().replace("\n", "\n> "))
            st.caption(f"[1] {passages[0].label}")
            with st.expander("Почему ответ модели не показан"):
                st.write("Проверка нашла: " + "; ".join(answer.checks["problems"]) + ".")
                st.caption("Отклонённый ответ модели (не является ответом системы):")
                st.text(answer.raw)


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
        corpus = build_corpus(lang)
        all_laws = sorted({c["law"] for c in corpus})
        laws = st.multiselect("Искать только в законах", all_laws, placeholder="Все законы")
        st.divider()
        st.caption(
            f"Корпус: {len(all_laws)} нормативных актов, "
            f"{len({c['id'].split('_fix_')[0] for c in corpus})} фрагментов, {len(corpus)} чанков "
            f"по {CHUNK_SIZE} символов. Данные: Adilet Legal QA "
            f"(названия законов восстановлены по шапкам страниц adilet.zan.kz)."
        )

    st.title("⚖️ Поиск по законодательству Республики Казахстан")
    st.write(
        "Задайте вопрос обычным языком — система найдёт фрагменты нормативных актов, "
        "в которых содержится ответ. Каждый результат — текст закона со ссылкой на источник; по желанию языковая "
        "модель сформулирует ответ только по найденным фрагментам."
    )

    examples = EXAMPLES[lang]
    st.caption("Примеры вопросов:")
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        if col.button(example, width="stretch"):
            st.session_state.query = example

    query = st.text_input("Ваш вопрос", key="query", placeholder="Например: кто может быть избран Президентом?")
    if not query.strip():
        return

    engine = load_engine(lang, model_name)
    if rerank:
        engine.reranker = load_reranker(default_reranker(lang))
        engine.rerank_pool = rerank_pool(lang)
    stems = query_stems(query)
    tab_search, tab_compare = st.tabs(["Результаты", "Сравнение методов"])

    with tab_search:
        t0 = time.time()
        # Модели нужны DEFAULT_TOP_K лучших фрагментов, даже если показать нужно меньше
        hits = run_search(engine, lang, query, mode, max(top_k, DEFAULT_TOP_K) if generate else top_k, laws, rerank)
        elapsed_ms = (time.time() - t0) * 1000
        if not hits:
            st.info("Ничего не найдено. Попробуйте переформулировать вопрос или снять фильтр по законам.")
        else:
            if generate:
                render_answer(query, hits[:DEFAULT_TOP_K], lang)
            label = MODE_LABELS[mode] + (" + реранкер" if rerank else "")
            st.caption(f"{label} · найдено {len(hits[:top_k])} за {elapsed_ms:.0f} мс")
            for rank, hit in enumerate(hits[:top_k], 1):
                render_hit(rank, hit, mode, lang, stems, rerank=rerank)

    with tab_compare:
        st.caption("Топ-3 каждого метода по одному и тому же вопросу.")
        # С включённым реранкером добавляется колонка Hybrid + реранкер
        columns = [(m, False) for m in MODE_LABELS] + ([("hybrid", True)] if rerank else [])
        for col, (m, m_rerank) in zip(st.columns(len(columns)), columns):
            with col:
                st.subheader(m.upper() + (" + RERANK" if m_rerank else ""))
                for rank, hit in enumerate(run_search(engine, lang, query, m, 3, laws, m_rerank), 1):
                    render_hit(rank, hit, m, lang, stems, compact=True, rerank=m_rerank)


if __name__ == "__main__":
    main()
