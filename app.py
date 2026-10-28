"""Веб-демо поиска по законодательству Республики Казахстан (Streamlit).

Интерфейс к тем же компонентам, что и консольный ask.py: Dense (E5 + FAISS),
BM25 и Hybrid (RRF). В отличие от ask.py, поиск идёт по чанкам 500 символов
(лучшая стратегия в бенчмарке run_adilet_eval.py), а для каждого результата
показываются закон, номер статьи, ссылка на adilet.zan.kz и полный фрагмент.

Запуск:
    streamlit run app.py
"""

import html
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import streamlit as st
import torch

from ask import LegalSearchEngine, load_legal_corpus
from run_adilet_eval import DEFAULT_MODEL
from src.chunking import get_fixed_size_chunks
from src.embeddings import EmbeddingModel
from src.retriever import tokenize_legal_text

CACHE_PARQUET = "data/raw/adilet_legal_qa_kz.parquet"
CACHE_DIR = "data/adilet/cache"
MODELS_DIR = Path("models")
CHUNK_SIZE, CHUNK_OVERLAP = 500, 50
# Сколько кандидатов достаём перед фильтрацией по законам
FILTER_POOL = 300

# Заголовок статьи для подписи результата. Мягче, чем src.chunking.ARTICLE_HEADING:
# в Конституции заголовки записаны без точки («Статья 41»).
ARTICLE_HEADING = re.compile(
    r"^[ \t]*(?:Статья\s+(?P<rus>\d+(?:-\d+)?)|(?P<kaz>\d+(?:-\d+)?)-бап)\b",
    re.IGNORECASE | re.MULTILINE,
)

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


# Название документа в шапке страницы adilet.zan.kz: «О языках в Республике Казахстан - ИПС "Әділет" ...»
PAGE_TITLE = re.compile(r'^\s*(.+?)\s+-\s+ИПС\s+"Әділет"')
TITLE_PREFIX = {"law": "Закон РК", "const_law": "Конституционный закон РК"}


@st.cache_data
def law_titles() -> Dict[str, str]:
    """Названия документов по source_id, взятые из шапки исходной страницы.

    Поле source_title датасета у 9 из 14 документов не соответствует тексту
    (например, Z950002444_ подписан как «О хозяйственных товариществах»,
    а это Закон «О банках и банковской деятельности»). Шапка страницы есть только
    в первом русском фрагменте каждого документа; казахские фрагменты получают
    название по тому же source_id.
    """
    df = pd.read_parquet(CACHE_PARQUET, columns=["context", "source_id", "doc_type", "language"])
    titles = {}
    for row in df[df["language"] == "rus"].itertuples():
        match = PAGE_TITLE.match(row.context)
        if match and row.source_id not in titles:
            title = match.group(1).replace("Республики Казахстан", "РК")
            prefix = TITLE_PREFIX.get(row.doc_type)
            titles[row.source_id] = f"{prefix} «{title}»" if prefix else title
    return titles


def article_label(match: re.Match) -> str:
    """Короткая подпись статьи: «ст. 10» для русского текста, «10-бап» для казахского."""
    return f"ст. {match.group('rus')}" if match.group("rus") else f"{match.group('kaz')}-бап"


@st.cache_data(show_spinner="Загрузка и разбиение корпуса законов...")
def build_corpus(lang: str) -> List[Dict[str, Any]]:
    """Разбивает фрагменты законов на чанки и определяет для каждого чанка статьи.

    Чанки совпадают с run_adilet_eval.py, поэтому используется уже готовый кэш эмбеддингов.
    """
    sub_df = load_legal_corpus(CACHE_PARQUET, lang)
    titles = law_titles()
    corpus: List[Dict[str, Any]] = []
    for idx, row in sub_df.iterrows():
        text = row["cleaned_context"]
        headings = list(ARTICLE_HEADING.finditer(text))
        search_from = 0
        for s_idx, chunk in enumerate(get_fixed_size_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            start = text.find(chunk, search_from)
            if start < 0:
                start = search_from
            end = start + len(chunk)
            search_from = start + 1

            # Статья, внутри которой начинается чанк, и статьи, начинающиеся внутри него
            before = [h for h in headings if h.start() <= start]
            inside = [h for h in headings if start < h.start() < end]
            covered = before[-1:] + inside
            title = covered[0].string[covered[0].start():].split("\n", 1)[0].strip() if covered else ""

            corpus.append({
                "id": f"ctx_{idx:04d}_fix_{s_idx:03d}",
                "text": chunk,
                "context": text,
                "law": titles.get(row["source_id"], row["source_title"]),
                "source_id": row["source_id"],
                "articles": ", ".join(article_label(h) for h in covered),
                "article_title": title,
            })
    return corpus


@st.cache_resource(show_spinner="Загрузка модели эмбеддингов...")
def load_embedder(model_name: str) -> EmbeddingModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return EmbeddingModel(model_name=model_name, device=device)


@st.cache_resource(show_spinner="Построение поискового индекса (при первом запуске векторизация занимает несколько минут)...")
def load_engine(lang: str, model_name: str) -> LegalSearchEngine:
    corpus = build_corpus(lang)
    safe_model = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name)
    # Дообученная модель может быть перезаписана: время изменения папки входит в ключ кэша
    model_path = Path(model_name)
    if model_path.exists():
        safe_model += f"_{int(model_path.stat().st_mtime)}"
    return LegalSearchEngine(
        docs=[c["text"] for c in corpus],
        doc_ids=[c["id"] for c in corpus],
        embedder=load_embedder(model_name),
        cache_dir=CACHE_DIR,
        cache_key=f"{safe_model}_{lang}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
    )


def available_models() -> List[str]:
    """Базовая модель с Hub и дообученные модели из папки models/."""
    local = sorted(str(p) for p in MODELS_DIR.glob("*") if (p / "config.json").exists()) if MODELS_DIR.exists() else []
    return [DEFAULT_MODEL] + local


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


def run_search(engine: LegalSearchEngine, lang: str, query: str, mode: str, top_k: int, laws: List[str]) -> List[Dict[str, Any]]:
    by_id = {c["id"]: c for c in build_corpus(lang)}
    pool = FILTER_POOL if laws else top_k
    hits = []
    for doc_id, _, score in engine.search(query, mode=mode, top_k=pool):
        chunk = by_id[doc_id]
        if laws and chunk["law"] not in laws:
            continue
        hits.append({**chunk, "score": score})
        if len(hits) == top_k:
            break
    return hits


def score_text(mode: str, score: float) -> str:
    if mode == "dense":
        return f"косинусное сходство {score:.3f}"
    if mode == "bm25":
        return f"BM25 {score:.2f}"
    return f"RRF {score:.4f}"


def render_hit(rank: int, hit: Dict[str, Any], mode: str, lang: str, stems: Set[str], compact: bool = False) -> None:
    url = f"https://adilet.zan.kz/{lang}/docs/{hit['source_id']}"
    with st.container(border=True):
        st.markdown(f"**{rank}. {hit['law']}**" + (f" · {hit['articles']}" if hit["articles"] else ""))
        if hit["article_title"] and not compact:
            st.caption(hit["article_title"])
        st.markdown(f"<div class='hit-text'>{highlight(hit['text'], stems)}</div>", unsafe_allow_html=True)
        st.caption(f"{score_text(mode, hit['score'])} · [открыть на adilet.zan.kz]({url})")
        if not compact:
            with st.expander("Весь фрагмент закона"):
                st.markdown(f"<div class='hit-text'>{highlight(hit['context'], stems)}</div>", unsafe_allow_html=True)


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
        model_name = st.selectbox("Модель эмбеддингов", available_models())
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
        "в которых содержится ответ. Генеративная модель не используется: каждый результат — текст закона со ссылкой на источник."
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
    stems = query_stems(query)
    tab_search, tab_compare = st.tabs(["Результаты", "Сравнение методов"])

    with tab_search:
        t0 = time.time()
        hits = run_search(engine, lang, query, mode, top_k, laws)
        elapsed_ms = (time.time() - t0) * 1000
        if not hits:
            st.info("Ничего не найдено. Попробуйте переформулировать вопрос или снять фильтр по законам.")
        else:
            st.caption(f"{MODE_LABELS[mode]} · найдено {len(hits)} за {elapsed_ms:.0f} мс")
            for rank, hit in enumerate(hits, 1):
                render_hit(rank, hit, mode, lang, stems)

    with tab_compare:
        st.caption("Топ-3 каждого метода по одному и тому же вопросу.")
        for col, m in zip(st.columns(3), MODE_LABELS):
            with col:
                st.subheader(m.upper())
                for rank, hit in enumerate(run_search(engine, lang, query, m, 3, laws), 1):
                    render_hit(rank, hit, m, lang, stems, compact=True)


if __name__ == "__main__":
    main()
