"""Интерактивный консольный поиск по законодательству Республики Казахстан.

Позволяет в реальном времени задавать любые юридические вопросы на русском или казахском языках
и мгновенно получать релевантные статьи нормативно-правовых актов (НПА).

Поддерживает три режима поиска:
1. 'dense' — семантический векторный поиск на базе FAISS (нейросеть E5);
2. 'bm25' — лексический поиск по ключевым словам и номерам статей;
3. 'hybrid' — гибридное ранжирование Reciprocal Rank Fusion (RRF) Dense + BM25.

Любой режим можно дополнить переранжированием кандидатов cross-encoder (флаг --rerank).

С флагом --generate (или командой ':answer on') над результатами поиска выводится ответ локальной
языковой модели (src/generation.py) со ссылками на статьи. Ответ строится по той же схеме, на которой
eval_generation.py подбирал пороги проверки: чанки 500 символов, гибридный поиск, реранкер, 3 лучших чанка.
Ответ, не подтверждённый текстом закона, не показывается.
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# Обеспечиваем корректный вывод UTF-8 в консоли Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.corpus import CHUNK_OVERLAP, CHUNK_SIZE, build_chunk_corpus, expand_hits, law_titles
from src.nli import load_entailment_scorer
from src.embeddings import EmbeddingModel
from src.generation import (
    DEFAULT_LLM, DEFAULT_TOP_K, STATUS_ANSWERED, STATUS_NO_CONTEXT, STATUS_REFUSED,
    Answer, GroundedAnswerer, OllamaClient, OllamaError, Passage, load_generation_params, reranker_relevance,
)
from src.retriever import VectorRetriever, BM25Retriever, HybridRetriever
from src.reranker import CrossEncoderReranker, DEFAULT_RERANKER, FINETUNED_RERANKER, default_reranker
from prepare_adilet_data import clean_adilet_context
from run_adilet_eval import get_or_compute_embeddings, model_cache_name, DEFAULT_MODEL

SEARCH_MODES = ("dense", "bm25", "hybrid")
# Пул реранкера для чанков 500 символов, если eval_reranker.py ещё не запускался
DEFAULT_CHUNK_RERANK_POOL = 30
# Модель, дообученная в notebooks/finetune_e5.ipynb
FINETUNED_MODEL = "models/e5-small-adilet-{lang}"
DEFAULT_HYBRID_PARAMS = {"dense_weight": 0.5, "sparse_weight": 0.5, "rrf_k": 60}
# Сколько кандидатов первого этапа переранжирует cross-encoder. Фрагменты законов в ask.py
# длинные (каждый — около 6 окон реранкера), поэтому пул меньше, чем у чанков в app.py
ASK_RERANK_POOL = 10


def default_model(lang: str) -> str:
    """Дообученная модель для языка, если она есть, иначе базовая с Hub."""
    path = FINETUNED_MODEL.format(lang=lang)
    return path if (Path(path) / "config.json").exists() else DEFAULT_MODEL


def load_bm25_params(lang: str) -> Dict[str, Any]:
    """Параметры BM25, подобранные tune_bm25.py на val для фиксированных чанков 500 символов.

    Пустой словарь (настройки BM25Retriever по умолчанию, как у прежнего rank_bm25), если подбора не было.
    На фрагментах законов целиком (ask.py) эти параметры хуже настроек по умолчанию — там их не используем.
    """
    path = Path(f"data/adilet/bm25_tuning_{lang}.json")
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8"))["best"])
    return {}


def load_hybrid_params(model_name: str, lang: str) -> Dict[str, Any]:
    """Параметры RRF, подобранные tune_hybrid.py на val для фиксированных чанков 500 символов."""
    path = Path(f"data/adilet/hybrid_tuning_{lang}.json")
    if path.exists():
        tuned = json.loads(path.read_text(encoding="utf-8"))["models"].get(Path(model_name).as_posix())
        if tuned:
            return {key: tuned[key] for key in DEFAULT_HYBRID_PARAMS}
    return dict(DEFAULT_HYBRID_PARAMS)


def load_rerank_pool(lang: str, reranker: Optional[str] = None) -> int:
    """Размер пула кандидатов реранкера, выбранный eval_reranker.py на val (для чанков 500 символов)."""
    path = Path(f"data/adilet/reranker_eval_{lang}.json")
    if path.exists():
        tuned = json.loads(path.read_text(encoding="utf-8"))["rerankers"].get(reranker or default_reranker(lang))
        if tuned:
            return tuned["best_pool"]
    return DEFAULT_CHUNK_RERANK_POOL


def load_legal_corpus(cache_parquet: str, lang: str) -> pd.DataFrame:
    """Загружает и очищает корпус законов."""
    if not os.path.exists(cache_parquet):
        raise FileNotFoundError(
            f"Файл {cache_parquet} не найден! Сначала запустите prepare_adilet_data.py"
        )
    df = pd.read_parquet(cache_parquet)
    sub_df = df[df["language"] == lang].copy().reset_index(drop=True)
    sub_df["cleaned_context"] = sub_df["context"].apply(clean_adilet_context)
    return sub_df


def format_preview(text: str, max_chars: int = 350) -> str:
    """Форматирует аккуратное превью текста статьи."""
    clean = re.sub(r'\s+', ' ', text).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(' ', 1)[0] + "..."


class LegalSearchEngine:
    """Поисковый движок по корпусу НПА с тремя режимами: dense, bm25, hybrid.

    BM25-индекс строится лениво — при первом обращении к режиму 'bm25' или 'hybrid'.
    Если задан reranker, search(..., rerank=True) переранжирует rerank_pool лучших
    кандидатов выбранного режима и возвращает оценки cross-encoder.
    """

    def __init__(
        self,
        docs: List[str],
        doc_ids: List[str],
        embedder: EmbeddingModel,
        cache_dir: str,
        cache_key: str,
        hybrid_params: Optional[Dict[str, Any]] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_pool: int = 30,
        bm25_params: Optional[Dict[str, Any]] = None,
    ):
        self.embedder = embedder
        self.hybrid_params = hybrid_params or dict(DEFAULT_HYBRID_PARAMS)
        self.bm25_params = dict(bm25_params or {})
        self.reranker = reranker
        self.rerank_pool = rerank_pool
        self.docs_by_id = dict(zip(doc_ids, docs))
        self._docs = docs
        self._doc_ids = doc_ids

        emb_matrix = get_or_compute_embeddings(
            embedder=embedder,
            texts=docs,
            cache_dir=cache_dir,
            cache_key=cache_key,
            use_cache=True
        )
        self.dense_retriever = VectorRetriever(embedding_dim=emb_matrix.shape[1])
        self.dense_retriever.add_documents(emb_matrix, docs, doc_ids)

        self._bm25_retriever = None
        self._hybrid_retriever = None

    @property
    def bm25_retriever(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever(**self.bm25_params)
            self._bm25_retriever.add_documents(self._docs, self._doc_ids)
        return self._bm25_retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        if self._hybrid_retriever is None:
            self._hybrid_retriever = HybridRetriever(
                vector_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
                **self.hybrid_params
            )
        return self._hybrid_retriever

    def search(self, query: str, mode: str, top_k: int, rerank: bool = False) -> List[Tuple[str, str, float]]:
        """Выполняет поиск и возвращает список троек (doc_id, текст, оценка)."""
        if rerank:
            if self.reranker is None:
                raise ValueError("Реранкер не задан")
            candidates = self.search(query, mode, max(top_k, self.rerank_pool))
            return self.rerank(query, candidates, top_k)

        if mode == "hybrid":
            q_emb = self.embedder.encode_queries([query])[0]
            pairs = self.hybrid_retriever.search_with_scores(query, q_emb, k=top_k)
        elif mode == "bm25":
            pairs = self.bm25_retriever.search(query, k=top_k)
        elif mode == "dense":
            q_emb = self.embedder.encode_queries([query])
            scores, indices = self.dense_retriever.search(q_emb, k=top_k)
            pairs = [
                (self.dense_retriever.get_doc_id(idx), float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
        else:
            raise ValueError(f"Неизвестный режим поиска: {mode}")

        return [(doc_id, self.docs_by_id[doc_id], score) for doc_id, score in pairs]

    def rerank(self, query: str, candidates: List[Tuple[str, str, float]], top_k: int) -> List[Tuple[str, str, float]]:
        """Переранжирует готовый список троек (doc_id, текст, оценка) cross-encoder."""
        pairs = self.reranker.rerank(query, [(doc_id, text) for doc_id, text, _ in candidates], top_k=top_k)
        return [(doc_id, self.docs_by_id[doc_id], score) for doc_id, score in pairs]


class RAGAnswerer:
    """Ответ на вопрос: гибридный поиск по чанкам 500 символов → реранкер → языковая модель с проверкой.

    Та же схема, что в веб-демо (app.py) и в eval_generation.py, поэтому подходят пороги проверки,
    подобранные на validation, и готовый кэш эмбеддингов чанков.
    """

    def __init__(
        self,
        lang: str,
        embedder: EmbeddingModel,
        model_name: str,
        reranker: CrossEncoderReranker,
        reranker_name: str,
        cache_dir: str,
        cache_parquet: str,
        llm: str = DEFAULT_LLM,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.lang = lang
        self.top_k = top_k
        titles = law_titles(pd.read_parquet(cache_parquet, columns=["context", "source_id", "doc_type", "language"]))
        corpus = build_chunk_corpus(load_legal_corpus(cache_parquet, lang), titles)
        self.chunks_by_id = {c["id"]: c for c in corpus}
        self.engine = LegalSearchEngine(
            docs=[c["text"] for c in corpus],
            doc_ids=[c["id"] for c in corpus],
            embedder=embedder,
            cache_dir=cache_dir,
            cache_key=f"{model_cache_name(model_name)}_{lang}_fixed_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
            hybrid_params=load_hybrid_params(model_name, lang),
            bm25_params=load_bm25_params(lang),
            reranker=reranker,
            rerank_pool=load_rerank_pool(lang, reranker_name),
        )
        # Пороги в шкале реранкера отключаются, если они подобраны для другой модели реранкера
        params = load_generation_params(lang, reranker_name)
        self.answerer = GroundedAnswerer(
            OllamaClient(llm), lang,
            min_retrieval_score=params["min_retrieval_score"],
            min_sentence_support=params["min_sentence_support"],
            min_answer_relevance=params["min_answer_relevance"],
            relevance_scorer=reranker_relevance(reranker),
            min_entailment=params["min_entailment"],
            entailment_scorer=load_entailment_scorer(params["min_entailment"]),
        )

    def ask(self, query: str) -> Answer:
        """Ищет top_k чанков и отвечает по ним (модель читает их с окружающим текстом); OllamaError — если модель недоступна."""
        hits = self.engine.search(query, mode="hybrid", top_k=self.top_k, rerank=True)
        chunks = [{**self.chunks_by_id[doc_id], "score": score} for doc_id, _, score in hits]
        return self.answerer.answer(query, [Passage.from_hit(h) for h in expand_hits(chunks)])


def format_answer(answer: Answer, answerer: GroundedAnswerer, lang: str) -> str:
    """Текст ответа для консоли: ответ с источниками, отказ или цитата закона с причинами."""
    def source(n: int) -> str:
        p = answer.passages[n - 1]
        return f"  [{n}] {p.label} — https://adilet.zan.kz/{lang}/docs/{p.source_id}"

    lines = ["💬 Ответ:"]
    if answer.status == STATUS_ANSWERED:
        lines.append(answer.text)
        lines.append("\nИсточники:")
        lines.extend(source(n) for n in answer.cited)
        relevance = answer.checks.get("answer_relevance")
        entailment = answer.checks.get("entailment")
        lines.append(
            f"✓ Проверено: все числа есть в тексте закона; слова каждого предложения найдены во фрагментах "
            f"не меньше чем на {answerer.min_sentence_support:.0%} (минимум {answer.checks['min_sentence_support']:.0%})"
            + (f"; соответствие вопросу {relevance:.2f}" if relevance is not None else "")
            + (f"; следование из фрагмента {entailment:.2f}" if entailment is not None else "")
            + f" · {answer.usage.get('latency_s') or 0:.1f} с"
        )
    elif answer.status == STATUS_NO_CONTEXT:
        lines.append(answer.text)
        top = answer.checks.get("top_score")
        if top is not None:
            lines.append(
                f"Модель не вызывалась: лучший фрагмент получил оценку реранкера {top:.2f} — ниже порога "
                f"{answerer.min_retrieval_score}. Скорее всего, нужного закона нет в корпусе."
            )
    elif answer.status == STATUS_REFUSED:
        lines.append(answer.text)
        lines.append("Модель прочитала найденные фрагменты и не нашла в них ответа.")
    else:
        lines.append(answer.text)
        lines.append("> " + answer.passages[0].text.strip().replace("\n", "\n> "))
        lines.append(source(1))
        lines.append("Ответ модели не показан, проверка нашла: " + "; ".join(answer.checks["problems"]) + ".")
    return "\n".join(lines)


def interactive_cli(
    lang: str = "rus",
    model_name: Optional[str] = None,
    search_mode: str = "hybrid",
    top_k: int = 3,
    cache_dir: str = "data/adilet/cache",
    cache_parquet: str = "data/raw/adilet_legal_qa_kz.parquet",
    rerank: bool = False,
    reranker_name: Optional[str] = None,
    rerank_pool: int = ASK_RERANK_POOL,
    generate: bool = False,
    llm: str = DEFAULT_LLM,
):
    model_name = model_name or default_model(lang)
    reranker_name = reranker_name or default_reranker(lang)
    rag: Optional[RAGAnswerer] = None
    print("=" * 80)
    print("  ИНТЕРАКТИВНЫЙ ПОИСК ПО БАЗЕ ЗАКОНОДАТЕЛЬСТВА РК (Adilet Legal QA)")
    print("=" * 80)
    print(f" Язык документов: {lang.upper()}")
    print(f" Модель поиска:   {model_name}")
    print(f" Режим поиска:    {search_mode.upper()}")
    print(f" Реранкер:        {f'{reranker_name} (топ-{rerank_pool} кандидатов)' if rerank else 'выключен'}")
    print(f" Количество результатов: {top_k}")
    print(f" Ответ модели:    {f'{llm} (Ollama) по {DEFAULT_TOP_K} лучшим чанкам' if generate else 'выключен'}")
    print("-" * 80)

    # 1. Загрузка корпуса
    print("\n[1/3] Загрузка статей законодательства...")
    sub_df = load_legal_corpus(cache_parquet, lang)

    docs = []
    doc_ids = []
    for idx, row in sub_df.iterrows():
        doc_id = f"ctx_{idx:04d}"
        header = f"Закон: {row['source_title']}\n"
        docs.append(header + row["cleaned_context"])
        doc_ids.append(doc_id)

    print(f"  Загружено {len(docs)} статей нормативно-правовых актов.")

    # 2. Инициализация модели и загрузка кэша эмбеддингов
    print("\n[2/3] Инициализация поискового индекса...")
    embedder = EmbeddingModel(model_name=model_name, device="cpu")
    # Веса гибрида (tune_hybrid.py) и параметры BM25 (tune_bm25.py) подобраны для чанков 500 символов,
    # а здесь индексируются фрагменты целиком, поэтому используются параметры по умолчанию
    # (подобранные для чанков BM25-параметры на целых фрагментах хуже: val MRR@10 0.676 против 0.713)
    engine = LegalSearchEngine(
        docs=docs,
        doc_ids=doc_ids,
        embedder=embedder,
        cache_dir=cache_dir,
        cache_key=f"{model_cache_name(model_name)}_{lang}_structural",
        rerank_pool=rerank_pool,
    )
    def get_reranker() -> CrossEncoderReranker:
        if engine.reranker is None:
            print(f"  Загрузка реранкера {reranker_name}...")
            engine.reranker = CrossEncoderReranker(reranker_name)
        return engine.reranker

    def get_rag() -> RAGAnswerer:
        nonlocal rag
        if rag is None:
            print("  Подготовка генерации ответа: индекс чанков 500 символов...")
            rag = RAGAnswerer(
                lang, embedder, model_name, get_reranker(), reranker_name, cache_dir, cache_parquet, llm=llm,
            )
        return rag

    if rerank:
        get_reranker()
    if generate:
        get_rag()

    print("\n[3/3] База знаний готова к работе!")
    print("=" * 80)
    print(" Введите ваш вопрос (например: 'С какого числа вступил в силу земельный кодекс?')")
    print(" Для переключения режима введите ':mode dense', ':mode bm25' или ':mode hybrid'")
    print(" Реранкер cross-encoder: ':rerank on' / ':rerank off'")
    print(f" Ответ языковой модели ({llm}): ':answer on' / ':answer off'")
    print(" Для выхода из программы введите 'exit' или 'q'")
    print("=" * 80)

    while True:
        try:
            label = search_mode.upper() + (" + RERANK" if rerank else "") + (" + ОТВЕТ" if generate else "")
            prompt_str = f"\n[{label}] Введите вопрос > "
            query = input(prompt_str).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход из программы.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q", "выход"):
            print("Завершение сеанса поиска. До свидания!")
            break

        # Команды смены режима
        if query.startswith(":mode"):
            parts = query.split()
            if len(parts) > 1 and parts[1].lower() in SEARCH_MODES:
                search_mode = parts[1].lower()
                print(f"Режим переключен на: {search_mode.upper()}")
            else:
                print("Неизвестный режим. Доступные: dense, bm25, hybrid")
            continue

        if query.startswith(":rerank"):
            parts = query.split()
            if len(parts) > 1 and parts[1].lower() in ("on", "off"):
                rerank = parts[1].lower() == "on"
                if rerank:
                    get_reranker()
                print(f"Реранкер {'включен' if rerank else 'выключен'}")
            else:
                print("Используйте ':rerank on' или ':rerank off'")
            continue

        if query.startswith(":answer"):
            parts = query.split()
            if len(parts) > 1 and parts[1].lower() in ("on", "off"):
                generate = parts[1].lower() == "on"
                if generate:
                    get_rag()
                print(f"Ответ модели {'включен' if generate else 'выключен'}")
            else:
                print("Используйте ':answer on' или ':answer off'")
            continue

        if generate:
            print(f"\n{llm} формулирует ответ по найденным фрагментам...")
            try:
                answer = get_rag().ask(query)
                print("-" * 80)
                print(format_answer(answer, rag.answerer, lang))
                print("-" * 80)
            except OllamaError as e:
                print(f"\n⚠ {e}")

        # Выполнение поиска
        t0 = time.time()
        results = engine.search(query, mode=search_mode, top_k=top_k, rerank=rerank)
        elapsed_ms = (time.time() - t0) * 1000.0

        if not results:
            print("\nНичего не найдено. Попробуйте переформулировать вопрос или сменить режим поиска.")
            continue

        # Вывод результатов
        print(f"\n🔎 Найдено {len(results)} релевантных статей за {elapsed_ms:.1f} мс:")
        print("-" * 80)

        for rank, (doc_id, text, score) in enumerate(results, 1):
            # Первая строка документа — заголовок «Закон: ...»
            source_line, _, body_text = text.partition("\n")
            preview = format_preview(body_text, max_chars=400)

            score_str = f"{score:.2f}" if search_mode == "bm25" and not rerank else f"{score:.4f}"
            print(f" [{rank}] {source_line} | ID: {doc_id} | Релевантность: {score_str}")
            print(f"     \"{preview}\"")
            print()


def main():
    parser = argparse.ArgumentParser(description="Интерактивный консольный поиск по НПА РК.")
    parser.add_argument("--lang", choices=["rus", "kaz"], default="rus", help="Язык ('rus' или 'kaz').")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Модель эмбеддингов (по умолчанию {FINETUNED_MODEL}, если есть, иначе {DEFAULT_MODEL}).")
    parser.add_argument("--mode", choices=list(SEARCH_MODES), default="hybrid", help="Режим поиска.")
    parser.add_argument("--top-k", type=int, default=3, help="Количество возвращаемых статей.")
    parser.add_argument("--cache-dir", type=str, default="data/adilet/cache", help="Папка с кэшем векторов.")
    parser.add_argument("--rerank", action="store_true", help="Переранжировать кандидатов cross-encoder.")
    parser.add_argument("--reranker", type=str, default=None,
                        help=f"Модель реранкера (по умолчанию {FINETUNED_RERANKER}, если есть, иначе {DEFAULT_RERANKER}).")
    parser.add_argument("--rerank-pool", type=int, default=ASK_RERANK_POOL,
                        help=f"Сколько кандидатов переранжировать (по умолчанию {ASK_RERANK_POOL}).")
    parser.add_argument("--generate", action="store_true",
                        help="Выводить ответ локальной языковой модели по найденным фрагментам (нужна Ollama).")
    parser.add_argument("--llm", type=str, default=DEFAULT_LLM, help=f"Модель Ollama (по умолчанию {DEFAULT_LLM}).")
    args = parser.parse_args()

    interactive_cli(
        lang=args.lang,
        model_name=args.model,
        search_mode=args.mode,
        top_k=args.top_k,
        cache_dir=args.cache_dir,
        rerank=args.rerank,
        reranker_name=args.reranker,
        rerank_pool=args.rerank_pool,
        generate=args.generate,
        llm=args.llm,
    )


if __name__ == "__main__":
    main()
