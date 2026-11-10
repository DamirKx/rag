"""Генерация ответа по найденным фрагментам закона с защитой от галлюцинаций.

Языковая модель (по умолчанию Qwen2.5-3B, запускается локально через Ollama) получает вопрос
и k лучших чанков после реранкера и формулирует ответ только по ним. Галлюцинацией считается
утверждение, которого нет в найденных фрагментах, а также ответ не на заданный вопрос
(модель переписала соседний фрагмент не по теме). Защита в три слоя:

1. Порог уверенности поиска: если лучший фрагмент получил от реранкера оценку ниже порога,
   нужного закона в найденном, скорее всего, нет — система отказывает, не вызывая модель.
2. Промпт: отвечать только по фрагменту, который отвечает на вопрос, ставить ссылки [n],
   при отсутствии ответа — фиксированная фраза отказа; температура 0 (детерминированный вывод).
   Ответ без текста или со ссылкой на несуществующий фрагмент запрашивается ещё один раз.
3. Проверка готового ответа (verify_answer и judge):
   - каждое число ответа (дата, срок, сумма, номер статьи или пункта) есть во фрагментах;
   - слова каждого предложения ответа в основном встречаются во фрагментах (доля основ
     слов не ниже порога) — модель не добавила сведений «от себя»;
   - кроме ссылок [n] в ответе есть текст; ссылки на несуществующие фрагменты (модель пишет
     номер пункта статьи вместо номера фрагмента) убираются из ответа, а не отклоняют его;
   - ответ относится к вопросу: тот же реранкер оценивает пару «вопрос — ответ»;
   - ответ следует из цитируемых фрагментов (NLI, src/nli.py; включается, когда порог подобран).
   Не прошедший проверку ответ не показывается: вместо него — цитата лучшего фрагмента закона.

Пороги подбирает eval_generation.py на validation: каждая проверка отбраковывает не больше 5%
хороших ответов (нужный фрагмент найден, модель ответила).

Номера статей в списке источников берутся из метаданных чанков (src/corpus.py), а не из текста модели.
"""

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.retriever import tokenize_legal_text
from src.text_norm import get_normalizer
from src.tracing import trace_answer

DEFAULT_LLM = "qwen2.5:3b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Сколько лучших чанков после реранкера получает модель: при Hit@1 ≈ 0,69 одного мало,
# а Hit@3 заметно выше; три чанка по 500 символов — около 700 токенов промпта
DEFAULT_TOP_K = 3

# Пороги по умолчанию — если eval_generation.py ещё не подбирал их на validation.
# min_retrieval_score и min_answer_relevance — в шкале оценок реранкера, для которого подобраны;
# min_entailment — вероятность NLI; пока порог не подобран (None), модель NLI не загружается
DEFAULT_PARAMS = {"min_retrieval_score": 0.5, "min_sentence_support": 0.6, "min_answer_relevance": 0.5, "min_entailment": None}
RERANKER_PARAMS = ("min_retrieval_score", "min_answer_relevance")

STATUS_ANSWERED = "answered"      # ответ прошёл проверку
STATUS_NO_CONTEXT = "no_context"  # поиск не нашёл уверенного фрагмента, модель не вызывалась
STATUS_REFUSED = "refused"        # модель сама ответила, что во фрагментах ответа нет
STATUS_UNVERIFIED = "unverified"  # ответ не прошёл проверку и заменён цитатой закона

REFUSAL = {
    "rus": "В найденных фрагментах закона ответа нет.",
    "kaz": "Табылған заң үзінділерінде жауап жоқ.",
}
UNVERIFIED = {
    "rus": "Не удалось сформировать ответ, полностью подтверждённый текстом закона. Ниже — наиболее подходящий фрагмент.",
    "kaz": "Заң мәтінімен толық расталған жауап құрастыру мүмкін болмады. Төменде — ең сәйкес үзінді.",
}
# Ядро фразы отказа для распознавания (модель может слегка изменить формулировку)
REFUSAL_CORE = {"rus": "ответа нет", "kaz": "жауап жоқ"}

SYSTEM_PROMPT = {
    "rus": (
        "Ты помощник по законодательству Республики Казахстан. Тебе даны пронумерованные фрагменты "
        "нормативных правовых актов и вопрос.\n"
        "Правила:\n"
        "1. Отвечай только на основе фрагментов. Не используй собственные знания и ничего не добавляй от себя.\n"
        "2. Найди фрагмент, в котором есть ответ на вопрос, и отвечай по нему. Ответ может быть записан другими "
        "словами, чем вопрос. Фрагменты не по теме не используй.\n"
        "3. Сначала напиши ответ полным предложением, а в конце предложения поставь номер фрагмента, из которого "
        "оно взято, в квадратных скобках. Ответ не может состоять только из ссылки.\n"
        "4. В квадратных скобках пиши только номер фрагмента — от 1 до {n}. Номера статей, пунктов и подпунктов "
        "в квадратные скобки не заключай.\n"
        "5. Числа, даты, сроки, суммы и номера статей переписывай точно так, как они записаны во фрагменте.\n"
        "6. Только если ни в одном фрагменте нет сведений по вопросу, ответь ровно одной фразой: «{refusal}»\n"
        "7. Отвечай кратко, по существу, на русском языке."
    ),
    "kaz": (
        "Сен Қазақстан Республикасының заңнамасы бойынша көмекшісің. Саған нормативтік құқықтық актілердің "
        "нөмірленген үзінділері мен сұрақ берілген.\n"
        "Ережелер:\n"
        "1. Тек үзінділер негізінде жауап бер. Өз біліміңді қолданба және өзіңнен ештеңе қоспа.\n"
        "2. Сұрақтың жауабы бар үзіндіні тауып, соған сүйеніп жауап бер. Жауап сұрақтағыдан басқа сөздермен жазылуы "
        "мүмкін. Тақырыпқа қатысы жоқ үзінділерді қолданба.\n"
        "3. Алдымен жауапты толық сөйлеммен жаз, содан кейін сөйлемнің соңында ол алынған үзіндінің нөмірін "
        "төртбұрышты жақшада көрсет. Жауап тек сілтемеден тұрмауы керек.\n"
        "4. Төртбұрышты жақшада тек үзіндінің нөмірін жаз — 1-ден {n}-ге дейін. Бап, тармақ және тармақша нөмірлерін "
        "төртбұрышты жақшаға алма.\n"
        "5. Сандарды, күндерді, мерзімдерді, сомаларды және бап нөмірлерін үзіндідегідей дәл жаз.\n"
        "6. Бірде-бір үзіндіде сұрақ бойынша мәлімет болмаған жағдайда ғана дәл бір сөйлеммен жауап бер: «{refusal}»\n"
        "7. Қысқа, мәні бойынша, қазақ тілінде жауап бер."
    ),
}

# Повторный запрос, если ответ без текста или со ссылкой на несуществующий фрагмент
RETRY_PROMPT = {
    "rus": (
        "Ответ не принят. Нужен ответ полным предложением по тексту фрагментов, а в квадратных скобках — "
        "только номер фрагмента от 1 до {n}. Напиши ответ заново."
    ),
    "kaz": (
        "Жауап қабылданбады. Үзінділер мәтіні бойынша толық сөйлеммен жауап беру керек, ал төртбұрышты жақшада "
        "тек 1-ден {n}-ге дейінгі үзінді нөмірі болуы керек. Жауапты қайта жаз."
    ),
}


@dataclass
class Passage:
    """Фрагмент закона, переданный модели."""

    id: str
    text: str
    law: str = ""
    articles: str = ""
    score: float = 0.0
    source_id: str = ""
    chunk_ids: List[str] = field(default_factory=list)  # чанки, из которых собран фрагмент (src/corpus.expand_hits)

    @property
    def label(self) -> str:
        """Подпись фрагмента: «Земельный кодекс РК, ст. 12»."""
        return ", ".join(part for part in (self.law, self.articles) if part)

    @property
    def ids(self) -> List[str]:
        """Идентификаторы всех чанков, вошедших во фрагмент."""
        return self.chunk_ids or [self.id]

    @classmethod
    def from_hit(cls, hit: Dict[str, Any]) -> "Passage":
        """Из результата поиска (словарь чанка src/corpus.py с полем score; см. также expand_hits)."""
        return cls(
            id=hit["id"], text=hit["text"], law=hit.get("law", ""), articles=hit.get("articles", ""),
            score=float(hit.get("score", 0.0)), source_id=hit.get("source_id", ""),
            chunk_ids=list(hit.get("chunk_ids", [])),
        )


@dataclass
class Answer:
    """Результат генерации с итогами проверок."""

    status: str
    text: str                       # что показать пользователю: ответ, отказ или пояснение к цитате
    passages: List[Passage]
    raw: str = ""                   # исходный ответ модели («» — модель не вызывалась)
    cited: List[int] = field(default_factory=list)  # номера фрагментов (с 1), на которые ссылается ответ
    checks: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.status == STATUS_ANSWERED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status, "text": self.text, "raw": self.raw, "cited": self.cited,
            "passages": [p.id for p in self.passages], "checks": self.checks, "usage": self.usage,
        }


# ---------------------------------------------------------------------------------------------
# Языковая модель
# ---------------------------------------------------------------------------------------------

class OllamaError(RuntimeError):
    """Ollama не запущена или модель не скачана."""


class OllamaClient:
    """Минимальный клиент HTTP API Ollama (/api/chat) на стандартной библиотеке.

    Ответы можно кэшировать в файл JSONL (cache_path): ключ — хеш модели, параметров и сообщений,
    поэтому повторный прогон оценки не вызывает модель заново.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM,
        url: str = OLLAMA_URL,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        num_predict: int = 400,
        seed: int = 0,
        timeout: float = 300.0,
        cache_path: Optional[str] = None,
        retries: int = 0,
    ):
        """retries — сколько раз повторить запрос, если Ollama не ответила за timeout секунд
        (бывает, когда видеопамять занята другой моделью); на прочие ошибки повторов нет."""
        self.model = model
        self.url = url.rstrip("/")
        self.options = {"temperature": temperature, "num_ctx": num_ctx, "num_predict": num_predict, "seed": seed}
        self.timeout = timeout
        self.retries = retries
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self._cache[record["key"]] = record["response"]

    def _key(self, messages: Sequence[Dict[str, str]]) -> str:
        payload = json.dumps([self.model, self.options, list(messages)], ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def chat(self, messages: Sequence[Dict[str, str]]) -> Dict[str, Any]:
        """Возвращает {"text", "prompt_tokens", "completion_tokens", "latency_s", "cached"}."""
        key = self._key(messages)
        if key in self._cache:
            return {**self._cache[key], "cached": True}

        body = {"model": self.model, "messages": list(messages), "stream": False, "options": self.options}
        request = urllib.request.Request(
            f"{self.url}/api/chat", data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        try:
            for attempt in range(self.retries + 1):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        data = json.loads(response.read().decode("utf-8"))
                    break
                except TimeoutError:
                    if attempt == self.retries:
                        raise
                    print(f"  Ollama не ответила за {self.timeout:.0f} с, повтор {attempt + 1}/{self.retries}...")
                    t0 = time.time()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 404:
                raise OllamaError(f"Модель {self.model} не найдена в Ollama: выполните `ollama pull {self.model}`") from e
            raise OllamaError(f"Ошибка Ollama {e.code}: {detail}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise OllamaError(
                f"Ollama недоступна по адресу {self.url}: запустите её (`ollama serve` или приложение Ollama) "
                f"и скачайте модель `ollama pull {self.model}`"
            ) from e

        result = {
            "text": data["message"]["content"].strip(),
            "prompt_tokens": data.get("prompt_eval_count"),
            "completion_tokens": data.get("eval_count"),
            "latency_s": round(time.time() - t0, 3),
        }
        self._cache[key] = result
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": result}, ensure_ascii=False) + "\n")
        return {**result, "cached": False}


def build_messages(question: str, passages: Sequence[Passage], lang: str = "rus") -> List[Dict[str, str]]:
    """Сообщения чата: системные правила и пользовательское сообщение с фрагментами и вопросом."""
    blocks = [f"[{i}] {p.label}\n{p.text.strip()}" for i, p in enumerate(passages, 1)]
    head, q = ("Фрагменты:", "Вопрос:") if lang == "rus" else ("Үзінділер:", "Сұрақ:")
    return [
        {"role": "system", "content": SYSTEM_PROMPT[lang].format(refusal=REFUSAL[lang], n=len(passages))},
        {"role": "user", "content": f"{head}\n\n" + "\n\n".join(blocks) + f"\n\n{q} {question.strip()}"},
    ]


# ---------------------------------------------------------------------------------------------
# Проверка ответа
# ---------------------------------------------------------------------------------------------

CITATION = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")
NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")
# Неразрывные пробелы и дефисы датасета («№ 221‑VII от 05.04.2023») приводятся к обычным
_SPACES = re.compile(r"[    ]")
_DASHES = re.compile(r"[‐‑‒–—−]")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*•][ \t]*)?\d{1,2}[.)][ \t]+")

MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
DATE_IN_WORDS = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})")

# Служебные и связующие слова, которые модель добавляет при пересказе — не считаются «новыми сведениями»
SUPPORT_STOPWORDS = {
    # вопросительные и местоимения
    "какой", "какая", "какие", "каких", "каком", "какого", "каким", "который", "которая", "которые", "которых",
    "этот", "этого", "этом", "эта", "эти", "этих", "того", "тому", "свой", "своих", "своего", "себя", "данный",
    "данного", "данном", "данной", "данные", "данных", "данным", "такой", "такие", "таких", "также", "кроме",
    # связки пересказа
    "согласно", "соответствии", "является", "являются", "означает", "относится", "относятся", "включает",
    "составляет", "составляют", "представляет", "представляют", "называется",
    "включают", "входят", "может", "могут", "должен", "должны", "должна", "указано", "указанных", "указанные",
    "указанным", "предусмотрено", "определяется", "определяются", "осуществляется", "осуществляются",
    "следующие", "следующих", "случае", "случаях", "числе", "этого", "этой", "если", "либо", "или", "когда",
    "фрагмент", "фрагменте", "фрагментах", "фрагмента", "контексте", "вопрос", "ответ",
    # казахские служебные
    "және", "немесе", "үшін", "бойынша", "сәйкес", "болып", "табылады", "мұнда", "осы", "сондай",
}


def normalize_for_check(text: str) -> str:
    """Нижний регистр, ё→е, обычные пробелы и дефисы."""
    text = _DASHES.sub("-", _SPACES.sub(" ", text)).lower().replace("ё", "е")
    return re.sub(r"[ \t]+", " ", text)


def extract_citations(answer: str) -> List[int]:
    """Номера фрагментов из ссылок вида [1], [1, 3] в порядке появления, без повторов."""
    seen: List[int] = []
    for group in CITATION.findall(answer):
        for n in re.split(r"\s*[,;]\s*", group):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


def strip_citations(answer: str) -> str:
    """Текст ответа без ссылок [n]."""
    return re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]*" + CITATION.pattern, "", answer)).strip()


def drop_invalid_citations(answer: str, n_passages: int) -> str:
    """Убирает из ответа ссылки на несуществующие фрагменты («[6]» при трёх фрагментах).

    Модель иногда пишет в скобках номер пункта статьи вместо номера фрагмента. Такая ссылка
    ничего не утверждает: проверки опоры и чисел сверяют ответ со всеми фрагментами.
    """
    def keep_valid(match: "re.Match") -> str:
        valid = [n for n in re.split(r"\s*[,;]\s*", match.group(1)) if 1 <= int(n) <= n_passages]
        return f"[{', '.join(valid)}]" if valid else ""

    cleaned = re.sub(r"[ \t]*" + CITATION.pattern, lambda m: (" " if keep_valid(m) else "") + keep_valid(m), answer)
    return re.sub(r"\s+([.,;:])", r"\1", cleaned).strip()


def _context_numbers(context: str) -> set:
    """Числа контекста и даты, записанные словами («1 июля 2023» → «01.07.2023»)."""
    norm = normalize_for_check(context)
    numbers = set(NUMBER.findall(norm))
    for day, month, year in DATE_IN_WORDS.findall(norm):
        numbers.add(f"{int(day):02d}.{MONTHS[month]:02d}.{year}")
    return numbers


def unsupported_numbers(answer: str, context: str) -> List[str]:
    """Числа ответа (кроме ссылок [n]), которых нет в контексте.

    Даты сравниваются без ведущих нулей («1.07.2023» = «01.07.2023»); дата во фрагменте
    может быть записана словами.
    """
    def canon(number: str) -> str:
        return re.sub(r"\b0+(\d)", r"\1", number)

    known = {canon(n) for n in _context_numbers(context)}
    # Нумерация списка в начале строки («1) ...», «2. ...») — оформление ответа, а не сведения из закона
    text = LIST_MARKER.sub("", normalize_for_check(strip_citations(answer)))
    missing = []
    for number in NUMBER.findall(text):
        if canon(number) not in known and number not in missing:
            missing.append(number)
    return missing


def _content_stems(text: str, normalize) -> List[str]:
    tokens = [
        t for t in tokenize_legal_text(normalize_for_check(text))
        if len(t) >= 4 and not any(ch.isdigit() for ch in t) and t not in SUPPORT_STOPWORDS
    ]
    return normalize(tokens)


def sentence_support(answer: str, context: str, lang: str = "rus") -> List[Tuple[str, float]]:
    """Доля основ значимых слов каждого предложения ответа, встречающихся в контексте.

    Основы: стеммер Snowball для русского, первые 5 букв для казахского. Предложения без
    значимых слов (например, «1)») пропускаются.
    """
    normalize = get_normalizer("snowball" if lang == "rus" else "prefix5")
    context_stems = set(_content_stems(context, normalize))
    result = []
    for sentence in SENTENCE_SPLIT.split(strip_citations(answer)):
        stems = _content_stems(sentence, normalize)
        if stems:
            result.append((sentence.strip(), sum(s in context_stems for s in stems) / len(stems)))
    return result


def answer_sentences(answer: str) -> List[str]:
    """Предложения ответа без ссылок [n]; строки без букв (нумерация «1)») пропускаются."""
    parts = SENTENCE_SPLIT.split(strip_citations(answer))
    return [part.strip() for part in parts if re.search(r"[^\W\d_]", part)]


def is_malformed(answer: str, n_passages: int, lang: str = "rus") -> bool:
    """Ответ, который стоит запросить заново: без текста или со ссылкой на несуществующий фрагмент."""
    if is_refusal(answer, lang):
        return False
    return not re.search(r"\w", strip_citations(answer)) or any(not 1 <= n <= n_passages for n in extract_citations(answer))


def cited_passages(answer: str, passages: Sequence[Passage]) -> List[Passage]:
    """Фрагменты, на которые ссылается ответ; без действительных ссылок — все."""
    cited = [passages[n - 1] for n in extract_citations(answer) if 1 <= n <= len(passages)]
    return cited or list(passages)


def is_refusal(answer: str, lang: str = "rus") -> bool:
    """Ответ модели — фраза отказа (а не содержательный ответ, где она случайно упомянута)."""
    norm = normalize_for_check(strip_citations(answer))
    core = REFUSAL_CORE[lang]
    return core in norm and len(norm) <= len(REFUSAL[lang]) + 40


def verify_answer(answer: str, passages: Sequence[Passage], lang: str = "rus") -> Dict[str, Any]:
    """Проверки ответа по контексту — всех переданных модели фрагментов вместе с подписями."""
    context = "\n".join(f"{p.label}\n{p.text}" for p in passages)
    cited = extract_citations(answer)
    support = sentence_support(answer, context, lang)
    return {
        "unsupported_numbers": unsupported_numbers(answer, context),
        "invalid_citations": [n for n in cited if not 1 <= n <= len(passages)],
        "min_sentence_support": round(min((s for _, s in support), default=1.0), 4),
        "weakest_sentence": min(support, key=lambda x: x[1])[0] if support else "",
        "n_sentences": len(support),
    }


# ---------------------------------------------------------------------------------------------
# Ответ с защитой от галлюцинаций
# ---------------------------------------------------------------------------------------------

def load_generation_params(lang: str, reranker: Optional[str] = None, path: Optional[str] = None) -> Dict[str, Any]:
    """Пороги, подобранные eval_generation.py на validation; иначе DEFAULT_PARAMS.

    Пороги в шкале реранкера (RERANKER_PARAMS) зависят от модели реранкера: если они
    подобраны для другой модели, эти проверки отключаются (None).
    """
    params = dict(DEFAULT_PARAMS)
    file = Path(path or f"data/adilet/generation_eval_{lang}.json")
    if file.exists():
        tuned = json.loads(file.read_text(encoding="utf-8")).get("params", {})
        params.update({k: tuned[k] for k in DEFAULT_PARAMS if k in tuned})
        if reranker is not None and tuned.get("reranker") not in (None, Path(reranker).as_posix()):
            params.update({k: None for k in RERANKER_PARAMS})
    return params


# Оценка соответствия ответа вопросу: (вопрос, ответ) -> число в шкале реранкера
RelevanceScorer = Callable[[str, str], float]
# Следование ответа из фрагментов: (вопрос, ответ, цитируемые фрагменты) -> вероятность (src/nli.py)
EntailmentScorer = Callable[[str, str, Sequence[Passage]], float]


def reranker_relevance(reranker) -> RelevanceScorer:
    """Соответствие ответа вопросу по оценке реранкера cross-encoder (src/reranker.py)."""
    return lambda question, answer: float(reranker.score(question, [strip_citations(answer)])[0])


class GroundedAnswerer:
    """Отвечает на вопрос по найденным фрагментам; не подтверждённый текстом закона ответ не выдаёт."""

    def __init__(
        self,
        client: Optional[OllamaClient],
        lang: str = "rus",
        min_retrieval_score: Optional[float] = DEFAULT_PARAMS["min_retrieval_score"],
        min_sentence_support: float = DEFAULT_PARAMS["min_sentence_support"],
        min_answer_relevance: Optional[float] = DEFAULT_PARAMS["min_answer_relevance"],
        relevance_scorer: Optional[RelevanceScorer] = None,
        verify: bool = True,
        min_entailment: Optional[float] = None,
        entailment_scorer: Optional[EntailmentScorer] = None,
    ):
        """
        Args:
            client: клиент языковой модели (None — только проверка готовых ответов, judge).
            lang: язык вопросов и корпуса ('rus' или 'kaz').
            min_retrieval_score: порог оценки реранкера лучшего фрагмента; None — без порога.
            min_sentence_support: минимальная доля основ слов предложения ответа, найденных во фрагментах.
            min_answer_relevance: порог соответствия ответа вопросу; None — без этой проверки.
            relevance_scorer: оценка соответствия ответа вопросу (reranker_relevance); без неё
                              проверка выполняется, только если оценка передана в judge.
            verify: проверять ответ (False — выдавать ответ модели как есть, для сравнения в оценке).
            min_entailment: порог вероятности того, что ответ следует из цитируемых фрагментов; None — без проверки.
            entailment_scorer: оценка следования (src/nli.py); нужна только вместе с min_entailment.
        """
        self.client = client
        self.lang = lang
        self.min_retrieval_score = min_retrieval_score
        self.min_sentence_support = min_sentence_support
        self.min_answer_relevance = min_answer_relevance
        self.relevance_scorer = relevance_scorer
        self.verify = verify
        self.min_entailment = min_entailment
        self.entailment_scorer = entailment_scorer

    def retrieval_confident(self, passages: Sequence[Passage]) -> bool:
        if not passages:
            return False
        if self.min_retrieval_score is None:
            return True
        return max(p.score for p in passages) >= self.min_retrieval_score

    def ask_model(self, question: str, passages: Sequence[Passage]) -> Dict[str, Any]:
        """Ответ модели; ответ без текста или с несуществующей ссылкой запрашивается ещё раз.

        Возвращает словарь OllamaClient.chat с полем retried; токены и время суммируются по обоим запросам.
        """
        messages = build_messages(question, passages, self.lang)
        first = self.client.chat(messages)
        if not is_malformed(first["text"], len(passages), self.lang):
            return {**first, "retried": False}
        retry = RETRY_PROMPT[self.lang].format(n=len(passages))
        second = self.client.chat(messages + [{"role": "assistant", "content": first["text"]}, {"role": "user", "content": retry}])
        merged = {key: (first.get(key) or 0) + (second.get(key) or 0) for key in ("prompt_tokens", "completion_tokens", "latency_s")}
        return {**second, **merged, "cached": first.get("cached") and second.get("cached"), "retried": True}

    def answer(self, question: str, passages: Sequence[Passage]) -> Answer:
        """Полный цикл: порог поиска → модель → проверка ответа."""
        passages = list(passages)
        top_score = max((p.score for p in passages), default=None)
        if not self.retrieval_confident(passages):
            result = Answer(
                status=STATUS_NO_CONTEXT, text=REFUSAL[self.lang], passages=passages,
                checks={"top_score": top_score},
            )
        else:
            response = self.ask_model(question, passages)
            usage = {k: response.get(k) for k in ("prompt_tokens", "completion_tokens", "latency_s", "cached", "retried")}
            result = self.judge(question, response["text"], passages, usage)
        trace_answer(question, result, build_messages(question, passages, self.lang), getattr(self.client, "model", None), self.lang)
        return result

    def judge(
        self,
        question: str,
        raw: str,
        passages: Sequence[Passage],
        usage: Optional[Dict[str, Any]] = None,
        relevance: Optional[float] = None,
        entailment: Optional[float] = None,
    ) -> Answer:
        """Проверяет готовый ответ модели и выбирает, что показать пользователю.

        relevance и entailment — готовые оценки (иначе считаются relevance_scorer и entailment_scorer).
        """
        passages = list(passages)
        usage = usage or {}
        checks: Dict[str, Any] = {"top_score": max((p.score for p in passages), default=None)}
        if is_refusal(raw, self.lang):
            return Answer(STATUS_REFUSED, REFUSAL[self.lang], passages, raw=raw, checks=checks, usage=usage)

        # Несуществующие ссылки не отклоняют ответ, а убираются из него (номера остаются в checks)
        text = drop_invalid_citations(raw, len(passages))
        checks.update(verify_answer(text, passages, self.lang))
        checks["invalid_citations"] = verify_answer(raw, passages, self.lang)["invalid_citations"]
        if relevance is None and self.relevance_scorer is not None:
            relevance = self.relevance_scorer(question, text)
        checks["answer_relevance"] = None if relevance is None else round(float(relevance), 4)
        cited = [n for n in extract_citations(text) if 1 <= n <= len(passages)]
        problems = []
        # Модель иногда отвечает одной ссылкой «[1]» без текста: проверять в ней нечего, но и ответа нет
        if not re.search(r"\w", strip_citations(text)):
            problems.append("ответ без текста — только ссылка на фрагмент")
        if checks["unsupported_numbers"]:
            problems.append("числа, которых нет в тексте закона: " + ", ".join(checks["unsupported_numbers"]))
        if checks["min_sentence_support"] < self.min_sentence_support:
            problems.append("утверждения, не подтверждённые текстом закона")
        if self.min_answer_relevance is not None and relevance is not None and relevance < self.min_answer_relevance:
            problems.append("ответ не относится к заданному вопросу")
        # NLI медленнее остальных проверок: не считается, если ответ уже отклонён
        if entailment is None and self.entailment_scorer is not None and self.min_entailment is not None and not problems:
            entailment = self.entailment_scorer(question, text, cited_passages(text, passages))
        checks["entailment"] = None if entailment is None else round(float(entailment), 4)
        if self.min_entailment is not None and entailment is not None and entailment < self.min_entailment:
            problems.append("ответ не следует из цитируемого фрагмента")
        checks["problems"] = problems

        if self.verify and problems:
            # Сам ответ (и выдуманные в нём числа) не показывается; причины — в checks["problems"]
            return Answer(STATUS_UNVERIFIED, UNVERIFIED[self.lang], passages, raw=raw, cited=[1], checks=checks, usage=usage)
        return Answer(STATUS_ANSWERED, text, passages, raw=raw, cited=cited or [1], checks=checks, usage=usage)
