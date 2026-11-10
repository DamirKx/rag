"""Проверка, что ответ следует из цитируемых фрагментов закона (Natural Language Inference).

Проверка по словам (src/generation.py) пропускает ответ, собранный из слов закона, но с перевёрнутым смыслом.
Готовая многоязычная модель NLI оценивает пару «фрагмент — утверждение» целиком. Это классификатор, а не
языковая модель: текста он не порождает. Ответ из соседнего пункта того же фрагмента NLI может не заметить:
такое утверждение верно по тексту, а связь с вопросом модель учитывает слабо.

Гипотеза — вопрос и предложение ответа вместе: у короткого ответа («в течение трёх дней») без вопроса
неизвестно, о чём он. Оценка ответа — минимум по предложениям, оценка предложения — максимум по цитируемым
фрагментам: каждое предложение должно следовать хотя бы из одного из них.

Модель работает на CPU: 4 ГБ видеопамяти заняты реранкером и языковой моделью. Казахского языка среди
обучающих языков модели нет, поэтому проверка применяется только к русскому корпусу (NLI_LANGS).
"""

from typing import List, Optional, Sequence, Tuple

from src.generation import EntailmentScorer, Passage, answer_sentences

DEFAULT_NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_LANGS = ("rus",)


class NLIChecker:
    """Вероятность следования гипотезы из посылки."""

    def __init__(self, model_name: str = DEFAULT_NLI, device: str = "cpu", batch_size: int = 8, max_length: int = 512):
        # Импорт внутри: модуль нужен только при подобранном пороге, а загрузка transformers небыстрая
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        labels = {name.lower(): idx for name, idx in self.model.config.label2id.items()}
        self.entailment_id = labels["entailment"]

    def entailment(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Вероятности следования для пар (посылка, гипотеза); обрезается только посылка."""
        import torch

        scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            premises, hypotheses = zip(*pairs[i:i + self.batch_size])
            batch = self.tokenizer(
                list(premises), list(hypotheses), truncation="only_first", max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probabilities = self.model(**batch).logits.softmax(dim=-1)
            scores.extend(probabilities[:, self.entailment_id].tolist())
        return scores


def entailment_scorer(checker) -> EntailmentScorer:
    """Оценка ответа для GroundedAnswerer: минимум по предложениям, максимум по фрагментам."""

    def score(question: str, answer: str, passages: Sequence[Passage]) -> float:
        sentences = answer_sentences(answer)
        if not sentences or not passages:
            return 1.0  # проверять нечего: ответ без текста отбраковывается отдельной проверкой
        pairs = [(p.text, f"{question.strip()} {sentence}") for sentence in sentences for p in passages]
        values = checker.entailment(pairs)
        n = len(passages)
        return min(max(values[i:i + n]) for i in range(0, len(values), n))

    return score


def load_entailment_scorer(min_entailment: Optional[float], device: str = "cpu") -> Optional[EntailmentScorer]:
    """Оценка следования для интерфейсов; пока порог не подобран, модель не загружается."""
    return None if min_entailment is None else entailment_scorer(NLIChecker(device=device))
