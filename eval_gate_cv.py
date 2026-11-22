"""Порог уверенности поиска для корпуса законов по ответам без порога (eval_laws_colloquial.py --answers --no-gate).

Прогон без порога сохраняет для каждого вопроса оценку реранкера лучшего фрагмента (top_score) и итог ответа,
поэтому любой порог считается без новых запросов к модели:
- таблица «порог → сколько ответов показано, сколько по эталонной статье, сколько на вопросы вне корпуса»;
- подбор на одной половине набора и проверка на другой (чётные и нечётные вопросы) — порог без подгонки
  под проверочные вопросы;
- порог по всему набору.

Правило подбора — как в 7.15: отбраковывается не больше 5% хороших ответов (показанных по эталонной статье),
но порог выше оценки любого вопроса вне корпуса, на который был бы показан ответ (отказ или отклонённый ответ
вне корпуса порога не требуют).

Запуск (видеокарта не нужна):
    python eval_gate_cv.py                                        # data/laws/colloquial_eval_100.json
    python eval_gate_cv.py --result data/laws/colloquial_eval.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval_generation import threshold_for

VARIANT = "+ пункты статей после отказа"  # CHOSEN в eval_laws_colloquial.py
THRESHOLDS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.676)


def shown_at(threshold: float, records: Sequence[Dict[str, Any]]) -> Tuple[int, int]:
    """Сколько ответов показано при пороге и сколько из них по эталонной статье."""
    shown = [r for r in records if r["top_score"] >= threshold and r["status"] == "answered"]
    return len(shown), sum(r["from_gold"] for r in shown)


def fit_threshold(records: Sequence[Dict[str, Any]]) -> float:
    """Порог по правилу 7.15 с условием «ни одного показанного ответа вне корпуса»."""
    good = [r["top_score"] for r in records if r["has_gold"] and r["status"] == "answered" and r["from_gold"]]
    bad = [r["top_score"] for r in records if not r["has_gold"] and r["status"] == "answered"]
    return round(max([threshold_for(good)] + [s + 1e-4 for s in bad]), 4)


def main() -> None:
    path = Path(sys.argv[sys.argv.index("--result") + 1] if "--result" in sys.argv else "data/laws/colloquial_eval_100.json")
    records: List[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))["answers_no_gate"][VARIANT]["records"]
    inside = [r for r in records if r["has_gold"]]
    outside = [r for r in records if not r["has_gold"]]

    print(f"{path}: {len(inside)} вопросов с ответом в корпусе, {len(outside)} вне корпуса\n")
    print(f"{'порог':>6s} {'показано':>9s} {'по эталону':>11s} {'не по эталону':>14s} {'вне корпуса':>12s}")
    for t in THRESHOLDS:
        s, g = shown_at(t, inside)
        print(f"{t:6.3f} {s:9d} {g:11d} {s - g:14d} {shown_at(t, outside)[0]:12d}")
    print("\nПоказанные ответы не по эталонной статье (проверить вручную — эталон мог не учесть другую статью):")
    for r in inside:
        if r["status"] == "answered" and not r["from_gold"]:
            print(f"  {r['id']} оценка {r['top_score']:.3f}: {r['answer'][:100]}")
    for r in outside:
        if r["status"] == "answered":
            print(f"  вне корпуса {r['id']} оценка {r['top_score']:.3f}: {r['answer'][:100]}")

    print("\nПодбор на одной половине, проверка на другой (чётные / нечётные вопросы):")
    halves = [[r for i, r in enumerate(records) if i % 2 == k] for k in (0, 1)]
    for k in (0, 1):
        t = fit_threshold(halves[k])
        test_in = [r for r in halves[1 - k] if r["has_gold"]]
        test_out = [r for r in halves[1 - k] if not r["has_gold"]]
        s, g = shown_at(t, test_in)
        print(f"  подбор на половине {k}: порог {t:.3f} → на другой показано {s} из {len(test_in)}, "
              f"по эталону {g}, вне корпуса {shown_at(t, test_out)[0]} из {len(test_out)}")
    print(f"\nПорог по всему набору: {fit_threshold(records):.3f}")


if __name__ == "__main__":
    main()
