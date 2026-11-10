"""Готовит корпус актуальных законов для веб-демо из PDF в папке data/laws/ (src/laws.py).

Запуск:
    python prepare_laws.py

Результат: data/laws/laws_rus.parquet. После добавления или замены PDF скрипт нужно запустить заново;
эмбеддинги пересчитаются при следующем поиске в веб-демо.
"""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.laws import LAWS_DIR, LAWS_PARQUET, read_laws


def main() -> None:
    df = read_laws(LAWS_DIR)
    df.to_parquet(LAWS_PARQUET, index=False)
    print(f"{'Акт':52s} {'статей':>7s} {'символов':>10s}  ссылка  редакция")
    for title, group in df.groupby("source_title", sort=False):
        print(f"{title[:52]:52s} {len(group):7d} {group['context'].str.len().sum():10,d}  "
              f"{'есть' if group['source_id'].iloc[0] else 'нет':6s}  {group['as_of'].iloc[0]}")
    print(f"Всего: {len(df)} фрагментов. Сохранено: {LAWS_PARQUET}")


if __name__ == "__main__":
    main()
