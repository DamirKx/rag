"""Замер ресурсов для трассировки: видеопамять, Qwen в Ollama, оперативная память и процессор.

Всё здесь дешёвое (единицы миллисекунд) и ничего не загружает: если процесс ещё не работал с видеокартой,
память карты не опрашивается — иначе torch создал бы контекст CUDA и занял ~300 МБ видеопамяти,
которых и так не хватает Qwen (видеокарта 4 ГБ).
"""

import json
import os
import sys
import time
import urllib.request
from typing import Any, Dict, Optional

MB = 2 ** 20


def gpu_memory() -> Dict[str, Any]:
    """Видеопамять: занято на всей карте и сколько держит PyTorch этого процесса (выделено и про запас)."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return {"gpu_used_by_process": False}
    free, total = torch.cuda.mem_get_info()
    return {
        "gpu_used_mb": round((total - free) / MB),
        "gpu_total_mb": round(total / MB),
        "torch_allocated_mb": round(torch.cuda.memory_allocated() / MB),
        "torch_reserved_mb": round(torch.cuda.memory_reserved() / MB),
    }


def ollama_placement(url: str, model: str, timeout: float = 2.0) -> Dict[str, Any]:
    """Где лежит языковая модель (/api/ps): доля на видеокарте. Меньше 100% — часть слоёв считает процессор."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/ps", timeout=timeout) as response:
            models = json.loads(response.read())["models"]
    except Exception:
        return {}
    entry = next((m for m in models if m.get("name") == model or m.get("model") == model), None)
    if entry is None or not entry.get("size"):
        return {"ollama_loaded": False}
    return {
        "ollama_model_mb": round(entry["size"] / MB),
        "ollama_vram_mb": round(entry.get("size_vram", 0) / MB),
        "ollama_on_gpu": round(entry.get("size_vram", 0) / entry["size"], 3),
        "ollama_context": entry.get("context_length"),
    }


class ResourceMeter:
    """Процессор и оперативная память за время запроса: with ResourceMeter() as meter: ...; meter.result."""

    def __init__(self):
        self.result: Dict[str, Any] = {}
        self._process = None

    def __enter__(self) -> "ResourceMeter":
        try:
            import psutil

            self._process = psutil.Process(os.getpid())
            self._cpu = self._process.cpu_times()
        except Exception:
            self._process = None
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        wall = time.perf_counter() - self._start
        self.result = {"wall_s": round(wall, 3)}
        if self._process is None:
            return
        try:
            import psutil

            cpu = self._process.cpu_times()
            cpu_s = (cpu.user - self._cpu.user) + (cpu.system - self._cpu.system)
            memory = psutil.virtual_memory()
            self.result.update({
                "cpu_s": round(cpu_s, 3),
                # Процессорное время всего процесса, включая фоновые потоки. 100% — одно ядро загружено всё время
                # запроса; больше 100% — несколько ядер (e5 на CPU и BM25 используют их одновременно)
                "cpu_percent_of_core": round(100 * cpu_s / wall) if wall > 0 else None,
                "cpu_cores": psutil.cpu_count(),
                "ram_process_mb": round(self._process.memory_info().rss / MB),
                "ram_used_percent": memory.percent,
                "ram_total_mb": round(memory.total / MB),
            })
        except Exception:
            pass


def snapshot(ollama_url: Optional[str] = None, ollama_model: Optional[str] = None) -> Dict[str, Any]:
    """Видеопамять и (если указана модель) размещение Qwen — для metadata трассы."""
    result = gpu_memory()
    if ollama_url and ollama_model:
        result.update(ollama_placement(ollama_url, ollama_model))
    return result
