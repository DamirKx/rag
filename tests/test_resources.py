"""Замер ресурсов для трассы: процессор и память запроса, размещение Qwen в Ollama, видеопамять."""

import io
import json
import sys
import time
import types

import src.resources as resources
from src.resources import ResourceMeter, gpu_memory, ollama_placement


def test_resource_meter_measures_cpu_and_ram():
    with ResourceMeter() as meter:
        sum(i * i for i in range(300_000))
        time.sleep(0.01)
    result = meter.result
    assert result["wall_s"] > 0
    assert result["cpu_s"] >= 0 and result["ram_process_mb"] > 0
    assert 0 < result["ram_used_percent"] <= 100


def test_gpu_memory_does_not_touch_unused_cuda(monkeypatch):
    """Если процесс не работал с видеокартой, контекст CUDA не создаётся (он занял бы ~300 МБ)."""
    fake_cuda = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False,
                                      mem_get_info=lambda: (_ for _ in ()).throw(AssertionError("CUDA тронута")))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    assert gpu_memory() == {"gpu_used_by_process": False}


def test_gpu_memory_reports_card_and_torch(monkeypatch):
    mb = 2 ** 20
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: True, mem_get_info=lambda: (1000 * mb, 4000 * mb),
        memory_allocated=lambda: 1100 * mb, memory_reserved=lambda: 1200 * mb,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    assert gpu_memory() == {"gpu_used_mb": 3000, "gpu_total_mb": 4000, "torch_allocated_mb": 1100, "torch_reserved_mb": 1200}


def test_ollama_placement(monkeypatch):
    mb = 2 ** 20
    payload = {"models": [{"name": "qwen2.5:3b", "size": 2000 * mb, "size_vram": 500 * mb, "context_length": 4096}]}
    monkeypatch.setattr(resources.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    assert ollama_placement("http://ollama", "qwen2.5:3b") == {
        "ollama_model_mb": 2000, "ollama_vram_mb": 500, "ollama_on_gpu": 0.25, "ollama_context": 4096,
    }
    assert ollama_placement("http://ollama", "другая-модель") == {"ollama_loaded": False}


def test_ollama_placement_unavailable(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Ollama не запущена")

    monkeypatch.setattr(resources.urllib.request, "urlopen", fail)
    assert ollama_placement("http://ollama", "qwen2.5:3b") == {}
