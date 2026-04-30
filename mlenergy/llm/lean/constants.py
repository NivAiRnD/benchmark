"""Internal implementation constants for the lean benchmark.

Tuning these is the only reason to edit this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ServerConfig field defaults
_DEFAULT_READY_TIMEOUT_S: Final[float] = 1800.0
_DEFAULT_VLLM_IMAGE: Final[str] = "vllm/vllm-openai:latest"
_CONFIG_BASE_DIR: Final[Path] = Path("configs/vllm")

# HTTP
_AIOHTTP_TIMEOUT_S: Final[float] = 6 * 60 * 60

# Polling / scraping intervals
_PROMETHEUS_INTERVAL_S: Final[float] = 1.0
_GPU_INTERVAL_S: Final[float] = 0.1
_CPU_INTERVAL_S: Final[float] = 0.1
_SCRAPE_TIMEOUT_S: Final[float] = 5.0
_DEFAULT_PERCENTILES: Final[tuple[float, ...]] = (50.0, 90.0, 95.0, 99.0)

# vLLM container paths and filenames
_VLLM_CONFIG_FILENAME: Final[str] = "monolithic.config.yaml"
_VLLM_ENV_FILENAME: Final[str] = "monolithic.env.yaml"
_CONTAINER_CONFIG_PATH: Final[str] = f"/vllm_config/{_VLLM_CONFIG_FILENAME}"
_CONTAINER_VLLM_CACHE_DIR: Final[str] = "/root/.cache/vllm"
_CONTAINER_NAME_PREFIX: Final[str] = "lean-benchmark-vllm-"
_DATA_PARALLEL_ARG: Final[str] = "--data-parallel-size"
_TENSOR_PARALLEL_ARG: Final[str] = "--tensor-parallel-size"

# vLLM log detection
_STARTUP_STRING: Final[str] = "Application startup complete"
_TERMINATION_STRINGS: Final[frozenset[str]] = frozenset(
    {
        "cuda error",
        "out of memory",
        "killed",
        "traceback (most recent call last)",
        "runtimeerror",
        "valueerror",
    }
)
