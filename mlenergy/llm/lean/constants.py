"""Internal implementation constants for the lean benchmark.

Tuning these is the only reason to edit this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final


class VLLMDefaults:
    READY_TIMEOUT_S: Final[float] = 1800.0
    IMAGE: Final[str] = "vllm/vllm-openai:latest"
    CONFIG_BASE_DIR: Final[Path] = Path("configs/vllm")


class VLLMContainerDefaults:
    CONFIG_FILENAME: Final[str] = "monolithic.config.yaml"
    ENV_FILENAME: Final[str] = "monolithic.env.yaml"
    CONFIG_PATH: Final[str] = f"/vllm_config/{CONFIG_FILENAME}"
    CACHE_DIR: Final[str] = "/root/.cache/vllm"
    NAME_PREFIX: Final[str] = "lean-benchmark-vllm-"
    DATA_PARALLEL_ARG: Final[str] = "--data-parallel-size"
    TENSOR_PARALLEL_ARG: Final[str] = "--tensor-parallel-size"
    STARTUP_STRING: Final[str] = "Application startup complete"
    TERMINATION_STRINGS: Final[frozenset[str]] = frozenset(
        {
            "cuda error",
            "out of memory",
            "killed",
            "traceback (most recent call last)",
            "runtimeerror",
            "valueerror",
        }
    )


class PrometheusDefaults:
    INTERVAL_S: Final[float] = 1.0
    SCRAPE_TIMEOUT_S: Final[float] = 5.0
    DEFAULT_PERCENTILES: Final[tuple[float, ...]] = (50.0, 90.0, 95.0, 99.0)


class RequestClientDefaults:
    TIMEOUT_S: Final[float] = 6 * 60 * 60


class HardwareMonitorDefaults:
    GPU_INTERVAL_S: Final[float] = 0.1
    CPU_INTERVAL_S: Final[float] = 0.1
