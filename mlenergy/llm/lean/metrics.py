"""Benchmark result types and metrics calculation.

Defines the data model for benchmark outputs and the function that reduces
raw request outputs into aggregate statistics.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

from mlenergy.llm.datasets import SampleRequest

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger("mlenergy.llm.lean")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricEntry:
    """A metric value with its display format. The label is its key in BenchmarkMetrics.entries."""

    value: float | int
    fmt: str = "%.2f"


@dataclass
class BenchmarkMetrics:
    """Named metric entries keyed by display label.

    Access by name:  metrics.entries["Completed requests"].value
    Iterate for log: for label, entry in metrics.entries.items(): ...

    Add new metrics in calculate_metrics — logging and serialization
    traverse entries automatically.
    """

    entries: dict[str, MetricEntry]


@dataclass
class RequestOutput:
    """Per-request outcome including latency and token metrics."""

    prompt: str | list[str] = ""
    output_text: str = ""
    reasoning_output_text: str = ""
    prompt_len: int = 0
    output_tokens: int = 0
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    error: str = ""


@dataclass
class BenchmarkResult:
    """Full benchmark result including metrics, energy, and per-request details."""

    metrics: BenchmarkMetrics
    energy_j: float
    energy_per_token_j: float
    benchmark_start_time: float
    benchmark_end_time: float
    prometheus_stats: dict[str, float]
    per_request: list[RequestOutput]
    power_trace: list[tuple[float, float]] = field(default_factory=list)

    def log(self) -> None:
        """Log all metrics and energy to the benchmark logger."""
        all_entries: dict[str, MetricEntry] = {
            **self.metrics.entries,
            "Energy (J)": MetricEntry(self.energy_j),
            "Energy per token (J)": MetricEntry(self.energy_per_token_j, "%.4f"),
        }
        sep = "=" * 51
        logger.info(sep)
        for label, entry in all_entries.items():
            logger.info("%-40s: " + entry.fmt, label, entry.value)
        logger.info(sep)


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------


def calculate_metrics(
    requests: list[SampleRequest],
    outputs: list[RequestOutput],
    tokenizer: PreTrainedTokenizerBase,
    duration_s: float,
) -> tuple[BenchmarkMetrics, int, list[int]]:
    """Reduce raw request outputs into aggregate benchmark metrics.

    Returns:
        metrics: Named entries for display and serialization.
        completed: Count of successful requests, for validation in the caller.
        actual_output_lens: Per-request output token counts (0 for failures).
    """
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    ttfts: list[float] = []
    tpots: list[float] = []
    e2els: list[float] = []

    for req, out in zip(requests, outputs):
        if not out.success:
            actual_output_lens.append(0)
            continue

        output_len = out.output_tokens or len(
            tokenizer(
                out.output_text + out.reasoning_output_text,
                add_special_tokens=False,
            ).input_ids
        )
        actual_output_lens.append(output_len)
        total_input += req.prompt_len
        ttfts.append(out.ttft)
        e2els.append(out.latency)
        if output_len > 1:
            tpots.append((out.latency - out.ttft) / (output_len - 1))
        completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed — check benchmark configuration.", stacklevel=2
        )

    def _ms(arr: list[float], fn: Any) -> float:
        return float(fn(arr or [0]) * 1000)

    def _p(arr: list[float], p: float) -> float:
        return float(np.percentile(arr or [0], p) * 1000)

    total_output = sum(actual_output_lens)

    entries: dict[str, MetricEntry] = {
        "Completed requests": MetricEntry(completed, "%d"),
        "Duration (s)": MetricEntry(duration_s),
        "Total input tokens": MetricEntry(total_input, "%d"),
        "Total output tokens": MetricEntry(total_output, "%d"),
        "Request throughput (req/s)": MetricEntry(completed / duration_s),
        "Output throughput (tok/s)": MetricEntry(total_output / duration_s),
        "Total token throughput (tok/s)": MetricEntry(
            (total_input + total_output) / duration_s
        ),
        "Mean TTFT (ms)": MetricEntry(_ms(ttfts, np.mean)),
        "Median TTFT (ms)": MetricEntry(_ms(ttfts, np.median)),
        "P90 TTFT (ms)": MetricEntry(_p(ttfts, 90)),
        "P99 TTFT (ms)": MetricEntry(_p(ttfts, 99)),
        "Mean TPOT (ms)": MetricEntry(_ms(tpots, np.mean)),
        "P90 TPOT (ms)": MetricEntry(_p(tpots, 90)),
        "Mean E2EL (ms)": MetricEntry(_ms(e2els, np.mean)),
        "Median E2EL (ms)": MetricEntry(_ms(e2els, np.median)),
        "P90 E2EL (ms)": MetricEntry(_p(e2els, 90)),
        "P99 E2EL (ms)": MetricEntry(_p(e2els, 99)),
    }
    return BenchmarkMetrics(entries=entries), completed, actual_output_lens
