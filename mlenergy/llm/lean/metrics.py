"""Benchmark result types and metrics calculation.

Defines the data model for benchmark outputs and the function that reduces
raw request outputs into aggregate statistics.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from mlenergy.llm.datasets import SampleRequest
from mlenergy.llm.lean.request import RequestOutput

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from mlenergy.llm.lean.config import BenchmarkConfig
    from mlenergy.llm.workloads import WorkloadConfig

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

    Add new metrics in BenchmarkMetrics.calculate — logging and serialization
    traverse entries automatically.
    """

    entries: dict[str, MetricEntry]

    @classmethod
    def calculate(
        cls,
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
        return cls(entries=entries), completed, actual_output_lens


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
    cpu_power_trace: list[tuple[float, float]] = field(default_factory=list)

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

    def save(
        self,
        output_dir: Path,
        task_dir: Path,
        config: BenchmarkConfig,
        workload: WorkloadConfig,
        run_id: str,
    ) -> None:
        """Write all run artifacts and update the task-level runs index.

        output_dir/
          git.diff       — uncommitted diff at run time (empty = clean)
          config.json    — workload + traffic + sampling params
          command.txt    — exact CLI invocation
          results.json   — metrics, energy, per-request data

        task_dir/
          runs.json      — one entry per run, for scanning across experiments
        """
        # --- Git provenance ---
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_commit = ""

        try:
            git_diff = subprocess.check_output(
                ["git", "diff", "HEAD"], text=True, stderr=subprocess.DEVNULL
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_diff = ""

        (output_dir / "git.diff").write_text(git_diff)

        # --- Config snapshot ---
        t = config.traffic
        config_data = {
            "workload": workload.model_dump(exclude={"run_id"}),
            "traffic": {
                "request_rate": t.request_rate if t.request_rate < float("inf") else "inf",
                "burstiness": t.burstiness,
                "max_concurrency": t.max_concurrency,
                "max_output_tokens": t.max_output_tokens,
                "ignore_eos": t.ignore_eos,
            },
            "sampling": {
                "temperature": config.sampling.temperature,
                "top_p": config.sampling.top_p,
            },
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2, default=str)

        (output_dir / "command.txt").write_text(" ".join(sys.argv))

        # --- Results ---
        results_data = {
            "run_id": run_id,
            "git_commit": git_commit,
            "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "model_id": workload.model_id,
            "gpu_model": workload.gpu_model,
            "num_gpus": workload.num_gpus,
            "num_requests": workload.num_requests,
            "num_request_repeats": workload.num_request_repeats,
            "seed": workload.seed,
            "max_num_seqs": workload.max_num_seqs,
            "max_num_batched_tokens": workload.max_num_batched_tokens,
            "request_rate": t.request_rate if t.request_rate < float("inf") else "inf",
            "burstiness": t.burstiness,
            "max_concurrency": t.max_concurrency,
            "max_output_tokens": t.max_output_tokens,
            "metrics": {label: entry.value for label, entry in self.metrics.entries.items()},
            "energy_j": self.energy_j,
            "energy_per_token_j": self.energy_per_token_j,
            "prometheus_stats": self.prometheus_stats,
            "power_trace": self.power_trace,
            "cpu_power_trace": self.cpu_power_trace,
            "per_request": [
                {
                    "ttft": o.ttft,
                    "latency": o.latency,
                    "output_tokens": o.output_tokens,
                    "success": o.success,
                }
                for o in self.per_request
            ],
        }
        result_file = output_dir / "results.json"
        with open(result_file, "w") as f:
            json.dump(results_data, f, indent=2)
        logger.info("Saved results to %s", result_file)

        # --- Task-level runs index ---
        runs_index_file = task_dir / "runs.json"
        runs_index: list[dict] = []
        if runs_index_file.exists():
            with open(runs_index_file) as f:
                runs_index = json.load(f)
        runs_index.append({
            "run_id": run_id,
            "git_commit": git_commit,
            "model_id": workload.model_id,
            "gpu_model": workload.gpu_model,
            **config_data,
        })
        with open(runs_index_file, "w") as f:
            json.dump(runs_index, f, indent=2, default=str)
        logger.info("Updated runs index at %s", runs_index_file)
