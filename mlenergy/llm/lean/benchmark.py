"""Lean LLM benchmark — orchestration, request dispatch, and result collection.

Usage::

    async with Benchmark(config, output_dir, max_num_seqs, tokenizer) as bm:
        result = await bm.run(requests)

The Benchmark context manager composes all sub-managers (vLLM container,
Prometheus scraper, aiohttp session) via AsyncExitStack so cleanup is always
LIFO and automatic regardless of errors.

Measurement window: from the moment the first task is dispatched to the moment
all tasks complete — capturing the full curve including prefill burst and transients.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import aiohttp
import numpy as np
from zeus.monitor import ZeusMonitor

from mlenergy.llm.datasets import SampleRequest
from mlenergy.llm.lean.config import BenchmarkConfig
from mlenergy.llm.lean.metrics import BenchmarkMetrics, BenchmarkResult, RequestOutput
from mlenergy.llm.lean.power import CPUPowerSampler, GPUPowerSampler, PowerSampler
from mlenergy.llm.lean.prometheus import PrometheusCollector
from mlenergy.llm.lean.request import RequestInput
from mlenergy.llm.lean.tracker import RequestTracker
from mlenergy.llm.lean.vllm_manager import VLLMManager

if TYPE_CHECKING:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger("mlenergy.llm.lean")

_AIOHTTP_TIMEOUT: Final[aiohttp.ClientTimeout] = aiohttp.ClientTimeout(
    total=6 * 60 * 60
)
_PROMETHEUS_INTERVAL_S: Final[float] = 1.0


# ---------------------------------------------------------------------------
# Managed dependencies
# ---------------------------------------------------------------------------


class _ManagedDeps:
    """Long-lived infrastructure that persists across all benchmark runs.

    These dependencies are entered once in Benchmark.__aenter__ and remain
    active for the entire Benchmark lifetime — including across multiple
    run() calls.  They represent the mandatory substrate that must be up
    before any measurement can happen: the vLLM server, the Prometheus
    scraper, and the HTTP session.

    Encapsulates construction order and the ready_event wiring between
    VLLMManager and PrometheusCollector.  Iterable over the three async
    context managers so callers can do:
        for dep in self._deps: await stack.enter_async_context(dep)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        log_path: Path,
        max_num_seqs: int,
        max_num_batched_tokens: int | None,
        monitor_cpu_power: bool,
    ) -> None:
        self.vllm = VLLMManager(
            config=config.server,
            log_path=log_path,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        self.prometheus = PrometheusCollector(
            metrics_url=config.server.base_url() + "/metrics",
            interval=_PROMETHEUS_INTERVAL_S,
            ready_event=self.vllm.ready_event,
        )
        self.session = aiohttp.ClientSession(
            timeout=_AIOHTTP_TIMEOUT,
            connector=aiohttp.TCPConnector(
                limit=0, ssl=False, keepalive_timeout=6 * 60 * 60
            ),
        )
        self.zeus = ZeusMonitor()
        self.power_sampler = GPUPowerSampler(self.zeus)
        self.cpu_sampler = CPUPowerSampler(self.zeus) if monitor_cpu_power else None

    def __iter__(self):
        return iter((self.vllm, self.prometheus, self.session))


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


class _RunContext:
    """Measurement window for a single benchmark run.

    Opened and closed once per run() call, independently of the long-lived
    _ManagedDeps infrastructure.  The window is intentionally tight: it starts
    the moment requests are dispatched and ends when all tasks complete, so
    energy and timing measurements exclude server startup and idle overhead.

    __aenter__: records start time, opens Zeus energy window, starts power samplers.
    __aexit__:  stops power samplers, closes Zeus window, computes metrics,
                builds BenchmarkResult, validates completion.
    """

    def __init__(
        self,
        requests: list[SampleRequest],
        deps: _ManagedDeps,
        tokenizer: PreTrainedTokenizerBase,
        max_concurrency: int | None,
    ) -> None:
        self.requests = requests
        self.tracker = RequestTracker(total=len(requests))
        self.semaphore: asyncio.Semaphore | contextlib.AbstractAsyncContextManager = (
            asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()
        )
        self.zeus = deps.zeus
        self.prometheus = deps.prometheus
        self.power_sampler = deps.power_sampler
        self.cpu_sampler = deps.cpu_sampler
        self.tokenizer = tokenizer
        self.outputs: list[RequestOutput] = []
        self.result: BenchmarkResult | None = None
        self._benchmark_start: float = 0.0

    async def __aenter__(self) -> _RunContext:
        self._benchmark_start = time.time()
        self.zeus.begin_window("benchmark", sync_execution=False)
        await self.power_sampler.__aenter__()
        if self.cpu_sampler is not None:
            await self.cpu_sampler.__aenter__()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        await self.power_sampler.__aexit__(None, None, None)
        if self.cpu_sampler is not None:
            await self.cpu_sampler.__aexit__(None, None, None)
        zeus_measurement = self.zeus.end_window("benchmark", sync_execution=False)
        benchmark_end = time.time()

        metrics, completed, _ = BenchmarkMetrics.calculate(
            requests=self.requests,
            outputs=self.outputs,
            tokenizer=self.tokenizer,
            duration_s=benchmark_end - self._benchmark_start,
        )

        energy_j = sum(zeus_measurement.gpu_energy.values())
        self.result = BenchmarkResult(
            metrics=metrics,
            energy_j=energy_j,
            energy_per_token_j=energy_j / (self.tracker.tokens_generated or 1),
            benchmark_start_time=self._benchmark_start,
            benchmark_end_time=benchmark_end,
            prometheus_stats=self.prometheus.calculate_stats(
                window_start=self._benchmark_start,
                window_end=benchmark_end,
                gauge_metrics={
                    "vllm:num_requests_running": "sum",
                    "vllm:kv_cache_usage_perc": "avg",
                },
                histogram_metrics=[
                    "vllm:request_prompt_tokens",
                    "vllm:request_generation_tokens",
                    "vllm:request_prefill_time_seconds",
                    "vllm:inter_token_latency_seconds",
                ],
            ),
            per_request=self.outputs,
            power_trace=self.power_sampler.trace,
            cpu_power_trace=self.cpu_sampler.trace if self.cpu_sampler else [],
        )
        self.result.log()

        if completed < len(self.requests):
            raise RuntimeError(
                f"Only {completed}/{len(self.requests)} requests completed. "
                "Treating run as failed."
            )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class Benchmark:
    """Async context manager for the full benchmark lifecycle.

    Owns all sub-managers (vLLM container, Prometheus scraper, aiohttp session)
    and enters/exits them via AsyncExitStack — cleanup is LIFO and automatic.

    Usage::

        async with Benchmark(config, output_dir, max_num_seqs, tokenizer) as bm:
            result = await bm.run(requests)
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        output_dir: Path,
        max_num_seqs: int,
        tokenizer: PreTrainedTokenizerBase,
        max_num_batched_tokens: int | None = None,
        monitor_cpu_power: bool = False,
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer

        log_path = output_dir / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        self._deps = _ManagedDeps(config, log_path, max_num_seqs, max_num_batched_tokens, monitor_cpu_power)
        self._stack: contextlib.AsyncExitStack | None = None

    async def __aenter__(self) -> "Benchmark":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        for dep in self._deps:
            await self._stack.enter_async_context(dep)
        await self._deps.vllm.wait_ready()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        assert self._stack is not None
        await self._stack.__aexit__(exc_type, exc_val, exc_tb)  # type: ignore[arg-type]

    async def run(self, requests: list[SampleRequest]) -> BenchmarkResult:
        """Dispatch all requests and collect results."""
        random.seed(self._config.seed)
        np.random.seed(self._config.seed)

        api_url = f"{self._config.server.base_url()}/v1/chat/completions"
        request_inputs = RequestInput.build_all(self._config, requests, api_url)

        async with _RunContext(requests, self._deps, self._tokenizer, self._config.traffic.max_concurrency) as ctx:
            ctx.outputs = await RequestInput.dispatch_all(
                requests=request_inputs,
                request_rate=self._config.traffic.request_rate,
                burstiness=self._config.traffic.burstiness,
                session=self._deps.session,
                tracker=ctx.tracker,
                semaphore=ctx.semaphore,
            )

        assert ctx.result is not None
        return ctx.result
