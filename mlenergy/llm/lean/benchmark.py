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
import functools
import json
import logging
import random
import time
import traceback
from collections.abc import AsyncGenerator, Callable, Awaitable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import aiohttp
import numpy as np
from zeus.monitor import ZeusMonitor

from mlenergy.llm.datasets import SampleRequest
from mlenergy.llm.lean.config import BenchmarkConfig
from mlenergy.llm.lean.metrics import BenchmarkResult, RequestOutput, calculate_metrics
from mlenergy.llm.lean.prometheus import PrometheusCollector
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
# Data types
# ---------------------------------------------------------------------------


@dataclass
class RequestInput:
    """Chat-completions request input, derived from a SampleRequest + config."""

    prompt: str | list[str]
    prompt_len: int
    output_len: int | None
    model_id: str
    api_url: str
    extra_body: dict
    system_prompt: str | None
    ignore_eos: bool


@dataclass
class BenchmarkDependencies:
    """All sub-managers for a benchmark run.

    Access by name: deps.zeus, deps.prometheus, deps.session, deps.vllm
    Enter context managers via deps.managed — zeus is excluded (not a CM).
    """

    vllm: VLLMManager
    prometheus: PrometheusCollector
    session: aiohttp.ClientSession
    zeus: ZeusMonitor

    @property
    def managed(self) -> tuple[VLLMManager, PrometheusCollector, aiohttp.ClientSession]:
        """Async context managers in enter order (LIFO cleanup: session, prometheus, vllm)."""
        return (self.vllm, self.prometheus, self.session)


@dataclass
class _RunContext:
    """Async context manager for a single benchmark run.

    __aenter__: records start time, opens Zeus energy window.
    __aexit__:  closes Zeus window, computes metrics, builds BenchmarkResult, validates completion.
    """

    tracker: RequestTracker
    semaphore: asyncio.Semaphore | contextlib.AbstractAsyncContextManager
    api_url: str
    requests: list[SampleRequest]
    deps: BenchmarkDependencies
    tokenizer: PreTrainedTokenizerBase
    outputs: list[RequestOutput] = field(default_factory=list)
    result: BenchmarkResult | None = None
    _benchmark_start: float = field(default=0.0, init=False)

    async def __aenter__(self) -> _RunContext:
        self._benchmark_start = time.time()
        self.deps.zeus.begin_window("benchmark", sync_execution=False)
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        zeus_measurement = self.deps.zeus.end_window("benchmark", sync_execution=False)
        benchmark_end = time.time()

        metrics, completed, _ = calculate_metrics(
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
            prometheus_stats=self.deps.prometheus.calculate_stats(
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
        )
        self.result.log()

        if completed < len(self.requests):
            raise RuntimeError(
                f"Only {completed}/{len(self.requests)} requests completed. "
                "Treating run as failed."
            )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def _build_dependencies(
    config: BenchmarkConfig,
    output_dir: Path,
    max_num_seqs: int,
    max_num_batched_tokens: int | None,
) -> BenchmarkDependencies:
    """Construct all benchmark sub-managers. Contexts are entered later in __aenter__."""
    log_path = output_dir / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return BenchmarkDependencies(
        vllm=VLLMManager(
            config=config.server,
            log_path=log_path,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        prometheus=PrometheusCollector(
            metrics_url=config.server.base_url() + "/metrics",
            interval=_PROMETHEUS_INTERVAL_S,
        ),
        session=aiohttp.ClientSession(
            timeout=_AIOHTTP_TIMEOUT,
            connector=aiohttp.TCPConnector(
                limit=0, ssl=False, keepalive_timeout=6 * 60 * 60
            ),
        ),
        zeus=ZeusMonitor(),
    )


async def _iter_requests(
    requests: list[SampleRequest],
    request_rate: float,
    burstiness: float,
) -> AsyncGenerator[SampleRequest, None]:
    """Yield requests spaced according to a gamma-distributed arrival process.

    When request_rate is inf, all requests are yielded immediately.
    Normalises cumulative delays to match the target total duration exactly,
    closing the ~1-2% gap that would otherwise accumulate from random samples.
    """
    assert burstiness > 0, f"burstiness must be positive, got {burstiness}"
    assert len(requests) > 0, "request list is empty"

    if request_rate == float("inf"):
        delays = [0.0] * len(requests)
    else:
        theta = 1.0 / (request_rate * burstiness)
        raw = [float(np.random.gamma(shape=burstiness, scale=theta)) for _ in requests]
        for i in range(1, len(raw)):
            raw[i] += raw[i - 1]
        target = len(requests) / request_rate
        factor = target / raw[-1] if raw[-1] != 0 else 1.0
        delays = [d * factor for d in raw]

    start = time.time()
    for i, request in enumerate(requests):
        sleep = start + delays[i] - time.time()
        if sleep > 0:
            await asyncio.sleep(sleep)
        yield request


def request_handler(
    fn: Callable[..., Awaitable[RequestOutput]],
) -> Callable[..., Awaitable[RequestOutput]]:
    """Lifecycle wrapper for request coroutines.

    Ensures tracker.notify_finished() always runs and catches all exceptions,
    returning a failed RequestOutput rather than propagating. The decorated
    function only needs to implement payload construction and stream parsing.
    """

    @functools.wraps(fn)
    async def wrapper(
        self: "Benchmark",
        req: RequestInput,
        tracker: RequestTracker,
    ) -> RequestOutput:
        try:
            return await fn(self, req, tracker)
        except Exception:
            return RequestOutput(
                prompt=req.prompt,
                prompt_len=req.prompt_len,
                success=False,
                error=traceback.format_exc(),
            )
        finally:
            tracker.notify_finished()

    return wrapper


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class Benchmark:
    """Aggregated async context manager for the full benchmark lifecycle.

    Composes VLLMManager, PrometheusCollector, and aiohttp.ClientSession via
    AsyncExitStack — cleanup is LIFO and automatic.

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
    ) -> None:
        self._config = config
        self._tokenizer = tokenizer
        self._deps: BenchmarkDependencies = _build_dependencies(
            config, output_dir, max_num_seqs, max_num_batched_tokens
        )
        self._stack: contextlib.AsyncExitStack | None = None

    async def __aenter__(self) -> "Benchmark":
        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        for dep in self._deps.managed:
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
        tasks: list[asyncio.Task[RequestOutput]] = []
        start_event = asyncio.Event()
        async with self._measurement_window(requests) as ctx:
            async for sample in _iter_requests(
                requests,
                self._config.traffic.request_rate,
                self._config.traffic.burstiness,
            ):
                req = self._make_request_input(sample, ctx.api_url)

                async def _limited(r: RequestInput = req) -> RequestOutput:
                    await start_event.wait()
                    async with ctx.semaphore:
                        return await self._send(r, ctx.tracker)

                tasks.append(asyncio.create_task(_limited()))

            start_event.set()
            ctx.outputs = list(await asyncio.gather(*tasks))

        assert ctx.result is not None
        return ctx.result

    @contextlib.asynccontextmanager
    async def _measurement_window(
        self, requests: list[SampleRequest]
    ) -> AsyncGenerator[_RunContext, None]:
        """Seeds RNG, constructs and enters _RunContext, yields it."""
        random.seed(self._config.seed)
        np.random.seed(self._config.seed)

        ctx = _RunContext(
            tracker=RequestTracker(total=len(requests)),
            semaphore=(
                asyncio.Semaphore(self._config.traffic.max_concurrency)
                if self._config.traffic.max_concurrency
                else contextlib.nullcontext()
            ),
            api_url=f"{self._config.server.base_url()}/v1/chat/completions",
            requests=requests,
            deps=self._deps,
            tokenizer=self._tokenizer,
        )
        async with ctx:
            yield ctx

    def _make_request_input(self, sample: SampleRequest, api_url: str) -> RequestInput:
        traffic = self._config.traffic
        sampling = self._config.sampling

        if traffic.max_output_tokens is None:
            output_len = None
        elif isinstance(traffic.max_output_tokens, int):
            output_len = traffic.max_output_tokens
        else:  # "dataset"
            output_len = sample.expected_output_len

        extra_body = {
            **sampling.extra_body,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
        }

        return RequestInput(
            prompt=sample.prompt,
            prompt_len=sample.prompt_len,
            output_len=output_len,
            model_id=self._config.server.model_id,
            api_url=api_url,
            extra_body=extra_body,
            system_prompt=sampling.system_prompt,
            ignore_eos=traffic.ignore_eos,
        )

    @request_handler
    async def _send(self, req: RequestInput, tracker: RequestTracker) -> RequestOutput:
        """Send a single chat-completions request and parse the streaming response."""
        assert self._deps.session is not None

        if isinstance(req.prompt, str):
            content: list[dict[str, Any]] = [{"type": "text", "text": req.prompt}]
            messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        else:
            content = [{"type": "text", "text": req.prompt[0]}]
            messages = [{"role": "user", "content": content}]
            for i, turn in enumerate(req.prompt[1:]):
                role = "user" if i % 2 == 1 else "assistant"
                messages.append(
                    {"role": role, "content": [{"type": "text", "text": turn}]}
                )

        if req.system_prompt:
            messages.insert(0, {"role": "system", "content": req.system_prompt})

        payload: dict[str, Any] = {
            "model": req.model_id,
            "messages": messages,
            "max_completion_tokens": req.output_len,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        if req.ignore_eos:
            payload["ignore_eos"] = True
        if req.extra_body:
            payload.update(req.extra_body)

        output = RequestOutput(prompt=req.prompt, prompt_len=req.prompt_len)
        output_text = ""
        reasoning_text = ""
        ttft = 0.0
        current_completion_tokens = 0
        st = time.perf_counter()
        most_recent_ts = st

        async with self._deps.session.post(
            req.api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            if response.status != 200:
                output.error = (await response.text()) or response.reason or ""
                output.success = False
                return output

            async for chunk_bytes in response.content:
                chunk_bytes = chunk_bytes.strip()
                if not chunk_bytes:
                    continue
                chunk = chunk_bytes.decode("utf-8")
                if chunk.startswith(":"):
                    continue
                chunk = chunk.removeprefix("data: ")
                if chunk == "[DONE]":
                    continue

                timestamp = time.perf_counter()
                data = json.loads(chunk)

                usage = data.get("usage")
                completion_tokens = usage and usage.get("completion_tokens")
                if not completion_tokens:
                    continue

                if choices := data.get("choices"):
                    delta = choices[0]["delta"]
                    content_text = delta.get("content")
                    reasoning_content = delta.get("reasoning_content")

                    inc = completion_tokens - current_completion_tokens
                    tracker.notify_tokens_generated(inc)
                    current_completion_tokens = completion_tokens

                    if ttft == 0.0:
                        ttft = timestamp - st
                        output.ttft = ttft
                        for _ in range(inc):
                            output.itl.append(0.0)
                    else:
                        output.itl.append(timestamp - most_recent_ts)
                        for _ in range(inc - 1):
                            output.itl.append(0.0)

                    most_recent_ts = timestamp
                    if content_text is not None:
                        output_text += content_text
                    if reasoning_content is not None:
                        reasoning_text += reasoning_content

                elif usage:
                    output.output_tokens = usage.get("completion_tokens", 0)

        output.output_text = output_text
        output.reasoning_output_text = reasoning_text
        output.latency = most_recent_ts - st
        output.success = ttft > 0.0
        if not output.success:
            output.error = "No valid chunks received — response produced no tokens."
        return output
