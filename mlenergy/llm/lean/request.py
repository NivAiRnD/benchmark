"""Single-request construction and dispatch for the lean benchmark.

Public API:
    RequestInput        — typed payload derived from SampleRequest + BenchmarkConfig.
    make_request_input  — builds a RequestInput from a sample and config.
    build_requests      — pre-builds all RequestInputs from a sample list.
    send                — sends one chat-completions SSE request and returns a RequestOutput.
    dispatch            — schedules and fires all requests with arrival timing, returns outputs.
    request_handler     — decorator that wraps lifecycle concerns (error handling, tracker).
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import time
import traceback
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Awaitable

import aiohttp
import numpy as np

from mlenergy.llm.lean.metrics import RequestOutput
from mlenergy.llm.lean.tracker import RequestTracker

if TYPE_CHECKING:
    from mlenergy.llm.datasets import SampleRequest
    from mlenergy.llm.lean.config import BenchmarkConfig


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
        session: aiohttp.ClientSession,
        req: RequestInput,
        tracker: RequestTracker,
    ) -> RequestOutput:
        try:
            return await fn(session, req, tracker)
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


def make_request_input(
    config: BenchmarkConfig, sample: SampleRequest, api_url: str
) -> RequestInput:
    traffic = config.traffic
    sampling = config.sampling

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
        model_id=config.server.model_id,
        api_url=api_url,
        extra_body=extra_body,
        system_prompt=sampling.system_prompt,
        ignore_eos=traffic.ignore_eos,
    )


@request_handler
async def send(
    session: aiohttp.ClientSession,
    req: RequestInput,
    tracker: RequestTracker,
) -> RequestOutput:
    """Send a single chat-completions request and parse the streaming response."""
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

    async with session.post(
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


def build_requests(
    config: BenchmarkConfig,
    samples: list[SampleRequest],
    api_url: str,
) -> list[RequestInput]:
    """Pre-build all RequestInputs before the measurement window opens."""
    return [make_request_input(config, s, api_url) for s in samples]


async def _iter_requests(
    requests: list[RequestInput],
    request_rate: float,
    burstiness: float,
) -> AsyncGenerator[RequestInput, None]:
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
    for i, req in enumerate(requests):
        sleep = start + delays[i] - time.time()
        if sleep > 0:
            await asyncio.sleep(sleep)
        yield req


async def dispatch(
    requests: list[RequestInput],
    request_rate: float,
    burstiness: float,
    session: aiohttp.ClientSession,
    tracker: RequestTracker,
    semaphore: asyncio.Semaphore | contextlib.AbstractAsyncContextManager,
) -> list[RequestOutput]:
    """Fire requests as they arrive according to the burstiness schedule."""
    tasks: list[asyncio.Task[RequestOutput]] = []

    async for req in _iter_requests(requests, request_rate, burstiness):
        async def _send(r: RequestInput = req) -> RequestOutput:
            async with semaphore:
                return await send(session, r, tracker)

        tasks.append(asyncio.create_task(_send()))

    return list(await asyncio.gather(*tasks))
