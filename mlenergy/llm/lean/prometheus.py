"""Prometheus metrics collection and analysis for vLLM benchmarks.

Scrapes /metrics periodically while the benchmark runs, stores snapshots,
and computes statistics over a specified time window.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Final, Literal

import aiohttp

logger = logging.getLogger("mlenergy.llm.lean")

_SCRAPE_TIMEOUT_S: Final[float] = 5.0
_DEFAULT_PERCENTILES: Final[tuple[float, ...]] = (50.0, 90.0, 95.0, 99.0)


@dataclass(frozen=True)
class Snapshot:
    timestamp: float
    metrics: str


class PrometheusCollector:
    """Async context manager that scrapes vLLM's /metrics endpoint periodically.

    Usage::

        async with PrometheusCollector(metrics_url) as collector:
            ...  # benchmark runs here

        stats = collector.calculate_stats(
            window_start=t0,
            window_end=t1,
            gauge_metrics={"vllm:num_requests_running": "sum"},
            histogram_metrics=["vllm:request_prompt_tokens"],
        )
    """

    def __init__(
        self,
        metrics_url: str,
        interval: float = 1.0,
        ready_event: asyncio.Event | None = None,
    ) -> None:
        self._metrics_url = metrics_url
        self._interval = interval
        self._ready_event = ready_event
        self._timeline: list[Snapshot] = []
        self._stop: asyncio.Event | None = None
        self._collect_task: asyncio.Task[None] | None = None
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> PrometheusCollector:
        self._stop = asyncio.Event()
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=_SCRAPE_TIMEOUT_S)
        )
        self._collect_task = asyncio.create_task(self._collect())
        logger.info(
            "Started Prometheus collection from %s (interval: %.1fs)",
            self._metrics_url,
            self._interval,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._stop is not None
        self._stop.set()
        if self._collect_task is not None:
            await self._collect_task
        if self._session is not None:
            await self._session.close()
        logger.info(
            "Stopped Prometheus collection. %d snapshots collected.",
            len(self._timeline),
        )

    @property
    def timeline(self) -> list[Snapshot]:
        return list(self._timeline)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def calculate_stats(
        self,
        window_start: float,
        window_end: float,
        gauge_metrics: dict[str, Literal["sum", "avg", "max"]],
        histogram_metrics: list[str] | None = None,
        percentiles: tuple[float, ...] = _DEFAULT_PERCENTILES,
    ) -> dict[str, float]:
        """Compute statistics over snapshots within [window_start, window_end].

        For gauges: average the per-snapshot aggregated value across all snapshots
        in the window.

        For histograms: use the last snapshot in the window (histograms are
        cumulative) and compute percentiles via linear bucket interpolation.

        Returns a flat dict, e.g.::

            {
                "vllm:num_requests_running": 42.3,
                "vllm:request_prompt_tokens_p50": 1500.0,
                "vllm:request_prompt_tokens_p99": 4100.0,
            }
        """
        window = [
            s for s in self._timeline if window_start <= s.timestamp <= window_end
        ]
        if not window:
            logger.warning(
                "No Prometheus snapshots in window [%.3f, %.3f]",
                window_start,
                window_end,
            )
            return {}

        logger.info(
            "Analysing %d snapshots in steady-state window [%.3f, %.3f]",
            len(window),
            window_start,
            window_end,
        )

        stats: dict[str, float] = {}

        for metric_name, aggregation in gauge_metrics.items():
            values = [
                v
                for s in window
                if (v := self._aggregate_gauge(s.metrics, metric_name, aggregation))
                is not None
            ]
            if values:
                stats[metric_name] = sum(values) / len(values)
                logger.info(
                    "%s: %.3f (agg=%s, n=%d)",
                    metric_name,
                    stats[metric_name],
                    aggregation,
                    len(values),
                )
            else:
                logger.warning("No values found for gauge metric: %s", metric_name)

        if histogram_metrics:
            last_snapshot = window[-1]
            for metric_name in histogram_metrics:
                histogram = self._parse_histogram(last_snapshot.metrics, metric_name)
                for p in percentiles:
                    value = self._histogram_percentile(histogram, p)
                    if value is not None:
                        p_key = f"p{int(p)}" if p == int(p) else f"p{p}"
                        full_key = f"{metric_name}_{p_key}"
                        stats[full_key] = value

                if any(f"{metric_name}_p" in k for k in stats):
                    p_str = ", ".join(
                        f"{k.split('_')[-1]}={v:.3f}"
                        for k, v in stats.items()
                        if k.startswith(metric_name)
                    )
                    logger.info("%s: %s", metric_name, p_str)
                else:
                    logger.warning(
                        "Could not calculate percentiles for: %s", metric_name
                    )

        return stats

    # ------------------------------------------------------------------
    # Private — collection
    # ------------------------------------------------------------------

    async def _collect(self) -> None:
        assert self._stop is not None and self._session is not None
        if self._ready_event is not None:
            await self._ready_event.wait()
        while not self._stop.is_set():
            try:
                async with self._session.get(self._metrics_url) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        self._timeline.append(
                            Snapshot(timestamp=time.time(), metrics=text)
                        )
                    else:
                        logger.warning(
                            "Prometheus scrape returned HTTP %d", resp.status
                        )
            except asyncio.TimeoutError:
                logger.warning("Prometheus scrape timed out")
            except Exception as exc:
                logger.warning("Prometheus scrape error: %s", exc)

            await asyncio.sleep(self._interval)

    # ------------------------------------------------------------------
    # Private — parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_gauge(metrics_text: str, metric_name: str) -> dict[str, float]:
        pattern = rf"^{re.escape(metric_name)}\{{([^}}]*)\}}\s+([\d.eE+-]+)"
        return {
            m.group(1): float(m.group(2))
            for m in re.finditer(pattern, metrics_text, re.MULTILINE)
        }

    @staticmethod
    def _parse_histogram(metrics_text: str, metric_name: str) -> dict[str, Any]:
        def _extract(pattern: str) -> dict[str, float]:
            return {
                m.group(1): float(m.group(2))
                for m in re.finditer(pattern, metrics_text, re.MULTILINE)
            }

        return {
            "buckets": _extract(
                rf"^{re.escape(metric_name)}_bucket\{{([^}}]*)\}}\s+([\d.eE+-]+)"
            ),
            "sum": _extract(
                rf"^{re.escape(metric_name)}_sum\{{([^}}]*)\}}\s+([\d.eE+-]+)"
            ),
            "count": _extract(
                rf"^{re.escape(metric_name)}_count\{{([^}}]*)\}}\s+([\d.eE+-]+)"
            ),
        }

    def _aggregate_gauge(
        self,
        metrics_text: str,
        metric_name: str,
        aggregation: Literal["sum", "avg", "max"],
    ) -> float | None:
        values = list(self._parse_gauge(metrics_text, metric_name).values())
        if not values:
            return None
        if aggregation == "sum":
            return sum(values)
        if aggregation == "avg":
            return sum(values) / len(values)
        if aggregation == "max":
            return max(values)
        raise ValueError(f"Unknown aggregation: {aggregation}")

    @staticmethod
    def _histogram_percentile(
        histogram: dict[str, Any], percentile: float
    ) -> float | None:
        buckets = histogram.get("buckets", {})
        if not buckets:
            return None

        # Build sorted list of (upper_bound, cumulative_count)
        bucket_data: list[tuple[float, float]] = []
        for labels, count in buckets.items():
            le_match = re.search(r'le="([^"]+)"', labels)
            if le_match:
                le_str = le_match.group(1)
                upper = float("inf") if le_str == "+Inf" else float(le_str)
                bucket_data.append((upper, count))
        if not bucket_data:
            return None

        bucket_data.sort(key=lambda x: x[0])
        total = bucket_data[-1][1]
        if total == 0:
            return None

        target = total * (percentile / 100.0)
        prev_upper, prev_count = 0.0, 0.0

        for upper, cumulative in bucket_data:
            if cumulative >= target:
                if prev_count == cumulative:
                    return prev_upper
                fraction = (target - prev_count) / (cumulative - prev_count)
                return prev_upper + fraction * (upper - prev_upper)
            prev_upper, prev_count = upper, cumulative

        return None
