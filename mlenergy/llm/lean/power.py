"""Async power samplers for GPU and CPU.

Both subclasses share the same async context manager lifecycle, polling loop,
and trace API via PowerSampler. Subclasses implement only `_read_watts` and,
if needed, `_on_enter` for pre-poll setup.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod

from zeus.monitor import ZeusMonitor

_GPU_INTERVAL_S: float = 0.1
_CPU_INTERVAL_S: float = 0.1


class PowerSampler(ABC):
    """Base async context manager that polls watts at a fixed interval.

    Records (seconds_since_start, watts) pairs. Subclasses implement
    `_read_watts` to return the current power reading (or None to skip).
    """

    def __init__(self, zeus: ZeusMonitor, interval: float) -> None:
        self._zeus = zeus
        self._interval = interval
        self._start: float = 0.0
        self._trace: list[tuple[float, float]] = []
        self._stop: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> PowerSampler:
        self._start = time.time()
        self._stop = asyncio.Event()
        await self._on_enter()
        self._task = asyncio.create_task(self._poll())
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    async def _on_enter(self) -> None:
        """Hook for subclass setup that must run before the poll loop starts."""

    async def _poll(self) -> None:
        loop = asyncio.get_event_loop()
        while not self._stop.is_set():
            await asyncio.sleep(self._interval)
            t = time.time() - self._start
            watts = await loop.run_in_executor(None, self._read_watts)
            if watts is not None:
                self._trace.append((t, watts))

    @abstractmethod
    def _read_watts(self) -> float | None:
        """Return current power in watts, or None to skip this sample."""

    @property
    def trace(self) -> list[tuple[float, float]]:
        return list(self._trace)


class GPUPowerSampler(PowerSampler):
    """Polls instant GPU power via Zeus."""

    def __init__(self, zeus: ZeusMonitor, interval: float = _GPU_INTERVAL_S) -> None:
        super().__init__(zeus, interval)

    def _read_watts(self) -> float:
        return sum(
            self._zeus.gpus.get_instant_power_usage(i) / 1000.0
            for i in self._zeus.gpu_indices
        )


class CPUPowerSampler(PowerSampler):
    """Polls CPU power via RAPL using delta-energy / delta-time."""

    def __init__(self, zeus: ZeusMonitor, interval: float = _CPU_INTERVAL_S) -> None:
        super().__init__(zeus, interval)
        self._prev_energy_j: float = 0.0
        self._prev_time: float = 0.0

    async def _on_enter(self) -> None:
        loop = asyncio.get_event_loop()
        self._prev_energy_j = await loop.run_in_executor(None, self._read_energy_j)
        self._prev_time = time.time()

    def _read_watts(self) -> float | None:
        t = time.time()
        energy_j = self._read_energy_j()
        dt = t - self._prev_time
        if dt <= 0:
            return None
        watts = (energy_j - self._prev_energy_j) / dt
        self._prev_energy_j = energy_j
        self._prev_time = t
        return watts

    def _read_energy_j(self) -> float:
        if not self._zeus.cpu_indices:
            return 0.0
        return sum(
            self._zeus.cpus.get_total_energy_consumption(i).cpu_mj / 1000.0
            for i in self._zeus.cpu_indices
        )
