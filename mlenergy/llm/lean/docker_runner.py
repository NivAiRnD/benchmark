"""Generic Docker container lifecycle management via asyncio subprocess."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    image: str
    name: str
    gpu_device_ids: str
    env_vars: dict[str, str]
    volumes: list[str]
    entrypoint_cmd: list[str]
    startup_string: str
    termination_strings: frozenset[str]
    ready_timeout_s: float


class DockerRunner(ABC):
    """Abstract async context manager for Docker container lifecycle.

    Subclasses implement _container_config() to provide engine-specific
    container setup. The base class handles subprocess spawning, log watching,
    and readiness/termination detection.

    Usage::

        async with MyRunner(..., log_path) as runner:
            await runner.wait_ready()
            ...  # container is running
        # container is terminated on exit
    """

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._proc: asyncio.subprocess.Process | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._ready_event: asyncio.Event = asyncio.Event()
        self._terminated_event: asyncio.Event = asyncio.Event()
        self._termination_line: str = ""
        self._cfg: ContainerConfig | None = None

    @abstractmethod
    def _container_config(self) -> ContainerConfig: ...

    async def __aenter__(self) -> DockerRunner:
        self._cfg = self._container_config()
        cmd = self._build_cmd(self._cfg)

        logger.info("Spawning container: %s", " ".join(cmd))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._watch_task = asyncio.create_task(self._watch_output())
        return self

    async def __aexit__(self, *_: object) -> None:
        self._validate_still_running()
        if self._proc is not None:
            self._proc.terminate()
            await self._proc.wait()
        if self._watch_task is not None:
            await self._watch_task

    @property
    def ready_event(self) -> asyncio.Event:
        return self._ready_event

    async def wait_ready(self) -> None:
        """Wait for the container to signal startup. Raises on timeout or detected crash."""
        assert self._cfg is not None
        ready_task = asyncio.create_task(self._ready_event.wait())
        terminated_task = asyncio.create_task(self._terminated_event.wait())

        done, pending = await asyncio.wait(
            {ready_task, terminated_task},
            timeout=self._cfg.ready_timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

        if not done:
            raise TimeoutError(
                f"Container did not become ready within {self._cfg.ready_timeout_s}s. "
                f"Check logs at {self._log_path}"
            )
        if terminated_task in done:
            raise RuntimeError(
                f"Container terminated during startup: {self._termination_line}\n"
                f"Check logs at {self._log_path}"
            )
        logger.info("Container is ready.")

    @staticmethod
    def _build_cmd(cfg: ContainerConfig) -> list[str]:
        env_flags: list[str] = []
        for k, v in cfg.env_vars.items():
            env_flags += ["-e", f"{k}={v}"]

        volume_flags: list[str] = []
        for mount in cfg.volumes:
            volume_flags += ["-v", mount]

        return [
            "docker", "run",
            "--rm",
            f"--name={cfg.name}",
            "--gpus", f'"device={cfg.gpu_device_ids}"',
            "--ipc=host",
            "--net=host",
            "--entrypoint", "",
            *env_flags,
            *volume_flags,
            cfg.image,
            *cfg.entrypoint_cmd,
        ]

    def _validate_still_running(self) -> None:
        if self._proc is not None and self._proc.returncode is not None:
            raise RuntimeError(
                f"Container exited unexpectedly with code {self._proc.returncode} "
                f"during the benchmark. Check logs at {self._log_path}"
            )

    async def _watch_output(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        assert self._cfg is not None

        with open(self._log_path, "w") as log_file:
            async for line_bytes in self._proc.stdout:
                line = line_bytes.decode(errors="replace")
                log_file.write(line)
                log_file.flush()

                if self._cfg.startup_string in line and not self._ready_event.is_set():
                    logger.info("Startup detected: %s", line.rstrip())
                    self._ready_event.set()

                elif any(s in line.lower() for s in self._cfg.termination_strings):
                    logger.error("Termination string detected: %s", line.rstrip())
                    self._termination_line = line.strip()
                    self._terminated_event.set()
