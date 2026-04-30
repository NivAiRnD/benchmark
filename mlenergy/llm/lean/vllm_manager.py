"""vLLM Docker container lifecycle management.

Responsibilities:
- Parse per-model vLLM YAML config and env vars from the config directory.
- Spawn the Docker container via asyncio subprocess with stdout/stderr piped.
- Detect readiness and termination from log output without polling.
- Validate the container is still alive after the benchmark completes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import yaml

from mlenergy.llm.lean.config import ServerConfig
from mlenergy.llm.lean.constants import (
    _CONTAINER_CONFIG_PATH,
    _CONTAINER_NAME_PREFIX,
    _CONTAINER_VLLM_CACHE_DIR,
    _DATA_PARALLEL_ARG,
    _STARTUP_STRING,
    _TENSOR_PARALLEL_ARG,
    _TERMINATION_STRINGS,
    _VLLM_CONFIG_FILENAME,
    _VLLM_ENV_FILENAME,
)

logger = logging.getLogger("mlenergy.llm.lean")


class VLLMManager:
    """Async context manager for the vLLM Docker container lifecycle.

    Usage::

        async with VLLMManager(server_config, log_path, max_num_seqs) as mgr:
            await mgr.wait_ready()
            ...  # run benchmark
        # container is terminated and validated on exit
    """

    def __init__(
        self,
        config: ServerConfig,
        log_path: Path,
        max_num_seqs: int,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        self._config = config
        self._log_path = log_path
        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens

        self._proc: asyncio.subprocess.Process | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._ready_event: asyncio.Event = asyncio.Event()
        self._terminated_event: asyncio.Event = asyncio.Event()
        self._termination_line: str = ""

    async def __aenter__(self) -> VLLMManager:
        hf_token = os.environ["HF_TOKEN"]
        hf_home = os.environ["HF_HOME"]
        vllm_cache_dir = os.environ.get("VLLM_CACHE_DIR")

        vllm_config_path = self._get_vllm_config_path()
        env_vars = self._load_env_vars()

        config_text = vllm_config_path.read_text()
        parallel_arg = _DATA_PARALLEL_ARG if _DATA_PARALLEL_ARG.lstrip("-") in config_text else _TENSOR_PARALLEL_ARG

        cmd = self._build_docker_command(
            config_path=vllm_config_path,
            env_vars=env_vars,
            hf_token=hf_token,
            hf_home=hf_home,
            vllm_cache_dir=vllm_cache_dir,
            parallel_arg=parallel_arg,
        )

        logger.info("Spawning vLLM container: %s", " ".join(cmd))
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._watch_task = asyncio.create_task(self._watch_output())
        return self

    async def __aexit__(self, *exc: object) -> None:
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
        """Wait for vLLM to signal startup. Raises on timeout or detected crash."""
        ready_task = asyncio.create_task(self._ready_event.wait())
        terminated_task = asyncio.create_task(self._terminated_event.wait())

        done, pending = await asyncio.wait(
            {ready_task, terminated_task},
            timeout=self._config.ready_timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

        if not done:
            raise TimeoutError(
                f"vLLM did not become ready within {self._config.ready_timeout_s}s. "
                f"Check logs at {self._log_path}"
            )
        if terminated_task in done:
            raise RuntimeError(
                f"vLLM terminated during startup: {self._termination_line}\n"
                f"Check logs at {self._log_path}"
            )
        logger.info("vLLM is ready.")

    def _get_vllm_config_path(self) -> Path:
        config = self._config
        path = (
            config.config_base_dir
            / config.workload
            / config.model_id
            / config.gpu_model
            / _VLLM_CONFIG_FILENAME
        ).absolute()
        if not path.exists():
            raise FileNotFoundError(
                f"vLLM config not found: {path}\n"
                f"Expected for model={config.model_id}, gpu={config.gpu_model}, workload={config.workload}"
            )
        return path

    def _load_env_vars(self) -> dict[str, str]:
        config = self._config
        path = (
            config.config_base_dir
            / config.workload
            / config.model_id
            / config.gpu_model
            / _VLLM_ENV_FILENAME
        )
        if not path.exists():
            return {}
        with open(path) as f:
            env_vars = yaml.safe_load(f) or {}
        return {k: str(v) for k, v in env_vars.items()}

    def _build_docker_command(
        self,
        config_path: Path,
        env_vars: dict[str, str],
        hf_token: str,
        hf_home: str,
        vllm_cache_dir: str | None,
        parallel_arg: str,
    ) -> list[str]:
        config = self._config
        container_name = f"{_CONTAINER_NAME_PREFIX}{''.join(str(g) for g in config.gpu_ids)}"

        all_env_vars = {
            "HF_TOKEN": hf_token,
            "HF_HOME": hf_home,
            "VLLM_ENGINE_READY_TIMEOUT_S": str(int(config.ready_timeout_s)),
            **env_vars,
        }
        if vllm_cache_dir:
            all_env_vars["VLLM_CACHE_DIR"] = vllm_cache_dir

        env_flags: list[str] = []
        for k, v in all_env_vars.items():
            env_flags += ["-e", f"{k}={v}"]

        gpu_device_ids = ",".join(str(g) for g in config.gpu_ids)
        bind_mounts: list[str] = [
            f"{config_path}:{_CONTAINER_CONFIG_PATH}:ro",
            f"{hf_home}:{hf_home}",
        ]
        if vllm_cache_dir:
            bind_mounts.append(f"{vllm_cache_dir}:{_CONTAINER_VLLM_CACHE_DIR}")
        volume_flags: list[str] = []
        for mount in bind_mounts:
            volume_flags += ["-v", mount]

        vllm_cmd: list[str] = [
            "vllm", "serve", config.model_id,
            f"--config={_CONTAINER_CONFIG_PATH}",
            f"--port={config.port}",
            f"{parallel_arg}={len(config.gpu_ids)}",
            f"--max-num-seqs={self._max_num_seqs}",
        ]
        if self._max_num_batched_tokens is not None:
            vllm_cmd.append(f"--max-num-batched-tokens={self._max_num_batched_tokens}")

        return [
            "docker", "run",
            "--rm",
            f"--name={container_name}",
            "--gpus", f'"device={gpu_device_ids}"',
            "--ipc=host",
            "--net=host",
            "--entrypoint", "",
            *env_flags,
            *volume_flags,
            config.image,
            *vllm_cmd,
        ]

    def _validate_still_running(self) -> None:
        if self._proc is not None and self._proc.returncode is not None:
            raise RuntimeError(
                f"vLLM container exited unexpectedly with code {self._proc.returncode} "
                f"during the benchmark. Check logs at {self._log_path}"
            )

    async def _watch_output(self) -> None:
        """Drain the container's stdout, write to log file, and set readiness/termination events."""
        assert self._proc is not None and self._proc.stdout is not None

        with open(self._log_path, "w") as log_file:
            async for line_bytes in self._proc.stdout:
                line = line_bytes.decode(errors="replace")
                log_file.write(line)
                log_file.flush()

                if _STARTUP_STRING in line and not self._ready_event.is_set():
                    logger.info("vLLM startup detected: %s", line.rstrip())
                    self._ready_event.set()

                elif any(s in line.lower() for s in _TERMINATION_STRINGS):
                    logger.error("vLLM termination string detected: %s", line.rstrip())
                    self._termination_line = line.strip()
                    self._terminated_event.set()
