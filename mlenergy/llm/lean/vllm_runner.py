"""vLLM-specific Docker container configuration."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from mlenergy.llm.lean.config import ServerConfig
from mlenergy.llm.lean.constants import VLLMContainerDefaults
from mlenergy.llm.lean.docker_runner import ContainerConfig, DockerRunner


class VLLMRunner(DockerRunner):
    """DockerRunner subclass for vLLM.

    Implements _container_config() to resolve vLLM config files and
    build the vllm serve command.

    Usage::

        async with VLLMRunner(server_config, log_path, max_num_seqs) as runner:
            await runner.wait_ready()
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
        super().__init__(log_path)
        self._server_config = config
        self._max_num_seqs = max_num_seqs
        self._max_num_batched_tokens = max_num_batched_tokens

    def _container_config(self) -> ContainerConfig:
        hf_token = os.environ["HF_TOKEN"]
        hf_home = os.environ["HF_HOME"]
        vllm_cache_dir = os.environ.get("VLLM_CACHE_DIR")

        config = self._server_config
        vllm_config_path = self._get_vllm_config_path()
        extra_env = self._load_env_vars()

        config_text = vllm_config_path.read_text()
        parallel_arg = VLLMContainerDefaults.DATA_PARALLEL_ARG if VLLMContainerDefaults.DATA_PARALLEL_ARG.lstrip("-") in config_text else VLLMContainerDefaults.TENSOR_PARALLEL_ARG

        env_vars = {
            "HF_TOKEN": hf_token,
            "HF_HOME": hf_home,
            "VLLM_ENGINE_READY_TIMEOUT_S": str(int(config.ready_timeout_s)),
            **extra_env,
        }
        if vllm_cache_dir:
            env_vars["VLLM_CACHE_DIR"] = vllm_cache_dir

        volumes = [
            f"{vllm_config_path}:{VLLMContainerDefaults.CONFIG_PATH}:ro",
            f"{hf_home}:{hf_home}",
        ]
        if vllm_cache_dir:
            volumes.append(f"{vllm_cache_dir}:{VLLMContainerDefaults.CACHE_DIR}")

        entrypoint_cmd = [
            "vllm", "serve", config.model_id,
            f"--config={VLLMContainerDefaults.CONFIG_PATH}",
            f"--port={config.port}",
            f"{parallel_arg}={len(config.gpu_ids)}",
            f"--max-num-seqs={self._max_num_seqs}",
        ]
        if self._max_num_batched_tokens is not None:
            entrypoint_cmd.append(f"--max-num-batched-tokens={self._max_num_batched_tokens}")

        return ContainerConfig(
            image=config.image,
            name=f"{VLLMContainerDefaults.NAME_PREFIX}{''.join(str(g) for g in config.gpu_ids)}",
            gpu_device_ids=",".join(str(g) for g in config.gpu_ids),
            env_vars=env_vars,
            volumes=volumes,
            entrypoint_cmd=entrypoint_cmd,
            startup_string=VLLMContainerDefaults.STARTUP_STRING,
            termination_strings=VLLMContainerDefaults.TERMINATION_STRINGS,
            ready_timeout_s=config.ready_timeout_s,
        )

    def _get_vllm_config_path(self) -> Path:
        config = self._server_config
        path = (
            config.config_base_dir
            / config.workload
            / config.model_id
            / config.gpu_model
            / VLLMContainerDefaults.CONFIG_FILENAME
        ).absolute()
        if not path.exists():
            raise FileNotFoundError(
                f"vLLM config not found: {path}\n"
                f"Expected for model={config.model_id}, gpu={config.gpu_model}, workload={config.workload}"
            )
        return path

    def _load_env_vars(self) -> dict[str, str]:
        config = self._server_config
        path = (
            config.config_base_dir
            / config.workload
            / config.model_id
            / config.gpu_model
            / VLLMContainerDefaults.ENV_FILENAME
        )
        if not path.exists():
            return {}
        with open(path) as f:
            env_vars = yaml.safe_load(f) or {}
        return {k: str(v) for k, v in env_vars.items()}
