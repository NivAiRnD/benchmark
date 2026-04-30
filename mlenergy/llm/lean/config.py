"""Benchmark configuration dataclasses.

All configs are frozen — set once at construction, never mutated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from mlenergy.llm.lean.constants import (
    _CONFIG_BASE_DIR,
    _DEFAULT_READY_TIMEOUT_S,
    _DEFAULT_VLLM_IMAGE,
)

if TYPE_CHECKING:
    from mlenergy.llm.lean.__main__ import Args
    from mlenergy.llm.workloads import WorkloadConfig

logger = logging.getLogger("mlenergy.llm.lean")

OutputTokens: TypeAlias = int | Literal["dataset"] | None


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for the vLLM server container.

    Attributes:
        model_id: HuggingFace model identifier, e.g. "meta-llama/Llama-3.1-8B-Instruct".
        gpu_model: GPU model name used to resolve config files, e.g. "H100", "B200".
        workload: Workload name used to resolve config files, e.g. "lm-arena-chat".
        gpu_ids: Physical GPU indices to pass to the container.
        image: Docker image to run, e.g. "vllm/vllm-openai:v0.11.1".
        port: Host port the vLLM server will listen on.
        ready_timeout_s: Seconds to wait for the server to become healthy before aborting.
        config_base_dir: Root directory that holds per-model vLLM YAML configs.
    """

    model_id: str
    gpu_model: str
    workload: str
    gpu_ids: list[int]
    image: str = _DEFAULT_VLLM_IMAGE
    port: int = 8000
    ready_timeout_s: float = _DEFAULT_READY_TIMEOUT_S
    config_base_dir: Path = _CONFIG_BASE_DIR

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def log_dir(self, output_root: Path) -> Path:
        return output_root / self.workload / self.model_id / self.gpu_model


@dataclass(frozen=True)
class TrafficConfig:
    """Configuration for request traffic shaping.

    Attributes:
        request_rate: Requests per second. Use float("inf") to send all at once.
        burstiness: Shape of inter-arrival distribution. 1.0 = Poisson, >1 = uniform, <1 = bursty.
        max_concurrency: Cap on in-flight requests. None means unlimited.
        max_output_tokens: Output length cap per request. None = no cap, "dataset" = use dataset value.
        ignore_eos: When True, requests generate tokens until max_output_tokens regardless of EOS.
    """

    request_rate: float = float("inf")
    burstiness: float = 1.0
    max_concurrency: int | None = None
    max_output_tokens: OutputTokens = None
    ignore_eos: bool = False


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters forwarded to vLLM on every request.

    Attributes:
        temperature: Sampling temperature. 0.0 = greedy decoding.
        top_p: Nucleus sampling probability threshold.
        extra_body: Additional fields merged into the request JSON body.
        system_prompt: Optional system message prepended to every conversation.
    """

    temperature: float = 0.8
    top_p: float = 0.95
    extra_body: dict = field(default_factory=dict)
    system_prompt: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration.

    Attributes:
        server: vLLM server and container settings.
        traffic: Request rate and concurrency settings.
        sampling: Sampling parameters sent with each request.
        seed: Random seed for reproducible request ordering and sampling.
    """

    server: ServerConfig
    traffic: TrafficConfig
    sampling: SamplingConfig
    seed: int = 42

    @classmethod
    def from_toml(cls, path: Path, workload: WorkloadConfig, gpu_ids: list[int]) -> BenchmarkConfig:
        import tomllib

        from mlenergy.llm.config import load_extra_body, load_system_prompt

        with open(path, "rb") as f:
            data = tomllib.load(f)

        s = data.get("server", {})
        t = data.get("traffic", {})
        p = data.get("sampling", {})
        extra_body = load_extra_body(
            model_id=workload.model_id,
            gpu_model=workload.gpu_model,
            workload=workload.normalized_name,
        )
        system_prompt = load_system_prompt(
            model_id=workload.model_id,
            gpu_model=workload.gpu_model,
            workload=workload.normalized_name,
        )
        return cls(
            server=ServerConfig(
                model_id=workload.model_id,
                gpu_model=workload.gpu_model,
                workload=workload.normalized_name,
                gpu_ids=gpu_ids,
                **{k: s[k] for k in ("image", "port", "ready_timeout_s") if k in s},
            ),
            traffic=TrafficConfig(**t),
            sampling=SamplingConfig(
                extra_body=extra_body,
                system_prompt=system_prompt,
                **{k: p[k] for k in ("temperature", "top_p") if k in p},
            ),
            seed=data.get("seed", workload.seed),
        )

    @classmethod
    def from_args(cls, args: Args, workload: WorkloadConfig, gpu_ids: list[int]) -> BenchmarkConfig:
        from mlenergy.llm.config import load_extra_body, load_system_prompt

        base = cls.from_toml(args.config, workload, gpu_ids) if args.config else None

        def pick(cli_val, base_val):
            return cli_val if cli_val is not None else base_val

        bt = base.traffic if base else TrafficConfig()
        bs = base.sampling if base else SamplingConfig()
        extra_body = base.sampling.extra_body if base else load_extra_body(
            model_id=workload.model_id,
            gpu_model=workload.gpu_model,
            workload=workload.normalized_name,
        )
        system_prompt = base.sampling.system_prompt if base else load_system_prompt(
            model_id=workload.model_id,
            gpu_model=workload.gpu_model,
            workload=workload.normalized_name,
        )
        return cls(
            server=ServerConfig(
                model_id=workload.model_id,
                gpu_model=workload.gpu_model,
                workload=workload.normalized_name,
                gpu_ids=gpu_ids,
                image=pick(args.server_image, base.server.image if base else _DEFAULT_VLLM_IMAGE),
            ),
            traffic=TrafficConfig(
                request_rate=pick(args.request_rate, bt.request_rate),
                burstiness=pick(args.burstiness, bt.burstiness),
                max_concurrency=pick(args.max_concurrency, bt.max_concurrency),
                max_output_tokens=pick(args.max_output_tokens, bt.max_output_tokens),
                ignore_eos=pick(args.ignore_eos, bt.ignore_eos),
            ),
            sampling=SamplingConfig(
                temperature=pick(args.temperature, bs.temperature),
                top_p=pick(args.top_p, bs.top_p),
                extra_body=extra_body,
                system_prompt=system_prompt,
            ),
            seed=base.seed if base else workload.seed,
        )
