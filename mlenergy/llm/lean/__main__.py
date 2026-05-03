"""CLI entry point for the lean LLM benchmark.

Usage with TOML config::

    sudo PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
      HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" \\
      .venv/bin/python -m mlenergy.llm.lean \\
      --config configs/vllm/lm-arena-chat/Qwen/Qwen3-8B/B200/bench.toml

Usage with CLI args::

    sudo PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
      HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" \\
      .venv/bin/python -m mlenergy.llm.lean \\
      workload:lm-arena-chat \\
        --workload.base-dir run/llm \\
        --workload.model-id Qwen/Qwen3-8B \\
        --workload.num-requests 100 \\
        --workload.gpu-model B200 \\
        --workload.num-gpus 1 \\
        --workload.max-num-seqs 8

Monitor in a second terminal::

    find run/llm -name "server.log" | xargs tail -f
"""


import os
import gc
import tyro
import logging
import asyncio
import tomllib
import resource

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generic, Literal, TypeVar

from mlenergy.llm.lean.benchmark import Benchmark
from mlenergy.llm.lean.config import BenchmarkConfig
from mlenergy.llm.lean.metrics import BenchmarkResult
from mlenergy.llm.lean.workloads import (
    GPQA,
    LMArenaChat,
    LengthControl,
    LeanWorkloadMixin,
    SourcegraphFIM,
)

WorkloadT = TypeVar("WorkloadT", bound=LeanWorkloadMixin)

logger = logging.getLogger(__spec__.parent)

_WORKLOAD_TYPES: dict[str, type[LeanWorkloadMixin]] = {
    "lm-arena-chat": LMArenaChat,
    "length-control": LengthControl,
    "sourcegraph-fim": SourcegraphFIM,
    "gpqa": GPQA,
}


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class Args(Generic[WorkloadT]):
    """CLI arguments for the lean LLM benchmark."""

    # Optional TOML config file; provides workload + benchmark defaults
    config: Path | None = None

    # Workload subcommand — optional when --config provides [workload]
    workload: WorkloadT | None = None

    # Server — None means "use config file value or built-in default"
    server_image: str | None = None

    # Traffic
    request_rate: float | None = None
    burstiness: float | None = None
    max_concurrency: int | None = None
    max_output_tokens: int | Literal["dataset"] | None = None
    ignore_eos: bool | None = None

    # Sampling
    temperature: float | None = None
    top_p: float | None = None

    # Run control (not in TOML)
    monitor_cpu_power: bool = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _workload_from_toml(data: dict) -> LeanWorkloadMixin:
    data = dict(data)
    cls = _WORKLOAD_TYPES[data.pop("type")]
    return cls(**data)


def _resolve_workload(args: Args) -> LeanWorkloadMixin:
    if args.workload is not None:
        return args.workload
    if args.config is not None:
        with open(args.config, "rb") as f:
            toml_data = tomllib.load(f)
        if "workload" not in toml_data:
            raise ValueError(f"--config provided but [{args.config}] has no [workload] section")
        return _workload_from_toml(toml_data["workload"])
    raise ValueError("Must provide either a workload subcommand or --config with a [workload] section")


def _validate_gpus(workload: LeanWorkloadMixin) -> list[int]:
    gpu_ids = [int(g) for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    if len(gpu_ids) != workload.num_gpus:
        raise ValueError(
            f"workload.num_gpus={workload.num_gpus} does not match "
            f"CUDA_VISIBLE_DEVICES ({len(gpu_ids)} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']})"
        )
    return gpu_ids


def _setup_logging(workload: LeanWorkloadMixin, args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(workload.to_path(of="driver_log"), mode="w"),
        ],
    )
    logger.info("%s", args)


async def _run_benchmark(config: BenchmarkConfig, workload: LeanWorkloadMixin, requests, monitor_cpu_power: bool) -> BenchmarkResult:
    async with Benchmark(
        config=config,
        output_dir=workload.run_dir,
        max_num_seqs=workload.max_num_seqs,
        tokenizer=workload.tokenizer,
        max_num_batched_tokens=workload.max_num_batched_tokens,
        monitor_cpu_power=monitor_cpu_power,
    ) as bm:
        return await bm.run(requests)


def main(args: Args) -> None:
    workload = _resolve_workload(args)
    gpu_ids = _validate_gpus(workload)

    workload.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    _setup_logging(workload, args)

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))

    requests = workload.load_requests()
    config = BenchmarkConfig.from_args(args, workload, gpu_ids)

    gc.collect()
    gc.freeze()

    result = asyncio.run(_run_benchmark(config, workload, requests, args.monitor_cpu_power))

    result.save(workload.run_dir, workload.base_dir / workload.normalized_name, config, workload, workload.run_id)


if __name__ == "__main__":
    args = tyro.cli(Args[LMArenaChat | LengthControl | SourcegraphFIM | GPQA])
    try:
        main(args)
    except Exception as e:
        logger.exception("Benchmark failed: %s", e)
        raise
