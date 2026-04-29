"""CLI entry point for the lean LLM benchmark.

Usage::

    sudo PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
      HF_TOKEN="$HF_TOKEN" HF_HOME="$HF_HOME" \\
      .venv/bin/python -m mlenergy.llm.lean \\
      --server-image vllm/vllm-openai:latest \\
      --max-output-tokens 128 \\
      workload:lm-arena-chat \\
        --workload.base-dir run/llm \\
        --workload.model-id Qwen/Qwen3-8B \\
        --workload.num-requests 100 \\
        --workload.gpu-model B200 \\
        --workload.num-gpus 1 \\
        --workload.max-num-seqs 8

Monitor in a second terminal::

    find run/llm/lean -name "server.log" | xargs tail -f
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import resource
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generic, Literal, TypeVar

import tyro

from mlenergy.llm.config import load_extra_body, load_system_prompt
from mlenergy.llm.lean.benchmark import Benchmark
from mlenergy.llm.lean.metrics import BenchmarkResult
from mlenergy.llm.lean.config import (
    BenchmarkConfig,
    SamplingConfig,
    ServerConfig,
    TrafficConfig,
)
from mlenergy.llm.workloads import (
    GPQA,
    LengthControl,
    LMArenaChat,
    SourcegraphFIM,
    WorkloadConfig,
)

WorkloadT = TypeVar("WorkloadT", bound=WorkloadConfig)

logger = logging.getLogger("mlenergy.llm.lean")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


@dataclass
class Args(Generic[WorkloadT]):
    """CLI arguments for the lean LLM benchmark."""

    workload: WorkloadT

    # Server
    server_image: str = "vllm/vllm-openai:v0.11.1"

    # Traffic
    request_rate: float = float("inf")
    burstiness: float = 1.0
    max_concurrency: int | None = None
    max_output_tokens: int | Literal["dataset"] | None = None
    ignore_eos: bool = False

    # Sampling
    temperature: float = 0.8
    top_p: float = 0.95

    # Run control
    overwrite_results: bool = False
    monitor_cpu_power: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_ids_from_env() -> list[int]:
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    return [int(g) for g in cuda_visible.split(",")]


def _output_dir(workload: WorkloadConfig) -> Path:
    return workload.base_dir / "lean" / workload.normalized_name / workload.model_id / workload.gpu_model


def _setup_logging(output_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "driver.log", mode="w"),
        ],
    )


def _build_config(args: Args, workload: WorkloadConfig, gpu_ids: list[int]) -> BenchmarkConfig:
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
    return BenchmarkConfig(
        server=ServerConfig(
            model_id=workload.model_id,
            gpu_model=workload.gpu_model,
            workload=workload.normalized_name,
            gpu_ids=gpu_ids,
            image=args.server_image,
        ),
        traffic=TrafficConfig(
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            max_concurrency=args.max_concurrency,
            max_output_tokens=args.max_output_tokens,
            ignore_eos=args.ignore_eos,
        ),
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            extra_body=extra_body,
            system_prompt=system_prompt,
        ),
        seed=workload.seed,
    )


def _save_result(result_file: Path, args: Args, workload: WorkloadConfig, result) -> None:
    data = {
        "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "model_id": workload.model_id,
        "gpu_model": workload.gpu_model,
        "num_gpus": workload.num_gpus,
        "num_requests": workload.num_requests,
        "num_request_repeats": workload.num_request_repeats,
        "seed": workload.seed,
        "max_num_seqs": workload.max_num_seqs,
        "max_num_batched_tokens": workload.max_num_batched_tokens,
        "request_rate": args.request_rate if args.request_rate < float("inf") else "inf",
        "burstiness": args.burstiness,
        "max_concurrency": args.max_concurrency,
        "max_output_tokens": args.max_output_tokens,
        "metrics": {label: entry.value for label, entry in result.metrics.entries.items()},
        "energy_j": result.energy_j,
        "energy_per_token_j": result.energy_per_token_j,
        "prometheus_stats": result.prometheus_stats,
        "power_trace": result.power_trace,
        "cpu_power_trace": result.cpu_power_trace,
        "per_request": [
            {
                "ttft": o.ttft,
                "latency": o.latency,
                "output_tokens": o.output_tokens,
                "success": o.success,
            }
            for o in result.per_request
        ],
    }
    with open(result_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved results to %s", result_file)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    assert isinstance(args.workload, WorkloadConfig)
    workload = args.workload

    gpu_ids = _gpu_ids_from_env()
    if len(gpu_ids) != workload.num_gpus:
        raise ValueError(
            f"--workload.num-gpus={workload.num_gpus} does not match "
            f"CUDA_VISIBLE_DEVICES ({len(gpu_ids)} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']})"
        )

    output_dir = _output_dir(workload)
    result_file = output_dir / "results.json"

    if result_file.exists() and not args.overwrite_results:
        print(
            f"Result file {result_file} already exists. Exiting. "
            "Pass --overwrite-results to rerun."
        )
        raise SystemExit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)
    logger.info("%s", args)

    # Raise open-file limit so many concurrent connections don't hit EMFILE
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))

    requests = workload.load_requests()
    config = _build_config(args, workload, gpu_ids)

    gc.collect()
    gc.freeze()

    async def _run() -> BenchmarkResult:
        async with Benchmark(
            config=config,
            output_dir=output_dir,
            max_num_seqs=workload.max_num_seqs,
            tokenizer=workload.tokenizer,
            max_num_batched_tokens=workload.max_num_batched_tokens,
            monitor_cpu_power=args.monitor_cpu_power,
        ) as bm:
            return await bm.run(requests)

    result = asyncio.run(_run())

    _save_result(result_file, args, workload, result)


if __name__ == "__main__":
    args = tyro.cli(Args[LMArenaChat | LengthControl | SourcegraphFIM | GPQA])
    try:
        main(args)
    except Exception as e:
        logger.exception("Benchmark failed: %s", e)
        raise
