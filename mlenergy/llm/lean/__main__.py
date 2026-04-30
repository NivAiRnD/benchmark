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
import logging
import os
import resource
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import tyro

from mlenergy.llm.lean.benchmark import Benchmark
from mlenergy.llm.lean.config import BenchmarkConfig
from mlenergy.llm.lean.metrics import BenchmarkResult
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
    server_image: str = "vllm/vllm-openai:latest"

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
# Main
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    assert isinstance(args.workload, WorkloadConfig)
    workload = args.workload

    gpu_ids = [int(g) for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    if len(gpu_ids) != workload.num_gpus:
        raise ValueError(
            f"--workload.num-gpus={workload.num_gpus} does not match "
            f"CUDA_VISIBLE_DEVICES ({len(gpu_ids)} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']})"
        )

    output_dir = workload.base_dir / "lean" / workload.normalized_name / workload.model_id / workload.gpu_model
    result_file = output_dir / "results.json"

    if result_file.exists() and not args.overwrite_results:
        print(
            f"Result file {result_file} already exists. Exiting. "
            "Pass --overwrite-results to rerun."
        )
        raise SystemExit(0)

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "driver.log", mode="w"),
        ],
    )
    logger.info("%s", args)

    # Raise open-file limit so many concurrent connections don't hit EMFILE
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65535, hard), hard))

    requests = workload.load_requests()
    config = BenchmarkConfig.from_args(args, workload, gpu_ids)

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

    result.save(result_file, args, workload)


if __name__ == "__main__":
    args = tyro.cli(Args[LMArenaChat | LengthControl | SourcegraphFIM | GPQA])
    try:
        main(args)
    except Exception as e:
        logger.exception("Benchmark failed: %s", e)
        raise
