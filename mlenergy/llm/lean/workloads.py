"""Lean-benchmark workload configurations.

Subclasses the shared WorkloadConfig types but overrides to_path() so that
all lean data — requests, tokenization, and run artifacts — lives under a
single self-contained base directory.

Set workload.run_id (a timestamp string) before calling to_path() for any
run-level path. Requests and tokenization paths are inherited unchanged.

Typical base_dir: run/llm/lean  →  everything under run/llm/lean/lm-arena-chat/
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import tyro

from mlenergy.llm.workloads import (
    GPQA as _GPQA,
    LMArenaChat as _LMArenaChat,
    LengthControl as _LengthControl,
    SourcegraphFIM as _SourcegraphFIM,
    WorkloadConfig,
)


class LeanWorkloadMixin(WorkloadConfig):
    """Path-override mixin for lean benchmark workloads.

    run_id is set at runtime (not a CLI arg) and determines the run directory.
    """

    run_id: Annotated[str, tyro.conf.Suppress] = ""

    @property
    def run_dir(self) -> Path:
        assert self.run_id, "set workload.run_id before accessing run_dir"
        return self.base_dir / self.normalized_name / self.model_id / self.gpu_model / self.run_id

    def to_path(
        self,
        of: Literal[
            "requests",
            "tokenization",
            "multimodal_dump",
            "results",
            "driver_log",
            "server_log",
            "prometheus",
            "config",
            "command",
            "git_diff",
        ],
        create_dirs: bool = True,
    ) -> Path:
        # Requests and tokenization are shared/cached — delegate to base.
        if of in ("requests", "tokenization", "multimodal_dump"):
            return super().to_path(of=of, create_dirs=create_dirs)  # type: ignore[arg-type]

        assert self.run_id, "set workload.run_id before calling to_path() for run paths"

        run_dir = self.run_dir

        match of:
            case "results":    
                path = run_dir / "results.json"
            case "driver_log": 
                path = run_dir / "driver.log"
            case "server_log": 
                path = run_dir / "server.log"
            case "prometheus": 
                path = run_dir / "prometheus.json"
            case "config":     
                path = run_dir / "config.json"
            case "command":    
                path = run_dir / "command.txt"
            case "git_diff":   
                path = run_dir / "git.diff"
            case _: 
                raise ValueError(f"Unknown path type: {of}")

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path


class LMArenaChat(LeanWorkloadMixin, _LMArenaChat): 
    pass
class LengthControl(LeanWorkloadMixin, _LengthControl): 
    pass
class SourcegraphFIM(LeanWorkloadMixin, _SourcegraphFIM): 
    pass
class GPQA(LeanWorkloadMixin, _GPQA): 
    pass
