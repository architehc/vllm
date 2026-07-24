# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in KTransformers low-token path for HY-V3 routed experts.

Small MoE invocations can run the routed experts on the CPU while the
tensor-parallel shared expert remains on the GPUs. Larger invocations continue
to use vLLM's normal GPU MoE path. The default threshold is one token; raising
it is useful for the uncached tail of a radix-cache hit.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLE_ENV = "VLLM_HYV3_KT_DECODE"
_LAYERS_ENV = "VLLM_HYV3_KT_LAYERS"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {parsed}")
    return parsed


def _parse_index_spec(spec: str, *, upper_bound: int, name: str) -> frozenset[int]:
    """Parse comma-separated indices and inclusive ranges."""
    normalized = spec.strip().lower()
    if normalized == "all":
        return frozenset(range(upper_bound))
    if not normalized:
        raise ValueError(f"{name} cannot be empty")

    result: set[int] = set()
    for part in normalized.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid empty component in {name}={spec!r}")
        if "-" in part:
            endpoints = part.split("-", maxsplit=1)
            try:
                start, end = (int(value) for value in endpoints)
            except ValueError as exc:
                raise ValueError(f"Invalid range {part!r} in {name}") from exc
            if start > end:
                raise ValueError(f"Descending range {part!r} in {name}")
            result.update(range(start, end + 1))
        else:
            try:
                result.add(int(part))
            except ValueError as exc:
                raise ValueError(f"Invalid index {part!r} in {name}") from exc

    invalid = sorted(index for index in result if not 0 <= index < upper_bound)
    if invalid:
        raise ValueError(
            f"{name} contains out-of-range indices {invalid}; "
            f"expected values in [0, {upper_bound - 1}]"
        )
    return frozenset(result)


def _parse_cpu_set(spec: str) -> frozenset[int]:
    cpu_count = os.cpu_count() or 1
    return _parse_index_spec(spec, upper_bound=cpu_count, name="VLLM_HYV3_KT_CPUSET")


def _parse_numa_nodes(
    spec: str | None, threadpool_count: int
) -> tuple[int, ...] | None:
    if spec is None:
        return None
    try:
        nodes = tuple(int(value.strip()) for value in spec.split(","))
    except ValueError as exc:
        raise ValueError(
            "VLLM_HYV3_KT_NUMA_NODES must be comma-separated integers"
        ) from exc
    if len(nodes) != threadpool_count or any(node < 0 for node in nodes):
        raise ValueError(
            "VLLM_HYV3_KT_NUMA_NODES must contain one nonnegative node per "
            f"thread pool ({threadpool_count}), got {nodes}"
        )
    return nodes


@dataclass(frozen=True)
class HYV3KTDecodeSettings:
    enabled: bool
    selected_layers: frozenset[int]
    weight_path: str
    package_dir: str | None
    cpu_threads: int
    threadpool_count: int
    numa_nodes: tuple[int, ...] | None
    cpu_set: frozenset[int] | None
    queue_cpu: int | None
    max_tokens: int

    @classmethod
    def from_env(
        cls, vllm_config: VllmConfig, *, num_hidden_layers: int
    ) -> HYV3KTDecodeSettings:
        enabled = _env_flag(_ENABLE_ENV)
        selected_layers = _parse_index_spec(
            os.environ.get(_LAYERS_ENV, "all"),
            upper_bound=num_hidden_layers,
            name=_LAYERS_ENV,
        )
        cpu_threads = _env_int("VLLM_HYV3_KT_THREADS", min(48, os.cpu_count() or 1))
        threadpool_count = _env_int("VLLM_HYV3_KT_THREADPOOLS", 1)
        if threadpool_count > cpu_threads:
            raise ValueError(
                "VLLM_HYV3_KT_THREADPOOLS cannot exceed "
                f"VLLM_HYV3_KT_THREADS ({cpu_threads})"
            )

        cpu_set_spec = os.environ.get("VLLM_HYV3_KT_CPUSET")
        queue_cpu_spec = os.environ.get("VLLM_HYV3_KT_QUEUE_CPU")
        queue_cpu = None
        if queue_cpu_spec is not None:
            try:
                queue_cpu = int(queue_cpu_spec)
            except ValueError as exc:
                raise ValueError("VLLM_HYV3_KT_QUEUE_CPU must be an integer") from exc

        model_config = vllm_config.model_config
        assert model_config is not None
        return cls(
            enabled=enabled,
            selected_layers=selected_layers,
            weight_path=os.environ.get(
                "VLLM_HYV3_KT_WEIGHT_PATH", str(model_config.model)
            ),
            package_dir=os.environ.get("VLLM_HYV3_KT_PACKAGE_DIR"),
            cpu_threads=cpu_threads,
            threadpool_count=threadpool_count,
            numa_nodes=_parse_numa_nodes(
                os.environ.get("VLLM_HYV3_KT_NUMA_NODES"), threadpool_count
            ),
            cpu_set=_parse_cpu_set(cpu_set_spec) if cpu_set_spec else None,
            queue_cpu=queue_cpu,
            max_tokens=_env_int("VLLM_HYV3_KT_MAX_TOKENS", 1),
        )


def _load_kt_kernel(package_dir: str | None) -> ModuleType:
    """Import kt_kernel, optionally from its package source directory."""
    try:
        return importlib.import_module("kt_kernel")
    except ModuleNotFoundError as exc:
        if exc.name != "kt_kernel" or package_dir is None:
            raise RuntimeError(
                "HY-V3 KTransformers decode requires the kt_kernel package. "
                "Install it or set VLLM_HYV3_KT_PACKAGE_DIR to the directory "
                "containing kt_kernel's __init__.py."
            ) from exc

    package_path = Path(package_dir).expanduser().resolve()
    init_path = package_path / "__init__.py"
    if not init_path.is_file():
        raise ValueError(
            f"VLLM_HYV3_KT_PACKAGE_DIR must contain __init__.py, got {package_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "kt_kernel", init_path, submodule_search_locations=[str(package_path)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create an import spec for {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kt_kernel"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("kt_kernel", None)
        raise
    return module


class HYV3KTDecode:
    """Run small HY-V3 routed-expert batches through KT's NVFP4 CPU kernel."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        layer_idx: int,
        num_hidden_layers: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.settings = HYV3KTDecodeSettings.from_env(
            vllm_config, num_hidden_layers=num_hidden_layers
        )
        self.enabled = (
            self.settings.enabled and self.layer_idx in self.settings.selected_layers
        )
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self._wrapper = None
        self._init_lock = threading.Lock()

        model_config = vllm_config.model_config
        assert model_config is not None
        if self.enabled and not model_config.enforce_eager:
            raise RuntimeError(
                "HY-V3 KTransformers decode currently requires --enforce-eager"
            )
        parallel_config = vllm_config.parallel_config
        if self.enabled and (
            parallel_config.data_parallel_size != 1
            or parallel_config.pipeline_parallel_size != 1
            or parallel_config.decode_context_parallel_size != 1
            or parallel_config.enable_expert_parallel
        ):
            raise RuntimeError(
                "HY-V3 KTransformers decode currently requires DP=PP=DCP=1 "
                "with expert parallelism disabled"
            )
        if self.enabled and model_config.dtype != torch.bfloat16:
            raise RuntimeError(
                "HY-V3 KTransformers NVFP4 decode requires bfloat16 activations"
            )

    def should_use(self, hidden_states: torch.Tensor) -> bool:
        return self.enabled and 0 < hidden_states.shape[0] <= self.settings.max_tokens

    def prepare(self) -> None:
        """Load selected CPU weights before the first distributed forward."""
        if self.enabled and self.tp_rank == 0:
            self._ensure_wrapper()

    def _configure_cpu_runtime(self) -> None:
        if self.settings.cpu_set is not None:
            if not hasattr(os, "sched_setaffinity"):
                raise RuntimeError("VLLM_HYV3_KT_CPUSET requires os.sched_setaffinity")
            os.sched_setaffinity(0, self.settings.cpu_set)
        if self.settings.queue_cpu is not None:
            os.environ.setdefault("KT_TASK_QUEUE_CPU", str(self.settings.queue_cpu))
        os.environ.setdefault("KT_KERNEL_CPU_VARIANT", "avx2")

    def _ensure_wrapper(self):
        if self.tp_rank != 0:
            raise RuntimeError("Only tensor-parallel rank 0 owns the KT wrapper")
        if self._wrapper is not None:
            return self._wrapper

        with self._init_lock:
            if self._wrapper is not None:
                return self._wrapper
            self._configure_cpu_runtime()
            kt_kernel = _load_kt_kernel(self.settings.package_dir)
            capture_batch_sizes = sorted({1, self.settings.max_tokens})
            kt_kernel.KTMoEWrapper.set_capture_batch_sizes(capture_batch_sizes)
            wrapper = kt_kernel.KTMoEWrapper(
                layer_idx=self.layer_idx,
                num_experts=self.num_experts,
                num_experts_per_tok=self.top_k,
                hidden_size=self.hidden_size,
                moe_intermediate_size=self.intermediate_size,
                gpu_experts_mask=None,
                cpuinfer_threads=self.settings.cpu_threads,
                threadpool_count=self.settings.threadpool_count,
                numa_nodes=(
                    list(self.settings.numa_nodes)
                    if self.settings.numa_nodes is not None
                    else None
                ),
                weight_path=self.settings.weight_path,
                chunked_prefill_size=self.settings.max_tokens,
                method="NVFP4",
            )
            physical_to_logical = torch.arange(
                self.num_experts, dtype=torch.int64, device="cpu"
            )
            wrapper.load_weights(physical_to_logical)
            self._wrapper = wrapper
            logger.info(
                "Loaded HY-V3 KT NVFP4 low-token layer %d on TP rank 0 "
                "(%d threads, %d pools, max_tokens=%d)",
                self.layer_idx,
                self.settings.cpu_threads,
                self.settings.threadpool_count,
                self.settings.max_tokens,
            )
        return self._wrapper

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        shared_mlp: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> torch.Tensor:
        if not self.should_use(hidden_states):
            raise RuntimeError("HYV3KTDecode.forward called for an ineligible batch")
        if not hidden_states.is_cuda:
            raise RuntimeError("HY-V3 KT decode expects CUDA hidden states")

        stream = torch.cuda.current_stream(hidden_states.device)
        wrapper = None
        if self.tp_rank == 0:
            wrapper = self._ensure_wrapper()
            wrapper.submit_forward(
                hidden_states, topk_ids, topk_weights, stream.cuda_stream
            )

        shared_output = (
            shared_mlp(hidden_states)
            if shared_mlp is not None
            else torch.zeros_like(hidden_states)
        )
        routed_output = (
            wrapper.sync_forward(hidden_states, stream.cuda_stream)
            if wrapper is not None
            else torch.zeros_like(hidden_states)
        )
        output = shared_output + routed_output
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return output
