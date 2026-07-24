# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TurboQuant Q4-NC decode at the per-rank Hy3 shape.

The physical cache is deliberately larger than L2 and timed calls rotate
through disjoint windows. This avoids reporting an unrealistically L2-hot 32K
cache while keeping the default allocation below 300 MiB.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass

import torch

from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)


@dataclass(frozen=True)
class KernelConfig:
    splits: int
    block_kv: int
    num_warps: int


CONFIGS = (
    KernelConfig(32, 4, 1),
    KernelConfig(32, 4, 2),
    KernelConfig(32, 4, 4),
    KernelConfig(32, 8, 4),
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    pos = fraction * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _fill_valid_q4_nc_metadata(cache: torch.Tensor) -> None:
    """Replace random metadata bytes with finite fp16 norm/scale/zero values."""
    # D=128 Q4-NC: key indices [0:64], norm [64:66], values [66:130],
    # scale [130:132], zero [132:134]. fp16 bit patterns are little-endian.
    cache[..., 64].zero_()
    cache[..., 65].fill_(0x3C)  # norm = 1.0 (0x3c00)
    cache[..., 130].zero_()
    cache[..., 131].fill_(0x30)  # scale = 0.125 (0x3000)
    cache[..., 132].zero_()
    cache[..., 133].fill_(0xBC)  # zero = -1.0 (0xbc00)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[32768, 262144])
    parser.add_argument("--physical-tokens", type=int, default=524288)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--max-extra-gib", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    torch.manual_seed(20260723)

    # GLM/Hy3 shape on one TP rank.
    batch = 1
    num_query_heads = 32
    num_kv_heads = 4
    head_dim = 128
    block_size = 16
    cfg = TurboQuantConfig.from_cache_dtype("turboquant_4bit_nc", head_dim)

    if args.physical_tokens % block_size:
        raise SystemExit("--physical-tokens must be divisible by block size 16")
    if any(seq <= 0 or seq % block_size for seq in args.seq_lens):
        raise SystemExit("all --seq-lens must be positive and divisible by 16")
    if max(args.seq_lens) > args.physical_tokens:
        raise SystemExit("--physical-tokens must cover the longest sequence")

    num_blocks = args.physical_tokens // block_size
    cache_bytes = args.physical_tokens * num_kv_heads * cfg.slot_size_aligned
    scratch_bytes = batch * num_query_heads * 32 * (head_dim + 1) * 4
    estimated_bytes = cache_bytes + scratch_bytes
    budget_bytes = int(args.max_extra_gib * 1024**3)
    if estimated_bytes > budget_bytes:
        raise SystemExit(
            f"planned tensor allocation {estimated_bytes / 1024**2:.1f} MiB "
            f"exceeds budget {budget_bytes / 1024**2:.1f} MiB"
        )

    free_before, total_memory = torch.cuda.mem_get_info(device)
    # Leave at least 512 MiB globally free after planned tensor allocations.
    safety_margin = 512 * 1024**2
    if free_before < estimated_bytes + safety_margin:
        raise SystemExit(
            f"only {free_before / 1024**2:.1f} MiB free; need "
            f"{(estimated_bytes + safety_margin) / 1024**2:.1f} MiB"
        )

    query = torch.randn(
        batch,
        num_query_heads,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    pi = torch.eye(head_dim, dtype=torch.float32, device=device)
    centroids = torch.linspace(-1.5, 1.5, 16, dtype=torch.float32, device=device)
    cache = torch.randint(
        0,
        256,
        (num_blocks, block_size, num_kv_heads, cfg.slot_size_aligned),
        dtype=torch.uint8,
        device=device,
    )
    _fill_valid_q4_nc_metadata(cache)

    mid_o = torch.empty(
        batch,
        num_query_heads,
        32,
        head_dim + 1,
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty_like(query)
    lse = torch.empty(batch, num_query_heads, dtype=torch.float32, device=device)

    results: list[dict[str, object]] = []
    reference_outputs: dict[int, torch.Tensor] = {}

    for seq_len in args.seq_lens:
        pages = seq_len // block_size
        window_count = args.physical_tokens // seq_len
        block_tables = torch.arange(
            window_count * pages, dtype=torch.int32, device=device
        ).reshape(window_count, pages)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

        def launch(
            kernel_cfg: KernelConfig,
            window: int,
            block_tables_for_seq: torch.Tensor = block_tables,
            seq_lens_for_seq: torch.Tensor = seq_lens,
        ) -> torch.Tensor:
            return triton_turboquant_decode_attention(
                query=query,
                kv_cache=cache,
                block_table=block_tables_for_seq[window : window + 1],
                seq_lens=seq_lens_for_seq,
                Pi=pi,
                centroids=centroids,
                scale=1.0 / math.sqrt(head_dim),
                mse_bits=cfg.key_mse_bits,
                key_packed_size=cfg.key_packed_size,
                value_quant_bits=cfg.effective_value_quant_bits,
                key_fp8=cfg.key_fp8,
                norm_correction=cfg.norm_correction,
                PiT=pi,
                mid_o_buf=mid_o,
                output_buf=output,
                lse_buf=lse,
                max_num_kv_splits=kernel_cfg.splits,
                stage1_block_kv=kernel_cfg.block_kv,
                stage1_num_warps=kernel_cfg.num_warps,
            )

        for kernel_cfg in CONFIGS:
            for warmup_idx in range(args.warmup):
                launch(kernel_cfg, warmup_idx % window_count)
            torch.cuda.synchronize(device)

            if kernel_cfg == CONFIGS[0]:
                reference_outputs[seq_len] = launch(kernel_cfg, 0).clone()
            else:
                candidate = launch(kernel_cfg, 0).clone()
                torch.testing.assert_close(
                    candidate,
                    reference_outputs[seq_len],
                    rtol=5e-3,
                    atol=5e-3,
                )
            torch.cuda.synchronize(device)

            samples_ms: list[float] = []
            for trial in range(args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                launch(kernel_cfg, trial % window_count)
                end.record()
                end.synchronize()
                samples_ms.append(start.elapsed_time(end))

            median_ms = statistics.median(samples_ms)
            results.append(
                {
                    "seq_len": seq_len,
                    **asdict(kernel_cfg),
                    "median_ms": median_ms,
                    "iqr_ms": _percentile(samples_ms, 0.75)
                    - _percentile(samples_ms, 0.25),
                    "min_ms": min(samples_ms),
                    "max_ms": max(samples_ms),
                    "samples_ms": samples_ms,
                    "unique_cache_gbps": (seq_len * num_kv_heads * cfg.slot_size)
                    / (median_ms * 1e6),
                }
            )

    torch.cuda.synchronize(device)
    free_after, _ = torch.cuda.mem_get_info(device)
    report = {
        "shape": {
            "batch": batch,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "preset": "turboquant_4bit_nc",
            "slot_bytes": cfg.slot_size_aligned,
        },
        "cache": {
            "physical_tokens": args.physical_tokens,
            "bytes": cache_bytes,
            "mib": cache_bytes / 1024**2,
        },
        "memory": {
            "global_free_before_mib": free_before / 1024**2,
            "global_free_after_mib": free_after / 1024**2,
            "device_total_mib": total_memory / 1024**2,
            "torch_max_allocated_mib": torch.cuda.max_memory_allocated(device)
            / 1024**2,
            "torch_max_reserved_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
