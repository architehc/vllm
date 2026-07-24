# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.hy_v3_kt_decode import (
    HYV3KTDecode,
    _parse_index_spec,
)
from vllm.model_executor.models.hy_v3_mtp import HYV3MultiTokenPredictor
from vllm.model_executor.offloader.prefetch import (
    PrefetchOffloader,
    _bypasses_expert_prefetch,
)


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("all", frozenset(range(8))),
        ("1", frozenset({1})),
        ("1,3-5,7", frozenset({1, 3, 4, 5, 7})),
    ],
)
def test_parse_kt_layer_spec(spec: str, expected: frozenset[int]):
    assert _parse_index_spec(spec, upper_bound=8, name="TEST_LAYERS") == expected


@pytest.mark.parametrize("spec", ["", "3-1", "1,,2", "8", "nope"])
def test_reject_invalid_kt_layer_spec(spec: str):
    with pytest.raises(ValueError):
        _parse_index_spec(spec, upper_bound=8, name="TEST_LAYERS")


class _DecodeOnlyModule(nn.Module):
    def __init__(self, max_tokens: int = 1):
        super().__init__()
        self.max_tokens = max_tokens

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        del positions, residual
        return hidden_states

    def should_bypass_offload_prefetch(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> bool:
        del positions, residual
        return hidden_states.shape[0] <= self.max_tokens


def test_kt_low_token_threshold():
    adapter = object.__new__(HYV3KTDecode)
    adapter.enabled = True
    adapter.settings = SimpleNamespace(max_tokens=16)

    assert adapter.should_use(torch.zeros(1, 16))
    assert adapter.should_use(torch.zeros(16, 16))
    assert not adapter.should_use(torch.zeros(17, 16))

    adapter.enabled = False
    assert not adapter.should_use(torch.zeros(1, 16))


class _PrepareCounter:
    def __init__(self):
        self.calls = 0

    def prepare(self):
        self.calls += 1


def test_mtp_prepares_kt_decode_before_drafting():
    kt_decode = _PrepareCounter()
    predictor = object.__new__(HYV3MultiTokenPredictor)
    predictor.layers = {
        "80": SimpleNamespace(
            mtp_block=SimpleNamespace(
                block_type="moe",
                mlp=SimpleNamespace(kt_decode=kt_decode),
            )
        ),
        "81": SimpleNamespace(mtp_block=SimpleNamespace(block_type="feedforward")),
    }

    predictor.prepare_kt_decode()

    assert kt_decode.calls == 1


def test_prefetch_bypass_is_decode_and_expert_only():
    module = _DecodeOnlyModule(max_tokens=16)
    positions = torch.zeros(1, dtype=torch.int64)
    decode_hidden = torch.zeros(1, 16)
    cached_tail_hidden = torch.zeros(16, 16)
    prefill_hidden = torch.zeros(17, 16)

    assert _bypasses_expert_prefetch(
        {"experts"}, module, positions, decode_hidden, None
    )
    assert _bypasses_expert_prefetch(
        {"experts"}, module, positions, cached_tail_hidden, None
    )
    assert not _bypasses_expert_prefetch(
        {"experts"}, module, positions, prefill_hidden, None
    )
    assert not _bypasses_expert_prefetch(set(), module, positions, decode_hidden, None)
    assert not _bypasses_expert_prefetch(
        {"experts", "self_attn"}, module, positions, decode_hidden, None
    )


def test_prefill_reprimes_a_copy_skipped_by_decode(monkeypatch: pytest.MonkeyPatch):
    modules = [_DecodeOnlyModule(), _DecodeOnlyModule()]
    entries = [
        SimpleNamespace(module=module, _prefetch_required=False) for module in modules
    ]
    offloader = object.__new__(PrefetchOffloader)
    offloader.offload_params = {"experts"}
    offloader.prefetch_step = 1
    offloader.module_offloaders = entries

    calls: list[tuple[str, int]] = []

    def start_prefetch(_tensor: torch.Tensor, index: int):
        calls.append(("start", index))
        entries[index]._prefetch_required = False

    def wait_prefetch(_tensor: torch.Tensor, index: int):
        calls.append(("wait", index))

    monkeypatch.setattr(torch.ops.vllm, "start_prefetch", start_prefetch)
    monkeypatch.setattr(torch.ops.vllm, "wait_prefetch", wait_prefetch)

    offloader._hook_module_forward(0, modules[0])
    offloader._hook_module_forward(1, modules[1])

    modules[0](torch.zeros(1), torch.zeros(1, 16), None)
    assert calls == []
    assert entries[1]._prefetch_required

    modules[1](torch.zeros(2), torch.zeros(2, 16), None)
    assert calls == [("start", 1), ("wait", 1), ("start", 0)]
    assert not entries[1]._prefetch_required
