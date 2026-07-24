# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.offloader.prefetch import (
    ParamInfo,
    _CpuParamOffloader,
    _ModuleOffloader,
    _PinnedStagingPool,
)


class _FakeEvent:
    def __init__(self, log: list[str], name: str):
        self.log = log
        self.name = name

    def record(self, _stream) -> None:
        self.log.append(f"record:{self.name}")

    def synchronize(self) -> None:
        self.log.append(f"sync:{self.name}")


class _FakeTensor:
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        stride: tuple[int, ...],
        dtype: torch.dtype,
        *,
        pinned: bool,
        log: list[str] | None = None,
    ):
        self.name = name
        self.shape = shape
        self._stride = stride
        self.dtype = dtype
        self._pinned = pinned
        self.log = log

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def is_pinned(self) -> bool:
        return self._pinned

    def copy_(self, source: "_FakeTensor", non_blocking: bool = False):
        if self.log is not None:
            self.log.append(f"copy:{source.name}->{self.name}:{non_blocking}")
        return self


def test_pinned_staging_pool_allocates_once_per_key_and_slot(monkeypatch):
    allocations: list[_FakeTensor] = []
    events: list[_FakeEvent] = []

    def empty_strided(*, size, stride, dtype, device, pin_memory):
        assert device == "cpu"
        assert pin_memory
        tensor = _FakeTensor(
            f"allocation-{len(allocations)}",
            tuple(size),
            tuple(stride),
            dtype,
            pinned=True,
        )
        allocations.append(tensor)
        return tensor

    def event_factory():
        event = _FakeEvent([], f"slot-{len(events)}")
        events.append(event)
        return event

    monkeypatch.setattr(torch, "empty_strided", empty_strided)
    monkeypatch.setattr(torch.cuda, "Event", event_factory)

    weight = ParamInfo("weight", (2, 2), (2, 1), torch.float16)
    duplicate_weight = ParamInfo("weight", (2, 2), (2, 1), torch.float16)
    scale = ParamInfo("scale", (3,), (1,), torch.float32)
    pool = _PinnedStagingPool([weight, duplicate_weight, scale], slot_capacity=2)

    assert len(allocations) == 4
    assert len(events) == 2
    assert pool.total_bytes == 40

    weight_slot_0 = pool.get_buffer(
        weight.name, weight.shape, weight.stride, weight.dtype, 0
    )
    weight_slot_1 = pool.get_buffer(
        weight.name, weight.shape, weight.stride, weight.dtype, 1
    )
    assert weight_slot_0 is pool.get_buffer(
        weight.name, weight.shape, weight.stride, weight.dtype, 2
    )
    assert weight_slot_0 is not weight_slot_1


def test_pinned_staging_pool_waits_before_slot_reuse(monkeypatch):
    log: list[str] = []
    event_count = 0

    def empty_strided(*, size, stride, dtype, device, pin_memory):
        return _FakeTensor(
            "staging", tuple(size), tuple(stride), dtype, pinned=pin_memory
        )

    def event_factory():
        nonlocal event_count
        event = _FakeEvent(log, f"slot-{event_count}")
        event_count += 1
        return event

    monkeypatch.setattr(torch, "empty_strided", empty_strided)
    monkeypatch.setattr(torch.cuda, "Event", event_factory)

    info = ParamInfo("weight", (2,), (1,), torch.float16)
    pool = _PinnedStagingPool([info], slot_capacity=2)
    stream = object()

    pool.wait_until_available(0)
    pool.record_h2d(0, stream)
    pool.wait_until_available(1)
    pool.wait_until_available(0)
    pool.record_h2d(0, stream)

    assert log == ["record:slot-0", "sync:slot-0", "record:slot-0"]


def test_pinned_staging_keeps_canonical_storage_pageable(monkeypatch):
    module = torch.nn.Linear(2, 2, bias=False)
    expected = module.weight.detach().clone()

    monkeypatch.setattr(
        "vllm.model_executor.offloader.prefetch.should_pin_memory", lambda: True
    )
    offloader = _CpuParamOffloader(module, "weight", pin_cpu_storage=False)

    assert offloader._cpu_storage is not None
    assert not offloader._cpu_storage.is_pinned()
    torch.testing.assert_close(offloader._cpu_storage, expected)


def test_pageable_to_pinned_copy_precedes_nonblocking_h2d(monkeypatch):
    log: list[str] = []
    cpu_storage = _FakeTensor(
        "pageable", (2,), (1,), torch.float16, pinned=False, log=log
    )
    staging = _FakeTensor("staging", (2,), (1,), torch.float16, pinned=True, log=log)
    gpu_buffer = _FakeTensor("gpu", (2,), (1,), torch.float16, pinned=False, log=log)

    class FakeStagingPool:
        def wait_until_available(self, slot_idx):
            log.append(f"wait-slot:{slot_idx}")

        def get_buffer(self, **kwargs):
            log.append(f"get-slot:{kwargs['slot_idx']}")
            return staging

        def record_h2d(self, slot_idx, stream):
            log.append(f"protect-slot:{slot_idx}")

    class FakeStream:
        def wait_event(self, _event):
            log.append("copy-stream-wait")

    class FakeCurrentStream:
        def record_event(self, _event):
            log.append("fork-record")

    copy_done = _FakeEvent(log, "layer-copy")
    module_offloader = object.__new__(_ModuleOffloader)
    module_offloader._buffer_pool = object()
    module_offloader._staging_pool = FakeStagingPool()
    module_offloader._buffer_slot_idx = 3
    module_offloader._param_offloaders = {
        "weight": SimpleNamespace(
            _cpu_storage=cpu_storage,
            _gpu_buffer=gpu_buffer,
        )
    }
    module_offloader.copy_stream = FakeStream()
    module_offloader._copy_done_event = copy_done
    module_offloader._prefetch_required = True

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: FakeCurrentStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

    module_offloader.start_onload_to_static()

    assert log == [
        "wait-slot:3",
        "fork-record",
        "copy-stream-wait",
        "get-slot:3",
        "copy:pageable->staging:False",
        "copy:staging->gpu:True",
        "protect-slot:3",
        "record:layer-copy",
    ]
    assert not module_offloader._prefetch_required


def test_pinned_staging_rejects_cuda_graph_capture(monkeypatch):
    module_offloader = object.__new__(_ModuleOffloader)
    module_offloader._buffer_pool = object()
    module_offloader._staging_pool = object()

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="--enforce-eager"):
        module_offloader.start_onload_to_static()
