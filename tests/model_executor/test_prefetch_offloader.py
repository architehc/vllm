# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.offloader.prefetch import (
    ParamInfo,
    PrefetchOffloader,
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


@pytest.mark.parametrize(
    ("num_modules", "prefetch_step"),
    [
        (0, 2),
        (4, 2),
        (5, 1),
    ],
)
def test_prefetch_static_slot_schedule_accepts_safe_counts(num_modules, prefetch_step):
    offloader = object.__new__(PrefetchOffloader)
    offloader.module_offloaders = [object()] * num_modules
    offloader.prefetch_step = prefetch_step

    offloader._validate_static_slot_schedule()


def test_prefetch_static_slot_schedule_rejected_before_post_init_work():
    sync_calls = 0

    class FakeModuleOffloader:
        def sync_cpu_storage(self):
            nonlocal sync_calls
            sync_calls += 1

    offloader = object.__new__(PrefetchOffloader)
    offloader.module_offloaders = [FakeModuleOffloader() for _ in range(3)]
    offloader.prefetch_step = 2

    with pytest.raises(
        ValueError,
        match=(
            r"3 offloaded modules cannot use prefetch_step=2.*"
            r"must be divisible by prefetch_step"
        ),
    ):
        offloader.post_init()

    assert sync_calls == 0


def test_prefetch_slot_owner_is_published_only_after_enqueue():
    log: list[tuple[str, object]] = []
    offloader = object.__new__(PrefetchOffloader)
    offloader._slot_owners = [0, 1]

    class FakeModuleOffloader:
        def __init__(self, layer_idx: int):
            self.layer_idx = layer_idx
            self._buffer_slot_idx = layer_idx % 2
            self._prefetch_required = layer_idx >= 2

        def start_onload_to_static(self):
            log.append(("enqueue", self.layer_idx))
            log.append(("owners_during_enqueue", tuple(offloader._slot_owners)))
            log.append(
                (
                    "required_during_enqueue",
                    tuple(
                        entry._prefetch_required
                        for entry in offloader.module_offloaders
                    ),
                )
            )
            self._prefetch_required = False

    offloader.module_offloaders = [FakeModuleOffloader(i) for i in range(4)]

    offloader._start_prefetch(2)

    assert log == [
        ("enqueue", 2),
        ("owners_during_enqueue", (None, 1)),
        ("required_during_enqueue", (True, False, True, True)),
    ]
    assert offloader._slot_owners == [2, 1]
    assert offloader.module_offloaders[0]._prefetch_required
    assert not offloader.module_offloaders[2]._prefetch_required


def test_slot_use_is_recorded_before_lookahead_prefetch(monkeypatch):
    log: list[str] = []

    class LoggingModule(torch.nn.Module):
        def forward(self, hidden_states):
            log.append("forward")
            return hidden_states

    modules = [LoggingModule() for _ in range(4)]
    offloader = object.__new__(PrefetchOffloader)
    offloader.offload_params = set()
    offloader.prefetch_step = 2
    offloader._slot_owners = [0, 1]
    offloader._slot_use_events = [object(), object()]
    offloader._slot_use_event_valid = [False, False]

    class FakeModuleOffloader:
        def __init__(self, layer_idx: int):
            self.module = modules[layer_idx]
            self._buffer_slot_idx = layer_idx % 2
            self._prefetch_required = False

    class FakeCurrentStream:
        def record_event(self, event):
            slot_idx = offloader._slot_use_events.index(event)
            log.append(f"record-slot:{slot_idx}")

    offloader.module_offloaders = [FakeModuleOffloader(i) for i in range(4)]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: FakeCurrentStream())
    monkeypatch.setattr(
        torch.ops.vllm,
        "wait_prefetch",
        lambda _tensor, index: log.append(f"wait:{index}"),
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "start_prefetch",
        lambda _tensor, index: log.append(f"start:{index}"),
    )
    offloader._hook_module_forward(0, modules[0])

    modules[0](torch.zeros(1, 2))

    assert log == ["wait:0", "forward", "record-slot:0", "start:2"]
    assert offloader._slot_use_event_valid == [True, False]


def test_prefetch_waits_for_use_event_of_target_static_slot():
    log: list[str] = []
    slot_events = [object(), object()]
    offloader = object.__new__(PrefetchOffloader)
    offloader._slot_owners = [0, 1]
    offloader._slot_use_events = slot_events
    offloader._slot_use_event_valid = [True, True]

    class FakeCopyStream:
        def wait_event(self, event):
            log.append(f"wait-slot:{slot_events.index(event)}")

    class FakeModuleOffloader:
        def __init__(self, layer_idx: int):
            self.layer_idx = layer_idx
            self._buffer_slot_idx = layer_idx % 2
            self._prefetch_required = layer_idx >= 2

        def start_onload_to_static(self):
            log.append(f"enqueue:{self.layer_idx}")
            self._prefetch_required = False

    offloader.copy_stream = FakeCopyStream()
    offloader.module_offloaders = [FakeModuleOffloader(i) for i in range(4)]

    offloader._start_prefetch(3)

    assert log == ["wait-slot:1", "enqueue:3"]
    assert offloader._slot_use_event_valid == [True, False]
    assert offloader._slot_owners == [0, 3]


def test_partial_traversal_reprimes_stale_slot_on_next_invocation(monkeypatch):
    calls: list[tuple[str, int]] = []
    modules = [torch.nn.Identity() for _ in range(4)]
    offloader = object.__new__(PrefetchOffloader)
    offloader.offload_params = set()
    offloader.prefetch_step = 2
    offloader._slot_owners = [0, 1]

    class FakeModuleOffloader:
        def __init__(self, layer_idx: int):
            self.module = modules[layer_idx]
            self._buffer_slot_idx = layer_idx % 2
            self._prefetch_required = layer_idx >= 2

        def start_onload_to_static(self):
            self._prefetch_required = False

    offloader.module_offloaders = [FakeModuleOffloader(i) for i in range(4)]

    def start_prefetch(_tensor: torch.Tensor, index: int):
        calls.append(("start", index))
        offloader._start_prefetch(index)

    def wait_prefetch(_tensor: torch.Tensor, index: int):
        calls.append(("wait", index))

    monkeypatch.setattr(torch.ops.vllm, "start_prefetch", start_prefetch)
    monkeypatch.setattr(torch.ops.vllm, "wait_prefetch", wait_prefetch)
    offloader._hook_module_forward(0, modules[0])

    # Stop after module 0: its lookahead overwrites slot 0 with module 2,
    # without reaching the circular tail that would normally restore module 0.
    modules[0](torch.zeros(1, 2))
    assert calls == [("wait", 0), ("start", 2)]
    assert offloader._slot_owners == [2, 1]

    calls.clear()
    modules[0](torch.zeros(1, 2))

    # The next invocation must restore module 0 before waiting/using the slot.
    assert calls == [("start", 0), ("wait", 0), ("start", 2)]


def test_wait_prefetch_defensively_reprimes_stale_slot(monkeypatch):
    calls: list[tuple[str, int]] = []
    offloader = object.__new__(PrefetchOffloader)
    offloader._slot_owners = [2, 1]
    offloader.copy_stream = object()

    class FakeModuleOffloader:
        def __init__(self, layer_idx: int):
            self.layer_idx = layer_idx
            self._buffer_slot_idx = layer_idx % 2
            self._prefetch_required = False
            self._event_valid_for_eager = True
            self._copy_done_event = object()

        def start_onload_to_static(self):
            calls.append(("start", self.layer_idx))
            self._prefetch_required = False

    class FakeCurrentStream:
        def wait_event(self, _event):
            calls.append(("wait", 0))

    offloader.module_offloaders = [FakeModuleOffloader(i) for i in range(4)]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: FakeCurrentStream())

    offloader._wait_for_layer(0)

    assert calls == [("start", 0), ("wait", 0)]
    assert offloader._slot_owners == [0, 1]
    assert offloader.module_offloaders[2]._prefetch_required


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
