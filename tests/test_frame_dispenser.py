"""Unit tests for FrameDispenser.

FrameDispenser replaces the former ``FovState.tracks_queue: SimpleQueue``
(and the ``stim_mask_queue: SimpleQueue`` via the same class) that handed
out entries FIFO by thread-arrival-order. That was the root cause of the
"frames processed out of acquisition order" race in
``ImageProcessingPipeline.run``. See
``faro/core/data_structures.py`` for the full rationale.

The dispenser supports two consumption modes:
  - ``get_predecessor(idx)`` — tracking needs frame N-1's df for frame N
  - ``wait_for_frame(idx)`` — stim masks need exactly this frame's mask
    (non-consuming so multiple readers share the same entry)
"""

from __future__ import annotations

import queue
import threading
import time

import pandas as pd
import pytest

from faro.core.data_structures import FrameDispenser


def _df(label: str) -> pd.DataFrame:
    """Tiny DataFrame to tag which put() produced it."""
    return pd.DataFrame({"tag": [label]})


class TestGetPredecessorSerial:

    def test_first_frame_returns_none(self):
        d = FrameDispenser()
        assert d.get_predecessor(0) is None

    def test_second_frame_returns_first_put(self):
        d = FrameDispenser()
        d.put_for_frame(0, _df("A"))
        got = d.get_predecessor(1)
        assert list(got["tag"]) == ["A"]

    def test_chain_of_frames(self):
        d = FrameDispenser()
        assert d.get_predecessor(0) is None
        d.put_for_frame(0, _df("F0"))
        for n in range(1, 5):
            prev = d.get_predecessor(n)
            assert list(prev["tag"]) == [f"F{n - 1}"]
            d.put_for_frame(n, _df(f"F{n}"))

    def test_continuation_with_offset(self):
        """Frames don't need to start at 0 (e.g. continue_experiment offsets them)."""
        d = FrameDispenser()
        d.put_for_frame(99, _df("last"))
        got = d.get_predecessor(100)
        assert list(got["tag"]) == ["last"]


class TestGetPredecessorSkip:

    def test_skipped_predecessor_returns_earlier_entry(self):
        d = FrameDispenser()
        d.put_for_frame(0, _df("F0"))
        d.skip_frame(1)
        got = d.get_predecessor(2)
        assert list(got["tag"]) == ["F0"]

    def test_all_skipped_returns_none(self):
        d = FrameDispenser()
        d.skip_frame(0)
        d.skip_frame(1)
        assert d.get_predecessor(2) is None


class TestWaitForFrame:
    """Non-consuming blocking read — used by stim masks so both the stim event
    and the pipeline's storage step can read the same entry.
    """

    def test_returns_put_value(self):
        d = FrameDispenser()
        d.put_for_frame(5, "mask-5")
        assert d.wait_for_frame(5) == "mask-5"

    def test_returns_none_for_skipped(self):
        d = FrameDispenser()
        d.skip_frame(5)
        assert d.wait_for_frame(5) is None

    def test_blocks_until_put(self):
        d = FrameDispenser()
        result = {}

        def worker():
            result["v"] = d.wait_for_frame(7, timeout=2.0)

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.05)
        assert t.is_alive()
        d.put_for_frame(7, "mask-7")
        t.join(timeout=1.0)
        assert not t.is_alive()
        assert result["v"] == "mask-7"

    def test_other_puts_dont_unblock(self):
        d = FrameDispenser()
        result = {}

        def worker():
            try:
                result["v"] = d.wait_for_frame(5, timeout=0.3)
            except queue.Empty:
                result["v"] = "TIMED_OUT"

        t = threading.Thread(target=worker)
        t.start()
        d.put_for_frame(4, "mask-4")
        d.put_for_frame(6, "mask-6")
        t.join(timeout=1.0)
        assert result["v"] == "TIMED_OUT"

    def test_timeout_raises_empty(self):
        d = FrameDispenser()
        with pytest.raises(queue.Empty):
            d.wait_for_frame(5, timeout=0.1)

    def test_does_not_consume(self):
        """Multiple readers see the same entry — wait_for_frame does not pop."""
        d = FrameDispenser()
        d.put_for_frame(5, "mask-5")
        assert d.wait_for_frame(5) == "mask-5"
        assert d.wait_for_frame(5) == "mask-5"
        assert d.peek_at_frame(5) == "mask-5"


class TestPeekAtFrame:
    """Non-blocking, non-consuming variant."""

    def test_returns_none_before_put(self):
        d = FrameDispenser()
        assert d.peek_at_frame(5) is None

    def test_returns_value_after_put(self):
        d = FrameDispenser()
        d.put_for_frame(5, "mask-5")
        assert d.peek_at_frame(5) == "mask-5"

    def test_returns_none_after_skip(self):
        d = FrameDispenser()
        d.skip_frame(5)
        assert d.peek_at_frame(5) is None

    def test_idempotent(self):
        d = FrameDispenser()
        d.put_for_frame(5, "mask-5")
        for _ in range(3):
            assert d.peek_at_frame(5) == "mask-5"


class TestPruneBelow:
    """Explicit cleanup — required when using wait_for_frame / peek_at_frame."""

    def test_strict_below(self):
        d = FrameDispenser()
        for i in range(6):
            d.put_for_frame(i, f"mask-{i}")
        d.prune_below(3)
        assert set(d._entries) == {3, 4, 5}

    def test_clears_skip_markers(self):
        d = FrameDispenser()
        for i in range(6):
            d.skip_frame(i)
        d.prune_below(3)
        assert d._skipped == {3, 4, 5}

    def test_empty_dispenser_is_noop(self):
        d = FrameDispenser()
        d.prune_below(5)
        assert d._entries == {} and d._skipped == set()

    def test_prune_below_zero_or_negative(self):
        d = FrameDispenser()
        d.put_for_frame(0, "mask-0")
        d.prune_below(-1)
        assert d._entries == {0: "mask-0"}
        d.prune_below(0)
        assert d._entries == {0: "mask-0"}


class TestPruning:

    def test_old_entries_pruned_after_get_predecessor(self):
        d = FrameDispenser()
        d.put_for_frame(0, _df("F0"))
        d.put_for_frame(1, _df("F1"))
        d.put_for_frame(2, _df("F2"))
        d.get_predecessor(3)
        assert list(d._entries.keys()) == []

    def test_pruning_during_normal_flow(self):
        """Steady-state memory is bounded to ~1 entry per FOV."""
        d = FrameDispenser()
        d.put_for_frame(0, _df("F0"))
        for n in range(1, 100):
            d.get_predecessor(n)
            d.put_for_frame(n, _df(f"F{n}"))
            assert len(d._entries) <= 1

    def test_empty_chain_prunes_skipped(self):
        """Trailing skipped frames are cleared when the chain walks to -1."""
        d = FrameDispenser()
        for n in range(5):
            d.skip_frame(n)
        assert d.get_predecessor(5) is None
        assert len(d._skipped) == 0


class TestConcurrent:

    def test_out_of_order_puts_resolve_correctly(self):
        """Frames 5 and 6 reach get_predecessor before 3 and 4 have put.

        The dispenser must hand out the correct predecessor to each.
        """
        d = FrameDispenser()
        d.put_for_frame(2, _df("F2"))
        results: dict[int, pd.DataFrame] = {}

        def worker(frame):
            results[frame] = d.get_predecessor(frame, timeout=3.0)

        threads = [threading.Thread(target=worker, args=(n,)) for n in (5, 6)]
        for t in threads:
            t.start()
        time.sleep(0.05)
        d.put_for_frame(3, _df("F3"))
        d.put_for_frame(4, _df("F4"))
        d.put_for_frame(5, _df("F5"))
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive()

        assert list(results[5]["tag"]) == ["F4"]
        assert list(results[6]["tag"]) == ["F5"]

    def test_many_concurrent_workers_in_order(self):
        """Stress test: 10 pipeline-style workers, each put after get."""
        d = FrameDispenser()
        d.put_for_frame(0, _df("F0"))
        N = 10
        barrier = threading.Barrier(N)
        results: dict[int, str] = {}
        errors: list[Exception] = []

        def worker(frame):
            try:
                barrier.wait()
                prev = d.get_predecessor(frame, timeout=5.0)
                results[frame] = list(prev["tag"])[0]
                d.put_for_frame(frame, _df(f"F{frame}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(1, N + 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive()

        assert not errors
        for n in range(1, N + 1):
            assert results[n] == f"F{n - 1}"


class TestReset:

    def test_reset_clears_state(self):
        d = FrameDispenser()
        d.put_for_frame(5, _df("F5"))
        d.reset()
        assert d.get_predecessor(0) is None

    def test_reset_wakes_waiters(self):
        d = FrameDispenser()

        def worker():
            with pytest.raises(queue.Empty):
                d.get_predecessor(5, timeout=0.5)

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.05)
        d.reset()
        t.join(timeout=1.0)
        assert not t.is_alive()
