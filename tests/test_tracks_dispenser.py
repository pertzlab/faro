"""Unit tests for TracksDispenser.

The dispenser replaces the former ``FovState.tracks_queue: SimpleQueue`` that
handed out df_tracked FIFO by thread-arrival-order. That was the root cause
of the "frames processed out of acquisition order" race in
``ImageProcessingPipeline.run``. See
``rtm_pymmcore/core/data_structures.py`` for the full rationale.
"""

from __future__ import annotations

import queue
import threading
import time

import pandas as pd
import pytest

from rtm_pymmcore.core.data_structures import TracksDispenser


def _df(label: str) -> pd.DataFrame:
    """Tiny DataFrame to tag which put() produced it."""
    return pd.DataFrame({"tag": [label]})


class TestTracksDispenserSerial:

    def test_first_frame_returns_empty(self):
        d = TracksDispenser()
        assert d.get_for_frame(0).empty

    def test_second_frame_returns_first_put(self):
        d = TracksDispenser()
        d.put_for_frame(0, _df("A"))
        got = d.get_for_frame(1)
        assert list(got["tag"]) == ["A"]

    def test_chain_of_frames(self):
        d = TracksDispenser()
        # Simulate a pipeline: for N = 0..4, get predecessor, put own.
        prev = d.get_for_frame(0)
        assert prev.empty
        d.put_for_frame(0, _df("F0"))
        for n in range(1, 5):
            prev = d.get_for_frame(n)
            assert list(prev["tag"]) == [f"F{n - 1}"]
            d.put_for_frame(n, _df(f"F{n}"))

    def test_continuation_with_offset(self):
        """Frames don't need to start at 0 (e.g. continue_experiment offsets them)."""
        d = TracksDispenser()
        d.put_for_frame(99, _df("last"))
        got = d.get_for_frame(100)
        assert list(got["tag"]) == ["last"]


class TestTracksDispenserSkip:

    def test_skipped_predecessor_returns_earlier_entry(self):
        d = TracksDispenser()
        d.put_for_frame(0, _df("F0"))
        d.skip_frame(1)
        got = d.get_for_frame(2)
        assert list(got["tag"]) == ["F0"]

    def test_all_skipped_returns_empty(self):
        d = TracksDispenser()
        d.skip_frame(0)
        d.skip_frame(1)
        assert d.get_for_frame(2).empty


class TestTracksDispenserPruning:

    def test_old_entries_pruned_after_get(self):
        d = TracksDispenser()
        d.put_for_frame(0, _df("F0"))
        d.put_for_frame(1, _df("F1"))
        d.put_for_frame(2, _df("F2"))
        # After consuming F2 via get_for_frame(3), entries <=2 are pruned
        d.get_for_frame(3)
        # Access the internals to confirm pruning
        assert list(d._entries.keys()) == []  # all consumed

    def test_pruning_during_normal_flow(self):
        """Steady-state memory is bounded to ~1 entry per FOV."""
        d = TracksDispenser()
        d.put_for_frame(0, _df("F0"))
        for n in range(1, 100):
            d.get_for_frame(n)
            d.put_for_frame(n, _df(f"F{n}"))
            # Before consuming, we have at most the most recent entry
            assert len(d._entries) <= 1


class TestTracksDispenserConcurrent:

    def test_get_blocks_until_put(self):
        d = TracksDispenser()
        result = {}

        def worker():
            result["df"] = d.get_for_frame(5, timeout=2.0)

        t = threading.Thread(target=worker)
        t.start()
        # Worker is blocked waiting for frame 4
        time.sleep(0.05)
        assert t.is_alive()
        # Provide predecessor
        d.put_for_frame(4, _df("F4"))
        t.join(timeout=1.0)
        assert not t.is_alive()
        assert list(result["df"]["tag"]) == ["F4"]

    def test_out_of_order_puts_resolve_correctly(self):
        """Simulate: frames 5 and 6 reach get_for_frame before 3 and 4 have put.

        The dispenser must hand out the correct predecessor to each, regardless
        of which thread called get first.
        """
        d = TracksDispenser()
        d.put_for_frame(2, _df("F2"))  # previous frame already done
        results: dict[int, pd.DataFrame] = {}

        def worker(frame):
            results[frame] = d.get_for_frame(frame, timeout=3.0)

        threads = [threading.Thread(target=worker, args=(n,)) for n in (5, 6)]
        for t in threads:
            t.start()
        time.sleep(0.05)  # let them reach the wait
        # Now fulfill predecessors in the "wrong" order: 5 is ready before 3/4.
        # Frames 3 and 4 must still run to unblock 5 and 6.
        d.put_for_frame(3, _df("F3"))
        d.put_for_frame(4, _df("F4"))
        d.put_for_frame(5, _df("F5"))
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive()

        assert list(results[5]["tag"]) == ["F4"]
        assert list(results[6]["tag"]) == ["F5"]

    def test_timeout_raises_empty(self):
        d = TracksDispenser()
        with pytest.raises(queue.Empty):
            d.get_for_frame(5, timeout=0.1)

    def test_many_concurrent_workers_in_order(self):
        """Stress test: 10 workers simulating a pipeline, each put after get."""
        d = TracksDispenser()
        d.put_for_frame(0, _df("F0"))  # seed
        N = 10
        barrier = threading.Barrier(N)
        results: dict[int, str] = {}
        errors: list[Exception] = []

        def worker(frame):
            try:
                barrier.wait()  # all start simultaneously
                prev = d.get_for_frame(frame, timeout=5.0)
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
            assert results[n] == f"F{n - 1}", (
                f"Frame {n} expected predecessor F{n - 1}, got {results[n]}"
            )


class TestTracksDispenserReset:

    def test_reset_clears_state(self):
        d = TracksDispenser()
        d.put_for_frame(5, _df("F5"))
        d.reset()
        # After reset, predecessor chain is gone
        assert d.get_for_frame(0).empty

    def test_reset_wakes_waiters(self):
        d = TracksDispenser()

        def worker():
            with pytest.raises(queue.Empty):
                d.get_for_frame(5, timeout=0.5)

        t = threading.Thread(target=worker)
        t.start()
        time.sleep(0.05)
        # reset notifies — but the waiter will re-check and still see
        # unresolved predecessor → keeps waiting → times out cleanly.
        d.reset()
        t.join(timeout=1.0)
        assert not t.is_alive()
