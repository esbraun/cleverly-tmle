"""The thread limiter: still limiting, and no longer rebuilding its controller.

``cleverly.learners._threads`` caches one ``ThreadpoolController`` for the process
because building one walks every loaded shared object, which measured 57% of a DR-TMLE
``retarget``.  Caching a piece of process-global state is exactly the kind of change that
is fast and wrong, so what is checked here is the *behaviour* the cache could break:

* the limit is still applied inside the block and restored after it, including when the
  block raises;
* ``set_thread_limit(None)`` still disables the limiting;
* nesting works, and the inner block's exit restores the outer block's limit rather than
  the original one;
* a pool loaded *after* the cache was built is picked up once something says so, which is
  the one real hazard and the reason :func:`refresh_thread_pools` exists;
* concurrent entry from several threads yields one controller, not one per thread.

The reuse itself is asserted by identity rather than by timing: "the same object came
back" is deterministic where "it was faster" is a measurement on a shared box.

Then a second thing the cache did *not* break and did not fix either: a limiter writes back
whatever it snapshotted, so two overlapping blocks used to restore each other's saved
state.  The concurrency test above cannot see that, because it never enters a limit at all
-- :class:`TestOverlappingBlocks` is what does, and
``test_nested_limits_restore_the_enclosing_one_not_the_original`` is what keeps the
single-threaded LIFO semantics the fix had to preserve.

That class also pins the reason the obvious version of the fix is wrong.  "The limits are
process-global" is only half true: the BLAS pools are, and OpenMP's count is an ICV per
*calling thread*, so a block that took a reference on another thread's instead of applying
its own would leave OpenMP unlimited.  Every block applies; only the outermost snapshot is
kept and restored.
"""

from __future__ import annotations

import threading
import time

import pytest

from cleverly.learners import _threads
from cleverly.learners._threads import (
    get_thread_limit,
    refresh_thread_pools,
    set_thread_limit,
    thread_limit,
)

threadpoolctl = pytest.importorskip("threadpoolctl")


@pytest.fixture(autouse=True)
def _restore_global_state():
    """Every test here mutates process-global state; put it back.

    The stack is asserted empty *after* each test rather than merely cleared, so a leaked
    entry is reported by the test that leaked it instead of by whichever unrelated one
    runs next and finds the process still limited.
    """
    limit = get_thread_limit()
    yield
    set_thread_limit(limit)
    refresh_thread_pools()
    assert _threads._STACK == [], "a block was left open"
    assert _threads._ROOT is None, "the root limiter was left holding the original"


def _limits() -> list[int | None]:
    """The current thread count of every pool threadpoolctl can see."""
    return [info["num_threads"] for info in threadpoolctl.threadpool_info()]


def _needs_a_multi_thread_pool() -> list[int | None]:
    """The pools, skipping unless one of them has room to be limited."""
    before = _limits()
    if not before or all(count == 1 for count in before):  # pragma: no cover
        pytest.skip("need a pool with more than one thread to tell the two apart")
    return before


def test_the_controller_is_built_once_and_reused():
    refresh_thread_pools()
    with thread_limit(1):
        pass
    first = _threads._CONTROLLER
    assert first is not None
    with thread_limit(1):
        pass
    assert _threads._CONTROLLER is first


def test_a_refresh_builds_a_new_one():
    with thread_limit(1):
        pass
    first = _threads._CONTROLLER
    refresh_thread_pools()
    assert _threads._CONTROLLER is None
    with thread_limit(1):
        pass
    assert _threads._CONTROLLER is not None
    assert _threads._CONTROLLER is not first


def test_the_limit_is_applied_and_restored():
    before = _limits()
    if not before:  # pragma: no cover - a build with no supported pool loaded
        pytest.skip("no thread pools visible to threadpoolctl")
    with thread_limit(1):
        assert all(count == 1 for count in _limits())
    assert _limits() == before


def test_the_limit_is_restored_after_an_exception():
    before = _limits()
    if not before:  # pragma: no cover - a build with no supported pool loaded
        pytest.skip("no thread pools visible to threadpoolctl")
    with pytest.raises(RuntimeError, match="deliberate"), thread_limit(1):
        raise RuntimeError("deliberate")
    assert _limits() == before


def test_nested_limits_restore_the_enclosing_one_not_the_original():
    before = _limits()
    if not before or all(count == 1 for count in before):  # pragma: no cover
        pytest.skip("need a pool with more than one thread to tell the two apart")
    with thread_limit(2):
        outer = _limits()
        with thread_limit(1):
            assert all(count == 1 for count in _limits())
        assert _limits() == outer
    assert _limits() == before


def test_none_disables_the_limiting_entirely():
    before = _limits()
    if not before or all(count == 1 for count in before):  # pragma: no cover
        pytest.skip("need a pool with more than one thread to tell the two apart")
    set_thread_limit(None)
    with thread_limit():
        assert _limits() == before


def test_an_explicit_limit_overrides_the_configured_default():
    before = _limits()
    if not before:  # pragma: no cover - a build with no supported pool loaded
        pytest.skip("no thread pools visible to threadpoolctl")
    set_thread_limit(None)
    with thread_limit(1):
        assert all(count == 1 for count in _limits())
    assert _limits() == before


def test_a_pool_loaded_after_the_cache_is_seen_once_something_says_so(monkeypatch):
    """The LightGBM hazard, with a fake pool standing in for the OpenMP one.

    A controller cached before a library is loaded cannot know about its pool.  Nothing
    detects that automatically -- the detection *is* the walk the cache exists to avoid --
    so the contract is that a refresh picks it up, and that the package calls the refresh
    where it imports a backend.  Both halves are checked: this one is the mechanism, and
    :func:`test_has_lightgbm_refreshes_the_controller` is the call site.
    """
    seen: list[int] = []

    class FakeController:
        def __init__(self) -> None:
            seen.append(len(seen))

    monkeypatch.setattr(_threads, "_ThreadpoolController", FakeController)
    refresh_thread_pools()
    assert isinstance(_threads._controller(), FakeController)
    assert _threads._controller() is _threads._controller()
    assert seen == [0]

    refresh_thread_pools()
    _threads._controller()
    assert seen == [0, 1]


def test_has_lightgbm_refreshes_the_controller(monkeypatch):
    """The one lazily-imported backend invalidates the cache, and only on the import."""
    from cleverly.learners import library

    calls: list[int] = []
    monkeypatch.setattr(library, "_LIGHTGBM", None)
    monkeypatch.setattr(library, "refresh_thread_pools", lambda: calls.append(1))

    first = library.has_lightgbm()
    assert library.has_lightgbm() is first
    # One refresh if the import succeeded, none if the extra is absent -- and never a
    # second, because the answer is cached.
    assert calls == ([1] if first else [])


def test_concurrent_entry_builds_one_controller(monkeypatch):
    """A thread-backed joblib enters this from several threads at once.

    The construction is made deliberately slow so that four threads are inside
    :func:`_controller` together: without the lock they would each build one, which is
    harmless but is precisely the walk the cache exists to avoid paying.
    """
    built: list[int] = []
    ready = threading.Barrier(4)

    class SlowController:
        def __init__(self) -> None:
            time.sleep(0.05)
            built.append(1)

    monkeypatch.setattr(_threads, "_ThreadpoolController", SlowController)
    refresh_thread_pools()

    results: list[object] = []
    lock = threading.Lock()

    def enter() -> None:
        ready.wait(timeout=10)  # all four arrive before any of them builds
        controller = _threads._controller()
        with lock:
            results.append(controller)

    threads = [threading.Thread(target=enter) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(results) == 4
    assert len({id(item) for item in results}) == 1
    assert built == [1]


def test_without_threadpoolctl_the_block_is_a_no_op(monkeypatch):
    monkeypatch.setattr(_threads, "_ThreadpoolController", None)
    refresh_thread_pools()
    with thread_limit(1):
        pass
    assert _threads._CONTROLLER is None


class TestOverlappingBlocks:
    """Two blocks open at once, which is what the limits being process-global costs.

    ``thread_limit`` is public and an ambient ``joblib.parallel_backend("threading")``
    reaches it in one step, so this is a supported use rather than a hypothetical one --
    even though nothing inside the package produces it, every ``map_parallel`` call
    leaving ``prefer=None`` and joblib therefore dispatching to processes.
    """

    def test_one_thread_leaving_does_not_unlimit_another_that_is_still_inside(self):
        """The failure the refcount exists for, and it failed on both counts.

        Without it: B snapshots A's limit rather than the process's, so A's exit writes
        back the original *while B is still in its block*, and B's exit then writes back
        A's limit and leaves the process limited for good.  Both assertions below are what
        that broke.
        """
        before = _needs_a_multi_thread_pool()
        both_inside = threading.Barrier(2, timeout=30)
        a_has_left = threading.Event()
        seen: dict[str, list[int | None]] = {}
        failures: list[BaseException] = []

        def first() -> None:
            try:
                with thread_limit(1):
                    both_inside.wait()
                    seen["a"] = _limits()
            except BaseException as error:  # pragma: no cover - reported below
                failures.append(error)
            finally:
                a_has_left.set()

        def second() -> None:
            try:
                with thread_limit(1):
                    both_inside.wait()
                    assert a_has_left.wait(timeout=30), "the first thread never left"
                    seen["b"] = _limits()
            except BaseException as error:  # pragma: no cover - reported below
                failures.append(error)

        threads = [threading.Thread(target=target) for target in (first, second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not failures, failures
        assert all(count == 1 for count in seen["a"]), seen
        # Read *after* the other thread's block closed: this is the assertion that fails
        # without the refcount, because the exit above restored the original.
        assert all(count == 1 for count in seen["b"]), seen
        assert _limits() == before

    def test_every_block_applies_and_only_the_outermost_snapshot_is_restored(self, monkeypatch):
        """The design in one assertion: refcount the *snapshots*, not the applies.

        Skipping the apply for a block that asks for the limit already in force is the
        obvious optimisation and it is wrong, because the limits are not uniformly
        process-global: OpenMP's count is an ICV that ``omp_set_num_threads`` sets for the
        calling thread, so a block that took a reference instead of applying would run its
        OpenMP regions unlimited.  ``test_a_second_thread_limits_its_own_thread_local_pool``
        is the same claim measured rather than argued.

        What must not repeat is the *restore*: only the outermost block's snapshot is the
        process's own setting, so it is the only one kept and the only one written back.
        """
        calls: list[str] = []

        class RecordingLimiter:
            def restore_original_limits(self) -> None:
                calls.append("restore")

        class RecordingController:
            def limit(self, *, limits: int) -> RecordingLimiter:
                calls.append(f"apply {limits}")
                return RecordingLimiter()

        monkeypatch.setattr(_threads, "_CONTROLLER", RecordingController())
        with thread_limit(1), thread_limit(1), thread_limit(1):
            pass

        # Three applies going in; coming out, two re-applies of the limit that is now
        # innermost, and exactly one restore -- of the first block's snapshot.
        assert calls == ["apply 1", "apply 1", "apply 1", "apply 1", "apply 1", "restore"]
        assert calls.count("restore") == 1

    def test_a_second_thread_limits_its_own_thread_local_pool(self):
        """OpenMP's thread count is per-thread, so a second block cannot be a no-op.

        Skipped where no OpenMP pool is loaded, which is a build without LightGBM or
        scikit-learn's histogram boosting -- there the claim has nothing to be about.  The
        BLAS pools alongside it *are* process-global, which is why the two have to be
        checked by name rather than by "all the pools are at 1".
        """
        pools = threadpoolctl.threadpool_info()
        openmp = [info for info in pools if info["user_api"] == "openmp"]
        if not openmp or all(info["num_threads"] == 1 for info in openmp):  # pragma: no cover
            pytest.skip("no OpenMP pool with more than one thread")

        seen: list[int] = []
        with thread_limit(1):

            def inner() -> None:
                with thread_limit(1):
                    seen.extend(
                        info["num_threads"]
                        for info in threadpoolctl.threadpool_info()
                        if info["user_api"] == "openmp"
                    )

            thread = threading.Thread(target=inner)
            thread.start()
            thread.join(timeout=30)

        assert seen and all(count == 1 for count in seen), seen

    def test_an_out_of_order_exit_still_restores_the_original(self):
        """Non-LIFO release, which one thread cannot produce and two can.

        The inner exit re-*applies* what is now outermost rather than restoring its own
        snapshot, and only the first block's limiter is kept -- so the process's own
        setting comes back whichever order the exits arrive in.
        """
        before = _needs_a_multi_thread_pool()

        outer = thread_limit(2)
        inner = thread_limit(1)
        outer.__enter__()
        inner.__enter__()
        assert all(count == 1 for count in _limits())

        outer.__exit__(None, None, None)  # the *enclosing* block leaves first
        assert all(count == 1 for count in _limits()), "the inner block lost its limit"
        inner.__exit__(None, None, None)

        assert _limits() == before

    def test_a_forked_child_starts_with_no_open_blocks(self):
        """Called directly rather than by forking: pytest under xdist is a poor place to.

        A child inheriting the stack would record blocks no frame in it will ever exit, so
        its own first block would refcount onto one of them and never restore.
        """
        with thread_limit(1):
            assert _threads._STACK
            _threads._before_fork()
            _threads._after_fork_in_child()
            assert _threads._STACK == []
            assert _threads._ROOT is None
            # The locks came back free, so the child's first entry does not deadlock.
            with thread_limit(1):
                pass
        assert _threads._STACK == []
