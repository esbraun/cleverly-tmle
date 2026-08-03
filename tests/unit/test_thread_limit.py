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
    """Every test here mutates process-global state; put it back."""
    limit = get_thread_limit()
    yield
    set_thread_limit(limit)
    refresh_thread_pools()


def _limits() -> list[int | None]:
    """The current thread count of every pool threadpoolctl can see."""
    return [info["num_threads"] for info in threadpoolctl.threadpool_info()]


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
