"""The Mammen multipliers drawn from raw generator words are the ``integers`` multipliers.

``_fill_multipliers(kind="mammen")`` reads PCG64's 64-bit outputs as pairs of ``uint32``
words, where it used to call ``Generator.integers(0, 2**32, dtype=np.uint32)`` (X106 part
K1c).  The equality rests on NumPy behaviour rather than a documented promise, so it is pinned
here against a frozen copy of the earlier route, on whatever NumPy CI resolves.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.inference import multiplier
from cleverly.inference.multiplier import (
    _MAMMEN_HIGH,
    _MAMMEN_LOW,
    _MAMMEN_THRESHOLD,
    _fill_multipliers,
    _uint32_words,
    multiplier_critical_value,
)


def _frozen_fill(rng: np.random.Generator, out: np.ndarray) -> None:
    """The Mammen fill as it was before X106 part K1c (after #692): ``integers`` words."""
    flags = rng.integers(0, 2**32, size=out.shape, dtype=np.uint32) < _MAMMEN_THRESHOLD
    np.multiply(flags, _MAMMEN_LOW - _MAMMEN_HIGH, out=out)
    np.add(out, _MAMMEN_HIGH, out=out)


def _state(rng: np.random.Generator) -> dict[str, object]:
    """The whole state dict, ``uinteger`` slot included."""
    return dict(rng.bit_generator.state)


def _next_draws(rng: np.random.Generator) -> bytes:
    return (
        rng.integers(0, 2**32, size=3, dtype=np.uint32).tobytes()
        + rng.random(2).tobytes()
        + rng.integers(0, 256, size=5, dtype=np.uint8).tobytes()
    )


SHAPES = [(4, 7), (8, 100_001), (256, 3), (3, 3), (1, 1), (5, 0)]


@pytest.mark.parametrize("shape", SHAPES)
def test_words_are_the_integers_words(shape: tuple[int, int]) -> None:
    frozen, fast = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(3):
        expected = frozen.integers(0, 2**32, size=shape, dtype=np.uint32)
        got = _uint32_words(fast, shape)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert got.tobytes() == expected.tobytes()
        assert _state(fast) == _state(frozen)
    assert _next_draws(fast) == _next_draws(frozen)


@pytest.mark.parametrize("shape", SHAPES)
def test_the_fill_is_the_frozen_fill(shape: tuple[int, int]) -> None:
    frozen, fast = np.random.default_rng(11), np.random.default_rng(11)
    expected, got = np.full(shape, np.nan), np.full(shape, np.nan)
    for _ in range(3):
        _frozen_fill(frozen, expected)
        _fill_multipliers(fast, got, "mammen")
        assert got.tobytes() == expected.tobytes()
        assert _state(fast) == _state(frozen)
    assert set(np.unique(got)) <= {_MAMMEN_HIGH, _MAMMEN_LOW}
    assert _next_draws(fast) == _next_draws(frozen)


def test_a_buffered_half_word_takes_the_integers_route() -> None:
    """After an odd-length draw a half word is buffered, and must come out first."""
    frozen, fast = np.random.default_rng(5), np.random.default_rng(5)
    frozen.integers(0, 2**32, size=1, dtype=np.uint32)
    fast.integers(0, 2**32, size=1, dtype=np.uint32)
    assert fast.bit_generator.state["has_uint32"] == 1
    expected = frozen.integers(0, 2**32, size=(4, 4), dtype=np.uint32)
    assert _uint32_words(fast, (4, 4)).tobytes() == expected.tobytes()
    assert _next_draws(fast) == _next_draws(frozen)


@pytest.mark.parametrize("make", [np.random.MT19937, np.random.Philox, np.random.SFC64])
def test_other_bit_generators_take_the_integers_route(make: type) -> None:
    frozen, fast = np.random.Generator(make(9)), np.random.Generator(make(9))
    expected, got = np.empty((6, 10)), np.empty((6, 10))
    _frozen_fill(frozen, expected)
    _fill_multipliers(fast, got, "mammen")
    assert got.tobytes() == expected.tobytes()
    assert _next_draws(fast) == _next_draws(frozen)


def test_the_fast_route_is_the_one_a_band_takes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The band's blocks are even-length PCG64 draws, so none falls back to ``integers``."""
    calls: list[tuple[int, ...]] = []
    original = np.random.Generator.integers

    class Counting(np.random.Generator):
        def integers(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(tuple(np.atleast_1d(kwargs.get("size", ()))))
            return original(self, *args, **kwargs)

    monkeypatch.setattr(
        multiplier.np.random,
        "default_rng",
        lambda seed=None: Counting(np.random.PCG64(seed)),
    )
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(1_000, 3))
    multiplier_critical_value(
        curves,
        curves.std(axis=0) / np.sqrt(1_000),
        n=1_000,
        n_replicates=400,
        kind="mammen",
        random_state=1,
    )
    assert calls == []


def _frozen_critical_value(
    curves: np.ndarray,
    se: np.ndarray,
    *,
    n: int,
    n_replicates: int,
    random_state: int,
    cluster: np.ndarray | None = None,
) -> float:
    """``multiplier_critical_value(kind="mammen")`` with the frozen fill."""
    original = multiplier._fill_multipliers

    def frozen(rng: np.random.Generator, out: np.ndarray, kind: str) -> None:
        assert kind == "mammen"
        _frozen_fill(rng, out)

    multiplier._fill_multipliers = frozen
    try:
        return multiplier_critical_value(
            curves,
            se,
            n=n,
            cluster=cluster,
            n_replicates=n_replicates,
            kind="mammen",
            random_state=random_state,
        )
    finally:
        multiplier._fill_multipliers = original


@pytest.mark.parametrize("n", [999, 10_000])
@pytest.mark.parametrize("n_replicates", [7, 10_000])
@pytest.mark.parametrize("clustered", [False, True])
def test_the_critical_value_is_bitwise_unchanged(
    n: int, n_replicates: int, clustered: bool
) -> None:
    rng = np.random.default_rng(n + n_replicates)
    curves = rng.normal(size=(n, 4)) * np.array([1.0, 2.0, 0.5, 3.0])
    curves[:, 3] = curves[:, 0] - curves[:, 1]
    se = curves.std(axis=0) / np.sqrt(n)
    cluster = rng.integers(0, n // 7, size=n) if clustered else None
    got = multiplier_critical_value(
        curves,
        se,
        n=n,
        cluster=cluster,
        n_replicates=n_replicates,
        kind="mammen",
        random_state=17,
    )
    expected = _frozen_critical_value(
        curves, se, n=n, n_replicates=n_replicates, random_state=17, cluster=cluster
    )
    assert repr(got) == repr(expected)


def test_swapped_half_words_are_caught() -> None:
    """Mutation control: high-then-low halves draw different multipliers than ``integers``."""
    frozen, mutated = np.random.default_rng(11), np.random.default_rng(11)
    expected = np.empty((8, 101))
    _frozen_fill(frozen, expected)
    words = mutated.bit_generator.random_raw(8 * 101 // 2).view(np.uint32).reshape(-1, 2)
    swapped = words[:, ::-1].reshape(8, 101) < _MAMMEN_THRESHOLD
    got = np.where(swapped, _MAMMEN_LOW, _MAMMEN_HIGH)
    assert got.tobytes() != expected.tobytes()
