"""Duplicate covariates are found through digests, confirmed byte for byte.

The test compares with the real digest and under forced digest collisions, with signed zeros,
a column that is both constant and duplicated, and the retained memory.
"""

from __future__ import annotations

import tracemalloc
import warnings

import numpy as np

from cleverly.data import validate
from cleverly.data.validate import check_covariates


def _checked(matrix: np.ndarray, names: list[str]):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = check_covariates(matrix, names)
    return result, [str(w.message) for w in caught]


def test_duplicates_are_found_through_digest_collisions(monkeypatch) -> None:
    rng = np.random.default_rng(5)
    a, b, c = rng.normal(size=(3, 50))
    zero = np.zeros(50)
    zero[1] = 1.0
    negative = zero.copy()
    zero[0], negative[0] = 0.0, -0.0
    matrix = np.column_stack([a, b, b, np.full(50, 2.0), zero, negative, c, a])
    names = ["a", "b", "b2", "const", "zero", "negzero", "c", "a2"]
    reference = _checked(matrix, names)
    monkeypatch.setattr(validate, "_column_digest", lambda column: 0)
    (kept, retained, dropped), messages = _checked(matrix, names)
    assert dropped == ["const", "b2", "a2"]  # constants first, then duplicates in order
    assert retained == ["a", "b", "zero", "negzero", "c"]  # signed zeros differ as bytes
    assert kept.tobytes() == reference[0][0].tobytes()
    assert (retained, dropped, messages) == (reference[0][1], reference[0][2], reference[1])


def test_a_constant_duplicate_is_dropped_as_a_constant() -> None:
    matrix = np.column_stack([np.arange(5.0), np.ones(5), np.ones(5)])
    (_, retained, dropped), _ = _checked(matrix, ["x", "one", "one2"])
    assert retained == ["x"]
    assert dropped == ["one", "one2"]


def test_stored_covariates_do_not_follow_the_callers_array() -> None:
    from cleverly.data import CausalData

    rng = np.random.default_rng(9)
    covariates = rng.normal(size=(200, 3))
    data = CausalData.from_arrays(
        outcome=rng.binomial(1, 0.5, 200).astype(float),
        treatment=rng.binomial(1, 0.5, 200).astype(float),
        covariates=covariates,
    )
    before = np.array(data.covariates, copy=True)
    covariates[:] = 0.0
    np.testing.assert_array_equal(data.covariates, before)


def test_duplicate_detection_keeps_no_copy_of_every_column() -> None:
    rng = np.random.default_rng(13)
    n, p = 200_000, 20
    matrix = rng.normal(size=(n, p))
    tracemalloc.start()
    try:
        check_covariates(matrix, [f"w{j}" for j in range(p)])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The retained output is one 8 * n * p copy. Byte keys of every column would add another
    # 8 * n * p on top; the digests keep at most a column or two of scratch.
    assert peak < 8 * n * p * 1.4
