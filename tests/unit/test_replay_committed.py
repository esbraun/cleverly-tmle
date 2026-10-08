"""The column comparison of the relabel check in ``tests/canonical/replay_committed.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.canonical.replay_committed import equal_entries


def test_a_last_bit_difference_passes_against_the_column_scale() -> None:
    left = pd.Series([1.0, 1e-3, 0.0])
    right = pd.Series([1.0 + 2e-16, 1e-3 + 2e-16, 1e-16])
    assert equal_entries(left, right, rtol=1e-12).all()
    assert not equal_entries(left, right, rtol=0.0).all()


def test_an_infinite_entry_does_not_widen_the_tolerance() -> None:
    """The mutation: a scale read with ``nanmax`` is infinite, and every row then passes."""
    left = pd.Series([np.inf, 1.0, 2.0])
    right = pd.Series([np.inf, 1.5, 2.0])
    assert equal_entries(left, right, rtol=1e-12).tolist() == [True, False, True]
    assert not equal_entries(pd.Series([np.inf]), pd.Series([-np.inf]), rtol=1e-12).any()


def test_missing_text_agrees_and_other_text_must_match() -> None:
    left = pd.Series(["a", None, "b"], dtype=object)
    right = pd.Series(["a", np.nan, "c"], dtype=object)
    assert equal_entries(left, right).tolist() == [True, True, False]
