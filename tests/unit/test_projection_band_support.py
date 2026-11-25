"""The projection band's own support (P1 PR 2, plan 3.3 and 4.3).

``effect_curve(grid=)`` attaches :func:`cleverly._support.band_support` to its band, read on
the coefficients' row with the triple ``(kind, "identity", "normal")`` and the draw count.
``projection-evidence`` registered the shipped band on ``T49``, ``T50``, ``T49.clustered`` and
``T49.repeats`` (``cleverly._support_rows.PROJECTION_BAND_CELLS``), so the band reads supported
on exactly that key and experimental on a neighbour key or on a row without the band.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pytest

from cleverly import CausalStudy
from cleverly import _support as support_module
from cleverly._summary import vocabulary
from tests.studies import conditional_projection as projection

GRID = {"W1": np.linspace(-0.8, 0.8, 5), "W3": np.zeros(5)}
OPTION = ("rademacher", "identity", "normal")


@functools.cache
def _fit() -> Any:
    group = projection.GROUP_BY_LABEL["obs_4k"]
    frame = projection.draw_from_seed("obs_4k", 800, 0)[0]
    effect = CausalStudy(frame, design=projection.design(group)).identify(
        projection.estimand(group)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(**{**projection.learners(group), "random_state": 0})


@functools.cache
def _pooled(cross_fit: bool = True) -> Any:
    """The same fit with one pooled outcome regression: row ``T49.pooled``, which has no band."""
    group = projection.GROUP_BY_LABEL["obs_4k"]
    frame = projection.draw_from_seed("obs_4k", 800, 0)[0]
    effect = CausalStudy(frame, design=projection.design(group)).identify(
        projection.estimand(group)
    )
    settings = {**projection.learners(group), "random_state": 0, "outcome_by_arm": False}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(**settings, **({} if cross_fit else {"cross_fit": False}))


def test_the_registered_row_lists_the_projection_triple_at_its_draws() -> None:
    row = next(row for row in support_module.registered_rows() if row.id == "T49")
    assert "band" in row.kinds and "projection-evidence" in row.evidence
    assert row.band_options == (OPTION,)
    assert row.band_option_draws == ((OPTION, 1000),)


def test_the_band_reads_supported_on_exactly_the_declared_key() -> None:
    assert {e.support.row for e in _fit().estimates.values()} == {"T49"}
    curve = _fit().effect_curve(grid=GRID, random_state=0)
    assert curve.status == "supported"
    assert curve.band.support.label == "supported", curve.band.support.reason
    assert curve.band.support.row == "T49"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"kind": "mammen"}, "multiplier_kind='mammen'"),
        ({"kind": "normal"}, "multiplier_kind='normal'"),
        ({"n_replicates": 999}, "measured this band at 1,000 multiplier draws"),
    ],
)
def test_a_neighbour_key_reads_experimental(change: dict[str, Any], reason: str) -> None:
    band = _fit().effect_curve(grid=GRID, random_state=0, **change).band
    assert band.support.label == "experimental"
    assert reason in band.support.reason
    # More draws than the row measured still read it.
    if "n_replicates" in change:
        more = _fit().effect_curve(grid=GRID, random_state=0, n_replicates=2000).band
        assert more.support.label == "supported"


def test_a_band_on_a_row_without_the_band_reads_experimental() -> None:
    """The pooled coefficients read their own row, which licenses no band."""
    pooled = _pooled()
    assert {e.support.label for e in pooled.estimates.values()} == {"supported"}
    assert {e.support.row for e in pooled.estimates.values()} == {"T49.pooled"}
    band = pooled.effect_curve(grid=GRID, random_state=0).band
    assert band.support.label == "experimental"
    assert "Row T49.pooled licenses no simultaneous band" in band.support.reason


def test_a_band_reads_its_coefficients_experimental_status() -> None:
    """A band whose coefficients match no row reads their status: the pooled fit in sample."""
    band = _pooled(cross_fit=False).effect_curve(grid=GRID, random_state=0).band
    assert band.support.label == "experimental"
    assert "No registered study covers this configuration" in band.support.reason


def test_a_band_built_outside_a_fit_carries_no_support() -> None:
    curve = _fit().effect_curve(random_state=0).model
    assert curve.band(GRID, random_state=0).support is None


def _band_note(curve: Any) -> str:
    return next(note.text for note in curve.summary().notes if note.code == "curve_band")


def test_the_note_names_the_configuration_of_an_unmeasured_band() -> None:
    text = _band_note(_pooled().effect_curve(grid=GRID, random_state=0))
    assert text.endswith(vocabulary.CURVE_BAND_UNMEASURED.format(kind="rademacher", draws="1,000"))
    assert "No registered study measures its joint coverage" in text


def test_the_note_of_a_measured_band_says_so() -> None:
    text = _band_note(_fit().effect_curve(grid=GRID, random_state=0))
    assert "No registered study" not in text
    assert text.endswith(vocabulary.CURVE_BAND_MEASURED.format(kind="rademacher", draws="1,000"))
