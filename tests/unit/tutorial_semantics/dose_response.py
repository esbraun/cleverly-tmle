"""The reviewed semantic callback for ``docs/examples/dose-response``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  Each assertion is a claim the prose makes about this draw, not a
statistical guarantee.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.unit.tutorial_semantics import figure_title

#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS: tuple[str, ...] = ()


def _rows(namespace: dict[str, Any], h: float) -> Any:
    table = namespace["table"]
    return table[table["bandwidth"] == h].set_index("dose")


def _check_data(namespace: dict[str, Any]) -> None:
    """The true curve rises steeply, then flattens; the simple comparison falls instead."""
    true_curve = namespace["true_curve"]
    early = true_curve(15) - true_curve(0)
    late = true_curve(30) - true_curve(15)
    assert early > 2 * late > 0, "most of the gain comes from the first 15 points"
    naive = namespace["naive"]
    assert np.all(np.diff(naive["share_renewed"]) < 0), "the raw share falls with the discount"
    assert np.all(np.diff(naive["true share at band middle"]) > 0), "the true curve rises"
    assert np.all(np.diff(naive["mean_usage"]) < 0), "bigger discounts go to less active users"


def _check_declaration(namespace: dict[str, Any]) -> None:
    """Twelve pairs, and a range inside the observed discounts with a margin."""
    curve, frame = namespace["curve"], namespace["frame"]
    assert len(curve.pairs) == 12
    lower, upper = curve.support
    assert frame["discount"].min() < lower and upper < frame["discount"].max()
    assert curve.nodes_per_piece == 8


def _check_estimate(namespace: dict[str, Any]) -> None:
    """Every interval and band covers the truth; the curve rises from 5% to 20% at bandwidth 5."""
    table = namespace["table"]
    assert ((table["lower"] <= table["true"]) & (table["true"] <= table["upper"])).all()
    assert ((table["band_lower"] <= table["true"]) & (table["true"] <= table["band_upper"])).all()
    wide = _rows(namespace, 5.0)
    assert wide.loc[5.0, "estimate"] < wide.loc[15.0, "estimate"] < wide.loc[20.0, "estimate"]
    assert 0.12 < wide.loc[20.0, "estimate"] - wide.loc[5.0, "estimate"] < 0.14, "about 13 points"
    narrow = _rows(namespace, 2.5)
    for rows in (wide, narrow):
        assert np.all(np.diff(rows.loc[5.0:20.0, "estimate"]) > 0), "both series rise to 20%"
    assert narrow.loc[30.0, "band_lower"] < 0.0 and narrow.loc[30.0, "band_upper"] > 1.0
    caught = [str(warning.message) for warning in namespace["caught"]]
    assert any("3 pair(s)" in message and "below 50" in message for message in caught)


def _check_band(namespace: dict[str, Any]) -> None:
    """The band separates 5% from 20% at both bandwidths; a wider window is not always narrower."""
    result = namespace["result"]
    assert result.band.critical_value > result.band.pointwise_critical_value
    wide, narrow = _rows(namespace, 5.0), _rows(namespace, 2.5)
    assert wide.loc[5.0, "band_upper"] < wide.loc[20.0, "band_lower"]
    assert narrow.loc[5.0, "band_upper"] < narrow.loc[20.0, "band_lower"]
    width = namespace["width"]
    assert width.loc[15.0, 5.0] < width.loc[15.0, 2.5], "usually narrower"
    assert width.loc[25.0, 5.0] > width.loc[25.0, 2.5], "but not at 25%"
    assert wide.loc[25.0, "kish"] < narrow.loc[25.0, "kish"]


def _check_support(namespace: dict[str, Any]) -> None:
    """The two pairs at 30% and the wider window at 25% are below the floor."""
    support, result, curve = namespace["support"], namespace["result"], namespace["curve"]
    flagged = support[support["low_support"]]
    assert sorted(flagged["dose"]) == [25.0, 30.0, 30.0]
    at_30 = flagged[flagged["dose"] == 30.0]
    assert ((at_30["kish"] > 10) & (at_30["kish"] < 25)).all(), "17.8 and 15.9"
    at_25 = flagged[flagged["dose"] == 25.0]
    assert (at_25["bandwidth"] == 5.0).all() and ((at_25["kish"] > 40) & (at_25["kish"] < 50)).all()
    assert set(flagged["status"]) == {"low_dose_support_plugin"}
    assert set(support.loc[~support["low_support"], "status"]) == {"influence_curve"}
    assert curve.support_floor == 50.0
    assert result.quadrature_shift < 0.05, "below the level at which the fit warns"
    # The page reads "supported" above the floor and "experimental" for the flagged pairs and
    # the band that covers them.
    assert set(flagged["label"]) == {"experimental"}
    assert set(support.loc[~support["low_support"], "label"]) == {"supported"}
    assert result.band.support.label == "experimental"
    assert figure_title(namespace["figure"]) == "Smoothed dose-response curve"


def _check_weights(namespace: dict[str, Any]) -> None:
    """The weight is non-negative in the middle and signed at the end of the range."""
    curve, doses = namespace["curve"], namespace["doses"]
    assert curve.weight(5.0, 15.0, doses).min() >= 0.0
    assert -0.01 < curve.weight(5.0, 30.0, doses).min() < 0.0
    end = curve.weight(5.0, 32.0, doses)
    middle = curve.weight(5.0, 15.0, doses)
    assert abs(end.min() + 0.92 / 5.0) < 0.005, "the reference's -0.92/h at the end"
    ratio = np.trapezoid(end**2, doses) / np.trapezoid(middle**2, doses)
    assert 7.3 < ratio < 7.7


def check(namespace: dict[str, Any]) -> None:
    """The dose-response tutorial's claims hold at the displayed sample size."""
    _check_data(namespace)
    _check_declaration(namespace)
    _check_estimate(namespace)
    _check_band(namespace)
    _check_support(namespace)
    _check_weights(namespace)
