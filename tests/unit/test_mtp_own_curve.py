"""The own curves of the boundaries and pricing fits, and the tail sizes read off them.

A cell's tail size is the larger of the efficient influence function's and the own curve's,
``D(Q_inf, r_0)``: the influence curve of the fit with its nuisances at their limits.  The two
studies' learners are parametric, so those limits solve population estimating equations.
``tests/studies/mtp_pricing_own_curve.py`` and ``tests/studies/mtp_boundaries_own_curve.py``
rebuild each fit in numpy so that it can be run on ten million rows (law P) or on the quadrature
grid (law B).  The checks below pin both against ``cleverly`` on a sample, where the two must agree
to rounding, and recompute one law-B tail size cheaply.  The committed tail sizes are in
``tests/diagnostics/fix5_heavy_ratios/law_p_moments.csv``, ``law_p_quadrature.csv`` and
``law_b_own_moments.csv``, written by that directory's ``run.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.studies import canonical_mtp_boundaries as boundaries
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_boundaries_common as law_b
from tests.studies import mtp_boundaries_own_curve as own_b
from tests.studies import mtp_pricing_common as law_p
from tests.studies import mtp_pricing_own_curve as own_p
from tests.studies import mtp_tail_size_common as tail

#: The largest difference from ``cleverly``, in standard deviations of the curve.
AGREEMENT = 1e-8


@pytest.mark.parametrize(
    ("tau", "estimator", "learners"),
    [
        (5, "ltmle", "primary"),
        (5, "sdr", "primary"),
        (3, "ltmle", "revenue"),
        (2, "ltmle", "property"),
    ],
)
def test_the_numpy_pricing_fit_is_cleverlys(tau: int, estimator: str, learners: str) -> None:
    frame = law_p.sample(2_000, 11 + tau, tau)
    settings = {}
    if learners != "primary":
        revenue = learners == "revenue"
        settings = {
            "outcome_learner": law_p.FinalOutcome(revenue=revenue),
            "pseudo_learner": law_p.SplinePseudo(),
            "revenue": revenue,
        }
    alpha = pricing.PRIMARY_ALPHA if estimator == "ltmle" and learners == "primary" else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pricing.fit(frame, tau, estimator=estimator, alpha=alpha, **settings)
    cols = own_p.columns(frame, tau)
    curve = own_p.OwnCurve(tau, estimator, learners).fit(cols)
    mine = curve.quantities(cols)
    scale = 10.0 if learners == "revenue" else 1.0
    for label in law_p.LABELS:
        estimate = result[f"ey_regimen[{label}]"]
        theirs = np.asarray(estimate.influence_curve, dtype=float)
        gap = np.max(np.abs(theirs - scale * mine[f"ey[{label}]"])) / np.std(theirs)
        assert gap < AGREEMENT, label
        assert abs(float(estimate.psi) - scale * curve.plans[label].psi) < AGREEMENT


@pytest.mark.parametrize(
    ("label_law", "labels", "estimator", "learners", "route"),
    [
        ("B", ("natural", "p1", "p4"), "ltmle", "primary", "classifier"),
        ("B", ("natural", "p4"), "sdr", "correct", "classifier"),
        ("B_atom", ("natural", "p5"), "ltmle", "primary", "density"),
        ("B_atom", ("natural", "p5"), "sdr", "correct", "density"),
    ],
)
def test_the_numpy_boundaries_fit_is_cleverlys(
    label_law: str, labels: tuple[str, ...], estimator: str, learners: str, route: str
) -> None:
    frame = law_b.SAMPLERS[label_law](2_000, 21)
    settings: dict[str, object] = {}
    if route == "classifier":
        settings = {"ratio": "classifier", "treatment_learner": law_b.OracleRatio()}
    if learners == "correct" and estimator == "sdr":
        settings["pseudo_learner"] = law_b.UnitRangePseudo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = boundaries.fit(
            frame, label_law, labels, estimator=estimator, outcome=learners, **settings
        )
    ratios = None
    if route == "density":
        ratios = {
            label: (
                result.fits[label].node_ratio[:, 0],
                result.fits[label].node_ratio[:, 1][:, None],
            )
            for label in labels
        }
    block = own_b.sample_block(frame, atom=label_law == "B_atom", ratios=ratios)
    curve = own_b.OwnCurve(estimator, learners, labels).fit(lambda: [block])
    for label in labels:
        estimate = result[f"ey_regimen[{label}]"]
        theirs = np.asarray(estimate.influence_curve, dtype=float)
        gap = np.max(np.abs(theirs - curve.curve(label, block)[:, 0])) / np.std(theirs)
        assert gap < AGREEMENT, label
        assert abs(float(estimate.psi) - curve.plans[label].psi) < AGREEMENT


# ------------------------------------------------------------- the committed tail sizes

DIAGNOSTIC = Path(__file__).resolve().parents[1] / "diagnostics" / "fix5_heavy_ratios"


def test_the_law_p_tail_sizes_are_declared_at_the_upper_bound() -> None:
    frame = pd.read_csv(DIAGNOSTIC / "law_p_moments.csv")
    expected = {
        (tau, learners, estimator)
        for tau, fits in tail.LAW_P_FITS.items()
        for learners, estimator in fits
    }
    assert set(zip(frame.tau, frame.learners, frame.estimator, strict=True)) == expected
    assert (frame.draws == tail.LAW_P_DRAWS).all()
    assert (frame.kappa_lower <= frame.kappa).all()
    assert (frame.kappa <= frame.kappa_upper).all()
    for row in frame.itertuples():
        assert row.n_kappa == tail.tail_size(row.kappa_upper)


def test_the_two_node_monte_carlo_meets_the_quadrature() -> None:
    """At ``tau = 2`` the two orders agree and the quadrature lies in the 99% band."""
    moments = pd.read_csv(DIAGNOSTIC / "law_p_moments.csv").set_index(
        ["tau", "learners", "estimator", "quantity"]
    )
    grid = pd.read_csv(DIAGNOSTIC / "law_p_quadrature.csv")
    orders = sorted(grid.order.unique())
    assert len(orders) == 2
    by_order = [
        grid[grid.order == order].set_index(["learners", "estimator", "quantity"])
        for order in orders
    ]
    np.testing.assert_allclose(by_order[0].kappa, by_order[1].kappa, rtol=1e-6)
    for key, row in by_order[1].iterrows():
        mc = moments.loc[(2, *key)]
        assert mc.kappa_lower <= row.kappa <= mc.kappa_upper, key


def test_the_law_b_own_tail_sizes_agree_across_orders() -> None:
    frame = pd.read_csv(DIAGNOSTIC / "law_b_own_moments.csv")
    keys = ["law", "learners", "estimator", "quantity"]
    low, high = (
        frame[frame.order == order].set_index(keys) for order in sorted(frame.order.unique())
    )
    assert set(low.index) == set(high.index)
    np.testing.assert_allclose(low.kappa, high.loc[low.index].kappa, rtol=1e-6)
    np.testing.assert_allclose(high.mass, 1.0, atol=1e-10)
    for row in high.itertuples():
        assert row.n_kappa == tail.tail_size(row.kappa)


def test_a_law_b_own_tail_size_recomputes() -> None:
    """``SDR`` with the calibration learners on law B, at a coarse order (about 12 s)."""
    committed = pd.read_csv(DIAGNOSTIC / "law_b_own_moments.csv").set_index(
        ["law", "learners", "estimator", "quantity", "order"]
    )
    moments = tail.law_b_own_moments("B", "correct", "sdr", order=16)
    for quantity in ("ey[p4]", "ate[p4]"):
        assert moments[quantity].kappa == pytest.approx(
            committed.loc[("B", "correct", "sdr", quantity, 40), "kappa"], rel=1e-4
        )
