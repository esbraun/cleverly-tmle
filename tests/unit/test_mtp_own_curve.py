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

import numpy as np
import pytest

from tests.studies import canonical_mtp_boundaries as boundaries
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_boundaries_common as law_b
from tests.studies import mtp_boundaries_own_curve as own_b
from tests.studies import mtp_pricing_common as law_p
from tests.studies import mtp_pricing_own_curve as own_p

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
