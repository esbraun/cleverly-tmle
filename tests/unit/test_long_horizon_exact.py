"""Exact witnesses of the X39 long-horizon law: the oracle, the plans and the weighted frame.

The law of :mod:`tests.discrete_law_longitudinal_horizon` has finite support, so one row per
support point weighted by its mass is the law itself.  A single-fold fit on that frame estimates
the law's own parameter.  These checks hold the oracle to an independent derivative, hold the
two plan representations to each other, and hold ``LTMLE`` and ``SDR`` with saturated learners to
the longhand g-formula and efficient influence function at three to five nodes.

Every fit here declares ``n_folds=1``: with the default ten folds the frame is cross-fitted over
support points, and a known-mechanism fit then misses the truth by 0.016 to 0.032 at K = 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.longitudinal import LTMLE, DynamicRegimen
from tests import discrete_law_longitudinal as two_node
from tests import discrete_law_longitudinal_horizon as H
from tests.studies import long_horizon_common as C
from tests.studies.fractional_glm import QuasiBinomialGLM

SETTINGS = [(3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3)]

PLANS = {
    2: ("never", "always", "respond", "switch", "coin", "coinlast"),
    3: ("low", "high", "respond", "switch", "mix", "mixlast", "taper"),
}

#: The plans that exist only as a mutation or a control candidate.
EXTRA = ("lagged", "reversed", "uniform", "mix_node1", "mix_norot", "mix_reversed", "mix_noprev")


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


# ----------------------------------------------------------------------------- the oracle
@pytest.mark.parametrize(
    ("K", "m", "rows"),
    [(3, 2, 424), (3, 3, 1380), (4, 2, 1704), (4, 3, 8292), (5, 2, 6824), (5, 3, 49764)],
)
def test_the_support_has_the_declared_size_and_mass(K: int, m: int, rows: int) -> None:
    assert H.support(K, m).rows == H.support_size(K, m) == rows
    assert H.masses(K, m, H.L_R).sum() == pytest.approx(1.0, abs=1e-12)
    assert H.masses(K, m, H.L_C).sum() == pytest.approx(1.0, abs=1e-12)


def test_the_truths_do_not_depend_on_the_mechanism() -> None:
    """``B``, ``s``, ``c0`` and ``gamma_C`` move the mechanism only; ``L`` and ``Y`` set the truth."""
    other = H.Law(0.25, 0.0, 5.5, 2.0)
    for name in PLANS[3]:
        assert H.truth(5, 3, name, other) == pytest.approx(H.truth(5, 3, name, H.L_R), abs=1e-15)
    # The mechanism does move: the same plan's curve differs between the two laws.
    assert np.max(np.abs(H.eif(3, 3, "mix", other) - H.eif(3, 3, "mix", H.L_R))) > 1e-2


def test_the_declared_truths_at_five_nodes() -> None:
    """``psi(always) = 0.8242`` and ``psi(never) = 0.5388`` at K = 5; the K = 5 static tail."""
    assert H.truth(5, 2, "always") == pytest.approx(0.8241611824, abs=1e-9)
    assert H.truth(5, 2, "never") == pytest.approx(0.5387997181, abs=1e-9)
    high = H.diagnostics(5, 2, "always", H.L_R)
    assert high["sigma"] == pytest.approx(3.5522744, rel=1e-6)
    assert high["kappa"] == pytest.approx(4719.968, rel=1e-6)
    assert high["n_kappa"] == 75_504


@pytest.mark.parametrize("plan", ["mix", "respond", "high", "taper", "lagged"])
def test_the_longhand_eif_is_the_derivative_of_the_g_formula(plan: str) -> None:
    """Complex-step derivative of the mass-only g-formula at each of the 1,380 support points."""
    K, m = 3, 3
    mass = H.masses(K, m, H.L_R)
    step = 1e-20
    derivative = np.empty(mass.size)
    for i in range(mass.size):
        bumped = mass.astype(complex)
        bumped[i] += 1j * step
        derivative[i] = H.truth_from_masses(K, m, plan, bumped).imag / step
    np.testing.assert_allclose(derivative, H.eif(K, m, plan), rtol=0.0, atol=1e-10)
    assert H.truth_from_masses(K, m, plan, mass) == pytest.approx(H.truth(K, m, plan), abs=1e-14)


def test_a_dropped_node_breaks_the_derivative_check() -> None:
    """Mutation: a curve that loses its node-2 term no longer equals the derivative."""
    K, m, plan = 3, 3, "mix"
    regressions = H.oracle_regressions(K, m, plan)
    exact = H.eif(K, m, plan)
    ratios = H._ratios(K, m, H.gformula(K, m, plan), H.L_R)
    ratios[:, 1] = 0.0
    dropped = H.curve(K, m, plan, H.L_R, regressions, ratios=ratios)
    assert np.max(np.abs(dropped - exact)) > 1e-2


def test_the_weighted_frame_is_the_law_the_sampler_draws() -> None:
    """Cell frequencies of 400,000 draws match the support masses at K = 3, m = 3."""
    K, m = 3, 3
    frame = H.draw(K, m, 400_000, 11, H.L_R)
    support = H.weighted_frame(K, m, H.L_R)
    keys = ["W"] + [
        c for t in range(1, K + 1) for c in ([f"L{t}"] if t > 1 else []) + [f"A{t}", f"C{t}"]
    ]
    keys.append("Y")
    observed = frame[keys].fillna(-1).value_counts(normalize=True)
    expected = support.assign(**{k: support[k].fillna(-1) for k in keys}).set_index(keys)["w"]
    joined = expected.to_frame("p").join(observed.rename("f"), how="left").fillna(0.0)
    assert len(joined) == len(support)
    z = (joined["f"] - joined["p"]) / np.sqrt(joined["p"] * (1 - joined["p"]) / 400_000)
    assert float(np.max(np.abs(z))) < 5.5
    # The probability columns are the law's, row by row.
    sample = frame.dropna(subset=["A2"]).head(50)
    for t in (1, 2):
        lt = sample[f"L{t}"].to_numpy() if t > 1 else np.zeros(len(sample))
        prev = sample[f"A{t - 1}"].to_numpy() if t > 1 else np.zeros(len(sample))
        g = H.treatment_probabilities(t, m, H.L_R, sample["W"].to_numpy(), lt, prev.astype(int))
        np.testing.assert_allclose(sample[[f"g{t}_{k}" for k in range(m)]].to_numpy(), g)


# ----------------------------------------------------------------------------- the two plan tables
@pytest.mark.parametrize(("K", "m"), SETTINGS)
def test_the_estimator_plans_are_the_oracle_plans(K: int, m: int) -> None:
    """Each plan's density on the frame rows equals the oracle's at the row's history."""
    frame = H.weighted_frame(K, m, H.L_R)
    support = H.support(K, m)
    tree = H.histories(K, m)
    names = PLANS[m] + EXTRA + (tuple(f"dur{j}" for j in range(K + 1)) if m == 2 else ())
    for name in names:
        density = H.plan(name, K, m)
        for t in range(1, K + 1):
            reached = support.end >= t
            oracle = density(t, tree)[support.path[reached, t - 1]]
            mine = C.plan_density(frame.loc[reached].reset_index(drop=True), name, K, m, t)
            np.testing.assert_array_equal(mine, oracle, err_msg=f"{name} node {t}")


def test_a_per_node_mutated_plan_differs_only_at_its_node() -> None:
    K, m = 4, 3
    tree = H.histories(K, m)
    for name, declared in (("lagged", "respond"), ("mix_noprev", "mix")):
        for node in range(2, K + 1):
            mutated = H.plan(name, K, m, nodes=(node,))
            base = H.plan(declared, K, m)
            for t in range(1, K + 1):
                same = np.array_equal(mutated(t, tree), base(t, tree))
                assert same == (t != node), (name, node, t)


# ----------------------------------------------------------------------------- the bridge
def test_the_weighted_support_realization_is_the_row_realization() -> None:
    """On the shipped two-node law, one weighted row per support point equals the ``N`` rows.

    A GLM outcome and a learned logistic mechanism, so the weights flow through every learner,
    every fluctuation and the curve.  Estimate and per-unit curve agree to ``rtol=1e-12``.
    """
    rows = two_node.frame()
    points = rows.iloc[two_node.first_row_of()].reset_index(drop=True)
    weighted = points.assign(w=two_node.COUNTS / two_node.N)
    kwargs: dict[str, Any] = {
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": QuasiBinomialGLM(),
        "treatment_learner": QuasiBinomialGLM(),
        "censoring_learner": QuasiBinomialGLM(),
        "n_folds": 1,
        "simultaneous": False,
        "random_state": 0,
    }
    roles: dict[str, Any] = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }
    specs = {
        label: spec
        if not isinstance(spec, tuple)
        else DynamicRegimen(label, spec, rule_kind="known" if any(map(callable, spec)) else None)
        for label, spec in two_node.REGIMEN_SPEC.items()
    }
    full = LTMLE(specs, **kwargs).fit(rows, **roles)
    small = LTMLE(specs, **kwargs).fit(weighted, weights="w", **roles)
    first = two_node.first_row_of()
    weight = weighted["w"].to_numpy()
    moved = 0.0
    for label in two_node.REGIMEN_ARMS:
        name = f"ey_regimen[{label}]"
        assert float(small[name].psi) == pytest.approx(float(full[name].psi), rel=1e-12)
        per_unit = C.per_unit_curve(small, name, weight)
        np.testing.assert_allclose(
            per_unit, np.asarray(full[name].influence_curve)[first], rtol=1e-12, atol=1e-13
        )
        moved = max(moved, abs(float(full[name].psi) - two_node.TRUTH[name]))
    # The GLM is not saturated on this law, so the agreement is not two exact fits agreeing.
    assert moved > 1e-4


# ----------------------------------------------------------------------------- saturated fits
@pytest.mark.parametrize(("K", "m"), SETTINGS)
@pytest.mark.parametrize("estimator", ["LTMLE", "SDR"])
def test_saturated_fits_return_the_g_formula_and_the_eif(K: int, m: int, estimator: str) -> None:
    """Every plan at every K: estimate equals ``truth()``, curve equals ``eif()`` row by row."""
    frame = H.weighted_frame(K, m, H.L_R)
    names = PLANS[m]
    result = C.fit(
        frame,
        K,
        m,
        names,
        estimator=estimator,
        outcome_learner=C.CellMeans(),
        pseudo_learner=C.CellMeans(),
        weights="w",
    )
    weight = frame["w"].to_numpy()
    for name in names:
        report = f"ey_regimen[{name}]"
        assert float(result[report].psi) == pytest.approx(H.truth(K, m, name), abs=1e-10)
        assert _rel(C.per_unit_curve(result, report, weight), H.eif(K, m, name)) < 1e-10


def test_the_frame_route_matches_on_a_mutated_law() -> None:
    """The saturated fit follows the frame, not a law it was told about: the null law moves it."""
    K, m = 3, 2
    null = H.Law(1.0, 0.1, 3.5, null_outcome=True)
    frame = H.weighted_frame(K, m, null)
    result = C.fit(
        frame,
        K,
        m,
        ("never", "always"),
        outcome_learner=C.CellMeans(),
        pseudo_learner=C.CellMeans(),
        weights="w",
    )
    assert float(result["ey_regimen[always]"].psi) == pytest.approx(
        H.truth(K, m, "always", null), abs=1e-12
    )
    assert abs(H.truth(K, m, "always", null) - H.truth(K, m, "always")) > 1e-2
    # Under the sharp null the policy and its last-node stop agree exactly.
    assert H.truth(5, 2, "coin", null) == pytest.approx(H.truth(5, 2, "coinlast", null), abs=1e-15)
    pd.testing.assert_index_equal(frame.columns, H.weighted_frame(K, m, H.L_R).columns)
