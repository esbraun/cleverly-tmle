"""The declared design of every joint-coverage cell, recomputed before any run reads it.

Each joint cell carries a pointwise control that must establish joint coverage below 0.95.
That needs the family's asymptotic pointwise joint coverage ``p0`` to sit far enough below
0.95 for the declared budget to resolve it.  This module recomputes ``p0``, the oracle band
critical value and the control power of every cell from the family's correlation:

==========================================  =================================================
source of the correlation                   cells
==========================================  =================================================
exact, from a finite-support law's EIF      ``regimens``, the survival and competing-risk
                                            cells, and both strata cells
the reported curves of ten fits at the      ``arms`` and every cell of
cell's own size (a design estimate)         ``default-simultaneous-bands``
==========================================  =================================================

``p0`` and the critical value come from 2,000,000 normal draws at seed 20261002
(:func:`tests.studies.default_band_properties.design`).  The control power is the exact
binomial probability that the 99% Clopper-Pearson upper endpoint falls below 0.95 when the
true joint coverage is ``p0``.  A cell whose correlation is a design estimate is held to the
power at ``p0 + 0.005``, which absorbs the design's own Monte Carlo error.

The strata law also carries two declared design conditions of its own: stratum ATEs at least
0.03 apart, and a marginal-fluctuation control displaced at least 1 SD at n = 2,000 in each
stratum its family reads.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from tests import discrete_law_competing as competing
from tests import discrete_law_longitudinal as longitudinal
from tests import discrete_law_survival as survival
from tests.studies import (
    composite_drtmle_properties,
    default_band_properties,
    ltmle_competing_properties,
    ltmle_properties,
    ltmle_survival_properties,
    multi_arm_mar_drtmle_properties,
    multi_arm_tmle_properties,
    stratified_law,
    stratified_tmle_properties,
)
from tests.studies import mar_arm_indexed_laws as arm_indexed_laws
from tests.studies.canonical_stratified_tmle import NECESSITY_STRATA
from tests.studies.default_band_properties import (
    MINIMUM_CONTROL_POWER,
    SHAPES,
    control_power,
    design,
    design_correlation,
)
from tests.studies.evidence.seeds import stream_seed

#: The allowance on ``p0`` for a correlation that is itself estimated.
DESIGN_ALLOWANCE = 0.005


def _correlation(probs: np.ndarray, curves: list[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(curves)
    covariance = (matrix * probs[:, None]).T @ matrix
    sd = np.sqrt(np.diag(covariance))
    return np.asarray(covariance / np.outer(sd, sd), dtype=float)


def _from_covariance(covariance: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.diag(covariance))
    return np.asarray(covariance / np.outer(sd, sd), dtype=float)


def _exact(module: object, names: tuple[str, ...]) -> np.ndarray:
    return _correlation(module.PROBS, [module.eif(name) for name in names])  # type: ignore[attr-defined]


#: Each exact cell: its correlation, its budget, and its pinned ``(p0, c)``.  The
#: provisional design numbers came from another draw of normals and agree within 0.002.
EXACT: dict[str, tuple[np.ndarray, int, tuple[float, float]]] = {
    "canonical-ltmle/regimens": (
        _exact(
            longitudinal,
            (
                "ey_regimen[never]",
                "ey_regimen[always]",
                "ey_regimen[treat_if_l2]",
                "ate_regimen[always vs never]",
                "ate_regimen[treat_if_l2 vs never]",
            ),
        ),
        ltmle_properties.CALIBRATION_REPLICATES,
        (0.8221, 2.501),
    ),
    "canonical-ltmle-survival/curve_always": (
        _exact(survival, ("risk_regimen[always @ t=1]", "risk_regimen[always @ t=2]")),
        ltmle_survival_properties.CURVE_REPLICATES,
        (0.9038, 2.232),
    ),
    "canonical-ltmle-survival/all_reported": (
        _exact(survival, survival.NAMES),
        ltmle_survival_properties.CALIBRATION_REPLICATES,
        (0.7644, 2.622),
    ),
    "canonical-ltmle-competing/all_reported": (
        _exact(competing, tuple(name for name in competing.NAMES if "t=2" in name)),
        ltmle_competing_properties.CALIBRATION_REPLICATES,
        (0.7576, 2.645),
    ),
    "canonical-ltmle-competing/curve_always": (
        _exact(
            competing,
            tuple(
                f"cif_regimen[always, {cause} @ t={horizon}]"
                for cause in competing.CAUSES
                for horizon in (1, 2)
            ),
        ),
        ltmle_competing_properties.CURVE_REPLICATES,
        (0.8371, 2.457),
    ),
    "canonical-stratified-tmle/strata": (
        _correlation(
            stratified_law.PROBS,
            [stratified_law.eif(name) for name in stratified_law.names()],
        ),
        stratified_tmle_properties.CALIBRATION_REPLICATES,
        (0.6210, 2.846),
    ),
    "canonical-stratified-tmle/crossfit_strata": (
        _correlation(
            stratified_law.PROBS,
            [
                stratified_law.eif(name)
                for name in stratified_law.names()
                if name.split("[V=")[0] in {"ey[1]", "ey[0]", "ate"}
            ],
        ),
        stratified_tmle_properties.CROSSFIT_REPLICATES,
        (0.6609, 2.778),
    ),
    # The nine names of L3 on the inference scale (log for rr and or), from the exact
    # efficient influence covariance of the law.
    "multi-arm-mar-drtmle/arms": (
        _from_covariance(
            arm_indexed_laws.influence_covariance(multi_arm_mar_drtmle_properties.LAW)
        ),
        multi_arm_mar_drtmle_properties.CALIBRATION_REPLICATES,
        (0.8223, 2.508),
    ),
    # The composite study: each scenario's reported names on the inference scale, from
    # the exact efficient influence covariance, whose arm terms divide by g_c = g pi_A pi.
    **{
        f"composite-missing-drtmle/{label}": (
            _from_covariance(
                arm_indexed_laws.influence_covariance(composite_drtmle_properties._keyed(scenario))
            ),
            composite_drtmle_properties.CALIBRATION_REPLICATES,
            declared,
        )
        for label, scenario, declared in (
            ("composite_observational", composite_drtmle_properties.OBSERVATIONAL, (0.8832, 2.326)),
            ("composite_binary", composite_drtmle_properties.BINARY, (0.8835, 2.324)),
            ("composite_three_arm", composite_drtmle_properties.THREE_ARM, (0.8220, 2.508)),
        )
    },
}


@pytest.mark.parametrize("cell", sorted(EXACT))
def test_each_exact_cell_has_a_discriminating_control(cell: str) -> None:
    correlation, replicates, declared = EXACT[cell]
    numbers = design(correlation, replicates)
    assert numbers.p0 == pytest.approx(declared[0], abs=5e-4)
    assert numbers.critical == pytest.approx(declared[1], abs=2e-3)
    assert numbers.p0 < 0.93
    assert numbers.control_power >= MINIMUM_CONTROL_POWER, numbers


def test_the_survival_duplicates_are_exact_identities() -> None:
    """The two duplicate pairs the ``all_reported`` page note describes."""
    for left, right in (
        ("risk_regimen[always @ t=1]", "risk_regimen[continue_if_l2 @ t=1]"),
        ("ate_regimen[always vs never @ t=1]", "ate_regimen[continue_if_l2 vs never @ t=1]"),
    ):
        assert survival.TRUTH[left] == survival.TRUTH[right]
        np.testing.assert_array_equal(survival.eif(left), survival.eif(right))


def _multi_arm_correlation() -> np.ndarray:
    total = None
    for index in range(default_band_properties.DESIGN_FITS):
        frame, _ = multi_arm_tmle_properties.multi_arm_properties.Sampler()(
            multi_arm_tmle_properties.JOINT_N,
            stream_seed(multi_arm_tmle_properties.STUDY, "design", "arms", index),
        )
        result = multi_arm_tmle_properties.fit_joint(frame)
        curves = np.column_stack([result[name].influence_curve for name in result.estimates])
        centred = curves - curves.mean(axis=0)
        covariance = centred.T @ centred / len(centred)
        total = covariance if total is None else total + covariance
    assert total is not None
    sd = np.sqrt(np.diag(total))
    return np.asarray(total / np.outer(sd, sd))


def test_the_multi_arm_cell_has_a_discriminating_control() -> None:
    numbers = design(_multi_arm_correlation(), multi_arm_tmle_properties.JOINT_REPLICATES)
    assert numbers.p0 < 0.93
    power = control_power(numbers.p0 + DESIGN_ALLOWANCE, multi_arm_tmle_properties.JOINT_REPLICATES)
    assert power >= MINIMUM_CONTROL_POWER, numbers


@pytest.mark.parametrize("label", [shape.label for shape in SHAPES])
def test_each_default_band_shape_has_a_discriminating_control(label: str) -> None:
    shape = default_band_properties.SHAPE_BY_LABEL[label]
    numbers = design(design_correlation(label), shape.replicates)
    power = control_power(numbers.p0 + DESIGN_ALLOWANCE, shape.replicates)
    if shape.pointwise_control:
        assert power >= MINIMUM_CONTROL_POWER, (label, numbers)
    else:
        assert power < MINIMUM_CONTROL_POWER, (label, numbers)


# ------------------------------------------------------------------------------ strata law


def _strata_limits() -> dict[int, tuple[float, float, float, float]]:
    """``(truth, control limit, both-wrong limit, SD at n = 2,000)`` per stratum.

    The control limit solves the population version of the longhand marginal fluctuation in
    ``stratified_tmle_properties.marginal_fluctuation``: the logistic projection of ``Y`` on
    ``(A, W)``, then one fluctuation per arm with clever covariate ``I(A = a) / g_a``.  The
    both-wrong limit is the stratum's unadjusted difference in means, which intercept-only
    outcome and treatment models reach.
    """
    support = np.asarray(stratified_law.SUPPORT, dtype=float)
    v, w, a, y = support.T
    probs = stratified_law.PROBS
    projection = LogisticRegression(C=1e12, max_iter=10_000, tol=1e-12).fit(
        np.column_stack([a, w]), y, sample_weight=probs
    )
    q1 = projection.predict_proba(np.column_stack([np.ones_like(w), w]))[:, 1]
    q0 = projection.predict_proba(np.column_stack([np.zeros_like(w), w]))[:, 1]
    g = stratified_law.propensity(w, v)

    def fluctuate(arm: np.ndarray, initial: np.ndarray, covariate: np.ndarray) -> np.ndarray:
        epsilon = 0.0
        for _ in range(100):
            fitted = expit(logit(initial) + epsilon * covariate)
            score = np.sum(probs * arm * covariate * (y - fitted))
            slope = np.sum(probs * arm * covariate**2 * fitted * (1.0 - fitted))
            epsilon += score / slope
        return np.asarray(expit(logit(initial) + epsilon * covariate))

    star1 = fluctuate(a, q1, 1.0 / g)
    star0 = fluctuate(1.0 - a, q0, 1.0 / (1.0 - g))
    out = {}
    for stratum in stratified_law.STRATA:
        inside = v == stratum
        weights = probs[inside] / probs[inside].sum()
        name = stratified_law.stratum_name("ate", stratum)
        unadjusted = np.sum(weights * a[inside] * y[inside]) / np.sum(weights * a[inside]) - np.sum(
            weights * (1.0 - a[inside]) * y[inside]
        ) / np.sum(weights * (1.0 - a[inside]))
        out[stratum] = (
            stratified_law.TRUTH[name],
            float(np.sum(weights * (star1 - star0)[inside])),
            float(unadjusted),
            stratified_law.EFFICIENCY_SD[name] / np.sqrt(stratified_tmle_properties.NECESSITY_N),
        )
    return out


def test_the_stratum_effects_are_resolvably_apart() -> None:
    effects = sorted(
        stratified_law.TRUTH[stratified_law.stratum_name("ate", stratum)]
        for stratum in stratified_law.STRATA
    )
    assert min(np.diff(effects)) >= 0.03


def test_the_marginal_fluctuation_control_is_displaced_in_each_outer_stratum() -> None:
    limits = _strata_limits()
    for stratum in NECESSITY_STRATA:
        truth, control, _, sd = limits[stratum]
        assert abs(control - truth) / sd >= 1.0, (stratum, (control - truth) / sd)


def test_the_middle_stratum_control_could_not_fail() -> None:
    """Why the family reads the outer strata: the middle displacement stays under 0.3 SD."""
    truth, control, _, sd = _strata_limits()[1]
    assert abs(control - truth) / sd < 0.3


def test_the_both_wrong_control_is_displaced_in_every_stratum() -> None:
    for stratum, (truth, _, unadjusted, sd) in _strata_limits().items():
        assert abs(unadjusted - truth) / sd >= 0.4, stratum
