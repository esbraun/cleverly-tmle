r"""C2, C2s, C4, C5 and C10 for a categorical tilt over time, on exactly realised laws.

Each law of :mod:`tests.discrete_law_longitudinal_categorical_incremental` is realised exactly
in its sample, so a saturated learner returns the law's own nuisances.  Kennedy's estimator
then reports the categorical odds-tilt functional, and its curve is the efficient influence
function.  The oracle states each parameter as a g-formula over finite-support masses and reads
the treatment mechanism off the perturbed law, so its complex-step derivative carries the
mechanism term at every tilt node.  A fit that dropped or mis-weighted the term fails the curve
comparison.

=====  ==========================================================================================
Id     Assertion
=====  ==========================================================================================
C2     estimate and curve on the end-of-study law, uncensored and with dropout
C2s    the same on the survival law and on the competing-risk law, per horizon
C4     all factors equal: the estimate is the sample mean and the curve is ``Y - mean``
C5     two levels: every declaration of one odds multiplier gives identical arrays, and the
       general formulas, called on a two-level input, equal the two-level ones
C10    ``IPSINodeRow`` recovers the mechanism and the node factor, and the factor lies inside
       the bound the declaration guarantees
C14    integer weights are row duplication, ``id=`` changes the variance only, five exact
       copies as five folds give the exact cross-fitted fit, and a bootstrap replicate is the
       fit on the resampled rows
=====  ==========================================================================================
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.inference.bootstrap import _bootstrap_design, _bootstrap_draw
from cleverly.longitudinal.ipsi import (
    mechanism_term_binary,
    mechanism_term_general,
    tilt_factor_binary,
    tilt_factor_general,
)
from cleverly.longitudinal.regimen import tilted_density_binary, tilted_density_general
from tests import discrete_law_longitudinal_categorical_incremental as law
from tests.unit import _categorical_ipsi_support as sup

pytestmark = pytest.mark.xdist_group("categorical_ipsi_exact")

NAMES = law.names(law.PLANS, law.REFERENCE)


def _end() -> tuple[Any, sup.Expected]:
    fit = sup.fit_ipsi(law.end_frame(), law.PLANS)
    return fit, sup.expected(NAMES, law.end_function, law.PROBS, law.COUNTS)


def _uncensored() -> tuple[Any, sup.Expected]:
    def function(name: str) -> Any:
        return law.end_function(name, base=law.UNCENSORED_PROBS)

    fit = sup.fit_ipsi(law.uncensored_frame(), law.PLANS, sup.UNCENSORED_COLUMNS)
    return fit, sup.expected(NAMES, function, law.UNCENSORED_PROBS, law.UNCENSORED_COUNTS)


def _survival() -> tuple[Any, sup.Expected]:
    names = law.names(law.EVENT_PLANS, law.REFERENCE, law.HORIZONS)
    fit = sup.fit_ipsi(
        law.survival_frame(),
        law.EVENT_PLANS,
        sup.SURVIVAL_COLUMNS,
        horizons=list(law.HORIZONS),
    )
    return fit, sup.expected(names, law.survival_function, law.SURVIVAL_PROBS, law.SURVIVAL_COUNTS)


def _competing() -> tuple[Any, sup.Expected]:
    fit = sup.fit_ipsi(
        law.competing_frame(),
        law.EVENT_PLANS,
        sup.competing_columns(),
        horizons=list(law.HORIZONS),
    )
    return fit, sup.expected(
        law.competing_names(), law.competing_function, law.COMPETING_PROBS, law.COMPETING_COUNTS
    )


CASES = {"end": _end, "uncensored": _uncensored, "survival": _survival, "competing": _competing}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request: Any) -> tuple[str, Any, sup.Expected]:
    fit, expected = CASES[request.param]()
    return request.param, fit, expected


def test_every_reported_parameter_is_checked(case: Any) -> None:
    _, fit, expected = case
    assert fit.config.estimator == "ipsi"
    assert set(fit.estimates) == {name for name, *_ in expected}


def test_c2_the_estimate_is_the_tilt_functional(case: Any) -> None:
    """C2 and C2s: the estimate is the g-formula under the categorical tilt, to ``1e-12``."""
    _, fit, expected = case
    for name, truth, _, _ in expected:
        assert fit.psi(name) == pytest.approx(truth, abs=sup.POINT), name


def test_c2_the_curve_is_the_gateaux_derivative(case: Any) -> None:
    """C2 and C2s: the curve is the efficient influence function, mechanism terms included."""
    _, fit, expected = case
    for name, _, eif, rows in expected:
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], eif, atol=sup.CURVE, rtol=0, err_msg=name
        )


def test_no_node_fits_a_fluctuation(case: Any) -> None:
    _, fit, _ = case
    for regimen_fit in fit.fits.values():
        for step in regimen_fit.steps:
            assert step.fluctuation is None
            np.testing.assert_array_equal(step.targeted, step.initial)


def test_the_estimate_is_the_mean_of_phi(case: Any) -> None:
    _, fit, _ = case
    for regimen_fit in fit.fits.values():
        first = regimen_fit.steps[0]
        mean = float(np.average(first.transform, weights=fit.data.weights))
        assert regimen_fit.psi_scaled == pytest.approx(mean, abs=1e-14)


def test_the_tilt_is_resolved_by_label_not_by_position() -> None:
    """The raw arm order and the sorted order differ, so a positional read lands elsewhere."""
    assert tuple(sorted(law.ARM_LABELS)) != law.ARM_LABELS
    fit, expected = _end()
    by_label = {name: truth for name, truth, _, _ in expected}
    assert fit.psi("ey_regimen[T_a]") == pytest.approx(by_label["ey_regimen[T_a]"], abs=sup.POINT)


# ------------------------------------------------------------------ C4 identity


def test_c4_equal_factors_are_the_natural_course_and_the_sample_mean() -> None:
    """All factors equal: the tilt changes nothing, so the estimate is :math:`\\bar Y`."""
    frame = law.uncensored_frame()
    equal = dict.fromkeys(law.ARM_LABELS, 3.0)
    plans = {"equal": (law.tilt(equal), law.tilt(equal)), "natural": law.PLANS["natural"]}
    fit = sup.fit_ipsi(frame, plans, sup.UNCENSORED_COLUMNS)
    y = frame["Y"].to_numpy(dtype=float)
    for label in plans:
        name = f"ey_regimen[{label}]"
        assert fit.psi(name) == pytest.approx(float(np.mean(y)), abs=sup.POINT)
        np.testing.assert_allclose(
            fit.influence_curves[name], y - np.mean(y), atol=sup.POINT, rtol=0
        )


# ------------------------------------------------------------------ C5 binary reduction


#: Declarations of one odds multiplier 2 on the second level.  The quotients are exact in
#: floating point, so the arrays are equal and not only close.
TWO_FORMS = [2.0, {1.0: 2.0}, {0.0: 0.5}, {0.0: 2.0, 1.0: 4.0}]


@pytest.mark.parametrize("declaration", TWO_FORMS[1:], ids=str)
def test_c5_every_declaration_of_one_multiplier_gives_identical_arrays(declaration: Any) -> None:
    reference = sup.binary_fit(2.0)
    other = sup.binary_fit(declaration)
    for name in ("ey_regimen[tilted]", "ate_regimen[tilted vs natural]"):
        assert other.psi(name) == reference.psi(name)
        np.testing.assert_array_equal(
            other.influence_curves[name], reference.influence_curves[name]
        )


def test_c5_the_first_level_factor_moves_the_fit() -> None:
    """A branch that read only the second factor would call ``{0: 2, 1: 4}`` a 4, not a 2."""
    twice = sup.binary_fit({0.0: 2.0, 1.0: 4.0})
    four = sup.binary_fit(4.0)
    assert abs(twice.psi("ey_regimen[tilted]") - four.psi("ey_regimen[tilted]")) > 1e-3


@pytest.mark.parametrize("factors", [(1.0, 2.0), (1.0, 0.25), (0.5, 3.0), (4.0, 0.5)])
def test_c5_the_general_formulas_equal_the_binary_ones_on_two_levels(
    factors: tuple[float, float],
) -> None:
    rng = np.random.default_rng(11)
    pi = rng.uniform(0.05, 0.95, size=40)
    g = np.column_stack([1.0 - pi, pi])
    f = np.asarray(factors)
    delta = f[1] / f[0]
    arm = rng.integers(0, 2, size=40).astype(float)
    by_arm = rng.uniform(0.0, 1.0, size=(40, 2))
    density = tilted_density_binary(pi, delta)
    np.testing.assert_allclose(tilted_density_general(g, f), density, atol=1e-12, rtol=0)
    # delta_A / D is scale free: the first-level factor divides numerator and normaliser alike.
    np.testing.assert_allclose(
        tilt_factor_general(arm, g, f), tilt_factor_binary(arm, pi, delta), atol=1e-12, rtol=0
    )
    marginal = (density * by_arm).sum(axis=1)
    general = mechanism_term_general(arm, g, by_arm, marginal, f)
    binary = mechanism_term_binary(arm, pi, by_arm, delta)
    assert np.max(np.abs(binary)) > 0.01, "a zero term would hide a wrong sign"
    np.testing.assert_allclose(general, binary, atol=1e-12, rtol=0)


# ------------------------------------------------------------------------------------ C14

#: A weight or a cluster must move the estimate by more than this, or the identity is vacuous.
WITNESS = 1e-3


def _gap(left: Any, right: Any) -> float:
    return max(abs(left.psi(name) - right.psi(name)) for name in NAMES)


class TestC14Compositions:
    """The compositions the binary fit supports, on the censored three-level law."""

    @staticmethod
    def _weights(n: int) -> np.ndarray:
        return np.random.default_rng(1).integers(1, 4, size=n).astype(float)

    def test_integer_weights_equal_row_duplication(self) -> None:
        frame = law.end_frame()
        weights = self._weights(len(frame))
        weighted = sup.fit_ipsi(frame.assign(wt=weights), law.PLANS, declared={"weights": "wt"})
        duplicated = sup.fit_ipsi(
            frame.loc[frame.index.repeat(weights.astype(int))].reset_index(drop=True), law.PLANS
        )
        assert _gap(weighted, duplicated) < 1e-12
        assert _gap(weighted, sup.fit_ipsi(frame, law.PLANS)) > WITNESS

    def test_singleton_clusters_are_the_iid_fit(self) -> None:
        frame = law.end_frame()
        iid = sup.fit_ipsi(frame, law.PLANS)
        singleton = sup.fit_ipsi(
            frame.assign(cid=np.arange(len(frame))), law.PLANS, declared={"id": "cid"}
        )
        for name in NAMES:
            assert singleton.psi(name) == pytest.approx(iid.psi(name), abs=1e-12)
            assert singleton.estimates[name].std_error == pytest.approx(
                iid.estimates[name].std_error, abs=1e-12
            )
            np.testing.assert_allclose(
                singleton.influence_curves[name], iid.influence_curves[name], atol=1e-12, rtol=0
            )

    def test_the_cluster_standard_error_is_the_longhand_formula(self) -> None:
        frame = law.end_frame()
        ids = np.arange(len(frame)) // 4
        clustered = sup.fit_ipsi(frame.assign(cid=ids), law.PLANS, declared={"id": "cid"})
        iid = sup.fit_ipsi(frame, law.PLANS)
        n, n_clusters = len(frame), int(ids.max()) + 1
        name = NAMES[0]
        sums = pd.Series(np.asarray(clustered.influence_curves[name])).groupby(ids).sum()
        longhand = np.sqrt(np.sum(sums.to_numpy() ** 2) * n_clusters / (n_clusters - 1)) / n
        assert clustered.estimates[name].std_error == pytest.approx(longhand, abs=1e-12)
        assert clustered.estimates[name].std_error > 1.5 * iid.estimates[name].std_error

    def test_five_exact_copies_as_five_folds_give_the_exact_cross_fitted_fit(self) -> None:
        copies = pd.concat(
            [law.end_frame().assign(cid=copy) for copy in range(5)], ignore_index=True
        )
        fit = sup.fit_ipsi(copies, law.PLANS, declared={"id": "cid"}, n_folds=5)
        for name, truth, _, _ in sup.expected(NAMES, law.end_function, law.PROBS, law.COUNTS):
            assert fit.psi(name) == pytest.approx(truth, abs=1e-12), name

    def test_a_bootstrap_replicate_is_the_fit_on_the_resampled_rows(self) -> None:
        frame = law.end_frame()
        plain = sup.fit_ipsi(frame, law.PLANS, random_state=0)
        fit = sup.fit_ipsi(frame, law.PLANS, random_state=0, n_bootstrap=3)
        name = NAMES[1]
        assert fit.psi(name) == plain.psi(name)
        summary = fit.estimates[name].bootstrap
        assert summary.n_failed == 0
        design = _bootstrap_design(fit.data, n_replicates=3, resampling="auto", random_state=0)
        for draw, value in zip(design.draws, summary.draws, strict=True):
            index, _ = _bootstrap_draw(
                fit.data.n, design.cluster, np.random.default_rng(draw.sequence)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resampled = sup.fit_ipsi(frame.iloc[index].reset_index(drop=True), law.PLANS)
            assert value == pytest.approx(resampled.psi(name), abs=1e-10)
        assert np.ptp(summary.draws) > WITNESS
