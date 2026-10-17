r"""Does the categorical machinery leave the two-level incremental surface where it was?

``Incremental`` on a treatment with two levels still means ``delta`` times the odds of the
second sorted level.  The categorical change adds a mapping form and a softmax path for three or
more levels.  These tests pin what must not move.

* T16: the first five replicates of the two committed incremental comparator studies
  (``npcausal-incremental`` and ``npcausal-stratified-incremental``), refitted here, equal the
  committed ``cleverly`` rows.  The rows were written by the code before this change.
  The comparison is ``rtol=1e-12``: the rows come from another platform, and the refit differs
  from them in the last bit on this one, as the refit of ``origin/main`` does.  The bits of
  the arithmetic are pinned by ``TestTheBinaryBranchRunsTheOperationsOfTheBinaryCode``.
* C5: every two-level declaration of one odds multiplier gives one tilt, and the general
  (three or more levels) formulas, called on a two-level input, equal the two-level ones.

The general formulas are reachable only for ``K > 2`` in a fit, so C5 calls them directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.mechanism import _covariate_binary, _covariate_general
from cleverly.inference.influence import _ipsi_term_binary, _ipsi_term_general
from cleverly.interventions import Incremental, IPSISet
from cleverly.interventions.incremental import _tilt_binary, _tilt_general
from tests.studies import canonical_incremental_interventions as primary
from tests.studies import canonical_stratified_incremental_msm as stratified
from tests.unit import _categorical_incremental_support as sup

REPLICATES = 5
#: An absolute floor for a value that may be zero.
FLOOR = 1e-12
COMMITTED = {
    primary.STUDY.implementation: "tests/canonical/npcausal_incremental/replicates.csv.gz",
    stratified.STUDY.implementation: (
        "tests/canonical/npcausal_stratified_incremental/replicates.csv.gz"
    ),
}


def _committed(implementation: str, replicate: int) -> pd.DataFrame:
    rows = pd.read_csv(COMMITTED[implementation], float_precision="round_trip")
    rows = rows[(rows["implementation"] == implementation) & (rows["replicate"] == replicate)]
    assert len(rows), "the committed artifact has no rows for this replicate"
    return rows.set_index("estimand")


def _agree(result: Any, committed: pd.DataFrame) -> None:
    for name, row in committed.iterrows():
        estimate = result[str(name)]
        assert estimate.psi == pytest.approx(row["estimate"], rel=1e-12, abs=FLOOR), name
        assert estimate.std_error == pytest.approx(row["std_error"], rel=1e-12, abs=FLOOR), name


class TestT16TheCommittedBinaryStudiesReplay:
    @pytest.mark.parametrize("replicate", range(REPLICATES))
    def test_the_unstratified_study_rows_come_back(self, replicate: int) -> None:
        frame, _ = primary.draw_scenario(primary.SCENARIO, 2000, replicate)
        result = primary.fit_cleverly(frame)
        _agree(result, _committed(primary.STUDY.implementation, replicate))

    @pytest.mark.parametrize("replicate", range(REPLICATES))
    def test_the_stratified_study_rows_come_back(self, replicate: int) -> None:
        frame, _ = stratified.draw_scenario(stratified.SCENARIO, 2000, replicate)
        result = stratified.fit_cleverly(frame)
        _agree(result, _committed(stratified.STUDY.implementation, replicate))


# ------------------------------------------------------------------------- bitwise witness


class TestTheBinaryBranchRunsTheOperationsOfTheBinaryCode:
    """``np.array_equal`` with the expressions of ``origin/main``'s two-level ``_tilt``.

    Elementwise IEEE operations give the same bits on every platform, so this holds in CI.
    """

    @pytest.mark.parametrize("delta", [1.7, 0.43, 6.0, 1.0])
    def test_the_arrays_of_a_two_level_tilt_are_those_of_the_binary_code(
        self, delta: float
    ) -> None:
        rng = np.random.default_rng(20261016)
        g1 = rng.uniform(0.02, 0.98, 3000)
        g = np.column_stack([1.0 - g1, g1])
        g0 = 1.0 - g1
        d = delta * g1 + g0
        values = np.column_stack([g0 / d, delta * g1 / d])
        weights = np.column_stack([1.0 / d, np.full_like(d, delta) / d])
        tilts = IPSISet.evaluate((Incremental(delta, name="t"),), _two_levels()[0], g)
        assert np.array_equal(tilts.values[:, :, 0], values)
        assert np.array_equal(tilts.weights[:, :, 0], weights)
        assert np.array_equal(tilts.binary_derivative[:, 0], delta / d**2)
        assert np.array_equal(tilts.normaliser[:, 0], d)

    def test_a_mapping_naming_the_first_level_gives_the_same_bits_as_its_multiplier(self) -> None:
        rng = np.random.default_rng(7)
        g1 = rng.uniform(0.02, 0.98, 3000)
        g = np.column_stack([1.0 - g1, g1])
        data = _two_levels()[0]
        scalar = IPSISet.evaluate((Incremental(6.0, name="t"),), data, g)
        mapped = IPSISet.evaluate((Incremental({0.0: 0.5, 1.0: 3.0}, name="t"),), data, g)
        assert np.array_equal(mapped.values, scalar.values)
        assert np.array_equal(mapped.weights, scalar.weights)
        assert np.array_equal(mapped.binary_derivative, scalar.binary_derivative)


# ----------------------------------------------------------------------------------- C5


def _two_levels() -> tuple[CausalData, np.ndarray]:
    """Two-level data from the three-arm law (``high`` against the rest) and its mechanism."""
    frame = sup.frame().assign(B=lambda f: (f["A"] == "high").astype(float))
    data = CausalData.from_frame(frame, outcome="Y", treatment="B", covariates=["W"])
    w = frame["W"].to_numpy().astype(int)
    treated = np.array([0.55, 0.30, 0.15])[w]
    return data, np.column_stack([1.0 - treated, treated])


#: Declarations that are one odds multiplier, 2, 4 or 0.25 on the second level.  The
#: factors are powers of two, so the quotients are exact.
SAME_TILT = [
    (2.0, [{1.0: 2.0}, {0.0: 0.5}, {0.0: 2.0, 1.0: 4.0}]),
    (4.0, [{1.0: 4.0}, {0.0: 0.25}, {0.0: 0.5, 1.0: 2.0}]),
    (0.25, [{1.0: 0.25}, {0.0: 4.0}, {0.0: 2.0, 1.0: 0.5}]),
]


class TestC5TwoLevelsAreTheBinaryTilt:
    @pytest.mark.parametrize(("delta", "forms"), SAME_TILT)
    def test_every_declaration_of_one_multiplier_evaluates_to_the_same_arrays(
        self, delta: float, forms: list[dict[float, float]]
    ) -> None:
        data, g = _two_levels()
        reference = IPSISet.evaluate((Incremental(delta, name="t"),), data, g)
        np.testing.assert_allclose(np.asarray(reference.factors[0]), [1.0, delta], atol=0, rtol=0)
        for form in forms:
            tilted = IPSISet.evaluate((Incremental(form, name="t"),), data, g)
            np.testing.assert_allclose(tilted.values, reference.values, atol=1e-14, rtol=0)
            np.testing.assert_allclose(tilted.weights, reference.weights, atol=1e-14, rtol=0)
            np.testing.assert_allclose(tilted.normaliser, reference.normaliser, atol=1e-14, rtol=0)
            np.testing.assert_allclose(
                tilted.binary_derivative, reference.binary_derivative, atol=1e-14, rtol=0
            )

    def test_the_first_level_factor_matters(self) -> None:
        """A branch that read only the second factor would call {0: 2, 1: 4} a 4, not a 2."""
        data, g = _two_levels()
        twice = IPSISet.evaluate((Incremental({0.0: 2.0, 1.0: 4.0}, name="t"),), data, g)
        four = IPSISet.evaluate((Incremental(4.0, name="t"),), data, g)
        assert np.max(np.abs(twice.values - four.values)) > 0.02

    @pytest.mark.parametrize("factors", [(1.0, 2.0), (1.0, 0.25), (0.5, 3.0), (4.0, 0.5)])
    def test_the_general_tilt_equals_the_binary_one_on_two_levels(
        self, factors: tuple[float, float]
    ) -> None:
        _, g = _two_levels()
        values_b, weights_b, normaliser_b = _tilt_binary((factors,), g)
        values_g, weights_g, normaliser_g = _tilt_general((factors,), g)
        np.testing.assert_allclose(values_g, values_b, atol=1e-12, rtol=0)
        np.testing.assert_allclose(weights_g, weights_b, atol=1e-12, rtol=0)
        # The general normaliser is sum_a f_a g_a, which is f_0 times Kennedy's D.
        np.testing.assert_allclose(normaliser_g / factors[0], normaliser_b, atol=1e-12, rtol=0)

    def _targeted(self, g: np.ndarray) -> InitialFit:
        w = np.arange(g.shape[0])
        at_zero = 0.30 + 0.20 * np.sin(w)
        at_one = 0.65 - 0.25 * np.cos(w)
        return InitialFit(observed=at_zero, arms={0.0: at_zero, 1.0: at_one})

    @pytest.mark.parametrize("factors", [(1.0, 2.0), (0.5, 3.0), (4.0, 0.5)])
    def test_the_general_covariate_differs_across_levels_by_the_binary_one(
        self, factors: tuple[float, float]
    ) -> None:
        """``H(1) - H(0)`` is ``delta (Q1 - Q0) / D^2``, the logistic covariate, exactly."""
        _, g = _two_levels()
        tilts = IPSISet(
            ("t",),
            (factors,),
            *_tilt_binary((factors,), g),
            g,
        )
        general_tilts = IPSISet(("t",), (factors,), *_tilt_general((factors,), g), g)
        fit = self._targeted(g)
        binary = _covariate_binary(fit, tilts)
        general = _covariate_general(fit, general_tilts)
        np.testing.assert_allclose(general[:, 1, :] - general[:, 0, :], binary, atol=1e-12, rtol=0)

    @pytest.mark.parametrize("factors", [(1.0, 2.0), (0.5, 3.0), (4.0, 0.5)])
    def test_the_general_mechanism_term_equals_the_binary_one(
        self, factors: tuple[float, float]
    ) -> None:
        """``h(A){Q(A) - m}`` is ``delta (Q1 - Q0)(A - g_1) / D^2``: ``E_g[h(a){Q(a) - m}]`` is 0."""
        data, g = _two_levels()
        tilts = IPSISet(("t",), (factors,), *_tilt_binary((factors,), g), g)
        general_tilts = IPSISet(("t",), (factors,), *_tilt_general((factors,), g), g)
        fit = self._targeted(g)
        predictions = np.column_stack([fit.arms[0.0], fit.arms[1.0]])
        treatment = np.asarray(data.treatment, dtype=float)
        mixture = np.einsum("ij,ij->i", tilts.values[:, :, 0], predictions)
        binary = _ipsi_term_binary(tilts, predictions, mixture, treatment, 0)
        general = _ipsi_term_general(general_tilts, predictions, mixture, treatment, 0)
        assert np.max(np.abs(binary)) > 0.01, "a zero term would hide a wrong sign"
        np.testing.assert_allclose(general, binary, atol=1e-12, rtol=0)

    def test_the_fit_does_not_depend_on_how_the_multiplier_is_declared(self) -> None:
        data, _ = _two_levels()
        frame = sup.frame().assign(B=lambda f: (f["A"] == "high").astype(float))
        fits = []
        for declaration in (2.0, {1.0: 2.0}, {0.0: 0.5}, {0.0: 2.0, 1.0: 4.0}):
            fits.append(
                TMLE(
                    incremental=[Incremental(declaration, name="t"), Incremental(1.0)],
                    outcome_learner=sup.MainTermsOutcome(width=3),
                    treatment_learner=sup.MainTermsTreatment(),
                    cross_fit=False,
                    simultaneous=False,
                    random_state=0,
                )
                .fit(frame, outcome="Y", treatment="B", covariates=["W"])
                .single()
            )
        assert data.n == len(frame)
        first = fits[0]["ey_ipsi[t]"]
        for other in fits[1:]:
            estimate = other["ey_ipsi[t]"]
            assert estimate.psi == pytest.approx(first.psi, abs=1e-12)
            np.testing.assert_allclose(
                estimate.influence_curve, first.influence_curve, atol=1e-12, rtol=0
            )
