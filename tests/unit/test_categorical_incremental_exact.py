r"""Is the categorical incremental fit the efficient estimator of the tilted mean?

``Incremental({level: factor, ...})`` on a treatment with three levels, against a
complex-step Gateaux derivative of the tilted functional

.. math::

    \Psi(\delta) = E_W\!\left[\sum_a \frac{\delta_a g(a \mid W)}{\sum_b \delta_b g(b \mid W)}
                     \bar Q(a, W)\right]

written longhand on the laws of :mod:`tests.discrete_law_multi` and
:mod:`tests.discrete_law_mar_multi`.  The oracle reads :math:`g` from the perturbed law, so the
mechanism term of the influence function is in the derivative without being told about it.

* C1: complete outcomes, oracle nuisances, estimate and curve against the oracle;
* C1m: outcomes missing at random (``delta=``);
* C1k: a declared known mechanism: no mechanism term, and the curve of the known
  :class:`~cleverly.interventions.Stochastic` regime at the tilted density;
* C1s: baseline strata, one parameter per stratum;
* C4: the identity, all factors equal, which is :math:`\bar Y` for any nuisances;
* C6: the collapsing identity, equal non-baseline factors against a binary fit;
* C7: the exact remainder, in both directions;
* C10: the overlap diagnostics;
* C13: an all-categorical vector treatment is one categorical treatment over its tuples.

The weak-learner witnesses, in which the targeting step moves, are in
:mod:`tests.unit.test_categorical_incremental_targeting`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

import tests.discrete_law_policy_point as vector_law
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import ipsi_means
from cleverly.interventions import Incremental, IPSISet, Stochastic, check_incremental_support
from tests import discrete_law_categorical_incremental as law
from tests import discrete_law_multi as base
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_longitudinal_multivalue import CellProbabilities
from tests.unit import _categorical_incremental_support as sup
from tests.unit._natural_course_support import Counting, CountingLogistic

NAMES = law.MEANS + law.CONTRASTS


@pytest.fixture(scope="module")
def point() -> Any:
    return sup.exact_fit()


@pytest.fixture(scope="module")
def missing() -> Any:
    return sup.exact_fit(missing=True)


def _sorted_factors(tilt: dict[str, float]) -> tuple[float, ...]:
    """The factors in arm-code order: ``CausalData`` sorts the labels."""
    return tuple(tilt.get(label, 1.0) for label in base.SORTED_LABELS)


# ----------------------------------------------------------------------- the premises


class TestTheLibraryAndTheOracleDescribeTheSameTilts:
    """The join between the oracle's labels and the codes the library resolves them to."""

    def test_the_resolved_factors_follow_the_sorted_labels(self, point: Any) -> None:
        tilts = point.nuisance.incremental
        assert tuple(point.data.treatment_levels) == base.SORTED_LABELS
        for name, factors in zip(tilts.names, tilts.factors, strict=True):
            assert factors == _sorted_factors(law.TILTS[name])

    def test_a_label_keyed_tilt_is_not_a_position_keyed_one(self) -> None:
        """The labels sort to ``(high, low, mid)``: by position T_b would tilt ``high``."""
        by_label = _sorted_factors(law.T_B)
        by_position = tuple(law.T_B.get(label, 1.0) for label in base.LABELS)
        assert by_label != by_position

    def test_the_tilted_density_is_the_odds_tilt_of_the_oracle_mechanism(self, point: Any) -> None:
        tilts = point.nuisance.incremental
        w = np.rint(point.data.covariates[:, 0]).astype(int)
        g = np.column_stack([base.G_EXACT[w, base.ARM_OF_CODE[c]] for c in range(base.K)])
        np.testing.assert_allclose(tilts.propensity, g, atol=1e-12, rtol=0)
        for r, _ in enumerate(tilts.names):
            f = np.asarray(tilts.factors[r])
            np.testing.assert_allclose(
                tilts.values[:, :, r], g * f / (g @ f)[:, None], atol=1e-14, rtol=0
            )

    def test_the_premises_of_the_oracle_hold(self) -> None:
        for name in NAMES:
            assert float((base.PROBS.reshape(-1) * law.eif_of(name)).sum()) == pytest.approx(
                0.0, abs=1e-12
            )


# ----------------------------------------------------------------------------- C1, C1m


@pytest.mark.parametrize("fixture", ["point", "missing"])
class TestC1TheFitIsTheEfficientEstimator:
    @staticmethod
    def _mode(fixture: str) -> bool:
        return fixture == "missing"

    @pytest.mark.parametrize("name", NAMES)
    def test_the_estimate_is_the_functional(
        self, request: pytest.FixtureRequest, fixture: str, name: str
    ) -> None:
        fit = request.getfixturevalue(fixture)
        expected = law.truth_of(name, missing=self._mode(fixture))
        assert fit.estimates[name].psi == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("name", NAMES)
    def test_the_curve_is_the_gateaux_derivative_at_every_support_point(
        self, request: pytest.FixtureRequest, fixture: str, name: str
    ) -> None:
        fit = request.getfixturevalue(fixture)
        missing = self._mode(fixture)
        reported = sup.at_support(fit.estimates[name].influence_curve, missing=missing)
        np.testing.assert_allclose(reported, law.eif_of(name, missing=missing), atol=1e-10, rtol=0)

    def test_the_mechanism_score_equation_is_solved(
        self, request: pytest.FixtureRequest, fixture: str
    ) -> None:
        fluctuation = request.getfixturevalue(fixture).fluctuations["ipsi"]
        assert fluctuation.mechanism is not None, "the ipsi group must target g too"
        assert fluctuation.mechanism.relative_score < 1e-9
        assert fluctuation.mechanism.propensity.shape[1] == 3

    def test_a_categorical_fit_is_labelled_experimental_until_a_study_covers_it(
        self, request: pytest.FixtureRequest, fixture: str
    ) -> None:
        fit = request.getfixturevalue(fixture)
        for name in NAMES:
            assert fit.estimates[name].support.label == "experimental"


# ---------------------------------------------------------------------------------- C1k


class TestC1kAKnownMechanismHasNoMechanismTerm:
    @pytest.fixture(scope="class")
    def known(self) -> Any:
        return sup.exact_fit(known=True)

    @pytest.mark.parametrize("name", law.MEANS)
    def test_the_curve_is_the_curve_without_the_mechanism_term(self, known: Any, name: str) -> None:
        tilt = law.TILTS[name[len("ey_ipsi[") : -1]]
        expected = law.eif(tilt) - law.mechanism_term(tilt)
        reported = sup.at_support(known.estimates[name].influence_curve)
        np.testing.assert_allclose(reported, expected, atol=1e-12, rtol=0)

    def test_no_mechanism_equation_is_solved(self, known: Any) -> None:
        assert known.nuisance.incremental.mechanism_known
        assert known.fluctuations["ipsi"].mechanism is None

    @pytest.mark.parametrize("name", law.MEANS)
    def test_it_is_the_curve_of_the_known_stochastic_regime(self, known: Any, name: str) -> None:
        """The known regime that assigns the tilted density: same mean, same curve."""
        label = name[len("ey_ipsi[") : -1]
        f = np.asarray(_sorted_factors(law.TILTS[label]))

        def density(covariates: pd.DataFrame) -> np.ndarray:
            w = np.rint(covariates["W"].to_numpy(dtype=float)).astype(int)
            g = np.column_stack([base.G[w, base.ARM_OF_CODE[c]] for c in range(base.K)])
            return np.asarray(g * f / (g @ f)[:, None])

        regime = (
            TMLE(
                outcome_learner=base.OracleMultiOutcome(),
                treatment_learner=base.OracleMultiTreatment(),
                cross_fit=False,
                interventions=[Stochastic(density, label, density_kind="known")],
                simultaneous=False,
                random_state=0,
            )
            .fit(base.frame(), outcome="Y", treatment="A", covariates=["W"])
            .single()
        )
        mean = regime.estimates[f"ey_regime[{label}]"]
        assert known.estimates[name].psi == pytest.approx(mean.psi, abs=1e-12)
        np.testing.assert_allclose(
            known.estimates[name].influence_curve, mean.influence_curve, atol=1e-12, rtol=0
        )


# ---------------------------------------------------------------------------------- C1s


class TestC1sBaselineStrata:
    @pytest.fixture(scope="class")
    def stratified(self) -> Any:
        return sup.exact_fit(strata=True)

    @pytest.mark.parametrize("stratum", [0, 1])
    @pytest.mark.parametrize("label", list(law.TILTS))
    def test_each_stratum_is_the_stratum_functional(
        self, stratified: Any, stratum: int, label: str
    ) -> None:
        estimate = stratified[f"ey_ipsi[{label}][V={stratum}]"]
        tilt = law.TILTS[label]
        assert estimate.psi == pytest.approx(law.truth_stratum(tilt, stratum), abs=1e-12)
        reported = sup.at_support(estimate.influence_curve)
        np.testing.assert_allclose(reported, law.eif_stratum(tilt, stratum), atol=1e-10, rtol=0)

    def test_the_stratum_curve_lives_inside_its_stratum(self) -> None:
        """The oracle curve is nonzero inside the stratum and zero outside it."""
        for stratum in (0, 1):
            curve = law.eif_stratum(law.T_A, stratum)
            inside = np.array([law.STRATA_OF_W[w] == stratum for w, _, _ in base.SUPPORT])
            assert np.max(np.abs(curve[inside])) > 0.02
            np.testing.assert_allclose(curve[~inside], 0.0, atol=1e-14)

    def test_an_absent_level_in_a_stratum_refuses_before_any_learner(self) -> None:
        """Stratum 1 holds no ``low`` or ``mid`` row: its softmax equation may have no root.

        The refusal is the data-only guard.  Whether a root exists depends on the fitted outcome
        regression, see :func:`~cleverly.estimators._strata.refuse_absent_tilt_arms`.
        """
        data = sup.frame(strata=True)
        data["A"] = np.where(data["V"] == 1, "high", data["A"])
        Counting.calls = 0
        with pytest.raises(
            DataError, match="treatment-mechanism score equation may have no finite root"
        ):
            TMLE(
                incremental=sup.declared(),
                outcome_learner=CountingLogistic(max_iter=1000),
                treatment_learner=CountingLogistic(max_iter=1000),
                cross_fit=False,
                simultaneous=False,
            ).fit(data, outcome="Y", treatment="A", covariates=["W", "V"], strata=["V"])
        assert Counting.calls == 0


# ----------------------------------------------------------------------------------- C4


class TestC4EqualFactorsAreTheIdentity:
    """Every factor equal leaves the mechanism alone: ``psi = mean(Y)``, curve ``Y - mean(Y)``."""

    @pytest.mark.parametrize("scale", [1.0, 3.0])
    def test_saturated_learners(self, scale: float) -> None:
        tilt = Incremental(dict.fromkeys(base.LABELS, scale), name="all equal")
        fit = sup.exact_fit([tilt])
        y = np.asarray(fit.data.outcome, dtype=float)
        estimate = fit.estimates["ey_ipsi[all equal]"]
        assert estimate.psi == pytest.approx(float(y.mean()), abs=1e-12)
        np.testing.assert_allclose(estimate.influence_curve, y - y.mean(), atol=1e-12, rtol=0)

    def test_the_default_name_is_the_odds_form_unless_every_factor_is_one(self) -> None:
        assert Incremental({"low": 2.0, "mid": 2.0}).name == "odds low x2, mid x2"
        assert Incremental({"low": 1.0, "mid": 1.0}).name == "natural course"


# ----------------------------------------------------------------------------------- C6


class _BinaryOutcome(BaseEstimator):
    """``E[Y | B, W]`` of the collapsed law: the base law's cell means pooled over ``A != low``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _BinaryOutcome:
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        b, w = design[:, 0].astype(int), np.rint(design[:, 1]).astype(int)
        return np.asarray(_COLLAPSED_Q[w, b])


class _BinaryTreatment(BaseEstimator):
    """``P(B = b | W)`` of the collapsed law."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _BinaryTreatment:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        w = np.rint(np.asarray(X, dtype=float)[:, 0]).astype(int)
        return np.column_stack([1.0 - _COLLAPSED_G[w], _COLLAPSED_G[w]])


def _collapse() -> tuple[np.ndarray, np.ndarray]:
    low = base.LABELS.index("low")
    others = [a for a in range(base.K) if a != low]
    g = base.G_EXACT[:, others].sum(axis=1)
    q = (base.G_EXACT[:, others] * base.Q_EXACT[:, others]).sum(axis=1) / g
    return g, np.column_stack([base.Q_EXACT[:, low], q])


_COLLAPSED_G, _COLLAPSED_Q = _collapse()


class TestC6TheCollapsingIdentity:
    """Equal factors on the non-baseline levels tilt ``1{A != low}`` like a binary treatment."""

    @staticmethod
    def _binary_fit(delta: float) -> Any:
        frame = base.frame().assign(B=lambda f: (f["A"] != "low").astype(float))
        return (
            TMLE(
                outcome_learner=_BinaryOutcome(),
                treatment_learner=_BinaryTreatment(),
                cross_fit=False,
                incremental=[Incremental(delta, name="collapsed")],
                simultaneous=False,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="B", covariates=["W"])
            .single()
        )

    def test_the_estimate_and_the_curve_equal_the_binary_fit(self) -> None:
        tilt = Incremental(law.COLLAPSED, name="collapsed")
        fit = sup.exact_fit([tilt])
        binary = self._binary_fit(4.0)
        left, right = fit.estimates["ey_ipsi[collapsed]"], binary.estimates["ey_ipsi[collapsed]"]
        assert left.psi == pytest.approx(right.psi, abs=1e-12)
        np.testing.assert_allclose(left.influence_curve, right.influence_curve, atol=1e-12, rtol=0)

    def test_unequal_factors_do_not_collapse(self) -> None:
        """The nonzero control: the same baseline with unequal factors differs by more than 0.01."""
        assert abs(law.truth(law.UNEQUAL) - law.truth(law.COLLAPSED)) > 0.01
        fit = sup.exact_fit([Incremental(law.UNEQUAL, name="unequal")])
        binary = self._binary_fit(4.0)
        gap = fit.estimates["ey_ipsi[unequal]"].psi - binary.estimates["ey_ipsi[collapsed]"].psi
        assert abs(gap) > 0.01


# ----------------------------------------------------------------------------------- C7

WRONG_G = law.WRONG_G
WRONG_Q = base.Q + np.array([[0.10, -0.15, 0.05], [-0.20, 0.10, 0.05], [0.05, 0.10, -0.15]])


def _expansion(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    """``R = psi_hat - psi + P_0 phi_hat`` per tilt, from the library on the exact sample.

    ``g_hat`` and ``q_hat`` are ``(W, label)`` tables on the law's own arm axis.
    """
    frame = base.frame()
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"])
    w = frame["W"].to_numpy().astype(int)
    order = list(base.ARM_OF_CODE)
    mechanism = g_hat[w][:, order]
    arms = {float(code): q_hat[w, order[code]] for code in range(base.K)}
    treatment = np.asarray(data.treatment, dtype=float)
    observed = np.array([arms[float(a)][i] for i, a in enumerate(treatment)])
    initial = InitialFit(observed=observed, arms=arms)
    tilts = IPSISet.evaluate(sup.declared(), data, mechanism)
    submodel = submodel_for(
        "ipsi", treatment, mechanism, arms=data.arm_codes, incremental=tilts.weights
    )
    means = ipsi_means(
        np.asarray(data.outcome, dtype=float),
        initial,
        submodel,
        tilts,
        treatment,
        np.ones(base.N),
    )
    return {
        name: means[float(r)].psi
        - law.truth(law.TILTS[name])
        + float(np.mean(means[float(r)].influence_curve))
        for r, name in enumerate(tilts.names)
    }


def _closed_form(g_hat: np.ndarray, q_hat: np.ndarray) -> dict[str, float]:
    r"""``E_0[(m_hat - m_0)(D_hat - D_0)/D_hat]`` per tilt, written longhand in the oracle."""
    return {name: law.remainder(tilt, g_hat, q_hat) for name, tilt in law.TILTS.items()}


class TestC7TheExactRemainder:
    @pytest.mark.parametrize("name", list(law.TILTS))
    def test_both_wrong_matches_the_closed_form(self, name: str) -> None:
        actual = _expansion(WRONG_G, WRONG_Q)[name]
        assert actual == pytest.approx(_closed_form(WRONG_G, WRONG_Q)[name], abs=1e-12)

    @pytest.mark.parametrize("name", ["T_a", "T_b"])
    def test_a_consistent_mechanism_kills_it_whatever_qbar_does(self, name: str) -> None:
        assert _expansion(base.G, WRONG_Q)[name] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("name", ["T_a", "T_b"])
    def test_a_consistent_qbar_does_not(self, name: str) -> None:
        actual = _expansion(WRONG_G, base.Q)[name]
        assert actual == pytest.approx(_closed_form(WRONG_G, base.Q)[name], abs=1e-12)
        assert abs(actual) > 3e-3, "the misspecification must be large enough to test anything"

    def test_the_natural_course_has_no_remainder_for_any_nuisances(self) -> None:
        assert _expansion(WRONG_G, WRONG_Q)["natural course"] == pytest.approx(0.0, abs=1e-12)

    def test_both_right_is_zero(self) -> None:
        for value in _expansion(base.G, base.Q).values():
            assert value == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------------- C10


class TestC10TheOverlapDiagnostics:
    def test_the_covariate_stays_inside_the_guaranteed_bounds(self, point: Any) -> None:
        reports = point.diagnostics.support()
        assert set(reports) == set(law.TILTS)
        for name, report in reports.items():
            f = np.asarray(_sorted_factors(law.TILTS[name]))
            low, high = report.guaranteed
            assert low == pytest.approx(f.min() / f.max(), abs=0)
            assert high == pytest.approx(f.max() / f.min(), abs=0)
            weights = point.nuisance.incremental.weights[:, :, list(law.TILTS).index(name)]
            assert weights.min() >= low - 1e-12
            assert weights.max() <= high + 1e-12
            assert report.delta == tuple(
                zip(base.SORTED_LABELS, _sorted_factors(law.TILTS[name]), strict=True)
            )

    def test_the_minimum_propensity_is_the_smallest_entry_of_the_mechanism(
        self, point: Any
    ) -> None:
        smallest = float(np.min(point.nuisance.incremental.propensity))
        for report in point.diagnostics.support().values():
            assert report.min_propensity == smallest
            (table,) = report.summary().tables
            assert table.rows[0][table.columns.index("min probability")].value == smallest

    def test_two_levels_report_the_binary_pair(self) -> None:
        frame = base.frame().assign(B=lambda f: (f["A"] != "low").astype(float))
        data = CausalData.from_frame(frame, outcome="Y", treatment="B", covariates=["W"])
        g = np.column_stack([1.0 - _COLLAPSED_G, _COLLAPSED_G])[frame["W"].to_numpy().astype(int)]
        for delta in (0.25, 1.0, 4.0):
            tilts = IPSISet.evaluate((Incremental(delta, name="t"),), data, g)
            report = check_incremental_support(tilts, data.treatment, levels=data.treatment_levels)[
                "t"
            ]
            assert report.guaranteed == (min(delta, 1.0 / delta), max(delta, 1.0 / delta))


# ---------------------------------------------------------------------------------- C13

#: Tilts by observed (flag, dose) tuple.  The doses are the labels of the law's component.
_TUPLE_FACTORS = {(0.0, 0.0): 0.5, (1.0, 1.0): 2.0}


def _vector_data(frame: pd.DataFrame) -> CausalData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalData.from_frame(frame, outcome="Y", treatment=["F", "D"], covariates=["W"])


def _combined_data(frame: pd.DataFrame) -> CausalData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"])


class TestC13AnAllCategoricalVectorIsOneCategoricalTreatment:
    """The tilt over observed tuples equals the tilt over the combined column."""

    @staticmethod
    def _fit(data: CausalData, factors: dict[Any, float]) -> Any:
        return (
            TMLE(
                outcome_learner=CellMeans(),
                treatment_learner=CellProbabilities(),
                cross_fit=False,
                incremental=[Incremental(factors, name="tilted")],
                estimands=["ey_ipsi"],
                simultaneous=False,
                random_state=0,
            )
            .fit(data)
            .single()
        )

    def test_the_vector_fit_equals_the_fit_on_the_combined_column(self) -> None:
        frame = vector_law.mixed_frame()
        frame = frame.assign(A=lambda f: 3.0 * f["F"] + f["D"])
        vector_data = _vector_data(frame)
        assert vector_data.treatment_kind == "discrete"
        levels = tuple(vector_data.treatment_levels)
        assert all(key in levels for key in _TUPLE_FACTORS), levels
        combined_factors = {3.0 * f + d: value for (f, d), value in _TUPLE_FACTORS.items()}
        vector = self._fit(vector_data, _TUPLE_FACTORS)
        combined = self._fit(_combined_data(frame), combined_factors)
        left, right = vector.estimates["ey_ipsi[tilted]"], combined.estimates["ey_ipsi[tilted]"]
        assert left.psi == pytest.approx(right.psi, abs=1e-12)
        np.testing.assert_allclose(left.influence_curve, right.influence_curve, atol=1e-12, rtol=0)
        # The tilt moved the mean: this is not the natural course in disguise.
        natural = float(np.mean(np.asarray(vector_data.outcome, dtype=float)))
        assert abs(left.psi - natural) > 1e-3

    def test_a_key_that_is_not_an_observed_tuple_is_refused(self) -> None:
        vector_data = _vector_data(vector_law.mixed_frame())
        with pytest.raises(DataError, match="not a level of"):
            TMLE(
                outcome_learner=CellMeans(),
                treatment_learner=CellProbabilities(),
                cross_fit=False,
                incremental=[Incremental({(9.0, 9.0): 2.0})],
            ).fit(vector_data)
