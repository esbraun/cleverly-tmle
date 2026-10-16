r"""C9, C10, C11, C12, C13 and the support label for a categorical odds tilt over time.

=====  ==========================================================================================
Id     Assertion
=====  ==========================================================================================
C9     a scalar at a node with other than two levels (R1) and a key that is no level of the node
       (R2) are refused before any learner, with the node named; the declaration resolves per
       node; a mixed node meets its own refusal first
C10    ``IPSINodeRow`` recovers the node factor and the mechanism share on the exact law, the
       factor lies in the bound the declaration guarantees, and the bound is the binary pair
       at two levels
C11    ``CausalStudy`` routes ``IncrementalEffect`` and ``IncrementalMean`` on a three-level
       longitudinal design to the direct fit
C12    a declared mechanism at a three-level tilt node: ``LTMLE`` equals the hand-built known
       ``Stochastic`` regimen, and the fit has no mechanism term there
C13    an all-categorical vector node is one categorical node over its observed tuples
-----  ------------------------------------------------------------------------------------------
label  a three-level tilt fit is experimental unless a ``G42`` row of the categorical-incremental
       study matches it
=====  ==========================================================================================
"""

from __future__ import annotations

import warnings
from functools import cache
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from cleverly import (
    CausalStudy,
    CrossFitting,
    IncrementalEffect,
    IncrementalMean,
    IPSIMethod,
    LongitudinalTreatment,
)
from cleverly._support import registered_rows
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import Incremental, Stochastic
from cleverly.longitudinal import IPSI, LTMLE, DynamicRegimen, ParametricGFormula
from tests import discrete_law_longitudinal_categorical_incremental as law
from tests import discrete_law_longitudinal_multivalue as multi
from tests.longitudinal_ipsi_categorical import SORTED_LABELS, sorted_factors
from tests.unit import _categorical_ipsi_support as sup
from tests.unit.test_parametric_gformula import NeverFit as GFormulaNeverFit
from tests.unit.test_parametric_gformula import panel_b, quiet_fit

pytestmark = pytest.mark.xdist_group("categorical_ipsi_declaration")

LEVELS = list(SORTED_LABELS)


class NeverFit:
    """A learner that fails the test if a refusal comes after the nuisance fit."""

    def fit(self, *args: Any, **kwargs: Any) -> NeverFit:
        raise AssertionError("a learner ran before the refusal")

    def predict(self, x: Any) -> Any:
        raise AssertionError("a learner ran before the refusal")

    def predict_proba(self, x: Any) -> Any:
        raise AssertionError("a learner ran before the refusal")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


def _never(regimens: Any, frame: pd.DataFrame) -> Any:
    """Fit ``regimens`` on ``frame`` with learners that fail the test if any one runs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IPSI(
            regimens,
            outcome_learner=NeverFit(),
            pseudo_learner=NeverFit(),
            treatment_learner=NeverFit(),
            n_folds=1,
            simultaneous=False,
        ).fit(frame, **sup.UNCENSORED_COLUMNS)


@cache
def _frame() -> pd.DataFrame:
    return law.uncensored_frame()


def _two_levels(node: str) -> pd.DataFrame:
    """The law's sample with ``low`` recoded to ``high`` at ``node``: that node has two levels."""
    frame = _frame().copy()
    frame[node] = frame[node].replace({"low": "high"})
    return frame


# ------------------------------------------------------------------ C9


class TestC9RefusalsAtANode:
    def test_r1_names_the_node_and_the_levels(self) -> None:
        with pytest.raises(DataError) as raised:
            _never({"t": Incremental(2.0)}, _frame())
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert raised.value.refusal.condition == (
            "a scalar odds multiplier names two arms; node 'A1' has 3 levels "
            "(['high', 'low', 'standard']). Declare one factor per level, for example "
            "Incremental({'high': 2.0}), which multiplies the odds of 'high' against every "
            "unnamed level."
        )

    def test_r2_names_the_key_the_node_and_the_levels(self) -> None:
        with pytest.raises(DataError) as raised:
            _never({"t": Incremental({"hi": 2.0})}, _frame())
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert raised.value.refusal.condition == (
            "Incremental names 'hi', which is not a level of node 'A1'; its levels are "
            "['high', 'low', 'standard']."
        )

    def test_a_position_is_not_a_label(self) -> None:
        with pytest.raises(DataError, match="not a level of node 'A1'"):
            _never({"t": Incremental({0: 2.0})}, _frame())

    def test_a_valid_mapping_reaches_the_first_learner(self) -> None:
        """The control: the same call with a level key is not refused."""
        with pytest.raises(AssertionError, match="a learner ran before the refusal"):
            _never({"t": Incremental({"high": 2.0})}, _frame())

    def test_one_declaration_is_resolved_at_each_node_with_that_nodes_levels(self) -> None:
        """A key absent at the second node meets R2 there, not at the first."""
        with pytest.raises(DataError) as raised:
            _never({"t": Incremental({"low": 2.0})}, _two_levels("A2"))
        assert raised.value.refusal.condition == (
            "Incremental names 'low', which is not a level of node 'A2'; its levels are "
            "['high', 'standard']."
        )

    def test_a_scalar_is_valid_at_a_two_level_node_and_refused_at_a_three_level_one(self) -> None:
        plans = {"t": DynamicRegimen("t", ("high", Incremental(2.0)), rule_kind="known")}
        with pytest.raises(AssertionError, match="a learner ran before the refusal"):
            _never(plans, _two_levels("A2"))
        reverse = {"t": DynamicRegimen("t", (Incremental(2.0), "high"), rule_kind="known")}
        with pytest.raises(DataError, match="node 'A1' has 3 levels"):
            _never(reverse, _frame())

    def test_the_first_bad_node_refuses_first(self) -> None:
        with pytest.raises(DataError, match="node 'A2' has 3 levels"):
            _never({"t": Incremental(2.0)}, _two_levels("A1"))

    def test_a_node_that_is_not_a_tilt_reads_no_factors(self) -> None:
        """A static label at a node reads no factor, so a scalar elsewhere is the only read."""
        plans = {"t": DynamicRegimen("t", ("low", Incremental({"high": 2.0})), rule_kind="known")}
        with pytest.raises(AssertionError, match="a learner ran before the refusal"):
            _never(plans, _frame())


# ------------------------------------------------------------------ the support label


class TestTheSupportLabel:
    def test_a_three_level_tilt_fit_is_experimental(self) -> None:
        fit = sup.fit_ipsi(_frame(), law.PLANS, sup.UNCENSORED_COLUMNS)
        assert fit.estimates
        for estimate in fit.estimates.values():
            assert estimate.support.label == "experimental"
            assert estimate.support.row is None

    def test_only_the_categorical_incremental_rows_cover_a_node_of_other_than_two_levels(
        self,
    ) -> None:
        """A three-level IPSI row exists only where the study ``categorical-incremental`` is."""
        rows = [row for row in registered_rows() if "ipsi" in row.key.engine]
        assert any(set(row.key.treatment) == {"2"} for row in rows), "the binary IPSI rows"
        wider = [row for row in rows if set(row.key.treatment) != {"2"}]
        assert wider, "the categorical-incremental study registers three-level rows"
        for row in wider:
            assert row.id.startswith("G42."), row.id
            assert set(row.key.treatment) == {"3"}, row.id


# ------------------------------------------------------------------ C10


def _factor_bound(factors: np.ndarray) -> tuple[float, float]:
    """The interval ``[min f / max f, max f / min f]`` that holds every node factor."""
    return float(factors.min() / factors.max()), float(factors.max() / factors.min())


def _node_factors(frame: pd.DataFrame, factors: dict[str, float]) -> dict[int, np.ndarray]:
    """``delta_A / D`` of every row at each node, from the sample's own cell frequencies."""
    weight = sorted_factors(factors)
    out = {}
    for time, (node, keys) in enumerate((("A1", ["W"]), ("A2", ["W", "A1", "L2"])), start=1):
        mechanism = np.column_stack(
            [
                frame.groupby(keys)[node].transform(lambda s, level=level: (s == level).mean())
                for level in SORTED_LABELS
            ]
        )
        own = frame[node].map(dict(zip(SORTED_LABELS, weight, strict=True))).to_numpy(dtype=float)
        out[time] = own / (mechanism @ weight)
    return out


class TestC10NodeDiagnostics:
    @pytest.fixture(scope="class")
    def fit(self) -> Any:
        return sup.fit_ipsi(_frame(), law.PLANS, sup.UNCENSORED_COLUMNS)

    @pytest.mark.parametrize("label", ["T_a", "T_b", "per_node"])
    def test_the_node_factor_is_recovered_from_the_fitted_density(
        self, fit: Any, label: str
    ) -> None:
        frame = _frame()
        plan = law.PLANS[label]
        rows = {
            (row.regimen, row.time): row.ipsi
            for row in fit.diagnostics.support().rows
            if row.ipsi is not None
        }
        first = _node_factors(frame, plan[0][1])[1]
        second = _node_factors(frame, plan[1][1])[2]
        for time, factors in ((1, first), (2, second)):
            report = rows[(label, time)]
            assert report.tilt_factor_min == pytest.approx(factors.min(), abs=1e-12)
            assert report.tilt_factor_max == pytest.approx(factors.max(), abs=1e-12)
            assert report.propensity_degenerate == 0.0

    @pytest.mark.parametrize("label", ["T_a", "T_b", "per_node"])
    def test_the_factor_lies_in_the_bound_the_declaration_guarantees(
        self, fit: Any, label: str
    ) -> None:
        plan = law.PLANS[label]
        rows = {
            (row.regimen, row.time): row.ipsi
            for row in fit.diagnostics.support().rows
            if row.ipsi is not None
        }
        for time, node in ((1, plan[0]), (2, plan[1])):
            low, high = _factor_bound(sorted_factors(node[1]))
            report = rows[(label, time)]
            assert low - 1e-12 <= report.tilt_factor_min <= report.tilt_factor_max <= high + 1e-12
            assert high > 1.5, "a bound near one would hold for a constant factor"

    def test_the_bound_is_attained_where_one_level_has_all_the_mass(self) -> None:
        """A degenerate mechanism puts every row on one level, so the factor is ``1``."""
        f = np.array([2.0, 0.5, 1.0])
        for level in range(3):
            g = np.eye(3)[level]
            assert f[level] / (g @ f) == pytest.approx(1.0)
        assert _factor_bound(f) == (0.25, 4.0)

    def test_the_bound_is_the_binary_pair_at_two_levels(self) -> None:
        for delta in (0.25, 0.5, 2.0, 7.0):
            assert _factor_bound(np.array([1.0, delta])) == (
                min(delta, 1.0 / delta),
                max(delta, 1.0 / delta),
            )

    def test_the_mechanism_share_is_the_oracle_share(self, fit: Any) -> None:
        names = law.names(law.PLANS, law.REFERENCE)
        assert names
        rows = {
            (row.regimen, row.time): row.ipsi
            for row in fit.diagnostics.support().rows
            if row.ipsi is not None
        }
        probs = law.UNCENSORED_PROBS
        for label in ("T_a", "per_node"):
            plan = law.PLANS[label]

            def at(p: Any, known: tuple[int, ...] = (), plan: Any = plan) -> Any:
                return law.functional(p, plan, base=probs, known=known)

            full = law.eif(at, probs)
            keep = probs > 0
            for node in (1, 2):
                term = full - law.eif(lambda p, node=node: at(p, (node,)), probs)
                mean_term = np.sum(probs[keep] * term[keep])
                var_term = np.sum(probs[keep] * (term[keep] - mean_term) ** 2)
                mean = np.sum(probs[keep] * full[keep])
                var = np.sum(probs[keep] * (full[keep] - mean) ** 2)
                assert rows[(label, node)].mechanism_share == pytest.approx(
                    var_term / var, abs=1e-9
                )

    def test_a_static_node_reports_no_factor(self, fit: Any) -> None:
        rows = {
            (row.regimen, row.time): row.ipsi
            for row in fit.diagnostics.support().rows
            if row.ipsi is not None
        }
        report = rows[("tilt_then_static", 2)]
        assert np.isnan(report.tilt_factor_min) and np.isnan(report.tilt_factor_max)
        assert report.mechanism_share == 0.0 or report.mechanism_share < 1e-12


# ------------------------------------------------------------------ C11


_TILTS = [
    Incremental({"high": 1.0}, name="natural"),
    Incremental({"high": 2.0, "low": 0.5}, name="ta"),
    Incremental({"standard": 0.25}, name="tb"),
]
_DESIGN = LongitudinalTreatment(
    outcome="Y",
    treatment=("A1", "A2"),
    baseline=("W",),
    time_varying=((), ("L2",)),
)
_LEARNERS: dict[str, Any] = {
    "outcome_learner": sup.CellMeans(),
    "pseudo_learner": sup.CellMeans(),
    "treatment_learner": multi.CellProbabilities(),
    "g_bounds": sup.NO_TRUNCATION,
    "simultaneous": False,
}


class TestC11CausalStudy:
    @pytest.mark.parametrize("estimand", [IncrementalEffect, IncrementalMean])
    def test_the_study_routes_to_the_direct_fit(self, estimand: Any) -> None:
        target = (
            estimand(_TILTS, reference="natural")
            if estimand is IncrementalEffect
            else (estimand(_TILTS))
        )
        effect = CausalStudy(_frame(), design=_DESIGN).identify(target)
        methods = {record.name: record for record in effect.available_methods()}
        assert methods["ipsi"].available and methods["ipsi"].reason is None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            workflow = effect.estimate(
                method=IPSIMethod(cross_fitting=CrossFitting(enabled=False)), **_LEARNERS
            )
            direct = IPSI(
                {tilt.name: tilt for tilt in _TILTS},
                reference="natural",
                n_folds=1,
                **_LEARNERS,
            ).fit(_frame(), **sup.UNCENSORED_COLUMNS)
        assert workflow.config.estimator == "ipsi"
        assert workflow.estimates
        for name, estimate in workflow.estimates.items():
            assert estimate.psi == direct[name].psi
            np.testing.assert_array_equal(estimate.influence_curve, direct[name].influence_curve)

    def test_a_key_that_is_no_level_is_refused_by_the_study_fit_before_a_learner(self) -> None:
        effect = CausalStudy(_frame(), design=_DESIGN).identify(
            IncrementalMean([Incremental({"nope": 2.0}, name="bad")])
        )
        with pytest.raises(DataError, match="'nope', which is not a level of node 'A1'"):
            effect.estimate(
                method=IPSIMethod(cross_fitting=CrossFitting(enabled=False)),
                outcome_learner=NeverFit(),
                pseudo_learner=NeverFit(),
                treatment_learner=NeverFit(),
                simultaneous=False,
            )

    def test_one_declaration_meets_each_node_through_the_study(self) -> None:
        effect = CausalStudy(_two_levels("A2"), design=_DESIGN).identify(
            IncrementalMean([Incremental({"low": 2.0}, name="low")])
        )
        with pytest.raises(DataError, match="which is not a level of node 'A2'"):
            effect.estimate(
                method=IPSIMethod(cross_fitting=CrossFitting(enabled=False)),
                outcome_learner=NeverFit(),
                pseudo_learner=NeverFit(),
                treatment_learner=NeverFit(),
                simultaneous=False,
            )


# ------------------------------------------------------------------ C12

_KNOWN_COLUMNS = {
    "outcome": "Y",
    "treatment": ("A1", "A2"),
    "baseline": ("W",),
    "time_varying": ((), ("L2",)),
}


def _known_density(node: int, factors: dict[str, float]) -> Any:
    """The tilted known mechanism of ``node``, as a density on the sorted label axis."""
    f = sorted_factors(factors)
    tables = law.UNCENSORED_TABLES
    order = [law.ARM_LABELS.index(label) for label in SORTED_LABELS]

    def density(history: pd.DataFrame) -> np.ndarray:
        w = np.rint(history["W"].to_numpy(dtype=float)).astype(int)
        if node == 1:
            g = tables["G1"][w][:, order]
        else:
            l2 = np.rint(history["L2"].to_numpy(dtype=float)).astype(int)
            raw1 = history["A1"].map(law.ARM_LABELS.index).to_numpy(dtype=int)
            g = tables["G2"][w, raw1, l2][:, order]
        return np.asarray(g * f[None, :] / (g @ f)[:, None])

    return density


class TestC12KnownMechanism:
    def _fit(self, regimens: Any, frame: pd.DataFrame, **declared: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LTMLE(
                regimens,
                outcome_learner=sup.CellMeans(),
                pseudo_learner=sup.CellMeans(),
                treatment_learner=multi.CellProbabilities(),
                n_folds=1,
                g_bounds=sup.NO_TRUNCATION,
                simultaneous=False,
            ).fit(frame, **_KNOWN_COLUMNS, **declared)

    def test_a_declared_mechanism_makes_the_tilt_a_known_stochastic_regimen(self) -> None:
        frame = sup.known_frame()
        known = sup.KNOWN_DECLARED
        factors = law.T_A
        tilt = self._fit({"ta": Incremental(factors, name="ta")}, frame, **known)
        by_hand = self._fit(
            {
                "ta": DynamicRegimen(
                    "ta",
                    (
                        Stochastic(_known_density(1, factors), "p1", density_kind="known"),
                        Stochastic(_known_density(2, factors), "p2", density_kind="known"),
                    ),
                )
            },
            frame,
            **known,
        )
        assert tilt.estimates
        for name, estimate in tilt.estimates.items():
            assert estimate.psi == pytest.approx(by_hand[name].psi, abs=1e-12), name
            np.testing.assert_allclose(
                estimate.influence_curve, by_hand[name].influence_curve, atol=1e-12, rtol=0
            )

    def test_the_known_fit_has_no_mechanism_term_and_differs_from_the_estimated_one(self) -> None:
        """The estimated-mechanism curve has the term, so the two curves differ."""
        frame = sup.known_frame()
        known = sup.KNOWN_DECLARED
        with_known = self._fit({"ta": Incremental(law.T_A, name="ta")}, frame, **known)
        probs = law.UNCENSORED_PROBS
        plan = law.PLANS["T_a"]
        full = law.eif(lambda p: law.functional(p, plan, base=probs), probs)
        held = law.eif(lambda p: law.functional(p, plan, base=probs, known=(1, 2)), probs)
        assert np.nanmax(np.abs(full - held)) > 0.02, "a vanishing term would hide a kept one"
        rows = law.first_rows(law.UNCENSORED_COUNTS)
        keep = probs > 0
        np.testing.assert_allclose(
            with_known.influence_curves["ey_regimen[ta]"][rows[keep]],
            held[keep],
            atol=1e-10,
            rtol=0,
        )


# ------------------------------------------------------------------ C13


def _vector_frame() -> pd.DataFrame:
    """The sample with each three-level node split into two binary components."""
    frame = _frame().copy()
    code = {"standard": (0.0, 0.0), "high": (1.0, 0.0), "low": (0.0, 1.0)}
    for time in (1, 2):
        frame[f"A{time}a"] = frame[f"A{time}"].map(lambda value: code[value][0])
        frame[f"A{time}b"] = frame[f"A{time}"].map(lambda value: code[value][1])
    return frame


class TestC13AnAllCategoricalVectorNode:
    _TUPLES: ClassVar[dict[str, tuple[float, float]]] = {
        "high": (1.0, 0.0),
        "low": (0.0, 1.0),
        "standard": (0.0, 0.0),
    }

    def _fit(self, factors: dict[str, float], *, vector: bool) -> Any:
        frame = _vector_frame()
        settings: dict[str, Any] = {
            "reference": "natural",
            "outcome_learner": sup.CellMeans(),
            "pseudo_learner": sup.CellMeans(),
            "treatment_learner": multi.CellProbabilities(),
            "n_folds": 1,
            "g_bounds": sup.NO_TRUNCATION,
            "simultaneous": False,
        }
        if vector:
            columns = {**sup.UNCENSORED_COLUMNS, "treatment": [["A1a", "A1b"], ["A2a", "A2b"]]}
            keyed = lambda mapping: {self._TUPLES[k]: v for k, v in mapping.items()}  # noqa: E731
        else:
            columns = dict(sup.UNCENSORED_COLUMNS)
            keyed = lambda mapping: dict(mapping)  # noqa: E731
        regimens = {
            "natural": Incremental(keyed({"high": 1.0})),
            "tilted": Incremental(keyed(factors)),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return IPSI(regimens, **settings).fit(frame, **columns)

    def test_the_vector_fit_equals_the_fit_on_the_combined_column(self) -> None:
        factors = {"high": 2.0, "low": 0.5}
        vector = self._fit(factors, vector=True)
        combined = self._fit(factors, vector=False)
        assert vector.data.treatment_levels[0] == ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0))
        assert vector.estimates.keys() == combined.estimates.keys()
        for name in vector.estimates:
            assert vector.psi(name) == pytest.approx(combined.psi(name), abs=1e-12)
            np.testing.assert_allclose(
                vector.influence_curves[name],
                combined.influence_curves[name],
                atol=1e-12,
                rtol=0,
            )
        natural = float(np.mean(_frame()["Y"].to_numpy(dtype=float)))
        assert abs(vector.psi("ey_regimen[tilted]") - natural) > 1e-3

    def test_a_tuple_that_is_no_observed_level_is_refused(self) -> None:
        frame = _vector_frame()
        columns = {**sup.UNCENSORED_COLUMNS, "treatment": [["A1a", "A1b"], ["A2a", "A2b"]]}
        with pytest.raises(DataError, match="not a level of node"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            IPSI(
                {"t": Incremental({(9.0, 9.0): 2.0})},
                outcome_learner=NeverFit(),
                pseudo_learner=NeverFit(),
                treatment_learner=NeverFit(),
                n_folds=1,
                simultaneous=False,
            ).fit(frame, **columns)


# ------------------------------------------------------------------ the parametric g-formula


class TestTheParametricGFormula:
    """The simulation form draws a two-level tilt only, and reads its factors from the plan."""

    def test_a_mapping_on_two_levels_is_the_scalar_multiplier(self) -> None:
        frame = panel_b(300, seed=5)
        results = {}
        for label, declaration in {
            "scalar": 2.0,
            "mapping": {1.0: 2.0},
            "first": {0.0: 0.5},
        }.items():
            estimator = ParametricGFormula(
                {"t": Incremental(declaration, name="t")}, n_draws=2, random_state=3
            )
            results[label] = quiet_fit(estimator, frame)
        for label in ("mapping", "first"):
            assert results[label].psi("ey_regimen[t]") == results["scalar"].psi("ey_regimen[t]")

    def test_a_categorical_tilt_is_redirected_before_any_learner(self) -> None:
        estimator = ParametricGFormula(
            {"t": Incremental({"high": 2.0})}, outcome_learner=GFormulaNeverFit()
        )
        roles = {**sup.UNCENSORED_COLUMNS, "treatment": ["A1", "A2"]}
        with pytest.raises(CapabilityError, match="odds tilt of a node with two levels") as raised:
            quiet_fit(estimator, _frame(), **roles)
        assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
        assert "Fit IPSI" in str(raised.value)
