r"""The projection of a conditional contrast by the per-arm outcome-adaptive C-TMLE (P1 PR 3).

``CTMLE(strategy="oat", oat_design="per_arm", effect_modification=...)`` fits
:math:`G_n(a \mid \bar Q_n(a, W))` per arm and targets the projection along
:math:`\varphi(V) s(a) / G_n(a)`. The point is the plug-in of the fluctuated regression, so it
is consistent when :math:`\bar Q_n` is, whatever :math:`G_n` converges to. No interval is
reported: condition (v) of Benkeser, Cai and van der Laan (2020) fails for an estimated outcome
regression, and the projection's covariate reads :math:`\varphi(V)`. The fit takes
``generated_design_plugin`` and keeps the curve as a diagnostic.

On :mod:`tests.discrete_law_oat` the per-arm design of arm 0 ties ``W = 0`` and ``W = 1``,
where :math:`g_0` differs, so the diagnostic curve read with :math:`g_0(a \mid W)` in place of
:math:`G_0(a \mid \bar Q_0(a))` departs on those rows: the nonzero witness of the design.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from scipy.optimize import root
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, CollaborativeTMLEMethod, ConditionalContrast, PointTreatment
from cleverly._inference_status import STATUS_RECORDS
from cleverly.assessment import CURVE_OPERATIONS
from cleverly.bases import Linear, Modifiers
from cleverly.effect_modification import CTMLE_CURVE_TOOLS_REFUSAL, EffectModification
from cleverly.estimators import CTMLE
from cleverly.estimators.ctmle import PROJECTION_DATA_REFUSAL, PROJECTION_STRATEGY_REFUSAL
from cleverly.exceptions import CapabilityError
from tests import discrete_law_oat as law
from tests.unit._natural_course_support import NeverFit

EXACT = 1e-10
FLOOR = 1e-3
STEM = {"difference": "cate_projection", "log_ratio": "log_crr_projection"}


class ArmShiftedOutcome(BaseEstimator):
    """The law's ``E[Y | A, W]`` on arm 0 and 0.5 on arm 1: a wrong, arm-specific ``Qbar``.

    Arm 1's design is then constant, so its mechanism is the marginal share and the
    fluctuation cannot repair the regression.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ArmShiftedOutcome:
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        arm = np.rint(design[:, 0]).astype(int)
        w = np.rint(design[:, 1]).astype(int)
        return np.where(arm == 1, 0.5, law.Q[w, arm])


def projection(scale: str = "difference") -> EffectModification:
    return EffectModification(Linear(["W"]), Modifiers(["W"]), scale)  # type: ignore[arg-type]


def exact_fit(scale: str = "difference", outcome: Any = None, **extra: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            CTMLE(
                strategy="oat",
                oat_design="per_arm",
                outcome_learner=law.OracleOutcome() if outcome is None else outcome,
                treatment_learner=law.SaturatedCategorical(),
                cross_fit=False,
                simultaneous=False,
                random_state=0,
                effect_modification=projection(scale),
                **extra,
            )
            .fit(law.frame(), outcome="Y", treatment="A")
            .single()
        )


def truth(scale: str, q: np.ndarray = law.Q) -> np.ndarray:
    """The projection of ``q[:, 1]`` against ``q[:, 0]`` on ``(1, w)`` under ``P_W``."""
    w = np.arange(4.0)
    basis = np.column_stack([np.ones(4), w])
    if scale == "difference":
        gram = basis.T @ (law.P_W[:, None] * basis)
        return np.asarray(np.linalg.solve(gram, basis.T @ (law.P_W * (q[:, 1] - q[:, 0]))))

    def equation(beta: np.ndarray) -> np.ndarray:
        e = expit(basis @ beta)
        return np.asarray(basis.T @ (law.P_W * ((q[:, 1] + q[:, 0]) * e - q[:, 1])))

    solved = root(equation, np.zeros(2), method="hybr", tol=1e-14)
    assert float(np.max(np.abs(equation(solved.x)))) < 1e-13
    return np.asarray(solved.x)


def closed_form_curve(design: np.ndarray) -> np.ndarray:
    r"""``M^{-1}[phi(tau - phi'beta) + phi sum_a s(a) 1{A=a}/G(a) (Y - Qbar_a)]`` per row.

    ``design`` holds the mechanism the covariate divides by, ``(4, 2)`` indexed ``[w, a]``.
    """
    frame = law.frame()
    w = frame["W"].to_numpy().astype(int)
    a = frame["A"].to_numpy().astype(int)
    y = frame["Y"].to_numpy()
    phi = np.column_stack([np.ones(w.size), w.astype(float)])
    beta = truth("difference")
    tau = law.Q[w, 1] - law.Q[w, 0]
    sign = np.where(a == 1, 1.0, -1.0)
    residual = sign / design[w, a] * (y - law.Q[w, a])
    rows = phi * (tau - phi @ beta + residual)[:, None]
    gram = phi.T @ phi / w.size
    return np.asarray(np.linalg.solve(gram, rows.T).T)


class TestTheExactLaw:
    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_the_point_is_the_truth_at_the_oracle_regression(self, scale: str) -> None:
        result = exact_fit(scale)
        expected = truth(scale)
        for j, term in enumerate(("(intercept)", "W")):
            assert result[f"{STEM[scale]}[{term}]"].psi == pytest.approx(expected[j], abs=EXACT)

    def test_the_diagnostic_curve_reads_the_per_arm_design(self) -> None:
        result = exact_fit()
        per_arm = np.column_stack([law.projection(0), law.projection(1)])
        expected = closed_form_curve(per_arm)
        for j, term in enumerate(("(intercept)", "W")):
            reported = np.asarray(result[f"cate_projection[{term}]"].influence_curve)
            np.testing.assert_allclose(reported, expected[:, j], rtol=0.0, atol=EXACT)
        # The witness: g_0(a | W) in place of G_0(a | Qbar_0(a)) departs on the tied rows.
        mutant = closed_form_curve(np.column_stack([1.0 - law.G1, law.G1]))
        w = law.frame()["W"].to_numpy()
        reported = np.asarray(result["cate_projection[W]"].influence_curve)
        assert float(np.max(np.abs(reported - mutant[:, 1])[w <= 1])) > FLOOR

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_a_wrong_regression_moves_the_point(self, scale: str) -> None:
        """The point needs a consistent ``Qbar``: it is not doubly robust."""
        result = exact_fit(scale, outcome=ArmShiftedOutcome())
        expected = truth(scale)
        slope = result[f"{STEM[scale]}[W]"].psi
        assert abs(slope - expected[1]) > 0.01

    def test_the_status_withholds_the_interval(self) -> None:
        result = exact_fit()
        estimate = result["cate_projection[W]"]
        assert estimate.inference == "generated_design_plugin"
        assert result.inference_status == "generated_design_plugin"
        curve = np.asarray(estimate.influence_curve)
        assert estimate.plugin_std_error == pytest.approx(
            np.sqrt(np.var(curve, ddof=1) / curve.size), rel=1e-10
        )
        with pytest.raises(Exception, match="interval"):
            estimate.ci  # noqa: B018
        assert STATUS_RECORDS["generated_design_plugin"].reason


def _never(**extra: Any) -> CTMLE:
    NeverFit.calls = 0
    return CTMLE(
        outcome_learner=NeverFit(),
        treatment_learner=NeverFit(),
        cross_fit=False,
        random_state=0,
        effect_modification=projection(),
        **extra,
    )


class TestTheRefusals:
    @pytest.mark.parametrize(
        "settings",
        [
            {"strategy": "greedy"},
            {"strategy": "derivative"},
            {"strategy": "oat", "oat_design": "shared"},
        ],
        ids=["greedy", "derivative", "shared"],
    )
    def test_every_other_strategy_refuses_before_any_learner(
        self, settings: dict[str, Any]
    ) -> None:
        with pytest.raises(CapabilityError) as caught:
            _never(**settings).fit(law.frame(), outcome="Y", treatment="A")
        assert caught.value.refusal == PROJECTION_STRATEGY_REFUSAL
        assert NeverFit.calls == 0

    def test_strata_and_missing_data_refuse(self) -> None:
        frame = law.frame().assign(S=lambda f: (f["W"] >= 2).astype(float))
        with pytest.raises(CapabilityError) as caught:
            _never(strategy="oat").fit(
                frame, outcome="Y", treatment="A", covariates=["W", "S"], strata=["S"]
            )
        assert caught.value.refusal == PROJECTION_DATA_REFUSAL
        missing = law.frame().assign(D=1.0)
        missing.loc[::7, "D"] = 0.0
        missing.loc[missing["D"] == 0.0, "Y"] = np.nan
        with pytest.raises(CapabilityError) as caught:
            _never(strategy="oat").fit(missing, outcome="Y", treatment="A", delta="D")
        assert caught.value.refusal == PROJECTION_DATA_REFUSAL
        assert NeverFit.calls == 0

    def test_an_intermediate_keeps_the_shipped_text(self) -> None:
        frame = law.frame().assign(Z=lambda f: (f["W"] % 2).astype(float))
        with pytest.raises(CapabilityError, match="does not compose either collaborative"):
            _never(strategy="oat").fit(frame, outcome="Y", treatment="A", intermediate="Z")
        assert NeverFit.calls == 0


def _effect() -> Any:
    study = CausalStudy(
        law.frame(), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
    )
    return study.identify(ConditionalContrast(Modifiers(["W"]), model=Linear(["W"])))


class TestTheStudyRoute:
    def test_the_record_is_available_and_the_routes_agree(self) -> None:
        effect = _effect()
        record = next(r for r in effect.available_methods() if r.name == "collaborative_tmle")
        assert record.available
        method = CollaborativeTMLEMethod(strategy="oat").with_overrides(
            outcome_learner=law.OracleOutcome(),
            treatment_learner=law.SaturatedCategorical(),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study_fit = effect.estimate(method=method)
        engine = exact_fit()
        for name, estimate in engine.estimates.items():
            assert study_fit[name].psi == pytest.approx(estimate.psi, abs=1e-12)

    def test_the_default_strategy_refuses_with_its_reason(self) -> None:
        with pytest.raises(CapabilityError) as caught:
            _effect().estimate(method="collaborative_tmle", outcome_learner=NeverFit())
        assert caught.value.refusal == PROJECTION_STRATEGY_REFUSAL

    def test_the_curve_checks_refuse_and_the_truncation_curve_runs(self) -> None:
        result = exact_fit()
        curve = result.effect_curve()
        assert curve.fitted_method == "collaborative_tmle"
        rows = {row.operation: row for row in curve.diagnostics.capabilities}
        for name in CURVE_OPERATIONS:
            assert rows[name].refusal == CTMLE_CURVE_TOOLS_REFUSAL
        with pytest.raises(CapabilityError) as caught:
            curve.diagnostics.calibration(n_folds=2)
        assert caught.value.refusal.condition.endswith(CTMLE_CURVE_TOOLS_REFUSAL.condition)
        assert rows["support"].available and rows["truncation_curve"].available
        frame = result.diagnostics.truncation_curve(bounds=[0.05, 0.2]).to_frame()
        assert set(frame["estimand"]) == set(result.estimates)
        assert any(column.startswith("plugin_") for column in frame.columns)


class TestTheAssessmentRows:
    def test_the_bound_keeps_the_selecting_premise_and_its_point_form(self) -> None:
        """The bound inherits the C-TMLE premise refusal; a consistent mechanism lifts it."""
        from cleverly.sensitivity import omitted_variable as ov

        result = exact_fit()
        with pytest.raises(CapabilityError, match="no nu\\^2 estimate for a 'collaborative_tmle'"):
            ov.omitted_variable_bounds(result, "cate_projection[W]", cf_y=0.02, cf_d=0.02)
        bound = ov.omitted_variable_bounds(
            result,
            "cate_projection[W]",
            cf_y=0.02,
            cf_d=0.02,
            assume_consistent_mechanism=True,
            mechanism_learner=LogisticRegression(max_iter=1000),
        )
        assert "projection" in bound.features

    def test_refute_runs_with_no_verdict(self) -> None:
        result = exact_fit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = result.diagnostics.refute(
                estimand="cate_projection[W]",
                tests=["placebo", "subset"],
                n_replicates=10,
                random_state=0,
            )
        assert all(test.passed is None for test in report.tests)
