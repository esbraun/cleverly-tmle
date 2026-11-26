r"""Refutations and the truncation curve of a conditional contrast's projection (P1 PR 3).

``refute`` refits the estimator and compares a reported coefficient with a stated reference.
No evidence row licenses a verdict for a projection, so every reading prints
``passed is None``. What this module checks is the reference: the generated truth of a
coefficient, summed over the held rows (the held design) or over the plasmode support, is
:math:`M_n^{-1} P_n[w \varphi \tau_{sim}]` on the difference scale and the root of the
empirical log-CRR equation on the log ratio, per stratum with ``strata=`` and at the declared
intermediate level for a controlled direct effect. Each truth is compared with a sum computed
here from the raw frame, which shares no code with the package (no basis, no stratum coding).

Three components can vanish on a convenient law, and each has a witness:

- the stratum mask: the stratum coefficients differ from the marginal ones;
- the intermediate level: with a nonzero intermediate effect the coefficients at ``Z = 0``
  differ from those at ``Z = 1``;
- the value and reference arms: the truth of the slope is nonzero, so a swap flips its sign.

The truncation curve of a projection fit retargets the coefficients at each bound, and each
row equals a refit at that bound with the same deterministic learners. A projection curve's
row reads its producing fit's curve; an EP curve keeps the shipped refusal, and a curve's
``refute`` row redirects to the producing fit.
"""

from __future__ import annotations

import functools
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import root
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, ControlledDirectEffect, PointTreatment
from cleverly.assessment import CURVE_REFUTE_REFUSAL, _projection_coefficient
from cleverly.bases import Linear, Modifiers
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.validation import BernoulliAdjustmentOutcome, GaussianAdjustmentOutcome
from cleverly.validation._point_generated import PlasmodeLaw, arm_means, held_weights

REFUTE = sys.modules["cleverly.validation.refute"]
EXACT = 1e-12
#: A witness must clear this floor.
FLOOR = 0.005
#: The process of a fit with no intermediate, and one whose intermediate term moves the CDE,
#: so the level of ``Z`` is witnessed.
PLAIN = BernoulliAdjustmentOutcome()
DIRECT = BernoulliAdjustmentOutcome(intermediate_effect=0.8)


def _process(result: Any) -> Any:
    return DIRECT if result.data.has_intermediate else PLAIN


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 300
    w = rng.normal(size=n)
    s = (rng.random(n) < 0.5).astype(float)
    a = rng.binomial(1, expit(0.4 * w))
    z = rng.binomial(1, expit(0.3 * w + 0.5 * a))
    y = rng.binomial(1, expit(-0.3 + 0.5 * w + a * (0.4 + 0.5 * w) + 0.3 * z))
    recorded = (rng.random(n) < 0.8).astype(float)
    observed = (rng.random(n) < 0.85).astype(float)
    return pd.DataFrame(
        {"Y": y, "A": a, "W": w, "S": s, "Z": z, "DeltaA": recorded, "Delta": observed}
    )


def _learners(**extra: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "random_state": 0,
        "cross_fit": False,
        **extra,
    }


def _contrast(scale: str) -> ConditionalContrast:
    return ConditionalContrast(Modifiers(["W"]), scale=scale, model=Linear(["W"]))  # type: ignore[arg-type]


#: The sweep's projection kinds rebuilt on a study, which records the identification that a
#: generated truth needs (a direct ``TMLE.fit`` records none).
KIND_SCALES = (
    ("plain", "difference"),
    ("plain", "log_ratio"),
    ("strata", "difference"),
    ("strata", "log_ratio"),
    ("missing_treatment", "difference"),
    ("intermediate", "difference"),
    ("intermediate", "log_ratio"),
)


@functools.cache
def fit(kind: str, scale: str) -> Any:
    frame = _frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "plain":
            design = PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
            return (
                CausalStudy(frame, design=design).identify(_contrast(scale)).estimate(**_learners())
            )
        if kind == "strata":
            design = PointTreatment(
                outcome="Y", treatment="A", adjustment=("W", "S"), strata=("S",)
            )
            return (
                CausalStudy(frame, design=design).identify(_contrast(scale)).estimate(**_learners())
            )
        if kind == "missing_treatment":
            design = PointTreatment(
                outcome="Y", treatment="A", adjustment=("W",), treatment_missingness="DeltaA"
            )
            frame = frame.assign(A=frame["A"].where(frame["DeltaA"] == 1))
            return (
                CausalStudy(frame, design=design)
                .identify(_contrast(scale))
                .estimate(**_learners(missingness_learner=LogisticRegression(max_iter=1000)))
            )
        level = 1.0 if kind == "intermediate_one" else 0.0
        missing = kind == "intermediate_missing"
        design = PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            intermediate="Z",
            missingness="Delta" if missing else None,
        )
        extra = {"missingness_learner": LogisticRegression(max_iter=1000)} if missing else {}
        return (
            CausalStudy(frame, design=design)
            .identify(ControlledDirectEffect(intermediate=level, contrast=_contrast(scale)))
            .estimate(**_learners(**extra))
        )


def _independent(
    frame: pd.DataFrame,
    value: np.ndarray,
    reference: np.ndarray,
    scale: str,
    rows: np.ndarray,
) -> np.ndarray:
    """The projection on ``(1, W)`` over ``rows``, by least squares or a generic root."""
    w = frame["W"].to_numpy()[rows]
    q_t, q_r = value[rows], reference[rows]
    basis = np.column_stack([np.ones(w.size), w])
    if scale == "difference":
        return np.asarray(np.linalg.lstsq(basis, q_t - q_r, rcond=None)[0])

    def equation(beta: np.ndarray) -> np.ndarray:
        return np.asarray(basis.T @ ((q_t + q_r) * expit(basis @ beta) - q_t))

    solved = root(equation, np.zeros(2), method="hybr", tol=1e-14)
    assert float(np.max(np.abs(equation(solved.x)))) < 1e-12
    return np.asarray(solved.x)


def _independent_means(result: Any, level: float | None) -> tuple[np.ndarray, np.ndarray]:
    """The process means of arms 1 and 0 at each row, from the frame's own columns."""
    data = result.data
    covariates = np.asarray(data.covariates, dtype=float)
    means = arm_means(_process(result), covariates, (0.0, 1.0), data, intermediate=level)
    return means[1.0], means[0.0]


def _rows(frame: pd.DataFrame, estimand: str) -> np.ndarray:
    if "[S=1.0]" in estimand:
        return np.asarray(frame["S"].to_numpy() == 1.0)
    if "[S=0.0]" in estimand:
        return np.asarray(frame["S"].to_numpy() == 0.0)
    return np.ones(len(frame), dtype=bool)


def _cases() -> list[tuple[str, str]]:
    return list(KIND_SCALES)


class TestGeneratedTruths:
    """Each coefficient's generated truth equals a sum computed here, to 1e-12."""

    @pytest.mark.parametrize(("kind", "scale"), _cases())
    @pytest.mark.parametrize("name", ["simulated_outcome", "plasmode"])
    def test_each_coefficient_equals_an_independent_sum(
        self, kind: str, scale: str, name: str
    ) -> None:
        result = fit(kind, scale)
        frame = _frame()
        level = 0.0 if kind == "intermediate" else None
        value, reference = _independent_means(result, level)
        assert len(result.estimates) == (6 if kind == "strata" else 2)
        for estimand in result.estimates:
            target = REFUTE._validate_generated_eligibility(
                result, estimand, _process(result), name
            )
            assert target.intermediate == level
            if name == "plasmode":
                means, weights, mixture = PlasmodeLaw.fit(result).means(
                    _process(result), result.data, target
                )
            else:
                covariates = np.asarray(result.data.covariates, dtype=float)
                means = arm_means(
                    _process(result), covariates, target.codes, result.data, intermediate=level
                )
                weights, mixture = held_weights(target, result.data), None
            truth = target.truth(means, weights, mixture)
            coefficients = _independent(frame, value, reference, scale, _rows(frame, estimand))
            term = 1 if "[W]" in estimand else 0
            assert truth == pytest.approx(coefficients[term], abs=EXACT), estimand

    def test_the_stratum_coefficients_differ_from_the_marginal_ones(self) -> None:
        """Witness of the stratum mask: dropping it would return the marginal truth."""
        for scale in ("difference", "log_ratio"):
            result = fit("strata", scale)
            stem = "cate_projection" if scale == "difference" else "log_crr_projection"
            truths = {}
            for estimand in (f"{stem}[W]", f"{stem}[W][S=0.0]", f"{stem}[W][S=1.0]"):
                target = REFUTE._validate_generated_eligibility(
                    result, estimand, PLAIN, "simulated_outcome"
                )
                covariates = np.asarray(result.data.covariates, dtype=float)
                means = arm_means(PLAIN, covariates, target.codes, result.data)
                truths[estimand] = target.truth(means, held_weights(target, result.data))
            values = list(truths.values())
            assert min(abs(values[0] - values[1]), abs(values[0] - values[2])) > FLOOR, truths

    def test_the_intermediate_level_moves_the_truth(self) -> None:
        """Witness of the intermediate thread: ``Z = 0`` and ``Z = 1`` give other truths."""
        for scale in ("difference", "log_ratio"):
            result = fit("intermediate", scale)
            stem = "cate_projection" if scale == "difference" else "log_crr_projection"
            target = REFUTE._validate_generated_eligibility(
                result, f"{stem}[(intercept)]", DIRECT, "simulated_outcome"
            )
            covariates = np.asarray(result.data.covariates, dtype=float)
            weights = held_weights(target, result.data)
            at = {
                level: target.truth(
                    arm_means(DIRECT, covariates, target.codes, result.data, intermediate=level),
                    weights,
                )
                for level in (0.0, 1.0)
            }
            assert abs(at[0.0] - at[1.0]) > FLOOR, at

    def test_the_slope_is_nonzero_so_an_arm_swap_is_seen(self) -> None:
        result = fit("plain", "log_ratio")
        target = REFUTE._validate_generated_eligibility(
            result, "log_crr_projection[W]", PLAIN, "simulated_outcome"
        )
        assert (target.value_code, target.reference_code) == (1.0, 0.0)
        covariates = np.asarray(result.data.covariates, dtype=float)
        means = arm_means(PLAIN, covariates, target.codes, result.data)
        assert abs(target.truth(means, held_weights(target, result.data))) > FLOOR


class TestRefutationsRun:
    """Every refutation runs on a projection coefficient and prints no verdict."""

    @pytest.mark.parametrize(("kind", "scale"), _cases())
    def test_each_test_runs_with_no_verdict(self, kind: str, scale: str) -> None:
        result = fit(kind, scale)
        estimand = next(name for name in result.estimates if "[W]" in name)
        tests = ["placebo", "subset", "random_common_cause", "plasmode", "simulated_outcome"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = result.diagnostics.refute(
                estimand=estimand,
                tests=tests,
                simulated_outcome=_process(result),
                n_replicates=10,
                random_state=0,
            )
        assert [test.name for test in report.tests] == tests
        assert all(test.passed is None for test in report.tests)

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_the_placebo_null_is_zero(self, scale: str) -> None:
        result = fit("plain", scale)
        for estimand in result.estimates:
            assert REFUTE._no_effect_null(result, estimand) == 0.0

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_a_call_that_names_no_coefficient_raises_the_row_s_sentence(self, scale: str) -> None:
        """A projection reports no ``ate``, so the bare call raises the row's deferral.

        The call's own default refusal, "estimand 'ate' was not requested in this fit",
        names none of the coefficients; the row names them all.  The preview raises the
        same record under its own name.
        """
        result = fit("plain", scale)
        row = result.diagnostics.capability("refute")
        assert row.status.value == "deferred" and row.refusal is not None
        assert row.refusal.condition == (
            f"refute answers for one estimand; choose an explicit estimand from "
            f"{list(result.estimates)}"
        )
        with pytest.raises(CapabilityError) as caught:
            result.diagnostics.refute()
        assert str(caught.value) == f"diagnostic 'refute' is deferred: {row.refusal.condition}"
        assert caught.value.refusal.reason is row.refusal.reason
        assert caught.value.refusal.alternative == row.refusal.alternative
        assert "'ate'" not in str(caught.value)
        with pytest.raises(CapabilityError) as preview:
            result.diagnostics.planned_refits()
        assert str(preview.value) == (
            "diagnostic 'planned_refits' previews 'refute', which is deferred: "
            f"{row.refusal.condition}"
        )

    def test_a_gaussian_outcome_on_the_log_ratio_refuses(self) -> None:
        result = fit("plain", "log_ratio")
        with pytest.raises(CapabilityError) as caught:
            result.diagnostics.refute(
                estimand="log_crr_projection[W]",
                tests=["simulated_outcome"],
                simulated_outcome=GaussianAdjustmentOutcome(),
                n_replicates=10,
                random_state=0,
            )
        assert "Gaussian" in str(caught.value) or "ratio" in str(caught.value)

    def test_a_plasmode_with_two_observation_mechanisms_refuses(self) -> None:
        result = fit("intermediate_missing", "difference")
        with pytest.raises(CapabilityError, match="plasmode"):
            REFUTE._validate_generated_eligibility(result, "cate_projection[W]", DIRECT, "plasmode")


class TestTruncationCurve:
    """Each truncation row equals a refit at its bound, and a curve reads its fit's rows."""

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_each_row_equals_a_refit_at_its_bound(self, scale: str) -> None:
        result = fit("plain", scale)
        curve = result.diagnostics.truncation_curve(bounds=[0.3, 0.4]).to_frame()
        assert set(curve["bound"]) == {0.3, 0.4}
        design = PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        effect = CausalStudy(_frame(), design=design).identify(_contrast(scale))
        moved = 0.0
        for bound in (0.3, 0.4):
            rows = curve[curve["bound"] == bound]
            upper = float(rows["upper_bound"].iloc[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                refit = effect.estimate(**_learners(g_bounds=(bound, upper)))
            for estimand, psi in zip(rows["estimand"], rows["psi"], strict=True):
                assert psi == pytest.approx(refit.estimates[estimand].psi, abs=1e-10)
                moved = max(moved, abs(psi - result.estimates[estimand].psi))
        # The bound binds on this law, so the equality is not the fitted estimate twice.
        assert moved > 1e-6

    def test_a_projection_curve_reads_its_fit_s_curve(self) -> None:
        result = fit("plain", "difference")
        curve = result.effect_curve()
        row = curve.diagnostics.capability("truncation_curve")
        assert row.available
        mine = curve.diagnostics.truncation_curve(bounds=[0.02, 0.1]).to_frame()
        theirs = result.diagnostics.truncation_curve(bounds=[0.02, 0.1]).to_frame()
        pd.testing.assert_frame_equal(mine, theirs)

    def test_an_ep_curve_redirects_to_the_projection_on_its_basis(self) -> None:
        """The truncation curve reports an interval at each bound, which an EP curve lacks."""
        from tests.unit.summary_fixtures.conditional import curve_ep

        curve = curve_ep()
        row = curve.diagnostics.capability("truncation_curve")
        assert not row.available
        assert row.refusal.reason is RefusalReason.API_REDIRECTION
        assert row.refusal.roadmap is None
        assert row.refusal.condition.startswith(
            "The truncation curve reports an estimate and its interval at each bound, and an "
            "EP curve has no interval. The projection on the same basis, curve.model.basis,"
        )
        assert row.refusal.condition.endswith("has both.")
        assert row.refusal.alternative == (
            "Fit ConditionalContrast(..., model=curve.model.basis) with method='tmle' and call "
            "truncation_curve() on that fit."
        )
        with pytest.raises(CapabilityError) as caught:
            curve.diagnostics.truncation_curve(bounds=[0.02, 0.1])
        assert caught.value.refusal.condition.endswith(row.refusal.condition)
        assert caught.value.refusal.alternative == row.refusal.alternative

    def test_an_ep_curve_carries_the_projection_s_live_truncation_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Witness: where the projection's truncation call refuses, the EP row says so."""
        assert _ep_truncation_record(monkeypatch) is not None

    def test_an_always_redirecting_ep_truncation_row_fails_the_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutation control: a row that redirects past a refusing gate fails the witness."""
        import cleverly.assessment as assessment
        from tests.unit._source_mutation import mutate

        mutate(
            monkeypatch,
            assessment,
            assessment,
            "_ep_truncation_row",
            [("if blocked is not None:", "if False:")],
        )
        assert _ep_truncation_record(monkeypatch) is None

    def test_a_curve_s_refute_row_redirects_to_its_fit(self) -> None:
        curve = fit("plain", "difference").effect_curve()
        row = curve.diagnostics.capability("refute")
        assert not row.available
        assert row.refusal is not None
        assert row.refusal.condition == CURVE_REFUTE_REFUSAL.condition
        assert row.refusal.alternative == (
            "Run refute on the fit that produced the curve, naming a coefficient such as "
            f"'{_projection_coefficient(curve)}'."
        )
        assert row.refusal.reason is RefusalReason.API_REDIRECTION
        assert row.refusal.roadmap is None


def _ep_truncation_record(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The EP curve's truncation record when its projection reports a jackknife variance.

    Returns the record when it carries the projection's live refusal, prefixed for the curve,
    and ``None`` when the row redirects instead.  The jackknife gate refuses the truncation
    curve on the projection while the method's static row runs, so only the live gate sees it.
    """
    import cleverly.assessment as assessment
    import cleverly.inference.cluster_jackknife as cluster_jackknife
    from cleverly.inference.cluster_jackknife import JACKKNIFE_TRUNCATION_REFUSAL
    from tests.unit.summary_fixtures.conditional import curve_ep

    curve = curve_ep()
    nuisance_fit = curve.fit.nuisance_fit
    row = next(
        item
        for item in assessment.assessment_capabilities(curve)
        if item.operation == "truncation_curve"
    )
    monkeypatch.setattr(cluster_jackknife, "jackknife_fit", lambda result: result is nuisance_fit)
    record = assessment._ep_truncation_row(curve, row).refusal
    if record.reason is not JACKKNIFE_TRUNCATION_REFUSAL.reason:
        return None
    if not record.condition.startswith(assessment.EP_NO_INFLUENCE_CURVE):
        return None
    tail = JACKKNIFE_TRUNCATION_REFUSAL.condition[1:]
    assert record.condition.endswith(tail)
    assert record.alternative == JACKKNIFE_TRUNCATION_REFUSAL.alternative
    return record
