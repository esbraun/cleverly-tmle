r"""The TMLE assessment of external scores and the paired comparison of two (X97 part B).

The checks of the X97 plan, section 4. Each exact check carries the witness that keeps it from
passing vacuously, or a deliberate mutation of the code that must make it fail:

- T1, T3 to T6: on the three-cell law of :mod:`tests.discrete_law`, the paired point is the
  functional and its curve the Gateaux derivative, on both scales; the sign follows the order;
  the second-moment term, the residual term and the cancelled comparator term are each
  checked by a mutation, with the witness that the term is not small;
- T2, T7, T8, T17: on a six-cell law where the fluctuation does not saturate (the test
  asserts the span ranks first), the paired row equals the difference of the single rows of one
  fit; with a wrong outcome regression and the oracle mechanism every TMLE estimate is the
  functional, while each of three mutations of the numerators misses it; the substitution
  equals the one-step at the targeted fit; and the log-ratio alternation fails loudly when it
  cannot settle;
- T9: the targeted regressions stay in range where the one-step's weights turn negative, and
  the log-ratio calibration is the unique minimiser of a convex loss;
- T10 to T12: missing outcomes, observation weights and clusters, with the three cluster cases;
- T14: the support labels and their tables;
- T16: the report, the frame and the repr.

The refusals (T13) sit with the other score refusals in
``tests/unit/test_conditional_contrast.py``.  T15, the bit-equality of the targeting curves
across the refactor onto :mod:`cleverly._pooled_targeting`, is a one-time record in the row's
progress log; ``tests/unit/test_targeting_curves.py`` keeps the curves' identity tests.
"""

from __future__ import annotations

import functools
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly import _pooled_targeting as pt
from cleverly import effect_curves as ec
from cleverly.bases import Linear, Modifiers
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from tests import discrete_law as law
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_oat import SaturatedCategorical
from tests.unit.test_conditional_contrast import SCORE, score_functional

#: The second score of the three-cell law: not affine in :data:`SCORE` on the three cells.
SCORE_B = np.array([1.00, 0.00, 1.00])

#: A mutation must move the number it targets by at least this much.
MUTATION_FLOOR = 1e-3

MODIFIERS = Modifiers(["W"], domains={"W": (0.0, 5.0)})


def _quiet(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args, **kwargs)

    return wrapped


def values(assessment: Any) -> np.ndarray:
    """Every estimate in report order: the column's three, the compare's three, the pair."""
    return np.array([estimate.psi for _, _, estimate in assessment.rows()])


def softplus(x: Any) -> Any:
    """``log(1 + e^x)``, analytic so the complex step differentiates it."""
    return np.log(1.0 + np.exp(x))


# ----------------------------------------------------------------- the three-cell law


def paired_functional(s1: Any, s2: Any, scale: str, *, missing: bool = False) -> Any:
    """Both scores' three parameters and ``R(s1) - R(s2)``, as one functional of the law.

    The paired row is the difference of the two risk differences: the constant's risk cancels.
    """
    first = score_functional(s1, scale, missing=missing)
    second = score_functional(s2, scale, missing=missing)

    def functional(probs: Any) -> Any:
        a, b = first(probs), second(probs)
        return np.concatenate([a, b, [a[0] - b[0]]])

    return functional


def law_frame() -> pd.DataFrame:
    frame = law.frame()
    cell = frame["W"].astype(int)
    frame["S"] = SCORE[cell]
    frame["SB"] = SCORE_B[cell]
    return frame


def law_effect(frame: pd.DataFrame | None = None, scale: str = "difference", **design: Any) -> Any:
    study = CausalStudy(
        law_frame() if frame is None else frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), **design),
    )
    return study.identify(
        ConditionalContrast(MODIFIERS, scale=scale, model=Linear(["W"]))  # type: ignore[arg-type]
    )


def law_oracle() -> dict[str, Any]:
    dgp = law.DiscreteLaw()
    return {
        "outcome_learner": OracleOutcome(dgp),
        "treatment_learner": OracleTreatment(dgp),
        "cross_fit": False,
        "random_state": 0,
    }


@functools.cache
def paired(scale: str, column: str = "S", compare: str = "SB") -> Any:
    return law_effect(scale=scale).assess_score(
        column,
        computed_from=["W"],
        score_kind="external",
        compare=compare,
        compare_computed_from=["W"],
        scale=scale,
        **law_oracle(),
    )


def law_curves(assessment: Any, rows: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.asarray(estimate.influence_curve)[rows] for _, _, estimate in assessment.rows()]
    )


def law_dtilde_terms(scale: str) -> tuple[np.ndarray, np.ndarray]:
    """At each support point, the residual terms of the two arms, ``1{A=a}/g (Y - Q_a)``."""
    w, a, y = (np.array(column) for column in zip(*law.SUPPORT, strict=True))
    q = law.Q_EXACT
    g = np.where(a == 1, law.G_EXACT[w], 1.0 - law.G_EXACT[w])
    residual = (y - q[w, a]) / g
    return np.where(a == 1, residual, 0.0), np.where(a == 0, residual, 0.0)


class TestTheThreeCellLaw:
    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_t1_the_paired_point_is_the_functional_and_its_curve_the_derivative(
        self, scale: str
    ) -> None:
        functional = paired_functional(SCORE, SCORE_B, scale)
        truth = functional(law.PROBS)
        # Witness: the paired risk difference is far from 0, on each scale.
        assert abs(truth[-1]) > (0.3 if scale == "difference" else 0.02)
        assessment = paired(scale)
        assert assessment.paired_name == "risk_difference[S - SB]"
        np.testing.assert_allclose(values(assessment), truth, atol=1e-11)
        np.testing.assert_allclose(
            law_curves(assessment, law.first_row_of()),
            law.gateaux_eif(functional),
            atol=1e-11,
            rtol=0,
        )

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_t3_swapping_the_scores_negates_the_paired_row(self, scale: str) -> None:
        forward, backward = paired(scale), paired(scale, "SB", "S")
        assert backward.paired_name == "risk_difference[SB - S]"
        ahead = forward[forward.paired_name]
        behind = backward[backward.paired_name]
        assert behind.psi == pytest.approx(-ahead.psi, abs=1e-12)
        np.testing.assert_allclose(
            np.asarray(behind.influence_curve), -np.asarray(ahead.influence_curve), atol=1e-12
        )
        # Mutation: a builder that reports R(compare) - R(column) misses T1 by twice the row.
        truth = paired_functional(SCORE, SCORE_B, scale)(law.PROBS)[-1]
        assert abs(-ahead.psi - truth) > (0.7 if scale == "difference" else 0.04)

    def test_t4_the_point_without_the_second_moments_misses_the_functional(self) -> None:
        assessment = paired("difference")
        truth = paired_functional(SCORE, SCORE_B, "difference")(law.PROBS)[-1]
        second_moments = float((law.P_W * (SCORE**2 - SCORE_B**2)).sum())
        mutated = assessment[assessment.paired_name].psi - second_moments
        assert abs(mutated - truth) > 0.6

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_t5_the_paired_curve_without_its_residual_term_misses_the_derivative(
        self, scale: str
    ) -> None:
        assessment = paired(scale)
        expected = law.gateaux_eif(paired_functional(SCORE, SCORE_B, scale))[:, -1]
        reported = law_curves(assessment, law.first_row_of())[:, -1]
        np.testing.assert_allclose(reported, expected, atol=1e-11)
        w = np.array([point[0] for point in law.SUPPORT])
        treated, control = law_dtilde_terms(scale)
        s1, s2 = SCORE[w], SCORE_B[w]
        if scale == "difference":
            residual = -2.0 * (s1 - s2) * (treated - control)
        else:
            residual = treated * (softplus(-s1) - softplus(-s2)) + control * (
                softplus(s1) - softplus(s2)
            )
        assert np.max(np.abs((reported - residual) - expected)) > 0.05

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_t6_a_comparator_term_left_in_the_paired_curve_misses_the_derivative(
        self, scale: str
    ) -> None:
        assessment = paired(scale)
        expected = law.gateaux_eif(paired_functional(SCORE, SCORE_B, scale))[:, -1]
        reported = law_curves(assessment, law.first_row_of())[:, -1]
        w = np.array([point[0] for point in law.SUPPORT])
        treated, control = law_dtilde_terms(scale)
        q = law.Q_EXACT
        tau = q[:, 1] - q[:, 0]
        if scale == "difference":
            ate = float((law.P_W * tau).sum())
            dtilde = tau[w] + treated - control
            term = 2.0 * ate * (dtilde - ate)
        else:
            c = float(np.log((law.P_W * q[:, 1]).sum() / (law.P_W * q[:, 0]).sum()))
            risk_c = float((law.P_W * (q[:, 1] * softplus(-c) + q[:, 0] * softplus(c))).sum())
            augmented = (q[w, 1] + treated) * softplus(-c) + (q[w, 0] + control) * softplus(c)
            term = -(augmented - risk_c)
        # Witness: the term is not small at some cell; it vanishes only when s1 = s2.
        assert np.max(np.abs(term)) > 0.1
        assert np.max(np.abs((reported + term) - expected)) > 0.1

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_t10_missing_outcomes_enter_the_paired_curve_through_the_response_mechanism(
        self, scale: str
    ) -> None:
        dgp = mar.DiscreteLaw()
        frame = mar.frame()
        cell = frame["W"].astype(int)
        frame["S"], frame["SB"] = SCORE[cell], SCORE_B[cell]
        effect = law_effect(frame, scale, missingness="Delta")
        assessment = effect.assess_score(
            "S",
            computed_from=["W"],
            score_kind="external",
            compare="SB",
            compare_computed_from=["W"],
            scale=scale,
            outcome_learner=OracleOutcome(dgp),
            treatment_learner=OracleTreatment(dgp),
            missingness_learner=OracleMissingness(dgp),
            cross_fit=False,
            random_state=0,
        )
        functional = paired_functional(SCORE, SCORE_B, scale, missing=True)
        np.testing.assert_allclose(values(assessment), functional(mar.PROBS), atol=1e-12)
        expected = law.contamination_eif(functional, mar.PROBS, mar.SUPPORT)
        reported = law_curves(assessment, mar.first_row_of())
        np.testing.assert_allclose(reported, expected, atol=1e-11, rtol=0)
        assert assessment.status == "experimental"
        assert "missing outcomes" in assessment.status_reason
        for _, _, estimate in assessment.rows():
            assert "Missing outcomes" in estimate.support.reason
        # Mutation: the paired residual term without 1 / Pi, which varies with (A, W) here.
        w, a, k = (np.array(column) for column in zip(*mar.SUPPORT, strict=True))
        observed = k != mar.UNOBSERVED
        y = (k == mar.OBSERVED_ONE).astype(float)
        g = np.where(a == 1, mar.G_EXACT[w], 1.0 - mar.G_EXACT[w])
        pi = mar.PI_EXACT[w, a]
        weight = np.where(observed, 1.0 / (g * pi), 0.0) * (y - mar.Q_EXACT[w, a])
        treated, control = np.where(a == 1, weight, 0.0), np.where(a == 0, weight, 0.0)
        s1, s2 = SCORE[w], SCORE_B[w]
        if scale == "difference":
            residual = -2.0 * (s1 - s2) * (treated - control)
        else:
            residual = treated * (softplus(-s1) - softplus(-s2)) + control * (
                softplus(s1) - softplus(s2)
            )
        dropped = reported[:, -1] - residual + residual * pi
        assert np.max(np.abs(dropped - expected[:, -1])) > 0.05


# ----------------------------------------------------------------- the six-cell witness law

PER_CELL = 80
SIX_G = np.array([0.25, 0.50, 0.75, 0.50, 0.25, 0.75])
#: ``Qbar_0(a, w)``, column ``a``.
SIX_Q = np.array(
    [[0.30, 0.55], [0.20, 0.60], [0.45, 0.70], [0.35, 0.40], [0.60, 0.85], [0.25, 0.50]]
)
#: A declared wrong outcome regression, column ``a``.
SIX_QI = np.array(
    [[0.65, 0.15], [0.65, 0.20], [0.70, 0.20], [0.75, 0.25], [0.80, 0.25], [0.80, 0.25]]
)
S6 = 0.1 + 0.1 * np.arange(6)
S6B = np.array([0.9, 0.0, 0.8, 0.1, 1.0, 0.2])
TIGHT_BOUNDS = (0.45, 0.55)


class TableOutcome(BaseEstimator):
    """The declared wrong regression :data:`SIX_QI`, read by ``(A, W)``; it learns nothing."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> TableOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        a, w = np.rint(design[:, 0]).astype(int), np.rint(design[:, 1]).astype(int)
        p = SIX_QI[w, a]
        return np.column_stack([1.0 - p, p])

    def predict(self, X: Any) -> Any:
        return self.predict_proba(X)[:, 1]


def six_frame(weights: bool = False) -> pd.DataFrame:
    """Eighty rows per cell; the treated share and both outcome shares are exact in every cell."""
    rows = []
    for w in range(6):
        treated = round(PER_CELL * SIX_G[w])
        for i in range(PER_CELL):
            a = int(i < treated)
            within, size = (i, treated) if a else (i - treated, PER_CELL - treated)
            ones = round(size * SIX_Q[w, a])
            rows.append((float(within < ones), a, w, 1 + i % 3 if weights else 1))
    frame = pd.DataFrame(rows, columns=["Y", "A", "W", "wt"])
    frame["S"], frame["SB"] = S6[frame["W"]], S6B[frame["W"]]
    return frame


def six_fit(
    frame: pd.DataFrame,
    *,
    g_bounds: tuple[float, float] | None = None,
    outcome: Any = None,
    weights: str | None = None,
) -> tuple[Any, Any]:
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W"], weights=weights
    )
    settings: dict[str, Any] = {} if g_bounds is None else {"g_bounds": g_bounds}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = (
            TMLE(
                outcome_learner=TableOutcome() if outcome is None else outcome,
                treatment_learner=SaturatedCategorical(),
                cross_fit=False,
                estimands=("ate",),
                reference=0.0,
                tol=1e-12,
                random_state=0,
                **settings,
            )
            .fit(data)
            .single()
        )
    return fit, data


@functools.cache
def six(g_bounds: tuple[float, float] | None = None) -> tuple[Any, Any, pd.DataFrame]:
    frame = six_frame()
    fit, data = six_fit(frame, g_bounds=g_bounds)
    return fit, data, frame


def six_scores(frame: pd.DataFrame, names: tuple[str, ...] = ("S", "SB")) -> dict[str, Any]:
    return {name: frame[name].to_numpy(float) for name in names}


def six_assess(scale: str, names: tuple[str, ...] = ("S", "SB"), **kwargs: Any) -> Any:
    fit, data, frame = six(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ec.score_assessment(fit, data, six_scores(frame, names), scale=scale)


def lt(theta: Any) -> Any:
    return np.logaddexp(0.0, -theta)


def lr(theta: Any) -> Any:
    return np.logaddexp(0.0, theta)


def six_calibration(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The log-CRR projection of the cell regressions onto ``(1, s)``, cells equally likely."""
    phi = np.column_stack([np.ones(6), s])
    beta = np.zeros(2)
    for _ in range(200):
        e = expit(phi @ beta)
        gradient = phi.T @ ((q[:, 0] + q[:, 1]) * e - q[:, 1])
        hessian = phi.T @ (((q[:, 0] + q[:, 1]) * e * (1 - e))[:, None] * phi)
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-15:
            break
    return beta


def six_truth(q: np.ndarray, scale: str, scores: tuple[np.ndarray, ...]) -> np.ndarray:
    """Each score's three parameters, then the paired row, under cell regressions ``q``."""
    q_r, q_t = q[:, 0], q[:, 1]
    out: list[float] = []
    risks = []
    for s in scores:
        phi = np.column_stack([np.ones(6), s])
        if scale == "difference":
            tau = q_t - q_r
            risk = float(np.mean(s**2 - 2 * s * tau))
            out.append(risk + float(np.mean(tau)) ** 2)
            out.extend(np.linalg.solve(phi.T @ phi, phi.T @ tau))
        else:
            c = float(np.log(np.mean(q_t) / np.mean(q_r)))
            risk = float(np.mean(q_t * lt(s) + q_r * lr(s)))
            out.append(risk - float(np.mean(q_t * lt(c) + q_r * lr(c))))
            out.extend(six_calibration(q, s))
        risks.append(risk)
    if len(scores) == 2:
        out.append(risks[0] - risks[1])
    return np.array(out)


def six_numerators(scale: str, scores: tuple[np.ndarray, ...]) -> np.ndarray:
    """The plan's numerator columns over the six cells, written out independently of the code.

    The difference scale returns ``(6, p)`` columns; the log-ratio scale the stacked ``(12, p)``
    pairs at the true calibrations.
    """
    if scale == "difference":
        return np.column_stack([np.ones(6), *scores])
    ones, zeros = np.ones(6), np.zeros(6)
    at_t, at_r = [ones, zeros], [zeros, ones]
    for s in scores:
        at_t.append(lt(s))
        at_r.append(lr(s))
        e = expit(np.column_stack([ones, s]) @ six_calibration(SIX_Q, s))
        for column in (ones, s):
            at_t.append(column * (e - 1))
            at_r.append(column * e)
    return np.vstack([np.column_stack(at_t), np.column_stack(at_r)])


class TestTheSixCellWitnessLaw:
    def test_the_law_is_exact_and_the_witnesses_are_not_small(self) -> None:
        frame = six_frame()
        cells = frame.groupby(["W", "A"])["Y"].agg(["mean", "size"]).reset_index()
        for _, row in cells.iterrows():
            w, a = int(row["W"]), int(row["A"])
            assert row["mean"] == pytest.approx(SIX_Q[w, a], abs=1e-12)
        treated = frame.groupby("W")["A"].mean().to_numpy()
        np.testing.assert_allclose(treated, SIX_G, atol=1e-12)
        pair = (S6, S6B)
        assert abs(six_truth(SIX_Q, "difference", pair)[-1]) > 0.1
        assert abs(six_truth(SIX_Q, "log_ratio", pair)[-1]) > 0.01
        affine = np.column_stack([np.ones(6), S6])
        residual = S6B - affine @ np.linalg.lstsq(affine, S6B, rcond=None)[0]
        assert np.max(np.abs(residual)) > 0.5

    def test_the_spans_do_not_saturate_the_cells(self) -> None:
        """A saturated span targets the truth whatever its numerators, so the mutations of T7
        would be blind; each span must leave directions of the law untouched."""
        rank = np.linalg.matrix_rank
        assert rank(six_numerators("difference", (S6, S6B))) == 3 < 6
        assert rank(six_numerators("log_ratio", (S6, S6B))) == 8 < 12
        assert rank(six_numerators("log_ratio", (S6,))) == 5 < 12

    @_quiet
    def test_the_fitted_spans_match_the_written_ones(self) -> None:
        assert six_assess("difference").span["rank"] == 3.0
        assert six_assess("log_ratio").span["rank"] == 8.0
        assert six_assess("log_ratio", ("S",)).span["rank"] == 5.0

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    @_quiet
    def test_t2_the_paired_row_is_the_difference_of_the_single_rows_of_one_fit(
        self, scale: str
    ) -> None:
        joint = six_assess(scale, g_bounds=TIGHT_BOUNDS)
        assert joint.compared is not None
        own, other = joint["risk_difference"], joint.compared["risk_difference"]
        pair = joint[joint.paired_name]
        assert pair.psi == pytest.approx(own.psi - other.psi, abs=1e-12)
        np.testing.assert_allclose(
            np.asarray(pair.influence_curve),
            np.asarray(own.influence_curve) - np.asarray(other.influence_curve),
            atol=1e-12,
        )
        # Mutation: the paired row from two separate single-score fluctuations.
        alone = six_assess(scale, ("S",), g_bounds=TIGHT_BOUNDS)["risk_difference"].psi
        alone_b = six_assess(scale, ("SB",), g_bounds=TIGHT_BOUNDS)["risk_difference"].psi
        assert abs((alone - alone_b) - pair.psi) > MUTATION_FLOOR

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    @_quiet
    def test_t7_every_targeted_estimate_is_the_functional(self, scale: str) -> None:
        truth = six_truth(SIX_Q, scale, (S6, S6B))
        np.testing.assert_allclose(values(six_assess(scale)), truth, atol=1e-8, rtol=0)
        single = six_truth(SIX_Q, scale, (S6,))
        np.testing.assert_allclose(values(six_assess(scale, ("S",))), single, atol=1e-8)
        # Witness: the untargeted plug-in at the wrong regression is far off.
        assert np.max(np.abs(six_truth(SIX_QI, scale, (S6, S6B)) - truth)) > 1e-2

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    @_quiet
    def test_t7_mutation_a_dropping_the_second_score_from_the_span(
        self, scale: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if scale == "difference":
            columns = ec._difference_columns
            monkeypatch.setattr(ec, "_difference_columns", lambda scores: columns(scores[:1]))
        else:
            numerators = ec._log_ratio_numerators
            monkeypatch.setattr(
                ec, "_log_ratio_numerators", lambda scores, betas: numerators(scores[:1], betas[:1])
            )
        truth = six_truth(SIX_Q, scale, (S6, S6B))[-1]
        assert abs(values(six_assess(scale))[-1] - truth) > MUTATION_FLOOR

    @_quiet
    def test_t7_mutation_b_contrast_numerators_on_the_log_ratio_scale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        numerators = ec._log_ratio_numerators

        def contrast(scores: Any, betas: Any) -> Any:
            at_t, _ = numerators(scores, betas)
            return at_t, -at_t

        monkeypatch.setattr(ec, "_log_ratio_numerators", contrast)
        truth = six_truth(SIX_Q, "log_ratio", (S6, S6B))
        assert np.max(np.abs(values(six_assess("log_ratio")) - truth)) > MUTATION_FLOOR

    @pytest.mark.parametrize("names", [("S", "SB"), ("S",)])
    @_quiet
    def test_t7_mutation_c_calibration_columns_frozen_at_the_initial_coefficients(
        self, names: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One fluctuation with the calibration columns at beta^0, and the projection of that
        frozen fit as the calibration: the alternation is what makes it exact."""
        fit, data, frame = six()
        arrays = [frame[name].to_numpy(float) for name in names]
        exact = [ec._target_scores(fit, data, arrays, scale="log_ratio", pair=(1.0, 0.0))]
        monkeypatch.setattr(ec, "DEFAULT_MAX_OUTER", 0)
        frozen = ec._target_scores(fit, data, arrays, scale="log_ratio", pair=(1.0, 0.0))
        weights = np.ones(data.n)
        for targeted, floor in ((exact[0], None), (frozen, MUTATION_FLOOR)):
            pooled = targeted.pooled
            misses = [
                np.max(
                    np.abs(
                        ec._log_ratio_projection(pooled.q_t, pooled.q_r, s, weights)
                        - six_calibration(SIX_Q, s[::PER_CELL])
                    )
                )
                for s in arrays
            ]
            if floor is None:
                assert max(misses) < 1e-8
            else:
                assert max(misses) > floor
        # The reported calibration of the frozen loop is beta^0 itself, the plug-in's.
        scores = tuple(S6 if name == "S" else S6B for name in names)
        truth = six_truth(SIX_Q, "log_ratio", scores)
        assert np.max(np.abs(values(six_assess("log_ratio", names)) - truth)) > MUTATION_FLOOR

    @_quiet
    def test_t7_a_slope_at_a_collinear_point_settles_on_the_reduced_span(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``score-assessment`` pilot draw whose calibration slope lands near 0.5, where the
        log-ratio numerators are collinear to about 2e-6 (relative): the full-span rounds cycle,
        and the rounds without the thinnest direction settle with every numerator within
        ``_REDUCED_SPAN_BOUND`` of the span."""
        from tests.studies import score_assessment as sa
        from tests.studies.evidence.seeds import stream_seed

        group = sa.PILOT_GROUP_BY_LABEL["single_crr_2k"]
        seed = stream_seed(sa.STUDY, "se_ratio_pilot", group.label, 198)
        frame, truths = sa.draw_from_seed(group.label, group.n, seed)
        effect = CausalStudy(sa.with_scores(group, frame), design=sa.design(group)).identify(
            ConditionalContrast(sa.MODIFIERS, scale="log_ratio", model=sa.BASIS)
        )

        def assess() -> Any:
            return effect.assess_score(
                "S",
                computed_from=["W1"],
                score_kind="external",
                **sa.learners(group),
                random_state=198,
            )

        settled = assess()
        assert settled.span["rank"] == settled.span["columns"] - 1
        assert 0.0 < settled.span["out_of_span"] <= ec._REDUCED_SPAN_BOUND
        slope = settled["calibration[S]"]
        assert abs(slope.psi - 0.5) < 0.01
        assert abs(slope.psi - truths["S_calibration_slope"]) < 4.0 * slope.std_error
        monkeypatch.setattr(ec, "_REDUCED_SPAN_BOUND", -1.0)
        with pytest.raises(DataError, match="did not settle"):
            assess()

    @_quiet
    def test_t18_a_reduced_span_fit_solves_every_score_equation_and_reads_experimental(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The T7a witness: the accepted reduced fit meets ``scaled_var`` on every original
        numerator and reads experimental.  Mutation: a retry that keeps one direction leaves
        the others' equations unsolved.  On the same draw from the declared wrong outcome
        regression of ``noweight_law``, whose full span also cycles, and with the geometric
        bound loosened, only the score test refuses it; skipping the test accepts it."""
        from sklearn.linear_model import LogisticRegression

        from tests.studies import conditional_projection as cp
        from tests.studies import score_assessment as sa
        from tests.studies.evidence.seeds import stream_seed

        group = sa.PILOT_GROUP_BY_LABEL["single_crr_2k"]
        seed = stream_seed(sa.STUDY, "se_ratio_pilot", group.label, 198)
        frame, _ = sa.draw_from_seed(group.label, group.n, seed)
        effect = CausalStudy(sa.with_scores(group, frame), design=sa.design(group)).identify(
            ConditionalContrast(sa.MODIFIERS, scale="log_ratio", model=sa.BASIS)
        )

        def assess(**learners: Any) -> Any:
            return effect.assess_score(
                "S",
                computed_from=["W1"],
                score_kind="external",
                **(learners or sa.learners(group)),
                random_state=198,
            )

        judged: list[float] = []
        ratio = ec._numerator_score_ratio

        def spy(*arguments: Any) -> float:
            judged.append(ratio(*arguments))
            return judged[-1]

        monkeypatch.setattr(ec, "_numerator_score_ratio", spy)
        settled = assess()
        assert settled.span["reduced"] >= 1
        assert judged and judged[-1] <= 1.0
        assert settled.status == "experimental"
        assert "dropping a collinear direction" in settled.status_reason
        for _, _, estimate in settled.rows():
            assert estimate.support.label == "experimental"
            assert "collinear direction" in estimate.support.reason

        basis = ec.arm_span_basis

        def one_direction(at_t: Any, at_r: Any, rank: int | None = None) -> Any:
            return basis(at_t, at_r, None if rank is None else 1)

        wrong = {
            "outcome_learner": cp.DeclaredOutcome("noweight"),
            "treatment_learner": LogisticRegression(max_iter=2_000),
        }
        monkeypatch.setattr(ec, "arm_span_basis", one_direction)
        monkeypatch.setattr(ec, "_REDUCED_SPAN_BOUND", math.inf)
        judged.clear()
        with pytest.raises(DataError, match="did not settle"):
            assess(**wrong)
        assert judged and min(judged) > 1.0
        monkeypatch.setattr(ec, "_numerator_score_ratio", lambda *arguments: 0.0)
        assert assess(**wrong).span["rank"] == 1

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    @_quiet
    def test_t8_the_substitution_equals_the_one_step_at_the_targeted_fit(self, scale: str) -> None:
        gap, initial_gap = self._identity_gaps(scale)
        assert gap < 1e-8
        # Witness: both nuisances are wrong (the bounds bind), so targeting moved the fit.
        assert initial_gap > 1e-2

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    @_quiet
    def test_t8_an_untruncated_mechanism_in_the_clever_covariate_breaks_the_identity(
        self, scale: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        propensity = pt.clever_propensity
        monkeypatch.setattr(
            pt, "clever_propensity", lambda nuisance, bounds: propensity(nuisance, (1e-6, 1 - 1e-6))
        )
        gap, _ = self._identity_gaps(scale)
        assert gap > MUTATION_FLOOR

    @staticmethod
    def _identity_gaps(scale: str) -> tuple[float, float]:
        fit, data, frame = six(TIGHT_BOUNDS)
        scores = six_scores(frame)
        arrays = list(scores.values())
        targeted = ec._target_scores(fit, data, arrays, scale=scale, pair=(1.0, 0.0))
        pooled = targeted.pooled
        assessment = ec.score_assessment(fit, data, scores, scale=scale)
        weights = np.asarray(data.weights, dtype=float)
        at_target = ec._one_step_estimates(
            pooled.q_t,
            pooled.q_r,
            pooled.targeted_t,
            pooled.targeted_r,
            scores,
            scale=scale,
            weights=weights,
        )
        initial = ec._one_step_estimates(
            pooled.initial_t,
            pooled.initial_r,
            pooled.d_t,
            pooled.d_r,
            scores,
            scale=scale,
            weights=weights,
        )

        def flat(estimates: dict[str, dict[str, float]]) -> np.ndarray:
            return np.array([value for name in scores for value in estimates[name].values()])

        reported = values(assessment)
        order = [0, 1, 2, 4, 5, 6, 3]  # the one-step dict holds the pair after the column's three
        return (
            float(np.max(np.abs(flat(at_target)[order] - reported))),
            float(np.max(np.abs(flat(initial)[order] - reported))),
        )

    @_quiet
    def test_t17_an_alternation_that_cannot_settle_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ec, "DEFAULT_MAX_OUTER", 1)
        with pytest.raises(DataError, match="did not settle") as caught:
            six_assess("log_ratio")
        # One round on the full span, then one on each reduced span (one per score).
        assert "after 3 rounds" in str(caught.value)

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    @_quiet
    def test_t11_integer_weights_equal_the_duplicated_frame(self, scale: str) -> None:
        weighted = six_frame(weights=True)
        duplicated = weighted.loc[weighted.index.repeat(weighted["wt"])].reset_index(drop=True)
        assessed = []
        for frame, weights in ((weighted, "wt"), (duplicated, None)):
            fit, data = six_fit(frame, g_bounds=TIGHT_BOUNDS, weights=weights)
            assessed.append(ec.score_assessment(fit, data, six_scores(frame), scale=scale))
        left, right = assessed
        np.testing.assert_allclose(values(left), values(right), atol=1e-11)
        # The weighted curve is the normalised weight times the duplicated curve at each row.
        mass = weighted["wt"].to_numpy(float)
        first_copy = np.concatenate([[0], np.cumsum(mass[:-1])]).astype(int)
        for (_, _, a), (_, _, b) in zip(left.rows(), right.rows(), strict=True):
            np.testing.assert_allclose(
                np.asarray(a.influence_curve),
                mass / mass.mean() * np.asarray(b.influence_curve)[first_copy],
                atol=1e-11,
            )
        assert left.status == "experimental" and "observation weights" in left.status_reason


# ----------------------------------------------------------------- bounds (T9)


@_quiet
def test_t9_the_targeted_fit_stays_in_range_where_one_step_weights_turn_negative() -> None:
    rng = np.random.default_rng(5)
    n = 800
    w = rng.uniform(-1, 1, n)
    a = rng.binomial(1, expit(3.5 * w))
    y = rng.binomial(1, expit(-0.4 + 0.8 * w + a * (0.5 - w)))
    frame = pd.DataFrame({"Y": y, "A": a, "W": w})
    data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"])
    fit = (
        TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            estimands=("ate",),
            reference=0.0,
            g_bounds=(0.01, 0.99),
            random_state=0,
        )
        .fit(data)
        .single()
    )
    score = {"S": 0.2 - 0.5 * w}
    pooled = ec._target_scores(fit, data, [score["S"]], scale="log_ratio", pair=(1.0, 0.0)).pooled
    tau = pooled.q_t - pooled.q_r
    assert np.all(np.abs(tau) <= 1.0) and np.all(pooled.q_t + pooled.q_r > 0.0)
    # Witness: the one-step loss has a negative weight on some row, so it is not convex there.
    assert np.min(pooled.d_t + pooled.d_r) < 0.0
    assessment = ec.score_assessment(fit, data, score, scale="log_ratio")
    beta = np.array([assessment["calibration[(intercept)]"].psi, assessment["calibration[S]"].psi])
    phi = np.column_stack([np.ones(n), score["S"]])
    e = expit(phi @ beta)
    total = pooled.q_t + pooled.q_r
    gradient = phi.T @ (total * e - pooled.q_t) / n
    hessian = phi.T @ ((total * e * (1 - e))[:, None] * phi) / n
    assert np.max(np.abs(gradient)) < 1e-8
    assert np.min(np.linalg.eigvalsh(hessian)) > 0.0
    # Mutation: the one-step calibration solves another equation, so its gradient is not 0.
    initial = ec._one_step_estimates(
        pooled.initial_t,
        pooled.initial_r,
        pooled.d_t,
        pooled.d_r,
        score,
        scale="log_ratio",
        weights=np.ones(n),
    )["S"]
    step = np.array([initial["calibration[(intercept)]"], initial["calibration[S]"]])
    e_step = expit(phi @ step)
    assert np.max(np.abs(phi.T @ (total * e_step - pooled.q_t) / n)) > 1e-4


# ----------------------------------------------------------------- clusters (T12)


def _clustered(clusters: int) -> Any:
    rng = np.random.default_rng(clusters)
    n = 400
    w = rng.uniform(-1, 1, n)
    a = rng.binomial(1, expit(0.5 * w))
    y = rng.binomial(1, expit(-0.2 + 0.6 * w + a * (0.3 + 0.5 * w)))
    frame = pd.DataFrame(
        {
            "Y": y,
            "A": a,
            "W": w,
            "site": np.arange(n) % clusters,
            "S": 0.1 + 0.2 * w,
            "SB": 0.1 + 0.1 * w**2,
        }
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), cluster="site"),
    )
    effect = study.identify(ConditionalContrast(Modifiers(["W"])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.assess_score(
            "S",
            computed_from=["W"],
            score_kind="external",
            compare="SB",
            compare_computed_from=["W"],
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            cross_fit=False,
            random_state=0,
        ), frame["site"].to_numpy()


@pytest.mark.parametrize(
    ("clusters", "status", "df"),
    [
        (20, "influence_curve", 18),
        (5, "few_cluster_plugin", 3),
        (2, "no_cluster_variance_plugin", None),
    ],
)
def test_t12_clustered_rows_sum_the_curve_and_take_the_few_cluster_reference(
    clusters: int, status: str, df: int | None
) -> None:
    assessment, site = _clustered(clusters)
    for _, _, estimate in assessment.rows():
        assert estimate.inference == status
        assert estimate.reference_df == df
        curve = np.asarray(estimate.influence_curve)
        n = curve.size
        sums = np.array([curve[site == j].sum() for j in range(clusters)])
        se = float(np.sqrt(clusters / (clusters - 1) * np.sum((sums - sums.mean()) ** 2) / n**2))
        if df is None:
            with pytest.raises(CapabilityError):
                _ = estimate.ci
            assert estimate.plugin_std_error == pytest.approx(se, rel=1e-10)
            continue
        assert estimate.std_error == pytest.approx(se, rel=1e-10)
        # The row-level SE differs: the cluster sum is what is reported.
        row_level = float(np.sqrt(np.sum(curve**2)) / n)
        assert abs(row_level - se) > 1e-3 * se
        low, high = estimate.ci
        half = (high - low) / 2.0
        assert half == pytest.approx(float(stats.t.ppf(0.975, df)) * se, rel=1e-10)
        assert half > 1.96 * se
    assert assessment.status == "experimental" and "clustered rows" in assessment.status_reason


# ----------------------------------------------------------------- support (T14)

_MEASURED_FLAGS = {
    "weighted": False,
    "clustered": False,
    "missing": False,
    "arms": 2,
    "cross_fitted": True,
}


@pytest.mark.parametrize("key", sorted(ec.SCORE_ASSESSMENT_MEASURED))
def test_t14_each_measured_configuration_reads_supported_with_its_studies(
    key: tuple[str, str, str],
) -> None:
    scale, outcome, mechanism = key
    single, pair = ec.score_assessment_support(
        scale=scale, outcome=outcome, known_mechanism=mechanism == "known", **_MEASURED_FLAGS
    )
    assert single.label == "supported"
    named = [item.split(":")[0].strip() for item in ec.SCORE_ASSESSMENT_MEASURED[key].split(";")]
    assert single.evidence == tuple(named)
    assert pair.label == ("supported" if key in ec.SCORE_PAIRED_MEASURED else "experimental")


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"weighted": True}, "observation weights"),
        ({"clustered": True}, "clustered rows"),
        ({"missing": True}, "missing outcomes"),
        ({"arms": 3}, "multi-arm"),
        ({"cross_fitted": False}, "not cross-fitted"),
        ({"reduced_span": True, "scale": "log_ratio"}, "collinear direction"),
        ({"outcome": "continuous", "scale": "log_ratio"}, "measures this configuration"),
        ({"known_mechanism": True, "scale": "log_ratio"}, "measures this configuration"),
    ],
)
def test_t14_each_departure_reads_experimental_with_its_reason(
    change: dict[str, Any], reason: str
) -> None:
    arguments: dict[str, Any] = {
        "scale": "difference",
        "outcome": "binary",
        "known_mechanism": False,
        **_MEASURED_FLAGS,
    }
    arguments.update(change)
    single, pair = ec.score_assessment_support(**arguments)
    assert single.label == pair.label == "experimental"
    assert reason in single.reason.lower() and single.evidence == ()


def test_t14_a_paired_row_outside_its_table_reads_experimental_beside_supported_singles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = ("difference", "binary", "estimated")
    monkeypatch.setattr(
        ec,
        "SCORE_PAIRED_MEASURED",
        {k: v for k, v in ec.SCORE_PAIRED_MEASURED.items() if k != key},
    )
    single, pair = ec.score_assessment_support(
        scale="difference", outcome="binary", known_mechanism=False, **_MEASURED_FLAGS
    )
    assert single.label == "supported"
    assert pair.label == "experimental"
    assert "paired risk difference" in pair.reason


@_quiet
def test_t14_a_fitted_assessment_carries_its_support_on_every_row() -> None:
    assessment = paired("difference")
    # The three-cell oracle fit is not cross-fitted, which no study measures.
    labels = {estimate.support.label for _, _, estimate in assessment.rows()}
    assert labels == {"experimental"}
    assert "not cross-fitted" in assessment.status_reason
    assert assessment.compared is not None
    assert assessment.compared.status == "experimental"


def _trial_effect(scale: str) -> Any:
    """A binary trial with the mechanism declared known, cross-fitted, iid and unweighted."""
    rng = np.random.default_rng(7)
    n = 600
    w = rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, expit(-0.3 + 0.6 * w + a * (0.3 + 0.5 * w)))
    frame = pd.DataFrame(
        {"Y": y, "A": a, "W": w, "p0": 0.5, "p1": 0.5, "S": 0.1 + 0.1 * w, "SB": 0.1 * w**2}
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            treatment_probabilities={0: "p0", 1: "p1"},
        ),
    )
    effect = study.identify(ConditionalContrast(Modifiers(["W"])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.assess_score(
            "S",
            computed_from=["W"],
            score_kind="external",
            compare="SB",
            compare_computed_from=["W"],
            scale=scale,
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
        )


def test_t14_a_fitted_known_mechanism_reads_its_own_configuration() -> None:
    difference = _trial_effect("difference")
    assert difference.status == "supported"
    for _, name, estimate in difference.rows():
        assert estimate.support.label == "supported"
        assert estimate.support.evidence == ("score-assessment",)
        assert "paired_trial_2k" in estimate.support.reason, name
    log_ratio = _trial_effect("log_ratio")
    assert log_ratio.status == "experimental"
    assert "measures this configuration" in log_ratio.status_reason
    assert {e.support.label for _, _, e in log_ratio.rows()} == {"experimental"}


def _measured_cells() -> list[tuple[str, str]]:
    cells = []
    for table in (ec.SCORE_ASSESSMENT_MEASURED, ec.SCORE_PAIRED_MEASURED):
        for entry in table.values():
            for item in entry.split(";"):
                study, groups = item.split(":", 1)
                cells += [(study.strip(), group.strip()) for group in groups.split(",")]
    return sorted(set(cells))


@pytest.mark.parametrize(("slug", "group"), _measured_cells())
def test_t14_every_group_a_table_names_ran_and_passed(slug: str, group: str) -> None:
    from tests.studies.evidence.registry import registered

    records = {record.slug: record for record in registered()}
    assert slug in records, f"{slug} is not a registered study"
    record = records[slug]
    if slug == "score-assessment":
        from tests.studies import score_assessment as sa

        # A group's cells by the study's own naming: the paired Wald cells carry no group prefix.
        cells = [
            cell for names in sa.group_cells(sa.GROUP_BY_LABEL[group]).values() for cell in names
        ]
    else:
        cells = [
            cell
            for names in record.property_cells.values()
            for cell in names
            if cell.startswith(group)
        ]
    assert cells, f"{slug} declares no property cell of group {group}"
    table = pd.read_csv(record.artifacts / "properties.csv")
    rows = table[table["cell"].isin(cells)]
    assert set(rows["cell"]) == set(cells)
    assert (rows["role"] == "positive").any()
    assert rows["passed"].astype(bool).all(), rows[["cell", "passed"]]
    assert rows["property_passed"].astype(bool).all(), rows[["cell", "property_passed"]]


# ----------------------------------------------------------------- report, frame, repr (T16)


@_quiet
def test_t16_the_frame_has_one_row_per_parameter_with_its_score() -> None:
    assessment = paired("difference")
    frame = assessment.to_frame()
    assert list(frame["score"]) == ["S"] * 3 + ["SB"] * 3 + ["S - SB"]
    assert list(frame["estimand"]) == [
        "risk_difference",
        "calibration[(intercept)]",
        "calibration[S]",
        "risk_difference",
        "calibration[(intercept)]",
        "calibration[SB]",
        "risk_difference[S - SB]",
    ]
    np.testing.assert_allclose(frame["psi"].to_numpy(float), values(assessment))
    assert {"support", "support_reason", "std_err", "ci_lower", "ci_upper"} <= set(frame.columns)
    single = law_effect().assess_score(
        "S", computed_from=["W"], score_kind="external", **law_oracle()
    )
    assert list(single.to_frame()["score"]) == ["S"] * 3
    assert single.compare is None and single.compared is None and single.paired_name is None


@_quiet
def test_t16_the_report_names_the_method_the_pair_and_the_status() -> None:
    assessment = paired("log_ratio")
    summary = assessment.summary()
    header = dict(summary.header)
    assert header["Method"].startswith("TMLE")
    assert "compared with SB" in header["Question"]
    assert header["Status"] == "experimental"
    (table,) = summary.tables
    assert table.columns[0] == "score" and table.columns[-1] == "status"
    assert len(table.rows) == 7
    assert set(summary.note_codes) >= {"score_external", "score_paired", "score_experimental"}
    full = str(assessment.summary(detail="full"))
    assert "P-values against 0" in full and "Model update" in full
    single = law_effect().assess_score(
        "S", computed_from=["W"], score_kind="external", **law_oracle()
    )
    assert "score_paired" not in single.summary().note_codes


@_quiet
def test_t16_the_repr_shows_the_paired_value() -> None:
    assessment = paired("difference")
    text = repr(assessment)
    assert text.startswith("ScoreAssessment('S' vs 'SB', paired=")
    assert f"{assessment[assessment.paired_name].psi:.4g}" in text
    single = law_effect().assess_score(
        "S", computed_from=["W"], score_kind="external", **law_oracle()
    )
    assert repr(single).startswith("ScoreAssessment('S', risk_difference=")


@_quiet
def test_cellmeans_learner_on_the_six_cell_law_is_the_oracle() -> None:
    """With the saturated regression the fit is exact, and every estimate is the functional."""
    fit, data = six_fit(six_frame(), outcome=CellMeans())
    frame = six_frame()
    assessment = ec.score_assessment(fit, data, six_scores(frame), scale="log_ratio")
    np.testing.assert_allclose(
        values(assessment), six_truth(SIX_Q, "log_ratio", (S6, S6B)), atol=1e-10
    )
