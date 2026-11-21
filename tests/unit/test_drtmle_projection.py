r"""The DR-TMLE projection of a conditional contrast on the difference scale (P1 PR 3, c1).

``DRTMLE(effect_modification=...)`` fits one block of reduced regressions per basis term and
solves, beside the projection's own score :math:`P_n[\varphi s(A)/g\,(Y - \bar Q^*)] = 0`,
equation (9) along ``2p`` mechanism columns and equation (10) along ``p`` outcome columns per
arm. The coefficient's curve is

.. math::

    M^{-1}\Big[\varphi\{\tau^* - \varphi^\top\beta\} + \varphi\,\frac{s(A)}{g_A}\{Y - \bar Q^*\}
        - \sum_a s(a)\{D_{A,a} + D_{Y,a}\}\Big],

with :math:`D_{A,j,a} = Q_{r,j,a}/g_a\,(1_a - g_a)` and
:math:`D_{Y,j,a} = 1_a\,g_{r,2,j,a}/g_{r,1,a}\,(Y - \bar Q^*_a)`, one per solved equation.

**The witnesses.** On law L-g (a propensity tie across ``w = 0, 1``) with a declared mechanism
and an arm-1 outcome error, :math:`D_A` is alive and :math:`D_Y` vanishes, and the reduced
regression pools the tied cells, so the curve differs from the efficient one and from both
weight mutants. On law L-Q (an outcome tie on arm 1) with arm 1's mechanism wrong and the
oracle outcome regression, :math:`D_Y` is alive and :math:`D_A` vanishes. Each closed form here
is computed from the cell counts, with no library code.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import root
from sklearn.base import BaseEstimator

from cleverly import CausalStudy, ConditionalContrast, DRTMLEMethod, PointTreatment
from cleverly.bases import Linear, Modifiers
from cleverly.effect_modification import EffectModification
from cleverly.estimators import DRTMLE, TMLE
from cleverly.estimators.drtmle import (
    PROJECTION_BIVARIATE_REFUSAL,
    PROJECTION_MISSING_REFUSAL,
    PROJECTION_ONE_STEP_REFUSAL,
    PROJECTION_RATIO_REFUSAL,
    PROJECTION_STRATA_REFUSAL,
)
from cleverly.exceptions import CapabilityError
from tests import discrete_law as law
from tests import discrete_law_multi as multi
from tests.conftest import OracleOutcome, OracleTreatment
from tests.discrete_law_longitudinal import CellMeans
from tests.studies.drtmle_one_wrong import CubicLeastSquares, CubicLogistic, OffsetLogistic
from tests.unit._natural_course_support import NeverFit
from tests.unit.test_influence_gateaux_drtmle import Misspecified
from tests.unit.test_reduced_projection import G_TIED, G_WRONG, Q_TIED, Q_WRONG_ARM1

EXACT = 1e-10
FLOOR = 1e-3
INERT = (1e-6, 1.0 - 1e-6)
TERMS = ("(intercept)", "W")


def projection(scale: str = "difference") -> EffectModification:
    return EffectModification(Linear(["W"]), Modifiers(["W"]), scale)  # type: ignore[arg-type]


def estimator(g_hat: np.ndarray, q_hat: np.ndarray, **options: Any) -> DRTMLE:
    dgp = Misspecified(g_hat, q_hat)
    settings: dict[str, Any] = {
        "outcome_learner": OracleOutcome(dgp),
        "treatment_learner": OracleTreatment(dgp),
        "reduced_outcome_learner": CellMeans(),
        "reduced_mechanism_residual_learner": CellMeans(),
        "reduced_treatment_learner": CellMeans(),
        "g_bounds": INERT,
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "effect_modification": projection(),
    }
    settings.update(options)
    return DRTMLE(**settings)


def fit(frame: pd.DataFrame, est: Any, known: np.ndarray | None = None, **extra: Any) -> Any:
    keywords: dict[str, Any] = dict(extra)
    if known is not None:
        keywords["treatment_probabilities"] = known[frame["W"].to_numpy().astype(int)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return est.fit(frame, outcome="Y", treatment="A", **keywords).single()


def curves(result: Any) -> np.ndarray:
    return np.column_stack(
        [np.asarray(result[f"cate_projection[{t}]"].influence_curve) for t in TERMS]
    )


# ------------------------------------------------------------------ the closed forms


class Law:
    """A variant of :mod:`tests.discrete_law` and its closed forms, from the counts alone."""

    def __init__(self, g: np.ndarray, q: np.ndarray) -> None:
        self.counts = law.cell_counts(g=g, q=q)
        self.frame = law.frame(self.counts)
        probs = self.counts / self.counts.sum()
        self.mass = probs.sum(axis=2)  # P(W = w, A = a)
        self.p_w = self.mass.sum(axis=1)
        self.g0 = self.mass[:, 1] / self.p_w
        self.q0 = probs[:, :, 1] / self.mass
        self.phi = np.column_stack([np.ones(3), np.arange(3.0)])
        self.gram = self.phi.T @ (self.p_w[:, None] * self.phi)
        self.beta = np.linalg.solve(
            self.gram, self.phi.T @ (self.p_w * (self.q0[:, 1] - self.q0[:, 0]))
        )
        self.w = self.frame["W"].to_numpy().astype(int)
        self.a = self.frame["A"].to_numpy().astype(int)
        self.y = self.frame["Y"].to_numpy()

    def targeted(self, q_hat: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
        """The projection fluctuation of ``q_hat`` along ``phi s(a)/g_a``, solved longhand."""
        mech = np.column_stack([1.0 - g_hat, g_hat])
        sign = np.array([-1.0, 1.0])
        eta = np.log(q_hat / (1.0 - q_hat))

        def star(eps: np.ndarray) -> np.ndarray:
            shift = (self.phi @ eps)[:, None] * sign[None, :] / mech
            return 1.0 / (1.0 + np.exp(-(eta + shift)))

        def score(eps: np.ndarray) -> np.ndarray:
            inner = self.mass * sign[None, :] / mech * (self.q0 - star(eps))
            return np.asarray(self.phi.T @ inner.sum(axis=1))

        solved = root(score, np.zeros(2), method="hybr", tol=1e-15)
        assert float(np.max(np.abs(score(solved.x)))) < 1e-14
        return star(solved.x)

    @staticmethod
    def _level(values: np.ndarray, weight: np.ndarray, design: np.ndarray) -> np.ndarray:
        out = np.empty_like(values)
        for level in np.unique(np.round(design, 12)):
            cells = np.round(design, 12) == level
            out[cells] = np.sum(weight[cells] * values[cells]) / np.sum(weight[cells])
        return out

    def qr(
        self, q_star: np.ndarray, g_hat: np.ndarray, arm: int, mode: str = "inside"
    ) -> np.ndarray:
        """``Q_{r,j}`` per cell ``(3, 2)``; ``mode`` names a mutant."""
        mech = g_hat if arm == 1 else 1.0 - g_hat
        resid = self.q0[:, arm] - q_star[:, arm]
        weight = self.mass[:, arm]
        if mode == "inside":
            return np.column_stack(
                [self._level(self.phi[:, j] * resid, weight, mech) for j in range(2)]
            )
        pooled = self._level(resid, weight, mech)
        if mode == "after":
            return self.phi * pooled[:, None]
        return np.column_stack([pooled, pooled])  # "dropped"

    def gr(self, q_star: np.ndarray, g_hat: np.ndarray, arm: int, mode: str = "inside") -> tuple:
        mech = g_hat if arm == 1 else 1.0 - g_hat
        g0 = self.g0 if arm == 1 else 1.0 - self.g0
        ratio = (g0 - mech) / mech
        design = q_star[:, arm]
        gr1 = self._level(g0, self.p_w, design)
        if mode == "inside":
            gr2 = np.column_stack(
                [self._level(self.phi[:, j] * ratio, self.p_w, design) for j in range(2)]
            )
        else:
            pooled = self._level(ratio, self.p_w, design)
            gr2 = self.phi * pooled[:, None] if mode == "after" else np.column_stack([pooled] * 2)
        return gr1, gr2

    def curve(
        self,
        q_star: np.ndarray,
        g_hat: np.ndarray,
        *,
        d_a: bool,
        d_y: bool,
        mode: str = "inside",
    ) -> np.ndarray:
        """The coefficient curve per row ``(n, 2)``, the corrections chosen by ``d_a``, ``d_y``."""
        w, a, y = self.w, self.a, self.y
        phi = self.phi[w]
        mech = np.column_stack([1.0 - g_hat, g_hat])[w]
        sign = np.where(a == 1, 1.0, -1.0)
        tau = q_star[w, 1] - q_star[w, 0]
        beta = np.linalg.solve(self.gram, self.phi.T @ (self.p_w * (q_star[:, 1] - q_star[:, 0])))
        rows = phi * (tau - phi @ beta)[:, None]
        rows += phi * (sign / mech[np.arange(w.size), a] * (y - q_star[w, a]))[:, None]
        for arm, s in ((1, 1.0), (0, -1.0)):
            took = (a == arm).astype(float)[:, None]
            g_a = mech[:, arm][:, None]
            if d_a:
                rows -= s * self.qr(q_star, g_hat, arm, mode)[w] / g_a * (took - g_a)
            if d_y:
                gr1, gr2 = self.gr(q_star, g_hat, arm, mode)
                rows -= s * took * gr2[w] / gr1[w][:, None] * (y - q_star[w, arm])[:, None]
        gram = phi.T @ phi / w.size
        return np.asarray(np.linalg.solve(gram, rows.T).T)


@pytest.fixture(scope="module")
def law_g() -> Law:
    return Law(G_TIED, law.Q)


@pytest.fixture(scope="module")
def law_q() -> Law:
    return Law(law.G, Q_TIED)


@pytest.fixture(scope="module")
def known_fit(law_g: Law) -> Any:
    """The registered composition: declared mechanism, wrong outcome regression, in sample."""
    return fit(law_g.frame, estimator(G_TIED, Q_WRONG_ARM1), known=G_TIED)


class TestTheKnownMechanism:
    def test_the_coefficients_are_the_truth(self, known_fit: Any, law_g: Law) -> None:
        for j, term in enumerate(TERMS):
            estimate = known_fit[f"cate_projection[{term}]"]
            assert estimate.psi == pytest.approx(law_g.beta[j], abs=EXACT)
            assert estimate.inference == "influence_curve"

    def test_the_curve_subtracts_the_weighted_mechanism_correction(
        self, known_fit: Any, law_g: Law
    ) -> None:
        q_star = law_g.targeted(Q_WRONG_ARM1, G_TIED)
        expected = law_g.curve(q_star, G_TIED, d_a=True, d_y=True)
        np.testing.assert_allclose(curves(known_fit), expected, rtol=0.0, atol=EXACT)

    def test_the_correction_is_alive_and_both_mutants_depart(self, law_g: Law) -> None:
        q_star = law_g.targeted(Q_WRONG_ARM1, G_TIED)
        right = law_g.curve(q_star, G_TIED, d_a=True, d_y=True)
        plain = law_g.curve(q_star, G_TIED, d_a=False, d_y=False)
        assert float(np.max(np.abs(right - plain))) > FLOOR
        w = law_g.w
        after = law_g.curve(q_star, G_TIED, d_a=True, d_y=True, mode="after")
        dropped = law_g.curve(q_star, G_TIED, d_a=True, d_y=True, mode="dropped")
        # The weight-after mutant departs on the w = 0 rows, the drop-weight one on w = 2.
        slope_qr = law_g.qr(q_star, G_TIED, 1)[:, 1]
        assert slope_qr[0] != pytest.approx(0.0, abs=FLOOR)
        assert float(np.max(np.abs(right - after)[w == 0])) > FLOOR
        assert float(np.max(np.abs(right - dropped)[w == 2])) > FLOOR

    def test_equation_nine_is_solved_per_arm_and_term(self, known_fit: Any) -> None:
        mechanism = known_fit.fluctuations["effect_modification"].mechanism
        assert mechanism is not None
        assert np.asarray(mechanism.score).shape == (4,)
        assert float(np.max(np.abs(mechanism.score))) < 1e-8


class TestTheEstimatedMechanism:
    def test_the_oracle_point_is_exact_and_the_interval_withheld(self, law_g: Law) -> None:
        result = fit(law_g.frame, estimator(G_TIED, Q_WRONG_ARM1))
        q_star = law_g.targeted(Q_WRONG_ARM1, G_TIED)
        expected = law_g.curve(q_star, G_TIED, d_a=True, d_y=True)
        for j, term in enumerate(TERMS):
            estimate = result[f"cate_projection[{term}]"]
            assert estimate.psi == pytest.approx(law_g.beta[j], abs=EXACT)
            assert estimate.inference == "reduced_regression_plugin"
            assert not estimate.supplies_inference
            closed = float(np.sqrt(np.var(expected[:, j], ddof=1) / law_g.w.size))
            assert estimate.plugin_std_error == pytest.approx(closed, rel=1e-9)
            with pytest.raises(CapabilityError, match="projection"):
                estimate.ci()


class TestTheOutcomeCorrection:
    """Law L-Q: arm 1's mechanism wrong, the oracle outcome regression; ``D_Y`` alive."""

    @pytest.fixture(scope="class")
    def result(self, law_q: Law) -> Any:
        return fit(law_q.frame, estimator(G_WRONG, law_q.q0))

    def test_the_point_is_exact_and_the_curve_weights_d_y(self, result: Any, law_q: Law) -> None:
        for j, term in enumerate(TERMS):
            assert result[f"cate_projection[{term}]"].psi == pytest.approx(law_q.beta[j], abs=EXACT)
        expected = law_q.curve(law_q.q0, G_WRONG, d_a=True, d_y=True)
        np.testing.assert_allclose(curves(result), expected, rtol=0.0, atol=EXACT)

    def test_both_mutants_depart_on_the_tied_rows(self, law_q: Law) -> None:
        right = law_q.curve(law_q.q0, G_WRONG, d_a=True, d_y=True)
        tied = law_q.w <= 1
        for mode in ("after", "dropped"):
            mutant = law_q.curve(law_q.q0, G_WRONG, d_a=True, d_y=True, mode=mode)
            assert float(np.max(np.abs(right - mutant)[tied])) > FLOOR
        assert (
            float(np.max(np.abs(right - law_q.curve(law_q.q0, G_WRONG, d_a=True, d_y=False))))
            > FLOOR
        )


class TestTheGuards:
    def test_an_empty_guard_is_the_tmle_projection_bit_for_bit(self, law_g: Law) -> None:
        dr = fit(law_g.frame, estimator(G_TIED, Q_WRONG_ARM1, guard=()), known=G_TIED)
        dgp = Misspecified(G_TIED, Q_WRONG_ARM1)
        plain = fit(
            law_g.frame,
            TMLE(
                outcome_learner=OracleOutcome(dgp),
                treatment_learner=OracleTreatment(dgp),
                g_bounds=INERT,
                cross_fit=False,
                simultaneous=False,
                random_state=0,
                effect_modification=projection(),
            ),
            known=G_TIED,
        )
        for term in TERMS:
            name = f"cate_projection[{term}]"
            assert dr[name].psi == plain[name].psi
            np.testing.assert_array_equal(dr[name].influence_curve, plain[name].influence_curve)
            assert dr[name].inference == plain[name].inference == "influence_curve"

    def test_an_empty_guard_keeps_its_interval_with_an_estimated_mechanism(
        self, law_g: Law
    ) -> None:
        result = fit(law_g.frame, estimator(G_TIED, Q_WRONG_ARM1, guard=()))
        for term in TERMS:
            assert result[f"cate_projection[{term}]"].inference == "influence_curve"
        assert result.nuisance.reduced is None

    def test_the_q_guard_subtracts_d_a_only(self, law_q: Law) -> None:
        result = fit(law_q.frame, estimator(G_WRONG, law_q.q0, guard=("Q",)))
        expected = law_q.curve(law_q.q0, G_WRONG, d_a=True, d_y=False)
        np.testing.assert_allclose(curves(result), expected, rtol=0.0, atol=EXACT)

    def test_the_g_guard_subtracts_d_y_only(self, law_g: Law) -> None:
        # On L-g with a wrong outcome regression and a wrong mechanism both corrections live.
        g_hat = G_TIED + np.array([0.05, 0.05, 0.0])
        result = fit(law_g.frame, estimator(g_hat, Q_WRONG_ARM1, guard=("g",)))
        assert result.fluctuations["effect_modification"].mechanism is None
        reduction = result.fluctuations["effect_modification"].reduction
        assert reduction is not None and reduction.guard == ("g",)
        q_star = result.fluctuations["effect_modification"].targeted
        w = law_g.w
        per_cell = np.column_stack(
            [
                [q_star.arms[0.0][w == k][0] for k in range(3)],
                [q_star.arms[1.0][w == k][0] for k in range(3)],
            ]
        )
        expected = law_g.curve(per_cell, g_hat, d_a=False, d_y=True)
        np.testing.assert_allclose(curves(result), expected, rtol=0.0, atol=1e-9)
        both = law_g.curve(per_cell, g_hat, d_a=True, d_y=True)
        assert float(np.max(np.abs(expected - both))) > FLOOR


class TestTheCrossFittedLaw:
    """Six copies grouped by cluster into three folds: every fold's training rows are the law."""

    @pytest.mark.parametrize("crossfit", ["foldwise", "pooled"])
    def test_the_stacked_fit_is_exact(self, law_g: Law, crossfit: str) -> None:
        copies = pd.concat([law_g.frame.assign(cid=k) for k in range(6)], ignore_index=True)
        est = estimator(G_TIED, Q_WRONG_ARM1, cross_fit=True, n_folds=3, reduced_crossfit=crossfit)
        result = fit(
            copies,
            est,
            known=G_TIED,
            id="cid",
        )
        assert result.extra["drtmle"].reduced_crossfit == crossfit
        q_star = law_g.targeted(Q_WRONG_ARM1, G_TIED)
        single = law_g.curve(q_star, G_TIED, d_a=True, d_y=True)
        for j, term in enumerate(TERMS):
            estimate = result[f"cate_projection[{term}]"]
            assert estimate.psi == pytest.approx(law_g.beta[j], abs=EXACT)
            np.testing.assert_allclose(
                np.asarray(estimate.influence_curve), np.tile(single[:, j], 6), atol=EXACT
            )


# ------------------------------------------------------------------- law P, the D_A witness

#: Checked at this seed: initial |P_n D_A,j| exceeds 1e-3 for both terms (P1 PR 3 plan, 4.1;
#: about 0.83 at a random seed).
LAW_P_SEED = 20261121


def law_p(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    g0 = 1.0 / (1.0 + np.exp(-(0.6 * w1 + 0.3 * w2)))
    a = rng.binomial(1, g0)
    eta = -0.2 + a * (0.5 + 0.6 * w1) - w1 + 2.0 * w2
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)))
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a.astype(float), "Y": y.astype(float), "g0": g0})


def law_p_estimator(**options: Any) -> DRTMLE:
    settings: dict[str, Any] = {
        "outcome_learner": OffsetLogistic(columns=(0, 1)),
        "treatment_learner": CubicLogistic(),
        "reduced_outcome_learner": CubicLeastSquares(),
        "reduced_mechanism_residual_learner": CubicLeastSquares(),
        "reduced_treatment_learner": CubicLogistic(),
        "g_bounds": (0.01, 0.99),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
        "effect_modification": EffectModification(Linear(["W1"]), Modifiers(["W1"]), "difference"),
    }
    settings.update(options)
    return DRTMLE(**settings)


def law_p_fit(frame: pd.DataFrame, **options: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            law_p_estimator(**options)
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                treatment_probabilities=frame["g0"].to_numpy(),
            )
            .single()
        )


class TestTheFiniteSampleWitness:
    def test_equation_nine_moves_the_declared_mechanism(self) -> None:
        frame = law_p(2000, LAW_P_SEED)
        result = law_p_fit(frame)
        reduced = result.nuisance.reduced
        assert reduced is not None and reduced.n_terms == 2
        g0 = frame["g0"].to_numpy()
        a = frame["A"].to_numpy()
        initial = np.zeros(2)
        for column, (arm, sign) in enumerate(((0.0, -1.0), (1.0, 1.0))):
            g_a = g0 if arm == 1.0 else 1.0 - g0
            took = (a == arm).astype(float)
            initial += sign * np.mean(
                reduced.qr[:, column, :] / g_a[:, None] * (took - g_a)[:, None], axis=0
            )
        assert float(np.min(np.abs(initial))) > FLOOR
        mechanism = result.fluctuations["effect_modification"].mechanism
        assert mechanism is not None
        assert float(np.max(np.abs(mechanism.score))) < 1e-8
        assert float(np.max(np.abs(np.asarray(mechanism.propensity) - g0))) > 1e-4


class TestTheEngineAndStudyRoutes:
    def test_they_agree(self) -> None:
        frame = law_p(1000, LAW_P_SEED).assign(g0c=lambda f: 1.0 - f["g0"])
        engine = law_p_fit(frame)
        study = CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W1", "W2"),
                treatment_probabilities={1.0: "g0", 0.0: "g0c"},
            ),
        )
        contrast = study.identify(ConditionalContrast(Modifiers(["W1"]), model=Linear(["W1"])))
        record = next(r for r in contrast.available_methods() if r.name == "drtmle")
        assert record.available
        method = DRTMLEMethod(
            reduced_outcome_learner=CubicLeastSquares(),
            reduced_mechanism_residual_learner=CubicLeastSquares(),
            reduced_treatment_learner=CubicLogistic(),
        ).with_overrides(
            outcome_learner=OffsetLogistic(columns=(0, 1)),
            treatment_learner=CubicLogistic(),
            g_bounds=(0.01, 0.99),
            cross_fit=False,
            simultaneous=False,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            routed = contrast.estimate(method=method)
        for term in ("(intercept)", "W1"):
            name = f"cate_projection[{term}]"
            assert routed[name].psi == pytest.approx(engine[name].psi, abs=1e-12)
            np.testing.assert_allclose(
                routed[name].influence_curve, engine[name].influence_curve, atol=1e-12
            )


class TestTheRefusals:
    """Each before any learner: the sentinel learners fail if fitted."""

    def sentinel(self, **options: Any) -> DRTMLE:
        settings: dict[str, Any] = {
            "outcome_learner": NeverFit(),
            "treatment_learner": NeverFit(),
            "reduced_outcome_learner": NeverFit(),
            "reduced_mechanism_residual_learner": NeverFit(),
            "reduced_treatment_learner": NeverFit(),
            "cross_fit": False,
            "effect_modification": projection(),
        }
        settings.update(options)
        return DRTMLE(**settings)

    def refused(self, frame: pd.DataFrame, est: DRTMLE, refusal: Any, **extra: Any) -> None:
        with pytest.raises(CapabilityError) as caught:
            est.fit(frame, outcome="Y", treatment="A", **extra)
        assert caught.value.refusal.condition == refusal.condition

    def test_the_log_ratio(self) -> None:
        self.refused(
            law.frame(),
            self.sentinel(effect_modification=projection("log_ratio")),
            PROJECTION_RATIO_REFUSAL,
        )

    def test_the_bivariate_reduction(self) -> None:
        self.refused(
            law.frame(), self.sentinel(reduction="bivariate"), PROJECTION_BIVARIATE_REFUSAL
        )

    def test_a_missing_outcome(self) -> None:
        frame = law.frame().assign(D=1.0)
        frame.loc[::7, "D"] = 0.0
        frame.loc[::7, "Y"] = np.nan
        self.refused(frame, self.sentinel(), PROJECTION_MISSING_REFUSAL, delta="D")

    def test_strata(self) -> None:
        frame = law.frame().assign(S=lambda f: (f.index % 2).astype(float))
        self.refused(frame, self.sentinel(), PROJECTION_STRATA_REFUSAL, strata=["S"])

    @pytest.mark.parametrize("option", [{"targeting": "one_step"}, {"evaluation": "companion"}])
    def test_one_step_and_evaluation(self, option: dict[str, Any]) -> None:
        if "evaluation" in option:
            option = {"evaluation": law.frame()}
        self.refused(law.frame(), self.sentinel(**option), PROJECTION_ONE_STEP_REFUSAL)


class WrongMultiOutcome(BaseEstimator):
    """The three-arm law's regression, off by ``+0.05`` on the "high" arm alone."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WrongMultiOutcome:
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        indicators = design[:, : multi.K - 1]
        code = np.where(indicators.any(axis=1), indicators.argmax(axis=1) + 1, 0)
        arm = np.array(multi.ARM_OF_CODE)[code]
        w = np.rint(design[:, multi.K - 1]).astype(int)
        return multi.Q_EXACT[w, arm] + 0.05 * (arm == multi.LABELS.index("high"))


class TestTheMultiArmPair:
    """A pair of a three-arm treatment: each pair arm's margin is tilted along its own ``p``."""

    def test_the_saturated_fit_is_exact(self) -> None:
        from tests.unit.test_effect_modification_exact import (
            curves_at,
            multi_first_rows,
            multi_functional,
        )

        est = DRTMLE(
            outcome_learner=WrongMultiOutcome(),
            treatment_learner=multi.OracleMultiTreatment(),
            reduced_outcome_learner=CellMeans(),
            reduced_mechanism_residual_learner=CellMeans(),
            reduced_treatment_learner=CellMeans(),
            g_bounds=INERT,
            cross_fit=False,
            simultaneous=False,
            random_state=0,
            effect_modification=EffectModification(
                Linear(["W"]), Modifiers(["W"]), "difference", treatment="high", reference="low"
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = est.fit(multi.frame(), outcome="Y", treatment="A").single()
        truth = multi_functional("difference")(multi.PROBS)
        np.testing.assert_allclose(
            [estimate.psi for estimate in result.estimates.values()], truth, atol=EXACT
        )
        # Saturated reductions on a mechanism that separates the cells: the corrected curve is
        # the efficient one, and the plain curve at the targeted wrong regression is not.
        expected = law.contamination_eif(multi_functional("difference"), multi.PROBS, multi.SUPPORT)
        np.testing.assert_allclose(curves_at(result, multi_first_rows()), expected, atol=1e-9)
        mechanism = result.fluctuations["effect_modification"].mechanism
        assert mechanism is not None and np.asarray(mechanism.score).shape == (6,)


class TestTheStudyAvailability:
    def _contrast(self, scale: str, frame: pd.DataFrame | None = None, **design: Any) -> Any:
        study = CausalStudy(
            law.frame() if frame is None else frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), **design),
        )
        return study.identify(
            ConditionalContrast(Modifiers(["W"]), scale=scale, model=Linear(["W"]))
        )

    def test_the_log_ratio_reads_r_dr_ratio(self) -> None:
        records = {r.name: r for r in self._contrast("log_ratio").available_methods()}
        assert not records["drtmle"].available
        assert records["drtmle"].refusal == PROJECTION_RATIO_REFUSAL

    def test_the_difference_scale_is_available(self) -> None:
        records = {r.name: r for r in self._contrast("difference").available_methods()}
        assert records["drtmle"].available


class TestTheCorrectionCheck:
    """``diagnostics.corrections()``: one row per arm, basis term and equation."""

    @pytest.fixture(scope="class")
    def estimated(self) -> Any:
        frame = law_p(2000, LAW_P_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return (
                law_p_estimator()
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )

    def test_every_identity_holds_per_term(self, estimated: Any) -> None:
        check = estimated.diagnostics.corrections()
        assert len(check.rows) == 2 * 2 * 2
        assert {row.term for row in check.rows} == {"(intercept)", "W1"}
        for row in check.rows:
            assert abs(row.residual) < 1e-8
            assert abs(row.reported) < 1e-6
        assert check.passed

    def test_every_correction_is_alive_before_targeting(self, estimated: Any) -> None:
        from cleverly.inference.influence import reduced_correction_parts

        nuisance = estimated.nuisance
        parts = reduced_correction_parts(
            nuisance.scaler.scale(estimated.data.outcome),
            nuisance.outcome,
            estimated.data.treatment,
            nuisance.reduced,
            nuisance.propensity.arm(1.0),
            bounds=(0.01, 0.99),
            guard=("Q", "g"),
        )
        for arm in (0.0, 1.0):
            assert float(np.min(np.abs(np.mean(parts.d_g[arm], axis=0)))) > 1e-4
            assert float(np.min(np.abs(np.mean(parts.d_q[arm], axis=0)))) > 1e-4

    def test_an_unsolved_term_reports_its_own_mean(self) -> None:
        from cleverly.estimators.tmle import correction_parts

        frame = law_p(2000, LAW_P_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                law_p_estimator(guard=("Q",))
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        fluctuation = result.fluctuations["effect_modification"]
        parts = correction_parts(
            result.data,
            result.nuisance,
            fluctuation,
            fluctuation.targeted,
            result.nuisance.scaler.scale(result.data.outcome),
        )
        assert parts is not None
        rows = [row for row in result.diagnostics.corrections().rows if row.equation == "D*_Q"]
        assert not any(row.solved for row in rows)
        for row in rows:
            j = ("(intercept)", "W1").index(row.term)
            expected = float(np.mean(parts.d_q[row.arm][:, j]))
            assert abs(expected) > 1e-5
            assert row.reported == pytest.approx(expected, rel=1e-12)
            assert row.label.endswith(f"[{row.term}]")


class TestTheCurve:
    """An effect curve of a DR-TMLE projection: its corrections and truncation rows route to
    the producing fit, and the five held-out rows refuse (R-dr-tools)."""

    def test_the_rows(self, known_fit: Any) -> None:
        from cleverly.assessment import CURVE_OPERATIONS
        from cleverly.effect_modification import DRTMLE_CURVE_TOOLS_REFUSAL

        curve = known_fit.effect_curve()
        assert curve.fitted_method == "drtmle"
        rows = {row.operation: row for row in curve.diagnostics.capabilities}
        for name in CURVE_OPERATIONS:
            assert rows[name].refusal == DRTMLE_CURVE_TOOLS_REFUSAL
        with pytest.raises(CapabilityError) as caught:
            curve.diagnostics.calibration(n_folds=2)
        assert caught.value.refusal.condition.endswith(DRTMLE_CURVE_TOOLS_REFUSAL.condition)
        assert rows["corrections"].available and rows["truncation_curve"].available
        routed = curve.diagnostics.corrections()
        assert routed.rows == known_fit.diagnostics.corrections().rows
        assert len(routed.rows) == 8

    def test_the_tmle_curve_keeps_its_corrections_text(self, law_g: Law) -> None:
        dgp = Misspecified(G_TIED, Q_WRONG_ARM1)
        plain = fit(
            law_g.frame,
            TMLE(
                outcome_learner=OracleOutcome(dgp),
                treatment_learner=OracleTreatment(dgp),
                cross_fit=False,
                effect_modification=projection(),
            ),
            known=G_TIED,
        )
        rows = {row.operation: row for row in plain.effect_curve().diagnostics.capabilities}
        assert not rows["corrections"].available
        assert (
            "solve no doubly-robust correction equations" in rows["corrections"].refusal.condition
        )


class TestTheOpenedPaths:
    """The plan 1.6 rows the deleted refusal opens beyond the registered composition."""

    def test_fixed_weights_equal_duplicated_rows(self, law_g: Law) -> None:
        # Weights that read the covariate only, so the declared mechanism holds in the target.
        frame = law_g.frame.assign(wt=1.0 + 2.0 * (law_g.frame["W"] == 2))
        duplicated = frame.loc[frame.index.repeat(frame["wt"].astype(int))].reset_index(drop=True)
        weighted = fit(
            frame,
            estimator(G_TIED, Q_WRONG_ARM1),
            known=G_TIED,
            weights="wt",
            mechanism_law="target",
            covariates=["W"],
        )
        copies = fit(duplicated, estimator(G_TIED, Q_WRONG_ARM1), known=G_TIED, covariates=["W"])
        for j, term in enumerate(TERMS):
            name = f"cate_projection[{term}]"
            # To the alternation's stopping tolerance: the two fits stop at different rounds.
            assert weighted[name].psi == pytest.approx(copies[name].psi, abs=1e-7)
            # The weights move the target, so the equality is not the unweighted truth again.
            assert abs(weighted[name].psi - law_g.beta[j]) > FLOOR

    def test_clusters_sum_the_curve(self, law_g: Law, known_fit: Any) -> None:
        frame = law_g.frame.assign(cluster=np.arange(len(law_g.frame)) // 4)
        clustered = fit(
            frame, estimator(G_TIED, Q_WRONG_ARM1), known=G_TIED, id="cluster", covariates=["W"]
        )
        for name, estimate in clustered.estimates.items():
            curve = np.asarray(known_fit[name].influence_curve)
            sums = np.bincount(frame["cluster"].to_numpy(), weights=curve)
            n, clusters = curve.size, sums.size
            expected = np.sqrt(np.sum(sums**2) / n**2 * clusters / (clusters - 1))
            assert estimate.psi == pytest.approx(known_fit[name].psi, abs=1e-14)
            assert estimate.std_error == pytest.approx(expected, rel=1e-6)
            assert estimate.std_error != pytest.approx(known_fit[name].std_error, rel=1e-3)

    def test_repeats_fit_and_read_experimental(self) -> None:
        result = law_p_fit(law_p(600, LAW_P_SEED), cross_fit=True, n_folds=3, repeats=2)
        assert len(result.repeats) == 2
        for estimate in result.estimates.values():
            assert np.isfinite(estimate.psi)
            assert estimate.inference == "influence_curve"
            assert estimate.support.label == "experimental"

    def test_the_bootstrap_reads_the_status_note(self) -> None:
        from cleverly._inference_status import STATUS_RECORDS
        from cleverly.inference.influence import bootstrap_line

        frame = law_p(600, LAW_P_SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                law_p_estimator(n_bootstrap=4)
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        note = STATUS_RECORDS["reduced_regression_plugin"].bootstrap_note
        for estimate in result.estimates.values():
            assert estimate.inference == "reduced_regression_plugin"
            assert estimate.support.label == "experimental"
            assert estimate.bootstrap is not None
            assert f"range [{estimate.bootstrap.ci[0]:.5g}" in bootstrap_line(estimate)
            assert bootstrap_line(estimate).endswith(f"({note})")

    def test_randomized_reads_its_mechanism(self) -> None:
        """Declared probabilities read known; without them the interval is withheld."""
        frame = law_p(600, LAW_P_SEED)
        known = law_p_fit(frame, randomized=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimated = (
                law_p_estimator(randomized=True)
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        assert {e.inference for e in known.estimates.values()} == {"influence_curve"}
        assert {e.inference for e in estimated.estimates.values()} == {"reduced_regression_plugin"}
        for name, estimate in known.estimates.items():
            assert estimate.support.label == "experimental"
            assert estimate.psi != pytest.approx(estimated[name].psi, abs=1e-6)

    def test_a_continuous_outcome_with_q_bounds_fits(self) -> None:
        from sklearn.linear_model import LinearRegression

        frame = law_p(600, LAW_P_SEED)
        noise = np.random.default_rng(1).normal(scale=0.3, size=len(frame))
        frame = frame.assign(Y=frame["Y"] + noise)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = (
                law_p_estimator(q_bounds=(-1.5, 2.5), outcome_learner=LinearRegression())
                .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
                .single()
            )
        assert getattr(result.nuisance.reduced, "n_terms", None) == 2
        for estimate in result.estimates.values():
            assert np.isfinite(estimate.psi) and np.isfinite(estimate.plugin_std_error)
            assert estimate.inference == "reduced_regression_plugin"
            assert estimate.support.label == "experimental"


def _truncation_estimator(kind: str, **options: Any) -> Any:
    if kind == "drtmle":
        return law_p_estimator(**options)
    scale: Literal["difference", "log_ratio"] = (
        "log_ratio" if kind == "tmle_log_ratio" else "difference"
    )
    return TMLE(
        outcome_learner=OffsetLogistic(columns=(0, 1)),
        treatment_learner=CubicLogistic(),
        cross_fit=False,
        simultaneous=False,
        random_state=0,
        effect_modification=EffectModification(Linear(["W1"]), Modifiers(["W1"]), scale),
        **options,
    )


@pytest.mark.parametrize("kind", ["tmle_difference", "tmle_log_ratio", "drtmle"])
def test_each_truncation_row_equals_a_refit_at_its_bound(kind: str) -> None:
    """Plan 4.3: the replay reuses the cached nuisances, and a DR-TMLE replay refits the
    reductions per bound, so every row is the fit at that bound; the 0.1 bound binds."""
    frame = law_p(1000, LAW_P_SEED)

    def fitted(bound: float) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = _truncation_estimator(kind, g_bounds=(bound, 1.0 - bound))
            return est.fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"]).single()

    result = fitted(0.01)
    curve = result.diagnostics.truncation_curve(bounds=[0.05, 0.1]).to_frame()
    for bound in (0.05, 0.1):
        rows = curve.loc[curve["bound"] == bound]
        values = dict(zip(rows["estimand"], rows["psi"], strict=True))
        refit = fitted(bound)
        for name, estimate in refit.estimates.items():
            assert values[name] == pytest.approx(estimate.psi, abs=1e-10)
    bound = curve.loc[curve["bound"] == 0.1]
    moved = np.abs(bound["psi"].to_numpy() - bound["fitted_psi"].to_numpy())
    assert float(moved.max()) > 1e-5
