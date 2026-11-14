r"""``ConditionalContrast`` through the study layer: refusals, results, curves and scores.

The projection arithmetic is pinned in ``tests/unit/test_effect_modification_exact.py``.
This module pins the contract around it: every refusal of the P1 plan's part (a) raises
before any learner with its text (R2 to R7, R9, R11, R13), the curve result predicts,
bands and persists, the heterogeneity test excludes the intercept, and the assessment of a
fixed external score is exact at the oracle with the influence curve of the plan's 2.3.
Its rev 1 curve, ``+ 2 (c - E s)(Dtilde - c)``, is checked to fail.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import cleverly
from cleverly import ATE, CausalStudy, ConditionalContrast, PointTreatment, sensitivity
from cleverly.bases import Constant, Cosine, Indicators, Linear, Modifiers, Sieve
from cleverly.effect_curves import EffectCurveResult
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.validation import refute
from tests import discrete_law as law
from tests import discrete_law_mar as mar
from tests.conftest import OracleMissingness, OracleOutcome, OracleTreatment
from tests.unit._natural_course_support import NeverFit

MODIFIERS = Modifiers(["W"], domains={"W": (0.0, 2.0)})

#: A score of ``W`` that is not the CATE, so the risk difference is not ``-Var(tau)``.
SCORE = np.array([0.10, 0.40, 0.20])


def law_study(**design: Any) -> CausalStudy:
    frame = law.frame()
    frame["S"] = SCORE[frame["W"].astype(int)]
    frame["T"] = (law.Q_EXACT[:, 1] - law.Q_EXACT[:, 0])[frame["W"].astype(int)]
    frame["M"] = (frame["W"] > 0).astype(float)
    return CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), **design)
    )


def oracle() -> dict[str, Any]:
    dgp = law.DiscreteLaw()
    return {
        "outcome_learner": OracleOutcome(dgp),
        "treatment_learner": OracleTreatment(dgp),
        "cross_fit": False,
        "random_state": 0,
    }


def never() -> dict[str, Any]:
    NeverFit.calls = 0
    return {"outcome_learner": NeverFit(), "treatment_learner": NeverFit()}


def linear_contrast(scale: str = "difference") -> ConditionalContrast:
    return ConditionalContrast(MODIFIERS, scale=scale, model=Linear(["W"]))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def projection_fit() -> Any:
    return law_study().identify(linear_contrast()).estimate(**oracle())


class TestTheStudyFit:
    def test_the_study_reports_the_engine_projection(self, projection_fit: Any) -> None:
        expected = [law.TRUTH[f"cate_projection[{t}]"] for t in law.EFFECT_MODIFICATION_TERMS]
        np.testing.assert_allclose(
            [estimate.psi for estimate in projection_fit.estimates.values()], expected, atol=1e-12
        )
        keys = projection_fit.parameter_keys
        assert [key.term for key in keys.values()] == ["(intercept)", "W"]
        assert {key.axis for key in keys.values()} == {"effect_modification"}

    def test_the_estimand_names_its_scale(self) -> None:
        assert linear_contrast().name == "cate"
        assert linear_contrast("log_ratio").name == "log_crr"
        assert "projection" in linear_contrast().definition

    def test_tmle_and_ep_are_available(self) -> None:
        methods = {m.name: m for m in law_study().identify(linear_contrast()).available_methods()}
        assert methods["tmle"].available
        assert methods["ep"].available
        assert not any(m.available for name, m in methods.items() if name not in ("tmle", "ep"))

    def test_other_estimands_read_the_ep_record(self) -> None:
        methods = {m.name: m for m in law_study().identify(ATE()).available_methods()}
        refusal = methods["ep"].refusal
        assert refusal.reason is RefusalReason.DIFFERENT_ESTIMAND
        assert refusal.condition == "EP estimates a ConditionalContrast."
        assert refusal.alternative == "Fit method='tmle' for this estimand."
        assert "'ep'" not in refusal.alternative


class TestRefusalsBeforeAnyLearner:
    def test_r2_tmle_without_a_model(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS))
        with pytest.raises(CapabilityError, match="not pathwise differentiable") as caught:
            effect.estimate(**never())
        assert NeverFit.calls == 0
        # The EP learner fits the curve without a model, so the alternative names it.
        assert "ep" in caught.value.refusal.alternative.split("'")

    def test_r3_a_basis_that_reads_an_undeclared_domain(self) -> None:
        with pytest.raises(CapabilityError, match="known function of the modifiers"):
            law_study().identify(ConditionalContrast(Modifiers(["W"]), model=Cosine(["W"], 2)))

    def test_r3_indicators_without_declared_levels(self) -> None:
        with pytest.raises((CapabilityError, DataError)):
            law_study().identify(ConditionalContrast(Modifiers(["W"]), model=Indicators("W")))

    def test_r4_a_sieve_sized_by_cross_validation(self) -> None:
        effect = law_study().identify(ConditionalContrast(MODIFIERS, model=Sieve(Cosine(["W"], 3))))
        with pytest.raises(CapabilityError, match="chosen by cross-validation") as caught:
            effect.estimate(**never())
        assert NeverFit.calls == 0
        assert "ep" in caught.value.refusal.alternative.split("'")

    def test_r4_a_declared_size_resolves(self) -> None:
        effect = law_study().identify(
            ConditionalContrast(MODIFIERS, model=Sieve(Cosine(["W"], 3), size=1))
        )
        assert effect.functional.effect_modification.terms == ("(intercept)", "cos1(W)")

    def test_r5_a_log_ratio_on_a_shifted_scale(self) -> None:
        frame = law.frame()
        frame["Y"] = 5.0 * frame["Y"] + 1.0
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        effect = study.identify(linear_contrast("log_ratio"))
        with pytest.raises(CapabilityError, match="These q_bounds start at 1"):
            effect.estimate(q_bounds=(1.0, 6.0), cross_fit=False, **never())
        assert NeverFit.calls == 0

    def test_r6_a_modifier_outside_the_adjustment_set(self) -> None:
        with pytest.raises(CapabilityError, match="must be an adjustment column") as caught:
            law_study().identify(ConditionalContrast(Modifiers(["S"]), model=Linear(["S"])))
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    @pytest.mark.parametrize(
        ("design", "axis"),
        [({"strata": ("W",)}, "strata="), ({"intermediate": "M"}, "an intermediate")],
    )
    def test_r7_compositions_outside_p1(self, design: dict, axis: str) -> None:
        with pytest.raises(CapabilityError, match=f"does not yet compose with {axis}") as caught:
            law_study(**design).identify(linear_contrast())
        assert caught.value.refusal.roadmap == "P1"

    @pytest.mark.parametrize(
        ("setting", "axis"),
        [({"repeats": 2}, "repeats above 1"), ({"n_bootstrap": 20}, "the full-refit bootstrap")],
    )
    def test_r7_settings_outside_p1(self, setting: dict, axis: str) -> None:
        effect = law_study().identify(linear_contrast())
        with pytest.raises(CapabilityError, match=f"does not yet compose with {axis}"):
            effect.estimate(**never(), **setting)
        assert NeverFit.calls == 0

    def test_r9_sensitivity_and_refutation_refuse_on_the_projection(
        self, projection_fit: Any
    ) -> None:
        rows = projection_fit.sensitivity.capabilities
        assert rows and not any(row.available for row in rows)
        for row in rows:
            assert row.refusal.condition.startswith(f"{row.operation} is defined for a mean")
            assert row.refusal.alternative == "Run it on ATE() or RiskRatio() on the same study."
        refute = projection_fit.diagnostics.capability("refute")
        assert not refute.available and refute.refusal.roadmap == "P1"
        with pytest.raises(CapabilityError, match="working-model coefficient"):
            projection_fit.sensitivity.evalue()

    def test_r9_the_evalue_row_a_request_resolves_is_the_declared_row(
        self, projection_fit: Any
    ) -> None:
        """The E-value rebuilt its row per request and read the generic arm-contrast refusal."""
        facade = projection_fit.sensitivity
        resolved = facade._capability_for_arguments("evalue", {})
        assert resolved.refusal == facade.capability("evalue").refusal
        assert resolved.refusal.roadmap == "P1"


#: Every free sensitivity or refutation function, called the way a user would on a fit.
FREE_FUNCTIONS = {
    "omitted_variable_bounds": lambda fit: sensitivity.omitted_variable_bounds(fit),
    "robustness_value": lambda fit: sensitivity.robustness_value(fit),
    "sensitivity_elements": lambda fit: sensitivity.sensitivity_elements(fit),
    "contour_data": lambda fit: sensitivity.contour_data(fit),
    "benchmark": lambda fit: sensitivity.benchmark(fit, ["W"]),
    "simulated_confounding": lambda fit: sensitivity.simulated_confounding(
        fit,
        grid=sensitivity.ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0, 0.1)),
        random_state=0,
    ),
    "refute": lambda fit: refute(fit, random_state=0),
}


@pytest.mark.parametrize("name", sorted(FREE_FUNCTIONS))
def test_r9_each_free_function_refuses_the_projection(projection_fit: Any, name: str) -> None:
    """Each one read an axis-keyed map that omitted the axis and raised ``KeyError``."""
    with pytest.raises(CapabilityError, match="working-model coefficient") as caught:
        FREE_FUNCTIONS[name](projection_fit)
    assert caught.value.refusal.roadmap == "P1"


class TestTheHeterogeneityTest:
    def test_it_tests_the_slope_alone(self, projection_fit: Any) -> None:
        test = projection_fit.heterogeneity_test()
        slope = projection_fit.estimates["cate_projection[W]"]
        assert test.df == 1 and test.terms == ("W",)
        assert test.statistic == pytest.approx((slope.psi / slope.std_error) ** 2, rel=1e-10)

    def test_including_the_intercept_changes_the_statistic(self, projection_fit: Any) -> None:
        """Mutation control: the joint statistic over both terms is a different number."""
        names = list(projection_fit.estimates)
        beta = np.array([projection_fit.estimates[name].psi for name in names])
        joint = float(beta @ np.linalg.solve(projection_fit.covariance(names), beta))
        assert joint - projection_fit.heterogeneity_test().statistic > 1.0

    def test_b1_a_basis_without_an_intercept_refuses(self) -> None:
        """On a constant effect the intercept-free slope is far from 0, so a test would reject.

        The law's effect is 0.2 in every cell. Without the intercept the coefficient of ``W``
        is ``0.2 E[W] / E[W^2]``, not 0, so a Wald test of it rejects a constant effect. With
        the intercept the slope is exactly 0.
        """
        q = np.array([[0.40, 0.60], [0.20, 0.40], [0.60, 0.80]])
        counts = law.cell_counts(q=q)
        dgp = law.DiscreteLaw(counts / law.N)
        study = CausalStudy(
            law.frame(counts), design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        settings = {
            **oracle(),
            "outcome_learner": OracleOutcome(dgp),
            "treatment_learner": OracleTreatment(dgp),
        }
        free = study.identify(
            ConditionalContrast(MODIFIERS, model=Linear(["W"], intercept=False))
        ).estimate(**settings)
        with pytest.raises(CapabilityError, match="no intercept column") as caught:
            free.heterogeneity_test()
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        slope = free.estimates["cate_projection[W]"]
        assert (slope.psi / slope.std_error) ** 2 > 10.0
        held = study.identify(ConditionalContrast(MODIFIERS, model=Linear(["W"]))).estimate(
            **settings
        )
        assert held.heterogeneity_test().statistic == pytest.approx(0.0, abs=1e-12)

    def test_a_constant_basis_has_nothing_to_test(self) -> None:
        fit = (
            law_study()
            .identify(ConditionalContrast(MODIFIERS, model=Constant()))
            .estimate(**oracle())
        )
        with pytest.raises(CapabilityError, match="no non-constant term"):
            fit.heterogeneity_test()

    def test_an_arm_fit_has_no_test(self) -> None:
        fit = law_study().identify(ATE()).estimate(**oracle())
        with pytest.raises(CapabilityError, match="projects no conditional contrast"):
            fit.heterogeneity_test()


# ----------------------------------------------------------------- L-disc, two modifiers

#: L-disc of plan 4.1: ``P(V, W2)``, ``g(V, W2)`` and ``Qbar(a, V, W2)``, each a multiple of
#: ``1 / 20``, ``1 / 10`` and ``1 / 10``, so ``LDISC_N`` rows realise the law exactly. The blip
#: varies with both modifiers.
LDISC_N = 10_000
LDISC_P = np.array([[0.20, 0.30], [0.15, 0.15], [0.10, 0.10]])
LDISC_G = np.array([[0.4, 0.6], [0.5, 0.3], [0.6, 0.5]])
LDISC_Q = np.array([[[0.3, 0.5], [0.4, 0.5]], [[0.2, 0.6], [0.5, 0.8]], [[0.4, 0.9], [0.3, 0.6]]])
LDISC_SUPPORT = tuple(
    (v, w2, a, y) for v in range(3) for w2 in range(2) for a in range(2) for y in range(2)
)


def ldisc_counts() -> np.ndarray:
    counts = np.zeros((3, 2, 2, 2))
    for v, w2, a, y in LDISC_SUPPORT:
        arm = LDISC_G[v, w2] if a else 1.0 - LDISC_G[v, w2]
        outcome = LDISC_Q[v, w2, a] if y else 1.0 - LDISC_Q[v, w2, a]
        counts[v, w2, a, y] = LDISC_P[v, w2] * arm * outcome * LDISC_N
    rounded = np.rint(counts)
    assert np.max(np.abs(counts - rounded)) < 1e-6
    return rounded.astype(int)


LDISC_COUNTS = ldisc_counts()
LDISC_PROBS = LDISC_COUNTS / LDISC_N


class LDiscLaw:
    """The oracle nuisances of L-disc, read off the realised cells."""

    def __init__(self) -> None:
        p = LDISC_PROBS
        self.g = p.sum(axis=3)[:, :, 1] / p.sum(axis=(2, 3))
        self.q = p[..., 1] / p.sum(axis=3)

    def propensity(self, covariates: Any) -> np.ndarray:
        cells = np.rint(np.asarray(covariates, dtype=float)).astype(int)
        return self.g[cells[:, 0], cells[:, 1]]

    def outcome_mean(self, covariates: Any, arm: float, intermediate: float | None) -> np.ndarray:
        cells = np.rint(np.asarray(covariates, dtype=float)).astype(int)
        return self.q[cells[:, 0], cells[:, 1], int(arm)]


def ldisc_frame() -> pd.DataFrame:
    cells = np.repeat(np.arange(len(LDISC_SUPPORT)), [LDISC_COUNTS[cell] for cell in LDISC_SUPPORT])
    columns = np.array(LDISC_SUPPORT, dtype=float)[cells]
    return pd.DataFrame(
        {"V": columns[:, 0], "W2": columns[:, 1], "A": columns[:, 2], "Y": columns[:, 3]}
    )


def ldisc_first_rows() -> np.ndarray:
    per_cell = np.array([LDISC_COUNTS[cell] for cell in LDISC_SUPPORT])
    return np.concatenate([[0], np.cumsum(per_cell)[:-1]])


def ldisc_beta(probs: Any) -> Any:
    """``beta`` of the CATE projection onto ``(1, V, W2)``, from the cell probabilities."""
    p_vw = probs.sum(axis=(2, 3))
    q = probs[..., 1] / probs.sum(axis=3)
    tau = q[..., 1] - q[..., 0]
    phi = np.array([[[1.0, v, w2] for w2 in range(2)] for v in range(3)])
    gram = np.einsum("vwp,vwq,vw->pq", phi, phi, p_vw)
    return np.linalg.solve(gram, np.einsum("vwp,vw->p", phi, p_vw * tau))


def ldisc_fit(**design: Any) -> Any:
    frame = ldisc_frame()
    for name, values in design.pop("columns", {}).items():
        frame[name] = values
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("V", "W2"), **design),
    )
    dgp = LDiscLaw()
    return study.identify(
        ConditionalContrast(Modifiers(["V", "W2"]), model=Linear(["V", "W2"]))
    ).estimate(
        outcome_learner=OracleOutcome(dgp),
        treatment_learner=OracleTreatment(dgp),
        cross_fit=False,
        random_state=0,
    )


@pytest.fixture(scope="module")
def ldisc() -> Any:
    return ldisc_fit()


class TestTheHeterogeneityTestOnTwoModifiers:
    """Plan 4.2: L-disc with ``Linear(V, W2)``, df 2 and the exact noncentrality."""

    def test_the_statistic_is_the_exact_noncentrality(self, ldisc: Any) -> None:
        beta = ldisc_beta(LDISC_PROBS)
        eif = law.contamination_eif(ldisc_beta, LDISC_PROBS, LDISC_SUPPORT)
        mass = np.array([LDISC_PROBS[cell] for cell in LDISC_SUPPORT])
        sigma = np.einsum("c,cp,cq->pq", mass, eif, eif)
        slopes = sigma[1:, 1:]
        noncentrality = LDISC_N * float(beta[1:] @ np.linalg.solve(slopes, beta[1:]))
        test = ldisc.heterogeneity_test()
        assert test.df == 2 and test.terms == ("V", "W2")
        # The reported covariance is the sample covariance with ``ddof=1``.
        expected = noncentrality * (LDISC_N - 1) / LDISC_N
        assert test.statistic == pytest.approx(expected, rel=1e-9)
        # The two slopes covary, so a diagonal covariance would give another number.
        diagonal = LDISC_N * float(np.sum(beta[1:] ** 2 / np.diag(slopes)))
        assert abs(diagonal - noncentrality) > 1.0
        assert abs(sigma[1, 2]) > 0.01

    def test_including_the_intercept_is_df_3_and_another_statistic(self, ldisc: Any) -> None:
        names = list(ldisc.estimates)
        beta = np.array([ldisc.estimates[name].psi for name in names])
        joint = float(beta @ np.linalg.solve(ldisc.covariance(names), beta))
        assert len(names) == 3
        assert joint - ldisc.heterogeneity_test().statistic > 10.0


#: Pairs of consecutive rows, which share a cell almost always, so the influence curves of a
#: pair agree and the cluster-level variance is close to twice the row-level one.
PAIRS = np.arange(LDISC_N) // 2


@pytest.fixture(scope="module")
def ldisc_clustered() -> Any:
    return ldisc_fit(cluster="C", columns={"C": PAIRS})


class TestTheClusteredProjection:
    def clustered_covariance(self, fit: Any) -> np.ndarray:
        curves = np.column_stack([np.asarray(e.influence_curve) for e in fit.estimates.values()])
        sums = np.zeros((LDISC_N // 2, curves.shape[1]))
        np.add.at(sums, PAIRS, curves)
        return (LDISC_N // 2) * np.cov(sums, rowvar=False, ddof=1) / LDISC_N**2

    def test_the_heterogeneity_test_sums_the_curves_within_clusters(
        self, ldisc_clustered: Any, ldisc: Any
    ) -> None:
        covariance = self.clustered_covariance(ldisc_clustered)
        beta = np.array([e.psi for e in ldisc_clustered.estimates.values()])
        expected = float(beta[1:] @ np.linalg.solve(covariance[1:, 1:], beta[1:]))
        test = ldisc_clustered.heterogeneity_test()
        assert test.statistic == pytest.approx(expected, rel=1e-10)
        # The row-level test reads about twice the statistic.
        assert ldisc.heterogeneity_test().statistic / test.statistic > 1.5

    @pytest.mark.parametrize(
        ("clusters", "status", "df"),
        [(5, "few_cluster_plugin", 3), (20, "influence_curve", 18)],
    )
    def test_a_few_cluster_heterogeneity_test_reads_the_hotelling_reference(
        self, clusters: int, status: str, df: int
    ) -> None:
        """Below 40 clusters the coefficients read t(nu), nu = J - 2; the joint test reads the
        Hotelling law, T (nu - q + 1) / (nu q) against F(q, nu - q + 1)."""
        labels = np.random.default_rng(clusters).permutation(LDISC_N) % clusters
        fit = ldisc_fit(cluster="C", columns={"C": labels})
        assert {e.inference for e in fit.estimates.values()} == {status}
        assert {e.reference_df for e in fit.estimates.values()} == {df}
        test = fit.heterogeneity_test()
        assert test.reference_df == df and test.df == 2
        assert test.p_value == pytest.approx(
            float(stats.f.sf(test.statistic * (df - 1) / (2 * df), 2, df - 1)), rel=1e-12
        )
        # The chi-square and the F(q, nu) readings it replaces are both anticonservative.
        assert test.p_value > float(stats.f.sf(test.statistic / 2, 2, df))
        assert test.p_value > float(stats.chi2.sf(test.statistic, 2))
        assert f"Hotelling F reference on 2 and {df - 1} df" in str(test)

    def test_a_many_cluster_heterogeneity_test_keeps_the_chi_square(
        self, ldisc_clustered: Any
    ) -> None:
        test = ldisc_clustered.heterogeneity_test()
        assert test.reference_df is None
        assert test.p_value == pytest.approx(float(stats.chi2.sf(test.statistic, 2)), rel=1e-12)

    @pytest.mark.parametrize("nu", [0, 1])
    def test_a_basis_with_more_terms_than_the_reference_df_is_refused(self, nu: int) -> None:
        """Two tested terms need nu = J - 2 >= 2: at nu = 1 (three clusters) the Hotelling
        law has no denominator degrees of freedom, and at nu = 0 no covariance either."""
        from cleverly.effect_curves import heterogeneity_test

        labels = np.random.default_rng(5).permutation(LDISC_N) % 5
        fit = ldisc_fit(cluster="C", columns={"C": labels})
        names = tuple(fit.estimates)
        estimates = {
            name: replace(estimate, reference_df=nu) for name, estimate in fit.estimates.items()
        }
        with pytest.raises(CapabilityError, match="no reference law") as caught:
            heterogeneity_test(
                estimates,
                names,
                ("(intercept)", "V", "W2"),
                (0,),
                scale="difference",
                cluster=fit.data.cluster,
            )
        assert caught.value.refusal.reason is RefusalReason.INFERENCE_GAP

    def test_the_hotelling_reference_at_one_term_is_the_squared_t(self) -> None:
        """At q = 1 the reference is F(1, nu), the square of the coefficients' t(nu)."""
        test = _null_cluster_test(np.random.default_rng(3), clusters=5, terms=1)
        t = np.sqrt(test.statistic)
        assert test.reference_df == 3
        assert test.p_value == pytest.approx(2 * float(stats.t.sf(t, 3)), rel=1e-12)

    def test_the_few_cluster_test_holds_its_size_at_five_clusters_and_three_terms(self) -> None:
        """Normal cluster totals under the null, J = 5 and q = 3 (nu = 3).  The Hotelling
        reference on nu = J - 2 holds the 5% size; the F(q, J - 2) reading it replaces rejects
        about three times too often (0.18 under the exact law at J - 1)."""
        rng = np.random.default_rng(20261112)
        draws = 4_000
        tests = [_null_cluster_test(rng, clusters=5, terms=3) for _ in range(draws)]
        size = np.mean([test.p_value < 0.05 for test in tests])
        old = np.mean([stats.f.sf(test.statistic / 3, 3, 3) < 0.05 for test in tests])
        # 99% binomial margin at 0.05 over 4,000 draws is about 0.009.
        assert size <= 0.05 + 0.009
        assert old > 0.10

    def test_the_band_draws_one_multiplier_per_cluster(self, ldisc_clustered: Any) -> None:
        from cleverly.inference.multiplier import multiplier_critical_value

        grid = {"V": np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]), "W2": np.repeat([0.0, 1.0], 3)}
        curve = ldisc_clustered.effect_curve(grid=grid, random_state=0)
        phi = np.column_stack([np.ones(6), grid["V"], grid["W2"]])
        covariance = self.clustered_covariance(ldisc_clustered)
        se = np.sqrt(np.einsum("ip,pq,iq->i", phi, covariance, phi))
        np.testing.assert_allclose(
            (curve.band.upper - curve.band.lower) / 2, curve.band.critical_value * se, rtol=1e-10
        )
        curves = (
            np.column_stack(
                [np.asarray(e.influence_curve) for e in ldisc_clustered.estimates.values()]
            )
            @ phi.T
        )
        settings = {"alpha": 0.05, "n_replicates": 1000, "kind": "rademacher", "random_state": 0}
        by_cluster = multiplier_critical_value(curves, se, n=LDISC_N, cluster=PAIRS, **settings)
        by_row = multiplier_critical_value(curves, se, n=LDISC_N, cluster=None, **settings)
        assert curve.band.critical_value == pytest.approx(by_cluster, rel=1e-12)
        assert by_row != pytest.approx(by_cluster, rel=1e-3)


#: The three ``W`` cells, as a grid of modifier values.
GRID = {"W": np.array([0.0, 1.0, 2.0])}


class TestTheCurve:
    def test_predictions_are_the_basis_times_the_coefficients(self, projection_fit: Any) -> None:
        curve = projection_fit.effect_curve()
        beta = np.array([estimate.psi for estimate in projection_fit.estimates.values()])
        np.testing.assert_allclose(curve.predict(GRID), law.EFFECT_MODIFICATION_DESIGN @ beta)
        frame = curve.predict(GRID, interval=True)
        covariance = projection_fit.covariance(list(projection_fit.estimates))
        phi = law.EFFECT_MODIFICATION_DESIGN
        np.testing.assert_allclose(
            frame["std_err"], np.sqrt(np.einsum("ip,pq,iq->i", phi, covariance, phi))
        )
        assert curve.to_frame().columns.tolist() == ["effect_curve:cate"]

    def test_the_pointwise_interval_reads_the_coefficients_reference(
        self, projection_fit: Any
    ) -> None:
        """A t reference on the coefficients widens the curve's interval to the t quantile.

        The fixture fit reads the normal, so the normal case alone cannot tell a curve that
        reads the coefficients' reference from one that always reads the normal.  The t
        model is the nonzero witness.
        """
        curve = projection_fit.effect_curve()
        assert curve.model.reference_df is None
        for df, quantile in ((None, stats.norm.ppf(0.975)), (4, stats.t.ppf(0.975, 4))):
            moved = replace(curve, model=replace(curve.model, reference_df=df))
            frame = moved.predict(GRID, interval=True)
            np.testing.assert_allclose(
                (frame["ci_upper"] - frame["ci_lower"]) / 2, quantile * frame["std_err"]
            )

    def test_the_band_is_wider_than_the_pointwise_interval(self, projection_fit: Any) -> None:
        curve = projection_fit.effect_curve(GRID, random_state=0)
        assert curve.band is not None and curve.band.critical_value > 1.96
        np.testing.assert_allclose(curve.band.estimate, curve.predict(GRID))
        assert "declared grid points" in str(curve.summary())

    def test_predict_checks_the_modifier_schema(self, projection_fit: Any) -> None:
        curve = projection_fit.effect_curve()
        with pytest.raises(DataError):
            curve.predict({"X": np.array([1.0])})
        with pytest.raises(DataError, match="domain"):
            curve.predict({"W": np.array([3.0])})

    def test_outside_clip_clips_to_the_domain(self) -> None:
        clipped = Modifiers(["W"], domains={"W": (0.0, 2.0)}, outside="clip")
        fit = (
            law_study()
            .identify(ConditionalContrast(clipped, model=Linear(["W"])))
            .estimate(**oracle())
        )
        curve = fit.effect_curve()
        np.testing.assert_allclose(
            curve.predict({"W": np.array([5.0])}), curve.predict({"W": np.array([2.0])})
        )

    def test_save_and_load_give_the_same_predictions(self, projection_fit: Any, tmp_path) -> None:
        curve = projection_fit.effect_curve()
        path = curve.save(tmp_path / "curve.joblib")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = cleverly.load(path)
        assert isinstance(loaded, EffectCurveResult)
        np.testing.assert_array_equal(loaded.predict(GRID), curve.predict(GRID))

    def test_the_outcome_scale_factor_reaches_the_prediction(self) -> None:
        """``q_bounds=(-2, 5)``: the curve is 7 times the binary curve."""

        class Scaled(law.DiscreteLaw):
            def __init__(self) -> None:
                super().__init__()
                self.q = 7.0 * self.q - 2.0

        frame = law.frame()
        frame["Y"] = 7.0 * frame["Y"] - 2.0
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        settings = {**oracle(), "outcome_learner": OracleOutcome(Scaled()), "q_bounds": (-2.0, 5.0)}
        curve = study.identify(linear_contrast()).estimate(**settings).effect_curve()
        binary = law_study().identify(linear_contrast()).estimate(**oracle()).effect_curve()
        np.testing.assert_allclose(curve.predict(GRID), 7.0 * binary.predict(GRID), atol=1e-10)
        assert curve.upper_bound - curve.lower_bound == pytest.approx(7.0)


def score_functional(score: np.ndarray, scale: str, *, missing: bool = False) -> Any:
    """``(risk difference, calibration intercept, calibration slope)`` of a fixed score.

    ``missing=True`` reads the ``(3, 2, 3)`` cells of :mod:`tests.discrete_law_mar`, whose
    outcome regression conditions on the outcome being observed.
    """
    phi = np.column_stack([np.ones(3), score])

    def functional(probs: Any) -> Any:
        p_w = probs.sum(axis=(1, 2))
        if missing:
            observed = probs[:, :, mar.OBSERVED_ZERO] + probs[:, :, mar.OBSERVED_ONE]
            q = probs[:, :, mar.OBSERVED_ONE] / observed
        else:
            q = probs[:, :, 1] / probs.sum(axis=2)
        if scale == "difference":
            tau = q[:, 1] - q[:, 0]
            ate = (p_w * tau).sum()
            risk = (p_w * (score**2 - 2.0 * score * tau)).sum() + ate**2
            gram = np.einsum("wp,wq,w->pq", phi, phi, p_w)
            beta = np.linalg.solve(gram, np.einsum("wp,w->p", phi, p_w * tau))
            return np.array([risk, beta[0], beta[1]])
        total = q[:, 1] + q[:, 0]
        c = np.log((p_w * q[:, 1]).sum() / (p_w * q[:, 0]).sum())

        def loss(theta: Any) -> Any:
            return (p_w * (total * np.log(1.0 + np.exp(theta)) - q[:, 1] * theta)).sum()

        risk = loss(score) - loss(c + 0.0 * score)
        beta = np.zeros(2, dtype=probs.dtype)
        for _ in range(law.MSM_NEWTON_STEPS):
            e = 1.0 / (1.0 + np.exp(-(phi @ beta)))
            equation = np.einsum("wp,w->p", phi, p_w * (total * e - q[:, 1]))
            jacobian = np.einsum("wp,wq,w->pq", phi, phi, p_w * total * e * (1.0 - e))
            beta = beta - np.linalg.solve(jacobian, equation)
        return np.array([risk, beta[0], beta[1]])

    return functional


SCORE_NAMES = ("risk_difference", "calibration[(intercept)]", "calibration[S]")


def assess(scale: str, column: str = "S") -> Any:
    effect = law_study().identify(linear_contrast(scale))
    return effect.assess_score(
        column, computed_from=["W"], score_kind="external", scale=scale, **oracle()
    )


def score_curves(assessment: Any, names: tuple[str, ...] = SCORE_NAMES) -> np.ndarray:
    rows = law.first_row_of()
    return np.column_stack([np.asarray(assessment[name].influence_curve)[rows] for name in names])


class TestTheScoreAssessment:
    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_the_point_is_the_functional_and_the_curve_its_derivative(self, scale: str) -> None:
        assessment = assess(scale)
        functional = score_functional(SCORE, scale)
        np.testing.assert_allclose(
            [assessment[name].psi for name in SCORE_NAMES], functional(law.PROBS), atol=1e-12
        )
        np.testing.assert_allclose(
            score_curves(assessment), law.gateaux_eif(functional), atol=1e-11, rtol=0
        )

    @pytest.mark.parametrize("scale", ["difference", "log_ratio"])
    def test_missing_outcomes_enter_through_the_observation_mechanism(self, scale: str) -> None:
        """The MAR law: the curve is the derivative only with ``1 / Pi`` in the residual term.

        On a law with every outcome observed ``Pi = 1``, so the factor vanishes there. Here it
        varies with ``W`` and ``A``, and dropping it moves the curve at the observed cells.
        """
        dgp = mar.DiscreteLaw()
        frame = mar.frame()
        frame["S"] = SCORE[frame["W"].astype(int)]
        study = CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y", treatment="A", adjustment=("W",), missingness="Delta"
            ),
        )
        assessment = study.identify(linear_contrast(scale)).assess_score(
            "S",
            computed_from=["W"],
            score_kind="external",
            scale=scale,
            outcome_learner=OracleOutcome(dgp),
            treatment_learner=OracleTreatment(dgp),
            missingness_learner=OracleMissingness(dgp),
            cross_fit=False,
            random_state=0,
        )
        functional = score_functional(SCORE, scale, missing=True)
        np.testing.assert_allclose(
            [assessment[name].psi for name in SCORE_NAMES], functional(mar.PROBS), atol=1e-12
        )
        rows = mar.first_row_of()
        reported = np.column_stack(
            [np.asarray(assessment[name].influence_curve)[rows] for name in SCORE_NAMES]
        )
        expected = law.contamination_eif(functional, mar.PROBS, mar.SUPPORT)
        np.testing.assert_allclose(reported, expected, atol=1e-11, rtol=0)

    def test_the_rev_1_risk_curve_misses_the_derivative(self) -> None:
        """Mutation: ``+ 2 (c - E s)(Dtilde - c)`` in place of ``+ 2 ATE (Dtilde - ATE)``."""
        assessment = assess("difference")
        q = law.Q_EXACT
        tau = q[:, 1] - q[:, 0]
        ate = float((law.P_W * tau).sum())
        mean_score = float((law.P_W * SCORE).sum())
        w, a, y = (np.array(column) for column in zip(*law.SUPPORT, strict=True))
        g = np.where(a == 1, law.G_EXACT[w], 1.0 - law.G_EXACT[w])
        dtilde = (2 * a - 1) / g * (y - q[w, a]) + tau[w]
        reported = score_curves(assessment)[:, 0]
        rev1 = reported - 2.0 * ate * (dtilde - ate) + 2.0 * (ate - mean_score) * (dtilde - ate)
        expected = law.gateaux_eif(score_functional(SCORE, "difference"))[:, 0]
        np.testing.assert_allclose(reported, expected, atol=1e-11, rtol=0)
        assert np.max(np.abs(rev1 - expected)) > 0.1

    def test_dropping_the_constant_block_misses_the_log_ratio_derivative(self) -> None:
        """Mutation: the risk of the score alone, without ``- [L(c) + Dlt(c) - R(c)]``."""
        functional = score_functional(SCORE, "log_ratio")

        def score_risk_only(probs: Any) -> Any:
            p_w = probs.sum(axis=(1, 2))
            q = probs[:, :, 1] / probs.sum(axis=2)
            total = q[:, 1] + q[:, 0]
            return (p_w * (total * np.log(1.0 + np.exp(SCORE)) - q[:, 1] * SCORE)).sum()

        full = law.gateaux_eif(functional)[:, 0]
        alone = law.gateaux_eif(score_risk_only)
        assert np.max(np.abs(full - alone)) > 0.05
        np.testing.assert_allclose(score_curves(assess("log_ratio"))[:, 0], full, atol=1e-11)

    def test_the_true_cate_is_calibrated_and_its_risk_is_minus_its_variance(self) -> None:
        assessment = assess("difference", column="T")
        tau = law.Q_EXACT[:, 1] - law.Q_EXACT[:, 0]
        variance = float((law.P_W * tau**2).sum() - (law.P_W * tau).sum() ** 2)
        assert assessment["calibration[(intercept)]"].psi == pytest.approx(0.0, abs=1e-10)
        assert assessment["calibration[T]"].psi == pytest.approx(1.0, abs=1e-10)
        assert assessment["risk_difference"].psi == pytest.approx(-variance, abs=1e-12)


class TestScoreRefusalsBeforeAnyLearner:
    def effect(self) -> Any:
        return law_study().identify(linear_contrast())

    def test_r13_iii_a_score_not_declared_external(self) -> None:
        with pytest.raises(CapabilityError, match="score_kind='external'"):
            self.effect().assess_score("S", computed_from=["W"], **never())
        assert NeverFit.calls == 0

    @pytest.mark.parametrize("inputs", [None, ["Y"], "W"])
    def test_r13_ii_inputs_outside_the_adjustment_set(self, inputs: Any) -> None:
        with pytest.raises(CapabilityError, match="known function of adjustment columns"):
            self.effect().assess_score("S", computed_from=inputs, score_kind="external", **never())
        assert NeverFit.calls == 0

    def test_r13_i_a_curve_fitted_on_these_rows(self) -> None:
        effect = self.effect()
        curve = effect.estimate(**oracle()).effect_curve()
        frame = law.frame()
        frame["fitted"] = curve.in_sample
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        refit = study.identify(linear_contrast())
        refit._register_curve(curve.in_sample)
        with pytest.raises(CapabilityError, match="fitted on the rows it would be assessed on"):
            refit.assess_score("fitted", computed_from=["W"], score_kind="external", **never())
        frame = frame.rename(columns={"fitted": "effect_curve:cate"})
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        with pytest.raises(CapabilityError, match="fitted on the rows it would be assessed on"):
            study.identify(linear_contrast()).assess_score(
                "effect_curve:cate", computed_from=["W"], score_kind="external", **never()
            )
        assert NeverFit.calls == 0

    def test_r13_i_a_second_identify_on_the_study_sees_the_curve(self) -> None:
        """The curves are kept under the study's data fingerprint, not one effect object."""
        study = law_study()
        curve = study.identify(linear_contrast()).estimate(**oracle()).effect_curve()
        frame = law.frame()
        frame["S"] = curve.in_sample
        other = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        assert any(
            np.array_equal(stored, curve.in_sample)
            for stored in other.identify(linear_contrast())._curves
        )
        with pytest.raises(CapabilityError, match="fitted on the rows it would be assessed on"):
            other.identify(linear_contrast()).assess_score(
                "S", computed_from=["W"], score_kind="external", **never()
            )
        assert NeverFit.calls == 0

    def test_r5_a_log_ratio_score_on_a_shifted_outcome(self) -> None:
        """A difference-scale estimand: ``scale='log_ratio'`` reads R5 before any learner."""
        frame = law.frame()
        frame["Y"] = 7.0 * frame["Y"] - 2.0
        frame["S"] = SCORE[frame["W"].astype(int)]
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        with pytest.raises(CapabilityError, match="q_bounds"):
            study.identify(linear_contrast()).assess_score(
                "S",
                computed_from=["W"],
                score_kind="external",
                scale="log_ratio",
                q_bounds=(-2.0, 5.0),
                **never(),
            )
        assert NeverFit.calls == 0

    def compare_effect(self) -> Any:
        """The three-cell study with a second score and a copy of the first."""
        frame = law.frame()
        frame["S"] = SCORE[frame["W"].astype(int)]
        frame["SB"] = np.array([1.0, 0.0, 1.0])[frame["W"].astype(int)]
        frame["S_copy"] = frame["S"]
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        return study.identify(linear_contrast())

    def compare(self, compare: Any = "SB", **settings: Any) -> Any:
        return self.compare_effect().assess_score(
            "S",
            computed_from=["W"],
            score_kind="external",
            compare=compare,
            compare_computed_from=["W"],
            **settings,
            **never(),
        )

    def test_x97_compare_names_the_score_column(self) -> None:
        with pytest.raises(DataError, match=r"^compare= names the same column as the score$"):
            self.compare("S")
        assert NeverFit.calls == 0

    def test_x97_compare_holds_the_same_values(self) -> None:
        text = (
            "the score column 'S' and the compare= column 'S_copy' hold the same value on "
            "every row, so their risk difference is 0 with no spread"
        )
        with pytest.raises(DataError) as caught:
            self.compare("S_copy")
        assert str(caught.value) == text
        assert NeverFit.calls == 0

    def test_x97_the_compare_column_passes_the_r13_checks(self) -> None:
        with pytest.raises(CapabilityError, match="known function of adjustment columns"):
            self.compare_effect().assess_score(
                "S", computed_from=["W"], score_kind="external", compare="SB", **never()
            )
        assert NeverFit.calls == 0
        frame = law.frame()
        frame["S"] = SCORE[frame["W"].astype(int)]
        frame["effect_curve:cate"] = frame["S"] + 1.0
        study = CausalStudy(
            frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",))
        )
        with pytest.raises(CapabilityError, match="fitted on the rows it would be assessed on"):
            study.identify(linear_contrast()).assess_score(
                "S",
                computed_from=["W"],
                score_kind="external",
                compare="effect_curve:cate",
                compare_computed_from=["W"],
                **never(),
            )
        assert NeverFit.calls == 0

    def test_x97_compare_given_as_a_list_of_one(self) -> None:
        with pytest.raises(DataError, match=r"compare= takes one column name; got \['SB'\]"):
            self.compare(["SB"])
        assert NeverFit.calls == 0

    def test_x97_compare_given_as_a_non_string_scalar(self) -> None:
        with pytest.raises(DataError, match="compare= takes one column name as a string; got 5"):
            self.compare(5)
        assert NeverFit.calls == 0

    def test_x97_several_compare_columns(self) -> None:
        from cleverly.study import SCORE_MANY_REFUSAL

        with pytest.raises(CapabilityError) as caught:
            self.compare(["SB", "M"])
        record = caught.value.refusal
        assert record == SCORE_MANY_REFUSAL
        assert record.reason is RefusalReason.IMPLEMENTATION_GAP
        assert record.roadmap == "X98"
        assert record.condition == (
            "assess_score compares one score with one other; a simultaneous comparison of "
            "three or more scores, with a joint interval over every pair, is not built."
        )
        assert record.alternative == (
            "Pass one column as compare=. Intervals from separate calls do not hold jointly."
        )
        assert NeverFit.calls == 0

    @pytest.mark.parametrize(
        ("settings", "reason", "condition", "alternative"),
        [
            (
                {"repeats": 2},
                RefusalReason.INFERENCE_GAP,
                "The assessment targets one cross-fitting split's outcome regression; a median "
                "over repeated splits of jointly targeted risks has no reviewed joint covariance, "
                "and no study measures it.",
                "Fit with repeats=1.",
            ),
            (
                {"n_bootstrap": 20},
                RefusalReason.INFERENCE_GAP,
                "The refit bootstrap of the score assessment is not built; its intervals come "
                "from the influence curves.",
                "Fit with n_bootstrap=0; the Wald intervals are reported.",
            ),
            (
                {"target_weights": True},
                RefusalReason.IMPLEMENTATION_GAP,
                "The assessment fluctuates along several score columns at once, and every row "
                "loads several of them, which the weighted form of the fluctuation cannot carry.",
                "Fit with target_weights=False.",
            ),
            (
                {"stopping": "sample_size"},
                RefusalReason.IMPLEMENTATION_GAP,
                "The assessment solves its fluctuation under the shipped relative stopping test; "
                "a declared stopping rule needs a stopping context the assessment does not build.",
                "Fit with stopping='relative'.",
            ),
            (
                {"targeting_scheme": "fold"},
                RefusalReason.IMPLEMENTATION_GAP,
                "The assessment fits one pooled fluctuation over every row from the cross-fitted "
                "nuisances, so fold-wise targeting or fold evaluation of the ATE fit would be "
                "ignored.",
                "Fit with targeting_scheme='pooled' and cv_evaluation=False.",
            ),
            (
                {"cv_evaluation": True},
                RefusalReason.IMPLEMENTATION_GAP,
                "The assessment fits one pooled fluctuation over every row from the cross-fitted "
                "nuisances, so fold-wise targeting or fold evaluation of the ATE fit would be "
                "ignored.",
                "Fit with targeting_scheme='pooled' and cv_evaluation=False.",
            ),
        ],
    )
    @pytest.mark.parametrize("paired", [False, True])
    def test_x97_settings_the_assessment_cannot_honour(
        self,
        settings: dict[str, Any],
        reason: RefusalReason,
        condition: str,
        alternative: str,
        paired: bool,
    ) -> None:
        with pytest.raises(CapabilityError) as caught:
            if paired:
                self.compare(**settings)
            else:
                self.effect().assess_score(
                    "S", computed_from=["W"], score_kind="external", **settings, **never()
                )
        record = caught.value.refusal
        assert record.reason is reason
        assert record.condition == condition
        assert record.alternative == alternative
        # An implementation gap names the roadmap item that tracks it; an inference gap names none.
        expected = "X98" if reason is RefusalReason.IMPLEMENTATION_GAP else None
        assert record.roadmap == expected
        assert NeverFit.calls == 0

    def test_an_effect_registers_the_curves_fitted_from_it(self) -> None:
        effect = self.effect()
        curve = effect.estimate(**oracle()).effect_curve()
        assert any(np.array_equal(stored, curve.in_sample) for stored in effect._curves)


class WeightSpy(ClassifierMixin, BaseEstimator):
    """A candidate that records the ``sample_weight`` each of its fits receives."""

    seen: ClassVar[list[Any]] = []

    def __init__(self, inner: Any = None) -> None:
        self.inner = inner

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WeightSpy:
        WeightSpy.seen.append(None if sample_weight is None else np.array(sample_weight))
        self.fitted_ = clone(self.inner).fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.fitted_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.fitted_.predict_proba(X)


def spied(learner: Any) -> Any:
    """The study's ensemble with each candidate's final estimator wrapped in a spy.

    A pipeline keeps its steps and gets the spy as its last step, so the weight reaches it
    as ``<step>__sample_weight`` (``src/cleverly/learners/_fitting.py``); the boosting
    candidate is wrapped whole.  The candidates are made cheap, not changed in kind.
    """
    library = []
    for name, candidate in learner.library:
        if isinstance(candidate, Pipeline):
            step, final = candidate.steps[-1]
            if hasattr(final, "Cs"):
                final = clone(final).set_params(Cs=3, cv=2)
            candidate = Pipeline([*candidate.steps[:-1], (step, WeightSpy(final))])
        else:
            candidate = WeightSpy(clone(candidate).set_params(max_iter=10))
        library.append((name, candidate))
    return clone(learner).set_params(library=library, n_folds=2)


def test_the_ensemble_hands_the_row_weights_to_every_candidate() -> None:
    """Plan 4.2: each candidate's fit receives the weights, in every outer and inner fold."""
    from tests.studies import conditional_projection as declared

    group = declared.GROUP_BY_LABEL["weighted_sl_2k"]
    frame = declared.sample("weighted", 300, np.random.default_rng(0))
    study = CausalStudy(frame, design=declared.design(group))
    WeightSpy.seen.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.identify(declared.estimand(group)).estimate(
            outcome_learner=spied(declared.ensemble((0.005, 0.995))),
            treatment_learner=spied(declared.ensemble((0.01, 0.99))),
            outcome_by_arm=True,
            n_folds=2,
            random_state=0,
        )
    # Two outer folds, three nuisances (two arms and the mechanism), three candidates, and
    # each candidate fitted on two inner folds and once on its whole training fold.
    assert len(WeightSpy.seen) == 2 * 3 * 3 * 3
    assert all(weights is not None for weights in WeightSpy.seen)
    assert all(np.ptp(weights) > 0.5 for weights in WeightSpy.seen)


def test_a_learned_fit_runs_end_to_end() -> None:
    """Cross-fitted GLM nuisances on a continuous modifier, both scales."""
    rng = np.random.default_rng(3)
    n = 600
    w = rng.uniform(-1, 1, n)
    a = rng.binomial(1, 0.5, n)
    y = rng.binomial(1, 0.3 + 0.2 * a * (w > 0))
    frame = pd.DataFrame({"Y": y, "A": a, "W": w})
    study = CausalStudy(frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)))
    for scale in ("difference", "log_ratio"):
        fit = study.identify(
            ConditionalContrast(Modifiers(["W"]), scale=scale, model=Linear(["W"]))  # type: ignore[arg-type]
        ).estimate(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            random_state=0,
        )
        assert fit.heterogeneity_test().df == 1
        assert np.isfinite(fit.effect_curve().in_sample).all()


def _null_cluster_test(rng: np.random.Generator, *, clusters: int, terms: int) -> Any:
    """The heterogeneity test of ``terms`` null coefficients from normal cluster totals.

    Every row of a cluster carries the cluster's total, so the estimate is the mean of the
    totals and the cluster-summed covariance is their sample covariance.
    """
    from cleverly.effect_curves import heterogeneity_test
    from cleverly.inference.influence import make_estimate

    rows = 4
    labels = np.repeat(np.arange(clusters), rows)
    totals = rng.normal(size=(clusters, terms + 1))
    curves = totals[labels]
    names = tuple(f"b{j}" for j in range(terms + 1))
    estimates = {
        name: make_estimate(
            name,
            float(curves[:, j].mean()),
            curves[:, j] - curves[:, j].mean(),
            n=len(labels),
            cluster=labels,
            inference="few_cluster_plugin",
            reference_df=clusters - 2,
            rule_widened=False,
        )
        for j, name in enumerate(names)
    }
    return heterogeneity_test(
        estimates,
        names,
        ("(intercept)", *(f"t{j}" for j in range(terms))),
        (0,),
        scale="difference",
        cluster=labels,
    )
