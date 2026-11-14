r"""The known-score ATT and ATC (``TMLE(conditional_form="known_score")``).

With the treatment mechanism declared, the ATT is the functional
:math:`\Psi_{g_0}(P)=E_P\{g_0(W)\tau_P(W)\}/E_P\{g_0(W)\}` of the model in which
:math:`g_0` is known. Its canonical gradient is

.. math::

    D_{KS} = \frac{1}{p}\Big[A(Y-\bar Q_1) - (1-A)\frac{g_0}{1-g_0}(Y-\bar Q_0)
             + g_0(W)(\tau-\psi)\Big],

the shipped ATT curve minus its projection :math:`(A-g_0)(\tau-\psi)/p` on the mechanism's
tangent space. Its variance is the known-score ATT bound that Chen, Hong and Tarozzi (2008,
Theorem 2) and Imbens (2004, Section 2.5) state.

The checks here derive the truth from the definition. :math:`\Psi_{g_0}` with :math:`g_0`
held fixed is the ATE of the law tilted by the baseline weight :math:`g_0(w)` (a tilt on
:math:`W` alone leaves every conditional law unchanged), so the Gateaux derivative of
that longhand functional on each finite-support law is the curve the fit must report. No
clever covariate and no library code enter the truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, MethodConfigurationError, RefusalReason
from tests import discrete_law as law
from tests import discrete_law_cde as cde
from tests import discrete_law_mar as mar
from tests import discrete_law_multi as multi
from tests.conftest import (
    OracleDirectOutcome,
    OracleIntermediate,
    OracleMissingness,
    OracleOutcome,
    OracleTreatment,
)
from tests.unit._exact_sensitivity_support import REFERENCE, multi_first_row_of
from tests.unit._natural_course_support import NeverFit

KNOWN = "known_score"


def _baseline_tilt(probs: Any, weight: Any) -> Any:
    """The law tilted by a weight of ``W`` alone (axis 0), analytic in ``probs``."""
    p = np.asarray(probs)
    w = np.asarray(weight).reshape((-1,) + (1,) * (p.ndim - 1))
    tilted = p * w
    return tilted / tilted.sum()


def _known(functional: Any, weight: np.ndarray) -> Any:
    """``P -> E_P[g(W) tau_P(W)] / E_P[g(W)]`` from a law's own ATE ``functional``."""
    return lambda p: functional(_baseline_tilt(p, weight))


# ----------------------------------------------------------------------------- fits


def _binary_fit(form: str, *, weights: str | None = None, probs: Any = None) -> Any:
    counts = None if probs is None else np.rint(np.asarray(probs) * law.N).astype(int)
    frame = law.frame(counts)
    dgp = law.DiscreteLaw(probs)
    kwargs: dict[str, Any] = {}
    if weights is not None:
        cells = law.cell_weights(law.WEIGHT_FUNCTIONS[weights])
        frame = frame.assign(w=law.row_weights(cells))
        kwargs["weights"] = "w"
    g = dgp.g[frame["W"].to_numpy().astype(int)]
    return (
        TMLE(
            outcome_learner=OracleOutcome(dgp),
            treatment_learner=OracleTreatment(dgp),
            cross_fit=False,
            estimands=("att", "atc"),
            simultaneous=False,
            random_state=0,
            conditional_form=form,
        )
        .fit(
            frame, outcome="Y", treatment="A", covariates=["W"], treatment_probabilities=g, **kwargs
        )
        .single()
    )


def _support_curve(result: Any, name: str, first_rows: np.ndarray) -> np.ndarray:
    return np.asarray(result.estimates[name].influence_curve)[first_rows]


# ----------------------------------------------------------------------------- exact law


class TestTheCurveIsTheKnownScoreGradient:
    @pytest.mark.parametrize(("name", "column"), [("att", 1), ("atc", 0)])
    def test_binary(self, name: str, column: int) -> None:
        result = _binary_fit(KNOWN)
        g = law.G_EXACT if column == 1 else 1.0 - law.G_EXACT
        truth = _known(lambda p: law.functional(p, "ate"), g)
        np.testing.assert_allclose(
            _support_curve(result, name, law.first_row_of()),
            law.gateaux_eif(truth),
            atol=1e-12,
            rtol=0,
        )
        assert result.estimates[name].psi == pytest.approx(float(truth(law.PROBS)), abs=1e-12)

    @pytest.mark.parametrize(("name", "column"), [("att", 1), ("atc", 0)])
    def test_weighted_on_the_baseline(self, name: str, column: int) -> None:
        # ``baseline`` weights tilt P(W) alone, so the declared G is the tilted law's
        # mechanism and the fixed-G functional is evaluated on the tilted law.
        result = _binary_fit(KNOWN, weights="baseline")
        cells = law.cell_weights(law.WEIGHT_FUNCTIONS["baseline"])
        g = law.G_EXACT if column == 1 else 1.0 - law.G_EXACT
        truth = _known(lambda p: law.functional(p, "ate"), g)
        expected = law.gateaux_eif(lambda p: truth(law.tilt(p, cells)))
        np.testing.assert_allclose(
            _support_curve(result, name, law.first_row_of()), expected, atol=1e-11, rtol=0
        )

    @pytest.mark.parametrize("stem", ["att", "atc"])
    @pytest.mark.parametrize("arm", [1, 2])
    def test_three_arms(self, stem: str, arm: int) -> None:
        frame = multi.frame()
        index = frame["W"].to_numpy().astype(int)
        declared = {label: multi.G_EXACT[index, a] for a, label in enumerate(multi.LABELS)}
        result = (
            TMLE(
                outcome_learner=multi.OracleMultiOutcome(),
                treatment_learner=multi.OracleMultiTreatment(),
                cross_fit=False,
                estimands=(stem,),
                reference=REFERENCE,
                simultaneous=False,
                random_state=0,
                conditional_form=KNOWN,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                treatment_probabilities=declared,
            )
            .single()
        )
        reference = multi.LABELS.index(REFERENCE)
        conditioning = arm if stem == "att" else reference
        truth = _known(
            lambda p: multi.functional(p, f"ate[{arm} vs {reference}]"),
            multi.G_EXACT[:, conditioning],
        )
        name = multi.reported_name(f"{stem}[{arm} vs {reference}]")
        expected = law.contamination_eif(truth, multi.PROBS, multi.SUPPORT)
        np.testing.assert_allclose(
            _support_curve(result, name, multi_first_row_of()), expected, atol=1e-12, rtol=0
        )

    @pytest.mark.parametrize(("name", "column"), [("att", 1), ("atc", 0)])
    def test_missing_outcomes(self, name: str, column: int) -> None:
        dgp = mar.DiscreteLaw()
        frame = mar.frame()
        g = mar.G_EXACT[frame["W"].to_numpy().astype(int)]
        result = (
            TMLE(
                outcome_learner=OracleOutcome(dgp),
                treatment_learner=OracleTreatment(dgp),
                missingness_learner=OracleMissingness(dgp),
                cross_fit=False,
                estimands=(name,),
                simultaneous=False,
                random_state=0,
                conditional_form=KNOWN,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W"],
                delta="Delta",
                treatment_probabilities=g,
            )
            .single()
        )
        weight = mar.G_EXACT if column == 1 else 1.0 - mar.G_EXACT
        truth = _known(lambda p: mar.functional(p, "ate"), weight)
        expected = law.contamination_eif(truth, mar.PROBS, mar.SUPPORT)
        np.testing.assert_allclose(
            _support_curve(result, name, mar.first_row_of()), expected, atol=1e-12, rtol=0
        )

    @pytest.mark.parametrize(("name", "column"), [("att", 1), ("atc", 0)])
    @pytest.mark.parametrize("level", cde.LEVELS)
    def test_controlled_direct_effect(self, name: str, column: int, level: int) -> None:
        dgp = cde.DiscreteLaw()
        frame = cde.frame()
        g = cde.G_EXACT[frame["W"].to_numpy().astype(int)]
        results = TMLE(
            outcome_learner=OracleDirectOutcome(dgp),
            treatment_learner=OracleTreatment(dgp),
            missingness_learner=OracleMissingness(dgp),
            intermediate_learner=OracleIntermediate(dgp),
            cross_fit=False,
            estimands=(name,),
            simultaneous=False,
            random_state=0,
            conditional_form=KNOWN,
        ).fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            intermediate="Z",
            treatment_probabilities=g,
        )
        weight = cde.G_EXACT if column == 1 else 1.0 - cde.G_EXACT
        truth = _known(lambda p: cde.functional(p, "ate", level), weight)
        expected = law.contamination_eif(truth, cde.PROBS, cde.SUPPORT)
        reported = _support_curve(results[float(level)], name, cde.first_row_of())
        np.testing.assert_allclose(reported, expected, atol=1e-12, rtol=0)


# ----------------------------------------------------------------------------- witnesses


def _tau_psi(probs: Any = None) -> tuple[np.ndarray, np.ndarray, float, float]:
    p = law.PROBS if probs is None else np.asarray(probs)
    p_w = p.sum(axis=(1, 2))
    g = p.sum(axis=2)[:, 1] / p_w
    q = p[:, :, 1] / p.sum(axis=2)
    tau = q[:, 1] - q[:, 0]
    share = float((p_w * g).sum())
    psi = float((p_w * g * tau).sum() / share)
    return g, tau, psi, share


class TestTheChangedTermIsNotZero:
    def test_the_projection_term_is_nonzero_on_the_law(self) -> None:
        arm_mean, known = _binary_fit("arm_mean"), _binary_fit(KNOWN)
        rows = law.first_row_of()
        gap = _support_curve(arm_mean, "att", rows) - _support_curve(known, "att", rows)
        g, tau, psi, share = _tau_psi()
        projection = np.array([(a - g[w]) * (tau[w] - psi) / share for w, a, _ in law.SUPPORT])
        np.testing.assert_allclose(gap, projection, atol=1e-12, rtol=0)
        assert np.max(np.abs(projection)) > 1e-2

    def test_the_arm_mean_curve_is_not_the_known_score_gradient(self) -> None:
        # Mutation control: a build that ignored the option would report this curve.
        truth = _known(lambda p: law.functional(p, "ate"), law.G_EXACT)
        reported = _support_curve(_binary_fit("arm_mean"), "att", law.first_row_of())
        assert np.max(np.abs(reported - law.gateaux_eif(truth))) > 1e-2


class TestTheVarianceWitness:
    """The acceptance witness: the known-score standard error is the smaller one."""

    @staticmethod
    def _variance_gap(probs: Any = None) -> tuple[float, float, float]:
        arm_mean = _binary_fit("arm_mean", probs=probs).estimates["att"]
        known = _binary_fit(KNOWN, probs=probs).estimates["att"]
        g, tau, psi, share = _tau_psi(probs)
        p = law.PROBS if probs is None else np.asarray(probs)
        p_w = p.sum(axis=(1, 2))
        gain = float((p_w * g * (1 - g) * (tau - psi) ** 2).sum() / share**2)
        # The reported variance is the sample variance (ddof=1) over n.
        n = law.N
        reported = (n - 1) * (arm_mean.std_error**2 - known.std_error**2)
        return reported, gain, known.std_error / arm_mean.std_error

    def test_the_gap_is_the_known_score_gain(self) -> None:
        reported, gain, ratio = self._variance_gap()
        assert reported == pytest.approx(gain, abs=1e-10)
        assert gain > 1e-4
        assert ratio < 1.0

    def test_a_constant_effect_gains_nothing(self) -> None:
        probs = law.cell_counts(q=[[0.4, 0.7], [0.2, 0.5], [0.6, 0.9]]) / law.N
        reported, gain, ratio = self._variance_gap(probs)
        assert gain == pytest.approx(0.0, abs=1e-15)
        assert reported == pytest.approx(0.0, abs=1e-12)
        assert ratio == pytest.approx(1.0, abs=1e-12)

    def test_a_constant_mechanism_still_gains(self) -> None:
        # A completely randomized trial: the known-score form averages every unit's
        # covariates, the arm-mean form only the treated units'.
        probs = law.cell_counts(g=[0.4, 0.4, 0.4]) / law.N
        reported, gain, ratio = self._variance_gap(probs)
        assert gain > 1e-4
        assert reported == pytest.approx(gain, abs=1e-10)
        assert ratio < 1.0


# ----------------------------------------------------------------------------- sampled frames


def _sampled_frame(n: int = 600, arms: int = 2, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.5, 1.5, size=n)
    logits = np.column_stack([np.zeros(n)] + [1.0 * k * w1 + 0.2 * k for k in range(1, arms)])
    g = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    a = (rng.uniform(size=n)[:, None] > np.cumsum(g, axis=1)).sum(axis=1).astype(float)
    # A binary outcome keeps the targeted arms on the reported scale.
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(0.8 * w1 + a * (0.5 + 2.5 * w1))))).astype(float)
    frame = pd.DataFrame({"W1": w1, "A": a, "Y": y})
    for k in range(arms):
        frame[f"p{k}"] = g[:, k]
    return frame


def _sampled_fit(frame: pd.DataFrame, form: str, estimands: Any, **settings: Any) -> Any:
    arms = sum(1 for column in frame if column.startswith("p"))
    base: dict[str, Any] = {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": NeverFit(),
        "cross_fit": False,
        "estimands": estimands,
        "simultaneous": False,
        "random_state": 0,
        "conditional_form": form,
    }
    base.update(settings)
    return (
        TMLE(**base)
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1"],
            treatment_probabilities={float(k): f"p{k}" for k in range(arms)},
        )
        .single()
    )


class TestThePlugIn:
    @pytest.mark.parametrize("arms", [2, 3])
    def test_the_estimate_is_the_score_weighted_mean(self, arms: int) -> None:
        frame = _sampled_frame(arms=arms)
        known = _sampled_fit(frame, KNOWN, ("att", "atc"))
        arm_mean = _sampled_fit(frame, "arm_mean", ("att", "atc"))
        for stem, group in (("att", "att"), ("atc", "atc")):
            targeted = known.fluctuations[group].targeted
            for arm in range(1, arms):
                name = stem if arms == 2 else f"{stem}[{float(arm):g} vs 0]"
                if name not in known.estimates:
                    name = next(k for k in known.estimates if k.startswith(stem) and f"{arm}" in k)
                conditioning = arm if stem == "att" else 0
                weight = frame[f"p{conditioning}"].to_numpy()
                tau = targeted.arms[float(arm)] - targeted.arms[0.0]
                expected = float((weight * tau).sum() / weight.sum())
                assert known.estimates[name].psi == pytest.approx(expected, abs=1e-12)
                # The two plug-ins differ by a sampling quantity of order n^(-1/2) times the
                # spread of tau; the floor is far above floating-point noise.
                assert abs(known.estimates[name].psi - arm_mean.estimates[name].psi) > 1e-4


# ----------------------------------------------------------------------------- refusals


class TestRefusals:
    def test_an_unknown_form(self) -> None:
        with pytest.raises(MethodConfigurationError, match="conditional_form must be one of"):
            TMLE(conditional_form="treated")

    def test_an_estimated_mechanism(self) -> None:
        frame = _sampled_frame()
        estimator = TMLE(
            outcome_learner=LinearRegression(),
            cross_fit=False,
            estimands=("att",),
            conditional_form=KNOWN,
        )
        with pytest.raises(CapabilityError, match="this fit estimates the mechanism") as info:
            estimator.fit(frame, outcome="Y", treatment="A", covariates=["W1"])
        assert info.value.refusal.reason == RefusalReason.INFERENCE_GAP

    def test_no_conditional_target(self) -> None:
        with pytest.raises(MethodConfigurationError, match="requests neither"):
            _sampled_fit(_sampled_frame(), KNOWN, ("ate",))

    def test_adaptive_iptw(self) -> None:
        from cleverly.estimators import AdaptiveIPTW

        with pytest.raises(MethodConfigurationError, match="conditional_form"):
            AdaptiveIPTW(conditional_form=KNOWN)


# ----------------------------------------------------------------------------- plumbing


class TestPlumbing:
    def test_config_and_summary_record_the_form(self) -> None:
        result = _sampled_fit(_sampled_frame(), KNOWN, ("att",))
        assert result.config.conditional_form == KNOWN
        assert "conditional_form='known_score'" in str(result.summary(detail="full"))

    def test_the_default_is_unchanged(self) -> None:
        frame = _sampled_frame()
        default = (
            TMLE(
                outcome_learner=LogisticRegression(),
                treatment_learner=NeverFit(),
                cross_fit=False,
                estimands=("att", "atc"),
                simultaneous=False,
                random_state=0,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1"],
                treatment_probabilities={0.0: "p0", 1.0: "p1"},
            )
            .single()
        )
        explicit = _sampled_fit(frame, "arm_mean", ("att", "atc"))
        for name in ("att", "atc"):
            assert default.estimates[name].psi == explicit.estimates[name].psi
        assert default.config.conditional_form == "arm_mean"

    def test_refit_and_retarget_keep_the_form(self) -> None:
        frame = _sampled_frame()
        result = _sampled_fit(frame, KNOWN, ("att",))
        estimator = TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=NeverFit(),
            cross_fit=False,
            estimands=("att",),
            simultaneous=False,
            random_state=0,
            conditional_form=KNOWN,
        )
        again = estimator.refit(result.data)
        assert again.estimates["att"].psi == result.estimates["att"].psi
        retargeted, _ = estimator.retarget(result.data, result.nuisance, estimands=("att",))
        assert retargeted["att"].psi == pytest.approx(result.estimates["att"].psi, abs=1e-12)

    def test_strata_slice_the_declared_mechanism(self) -> None:
        frame = _sampled_frame(n=800).assign(S=lambda f: (f["W1"] > 0).astype(int))
        result = (
            TMLE(
                outcome_learner=LinearRegression(),
                cross_fit=False,
                estimands=("att",),
                simultaneous=False,
                random_state=0,
                conditional_form=KNOWN,
            )
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "S"],
                strata=["S"],
                treatment_probabilities={0.0: "p0", 1.0: "p1"},
            )
            .single()
        )
        targeted = result.fluctuations["att"].targeted
        tau = targeted.arms[1.0] - targeted.arms[0.0]
        g = frame["p1"].to_numpy()
        stratum = frame["S"].to_numpy() == 1
        expected = float((g * tau)[stratum].sum() / g[stratum].sum())
        name = next(k for k in result.estimates if k.startswith("att") and k != "att")
        stratum_estimates = {k: v.psi for k, v in result.estimates.items() if k.startswith("att")}
        assert any(abs(v - expected) < 1e-10 for v in stratum_estimates.values()), name


# ----------------------------------------------------------------------------- readers


def _mar_known_fit() -> Any:
    dgp = mar.DiscreteLaw()
    frame = mar.frame()
    return (
        TMLE(
            outcome_learner=OracleOutcome(dgp),
            treatment_learner=OracleTreatment(dgp),
            missingness_learner=OracleMissingness(dgp),
            cross_fit=False,
            estimands=("ate", "att"),
            simultaneous=False,
            random_state=0,
            conditional_form=KNOWN,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            treatment_probabilities=mar.G_EXACT[frame["W"].to_numpy().astype(int)],
        )
        .single()
    )


class TestReaders:
    """Each reader of the ATT either carries the known-score form or refuses it."""

    def test_the_mnar_tilt_refuses_the_att_and_tilts_the_rest(self) -> None:
        from cleverly.sensitivity import missingness_tilt

        result = _mar_known_fit()
        with pytest.raises(CapabilityError, match="the MNAR tilt recomputes the ATT") as info:
            missingness_tilt(result, [0.0, 0.5], estimands=["att"])
        assert info.value.refusal.reason == RefusalReason.INFERENCE_GAP
        frame = missingness_tilt(result, [0.0, 0.5]).to_frame()
        assert set(frame["estimand"]) == {"ate"}

    def test_the_omitted_variable_bound_refuses_the_att(self) -> None:
        from cleverly.sensitivity import omitted_variable_bounds

        result = _sampled_fit(_sampled_frame(), KNOWN, ("ate", "att"))
        with pytest.raises(CapabilityError, match="the omitted-variable bound recomputes"):
            omitted_variable_bounds(result, "att")

    def test_the_marginal_bound_refuses_the_att(self) -> None:
        from cleverly.sensitivity import marginal_sensitivity_bounds

        result = _sampled_fit(_sampled_frame(), KNOWN, ("ate", "att"))
        with pytest.raises(CapabilityError, match="the marginal sensitivity bound recomputes"):
            marginal_sensitivity_bounds(result, 1.5, estimand="att")

    def test_the_placebo_and_subset_refutations_carry_the_form(self) -> None:
        from cleverly.validation import refute

        result = _sampled_fit(_sampled_frame(), KNOWN, ("att",))
        report = refute(
            result, estimand="att", tests=("placebo", "subset"), n_replicates=2, random_state=1
        )
        assert {test.name for test in report.tests} == {"placebo", "subset"}

    def test_the_bootstrap_carries_the_form(self) -> None:
        frame = _sampled_frame()
        result = _sampled_fit(frame, KNOWN, ("att",), n_bootstrap=20)
        assert result.bootstrap is not None
        assert np.all(np.isfinite(result.bootstrap.draws["att"]))


# ----------------------------------------------------------------------------- study API


class TestStudyAPI:
    def test_targeting_carries_the_form_to_the_fit(self) -> None:
        from cleverly import ATT, CausalStudy, PointTreatment, TMLEMethod

        frame = _sampled_frame()
        design = PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1",),
            treatment_probabilities={0.0: "p0", 1.0: "p1"},
        )
        effect = CausalStudy(frame, design=design).identify(ATT())
        method = TMLEMethod().with_overrides(
            outcome_learner=LogisticRegression(),
            treatment_learner=NeverFit(),
            cross_fit=False,
            random_state=0,
            conditional_form=KNOWN,
        )
        result = effect.estimate(method)
        assert result.config.conditional_form == KNOWN
        direct = _sampled_fit(frame, KNOWN, ("att",))
        assert result.estimates["att"].psi == pytest.approx(direct.estimates["att"].psi, abs=1e-12)
