r"""The projection of a controlled direct effect onto a declared basis (P1 PR 2, part a).

``ControlledDirectEffect(intermediate=z, contrast=ConditionalContrast(..., model=basis))``
estimates :math:`\beta_z = M^{-1} E[\varphi(V)\{\bar Q(t, z, W) - \bar Q(r, z, W)\}]` (difference)
or the root of the log-CRR projection's equation at :math:`\bar Q(\cdot, z, W)` (log ratio).
Its clever covariate is :math:`c_a(V)\,1\{A = a\}1\{Z = z\}\Delta/\{g\,q_z\,\pi\}` on the observed
row and :math:`c_a(V)/\{g\,q_z\,\pi\}` on the counterfactual rows.

The law is :mod:`tests.discrete_law_cde_projection`, whose :math:`\tau_z` is not linear in
``W``, so averaging over the wrong rows is visible under ``Linear(W)``.  Five components can
vanish at the truth, and each has a witness here:

- the plug-in stratum (averaging the contrast over the ``Z = z`` rows), and the
  counterfactual ``Z = z`` block of the outcome design: law-level witnesses that the mutant
  functional differs from the truth the fit reproduces;
- the covariate's selection ``1{Z = z}``, its density ``q_z`` and its ``1/pi``: each vanishes
  at a correct outcome regression, so each is mutated at an arm-specific wrong ``Qbar``
  (``+0.125 W`` on arm 1), where the full covariate still recovers the truth exactly and each
  mutant misses it by its tabled vector.

Then every ``isinstance`` site and pseudo-outcome caller the composition reaches, every
composition the deleted refusal makes reachable, and the refusals that stay or are new.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from cleverly import CausalStudy, ConditionalContrast, ControlledDirectEffect, PointTreatment
from cleverly.bases import Linear, Modifiers
from cleverly.effect_modification import (
    CDE_CURVE_TOOLS_REFUSAL,
    EP_INTERMEDIATE_REFUSAL,
    INTERMEDIATE_STRATA_REFUSAL,
)
from cleverly.estimators import targeting
from cleverly.estimators._nuisance import NuisanceEstimates
from cleverly.exceptions import CapabilityError, DataError
from tests import discrete_law_cde_projection as law
from tests.conftest import (
    OracleDirectOutcome,
    OracleIntermediate,
    OracleMissingness,
    OracleTreatment,
)
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_oat import SaturatedCategorical
from tests.unit._natural_course_support import NeverFit

MODIFIERS = Modifiers(["W"], domains={"W": (0.0, 2.0)})
STEM = {"difference": "cate_projection", "log_ratio": "log_crr_projection"}
SCALES = ("difference", "log_ratio")
LEVELS = (0, 1)

#: Exact checks hold to this tolerance; a witness must clear the floor.
EXACT = 1e-10
FLOOR = 0.005

#: Each covariate mutant's displacement from the truth at the wrong outcome regression, by
#: level, from the plan's independent computation (``disc_mut.py``, rows/P1/plan-pr2.md 4.1).
MUTANT_DISPLACEMENT = {
    "selection": {0: (0.0128, 0.0271), 1: (-0.0352, 0.0111)},
    "density": {0: (-0.0375, 0.0174), 1: (0.0086, 0.0088)},
    "response": {0: (0.0060, 0.0136), 1: (-0.0007, 0.0131)},
}


def quiet(function: Any) -> Any:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return function(*args, **kwargs)

    return wrapped


def design(*, missing: bool = True, **extra: Any) -> PointTreatment:
    return PointTreatment(
        outcome="Y",
        treatment="A",
        adjustment=("W", "S") if "strata" in extra else ("W",),
        missingness="Delta" if missing else None,
        intermediate="Z",
        **extra,
    )


def contrast(scale: str = "difference", **settings: Any) -> ConditionalContrast:
    return ConditionalContrast(MODIFIERS, scale=scale, model=Linear(["W"]), **settings)  # type: ignore[arg-type]


def effect(
    level: int,
    scale: str = "difference",
    *,
    frame: pd.DataFrame | None = None,
    missing: bool = True,
    **extra: Any,
) -> Any:
    data = law.frame(complete=not missing) if frame is None else frame
    study = CausalStudy(data, design=design(missing=missing, **extra))
    return study.identify(
        ControlledDirectEffect(intermediate=float(level), contrast=contrast(scale))
    )


def oracle(dgp: Any = None, outcome: Any = None) -> dict[str, Any]:
    dgp = law.law() if dgp is None else dgp
    return {
        "outcome_learner": OracleDirectOutcome(dgp if outcome is None else outcome),
        "treatment_learner": OracleTreatment(dgp),
        "missingness_learner": OracleMissingness(dgp),
        "intermediate_learner": OracleIntermediate(dgp),
        "cross_fit": False,
        "random_state": 0,
    }


def never() -> dict[str, Any]:
    NeverFit.calls = 0
    return {
        "outcome_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "missingness_learner": NeverFit(),
        "intermediate_learner": NeverFit(),
    }


def beta(fit: Any) -> np.ndarray:
    return np.array([estimate.psi for estimate in fit.estimates.values()])


@functools.cache
@quiet
def exact_fit(level: int, scale: str) -> Any:
    """The in-sample missing-outcome fit at the oracle nuisances."""
    return effect(level, scale).estimate(**oracle())


def wrong_outcome() -> Any:
    """The arm-specific wrong ``Qbar``: ``+0.125 W`` on arm 1, at both levels."""
    bad = law.law()
    bad.q = bad.q.copy()
    bad.q[:, 1, :] += 0.125 * np.arange(3.0)[:, None]
    return bad


def wrong_complete_outcome() -> Any:
    """:func:`wrong_outcome` on the complete-row law."""
    bad = law.law(complete=True)
    bad.q = bad.q.copy()
    bad.q[:, 1, :] += 0.125 * np.arange(3.0)[:, None]
    return bad


def closed_form_curve(
    level: int,
    scale: str,
    coefficients: np.ndarray,
    *,
    frame: pd.DataFrame | None = None,
    dgp: Any = None,
) -> np.ndarray:
    """The influence curve of rows/P1/plan-pr2.md 2.1 at the oracle nuisances, row by row."""
    frame = law.frame() if frame is None else frame
    dgp = law.law() if dgp is None else dgp
    w = frame["W"].to_numpy().astype(int)
    a = frame["A"].to_numpy().astype(int)
    z = frame["Z"].to_numpy()
    delta = frame["Delta"].to_numpy() if "Delta" in frame else np.ones(len(frame))
    y = np.nan_to_num(frame["Y"].to_numpy())
    phi = np.column_stack([np.ones(len(frame)), w])
    q_t, q_r = dgp.q[w, 1, level], dgp.q[w, 0, level]
    g = np.where(a == 1, dgp.g[w], 1.0 - dgp.g[w])
    q_one = dgp.qz[w, a]
    q_z = q_one if level == 1 else 1.0 - q_one
    residual = (z == level) * delta / (g * q_z * dgp.pi[w, a]) * (y - dgp.q[w, a, level])
    n = len(frame)
    if scale == "difference":
        gram = phi.T @ phi / n
        sign = np.where(a == 1, 1.0, -1.0)
        inner = phi * ((q_t - q_r) - phi @ coefficients)[:, None] + phi * (sign * residual)[:, None]
        return inner @ np.linalg.inv(gram).T
    m = expit(phi @ coefficients)
    total = q_t + q_r
    jacobian = phi.T @ (phi * (total * m * (1.0 - m))[:, None]) / n
    numerator = np.where(a == 1, m - 1.0, m)
    inner = phi * (total * m - q_t)[:, None] + phi * (numerator * residual)[:, None]
    return -inner @ np.linalg.inv(jacobian).T


# ---------------------------------------------------------------- the exact law


class TestTheExactLaw:
    @pytest.mark.parametrize("scale", SCALES)
    @pytest.mark.parametrize("level", LEVELS)
    def test_the_projection_is_the_truth_and_its_curve_the_closed_form(
        self, level: int, scale: str
    ) -> None:
        fit = exact_fit(level, scale)
        assert list(fit.estimates) == [f"{STEM[scale]}[(intercept)]", f"{STEM[scale]}[W]"]
        np.testing.assert_allclose(beta(fit), law.truth(level, scale), rtol=0, atol=EXACT)
        curves = np.column_stack([e.influence_curve for e in fit.estimates.values()])
        np.testing.assert_allclose(curves.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(
            curves, closed_form_curve(level, scale, beta(fit)), rtol=0, atol=1e-9
        )

    def test_the_two_levels_are_different_parameters(self) -> None:
        """Confusing the levels moves the answer by far more than the floor on both scales."""
        for scale in SCALES:
            gap = law.truth(0, scale) - law.truth(1, scale)
            assert np.max(np.abs(gap)) > 0.3

    @pytest.mark.parametrize("scale", SCALES)
    @quiet
    def test_in_sample_with_complete_outcomes(self, scale: str) -> None:
        dgp = law.law(complete=True)
        for level in LEVELS:
            fit = effect(level, scale, missing=False).estimate(**oracle(dgp))
            np.testing.assert_allclose(
                beta(fit), law.truth(level, scale, law.complete_probs()), rtol=0, atol=EXACT
            )

    @quiet
    def test_one_engine_fit_shares_the_nuisances_over_both_levels(self) -> None:
        """Both levels come from one pass of the four learners (``tmle.py`` shares them)."""
        from cleverly.effect_modification import EffectModification
        from cleverly.estimators import TMLE

        calls: list[str] = []

        class Counting(OracleDirectOutcome):
            def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
                calls.append("outcome")
                return super().fit(X, y, sample_weight)

        dgp = law.law()
        settings = oracle(dgp)
        settings["outcome_learner"] = Counting(dgp)
        estimator = TMLE(
            effect_modification=EffectModification(Linear(["W"]), MODIFIERS, "difference"),
            **settings,
        )
        results = estimator.fit(
            law.frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            delta="Delta",
            intermediate="Z",
        )
        assert len(calls) == 1
        for level in LEVELS:
            np.testing.assert_allclose(
                beta(results[float(level)]), law.truth(level), rtol=0, atol=EXACT
            )


# ---------------------------------------------------------------- witnesses


class TestTheTermsThatVanishAtTheTruth:
    @pytest.mark.parametrize("level", LEVELS)
    def test_averaging_the_contrast_over_the_stratum_would_be_caught(self, level: int) -> None:
        """Plug-in stratum: ``M_z^{-1} P_n[1{Z = z} phi (Q*_t - Q*_r)]`` is another number."""
        frame = law.frame()
        dgp = law.law()
        w = frame["W"].to_numpy().astype(int)
        keep = frame["Z"].to_numpy() == level
        phi = np.column_stack([np.ones(len(frame)), w])[keep]
        tau = (dgp.q[w, 1, level] - dgp.q[w, 0, level])[keep]
        mutant = np.linalg.solve(phi.T @ phi, phi.T @ tau)
        np.testing.assert_allclose(mutant, law.plugin_stratum(level), atol=1e-12)
        expected = {0: (0.0882, -0.2647), 1: (0.4474, -0.3816)}[level]
        np.testing.assert_allclose(mutant, expected, atol=5e-5)
        assert np.max(np.abs(mutant - beta(exact_fit(level, "difference")))) > 0.03

    def test_predicting_at_the_observed_intermediate_would_be_caught(self) -> None:
        """Counterfactual block: the outcome at each row's own ``Z`` gives (0.3523, -0.3864)."""
        frame = law.frame()
        dgp = law.law()
        w = frame["W"].to_numpy().astype(int)
        z = frame["Z"].to_numpy().astype(int)
        phi = np.column_stack([np.ones(len(frame)), w])
        tau = dgp.q[w, 1, z] - dgp.q[w, 0, z]
        mutant = np.linalg.solve(phi.T @ phi, phi.T @ tau)
        np.testing.assert_allclose(mutant, law.observed_z_prediction(), atol=1e-12)
        np.testing.assert_allclose(mutant, (0.3523, -0.3864), atol=5e-5)
        for level in LEVELS:
            assert np.max(np.abs(mutant - beta(exact_fit(level, "difference")))) > 0.05

    @pytest.mark.parametrize("level", LEVELS)
    @quiet
    def test_the_full_covariate_recovers_the_truth_at_a_wrong_outcome_regression(
        self, level: int
    ) -> None:
        fit = effect(level).estimate(**oracle(outcome=wrong_outcome()))
        np.testing.assert_allclose(beta(fit), law.truth(level), rtol=0, atol=EXACT)
        # Witness: the wrong regression's plug-in is off by more than the floor.
        plug_in = law.projection(
            law.PROBS.sum(axis=(1, 2, 3)),
            wrong_outcome().q[:, 1, level] - wrong_outcome().q[:, 0, level],
        )
        assert np.max(np.abs(plug_in - law.truth(level))) > 0.05

    @pytest.mark.parametrize("mutation", sorted(MUTANT_DISPLACEMENT))
    @pytest.mark.parametrize("level", LEVELS)
    @quiet
    def test_each_covariate_mutant_misses_the_truth_by_its_tabled_vector(
        self, mutation: str, level: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inputs = targeting.clever_covariate_inputs
        if mutation == "selection":
            monkeypatch.setattr(
                targeting, "clever_covariate_inputs", lambda *args: (inputs(*args)[0], None)
            )
        elif mutation == "density":
            monkeypatch.setattr(
                targeting, "clever_covariate_inputs", lambda *args: (None, inputs(*args)[1])
            )
        else:
            monkeypatch.setattr(
                NuisanceEstimates,
                "bounded_missingness",
                lambda self, lower: np.ones_like(self.missingness),
            )
        fit = effect(level).estimate(**oracle(outcome=wrong_outcome()))
        moved = beta(fit) - law.truth(level)
        np.testing.assert_allclose(moved, MUTANT_DISPLACEMENT[mutation][level], atol=5e-5)
        assert np.max(np.abs(moved)) >= FLOOR


# ---------------------------------------------------------------- the study sites


class TestTheStudySites:
    def test_the_target_reads_the_unwrapped_contrast_and_the_level(self) -> None:
        identified = effect(0)
        functional = identified.functional
        assert functional.target == "cate_projection"
        assert functional.intermediate == 0.0
        assert functional.effect_modification is not None
        assert functional.effect_modification.scale == "difference"
        assumptions = identified.identification.assumptions
        assert any("intermediate positivity at Z = 0" in item for item in assumptions)
        assert any("no unmeasured intermediate confounding" in item for item in assumptions)
        assert any("response positivity for Delta" in item for item in assumptions)

    def test_available_methods_list_tmle_and_refuse_ep_with_r_ep_int(self) -> None:
        methods = {m.name: m for m in effect(1).available_methods()}
        assert methods["tmle"].available
        assert not methods["ep"].available
        assert methods["ep"].refusal == EP_INTERMEDIATE_REFUSAL

    @pytest.mark.parametrize("scale", SCALES)
    def test_ep_refuses_before_any_learner(self, scale: str) -> None:
        with pytest.raises(CapabilityError, match="intermediate mechanism") as caught:
            effect(0, scale).estimate("ep", **never())
        refusal = caught.value.refusal
        assert refusal.condition.endswith(EP_INTERMEDIATE_REFUSAL.condition)
        assert (refusal.reason, refusal.roadmap) == (EP_INTERMEDIATE_REFUSAL.reason, "P1")
        assert NeverFit.calls == 0

    def test_the_ep_engine_refuses_an_intermediate(self) -> None:
        from cleverly.effect_modification import effect_modification_ep_refusal

        study = CausalStudy(law.frame(), design=design())
        assert effect_modification_ep_refusal(study.data) == EP_INTERMEDIATE_REFUSAL

    @pytest.mark.parametrize("tool", ["assess_score", "targeting_curve"])
    def test_the_score_tools_refuse_before_any_learner(self, tool: str) -> None:
        frame = law.frame()
        frame["S"] = frame["W"]
        identified = effect(0, frame=frame)
        with pytest.raises(CapabilityError) as caught:
            getattr(identified, tool)("S", computed_from=["W"], score_kind="external", **never())
        assert caught.value.refusal == CDE_CURVE_TOOLS_REFUSAL
        assert NeverFit.calls == 0

    @pytest.mark.parametrize("scale", SCALES)
    def test_the_summary_says_each_row_is_the_controlled_effect_at_its_level(
        self, scale: str
    ) -> None:
        from cleverly.estimators._point_summary import _meaning

        for level in LEVELS:
            fit = exact_fit(level, scale)
            for name in fit.estimates:
                meaning = _meaning(fit, name)
                assert f"Y(t, {level})" in meaning and f"Y(r, {level})" in meaning, meaning
                assert f"with Z held at {level}" in meaning
                assert "E[Y^t" not in meaning
            text = " ".join(str(fit.summary()).split())
            assert "E[Y^t - Y^r | V]" not in text

    def test_the_curve_carries_the_level_and_refuses_its_held_out_checks(self) -> None:
        curve = exact_fit(1, "difference").effect_curve()
        assert curve.target_sentence.endswith("with the intermediate Z held at 1")
        np.testing.assert_allclose(
            curve.predict({"W": np.array([0.0, 2.0])}),
            law.truth(1) @ np.array([[1.0, 1.0], [0.0, 2.0]]),
            atol=EXACT,
        )
        with pytest.raises(CapabilityError) as caught:
            curve.diagnostics  # noqa: B018 - the property access is the call
        assert caught.value.refusal == CDE_CURVE_TOOLS_REFUSAL

    def test_a_bare_conditional_contrast_on_an_intermediate_design_still_refuses(self) -> None:
        study = CausalStudy(law.frame(), design=design())
        with pytest.raises(CapabilityError, match="must identify ControlledDirectEffect"):
            study.identify(contrast())

    def test_a_modifier_equal_to_the_intermediate_column_refuses(self) -> None:
        study = CausalStudy(law.frame(), design=design())
        modifiers = Modifiers(["Z"], levels={"Z": (0.0, 1.0)})
        from cleverly.bases import Indicators

        with pytest.raises(CapabilityError, match="not adjustment columns"):
            study.identify(
                ControlledDirectEffect(
                    intermediate=0.0,
                    contrast=ConditionalContrast(modifiers, model=Indicators("Z")),
                )
            )

    def test_a_string_contrast_still_names_the_typed_ones(self) -> None:
        with pytest.raises(DataError, match="ConditionalContrast"):
            ControlledDirectEffect(intermediate=0.0, contrast="cate")  # type: ignore[arg-type]


# ---------------------------------------------------------------- the compositions


def saturated(**extra: Any) -> dict[str, Any]:
    """Saturated learners: on a sample that realises its law exactly they are the oracle."""
    return {
        "outcome_learner": CellMeans(),
        "treatment_learner": SaturatedCategorical(),
        "missingness_learner": SaturatedCategorical(),
        "intermediate_learner": SaturatedCategorical(),
        "cross_fit": False,
        "random_state": 0,
        **extra,
    }


def empirical_truth(frame: pd.DataFrame, level: int, pair: tuple[int, int]) -> np.ndarray:
    """``beta_z`` from the empirical cell means of ``frame``, weighted by ``n`` if present."""
    weights = frame["n"].to_numpy() if "n" in frame else np.ones(len(frame))
    observed = frame["Delta"].to_numpy() == 1.0
    p_w = np.array([weights[frame["W"] == w].sum() for w in range(3)]) / weights.sum()
    q = np.empty((3, 2))
    for w in range(3):
        for column, arm in enumerate(pair):
            cell = (
                observed
                & (frame["W"].to_numpy() == w)
                & (frame["A"].to_numpy() == arm)
                & (frame["Z"].to_numpy() == level)
            )
            q[w, column] = np.sum(weights[cell] * frame["Y"].to_numpy()[cell]) / weights[cell].sum()
    return law.projection(p_w, q[:, 0] - q[:, 1])


class TestTheCompositions:
    @quiet
    def test_a_pooled_and_a_per_arm_outcome_regression_both_reach_the_truth(self) -> None:
        """The oracle reads the pooled design; the saturated learner is the per-arm oracle."""
        pooled = effect(0).estimate(**oracle(), outcome_by_arm=False)
        per_arm = effect(0).estimate(**saturated(outcome_by_arm=True))
        for fit in (pooled, per_arm):
            np.testing.assert_allclose(beta(fit), law.truth(0), rtol=0, atol=EXACT)

    @quiet
    def test_row_weights_reach_the_weighted_truth(self) -> None:
        """Weights that move with ``Y`` and ``A``, so the weighted law differs from the oracle's;
        saturated learners are the weighted law's oracle."""
        frame = law.frame()
        frame["n"] = 1.0 + 0.5 * frame["Y"].fillna(0.0) + 0.75 * frame["A"] + 0.25 * frame["Z"]
        fit = effect(1, frame=frame, weights="n").estimate(**saturated())
        truth = empirical_truth(frame, 1, (1, 0))
        np.testing.assert_allclose(beta(fit), truth, rtol=0, atol=EXACT)
        # Witness: the unweighted truth is another number.
        assert np.max(np.abs(truth - law.truth(1))) > FLOOR

    @quiet
    def test_clusters_sum_the_curve_within_each_cluster(self) -> None:
        frame = law.frame()
        frame["C"] = np.arange(len(frame)) // 4
        fit = effect(0, frame=frame, cluster="C").estimate(**oracle())
        np.testing.assert_allclose(beta(fit), law.truth(0), rtol=0, atol=EXACT)
        curves = np.column_stack([np.asarray(e.influence_curve) for e in fit.estimates.values()])
        n, clusters = len(frame), len(frame) // 4
        sums = np.zeros((clusters, 2))
        np.add.at(sums, frame["C"].to_numpy(), curves)
        expected = clusters * np.cov(sums, rowvar=False, ddof=1) / n**2
        variances = np.array([e.std_error**2 for e in fit.estimates.values()])
        np.testing.assert_allclose(variances, np.diag(expected), rtol=1e-10)
        row_level = np.cov(curves, rowvar=False, ddof=1).diagonal() / n
        assert np.max(np.abs(variances / row_level - 1.0)) > 0.01

    @quiet
    def test_a_pair_of_a_three_arm_treatment(self) -> None:
        """A third arm (a copy of arm 0's rows with arm 1's outcomes swapped in) that the pair
        (1, 0) does not read; saturated learners make the fit exact on the empirical law."""
        frame = law.frame()
        extra = frame[frame["A"] == 0.0].copy()
        extra["A"] = 2.0
        extra["Y"] = np.where(extra["Delta"] == 1.0, 1.0 - extra["Y"], np.nan)
        three = pd.concat([frame, extra], ignore_index=True)
        study = CausalStudy(three, design=design())
        identified = study.identify(
            ControlledDirectEffect(
                intermediate=0.0,
                contrast=ConditionalContrast(
                    MODIFIERS,
                    model=Linear(["W"]),  # type: ignore[arg-type]
                    treatment=1.0,
                    reference=0.0,
                ),
            )
        )
        fit = identified.estimate(**saturated())
        truth = empirical_truth(three, 0, (1, 0))
        np.testing.assert_allclose(beta(fit), truth, rtol=0, atol=EXACT)
        assert np.max(np.abs(truth - law.truth(0))) > FLOOR

    @quiet
    def test_repeats_report_the_median_rule_over_the_per_repeat_fits(self) -> None:
        frame = law.frame(complete=True)
        identified = effect(0, frame=frame, missing=False)
        learners = {
            "outcome_learner": LogisticRegression(max_iter=2000),
            "treatment_learner": LogisticRegression(max_iter=2000),
            "intermediate_learner": LogisticRegression(max_iter=2000),
        }
        fit = identified.estimate(**learners, cross_fit=True, n_folds=3, repeats=3, random_state=4)
        draws = [repeat.projection for repeat in fit.repeats]
        assert len(draws) == 3
        for key, estimate in fit.estimates.items():
            values = [draw[key].psi for draw in draws]
            assert len({round(value, 12) for value in values}) == 3
            assert estimate.psi == pytest.approx(float(np.median(values)), abs=1e-14)
            expected = float(np.median([draw[key].std_error for draw in draws]))
            assert fit.se_without_displacement[key] == pytest.approx(expected, rel=1e-12)
        assert fit.effect_curve().model.repeated
        assert all(e.support.label == "experimental" for e in fit.estimates.values())

    @quiet
    def test_a_full_refit_bootstrap_reads_the_bootstrap_support_rule(self) -> None:
        fit = effect(0).estimate(**oracle(), n_bootstrap=4)
        for estimate in fit.estimates.values():
            assert len(estimate.bootstrap.draws) == 4
            assert estimate.bootstrap.support.label == "experimental"

    @pytest.mark.parametrize("scale", SCALES)
    @pytest.mark.parametrize("level", LEVELS)
    @quiet
    def test_the_cross_fitted_stacked_fit_is_exact_and_its_curve_the_closed_form(
        self, level: int, scale: str
    ) -> None:
        """The composition ``projection-evidence`` measures: stacked CV-TMLE, complete outcomes.

        The oracle learners ignore their training rows, so every fold's nuisances are the
        law's and the stacked fit reproduces the projection exactly; at the wrong outcome
        regression the pooled fluctuation still recovers it.
        """
        dgp = law.law(complete=True)
        truth = law.truth(level, scale, law.complete_probs())
        settings = {"cross_fit": True, "n_folds": 4}
        fit = effect(level, scale, missing=False).estimate(**{**oracle(dgp), **settings})
        np.testing.assert_allclose(beta(fit), truth, rtol=0, atol=EXACT)
        curves = np.column_stack([e.influence_curve for e in fit.estimates.values()])
        expected = closed_form_curve(
            level, scale, beta(fit), frame=law.frame(complete=True), dgp=dgp
        )
        np.testing.assert_allclose(curves, expected, rtol=0, atol=1e-9)
        wrong = effect(level, scale, missing=False).estimate(
            **{**oracle(dgp, outcome=wrong_complete_outcome()), **settings}
        )
        np.testing.assert_allclose(beta(wrong), truth, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("scale", SCALES)
    @pytest.mark.parametrize("level", LEVELS)
    @quiet
    def test_fold_wise_targeting_is_exact_when_every_fold_is_the_law(
        self, level: int, scale: str
    ) -> None:
        """``targeting_scheme="fold"``: one fluctuation per fold, read on that fold's rows.

        Each fold's epsilon solves its own score equation, so on a random split of the law's
        rows a fold's cell imbalance moves it. Six whole copies of the complete law, grouped as
        clusters into three folds, make every fold the law itself: the fold-wise fit is then
        exact, at the oracle and at the wrong outcome regression, where each fold's epsilon is
        not zero. It reads ``experimental``: no study measures it.
        """
        copies = pd.concat(
            [law.frame(complete=True).assign(C=c) for c in range(6)], ignore_index=True
        )
        dgp = law.law(complete=True)
        truth = law.truth(level, scale, law.complete_probs())
        settings = {"cross_fit": True, "n_folds": 3, "targeting_scheme": "fold"}
        for outcome in (None, wrong_complete_outcome()):
            fit = effect(level, scale, frame=copies, missing=False, cluster="C").estimate(
                **{**oracle(dgp, outcome=outcome), **settings}
            )
            np.testing.assert_allclose(beta(fit), truth, rtol=0, atol=1e-9)
            assert all(e.support.label == "experimental" for e in fit.estimates.values())
        # Witness: the wrong regression's plug-in is off by more than the floor.
        bad = wrong_complete_outcome().q[:, :, level]
        assert (
            np.max(
                np.abs(
                    law.projection(law.complete_probs().sum(axis=(1, 2, 3)), bad[:, 1] - bad[:, 0])
                    - law.truth(level, "difference", law.complete_probs())
                )
            )
            > 0.05
        )

    @pytest.mark.parametrize("scale", SCALES)
    @quiet
    def test_a_declared_treatment_law_is_exact(self, scale: str) -> None:
        """``treatment_probabilities=`` replaces the treatment learner by the declared law."""
        frame = law.frame(complete=True)
        dgp = law.law(complete=True)
        frame["g1"] = dgp.g[frame["W"].to_numpy().astype(int)]
        frame["g0"] = 1.0 - frame["g1"]
        declared = {"treatment_probabilities": {1.0: "g1", 0.0: "g0"}, "mechanism_law": "sample"}
        for level in LEVELS:
            study = CausalStudy(frame, design=design(missing=False, **declared))
            identified = study.identify(
                ControlledDirectEffect(intermediate=float(level), contrast=contrast(scale))
            )
            learners = oracle(dgp, outcome=wrong_complete_outcome())
            learners["treatment_learner"] = NeverFit()
            fit = identified.estimate(**learners)
            truth = law.truth(level, scale, law.complete_probs())
            np.testing.assert_allclose(beta(fit), truth, rtol=0, atol=1e-9)
            assert all(e.support.label == "experimental" for e in fit.estimates.values())

    @pytest.mark.parametrize("scale", SCALES)
    @quiet
    def test_a_randomized_design_is_exact(self, scale: str) -> None:
        """``randomized=True`` with the treatment learner at the law's mechanism."""
        dgp = law.law(complete=True)
        for level in LEVELS:
            study = CausalStudy(
                law.frame(complete=True), design=design(missing=False, randomized=True)
            )
            identified = study.identify(
                ControlledDirectEffect(intermediate=float(level), contrast=contrast(scale))
            )
            fit = identified.estimate(**oracle(dgp, outcome=wrong_complete_outcome()))
            truth = law.truth(level, scale, law.complete_probs())
            np.testing.assert_allclose(beta(fit), truth, rtol=0, atol=1e-9)
            assert all(e.support.label == "experimental" for e in fit.estimates.values())

    @quiet
    def test_the_cross_fitted_stacked_fit_runs_and_reads_experimental_until_registration(
        self,
    ) -> None:
        identified = effect(1, "log_ratio", missing=False)
        fit = identified.estimate(
            outcome_learner=LogisticRegression(max_iter=2000),
            treatment_learner=LogisticRegression(max_iter=2000),
            intermediate_learner=LogisticRegression(max_iter=2000),
            cross_fit=True,
            n_folds=3,
            outcome_by_arm=True,
            random_state=0,
        )
        assert np.all(np.isfinite(beta(fit)))
        assert all(e.support.label == "experimental" for e in fit.estimates.values())


# ---------------------------------------------------------------- refusals


class TestTheRefusals:
    def test_cross_fitting_with_missing_outcomes_refuses_with_x53(self) -> None:
        with pytest.raises(CapabilityError, match="cross-fitting with missing outcomes"):
            effect(0).estimate(**never(), cross_fit=True)
        assert NeverFit.calls == 0

    def test_strata_refuse_before_any_learner(self) -> None:
        frame = law.frame()
        frame["S"] = (frame["W"] > 0).astype(float)
        study = CausalStudy(frame, design=design(strata="S"))
        with pytest.raises(CapabilityError) as caught:
            study.identify(ControlledDirectEffect(intermediate=0.0, contrast=contrast()))
        assert caught.value.refusal == INTERMEDIATE_STRATA_REFUSAL

    def test_fold_evaluated_strata_meet_the_same_refusal_first(self) -> None:
        """X28 refuses an intermediate beside fold-evaluated strata on the engine route; on a
        conditional contrast R-int-strata refuses earlier, at identification."""
        from cleverly.effect_modification import EffectModification
        from cleverly.estimators import TMLE

        frame = law.frame(complete=True)
        frame["S"] = (frame["W"] > 0).astype(float)
        estimator = TMLE(
            effect_modification=EffectModification(Linear(["W"]), MODIFIERS, "difference"),
            **never(),
            cv_evaluation=True,
        )
        with pytest.raises(CapabilityError) as caught:
            estimator.fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W", "S"],
                intermediate="Z",
                strata="S",
            )
        assert caught.value.refusal == INTERMEDIATE_STRATA_REFUSAL
        assert NeverFit.calls == 0

    def test_a_missing_treatment_refuses_with_f33(self) -> None:
        frame = law.frame()
        frame["R"] = 1.0
        frame.loc[::7, "R"] = 0.0
        frame.loc[frame["R"] == 0.0, "A"] = np.nan
        with pytest.raises(CapabilityError, match="controlled direct effect") as caught:
            effect(0, frame=frame, treatment_missingness="R").estimate(**never())
        assert caught.value.refusal.roadmap == "F33"
        assert NeverFit.calls == 0

    def test_a_level_outside_zero_and_one_refuses_with_x63(self) -> None:
        study = CausalStudy(law.frame(), design=design())
        with pytest.raises(DataError, match="fixed at 0 or 1"):
            study.identify(ControlledDirectEffect(intermediate=0.5, contrast=contrast()))
