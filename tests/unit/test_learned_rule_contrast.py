"""The learned rule beside known regimes, and their contrasts (X11 (c)).

``TMLE(learned_rule=..., interventions=[...])`` adds each known regime as one more column of the
learned-rule fit's regime set, on the same folds, so one fluctuation targets every column.  The
fit reports the learned-rule value ``ey_learned_rule``, each known regime's value
``ey_known_regime`` and each contrast ``ate_learned_rule``: the learned rule minus the known regime,
or the known regime minus the learned rule when ``reference=`` names the rule.  The contrast's curve
is the difference of the two curves on the same folds, which is the package's joint step: each
estimator is asymptotically linear, so the stack is jointly normal.

The witnesses are the plan's (``X11-wave2-plan.md`` rev 2.1, section 4):

- W-c1 (identity): the contrast point is the learned point minus the known point, and the contrast
  variance is :math:`V^{-2}\\sum_v n_v^{-2}\\sum_{i\\in v}(D_{L,i}-D_{K,i})^2`, with both curves
  rebuilt longhand from the targeted fit.  Under the stacked report it is the centred rule of the
  difference of the two curves.
- W-c2 (reduction): with a stub outcome regression whose blip has the sign of ``W1``, every fold
  rule is ``1{W1 > 0}``, and every reported value, contrast, curve and bound equals the shipped
  regime fit of ``Rule(1{W1 > 0})`` and ``Static(1)`` on the same folds, bit for bit.
- W-c3 (aggregation): ``repeats=4`` reports the mean of four single-split fits for every name.  The
  same known regime fitted alone reports the median, and the two differ on the witness draw.
- W-c4 (covariance, nonzero): the joint standard error differs from :math:`|se_L-se_K|` and from
  :math:`\\sqrt{se_L^2+se_K^2}` by declared factors, and the mutation that misaligns the two
  curves by one row changes it.
- W-c5 (scores): each column's fold-average score is solved by the joint fluctuation.  The
  mutation that targets the learned column alone leaves the known regime's score unsolved.

The label: a qualifier ``known_regimes`` keeps every learned-rule row from covering a contrast fit,
so each such fit reads experimental until a study registers.
"""

from __future__ import annotations

from functools import cache
from typing import Any, ClassVar

import numpy as np
import pytest

from cleverly._inference_status import NO_KNOWN_REGIME_BANDS
from cleverly._summary import vocabulary
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.inference import influence_variance, simultaneous_bands
from cleverly.inference.multiplier import KNOWN_REGIME_BAND_REFUSAL
from cleverly.interventions import LearnedRule, Rule, Static
from cleverly.interventions.learned import freeze_learned_rule, frozen_rule_replay
from cleverly.sensitivity import omitted_variable_bounds
from cleverly.validation import refute
from cleverly.validation.refute import _placebo_null
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit.test_learned_rule_in_sample import SignStub
from tests.unit.test_learned_rule_strata import FIT as STRATA_FIT
from tests.unit.test_learned_rule_strata import _strata_frame

#: The reported names of a fit with the default rule beside ``Static(1)``.
LEARNED = support.NAME
KNOWN = "ey_known_regime[always 1]"
CONTRAST = "ate_learned_rule[learned rule vs always 1]"
FLIPPED = "ate_learned_rule[always 1 vs learned rule]"

FIT: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}

#: The declared factors of W-c4, each below what the fixed draw gives.
W_C4_PERFECT = 5.0  # observed se_C / |se_L - se_K| = 11.7
W_C4_INDEPENDENT = 0.7  # observed se_C / sqrt(se_L^2 + se_K^2) = 0.44
W_C4_ROLL = 1.5  # observed 2.28 for the curves misaligned by one row
#: The declared gap of W-c3 between the median and the mean of four split values.
W_C3_GAP = 1e-3  # observed 0.0021 for the known regime
#: The declared floor of W-c5 for an unsolved score.
W_C5_UNSOLVED = 1e-2  # observed 0.050 under the learned-column-only fluctuation


@cache
def _planted(cv_evaluation: bool = True) -> Any:
    """A misspecified outcome regression at unequal folds (n = 203, V = 5), beside ``Static(1)``."""
    return support.fit(
        support.law_frame(203, 1),
        outcome_learner=support.misspecified_learner(),
        interventions=[Static(1)],
        cv_evaluation=cv_evaluation,
    )


def _known_pieces(result: Any) -> tuple[np.ndarray, np.ndarray]:
    """The clever covariate of ``Static(1)`` at the observed arm, and ``Qbar*(1, W)``."""
    nuisance = result.nuisance
    lower, upper = result.config.g_bounds
    g1 = np.clip(np.asarray(nuisance.propensity.arm(1.0), dtype=float), lower, upper)
    treatment = np.asarray(result.data.treatment, dtype=float)
    h = (treatment == 1.0).astype(float) / g1
    q = np.asarray(result.fluctuations["regime"].targeted.arms[1.0], dtype=float)
    return h, q


def _fold_curve(
    h: np.ndarray, rule: np.ndarray, pieces: dict[str, Any], tests: list[np.ndarray]
) -> np.ndarray:
    """``D_i = H_i (Y_i - Qbar*_i) + Qbar*(d, W_i) - psi_v``, centred at the fold's plug-in."""
    curve = np.empty(h.size)
    residual = h * (pieces["y"] - pieces["q_star_observed"])
    for test in tests:
        curve[test] = residual[test] + rule[test] - np.mean(rule[test])
    return curve


def _cross_validated(curve: np.ndarray, tests: list[np.ndarray]) -> float:
    """``V^-2 sum_v n_v^-2 sum_{i in v} D_i^2``."""
    return float(
        np.sum([np.sum(curve[test] ** 2) / test.size**2 for test in tests]) / len(tests) ** 2
    )


@cache
def _curves() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """W-c1's longhand curves ``D_L`` and ``D_K`` of :func:`_planted`, and its folds."""
    result = _planted()
    pieces = support.validation_pieces(result)
    tests = pieces["tests"]
    h, q = _known_pieces(result)
    learned = _fold_curve(pieces["h"], pieces["q_star_rule"], pieces, tests)
    known = _fold_curve(h, q, pieces, tests)
    return learned, known, tests


class TestTheFixtureHasTeeth:
    def test_the_folds_are_unequal_and_the_fluctuation_moves(self) -> None:
        result = _planted()
        sizes = {int(np.asarray(test).size) for _, test in result.nuisance.folds}
        assert len(sizes) > 1
        assert float(np.min(np.abs(result.fluctuations["regime"].epsilon))) > 1e-2

    def test_the_rule_and_the_known_regime_disagree_on_some_rows(self) -> None:
        values = _planted().nuisance.regimes.values
        assert 0.0 < float(np.mean(values[:, 1, 0])) < 1.0
        np.testing.assert_array_equal(values[:, 1, 1], np.ones(values.shape[0]))

    def test_the_regime_set_holds_the_rule_then_the_known_regime(self) -> None:
        regimes = _planted().nuisance.regimes
        assert regimes.names == ("learned rule", "always 1")
        assert regimes.reference == 1.0
        assert regimes.fold_rules.shape == (203, 5)


# ------------------------------------------------------------------------------- W-c1


class TestWc1Identity:
    def test_the_point_is_the_difference_of_the_two_values(self) -> None:
        estimates = _planted().estimates
        assert estimates[CONTRAST].psi == pytest.approx(
            estimates[LEARNED].psi - estimates[KNOWN].psi, rel=1e-12, abs=1e-15
        )

    def test_the_variance_is_the_cross_validated_form_of_the_difference(self) -> None:
        learned, known, tests = _curves()
        estimates = _planted().estimates
        assert estimates[CONTRAST].variance == pytest.approx(
            _cross_validated(learned - known, tests), rel=1e-12
        )
        assert estimates[LEARNED].variance == pytest.approx(
            _cross_validated(learned, tests), rel=1e-12
        )
        assert estimates[KNOWN].variance == pytest.approx(_cross_validated(known, tests), rel=1e-12)

    def test_the_stored_curve_is_the_rescaled_difference(self) -> None:
        learned, known, tests = _curves()
        stored = np.asarray(_planted().estimates[CONTRAST].influence_curve, dtype=float).copy()
        for test in tests:
            stored[test] *= len(tests) * test.size / stored.size
        np.testing.assert_allclose(stored, learned - known, rtol=0, atol=1e-12)

    def test_the_stacked_variance_is_the_centred_rule_of_the_difference(self) -> None:
        estimates = _planted(cv_evaluation=False).estimates
        difference = np.asarray(estimates[LEARNED].influence_curve) - np.asarray(
            estimates[KNOWN].influence_curve
        )
        np.testing.assert_allclose(
            estimates[CONTRAST].influence_curve, difference, rtol=0, atol=1e-14
        )
        assert estimates[CONTRAST].variance == pytest.approx(
            influence_variance(difference), rel=1e-12
        )

    def test_reference_naming_the_rule_flips_the_sign_only(self) -> None:
        flipped = support.fit(
            support.law_frame(203, 1),
            outcome_learner=support.misspecified_learner(),
            interventions=[Static(1)],
            reference="learned rule",
        )
        base = _planted().estimates[CONTRAST]
        estimate = flipped.estimates[FLIPPED]
        assert CONTRAST not in flipped.estimates
        assert estimate.psi == pytest.approx(-base.psi, rel=1e-12)
        assert estimate.variance == pytest.approx(base.variance, rel=1e-12)
        np.testing.assert_allclose(
            estimate.influence_curve, -np.asarray(base.influence_curve), rtol=0, atol=1e-14
        )

    def test_every_known_regime_gets_its_own_contrast_with_the_rule(self) -> None:
        result = support.fit(interventions=[Static(1), Static(0)])
        assert list(result.estimates) == [
            LEARNED,
            KNOWN,
            "ey_known_regime[always 0]",
            CONTRAST,
            "ate_learned_rule[learned rule vs always 0]",
        ]


# ------------------------------------------------------------------------------- W-c2


def _w1_rule() -> Rule:
    return Rule(lambda w: (w["W1"] > 0).astype(float), name="w1", rule_kind="known")


@cache
def _reduction(cv_evaluation: bool) -> tuple[Any, Any]:
    """The stub learned-rule contrast fit and the shipped regime contrast fit, on one draw."""
    frame = support.law_frame(400, 0)
    common: dict[str, Any] = {
        "outcome_learner": SignStub(),
        "treatment_learner": support.treatment_learner(),
        "cross_fit": True,
        "cv_evaluation": cv_evaluation,
        "n_folds": 5,
        "simultaneous": False,
        "random_state": 0,
    }
    learned = (
        TMLE(learned_rule=LearnedRule(), interventions=[Static(1)], **common)
        .fit(frame, **FIT)
        .single()
    )
    regime = (
        TMLE(interventions=[_w1_rule(), Static(1)], reference="always 1", **common)
        .fit(frame, **FIT)
        .single()
    )
    return learned, regime


#: Each reported name of the learned-rule contrast fit, and its name in the regime fit.
REDUCTION = {
    LEARNED: "ey_regime[w1]",
    KNOWN: "ey_regime[always 1]",
    CONTRAST: "ate_regime[w1 vs always 1]",
}


@pytest.mark.parametrize("cv_evaluation", [True, False], ids=["fold_evaluated", "stacked"])
class TestWc2Reduction:
    def test_the_folds_and_the_rule_are_the_regime_fit_s(self, cv_evaluation: bool) -> None:
        learned, regime = _reduction(cv_evaluation)
        for (_, a), (_, b) in zip(learned.nuisance.folds, regime.nuisance.folds, strict=True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            learned.nuisance.regimes.values, regime.nuisance.regimes.values
        )

    def test_the_fluctuation_moves(self, cv_evaluation: bool) -> None:
        """The nonzero witness: the comparison runs through a nonzero epsilon."""
        learned, _ = _reduction(cv_evaluation)
        assert float(np.max(np.abs(learned.fluctuations["regime"].epsilon))) > 1e-2

    @pytest.mark.parametrize("name", list(REDUCTION))
    def test_each_estimate_equals_the_regime_fit_s(self, cv_evaluation: bool, name: str) -> None:
        learned, regime = _reduction(cv_evaluation)
        a, b = learned.estimates[name], regime.estimates[REDUCTION[name]]
        assert a.psi == pytest.approx(b.psi, rel=1e-12, abs=1e-15)
        assert a.variance == pytest.approx(b.variance, rel=1e-12)
        np.testing.assert_allclose(a.influence_curve, b.influence_curve, rtol=0, atol=1e-13)


class TestWc2Bound:
    """The bound of each value runs, and on the stub it equals the regime fit's bound."""

    STRENGTHS: ClassVar[dict[str, Any]] = {
        "cf_y": 0.05,
        "cf_d": 0.05,
        "nu2_estimator": "doubly_robust",
    }

    def test_the_known_regime_bound_equals_the_regime_bound_with_limits(self) -> None:
        learned, regime = _reduction(True)
        a = omitted_variable_bounds(learned, KNOWN, **self.STRENGTHS)
        b = omitted_variable_bounds(regime, REDUCTION[KNOWN], **self.STRENGTHS)
        for field in ("max_bias", "lower", "upper", "ci_lower", "ci_upper"):
            assert getattr(a, field) == pytest.approx(getattr(b, field), rel=1e-10), field
        assert a.max_bias > 0.0

    def test_the_learned_rule_bound_equals_the_rule_s_point_bound(self) -> None:
        learned, regime = _reduction(True)
        a = omitted_variable_bounds(learned, LEARNED, **self.STRENGTHS)
        b = omitted_variable_bounds(regime, REDUCTION[LEARNED], **self.STRENGTHS)
        for field in ("max_bias", "lower", "upper"):
            assert getattr(a, field) == pytest.approx(getattr(b, field), rel=1e-10), field
        assert a.limit_refusal is not None

    def test_the_bound_of_a_contrast_refuses_by_name(self) -> None:
        learned, _ = _reduction(True)
        with pytest.raises(CapabilityError, match=r"X11 \(n2\)") as caught:
            omitted_variable_bounds(learned, CONTRAST, **self.STRENGTHS)
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        assert "ey_known_regime" in str(caught.value)


# ------------------------------------------------------------------------------- W-c3


@cache
def _repeated() -> tuple[Any, list[Any]]:
    """A contrast fit of four splits, and the single-split fits at its split seeds."""
    repeated = support.fit(interventions=[Static(1)], repeats=4)
    singles = [
        support.fit(interventions=[Static(1)], random_state=repeat.seed)
        for repeat in repeated.repeats
    ]
    return repeated, singles


class TestWc3Aggregation:
    @pytest.mark.parametrize("name", [LEARNED, KNOWN, CONTRAST])
    def test_each_name_is_the_mean_of_the_single_split_fits(self, name: str) -> None:
        repeated, singles = _repeated()
        values = [single.estimates[name].psi for single in singles]
        assert repeated.estimates[name].psi == pytest.approx(np.mean(values), rel=1e-12)

    def test_the_contrast_curve_is_the_difference_of_the_mean_curves(self) -> None:
        estimates = _repeated()[0].estimates
        difference = np.asarray(estimates[LEARNED].influence_curve) - np.asarray(
            estimates[KNOWN].influence_curve
        )
        np.testing.assert_allclose(
            estimates[CONTRAST].influence_curve, difference, rtol=0, atol=1e-14
        )

    def test_the_known_regime_fitted_alone_reports_the_median(self) -> None:
        """The contract's sentence: the mean beside a learned rule, the median alone."""
        alone = support.learned_tmle(learned_rule=None, interventions=[Static(1)], repeats=4)
        result = alone.fit(support.law_frame(), **FIT).single()
        singles = [
            support.learned_tmle(learned_rule=None, interventions=[Static(1)], random_state=seed)
            .fit(support.law_frame(), **FIT)
            .single()
            .estimates["ey_regime[always 1]"]
            .psi
            for seed in (repeat.seed for repeat in result.repeats)
        ]
        assert result.estimates["ey_regime[always 1]"].psi == pytest.approx(
            np.median(singles), rel=1e-12
        )
        repeated, split_fits = _repeated()
        values = [single.estimates[KNOWN].psi for single in split_fits]
        assert abs(float(np.median(values)) - repeated.estimates[KNOWN].psi) > W_C3_GAP

    def test_clusters_with_repeats_keep_row_9(self) -> None:
        frame = support.law_frame(120, 3).assign(c=np.arange(120) % 20)
        estimator = support.learned_tmle(
            interventions=[Static(1)], repeats=2, **never_fit_learners()
        )
        NeverFit.calls = 0
        with pytest.raises(CapabilityError, match=r"X11 \(i2\)"):
            estimator.fit(frame, id="c", **FIT)
        assert NeverFit.calls == 0


# ------------------------------------------------------------------------------- W-c4


class TestWc4Covariance:
    def _se(self) -> tuple[float, float, float]:
        estimates = _planted().estimates
        se_l, se_k, se_c = (
            float(np.sqrt(estimates[n].variance)) for n in (LEARNED, KNOWN, CONTRAST)
        )
        return se_l, se_k, se_c

    def test_the_joint_se_is_not_the_perfect_correlation_se(self) -> None:
        se_l, se_k, se_c = self._se()
        assert se_c > W_C4_PERFECT * abs(se_l - se_k)

    def test_the_joint_se_is_not_the_independent_se(self) -> None:
        se_l, se_k, se_c = self._se()
        assert se_c < W_C4_INDEPENDENT * float(np.hypot(se_l, se_k))

    def test_control_misaligned_rows_change_the_se(self) -> None:
        """A power check that the joint SE sees the pairing of rows, not a library mutation.

        W-c1 is the witness for pairing: it rebuilds the contrast variance from the two
        curves at rel 1e-12, so a known column taken from another split, which moves the SE
        by about 2%, fails it already.
        """
        learned, known, tests = _curves()
        se_c = self._se()[2]
        rolled = float(np.sqrt(_cross_validated(learned - np.roll(known, 1), tests)))
        assert rolled > W_C4_ROLL * se_c


# ------------------------------------------------------------------------------- W-c5


def _fold_score(h: np.ndarray, residual: np.ndarray, tests: list[np.ndarray]) -> float:
    """``(1/V) sum_v (1/n_v) sum_{i in v} H_i (Y_i - Qbar_i)``."""
    return float(np.mean([np.mean((h * residual)[test]) for test in tests]))


class TestWc5Scores:
    def test_each_column_s_fold_average_score_is_solved(self) -> None:
        result = _planted()
        pieces = support.validation_pieces(result)
        h_known, _ = _known_pieces(result)
        residual = pieces["y"] - pieces["q_star_observed"]
        for h in (pieces["h"], h_known):
            assert abs(_fold_score(h, residual, pieces["tests"])) < 1e-10

    def test_control_the_initial_fit_leaves_both_scores_unsolved(self) -> None:
        result = _planted()
        pieces = support.validation_pieces(result)
        h_known, _ = _known_pieces(result)
        residual = pieces["y"] - np.asarray(result.nuisance.outcome.observed, dtype=float)
        for h in (pieces["h"], h_known):
            assert abs(_fold_score(h, residual, pieces["tests"])) > W_C5_UNSOLVED

    def test_control_the_learned_column_alone_leaves_the_known_score_unsolved(self) -> None:
        """Targeting the learned rule alone, as a fit without interventions= does."""
        result = _planted()
        alone = support.fit(
            support.law_frame(203, 1), outcome_learner=support.misspecified_learner()
        )
        pieces = support.validation_pieces(result)
        h_known, _ = _known_pieces(result)
        residual = pieces["y"] - np.asarray(alone.fluctuations["regime"].targeted.observed)
        assert abs(_fold_score(h_known, residual, pieces["tests"])) > W_C5_UNSOLVED


# ------------------------------------------------------------------------------- label


class TestTheLabel:
    @pytest.mark.parametrize(
        "settings",
        [{}, {"cv_evaluation": False}],
        ids=["fold_evaluated", "stacked"],
    )
    def test_a_contrast_fit_keys_known_regimes_and_reads_experimental(
        self, settings: dict[str, Any]
    ) -> None:
        alone = support.fit(**settings)
        contrast = support.fit(interventions=[Static(1)], **settings)
        assert alone.estimates[LEARNED].support.label == "supported"
        key = contrast.config.support_key
        assert "known_regimes" in key.qualifiers
        assert "known_regimes" not in alone.config.support_key.qualifiers
        assert key.family == alone.config.support_key.family
        for estimate in contrast.estimates.values():
            assert estimate.support.label == "experimental"
            assert estimate.support.row is None

    @pytest.mark.filterwarnings("ignore::cleverly.exceptions.InSampleLearnerWarning")
    def test_the_in_sample_contrast_keeps_the_learner_families(self) -> None:
        result = support.fit(interventions=[Static(1)], cross_fit=False, cv_evaluation=False)
        key = result.config.support_key
        assert key.splitting == ("in_sample",)
        assert "known_regimes" in key.qualifiers
        assert dict(key.options)["outcome_learner_family"] == "adaptive"
        assert result.extra["learned_rule"].scheme == "in_sample"

    @pytest.mark.parametrize(
        "settings",
        [
            {"targeting_scheme": "fold"},
            {"cv_evaluation": False, "targeting_scheme": "fold"},
        ],
        ids=["fold_targeted", "stacked_fold_targeted"],
    )
    def test_fold_targeting_runs_experimental(self, settings: dict[str, Any]) -> None:
        result = support.fit(interventions=[Static(1)], **settings)
        assert set(result.estimates) == {LEARNED, KNOWN, CONTRAST}
        assert all(e.support.label == "experimental" for e in result.estimates.values())


class TestTheCompositions:
    """Weights, clusters and strata run beside known regimes, each experimental."""

    def test_weights(self) -> None:
        frame = support.law_frame(400, 0)
        frame["w"] = np.random.default_rng(1).uniform(0.5, 2.0, len(frame))
        result = (
            support.learned_tmle(interventions=[Static(1)]).fit(frame, weights="w", **FIT).single()
        )
        assert result.config.support_key.weighting == "fixed"
        _assert_the_difference(result.estimates, "", cv_evaluation=True)

    def test_clusters(self) -> None:
        frame = support.law_frame(400, 0).assign(c=np.arange(400) // 4)
        result = support.learned_tmle(interventions=[Static(1)]).fit(frame, id="c", **FIT).single()
        assert result.config.support_key.dependence == "clustered"
        assert "known_regimes" in result.config.support_key.qualifiers
        assert result.estimates[CONTRAST].support.label == "experimental"

    @pytest.mark.parametrize("cv_evaluation", [True, False], ids=["fold_evaluated", "stacked"])
    def test_strata_report_each_stratum_s_contrast(self, cv_evaluation: bool) -> None:
        result = (
            support.learned_tmle(interventions=[Static(1)], cv_evaluation=cv_evaluation)
            .fit(support.law_frame(400, 0), strata=["W2"], **FIT)
            .single()
        )
        for label in ("[W2=0.0]", "[W2=1.0]"):
            _assert_the_difference(result.estimates, label, cv_evaluation=cv_evaluation)
        assert {"known_regimes", "strata"} <= result.config.support_key.qualifiers


def _assert_the_difference(estimates: Any, label: str, *, cv_evaluation: bool) -> None:
    """The contrast's point and curve are the differences, and its variance reads its curve.

    The variance is the stored-curve rule of the report: :math:`n^{-2}\\sum_i c_i^2` of the
    fold-evaluated curve, and the centred rule under the stacked report.
    """
    contrast, learned, known = (estimates[f"{name}{label}"] for name in (CONTRAST, LEARNED, KNOWN))
    assert contrast.psi == pytest.approx(learned.psi - known.psi, rel=1e-12, abs=1e-15)
    curve = np.asarray(contrast.influence_curve, dtype=float)
    difference = np.asarray(learned.influence_curve) - np.asarray(known.influence_curve)
    np.testing.assert_allclose(curve, difference, rtol=0, atol=1e-14)
    # Nonzero witness: the two curves differ, so the difference is not either one.
    assert float(np.max(np.abs(difference))) > 1e-2
    expected = (
        float(np.sum(curve**2)) / curve.size**2 if cv_evaluation else influence_variance(curve)
    )
    assert contrast.variance == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------------------------------- refusals


def _engine(**overrides: Any) -> TMLE:
    return support.learned_tmle(**{**never_fit_learners(), **overrides})


def _flat(text: Any) -> str:
    """``text`` with its line wrapping undone, so a sentence is found across lines."""
    return " ".join(str(text).split())


def _default_fit(frame: Any = None, **overrides: Any) -> Any:
    """A contrast fit with the constructor's defaults, except the learners and the folds."""
    estimator = TMLE(
        learned_rule=LearnedRule(),
        interventions=[Static(1)],
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        random_state=0,
        **{**support.FOLD_EVALUATED, **overrides},
    )
    assert estimator.simultaneous
    frame = support.law_frame() if frame is None else frame
    return estimator.fit(frame, **FIT).single()


class TestTheBand:
    """The default band is skipped beside known regimes, and a direct band call refuses."""

    def test_the_default_fit_skips_the_band_and_says_why(self) -> None:
        result = _default_fit()
        assert list(result.estimates) == [LEARNED, KNOWN, CONTRAST]
        assert result.simultaneous is None
        assert _flat(NO_KNOWN_REGIME_BANDS) in _flat(result.summary(detail="full"))

    def test_control_without_known_regimes_the_band_runs(self) -> None:
        """Nonzero witness: the same default fit with strata and no known regime builds one."""
        lone = (
            TMLE(
                learned_rule=LearnedRule(),
                outcome_learner=support.outcome_learner(),
                treatment_learner=support.treatment_learner(),
                n_folds=5,
                random_state=0,
                **support.FOLD_EVALUATED,
            )
            .fit(support.law_frame(), strata=["W2"], **FIT)
            .single()
        )
        assert lone.simultaneous is not None
        assert _flat(NO_KNOWN_REGIME_BANDS) not in _flat(lone.summary(detail="full"))

    @pytest.mark.parametrize("settings", [{"repeats": 2}, {"cv_evaluation": False}])
    def test_repeats_and_the_stacked_report_skip_it_too(self, settings: dict[str, Any]) -> None:
        result = _default_fit(**settings)
        assert result.simultaneous is None
        assert _flat(NO_KNOWN_REGIME_BANDS) in _flat(result.summary(detail="full"))

    def test_strata_beside_known_regimes_skip_it(self) -> None:
        result = (
            TMLE(
                learned_rule=LearnedRule(),
                interventions=[Static(1)],
                outcome_learner=support.outcome_learner(),
                treatment_learner=support.treatment_learner(),
                n_folds=5,
                random_state=0,
                **support.FOLD_EVALUATED,
            )
            .fit(support.law_frame(), strata=["W2"], **FIT)
            .single()
        )
        assert f"{CONTRAST}[W2=0.0]" in result.estimates
        assert result.simultaneous is None

    def test_the_learned_value_s_stratified_band_runs_beside_known_regimes(self) -> None:
        """Only the learned-rule value requested: its shipped stratified band still runs."""
        result = (
            TMLE(
                learned_rule=LearnedRule(),
                interventions=[Static(1)],
                estimands=("ey_learned_rule",),
                outcome_learner=support.outcome_learner(),
                treatment_learner=support.treatment_learner(),
                n_folds=5,
                random_state=0,
                **support.FOLD_EVALUATED,
            )
            .fit(support.law_frame(), strata=["W2"], **FIT)
            .single()
        )
        assert set(result.estimates) == {LEARNED, f"{LEARNED}[W2=0.0]", f"{LEARNED}[W2=1.0]"}
        assert result.simultaneous is not None
        assert set(result.simultaneous.bands) == set(result.estimates)

    @pytest.mark.parametrize(
        "names",
        [
            [LEARNED, KNOWN, CONTRAST],
            [LEARNED, KNOWN],
            [LEARNED, CONTRAST],
            [CONTRAST],
        ],
        ids=["all", "value_and_known", "value_and_contrast", "contrast_alone"],
    )
    def test_simultaneous_bands_refuses_a_known_value_or_contrast(self, names: list[str]) -> None:
        estimates = _default_fit().estimates
        with pytest.raises(CapabilityError) as caught:
            simultaneous_bands({name: estimates[name] for name in names})
        assert caught.value.refusal == KNOWN_REGIME_BAND_REFUSAL
        assert caught.value.refusal.reason is RefusalReason.INFERENCE_GAP
        assert "X11 (c2)" in str(caught.value)

    def test_simultaneous_bands_refuses_a_stratum_alias(self) -> None:
        result = (
            support.learned_tmle(interventions=[Static(1)])
            .fit(support.law_frame(), strata=["W2"], **FIT)
            .single()
        )
        aliases = [f"{CONTRAST}[W2=0.0]", f"{CONTRAST}[W2=1.0]"]
        with pytest.raises(CapabilityError) as caught:
            simultaneous_bands({name: result.estimates[name] for name in aliases})
        assert caught.value.refusal == KNOWN_REGIME_BAND_REFUSAL

    def test_one_estimate_builds_no_band_and_runs(self) -> None:
        """A band needs two estimates, so one requested value meets no refusal."""
        result = _default_fit(estimands=("ey_learned_rule",))
        assert list(result.estimates) == [LEARNED]
        assert result.simultaneous is None

    def test_the_learned_rule_value_is_always_reported(self) -> None:
        """Every contrast reads it and the record describes it, so a contrast-only list adds it."""
        result = support.fit(interventions=[Static(1)], estimands=("ate_learned_rule",))
        assert list(result.estimates) == [LEARNED, CONTRAST]
        assert result.extra["learned_rule"].splits[0].estimate == result.estimates[LEARNED].psi


class TestTheOtherSurfaces:
    def test_the_frozen_rule_replay_carries_the_known_regimes(self) -> None:
        result = _planted()
        replay = frozen_rule_replay(result.estimator, freeze_learned_rule(result))
        again = replay.fit(support.law_frame(203, 1), **FIT).single()
        assert again.nuisance.regimes.names == result.nuisance.regimes.names
        for name, estimate in result.estimates.items():
            assert again.estimates[name].psi == pytest.approx(estimate.psi, rel=1e-12), name

    def test_the_placebo_reads_a_contrast_against_zero_and_a_value_against_the_mean(
        self,
    ) -> None:
        """The placebo law gives every rule the outcome mean, so a contrast's null is 0."""
        result = support.fit(interventions=[Static(1)])
        outcome_mean = float(np.mean(result.data.outcome))
        assert outcome_mean > 0.1
        for name, null in ((LEARNED, outcome_mean), (KNOWN, outcome_mean), (CONTRAST, 0.0)):
            assert _placebo_null(result, name) == pytest.approx(null, rel=1e-12, abs=1e-15)
        test = refute(result, estimand=CONTRAST, tests=["placebo"], n_replicates=3).tests[0]
        assert abs(float(np.mean(test.values))) < 0.1

    def test_the_record_describes_the_learned_column(self) -> None:
        result = _planted()
        alone = support.fit(
            support.law_frame(203, 1), outcome_learner=support.misspecified_learner()
        )
        split, solo = result.extra["learned_rule"].splits[0], alone.extra["learned_rule"].splits[0]
        assert split.arm_shares == solo.arm_shares
        assert split.estimate == result.estimates[LEARNED].psi

    def test_a_round_trip_keeps_every_name(self) -> None:
        from cleverly.estimators.serialize import dumps, loads

        result = _planted()
        loaded = loads(dumps(result))
        assert list(loaded.estimates) == list(result.estimates)
        assert loaded.nuisance.regimes.names == ("learned rule", "always 1")

    def test_a_stratum_placebo_reads_the_stratum_mean_and_a_stratum_contrast_zero(
        self,
    ) -> None:
        """A known-regime stratum alias reads its stratum's outcome mean, its contrast 0."""
        result = (
            support.learned_tmle(interventions=[Static(1)])
            .fit(_strata_frame(), strata=("S",), **STRATA_FIT)
            .single()
        )
        data = result.data
        means = {
            data.stratum_label(code): float(np.mean(data.outcome[data.strata == code]))
            for code in range(data.n_strata)
        }
        # Nonzero witness: the strata's outcome means differ, so E[Y] cannot pass for them.
        assert max(means.values()) - min(means.values()) > 0.2, means
        for label, mean in means.items():
            for name in (LEARNED, KNOWN):
                assert _placebo_null(result, f"{name}[{label}]") == pytest.approx(mean, rel=1e-12)
            assert _placebo_null(result, f"{CONTRAST}[{label}]") == 0.0


class TestTheSummary:
    """A contrast fit's summary names the fixed rules, the band, the bound and the reason."""

    @pytest.fixture(scope="class")
    def full(self) -> str:
        return str(_planted().summary(detail="full"))

    def test_the_method_details_name_the_fixed_rules(self, full: str) -> None:
        line = vocabulary.LEARNED_RULE_KNOWN_REGIMES.format(
            names="always 1", contrast=vocabulary.LEARNED_RULE_CONTRAST_DIRECTION[False]
        )
        assert _flat(line) in _flat(full)

    def test_reference_naming_the_rule_flips_the_described_direction(self) -> None:
        result = support.fit(interventions=[Static(1)], reference="learned rule")
        text = _flat(result.summary(detail="full"))
        assert vocabulary.LEARNED_RULE_CONTRAST_DIRECTION[True] in text
        assert vocabulary.LEARNED_RULE_CONTRAST_DIRECTION[False] not in text

    def test_the_bound_pointer_stays(self, full: str) -> None:
        """Each value can be bounded, so the pointer names the one-row call."""
        assert _flat(vocabulary.SENSITIVITY_POINTER_ONE_ROW) in _flat(full)

    def test_the_experimental_reason_names_the_fixed_rules(self, full: str) -> None:
        text = _flat(full)
        assert vocabulary.SUPPORT_QUALIFIERS["known_regimes"] in text
        assert vocabulary.SUPPORT_AXES["qualifiers"] not in text
        reason = _planted().estimates[LEARNED].support.reason
        assert "differing on qualifiers (known_regimes)" in reason

    def test_the_repeats_note_says_a_fixed_rule_s_target_is_fixed(self) -> None:
        text = _flat(_repeated()[0].summary())
        assert "A fixed rule's value has the same target in every split." in text
        alone = _flat(support.fit(repeats=2).summary())
        assert "A fixed rule's value" not in alone
        assert "so each split has its own target" in alone
