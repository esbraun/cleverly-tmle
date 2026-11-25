"""The natural course beside a learned rule: ``TMLE(learned_rule=..., interventions=[NaturalCourse()])``.

A :class:`~cleverly.interventions.NaturalCourse` is the observed course, a comparator of the
learned rule only.  The fit reports its mean ``ey_natural_course`` from the observed outcomes,
under the report's fold weights, and the contrast ``ate_learned_rule[<rule> vs natural course]``
from the difference of the two curves.  The mean sits outside the regime set and its
fluctuation.  The witnesses:

- W-d5 (fold weighting): the fold-evaluated point is :math:`\\sum_v V^{-1}\\bar Y_v` at unequal
  folds and differs from :math:`\\bar Y`; its curve is :math:`Y - \\bar Y_{v(i)}` under the
  report's row scaling; the stacked point is :math:`\\bar Y`.  The weighted and stratum forms
  are checked too.
- W-d6 (covariance): :math:`D_L` and :math:`D_N` are correlated, the contrast's SE is the
  report's rule on :math:`D_L - D_N`, and the curves misaligned by one row, or read as
  independent, give another SE.
- W-d7 (sign): with a stub regression every fold rule is ``1{W1 > 0}``, the learned value
  equals the regime fit of ``Rule(1{W1 > 0})`` on the same folds, and the contrast equals that
  value minus :math:`\\sum_v V^{-1}\\bar Y_v`, longhand.
- W-d12 (outside the fluctuation): the learned and regime columns of a fit with the natural
  course equal those of the same fit without it, bit for bit, at a nonzero coefficient.
- W-d13 (outside the density array): the regime densities are the same with and without it,
  and the omitted-variable bound of each regime value is unchanged.
- The refusals of a natural course that is not beside a learned rule (R-d8a, R-d8b), beside an
  intermediate (R-d7), and of ``ey_natural_course`` without one (R-d9), before any learner.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import LearnedRule, NaturalCourse, Rule, Static, Stochastic
from cleverly.interventions.base import NATURAL_COURSE_ALONE_REFUSAL
from cleverly.longitudinal import LTMLE
from cleverly.sensitivity import omitted_variable_bounds
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners
from tests.unit.test_learned_rule_in_sample import SignStub
from tests.unit.test_learned_rule_strata import FIT as _STRATA_FIT
from tests.unit.test_learned_rule_strata import _strata_frame

FIT: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}
STRATA_FIT: dict[str, Any] = dict(_STRATA_FIT)
LEARNED = support.NAME
NATURAL = "ey_natural_course[natural course]"
CONTRAST = "ate_learned_rule[learned rule vs natural course]"
FLIPPED = "ate_learned_rule[natural course vs learned rule]"

#: The declared factors of W-d6, each below what the fixed draw gives.
W_D6_CORRELATION = 0.1  # observed corr(D_L, D_N) = 0.68 fold-evaluated, 0.70 stacked
W_D6_ROLL = 1.25  # observed 1.63 and 1.66 for the curves misaligned by one row
W_D6_INDEPENDENT = 1.10  # observed 1.66 and 1.69 for sqrt(se_L^2 + se_N^2) over the joint SE


def _coin() -> Stochastic:
    return Stochastic(
        lambda w: np.column_stack([np.full(len(w), 0.5), np.full(len(w), 0.5)]),
        name="coin",
        density_kind="known",
    )


def _offer() -> Stochastic:
    def density(w: pd.DataFrame) -> np.ndarray:
        p = 0.5 * (np.asarray(w["W1"]) > 0)
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, name="offer", density_kind="known")


@cache
def _fit(comparators: str, cv_evaluation: bool = True) -> Any:
    """A misspecified outcome regression at unequal folds (n = 203, V = 5)."""
    declared = {
        "natural": [NaturalCourse()],
        "static": [Static(1)],
        "static_natural": [Static(1), NaturalCourse()],
        "stochastic": [Static(0), _coin(), _offer()],
        "stochastic_natural": [Static(0), _coin(), _offer(), NaturalCourse()],
    }[comparators]
    return support.fit(
        support.law_frame(203, 1),
        outcome_learner=support.misspecified_learner(),
        interventions=declared,
        cv_evaluation=cv_evaluation,
    )


def _folds(result: Any) -> list[np.ndarray]:
    return [np.asarray(test) for _, test in result.nuisance.folds]


# ------------------------------------------------------------------------------- names


@pytest.mark.parametrize("cv_evaluation", [True, False])
def test_the_natural_course_alone_reports_its_value_and_contrast(cv_evaluation: bool) -> None:
    """``(NaturalCourse(),)`` alone is a comparator: the fit keeps its contrast."""
    result = _fit("natural", cv_evaluation)
    assert list(result.estimates) == [LEARNED, NATURAL, CONTRAST]
    assert result.estimator.interventions == ()
    assert result.estimator.natural_course == NaturalCourse()


def test_reference_naming_the_rule_flips_the_natural_contrast() -> None:
    result = support.fit(
        support.law_frame(203, 1),
        outcome_learner=support.misspecified_learner(),
        interventions=[NaturalCourse()],
        reference="learned rule",
    )
    plain = _fit("natural")
    assert list(result.estimates) == [LEARNED, NATURAL, FLIPPED]
    assert result.psi(FLIPPED) == pytest.approx(-plain.psi(CONTRAST), abs=1e-15)
    np.testing.assert_allclose(
        result.estimates[FLIPPED].influence_curve,
        -np.asarray(plain.estimates[CONTRAST].influence_curve),
        atol=1e-15,
        rtol=0,
    )


def test_a_fit_with_known_regimes_and_no_natural_course_does_not_list_it() -> None:
    assert NATURAL not in _fit("static").estimates
    assert not any("natural" in name for name in _fit("stochastic").estimates)


# ------------------------------------------------------------------------------- W-d5


def test_w_d5_the_fold_evaluated_point_averages_the_fold_means() -> None:
    result = _fit("natural")
    y = np.asarray(result.data.outcome, dtype=float)
    folds = _folds(result)
    n, v = y.size, len(folds)
    sizes = {test.size for test in folds}
    assert len(sizes) > 1, "the witness needs unequal folds"
    estimate = result.estimates[NATURAL]
    assert estimate.psi == pytest.approx(np.mean([y[test].mean() for test in folds]), abs=1e-15)
    # Unequal folds weight a fold's mean by 1/V, not by its share of rows.
    assert abs(estimate.psi - y.mean()) > 1e-4
    expected = np.empty(n)
    for test in folds:
        expected[test] = (y[test] - y[test].mean()) * n / (v * test.size)
    np.testing.assert_allclose(estimate.influence_curve, expected, atol=1e-15, rtol=0)
    variance = sum(np.sum((y[t] - y[t].mean()) ** 2) / t.size**2 for t in folds) / v**2
    assert estimate.std_error**2 == pytest.approx(variance, rel=1e-12)


def test_w_d5_the_stacked_point_is_the_observed_mean() -> None:
    result = _fit("natural", cv_evaluation=False)
    y = np.asarray(result.data.outcome, dtype=float)
    estimate = result.estimates[NATURAL]
    assert estimate.psi == pytest.approx(y.mean(), abs=1e-15)
    np.testing.assert_allclose(estimate.influence_curve, y - y.mean(), atol=1e-15, rtol=0)


@pytest.mark.parametrize("cv_evaluation", [True, False])
def test_w_d5_the_weighted_mean_reads_the_weights(cv_evaluation: bool) -> None:
    frame = _strata_frame(203, 1)
    result = (
        support.learned_tmle(
            outcome_learner=support.misspecified_learner(),
            interventions=[NaturalCourse()],
            cv_evaluation=cv_evaluation,
        )
        .fit(frame, weights="w", **STRATA_FIT)
        .single()
    )
    y = np.asarray(result.data.outcome, dtype=float)
    w = np.asarray(result.data.weights, dtype=float)
    folds = _folds(result)
    expected = (
        np.mean([np.average(y[test], weights=w[test]) for test in folds])
        if cv_evaluation
        else np.average(y, weights=w)
    )
    assert result.psi(NATURAL) == pytest.approx(expected, abs=1e-15)
    assert abs(result.psi(NATURAL) - y.mean()) > 1e-4


def test_w_d5_each_stratum_reports_its_own_observed_mean() -> None:
    """Under the stacked report each stratum's mean is the stratum's observed mean."""
    frame = _strata_frame(203, 1)
    result = (
        support.learned_tmle(
            outcome_learner=support.misspecified_learner(),
            interventions=[NaturalCourse()],
            cv_evaluation=False,
        )
        .fit(frame, strata=("S",), **STRATA_FIT)
        .single()
    )
    y = np.asarray(result.data.outcome, dtype=float)
    stratum = frame["S"].to_numpy()
    for level in np.unique(stratum):
        name = f"{NATURAL}[S={level}]"
        assert result.psi(name) == pytest.approx(y[stratum == level].mean(), abs=1e-15)
        contrast = f"{CONTRAST}[S={level}]"
        assert result.psi(contrast) == pytest.approx(
            result.psi(f"{LEARNED}[S={level}]") - result.psi(name), abs=1e-15
        )


def test_fold_evaluated_strata_refuse_the_natural_course_before_any_learner() -> None:
    """No fold-evaluated stratum construction of the natural course is derived (X28)."""
    with pytest.raises(CapabilityError) as caught:
        support.learned_tmle(interventions=[NaturalCourse()], **never_fit_learners()).fit(
            _strata_frame(203, 1), strata=("S",), **STRATA_FIT
        )
    assert "ey_natural_course" in str(caught.value)
    assert caught.value.refusal.roadmap == "X28"
    assert NeverFit.calls == 0


# ------------------------------------------------------------------------------- W-d6


@pytest.mark.parametrize("cv_evaluation", [True, False])
def test_w_d6_the_contrast_reads_the_covariance(cv_evaluation: bool) -> None:
    result = _fit("natural", cv_evaluation)
    learned = np.asarray(result.estimates[LEARNED].influence_curve, dtype=float)
    natural = np.asarray(result.estimates[NATURAL].influence_curve, dtype=float)
    contrast = result.estimates[CONTRAST]
    assert np.corrcoef(learned, natural)[0, 1] > W_D6_CORRELATION
    np.testing.assert_allclose(contrast.influence_curve, learned - natural, atol=1e-15, rtol=0)
    joint = contrast.std_error
    # The report's rule on the difference: the same rule the value reports on its own curve.
    scale = result.estimates[LEARNED].std_error / np.sqrt(np.sum(learned**2))
    if cv_evaluation:
        folds = _folds(result)
        n, v = learned.size, len(folds)

        def rule(curve: np.ndarray) -> float:
            raw = [curve[t] * v * t.size / n for t in folds]
            return float(np.sqrt(sum(np.sum(r**2) / r.size**2 for r in raw) / v**2))

        assert joint == pytest.approx(rule(learned - natural), rel=1e-12)
        rolled = rule(learned - np.roll(natural, 1))
    else:
        diff = learned - natural
        assert joint == pytest.approx(scale * np.sqrt(np.sum(diff**2)), rel=1e-12)
        rolled = scale * float(np.sqrt(np.sum((learned - np.roll(natural, 1)) ** 2)))
    assert rolled / joint > W_D6_ROLL
    independent = np.hypot(result.estimates[LEARNED].std_error, result.estimates[NATURAL].std_error)
    assert independent / joint > W_D6_INDEPENDENT


# ------------------------------------------------------------------------------- W-d7


def _w1_rule() -> Rule:
    return Rule(lambda w: (w["W1"] > 0).astype(float), name="w1", rule_kind="known")


@cache
def _sign(cv_evaluation: bool) -> tuple[Any, Any]:
    """The stub learned-rule fit beside the natural course, and the regime fit of the rule."""
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
        TMLE(learned_rule=LearnedRule(), interventions=[NaturalCourse()], **common)
        .fit(frame, **FIT)
        .single()
    )
    regime = TMLE(interventions=[_w1_rule()], **common).fit(frame, **FIT).single()
    return learned, regime


@pytest.mark.parametrize("cv_evaluation", [True, False])
def test_w_d7_the_natural_contrast_is_the_rule_value_minus_the_fold_means(
    cv_evaluation: bool,
) -> None:
    learned, regime = _sign(cv_evaluation)
    w1 = np.asarray(learned.data.covariates[:, 0]) > 0
    np.testing.assert_array_equal(learned.nuisance.regimes.values[:, 1, 0], w1.astype(float))
    value = regime.psi("ey_regime[w1]")
    assert learned.psi(LEARNED) == value
    y = np.asarray(learned.data.outcome, dtype=float)
    folds = _folds(learned)
    observed = np.mean([y[t].mean() for t in folds]) if cv_evaluation else y.mean()
    assert learned.psi(CONTRAST) == pytest.approx(value - observed, abs=1e-14)
    # The rule wins on this law, so the contrast is positive: a flipped stem would fail.
    assert learned.psi(CONTRAST) > 0.02


# ------------------------------------------------------------------------------- W-d12


@pytest.mark.parametrize("cv_evaluation", [True, False])
@pytest.mark.parametrize(
    "pair", [("static", "static_natural"), ("stochastic", "stochastic_natural")]
)
def test_w_d12_the_natural_course_never_enters_the_fluctuation(
    pair: tuple[str, str], cv_evaluation: bool
) -> None:
    without, with_natural = (_fit(name, cv_evaluation) for name in pair)
    assert float(np.min(np.abs(with_natural.fluctuations["regime"].epsilon))) > 1e-3
    np.testing.assert_array_equal(
        with_natural.fluctuations["regime"].epsilon, without.fluctuations["regime"].epsilon
    )
    for name, estimate in without.estimates.items():
        other = with_natural.estimates[name]
        assert other.psi == estimate.psi, name
        assert other.std_error == estimate.std_error, name
        np.testing.assert_array_equal(other.influence_curve, estimate.influence_curve)
    # Whatever regimes are present, the natural course's point is the longhand form.
    y = np.asarray(with_natural.data.outcome, dtype=float)
    folds = _folds(with_natural)
    observed = np.mean([y[t].mean() for t in folds]) if cv_evaluation else y.mean()
    assert with_natural.psi(NATURAL) == pytest.approx(observed, abs=1e-15)
    assert with_natural.psi(NATURAL) == _fit("natural", cv_evaluation).psi(NATURAL)


# ------------------------------------------------------------------------------- W-d13


def test_w_d13_the_density_array_excludes_the_natural_course() -> None:
    without, with_natural = _fit("stochastic"), _fit("stochastic_natural")
    values = np.asarray(with_natural.nuisance.regimes.values)
    assert values.shape == (203, 2, 4)
    np.testing.assert_array_equal(values, np.asarray(without.nuisance.regimes.values))
    assert with_natural.nuisance.regimes.names == without.nuisance.regimes.names


def test_the_support_report_names_the_natural_course_and_keeps_its_rows() -> None:
    """The natural course has no density, so it has no row of the regime support report; the
    report's summary names it as using no treatment weights."""
    without = _fit("stochastic").diagnostics.support()
    report = _fit("stochastic_natural").diagnostics.support()
    assert list(report.regimes) == list(without.regimes)
    assert "natural course" not in report.regimes
    assert report.natural_course == "natural course" and without.natural_course is None
    text = report.summary().note("natural_course_no_weights").text
    assert text.startswith("Natural course has no row") and "no treatment weights" in text
    assert "natural_course_no_weights" not in without.summary().note_codes


def test_the_score_check_lists_no_equation_for_the_natural_course() -> None:
    """No targeting equation reads the natural course: the fluctuation rows of the fit with it
    are those of the fit without it, its value and contrast have only influence-curve rows, and
    the summary says why."""
    without = _fit("stochastic").diagnostics.score_equations()
    check = _fit("stochastic_natural").diagnostics.score_equations()
    fluctuation = [row.name for row in check.rows if row.kind == "fluctuation"]
    assert fluctuation == [row.name for row in without.rows if row.kind == "fluctuation"]
    named = {row.name: row.kind for row in check.rows if "natural course" in row.name}
    assert named == {NATURAL: "influence curve", CONTRAST: "influence curve"}
    assert check.natural_course == "natural course" and without.natural_course is None
    text = check.summary().note("natural_course_no_equation").text
    assert "no equation the correction step solves" in text and "ey_natural_course" in text
    assert "natural_course_no_equation" not in without.summary().note_codes


def test_w_d13_the_omitted_bound_of_a_regime_value_is_unchanged() -> None:
    without, with_natural = _fit("static"), _fit("static_natural")
    name = "ey_known_regime[always 1]"
    bound = omitted_variable_bounds(without, estimand=name, cf_y=0.05, cf_d=0.05)
    other = omitted_variable_bounds(with_natural, estimand=name, cf_y=0.05, cf_d=0.05)
    assert (other.max_bias, other.lower, other.upper) == (bound.max_bias, bound.lower, bound.upper)


def test_the_omitted_bound_of_the_natural_course_has_nothing_to_bound() -> None:
    result = _fit("natural")
    with pytest.raises(CapabilityError) as caught:
        omitted_variable_bounds(result, estimand=NATURAL, cf_y=0.05, cf_d=0.05)
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    assert "nothing to bound" in str(caught.value)
    with pytest.raises(CapabilityError) as contrast:
        omitted_variable_bounds(result, estimand=CONTRAST, cf_y=0.05, cf_d=0.05)
    assert contrast.value.refusal.roadmap == "X11"
    assert "natural course" in str(contrast.value)


# ------------------------------------------------------------------------------- labels


def test_the_key_declares_the_natural_course() -> None:
    qualifiers = _fit("natural").config.support_key.qualifiers
    assert {"known_regimes", "natural_course"} <= qualifiers
    assert "known_stochastic_regimes" not in qualifiers
    combined = _fit("stochastic_natural").config.support_key.qualifiers
    assert {"known_regimes", "known_stochastic_regimes", "natural_course"} <= combined


def test_every_natural_course_estimate_reads_experimental() -> None:
    """No registered row declares ``natural_course`` until the comparators study registers."""
    for name in ("natural", "static_natural", "stochastic_natural"):
        result = _fit(name)
        assert {e.support.label for e in result.estimates.values()} == {"experimental"}


# ------------------------------------------------------------------------------- refusals


def test_r_d8b_a_natural_course_alone_without_a_rule_redirects() -> None:
    with pytest.raises(CapabilityError) as caught:
        TMLE(interventions=[NaturalCourse()], **never_fit_learners())
    assert caught.value.refusal.reason is RefusalReason.API_REDIRECTION
    assert str(caught.value).startswith(NATURAL_COURSE_ALONE_REFUSAL.condition)
    assert "NaturalCourseMean()" in str(caught.value.refusal.alternative)
    assert NeverFit.calls == 0


@pytest.mark.parametrize("declared", [[Static(0), NaturalCourse()], [NaturalCourse(), 1]])
def test_r_d8a_a_natural_course_among_regimes_without_a_rule_refuses(
    declared: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Any] = []
    original = Static.__init__

    def spy(self: Any, level: Any, *args: Any, **kwargs: Any) -> None:
        built.append(level)
        original(self, level, *args, **kwargs)

    monkeypatch.setattr(Static, "__init__", spy)
    with pytest.raises(CapabilityError) as caught:
        TMLE(interventions=declared, **never_fit_learners())
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert caught.value.refusal.roadmap == "X11"
    assert "interventions= holds NaturalCourse(), the observed course" in str(caught.value)
    assert not any(isinstance(level, NaturalCourse) for level in built)
    assert NeverFit.calls == 0


def test_two_natural_courses_refuse() -> None:
    with pytest.raises(DataError, match="declares 2 NaturalCourse"):
        TMLE(learned_rule=LearnedRule(), interventions=[NaturalCourse(), NaturalCourse("x")])


def test_a_natural_course_named_like_another_comparator_refuses() -> None:
    with pytest.raises(CapabilityError) as caught:
        support.learned_tmle(
            interventions=[Static(1), NaturalCourse("always 1")], **never_fit_learners()
        ).fit(support.law_frame(203, 1), **FIT)
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    assert "'always 1'" in str(caught.value)
    assert NeverFit.calls == 0


def test_a_natural_course_named_like_the_rule_refuses() -> None:
    with pytest.raises(CapabilityError) as caught:
        support.learned_tmle(
            interventions=[NaturalCourse("learned rule")], **never_fit_learners()
        ).fit(support.law_frame(203, 1), **FIT)
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    assert NeverFit.calls == 0


def test_r_d9_ey_natural_course_without_a_natural_course_refuses() -> None:
    with pytest.raises(CapabilityError) as caught:
        support.learned_tmle(
            interventions=[Static(1)], estimands=["ey_natural_course"], **never_fit_learners()
        ).fit(support.law_frame(203, 1), **FIT)
    assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
    assert str(caught.value).startswith(
        "estimands= names ey_natural_course, and interventions= declares no NaturalCourse()"
    )
    assert caught.value.refusal.alternative == (
        "Add NaturalCourse() to interventions= beside the learned rule."
    )
    assert NeverFit.calls == 0


def test_r_d7_a_natural_course_beside_an_intermediate_refuses() -> None:
    frame = support.law_frame(203, 1).assign(Z=lambda f: (f["W2"] > 0).astype(int))
    with pytest.raises(CapabilityError) as caught:
        support.learned_tmle(interventions=[NaturalCourse()], **never_fit_learners()).fit(
            frame, intermediate="Z", **FIT
        )
    assert caught.value.refusal.reason is RefusalReason.DIFFERENT_ESTIMAND
    assert "NaturalCourse() reads the observed course, in which Z is not held" in str(caught.value)
    assert str(caught.value.refusal.alternative).startswith(
        "Compare with a known regime at the held level"
    )
    assert NeverFit.calls == 0


def test_a_longitudinal_regimen_node_refuses_the_natural_course() -> None:
    with pytest.raises(CapabilityError) as caught:
        LTMLE(regimens={"current": [NaturalCourse(), 1]}).fit(
            pd.DataFrame({"L0": [0.0, 1.0], "A1": [0, 1], "A2": [1, 0], "Y": [0, 1]}),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["L0"],
        )
    assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert "regimen 'current' holds NaturalCourse()" in str(caught.value)


def test_the_natural_course_takes_a_non_empty_name() -> None:
    assert NaturalCourse().name == "natural course"
    with pytest.raises(DataError):
        NaturalCourse("")
