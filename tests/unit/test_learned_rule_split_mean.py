"""X11 (i): the learned-rule value with ``repeats=R`` reports the mean over the ``R`` splits.

Each split ``r`` learns its own fold rules, so it estimates its own target
``psi_tilde^(r)``.  The reported target is the mean of the split targets.  By linearity over
a fixed ``R``, the estimate is the mean of the split estimates, and its curve is the mean
``phibar_i`` of the split curves ``phi_{r,i} = (n / (V n_v)) D_{r,i}``, so the variance is
``n^-2 sum_i phibar_i^2``.  The median of the split estimates estimates no defined parameter.

* W-i1 (identity): the reported point is the mean of ``R`` single-split fits at the same split
  seeds, bit for bit, and the variance is rebuilt from their curves.
* W-i2 (nonzero): on the witness draw the mean and the median of the four split estimates
  differ by more than :data:`MEAN_MEDIAN_GAP`, so a median mutation shows at unit level.
* The clusters row: ``repeats`` above 1 with ``id=`` refuses before any learner and cites
  X11 (i2). Its remedy, ``repeats=1``, fits as written.
* The surfaces that read one rule per row refuse a fit of several splits, and each remedy,
  a refit with ``repeats=1``, runs as written.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any

import numpy as np
import pytest

from cleverly import CausalStudy, CrossFitting, LearnedRuleValue, PointTreatment, TMLEMethod
from cleverly.exceptions import CapabilityError
from cleverly.inference.influence import median_estimates
from cleverly.interventions import LearnedRuleRecord
from cleverly.interventions.learned import freeze_learned_rule
from cleverly.methods import ModelSpec
from cleverly.sensitivity._simulated_confounding_request import simulated_confounding_refusal
from cleverly.validation.refute import refute, refute_refusal
from tests.unit import _learned_rule_support as support
from tests.unit._natural_course_support import NeverFit, never_fit_learners

#: The split count of the witness fit.
R = 4

#: The declared floor of ``|mean - median|`` of the four split estimates on the witness
#: draw (``law_frame(400, 0)``, ``random_state=0``).  The draw gives about 0.0028, roughly
#: 0.09 of the reported standard error.
MEAN_MEDIAN_GAP = 0.002

#: The rule-holding refutation tests (X82 part F4).
HOLDING_TESTS = ("random_common_cause", "subset", "bootstrap_stability")


@cache
def _repeated() -> Any:
    return support.fit(repeats=R)


@cache
def _singles() -> tuple[Any, ...]:
    """One single-split fit at each split seed of :func:`_repeated`."""
    return tuple(support.fit(random_state=repeat.seed) for repeat in _repeated().repeats)


def _with_repeats_one(message: str) -> dict[str, Any]:
    """The setting that the remedy of ``message`` prints, found there verbatim."""
    assert "repeats=1" in message, message
    return {"repeats": 1}


class TestTheMeanOverSplits:
    """W-i1 and W-i2."""

    def test_the_point_is_the_mean_of_single_split_fits_bit_for_bit(self) -> None:
        reported = _repeated().estimates[support.NAME]
        singles = [fit.estimates[support.NAME] for fit in _singles()]
        assert reported.psi == float(np.mean([single.psi for single in singles]))
        assert [split.seed for split in _repeated().extra["learned_rule"].splits] == [
            repeat.seed for repeat in _repeated().repeats
        ]

    def test_the_variance_is_the_second_moment_of_the_mean_curve(self) -> None:
        reported = _repeated().estimates[support.NAME]
        n = _repeated().data.n
        phibar = np.mean(
            [fit.estimates[support.NAME].influence_curve for fit in _singles()], axis=0
        )
        np.testing.assert_array_equal(reported.influence_curve, phibar)
        assert reported.variance == float(np.sum(np.square(phibar)) / n**2)
        # Each single curve is phi_{r,i} = (n / (V n_v)) D_{r,i}: its second moment is the
        # split's own cross-validated variance.
        for fit in _singles():
            curve = fit.estimates[support.NAME].influence_curve
            assert np.sum(curve**2) / n**2 == pytest.approx(
                fit.estimates[support.NAME].variance, rel=1e-12
            )

    def test_the_variance_is_at_most_the_mean_split_variance(self) -> None:
        """Jensen's inequality, a statement about the two estimates on every draw."""
        reported = _repeated().estimates[support.NAME]
        split_variances = [fit.estimates[support.NAME].variance for fit in _singles()]
        assert reported.variance <= float(np.mean(split_variances))
        assert reported.variance < min(split_variances)

    def test_the_fold_report_takes_the_mean_too(self) -> None:
        detail = _repeated().cv_targeting
        assert detail.repeats == R and detail.aggregate == "mean"
        assert detail.fold_evaluated[support.NAME].psi == _repeated().estimates[support.NAME].psi
        pooled = [fit.cv_targeting.pooled[support.NAME] for fit in _singles()]
        assert detail.pooled[support.NAME].psi == float(np.mean([p.psi for p in pooled]))
        curve = np.mean([p.influence_curve for p in pooled], axis=0)
        assert detail.pooled[support.NAME].variance == float(np.var(curve, ddof=1) / curve.size)

    def test_the_mean_and_the_median_differ_on_the_witness_draw(self) -> None:
        """W-i2: the median of the split estimates is a different number."""
        splits = [fit.estimates[support.NAME].psi for fit in _singles()]
        reported = _repeated().estimates[support.NAME].psi
        assert abs(reported - float(np.median(splits))) > MEAN_MEDIAN_GAP
        # Mutation: the median rule over the same split reports leaves the mean.
        median = median_estimates([fit.estimates for fit in _singles()])[support.NAME].psi
        assert abs(median - reported) > MEAN_MEDIAN_GAP

    def test_one_split_is_bit_for_bit_the_ordinary_fit(self) -> None:
        once = support.fit(repeats=1)
        ordinary = support.fit()
        assert once.estimates[support.NAME].psi == ordinary.estimates[support.NAME].psi
        assert once.extra["learned_rule"].n_splits == 1


class TestTheRecord:
    """One record per split, and a summary line that gives the range over splits."""

    def test_one_record_per_split(self) -> None:
        record = _repeated().extra["learned_rule"]
        assert isinstance(record, LearnedRuleRecord)
        assert record.n_splits == R
        for split, single in zip(record.splits, _singles(), strict=True):
            (alone,) = single.extra["learned_rule"].splits
            assert split == alone
            assert split.estimate == single.estimates[support.NAME].psi
        assert record.fold_sizes == record.splits[0].fold_sizes

    def test_the_summary_line_gives_the_range_over_splits(self) -> None:
        record = _repeated().extra["learned_rule"]
        line = record.describe()
        estimates = [split.estimate for split in record.splits]
        assert f"mean over {R} random splits" in line
        assert f"{min(estimates):.4g} to {max(estimates):.4g}" in line
        shares = [share for split in record.splits for share in split.treated_shares]
        assert f"{min(shares):.3g} to {max(shares):.3g}" in line

    def test_the_summary_notes_say_mean(self) -> None:
        note = _repeated().summary().note("repeated_splits").text
        assert f"mean over {R} random splits" in note and "median" not in note
        cv_note = _repeated().cv_targeting.summary().note("cv_first_split").text
        assert "by their mean" in cv_note


def _clustered(n: int = 120) -> Any:
    return support.law_frame(n, 3).assign(c=np.arange(n) % 20)


_FIT = {"outcome": "Y", "treatment": "A", "covariates": support.COVARIATES}


class TestTheClustersRow:
    """``repeats`` above 1 with ``id=`` refuses and cites X11 (i2)."""

    def _refused(self, call: Any, item: str) -> str:
        NeverFit.calls = 0
        with pytest.raises(CapabilityError) as caught:
            call()
        assert NeverFit.calls == 0
        message = str(caught.value)
        assert item in message, message
        return message

    def test_the_engine_refuses_before_any_learner(self) -> None:
        estimator = support.learned_tmle(repeats=2, **never_fit_learners())
        message = self._refused(lambda: estimator.fit(_clustered(), id="c", **_FIT), "X11 (i2)")
        assert "X11 (i2) in docs/roadmap.md" in message

    def test_the_remedy_fits_as_written(self) -> None:
        estimator = support.learned_tmle(repeats=2, **never_fit_learners())
        message = self._refused(lambda: estimator.fit(_clustered(), id="c", **_FIT), "X11 (i2)")
        result = support.learned_tmle(**_with_repeats_one(message)).fit(
            _clustered(), id="c", **_FIT
        )
        assert np.isfinite(result.single().estimates[support.NAME].psi)

    def test_a_data_row_comes_first(self) -> None:
        estimator = support.learned_tmle(repeats=2, **never_fit_learners())
        self._refused(
            lambda: estimator.fit(_clustered(), id="c", strata=("W2",), **_FIT), "X11 (m)"
        )

    def test_a_fold_policy_row_comes_first(self) -> None:
        """The stacked row's remedy keeps ``repeats=2``, which then meets this later row."""
        estimator = support.learned_tmle(repeats=2, cv_evaluation=False, **never_fit_learners())
        self._refused(lambda: estimator.fit(_clustered(), id="c", **_FIT), "X11 (e)")

    def test_the_method_refuses_with_the_same_row(self) -> None:
        frame = _clustered()
        study = CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y", treatment="A", adjustment=support.COVARIATES, cluster="c"
            ),
        )
        method = TMLEMethod(
            models=ModelSpec(**never_fit_learners()),
            cross_fitting=CrossFitting(n_folds=5, fold_evaluation=True, repeats=2),
        )
        effect = study.identify(LearnedRuleValue())
        message = self._refused(lambda: effect.estimate(method, simultaneous=False), "X11 (i2)")
        assert "CrossFitting(repeats=1)" in message
        remedy = TMLEMethod(
            models=ModelSpec(
                outcome_learner=support.outcome_learner(),
                treatment_learner=support.treatment_learner(),
            ),
            cross_fitting=CrossFitting(n_folds=5, fold_evaluation=True, repeats=1),
        )
        assert np.isfinite(effect.estimate(remedy, simultaneous=False)[support.NAME].psi)


@cache
def _study_fit(repeats: int) -> Any:
    study = CausalStudy(
        support.law_frame(n=403),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    return study.identify(LearnedRuleValue()).estimate(
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        cv_evaluation=True,
        simultaneous=False,
        random_state=0,
        repeats=repeats,
    )


def _remedy_names_repeats_one(message: str, operation: str) -> None:
    assert re.search(r"Refit with repeats=1 \(CrossFitting\(repeats=1\) on the method\)", message)
    assert f"run {operation} on that fit" in message


class TestTheSurfacesThatReadOneRulePerRow:
    """Each refuses a fit of several splits, and its remedy, ``repeats=1``, runs."""

    def test_freeze_learned_rule(self) -> None:
        with pytest.raises(CapabilityError) as caught:
            freeze_learned_rule(_repeated())
        message = str(caught.value)
        _remedy_names_repeats_one(message, "freeze_learned_rule")
        once = support.fit(**_with_repeats_one(message))
        assert freeze_learned_rule(once).treat.shape == (once.data.n,)

    def test_simulated_confounding(self) -> None:
        refusal = simulated_confounding_refusal(_study_fit(2), support.NAME)
        assert refusal is not None
        _remedy_names_repeats_one(refusal.message(), "simulated_confounding")
        row = _study_fit(2).sensitivity.capability("simulated_confounding")
        assert not row.available
        assert simulated_confounding_refusal(_study_fit(1), support.NAME) is None

    def test_the_rule_holding_refute_tests(self) -> None:
        for name in HOLDING_TESTS:
            refusal = refute_refusal(_repeated(), estimand=support.NAME, tests=(name,))
            assert refusal is not None, name
            _remedy_names_repeats_one(refusal.message(), f"refute(tests={[name]!r})")
        # The placebo test relearns the rules, so it runs on every split.
        assert refute_refusal(_repeated(), estimand=support.NAME, tests=("placebo",)) is None
        with pytest.raises(CapabilityError):
            refute(_repeated(), estimand=support.NAME, tests=("subset",), n_replicates=2)
        once = support.fit(repeats=1)
        report = refute(once, estimand=support.NAME, tests=("subset",), n_replicates=2)
        assert len(report.tests[0].values) == 2

    def test_the_placebo_test_runs_on_every_split(self) -> None:
        report = refute(_repeated(), estimand=support.NAME, tests=("placebo",), n_replicates=2)
        assert len(report.tests[0].values) == 2

    def test_the_support_report(self) -> None:
        row = _repeated().diagnostics.capability("support")
        assert not row.available
        with pytest.raises(CapabilityError) as caught:
            _repeated().diagnostics.support()
        _remedy_names_repeats_one(str(caught.value), "diagnostics.support()")
        assert support.fit(repeats=1).diagnostics.support() is not None

    def test_the_truncation_curve(self) -> None:
        with pytest.raises(CapabilityError) as caught:
            _repeated().diagnostics.truncation_curve()
        message = str(caught.value)
        _remedy_names_repeats_one(message, "truncation_curve")
        assert "median" in message
        curve = support.fit(repeats=1).diagnostics.truncation_curve()
        assert len(curve.to_frame()) > 0

    def test_the_omitted_variable_bound_refuses_every_learned_rule_fit(self) -> None:
        """X11 (n) holds the bound, so a fit of several splits meets that refusal too."""
        row = _repeated().sensitivity.capability("omitted_confounding")
        assert not row.available and "X11 (n) in docs/roadmap.md" in (row.reason or "")
