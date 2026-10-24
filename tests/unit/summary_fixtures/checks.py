"""Check and refutation reports whose summaries the contract test checks (X89 T3).

One fixture per kind of row, table and note a refutation report can print, so each of its
note codes fires on at least one fixture. Fits are compact (300 rows) with linear and
logistic learners, and each test runs the fewest repeats it accepts. The failed and passed
reports are built from real tests with ``dataclasses.replace``: a correct fit at this size
gives no pass or fail, and the summary has to print the other two results too.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly.datasets import make_binary_outcome, make_linear_ate
from cleverly.estimators import TMLE
from cleverly.validation.outcome_processes import GaussianAdjustmentOutcome
from cleverly.validation.refute import (
    BootstrapMeasurementError,
    RefutationResult,
    refute,
)
from cleverly.validation.simulation import ReplicationFailure

from . import Case
from ._check_fidelity import read_refutation

__all__ = ["FIXTURES"]

_COVARIATES = ("W1", "W2", "W3", "W4")


def _learners() -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "random_state": 0,
    }


@functools.cache
def _frame() -> Any:
    return make_linear_ate(n=300, seed=1)[0]


@functools.cache
def _studied() -> Any:
    """A fit through a study, which records what the generated-outcome tests need."""
    design = PointTreatment(outcome="Y", treatment="A", adjustment=_COVARIATES)
    return CausalStudy(_frame(), design=design).identify(ATE()).estimate(**_learners())


@functools.cache
def default_tests() -> RefutationResult:
    """The three default tests, none of which gives a pass or fail at this size."""
    return refute(_studied(), n_replicates=10, random_state=3)


@functools.cache
def ratio() -> RefutationResult:
    """The default tests of a risk ratio, which print geometric means and a log SD."""
    frame = make_binary_outcome(n=300, seed=1)[0]
    learners = {**_learners(), "outcome_learner": LogisticRegression(max_iter=1000)}
    fit = TMLE(**learners).fit(frame, outcome="Y", treatment="A").single()
    return refute(fit, estimand="rr", n_replicates=10, random_state=3)


@functools.cache
def generated() -> RefutationResult:
    """The noise and simulated outcomes, which print readings against their bands."""
    return refute(
        _studied(),
        tests=("dummy_outcome", "simulated_outcome"),
        n_replicates=10,
        simulated_outcome=GaussianAdjustmentOutcome(),
        random_state=3,
    )


@functools.cache
def negative_control() -> RefutationResult:
    """One negative-control refit, which prints a z statistic."""
    noise = np.random.default_rng(0).normal(size=300)
    control = 0.3 * np.asarray(_frame()["W1"], dtype=float) + noise
    return refute(
        _studied(),
        tests=("negative_control_outcome",),
        negative_control_outcome=control,
        random_state=3,
    )


@functools.cache
def measurement_error() -> RefutationResult:
    """Measurement error on one covariate, at the fewest draws its default rule accepts."""
    return refute(
        _studied(),
        tests=("bootstrap_measurement_error",),
        n_replicates=40,
        bootstrap_measurement_error=BootstrapMeasurementError(("W1",)),
        random_state=3,
    )


@functools.cache
def stability_and_plasmode() -> RefutationResult:
    """A bootstrap of whole units and a plasmode outcome, the two tests that resample rows
    the way the fit's design does."""
    return refute(
        _studied(), tests=("bootstrap_stability", "plasmode"), n_replicates=10, random_state=3
    )


def failed() -> RefutationResult:
    """A failed shuffled-treatment test, a measurement-error test with two failed draws, and
    a subsample test whose refits report only a plug-in standard error."""
    tests = {test.name: test for test in default_tests().tests}
    error = measurement_error().tests[0]
    lost = tuple(
        ReplicationFailure(record.replicate, record.seed, "LinAlgError", "singular matrix")
        for record in error.records[:2]
    )
    return replace(
        default_tests(),
        tests=(
            replace(tests["placebo"], passed=False, rung="L2", evidence="A study sets it."),
            replace(error, records=error.records[2:], failures=lost),
            replace(tests["subset"], inference="estimated_mechanism_plugin"),
        ),
    )


def passed() -> RefutationResult:
    """Every test passed, which only a fit inside a test's studied scope can print."""
    return replace(
        default_tests(),
        tests=tuple(
            replace(test, passed=True, rung="L2", evidence="A simulation study sets it.")
            for test in default_tests().tests
        ),
    )


def _case(build: Any, names: tuple[str, ...] = ()) -> Case:
    return Case(build, names=(*_COVARIATES, *names), fidelity=read_refutation)


FIXTURES = {
    "refute_default": _case(default_tests),
    "refute_ratio": _case(ratio),
    "refute_generated": _case(generated),
    "refute_negative_control": _case(negative_control),
    "refute_measurement_error": _case(measurement_error),
    "refute_stability_and_plasmode": _case(stability_and_plasmode),
    "refute_failed": _case(failed),
    "refute_passed": _case(passed),
}
