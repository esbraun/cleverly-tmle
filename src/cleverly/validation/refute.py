r"""Refutation tests: try to break the fit on purpose.

Diagnostics tell you about the fit you have.  Refutation tests -- in the spirit of
DoWhy's refuters -- perturb the *problem* in ways whose correct answer is known in
advance, and check that the estimator gives that answer.  They cannot prove the
identification assumptions hold, but they do catch real defects: leakage, a
mis-specified estimand, an implementation that is quietly picking up an artefact.

The tests, and what a failure means:

``placebo``
    Replace treatment with a random permutation of itself, destroying any real
    effect while preserving its marginal distribution.  The estimate should be
    indistinguishable from zero.  A non-null placebo estimate means the pipeline is
    manufacturing an effect -- leakage between folds, or an estimand that is not
    what it claims.

``random_common_cause``
    Add an irrelevant random covariate.  The estimate should barely move; the
    adjustment set is supposed to be robust to noise.  A large shift means the
    nuisance models are unstable at this sample size.

``subset``
    Refit on random subsamples.  The estimates should scatter around the full-sample
    estimate by roughly the standard error.  Scatter far exceeding it points to
    influential observations -- usually the same handful of units the positivity
    diagnostics flag.

``negative_control_outcome``
    Refit with an outcome that the treatment cannot plausibly affect.  A non-null
    estimate is direct evidence of residual confounding, and unlike the other three
    it tests an *identification* assumption rather than the implementation.

Each test refits the model, so a full run costs several times a single fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..estimators.base import format_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = ["RefutationResult", "RefutationTest", "refute"]

#: Tests run by default.  ``negative_control_outcome`` is excluded because it needs a
#: control outcome only the analyst can supply.
DEFAULT_TESTS: tuple[str, ...] = ("placebo", "random_common_cause", "subset")


@dataclass(frozen=True)
class RefutationTest:
    """The outcome of one refutation test."""

    name: str
    estimand: str
    original: float
    values: tuple[float, ...]
    expectation: str
    passed: bool
    detail: str

    @property
    def mean(self) -> float:
        finite = np.asarray([v for v in self.values if np.isfinite(v)])
        return float(finite.mean()) if finite.size else float("nan")

    @property
    def spread(self) -> float:
        finite = np.asarray([v for v in self.values if np.isfinite(v)])
        return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")


@dataclass(frozen=True)
class RefutationResult:
    """All refutation tests run on a fit."""

    tests: tuple[RefutationTest, ...]
    estimand: str

    @property
    def passed(self) -> bool:
        return all(test.passed for test in self.tests)

    def __bool__(self) -> bool:
        return self.passed

    def __getitem__(self, name: str) -> RefutationTest:
        for test in self.tests:
            if test.name == name:
                return test
        raise KeyError(f"no test named {name!r}; have {[t.name for t in self.tests]}")

    def to_frame(self, data: Any = None) -> Any:
        from ..utils.frames import frame_from_dict

        payload = {
            "test": [test.name for test in self.tests],
            "estimand": [test.estimand for test in self.tests],
            "original": [test.original for test in self.tests],
            "refuted_mean": [test.mean for test in self.tests],
            "refuted_sd": [test.spread for test in self.tests],
            "expectation": [test.expectation for test in self.tests],
            "passed": [test.passed for test in self.tests],
        }
        if data is not None:
            return data.frame_like(payload)
        return frame_from_dict(payload)

    def summary(self) -> str:
        lines = [
            f"Refutation tests for {self.estimand!r}",
            "-" * 34,
            format_table(
                ["test", "original", "refuted (mean)", "sd", "expected", "ok"],
                [
                    [
                        test.name,
                        f"{test.original:.5g}",
                        f"{test.mean:.5g}",
                        f"{test.spread:.4g}" if np.isfinite(test.spread) else "-",
                        test.expectation,
                        "yes" if test.passed else "NO",
                    ]
                    for test in self.tests
                ],
            ),
        ]
        for test in self.tests:
            if not test.passed:
                lines.append("")
                lines.append(f"{test.name}: {test.detail}")
        if self.passed:
            lines.append("")
            lines.append("VERDICT: all refutation tests behaved as they should.")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def refute(
    result: TMLEResult,
    *,
    estimand: str = "ate",
    tests: Sequence[str] = DEFAULT_TESTS,
    n_replicates: int = 5,
    subset_fraction: float = 0.7,
    negative_control_outcome: Any = None,
    random_state: int | None = None,
    tolerance: float = 3.0,
) -> RefutationResult:
    """Run refutation tests against a fitted model.

    Parameters
    ----------
    n_replicates:
        Replicates per randomised test.  The default is deliberately small because each
        one is a full refit; raise it when a borderline result needs resolving.
    tolerance:
        How many standard errors a null test may deviate before failing.  The default of
        3 keeps the false-alarm rate low across several tests.
    negative_control_outcome:
        An outcome the treatment cannot affect.  Required to run that test.
    """
    estimator = result.estimator
    if estimator is None:
        raise ValueError("refute needs the fitted estimator that produced the result")
    if estimand not in result.estimates:
        raise ValueError(f"estimand {estimand!r} was not requested in this fit")

    original = result[estimand].psi
    std_error = result[estimand].std_error
    rng = np.random.default_rng(random_state)
    data = result.data
    outcomes: list[RefutationTest] = []

    def refit(replacement: Any) -> float:
        refitted = estimator._fit_single(replacement, intermediate_value=result.intermediate_value)
        return refitted[estimand].psi

    for name in tests:
        if name == "placebo":
            values = tuple(
                refit(data.with_treatment(rng.permutation(data.treatment)))
                for _ in range(n_replicates)
            )
            # Permuting treatment removes the effect, so each replicate is a draw from a
            # null distribution whose spread is about the original standard error.
            threshold = tolerance * std_error / np.sqrt(max(1, n_replicates))
            mean = float(np.mean(values))
            passed = bool(abs(mean) <= max(threshold, tolerance * std_error / 2))
            outcomes.append(
                RefutationTest(
                    name=name,
                    estimand=estimand,
                    original=original,
                    values=values,
                    expectation="~ 0",
                    passed=passed,
                    detail=(
                        f"mean placebo estimate {mean:+.5g} is more than {tolerance:g} standard "
                        "errors from zero, which suggests the pipeline is producing an effect "
                        "where none exists (fold leakage, or a misdefined estimand)"
                    ),
                )
            )

        elif name == "random_common_cause":
            values = tuple(
                refit(data.with_extra_covariate(rng.normal(size=data.n), f"_noise_{index}"))
                for index in range(n_replicates)
            )
            shift = float(np.mean(values)) - original
            passed = bool(abs(shift) <= tolerance * std_error)
            outcomes.append(
                RefutationTest(
                    name=name,
                    estimand=estimand,
                    original=original,
                    values=values,
                    expectation=f"~ {original:.4g}",
                    passed=passed,
                    detail=(
                        f"adding an irrelevant covariate moved the estimate by {shift:+.5g} "
                        f"({abs(shift) / std_error:.1f} standard errors); the nuisance models are "
                        "unstable at this sample size"
                    ),
                )
            )

        elif name == "subset":
            size = max(20, int(subset_fraction * data.n))
            values = tuple(
                refit(data.subset(rng.choice(data.n, size=size, replace=False)))
                for _ in range(n_replicates)
            )
            expected = std_error * np.sqrt(1.0 / subset_fraction - 1.0 + 1.0)
            spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            passed = bool(spread <= tolerance * expected)
            outcomes.append(
                RefutationTest(
                    name=name,
                    estimand=estimand,
                    original=original,
                    values=values,
                    expectation=f"scatter <~ {tolerance * expected:.4g}",
                    passed=passed,
                    detail=(
                        f"subsample estimates scatter by {spread:.5g}, far more than the "
                        f"{expected:.5g} expected from sampling alone; a few influential "
                        "observations are driving the estimate"
                    ),
                )
            )

        elif name == "negative_control_outcome":
            if negative_control_outcome is None:
                raise ValueError(
                    "the negative_control_outcome test needs an outcome array that treatment "
                    "cannot affect; pass negative_control_outcome=<array>"
                )
            control = np.asarray(negative_control_outcome, dtype=float).reshape(-1)
            if control.shape[0] != data.n:
                raise ValueError(
                    f"negative_control_outcome has length {control.shape[0]}, expected {data.n}"
                )
            from dataclasses import replace as dataclass_replace

            from ..data.validate import infer_family

            replacement = dataclass_replace(
                data,
                outcome=np.where(data.observed, control, 0.0),
                family=infer_family(control, data.observed),
            )
            refitted = estimator._fit_single(
                replacement, intermediate_value=result.intermediate_value
            )
            value = refitted[estimand].psi
            control_se = refitted[estimand].std_error
            passed = bool(abs(value) <= tolerance * control_se)
            outcomes.append(
                RefutationTest(
                    name=name,
                    estimand=estimand,
                    original=original,
                    values=(value,),
                    expectation="~ 0",
                    passed=passed,
                    detail=(
                        f"the negative-control outcome shows an effect of {value:+.5g} "
                        f"({abs(value) / control_se:.1f} standard errors). Unlike the other "
                        "tests this is evidence about identification: residual confounding is "
                        "the most likely explanation"
                    ),
                )
            )

        else:
            raise ValueError(
                f"unknown refutation test {name!r}; choose from "
                f"{['placebo', 'random_common_cause', 'subset', 'negative_control_outcome']}"
            )

    return RefutationResult(tests=tuple(outcomes), estimand=estimand)
