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
    Refit with an outcome that the treatment cannot plausibly affect.  Under a valid,
    comparable negative-control design, a non-null estimate can reveal residual bias.
    It can also reflect a bad control or a bias that does not affect the primary outcome.
    A null estimate does not prove that unmeasured confounding is absent.

``dummy_outcome``
    Generate Gaussian noise independent of treatment and adjustment variables, then
    fully refit.  Its declared effect is zero; exclusion from the empirical refit
    distribution reveals a pipeline that manufactures signal from a null outcome.

``simulated_outcome``
    Generate a Gaussian outcome from a recorded adjustment function, additive treatment
    term, and noise law, then fully refit.  The declared treatment effect must lie inside
    the empirical refit distribution.  Failure reveals that the pipeline cannot recover
    a known effect under its first registered generated-outcome process.

Each test refits the model, so a full run costs several times a single fit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from ..exceptions import CapabilityError
from ..inference.bootstrap import Resampling, _bootstrap_design
from ..utils.frames import emit_frame
from ..utils.text import format_table
from .simulation import ReplicationFailure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult

__all__ = [
    "DEFAULT_OUTCOME_REPLICATES",
    "DEFAULT_TESTS",
    "BootstrapMeasurementError",
    "EmpiricalInclusionRule",
    "EmpiricalRefitRecord",
    "GaussianAdjustmentOutcome",
    "GaussianIndependentOutcome",
    "GaussianNoise",
    "GeneratedOutcomeRecord",
    "RefutationResult",
    "RefutationTest",
    "RelativeGaussianNoise",
    "refute",
]

#: Tests run by default.  ``negative_control_outcome`` is excluded because it needs a
#: control outcome only the analyst can supply.
DEFAULT_TESTS: tuple[str, ...] = ("placebo", "random_common_cause", "subset")

#: Generated outcomes need an empirical distribution rather than the small smoke-test
#: budget used by the older perturbations.
DEFAULT_OUTCOME_REPLICATES = 100
_DEFAULT_LEGACY_REPLICATES = 5


@dataclass(frozen=True)
class GaussianNoise:
    """Declare one Gaussian noise law for a generated outcome.

    Parameters
    ----------
    mean : float
        Mean of each independent noise draw.
    standard_deviation : float
        Positive standard deviation of each independent noise draw.
    """

    mean: float = 0.0
    standard_deviation: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("Gaussian noise mean must be finite")
        if not np.isfinite(self.standard_deviation) or self.standard_deviation <= 0.0:
            raise ValueError("Gaussian noise standard_deviation must be finite and positive")

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw independent values from the declared law.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator that supplies the draw.
        size : int
            Number of values to draw.

        Returns
        -------
        ndarray
            Independent Gaussian values.
        """
        return np.asarray(rng.normal(self.mean, self.standard_deviation, size=size), dtype=float)


@dataclass(frozen=True)
class RelativeGaussianNoise:
    """Declare mean-zero numeric noise relative to a sampled variable.

    Parameters
    ----------
    standard_deviation : float
        Nonnegative multiple of the variable's bootstrap-sample standard deviation.
    """

    standard_deviation: float = 0.1

    def __post_init__(self) -> None:
        if not np.isfinite(self.standard_deviation) or self.standard_deviation < 0.0:
            raise ValueError("relative Gaussian standard_deviation must be finite and nonnegative")

    def draw(self, rng: np.random.Generator, values: Any) -> np.ndarray:
        """Draw noise on the declared bootstrap-sample scale.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator that supplies the draw.
        values : array-like
            Sampled numeric variable whose population standard deviation sets the scale.

        Returns
        -------
        ndarray
            One mean-zero Gaussian draw per value.
        """
        sample = np.asarray(values, dtype=float).reshape(-1)
        scale = self.standard_deviation * float(np.std(sample, ddof=0))
        return np.asarray(rng.normal(0.0, scale, size=sample.size), dtype=float)


@dataclass(frozen=True)
class BootstrapMeasurementError:
    """Declare bootstrap measurement-error validation for adjustment variables.

    Parameters
    ----------
    variables : tuple of str
        Original logical adjustment-variable names to perturb.
    numeric_noise : RelativeGaussianNoise
        Relative law for each selected numeric variable.
    categorical_change_probability : float
        Probability that a selected categorical value changes to another level.
    resampling : {"auto", "iid", "cluster"}
        Bootstrap unit. ``"auto"`` uses clusters when the fit declares them.
    """

    variables: tuple[str, ...]
    numeric_noise: RelativeGaussianNoise = RelativeGaussianNoise()
    categorical_change_probability: float = 0.1
    resampling: Resampling = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.variables, str):
            raise ValueError("bootstrap measurement error variables must be a sequence of names")
        object.__setattr__(self, "variables", tuple(self.variables))
        if not self.variables:
            raise ValueError("bootstrap measurement error variables must not be empty")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("bootstrap measurement error variables must not repeat")
        if not all(isinstance(name, str) and name for name in self.variables):
            raise ValueError("bootstrap measurement error variables must be nonempty names")
        if type(self.numeric_noise) is not RelativeGaussianNoise:
            raise ValueError("numeric_noise must be a RelativeGaussianNoise declaration")
        probability = self.categorical_change_probability
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("categorical_change_probability must be between zero and one")
        if self.resampling not in ("auto", "iid", "cluster"):
            raise ValueError("resampling must be 'auto', 'iid', or 'cluster'")


@dataclass(frozen=True)
class GaussianIndependentOutcome:
    """Declare an outcome independent of treatment and adjustment variables.

    Parameters
    ----------
    noise : GaussianNoise
        Gaussian law that generates every outcome value.
    """

    noise: GaussianNoise = GaussianNoise()
    name: str = field(default="gaussian_independent_noise", init=False)
    adjustment_function: str = field(default="none", init=False)
    treatment_term: str = field(default="none", init=False)
    family: str = field(default="gaussian", init=False)
    known_effect: float = field(default=0.0, init=False)

    @property
    def noise_law(self) -> GaussianNoise:
        """Return the recorded Gaussian noise declaration."""
        return self.noise

    def draw(self, data: Any, *, seed: int) -> np.ndarray:
        """Draw one independent outcome vector.

        Parameters
        ----------
        data : CausalData
            Analysis rows that determine the output length.
        seed : int
            Child seed for this draw.

        Returns
        -------
        ndarray
            One generated outcome per analysis row.
        """
        return self.noise.draw(np.random.default_rng(seed), data.n)


@dataclass(frozen=True)
class GaussianAdjustmentOutcome:
    r"""Declare ``f(W) + effect * A + epsilon`` with Gaussian noise.

    The adjustment function standardizes each nonconstant covariate, sums the standardized
    columns, divides by the square root of how many columns it summed, and multiplies the
    result by ``adjustment_scale``. The root-count divisor holds the adjustment standard
    deviation at about ``adjustment_scale`` for any number of approximately independent
    covariates, so the confounding signal does not shrink against the fixed noise scale as
    the adjustment set widens. Correlated columns raise that standard deviation instead,
    which only makes a failure to adjust easier to detect. The known binary arm contrast is
    ``effect`` because the treatment term is additive.

    Parameters
    ----------
    effect : float
        Declared effect of treatment code one against code zero.
    adjustment_scale : float
        Multiplier on the root-count normalized standardized covariate sum.
    noise : GaussianNoise
        Gaussian law for ``epsilon``.
    """

    effect: float = 1.0
    adjustment_scale: float = 1.0
    noise: GaussianNoise = GaussianNoise()
    name: str = field(default="gaussian_adjustment_dependent", init=False)
    adjustment_function: str = field(
        default="scaled_standardized_covariate_sum_over_root_count", init=False
    )
    treatment_term: str = field(default="effect_times_treatment_code", init=False)
    family: str = field(default="gaussian", init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.effect):
            raise ValueError("Gaussian adjustment outcome effect must be finite")
        if not np.isfinite(self.adjustment_scale):
            raise ValueError("Gaussian adjustment outcome adjustment_scale must be finite")

    @property
    def known_effect(self) -> float:
        """Return the derived code-one against code-zero mean contrast."""
        return float(self.effect)

    @property
    def noise_law(self) -> GaussianNoise:
        """Return the recorded Gaussian noise declaration."""
        return self.noise

    def adjustment(self, covariates: Any) -> np.ndarray:
        """Evaluate the declared adjustment function.

        Parameters
        ----------
        covariates : array-like
            Covariate matrix with one row per observation.

        Returns
        -------
        ndarray
            Adjustment value for each row.
        """
        values = np.asarray(covariates, dtype=float)
        center = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        active = scale > 0.0
        if not np.any(active):
            return np.zeros(values.shape[0], dtype=float)
        standardized = (values[:, active] - center[active]) / scale[active]
        # Sum over root count rather than mean: a mean of p standardized columns has
        # standard deviation about 1/sqrt(p), so the confounding signal would vanish
        # against the fixed noise scale on a wide adjustment set.
        divisor = np.sqrt(float(np.count_nonzero(active)))
        return np.asarray(
            self.adjustment_scale * np.sum(standardized, axis=1) / divisor, dtype=float
        )

    def draw(self, data: Any, *, seed: int) -> np.ndarray:
        """Draw one adjustment-dependent outcome vector.

        Parameters
        ----------
        data : CausalData
            Analysis rows that supply covariates and treatment.
        seed : int
            Child seed for the Gaussian noise.

        Returns
        -------
        ndarray
            One generated outcome per analysis row.
        """
        noise = self.noise.draw(np.random.default_rng(seed), data.n)
        return np.asarray(
            self.adjustment(data.covariates) + self.effect * data.treatment + noise,
            dtype=float,
        )


@dataclass(frozen=True)
class EmpiricalInclusionRule:
    """Declare a two-sided empirical inclusion rule for generated outcomes.

    Parameters
    ----------
    alpha : float
        Two-sided tail probability.
    minimum_draws : int
        Minimum successful draw count needed to apply the empirical rule. The rule refuses
        a count below ``2 / alpha``, where the declared level is unreachable.
    failure_policy : {"fail"}
        Policy for a failed refit. The first catalog always fails if any refit fails.
    """

    alpha: float = 0.05
    minimum_draws: int = 40
    failure_policy: Literal["fail"] = "fail"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("empirical inclusion alpha must be between zero and one")
        if self.minimum_draws < 1:
            raise ValueError("empirical inclusion minimum_draws must be positive")
        # The smallest positive value ``pvalue`` can return on n draws is 2/n.  When that
        # exceeds alpha the rule can only reject a sample whose draws all fall on one side
        # of the declared effect, which leaves it with almost no power: it stops reading the
        # tail of the refit distribution and reads only whether the truth is outside its
        # whole range.
        needed = int(np.ceil(2.0 / self.alpha - 1e-9))
        if self.minimum_draws < needed:
            raise ValueError(
                f"empirical inclusion needs minimum_draws >= {needed} at alpha={self.alpha:g}; "
                f"{self.minimum_draws} draws cannot reach a two-sided p-value at or below "
                "alpha except on an all-on-one-side sample"
            )
        if self.failure_policy != "fail":
            raise ValueError("generated-outcome refutations support only failure_policy='fail'")

    def pvalue(self, values: Sequence[float], truth: float) -> float:
        """Return a two-sided empirical rank p-value with inclusive half-ties.

        Parameters
        ----------
        values : sequence of float
            Successful refitted estimates.
        truth : float
            Declared process effect.

        Returns
        -------
        float
            Two-sided empirical rank probability. An empty sample returns ``0.0``, which is
            a rank over nothing rather than a rejection. :meth:`evaluate` reports ``None``
            for that case, and it is the value a report carries.
        """
        estimates = np.asarray(values, dtype=float)
        if estimates.size == 0:
            return 0.0
        below = float(np.count_nonzero(estimates < truth))
        above = float(np.count_nonzero(estimates > truth))
        ties = float(estimates.size - below - above)
        return float(min(1.0, 2.0 * min(below + 0.5 * ties, above + 0.5 * ties) / estimates.size))

    @property
    def minimum_draw_count(self) -> int:
        """Return the minimum successful draw count under this rule."""
        return self.minimum_draws

    def evaluate(
        self,
        values: Sequence[float],
        truth: float,
        failures: Sequence[ReplicationFailure],
    ) -> tuple[bool, float | None, str]:
        """Apply the declared draw-count, failure, and empirical-tail gates.

        Parameters
        ----------
        values : sequence of float
            Successful refitted estimates.
        truth : float
            Declared process effect.
        failures : sequence of ReplicationFailure
            Failed refits retained by the operation.

        Returns
        -------
        tuple of bool, float or None, str
            Pass status, empirical probability, and readable detail. The probability is
            ``None`` when no draw succeeded, because an empty sample carries no rank
            information and a reported zero would read as a strong rejection.
        """
        # ``None`` rather than a number the empty sample cannot support.
        probability = self.pvalue(values, truth) if len(values) else None
        if failures:
            return (
                False,
                probability,
                f"{len(failures)} refit(s) failed; failure_policy='fail' keeps every failure",
            )
        if len(values) < self.minimum_draws:
            return (
                False,
                probability,
                f"only {len(values)} successful draw(s); the rule requires {self.minimum_draws}",
            )
        probability = self.pvalue(values, truth)
        return (
            probability > self.alpha,
            probability,
            f"two-sided empirical p={probability:.5g} with inclusive half-ties "
            f"(alpha={self.alpha:g})",
        )


@dataclass(frozen=True)
class EmpiricalRefitRecord:
    """One successful empirical refit.

    Parameters
    ----------
    replicate : int
        Zero-based draw index.
    seed : int
        Child seed used for both outcome generation and estimator refit.
    estimand : str
        Structured result alias selected for the refit.
    estimate : float
        Refitted point estimate.
    std_error : float
        Refitted standard error.
    family : str
        Outcome family supplied by the refitted result.
    """

    replicate: int
    seed: int
    estimand: str
    estimate: float
    std_error: float
    family: str


# Compatibility alias. Saved reports and callers that imported the established name keep
# the same class object while new empirical operations use the generic declaration.
GeneratedOutcomeRecord = EmpiricalRefitRecord


def _format_number(value: float | None) -> str:
    """Format one optional number for a summary line.

    Parameters
    ----------
    value : float or None
        Number to format.

    Returns
    -------
    str
        Five significant figures, and ``"-"`` for a missing value.
    """
    return "-" if value is None else f"{value:.5g}"


@dataclass(frozen=True)
class RefutationTest:
    """The outcome of one refutation test.

    Parameters
    ----------
    name : str
        Which refutation was run.
    estimand : str
        Alias the test was run for.
    original : float
        The estimate before the refutation.
    values : tuple of float
        Estimates the refutation produced.
    expectation : str
        What those values should look like if the fit is sound.
    passed : bool
        Whether they did.
    detail : str
        What was seen, in the test's own terms.
    standard_errors : tuple of float
        Standard errors from successful generated-outcome refits.
    declaration : object or None
        Immutable outcome-process declaration, for a generated-outcome test.
    rule : EmpiricalInclusionRule or None
        Recorded inclusion rule, for a generated-outcome test.
    records : tuple of EmpiricalRefitRecord
        Successful empirical refit draws.
    failures : tuple of ReplicationFailure
        Failed generated-outcome draws retained by the shared failure contract.
    family : str or None
        Family supplied by successful refitted results.
    empirical_pvalue : float or None
        Two-sided empirical rank p-value for the declared effect. ``None`` when no draw
        succeeded, and on an older refuter.
    declared_effect : float or None
        Effect the declared process implies for this test's own contrast direction, and
        ``None`` for an older refuter. It carries the sign of the parameter key: a fit
        that reports code zero against code one declares the negated process effect.
    requested_draws : int or None
        Requested empirical draw count.
    resampling : str or None
        Resolved bootstrap mode for a bootstrap-based refuter.
    """

    name: str
    estimand: str
    original: float
    values: tuple[float, ...]
    expectation: str
    passed: bool
    detail: str
    standard_errors: tuple[float, ...] = ()
    declaration: Any = None
    rule: EmpiricalInclusionRule | None = None
    records: tuple[EmpiricalRefitRecord, ...] = ()
    failures: tuple[ReplicationFailure, ...] = ()
    family: str | None = None
    empirical_pvalue: float | None = None
    declared_effect: float | None = None
    requested_draws: int | None = None
    resampling: str | None = None

    @property
    def mean(self) -> float:
        """Return the mean estimate across this test's replicates.

        Returns
        -------
        float
            Mean over the finite replicates, and ``nan`` when none are finite. A
            replicate that failed to converge is dropped rather than propagated.
        """
        finite = np.asarray([v for v in self.values if np.isfinite(v)])
        return float(finite.mean()) if finite.size else float("nan")

    @property
    def spread(self) -> float:
        """Return how far this test's replicates spread around their mean.

        Returns
        -------
        float
            Sample standard deviation over the finite replicates, and ``nan`` when
            fewer than two are finite.
        """
        finite = np.asarray([v for v in self.values if np.isfinite(v)])
        return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")

    @property
    def n_failed(self) -> int:
        """Return the number of retained failed generated-outcome draws."""
        return len(self.failures)

    @property
    def n_replicates(self) -> int:
        """Return the number of requested draws represented by this test."""
        if self.declaration is not None:
            return len(self.records) + len(self.failures)
        return len(self.values)

    @property
    def process(self) -> Any:
        """Return the generated outcome declaration, or ``None`` for older refuters."""
        return self.declaration

    @property
    def child_seeds(self) -> tuple[int, ...]:
        """Return every successful and failed draw seed in draw-index order."""
        indexed = [(item.replicate, item.seed) for item in self.records]
        indexed.extend((item.replicate, item.seed) for item in self.failures)
        return tuple(seed for _, seed in sorted(indexed))

    def to_frame(self, data: Any = None, *, backend: str | None = None) -> Any:
        """Return one row per generated-outcome draw.

        Parameters
        ----------
        data : Any
            Dataframe or fitted container whose backend to match.
        backend : str or None
            Backend name used when ``data`` is omitted.

        Returns
        -------
        dataframe
            Successful estimates and standard errors alongside retained failures.
        """
        successes = {item.replicate: item for item in self.records}
        failures = {item.replicate: item for item in self.failures}
        indices = sorted((*successes, *failures))
        payload: dict[str, list[Any]] = {
            "test": [],
            "replicate": [],
            "seed": [],
            "estimate": [],
            "std_error": [],
            "family": [],
            "error_type": [],
            "message": [],
        }
        for index in indices:
            record = successes.get(index)
            failure = failures.get(index)
            # Every index comes from one of the two maps, so exactly one lookup can miss.
            # Read the seed off whichever hit rather than narrowing with an ``assert``,
            # which ``-O`` strips.
            seed = record.seed if record is not None else failures[index].seed
            payload["test"].append(self.name)
            payload["replicate"].append(index)
            payload["seed"].append(seed)
            payload["estimate"].append(None if record is None else record.estimate)
            payload["std_error"].append(None if record is None else record.std_error)
            payload["family"].append(None if record is None else record.family)
            payload["error_type"].append(None if failure is None else failure.error_type)
            payload["message"].append(None if failure is None else failure.message)
        return emit_frame(payload, data, backend=backend)


@dataclass(frozen=True)
class RefutationResult:
    """All refutation tests run on a fit.

    Parameters
    ----------
    tests : tuple of RefutationTest
        One record per refutation run.
    estimand : str
        Alias the tests were run for.
    backend : str or None
        Dataframe backend :meth:`to_frame` returns when ``data`` is omitted.
    random_state : int or None
        Seed this report ran under.  Pass it back to :func:`refute` to obtain the report
        again, whether or not the fit carries a seed of its own.  ``None`` only on a
        report saved before this field existed.
    """

    tests: tuple[RefutationTest, ...]
    estimand: str
    #: Name of the dataframe backend the fit's data arrived in, so that
    #: :meth:`to_frame` honours "results come back in the backend you passed in"
    #: without a caller having to thread the container back in by hand.
    backend: str | None = None
    #: Seed this report ran under, resolved rather than requested: an explicit seed, else
    #: the fit's own, else one drawn here.  It pins the perturbations and the refits they
    #: feed, so the report is enough to obtain the report again.
    random_state: int | None = None

    @property
    def passed(self) -> bool:
        """Return whether every required check passed."""
        return all(test.passed for test in self.tests)

    def __bool__(self) -> bool:
        return self.passed

    def __getitem__(self, name: str) -> RefutationTest:
        for test in self.tests:
            if test.name == name:
                return test
        raise KeyError(f"no test named {name!r}; have {[t.name for t in self.tests]}")

    def to_frame(self, data: Any = None) -> Any:
        """Return tabular output in the input dataframe backend.

        Parameters
        ----------
        data : Any
            A dataframe or fitted container whose backend to match. ``None`` uses the
            backend recorded on this object.

        Returns
        -------
        dataframe
            One row per refutation test.
        """
        payload = {
            "test": [test.name for test in self.tests],
            "estimand": [test.estimand for test in self.tests],
            "original": [test.original for test in self.tests],
            "refuted_mean": [test.mean for test in self.tests],
            "refuted_sd": [test.spread for test in self.tests],
            "declared_effect": [test.declared_effect for test in self.tests],
            "family": [test.family for test in self.tests],
            "successful_draws": [len(test.records) for test in self.tests],
            "failed_draws": [test.n_failed for test in self.tests],
            "empirical_pvalue": [test.empirical_pvalue for test in self.tests],
            "requested_draws": [test.requested_draws for test in self.tests],
            "resampling": [test.resampling for test in self.tests],
            "expectation": [test.expectation for test in self.tests],
            "passed": [test.passed for test in self.tests],
        }
        return emit_frame(payload, data, backend=self.backend)

    def draws_frame(self, name: str, data: Any = None) -> Any:
        """Return successful and failed draws for one generated-outcome test.

        Parameters
        ----------
        name : str
            Name of a generated-outcome refutation in this report.
        data : Any
            Dataframe or fitted container whose backend to match. ``None`` uses the
            backend recorded by the report.

        Returns
        -------
        dataframe
            One row per requested generated-outcome draw.
        """
        test = self[name]
        if test.declaration is None:
            raise ValueError(f"refutation {name!r} has no generated-outcome draw records")
        return test.to_frame(data, backend=self.backend)

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable table, one line per refutation test.
        """
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
            if test.declaration is not None:
                lines.append("")
                if isinstance(test.declaration, BootstrapMeasurementError):
                    variables = ", ".join(test.declaration.variables)
                    lines.append(
                        f"{test.name}: variables={variables}; resampling={test.resampling}; "
                        f"original={_format_number(test.declared_effect)}; "
                        f"successful={len(test.records)}; failed={test.n_failed}"
                    )
                else:
                    lines.append(
                        f"{test.name}: process={test.declaration.name}; family={test.family}; "
                        f"effect={_format_number(test.declared_effect)}; "
                        f"successful={len(test.records)}; failed={test.n_failed}"
                    )
                lines.append(f"rule: {test.detail}")
            if not test.passed and test.declaration is None:
                lines.append("")
                lines.append(f"{test.name}: {test.detail}")
        if self.passed:
            lines.append("")
            lines.append("VERDICT: all refutation tests behaved as they should.")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


_KNOWN_TESTS = (
    "placebo",
    "random_common_cause",
    "subset",
    "negative_control_outcome",
    "dummy_outcome",
    "simulated_outcome",
    "bootstrap_measurement_error",
)
_GENERATED_TESTS = ("dummy_outcome", "simulated_outcome")
_CHILD_SEED_TAGS = {"dummy_outcome": 1, "simulated_outcome": 2}
_ADDITIVE_MEAN_CONTRASTS = {"ate", "att", "atc"}


def _resolve_seed(result: Any, random_state: int | None) -> int:
    estimator = result.estimator
    if random_state is not None:
        return int(random_state)
    if estimator.random_state is not None:
        return int(estimator.random_state)
    return int(np.random.SeedSequence().generate_state(1)[0])


def _generated_child_seeds(root_seed: int, name: str, count: int) -> tuple[int, ...]:
    sequence = np.random.SeedSequence([root_seed, _CHILD_SEED_TAGS[name]])
    return tuple(int(child.generate_state(1)[0]) for child in sequence.spawn(count))


def _classification_only_learner(learner: Any) -> Any | None:
    """Return a saved learner only when it has no regression-capable route."""
    if learner is None:
        return None
    from sklearn.base import is_classifier

    if getattr(learner, "task", None) == "classification" or is_classifier(learner):
        return learner
    library = getattr(learner, "library", None)
    if library is None:
        return None
    incompatible_candidates = []
    for item in library:
        candidate = item[1] if isinstance(item, tuple) and len(item) == 2 else item
        incompatible_candidates.append(_classification_only_learner(candidate) is not None)
    return learner if incompatible_candidates and all(incompatible_candidates) else None


def _validate_generated_process(process: Any, name: str) -> None:
    expected = GaussianIndependentOutcome if name == "dummy_outcome" else GaussianAdjustmentOutcome
    if type(process) is not expected:
        raise CapabilityError(
            f"{name} has no registered effect derivation and family validation for "
            f"{type(process).__name__}; use the exact registered {expected.__name__} declaration"
        )
    if process.family != "gaussian":
        raise CapabilityError(
            f"{name} has family={process.family!r}; only the Gaussian process family has "
            "an implemented effect derivation"
        )


def _resolve_arm_code(data: Any, endpoint: Any, role: str, name: str) -> float:
    """Resolve one parameter-key endpoint to its internal arm code.

    Parameters
    ----------
    data : CausalData
        Analysis rows that declare the arm codes and their labels.
    endpoint : Any
        Label or code recorded on the parameter key.
    role : str
        Which endpoint this is, named in a refusal.
    name : str
        Refutation name, named in a refusal.

    Returns
    -------
    float
        The internal arm code for ``endpoint``.

    Raises
    ------
    CapabilityError
        If no declared arm matches the endpoint.
    """
    # Same label-or-code match the estimator's reference-arm resolution uses, so a key
    # written in the user's own levels and one written in internal codes agree.
    for code, label in zip(data.arm_codes, data.treatment_levels, strict=True):
        if label == endpoint or code == endpoint:
            return float(code)
    raise CapabilityError(
        f"{name} has no effect derivation for {role}={endpoint!r}; it is not an arm of "
        f"{data.treatment_name} whose levels are {list(data.treatment_levels)}"
    )


def _validate_generated_eligibility(
    result: Any, estimand: str, process: Any, name: str
) -> tuple[float, float]:
    """Refuse an ineligible generated-outcome request and resolve its contrast direction.

    Parameters
    ----------
    result : TMLEResult
        The fitted result the refutation would refit.
    estimand : str
        Alias the refutation runs for.
    process : object
        Registered outcome-process declaration.
    name : str
        Refutation name, named in every refusal.

    Returns
    -------
    tuple of float, float
        Value and reference arm codes the parameter key contrasts, in that order.

    Raises
    ------
    CapabilityError
        If the fit, the estimand, or the process has no registered effect derivation.
    """
    _validate_generated_process(process, name)
    from ..study import BackdoorMeanContrast, ExplicitAdjustmentProvider, ParameterKey
    from ..targets import TARGETS

    identified = getattr(result, "identified_effect", None)
    if identified is None:
        raise CapabilityError(
            f"{name} needs identification metadata for a backdoor additive mean contrast; "
            "this legacy fit records none"
        )
    functional = identified.functional
    if type(functional) is not BackdoorMeanContrast:
        raise CapabilityError(
            f"{name} has no outcome process or effect derivation for "
            f"{type(functional).__name__}; use a backdoor-identified effect"
        )
    provider = getattr(identified, "provider", None)
    if type(provider) is not ExplicitAdjustmentProvider:
        raise CapabilityError(
            f"{name} needs registered backdoor provider provenance; "
            f"{type(provider).__name__} does not explicitly provide it"
        )
    data = result.data
    if data.family != process.family:
        raise CapabilityError(
            f"{name} cannot replace an original outcome family={data.family!r} with the "
            f"registered {process.family!r} process; the first catalog requires matching "
            "Gaussian outcome configurations"
        )
    estimator = result.estimator
    configured_family = getattr(estimator, "family", "auto")
    if configured_family not in {"auto", process.family}:
        raise CapabilityError(
            f"{name} cannot refit an estimator configured with family={configured_family!r} "
            f"under the declared {process.family!r} process"
        )
    outcome_learner = getattr(estimator, "outcome_learner", None)
    incompatible_learner = _classification_only_learner(outcome_learner)
    if incompatible_learner is not None:
        raise CapabilityError(
            f"{name} needs a regression-capable saved outcome learner for its Gaussian "
            f"process; {type(incompatible_learner).__name__} is configured for classification"
        )
    if getattr(functional, "longitudinal", False):
        raise CapabilityError(
            f"{name} has no longitudinal outcome process or sequential effect derivation"
        )
    if not getattr(data, "is_binary_treatment", False):
        raise CapabilityError(
            f"{name} has no effect derivation for a non-binary treatment; the first "
            "process catalog covers code one against code zero"
        )
    if getattr(data, "has_intermediate", False) or functional.intermediate is not None:
        raise CapabilityError(
            f"{name} has no controlled-direct-effect outcome process or effect derivation"
        )
    if getattr(data, "has_missing_outcome", False):
        raise CapabilityError(
            f"{name} has no missing-outcome process or observation-model effect derivation"
        )
    if functional.msm is not None or functional.axis == "msm":
        raise CapabilityError(f"{name} has no MSM outcome process or coefficient derivation")
    if functional.axis != "arm" or functional.interventions:
        raise CapabilityError(
            f"{name} has no intervention-indexed outcome process or effect derivation"
        )
    key = result.parameter_keys.get(estimand)
    if type(key) is not ParameterKey:
        raise CapabilityError(
            f"{name} needs a structured parameter key for {estimand!r}; choose one of "
            f"{list(result.parameter_keys)}"
        )
    if key.estimand not in _ADDITIVE_MEAN_CONTRASTS:
        kind = "ratio" if key.estimand in {"rr", "or"} else key.estimand
        raise CapabilityError(
            f"{name} has no {kind} effect derivation; choose an additive mean contrast "
            "(ate, att, or atc)"
        )
    declared_estimand = getattr(getattr(identified, "estimand", None), "name", None)
    if functional.target != key.estimand or declared_estimand != key.estimand:
        raise CapabilityError(
            f"{name} found inconsistent registered target provenance: functional target "
            f"{functional.target!r}, identified estimand {declared_estimand!r}, and parameter "
            f"key {key.estimand!r} must agree"
        )
    registered = TARGETS.get(functional.target)
    if (
        registered is None
        or getattr(identified, "identification", None) != registered.identification
    ):
        raise CapabilityError(
            f"{name} needs the registered identification artifact for target {functional.target!r}"
        )
    # The key names the contrast direction, and the process declares its effect for code
    # one against code zero.  A fit reported against a non-default reference therefore
    # declares the negated effect, and reading the key is the only way to see it.
    reference_endpoint = key.reference
    reference_code = (
        float(data.arm_codes[0])
        if reference_endpoint is None
        else _resolve_arm_code(data, reference_endpoint, "the parameter key reference", name)
    )
    value_endpoint = key.value
    if value_endpoint is None:
        remaining = [float(code) for code in data.arm_codes if float(code) != reference_code]
        if len(remaining) != 1:
            raise CapabilityError(
                f"{name} has no effect derivation for a parameter key with no value arm; "
                f"{data.treatment_name} does not leave exactly one other arm"
            )
        value_code = remaining[0]
    else:
        value_code = _resolve_arm_code(data, value_endpoint, "the parameter key value", name)
    if value_code == reference_code:
        raise CapabilityError(
            f"{name} has no effect derivation for a parameter key whose value and reference "
            f"are the same arm ({value_code:g}); the catalog covers a two-arm contrast"
        )
    return value_code, reference_code


def _run_empirical_refits(
    result: Any,
    *,
    estimand: str,
    draws: Sequence[tuple[int, int]],
    replacement: Callable[[int, int], Any],
    expected_family: str,
) -> tuple[tuple[EmpiricalRefitRecord, ...], tuple[ReplicationFailure, ...]]:
    """Run, validate, and retain empirical refits through one failure contract."""
    records: list[EmpiricalRefitRecord] = []
    failures: list[ReplicationFailure] = []
    for replicate, child_seed in draws:
        prepared = replacement(replicate, child_seed)
        try:
            refitted = result.estimator.refit(
                prepared,
                intermediate_value=result.intermediate_value,
                random_state=child_seed,
            )
        except Exception as error:  # an estimator failure belongs in the report
            failures.append(
                ReplicationFailure(
                    replicate=replicate,
                    seed=child_seed,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            continue
        fitted_family = str(refitted.data.family)
        if fitted_family != expected_family:
            raise RuntimeError(
                f"empirical refit reported authoritative family={fitted_family!r}; expected "
                f"{expected_family!r}"
            )
        estimate = refitted[estimand]
        if not np.isfinite(estimate.psi) or not np.isfinite(estimate.std_error):
            failures.append(
                ReplicationFailure(
                    replicate=replicate,
                    seed=child_seed,
                    error_type="ValueError",
                    message="refitted estimate or standard error is non-finite",
                )
            )
            continue
        records.append(
            EmpiricalRefitRecord(
                replicate=replicate,
                seed=child_seed,
                estimand=estimand,
                estimate=float(estimate.psi),
                std_error=float(estimate.std_error),
                family=fitted_family,
            )
        )
    return tuple(records), tuple(failures)


def _measurement_encodings(data: Any) -> dict[str, Any]:
    return {encoding.column: encoding for encoding in data.encodings}


def _validate_measurement_error_eligibility(
    result: Any, declaration: BootstrapMeasurementError, n_replicates: int
) -> Any:
    """Validate the complete bootstrap measurement-error request before refitting."""
    from ..data import CausalData

    data = getattr(result, "data", None)
    if type(data) is not CausalData:
        raise CapabilityError(
            "bootstrap_measurement_error supports point-treatment CausalData results; "
            f"got result family {type(result).__name__} with data family {type(data).__name__}"
        )
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    if declaration.resampling == "cluster" and data.cluster is None:
        raise CapabilityError("resampling='cluster' requires the data to carry cluster ids")
    selected_strata = set(declaration.variables).intersection(data.strata_names)
    if selected_strata:
        raise CapabilityError(
            "bootstrap_measurement_error cannot perturb selected strata variables while "
            f"preserving target metadata: {sorted(selected_strata)}"
        )
    encodings = _measurement_encodings(data)
    generated = {
        name: encoding.column
        for encoding in data.encodings
        for name in encoding.generated
        if name != encoding.column
    }
    numeric = set(data.covariate_names).difference(generated)
    original = numeric.union(encodings)
    unknown = [name for name in declaration.variables if name not in original]
    if unknown:
        indicator = next((name for name in unknown if name in generated), None)
        if indicator is not None:
            raise CapabilityError(
                f"{indicator!r} is a generated indicator for original categorical variable "
                f"{generated[indicator]!r}; select the original variable"
            )
        raise CapabilityError(
            f"unknown adjustment variables {unknown}; choose from {sorted(original)}"
        )
    retained = set(data.covariate_names)
    for name in declaration.variables:
        encoding = encodings.get(name)
        if encoding is None:
            continue
        missing = [indicator for indicator in encoding.generated if indicator not in retained]
        if missing:
            raise CapabilityError(
                "bootstrap_measurement_error cannot perturb categorical variable "
                f"{name!r} because duplicate-column removal left its encoded block "
                f"incomplete; missing indicators {missing}"
            )
    return data


def _categorical_codes(data: Any, encoding: Any) -> np.ndarray:
    columns = [data.covariate_names.index(name) for name in encoding.generated]
    block = np.asarray(data.covariates[:, columns], dtype=float)
    valid = np.all(np.isin(block, (0.0, 1.0)), axis=1) & (np.sum(block, axis=1) <= 1.0)
    if not np.all(valid):
        raise RuntimeError(
            f"encoded block for {encoding.column!r} is not a valid drop-first indicator block"
        )
    active = np.argmax(block, axis=1) + 1
    return np.where(np.sum(block, axis=1) == 0.0, 0, active).astype(np.int64)


def _perturb_measurement_error(
    data: Any, declaration: BootstrapMeasurementError, *, seed: int
) -> Any:
    """Perturb sampled adjustment variables and rebuild complete categorical blocks."""
    rng = np.random.default_rng(seed)
    values = np.array(data.covariates, dtype=float, copy=True)
    encodings = _measurement_encodings(data)
    for name in declaration.variables:
        encoding = encodings.get(name)
        if encoding is None:
            column = data.covariate_names.index(name)
            values[:, column] += declaration.numeric_noise.draw(rng, values[:, column])
            continue
        columns = [data.covariate_names.index(item) for item in encoding.generated]
        codes = _categorical_codes(data, encoding)
        change = rng.random(data.n) < declaration.categorical_change_probability
        if np.any(change):
            alternatives = rng.integers(0, len(encoding.levels) - 1, size=int(np.sum(change)))
            current = codes[change]
            codes[change] = alternatives + (alternatives >= current)
        rebuilt = np.column_stack(
            [np.asarray(codes == code, dtype=float) for code in range(1, len(encoding.levels))]
        )
        values[:, columns] = rebuilt
    return data.with_covariates(values, name="measurement-error covariates")


def _bootstrap_measurement_error_test(
    result: Any,
    *,
    estimand: str,
    original: float,
    declaration: BootstrapMeasurementError,
    rule: EmpiricalInclusionRule,
    n_replicates: int,
    root_seed: int,
) -> RefutationTest:
    data = _validate_measurement_error_eligibility(result, declaration, n_replicates)
    design = _bootstrap_design(
        data,
        n_replicates=n_replicates,
        resampling=declaration.resampling,
        random_state=root_seed,
    )
    draw_by_replicate = {draw.replicate: draw for draw in design.draws}
    records, failures = _run_empirical_refits(
        result,
        estimand=estimand,
        draws=tuple((draw.replicate, draw.seed) for draw in design.draws),
        replacement=lambda replicate, seed: _perturb_measurement_error(
            design.sample(data, draw_by_replicate[replicate]), declaration, seed=seed
        ),
        expected_family=data.family,
    )
    values = tuple(item.estimate for item in records)
    passed, probability, detail = rule.evaluate(values, original, failures)
    families = {item.family for item in records}
    family = next(iter(families)) if len(families) == 1 else None
    if len(families) > 1:
        passed = False
        detail = f"successful refits reported inconsistent outcome families {sorted(families)}"
    return RefutationTest(
        name="bootstrap_measurement_error",
        estimand=estimand,
        original=original,
        values=values,
        expectation=f"includes {original:.5g}",
        passed=passed,
        detail=detail,
        standard_errors=tuple(item.std_error for item in records),
        declaration=declaration,
        rule=rule,
        records=records,
        failures=failures,
        family=family,
        empirical_pvalue=probability,
        declared_effect=original,
        requested_draws=n_replicates,
        resampling=design.resampling,
    )


def _generated_outcome_test(
    result: Any,
    *,
    name: str,
    estimand: str,
    original: float,
    process: GaussianIndependentOutcome | GaussianAdjustmentOutcome,
    rule: EmpiricalInclusionRule,
    n_replicates: int,
    root_seed: int,
    value_code: float,
    reference_code: float,
) -> RefutationTest:
    data = result.data
    seeds = _generated_child_seeds(root_seed, name, n_replicates)
    records, failures = _run_empirical_refits(
        result,
        estimand=estimand,
        draws=tuple(enumerate(seeds)),
        replacement=lambda _replicate, seed: data.with_outcome(
            process.draw(data, seed=seed), family=process.family
        ),
        expected_family=process.family,
    )
    values = tuple(item.estimate for item in records)
    # The additive treatment term makes the arm contrast linear in the code difference, so
    # the declared effect is the process effect signed by the key's own direction.
    declared_effect = float(process.known_effect * (value_code - reference_code))
    passed, probability, detail = rule.evaluate(values, declared_effect, failures)
    families = {item.family for item in records}
    family = next(iter(families)) if len(families) == 1 else None
    if len(families) > 1:
        passed = False
        detail = f"successful refits reported inconsistent outcome families {sorted(families)}"
    return RefutationTest(
        name=name,
        estimand=estimand,
        original=original,
        values=values,
        expectation=f"includes {declared_effect:.5g}",
        passed=passed,
        detail=detail,
        standard_errors=tuple(item.std_error for item in records),
        declaration=process,
        rule=rule,
        records=records,
        failures=failures,
        family=family,
        empirical_pvalue=probability,
        declared_effect=declared_effect,
        requested_draws=n_replicates,
    )


def refute(
    result: TMLEResult,
    *,
    estimand: str = "ate",
    tests: Sequence[str] = DEFAULT_TESTS,
    n_replicates: int | None = None,
    subset_fraction: float = 0.7,
    negative_control_outcome: Any = None,
    dummy_outcome: GaussianIndependentOutcome | None = None,
    simulated_outcome: GaussianAdjustmentOutcome | None = None,
    outcome_rule: EmpiricalInclusionRule = EmpiricalInclusionRule(),
    bootstrap_measurement_error: BootstrapMeasurementError | None = None,
    measurement_error_rule: EmpiricalInclusionRule = EmpiricalInclusionRule(),
    random_state: int | None = None,
    tolerance: float = 3.0,
) -> RefutationResult:
    """Run refutation tests against a fitted model.

    Parameters
    ----------
    result : TMLEResult
        A fitted result to refute.
    estimand : str
        Alias the tests are run for.
    tests : sequence of str
        Which refutations to run.
    n_replicates : int or None
        Replicates per randomized test. ``None`` uses five for each established
        perturbation and 100 for each generated-outcome test. A generated-outcome test
        refuses a budget below its rule's ``minimum_draws`` before any refit.
    subset_fraction : float
        Share of rows the subset test refits on.
    negative_control_outcome : str or None
        An outcome the treatment cannot affect.  Required to run that test.
    dummy_outcome : GaussianIndependentOutcome or None
        Independent Gaussian process declaration. ``None`` uses the default declaration.
    simulated_outcome : GaussianAdjustmentOutcome or None
        Adjustment-dependent additive process. ``None`` uses the default declaration.
    outcome_rule : EmpiricalInclusionRule
        Recorded rule applied to generated-outcome refits.
    bootstrap_measurement_error : BootstrapMeasurementError or None
        Measurement-error declaration. Required when the named test is requested.
    measurement_error_rule : EmpiricalInclusionRule
        Recorded rule that compares the original estimate with measurement-error refits.
    random_state : int or None
        Seed for the randomised tests.  ``None`` uses the seed the fit was run with, so a
        seeded fit gives the same refutation every time.  An unseeded fit draws a seed, and
        the refits run under it too.  Either way the report records it under
        ``random_state``, and passing that value back repeats the report.
    tolerance : float
        How many standard errors a null test may deviate before failing.  The default of
        3 keeps the false-alarm rate low across several tests.

    Returns
    -------
    RefutationResult
        One record per test run, with what it expected and what it saw.
    """
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError("refute needs the fitted estimator that produced the result")
    if estimand not in result.estimates:
        raise CapabilityError(f"estimand {estimand!r} was not requested in this fit")
    requested = tuple(tests)
    unknown = [name for name in requested if name not in _KNOWN_TESTS]
    if unknown:
        raise ValueError(
            f"unknown refutation test {unknown[0]!r}; choose from {list(_KNOWN_TESTS)}"
        )
    if n_replicates is not None and (
        isinstance(n_replicates, bool)
        or not isinstance(n_replicates, (int, np.integer))
        or n_replicates < 1
    ):
        raise ValueError("n_replicates must be positive and an integer")

    processes: dict[str, GaussianIndependentOutcome | GaussianAdjustmentOutcome] = {
        "dummy_outcome": (GaussianIndependentOutcome() if dummy_outcome is None else dummy_outcome),
        "simulated_outcome": (
            GaussianAdjustmentOutcome() if simulated_outcome is None else simulated_outcome
        ),
    }
    # Validate every generated operation before any requested operation can refit. A mixed
    # call must not pay for placebo fits and only then discover that its outcome process has
    # no effect derivation.
    generated_requested = any(name in _GENERATED_TESTS for name in requested)
    if generated_requested and type(outcome_rule) is not EmpiricalInclusionRule:
        raise CapabilityError(
            "generated-outcome refutations require the exact registered "
            "EmpiricalInclusionRule declaration"
        )
    contrasts: dict[str, tuple[float, float]] = {}
    for name in requested:
        if name in _GENERATED_TESTS:
            contrasts[name] = _validate_generated_eligibility(
                result, estimand, processes[name], name
            )
    measurement_requested = "bootstrap_measurement_error" in requested
    if measurement_requested:
        if type(bootstrap_measurement_error) is not BootstrapMeasurementError:
            raise CapabilityError(
                "bootstrap_measurement_error requires the exact registered "
                "BootstrapMeasurementError declaration"
            )
        if type(measurement_error_rule) is not EmpiricalInclusionRule:
            raise CapabilityError(
                "bootstrap_measurement_error requires the exact registered "
                "EmpiricalInclusionRule declaration"
            )
        measurement_budget = (
            n_replicates if n_replicates is not None else DEFAULT_OUTCOME_REPLICATES
        )
        _validate_measurement_error_eligibility(
            result, bootstrap_measurement_error, measurement_budget
        )
        if measurement_budget < measurement_error_rule.minimum_draws:
            raise CapabilityError(
                f"bootstrap_measurement_error was asked for {measurement_budget} draw(s) "
                f"under a rule that requires {measurement_error_rule.minimum_draws}; raise "
                "n_replicates or declare a rule with a smaller minimum_draws"
            )
    if generated_requested:
        # A draw budget below the rule's own floor can only end in "too few draws", and the
        # caller can see that before any refit is paid for.
        budget = n_replicates if n_replicates is not None else DEFAULT_OUTCOME_REPLICATES
        if budget < outcome_rule.minimum_draws:
            raise CapabilityError(
                f"a generated-outcome refutation was asked for {budget} draw(s) under a rule "
                f"that requires {outcome_rule.minimum_draws}; raise n_replicates to at least "
                f"{outcome_rule.minimum_draws}, or declare a rule with a smaller minimum_draws"
            )

    original = result[estimand].psi
    std_error = result[estimand].std_error
    # The package convention for a stochastic operation on a fitted object: an explicit seed
    # wins, ``None`` inherits the fit's own.  ``is None`` rather than truthiness, because
    # ``random_state=0`` is falsy and still has to win.  Same form as ``ctmle.py:846``.
    #
    # An unseeded fit draws one seed here rather than handing ``None`` to the generator, so
    # the report can name it.  The seed then goes to the refits as well as to the draws
    # below.  A perturbation is only half of a refutation: ``refit`` re-learns the
    # nuisances, and an estimator with no ``random_state`` redraws its folds every time, so
    # seeding the draws alone would leave the report unrepeatable.
    seed = _resolve_seed(result, random_state)
    rng = np.random.default_rng(seed)
    data = result.data
    outcomes: list[RefutationTest] = []

    def refit(replacement: Any) -> float:
        refitted = estimator.refit(
            replacement,
            intermediate_value=result.intermediate_value,
            random_state=seed,
        )
        return refitted[estimand].psi

    for name in requested:
        replicate_count = (
            n_replicates
            if n_replicates is not None
            else (
                DEFAULT_OUTCOME_REPLICATES
                if name in _GENERATED_TESTS or name == "bootstrap_measurement_error"
                else _DEFAULT_LEGACY_REPLICATES
            )
        )
        if name == "placebo":
            values = tuple(
                refit(data.with_treatment(rng.permutation(data.treatment)))
                for _ in range(replicate_count)
            )
            # Permuting treatment removes the effect, so each replicate is a draw from a
            # null distribution whose spread is about the original standard error.
            threshold = tolerance * std_error / np.sqrt(max(1, replicate_count))
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
                for index in range(replicate_count)
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
                for _ in range(replicate_count)
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
            replacement = data.with_outcome(
                negative_control_outcome, name="negative_control_outcome"
            )
            refitted = estimator.refit(
                replacement,
                intermediate_value=result.intermediate_value,
                random_state=seed,
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
                        f"({abs(value) / control_se:.1f} standard errors). Under a valid, "
                        "comparable control design, this flags residual bias. It can also "
                        "indicate that the negative-control assumptions fail"
                    ),
                )
            )

        elif name in _GENERATED_TESTS:
            outcomes.append(
                _generated_outcome_test(
                    result,
                    name=name,
                    estimand=estimand,
                    original=original,
                    process=processes[name],
                    rule=outcome_rule,
                    n_replicates=replicate_count,
                    root_seed=seed,
                    value_code=contrasts[name][0],
                    reference_code=contrasts[name][1],
                )
            )
        else:
            # The preflight above proves the declaration's exact type.
            assert bootstrap_measurement_error is not None
            outcomes.append(
                _bootstrap_measurement_error_test(
                    result,
                    estimand=estimand,
                    original=original,
                    declaration=bootstrap_measurement_error,
                    rule=measurement_error_rule,
                    n_replicates=replicate_count,
                    root_seed=seed,
                )
            )

    return RefutationResult(
        tests=tuple(outcomes),
        estimand=estimand,
        backend=result.data.backend,
        random_state=seed,
    )
