r"""Refutation tests: try to break the fit on purpose.

Diagnostics tell you about the fit you have.  Refutation tests -- in the spirit of
DoWhy's refuters -- perturb the *problem* in ways whose correct answer is known in
advance, and read what the estimator gives.  They cannot prove the identification
assumptions hold, but they can show a defect: leakage, a mis-specified estimand, an
implementation that is quietly picking up an artefact.

A test prints a verdict only where evidence licenses its rule.  Each test computes a
statistic against a stated reference.  :mod:`cleverly.validation._interpretation` holds
the registry of evidence, and a rule that no registered row covers prints its numbers and
no verdict (``passed`` is ``None``).  The rungs are ``L0`` (a description), ``L1`` (a stated
reference with no measured size), ``L2`` (an exact reference, or a measured size and power
on the fit's configuration) and ``L3`` (``L2`` with a comparator beside it).

The tests, and what a reading means:

``placebo``
    Replace treatment with a random permutation of itself (or independent draws from its
    marginal), destroying any real effect while preserving its marginal distribution.  The
    refit is exactly null on the permuted law, so the one-sample ``t`` of the draws against
    the null value is the reading.  It checks that the estimator returns the null when
    treatment is independent of covariates and outcome.  It says nothing about confounding
    in your data.

``random_common_cause``
    Add an irrelevant random covariate.  The target does not change, so the paired shift
    (each noise refit against the original fit, under its seed and folds) is a description of the
    learners' stability at this sample size.

``subset``
    Refit on random subsamples that keep the original fold labels of the drawn units.  Given
    the data, the scatter of an asymptotically linear estimator has the reference
    ``SE * sqrt((n/m - 1) n/(n-1))``, and ``(R - 1) rho^2`` is approximately chi-squared.
    A large ratio reads "the estimator varies more than its standard error says".

``negative_control_outcome``
    Refit with an outcome that the treatment cannot plausibly affect.  Under a valid,
    comparable negative-control design, a non-null estimate can reveal residual bias.
    It can also reflect a bad control or a bias that does not affect the primary outcome.
    A null estimate does not prove that unmeasured confounding is absent.

``dummy_outcome``
    Generate Gaussian noise independent of treatment and adjustment variables, hold the
    design, and refit ``n_replicates`` times.  Its declared effect is zero.  The draws are
    i.i.d. given the held design, so the bias, the standard-error ratio and the coverage have
    exact Monte Carlo references.  Each reading passes when its 95% interval lies inside an
    equivalence band (:class:`EquivalenceBands`).

``simulated_outcome``
    The same readings for a Gaussian outcome generated from a recorded adjustment function,
    additive treatment term, and noise law.

``bootstrap_measurement_error``
    Draw a bootstrap sample, add declared measurement error to the selected adjustment
    variables, then fully refit.  The noise changes the target, so the draws describe how far
    the estimate moves under that error.  A draw that fails to sample, perturb, or refit is
    retained as a failure rather than raised, because a bootstrap draw fails per sample.

``plasmode``
    Resample units from the empirical law of the covariates, draw treatment from a frozen
    model and the outcome from the ``simulated_outcome`` declaration, then refit.  The truth
    is a finite sum over the support points, so bias, standard-error ratio and coverage are
    read against their equivalence bands.  Arm targets, regimes and MSM coefficients have an
    exact truth in both designs.  An incremental tilt reads the true treatment mechanism, so
    only the plasmode, whose frozen model is that mechanism, reads it.  A modified treatment
    policy moves the natural treatment: the held design reads each row's held arm or dose,
    and the plasmode of an arm treatment reads its frozen model.  On an end-of-study
    longitudinal panel the plasmode draws the whole history from frozen node models, and its
    truth is the sequential g-formula of that law.

``bootstrap_stability``
    Refit on bootstrap samples with no added noise.  Each copy of a unit keeps the unit's
    original fold and every refit reuses the original fit's seeds, so the shift of the mean
    against the original fit carries no fold or seed noise.  The report is descriptive.

Each test refits the model, so a full run costs several times a single fit.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .._inference_status import InferenceStatus, supplies_inference
from .._summary import Detail, Summary, check_detail
from ..exceptions import NO_ALTERNATIVE, CapabilityError, CleverlyError, Refusal, RefusalReason
from ..inference.bootstrap import Resampling, _bootstrap_design
from ..inference.delta import two_sided_pvalue
from ..inference.influence import spread_name
from ..utils.frames import emit_frame
from ..utils.random import resolve_assessment_seed
from . import _interpretation
from ._band_placebo import (
    BAND_SENTENCE,
    NEAREST_DONORS,
    PanelBands,
    PointBands,
    placebo_bands,
)
from ._interpretation import FitConfiguration, Resolution, Rung
from ._markov_placebo import MarkovKernel, placebo_known_mechanisms
from ._point_generated import (
    PROJECTION_LINKS,
    InterventionLaw,
    PointTarget,
    Projection,
    TargetKind,
    arm_means,
    held_mixture,
    held_weights,
)
from ._point_generated import PlasmodeLaw as _PlasmodeLaw
from ._readings import (
    DEFAULT_BAND_REPLICATES,
    BandReading,
    EquivalenceBands,
    SubsetReading,
    band_readings,
    paired_shift,
    placebo_reading,
    subset_reading,
)
from ._structural_refute import (
    is_structural,
    structural_fit_seeds,
    structural_frozen_folds,
    structural_null,
    structural_subset_refusal,
    structural_tests_refusal,
)
from .outcome_processes import (
    BernoulliAdjustmentOutcome,
    BoundedAdjustmentOutcome,
    GaussianAdjustmentOutcome,
    GaussianIndependentOutcome,
    GaussianNoise,
    LogisticHazardOutcome,
    MultinomialHazardOutcome,
    arm_term,
)
from .simulation import ReplicationFailure

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators.base import TMLEResult
    from ..interventions.learned import FrozenLearnedRule

__all__ = [
    "DEFAULT_OUTCOME_REPLICATES",
    "DEFAULT_TESTS",
    "BandReading",
    "BootstrapMeasurementError",
    "EmpiricalInclusionRule",
    "EmpiricalRefitRecord",
    "EquivalenceBands",
    "GaussianAdjustmentOutcome",
    "GaussianIndependentOutcome",
    "GaussianNoise",
    "RefutationResult",
    "RefutationTest",
    "RelativeGaussianNoise",
    "refute",
]

#: Tests run by default.  ``negative_control_outcome`` is excluded because it needs a
#: control outcome only the analyst can supply.
DEFAULT_TESTS: tuple[str, ...] = ("placebo", "random_common_cause", "subset")

#: A generated-outcome report needs 400 draws: at that count a calibrated estimator passes
#: all three equivalence readings with probability 0.93, and at 100 draws with probability
#: 0.003 (X82 plan 4.2.1). Below it the readings print at ``L1`` and state no verdict.
DEFAULT_OUTCOME_REPLICATES = DEFAULT_BAND_REPLICATES

#: Draws of ``bootstrap_measurement_error``, a description with no verdict rule.
_DEFAULT_MEASUREMENT_REPLICATES = 100

#: Eric, 2026-10-22 01:51Z: a check that needs more than 20 refits is opt-in. The default
#: placebo and subset therefore refit 20 times (``t`` with 19 degrees of freedom), which is
#: the count the short studies calibrate.
_DEFAULT_PERTURBATION_REPLICATES = 20

#: The random-common-cause report refits each noise draw under the original fit's seed, so the
#: original fit is the paired baseline and the report costs one refit per draw.
_DEFAULT_NOISE_REPLICATES = 20

#: The fewest draws at which the equivalence intervals are defined and worth printing.
_MINIMUM_BAND_DRAWS = 10


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
        Original logical adjustment-variable names to perturb. A point fit holds a declared
        strata variable at its recorded values. A longitudinal fit perturbs a baseline or
        node covariate at every node where it appears; a long panel's column ``l`` covers
        ``l@1``, ``l@2`` and so on.
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
        try:
            selected = tuple(self.variables)
        except TypeError:
            # ``tuple`` on a non-iterable raises ``TypeError``, which would escape the
            # module's ``ValueError`` convention for a rejected declaration.
            raise ValueError(
                "bootstrap measurement error variables must be a sequence of names"
            ) from None
        object.__setattr__(self, "variables", selected)
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
        Child seed for this draw. It seeds the estimator refit, and the perturbation the
        operation applies before it: outcome generation, or bootstrap measurement error.
    estimand : str
        Structured result alias selected for the refit.
    estimate : float
        Refitted point estimate.
    std_error : float
        Refitted standard error, read through
        :attr:`~cleverly.ParameterEstimate.plugin_std_error`. On a fit that supplies no
        inference it is that diagnostic, and not a standard error.
    family : str
        Outcome family supplied by the refitted result.
    """

    replicate: int
    seed: int
    estimand: str
    estimate: float
    std_error: float
    family: str


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
    passed : bool or None
        Whether they did, where evidence licenses a verdict. ``None`` is a descriptive
        reading: the statistic is printed against its reference, and no registered
        calibration covers this fit (:attr:`rung`, :attr:`scope_sentence`).
    detail : str
        What was seen, in the test's own terms.
    standard_errors : tuple of float
        Standard errors from successful empirical refits.
    declaration : object or None
        Immutable declaration, for an empirical refit test. It holds the outcome process,
        or the measurement-error request.
    rule : EmpiricalInclusionRule or None
        Recorded inclusion rule, for an empirical refit test.
    records : tuple of EmpiricalRefitRecord
        Successful empirical refit draws.
    failures : tuple of ReplicationFailure
        Failed empirical draws retained by the shared failure contract. Every operation
        retains a refit failure. ``bootstrap_measurement_error`` also retains a draw it
        cannot sample or perturb.
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
    inference : str
        Inference status of the refitted estimates. One of
        :data:`~cleverly.inference.influence.InferenceStatus`, which the
        :doc:`inference reference </technical-reference/inference>` lists. On a fit
        that supplies no inference, ``standard_errors`` and each record's ``std_error``
        hold the plug-in diagnostic
        :attr:`~cleverly.ParameterEstimate.plugin_std_error`, and :meth:`to_frame`
        publishes that column as ``plugin_std_error``.
    mean_kind : str
        ``"geometric"`` for a ratio placebo or noise refutation tested on the log
        scale. Other tests use the arithmetic mean. The underlying ``values`` remain
        on the reported scale.
    spread_kind : str
        ``"log"`` for ratio subset scatter tested on the log scale. Other reports use
        reported-scale scatter. The underlying ``values`` remain on the reported scale.
    rung : {"L0", "L1", "L2", "L3"}
        The evidence rung the verdict rule stands on for this fit. ``"L0"`` is a
        description, and a verdict needs ``"L2"`` or above.
    scope_sentence : str
        The only interpretation text this test prints: the covering registry row's sentence,
        or the line that names the covered configuration.
    statistic : float or None
        The test statistic: ``t`` (placebo), the scatter ratio ``rho`` (subset), the paired
        shift in standard errors (random common cause), or ``z`` (negative control).
    statistic_name : str
        Name of :attr:`statistic`.
    statistic_pvalue : float or None
        Two-sided probability of :attr:`statistic` under its stated reference law, printed
        for the reader. A descriptive reading states no verdict from it.
    readings : tuple of BandReading
        The equivalence readings of a generated-outcome report, each with its interval and
        band.
    extras : tuple of tuple
        Further named numbers the test prints beside its statistic, such as the scale ratio
        of the placebo draws.
    conditions : tuple of str
        The sentences that say what a reading of this test can and cannot show, which its
        printed summary keeps beside the numbers.
    evidence : str
        What sets this test's pass or fail, or how this fit differs from the fits a study
        covers, in plain words.
    withheld : str
        Why a test that a study covers for this kind of fit gives no pass or fail on this
        report, as a clause in plain words. Empty when it gives one, or when no study
        covers it.
    """

    name: str
    estimand: str
    original: float
    values: tuple[float, ...]
    expectation: str
    passed: bool | None
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
    inference: InferenceStatus = "influence_curve"
    mean_kind: Literal["arithmetic", "geometric"] = "arithmetic"
    spread_kind: Literal["reported", "log"] = "reported"
    rung: Rung = "L0"
    scope_sentence: str = ""
    statistic: float | None = None
    statistic_name: str = ""
    statistic_pvalue: float | None = None
    readings: tuple[BandReading, ...] = ()
    extras: tuple[tuple[str, float], ...] = ()
    conditions: tuple[str, ...] = ()
    evidence: str = ""
    withheld: str = ""

    @property
    def verdict(self) -> str:
        """Return ``"passed"``, ``"failed"`` or ``"descriptive"``."""
        if self.passed is None:
            return "descriptive"
        return "passed" if self.passed else "failed"

    @property
    def mean(self) -> float:
        """Return the displayed mean of refitted estimates.

        Returns
        -------
        float
            Arithmetic mean by default, or geometric mean for a ratio placebo or noise
            refutation.
            ``nan`` when no replicate is finite. Failed replicates are dropped.
        """
        finite = np.asarray([v for v in self.values if np.isfinite(v)])
        if not finite.size:
            return float("nan")
        if self.mean_kind == "geometric":
            return float(np.exp(np.mean(np.log(finite))))
        return float(finite.mean())

    @property
    def spread(self) -> float:
        """Return how far this test's replicates spread on the verdict scale.

        Returns
        -------
        float
            Sample standard deviation over the finite replicates, on the log scale
            for a ratio subset test. Returns ``nan`` with fewer than two values.
        """
        finite = np.asarray([v for v in self.values if np.isfinite(v)])
        if self.spread_kind == "log":
            finite = np.log(finite)
        return float(finite.std(ddof=1)) if finite.size > 1 else float("nan")

    @property
    def n_failed(self) -> int:
        """Return the number of retained failed empirical draws."""
        return len(self.failures)

    @property
    def n_replicates(self) -> int:
        """Return the number of requested draws represented by this test."""
        if self.declaration is not None:
            return len(self.records) + len(self.failures)
        return len(self.values)

    @property
    def process(self) -> Any:
        """Return the empirical refit declaration, or ``None`` for older refuters."""
        return self.declaration

    @property
    def child_seeds(self) -> tuple[int, ...]:
        """Return every successful and failed draw seed in draw-index order."""
        indexed = [(item.replicate, item.seed) for item in self.records]
        indexed.extend((item.replicate, item.seed) for item in self.failures)
        return tuple(seed for _, seed in sorted(indexed))

    def __repr__(self) -> str:
        from ._refute_summary import test_repr

        return test_repr(self)

    def to_frame(self, data: Any = None, *, backend: str | None = None) -> Any:
        """Return one row per empirical refit draw.

        Parameters
        ----------
        data : Any
            Dataframe or fitted container whose backend to match.
        backend : str or None
            Backend name used when ``data`` is omitted.

        Returns
        -------
        dataframe
            Successful estimates and standard errors alongside retained failures. The
            standard-error column is ``plugin_std_error`` when :attr:`inference` is a
            diagnostic status.
        """
        successes = {item.replicate: item for item in self.records}
        failures = {item.replicate: item for item in self.failures}
        indices = sorted((*successes, *failures))
        error_column = spread_name("std_error", self.inference)
        payload: dict[str, list[Any]] = {
            "test": [],
            "replicate": [],
            "seed": [],
            "estimate": [],
            error_column: [],
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
            payload[error_column].append(None if record is None else record.std_error)
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
        report built by hand.
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
    def passed(self) -> bool | None:
        """Return the verdict of the whole report.

        Returns
        -------
        bool or None
            ``False`` when any test failed. ``None`` when no test failed and at least one is
            descriptive, because a descriptive test states no verdict. ``True`` when every
            test passed.
        """
        if any(test.passed is False for test in self.tests):
            return False
        if any(test.passed is None for test in self.tests):
            return None
        return True

    def __bool__(self) -> bool:
        verdict = self.passed
        if verdict is None:
            raise TypeError(
                "this refutation report has a descriptive test, so its truth value is "
                "undefined. Read report.tests and each test's rung and scope_sentence"
            )
        return verdict

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
            "statistic_name": [test.statistic_name or None for test in self.tests],
            "statistic": [test.statistic for test in self.tests],
            "statistic_pvalue": [test.statistic_pvalue for test in self.tests],
            "rung": [test.rung for test in self.tests],
            "verdict": [test.verdict for test in self.tests],
            "passed": [test.passed for test in self.tests],
        }
        return emit_frame(payload, data, backend=self.backend)

    def draws_frame(self, name: str, data: Any = None) -> Any:
        """Return successful and failed draws for one empirical refit test.

        Parameters
        ----------
        name : str
            Name of a generated-outcome or measurement-error refutation in this report.
        data : Any
            Dataframe or fitted container whose backend to match. ``None`` uses the
            backend recorded by the report.

        Returns
        -------
        dataframe
            One row per requested empirical draw.
        """
        test = self[name]
        if test.declaration is None:
            raise ValueError(f"refutation {name!r} has no empirical draw records")
        return test.to_frame(data, backend=self.backend)

    def summary(self, detail: Detail = "short") -> Summary:
        """Return the printed summary of the tests.

        Parameters
        ----------
        detail : {"short", "full"}
            ``"short"`` prints the header, one row per test, the statistics behind each
            reading and one note per test. ``"full"`` adds what each test was compared
            with and the repeats of each test.

        Returns
        -------
        Summary
            A ``str`` whose value is the printed text. Its ``tables``, ``notes`` and
            ``reading`` hold the parts as data. A test that gives no pass or fail prints
            "no pass or fail" in its result column.
        """
        from ._refute_summary import refutation_summary

        return refutation_summary(self, check_detail(detail))

    def _repr_html_(self) -> str:
        return self.summary().as_html()

    def __repr__(self) -> str:
        from ._refute_summary import report_repr

        return report_repr(self)


_KNOWN_TESTS = (
    "placebo",
    "random_common_cause",
    "subset",
    "negative_control_outcome",
    "dummy_outcome",
    "simulated_outcome",
    "bootstrap_measurement_error",
    "bootstrap_stability",
    "plasmode",
    "future_treatment",
)
#: The negative-control outcome forms: ``None`` for an outcome measured after treatment, and
#: ``"pre_treatment"`` for one measured before the first treatment (X82 plan 5.7.1).
_NEGATIVE_CONTROL_KINDS = (None, "pre_treatment")
_GENERATED_TESTS = ("dummy_outcome", "simulated_outcome", "plasmode")
#: The refutations whose refit sees a different row set from the fit's own. ``subset`` drops
#: rows, so the count changes; ``bootstrap_measurement_error`` resamples with replacement
#: through :meth:`~cleverly.inference.bootstrap._BootstrapDesign.sample`, so the count is
#: unchanged but a different unit sits at each position -- the same reason the targeted
#: bootstrap refuses a supplied plan outright. Every other test here replaces or perturbs a
#: column and keeps each row in its position, which is what makes a supplied
#: :class:`~cleverly.SplitPlan` still apply.
_ROW_SET_TESTS = ("subset", "bootstrap_measurement_error", "bootstrap_stability")
#: ``subset`` refits on the original fold labels of the drawn units, so a supplied plan
#: labels the refit's rows by position and the test does not refuse it.
_PLAN_REFUSING_TESTS = ("bootstrap_measurement_error", "bootstrap_stability")
_CHILD_SEED_TAGS = {
    "dummy_outcome": 1,
    "simulated_outcome": 2,
    "random_common_cause": 3,
    "plasmode": 4,
}
_ADDITIVE_MEAN_CONTRASTS = {"ate", "att", "atc"}


def _permuted_treatment(data: Any, rng: np.random.Generator) -> np.ndarray:
    """The placebo treatment: the recorded codes permuted among the recorded rows.

    A vector treatment with a continuous component permutes its component rows.  A row
    with an unrecorded treatment keeps its ``NaN`` code, so the placebo keeps the treatment
    observation pattern and permutes only what was recorded.  Without a missing treatment
    this is a permutation of the whole column, drawn exactly as before.
    """
    if data.treatment_kind == "vector":
        # The rows of the component block move together, so the placebo keeps the joint
        # law of the components and breaks only their link to the covariates and outcome.
        return np.asarray(rng.permutation(data.treatment_components, axis=0), dtype=float)
    if not data.has_missing_treatment:
        return np.asarray(rng.permutation(data.treatment), dtype=float)
    recorded = data.treatment_recorded
    permuted = np.asarray(data.treatment, dtype=float).copy()
    permuted[recorded] = rng.permutation(permuted[recorded])
    return permuted


def _validated_tests(
    tests: Sequence[str],
    n_replicates: int | None,
    negative_control_outcome: Any = None,
    placebo_mode: str | None = None,
    negative_control_kind: str | None = None,
    negative_control_floor: float = 0.01,
    future_treatment_control: Any = None,
    future_treatment_node: int = 1,
) -> tuple[str, ...]:
    """Refuse malformed arguments of :func:`refute` before any refit.

    A malformed argument is reported before any refusal, so a caller who misspelled a test
    name hears that before a refusal that the corrected name might not meet.  :func:`refute`
    and the ``refute`` method of :class:`~cleverly.assessment.DiagnosticsFacade` each call
    this first, and :func:`refute_refusal` returns ``None`` where it raises.

    Parameters
    ----------
    tests : sequence of str
        The requested test names.
    n_replicates : int or None
        Replicates per randomized test.
    negative_control_outcome : Any
        The outcome required by the negative-control test.
    placebo_mode : str or None
        How the placebo treatment is drawn. ``None`` is the result family's own draw.
    negative_control_kind : str or None
        ``None``, or ``"pre_treatment"`` for a control measured before treatment.
    negative_control_floor : float
        The residual-variance floor of a pre-treatment control, a share in ``(0, 1)``.
    future_treatment_control : Any
        The control the future-treatment test reads.
    future_treatment_node : int
        The node ``t`` whose next treatment is the exposure.

    Returns
    -------
    tuple of str
        The requested test names.

    Raises
    ------
    ValueError
        If no test is named, a test name is unknown, ``n_replicates`` is not a positive integer, the
        placebo mode or the control kind is unknown, the floor is outside ``(0, 1)``, the
        node is not a positive integer, or a control test has no control.
    """
    requested = tuple(tests)
    if not requested:
        raise ValueError(f"name at least one refutation test; choose from {list(_KNOWN_TESTS)}")
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
    if placebo_mode is not None and placebo_mode not in _PLACEBO_MODES:
        raise ValueError(
            f"unknown placebo_mode {placebo_mode!r}; choose from {list(_PLACEBO_MODES)}"
        )
    if "negative_control_outcome" in requested and negative_control_outcome is None:
        raise ValueError(
            "the negative_control_outcome test needs an outcome array that treatment "
            "cannot affect; pass negative_control_outcome=<array>"
        )
    if negative_control_kind not in _NEGATIVE_CONTROL_KINDS:
        raise ValueError(
            f"unknown negative_control_kind {negative_control_kind!r}; choose None or "
            "'pre_treatment'"
        )
    if not 0.0 < float(negative_control_floor) < 1.0:
        raise ValueError("negative_control_floor must lie strictly between 0 and 1")
    if (
        isinstance(future_treatment_node, bool)
        or not isinstance(future_treatment_node, (int, np.integer))
        or future_treatment_node < 1
    ):
        raise ValueError("future_treatment_node must be a positive integer")
    if "future_treatment" in requested and future_treatment_control is None:
        raise ValueError(
            "the future_treatment test needs a control measured before the later treatment "
            "and outside its declared history; pass future_treatment_control=<array>"
        )
    return requested


@dataclass(frozen=True)
class _RefuteRequest:
    """The arguments of one :func:`refute` request that a rule of the table reads.

    Parameters
    ----------
    estimand : str
        The alias the tests would refit.
    tests : tuple of str
        The requested test names, already validated.
    n_replicates : int or None
        Replicates per randomized test.
    dummy_outcome : GaussianIndependentOutcome or None
        The independent process declaration, or ``None`` for the default.
    simulated_outcome : GaussianAdjustmentOutcome or None
        The adjustment-dependent process declaration, or ``None`` for the default.
    outcome_bands : EquivalenceBands
        The equivalence bands applied to generated-outcome refits.
    bootstrap_measurement_error : BootstrapMeasurementError or None
        The measurement-error declaration.
    measurement_error_rule : EmpiricalInclusionRule
        The rule applied to measurement-error refits.
    placebo_mode : {"permute", "draw", "markov"} or None
        How the placebo treatment is drawn. ``None`` is the result family's own draw.
    noise_nodes : bool
        Whether ``random_common_cause`` also appends a noise column to every node of a
        longitudinal fit.
    negative_control_outcome : Any
        The negative-control outcome, or ``None``.
    negative_control_kind : str or None
        ``"pre_treatment"`` for a control measured before treatment, else ``None``.
    future_treatment_control : Any
        The control of the future-treatment test, or ``None``.
    future_treatment_node : int
        The node ``t`` whose next treatment is the exposure.
    negative_control_floor : float
        The least cross-fitted residual variance share of a pre-treatment or future-treatment
        control.
    placebo_bands : dict
        Holds the band-rank draw of the fit once the placebo rule has built it, under the key
        ``"bands"``, so :func:`refute` does not build it twice. Not part of the request's value.
    """

    estimand: str
    tests: tuple[str, ...]
    n_replicates: int | None = None
    dummy_outcome: Any = None
    simulated_outcome: Any = None
    outcome_bands: Any = field(default_factory=EquivalenceBands)
    bootstrap_measurement_error: Any = None
    measurement_error_rule: Any = field(default_factory=EmpiricalInclusionRule)
    placebo_mode: str | None = None
    noise_nodes: bool = False
    negative_control_outcome: Any = None
    negative_control_kind: str | None = None
    future_treatment_control: Any = None
    future_treatment_node: int = 1
    negative_control_floor: float = 0.01
    placebo_bands: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def generated(self) -> tuple[str, ...]:
        """The requested generated-outcome tests, in request order."""
        return tuple(name for name in self.tests if name in _GENERATED_TESTS)

    @property
    def draws(self) -> int:
        """The draw budget of a measurement-error test."""
        return _DEFAULT_MEASUREMENT_REPLICATES if self.n_replicates is None else self.n_replicates

    @property
    def generated_draws(self) -> int:
        """The draw budget of a generated-outcome test."""
        return DEFAULT_OUTCOME_REPLICATES if self.n_replicates is None else self.n_replicates

    def process(self, name: str) -> Any:
        """The outcome process a generated-outcome test draws from.

        Parameters
        ----------
        name : str
            ``"dummy_outcome"``, ``"simulated_outcome"`` or ``"plasmode"``. The plasmode
            draws its outcome from the ``simulated_outcome`` declaration.

        Returns
        -------
        GaussianIndependentOutcome or GaussianAdjustmentOutcome
            The declared process, or the default one when the request declares none.
        """
        if name == "dummy_outcome":
            return (
                GaussianIndependentOutcome() if self.dummy_outcome is None else self.dummy_outcome
            )
        return (
            GaussianAdjustmentOutcome()
            if self.simulated_outcome is None
            else self.simulated_outcome
        )


#: What a caller can run when a generated-outcome test has no derivation for the fit.
_STABILITY_ALTERNATIVE = (
    "The stability tests run on this fit: refute(result, estimand=..., "
    "tests=('random_common_cause', 'subset'))."
)


def _invalid(text: str) -> Refusal:
    """The refusal of a request whose declaration the test cannot run on."""
    return Refusal(RefusalReason.INVALID_DECLARATION, text)


def _refuse_missing_estimator(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a result that no longer holds the estimator every test refits."""
    if result.estimator is not None:
        return None
    return _invalid("refute needs the fitted estimator that produced the result")


def _refuse_unreported_estimand(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse an estimand the fit did not report, which no refit can compare against."""
    if request.estimand in result.estimates:
        return None
    return _invalid(f"estimand {request.estimand!r} was not requested in this fit")


def _is_longitudinal(result: Any) -> bool:
    """Whether ``result`` is a longitudinal fit, which the longitudinal adapter refits."""
    return getattr(result, "assessment_family", None) == "longitudinal"


def _is_panel(result: Any) -> bool:
    """Whether ``result`` fits a treatment panel: a longitudinal or a structural fit."""
    return _is_longitudinal(result) or is_structural(result)


def _refuse_structural_tests(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a test the structural adapter does not run (X82 part D).

    A continuous negative control is a regression target. A structural fit regresses it with
    ``regression_learner``, which defaults to ``outcome_learner``, so a fit whose outcome
    learner only classifies and that declares no regression learner has no learner for it.
    """
    if not is_structural(result):
        return None
    refusal = structural_tests_refusal(request.tests)
    if refusal is not None or "negative_control_outcome" not in request.tests:
        return refusal
    values = np.asarray(request.negative_control_outcome, dtype=float).reshape(-1)
    binary = bool(np.all(np.isin(values[np.isfinite(values)], (0.0, 1.0))))
    estimator = result.estimator
    outcome_learner = getattr(estimator, "outcome_learner", None)
    # A binomial GLM accepts only targets on [0, 1], so it cannot regress the control either.
    classifies = _classification_only_learner(outcome_learner) is not None or (
        str(getattr(outcome_learner, "family", "")) == "binomial"
    )
    if not binary and getattr(estimator, "regression_learner", None) is None and classifies:
        return _invalid(
            "the negative-control outcome is continuous, and this structural fit regresses a "
            "continuous target with its outcome learner, which only classifies. Refit with "
            "regression_learner= set to a regression learner, or pass a binary control"
        )
    return None


#: The tests the longitudinal adapter runs (X82 parts C1 and E4).
_LONGITUDINAL_TESTS = (
    "placebo",
    "random_common_cause",
    "subset",
    "bootstrap_stability",
    "negative_control_outcome",
    "future_treatment",
    "dummy_outcome",
    "simulated_outcome",
    "plasmode",
    "bootstrap_measurement_error",
)


def _refuse_longitudinal_tests(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a test the longitudinal adapter does not run yet."""
    if not _is_longitudinal(result):
        return None
    pending = [name for name in request.tests if name not in _LONGITUDINAL_TESTS]
    if not pending:
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"refutation test(s) {pending} have no longitudinal adapter yet.",
        alternative=(
            "The placebo, random_common_cause, subset, bootstrap_stability, "
            "negative_control_outcome, future_treatment, dummy_outcome, simulated_outcome, "
            "plasmode and bootstrap_measurement_error tests run on this fit."
        ),
        roadmap="X82",
    )


def _refuse_placebo_mode(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a placebo draw that the result's family has no exact null for."""
    if "placebo" not in request.tests or request.placebo_mode is None:
        return None
    if _is_panel(result):
        if request.placebo_mode == "markov":
            return None
        return _invalid(
            f"placebo_mode={request.placebo_mode!r} is a point-treatment draw. On a "
            "longitudinal panel a permuted or redrawn trajectory gives a unit at risk the "
            "missing treatment of a unit that left earlier, and a draw stratified by "
            "follow-up makes the placebo depend on the event process. The longitudinal "
            "placebo draws each trajectory from a Markov kernel: leave placebo_mode=None or "
            "pass 'markov'"
        )
    if request.placebo_mode != "markov":
        return None
    return _invalid(
        "placebo_mode='markov' draws treatment trajectories of a longitudinal panel; a "
        "point-treatment fit permutes ('permute') or draws ('draw') its treatment"
    )


def _refuse_placebo_kernel(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse the placebo on a dose that has no Markov kernel or no band-rank draw.

    A dose whose declared support is a function of the covariates or the history takes the
    band-rank draw (:mod:`~cleverly.validation._band_placebo`). The cases that draw does not
    implement, and a band declaration it cannot read, are refused here, before any refit.
    """
    if "placebo" not in request.tests:
        return None
    if _is_panel(result) and getattr(result.data, "has_mixed_node", False):
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "the longitudinal placebo has no Markov kernel for a vector node with a continuous "
            "component, whose components a drawn code cannot replace.",
            alternative=(
                "The random_common_cause, subset and bootstrap_stability tests run on this fit."
            ),
            roadmap="X82",
        )
    try:
        request.placebo_bands["bands"] = placebo_bands(result)
    except CapabilityError as error:
        return error.refusal
    return None


def _refuse_noise_nodes(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse per-node noise columns where no node or no law for them exists."""
    if not request.noise_nodes or "random_common_cause" not in request.tests:
        return None
    if not _is_panel(result):
        return _invalid(
            "noise_nodes=True appends a noise column to every node of a longitudinal fit; a "
            "point-treatment fit has no nodes. Leave noise_nodes=False"
        )
    if getattr(result.config, "estimator", None) == "parametric_gformula":
        return _invalid(
            "noise_nodes=True on a parametric g-formula fit would add a time-varying "
            "covariate that the forward simulation must model, and the fit declares no law "
            "for it. Leave noise_nodes=False: the baseline noise column runs on this fit"
        )
    return None


def _matching_column(values: np.ndarray, matrix: np.ndarray, rows: np.ndarray) -> int | None:
    """The first column of ``matrix`` equal to ``values`` on ``rows``, or ``None``."""
    target = values[rows]
    for column in range(matrix.shape[1]):
        candidate = np.asarray(matrix[rows, column], dtype=float)
        if np.allclose(candidate, target, rtol=0.0, atol=1e-12):
            return column
    return None


def _adjustment_set(data: Any, longitudinal: bool) -> tuple[np.ndarray, tuple[str, ...], Any]:
    """The adjustment set a pre-treatment control is read against, and its rows.

    A point fit adjusts for its covariates. A longitudinal pre-treatment control precedes
    every node, so its adjustment set is the baseline. The rows are those whose outcome is
    observed.
    """
    if longitudinal:
        rows = ~np.isnan(np.asarray(data.outcome, dtype=float))
        return np.asarray(data.baseline, dtype=float), tuple(data.baseline_names), rows
    return (
        np.asarray(data.covariates, dtype=float),
        tuple(data.covariate_names),
        np.asarray(data.observed, dtype=bool),
    )


def _refuse_negative_control_design(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a negative-control outcome that no refit can read (X82 plan 5.7.1)."""
    if "negative_control_outcome" not in request.tests:
        return None
    # A structural fit holds a panel with one end-of-study outcome, read as a longitudinal one.
    longitudinal = _is_panel(result)
    data: Any = result.data
    if longitudinal and data.is_survival:
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "a negative-control outcome on a survival or competing-risk fit must be an "
            "absorbing indicator process that keeps the risk sets, and no such replacement "
            "is written.",
            alternative=NO_ALTERNATIVE,
            roadmap="X82",
        )
    if request.negative_control_kind != "pre_treatment":
        return None
    if longitudinal and data.is_repeated:
        return _invalid(
            "a pre-treatment control is one value per unit, and a repeated outcome reads one "
            "value per node. Pass negative_control_kind=None for a control at every node"
        )
    matrix, names, rows = _adjustment_set(data, longitudinal)
    values = np.asarray(request.negative_control_outcome, dtype=float).reshape(-1)
    if values.size != data.n or not np.all(np.isfinite(values[rows])):
        # ``with_outcome`` reports the shape and the missing values in its own words.
        return None
    column = _matching_column(values, matrix, rows)
    if column is not None:
        return _invalid(
            f"the pre-treatment control equals the adjustment column {names[column]!r}. The "
            "control fit would drop a measured confounder and give a false alarm by "
            "construction. Pass a control outside the adjustment set"
        )
    reason = _control_residual_reason(values[rows], matrix[rows], request.negative_control_floor)
    return None if reason is None else _invalid(reason)


def _future_history(data: Any, node: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """The history the fit adjusts the treatment at ``node`` for, as one matrix.

    The covariate history ``(W, L_1, ..., L_node)``, with each earlier repeated outcome,
    and the earlier treatments: one indicator per non-reference level of a categorical node,
    and the dose of a continuous node.
    """
    blocks = [np.asarray(data.covariate_history(node), dtype=float)]
    names = list(data.history_names(node))
    for earlier in range(1, node):
        codes = np.asarray(data.treatment[:, earlier - 1], dtype=float)
        name = data.treatment_names[earlier - 1]
        if data.node_kind(earlier) == "categorical":
            levels = data.treatment_levels[earlier - 1]
            for code in range(1, len(levels)):
                blocks.append((codes == code).astype(float)[:, None])
                names.append(f"{name}={levels[code]}")
        else:
            blocks.append(np.nan_to_num(codes)[:, None])
            names.append(name)
    return np.column_stack(blocks), tuple(names)


def _refuse_future_treatment(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a future-treatment exposure that has no valid control (X82 plan 5.7.2)."""
    if "future_treatment" not in request.tests:
        return None
    if not _is_longitudinal(result):
        return _invalid(
            "the future_treatment test reads a later treatment node of a longitudinal fit, "
            "and a point-treatment fit has none"
        )
    data: Any = result.data
    node = request.future_treatment_node + 1
    if data.treatment_held or data.n_decisions < 2:
        return _invalid(
            "no valid control for the future-treatment exposure: the fit has one treatment "
            "decision, so no later treatment exists"
        )
    if node > data.n_decisions:
        return _invalid(
            f"future_treatment_node={request.future_treatment_node} has no later treatment "
            f"node; choose a node below {data.n_decisions}"
        )
    if (
        data.has_mixed_node
        or data.node_kind(node) != "categorical"
        or len(data.treatment_levels[node - 1]) != 2
    ):
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "the future-treatment exposure is a binary treatment at the later node; a node "
            "with more levels, a dose or a mixed node has no contrast written for it.",
            alternative=NO_ALTERNATIVE,
            roadmap="X82",
        )
    control = np.asarray(request.future_treatment_control, dtype=float).reshape(-1)
    if control.size != data.n:
        return _invalid(
            f"future_treatment_control has {control.size} value(s); expected one per unit "
            f"({data.n})"
        )
    rows = data.node_present(node)
    if not np.all(np.isfinite(control[rows])):
        return _invalid(
            f"future_treatment_control must be finite on every unit at risk at node {node}"
        )
    history, names = _future_history(data, node)
    column = _matching_column(control, history, rows)
    if column is not None:
        return _invalid(
            f"the control equals {names[column]!r}, a member of the history that the fit "
            f"adjusts the treatment at node {node} for. A declared cause of that treatment is "
            "associated with it by design, so its estimate is no confounding signal"
        )
    # A control that is nearly a function of the history is associated with the later
    # treatment by design too, for example the sum of two history columns.
    reason = _control_residual_reason(
        control[rows],
        history[rows],
        request.negative_control_floor,
        noun="the control",
        adjustment=f"the history of the treatment at node {node}",
    )
    if reason is not None:
        return _invalid(reason)
    values = control[rows]
    mean = float(np.mean(values)) if values.size else 0.0
    if np.unique(values).size <= 1 or float(np.var(values)) <= 1e-12 * max(1.0, mean**2):
        return _invalid(
            "the control is constant on the analysis set, so its estimate is 0 by construction"
        )
    arms = np.unique(np.asarray(data.treatment[rows, node - 1], dtype=float))
    if arms.size < 2:
        return _invalid(
            f"every unit at risk at node {node} holds one treatment level, so the exposure "
            "has no contrast"
        )
    return None


def _longitudinal_null(result: Any, estimand: str) -> float | None:
    """Return the no-effect value of a longitudinal parameter, if it has one."""
    estimate = result.estimates.get(estimand) if hasattr(result, "estimates") else None
    if estimate is None:
        return None
    scale = getattr(estimate, "scale", None)
    if scale == "difference":
        return 0.0
    if scale == "ratio":
        return 1.0
    return None


def _target_declaration(result: TMLEResult, estimand: str) -> Any:
    """Read the registered target through a key or the reported alias convention."""
    from ..targets import TARGETS
    from ..targets.base import parameter_stem

    keys = getattr(result, "parameter_keys", {})
    key = keys.get(estimand) if keys else None
    target = getattr(key, "estimand", parameter_stem(estimand))
    return TARGETS.get(target)


def _is_learned_rule_fit(result: Any) -> bool:
    return getattr(getattr(result, "config", None), "parameter_axis", None) == "learned_rule"


def _frozen_rule(result: Any) -> FrozenLearnedRule | None:
    """The realized fold rules of a learned-rule fit, or ``None`` for any other fit.

    A refit that changes no covariate value holds these rules fixed, so it estimates the
    value of the fit's own rules and not of rules relearned on the refit data (X82 part F4).
    """
    from ..interventions.learned import freeze_learned_rule

    return freeze_learned_rule(result) if _is_learned_rule_fit(result) else None


def _learned_rule_reference(data: Any) -> float:
    """Return the weighted mean of the observed outcomes, the placebo value of any rule.

    Under the placebo law the value of every rule, learned or fixed, is ``E[Y]``, and this
    mean estimates it. It is constant across permutations, so the ``t`` rule applies
    unchanged. A weighted fit weights it, because the fit's own value is weighted (X82 plan
    5.1.1). A learned-rule fit refuses missing outcomes (F27), so every outcome is observed.
    """
    observed = np.asarray(data.observed, dtype=bool)
    return float(np.average(data.outcome[observed], weights=data.weights[observed]))


def _no_effect_null(result: TMLEResult, estimand: str) -> float | None:
    """Return the reported-scale no-effect value of a registered parameter, if defined."""
    if is_structural(result):
        return structural_null(result, estimand)
    if _is_longitudinal(result):
        return _longitudinal_null(result, estimand)
    declaration = _target_declaration(result, estimand)
    if declaration is None:
        return None
    if declaration.parameter_axis == "msm":
        return None
    if declaration.scale == "ratio":
        return 1.0
    if declaration.scale == "difference" or declaration.name == "paf":
        return 0.0
    return None


@dataclass(frozen=True)
class _MSMReference:
    """The placebo value of one MSM coefficient, and whether it is exact.

    ``exact`` holds when the coefficient is 0 for every function of the covariates the
    placebo law can put in the outcome regression. Otherwise ``value`` is the projection
    of the fitted regression ``E[Y | W]``, a computed number and not an exact null.
    """

    value: float
    exact: bool


def _msm_reference(result: Any, estimand: str) -> _MSMReference | None:
    """Return the placebo reference of a point-treatment MSM coefficient, or ``None``.

    Under the placebo law ``m(a, W) = q(W)`` for every arm ``a`` (X82 plan 5.1.4). The
    coefficient is the weighted projection ``M^-1 sum_i w_i sum_j h_ij phi_ij q_i / n`` of
    ``q``. A coefficient is exactly 0 when its row of ``M^-1 sum_j h_ij phi_ij`` vanishes on
    every unit, whatever ``q`` is. That holds for a product weight and a tensor design whose
    arm factor contains an intercept, and the identity link keeps the projection linear. The
    check reads the realised arrays, so it needs no declaration of the structure. Any other
    coefficient takes the projection of the fitted mixture ``sum_a g(a | W) Qbar(a, W)``.
    """
    from ..msm import solve_projection

    if getattr(getattr(result, "config", None), "parameter_axis", None) != "msm":
        return None
    nuisance = getattr(result, "nuisance", None)
    model = getattr(nuisance, "msm", None)
    fluctuation = getattr(result, "fluctuations", {}).get("msm")
    if nuisance is None or model is None or fluctuation is None or model.dose_values:
        return None
    labels = [f"msm[{term}]" for term in model.terms]
    if estimand not in labels:
        return None
    column = labels.index(estimand)
    design = np.asarray(model.design, dtype=float)
    weights = np.asarray(model.weights, dtype=float)
    sampling = np.asarray(result.data.weights, dtype=float)
    if model.link == "identity":
        normal = np.einsum("i,ij,ijp,ijq->pq", sampling, weights, design, design)
        rows = np.einsum("i,ij,ijp->pi", sampling, weights, design)
        operator = np.linalg.solve(normal, rows)
        if np.max(np.abs(operator[column])) <= 1e-9 * max(1.0, float(np.max(np.abs(operator)))):
            return _MSMReference(0.0, True)
    arms = model.arms
    targeted = fluctuation.targeted
    propensity = nuisance.propensity
    mixture = sum(
        np.asarray(propensity.arm(arm), dtype=float)
        * np.asarray(nuisance.scaler.unscale_level(targeted.arms[arm]), dtype=float)
        for arm in arms
    )
    predictions = np.repeat(np.asarray(mixture, dtype=float)[:, None], len(arms), axis=1)
    beta = solve_projection(design, weights, predictions, sampling, str(model.link)).beta
    return _MSMReference(float(beta[column]), False)


def _longitudinal_msm_reference(result: Any, estimand: str) -> _MSMReference | None:
    """Return the placebo reference of a longitudinal working-model coefficient, or ``None``.

    Under the placebo law every regimen of a cell has the same mean ``q(V, t)``, a function
    of the unit and the cell's horizon (X82 plan 5.1.2 and 5.1.4). Under the identity link a
    coefficient is exactly 0 when, for every unit and every horizon, its row of
    ``M^-1 sum_c h_c phi_c`` over the cells of that horizon vanishes, whatever ``q`` is. Any
    other coefficient, or any other link, has no exact placebo value, and the value of the
    reference is ``nan``.
    """
    from ..longitudinal.estimator import CAUSE_INFIX

    for msm_fit in getattr(result, "msm_fits", ()):
        model = msm_fit.model
        for column, term in enumerate(model.terms):
            inside = term if msm_fit.cause is None else f"{term}{CAUSE_INFIX}{msm_fit.cause}"
            if f"msm_regimen[{inside}]" != estimand:
                continue
            if str(model.link) != "identity":
                return _MSMReference(float("nan"), False)
            design = np.asarray(model.design, dtype=float)
            weights = np.asarray(model.weights, dtype=float)
            sampling = np.asarray(result.data.weights, dtype=float)
            normal = np.einsum("i,ik,ikp,ikq->pq", sampling, weights, design, design)
            horizons = np.asarray([cell.horizon for cell in model.cells])
            for horizon in np.unique(horizons):
                cells = horizons == horizon
                rows = np.einsum("i,ik,ikp->pi", sampling, weights[:, cells], design[:, cells, :])
                operator = np.linalg.solve(normal, rows)
                scale = max(1.0, float(np.max(np.abs(operator))))
                if np.max(np.abs(operator[column])) > 1e-9 * scale:
                    return _MSMReference(float("nan"), False)
            return _MSMReference(0.0, True)
    return None


def _working_model_reference(result: Any, estimand: str) -> _MSMReference | None:
    """The placebo reference of a working-model coefficient; a structural fit has none."""
    if is_structural(result):
        return None
    if _is_longitudinal(result):
        return _longitudinal_msm_reference(result, estimand)
    return _msm_reference(result, estimand)


def _placebo_null(result: TMLEResult, estimand: str) -> float | None:
    """Return the value the placebo reads the refit against, or ``None`` when it has none.

    A contrast has its fixed no-effect value. A learned-rule level reads against the outcome
    mean, because the placebo law gives every rule that value. An MSM coefficient reads
    against its placebo projection (:func:`_msm_reference`).
    """
    if _is_learned_rule_fit(result) and estimand in result.estimates:
        return _learned_rule_reference(result.data)
    reference = _working_model_reference(result, estimand)
    if reference is not None:
        return reference.value
    return _no_effect_null(result, estimand)


def _no_effect_difference(value: float, null: float) -> float:
    """Return the signed difference on the estimate's inference scale."""
    if null == 1.0:
        return float(np.log(value))
    return value - null


def _refuse_no_effect_null(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse null-effect tests on parameters with no fixed reported-scale null."""
    chosen = [name for name in request.tests if name in ("placebo", "negative_control_outcome")]
    chosen = [
        name
        for name in chosen
        if (_placebo_null if name == "placebo" else _no_effect_null)(result, request.estimand)
        is None
    ]
    if not chosen:
        return None
    keys = getattr(result, "parameter_keys", {})
    key = keys.get(request.estimand) if keys else None
    from ..targets.base import parameter_stem

    target = getattr(key, "estimand", parameter_stem(request.estimand))
    label = "NaturalCourseMean" if target == "ey_obs" else repr(request.estimand)
    prefix = "the placebo refutation and " if target == "ey_obs" and "placebo" in chosen else ""
    return _invalid(
        f"{prefix}refutation test(s) {chosen} require a fixed no-effect value on the reported "
        f"scale, but {label} has none. An outcome level or policy mean need not approach "
        "zero, and an arbitrary MSM coefficient has no universal null. Drop these "
        "tests from tests=; stability tests keep their usual interpretation."
    )


def _refuse_row_set_under_plan(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a test that refits on another row set when the fit was given a split plan.

    A fit that was given a :class:`~cleverly.SplitPlan` cannot refit on a different row
    set: the plan holds one fold label per row of the original data, and there is no rule
    here for which labels a resampled row inherits.  Left to the refit, ``subset`` gives the
    caller a row-count ``DataError`` naming a "split plan" they did not mention in this
    call, after paying for the tests that ran first, and ``bootstrap_measurement_error``
    gives no error at all: its draw keeps the row count, so the labels land on resampled
    units and the refutation silently reports a split the fit never ran.

    ``getattr`` because ``TMLEResult.estimator`` is annotated ``Any`` and :func:`refute`
    reads it as a refit *seam*: anything with the ``refit`` signature serves, and
    ``_validate_generated_eligibility`` reads ``getattr(estimator, "family", "auto")`` off
    the same object for the same reason.  The stand-ins in
    ``tests/unit/test_generated_outcome_refutation.py`` and
    ``tests/unit/test_bootstrap_measurement_error.py`` hold no plan.
    """
    supplied_plan = getattr(result.estimator, "split_plan", None)
    if supplied_plan is None:
        return None
    resampled = [name for name in request.tests if name in _PLAN_REFUSING_TESTS]
    if not resampled:
        return None
    return _invalid(
        f"refutation test(s) {resampled} refit on rows this fit did not run, and "
        f"this fit declared split_plan={supplied_plan!r}. A supplied plan labels "
        "the rows it was realised on, by position, so it cannot label a refit that "
        "drops rows or draws them with replacement, and dropping the plan for those "
        "refits would refute a different split from the one the fit ran. Drop "
        f"{resampled} from tests=, or refit the result without split_plan= to refute "
        "a fit whose folds are drawn from the data each time"
    )


#: The tests that hold a learned-rule fit's realized rules fixed on every refit (X82 part F4).
_RULE_HOLDING_TESTS = (
    "random_common_cause",
    "subset",
    "bootstrap_stability",
    "bootstrap_measurement_error",
)


def _refuse_learned_rule_splits(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a rule-holding test on a learned-rule fit of several splits.

    Those tests freeze one realized rule per row, and a fit of ``repeats`` above 1 has one
    rule per row in each split.  ``placebo`` relearns the rules, so it runs.
    """
    if not _is_learned_rule_fit(result):
        return None
    held = [name for name in request.tests if name in _RULE_HOLDING_TESTS]
    if not held:
        return None
    from ..interventions.learned import learned_rule_splits_refusal

    return learned_rule_splits_refusal(f"refute(tests={held})", result.n_repeats)


def _refuse_generated_rule(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a generated-outcome test under bands other than the registered ones."""
    if not request.generated or type(request.outcome_bands) is EquivalenceBands:
        return None
    return _invalid(
        "generated-outcome refutations require the exact registered EquivalenceBands declaration"
    )


def _refuse_generated_eligibility(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a generated-outcome test whose fit, estimand or process has no derivation.

    The sentence is the one ``_validate_generated_eligibility`` raises.  This rule calls
    that check rather than copying it, and :func:`refute` calls it again to read the
    contrast direction it returns.
    """
    for name in request.generated:
        try:
            _validate_generated_eligibility(result, request.estimand, request.process(name), name)
        except CapabilityError as error:
            return error.refusal
    return None


def _refuse_measurement_declaration(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse ``bootstrap_measurement_error`` without the registered declaration."""
    if "bootstrap_measurement_error" not in request.tests:
        return None
    if type(request.bootstrap_measurement_error) is BootstrapMeasurementError:
        return None
    return _invalid(
        "bootstrap_measurement_error requires the exact registered "
        "BootstrapMeasurementError declaration"
    )


def _refuse_measurement_rule(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse ``bootstrap_measurement_error`` under a rule other than the registered one."""
    if "bootstrap_measurement_error" not in request.tests:
        return None
    if type(request.measurement_error_rule) is EmpiricalInclusionRule:
        return None
    return _invalid(
        "bootstrap_measurement_error requires the exact registered "
        "EmpiricalInclusionRule declaration"
    )


def _refuse_measurement_eligibility(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a measurement-error declaration that this fit's data cannot carry.

    The sentence is the one ``_validate_measurement_error_eligibility`` raises, and this
    rule calls that check rather than copying it.
    """
    if "bootstrap_measurement_error" not in request.tests:
        return None
    try:
        _validate_measurement_error_eligibility(
            result, request.bootstrap_measurement_error, request.draws
        )
    except CapabilityError as error:
        return error.refusal
    return None


def _refuse_stability_eligibility(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse ``bootstrap_stability`` on a fit whose data it cannot resample, or on one draw."""
    if "bootstrap_stability" not in request.tests:
        return None
    try:
        _validate_stability_eligibility(result, request.draws)
    except CapabilityError as error:
        return error.refusal
    return None


def _refuse_measurement_budget(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a measurement-error draw budget below its rule's floor."""
    if "bootstrap_measurement_error" not in request.tests:
        return None
    floor = request.measurement_error_rule.minimum_draws
    if request.draws >= floor:
        return None
    return _invalid(
        f"bootstrap_measurement_error was asked for {request.draws} draw(s) "
        f"under a rule that requires {floor}; raise "
        "n_replicates or declare a rule with a smaller minimum_draws"
    )


def _refuse_generated_budget(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """Refuse a generated-outcome draw budget too small for the equivalence intervals.

    The intervals of the three readings need a spread of draws to exist, and the caller
    can see that before any refit is paid for. A budget below the bands' calibrated
    minimum (400) still runs, and its readings print at ``L1``.
    """
    floor = _MINIMUM_BAND_DRAWS
    if not request.generated or request.generated_draws >= floor:
        return None
    return _invalid(
        f"a generated-outcome refutation was asked for {request.generated_draws} draw(s); "
        f"its equivalence intervals need at least {floor}. Raise n_replicates to at least "
        f"{floor} ({DEFAULT_OUTCOME_REPLICATES} is the calibrated minimum)"
    )


#: Every refusal of a :func:`refute` request that its arguments and the result decide
#: before any refit, in the one order the call and the ``refute`` capability row both use.
#: Each rule assumes its predecessors returned ``None``.  The names are the introspection
#: contract, so a test pins the order without respelling a message.
_REQUEST_RULES: tuple[tuple[str, Callable[[TMLEResult, _RefuteRequest], Refusal | None]], ...] = (
    ("estimator", _refuse_missing_estimator),
    ("estimand", _refuse_unreported_estimand),
    ("structural_tests", _refuse_structural_tests),
    ("longitudinal_tests", _refuse_longitudinal_tests),
    ("placebo_mode", _refuse_placebo_mode),
    ("placebo_kernel", _refuse_placebo_kernel),
    ("noise_nodes", _refuse_noise_nodes),
    ("negative_control_design", _refuse_negative_control_design),
    ("future_treatment", _refuse_future_treatment),
    ("no_effect_null", _refuse_no_effect_null),
    ("row_set_under_plan", _refuse_row_set_under_plan),
    ("learned_rule_splits", _refuse_learned_rule_splits),
    ("generated_rule", _refuse_generated_rule),
    ("generated_eligibility", _refuse_generated_eligibility),
    ("measurement_declaration", _refuse_measurement_declaration),
    ("measurement_rule", _refuse_measurement_rule),
    ("measurement_eligibility", _refuse_measurement_eligibility),
    ("stability_eligibility", _refuse_stability_eligibility),
    ("measurement_budget", _refuse_measurement_budget),
    ("generated_budget", _refuse_generated_budget),
)


def _first_refusal(result: TMLEResult, request: _RefuteRequest) -> Refusal | None:
    """The record of the first rule of :data:`_REQUEST_RULES` that refuses ``request``."""
    for _name, rule in _REQUEST_RULES:
        refusal = rule(result, request)
        if refusal is not None:
            return refusal
    return None


def refute_refusal(
    result: TMLEResult,
    *,
    estimand: str,
    tests: Sequence[str],
    n_replicates: int | None = None,
    negative_control_outcome: Any = None,
    dummy_outcome: (
        GaussianIndependentOutcome | LogisticHazardOutcome | MultinomialHazardOutcome | None
    ) = None,
    simulated_outcome: (
        GaussianAdjustmentOutcome | LogisticHazardOutcome | MultinomialHazardOutcome | None
    ) = None,
    outcome_bands: EquivalenceBands = EquivalenceBands(),
    bootstrap_measurement_error: BootstrapMeasurementError | None = None,
    measurement_error_rule: EmpiricalInclusionRule = EmpiricalInclusionRule(),
    placebo_mode: str | None = None,
    noise_nodes: bool = False,
    negative_control_kind: str | None = None,
    negative_control_floor: float = 0.01,
    future_treatment_control: Any = None,
    future_treatment_node: int = 1,
) -> Refusal | None:
    """Return the refusal a :func:`refute` request meets before any refit.

    :func:`refute` raises the record this function returns, and the ``refute`` capability
    row of :class:`~cleverly.assessment.DiagnosticsFacade` quotes it for the same request.
    The row defers on ``tests`` when the default tests are refused and a smaller set runs,
    so a combined report names the argument instead of calling a refutation the call
    refuses.  The keyword arguments are those of :func:`refute` that decide a refusal.

    A malformed argument is not a refusal. An unknown test name, an invalid
    ``n_replicates``, or a missing negative-control outcome raises ``ValueError`` from
    the call before any refusal. This function returns ``None`` for those requests.

    Parameters
    ----------
    result : TMLEResult
        A fitted point-treatment result.
    estimand : str
        The alias the tests would refit.
    tests : sequence of str
        The requested test names.
    n_replicates : int or None
        Replicates per randomized test.
    negative_control_outcome : Any
        The outcome required by the negative-control test.
    dummy_outcome : GaussianIndependentOutcome, LogisticHazardOutcome or None
        Independent process declaration.
    simulated_outcome : GaussianAdjustmentOutcome, LogisticHazardOutcome or None
        Adjustment-dependent process declaration.
    outcome_bands : EquivalenceBands
        Bands applied to generated-outcome readings.
    bootstrap_measurement_error : BootstrapMeasurementError or None
        Measurement-error declaration.
    measurement_error_rule : EmpiricalInclusionRule
        Rule applied to measurement-error refits.
    placebo_mode : {"permute", "draw", "markov"} or None
        How the placebo treatment is drawn. ``None`` is the result family's own draw.
    noise_nodes : bool
        Whether ``random_common_cause`` also appends a noise column to every node.
    negative_control_kind : str or None
        ``"pre_treatment"`` for a control measured before treatment, else ``None``.
    negative_control_floor : float
        The residual-variance floor of a pre-treatment or future-treatment control.
    future_treatment_control : Any
        The control of the future-treatment test.
    future_treatment_node : int
        The node ``t`` whose next treatment is the exposure.

    Returns
    -------
    Refusal or None
        The exact refusal record, or ``None`` when no rule refuses the request or an
        argument is malformed.
    """
    try:
        requested = _validated_tests(
            tests,
            n_replicates,
            negative_control_outcome,
            placebo_mode,
            negative_control_kind,
            negative_control_floor,
            future_treatment_control,
            future_treatment_node,
        )
    except ValueError:
        return None
    request = _RefuteRequest(
        estimand,
        requested,
        n_replicates,
        dummy_outcome,
        simulated_outcome,
        outcome_bands,
        bootstrap_measurement_error,
        measurement_error_rule,
        placebo_mode,
        bool(noise_nodes),
        negative_control_outcome=negative_control_outcome,
        negative_control_kind=negative_control_kind,
        future_treatment_control=future_treatment_control,
        future_treatment_node=int(future_treatment_node),
        negative_control_floor=float(negative_control_floor),
    )
    return _first_refusal(result, request)


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


#: The point processes each generated test accepts. The independent test also accepts a
#: Bernoulli or bounded declaration whose ``independent`` property holds.
_INDEPENDENT_PROCESSES = (
    GaussianIndependentOutcome,
    BernoulliAdjustmentOutcome,
    BoundedAdjustmentOutcome,
)
_ADJUSTMENT_PROCESSES = (
    GaussianAdjustmentOutcome,
    BernoulliAdjustmentOutcome,
    BoundedAdjustmentOutcome,
)


def _validate_generated_process(process: Any, name: str) -> None:
    accepted = _INDEPENDENT_PROCESSES if name == "dummy_outcome" else _ADJUSTMENT_PROCESSES
    if type(process) not in accepted:
        raise CapabilityError(
            f"{name} has no registered effect derivation and family validation for "
            f"{type(process).__name__}; use one of the registered declarations "
            f"{[kind.__name__ for kind in accepted]}",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if name == "dummy_outcome" and not process.independent:
        raise CapabilityError(
            f"dummy_outcome draws an outcome independent of everything, and this "
            f"{type(process).__name__} reads the covariates or the treatment. Declare "
            "effect=0 and adjustment_scale=0, or run it as simulated_outcome",
            reason=RefusalReason.INVALID_DECLARATION,
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
        f"{data.treatment_name} whose levels are {list(data.treatment_levels)}",
        reason=RefusalReason.INVALID_DECLARATION,
    )


def _validate_plasmode_eligibility(result: Any) -> None:
    """Refuse a plasmode on data whose generated law Part A does not build (plan 5.6)."""
    data = result.data
    if data.weights_name is not None:
        raise CapabilityError(
            "plasmode has no generated law for a fit with weights: rows drawn from the "
            "unweighted empirical law that keep their weights describe a mixed law.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    observation = [
        label
        for present, label in (
            (bool(getattr(data, "has_missing_outcome", False)), "a missing outcome"),
            (bool(getattr(data, "has_missing_treatment", False)), "a missing treatment"),
            (bool(getattr(data, "has_intermediate", False)), "an intermediate"),
        )
        if present
    ]
    if len(observation) > 1:
        raise CapabilityError(
            f"plasmode has no generated law for a fit with {' and '.join(observation)}: each "
            "frozen observation model reads (W, A) alone, so two of them together would "
            "drop the dependence of one on the other.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )


@dataclass(frozen=True)
class _LongitudinalGenerated:
    """A generated outcome on the held history of a longitudinal fit (X82 plan 5.5).

    The draw keeps ``(W, L, A, C)`` and replaces the outcome. The outcome at node ``t`` is
    ``f(W) + effect * sum_s A_s + epsilon_t``, where the sum runs over the decision nodes up
    to ``t``, and ``f`` and ``effect`` are 0 for the independent process. The outcome reads no
    ``L``, so a static regimen ``a`` has the exact level ``mean_i f(W_i) + mu + effect *
    sum_s a_s`` and every held censoring pattern stays at random.

    Parameters
    ----------
    process : GaussianIndependentOutcome or GaussianAdjustmentOutcome
        The declared outcome process.
    truth : float
        The sample-conditional value of the estimand under the process.
    covariate_part : bool
        Whether the covariate part of the influence curve vanishes, so the standard-error
        ratio and the coverage have the held-design reference.
    shares : tuple of float
        The uncensored share at each node.
    """

    process: Any
    truth: float
    covariate_part: bool
    shares: tuple[float, ...]

    def draw(self, data: Any, *, seed: int, name: str) -> Any:
        """The container with one generated outcome in place of the original.

        Parameters
        ----------
        data : LongitudinalData
            The held panel.
        seed : int
            The child seed of this draw.
        name : str
            The test name, named in a refusal.

        Returns
        -------
        LongitudinalData
            The panel with the generated outcome.
        """
        rng = np.random.default_rng(seed)
        decisions = int(data.n_decisions)
        codes = np.nan_to_num(np.asarray(data.treatment[:, :decisions], dtype=float))
        dose = np.cumsum(codes, axis=1)
        shift = _generated_adjustment(self.process, data)
        effect = float(self.process.known_effect)
        if data.repeated is None:
            noise = self.process.noise.draw(rng, data.n)
            values = shift + effect * dose[:, -1] + noise
        else:
            nodes = int(data.n_times)
            noise = self.process.noise.draw(rng, data.n * nodes).reshape(data.n, nodes)
            reached = np.minimum(np.arange(1, nodes + 1), decisions) - 1
            values = shift[:, None] + effect * dose[:, reached] + noise
        return data.with_outcome(values, name=name)


def _generated_adjustment(process: Any, data: Any) -> np.ndarray:
    """``f(W)`` of a generated process on the baseline of ``data``; 0 without one."""
    if not isinstance(process, GaussianAdjustmentOutcome):
        return np.zeros(data.n, dtype=float)
    baseline = np.asarray(data.baseline, dtype=float)
    if not baseline.shape[1]:
        return np.zeros(data.n, dtype=float)
    return process.adjustment(baseline)


def _zero_one_levels(levels: Sequence[object]) -> bool:
    """Whether a node's levels are the numbers 0 and 1, in that order."""
    try:
        return tuple(float(level) for level in levels) == (0.0, 1.0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _refuse_longitudinal_plasmode(name: str, structure: str) -> None:
    if name == "plasmode":
        raise CapabilityError(
            f"the longitudinal plasmode has no draw for {structure}. It draws an end-of-study, "
            "repeated-outcome or survival panel with binary treatment nodes.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="dummy_outcome and simulated_outcome run on the held history.",
            roadmap="X82",
        )


@dataclass(frozen=True, eq=False)
class _LongitudinalPlasmode:
    """A longitudinal plasmode target: the frozen law, its g-formula truth and its draw.

    Parameters
    ----------
    law : LongitudinalPlasmodeLaw
        The frozen law.
    process : object
        The declared outcome process.
    truth : float
        The g-formula value of the estimand under the law.
    monte_carlo_se : float
        The Monte Carlo standard error of ``truth``, 0 when every branch is enumerated.
    draws : int or None
        The histories per support point of a Monte Carlo truth, or ``None``.
    covariate_part : bool
        Always ``True``: the plasmode resamples ``W``, so its readings have the
        unconditional reference.
    """

    law: Any
    process: Any
    truth: float
    monte_carlo_se: float
    draws: int | None
    covariate_part: bool = True

    def draw(self, data: Any, *, seed: int, name: str) -> Any:
        """Draw one panel of the natural course of the frozen law.

        Parameters
        ----------
        data : LongitudinalData
            The held panel. The law holds it as its support.
        seed : int
            The child seed of this draw.
        name : str
            The test name.

        Returns
        -------
        LongitudinalData
            The drawn panel.
        """
        return self.law.draw(self.process, seed=seed, name=name)


def _longitudinal_plasmode_target(
    result: Any, estimand: str, process: Any, name: str
) -> _LongitudinalPlasmode:
    """Refuse a panel the longitudinal plasmode cannot draw, else freeze its law and truth."""
    from ..longitudinal.regimen import declares_tilt, declares_tuned
    from ._longitudinal_plasmode import (
        MAX_DRAWS,
        MONTE_CARLO_SHARE,
        PILOT_DRAWS,
        LongitudinalPlasmodeLaw,
    )

    data = result.data
    structures = (
        (bool(data.treatment_held), "a held baseline decision"),
        (int(data.n_decisions) != int(data.n_times), "a node without a treatment decision"),
        (data.known_mechanisms is not None, "a declared known mechanism"),
        (data.cluster is not None, "a clustered panel"),
        (bool(data.is_weighted), "a weighted panel"),
        (declares_tuned(result.config.regimens), "a regimen with a tuned cap"),
        (declares_tilt(result.config.regimens), "an odds tilt of the treatment mechanism"),
        (
            bool(data.has_mixed_node)
            or not all(_zero_one_levels(levels) for levels in data.treatment_levels),
            "a non-binary treatment node",
        ),
    )
    for present, structure in structures:
        if present:
            _refuse_longitudinal_plasmode(name, structure)
    label, cause, horizon = _held_parameter(result, estimand, name)
    regimens = {regimen.label: regimen for regimen in result.config.regimens}
    parts = label.split(" vs ") if " vs " in label else [label]
    law = LongitudinalPlasmodeLaw.fit(result)
    # A survival risk or a repeated mean reads the nodes through its horizon, and a risk
    # reads the cause of its parameter.
    reading: dict[str, int] = {}
    if data.is_survival or data.repeated is not None:
        reading["horizon"] = horizon
    if data.is_survival:
        reading["cause"] = 0 if cause is None else list(data.cause_labels).index(cause)
    draws: int | None = None
    if not law.enumerable:
        # Size N so the Monte Carlo error is at most 2% of the fit's standard error (plan 6).
        pilot = sum(
            law.truth(process, regimens[part], draws=PILOT_DRAWS, seed=index, **reading)[1] ** 2
            for index, part in enumerate(parts)
        )
        allowed = MONTE_CARLO_SHARE * float(result[estimand].plugin_std_error)
        needed = PILOT_DRAWS * pilot / allowed**2 if allowed > 0.0 else float(MAX_DRAWS)
        draws = int(min(max(np.ceil(needed), PILOT_DRAWS), MAX_DRAWS))
    levels = [
        law.truth(process, regimens[part], draws=draws, seed=1000 + index, **reading)
        for index, part in enumerate(parts)
    ]
    truth = levels[0][0] - levels[1][0] if len(levels) == 2 else levels[0][0]
    error = float(np.sqrt(sum(level[1] ** 2 for level in levels)))
    return _LongitudinalPlasmode(
        law=law, process=process, truth=float(truth), monte_carlo_se=error, draws=draws
    )


def _held_parameter(result: Any, estimand: str, name: str) -> tuple[str, str | None, int]:
    """The regimen label, cause and horizon of ``estimand``, after the shared refusals.

    Each node must have the levels 0 and 1, because a process adds its effect once per
    treated node. An MSM coefficient has no held-history truth.
    """
    data = result.data
    index = getattr(result, "parameter_index", None) or {}
    if getattr(result, "msm", None) is not None or estimand not in index:
        raise CapabilityError(
            f"{name} has no held-history truth for {estimand!r}; the derivation covers the "
            "regimen levels and their difference contrasts.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="X82",
        )
    decisions = int(data.n_decisions)
    binary = not data.has_mixed_node and all(
        data.node_kind(node) == "categorical" and _zero_one_levels(data.treatment_levels[node - 1])
        for node in range(1, decisions + 1)
    )
    if not binary:
        raise CapabilityError(
            f"{name} adds the effect once per treated node, so each node must have the "
            "levels 0 and 1; another node has no derivation.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    label, cause, horizon = index[estimand]
    return str(label), cause, int(horizon)


def _regimen_doses(
    result: Any, label: str, horizon: int, name: str, *, static: bool
) -> list[np.ndarray | None]:
    """Per policy of ``label``, the treated decision nodes up to each node through ``horizon``.

    ``None`` stands for a policy the process does not read, which only an independent
    process admits. A process that reads the treatment needs a static regimen, because any
    other policy reads the held covariates or the natural treatment.
    """
    from ..longitudinal.regimen import Regimen

    regimens = {regimen.label: regimen for regimen in result.config.regimens}
    decisions = int(result.data.n_decisions)
    doses: list[np.ndarray | None] = []
    for part in label.split(" vs ") if " vs " in label else [label]:
        regimen = regimens.get(part)
        if not static:
            doses.append(None)
            continue
        if type(regimen) is not Regimen:
            raise CapabilityError(
                f"{name} has an exact truth for a static regimen only. Policy {part!r} reads "
                "the held covariates or the natural treatment, whose law under the policy is "
                "unknown when the history is held.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="dummy_outcome has the exact null for every policy.",
                roadmap="X82",
            )
        values = np.asarray(regimen.values, dtype=float).reshape(-1)
        arms = np.resize(values, decisions) if values.size < decisions else values[:decisions]
        reached = np.minimum(np.arange(1, horizon + 1), decisions) - 1
        doses.append(np.cumsum(arms)[reached])
    return doses


def _held_shares(uncensored: Any) -> tuple[float, ...]:
    kept = np.asarray(uncensored, dtype=bool)
    return tuple(float(np.mean(kept[:, node])) for node in range(kept.shape[1]))


def _longitudinal_generated_target(
    result: Any, estimand: str, process: Any, name: str
) -> _LongitudinalGenerated | _SurvivalGenerated | _LongitudinalPlasmode:
    """Refuse a generated outcome a longitudinal fit cannot read, else build its target."""
    data = result.data
    if data.is_survival:
        if name == "plasmode":
            _validate_hazard_process(data, process, name)
            return _longitudinal_plasmode_target(result, estimand, process, name)
        return _survival_generated_target(result, estimand, process, name)
    _validate_generated_process(process, name)
    if not isinstance(process, (GaussianIndependentOutcome, GaussianAdjustmentOutcome)):
        raise CapabilityError(
            f"{name} on a longitudinal panel has a held-history truth for the Gaussian "
            f"processes only; {type(process).__name__} has none.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    if getattr(process, "arm_effects", None) is not None:
        raise CapabilityError(
            f"{name} on a longitudinal panel adds its effect once per treated node; "
            "arm_effects declares a point-treatment arm term. Declare effect= instead",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if getattr(process, "intermediate_effect", 0.0) or getattr(process, "cluster_sd", 0.0):
        raise CapabilityError(
            f"{name} on a longitudinal panel has no intermediate and no clustered plasmode; "
            "intermediate_effect and cluster_sd declare point-treatment terms. Leave both 0",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if data.family != process.family:
        raise CapabilityError(
            f"{name} cannot replace an original outcome family={data.family!r} with the "
            f"registered {process.family!r} process; the first catalog requires matching "
            "Gaussian outcome configurations.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    estimator = result.estimator
    if getattr(estimator, "q_bounds", None) is not None:
        raise CapabilityError(
            f"{name} draws a Gaussian outcome, whose support is the whole real line, and this "
            f"fit declares q_bounds={estimator.q_bounds!r}. A cross-fitted continuous panel "
            "must declare them, so refute a continuous outcome on a fit with n_folds=1 and "
            "q_bounds=None",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    incompatible = _classification_only_learner(getattr(estimator, "outcome_learner", None))
    if incompatible is not None:
        raise CapabilityError(
            f"{name} needs a regression-capable outcome learner for its Gaussian process; "
            f"{type(incompatible).__name__} is configured for classification",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if name == "plasmode":
        return _longitudinal_plasmode_target(result, estimand, process, name)
    if getattr(process, "history_effect", 0.0):
        raise CapabilityError(
            f"{name} keeps the held covariate history, whose law under a regimen is unknown, "
            "so an outcome that reads it has no held-history truth. Leave history_effect at 0, "
            "or run tests=('plasmode',), which draws the history from frozen models",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    label, _cause, horizon = _held_parameter(result, estimand, name)
    adjusted = isinstance(process, GaussianAdjustmentOutcome)
    doses = _regimen_doses(result, label, horizon, name, static=adjusted)
    shift = _generated_adjustment(process, data)
    baseline = float(np.average(shift, weights=np.asarray(data.weights, dtype=float)))
    effect = float(process.known_effect)
    levels = [
        baseline + process.noise.mean + (0.0 if dose is None else effect * float(dose[-1]))
        for dose in doses
    ]
    contrast = len(levels) == 2
    truth = levels[0] - levels[1] if contrast else levels[0]
    return _LongitudinalGenerated(
        process=process,
        truth=float(truth),
        covariate_part=contrast or not adjusted,
        shares=_held_shares(data.uncensored),
    )


@dataclass(frozen=True)
class _SurvivalGenerated:
    """Generated events on the held history of a survival fit (X82 plan 5.5.2).

    The draw keeps ``(W, L, A, C)``. An original event at node ``t`` becomes censoring at
    ``t``, so the generated outcome is not observed at that node. Original censoring stays.
    While a unit is under observation, the generated event at node ``t`` reads ``W`` and the
    treated decision nodes up to ``t`` through the declared hazard, and fresh noise. The
    recoded censoring reads the original outcome, and the generated event does not, so the
    censoring stays at random given the recorded history.

    Parameters
    ----------
    process : LogisticHazardOutcome or MultinomialHazardOutcome
        The declared hazard.
    truth : float
        The sample-conditional value of the estimand under the process.
    covariate_part : bool
        Whether the covariate part of the influence curve vanishes.
    shares : tuple of float
        The uncensored share at each node after the recode.
    """

    process: Any
    truth: float
    covariate_part: bool
    shares: tuple[float, ...]

    def draw(self, data: Any, *, seed: int, name: str) -> Any:
        """The container with generated events in place of the original ones.

        Parameters
        ----------
        data : LongitudinalData
            The held panel.
        seed : int
            The child seed of this draw.
        name : str
            The test name. The survival draw names nothing.

        Returns
        -------
        LongitudinalData
            The panel with the generated events and the recoded exits.
        """
        rng = np.random.default_rng(seed)
        kept = _recoded_uncensored(data)
        shift = _hazard_adjustment(self.process, data)
        decisions = int(data.n_decisions)
        codes = np.nan_to_num(np.asarray(data.treatment[:, :decisions], dtype=float))
        dose = np.cumsum(codes, axis=1)
        nodes = int(data.n_times)
        causes = int(self.process.causes)
        event = np.zeros((data.n, nodes), dtype=bool)
        cause_event = np.zeros((data.n, nodes, causes), dtype=bool)
        alive = np.ones(data.n, dtype=bool)
        for node in range(nodes):
            observed = alive & kept[:, node]
            probabilities = self.process.probabilities(shift, dose[:, min(node + 1, decisions) - 1])
            draw = rng.uniform(size=data.n)
            edges = np.cumsum(probabilities, axis=1)
            cause = np.sum(draw[:, None] >= edges, axis=1)
            hit = observed & (cause < causes)
            for index in range(causes):
                cause_event[hit & (cause == index), node:, index] = True
            event[hit, node:] = True
            alive &= ~hit
            kept[~alive, node + 1 :] = False
        return data.with_events(event, kept, cause_event if data.cause_event is not None else None)


def _recoded_uncensored(data: Any) -> np.ndarray:
    """The censoring of ``data`` with each original event recoded as censoring at its node."""
    kept = np.asarray(data.uncensored, dtype=bool).copy()
    event = np.asarray(data.event, dtype=bool)
    first = event.copy()
    first[:, 1:] &= ~event[:, :-1]
    for node in range(kept.shape[1]):
        kept[first[:, node], node:] = False
    return kept


def _hazard_adjustment(process: Any, data: Any) -> np.ndarray:
    from .outcome_processes import standardized_sum

    return standardized_sum(np.asarray(data.baseline, dtype=float), process.adjustment_scale)


def _validate_hazard_process(data: Any, process: Any, name: str) -> None:
    """Refuse a declaration that is not the hazard of the fit's outcome."""
    competing = bool(data.is_competing)
    expected = MultinomialHazardOutcome if competing else LogisticHazardOutcome
    if type(process) is not expected:
        # The plasmode has no keyword of its own: it reads the simulated_outcome declaration.
        keyword = "simulated_outcome" if name == "plasmode" else name
        raise CapabilityError(
            f"{name} on a {'competing-risk' if competing else 'survival'} fit draws its "
            f"events from a declared hazard; pass {keyword}={expected.__name__}(...)",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if name == "dummy_outcome" and not process.independent:
        raise CapabilityError(
            "dummy_outcome draws events that read nothing; declare effect 0 and "
            "adjustment_scale 0, or run simulated_outcome for a hazard that reads W and A",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if competing and process.causes != len(data.cause_labels):
        raise CapabilityError(
            f"{name} declares {process.causes} cause(s), and the fit has "
            f"{len(data.cause_labels)}: {list(data.cause_labels)}",
            reason=RefusalReason.INVALID_DECLARATION,
        )


def _survival_generated_target(
    result: Any, estimand: str, process: Any, name: str
) -> _SurvivalGenerated:
    """Refuse a hazard a survival fit cannot read, else build its sample-conditional truth."""
    data = result.data
    _validate_hazard_process(data, process, name)
    if process.history_effect:
        raise CapabilityError(
            f"{name} keeps the held covariate history, whose law under a regimen is unknown, "
            "so a hazard that reads it has no held-history truth. Leave history_effect at 0, "
            "or run tests=('plasmode',), which draws the history from frozen models",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    known = getattr(data, "known_mechanisms", None)
    if (
        known is not None
        and data.censoring_names
        and any(known.censoring_at(node) is not None for node in range(1, data.n_times + 1))
    ):
        raise CapabilityError(
            f"{name} recodes each original event as censoring, so the censoring law is no "
            "longer the known one the fit declares",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    label, cause, horizon = _held_parameter(result, estimand, name)
    doses = _regimen_doses(result, label, horizon, name, static=not process.independent)
    shift = _hazard_adjustment(process, data)
    weights = np.asarray(data.weights, dtype=float)
    column = 0 if cause is None else list(data.cause_labels).index(cause)
    levels: list[float] = []
    for dose in doses:
        survival = np.ones(data.n, dtype=float)
        risk = np.zeros(data.n, dtype=float)
        for node in range(horizon):
            exposure = np.zeros(data.n) if dose is None else np.full(data.n, float(dose[node]))
            probabilities = process.probabilities(shift, exposure)
            risk += survival * probabilities[:, column]
            survival *= 1.0 - probabilities.sum(axis=1)
        levels.append(float(np.average(risk, weights=weights)))
    contrast = len(levels) == 2
    truth = levels[0] - levels[1] if contrast else levels[0]
    return _SurvivalGenerated(
        process=process,
        truth=float(truth),
        covariate_part=bool(process.independent),
        shares=_held_shares(_recoded_uncensored(data)),
    )


def _validate_generated_eligibility(
    result: Any, estimand: str, process: Any, name: str
) -> PointTarget | _LongitudinalGenerated | _SurvivalGenerated | _LongitudinalPlasmode:
    """Refuse an ineligible generated-outcome request and resolve its target.

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
    PointTarget, _LongitudinalGenerated or _SurvivalGenerated
        The point target with its arm codes and population. A longitudinal fit returns its
        held-history target instead.

    Raises
    ------
    CapabilityError
        If the fit, the estimand, or the process has no registered effect derivation.
    """
    if _is_longitudinal(result):
        return _longitudinal_generated_target(result, estimand, process, name)
    _validate_generated_process(process, name)
    if name == "plasmode":
        _validate_plasmode_eligibility(result)
    from ..study import (
        BackdoorMeanContrast,
        ExplicitAdjustmentProvider,
        ParameterKey,
        _matches_registered_point_identification,
    )
    from ..targets import TARGETS

    identified = getattr(result, "identified_effect", None)
    if identified is None:
        raise CapabilityError(
            f"{name} needs identification metadata for a backdoor additive mean contrast; "
            "this result records none",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    functional = identified.functional
    if type(functional) is not BackdoorMeanContrast:
        raise CapabilityError(
            f"{name} has no outcome process or effect derivation for {type(functional).__name__}.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Use a backdoor-identified effect.",
            roadmap="X82",
        )
    provider = getattr(identified, "provider", None)
    if type(provider) is not ExplicitAdjustmentProvider:
        raise CapabilityError(
            f"{name} needs registered backdoor provider provenance; "
            f"{type(provider).__name__} does not explicitly provide it",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    data = result.data
    if data.family != process.family:
        raise CapabilityError(
            f"{name} cannot replace an original outcome family={data.family!r} with the "
            f"registered {process.family!r} process {type(process).__name__}; declare the "
            "process of the fit's family (Gaussian or bounded for a continuous outcome, "
            "Bernoulli for a binary one).",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    estimator = result.estimator
    configured_family = getattr(estimator, "family", "auto")
    if configured_family not in {"auto", process.family}:
        raise CapabilityError(
            f"{name} cannot refit an estimator configured with family={configured_family!r} "
            f"under the declared {process.family!r} process",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    q_bounds = getattr(estimator, "q_bounds", None)
    bounded = isinstance(process, BoundedAdjustmentOutcome)
    if q_bounds is not None and process.family == "gaussian" and not bounded:
        raise CapabilityError(
            f"{name} replaces the outcome with a draw from the registered Gaussian "
            "process, whose support is the whole real line, and this fit declares "
            f"q_bounds={q_bounds!r}. The refit would scale the generated "
            "outcome into a range it is not confined to and report the refutation under "
            "bounds the process does not satisfy. Declare a BoundedAdjustmentOutcome, "
            "which draws inside the fit's q_bounds, or refute a fit with q_bounds=None "
            "(Targeting(q_bounds=None)) and cross_fit=False (CrossFitting(enabled=False))",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if bounded and q_bounds is None:
        raise CapabilityError(
            f"{name} with a BoundedAdjustmentOutcome draws inside the fit's q_bounds, and "
            "this fit declares none. Declare a GaussianAdjustmentOutcome for a fit with "
            "q_bounds=None",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if process.family == "gaussian":
        outcome_learner = getattr(estimator, "outcome_learner", None)
        incompatible_learner = _classification_only_learner(outcome_learner)
        if incompatible_learner is not None:
            raise CapabilityError(
                f"{name} needs a regression-capable saved outcome learner for its continuous "
                f"process; {type(incompatible_learner).__name__} is configured for "
                "classification",
                reason=RefusalReason.INVALID_DECLARATION,
            )
    if getattr(functional, "longitudinal", False):
        raise CapabilityError(
            f"{name} has no longitudinal outcome process or sequential effect derivation.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="X82",
        )
    if getattr(data, "is_continuous_treatment", False):
        _refuse_continuous_point_target(functional, process, name)
    if getattr(process, "history_effect", 0.0):
        raise CapabilityError(
            f"{name} declares history_effect={process.history_effect:g}, which only the "
            "longitudinal plasmode reads: a point fit has no time-varying covariates. Leave "
            "history_effect at 0",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    has_intermediate = bool(getattr(data, "has_intermediate", False))
    if getattr(process, "intermediate_effect", 0.0) and not has_intermediate:
        raise CapabilityError(
            f"{name} declares intermediate_effect={process.intermediate_effect:g}, and this "
            "fit has no intermediate for it to read. Leave intermediate_effect at 0",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if getattr(process, "cluster_sd", 0.0) and (name != "plasmode" or data.cluster is None):
        raise CapabilityError(
            f"{name} declares cluster_sd={process.cluster_sd:g}, which only the plasmode of a "
            "clustered fit draws: the held design keeps every row and draws independent "
            "noise. Leave cluster_sd at 0",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    axis = functional.axis
    if axis not in _POINT_TARGET_AXES or (axis == "arm" and functional.interventions):
        raise CapabilityError(
            f"{name} has no outcome process or effect derivation for the {axis!r} axis. "
            "The generated laws cover arm targets, regimes, incremental tilts, modified "
            "treatment policies and MSM coefficients. A risk-ratio tilt reads the true "
            "treatment mechanism, so only a plasmode can read it, and that plasmode is not "
            "built yet; learned rules have no generated truth yet.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    if (functional.msm is not None) != (axis == "msm"):
        raise CapabilityError(
            f"{name} found inconsistent registered target provenance: an MSM on the {axis!r} axis",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if axis == "ipsi" and name != "plasmode":
        raise CapabilityError(
            f"{name} has no truth for an incremental tilt: the tilt multiplies the odds of "
            "the true treatment mechanism, which the held design does not know.",
            reason=RefusalReason.API_REDIRECTION,
            alternative=(
                "Run tests=('plasmode',), whose frozen treatment model is the true "
                "mechanism of its generated law."
            ),
        )
    if axis == "policy" and name != "plasmode" and getattr(data, "has_missing_treatment", False):
        # A policy moves each row's natural arm, and the held design reads the held arm as
        # that natural arm. An unrecorded row has none.
        raise CapabilityError(
            f"{name} has no held-design truth for a modified treatment policy on a fit with "
            "a missing treatment: the policy moves each row's natural treatment, and an "
            "unrecorded row's treatment is unknown.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    key = result.parameter_keys.get(estimand)
    if type(key) is not ParameterKey:
        raise CapabilityError(
            f"{name} needs a structured parameter key for {estimand!r}; choose one of "
            f"{list(result.parameter_keys)}",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    kind = _POINT_TARGET_KINDS.get(key.estimand)
    if kind is None:
        raise CapabilityError(
            f"{name} has no {key.estimand} effect derivation.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=(
                "Choose an arm contrast (ate, att, atc), an arm level (ey, ey1, ey0), a "
                "ratio (rr, or), a natural-course target (ey_obs, par, paf), a regime "
                "target (ey_regime, ate_regime), an incremental target (ey_ipsi, ate_ipsi), "
                "a policy target (ey_policy, ate_policy) or an MSM coefficient (msm)."
            ),
            roadmap="X82",
        )
    if key.estimand in ("att", "atc") and getattr(data, "has_missing_treatment", False):
        # The held design knows neither the arm of an unrecorded row nor g(v | W), so it
        # cannot sum the treated or the control population (part E plan review, R1).
        raise CapabilityError(
            f"{name} has no held-design truth for {key.estimand} on a fit with a missing "
            "treatment: an unrecorded row's arm, and so the target population, is unknown.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    declared_estimand = getattr(getattr(identified, "estimand", None), "name", None)
    expected_estimand = "ey" if key.estimand in ("ey1", "ey0") else key.estimand
    if functional.intermediate is not None:
        expected_estimand = "controlled_direct_effect"
    if functional.target != key.estimand or declared_estimand != expected_estimand:
        raise CapabilityError(
            f"{name} found inconsistent registered target provenance: functional target "
            f"{functional.target!r}, identified estimand {declared_estimand!r}, and parameter "
            f"key {key.estimand!r} must agree",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    registered = TARGETS.get(functional.target)
    declared_identification = getattr(identified, "identification", None)
    if (
        registered is None
        or declared_identification is None
        or not _matches_registered_point_identification(
            declared_identification,
            registered.identification,
            identified,
            data,
            functional.axis,
        )
    ):
        raise CapabilityError(
            f"{name} needs the registered identification artifact for target {functional.target!r}",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if has_intermediate and functional.intermediate is None:
        raise CapabilityError(
            f"{name} found an intermediate on the data and none on the functional; the "
            "controlled direct effect fixes it at a declared value",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    shared: dict[str, Any] = {
        "outcome_bounds": _bounds_tuple(q_bounds) if bounded else None,
        "stratum": _stratum_code(data, key, name),
        "intermediate": (
            None if functional.intermediate is None else float(result.intermediate_value)
        ),
    }
    codes = tuple(float(code) for code in data.arm_codes)
    if kind == "level":
        endpoint = key.value
        if endpoint is None:
            endpoint = 1 if key.estimand == "ey1" else 0
        value_code = _resolve_arm_code(data, endpoint, "the parameter key value", name)
        target = PointTarget(
            kind="level", population="all", value_code=value_code, reference_code=None, **shared
        )
    elif kind == "natural":
        target = PointTarget(
            kind="natural",
            population="all",
            value_code=None,
            reference_code=None,
            arms=codes,
            **shared,
        )
    elif kind in ("intervention", "intervention_difference"):
        if axis == "policy":
            laws = _policy_laws(result, functional, key, name)
        else:
            laws = _intervention_laws(result, functional, key, name)
        laws.setdefault("arms", codes)
        target = PointTarget(
            kind=kind,
            population="all",
            value_code=None,
            reference_code=None,
            **shared,
            **laws,
        )
    elif kind == "msm":
        target = PointTarget(
            kind="msm",
            population="all",
            value_code=None,
            reference_code=None,
            arms=codes,
            projection=_msm_projection(result, functional, key, name),
            **shared,
        )
    elif kind in ("par", "paf"):
        # The key value of ``par`` and ``paf`` is the reference arm r of ``ey_obs - ey[r]``.
        endpoint = key.value if key.value is not None else data.arm_codes[0]
        value_code = _resolve_arm_code(data, endpoint, "the parameter key value", name)
        target = PointTarget(
            kind=kind,
            population="all",
            value_code=value_code,
            reference_code=None,
            arms=codes,
            **shared,
        )
    else:
        value_code, reference_code = _contrast_codes(data, key, name)
        population = {"att": "value", "atc": "reference"}.get(key.estimand, "all")
        target = PointTarget(
            kind=kind,
            population=population,  # type: ignore[arg-type]
            value_code=value_code,
            reference_code=reference_code,
            **shared,
        )
    _validate_generated_law(target, process, data, name)
    return target


def _stratum_code(data: Any, key: Any, name: str) -> int | None:
    """Return the code of the stratum a parameter key names, or ``None`` for every row."""
    if key.stratum is None:
        return None
    levels = tuple(getattr(data, "strata_levels", ()))
    if tuple(key.stratum) not in levels:
        raise CapabilityError(
            f"{name} found a parameter key for stratum {key.stratum!r}, which the data does "
            f"not declare among {list(levels)}",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    return levels.index(tuple(key.stratum))


#: The point targets of the catalog and the functional of the arm levels each reads.
_POINT_TARGET_KINDS: dict[str, TargetKind] = {
    "ate": "difference",
    "att": "difference",
    "atc": "difference",
    "ey": "level",
    "ey1": "level",
    "ey0": "level",
    "rr": "rr",
    "or": "or",
    "ey_obs": "natural",
    "par": "par",
    "paf": "paf",
    "ey_regime": "intervention",
    "ate_regime": "intervention_difference",
    "ey_ipsi": "intervention",
    "ate_ipsi": "intervention_difference",
    "ey_policy": "intervention",
    "ate_policy": "intervention_difference",
    "msm": "msm",
}

#: The parameter axes whose targets have a point generated law (part E3).
_POINT_TARGET_AXES = ("arm", "regime", "ipsi", "policy", "msm")


def _refuse_continuous_point_target(functional: Any, process: Any, name: str) -> None:
    """Refuse a continuous-treatment request that has no generated truth (part E3b).

    The held design has a policy truth on a continuous treatment, because each row's held
    dose is its natural dose. An arm target has no arms to sum over. The plasmode would need
    a frozen law of the dose, which part E3b does not draw.
    """
    if functional.axis != "policy":
        raise CapabilityError(
            f"{name} has no {functional.axis!r} target on a continuous treatment; the "
            "arm-indexed truths need declared arms. A modified treatment policy has a "
            "held-design truth on a continuous treatment.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        )
    if name == "plasmode":
        raise CapabilityError(
            f"{name} has no law of a continuous treatment: the plasmode draws each arm from "
            "a frozen treatment model, and a continuous dose has no frozen model yet.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=(
                "Run tests=('simulated_outcome',), whose held design reads each row's held "
                "dose as its natural dose."
            ),
            roadmap="X82",
        )
    if getattr(process, "arm_effects", None) is not None:
        raise CapabilityError(
            f"{name} declares arm_effects on a continuous treatment, which has no arms. "
            "Declare the scalar effect, whose term is effect times the dose",
            reason=RefusalReason.INVALID_DECLARATION,
        )


def _policy_laws(result: Any, functional: Any, key: Any, name: str) -> dict[str, Any]:
    """Return the value and reference laws of a modified treatment policy (part E3b).

    On an arm treatment each law is the policy's transition ``T[i, s, a]``, the probability
    that the policy moves natural arm ``s`` to arm ``a`` at row ``i``. On a continuous
    treatment each branch of a policy gets its own code, the code's dose at each row is the
    branch applied to the held dose, and the law puts the branch's probability on that code.
    """
    from ..interventions.policy import discrete_assignments, lazy_frame, policy_branches

    data = result.data
    declared: tuple[Any, ...] = tuple(functional.interventions)
    labels = [str(policy.name) for policy in declared]
    reference = key.reference if key.estimand == "ate_policy" else None
    if key.value not in labels or (reference is not None and reference not in labels):
        raise CapabilityError(
            f"{name} found a parameter key for {key.value!r} against {reference!r}, which "
            f"the declared policies {labels} do not name",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    wanted = [key.value] if reference is None else [key.value, reference]
    policies = {str(policy.name): policy for policy in declared}
    frame = lazy_frame(
        lambda: data.frame_like(
            {column: data.covariates[:, j] for j, column in enumerate(data.covariate_names)}
        )
    )
    try:
        if data.is_continuous_treatment:
            return _continuous_policy_laws(policies, wanted, data, frame, policy_branches)
        levels = tuple(data.treatment_levels)
        rows = np.ones(data.n, dtype=bool)
        laws: dict[str, Any] = {}
        for label in wanted:
            transition = np.zeros((data.n, len(levels), len(levels)), dtype=float)
            assignments = discrete_assignments(policies[label], levels, frame, rows)
            for probability, assigned in assignments:
                np.add.at(
                    transition,
                    (np.arange(data.n)[:, None], np.arange(len(levels))[None, :], assigned),
                    probability,
                )
            laws[label] = InterventionLaw(label, transition=transition)
    except CapabilityError as error:
        raise CapabilityError(
            f"{name} cannot evaluate the declared policy on the fitted rows: {error}",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=_STABILITY_ALTERNATIVE,
            roadmap="X82",
        ) from error
    return {
        "value_law": laws[key.value],
        "reference_law": None if reference is None else laws[reference],
    }


def _continuous_policy_laws(
    policies: dict[str, Any],
    wanted: list[str],
    data: Any,
    frame: Any,
    policy_branches: Any,
) -> dict[str, Any]:
    """Return the branch codes, their doses, and the laws of continuous-dose policies."""
    held = np.asarray(data.treatment, dtype=float)
    doses: dict[float, np.ndarray] = {}
    columns: dict[str, list[tuple[float, float]]] = {}
    for label in wanted:
        columns[label] = []
        for probability, branch in policy_branches(policies[label]):
            code = float(len(doses))
            doses[code] = np.asarray(branch.assign(held, frame), dtype=float).reshape(-1)
            columns[label].append((code, float(probability)))
    arms = tuple(doses)
    laws: dict[str, InterventionLaw] = {}
    for label in wanted:
        values = np.zeros((data.n, len(arms)), dtype=float)
        for code, probability in columns[label]:
            values[:, arms.index(code)] = probability
        laws[label] = InterventionLaw(label, values=values)
    reference = wanted[1] if len(wanted) > 1 else None
    return {
        "arms": arms,
        "doses": doses,
        "value_law": laws[wanted[0]],
        "reference_law": None if reference is None else laws[reference],
    }


def _intervention_laws(result: Any, functional: Any, key: Any, name: str) -> dict[str, Any]:
    """Return the value and reference laws of a regime or incremental target (part E3).

    A regime's law is its declared density on the fitted rows, which the regime evaluates
    from ``W`` alone. An incremental tilt's law is its odds factors, and the truth applies
    them to the true mechanism of the generated law.
    """
    from ..interventions import RegimeSet
    from ..interventions.base import as_interventions

    data = result.data
    declarations: tuple[Any, ...] = tuple(functional.interventions)
    if functional.axis == "regime":
        declarations = as_interventions(functional.interventions)
    labels = tuple(str(item.name) for item in declarations)
    if functional.axis == "regime":
        evaluated = RegimeSet.evaluate(declarations, data, reference=functional.reference)
        laws = {
            label: InterventionLaw(label, values=np.asarray(evaluated.values[:, :, index]))
            for index, label in enumerate(evaluated.names)
        }
    else:
        laws = {
            str(item.name): InterventionLaw(
                str(item.name),
                factors=np.asarray(item.factors(data.treatment_levels), dtype=float),
            )
            for item in declarations
        }
    reference = None
    if key.estimand.startswith("ate_"):
        reference = key.reference
    if key.value not in laws or (reference is not None and reference not in laws):
        raise CapabilityError(
            f"{name} found a parameter key for {key.value!r} against {reference!r}, which "
            f"the declared interventions {list(labels)} do not name",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    return {
        "value_law": laws[key.value],
        "reference_law": None if reference is None else laws[reference],
    }


def _msm_projection(result: Any, functional: Any, key: Any, name: str) -> Projection:
    """Return the working model of an MSM coefficient on the fitted rows (part E3)."""
    from ..msm import MSMSet

    model = functional.msm
    if str(model.link) not in PROJECTION_LINKS:
        raise CapabilityError(
            f"{name} has no projection truth for an MSM with link {model.link!r}.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=f"Declare the working model with a link in {list(PROJECTION_LINKS)}.",
            roadmap="X82",
        )
    evaluated = MSMSet.evaluate(model, result.data)
    if key.term not in evaluated.terms:
        raise CapabilityError(
            f"{name} found a parameter key for the term {key.term!r}, which the working "
            f"model {list(evaluated.terms)} does not name",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    return Projection(
        design=np.asarray(evaluated.design, dtype=float),
        weights=np.asarray(evaluated.weights, dtype=float),
        link=str(model.link),
        term=evaluated.terms.index(key.term),
    )


def _bounds_tuple(bounds: Any) -> tuple[float, float]:
    return (float(bounds[0]), float(bounds[1]))


def _contrast_codes(data: Any, key: Any, name: str) -> tuple[float, float]:
    """Resolve the value and reference arm codes of a contrast key."""
    # The key names the contrast direction, and the process declares its arm term per arm
    # code. A fit reported against a non-default reference therefore has the negated
    # effect, and reading the key is the only way to see it.
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
                f"{data.treatment_name} does not leave exactly one other arm",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        value_code = remaining[0]
    else:
        value_code = _resolve_arm_code(data, value_endpoint, "the parameter key value", name)
    if value_code == reference_code:
        raise CapabilityError(
            f"{name} has no effect derivation for a parameter key whose value and reference "
            f"are the same arm ({value_code:g}); the catalog covers a two-arm contrast",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    return value_code, reference_code


def _validate_generated_law(target: PointTarget, process: Any, data: Any, name: str) -> None:
    """Refuse a declared law that is not a law on these rows, before any refit.

    Every arm mean must be admissible at every row: a risk in ``[0, 1]``, or a bounded mean
    strictly inside ``q_bounds``. The plasmode resamples these rows, so the check covers it.
    A ratio needs positive levels, and an odds ratio levels below 1.
    """
    covariates = np.asarray(data.covariates, dtype=float)
    codes = tuple(float(code) for code in data.arm_codes)
    if target.doses is not None:
        # A continuous treatment has no arms. Its codes are the policy branches, and the
        # law must be admissible at the dose each branch assigns.
        codes = target.codes
    # A process with an intermediate term is a law only if its mean is admissible at both
    # values of the binary intermediate, since the held rows and the plasmode read both.
    settings: tuple[float | None, ...] = (None,)
    if getattr(data, "has_intermediate", False):
        settings = (0.0, 1.0)
    admissible = getattr(process, "admissible", None)
    for setting in settings:
        try:
            means = arm_means(
                process,
                covariates,
                codes,
                data,
                bounds=target.outcome_bounds,
                intermediate=setting,
                doses=target.doses,
            )
        except ValueError as error:
            raise CapabilityError(
                f"{name} cannot evaluate {type(process).__name__} on this fit: {error}",
                reason=RefusalReason.INVALID_DECLARATION,
            ) from error
        if admissible is None:
            continue
        for code, values in means.items():
            if not admissible(values, bounds=target.outcome_bounds):
                place = "[0, 1]" if target.outcome_bounds is None else "inside q_bounds"
                raise CapabilityError(
                    f"{name} declares a {type(process).__name__} whose mean leaves {place} "
                    f"on some row of arm {code:g}; the declaration is not a law on these "
                    "rows. Lower adjustment_scale or move the intercept or effects",
                    reason=RefusalReason.INVALID_DECLARATION,
                )
    if target.ratio or target.kind == "paf":
        means = arm_means(
            process,
            covariates,
            target.codes,
            data,
            bounds=target.outcome_bounds,
            intermediate=target.intermediate,
            doses=target.doses,
        )
        levels = target.levels(means, held_weights(target, data), held_mixture(target, data))
        upper = 1.0 if target.kind == "or" else np.inf
        if not all(0.0 < level < upper for level in levels):
            raise CapabilityError(
                f"{name} has no {target.kind} truth under this declaration: a level "
                f"{[round(level, 6) for level in levels]} it divides by is outside "
                f"(0, {upper:g})",
                reason=RefusalReason.INVALID_DECLARATION,
            )


def _run_empirical_refits(
    result: Any,
    *,
    estimand: str,
    draws: Sequence[tuple[int, int]],
    replacement: Callable[[int, int], Any],
    expected_family: str,
    retain_preparation_failures: bool = False,
    refit_options: Callable[[int], dict[str, Any]] | None = None,
    refit_estimator: Callable[[int], Any] | None = None,
) -> tuple[tuple[EmpiricalRefitRecord, ...], tuple[ReplicationFailure, ...]]:
    """Run, validate, and retain empirical refits through one failure contract.

    Parameters
    ----------
    result : TMLEResult
        Fitted result that supplies the estimator and its intermediate value.
    estimand : str
        Structured result alias read off each refit.
    draws : sequence of tuple of int
        One ``(replicate, seed)`` pair per requested draw.
    replacement : callable
        Builds the data for one draw from its replicate index and seed.
    expected_family : str
        Outcome family every successful refit must report.
    retain_preparation_failures : bool
        Whether a failure inside ``replacement`` becomes a retained failure. ``False``
        raises it, because a deterministic preparation fails the same way on every draw
        and the caller must see that declaration or data error at once. ``True`` retains
        it, because a bootstrap draw fails per sample: a resample can hold too few rows,
        and ``run_bootstrap`` drops exactly that draw. Keep the two paths apart. A refit
        failure is retained either way.
    refit_options : callable or None
        Maps a replicate index to extra keyword arguments of the refit, for example the
        frozen fold labels of a resample.
    refit_estimator : callable or None
        Maps a replicate index to the estimator that refits it, for example the frozen
        rules of a learned-rule fit on the rows of a resample. ``None`` refits with
        ``result.estimator``.

    Returns
    -------
    tuple of tuple
        Successful records, then the retained failures.
    """
    records: list[EmpiricalRefitRecord] = []
    failures: list[ReplicationFailure] = []
    for replicate, child_seed in draws:
        try:
            prepared = replacement(replicate, child_seed)
        except Exception as error:
            if not retain_preparation_failures:
                raise
            # A bootstrap sample this draw cannot build, or a perturbation the sample
            # refuses, is one failed draw among many. Raising would discard every draw
            # that already succeeded, so record it with the fields a refit failure carries.
            failures.append(
                ReplicationFailure(
                    replicate=replicate,
                    seed=child_seed,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            )
            continue
        try:
            estimator = result.estimator if refit_estimator is None else refit_estimator(replicate)
            refitted = estimator.refit(
                prepared,
                **_intermediate_option(result),
                random_state=child_seed,
                **({} if refit_options is None else refit_options(replicate)),
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
        if not np.isfinite(estimate.psi) or not np.isfinite(estimate.plugin_std_error):
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
                std_error=float(estimate.plugin_std_error),
                family=fitted_family,
            )
        )
    return tuple(records), tuple(failures)


def _measurement_encodings(data: Any) -> dict[str, Any]:
    return {encoding.column: encoding for encoding in data.encodings}


def _validate_numeric_measurement_column(data: Any, name: str) -> None:
    """Refuse a selected column that relative Gaussian noise cannot perturb coherently.

    Parameters
    ----------
    data : CausalData
        Original data the fit ran on.
    name : str
        Selected adjustment variable that carries no ``CategoricalEncoding``.
    """
    column = data.covariate_names.index(name)
    values = np.asarray(data.covariates[:, column], dtype=float)
    # A column with no encoding is treated as numeric, and ``CausalData.from_arrays``
    # records no encoding at all.  An undeclared 0/1 column therefore reaches the numeric
    # path, where continuous noise would make it real-valued, and a hand-built dummy block
    # would have each indicator perturbed on its own.  Refuse both here, before any refit.
    if bool(np.all(np.isin(values, (0.0, 1.0)))):
        raise CapabilityError(
            f"bootstrap_measurement_error cannot perturb {name!r}: its values are all 0 or 1 "
            "but it carries no CategoricalEncoding, so the operation cannot tell an "
            "undeclared binary variable from one indicator of a dummy block, and relative "
            "Gaussian noise would make it real-valued. Supply the variable as a boolean or "
            "categorical dataframe column to CausalData.from_frame, so it carries the "
            "logical metadata this operation perturbs by",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if float(np.std(values, ddof=0)) == 0.0:
        raise CapabilityError(
            f"bootstrap_measurement_error cannot perturb {name!r}: the variable is constant, "
            "so its relative noise scale is exactly zero and every draw would return the "
            "original column while the report still named the variable as perturbed. Remove "
            "it from the declaration, or declare an absolute noise law for it",
            reason=RefusalReason.INVALID_DECLARATION,
        )


def _validate_measurement_error_eligibility(
    result: Any, declaration: BootstrapMeasurementError, n_replicates: int
) -> Any:
    """Validate the complete bootstrap measurement-error request before refitting."""
    from ..data import CausalData
    from ..longitudinal.data import LongitudinalData

    data: Any = getattr(result, "data", None)
    if type(data) not in (CausalData, LongitudinalData):
        raise CapabilityError(
            "bootstrap_measurement_error supports CausalData and LongitudinalData results; "
            f"got result family {type(result).__name__} with data family {type(data).__name__}.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="X82",
        )
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    if declaration.resampling == "cluster" and data.cluster is None:
        raise CapabilityError(
            "resampling='cluster' requires the data to carry cluster ids",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if type(data) is LongitudinalData:
        _validate_node_measurement(data, declaration)
        return data
    if not _perturbed_variables(data, declaration):
        raise CapabilityError(
            "bootstrap_measurement_error holds the strata variables at their recorded values, "
            "so the reported strata keep their meaning, and this declaration names only "
            f"strata variables {sorted(declaration.variables)}. No column is left to perturb.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative=(
                "Declare BootstrapMeasurementError(variables=...) with at least one adjustment "
                "variable other than the strata."
            ),
        )
    encodings = _measurement_encodings(data)
    generated = data.indicator_sources
    numeric = set(data.covariate_names).difference(generated)
    original = numeric.union(encodings)
    unknown = [name for name in declaration.variables if name not in original]
    if unknown:
        indicator = next((name for name in unknown if name in generated), None)
        if indicator is not None:
            raise CapabilityError(
                f"{indicator!r} is a generated indicator for original categorical variable "
                f"{generated[indicator]!r}.",
                reason=RefusalReason.API_REDIRECTION,
                alternative=f"Select the original variable {generated[indicator]!r}.",
            )
        raise CapabilityError(
            f"unknown adjustment variables {unknown}; choose from {sorted(original)}",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    retained = set(data.covariate_names)
    for name in _perturbed_variables(data, declaration):
        encoding = encodings.get(name)
        if encoding is None:
            _validate_numeric_measurement_column(data, name)
            continue
        missing = [indicator for indicator in encoding.generated if indicator not in retained]
        if missing:
            raise CapabilityError(
                "bootstrap_measurement_error cannot perturb categorical variable "
                f"{name!r} because duplicate-column removal left its encoded block "
                f"incomplete; missing indicators {missing}",
                reason=RefusalReason.INVALID_DECLARATION,
            )
    return data


def _perturbed_variables(data: Any, declaration: BootstrapMeasurementError) -> tuple[str, ...]:
    """Return the declared variables a point draw perturbs: every one but the strata."""
    held = set(getattr(data, "strata_names", ()))
    return tuple(name for name in declaration.variables if name not in held)


def _held_strata(data: Any, declaration: BootstrapMeasurementError) -> tuple[str, ...]:
    """Return the declared strata variables, which a point draw holds at recorded values."""
    held = set(getattr(data, "strata_names", ()))
    return tuple(name for name in declaration.variables if name in held)


def _node_columns(data: Any, name: str) -> tuple[tuple[int | None, int], ...]:
    """Locate ``name`` in a panel: ``(None, j)`` for a baseline column, ``(t, j)`` at node t."""
    places: list[tuple[int | None, int]] = []
    if name in data.baseline_names:
        places.append((None, data.baseline_names.index(name)))
    for time, names in enumerate(data.time_varying_names):
        for column, label in enumerate(names):
            # A long panel names a time-varying column ``name@t`` at node t.
            if label == name or label.rpartition("@")[0] == name:
                places.append((time, column))
    return tuple(places)


def _node_values(data: Any, place: tuple[int | None, int]) -> np.ndarray:
    time, column = place
    block = data.baseline if time is None else data.time_varying[time]
    return np.asarray(block[:, column], dtype=float)


def _validate_node_measurement(data: Any, declaration: BootstrapMeasurementError) -> None:
    """Refuse a panel declaration that names no perturbable covariate column."""
    outcomes = set(getattr(data, "repeated_names", ())) | {data.outcome_name}
    node_names = {
        label.rpartition("@")[0] or label for names in data.time_varying_names for label in names
    }
    available = sorted(set(data.baseline_names).union(node_names).difference(outcomes))
    for name in declaration.variables:
        if name in outcomes:
            raise CapabilityError(
                f"bootstrap_measurement_error cannot perturb {name!r}: it is an outcome of "
                "this panel. A later node reads it as a covariate, but the error would then "
                "differ between the outcome and its copy",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        places = _node_columns(data, name)
        if not places:
            raise CapabilityError(
                f"unknown covariate {name!r}; choose from the baseline and node covariates "
                f"{available}",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        for place in places:
            values = _node_values(data, place)
            finite = values[np.isfinite(values)]
            where = "baseline" if place[0] is None else f"node {place[0] + 1}"
            if finite.size and bool(np.all(np.isin(finite, (0.0, 1.0)))):
                raise CapabilityError(
                    f"bootstrap_measurement_error cannot perturb {name!r} at {where}: its "
                    "values are all 0 or 1, and a panel carries no categorical metadata, so "
                    "relative Gaussian noise would make a binary column real-valued",
                    reason=RefusalReason.INVALID_DECLARATION,
                )
            if not finite.size or float(np.std(finite)) == 0.0:
                raise CapabilityError(
                    f"bootstrap_measurement_error cannot perturb {name!r} at {where}: the "
                    "column is constant there, so its relative noise scale is zero",
                    reason=RefusalReason.INVALID_DECLARATION,
                )


def _perturb_node_measurement(
    data: Any, declaration: BootstrapMeasurementError, rng: np.random.Generator
) -> Any:
    """Perturb each declared panel column at every node where it appears.

    The noise scale of each appearance is the relative scale times the column's standard
    deviation at that node, on the rows where it is recorded. A missing value stays missing.
    """
    baseline = np.array(data.baseline, dtype=float, copy=True)
    blocks = [np.array(block, dtype=float, copy=True) for block in data.time_varying]
    for name in declaration.variables:
        for time, column in _node_columns(data, name):
            target = baseline if time is None else blocks[time]
            values = target[:, column]
            present = np.isfinite(values)
            values[present] += declaration.numeric_noise.draw(rng, values[present])
    return replace(data, baseline=baseline, time_varying=tuple(blocks))


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
    """Perturb sampled adjustment variables and rebuild complete categorical blocks.

    A panel perturbs each declared column at every node where it appears. A point draw
    holds the strata variables at their recorded values.
    """
    rng = np.random.default_rng(seed)
    if hasattr(data, "time_varying"):
        return _perturb_node_measurement(data, declaration, rng)
    values = np.array(data.covariates, dtype=float, copy=True)
    encodings = _measurement_encodings(data)
    for name in _perturbed_variables(data, declaration):
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


class _CopyFolds:
    """Keep every bootstrap copy of a unit in the fold of the original unit.

    A refit that redrew its folds on a resample would put copies of one unit in different
    folds, so a copy would train the nuisance that predicts another copy. Each resampled row
    takes the fold label of the row it copies (X82 plan 5.4). A stand-in result that holds
    no repeats leaves the refit to draw its own folds.
    """

    def __init__(self, result: Any, data: Any, design: Any, draws: Mapping[int, Any]) -> None:
        self._result = result
        self._data = data
        self._design = design
        self._draws = draws
        self._index: dict[int, np.ndarray] = {}
        #: Repeat seeds of the original fit that every refit reuses, or ``None`` to let the
        #: refit draw its own.
        self.frozen_seeds: Sequence[int] | None = None

    def sample(self, replicate: int) -> Any:
        index, codes = self._design.indices(self._data, self._draws[replicate])
        self._index[replicate] = index
        sample = self._data.subset(index)
        return sample if codes is None else replace(sample, cluster=codes)

    def rows(self, replicate: int) -> np.ndarray:
        """The original row of each row of the sample drawn for ``replicate``."""
        return self._index[replicate]

    def options(self, replicate: int) -> dict[str, Any]:
        sliced = _sliced_folds(self._result, self._index[replicate])
        options: dict[str, Any] = {} if sliced is None else {"frozen_folds": sliced}
        if self.frozen_seeds is not None:
            options["frozen_seeds"] = self.frozen_seeds
        return options


def _validate_stability_eligibility(result: Any, n_replicates: int) -> Any:
    """Validate a ``bootstrap_stability`` request before it refits."""
    from ..data import CausalData
    from ..longitudinal.data import LongitudinalData

    data = getattr(result, "data", None)
    if type(data) not in (CausalData, LongitudinalData):
        raise CapabilityError(
            "bootstrap_stability supports CausalData and LongitudinalData results; "
            f"got result family {type(result).__name__} with data family {type(data).__name__}.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative=NO_ALTERNATIVE,
            roadmap="X82",
        )
    if n_replicates < 2:
        raise CapabilityError(
            f"bootstrap_stability was asked for {n_replicates} draw(s); a spread needs at least 2",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    return data


def _bootstrap_stability_test(
    result: Any,
    *,
    estimand: str,
    original: float,
    n_replicates: int,
    root_seed: int,
    is_ratio: bool = False,
) -> RefutationTest:
    """Refit on bootstrap resamples that keep each unit's fold, and describe the spread.

    The original fit is the same-seed full-data baseline: every refit reuses its repeat
    seeds and each copy of a unit keeps the unit's original fold, so the shift of the mean
    against the original fit carries no fold or seed noise. No theorem covers a refit
    bootstrap of a cross-fitted estimator, so the report stays descriptive (X82 plan 5.4).
    A ratio reads on the log scale, with a geometric mean, as ``subset`` does.

    A fit given ``split_plan=`` refuses this test. The frozen folds could carry the plan's
    labels the way ``subset`` does, which is a possible extension.
    """
    data = _validate_stability_eligibility(result, n_replicates)
    design = _bootstrap_design(
        data, n_replicates=n_replicates, resampling="auto", random_state=root_seed
    )
    draw_by_replicate = {draw.replicate: draw for draw in design.draws}
    folds = _CopyFolds(result, data, design, draw_by_replicate)
    repeats = getattr(result, "repeats", None)
    folds.frozen_seeds = [item.seed for item in repeats] if repeats else None
    if is_structural(result):
        folds.frozen_seeds = structural_fit_seeds(result)
    from ..interventions.learned import frozen_rule_replay

    frozen = _frozen_rule(result)
    records, failures = _run_empirical_refits(
        result,
        estimand=estimand,
        draws=tuple((draw.replicate, draw.seed) for draw in design.draws),
        replacement=lambda replicate, seed: folds.sample(replicate),
        refit_options=folds.options,
        refit_estimator=(
            None
            if frozen is None
            else lambda replicate: frozen_rule_replay(
                result.estimator, frozen, folds.rows(replicate)
            )
        ),
        expected_family=data.family,
        retain_preparation_failures=True,
    )
    values = tuple(item.estimate for item in records)
    scaled = np.log(values) if is_ratio else np.asarray(values, dtype=float)
    centre = float(np.log(original)) if is_ratio else float(original)
    shift = float(np.mean(scaled) - centre) if values else float("nan")
    spread = float(np.std(scaled, ddof=1)) if len(values) > 1 else float("nan")
    scale = " on the log scale" if is_ratio else ""
    return RefutationTest(
        name="bootstrap_stability",
        estimand=estimand,
        original=original,
        values=values,
        expectation=f"spread {spread:.4g}{scale} around the original fit",
        passed=None,
        detail=(
            f"{len(values)} of {n_replicates} resample refits succeeded; the mean moved "
            f"{shift:.4g}{scale} from the same-seed full-data fit"
        ),
        standard_errors=tuple(item.std_error for item in records),
        failures=failures,
        requested_draws=n_replicates,
        resampling=design.resampling,
        rung="L0",
        scope_sentence=_STABILITY_SENTENCE,
        extras=(("shift against the original fit", shift),),
        conditions=(_STABILITY_SENTENCE,),
        evidence=_interpretation.PLAIN_NONE,
        mean_kind="geometric" if is_ratio else "arithmetic",
        spread_kind="log" if is_ratio else "reported",
    )


_STABILITY_SENTENCE = (
    "The draws describe how far the estimate moves under resampling of units that keep their "
    "folds. Copy-fold resampling trains on fewer distinct units, so the spread tends to exceed "
    "the standard error. No result covers a refit bootstrap of a cross-fitted estimator, so "
    'the report states no verdict. The random-subsamples test ("subset") is the calibrated '
    "stability check."
)


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
    folds = _CopyFolds(result, data, design, draw_by_replicate)
    from ..interventions.learned import frozen_rule_replay

    # A learned-rule fit holds each source unit's realized rule, the rule at its true
    # covariates, so every refit targets the value of the fit's own rules (X82 part F4).
    frozen = _frozen_rule(result)
    records, failures = _run_empirical_refits(
        result,
        estimand=estimand,
        draws=tuple((draw.replicate, draw.seed) for draw in design.draws),
        replacement=lambda replicate, seed: _perturb_measurement_error(
            folds.sample(replicate), declaration, seed=seed
        ),
        refit_options=folds.options,
        refit_estimator=(
            None
            if frozen is None
            else lambda replicate: frozen_rule_replay(
                result.estimator, frozen, folds.rows(replicate)
            )
        ),
        expected_family=data.family,
        # A cluster or stratum bootstrap draw can hold too few rows to subset, and that is
        # a property of the draw rather than of the request. ``run_bootstrap`` drops such a
        # draw, so this operation retains it.
        retain_preparation_failures=True,
    )
    values = tuple(item.estimate for item in records)
    # The rule's tail probability is a number to print. The declared noise changes the
    # target, so "the draws include the original" describes the movement and tests nothing.
    _, probability, detail = rule.evaluate(values, original, failures)
    families = {item.family for item in records}
    family = next(iter(families)) if len(families) == 1 else None
    if len(families) > 1:
        detail = f"successful refits reported inconsistent outcome families {sorted(families)}"
    sentences = [_MEASUREMENT_ERROR_SENTENCE]
    if hasattr(data, "time_varying"):
        located = []
        for name in declaration.variables:
            nodes = [
                "baseline" if time is None else f"node {time + 1}"
                for time, _ in _node_columns(data, name)
            ]
            located.append(f"{name} at {', '.join(nodes)}")
        sentences.append(
            "The perturbation acts at every node where a declared column appears: "
            + "; ".join(located)
            + "."
        )
    else:
        held = _held_strata(data, declaration)
        if held:
            sentences.append(
                f"The strata variables {list(held)} keep their recorded values, so each "
                "reported stratum keeps its rows."
            )
    return RefutationTest(
        name="bootstrap_measurement_error",
        estimand=estimand,
        original=original,
        values=values,
        expectation=f"moves from {original:.5g} under the declared error",
        passed=None,
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
        rung="L0",
        scope_sentence=" ".join(sentences),
        conditions=tuple(sentences),
        evidence=_interpretation.PLAIN_NONE,
    )


_MEASUREMENT_ERROR_SENTENCE = (
    "The declared measurement error changes the target, toward the confounded contrast. The "
    "draws describe how far the estimate moves under that error, and the report states no "
    "verdict."
)


def _read_configuration(result: Any, estimand: str) -> FitConfiguration | None:
    """Read the scope axes of ``result``, or ``None`` when a stand-in result lacks them.

    A fit that cannot be read matches only a registry row of unrestricted scope, which
    exists for exact references and nothing else.
    """
    try:
        return _interpretation.fit_configuration(result, estimand)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None


#: The printed name of each band reading. The bias reading is the bias over the standard
#: deviation of the draws, so its label says so.
_READING_LABELS = {"bias": "standardized bias", "se_ratio": "se_ratio", "coverage": "coverage"}


def _band_sentence(sentence: str, inside: bool) -> str:
    """The registry sentence of a band reading, stated for the side of the band it fell on.

    Every ``generated_outcome`` and ``plasmode`` row words its sentence for a reading that
    lies inside its band. A reading outside the band reads ``lies outside`` instead.
    """
    if inside:
        return sentence
    outside = sentence.replace("lies inside", "lies outside")
    if outside == sentence:  # pragma: no cover - the registry test pins the wording
        raise ValueError(f"a band sentence must say 'lies inside': {sentence!r}")
    return outside


def _resolve(
    instrument: str,
    rule: str,
    fit: FitConfiguration | None,
    replicates: int | None = None,
    placebo_draw: str | None = None,
) -> Resolution:
    """Ask the registry, reading :data:`~cleverly.validation._interpretation.VERDICT_EVIDENCE`
    at call time so a test may patch it. A row covers a report only at its replicate count,
    and a placebo row only on the draw it calibrated."""
    if fit is not None:
        fit = replace(fit, replicates=replicates, placebo_draw=placebo_draw)
    return _interpretation.resolve(instrument, rule, fit)


def _generated_outcome_test(
    result: Any,
    *,
    name: str,
    estimand: str,
    original: float,
    process: Any,
    bands: EquivalenceBands,
    n_replicates: int,
    root_seed: int,
    fit: FitConfiguration | None,
    target: PointTarget | _LongitudinalGenerated | _SurvivalGenerated | _LongitudinalPlasmode,
) -> RefutationTest:
    data = result.data
    seeds = _generated_child_seeds(root_seed, name, n_replicates)
    plasmode = name == "plasmode"
    ratio = isinstance(target, PointTarget) and target.ratio
    law: _PlasmodeLaw | None = None
    if not isinstance(target, PointTarget):
        held = target

        def replacement(_replicate: int, seed: int) -> Any:
            return held.draw(data, seed=seed, name=name)

        declared_effect = float(held.truth)
        covariate_part = bool(held.covariate_part)
    elif plasmode:
        law = _PlasmodeLaw.fit(result)
        frozen = law
        means, weights, mixture = frozen.means(process, data, target)

        def replacement(_replicate: int, seed: int) -> Any:
            return frozen.draw(data, process, seed=seed, target=target)

        declared_effect = target.truth(means, weights, mixture)
        # The plasmode resamples W, so its SE ratio and coverage have the unconditional
        # reference for every target (plan 5.6) and need no covariate-part check.
        covariate_part = True
    else:
        covariates = np.asarray(data.covariates, dtype=float)
        terms = arm_term(
            process, data.treatment, arm_codes=data.arm_codes, levels=data.treatment_levels
        )
        if data.has_intermediate:
            # The held design keeps Z, and the process adds gamma * Z with no arm interaction.
            terms = terms + float(process.intermediate_effect) * np.asarray(
                data.intermediate, dtype=float
            )
        observed = process.mean(covariates, terms, bounds=target.outcome_bounds)
        unrecorded = ~np.asarray(data.treatment_recorded, dtype=bool)
        first_recorded = int(np.flatnonzero(~unrecorded)[0])

        def replacement(_replicate: int, seed: int) -> Any:
            drawn = process.sample(
                np.random.default_rng(seed), observed, bounds=target.outcome_bounds
            )
            if np.any(unrecorded):
                # An unrecorded row's outcome is never read, but a Gaussian fit scales the
                # outcome by its range over every row. Copying a recorded row's value keeps
                # that range a function of the recorded rows (part E plan review, R2).
                drawn = np.where(unrecorded, drawn[first_recorded], drawn)
            return data.with_outcome(drawn, family=process.family)

        means = arm_means(
            process,
            covariates,
            target.codes,
            data,
            bounds=target.outcome_bounds,
            intermediate=target.intermediate,
            doses=target.doses,
        )
        weights = held_weights(target, data)
        mixture = held_mixture(target, data)
        # The truth is the sample-conditional functional of the held rows. The held design
        # reads the SE ratio and coverage only where the covariate part of the curve is
        # identically 0 on this law, which the law itself decides (plan 5.5.1).
        declared_effect = target.truth(means, weights, mixture)
        covariate_part = target.covariate_part_vanishes(means, weights, mixture)

    records, failures = _run_empirical_refits(
        result,
        estimand=estimand,
        draws=tuple(enumerate(seeds)),
        replacement=replacement,
        expected_family=process.family,
    )
    if ratio:
        # A ratio reads on the log scale, where its standard error is reported. A refit with
        # a nonpositive ratio has no log and joins the retained failures.
        kept = [item for item in records if item.estimate > 0.0]
        failures = failures + tuple(
            ReplicationFailure(
                replicate=item.replicate,
                seed=item.seed,
                error_type="ValueError",
                message="refitted ratio is not positive, so it has no log-scale reading",
            )
            for item in records
            if not item.estimate > 0.0
        )
        records = tuple(kept)
    values = tuple(item.estimate for item in records)
    errors = tuple(item.std_error for item in records)
    scaled = tuple(float(np.log(value)) for value in values) if ratio else values
    reference_value = float(np.log(declared_effect)) if ratio else declared_effect
    families = {item.family for item in records}
    family = next(iter(families)) if len(families) == 1 else None

    readings: tuple[BandReading, ...] = ()
    passed: bool | None = None
    rung: Rung = "L0"
    sentences: list[str] = []
    # The sentences about what licenses a reading, which the plain ``evidence`` restates;
    # every other sentence is a condition the printed summary keeps.
    resolved: set[str] = set()
    evidence = _interpretation.PLAIN_NONE
    withheld = ""
    licensed_plain = ""
    not_read = "No readings were computed, so there is nothing to compare."
    if len(families) > 1:
        detail = f"successful refits reported inconsistent outcome families {sorted(families)}"
        sentences.append("The refits reported different outcome families, so no reading runs.")
        evidence = not_read
    elif failures:
        detail = (
            f"{len(failures)} refit(s) failed and failure_policy='fail' keeps every failure; "
            "the equivalence readings need every draw"
        )
        sentences.append(f"{len(failures):,} refits failed, and the readings need every draw.")
        evidence = not_read
    elif len(scaled) < _MINIMUM_BAND_DRAWS or float(np.std(scaled, ddof=1)) <= 0.0:
        detail = (
            f"only {len(values)} usable draw(s), or draws that do not vary; the equivalence "
            "intervals are undefined"
        )
        sentences.append(
            f"{len(values):,} usable draws, or draws that do not vary, leave the readings "
            "undefined."
        )
        evidence = not_read
    else:
        # A Monte Carlo truth widens the bias and coverage intervals in quadrature.
        truth_variance = (
            target.monte_carlo_se**2 if isinstance(target, _LongitudinalPlasmode) else 0.0
        )
        shown = band_readings(scaled, errors, reference_value, bands, truth_variance=truth_variance)
        # A fit whose status withholds an interval has no standard error or coverage to
        # read; its plug-in spread is a diagnostic (X82 plan, matrix notes 8 and 15).
        status = result[estimand].inference
        interval = supplies_inference(status)
        by_rule = {"bias": "bias_band", "se_ratio": "se_ratio_band", "coverage": "coverage_band"}
        readings = tuple(
            reading for reading in shown if reading.name == "bias" or (covariate_part and interval)
        )
        calibrated = len(values) >= bands.minimum_draws
        licensed_inside: list[bool] = []
        for reading in readings:
            resolution = _resolve(
                "plasmode" if plasmode else "generated_outcome",
                by_rule[reading.name],
                fit,
                len(values),
            )
            evidence = resolution.plain or evidence
            if resolution.licensed:
                licensed_plain = licensed_plain or resolution.plain
            if resolution.licensed and calibrated:
                licensed_inside.append(reading.inside)
                sentences.append(_band_sentence(resolution.sentence, reading.inside))
                resolved.add(sentences[-1])
        if licensed_inside:
            passed = all(licensed_inside)
            rung = "L2"
            # The caveat holds for every band reading, so it is stated once.
            sentences.append(
                "The plasmode resembles your data and does not estimate its truth."
                if plasmode
                else "Your own outcome law is not this one."
            )
        elif not calibrated:
            rung = "L1"
            sentences.append(
                f"{len(values)} draws are below the calibrated minimum of "
                f"{bands.minimum_draws}, so the readings state no verdict."
            )
            resolved.add(sentences[-1])
            if licensed_plain:
                evidence = (
                    f"Its {len(values)} repeats are fewer than the {bands.minimum_draws} its "
                    "simulation study needs, so it gives no pass or fail."
                )
            else:
                # No study row covers this fit, so more repeats would not give a verdict either.
                evidence = (
                    f"Its {len(values)} repeats are fewer than the {bands.minimum_draws} a "
                    "verdict needs, and no simulation study backs a verdict for this kind of "
                    "fit at any count, so it gives no pass or fail."
                )
            if licensed_plain:
                withheld = (
                    f"its {len(values):,} repeats are fewer than the "
                    f"{bands.minimum_draws:,} its study used"
                )
        else:
            sentences.append(_interpretation.DESCRIPTIVE_SENTENCE)
            resolved.add(sentences[-1])
        detail = "; ".join(
            f"{_READING_LABELS[reading.name]} {reading.estimate:.4g} "
            f"[{reading.lower:.4g}, {reading.upper:.4g}] "
            f"{'inside' if reading.inside else 'outside'} "
            f"[{reading.band[0]:g}, {reading.band[1]:g}]"
            for reading in readings
        )
        if ratio:
            detail += "; on the log scale"
        if not interval:
            detail += (
                "; standard-error ratio and coverage withheld (the fit reports no interval: "
                f"{status})"
            )
        elif not covariate_part:
            detail += "; standard-error ratio and coverage withheld (covariate part of the curve)"
        design = "the held design"
        if isinstance(target, _LongitudinalPlasmode):
            design = (
                "resampled baselines and a history drawn from frozen node models, with the "
                f"treatment clipped to [{target.law.lower:g}, {1.0 - target.law.lower:g}]"
            )
            if target.draws is None:
                sentences.append(
                    "The truth enumerates every branch of every history, so it is exact."
                )
            else:
                sentences.append(
                    f"The truth averages {target.draws} simulated histories per support point; "
                    f"its Monte Carlo standard error is {target.monte_carlo_se:.3g}, which the "
                    "bias and coverage readings add in quadrature."
                )
        elif not isinstance(target, PointTarget):
            design = "the held history"
            if target.shares:
                detail += "; uncensored share by node " + ", ".join(
                    f"{share:.3f}" for share in target.shares
                )
            sentences.append(
                _RECODED_EXIT_SENTENCE
                if isinstance(target, _SurvivalGenerated)
                else _HELD_HISTORY_SENTENCE
            )
        if law is not None:
            clip = (
                "a known mechanism"
                if law.bounds is None
                else f"a frozen treatment model clipped to [{law.bounds[0]:g}, {law.bounds[1]:g}]"
            )
            if len(law.arm_codes) > 2:
                clip += f", smallest arm probability {law.floor:.3g}"
            for model in law.observation_models:
                clip += f", a frozen {model} model"
            units = "covariates"
            if law.clusters is not None:
                units = "whole clusters"
                cluster_sd = float(getattr(process, "cluster_sd", 0.0))
                if cluster_sd > 0.0:
                    units += f" with a cluster intercept of SD {cluster_sd:g}"
            design = f"resampled {units} and {clip}"
        precision = getattr(process, "precision", None)
        law_name = process.name if precision is None else f"{process.name} (phi {precision:g})"
        law_sentence = (
            f"Generated law: {law_name}, {len(values)} draws on {design}, "
            f"truth {declared_effect:.5g}."
        )
        sentences.append(law_sentence)
    return RefutationTest(
        name=name,
        estimand=estimand,
        original=original,
        values=values,
        expectation=f"readings inside their bands around {declared_effect:.5g}",
        passed=passed,
        detail=detail,
        standard_errors=errors,
        declaration=process,
        rule=None,
        records=records,
        failures=failures,
        family=family,
        empirical_pvalue=None,
        declared_effect=declared_effect,
        requested_draws=n_replicates,
        rung=rung,
        scope_sentence=" ".join(dict.fromkeys(sentences)),
        readings=readings,
        # The generated law prints in the full summary's repeats, in plain words.
        conditions=tuple(
            item
            for item in dict.fromkeys(sentences)
            if item not in resolved and not item.startswith("Generated law: ")
        ),
        evidence=licensed_plain if passed is not None else evidence,
        withheld=withheld,
    )


_PLACEBO_MODES = ("permute", "draw", "markov")

#: The node a survival recode drops (X82 plan 5.5.2, z1).
_RECODED_EXIT_SENTENCE = (
    "The draw keeps the covariates, treatments and censoring of each unit and replaces the "
    "events. An original event at node t censors the unit at t, so the generated outcome is "
    "not observed at that node. The hazard reads W and the treatments only, so this censoring "
    "stays at random. A risk under a hazard that reads W has a covariate part in its "
    "influence curve, so it reads bias only. Coverage also needs a correct censoring model, "
    "so the report prints the uncensored share at each node."
)

#: Why a held-history level reads bias only (X82 plan 5.5.1).
_HELD_HISTORY_SENTENCE = (
    "The draw keeps the covariates, treatments and censoring of each unit and replaces the "
    "outcome. With the covariates held, the truth of a level is the sample mean of f(W), while "
    "the standard error estimates the marginal variance, so a level reads bias only. A "
    "difference contrast cancels f(W), so its standard-error ratio and coverage have the "
    "held-design reference. Coverage also needs a correct censoring model, so the report "
    "prints the uncensored share at each node."
)


def _reported_spread(status: InferenceStatus) -> str:
    """The short name of the fit's standard error in a placebo label, at its status."""
    return "reported SE" if supplies_inference(status) else "plug-in SE"


def _plugin_reference_clause(status: InferenceStatus) -> str:
    """The clause that marks a sampling reference built from a diagnostic spread.

    A fit whose status withholds an interval reports ``plugin_std_error`` only. A reference
    built from it is a description, and the registry covers no withheld status.
    """
    if supplies_inference(status):
        return ""
    return (
        "; the fit reports no interval, so the reference uses the plug-in spread of its "
        "curve, a diagnostic"
    )


def _placebo_conditions(clauses: str) -> tuple[str, ...]:
    """The conditions a placebo reading carries: the shared sentence, then the clauses its
    detail appends for this fit's design, as one sentence-ended string."""
    extra = clauses.lstrip(". ").strip()
    if not extra:
        return (_PLACEBO_SENTENCE,)
    return (_PLACEBO_SENTENCE, extra if extra.endswith(".") else extra + ".")


_PLACEBO_SENTENCE = (
    "This reads whether the estimator returns the null when treatment is independent of "
    "covariates and outcome. It does not test the effect, and it says nothing about "
    "confounding in your data."
)
_SUBSET_SENTENCE = (
    "This compares the scatter of subsample refits with the sampling reference for your "
    "standard error. It cannot indicate bias."
)
_NOISE_SENTENCE = (
    "The target does not change when an irrelevant column joins the adjustment set. The "
    "paired shift describes how far the learners move at this sample size, and no reference "
    "law gives it a verdict."
)
_NEGATIVE_CONTROL_SENTENCE = (
    "Under negative-control assumptions that this report does not check, a non-null "
    "estimate can reveal residual bias. It can also mean the control is a bad one. A null "
    "estimate does not show that unmeasured confounding is absent."
)
#: The three assumptions of a pre-treatment control (X82 plan 5.7.1, n8).
_PRE_TREATMENT_SENTENCE = (
    "The pre-treatment reading assumes three things: no causal path from treatment to the "
    "control, which time order gives; the confounding structure of the primary outcome; and "
    "no effect of the control on treatment given the adjustment set. A lagged outcome that "
    "drives treatment and sits outside the adjustment set breaks the third, and its estimate "
    "is then nonzero with no unmeasured confounding."
)
#: What the future-treatment reading assumes (X82 plan 5.7.2).
_FUTURE_TREATMENT_SENTENCE = (
    "The reading assumes that the control was measured before the later treatment, which the "
    "code cannot check from an array. It also assumes no feedback: the history the fit "
    "declares for the later treatment "
    "is its full set of parents, so the treatment is independent of the control given that "
    "history. The code cannot check this. A nonzero estimate means an unmeasured common cause "
    "of the later treatment and the control, or feedback that the declaration excludes. The "
    "report states no verdict."
)


def _control_residual_reason(
    values: np.ndarray,
    features: np.ndarray,
    floor: float,
    *,
    noun: str = "the pre-treatment control",
    adjustment: str = "the adjustment set",
) -> str | None:
    """Why a pre-treatment control is nearly a function of the adjustment set, or ``None``.

    The share is the two-fold cross-fitted residual variance of a linear regression of the
    control on the adjustment set, over the control's variance (X82 plan 5.7.1). Below
    ``floor`` the control estimate is 0 because the component vanishes, not because
    confounding is absent. The folds use a fixed seed, so the capability row and the call
    read the same share.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    total = float(np.var(values))
    if total <= 0.0:
        return f"{noun} is constant on the analysis rows, so its estimate is 0 by construction"
    if not features.shape[1] or values.size < 4:
        return None
    residual = np.empty_like(values)
    for train, test in KFold(n_splits=2, shuffle=True, random_state=0).split(features):
        model = LinearRegression().fit(features[train], values[train])
        residual[test] = values[test] - model.predict(features[test])
    share = float(np.mean(residual**2)) / total
    if share >= floor:
        return None
    return (
        f"{noun} is nearly a function of {adjustment}: its "
        f"cross-fitted residual variance is {share:.3g} of its variance, below the floor "
        f"{floor:g}. Its estimate would be 0 because that component vanishes, not because "
        "confounding is absent"
    )


def _control_values(data: Any, longitudinal: bool) -> np.ndarray:
    """The observed values of a control that replaced the outcome of ``data``."""
    if longitudinal:
        values = np.asarray(
            data.repeated if data.repeated is not None else data.outcome, dtype=float
        )
        return values[np.isfinite(values)]
    return np.asarray(data.outcome, dtype=float)[np.asarray(data.observed, dtype=bool)]


def _control_estimator(
    estimator: Any, family: str, values: np.ndarray
) -> tuple[Any, tuple[float, float] | None]:
    """A copy of ``estimator`` whose outcome bounds fit a control, and those bounds.

    A control is an auxiliary fit that the package builds, so the package sets its scale. A
    binary control takes no bounds. A continuous control on a cross-fitted fit, or on a fit
    that declares bounds for its own outcome, takes the control's observed range, which is
    the usual TMLE choice (Gruber and van der Laan 2010). An in-sample fit with no bounds
    keeps none. The report prints the bounds it set. A continuous control on a fit whose
    outcome learner is a classifier regresses with the fit's pseudo-outcome learner. An
    :class:`~cleverly.estimators.AdaptiveIPTW` weights the observed control and scales no
    outcome, so it refits unchanged and takes no bounds.
    """
    from ..estimators.iptw import AdaptiveIPTW

    if isinstance(estimator, AdaptiveIPTW):
        return estimator, None
    declared = getattr(estimator, "q_bounds", None)
    cross_fitted = bool(getattr(estimator, "cross_fit", True)) and (
        int(getattr(estimator, "n_folds", 1) or 1) > 1
    )
    bounds: tuple[float, float] | None = None
    if family != "binomial" and values.size and (cross_fitted or declared is not None):
        bounds = (float(np.min(values)), float(np.max(values)))
    learner = getattr(estimator, "outcome_learner", None)
    swap = family != "binomial" and _classification_only_learner(learner) is not None
    if bounds == declared and not swap:
        return estimator, bounds
    copied = copy.copy(estimator)
    copied.q_bounds = bounds
    if swap:
        # A classifier cannot regress a continuous control. A longitudinal fit's
        # pseudo-outcome learner is a regression learner; a point fit has none and keeps
        # its learner, whose refit then reports the failure.
        copied.outcome_learner = getattr(estimator, "pseudo_learner", learner)
    return copied, bounds


def _bounds_clause(bounds: tuple[float, float] | None) -> str:
    """The report clause that names the outcome bounds a control fit used."""
    if bounds is None:
        return ""
    return (
        "; the control fit bounds its outcome to its observed range "
        f"[{bounds[0]:.5g}, {bounds[1]:.5g}]"
    )


def _future_treatment_test(
    result: Any, *, estimand: str, original: float, control: Any, node: int, seed: int
) -> RefutationTest:
    """Regress the control on the treatment at ``node``, adjusted for its declared history.

    The fit is a point-treatment TMLE of the control on the binary treatment at ``node``,
    adjusted for the history the longitudinal fit declares for that treatment, on the units
    at risk at ``node``. It reuses the fit's outcome and treatment learners and its fold
    count. The reference is 0 (X82 plan 5.7.2), and the reading is descriptive.
    """
    from ..data import CausalData
    from ..estimators.tmle import TMLE

    data = result.data
    rows = data.node_present(node)
    history, names = _future_history(data, node)
    values = np.asarray(control, dtype=float).reshape(-1)[rows]
    codes = np.asarray(data.treatment[rows, node - 1], dtype=float)
    weights = np.asarray(data.weights, dtype=float)[rows]
    cluster = None if data.cluster is None else np.asarray(data.cluster)[rows]
    point = CausalData.from_arrays(
        values,
        codes,
        history[rows],
        covariate_names=[f"h{index}" for index in range(len(names))],
        weights=None if np.allclose(weights, 1.0) else weights,
        id=cluster,
    )
    template = result.estimator
    folds = max(1, int(getattr(template, "n_folds", 1) or 1))
    binary = bool(np.isin(values, (0.0, 1.0)).all())
    cross_fit = folds > 1
    # A continuous control on a cross-fitted fit takes its observed range as its bounds.
    bounds = None if binary or not cross_fit else (float(np.min(values)), float(np.max(values)))
    outcome_learner = getattr(template, "outcome_learner", None)
    if not binary and _classification_only_learner(outcome_learner) is not None:
        # A classifier cannot regress a continuous control. The fit's regression learner
        # for its pseudo-outcomes can, and the default library stands in without one.
        outcome_learner = getattr(template, "pseudo_learner", None)
    fitted = TMLE(
        outcome_learner=outcome_learner,
        treatment_learner=getattr(template, "treatment_learner", None),
        cross_fit=cross_fit,
        n_folds=folds,
        q_bounds=bounds,
        random_state=seed,
    ).fit(point)
    estimate = fitted.single()["ate"]
    value = float(estimate.psi)
    std_error = float(estimate.plugin_std_error)
    z = value / std_error if std_error > 0 else float("nan")
    pvalue = float(two_sided_pvalue(z, 1.0)) if np.isfinite(z) else float("nan")
    return RefutationTest(
        name="future_treatment",
        estimand=estimand,
        original=original,
        values=(value,),
        expectation="~ 0",
        passed=None,
        detail=(
            f"the control moves by {value:+.5g} with the treatment at node {node} (z = "
            f"{z:+.3g} standard errors from 0), adjusted for the {len(names)} column(s) of "
            f"that treatment's declared history on {int(rows.sum())} units at risk, "
            f"{'cross-fitted in ' + str(folds) + ' folds' if cross_fit else 'fitted in sample'}"
            + _bounds_clause(bounds)
        ),
        standard_errors=(std_error,),
        rung="L0",
        scope_sentence=_FUTURE_TREATMENT_SENTENCE,
        conditions=(_FUTURE_TREATMENT_SENTENCE,),
        evidence=_interpretation.PLAIN_NONE,
        statistic=z,
        statistic_name="z",
        statistic_pvalue=pvalue,
    )


def _drawn_treatment(data: Any, rng: np.random.Generator) -> np.ndarray:
    """The independent-draw placebo: i.i.d. draws from the empirical marginal of treatment.

    Every arm of a categorical treatment stays present, because the refit must estimate the
    parameter of the same declaration. A vector treatment draws whole component rows, so the
    joint law of the components is the empirical one. A continuous dose draws from observed
    values and so keeps the observed support.
    """
    if data.treatment_kind == "vector":
        rows = rng.integers(0, data.n, size=data.n)
        return np.asarray(data.treatment_components, dtype=float)[rows]
    recorded = (
        data.treatment_recorded if data.has_missing_treatment else np.ones(data.n, dtype=bool)
    )
    source = np.asarray(data.treatment, dtype=float)[recorded]
    arms = None if data.is_continuous_treatment else np.unique(source)
    for _ in range(1000):
        drawn = rng.choice(source, size=source.size, replace=True)
        if arms is None or np.array_equal(np.unique(drawn), arms):
            break
    else:  # pragma: no cover - a 1000-fold failure needs an arm share near zero
        raise RuntimeError("the placebo draw dropped an arm of the treatment 1000 times")
    placebo = np.asarray(data.treatment, dtype=float).copy()
    placebo[recorded] = drawn
    return placebo


def _declares_known_treatment(data: Any) -> bool:
    """Whether ``data`` declares a known treatment mechanism, on either family."""
    known = getattr(data, "known_mechanisms", None)
    if known is not None:
        return bool(known.declares_treatment)
    return getattr(data, "known_treatment", None) is not None


def _placebo_data(
    data: Any, rng: np.random.Generator, mode: str | None, kernel: MarkovKernel | None = None
) -> Any:
    """Return ``data`` with the placebo treatment, and its exact mechanism where declared.

    A fit that declared a known treatment mechanism keeps one on the placebo law. That law
    draws treatment independently of the covariates with the empirical arm shares, so the
    exact mechanism is those shares, constant in ``W``. A refit that dropped the declaration
    and re-estimated the mechanism would be another configuration of the estimator
    (a known-mechanism fit would become an estimated-mechanism fit).

    A longitudinal panel draws its trajectories from ``kernel``, and each node it declares
    known takes the kernel's law given the drawn previous treatment (X82 plan 5.1.2).
    """
    if kernel is not None:
        placebo_matrix = kernel.placebo(data, rng)
        return data.with_treatment(
            placebo_matrix,
            known_mechanisms=placebo_known_mechanisms(data, kernel, placebo_matrix),
        )
    placebo = _drawn_treatment(data, rng) if mode == "draw" else _permuted_treatment(data, rng)
    replaced = data.with_treatment(placebo)
    if data.known_treatment is None:
        return replaced
    from ..data.known_mechanism import KnownMechanism

    # The exact mechanism of the placebo law is the empirical arm shares of the original
    # treatment column, for the permutation and the independent draw alike. The realized
    # shares of one drawn sample are not the law's.
    treatment = np.asarray(data.treatment, dtype=float)
    recorded = data.treatment_recorded if data.has_missing_treatment else slice(None)
    shares = np.asarray([float(np.mean(treatment[recorded] == code)) for code in data.arm_codes])
    values = np.tile(shares, (replaced.n, 1))
    return replace(replaced, known_treatment=KnownMechanism(values, "array"))


def _on_draw(error: CleverlyError, label: str) -> CleverlyError:
    """``error`` with its condition prefixed by the placebo draw that raised it."""
    refusal = replace(error.refusal, condition=f"{label}: {error.refusal.condition}")
    return type(error)(refusal=refusal)


def _band_placebo_sets(
    result: Any,
    bands: PointBands | PanelBands,
    rng: np.random.Generator,
    mode: str | None,
    count: int,
) -> list[Any]:
    """Draw every band-rank placebo data set, each through the refit's preflight.

    The refit of each data set would run the estimator's declaration checks before any
    learner: the declared support (an assertion here, since every placebo dose lies in its
    band), the undeclared and declared atoms, and each policy's partition, monotonicity,
    collapse, tie, atom-image and support checks. A placebo dose takes new values, so a check
    the original fit passed can fail on one draw. Every draw is checked here first, so such a
    draw refuses the call before any refit, with its own reason code. The pooled extreme
    positions are checked as well, which makes the support check independent of the seed for
    a monotone policy (``X82-J-plan.md`` section 3.3).
    """
    data = result.data
    sets: list[Any] = []
    for draw in range(1, count + 1):
        label = f"on placebo draw {draw} of the band-rank placebo"
        try:
            if isinstance(bands, PanelBands):
                doses, drawn = bands.placebo(data, rng)
                replacement = data.with_treatment(
                    doses, known_mechanisms=placebo_known_mechanisms(data, bands.kernel, drawn)
                )
            else:
                replacement = data.with_treatment(bands.draw(data, rng, mode))
            _placebo_preflight(result, replacement)
        except CleverlyError as error:
            raise _on_draw(error, label) from error
        sets.append(replacement)
    try:
        _band_extremes_check(result, bands)
    except CleverlyError as error:
        raise _on_draw(
            error, "at a pooled extreme position of the band-rank placebo, on every row"
        ) from error
    return sets


def _placebo_preflight(result: Any, replacement: Any) -> None:
    """Run the refit's own declaration checks on ``replacement``, fitting nothing.

    Every longitudinal estimator defines ``_preflight_policies``, so a panel fit never skips the
    check. A point estimator without ``_preflight_fit_configuration`` declares no policy.
    """
    estimator = result.estimator
    if _is_panel(result):
        estimator._preflight_policies(replacement)
        return
    check = getattr(estimator, "_preflight_fit_configuration", None)
    if check is not None:
        check(replacement)


def _band_extremes_check(result: Any, bands: PointBands | PanelBands) -> None:
    """Refuse a policy that leaves the band at a pooled extreme position (R-a1).

    The check is deterministic. On a point fit it maps the smallest and the largest pooled
    position on every row. On a panel it maps each corner of :meth:`PanelBands.extremes`, so a
    band that reads an earlier dose is checked with that dose low and high. A tuned point policy
    is checked at its in-sample cap. A tuned regimen is not checked here: its cap is fitted on the
    refit's split, so it relies on the per-draw preflight of :func:`_band_placebo_sets`.
    """
    from ..interventions.policy import (
        ModifiedPolicy,
        Piecewise,
        Scale,
        Shift,
        _check_declared_support,
        lazy_frame,
    )

    data = result.data
    estimator = result.estimator
    moving = (Shift, Scale, Piecewise, ModifiedPolicy)
    if isinstance(bands, PointBands):
        extremes = bands.extremes()
        policies = tuple(getattr(estimator, "policies", None) or ())
        if extremes is None or not policies:
            return
        from ..estimators._tuned_policies import declares_tuned, in_sample_policies

        if declares_tuned(policies):
            policies = in_sample_policies(policies, data)
        frame = lazy_frame(
            lambda: data.frame_like(
                {name: data.covariates[:, j] for j, name in enumerate(data.covariate_names)}
            )
        )
        rows = np.zeros(data.n, dtype=bool)
        rows[bands.pool] = True
        for doses in extremes:
            for policy in policies:
                if isinstance(policy, moving):
                    _check_declared_support(
                        policy,
                        np.nan_to_num(doses, nan=0.0),
                        frame,
                        rows,
                        data.treatment_support,
                        regimen=getattr(policy, "name", repr(policy)),
                        node=str(data.treatment_name),
                    )
        return
    from functools import partial

    from ..longitudinal.regimen import DynamicRegimen, resolve_regimens
    from ..longitudinal.regimen import declares_tuned as declares_tuned_regimens

    declared = getattr(estimator, "regimens", None)
    if declared is None:
        return
    regimens = resolve_regimens(
        declared, data.n_decisions, held_nodes=data.n_times if data.treatment_held else None
    )
    if declares_tuned_regimens(regimens):
        return
    for matrix in bands.extremes(data):
        extreme = replace(data, treatment=matrix)
        for time, form in enumerate(bands.forms, start=1):
            if form is None:
                continue
            column = matrix[:, time - 1]
            rows = ~np.isnan(column)
            frame = lazy_frame(partial(extreme.policy_frame, time))
            for regimen in regimens:
                if not isinstance(regimen, DynamicRegimen) or not regimen.is_mtp(time):
                    continue
                policy = regimen.plan[time - 1]
                if not isinstance(policy, moving):
                    continue
                _check_declared_support(
                    policy,
                    np.nan_to_num(column, nan=0.0),
                    frame,
                    rows,
                    data.node_support(time),
                    regimen=regimen.label,
                    node=str(data.treatment_names[time - 1]),
                )


def _noise_data(data: Any, rng: np.random.Generator, index: int, nodes: bool) -> Any:
    """Return ``data`` with an independent standard-normal column appended.

    A point-treatment fit and a longitudinal fit both take one baseline column. With
    ``nodes``, a longitudinal fit also takes one column per node, appended to the covariates
    measured before that node's treatment (X82 plan 5.2).
    """
    noisy = data.with_extra_covariate(rng.normal(size=data.n), f"_noise_{index}")
    if not nodes:
        return noisy
    return noisy.with_node_covariates(
        rng.normal(size=(data.n, data.n_times)),
        [f"_noise_{index}_node_{time}" for time in range(1, data.n_times + 1)],
    )


def _subset_draw(
    data: Any, rng: np.random.Generator, fraction: float
) -> tuple[np.ndarray, int, int]:
    """Draw the rows of one subset, whole clusters on a clustered fit.

    Returns
    -------
    tuple
        The selected row positions, the number of sampling units, and the number drawn.
    """
    if data.cluster is None:
        units = int(data.n)
        size = min(units, max(20, int(fraction * units)))
        return np.sort(rng.choice(units, size=size, replace=False)), units, size
    labels = np.unique(data.cluster)
    units = int(labels.size)
    size = min(units, max(2, int(fraction * units)))
    chosen = rng.choice(labels, size=size, replace=False)
    return np.flatnonzero(np.isin(data.cluster, chosen)), units, size


def _placebo_positivity_gap(result: Any, kernel: MarkovKernel) -> str | None:
    """Name a static regimen that the placebo kernel never draws, or return ``None``.

    The exact null needs ``K(a) > 0`` for each regimen the fit evaluates (X82 plan 5.1.2).
    The check reads static regimens on an all-categorical panel. On a held design the
    trajectory is the regimen's node-1 arm at every node. A dynamic regimen or a dose
    node has no single trajectory to read.
    """
    from ..longitudinal.regimen import Regimen

    if not all(kernel.categorical):
        return None
    data = result.data
    for regimen in getattr(result.config, "regimens", ()):
        if type(regimen) is not Regimen:
            continue
        encoded = data.encode_assignment(np.asarray(regimen.values, dtype=object), regimen.label)
        codes = tuple(int(code) for code in encoded[0])
        if kernel.held:
            # A held regimen resolves on node 1 only. Its later columns are identity nodes,
            # which ``encode_assignment`` fills with each unit's own observed arm, so row 0
            # says nothing about the regimen there. The held kernel copies node 1 forward.
            codes = (codes[0],) * kernel.n_times
        if kernel.trajectory_probability(codes) == 0.0:
            return (
                f"the Markov kernel of the observed treatments gives regimen "
                f"{regimen.label!r} probability 0, because no unit follows one of its "
                "transitions. The placebo law then has no support for the regimen, so no "
                "placebo refit runs"
            )
    return None


def _intermediate_option(result: Any) -> dict[str, Any]:
    """The ``intermediate_value`` a point refit holds fixed; a longitudinal fit has none."""
    if not hasattr(result, "intermediate_value"):
        return {}
    return {"intermediate_value": result.intermediate_value}


def _sliced_folds(result: Any, index: np.ndarray) -> tuple[Any, ...] | None:
    """The original fold labels of the drawn rows, one split per repeat.

    A subset refit that redrew its folds would mix a new source of fold noise into the
    scatter it measures. Each drawn unit keeps its original label, so the fold sizes vary
    slightly and no split is redrawn. ``None`` for a stand-in result that holds no repeats.
    """
    from ..learners.crossfit import Folds

    if is_structural(result):
        return structural_frozen_folds(result, index)
    repeats = getattr(result, "repeats", None)
    if not repeats:
        return None
    sliced: list[Folds] = []
    for repeat in repeats:
        folds = repeat.folds
        if folds.is_single:
            sliced.append(Folds.single(int(index.size)))
            continue
        labels = np.asarray(folds.assignment)[index]
        _, compact = np.unique(labels, return_inverse=True)
        sliced.append(Folds(compact.astype(np.int64), int(compact.max()) + 1))
    return tuple(sliced)


def refute(
    result: TMLEResult,
    *,
    estimand: str = "ate",
    tests: Sequence[str] = DEFAULT_TESTS,
    n_replicates: int | None = None,
    subset_fraction: float = 0.7,
    placebo_mode: Literal["permute", "draw", "markov"] | None = None,
    seed_spread: bool = False,
    noise_nodes: bool = False,
    negative_control_outcome: Any = None,
    negative_control_kind: Literal["pre_treatment"] | None = None,
    negative_control_floor: float = 0.01,
    future_treatment_control: Any = None,
    future_treatment_node: int = 1,
    dummy_outcome: (
        GaussianIndependentOutcome | LogisticHazardOutcome | MultinomialHazardOutcome | None
    ) = None,
    simulated_outcome: (
        GaussianAdjustmentOutcome | LogisticHazardOutcome | MultinomialHazardOutcome | None
    ) = None,
    outcome_bands: EquivalenceBands = EquivalenceBands(),
    bootstrap_measurement_error: BootstrapMeasurementError | None = None,
    measurement_error_rule: EmpiricalInclusionRule = EmpiricalInclusionRule(),
    random_state: int | None = None,
) -> RefutationResult:
    """Run refutation tests against a fitted model.

    Each test prints its statistic against a stated reference. It prints a verdict
    (``passed``) only where the registry of evidence covers the fit. Otherwise ``passed`` is
    ``None`` and the report says what the numbers describe. The generated-outcome tests have
    exact references and print a verdict from the equivalence bands.

    Parameters
    ----------
    result : TMLEResult
        A fitted result to refute.
    estimand : str
        Alias the tests are run for.
    tests : sequence of str
        Which refutations to run. Parameters without a fixed reported-scale no-effect
        value refuse ``placebo`` and ``negative_control_outcome`` before any refit.
        Differences use zero; risk and odds ratios use one. A fit
        given ``split_plan=`` refuses ``bootstrap_measurement_error``
        before any refit. ``refute_refusal`` holds these refusals, and every refusal of a
        generated-outcome or measurement-error request. The ``refute`` capability row
        reads it for the same request, and defers on ``tests`` when the default tests meet
        one and a smaller set runs. ``random_common_cause`` adds a column of independent
        noise. A :class:`~cleverly.CTMLE` with an explicit ``ordering=`` places that column
        after the declared ordering, and a :class:`~cleverly.DRTMLE` refits without an
        ``evaluation=`` companion that lacks it.
    n_replicates : int or None
        Replicates per randomized test. ``None`` uses 20 for ``placebo``, ``subset`` and
        ``random_common_cause`` (``t`` with 19 degrees of freedom for the placebo), 100 for
        ``bootstrap_measurement_error``, and 400 for each generated-outcome test. Any check
        that needs more than 20 refits is opt-in. The generated-outcome readings print at
        ``L1`` below 400 draws, and a budget below ten is refused before any refit.
    subset_fraction : float
        Share of rows the subset test refits on.
    placebo_mode : {"permute", "draw", "markov"} or None
        On a point-treatment fit, ``"permute"`` permutes the recorded treatment and
        ``"draw"`` draws i.i.d. from its empirical marginal. Both keep the marginal law, and
        both give the same null. On a longitudinal fit, ``"markov"`` draws each unit's
        treatment trajectory from a first-order kernel fitted to the observed treatments,
        and masks it by the observed at-risk pattern. ``None`` is ``"permute"`` on a
        point-treatment fit and ``"markov"`` on a longitudinal fit. A dose whose declared
        support is a function of the covariates or the history takes the band-rank form of
        the draw: each dose's position within its own band is permuted, drawn or
        kernel-drawn, then mapped into the receiving unit's band, so every placebo dose meets
        the support the refit checks. Each placebo data set meets the refit's declaration
        checks before any refit, and the placebo reading reports the band-rank form.
    seed_spread : bool
        Whether ``random_common_cause`` also refits ``n_replicates`` baselines at fresh seeds
        and prints their spread, the movement that fold noise alone gives. It adds that many
        refits, so it is opt-in.
    noise_nodes : bool
        On a longitudinal fit, whether ``random_common_cause`` also appends one noise column
        to the covariates of every node. ``False`` appends one baseline column. A
        point-treatment fit refuses ``True``.
    negative_control_outcome : array-like or None
        An outcome the treatment cannot affect.  Required to run that test. On a
        longitudinal fit it has the outcome's shape: ``(n,)`` at the end of study and
        ``(n, T)`` for a repeated outcome. A survival or competing-risk fit refuses it.
    negative_control_kind : {"pre_treatment"} or None
        ``"pre_treatment"`` declares a control measured before the first treatment. The test
        then refuses a control that equals an adjustment column, or whose cross-fitted
        residual variance on the adjustment set is below ``negative_control_floor`` of its
        variance. ``None`` declares a control measured after treatment.
    negative_control_floor : float
        The cross-fitted residual-variance share below which a pre-treatment control, or a
        future-treatment control regressed on its history, is refused.
    future_treatment_control : array-like or None
        The control of the ``future_treatment`` test on a longitudinal fit: one value per
        unit, measured before the treatment at node ``future_treatment_node + 1`` and outside
        the history that the fit adjusts that treatment for. The code cannot check the time
        of measurement from an array. Required to run that test.
    future_treatment_node : int
        The node ``t``. The test regresses the control on the treatment at ``t + 1``,
        adjusted for that treatment's history, on the units at risk at ``t + 1``.
    dummy_outcome : object or None
        Independent process declaration: a :class:`GaussianIndependentOutcome`, or a
        Bernoulli or bounded process with ``effect=0`` and ``adjustment_scale=0``. ``None``
        uses the default Gaussian declaration. A survival fit takes a hazard with
        ``effect=0`` and ``adjustment_scale=0``, and a competing-risk fit takes a
        multinomial hazard with every effect 0.
    simulated_outcome : object or None
        Adjustment-dependent process, which ``simulated_outcome`` and ``plasmode`` draw. A
        point fit takes a :class:`GaussianAdjustmentOutcome`, a
        :class:`BernoulliAdjustmentOutcome` or a :class:`BoundedAdjustmentOutcome` of its
        outcome family. It covers arm contrasts, arm levels, ratios, ``ey_obs``, ``par``
        and ``paf``, on fits with strata, missing outcomes, a missing treatment or an
        intermediate. ``None`` uses the default Gaussian declaration. A survival fit takes a
        :class:`LogisticHazardOutcome`, and a competing-risk fit takes a
        :class:`MultinomialHazardOutcome`.
    outcome_bands : EquivalenceBands
        Equivalence bands applied to the generated-outcome readings.
    bootstrap_measurement_error : BootstrapMeasurementError or None
        Measurement-error declaration. Required when the named test is requested.
    measurement_error_rule : EmpiricalInclusionRule
        Recorded rule that compares the original estimate with measurement-error refits.
        It supplies the tail probability the report prints, and no verdict.
    random_state : int or None
        Seed for the randomised tests.  ``None`` uses the seed the fit was run with, so a
        seeded fit gives the same refutation every time.  An unseeded fit draws a seed, and
        the refits run under it too.  Either way the report records it under
        ``random_state``, and passing that value back repeats the report.  On a fit with a
        band-rank placebo the placebo sets are drawn first, so the draws of the other tests
        depend on whether ``tests`` holds ``"placebo"``.

    Returns
    -------
    RefutationResult
        One record per test run, with what it expected, what it saw, the rung of evidence
        behind its verdict rule, and the verdict or ``None``.

    Raises
    ------
    ValueError
        If no test is named, a test name is unknown, ``n_replicates`` is not a positive
        integer, or the negative-control test has no outcome. These checks precede every
        refusal and refit.
    CapabilityError
        If ``refute_refusal`` refuses the request. Each such refusal precedes every refit.
    """
    # The malformed arguments first, and the refusals of a well-formed request after them.
    # Every refusal is decided before any refit, so a mixed call does not pay for placebo
    # fits and only then discover that its outcome process has no effect derivation. The
    # ``refute`` capability row reads the same table, so a row that reads available is a
    # request this line admits.
    request = _RefuteRequest(
        estimand,
        _validated_tests(
            tests,
            n_replicates,
            negative_control_outcome,
            placebo_mode,
            negative_control_kind,
            negative_control_floor,
            future_treatment_control,
            future_treatment_node,
        ),
        n_replicates,
        dummy_outcome,
        simulated_outcome,
        outcome_bands,
        bootstrap_measurement_error,
        measurement_error_rule,
        placebo_mode,
        bool(noise_nodes),
        negative_control_outcome=negative_control_outcome,
        negative_control_kind=negative_control_kind,
        future_treatment_control=future_treatment_control,
        future_treatment_node=int(future_treatment_node),
        negative_control_floor=float(negative_control_floor),
    )
    refusal = _first_refusal(result, request)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    requested = request.tests
    estimator = result.estimator
    processes = {name: request.process(name) for name in _GENERATED_TESTS}
    # The table admitted each generated test, so this reads the contrast direction and
    # refuses nothing.
    contrasts = {
        name: _validate_generated_eligibility(result, estimand, processes[name], name)
        for name in request.generated
    }

    original = result[estimand].psi
    no_effect_null = _no_effect_null(result, estimand)
    placebo_null = _placebo_null(result, estimand)
    msm_reference = _working_model_reference(result, estimand)
    estimate_scale = getattr(result[estimand], "scale", None)
    declaration = _target_declaration(result, estimand)
    is_ratio = (estimate_scale or getattr(declaration, "scale", None)) == "ratio"
    # The standard error on the inference scale (log for a ratio). The plug-in accessor
    # answers on a selector-path collaborative fit too, and the reading then says so.
    std_error = result[estimand].plugin_std_error
    # Every refit reruns this estimator, so each test's standard errors carry the fit's
    # own inference status. ``RefutationTest.to_frame`` names its column from it, and a
    # detail below names the unit it counts in from it.
    status = result[estimand].inference
    unit = spread_name("standard errors", status)
    fit = _read_configuration(result, estimand)
    # The package convention for a stochastic operation on a fitted object: an explicit seed
    # wins, ``None`` inherits the fit's own.  ``is None`` rather than truthiness, because
    # ``random_state=0`` is falsy and still has to win.  Same form as ``ctmle.py:846``.
    #
    # An unseeded fit draws one seed here rather than handing ``None`` to the generator, so
    # the report can name it.  The seed then goes to the refits as well as to the draws
    # below.  A perturbation is only half of a refutation: ``refit`` re-learns the
    # nuisances, and an estimator with no ``random_state`` redraws its folds every time, so
    # seeding the draws alone would leave the report unrepeatable.
    seed = resolve_assessment_seed(result, random_state)
    rng = np.random.default_rng(seed)
    data = result.data
    outcomes: list[RefutationTest] = []
    longitudinal = _is_longitudinal(result)
    structural = is_structural(result)
    panel = longitudinal or structural
    kernel = MarkovKernel.fit(data) if panel and "placebo" in requested else None
    # A dose whose declared support is a function takes the band-rank draw. Its placebo data
    # sets are drawn here, first, and each meets the refit's whole preflight, so a draw the
    # refit would refuse stops the call before any refit (X82 part J, section 3.3).
    bands = (
        (
            request.placebo_bands["bands"]
            if "bands" in request.placebo_bands
            else placebo_bands(result)
        )
        if "placebo" in requested
        else None
    )
    placebo_draw: str = "markov" if panel else (placebo_mode or "permute")
    banded_sets: list[Any] | None = None
    if bands is not None:
        placebo_draw = f"band_{placebo_draw}"
        banded_sets = _band_placebo_sets(
            result,
            bands,
            rng,
            placebo_mode,
            n_replicates if n_replicates is not None else _DEFAULT_PERTURBATION_REPLICATES,
        )

    from ..interventions.learned import frozen_rule_replay

    # Frozen on the first refit that holds the rule: a learned-rule fit of several splits
    # has no single rule per row, and the request rules refuse every test that holds one.
    frozen_rules: list[FrozenLearnedRule | None] = []

    def frozen_rule() -> FrozenLearnedRule | None:
        if not frozen_rules:
            frozen_rules.append(_frozen_rule(result))
        return frozen_rules[0]

    def refit(
        replacement: Any,
        *,
        seed: int = seed,
        frozen_folds: Sequence[Any] | None = None,
        frozen_seeds: Sequence[int] | None = None,
        rule_rows: np.ndarray | None = None,
        hold_rule: bool = False,
    ) -> tuple[float, float]:
        """Refit ``replacement``; ``hold_rule`` holds a learned-rule fit's realized rules.

        ``rule_rows`` gives the original row of each row of ``replacement`` on a subset.
        """
        options: dict[str, Any] = {}
        if frozen_folds is not None:
            options["frozen_folds"] = frozen_folds
        if frozen_seeds is not None:
            options["frozen_seeds"] = frozen_seeds
        held = frozen_rule() if hold_rule else None
        fitter = estimator if held is None else frozen_rule_replay(estimator, held, rule_rows)
        refitted = fitter.refit(
            replacement,
            **_intermediate_option(result),
            random_state=seed,
            **options,
        )
        estimate = refitted[estimand]
        return float(estimate.psi), float(estimate.plugin_std_error)

    def on_scale(value: float) -> float:
        return float(np.log(value)) if is_ratio else float(value)

    for name in requested:
        if n_replicates is not None:
            replicate_count = n_replicates
        elif name in _GENERATED_TESTS:
            replicate_count = DEFAULT_OUTCOME_REPLICATES
        elif name in ("bootstrap_measurement_error", "bootstrap_stability"):
            replicate_count = _DEFAULT_MEASUREMENT_REPLICATES
        elif name == "random_common_cause":
            replicate_count = _DEFAULT_NOISE_REPLICATES
        else:
            replicate_count = _DEFAULT_PERTURBATION_REPLICATES
        mean_kind: Literal["arithmetic", "geometric"] = "geometric" if is_ratio else "arithmetic"
        if name == "placebo":
            assert placebo_null is not None
            # A structural contrast is read from the blips, which are 0 on the placebo law
            # whatever regimen the kernel draws, so only a longitudinal fit checks the gap.
            gap = (
                None
                if not longitudinal or kernel is None
                else (_placebo_positivity_gap(result, kernel))
            )
            if gap is not None:
                # A regimen the kernel never draws has no placebo value to refit toward,
                # so the test reports the reason and runs no refit (X82 plan 5.1.2).
                outcomes.append(
                    RefutationTest(
                        name=name,
                        estimand=estimand,
                        original=original,
                        values=(),
                        expectation="no exact value",
                        passed=None,
                        detail=gap,
                        rung="L0",
                        scope_sentence=_PLACEBO_SENTENCE,
                        conditions=(_PLACEBO_SENTENCE, gap[0].upper() + gap[1:] + "."),
                        evidence="No refit ran, so there is nothing to compare.",
                    )
                )
                continue
            fits = (
                [refit(replacement) for replacement in banded_sets]
                if banded_sets is not None
                else [
                    refit(_placebo_data(data, rng, placebo_mode, kernel))
                    for _ in range(replicate_count)
                ]
            )
            values = tuple(psi for psi, _ in fits)
            learned = _is_learned_rule_fit(result)
            # A learned-rule level and an MSM coefficient read on their reported scale; only
            # a registered contrast can sit on the log scale.
            direct = learned or msm_reference is not None
            computed = msm_reference is not None and not msm_reference.exact
            if replicate_count < 2 or computed:
                # A computed projection is one offset shared by all draws and estimated from
                # the same data, so the t reference does not hold for it (X82 plan 5.1.4).
                reading = None
            else:
                reading = placebo_reading(
                    [
                        psi - placebo_null if direct else _no_effect_difference(psi, placebo_null)
                        for psi in values
                    ],
                    std_error=std_error,
                    plugin_std_errors=[se for _, se in fits],
                    scale=original,
                )
            resolution = _resolve("placebo", "t", fit, replicate_count, placebo_draw)
            passed: bool | None = None
            if reading is not None and resolution.licensed:
                passed = not reading.rejects
            placebo_withheld = ""
            if computed:
                placebo_withheld = (
                    "its expected value is computed from the fitted outcome model rather than "
                    "exact, so the t reference does not hold"
                )
            elif reading is None:
                placebo_withheld = "one shuffled copy has no spread"
            mean = float(np.exp(np.mean(np.log(values)))) if is_ratio else float(np.mean(values))
            if computed and np.isnan(placebo_null):
                detail = (
                    f"{mean_kind} mean placebo estimate {mean:+.5g}. This working-model "
                    "coefficient has no exact placebo value under this design, weight and "
                    "link, and no projection of a longitudinal fit is computed, so the report "
                    "states no verdict"
                )
            elif computed:
                detail = (
                    f"{mean_kind} mean placebo estimate {mean:+.5g} beside the computed "
                    f"projection {placebo_null:.5g} of the fitted outcome regression. This "
                    "coefficient has no exact placebo value under this design and weight, so "
                    "the report states no verdict"
                )
            elif reading is None:
                detail = "one placebo draw carries no spread, so the t statistic is undefined"
            elif reading.kind == "constant_zero":
                detail = (
                    f"{mean_kind} mean placebo estimate {mean:+.5g}: the estimate does not "
                    "move with the treatment"
                )
            else:
                detail = (
                    f"{mean_kind} mean placebo estimate {mean:+.5g} against the exact null "
                    f"{placebo_null:g}; the t statistic of the {replicate_count} draws is "
                    f"{reading.t:+.4g} on {reading.df} degrees of freedom"
                )
            plain_length = len(detail)
            if learned:
                detail += (
                    f". The reference is the outcome mean {placebo_null:.5g}, the value of "
                    "every rule under the placebo law"
                )
            if msm_reference is not None and msm_reference.exact:
                detail += (
                    ". The weight and the design separate the arm from the covariates, so the "
                    "coefficient is exactly 0 under the placebo law"
                )
            if longitudinal:
                detail += (
                    ". Each trajectory is drawn from a first-order Markov kernel of the "
                    "observed treatments and masked by the observed at-risk pattern"
                )
            if structural:
                detail += (
                    ". Each trajectory is drawn from a first-order Markov kernel of the "
                    "observed treatments, so every blip and every regimen contrast is "
                    "exactly 0 on the placebo law"
                )
            if bands is not None:
                detail += f". {BAND_SENTENCE}"
                if panel:
                    detail += (
                        ". At each banded node the kernel draws the position from one of the "
                        f"{NEAREST_DONORS} donors whose previous value is nearest"
                    )
            if _declares_known_treatment(data):
                detail += ". The refit keeps a known mechanism: the placebo law's own " + (
                    "kernel" if longitudinal else "arm shares"
                )
            outcomes.append(
                RefutationTest(
                    name=name,
                    estimand=estimand,
                    original=original,
                    values=values,
                    expectation=(
                        "no exact value" if np.isnan(placebo_null) else f"~ {placebo_null:g}"
                    ),
                    passed=passed,
                    detail=detail,
                    standard_errors=tuple(se for _, se in fits),
                    mean_kind=mean_kind,
                    rung=resolution.rung,
                    scope_sentence=f"{_PLACEBO_SENTENCE} {resolution.sentence}",
                    conditions=_placebo_conditions(detail[plain_length:]),
                    evidence=resolution.plain,
                    withheld=placebo_withheld if resolution.licensed else "",
                    statistic=None if reading is None else reading.t,
                    statistic_name="t",
                    statistic_pvalue=None if reading is None else reading.pvalue,
                    extras=()
                    if reading is None or reading.scale_ratio is None
                    else (
                        (
                            f"mean {_reported_spread(status)} over SD of the draws",
                            reading.scale_ratio,
                        ),
                    ),
                )
            )

        elif name == "random_common_cause":
            repeats = getattr(result, "repeats", None)
            fit_seed = repeats[0].seed if repeats else seed
            repeat_seeds = [item.seed for item in repeats] if repeats else None
            # A structural fit holds one split and one fit seed. Each noise refit reuses both,
            # so the shift carries no fold or learner-seed noise on an unseeded fit.
            noise_folds = structural_frozen_folds(result) if structural else None
            if structural:
                repeat_seeds = structural_fit_seeds(result)  # type: ignore[assignment]
            noise_fits = [
                refit(
                    _noise_data(data, rng, index, noise_nodes),
                    seed=fit_seed,
                    frozen_folds=noise_folds,
                    frozen_seeds=repeat_seeds,
                    hold_rule=True,
                )
                for index in range(replicate_count)
            ]
            values = tuple(psi for psi, _ in noise_fits)
            baseline = [on_scale(original)] * replicate_count
            shift, shift_sd, _ = paired_shift(
                [on_scale(psi) for psi in values], baseline, std_error=std_error
            )
            extras = [("spread of the paired differences (SE units)", shift_sd)]
            if seed_spread:
                child = _generated_child_seeds(seed, name, replicate_count)
                base_fits = [
                    refit(data, seed=child[index], hold_rule=True)
                    for index in range(replicate_count)
                ]
                _, _, seed_only = paired_shift(
                    [on_scale(psi) for psi, _ in base_fits],
                    [on_scale(psi) for psi, _ in base_fits],
                    std_error=std_error,
                )
                extras.append(("spread across seeds with no noise column (SE units)", seed_only))
            outcomes.append(
                RefutationTest(
                    name=name,
                    estimand=estimand,
                    original=original,
                    values=values,
                    expectation=f"~ {original:.4g}",
                    passed=None,
                    detail=(
                        f"adding an irrelevant covariate moved the estimate by {shift:+.4g} "
                        f"{unit} {'on the log scale ' if is_ratio else ''}against the "
                        "original fit, under its seed and folds"
                    ),
                    standard_errors=tuple(se for _, se in noise_fits),
                    mean_kind=mean_kind,
                    rung="L0",
                    scope_sentence=_NOISE_SENTENCE,
                    conditions=(_NOISE_SENTENCE,),
                    evidence=_interpretation.PLAIN_NONE,
                    statistic=shift,
                    statistic_name=f"paired shift in {unit}",
                    extras=tuple(extras),
                )
            )

        elif name == "subset":
            drawn = [_subset_draw(data, rng, subset_fraction) for _ in range(replicate_count)]
            thin = structural_subset_refusal(result, [d[0] for d in drawn]) if structural else None
            if thin is not None:
                # Every draw is read before the first subset refit, so a draw that would stop
                # at the Jacobian floor part way through the test refuses it first.
                raise CapabilityError.from_refusal(thin)
            fits = [
                refit(
                    data.subset(index),
                    frozen_folds=_sliced_folds(result, index),
                    rule_rows=index,
                    hold_rule=True,
                )
                for index, _, _ in drawn
            ]
            values = tuple(psi for psi, _ in fits)
            units, size = drawn[0][1], drawn[0][2]
            reading_s: SubsetReading | None = (
                None
                if replicate_count < 2
                else subset_reading(
                    [on_scale(psi) for psi in values], std_error=std_error, n=units, m=size
                )
            )
            resolution = _resolve("subset", "rho", fit, replicate_count)
            passed = None
            if reading_s is not None and np.isfinite(reading_s.rho) and resolution.licensed:
                passed = bool(reading_s.pvalue >= 0.05)
            spread = float("nan") if reading_s is None else reading_s.spread
            expectation = (
                (
                    f"log-scale scatter ~ {reading_s.reference_sd:.4g}"
                    if is_ratio
                    else f"scatter ~ {reading_s.reference_sd:.4g}"
                )
                if reading_s is not None and np.isfinite(reading_s.reference_sd)
                else "scatter"
            )
            outcomes.append(
                RefutationTest(
                    name=name,
                    estimand=estimand,
                    original=original,
                    values=values,
                    expectation=expectation,
                    passed=passed,
                    detail=(
                        f"subsample estimates scatter by {spread:.5g} "
                        f"{'on the log scale ' if is_ratio else ''}against "
                        f"{'nan' if reading_s is None else format(reading_s.reference_sd, '.5g')} "
                        f"from sampling {size} of {units} units alone"
                        + _plugin_reference_clause(status)
                    ),
                    standard_errors=tuple(se for _, se in fits),
                    spread_kind="log" if is_ratio else "reported",
                    rung=resolution.rung,
                    scope_sentence=f"{_SUBSET_SENTENCE} {resolution.sentence}",
                    conditions=(_SUBSET_SENTENCE,),
                    evidence=resolution.plain,
                    withheld=(
                        ""
                        if passed is not None or not resolution.licensed
                        else "fewer than two subsamples have no spread"
                        if reading_s is None
                        else "the spread ratio is undefined"
                    ),
                    statistic=None if reading_s is None else reading_s.rho,
                    statistic_name="scatter ratio rho",
                    statistic_pvalue=None if reading_s is None else reading_s.pvalue,
                )
            )

        elif name == "negative_control_outcome":
            assert no_effect_null is not None
            replacement = data.with_outcome(
                negative_control_outcome, name="negative_control_outcome"
            )
            pre_treatment = negative_control_kind == "pre_treatment"
            if structural:
                # g-estimation bounds no outcome, and the control refit keeps the original
                # split and fit seed, so the reading compares like with like.
                refitted = estimator.refit(
                    replacement,
                    random_state=seed,
                    frozen_folds=structural_frozen_folds(result),
                    frozen_seeds=structural_fit_seeds(result),
                )
                control_bounds = None
            else:
                control_estimator, control_bounds = _control_estimator(
                    estimator, str(replacement.family), _control_values(replacement, longitudinal)
                )
                refitted = control_estimator.refit(
                    replacement,
                    **_intermediate_option(result),
                    random_state=seed,
                )
            value = refitted[estimand].psi
            control_se = refitted[estimand].plugin_std_error
            deviation = _no_effect_difference(value, no_effect_null)
            z = deviation / control_se if control_se > 0 else float("nan")
            pvalue = float(two_sided_pvalue(z, 1.0)) if np.isfinite(z) else float("nan")
            # The control's own outcome family and event count are the scope axes of a
            # negative-control reading, not the primary outcome's.
            control_fit = (
                None
                if fit is None
                else replace(
                    fit,
                    outcome_family=str(replacement.family),
                    min_event_count=(
                        _interpretation._longitudinal_event_count(replacement)
                        if panel
                        else _interpretation.event_count(replacement)
                    ),
                )
            )
            resolution = _resolve("negative_control_outcome", "z", control_fit, 1)
            passed = bool(pvalue >= 0.05) if resolution.licensed and np.isfinite(z) else None
            outcomes.append(
                RefutationTest(
                    name=name,
                    estimand=estimand,
                    original=original,
                    values=(value,),
                    expectation=f"~ {no_effect_null:g}",
                    passed=passed,
                    detail=(
                        f"the negative-control outcome shows an estimate of {value:+.5g} "
                        f"(z = {z:+.3g} {unit} from the no-effect value {no_effect_null:g})"
                        + _bounds_clause(control_bounds)
                    ),
                    rung=resolution.rung,
                    scope_sentence=(
                        f"{_NEGATIVE_CONTROL_SENTENCE} "
                        + (f"{_PRE_TREATMENT_SENTENCE} " if pre_treatment else "")
                        + resolution.sentence
                    ),
                    conditions=(
                        _NEGATIVE_CONTROL_SENTENCE,
                        *((_PRE_TREATMENT_SENTENCE,) if pre_treatment else ()),
                    ),
                    evidence=resolution.plain,
                    statistic=z,
                    statistic_name="z",
                    statistic_pvalue=pvalue,
                )
            )

        elif name == "future_treatment":
            outcomes.append(
                _future_treatment_test(
                    result,
                    estimand=estimand,
                    original=original,
                    control=future_treatment_control,
                    node=int(future_treatment_node) + 1,
                    seed=seed,
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
                    bands=outcome_bands,
                    n_replicates=replicate_count,
                    root_seed=seed,
                    fit=fit,
                    target=contrasts[name],
                )
            )
        elif name == "bootstrap_stability":
            outcomes.append(
                _bootstrap_stability_test(
                    result,
                    estimand=estimand,
                    original=original,
                    n_replicates=replicate_count,
                    root_seed=seed,
                    is_ratio=is_ratio,
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
        tests=tuple(replace(test, inference=status) for test in outcomes),
        estimand=estimand,
        backend=result.data.backend,
        random_state=seed,
    )
