r"""Simulation studies: does the estimator work, and does its interval cover?

Everything else in this package validates a *fit*.  This validates the *estimator*:
repeat the whole procedure on fresh data from a process whose truth is known, and
measure what comes out.

The three numbers that matter, in order:

**Coverage.** The fraction of replications whose confidence interval contains the
truth.  A nominal 95% interval that covers 80% of the time is not conservative or
approximate -- it is wrong, and every p-value derived from it is wrong too.  This is
the single most informative check available, and it is the one most often skipped.

**Bias, scaled by root-n.** Absolute bias shrinks with :math:`n` for almost any
estimator, so it says little on its own.  What distinguishes an efficient estimator
is that :math:`\sqrt{n}\,\mathrm{bias}` stays bounded as :math:`n` grows, and that is
what :attr:`StudyResult.root_n_bias` reports.  A value that grows with sample size is
the signature of a bias term that does not vanish fast enough -- exactly what the
targeting step is supposed to remove.

**Standard-error accuracy.** The ratio of the mean estimated standard error to the
actual standard deviation of the estimates across replications.  Near 1 means the
reported uncertainty is honest; below 1 means it is optimistic, which is how a
coverage failure usually arises.

This harness is public API, not just test scaffolding: running it on a DGP resembling
your own data is the most direct way to find out whether a particular
estimator configuration is trustworthy for your problem before you rely on it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import FloatArray
from ..estimators.base import TMLEResultSet
from ..estimators.direct_effect import check_level
from ..utils.parallel import map_parallel
from ..utils.text import format_table

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..datasets.synthetic import DGP
    from ..estimators.tmle import TMLE
    from ..longitudinal import LTMLE

__all__ = ["CoverageStudy", "EstimandSummary", "StudyResult"]


@dataclass(frozen=True)
class EstimandSummary:
    """Simulation results for one estimand."""

    estimand: str
    truth: float
    n: int
    n_replicates: int
    estimates: FloatArray
    std_errors: FloatArray
    covered: FloatArray
    rejected: FloatArray
    #: The point estimates on the **inference scale** -- the scale ``std_errors`` is on, which
    #: for a ratio estimand is the log scale and for everything else is ``estimates`` itself.
    #: ``None`` means the two coincide, which is the case for every difference and level.
    #: Only :attr:`se_ratio` reads it; see there for why it has to exist.
    inference_estimates: FloatArray | None = None

    @property
    def mean_estimate(self) -> float:
        return float(np.mean(self.estimates))

    @property
    def inference_scale_estimates(self) -> FloatArray:
        """:attr:`estimates` on the scale the reported standard error is on."""
        if self.inference_estimates is None:
            return self.estimates
        return self.inference_estimates

    @property
    def bias(self) -> float:
        return float(np.mean(self.estimates) - self.truth)

    @property
    def root_n_bias(self) -> float:
        r"""``sqrt(n) * bias``: bounded as ``n`` grows for a root-n consistent estimator."""
        return float(np.sqrt(self.n) * self.bias)

    @property
    def monte_carlo_se(self) -> float:
        """Standard deviation of the estimates across replications: the *actual* variability."""
        return float(np.std(self.estimates, ddof=1))

    @property
    def bias_se(self) -> float:
        """Standard error of the estimated bias, for judging whether it is real."""
        return float(self.monte_carlo_se / np.sqrt(self.n_replicates))

    @property
    def mean_std_error(self) -> float:
        """Mean *reported* standard error."""
        return float(np.mean(self.std_errors))

    @property
    def se_ratio(self) -> float:
        """Reported over actual standard error; 1.0 is honest, below 1 is optimistic.

        Both halves are taken on the **inference scale**, which is the scale
        :attr:`~cleverly.inference.ParameterEstimate.std_error` is defined on -- the log scale
        for a ratio, and the reporting scale for everything else.  For a ratio the two come
        apart badly: ``psi`` is the odds ratio while ``std_error`` is ``SE(log OR)``, and by
        the delta method ``sd(OR) ~ psi * sd(log OR)``, so dividing one by the other returns
        roughly ``1 / psi`` and says nothing whatever about calibration.  A well-behaved
        ``or`` of ``0.42`` came back as ``2.82``, which read as an interval three times too
        wide and was only the reciprocal of the truth.

        :attr:`monte_carlo_se` deliberately stays on the reporting scale, because that is
        where :attr:`bias` and :attr:`rmse` are meaningful; the two differ only for a ratio.
        """
        spread = float(np.std(self.inference_scale_estimates, ddof=1))
        if spread <= 0:
            return float("nan")
        return float(self.mean_std_error / spread)

    @property
    def coverage(self) -> float:
        """Fraction of confidence intervals containing the truth."""
        return float(np.mean(self.covered))

    @property
    def coverage_se(self) -> float:
        """Standard error of the coverage estimate -- needed to judge a near-miss."""
        p = self.coverage
        return float(np.sqrt(p * (1.0 - p) / self.n_replicates))

    @property
    def rejection_rate(self) -> float:
        """Fraction of replications rejecting the null.

        Power when the truth is non-zero; the type I error rate when it is zero.
        """
        return float(np.mean(self.rejected))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean((self.estimates - self.truth) ** 2)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimand": self.estimand,
            "truth": self.truth,
            "n": self.n,
            "n_replicates": self.n_replicates,
            "mean_estimate": self.mean_estimate,
            "bias": self.bias,
            "bias_se": self.bias_se,
            "root_n_bias": self.root_n_bias,
            "rmse": self.rmse,
            "monte_carlo_se": self.monte_carlo_se,
            "mean_std_error": self.mean_std_error,
            "se_ratio": self.se_ratio,
            "coverage": self.coverage,
            "coverage_se": self.coverage_se,
            "rejection_rate": self.rejection_rate,
        }


@dataclass(frozen=True)
class StudyResult:
    """The full output of a :class:`CoverageStudy`."""

    summaries: dict[str, EstimandSummary]
    n: int
    n_replicates: int
    n_failed: int
    alpha: float
    label: str

    def __getitem__(self, estimand: str) -> EstimandSummary:
        return self.summaries[estimand]

    def to_frame(self, backend: str | None = None) -> Any:
        from ..utils.frames import frame_from_dict

        rows = [summary.to_dict() for summary in self.summaries.values()]
        payload = {key: [row[key] for row in rows] for key in rows[0]}
        return frame_from_dict(payload, backend=backend)

    def summary(self) -> str:
        level = 1.0 - self.alpha
        lines = [
            f"Simulation study: {self.label}",
            "-" * (18 + len(self.label)),
            f"n = {self.n}; replications = {self.n_replicates}"
            + (f" ({self.n_failed} failed)" if self.n_failed else ""),
            "",
            format_table(
                [
                    "estimand",
                    "truth",
                    "mean",
                    "bias",
                    "sqrt(n)*bias",
                    "mc se",
                    "mean se",
                    "se ratio",
                    f"{level:.0%} cover",
                    "reject",
                ],
                [
                    [
                        summary.estimand,
                        f"{summary.truth:.4f}",
                        f"{summary.mean_estimate:.4f}",
                        f"{summary.bias:+.4f}",
                        f"{summary.root_n_bias:+.3f}",
                        f"{summary.monte_carlo_se:.4f}",
                        f"{summary.mean_std_error:.4f}",
                        f"{summary.se_ratio:.3f}",
                        f"{summary.coverage:.3f}",
                        f"{summary.rejection_rate:.3f}",
                    ]
                    for summary in self.summaries.values()
                ],
            ),
            "",
            self.verdict(),
        ]
        return "\n".join(lines)

    def verdict(self) -> str:
        """Reading of the study, naming the specific failure mode where there is one."""
        notes: list[str] = []
        target = 1.0 - self.alpha
        for summary in self.summaries.values():
            shortfall = target - summary.coverage
            if shortfall > 3.0 * summary.coverage_se and shortfall > 0.02:
                notes.append(
                    f"{summary.estimand}: coverage {summary.coverage:.3f} is below the nominal "
                    f"{target:.0%} by more than Monte Carlo error"
                    + (
                        f"; the reported standard error is {1 / summary.se_ratio:.2f}x too small"
                        if summary.se_ratio < 0.95
                        else "; the standard error looks right, so this is bias"
                    )
                )
            if abs(summary.bias) > 3.0 * summary.bias_se and abs(summary.bias) > 0.02 * max(
                1.0, abs(summary.truth)
            ):
                notes.append(
                    f"{summary.estimand}: bias {summary.bias:+.4f} is "
                    f"{abs(summary.bias) / summary.bias_se:.1f} Monte Carlo standard errors from "
                    "zero"
                )
        if not notes:
            return "VERDICT: coverage and bias are consistent with a correctly working estimator."
        return "VERDICT:\n" + "\n".join(f"  - {note}" for note in notes)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


class CoverageStudy:
    """Repeatedly fit an estimator on fresh data and measure bias and coverage.

    Parameters
    ----------
    dgp:
        A :class:`~cleverly.datasets.DGP`, or a callable
        ``(n, seed) -> (frame, truth)`` following the convention of the generators in
        :mod:`cleverly.datasets`.
    estimator:
        A zero-argument factory returning a fresh estimator per replication.  A factory
        rather than an instance, so replications cannot share fitted state.
    n, n_replicates:
        Sample size per replication, and how many replications.
    estimands:
        Which estimands to summarise; defaults to whatever the first fit reports.
    fit_kwargs:
        Passed to ``fit`` -- column names, ``delta=``, ``id=`` and so on.
    truth_key:
        ``"population"`` (default) compares against the population estimand, fixed
        across replications; ``"sample"`` compares against each replication's realised
        sample estimand, which removes one source of variability but changes what
        coverage means.
    intermediate_value:
        The level of the intermediate variable to study, for a controlled direct effect.
        Required when ``fit_kwargs`` contains ``intermediate=``, because such a fit
        returns one result per level and each level is a *different parameter*: the
        study picks that level out of the result set and asks the process for the truth
        at the same level, so the two cannot silently disagree.  See
        :mod:`cleverly.estimators.direct_effect`.

    Example
    -------
    >>> from cleverly.estimators import TMLE
    >>> from cleverly.datasets import nonlinear_dgp
    >>> from cleverly.validation import CoverageStudy
    >>> study = CoverageStudy(
    ...     dgp=nonlinear_dgp(),
    ...     estimator=lambda: TMLE(outcome_learner="glm", treatment_learner="glm"),
    ...     n=500,
    ...     n_replicates=50,
    ...     seed=0,
    ... )
    >>> print(study.run().summary())                                     # doctest: +SKIP
    """

    def __init__(
        self,
        dgp: DGP | Callable[..., tuple[Any, dict[str, float]]],
        estimator: Callable[[], TMLE | LTMLE],
        *,
        n: int = 1000,
        n_replicates: int = 100,
        estimands: Sequence[str] | None = None,
        fit_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        n_jobs: int = 1,
        truth_key: str = "population",
        intermediate_value: float | None = None,
        label: str | None = None,
    ) -> None:
        if n_replicates < 2:
            raise ValueError(f"n_replicates must be at least 2; got {n_replicates}")
        if truth_key not in ("population", "sample"):
            raise ValueError(f"truth_key must be 'population' or 'sample'; got {truth_key!r}")
        if intermediate_value is not None:
            intermediate_value = check_level(intermediate_value)
        # Checked here rather than in the replication loop, which swallows every
        # exception to keep one bad draw from killing a study: a mismatch between the
        # fit and the level asked for is a configuration error, and reporting it as
        # "every replication failed" would bury it.
        targets_a_level = (fit_kwargs or {}).get("intermediate") is not None
        if targets_a_level and intermediate_value is None:
            raise ValueError(
                "fit_kwargs names an intermediate variable, so each fit returns one "
                "result per level and each level is a different parameter. Pass "
                "intermediate_value=0.0 or 1.0 to say which one this study measures."
            )
        if intermediate_value is not None and not targets_a_level:
            raise ValueError(
                f"intermediate_value={intermediate_value} was given but fit_kwargs does "
                "not name an intermediate variable, so the fit has no level to select."
            )
        self.dgp = dgp
        self.estimator = estimator
        self.n = n
        self.n_replicates = n_replicates
        self.estimands = estimands
        self.fit_kwargs = fit_kwargs or {"outcome": "Y", "treatment": "A"}
        self.seed = seed
        self.n_jobs = n_jobs
        self.truth_key = truth_key
        self.intermediate_value = intermediate_value
        self.label = str(label or getattr(dgp, "name", getattr(dgp, "__name__", "study")))

    def _draw(self, seed: int) -> tuple[Any, dict[str, float]]:
        from ..datasets.synthetic import DGP as DGPClass

        if isinstance(self.dgp, DGPClass):
            # Passing the level through is what keeps the comparison honest: the truth a
            # process reports for a controlled direct effect is the effect *at a level*,
            # and DGP.sample silently defaults it to 0 when it is not told which one.
            return self.dgp.sample(self.n, seed=seed, intermediate_value=self.intermediate_value)
        if self.intermediate_value is None:
            return self.dgp(self.n, seed=seed)
        return self.dgp(self.n, seed=seed, intermediate_value=self.intermediate_value)

    def _select(self, result: Any) -> Any:
        """Pick the single result a replication is summarising.

        A fit with ``intermediate=`` returns one result per level, and the levels are
        different parameters rather than two views of one, so the study has to be told
        which it is measuring coverage for rather than guessing.  That is the set's own
        key lookup: ``intermediate_value`` is ``None`` for an ordinary fit, which is
        exactly the key such a fit uses, so a mismatch in either direction surfaces as a
        ``KeyError`` naming the levels that are available.

        An estimator that does not return a *set* -- :class:`~cleverly.LTMLE` returns one
        :class:`~cleverly.LongitudinalResult` -- is already the single result, and is
        passed through.  Keying into it would ask for the parameter named ``None`` and
        get a ``KeyError`` that the replication loop swallows, so the whole study would
        fail with "every replication failed; check the estimator configuration" while the
        estimator was configured correctly.
        """
        if not isinstance(result, TMLEResultSet):
            if self.intermediate_value is not None:
                raise TypeError(
                    f"intermediate_value={self.intermediate_value!r} was given, but the "
                    f"estimator returned a {type(result).__name__} rather than a result "
                    "set with one entry per level"
                )
            return result
        return result[self.intermediate_value]

    def run(self) -> StudyResult:
        """Execute the study."""
        import warnings

        seeds = np.random.SeedSequence(self.seed).generate_state(self.n_replicates)

        def replicate(seed: int) -> dict[str, tuple[float, float, float, float, float]] | None:
            try:
                frame, truth = self._draw(int(seed))
                with warnings.catch_warnings():
                    # Individual replications routinely trip positivity warnings; the
                    # aggregate coverage is the diagnostic here, not the per-fit warnings.
                    warnings.simplefilter("ignore")
                    fitted = self.estimator().fit(frame, **self.fit_kwargs)
                result = self._select(fitted)
                names = self.estimands or tuple(result.estimates)
                out: dict[str, tuple[float, float, float, float, float]] = {}
                for name in names:
                    estimate = result[name]
                    prefix = "" if self.truth_key == "population" else "sample_"
                    reference = truth[f"{prefix}{name}"]
                    low, high = estimate.ci
                    # The fifth entry is the estimate on the scale `std_error` is on, which is
                    # the log scale for a ratio -- taken off `log_psi`, the same field `ci`
                    # builds the interval from, rather than re-derived here. `se_ratio` is the
                    # only consumer, and it is the only summary that has to compare the two.
                    out[name] = (
                        estimate.psi,
                        estimate.std_error,
                        float(low <= reference <= high),
                        float(estimate.pvalue < result.config.alpha_sig),
                        float(
                            estimate.log_psi
                            if estimate.scale == "ratio" and estimate.log_psi is not None
                            else estimate.psi
                        ),
                    )
                    out[f"__truth__{name}"] = (reference, 0.0, 0.0, 0.0, 0.0)
                return out
            except Exception:
                return None

        outcomes = map_parallel(replicate, seeds.tolist(), n_jobs=self.n_jobs)
        successes = [row for row in outcomes if row is not None]
        if not successes:
            raise RuntimeError("every replication failed; check the estimator configuration")

        names = [key for key in successes[0] if not key.startswith("__truth__")]
        alpha = 0.05
        summaries: dict[str, EstimandSummary] = {}
        for name in names:
            estimates = np.array([row[name][0] for row in successes])
            std_errors = np.array([row[name][1] for row in successes])
            covered = np.array([row[name][2] for row in successes])
            rejected = np.array([row[name][3] for row in successes])
            truths = np.array([row[f"__truth__{name}"][0] for row in successes])
            inference = np.array([row[name][4] for row in successes])
            summaries[name] = EstimandSummary(
                estimand=name,
                truth=float(np.mean(truths)),
                n=self.n,
                n_replicates=len(successes),
                estimates=estimates,
                std_errors=std_errors,
                covered=covered,
                rejected=rejected,
                # `None` when the two scales coincide, so a difference estimand's `se_ratio`
                # is arithmetically what it always was rather than merely close to it.
                inference_estimates=None if np.array_equal(inference, estimates) else inference,
            )

        return StudyResult(
            summaries=summaries,
            n=self.n,
            n_replicates=len(successes),
            n_failed=len(outcomes) - len(successes),
            alpha=alpha,
            label=self.label,
        )
