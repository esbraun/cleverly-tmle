"""Repeated-sampling property studies: the machinery, not the claims.

A property study asks whether a complete estimator behaves the way its source theory says
it does when applied to samples from a known law.  The cells, laws and learners are the
study's; the sampling loop, the replication accounting, the per-cell verdicts and the rate
estimator are shared, because every method that gets an evidence row needs the same four.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cleverly.validation import CoverageStudy, ReplicationRecord, summarize_replications
from tests.studies.evidence.inference import (
    Interval,
    clopper_pearson,
    percentile_interval,
    standardized_bias_verdict,
)

REPLICATE_COLUMNS = (
    "property",
    "cell",
    "role",
    "replicate",
    "n",
    "requested_replicates",
    "failed_replicates",
    "truth",
    "estimate",
    "std_error",
    "covered",
    "rejected",
)


@dataclass(frozen=True)
class ReplicationSpec:
    """One repeated-sampling configuration before it expands into seeded replications."""

    property: str
    cell: str
    n: int
    replicates: int
    configuration: str


def finite_support_sample(
    probs: np.ndarray,
    support: Sequence[Sequence[float | int]],
    n: int,
    seed: int,
    *,
    columns: Sequence[str],
    kind_axis: int | None = None,
    unobserved: int | None = None,
) -> pd.DataFrame:
    """Draw rows from a finite law, with optional missing-outcome recoding."""
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(support), size=n, p=np.asarray(probs).reshape(-1))
    values = np.asarray(support, dtype=float)[cells]
    if kind_axis is None:
        return pd.DataFrame(values, columns=columns)
    if unobserved is None:
        raise ValueError("unobserved is required when kind_axis is set")
    observed_axes = [axis for axis in range(values.shape[1]) if axis != kind_axis]
    if len(columns) != len(observed_axes):
        raise ValueError("columns must name every support axis except kind_axis")
    frame = pd.DataFrame(values[:, observed_axes], columns=columns)
    kind = values[:, kind_axis]
    frame["Y"] = np.where(kind == unobserved, np.nan, kind)
    frame["Delta"] = np.where(kind == unobserved, 0.0, 1.0)
    return frame


def property_role(
    configuration: str,
    *,
    controls: Collection[str],
    property_name: str,
    n: int,
    rate_sizes: Sequence[int],
) -> str:
    """Classify a property row from its nuisance configuration and rate rung."""
    if property_name == "root_n_and_efficiency" and n == min(rate_sizes):
        return "control"
    return "control" if configuration in controls else "positive"


def replication_payloads(
    record: Any,
    specs: Sequence[ReplicationSpec],
) -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    """Expand property specifications into the tuple shape used by parallel workers."""
    from tests.studies.evidence.seeds import stream_seed

    out: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for spec in specs:
        for replicate in range(spec.replicates):
            seed = stream_seed(
                record,
                "property_sample",
                spec.property,
                spec.cell,
                replicate,
            )
            payload = (
                spec.property,
                spec.cell,
                replicate,
                spec.n,
                spec.replicates,
                seed,
                spec.configuration,
            )
            out.append((payload,))
    return out


@dataclass(frozen=True)
class PropertyCell:
    """One repeated-sampling cell: a law, a nuisance configuration, a size, a seed.

    ``role`` is what stops a control from being read as a claim.  A cell fit with both
    nuisances wrong, or with deliberately in-sample predictions, is *supposed* to fail; its
    verdict records that it failed in the required direction.  Published without the
    distinction, a ``passed`` column says the same word about a valid estimator and about
    one that was broken on purpose, and the rule printed beside it is the positive cell's.
    """

    property: str
    cell: str
    dgp: Any
    outcome_learner: Callable[[], Any]
    treatment_learner: Callable[[], Any]
    n: int
    replicates: int
    seed: int
    #: ``"positive"`` for a cell whose rule asserts the estimator behaved, ``"control"`` for
    #: one whose rule asserts it broke in the direction the property predicts.
    role: str = "positive"
    estimand: str = "ate"
    fit_kwargs: dict[str, Any] = field(default_factory=lambda: {"outcome": "Y", "treatment": "A"})


def run_cells(
    cells: Sequence[PropertyCell],
    estimator: Callable[[PropertyCell], Callable[[], Any]],
    *,
    n_jobs: int,
) -> pd.DataFrame:
    """Run every cell and return the per-replication rows.

    Cells run one at a time with the whole core budget handed to the replication loop inside
    each, rather than several cells at once each with a slice.  Two levels of process pool
    over the same cores oversubscribe every one of them; a single level with a few hundred
    independent fits already keeps every worker fed.

    ``requested_replicates`` and ``failed_replicates`` travel on every row.  A replication
    whose fit raises is dropped by :class:`~cleverly.validation.CoverageStudy` -- correctly,
    so one bad draw cannot kill a study -- and a dropped replication silently widens the
    Monte Carlo error of every cell it touches.  Recording the count is what lets the
    verdicts refuse to be computed on a study that quietly shrank.
    """
    frames: list[pd.DataFrame] = []
    for cell in cells:
        result = CoverageStudy(
            dgp=cell.dgp,
            estimator=estimator(cell),
            n=cell.n,
            n_replicates=cell.replicates,
            estimands=(cell.estimand,),
            fit_kwargs=dict(cell.fit_kwargs),
            seed=cell.seed,
            n_jobs=n_jobs,
        ).run()
        records = tuple(
            record for record in result.replications if record.estimand == cell.estimand
        )
        frames.append(
            pd.DataFrame(
                {
                    "property": cell.property,
                    "cell": cell.cell,
                    "role": cell.role,
                    "replicate": [record.replicate for record in records],
                    "n": cell.n,
                    "requested_replicates": cell.replicates,
                    "failed_replicates": result.n_failed,
                    "truth": [record.truth for record in records],
                    "estimate": [record.estimate for record in records],
                    "std_error": [record.std_error for record in records],
                    "covered": [int(record.covered) for record in records],
                    "rejected": [int(record.rejected) for record in records],
                }
            )
        )
    return pd.concat(frames, ignore_index=True).loc[:, list(REPLICATE_COLUMNS)]


def coverage_gain_interval(
    positive: pd.DataFrame,
    control: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Resampling interval for the coverage a positive cell buys over its paired control.

    Paired on ``replicate``, because the two cells are run on the same draws precisely so
    this difference is not two independent rates subtracted.  Shared rather than owned by
    one study family: two of them now make a claim about a pair of cells, and a statistic
    written twice is a statistic that can be changed once.
    """
    paired = positive[["replicate", "covered"]].merge(
        control[["replicate", "covered"]], on="replicate", suffixes=("_positive", "_control")
    )
    differences = paired["covered_positive"].to_numpy(dtype=float) - paired[
        "covered_control"
    ].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(differences), size=(replicates, len(differences)))
    interval = percentile_interval(
        differences[picks].mean(axis=1), confidence_level=confidence_level
    )
    return interval.low, interval.high


def replicate_row(
    *,
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: Any,
    alpha: float,
) -> dict[str, Any]:
    """One :data:`REPLICATE_COLUMNS` row, off an estimate that carries its own interval.

    Shared with :func:`run_cells`, which builds the same row from a
    :class:`~cleverly.validation.ReplicationRecord`.  A study whose fit reports several
    parameters per replication cannot use ``run_cells`` yet and assembles its rows itself, so
    this is the piece both paths need: the schema in one place rather than one copy per study
    that hand-rolls its loop.
    """
    low, high = estimate.ci
    return {
        "property": property_name,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
        "covered": int(low <= truth <= high),
        "rejected": int(estimate.pvalue < alpha),
    }


def control_row(
    *,
    property_name: str,
    cell: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: float,
    standard_error: float,
    critical: float,
    role: str = "control",
) -> dict[str, Any]:
    """Build a control row from an explicit point estimate and standard error.

    An unfluctuated plug-in and a survivor-only recursion are numbers, not fits: neither has
    an influence curve to report, and inventing one would make the control a claim about
    inference where it is a claim about bias. Those callers supply the paired positive arm's
    standard error, and their families gate on bias alone. An inference control can instead
    supply a second variance calculation for the same point estimate and gate the resulting
    coverage and standard-error ratio.

    ``role`` is here because a family may read one statistic at two points and gate on the
    pair, rather than pairing a fit against a control.  ``correction_necessity`` reads the
    DR-TMLE correction score before and after the cycle, so its "after" arm is the positive
    one; without the argument its caller patched the returned dictionary, which put the
    schema's meaning in the caller rather than here.

    A bias-only caller has no scale to supply either, and passes a unit placeholder.  The
    consequence is on the record rather than hidden: ``mean_std_error``, ``se_ratio``,
    ``coverage`` and ``rejection_rate`` are then arithmetic on that placeholder and mean
    nothing. Renderers and verdicts select on the family, so they do not publish those values.
    """
    half = critical * standard_error
    return {
        "property": property_name,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": estimate,
        "std_error": standard_error,
        "covered": int(estimate - half <= truth <= estimate + half),
        "rejected": int(abs(estimate / standard_error) > critical),
    }


def summary_interval(summary: pd.DataFrame, index: Any, prefix: str) -> Interval:
    """One already-computed interval, read back off a summary row by column prefix."""
    return Interval(
        float(summary.loc[index, f"{prefix}_ci_lower"]),
        float(summary.loc[index, f"{prefix}_ci_upper"]),
    )


def paired_displacement(rows: pd.DataFrame, family: str, left: str, right: str) -> float:
    """How far ``right``'s mean sits from ``left``'s, in ``left``'s empirical spread.

    The statistic a necessity family's *joint* claim is made of.  Each row's own endpoint says
    the positive arm's bias is inside the margin and the control's is outside it, and a step
    that did nothing at all would satisfy neither arm's rule in a way that distinguishes it --
    because the control would then simply be the estimate.  So the pair needs a statement of
    its own, and this is it.

    Refuses arms that are not aligned on ``replicate`` rather than subtracting two means that
    came off different draws, which is the whole reason the two arms are emitted from one
    replication.
    """
    arms = {
        name: rows.loc[(rows["property"] == family) & (rows["cell"] == cell)].sort_values(
            "replicate"
        )
        for name, cell in (("left", left), ("right", right))
    }
    if not np.array_equal(arms["left"]["replicate"], arms["right"]["replicate"]):
        raise ValueError(f"the {family} arms {left} and {right} are not paired on replication")
    spread = float(arms["left"]["estimate"].std(ddof=1))
    moved = float(arms["right"]["estimate"].mean() - arms["left"]["estimate"].mean())
    return abs(moved) / spread


def require_complete(rows: pd.DataFrame) -> None:
    """Refuse a property table that lost replications.

    Every verdict below standardizes by a spread or a rate estimated from these rows, so a
    cell that lost replications is not merely noisier -- for a bias claim it is *easier*,
    because the interval it has to sit inside is estimated from the same shrunken sample.
    """
    for (property_name, cell), group in rows.groupby(["property", "cell"], sort=True):
        requested = int(group["requested_replicates"].iloc[0])
        failed = int(group["failed_replicates"].iloc[0])
        if failed or len(group) != requested:
            raise ValueError(
                f"{property_name}/{cell} has {len(group)} of {requested} replications "
                f"({failed} fits failed); a study that lost replications cannot be summarised "
                f"as though it had not"
            )


def summarize_cells(
    rows: pd.DataFrame,
    *,
    margin: float,
    confidence_level: float,
    alpha: float,
) -> pd.DataFrame:
    """Descriptive summary plus the per-cell interval verdicts, one row per cell."""
    require_complete(rows)
    records: list[dict[str, Any]] = []
    for (property_name, cell), group in rows.groupby(["property", "cell"], sort=True):
        estimates = group["estimate"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        replicates = len(group)
        canonical = summarize_replications(
            tuple(
                ReplicationRecord(
                    replicate=int(row.replicate),
                    seed=-1,
                    estimand="cell",
                    truth=float(row.truth),
                    estimate=float(row.estimate),
                    std_error=float(row.std_error),
                    covered=bool(row.covered),
                    rejected=bool(row.rejected),
                    inference_estimate=float(row.estimate),
                    alpha=alpha,
                )
                for row in group.itertuples(index=False)
            ),
            estimand="cell",
            n=int(group["n"].iloc[0]),
        )
        bias = standardized_bias_verdict(
            estimates - truth, margin=margin, confidence_level=confidence_level
        )
        coverage = clopper_pearson(
            int(group["covered"].sum()), replicates, confidence_level=confidence_level
        )
        rejection = clopper_pearson(
            int(group["rejected"].sum()), replicates, confidence_level=confidence_level
        )
        empirical_se = bias.scale
        records.append(
            {
                "property": property_name,
                "cell": cell,
                "role": str(group["role"].iloc[0]),
                "n": int(group["n"].iloc[0]),
                "replicates": replicates,
                "failed_replicates": int(group["failed_replicates"].iloc[0]),
                "truth": truth,
                "mean_estimate": canonical.mean_estimate,
                "bias": canonical.bias,
                "bias_se": canonical.bias_se,
                "bias_ci_lower": bias.interval.low,
                "bias_ci_upper": bias.interval.high,
                "bias_margin": bias.margin,
                "standardized_bias": bias.standardized,
                "bias_equivalent": bias.equivalent,
                "bias_discriminated": bias.discriminated,
                "root_n_bias": canonical.root_n_bias,
                "empirical_se": empirical_se,
                "mean_std_error": canonical.mean_std_error,
                "se_ratio": canonical.se_ratio,
                "coverage": canonical.coverage,
                "coverage_ci_lower": coverage.low,
                "coverage_ci_upper": coverage.high,
                "rejection_rate": canonical.rejection_rate,
                "rejection_ci_lower": rejection.low,
                "rejection_ci_upper": rejection.high,
                "nominal_size": alpha,
            }
        )
    return pd.DataFrame.from_records(records)


def ratio_intervals(
    group: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    bound: float | None = None,
) -> dict[str, Interval]:
    """Every spread ratio a calibration cell reports, off **one** set of draws.

    Always returns ``se_ratio``: mean reported SE over the empirical spread of the estimates.
    That point ratio is a quotient of two statistics of the same replications, so its Monte
    Carlo error is dominated by the standard deviation in the denominator and is not available
    in closed form.  Resampling the replications jointly keeps numerator and denominator on the
    same draws, which is what makes the interval an interval for the ratio rather than for two
    unrelated quantities.

    ``bound`` is a study's independently computed efficiency bound -- on a finite-support law,
    :math:`\\sqrt{E_P[D^*(O)^2]}` taken from a Gateaux derivative rather than from anything the
    estimator reports.  Given one, two more intervals come back: ``efficiency_empirical`` for
    :math:`\\sqrt{n}` times the sampling spread over the bound, and ``efficiency_reported`` for
    the same over the mean reported standard error.

    All three ride the *same* index draws, and deliberately.  Resampled apart they would each
    be valid alone while disagreeing about their own arithmetic: ``se_ratio`` is
    ``efficiency_reported / efficiency_empirical`` replication by replication, and three
    independent seeds leave three intervals that no single resampled world produces.  Sharing
    the draws also costs a third of the work, which is what a ``(10,000 x 2,400)`` gather makes
    worth counting.
    """
    values = group[["estimate", "std_error"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[picks]
    spread = draws[:, :, 0].std(axis=1, ddof=1)
    reported = draws[:, :, 1].mean(axis=1)
    intervals = {
        "se_ratio": percentile_interval(reported / spread, confidence_level=confidence_level)
    }
    if bound is not None:
        scale = float(np.sqrt(int(group["n"].iloc[0]))) / bound
        intervals["efficiency_empirical"] = percentile_interval(
            spread * scale, confidence_level=confidence_level
        )
        intervals["efficiency_reported"] = percentile_interval(
            reported * scale, confidence_level=confidence_level
        )
    return intervals


def se_ratio_interval(
    group: pd.DataFrame, *, replicates: int, confidence_level: float, seed: int
) -> Interval:
    """Just the reported-over-empirical ratio of :func:`ratio_intervals`.

    Kept as its own name because most studies claim nothing about an efficiency bound, and a
    caller that wants one number should not have to know that two more are available or index
    a dictionary to say so.
    """
    return ratio_intervals(
        group, replicates=replicates, confidence_level=confidence_level, seed=seed
    )["se_ratio"]


def se_ratio_deficit_interval(
    subject: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> Interval:
    """Resampling interval for one cell's SE ratio *minus* a paired cell's.

    Both cells are run on the same draws, so the two ratios are resampled on one shared
    set of replication indices rather than independently.  That is what makes the interval
    an interval for the difference: the Monte Carlo error common to both -- the empirical
    spread of a shared sampling distribution -- cancels instead of being added twice, and a
    deficit of a few percent is resolvable at replication counts where each ratio on its own
    is not.

    Negative values mean ``subject`` reports a smaller standard error, relative to its own
    spread, than ``reference`` does.
    """
    merged = subject[["replicate", "estimate", "std_error"]].merge(
        reference[["replicate", "estimate", "std_error"]],
        on="replicate",
        suffixes=("_subject", "_reference"),
    )
    if len(merged) != len(subject) or len(merged) != len(reference):
        raise ValueError("the two cells are not paired on replication")
    values = merged[
        ["estimate_subject", "std_error_subject", "estimate_reference", "std_error_reference"]
    ].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(values), size=(replicates, len(values)))
    draws = values[picks]
    subject_ratio = draws[:, :, 1].mean(axis=1) / draws[:, :, 0].std(axis=1, ddof=1)
    reference_ratio = draws[:, :, 3].mean(axis=1) / draws[:, :, 2].std(axis=1, ddof=1)
    return percentile_interval(subject_ratio - reference_ratio, confidence_level=confidence_level)


@dataclass(frozen=True)
class Rate:
    """A fitted ``log(quantity) ~ log(n)`` slope with a bootstrap interval."""

    slope: float
    interval: Interval

    def equivalent_to(self, expected: float, margin: float) -> bool:
        """The whole interval lies within ``margin`` of ``expected`` -- the accept verdict.

        Margin-bounded rather than a containment test, for the reason the package docstring
        gives: ``interval.contains(expected)`` is a test against a point, so it gets *harder*
        as replications are added and eventually fails any estimator whose fitted rate is not
        exactly the asymptotic one.  :meth:`consistent_with` below is that rule, kept because
        :mod:`tests.unit.test_evidence_framework` holds the two side by side and asserts which
        way each one moves.
        """
        return self.interval.within(expected - margin, expected + margin)

    def consistent_with(self, expected: float) -> bool:
        """Does the interval contain ``expected``?  The rule :meth:`equivalent_to` replaced."""
        return self.interval.contains(expected)

    def excludes(self, value: float) -> bool:
        return not self.interval.contains(value)


def _positive(values: Any) -> Any:
    """Floor a magnitude away from zero so its logarithm stays finite.

    A bootstrap draw of an absolute bias can land on exactly zero, and one ``-inf`` would take
    the whole fitted slope with it.  The floor is far below any bias a study of this size can
    resolve -- a bias of ``1e-300`` and a bias of ``0`` are the same statement -- so it changes
    no verdict and removes a failure mode that has nothing to do with the estimator.
    """
    return np.maximum(values, np.finfo(float).tiny)


def _slope(sizes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Least-squares slope of ``values`` on ``log(sizes)``, vectorised over leading axes."""
    x = np.log(sizes)
    centred = x - x.mean()
    return (values * centred).sum(axis=-1) / (centred**2).sum()


def rate(
    rows: pd.DataFrame,
    *,
    property_name: str,
    statistic: str = "spread",
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> Rate:
    """How fast the sampling distribution contracts as ``n`` grows.

    ``statistic="spread"`` regresses the log *empirical* standard deviation of the estimates
    on log ``n``; root-n asymptotics predict a slope of :math:`-1/2`.  This is a property of
    the estimator's sampling distribution and can come out wrong.

    ``statistic="reported"`` does the same for the mean reported standard error.  That one is
    close to arithmetic -- an influence-curve standard error is
    :math:`\\hat\\sigma/\\sqrt{n}`, so any estimator that divides by the right power of ``n``
    produces :math:`-1/2` whether or not it is consistent.  It is kept because it does catch
    a standard error carrying the wrong power of ``n``, and labelled so it is not mistaken
    for evidence of root-n consistency.

    ``statistic="bias"`` regresses the log *absolute bias* on log ``n``, and it is the one
    that separates the two ways a level test can come out red.  A bias that is a second-order
    remainder contracts at :math:`n^{-1}` and one that is first order at :math:`n^{-1/2}`;
    both leave the standardized bias vanishing and the Wald interval eventually valid.  An
    inconsistent estimator's bias does not contract at all, so its slope is zero.  A single
    cell at a single ``n`` cannot tell those apart, which is why a red level cell used to be
    unreadable: the number said the margin was exceeded and nothing said whether that was a
    finite-sample remainder or a broken estimator.

    The bias is taken against the ``truth`` column, which every replication row carries.  It
    is a *signed* mean before the absolute value, so a cell whose bias is genuinely near zero
    produces a log of something near zero and a slope dominated by noise -- correctly, since
    there is no rate to estimate when there is no bias.  Read the slope beside the level cell
    rather than instead of it.
    """
    subset = rows.loc[rows["property"] == property_name]
    grouped = [(int(group["n"].iloc[0]), group) for _, group in subset.groupby("cell", sort=True)]
    grouped.sort(key=lambda item: item[0])
    if len(grouped) < 3:
        raise ValueError(
            f"{property_name} has {len(grouped)} sizes; a rate needs at least three so the "
            f"slope is estimated rather than read off one ratio"
        )
    sizes = np.array([size for size, _ in grouped], dtype=float)
    column = "std_error" if statistic == "reported" else "estimate"
    samples = [group[column].to_numpy(dtype=float) for _, group in grouped]
    # One truth per rung.  Read per rung rather than once, because a ladder is entitled to
    # a size-dependent truth and reading the first rung's would silently bias the others.
    truths = [float(group["truth"].iloc[0]) for _, group in grouped]

    def observed(values: np.ndarray, truth: float) -> float:
        if statistic == "spread":
            return float(np.log(np.std(values, ddof=1)))
        if statistic == "bias":
            return float(np.log(_positive(abs(values.mean() - truth))))
        return float(np.log(values.mean()))

    point = _slope(
        sizes,
        np.array([observed(values, truth) for values, truth in zip(samples, truths, strict=True)]),
    )

    draws = np.empty((bootstrap_replicates, len(sizes)), dtype=float)
    # A separate stream per size: the sizes are independent runs, and resampling them with
    # shared indices would pretend they were paired.  Spawned rather than ``seed + index``,
    # which only *looks* separate: two callers whose base seeds differ by one -- which is
    # exactly what the two published rate rows had -- then share every stream but the first.
    children = np.random.SeedSequence(seed).spawn(len(samples))
    for index, values in enumerate(samples):
        rng = np.random.default_rng(children[index])
        picks = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
        resampled = values[picks]
        if statistic == "spread":
            draws[:, index] = np.log(resampled.std(axis=1, ddof=1))
        elif statistic == "bias":
            draws[:, index] = np.log(_positive(np.abs(resampled.mean(axis=1) - truths[index])))
        else:
            draws[:, index] = np.log(resampled.mean(axis=1))
    return Rate(
        slope=float(point),
        interval=percentile_interval(_slope(sizes, draws), confidence_level=confidence_level),
    )
