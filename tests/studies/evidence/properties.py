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
from cleverly.validation._seeds import sample_seed_streams
from tests.studies.evidence.inference import (
    Interval,
    clopper_pearson,
    percentile_interval,
    standardized_bias_verdict,
)
from tests.studies.evidence.schema import reported_inference, reported_pvalue

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


#: Gathered elements one bootstrap block may materialize at once.  A block of ``b``
#: replicates over ``n`` rows of width ``w`` gathers ``b * n * w`` floats, so this caps a
#: block near 32 MB and the index matrix that feeds it near half that.
BOOTSTRAP_BLOCK_ELEMENTS = 4_000_000


def bootstrap_draw_blocks(values: np.ndarray, *, replicates: int, rng: np.random.Generator) -> Any:
    """Yield resampled blocks of ``values`` that together form ``replicates`` draws.

    ``rng.integers`` fills its output in C order, so drawing the index matrix in row blocks
    consumes exactly the stream one ``(replicates, n)`` call consumes.  Every block is
    therefore bit-identical to the corresponding rows of the single allocation this
    replaces, and no published interval moves.

    Blocking is what makes the cells with 40,000 replications affordable.  A ``(10,000 x
    40,000)`` index matrix and the ``(10,000 x 40,000 x 2)`` gather it feeds cost about 9.6
    GB together, which is fine alone and exhausts the machine once the test session runs
    sixteen workers at a time.

    Parameters
    ----------
    values : ndarray
        Rows to resample, either one-dimensional or one row per observation.
    replicates : int
        Total bootstrap replicates to draw.
    rng : numpy.random.Generator
        Generator that supplies the index draws.

    Yields
    ------
    ndarray
        One block of resampled draws, indexed by replicate first.
    """
    rows = len(values)
    width = max(1, values.size // max(1, rows))
    block = max(1, BOOTSTRAP_BLOCK_ELEMENTS // max(1, rows * width))
    drawn = 0
    while drawn < replicates:
        count = min(block, replicates - drawn)
        yield values[rng.integers(0, rows, size=(count, rows))]
        drawn += count


@dataclass(frozen=True)
class ReplicationSpec:
    """One repeated-sampling configuration before it expands into seeded replications.

    ``seed_key`` names the sample stream when it differs from ``cell``.  Two cells that share
    one key draw identical samples, which is how a paired comparison holds the data fixed.
    """

    property: str
    cell: str
    n: int
    replicates: int
    configuration: str
    seed_key: str | None = None


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
                spec.cell if spec.seed_key is None else spec.seed_key,
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


@dataclass(frozen=True)
class PropertyBatch:
    """One actual sampling call, including the factory shared by full and sparse runs."""

    name: str
    cells: tuple[PropertyCell, ...]
    estimator: Callable[[PropertyCell], Callable[[], Any]]

    def run(self, *, n_jobs: int) -> pd.DataFrame:
        """Run the complete declared batch."""
        return run_cells(self.cells, self.estimator, n_jobs=n_jobs)


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

    Distinct root seeds receive distinct sample seeds across the cells.  Equal roots share
    one prefix, including when their budgets differ, to retain intentionally paired draws.
    Planning by replication index preserves the old prefix when a rung budget increases.
    """
    budgets: dict[int, int] = {}
    for cell in cells:
        budgets[cell.seed] = max(budgets.get(cell.seed, 0), cell.replicates)
    sample_seeds = sample_seed_streams(budgets)
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
            sample_seeds=sample_seeds[cell.seed][: cell.replicates],
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
    means = np.concatenate(
        [
            block.mean(axis=1)
            for block in bootstrap_draw_blocks(differences, replicates=replicates, rng=rng)
        ]
    )
    interval = percentile_interval(means, confidence_level=confidence_level)
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

    The interval, the standard error and the p-value are read through
    :func:`~tests.studies.evidence.schema.reported_inference` and
    :func:`~tests.studies.evidence.schema.reported_pvalue`, so a row is written under any
    inference status and an inferential row keeps its numbers bit for bit.
    """
    std_error, low, high = reported_inference(estimate)
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
        "std_error": std_error,
        "covered": int(low <= truth <= high),
        "rejected": int(reported_pvalue(estimate) < alpha),
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


def spread_values(group: pd.DataFrame, *, truth_varies: bool) -> np.ndarray:
    """The values whose spread a statistic reads: the estimates, or each row's error.

    A record with one truth reads the estimates, bit for bit as before.  A record whose truth
    varies by replication (:attr:`StudyRecord.truth_varies_by_replicate`) reads the error
    ``estimate - truth`` of each row, because the spread of the estimate then also carries
    the spread of the target.
    """
    estimates = group["estimate"].to_numpy(dtype=float)
    if not truth_varies:
        return estimates
    return estimates - group["truth"].to_numpy(dtype=float)


def summary_interval(summary: pd.DataFrame, index: Any, prefix: str) -> Interval:
    """One already-computed interval, read back off a summary row by column prefix."""
    return Interval(
        float(summary.loc[index, f"{prefix}_ci_lower"]),
        float(summary.loc[index, f"{prefix}_ci_upper"]),
    )


def paired_displacement(
    rows: pd.DataFrame, family: str, left: str, right: str, *, truth_varies: bool = False
) -> float:
    """How far ``right``'s mean sits from ``left``'s, in ``left``'s empirical spread.

    ``truth_varies`` reads each arm's error instead of its estimate, each arm subtracting its
    own truth, as :func:`spread_values` does.

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
    if truth_varies:
        values = {
            name: pd.Series(spread_values(arm, truth_varies=True)) for name, arm in arms.items()
        }
    else:
        values = {name: arm["estimate"] for name, arm in arms.items()}
    spread = float(values["left"].std(ddof=1))
    moved = float(values["right"].mean() - values["left"].mean())
    return abs(moved) / spread


def require_complete(rows: pd.DataFrame) -> None:
    """Refuse a property table that lost replications.

    Every verdict below standardizes by a spread or a rate estimated from these rows, so a
    cell that lost replications is not merely noisier -- for a bias claim it is *easier*,
    because the interval it has to sit inside is estimated from the same shrunken sample.
    """

    def count(value: Any, *, field: str, property_name: str, cell: str) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{property_name}/{cell} has a non-integer {field}: {value!r}"
            ) from error
        if not np.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
            raise ValueError(f"{property_name}/{cell} has a non-integer {field}: {value!r}")
        return int(numeric)

    for (property_name, cell), group in rows.groupby(["property", "cell"], sort=True):
        requested_values = group["requested_replicates"].unique()
        if len(requested_values) != 1:
            raise ValueError(
                f"{property_name}/{cell} has inconsistent requested replication counts: "
                f"{sorted(int(value) for value in requested_values)}"
            )
        failed_values = group["failed_replicates"].unique()
        if len(failed_values) != 1:
            raise ValueError(
                f"{property_name}/{cell} has inconsistent failed replication counts: "
                f"{sorted(int(value) for value in failed_values)}"
            )
        requested = count(
            requested_values[0],
            field="requested replication count",
            property_name=str(property_name),
            cell=str(cell),
        )
        failed = count(
            failed_values[0],
            field="failed replication count",
            property_name=str(property_name),
            cell=str(cell),
        )
        if failed or len(group) != requested:
            raise ValueError(
                f"{property_name}/{cell} has {len(group)} of {requested} replications "
                f"({failed} fits failed); a study that lost replications cannot be summarised "
                f"as though it had not"
            )
        replicate_ids = np.sort(group["replicate"].to_numpy())
        expected_ids = np.arange(requested)
        if not np.array_equal(replicate_ids, expected_ids):
            raise ValueError(
                f"{property_name}/{cell} has duplicate or missing replicate ids; expected "
                f"0..{requested - 1} exactly once"
            )


def error_summary(
    errors: np.ndarray,
    std_errors: np.ndarray,
    covered: np.ndarray,
    rejected: np.ndarray,
    *,
    n: int,
    alpha: float,
) -> Any:
    """The canonical summary of a cell, computed on its errors rather than its estimates.

    Each record carries its error as the estimate and a truth of zero, so the canonical
    bias, spread, SE ratio and RMSE are the error's.  The caller publishes the mean estimate
    and the mean truth itself, because this summary's are the error's mean and zero.
    """
    return summarize_replications(
        tuple(
            ReplicationRecord(
                replicate=index,
                seed=-1,
                estimand="cell",
                truth=0.0,
                estimate=float(error),
                std_error=float(std_error),
                covered=bool(hit),
                rejected=bool(reject),
                inference_estimate=float(error),
                alpha=alpha,
            )
            for index, (error, std_error, hit, reject) in enumerate(
                zip(errors, std_errors, covered, rejected, strict=True)
            )
        ),
        estimand="cell",
        n=n,
    )


def summarize_cells(
    rows: pd.DataFrame,
    *,
    margin: float,
    confidence_level: float,
    alpha: float,
    truth_varies: bool = False,
) -> pd.DataFrame:
    """Descriptive summary plus the per-cell interval verdicts, one row per cell.

    ``truth_varies`` is :attr:`StudyRecord.truth_varies_by_replicate`.  Given it, the bias,
    the spread, the SE ratio and the RMSE read each row's error, ``truth`` publishes the mean
    truth, and ``truth_min`` and ``truth_max`` follow it.
    """
    require_complete(rows)
    records: list[dict[str, Any]] = []
    for (property_name, cell), group in rows.groupby(["property", "cell"], sort=True):
        estimates = group["estimate"].to_numpy(dtype=float)
        replicates = len(group)
        if truth_varies:
            truths = group["truth"].to_numpy(dtype=float)
            truth = float(np.mean(truths))
            errors = estimates - truths
            canonical = error_summary(
                errors,
                group["std_error"].to_numpy(dtype=float),
                group["covered"].to_numpy(),
                group["rejected"].to_numpy(),
                n=int(group["n"].iloc[0]),
                alpha=alpha,
            )
        else:
            truth = float(group["truth"].iloc[0])
            errors = estimates - truth
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
        bias = standardized_bias_verdict(errors, margin=margin, confidence_level=confidence_level)
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
                **(
                    {"truth_min": float(np.min(truths)), "truth_max": float(np.max(truths))}
                    if truth_varies
                    else {}
                ),
                "mean_estimate": float(np.mean(estimates))
                if truth_varies
                else canonical.mean_estimate,
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
    truth_varies: bool = False,
) -> dict[str, Interval]:
    """Every spread ratio a calibration cell reports, off **one** set of draws.

    ``truth_varies`` reads the spread of each row's error, as :func:`spread_values` does.

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
    return {
        name: percentile_interval(draws, confidence_level=confidence_level)
        for name, draws in ratio_draws(
            group, replicates=replicates, seed=seed, bound=bound, truth_varies=truth_varies
        ).items()
    }


def ratio_draws(
    group: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    bound: float | None = None,
    truth_varies: bool = False,
) -> dict[str, np.ndarray]:
    """The bootstrap draws behind :func:`ratio_intervals`, one array per ratio.

    A caller that compares two independent cells needs the draws rather than the intervals,
    because the interval of a difference is read off the draw-by-draw difference.
    :func:`ratio_intervals` takes its percentiles from exactly these arrays.

    ``truth_varies`` resamples each row's error and its SE together, by row.
    """
    if truth_varies:
        values = np.column_stack(
            [spread_values(group, truth_varies=True), group["std_error"].to_numpy(dtype=float)]
        )
    else:
        values = group[["estimate", "std_error"]].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    blocks = [
        (draws[:, :, 0].std(axis=1, ddof=1), draws[:, :, 1].mean(axis=1))
        for draws in bootstrap_draw_blocks(values, replicates=replicates, rng=rng)
    ]
    spread = np.concatenate([block[0] for block in blocks])
    reported = np.concatenate([block[1] for block in blocks])
    draws = {"se_ratio": reported / spread}
    if bound is not None:
        scale = float(np.sqrt(int(group["n"].iloc[0]))) / bound
        draws["efficiency_empirical"] = spread * scale
        draws["efficiency_reported"] = reported * scale
    return draws


def se_ratio_interval(
    group: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    truth_varies: bool = False,
) -> Interval:
    """Just the reported-over-empirical ratio of :func:`ratio_intervals`.

    Kept as its own name because most studies claim nothing about an efficiency bound, and a
    caller that wants one number should not have to know that two more are available or index
    a dictionary to say so.
    """
    return ratio_intervals(
        group,
        replicates=replicates,
        confidence_level=confidence_level,
        seed=seed,
        truth_varies=truth_varies,
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
    gaps = np.concatenate(
        [
            draws[:, :, 1].mean(axis=1) / draws[:, :, 0].std(axis=1, ddof=1)
            - draws[:, :, 3].mean(axis=1) / draws[:, :, 2].std(axis=1, ddof=1)
            for draws in bootstrap_draw_blocks(values, replicates=replicates, rng=rng)
        ]
    )
    return percentile_interval(gaps, confidence_level=confidence_level)


@dataclass(frozen=True)
class PairedSpreadRatio:
    """A paired spread ratio and its percentile-bootstrap interval."""

    ratio: float
    interval: Interval


def paired_spread_ratio_interval(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> PairedSpreadRatio:
    """Bootstrap one ratio of paired across-replication standard deviations.

    The two inputs must contain each replication exactly once and must have identical
    replication keys. The bootstrap samples paired rows with one shared index matrix. This
    preserves any dependence between the two estimates instead of resampling two marginal
    spreads independently.

    A zero denominator is undefined. The helper refuses it in either the observed statistic
    or a resampled draw instead of dropping that draw and changing the bootstrap law.
    """
    required = {"replicate", "estimate"}
    for label, frame in (("numerator", numerator), ("denominator", denominator)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"the {label} is missing required columns {sorted(missing)}")
        if frame["replicate"].duplicated().any():
            raise ValueError(f"the {label} contains duplicate replication keys")

    ordered_numerator = numerator.sort_values("replicate")
    ordered_denominator = denominator.sort_values("replicate")
    numerator_keys = ordered_numerator["replicate"].to_numpy()
    denominator_keys = ordered_denominator["replicate"].to_numpy()
    if not np.array_equal(numerator_keys, denominator_keys):
        raise ValueError("the numerator and denominator are not paired on replication")
    if len(numerator_keys) < 2:
        raise ValueError("a spread ratio needs at least two paired replications")

    values = np.column_stack(
        [
            ordered_numerator["estimate"].to_numpy(dtype=float),
            ordered_denominator["estimate"].to_numpy(dtype=float),
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("the paired estimates must all be finite")
    observed_spreads = values.std(axis=0, ddof=1)
    if observed_spreads[1] <= 0.0:
        raise ValueError("the observed denominator spread is zero")

    rng = np.random.default_rng(seed)
    spreads = np.concatenate(
        [
            draws.std(axis=1, ddof=1)
            for draws in bootstrap_draw_blocks(values, replicates=replicates, rng=rng)
        ]
    )
    if np.any(spreads[:, 1] <= 0.0):
        raise ValueError("a bootstrap draw has zero denominator spread")
    ratios = spreads[:, 0] / spreads[:, 1]
    return PairedSpreadRatio(
        ratio=float(observed_spreads[0] / observed_spreads[1]),
        interval=percentile_interval(ratios, confidence_level=confidence_level),
    )


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
    truth_varies: bool = False,
) -> Rate:
    """How fast the sampling distribution contracts as ``n`` grows.

    ``truth_varies`` reads each row's error in place of its estimate, so ``spread`` is the
    log SD of the error and ``bias`` the log absolute mean error.  ``reported`` is unchanged.

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
    if statistic != "reported" and truth_varies:
        # Each row's error, whose truth is zero by construction.
        samples = [spread_values(group, truth_varies=True) for _, group in grouped]
        truths = [0.0 for _ in grouped]
    else:
        column = "std_error" if statistic == "reported" else "estimate"
        samples = [group[column].to_numpy(dtype=float) for _, group in grouped]
        # One truth per rung.  Read per rung rather than once, because a ladder is entitled
        # to a size-dependent truth and reading the first rung's would silently bias the
        # others.
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
        filled = 0
        for resampled in bootstrap_draw_blocks(values, replicates=bootstrap_replicates, rng=rng):
            stop = filled + len(resampled)
            if statistic == "spread":
                draws[filled:stop, index] = np.log(resampled.std(axis=1, ddof=1))
            elif statistic == "bias":
                draws[filled:stop, index] = np.log(
                    _positive(np.abs(resampled.mean(axis=1) - truths[index]))
                )
            else:
                draws[filled:stop, index] = np.log(resampled.mean(axis=1))
            filled = stop
    return Rate(
        slope=float(point),
        interval=percentile_interval(_slope(sizes, draws), confidence_level=confidence_level),
    )
