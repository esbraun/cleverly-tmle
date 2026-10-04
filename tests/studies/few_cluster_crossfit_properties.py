"""The ``few_cluster_reference`` family of ``few-cluster-cross-fitted-ltmle``.

Each draw holds ``J`` clusters, with ``J`` in 20 and 30, of ``equal40`` or ``unequal40``
sizes, from the end-of-study law with the scaled latent. One cross-fitted clustered fit per
draw, five whole-cluster folds, so a validation fold holds 4 to 6 clusters. The stream is
``stream_seed(record, "few_cluster", f"{sizes}/J{J}", replicate)``.

The fit publishes three arms on ``ate_regimen[always vs never]``. ``t_reference`` (positive)
is the reported interval, on t with ``J - 2`` degrees of freedom: the 99% exact coverage lower
bound must reach 0.90, and the bias must lie within 0.25 empirical SD. ``iid_t_control``
(control) keeps the point estimate and the t quantile and uses the IID row standard error:
its SE-ratio upper bound must fall below 0.80. ``normal_reference`` (reported) keeps the
cluster-robust standard error with the normal quantile. The rule is the one
``clustered-few-cluster-tmle`` states, applied to this record.

A fit that raises, or that returns a non-finite estimate or standard error, is a failed
replication. The study publishes the count in ``failed_replicates``, and the framework then
refuses to summarise the cell. The declared failure-only probe found none (the design test
pins its record). The policy is ``reporting``, and F28 owns every red cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from cleverly.inference import influence_variance
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.properties import (
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    FEW_CLUSTER_IID_CONTROL_CEILING,
    apply_shared_verdicts,
    few_cluster_reference_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.few_cluster_crossfit_ltmle import (
    ARMS,
    CLUSTER_COUNTS,
    FAMILY,
    PROPERTY_REPLICATES,
    SIZE_LAWS,
    STUDY,
    TARGET,
    cell_name,
    draw,
    fit_cleverly,
)

NORMAL_CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: What the ``iid_t_control`` SE-ratio upper bound must fall below.
IID_CONTROL_CEILING = FEW_CLUSTER_IID_CONTROL_CEILING


def t_critical(df: int) -> float:
    """The two-sided t quantile with ``df`` degrees of freedom."""
    return float(t.ppf(1.0 - STUDY.margins.alpha / 2.0, df))


def nominal_n(sizes: str, clusters: int) -> int:
    """The nominal row count of a cell: 40 rows per cluster, the mean of both size laws."""
    del sizes
    return 40 * clusters


@dataclass(frozen=True)
class DeclaredLaw:
    """The law one cell reads, with the exact truth of the target.

    Parameters
    ----------
    name : str
        The size law and the cluster count, ``"<sizes>/J<count>"``.
    """

    name: str

    def truth(self) -> dict[str, float]:
        """``make_longitudinal``'s truth of the static contrast."""
        _, truth = draw("equal1", 2, 0)
        return {TARGET: float(truth[TARGET])}


def _seed(sizes: str, clusters: int, replicate: int) -> int:
    return stream_seed(STUDY, "few_cluster", f"{sizes}/J{clusters}", replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its law, its estimand and the root of its stream."""
    return tuple(
        PropertyCell(
            property=FAMILY,
            cell=cell_name(sizes, clusters, arm),
            dgp=DeclaredLaw(f"{sizes}/J{clusters}"),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=nominal_n(sizes, clusters),
            replicates=PROPERTY_REPLICATES,
            seed=_seed(sizes, clusters, 0),
            role=role,
            estimand=TARGET,
        )
        for sizes in SIZE_LAWS
        for clusters in CLUSTER_COUNTS
        for arm, role in ARMS
    )


def fit_cell(sizes: str, clusters: int, replicate: int) -> tuple[Any, float]:
    """One replication's estimate of the target, and its truth."""
    frame, truth = draw(sizes, clusters, _seed(sizes, clusters, replicate))
    return fit_cleverly(frame)[TARGET], float(truth[TARGET])


def _cell_rows(payload: tuple[str, int, int, int]) -> tuple[list[dict[str, Any]], bool]:
    """The three arms of one draw, or no rows and a failure."""
    sizes, clusters, replicate, requested = payload
    try:
        estimate, truth = fit_cell(sizes, clusters, replicate)
        standard_error = estimate.plugin_std_error
        if not (np.isfinite(estimate.psi) and np.isfinite(standard_error)):
            raise ValueError("a non-finite estimate or standard error")
    except Exception:
        return [], True
    if estimate.reference_df != clusters - 2:
        raise AssertionError(f"J = {clusters} reports reference_df {estimate.reference_df}")
    n = nominal_n(sizes, clusters)
    common = {"property_name": FAMILY, "replicate": replicate, "requested": requested}
    return [
        replicate_row(
            cell=cell_name(sizes, clusters, "t_reference"),
            role="positive",
            n=n,
            truth=truth,
            estimate=estimate,
            alpha=STUDY.margins.alpha,
            **common,
        ),
        control_row(
            cell=cell_name(sizes, clusters, "iid_t_control"),
            n=n,
            truth=truth,
            estimate=float(estimate.psi),
            standard_error=float(np.sqrt(influence_variance(estimate.influence_curve))),
            critical=t_critical(clusters - 2),
            **common,
        ),
        control_row(
            cell=cell_name(sizes, clusters, "normal_reference"),
            n=n,
            truth=truth,
            estimate=float(estimate.psi),
            standard_error=float(standard_error),
            critical=NORMAL_CRITICAL,
            role=DIAGNOSTIC_ROLE,
            **common,
        ),
    ], False


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every replication; ``budget`` caps each cell for a pre-run check.

    The count of failed fits of each ``(sizes, J)`` is written into every row of its three
    cells, so a cell that lost a replication is refused when it is summarised.
    """
    replicates = PROPERTY_REPLICATES if budget is None else budget
    payloads = [
        ((sizes, clusters, replicate, replicates),)
        for sizes in SIZE_LAWS
        for clusters in CLUSTER_COUNTS
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_cell_rows, payloads, n_jobs=n_jobs)
    rows = pd.DataFrame([row for result, _ in outcomes for row in result])
    failed: dict[tuple[str, int], int] = {}
    for (payload,), (_, failure) in zip(payloads, outcomes, strict=True):
        if failure:
            key = (payload[0], payload[1])
            failed[key] = failed.get(key, 0) + 1
    for (sizes, clusters), count in failed.items():
        for arm, _ in ARMS:
            mask = rows["cell"] == cell_name(sizes, clusters, arm)
            rows.loc[mask, "failed_replicates"] = count
    return rows


def failure_probe(draws: int, *, n_jobs: int = 1) -> dict[str, int]:
    """The declared failure-only probe: failed fits per cell over the first ``draws`` streams.

    It records no estimate. A cell with any failure is dropped before the declared run.
    """
    payloads = [
        ((sizes, clusters, replicate, draws),)
        for sizes in SIZE_LAWS
        for clusters in CLUSTER_COUNTS
        for replicate in range(draws)
    ]
    outcomes = map_parallel(_probe_one, payloads, n_jobs=n_jobs)
    counts = {f"{sizes}/J{clusters}": 0 for sizes in SIZE_LAWS for clusters in CLUSTER_COUNTS}
    for (payload,), failure in zip(payloads, outcomes, strict=True):
        counts[f"{payload[0]}/J{payload[1]}"] += int(failure)
    return counts


def _probe_one(payload: tuple[str, int, int, int]) -> bool:
    sizes, clusters, replicate, _ = payload
    try:
        estimate, _ = fit_cell(sizes, clusters, replicate)
        return not (np.isfinite(estimate.psi) and np.isfinite(estimate.plugin_std_error))
    except Exception:
        return True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the declared rules; the policy is ``reporting``, so red cells are published."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    few_cluster_reference_verdicts(summary, rows, STUDY)
    return finish(summary, rates)
