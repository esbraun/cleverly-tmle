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

A fit that raises a recordable error, or that returns a non-finite estimate or standard error,
is a failed replication. The property loop records it and the accounting reads it. The sample
is drawn outside the fit, so a sampler error stops the run. The declared failure-only probe
found none (the design test pins its record). The policy is ``reporting``, and F28 owns every
red cell.
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
from tests.studies.evidence import property_verdicts
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
from tests.studies.evidence.replication import fit_properties
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


_Key = tuple[str, int, int, int]
_Drawn = tuple[pd.DataFrame, dict[str, float]]


def draw_cell(sizes: str, clusters: int, replicate: int) -> _Drawn:
    """One replication's sample and its truth."""
    return draw(sizes, clusters, _seed(sizes, clusters, replicate))


def fit_drawn(drawn: _Drawn) -> tuple[Any, float]:
    """The estimate of the target on a drawn sample, and its truth."""
    frame, truth = drawn
    return fit_cleverly(frame)[TARGET], float(truth[TARGET])


def fit_cell(sizes: str, clusters: int, replicate: int) -> tuple[Any, float]:
    """One replication's estimate of the target, and its truth."""
    return fit_drawn(draw_cell(sizes, clusters, replicate))


def _draw(key: _Key) -> _Drawn:
    """The sample a key fits, a function of the key alone."""
    sizes, clusters, replicate, _ = key
    return draw_cell(sizes, clusters, replicate)


def _cells_of(key: _Key) -> list[str]:
    """The three ``family/cell`` names one draw fills."""
    sizes, clusters = key[:2]
    return [f"{FAMILY}/{cell_name(sizes, clusters, arm)}" for arm, _ in ARMS]


def _cell_of(key: _Key) -> str:
    return f"{FAMILY}/{cell_name(key[0], key[1], 't_reference')}"


def _replicate_of(key: _Key) -> int:
    return key[2]


def _cell_rows(key: _Key, drawn: _Drawn) -> list[dict[str, Any]]:
    """The three arms of one draw."""
    sizes, clusters, replicate, requested = key
    estimate, truth = fit_drawn(drawn)
    standard_error = estimate.plugin_std_error
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
    ]


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every replication; ``budget`` caps each cell for a pre-run check.

    A key whose fit fails costs its three cells one replication each. The property loop
    records the failure, and the accounting reads the records.
    """
    replicates = PROPERTY_REPLICATES if budget is None else budget
    keys: list[_Key] = [
        (sizes, clusters, replicate, replicates)
        for sizes in SIZE_LAWS
        for clusters in CLUSTER_COUNTS
        for replicate in range(replicates)
    ]
    outcomes = fit_properties(
        keys,
        _cell_rows,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
    )
    return pd.DataFrame([row for result in outcomes for row in result])


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
