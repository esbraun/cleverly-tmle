"""The ``few_cluster_reference`` family of ``clustered-few-cluster-tmle``.

Each draw holds ``J`` clusters, with ``J`` in 10, 20 and 30, of one of two size laws:

* ``equal10``: clusters of 10 rows from the law of ``clustered_dgp(10, "binomial")``, which
  :func:`tests.studies.clustered_unequal_laws.draw_informative` reproduces with the
  arm-by-latent coefficient 8, no size terms and fixed sizes;
* ``unequal_informative``: the declared informative law of that module.

Five fits share each draw, four at J = 10, where in-sample LTMLE is not in the grid (its
interval floor is 20 clusters). Every fit of one ``(sizes, J)`` cell reads the same samples. The
stream is ``stream_seed(record, "few_cluster", f"{sizes}/J{J}", replicate)``:

==========================  ==================================================================
fit                         construction
==========================  ==================================================================
``tmle_crossfit``           stacked CV-TMLE, 5 grouped folds
``tmle_cv_evaluation``      fold-evaluated CV-TMLE, 5 folds, so each validation fold holds
                            at least 2 clusters
``tmle_in_sample``          in-sample TMLE
``drtmle_crossfit``         cross-fitted DR-TMLE, the folds of ``tmle_crossfit``
``ltmle_in_sample``         in-sample LTMLE on one node, the primary construction
==========================  ==================================================================

Every TMLE and DR-TMLE fit uses the unpenalized logistic outcome regression and the exact
propensity of :mod:`tests.studies.clustered_unequal_cvtmle`. The registered evidence is
parametric by design: Wang et al. (2024), Remark 3, caution against complex learners at about
20 clusters.

Each fit publishes three arms. ``t_reference`` (positive) is the reported interval, on t with
``J - 2`` degrees of freedom: the 99% exact coverage lower bound must reach 0.90, and the bias
must lie within 0.25 empirical SD. ``iid_t_control`` (control) keeps the point estimate and the
t quantile and uses the IID row standard error: its SE-ratio upper bound must fall below 0.80.
``normal_reference`` (reported) keeps the cluster-robust standard error with the normal
quantile. Every cell also reports its SE-ratio interval and its coverage against the 0.99
over-coverage ceiling, because at ``t_8`` excess width is as likely as under-coverage.

The fold-evaluated fit reports t with ``min(J - 2, J - V)`` degrees of freedom: its variance
centres the cluster totals in each of the ``V = 5`` folds, so it has ``J - V``. Its
``t_reference`` cells measure that rule, and a fourth, reported arm,
``t_j_minus_2_reference``, keeps ``J - 2`` on the same fits, so the run shows what the rule buys
at 2, 4 and 6 clusters per fold.

A fit that raises, or that returns a non-finite estimate or standard error, is a failed
replication. The study publishes the count in ``failed_replicates`` and never drops one
silently; the framework then refuses to summarise the cell. The policy is ``reporting``, the
owner of every red cell is F28, and nothing changes after a result is seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.linear_model import LinearRegression

from cleverly.estimators import DRTMLE, TMLE
from cleverly.inference import influence_variance
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_unequal_laws as laws
from tests.studies.canonical_drtmle import G_BOUNDS as DRTMLE_G_BOUNDS
from tests.studies.canonical_drtmle import ColumnLogistic
from tests.studies.clustered_few_cluster_tmle import (
    CLUSTER_COUNTS,
    FAMILY,
    N_FOLDS,
    PROPERTY_REPLICATES,
    SIZE_LAWS,
    STUDY,
    arms,
    cell_name,
    expected_reference_df,
    fit_cleverly,
    fits_at,
)
from tests.studies.clustered_unequal_cvtmle import tmle_settings
from tests.studies.evidence.properties import (
    PropertyCell,
    control_row,
    replicate_row,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

NORMAL_CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: What the ``iid_t_control`` SE-ratio upper bound must fall below.
IID_CONTROL_CEILING = 0.80
#: The estimand each fit reports the ATE under.
TARGET = {
    "tmle_crossfit": "ate",
    "tmle_cv_evaluation": "ate",
    "tmle_in_sample": "ate",
    "drtmle_crossfit": "ate",
    "ltmle_in_sample": "ate_regimen[always vs never]",
}


def t_critical(df: int) -> float:
    """The two-sided t quantile with ``df`` degrees of freedom."""
    return float(t.ppf(1.0 - STUDY.margins.alpha / 2.0, df))


def size_law_parameters(sizes: str) -> dict[str, float]:
    """The keyword arguments of :func:`~tests.studies.clustered_unequal_laws.draw_informative`."""
    if sizes == "equal10":
        return {"delta": 0.0, "gamma": 0.0, "effect_modifier": 8.0}
    if sizes == "unequal_informative":
        return {"delta": laws.DELTA, "gamma": laws.GAMMA, "effect_modifier": laws.EFFECT_MODIFIER}
    raise KeyError(sizes)


@dataclass(frozen=True)
class SizeLaw:
    """The law one cell reads, with its exact ATE.

    Parameters
    ----------
    name : str
        One of ``SIZE_LAWS``, with the cluster count after a slash.
    """

    name: str

    def truth(self) -> dict[str, float]:
        """The exact ATE of the law, under each name a fit reports it."""
        sizes = self.name.split("/", 1)[0]
        value = laws.informative_truth(**size_law_parameters(sizes))["ate"]
        return {"ate": value, "ate_regimen[always vs never]": value}


def draw(sizes: str, clusters: int, seed: int) -> pd.DataFrame:
    """One draw of ``clusters`` clusters of ``sizes``."""
    fixed = np.full(clusters, 10) if sizes == "equal10" else None
    return laws.draw_informative(
        clusters, np.random.default_rng(seed), sizes=fixed, **size_law_parameters(sizes)
    )


def _seed(sizes: str, clusters: int, replicate: int) -> int:
    return stream_seed(STUDY, "few_cluster", f"{sizes}/J{clusters}", replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with its law, its estimand and the root of its stream."""
    return tuple(
        PropertyCell(
            property=FAMILY,
            cell=cell_name(fit, sizes, clusters, arm),
            dgp=SizeLaw(f"{sizes}/J{clusters}"),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=10 * clusters,
            replicates=PROPERTY_REPLICATES,
            seed=_seed(sizes, clusters, 0),
            role=role,
            estimand=TARGET[fit],
        )
        for clusters in CLUSTER_COUNTS
        for fit in fits_at(clusters)
        for sizes in SIZE_LAWS
        for arm, role in arms(fit)
    )


def published_cells() -> tuple[str, ...]:
    """The cell names, in the order :func:`declared_cells` declares them."""
    return tuple(cell.cell for cell in declared_cells())


def fold_count(fit: str, clusters: int) -> int:
    """Five folds, which every grid count of at least 10 clusters supports."""
    del fit, clusters
    return N_FOLDS


def fit_estimate(fit: str, frame: pd.DataFrame, clusters: int) -> Any:
    """The ATE estimate of one fit."""
    roles = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"], "id": "cluster"}
    if fit == "ltmle_in_sample":
        return fit_cleverly(frame)[TARGET[fit]]
    if fit == "tmle_in_sample":
        settings = tmle_settings(cross_fit=False, estimands=("ate",))
        return TMLE(**settings).fit(frame, **roles).single()["ate"]
    folds = fold_count(fit, clusters)
    if fit == "tmle_crossfit":
        settings = tmle_settings(n_folds=folds, estimands=("ate",))
        return TMLE(**settings).fit(frame, **roles).single()["ate"]
    if fit == "tmle_cv_evaluation":
        settings = tmle_settings(n_folds=folds, cv_evaluation=True, estimands=("ate",))
        return TMLE(**settings).fit(frame, **roles).single()["ate"]
    if fit == "drtmle_crossfit":
        settings = tmle_settings(
            n_folds=folds,
            estimands=("ate",),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=ColumnLogistic(),
            # The canonical DR-TMLE study's bounds: the reduced treatment regression is
            # fitted, not exact, and 1e-9 let one curve entry reach 4.7e5.
            g_bounds=DRTMLE_G_BOUNDS,
        )
        return DRTMLE(**settings).fit(frame, **roles).single()["ate"]
    raise KeyError(fit)


def _cell_rows(
    payload: tuple[str, int, int, int],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, int]]]:
    """Every fit of one draw: its arms' rows, and the fits that failed."""
    sizes, clusters, replicate, requested = payload
    frame = draw(sizes, clusters, _seed(sizes, clusters, replicate))
    truth = SizeLaw(f"{sizes}/J{clusters}").truth()["ate"]
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str, int]] = []
    common = {"property_name": FAMILY, "replicate": replicate, "requested": requested}
    for fit in fits_at(clusters):
        try:
            estimate = fit_estimate(fit, frame, clusters)
            standard_error = estimate.plugin_std_error
            if not (np.isfinite(estimate.psi) and np.isfinite(standard_error)):
                raise ValueError("a non-finite estimate or standard error")
        except Exception:
            failures.append((fit, sizes, clusters))
            continue
        if estimate.reference_df != expected_reference_df(fit, clusters):
            raise AssertionError(
                f"{fit} at J = {clusters} reports reference_df {estimate.reference_df}"
            )
        n = 10 * clusters
        rows.append(
            replicate_row(
                cell=cell_name(fit, sizes, clusters, "t_reference"),
                role="positive",
                n=n,
                truth=truth,
                estimate=estimate,
                alpha=STUDY.margins.alpha,
                **common,
            )
        )
        rows.append(
            control_row(
                cell=cell_name(fit, sizes, clusters, "iid_t_control"),
                n=n,
                truth=truth,
                estimate=float(estimate.psi),
                standard_error=float(np.sqrt(influence_variance(estimate.influence_curve))),
                critical=t_critical(estimate.reference_df),
                **common,
            )
        )
        rows.append(
            control_row(
                cell=cell_name(fit, sizes, clusters, "normal_reference"),
                n=n,
                truth=truth,
                estimate=float(estimate.psi),
                standard_error=float(standard_error),
                critical=NORMAL_CRITICAL,
                role=DIAGNOSTIC_ROLE,
                **common,
            )
        )
        if fit == "tmle_cv_evaluation":
            rows.append(
                control_row(
                    cell=cell_name(fit, sizes, clusters, "t_j_minus_2_reference"),
                    n=n,
                    truth=truth,
                    estimate=float(estimate.psi),
                    standard_error=float(standard_error),
                    critical=t_critical(clusters - 2),
                    role=DIAGNOSTIC_ROLE,
                    **common,
                )
            )
    return rows, failures


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every replication; ``budget`` caps each cell for a pre-run check.

    The count of failed fits of each ``(fit, sizes, J)`` is written into every row of its three
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
    failed: dict[tuple[str, str, int], int] = {}
    for _, failures in outcomes:
        for key in failures:
            failed[key] = failed.get(key, 0) + 1
    for (fit, sizes, clusters), count in failed.items():
        for arm, _ in arms(fit):
            mask = rows["cell"] == cell_name(fit, sizes, clusters, arm)
            rows.loc[mask, "failed_replicates"] = count
    return rows


def few_cluster_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The positive, control and reported rules of every cell, with SE-ratio intervals."""
    margins = STUDY.margins
    family = summary["property"] == FAMILY
    for index in summary.index[family.to_numpy()]:
        cell = str(summary.loc[index, "cell"])
        role = str(summary.loc[index, "role"])
        group = rows.loc[(rows["property"] == FAMILY) & (rows["cell"] == cell)]
        ratio = se_ratio_interval(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, FAMILY, cell),
            truth_varies=False,
        )
        summary.loc[index, "se_ratio_ci_lower"] = ratio.low
        summary.loc[index, "se_ratio_ci_upper"] = ratio.high
        if role == "positive":
            summary.loc[index, "passed"] = bool(
                summary.loc[index, "coverage_ci_lower"] >= margins.coverage_floor
                and summary.loc[index, "bias_equivalent"]
            )
        elif role == "control":
            summary.loc[index, "passed"] = bool(ratio.high < IID_CONTROL_CEILING)
        else:
            summary.loc[index, "passed"] = True
        summary.loc[index, "property_passed"] = summary.loc[index, "passed"]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the declared rules; the policy is ``reporting``, so red cells are published."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    few_cluster_verdicts(summary, rows)
    return finish(summary, rates)
