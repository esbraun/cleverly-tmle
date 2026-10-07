"""Property families of ``clustered-unequal-cvtmle``.

Three families, each with its own sample stream:

``clustered_inference`` (gated)
    Four pairs of a cluster-robust arm and an IID control, each pair one fit on one draw:
    the stacked fit, the fold-evaluated fit (``cv_evaluation=True``), the cross-fitted
    ``DRTMLE``, all on the informative law at 200 clusters; and the fold-evaluated fit on the
    cluster-level covariate law at 40 clusters of 30 rows in 10 folds, for the treated mean
    ``ey1``. That last pair is the
    configuration of the shipped fold-evaluated variance defect that commit 6c237420 fixed.
    The rules are :func:`~tests.studies.evidence.property_verdicts.clustered_inference_verdicts`.
``estimand_weighting`` (gated)
    One draw of the informative law, fitted twice: unweighted, and with ``weights`` equal to
    one over the cluster size. The unweighted interval must cover the row-weighted mean
    :math:`\\mu_I` (``individual_average``) and must not cover the cluster-average mean
    :math:`\\mu_C` (``cluster_average_truth``). The weighted interval must do the reverse. A
    positive cell needs the 99% lower coverage bound at 0.90; a control needs the 99% upper
    coverage bound below 0.50.
``cluster_aggregation_rule`` (reported)
    The same stacked curve aggregated by cluster sums, as this package and R ``ltmle`` do,
    and by cluster means, as R ``tmle`` 2.1.1 and ``tmle3`` do. Reported, not gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import DRTMLE, TMLE
from cleverly.inference import influence_variance
from cleverly.utils.parallel import map_parallel
from tests.conftest import OracleTreatment
from tests.parallel import STUDY_JOBS
from tests.studies import clustered_unequal_laws as laws
from tests.studies.canonical_drtmle import G_BOUNDS as DRTMLE_G_BOUNDS
from tests.studies.canonical_drtmle import ColumnLogistic
from tests.studies.clustered_unequal_cvtmle import (
    CLUSTER_AVERAGE,
    INFORMATIVE,
    PRIMARY_N,
    PROPERTY_REPLICATES,
    STUDY,
    ExactPropensity,
    draw_from_seed,
    fit_cleverly,
    tmle_settings,
)
from tests.studies.evidence.properties import PropertyCell, control_row, replicate_row
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    clustered_inference_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

TARGET = "ate"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
CLUSTERED = "clustered_inference"
WEIGHTING = "estimand_weighting"
AGGREGATION = "cluster_aggregation_rule"
#: The cluster-level covariate law of the fold-variance pair: 40 clusters in 10 folds.
COVARIATE_CLUSTERS = 40
COVARIATE_FOLDS = 10
#: The target of the fold-variance pair. The cluster-shared covariate enters the plug-in part
#: of the arm mean, which is where the defect lived; the ATE's plug-in parts largely cancel
#: (a pre-run probe of 60 fits: ey1 IID SE / SD 0.39, ATE 0.96 against robust 1.14 and 0.98).
COVARIATE_TARGET = "ey1"
#: The 99% upper coverage bound a ``*_truth`` control must fall below.
CONTROL_COVERAGE_CEILING = 0.50

#: Each clustered pair: (positive cell, control cell, fit kind).
PAIRS = (
    ("cluster_robust", "iid_control", "stacked"),
    ("cluster_robust_fold_evaluated", "iid_control_fold_evaluated", "fold_evaluated"),
    ("cluster_robust_drtmle", "iid_control_drtmle", "drtmle"),
    (
        "cluster_robust_fold_evaluated_cluster_covariate",
        "iid_control_fold_evaluated_cluster_covariate",
        "covariate_fold_evaluated",
    ),
)
WEIGHTING_CELLS = (
    ("individual_average", "unweighted", "ate", "positive"),
    ("cluster_average_truth", "unweighted", "ate_cluster", "control"),
    ("cluster_average_weights", "weighted", "ate_cluster", "positive"),
    ("individual_average_truth", "weighted", "ate", "control"),
)


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every reported name.

    Parameters
    ----------
    name : str
        ``"informative"`` or ``"cluster_covariate"``, with the fit kind after a colon so two
        families never share a ``(name, seed)`` stream.
    """

    name: str

    def truth(self) -> dict[str, float]:
        """The exact ``ate`` and the cluster-average ``ate_cluster`` of the law."""
        if self.name.startswith("cluster_covariate"):
            return {"ey1": laws.covariate_truth()["ey1"]}
        truth = laws.informative_truth()
        return {"ate": truth["ate"], "ate_cluster": truth["ate_cluster"]}


def _law_name(kind: str) -> str:
    base = "cluster_covariate" if kind == "covariate_fold_evaluated" else "informative"
    return f"{base}:{kind}"


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with the law it reads, its estimand and its stream root.

    The truth of each published row is the declared law's own:
    ``tests/unit/test_clustered_unequal_study.py`` runs every cell at a budget of two and
    checks the published set and truths against this declaration.
    """
    cells: list[PropertyCell] = []
    for positive, control, kind in PAIRS:
        for cell, role in ((positive, "positive"), (control, "control")):
            estimand = COVARIATE_TARGET if kind == "covariate_fold_evaluated" else TARGET
            cells.append(_cell(CLUSTERED, cell, role, kind, kind, estimand))
    for cell, _, estimand, role in WEIGHTING_CELLS:
        cells.append(_cell(WEIGHTING, cell, role, "weighting", "paired", estimand))
    for cell in ("cluster_sum", "cluster_mean"):
        cells.append(_cell(AGGREGATION, cell, DIAGNOSTIC_ROLE, "aggregation", "paired", "ate"))
    return tuple(cells)


def _cell(family: str, cell: str, role: str, kind: str, label: str, estimand: str) -> Any:
    return PropertyCell(
        property=family,
        cell=cell,
        dgp=DeclaredLaw(_law_name(kind)),
        outcome_learner=lambda: None,
        treatment_learner=lambda: None,
        n=PRIMARY_N,
        replicates=PROPERTY_REPLICATES,
        seed=_seed(family, label, 0),
        role=role,
        estimand=estimand,
    )


def _iid(estimate: Any) -> float:
    return float(np.sqrt(influence_variance(estimate.influence_curve)))


def _covariate_fit(seed: int) -> tuple[Any, float]:
    frame = laws.draw_cluster_covariate(COVARIATE_CLUSTERS, np.random.default_rng(seed))
    result = (
        TMLE(
            outcome_learner=LogisticRegression(C=1e6, max_iter=1000),
            treatment_learner=LogisticRegression(C=1e6, max_iter=1000),
            cross_fit=True,
            n_folds=COVARIATE_FOLDS,
            cv_evaluation=True,
            stratify_folds="none",
            estimands=(COVARIATE_TARGET,),
            simultaneous=False,
            g_bounds=(1e-9, 1.0 - 1e-9),
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
        .single()
    )
    return result[COVARIATE_TARGET], laws.covariate_truth()[COVARIATE_TARGET]


def _informative_fit(kind: str, seed: int) -> tuple[Any, float]:
    frame, truth = draw_from_seed(INFORMATIVE, PRIMARY_N, seed)
    if kind == "stacked":
        return fit_cleverly(frame, INFORMATIVE)[TARGET], truth[TARGET]
    if kind == "fold_evaluated":
        return fit_cleverly(frame, INFORMATIVE, cv_evaluation=True)[TARGET], truth[TARGET]
    if kind == "drtmle":
        settings = tmle_settings(
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=ColumnLogistic(),
            # The canonical DR-TMLE study's bounds: the reduced treatment regression is
            # fitted, not exact, and 1e-9 let one curve entry reach 4.7e5.
            g_bounds=DRTMLE_G_BOUNDS,
        )
        result = (
            DRTMLE(**settings)
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
            .single()
        )
        return result[TARGET], truth[TARGET]
    raise KeyError(kind)


def _pair_rows(payload: tuple[int, int, int, str, str, str]) -> list[dict[str, Any]]:
    replicate, requested, seed, positive, control, kind = payload
    if kind == "covariate_fold_evaluated":
        estimate, truth = _covariate_fit(seed)
    else:
        estimate, truth = _informative_fit(kind, seed)
    robust = replicate_row(
        property_name=CLUSTERED,
        cell=positive,
        role="positive",
        replicate=replicate,
        n=PRIMARY_N,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    iid = control_row(
        property_name=CLUSTERED,
        cell=control,
        replicate=replicate,
        n=PRIMARY_N,
        requested=requested,
        truth=truth,
        estimate=float(estimate.psi),
        standard_error=_iid(estimate),
        critical=CRITICAL,
    )
    return [robust, iid]


def _weighting_rows(payload: tuple[int, int, int]) -> list[dict[str, Any]]:
    replicate, requested, seed = payload
    frame, _ = draw_from_seed(INFORMATIVE, PRIMARY_N, seed)
    truth = laws.informative_truth()
    fits = {
        "unweighted": fit_cleverly(frame, INFORMATIVE)[TARGET],
        "weighted": fit_cleverly(frame, CLUSTER_AVERAGE)[TARGET],
    }
    return [
        replicate_row(
            property_name=WEIGHTING,
            cell=cell,
            role=role,
            replicate=replicate,
            n=PRIMARY_N,
            requested=requested,
            truth=float(truth[estimand]),
            estimate=fits[fit],
            alpha=STUDY.margins.alpha,
        )
        for cell, fit, estimand, role in WEIGHTING_CELLS
    ]


def cluster_mean_standard_error(curve: Any, cluster: Any) -> float:
    """``sqrt(var(by(IC, id, mean)) / J)``, the cluster-mean rule of R ``tmle`` 2.1.1."""
    codes = np.asarray(cluster)
    values = np.asarray(curve, dtype=float)
    means = np.array([values[codes == code].mean() for code in np.unique(codes)])
    return float(np.sqrt(np.var(means, ddof=1) / means.size))


def _aggregation_rows(payload: tuple[int, int, int]) -> list[dict[str, Any]]:
    replicate, requested, seed = payload
    frame, truth = draw_from_seed(INFORMATIVE, PRIMARY_N, seed)
    estimate = fit_cleverly(frame, INFORMATIVE)[TARGET]
    cluster = frame["cluster"].to_numpy()
    rules = {
        "cluster_sum": estimate.std_error,
        "cluster_mean": cluster_mean_standard_error(estimate.influence_curve, cluster),
    }
    return [
        control_row(
            property_name=AGGREGATION,
            cell=cell,
            replicate=replicate,
            n=PRIMARY_N,
            requested=requested,
            truth=float(truth[TARGET]),
            estimate=float(estimate.psi),
            standard_error=float(rules[cell]),
            critical=CRITICAL,
            role=DIAGNOSTIC_ROLE,
        )
        for cell in ("cluster_sum", "cluster_mean")
    ]


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    replicates = PROPERTY_REPLICATES if budget is None else budget
    out: list[tuple[Any, ...]] = []
    for positive, control, kind in PAIRS:
        out.extend(
            ((_pair_rows, (r, replicates, _seed(CLUSTERED, kind, r), positive, control, kind)),)
            for r in range(replicates)
        )
    out.extend(
        ((_weighting_rows, (r, replicates, _seed(WEIGHTING, "paired", r))),)
        for r in range(replicates)
    )
    out.extend(
        ((_aggregation_rows, (r, replicates, _seed(AGGREGATION, "paired", r))),)
        for r in range(replicates)
    )
    return out


def _run(job: tuple[Any, tuple[Any, ...]]) -> list[dict[str, Any]]:
    function, payload = job
    return list(function(payload))


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each family for a pre-run check."""
    outcomes = map_parallel(_run, _payloads(budget), n_jobs=n_jobs)
    return pd.DataFrame([row for rows in outcomes for row in rows])


def estimand_weighting_verdicts(summary: pd.DataFrame) -> None:
    """Positive cells clear the coverage floor; controls fall below the declared ceiling."""
    margins = STUDY.margins
    family = summary["property"] == WEIGHTING
    positive = family & (summary["role"] == "positive")
    control = family & (summary["role"] == "control")
    summary.loc[positive, "passed"] = summary.loc[positive, "coverage_ci_lower"] >= (
        margins.coverage_floor
    )
    summary.loc[control, "passed"] = (
        summary.loc[control, "coverage_ci_upper"] < CONTROL_COVERAGE_CEILING
    )
    summary.loc[family, "property_passed"] = bool(summary.loc[family, "passed"].all())


def aggregation_diagnostics(summary: pd.DataFrame) -> None:
    """A reported family: the numbers are published, and no verdict is formed."""
    family = summary["property"] == AGGREGATION
    roles = set(summary.loc[family, "role"].astype(str))
    if family.any() and roles != {DIAGNOSTIC_ROLE}:
        raise ValueError(f"{AGGREGATION} publishes roles {sorted(roles)}")
    summary.loc[family, "passed"] = True
    summary.loc[family, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """Apply the declared clustered, weighting and aggregation rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
        rate_labels=(),
    )
    for positive, control, _ in PAIRS:
        clustered_inference_verdicts(
            summary, rows, STUDY, positive_cell=positive, control_cell=control
        )
    estimand_weighting_verdicts(summary)
    aggregation_diagnostics(summary)
    return finish(summary, rates)


def exact_propensity() -> OracleTreatment:
    """The exact treatment mechanism of the informative law."""
    return OracleTreatment(ExactPropensity())
