"""Property cells for the full-refit bootstrap and the derived post-fit contrasts.

Every cell sits in the shared ``interval_calibration`` family, so the shared calibration
verdicts apply unchanged: a positive cell needs its SE-ratio and coverage intervals inside the
calibration bands, and a shrunken control needs its SE-ratio interval below the band.

What a row measures depends on the cell.

* A ratio row records the log ratio, its log-scale standard error and whether the
  exponentiated interval covers the true ratio.
* An RMST row records the RMST or the RMST contrast with its influence-curve interval.
* A bootstrap row records the point estimate, the bootstrap standard deviation as
  ``std_error``, and whether the two-sided percentile interval covers the truth.  The
  shrunken control shrinks that interval about the median of the draws.

Each bootstrap row also records how many bootstrap replicates were requested and how many
failed.  :func:`summarize_properties` publishes the mean failed share per cell, and a cell above
:data:`~tests.studies.canonical_full_refit_bootstrap.FAILED_REPLICATE_CAP` claims no coverage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.estimators import TMLE
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law as point_law
from tests import discrete_law_longitudinal as end_law
from tests.parallel import STUDY_JOBS
from tests.studies import survival_grid_law as grid
from tests.studies.canonical_full_refit_bootstrap import (
    BOOTSTRAP_REPLICATES,
    CLUSTER_SHIFT,
    CLUSTER_SIZE,
    CLUSTERS,
    FAILED_REPLICATE_CAP,
    SHRUNKEN_SE_FACTOR,
    STUDY,
    fit_end_of_study,
    groups,
    sample_end_of_study,
)
from tests.studies.canonical_ltmle import G_BOUNDS
from tests.studies.evidence.properties import REPLICATE_COLUMNS, finite_support_sample
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

#: The extra replicate columns a bootstrap row carries.
BOOTSTRAP_COLUMNS = ("bootstrap_requested", "bootstrap_failed")
#: The extra summary columns.
SUMMARY_COLUMNS = ("bootstrap_failed_fraction", "bootstrap_conditional")

CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
GRID_REGIMENS = {"never": list(grid.REGIMENS["never"]), "always": list(grid.REGIMENS["always"])}
GRID_HORIZON = grid.K
RMST_HORIZON = grid.K + 1


class MarkovCellMeans(end_law.CellMeans):
    """Cell means over the columns each true conditional of the four-node law reads.

    The law's outcome hazard reads ``(W, L_t, A_t)``, its treatment ``(W, L_t, A_{t-1})`` and
    its censoring ``(W, L_t, A_t)``.  Among a regimen's followers ``A_t`` is fixed, so the
    sequential regression reads ``(W, L_t)``.  A saturated learner over the whole history
    would also be correctly specified, but it splits a sample of 1,000 into hundreds of
    cells, and its fitted mechanism then reaches 0 or 1.

    Parameters
    ----------
    role : {"outcome", "treatment", "censoring"}
        Which design the learner is handed: ``[W, L_1..L_t]``, ``[W, L_1..L_t, A_1..A_{t-1}]``
        or ``[W, L_1..L_t, A_1..A_t]``.
    """

    def __init__(self, role: str = "outcome") -> None:
        self.role = role

    def _select(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        width = matrix.shape[1]
        if self.role == "outcome":
            columns = [0, width - 1]
        elif self.role == "treatment":
            t = width // 2
            columns = [0, t] + ([width - 1] if t > 1 else [])
        elif self.role == "censoring":
            t = (width - 1) // 2
            columns = [0, t, width - 1]
        else:
            raise ValueError(f"unknown role {self.role!r}")
        return matrix[:, columns]

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> MarkovCellMeans:
        super().fit(self._select(X), y, sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(self._select(X))


def fit_grid(frame: pd.DataFrame, *, n_folds: int = 1, **settings: Any) -> Any:
    """A fit of the four-node survival law with the correctly specified learners."""
    return LTMLE(
        GRID_REGIMENS,
        reference="never",
        outcome_learner=MarkovCellMeans("outcome"),
        pseudo_learner=MarkovCellMeans("outcome"),
        treatment_learner=MarkovCellMeans("treatment"),
        censoring_learner=MarkovCellMeans("censoring"),
        n_folds=n_folds,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        n_jobs=1,
        **settings,
    ).fit(frame, **grid.fit_columns())


def sample_clustered(seed: int) -> pd.DataFrame:
    """The end-of-study law in clusters that share a latent effect modifier.

    Each cluster draws a sign ``U``, and an uncensored unit's outcome is redrawn with
    probability ``Q + CLUSTER_SHIFT * U * (A1 + A2 - 1)``.  ``U`` has mean zero and moves no
    treatment or censoring node, so every regimen mean is the law's own and the rows of a
    cluster are correlated.
    """
    rng = np.random.default_rng(seed)
    frame = sample_end_of_study(CLUSTERS * CLUSTER_SIZE, int(rng.integers(2**31)))
    cluster = np.repeat(np.arange(CLUSTERS), CLUSTER_SIZE)
    latent = rng.choice((-1.0, 1.0), size=CLUSTERS)[cluster]
    observed = frame["C2"].to_numpy() == 1.0
    w = frame["W"].to_numpy().astype(int)
    a1 = frame["A1"].to_numpy()
    l2 = np.nan_to_num(frame["L2"].to_numpy()).astype(int)
    a2 = np.nan_to_num(frame["A2"].to_numpy()).astype(int)
    a1_index = np.nan_to_num(a1).astype(int)
    probability = end_law.Q[w, a1_index, l2, a2] + CLUSTER_SHIFT * latent * (a1_index + a2 - 1)
    redrawn = (rng.random(len(frame)) < probability).astype(float)
    frame["Y"] = np.where(observed, redrawn, np.nan)
    frame["cluster"] = cluster
    # Kept for the design test's within-cluster witness; no fit reads it.
    frame["latent"] = latent
    return frame


def _truths() -> dict[str, float]:
    always = float(end_law.TRUTH["ey_regimen[always]"])
    never = float(end_law.TRUTH["ey_regimen[never]"])
    risk = {
        (label, t): grid.risk_truth(label, t) for label in GRID_REGIMENS for t in (2, GRID_HORIZON)
    }
    f_a, f_n = risk["always", GRID_HORIZON], risk["never", GRID_HORIZON]
    return {
        "rr_end_of_study": float(np.log(always / never)),
        "or_end_of_study": float(np.log((always / (1 - always)) / (never / (1 - never)))),
        "rr_survival": float(np.log(f_a / f_n)),
        "survival_rr_survival": float(np.log((1 - f_a) / (1 - f_n))),
        "rmst_survival": grid.rmst_truth("always", RMST_HORIZON),
        "rmst_contrast_survival": grid.rmst_truth("always", RMST_HORIZON)
        - grid.rmst_truth("never", RMST_HORIZON),
        "rmst_crossfit": grid.rmst_truth("always", RMST_HORIZON),
        "rmst_contrast_crossfit": grid.rmst_truth("always", RMST_HORIZON)
        - grid.rmst_truth("never", RMST_HORIZON),
        "boot_ey_end_of_study": always,
        "boot_ate_end_of_study": always - never,
        "boot_ey_crossfit": always,
        "boot_ate_crossfit": always - never,
        "boot_risk2_survival": risk["always", 2],
        "boot_risk4_survival": risk["always", GRID_HORIZON],
        "boot_rmst_survival": grid.rmst_truth("always", RMST_HORIZON),
        "boot_ey_clustered": always,
        "boot_ate_clustered": always - never,
        "boot_ate_point_tmle": float(point_law.TRUTH["ate"]),
    }


TRUTHS = _truths()
GROUPS = groups()


def _base_row(spec: dict[str, Any], replicate: int, truth: float) -> dict[str, Any]:
    return {
        "property": "interval_calibration",
        "cell": f"{spec['label']}__correctly_specified",
        "role": "positive",
        "replicate": replicate,
        "n": spec["n"],
        "requested_replicates": spec["replicates"],
        "failed_replicates": 0,
        "truth": truth,
        "bootstrap_requested": spec["bootstrap"],
        "bootstrap_failed": 0,
    }


def _wald_rows(
    spec: dict[str, Any], replicate: int, truth: float, value: float, se: float
) -> list[dict[str, Any]]:
    """A positive row on the inference scale, and its shrunken control when declared."""
    rows = []
    for factor, suffix, role in ((1.0, "correctly_specified", "positive"),) + (
        ((SHRUNKEN_SE_FACTOR, "shrunken_se_control", "control"),) if spec["control"] else ()
    ):
        row = _base_row(spec, replicate, truth)
        half = CRITICAL * se * factor
        row.update(
            cell=f"{spec['label']}__{suffix}",
            role=role,
            estimate=value,
            std_error=se * factor,
            covered=int(value - half <= truth <= value + half),
            rejected=int(abs(value) / (se * factor) > CRITICAL),
        )
        rows.append(row)
    return rows


def _bootstrap_rows(
    spec: dict[str, Any], replicate: int, truth: float, estimate: Any, failed: int
) -> list[dict[str, Any]]:
    """A positive row read off the shipped bootstrap summary, and its shrunken control.

    The positive row records ``estimate.bootstrap.std_error`` and whether
    ``estimate.bootstrap.ci`` covers the truth, so the cell measures what a user reads.  The
    control shrinks that interval about the median of the shipped draws.
    """
    summary = estimate.bootstrap
    if summary is None:
        raise RuntimeError(f"{estimate.name} carries no bootstrap summary")
    low, high = summary.ci
    median = float(np.median(summary.draws))
    rows = []
    for factor, suffix, role in ((1.0, "correctly_specified", "positive"),) + (
        ((SHRUNKEN_SE_FACTOR, "shrunken_se_control", "control"),) if spec["control"] else ()
    ):
        lo = median + factor * (low - median)
        hi = median + factor * (high - median)
        row = _base_row(spec, replicate, truth)
        row.update(
            cell=f"{spec['label']}__{suffix}",
            role=role,
            estimate=float(estimate.psi),
            std_error=float(summary.std_error) * factor,
            covered=int(lo <= truth <= hi),
            rejected=int(not lo <= 0.0 <= hi),
            bootstrap_failed=failed,
        )
        rows.append(row)
    return rows


def _fit(group: str, spec: dict[str, Any], replicate: int) -> Any:
    """One sample and one fit, shared by every cell of ``group``."""
    seed = stream_seed(STUDY, "property_sample", group, replicate)
    fit_seed = stream_seed(STUDY, "fit_seed", group, replicate)
    boot = {"n_bootstrap": spec["bootstrap"]} if spec["bootstrap"] else {}
    if spec["law"] == "point":
        frame = finite_support_sample(
            point_law.PROBS, point_law.SUPPORT, spec["n"], seed, columns=("W", "A", "Y")
        )
        return (
            TMLE(
                outcome_learner=end_law.CellMeans(),
                treatment_learner=end_law.CellMeans(),
                cross_fit=False,
                estimands=("ate",),
                simultaneous=False,
                random_state=fit_seed,
                n_jobs=1,
                **boot,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W"])
            .single()
        )
    if spec["law"] == "end_of_study":
        frame = sample_end_of_study(spec["n"], seed)
        return fit_end_of_study(frame, n_folds=spec["folds"], random_state=fit_seed, **boot)
    if spec["law"] == "clustered":
        frame = sample_clustered(seed)
        return fit_end_of_study(frame, n_folds=spec["folds"], random_state=fit_seed, **boot)
    frame = grid.sample(spec["n"], np.random.default_rng(seed))
    return fit_grid(frame, n_folds=spec["folds"], random_state=fit_seed, **boot)


def _rows(spec: dict[str, Any], result: Any, replicate: int) -> list[dict[str, Any]]:
    """The rows of one cell, read off its group's fit."""
    label = spec["label"]
    truth = TRUTHS[label]
    if label in ("rr_end_of_study", "or_end_of_study"):
        kind = "or" if label.startswith("or_") else "rr"
        ratio = result.ratio("ey_regimen[always]", "ey_regimen[never]", kind=kind)
        return _wald_rows(spec, replicate, truth, ratio.log_psi, ratio.plugin_std_error)
    a = f"risk_regimen[always @ t={GRID_HORIZON}]"
    b = f"risk_regimen[never @ t={GRID_HORIZON}]"
    if label in ("rr_survival", "survival_rr_survival"):
        view = "survival" if label.startswith("survival_") else "risk"
        ratio = result.ratio(a, b, view=view)
        return _wald_rows(spec, replicate, truth, ratio.log_psi, ratio.plugin_std_error)
    if label.startswith("rmst_"):
        versus = "never" if "contrast" in label else None
        rmst = result.rmst("always", RMST_HORIZON, versus=versus)
        return _wald_rows(spec, replicate, truth, rmst.psi, rmst.plugin_std_error)
    failed = result.bootstrap.n_failed
    if label == "boot_rmst_survival":
        # The shipped RMST carries the bootstrap its replicate risks imply.
        return _bootstrap_rows(spec, replicate, truth, result.rmst("always", RMST_HORIZON), failed)
    names = {
        "boot_risk2_survival": "risk_regimen[always @ t=2]",
        "boot_risk4_survival": a,
        "boot_ate_point_tmle": "ate",
    }
    name = names.get(label) or (
        "ey_regimen[always]" if "_ey_" in label else "ate_regimen[always vs never]"
    )
    return _bootstrap_rows(spec, replicate, truth, result[name], failed)


def _measure(group: str, replicate: int) -> list[dict[str, Any]]:
    """Every row of ``group`` at ``replicate``."""
    specs = GROUPS[group]
    result = _fit(group, specs[0], replicate)
    return [row for spec in specs for row in _rows(spec, result, replicate)]


def _fit_replication(payload: tuple[str, int]) -> list[dict[str, Any]]:
    group, replicate = payload
    return _measure(group, replicate)


def payloads(replicates: int | None = None) -> list[tuple[tuple[str, int]]]:
    """Every (group, replicate) pair, optionally capped for a smoke pass."""
    return [
        ((group, replicate),)
        for group, specs in GROUPS.items()
        for replicate in range(specs[0]["replicates"] if replicates is None else replicates)
    ]


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.loc[:, [*REPLICATE_COLUMNS, *BOOTSTRAP_COLUMNS]].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared calibration verdicts, plus the failed-replicate cap.

    A bootstrap cell whose mean failed share exceeds
    :data:`~tests.studies.canonical_full_refit_bootstrap.FAILED_REPLICATE_CAP` publishes
    ``bootstrap_conditional`` and fails, because its coverage is conditional on the
    replicates that ran.
    """
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=SUMMARY_COLUMNS, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    summary["bootstrap_conditional"] = summary["bootstrap_conditional"].astype(object)
    for index in summary.index:
        group = rows.loc[
            (rows["property"] == summary.loc[index, "property"])
            & (rows["cell"] == summary.loc[index, "cell"])
        ]
        requested = group.get("bootstrap_requested")
        if requested is None or not (requested > 0).any():
            summary.loc[index, "bootstrap_conditional"] = False
            continue
        share = float((group["bootstrap_failed"] / group["bootstrap_requested"]).mean())
        conditional = share > FAILED_REPLICATE_CAP
        summary.loc[index, "bootstrap_failed_fraction"] = share
        summary.loc[index, "bootstrap_conditional"] = conditional
        if conditional and summary.loc[index, "role"] == "positive":
            summary.loc[index, "passed"] = False
    return finish(summary, rates)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "MarkovCellMeans",
    "fit_grid",
    "generate_property_rows",
    "payloads",
    "sample_clustered",
    "summarize_properties",
]
