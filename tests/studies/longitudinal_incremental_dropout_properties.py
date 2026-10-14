r"""Property families of ``longitudinal-incremental-dropout`` (S2).

Every fit is :class:`~cleverly.longitudinal.IPSI` with five folds and saturated cell means at
every slot, with 3,000 replications, on a law of
:mod:`tests.discrete_law_longitudinal_incremental`.  The size is n = 16,000, except 48,000 on
the law ``D`` band draws and 96,000 on law ``Dc``.  ``x2`` is the plan that doubles the odds
of treatment at every node, and ``contrast_x2`` is its contrast with the natural course.

============================  =================================================================
family                        cells
============================  =================================================================
``interval_calibration``      seven labels (below), each with the two derived controls
``simultaneous_coverage``     the band over the risks of three plans at horizons 1 to 3 and
                              their six contrasts on law ``D``, and its pointwise control
``clustered_inference``       ``ey_regimen[odds x2]`` on law ``Ck`` in 800 clusters of 10,
                              3,000 replications: the cluster-robust standard error, and the
                              standard error that treats each unit as independent (control)
``censoring_complete_case``   ``ey_regimen[odds x2]`` on law ``C`` fitted to the rows that never
                              drop out, without a censoring model (reported, no verdict)
============================  =================================================================

The calibration labels:

==============================  ==========================================================
label                           law and estimand
==============================  ==========================================================
``dropout_contrast_x2``         ``C``, ``contrast_x2`` at the end of study
``survival_contrast_x2_h3``     ``D``, ``contrast_x2`` of the risk by horizon 3, read off
                                the band fit
``dropout_contrast_x2_in_sample``  ``C``, ``contrast_x2``, one fold (Kennedy Algorithm 2)
``survival_contrast_x2_h3_in_sample``  ``D``, ``contrast_x2`` of the risk by horizon 3, one
                                fold, on the band draws
``competing_contrast_x2_h3``    ``Dc``, ``contrast_x2`` of the cause-1 incidence by
                                horizon 3, at n = 96,000
``msm_log_delta``               ``C``, the slope of a working model linear in
                                :math:`\log\delta` over the three plans (identity link)
``weighted_contrast_x0_5``      ``C`` with the informative weight :func:`weight`,
                                ``odds x0.5 vs natural course``; the truth is the weighted
                                law's
==============================  ==========================================================

``dropout_contrast_x2``, its one-fold fit, ``msm_log_delta``, ``weighted_contrast_x0_5`` and
the complete-case fit read one draw of law ``C`` each replication.

**Rules.**  ``interval_calibration`` and ``simultaneous_coverage`` read the shared verdicts,
with :data:`EFFICIENCY_RATIO_BAND` around the exact influence-function standard deviation.
``clustered_inference`` reads the shared paired rule: the cluster-robust arm is calibrated,
the independent-unit arm's SE-ratio interval lies below 0.80, and the pair's coverage gain is
at least 0.03.  The design test computes the independent-unit arm's exact SE ratio, 0.60.
``censoring_complete_case`` publishes its bias beside the exact complete-case limit of
:func:`complete_case_limit` and states no verdict.  The one-fold cells are gated like the
cross-fitted ones.  Kennedy (2019), Algorithm 2 (Appendix 8.3), covers a one-fold fit when
the nuisance class is Donsker, and saturated cell means on a finite history set are.  Kim et
al. (2021), Theorem 5.1, states the dropout result with sample splitting; the same Donsker
argument carries it to one fold.

A replication that raises is not redrawn.  :func:`failure_probe` counts failed fits per fit
set and records no estimate.  A raise in the declared run stops it; the failing cell then drops
to its red-cell owner with its failure count published, and the run repeats without it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.msm import MSM
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal_incremental as law
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_incremental_common as common
from tests.studies.canonical_longitudinal_incremental_dropout import (
    HORIZONS,
    PLANS,
    REFERENCE,
    STUDY,
)
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    clustered_inference_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells

N_FOLDS = 5
N = 16_000
#: A calibration cell whose SE ratio and coverage equal their targets passes with probability
#: 0.935 at 2,000 replications and 0.996 at 3,000.  Eight cells read the rule, so 3,000.
REPLICATES = 3_000
#: The cluster-robust arm reads the calibration rule, which cannot pass at 1,000 replications
#: (the 99% intervals are wider than the bands).  ``tests/diagnostics/x19_sizing`` reads the
#: pair at this size.
CLUSTER_N = 8_000
CLUSTER_REPLICATES = 3_000
#: The competing-risk cell needs a larger size.  At n = 8,000 a training fold can hold one
#: or two treated units of a rare history, and the saturated censoring fit then reaches its
#: lower bound.  That tail is gone at 32,000, where the SE ratio reads 0.948, and at 64,000 it
#: reads 0.980: a deficit that falls as 1/n.  Projected to 96,000, the cell passes with
#: probability 0.97 and S2's calibration cells pass jointly with at least 0.90
#: (``tests/diagnostics/x19_sizing``, ``tests/unit/test_longitudinal_incremental_design.py``).
COMPETING_N = 96_000
#: The band draws of law ``D``.  They also carry the two survival calibration labels, which
#: read an SE ratio of 0.98 at n = 16,000.  At 3,000 replications each passes the calibration
#: rule with probability 0.94 at 16,000 and 0.99 here.  The joint rule sets the size: with
#: the band at 16,000, S2's calibration cells pass together with probability about 0.85
#: (``tests/unit/test_longitudinal_incremental_design.py``).
BAND_N = 48_000
CLUSTER_SIZE = 10
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

X2 = "odds x2"
X0_5 = "odds x0.5"
MEAN_X2 = f"ey_regimen[{X2}]"
CONTRAST_X2 = f"ate_regimen[{X2} vs {REFERENCE}]"
CONTRAST_X0_5 = f"ate_regimen[{X0_5} vs {REFERENCE}]"
SURVIVAL_X2 = f"ate_regimen[{X2} vs {REFERENCE} @ t=3]"
COMPETING_X2 = f"ate_regimen[{X2} vs {REFERENCE}, event @ t=3]"
SLOPE = "msm_regimen[log_delta]"
LOG_DELTA = {label: float(np.log(delta)) for label, delta in law.DELTAS.items()}
#: The projection of the three plan means onto ``(1, log delta)``: row ``j`` holds the plan
#: weights of coefficient ``j``.
PROJECTION = np.linalg.pinv(np.array([[1.0, LOG_DELTA[label]] for label in PLANS]))
SLOPE_WEIGHTS = dict(zip(PLANS, PROJECTION[1], strict=True))

BAND_LABEL = "survival_grid"
BAND_NAMES = common.names(PLANS, REFERENCE, horizons=HORIZONS)


def weight(values: Mapping[str, float]) -> float:
    """The informative observation weight: it reads the first arm and the baseline.

    Both are observed on every row.  The weight moves the first-node mechanism (the M18
    witness) and the outcome distribution, so the weighted truth differs from the unweighted
    one by 5.5 standard errors at n = 16,000 (design test).
    """
    return 1.0 + values["A1"] + 0.5 * values["W"]


def cell_weights(target: Any, function: Callable[[Mapping[str, float]], float]) -> np.ndarray:
    """``function`` on every cell of ``target``, in the law's shape."""
    names = [axis.name for axis in target.axes]
    out = np.zeros(target.shape)
    for cell in target.cells:
        out[cell] = function(dict(zip(names, cell, strict=True)))
    return out


WEIGHTS = cell_weights(law.LAW_C, weight)


def complete_case_limit(target: Any, plan: common.Plan) -> float:
    """The exact limit of a fit to the rows that never drop out, with no censoring model.

    The g-formula of ``plan`` evaluated on the law conditioned on staying to the end.
    """
    stay = cell_weights(
        target,
        lambda values: float(
            all(values[axis.name] == 1 for axis in target.axes if axis.kind == "censoring")
        ),
    )
    return law.truth(target, plan, weights=stay)


def _working_model(label: str, horizon: int, frame: pd.DataFrame) -> np.ndarray:
    del horizon
    return np.tile([1.0, LOG_DELTA[label]], (len(frame), 1))


MSM_LOG_DELTA = MSM(
    design=_working_model, terms=("1", "log_delta"), link="identity", design_kind="known"
)

CALIBRATION_LABELS = (
    "dropout_contrast_x2",
    "survival_contrast_x2_h3",
    "dropout_contrast_x2_in_sample",
    "survival_contrast_x2_h3_in_sample",
    "competing_contrast_x2_h3",
    "msm_log_delta",
    "weighted_contrast_x0_5",
)
#: ``label -> (draw stream, estimand)``.
CALIBRATION = {
    "dropout_contrast_x2": ("law_c", CONTRAST_X2),
    "survival_contrast_x2_h3": (BAND_LABEL, SURVIVAL_X2),
    "dropout_contrast_x2_in_sample": ("law_c", CONTRAST_X2),
    "survival_contrast_x2_h3_in_sample": (BAND_LABEL, SURVIVAL_X2),
    "competing_contrast_x2_h3": ("law_dc", COMPETING_X2),
    "msm_log_delta": ("law_c", SLOPE),
    "weighted_contrast_x0_5": ("law_c", CONTRAST_X0_5),
}
TRUTH = {
    "dropout_contrast_x2": common.truth(law.LAW_C, PLANS, CONTRAST_X2),
    "survival_contrast_x2_h3": common.truth(law.LAW_D, PLANS, SURVIVAL_X2),
    "competing_contrast_x2_h3": common.truth(law.LAW_DC, PLANS, COMPETING_X2),
    "msm_log_delta": float(
        sum(weight * law.truth(law.LAW_C, PLANS[label]) for label, weight in SLOPE_WEIGHTS.items())
    ),
    "weighted_contrast_x0_5": common.truth(law.LAW_C, PLANS, CONTRAST_X0_5, weights=WEIGHTS),
}
EFFICIENCY_SD = {
    "dropout_contrast_x2": common.eif_sd(law.LAW_C, PLANS, CONTRAST_X2),
    "survival_contrast_x2_h3": common.eif_sd(law.LAW_D, PLANS, SURVIVAL_X2),
    "competing_contrast_x2_h3": common.eif_sd(law.LAW_DC, PLANS, COMPETING_X2),
    "msm_log_delta": common.combined_sd(law.LAW_C, PLANS, SLOPE_WEIGHTS),
    "weighted_contrast_x0_5": common.eif_sd(law.LAW_C, PLANS, CONTRAST_X0_5, weights=WEIGHTS),
}
for _label in ("dropout_contrast_x2", "survival_contrast_x2_h3"):
    TRUTH[f"{_label}_in_sample"] = TRUTH[_label]
    EFFICIENCY_SD[f"{_label}_in_sample"] = EFFICIENCY_SD[_label]
BAND_TRUTH = common.truths(law.LAW_D, PLANS, BAND_NAMES)
CLUSTER_TRUTH = common.truth(law.LAW_CK.marginal, PLANS, MEAN_X2)
COMPLETE_CASE_TRUTH = common.truth(law.LAW_C, PLANS, MEAN_X2)
COMPLETE_CASE_LIMIT = complete_case_limit(law.LAW_C, PLANS[X2])


# ---------------------------------------------------------------------------- fits


def _fit(frame: pd.DataFrame, target: Any, **settings: Any) -> Any:
    settings.setdefault("n_folds", N_FOLDS)
    settings.setdefault("reference", REFERENCE)
    return common.fit(frame, target, PLANS, **settings)


def complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows that never drop out, without their censoring columns."""
    stay = np.ones(len(frame), dtype=bool)
    for t in range(1, law.LAW_C.n_times + 1):
        stay &= frame[f"C{t}"].to_numpy() == 1
    kept = frame.loc[stay].drop(columns=[f"C{t}" for t in range(1, law.LAW_C.n_times + 1)])
    return kept.reset_index(drop=True)


def iid_standard_error(estimate: Any) -> float:
    """The standard error that treats every unit as independent, off the unit curve."""
    curve = np.asarray(estimate.influence_curve, dtype=float)
    return float(np.std(curve, ddof=1) / np.sqrt(len(curve)))


# ---------------------------------------------------------------------------- the grid


def _size(label: str) -> int:
    """The declared size of a calibration label: :data:`N` unless its stream declares one."""
    stream = CALIBRATION[label][0]
    return {"law_dc": COMPETING_N, BAND_LABEL: BAND_N}.get(stream, N)


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by its draw stream, with its exact truths.

    Parameters
    ----------
    name : str
        A draw stream: ``"law_c"``, ``"law_dc"``, ``"law_ck"`` or ``"survival_grid"``.
    """

    name: str

    def truth(self) -> dict[str, float]:
        if self.name == "law_ck":
            return {MEAN_X2: CLUSTER_TRUTH}
        labels = [label for label, (stream, _) in CALIBRATION.items() if stream == self.name]
        out = {CALIBRATION[label][1]: TRUTH[label] for label in labels}
        if self.name == "law_c":
            out[MEAN_X2] = COMPLETE_CASE_TRUTH
        return out


#: Every fit set: ``(family, draw stream, n, replicates)``.
FIT_SETS: tuple[tuple[str, str, int, int], ...] = (
    ("interval_calibration", "law_c", N, REPLICATES),
    ("interval_calibration", "law_dc", COMPETING_N, REPLICATES),
    ("clustered_inference", "law_ck", CLUSTER_N, CLUSTER_REPLICATES),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream.

    The two calibration controls of each label are derived from its rows, so they are not
    here.
    """
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        stream: str,
        estimand: str,
        seed: int,
        *,
        n: int = N,
        replicates: int = REPLICATES,
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(stream),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=replicates,
                seed=seed,
                role=role,
                estimand=estimand,
            )
        )

    for label in CALIBRATION_LABELS:
        stream, estimand = CALIBRATION[label]
        family = JOINT if stream == BAND_LABEL else "interval_calibration"
        add(
            "interval_calibration",
            f"{label}__correctly_specified",
            "positive",
            stream,
            estimand,
            _seed(family, stream, 0),
            n=_size(label),
        )
    for cell, role in (
        ("mean_x2__cluster_robust", "positive"),
        ("mean_x2__iid_control", "control"),
    ):
        add(
            "clustered_inference",
            cell,
            role,
            "law_ck",
            MEAN_X2,
            _seed("clustered_inference", "law_ck", 0),
            n=CLUSTER_N,
            replicates=CLUSTER_REPLICATES,
        )
    add(
        "censoring_complete_case",
        "mean_x2__complete_case",
        DIAGNOSTIC_ROLE,
        "law_c",
        MEAN_X2,
        _seed("interval_calibration", "law_c", 0),
    )
    cells.extend(
        joint_property_cells(
            BAND_LABEL, n=BAND_N, replicates=REPLICATES, seed=_seed(JOINT, BAND_LABEL, 0)
        )
    )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _fit_set_rows(payload: tuple[str, str, int, int, int]) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes."""
    family, stream, replicate, n, requested = payload
    seed = _seed(family, stream, replicate)
    common_row = {"replicate": replicate, "n": n, "requested": requested}

    def row(
        cell: str, role: str, truth: float, estimate: Any, property_name: str
    ) -> dict[str, Any]:
        return replicate_row(
            property_name=property_name,
            cell=cell,
            role=role,
            truth=truth,
            estimate=estimate,
            alpha=STUDY.margins.alpha,
            **common_row,
        )

    if stream == "law_dc":
        frame = law.sample(law.LAW_DC, n, seed)
        result = _fit(frame, law.LAW_DC, horizons=[3])
        label = "competing_contrast_x2_h3"
        return [
            row(
                f"{label}__correctly_specified",
                "positive",
                TRUTH[label],
                result[COMPETING_X2],
                family,
            )
        ]
    if stream == "law_ck":
        frame = law.sample_clustered(n, CLUSTER_SIZE, seed)
        estimate = _fit(frame, law.LAW_CK.marginal, id="id")[MEAN_X2]
        return [
            row("mean_x2__cluster_robust", "positive", CLUSTER_TRUTH, estimate, family),
            control_row(
                property_name=family,
                cell="mean_x2__iid_control",
                truth=CLUSTER_TRUTH,
                estimate=float(estimate.psi),
                standard_error=iid_standard_error(estimate),
                critical=CRITICAL,
                **common_row,
            ),
        ]
    frame = law.sample(law.LAW_C, n, seed)
    main = _fit(frame, law.LAW_C)
    slope = _fit(frame, law.LAW_C, reference=None, msm=MSM_LOG_DELTA)
    weighted = _fit(frame.assign(weight=weight(frame)), law.LAW_C, weights="weight")
    complete = _fit(complete_cases(frame), law.LAW_A)
    one_fold = _fit(frame, law.LAW_C, n_folds=1)
    return [
        row(
            "dropout_contrast_x2__correctly_specified",
            "positive",
            TRUTH["dropout_contrast_x2"],
            main[CONTRAST_X2],
            family,
        ),
        row(
            "msm_log_delta__correctly_specified",
            "positive",
            TRUTH["msm_log_delta"],
            slope[SLOPE],
            family,
        ),
        row(
            "weighted_contrast_x0_5__correctly_specified",
            "positive",
            TRUTH["weighted_contrast_x0_5"],
            weighted[CONTRAST_X0_5],
            family,
        ),
        row(
            "mean_x2__complete_case",
            DIAGNOSTIC_ROLE,
            COMPLETE_CASE_TRUTH,
            complete[MEAN_X2],
            "censoring_complete_case",
        ),
        row(
            "dropout_contrast_x2_in_sample__correctly_specified",
            "positive",
            TRUTH["dropout_contrast_x2_in_sample"],
            one_fold[CONTRAST_X2],
            family,
        ),
    ]


def _band_rows(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, requested = payload
    frame = law.sample(law.LAW_D, BAND_N, _seed(JOINT, BAND_LABEL, replicate))
    result = _fit(frame, law.LAW_D, horizons=list(HORIZONS), simultaneous=True)
    one_fold = _fit(frame, law.LAW_D, horizons=[3], n_folds=1)
    label = "survival_contrast_x2_h3"
    return [
        *joint_coverage_rows(
            result,
            BAND_TRUTH,
            BAND_NAMES,
            label=BAND_LABEL,
            replicate=replicate,
            n=BAND_N,
            requested=requested,
            pointwise_critical=CRITICAL,
        ),
        replicate_row(
            property_name="interval_calibration",
            cell=f"{label}__correctly_specified",
            role="positive",
            truth=TRUTH[label],
            estimate=result[SURVIVAL_X2],
            alpha=STUDY.margins.alpha,
            replicate=replicate,
            n=BAND_N,
            requested=requested,
        ),
        replicate_row(
            property_name="interval_calibration",
            cell=f"{label}_in_sample__correctly_specified",
            role="positive",
            truth=TRUTH[f"{label}_in_sample"],
            estimate=one_fold[SURVIVAL_X2],
            alpha=STUDY.margins.alpha,
            replicate=replicate,
            n=BAND_N,
            requested=requested,
        ),
    ]


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for family, stream, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(((_fit_set_rows, (family, stream, r, n, requested)),) for r in range(requested))
    band = REPLICATES if budget is None else budget
    out.extend(((_band_rows, (r, band)),) for r in range(band))
    return out


def _run(job: tuple[Any, tuple[Any, ...]]) -> list[dict[str, Any]]:
    function, payload = job
    return list(function(payload))


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = map_parallel(_run, _payloads(budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            # One call per size: the noise control adds the efficient spread at its cell's n.
            *(
                calibration_controls(
                    rows,
                    STUDY,
                    labels=[label for label in CALIBRATION_LABELS if _size(label) == size],
                    efficiency_bounds=EFFICIENCY_SD,
                    calibration_n=size,
                    shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                    critical=CRITICAL,
                )
                for size in sorted({_size(label) for label in CALIBRATION_LABELS})
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def failure_probe(draws: int, *, n_jobs: int = 1, start: int = 0) -> dict[str, int]:
    """Failed fits per fit set over streams ``start`` to ``start + draws - 1``.

    It records no estimate.  A fit set with any failure is dropped before the declared run.
    """
    jobs = [
        (_fit_set_rows, (family, stream, r, n, draws))
        for family, stream, n, _ in FIT_SETS
        for r in range(start, start + draws)
    ]
    jobs += [(_band_rows, (r, draws)) for r in range(start, start + draws)]
    outcomes = map_parallel(_probe_one, [(job,) for job in jobs], n_jobs=n_jobs)
    counts = {f"{family}/{stream}": 0 for family, stream, *_ in FIT_SETS}
    counts[f"{JOINT}/{BAND_LABEL}"] = 0
    for (function, payload), failed in zip(jobs, outcomes, strict=True):
        key = f"{JOINT}/{BAND_LABEL}" if function is _band_rows else f"{payload[0]}/{payload[1]}"
        counts[key] += int(failed)
    return counts


def _probe_one(job: tuple[Any, tuple[Any, ...]]) -> bool:
    try:
        rows = _run(job)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


# ---------------------------------------------------------------------------- verdicts

#: The column that publishes a cell's exact-law limit of its mean estimate.
LIMIT_COLUMN = "exact_limit"


def reported_families(summary: pd.DataFrame) -> None:
    """The two reported families state no verdict; the complete case publishes its limit."""
    complete = summary["property"] == "censoring_complete_case"
    summary.loc[complete, LIMIT_COLUMN] = COMPLETE_CASE_LIMIT
    summary.loc[complete, "passed"] = True
    summary.loc[complete, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
            LIMIT_COLUMN,
        ),
        rate_labels=(),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    clustered_inference_verdicts(
        summary,
        rows,
        STUDY,
        positive_cell="mean_x2__cluster_robust",
        control_cell="mean_x2__iid_control",
    )
    reported_families(summary)
    return finish(summary, rates)


__all__ = [
    "FIT_SETS",
    "declared_cells",
    "failure_probe",
    "generate_property_rows",
    "summarize_properties",
]
