r"""Property families of ``longitudinal-incremental`` (S1).

Every fit is :class:`~cleverly.longitudinal.IPSI` with five folds and saturated cell means at
every slot, on law ``A`` of :mod:`tests.discrete_law_longitudinal_incremental`, unless the table
names another setting.  ``x2`` is the plan that doubles the odds of treatment at every node, and
``contrast_x2`` is ``ate_regimen[odds x2 vs natural course]``.

==============================  ===============================================================
family                          cells, replications and size
==============================  ===============================================================
``interval_calibration``        seven labels (below); 2,000 at n = 4,000 each, with the two
                                derived controls of each
``root_n_and_efficiency``       ``contrast_x2`` at n = 1,000 (control: the smallest rung),
                                4,000 and 16,000; 700 each
``root_n_rate``                 the two rate rows of ``contrast_x2``, from the ladder
``type_i_error``                ``contrast_x2`` on law ``A0`` (every tilt has one mean); 1,000
                                at n = 4,000
``power``                       ``odds x1.5 vs natural course`` on law ``Aw``; 1,000 at
                                n = 4,000
``simultaneous_coverage``       the band over nine tilts on ``geomspace(0.25, 4, 9)`` (nine
                                means and eight contrasts) and its pointwise control; 2,000 at
                                n = 5,000
``mechanism_term_necessity``    the ``contrast_x2`` calibration fits: the full curve, and the
                                curve without the mechanism term (control)
``single_robustness``           the regressions constant and the mechanism right at n = 4,000
                                and 16,000 (positive, two-size rule); the mechanism constant
                                and the regressions right at n = 4,000 (no verdict); 1,000 each
``targeting_necessity``         the n = 16,000 draws of ``single_robustness``: the one-step
                                estimate, and the plug-in mean of the first-node regression
                                (control)
==============================  ===============================================================

The calibration labels:

=========================  ======================================================  ============
label                      law, plan and estimand                                  bound
=========================  ======================================================  ============
``mean_x2``                ``A``, ``ey_regimen[odds x2]``                          EIF
``contrast_x2``            ``A``, ``contrast_x2``                                  EIF
``contrast_x2_in_sample``  ``A``, ``contrast_x2``, one fold (Kennedy Algorithm 2)  EIF
``known_x2``               ``A``, ``ey_regimen[odds x2]``, the law's own           known-π EIF
                           :math:`\pi_t` declared (no mechanism term)
``no_positivity``          ``Ax`` (exact 0 and 1 propensities, and 0.02 cells),    EIF
                           ``contrast_x2`` over two nodes
``per_node``               ``A``, the plan :math:`(\delta_1, \delta_2, \delta_3)    EIF
                           = (2, 0.5, 1.5)`, its mean
``tilt_then_static``       ``A``, tilts by 2 at nodes 1 and 2, then arm 1 at node  EIF
                           3, its mean
=========================  ======================================================  ============

``mean_x2``, ``contrast_x2``, ``contrast_x2_in_sample`` and ``known_x2`` read one draw each
replication, fitted three ways.  ``per_node`` and ``tilt_then_static`` share one fit.

**Rules.**  ``interval_calibration``, ``root_n_and_efficiency``, ``root_n_rate``,
``type_i_error``, ``power`` and ``simultaneous_coverage`` read the shared verdicts.
``mechanism_term_necessity`` passes when the full curve's SE-ratio interval lies inside
:data:`EFFICIENCY_RATIO_BAND` and the control's lies wholly below the calibration band: the
design test computes the control's exact SD ratio, 0.78.  ``single_robustness``'s positive
pair passes when :math:`|\bar b| + 2\,\mathrm{MCSE}` at n = 16,000 is below
:data:`ABSOLUTE_BIAS_TOLERANCE` and below the same bound at n = 4,000 (the ``X4-plan.md`` 6.8
rule).  Its diagnostic cell publishes its bias beside the exact-law limit of
:func:`~tests.studies.longitudinal_incremental_common.constant_mechanism_limit`.
``targeting_necessity`` reads the shared necessity rule at :data:`TARGETING_DISPLACEMENT`.

A replication that raises is not redrawn.  :func:`failure_probe` counts failed fits per fit
set and records no estimate.  A raise in the declared run stops it; the failing cell then drops
to its red-cell owner with its failure count published, and the run repeats without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal_incremental as law
from tests.parallel import STUDY_JOBS
from tests.studies import longitudinal_incremental_common as common
from tests.studies.canonical_longitudinal_incremental import PLANS, REFERENCE, STUDY
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    ratio_intervals,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells

N_FOLDS = 5
CALIBRATION_REPLICATES = 2_000
CALIBRATION_N = 4_000
RATE_REPLICATES = 700
RATE_SIZES = (1_000, 4_000, 16_000)
NULL_REPLICATES = 1_000
NULL_N = 4_000
BAND_REPLICATES = 2_000
BAND_N = 5_000
REQUIREMENT_REPLICATES = 1_000
REQUIREMENT_SIZES = (4_000, 16_000)
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
#: The absolute bias bound of the positive ``single_robustness`` pair at n = 16,000.
ABSOLUTE_BIAS_TOLERANCE = 0.01
TARGETING_DISPLACEMENT = 0.25
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

X2 = "odds x2"
NATURAL = REFERENCE
MEAN_X2 = f"ey_regimen[{X2}]"
CONTRAST_X2 = f"ate_regimen[{X2} vs {NATURAL}]"
POWER_PLANS = common.constant_plans({NATURAL: 1.0, "odds x1.5": 1.5})
POWER_NAME = f"ate_regimen[odds x1.5 vs {NATURAL}]"
AX_PLANS = common.constant_plans(law.DELTAS, n_times=2)
MIXED_PLANS: dict[str, common.Plan] = {
    "per_node": (law.tilt(2.0), law.tilt(0.5), law.tilt(1.5)),
    "tilt_then_static": (law.tilt(2.0), law.tilt(2.0), law.arm(1)),
}
BAND_DELTAS = {f"odds x{delta:.3g}": float(delta) for delta in np.geomspace(0.25, 4.0, 9)}
BAND_REFERENCE = "odds x1"
BAND_PLANS = common.constant_plans(BAND_DELTAS)
BAND_LABEL = "nine_tilts"
BAND_NAMES = common.names(BAND_PLANS, BAND_REFERENCE)

CALIBRATION_LABELS = (
    "mean_x2",
    "contrast_x2",
    "contrast_x2_in_sample",
    "known_x2",
    "no_positivity",
    "per_node",
    "tilt_then_static",
)
#: ``(law, plans, estimand, known nodes)`` of each calibration label.
CALIBRATION: dict[str, tuple[Any, dict[str, common.Plan], str, tuple[int, ...]]] = {
    "mean_x2": (law.LAW_A, PLANS, MEAN_X2, ()),
    "contrast_x2": (law.LAW_A, PLANS, CONTRAST_X2, ()),
    "contrast_x2_in_sample": (law.LAW_A, PLANS, CONTRAST_X2, ()),
    "known_x2": (law.LAW_A, PLANS, MEAN_X2, (1, 2, 3)),
    "no_positivity": (law.LAW_AX, AX_PLANS, CONTRAST_X2, ()),
    "per_node": (law.LAW_A, MIXED_PLANS, "ey_regimen[per_node]", ()),
    "tilt_then_static": (law.LAW_A, MIXED_PLANS, "ey_regimen[tilt_then_static]", ()),
}
#: The draw stream each calibration label reads.
CALIBRATION_STREAM = {
    "mean_x2": "law_a",
    "contrast_x2": "law_a",
    "contrast_x2_in_sample": "law_a",
    "known_x2": "law_a",
    "no_positivity": "law_ax",
    "per_node": "law_a_mixed",
    "tilt_then_static": "law_a_mixed",
}
TRUTH = {
    label: common.truth(target, plans, name, known=known)
    for label, (target, plans, name, known) in CALIBRATION.items()
}
EFFICIENCY_SD = {
    label: common.eif_sd(target, plans, name, known=known)
    for label, (target, plans, name, known) in CALIBRATION.items()
}
NULL_TRUTH = common.truth(law.LAW_A0, PLANS, CONTRAST_X2)
POWER_TRUTH = common.truth(law.LAW_AW, POWER_PLANS, POWER_NAME)
CONTRAST_TRUTH = TRUTH["contrast_x2"]
BAND_TRUTH = common.truths(law.LAW_A, BAND_PLANS, BAND_NAMES)


# ---------------------------------------------------------------------------- fits


def _fit(frame: pd.DataFrame, target: Any, plans: Any, **settings: Any) -> Any:
    settings.setdefault("n_folds", N_FOLDS)
    settings.setdefault("reference", REFERENCE if REFERENCE in plans else None)
    return common.fit(frame, target, plans, **settings)


def mechanism_free_se(result: Any, name: str = CONTRAST_X2) -> float:
    """The standard error of ``name`` from its curve without the mechanism term."""
    left, right, _, _ = common.parse(name)
    curve = np.asarray(result.influence_curves[name], dtype=float)
    term = result.fits[left].mechanism_term.sum(axis=1)
    if right is not None:
        term = term - result.fits[right].mechanism_term.sum(axis=1)
    reduced = curve - (term - np.mean(term))
    return float(np.std(reduced, ddof=1) / np.sqrt(len(reduced)))


# ---------------------------------------------------------------------------- the grid


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        A draw stream: ``"law_a"``, ``"law_ax"``, ``"law_a_mixed"``, ``"rate"``,
        ``"null"``, ``"power"``, ``"requirement"`` or ``"diagnostic"``.
    """

    name: str

    def truth(self) -> dict[str, float]:
        if self.name in {"law_a", "law_ax", "law_a_mixed"}:
            labels = [key for key, stream in CALIBRATION_STREAM.items() if stream == self.name]
            return {CALIBRATION[key][2]: TRUTH[key] for key in labels}
        if self.name == "null":
            return {CONTRAST_X2: NULL_TRUTH}
        if self.name == "power":
            return {POWER_NAME: POWER_TRUTH}
        if self.name == "diagnostic":
            return {CONTRAST_X2: CONTRAST_TRUTH}
        return {CONTRAST_X2: CONTRAST_TRUTH}


#: Every fit set: ``(family, stream label, configuration, n, replicates, law)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    (
        "interval_calibration",
        "law_a",
        "calibration",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "law_a",
    ),
    (
        "interval_calibration",
        "law_ax",
        "calibration",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "law_ax",
    ),
    (
        "interval_calibration",
        "law_a_mixed",
        "calibration",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        "law_a_mixed",
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "rate", size, RATE_REPLICATES, "rate")
        for size in RATE_SIZES
    ),
    ("type_i_error", "sharp_null", "null", NULL_N, NULL_REPLICATES, "null"),
    ("power", "alternative", "power", NULL_N, NULL_REPLICATES, "power"),
    *(
        (
            "single_robustness",
            f"outcome_constant__n_{size}",
            "outcome_constant",
            size,
            REQUIREMENT_REPLICATES,
            "requirement",
        )
        for size in REQUIREMENT_SIZES
    ),
    (
        "single_robustness",
        "mechanism_constant",
        "mechanism_constant",
        REQUIREMENT_SIZES[0],
        REQUIREMENT_REPLICATES,
        "diagnostic",
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream.

    The ``root_n_rate`` rows are fitted from the ladder's rows rather than sampled, and the
    two calibration controls of each label are derived from its rows, so neither is here.
    """
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        law_name: str,
        n: int,
        replicates: int,
        stream: str,
        estimand: str,
        *,
        stream_family: str | None = None,
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(law_name),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=replicates,
                seed=_seed(stream_family or family, stream, 0),
                role=role,
                estimand=estimand,
            )
        )

    for label in CALIBRATION_LABELS:
        stream = CALIBRATION_STREAM[label]
        add(
            "interval_calibration",
            f"{label}__correctly_specified",
            "positive",
            stream,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            stream,
            CALIBRATION[label][2],
        )
    for size in RATE_SIZES:
        add(
            "root_n_and_efficiency",
            f"contrast_x2__n_{size}",
            "control" if size == min(RATE_SIZES) else "positive",
            "rate",
            size,
            RATE_REPLICATES,
            f"n_{size}",
            CONTRAST_X2,
        )
    add(
        "type_i_error",
        "contrast_x2__sharp_null",
        "positive",
        "null",
        NULL_N,
        NULL_REPLICATES,
        "sharp_null",
        CONTRAST_X2,
    )
    add(
        "power",
        "contrast_x1_5__alternative",
        "positive",
        "power",
        NULL_N,
        NULL_REPLICATES,
        "alternative",
        POWER_NAME,
    )
    for cell, role in (
        ("contrast_x2__full_eif", "positive"),
        ("contrast_x2__mechanism_term_dropped", "control"),
    ):
        add(
            "mechanism_term_necessity",
            cell,
            role,
            "law_a",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            "law_a",
            CONTRAST_X2,
            stream_family="interval_calibration",
        )
    for size in REQUIREMENT_SIZES:
        add(
            "single_robustness",
            f"contrast_x2__mechanism_correct__n_{size}",
            "positive",
            "requirement",
            size,
            REQUIREMENT_REPLICATES,
            f"outcome_constant__n_{size}",
            CONTRAST_X2,
        )
    add(
        "single_robustness",
        "contrast_x2__mechanism_constant",
        DIAGNOSTIC_ROLE,
        "diagnostic",
        REQUIREMENT_SIZES[0],
        REQUIREMENT_REPLICATES,
        "mechanism_constant",
        CONTRAST_X2,
    )
    for cell, role in (("contrast_x2__one_step", "positive"), ("contrast_x2__plug_in", "control")):
        add(
            "targeting_necessity",
            cell,
            role,
            "requirement",
            REQUIREMENT_SIZES[1],
            REQUIREMENT_REPLICATES,
            f"outcome_constant__n_{REQUIREMENT_SIZES[1]}",
            CONTRAST_X2,
            stream_family="single_robustness",
        )
    cells.extend(
        joint_property_cells(
            BAND_LABEL, n=BAND_N, replicates=BAND_REPLICATES, seed=_seed(JOINT, BAND_LABEL, 0)
        )
    )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _calibration_rows(
    law_name: str, frame: pd.DataFrame, row: Any, common_row: dict[str, Any]
) -> list[dict[str, Any]]:
    if law_name == "law_ax":
        result = _fit(frame, law.LAW_AX, AX_PLANS)
        return [
            row(
                "no_positivity__correctly_specified",
                "positive",
                TRUTH["no_positivity"],
                result[CONTRAST_X2],
            )
        ]
    if law_name == "law_a_mixed":
        result = _fit(frame, law.LAW_A, MIXED_PLANS, reference=None)
        return [
            row(
                f"{label}__correctly_specified",
                "positive",
                TRUTH[label],
                result[CALIBRATION[label][2]],
            )
            for label in ("per_node", "tilt_then_static")
        ]
    result = _fit(frame, law.LAW_A, PLANS)
    in_sample = _fit(frame, law.LAW_A, PLANS, n_folds=1)
    known = _fit(
        frame,
        law.LAW_A,
        PLANS,
        treatment_probabilities=common.known_treatment(frame, law.LAW_A),
    )
    contrast = result[CONTRAST_X2]
    return [
        row("mean_x2__correctly_specified", "positive", TRUTH["mean_x2"], result[MEAN_X2]),
        row("contrast_x2__correctly_specified", "positive", TRUTH["contrast_x2"], contrast),
        row(
            "contrast_x2_in_sample__correctly_specified",
            "positive",
            TRUTH["contrast_x2_in_sample"],
            in_sample[CONTRAST_X2],
        ),
        row("known_x2__correctly_specified", "positive", TRUTH["known_x2"], known[MEAN_X2]),
        row(
            "contrast_x2__full_eif",
            "positive",
            CONTRAST_TRUTH,
            contrast,
            "mechanism_term_necessity",
        ),
        control_row(
            property_name="mechanism_term_necessity",
            cell="contrast_x2__mechanism_term_dropped",
            truth=CONTRAST_TRUTH,
            estimate=float(contrast.psi),
            standard_error=mechanism_free_se(result),
            critical=CRITICAL,
            **common_row,
        ),
    ]


def _fit_set_rows(payload: tuple[str, str, str, int, int, int, str]) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes."""
    family, label, configuration, replicate, n, requested, law_name = payload
    seed = _seed(family, label, replicate)
    alpha = STUDY.margins.alpha
    common_row = {"replicate": replicate, "n": n, "requested": requested}

    def row(
        cell: str, role: str, truth: float, estimate: Any, property_name: str = family
    ) -> dict[str, Any]:
        return replicate_row(
            property_name=property_name,
            cell=cell,
            role=role,
            truth=truth,
            estimate=estimate,
            alpha=alpha,
            **common_row,
        )

    if family == "interval_calibration":
        target = law.LAW_AX if law_name == "law_ax" else law.LAW_A
        return _calibration_rows(law_name, law.sample(target, n, seed), row, common_row)
    if family == "root_n_and_efficiency":
        result = _fit(law.sample(law.LAW_A, n, seed), law.LAW_A, PLANS)
        role = "control" if n == min(RATE_SIZES) else "positive"
        return [row(f"contrast_x2__n_{n}", role, CONTRAST_TRUTH, result[CONTRAST_X2])]
    if family == "type_i_error":
        result = _fit(law.sample(law.LAW_A0, n, seed), law.LAW_A0, PLANS)
        return [row("contrast_x2__sharp_null", "positive", NULL_TRUTH, result[CONTRAST_X2])]
    if family == "power":
        result = _fit(law.sample(law.LAW_AW, n, seed), law.LAW_AW, POWER_PLANS)
        return [row("contrast_x1_5__alternative", "positive", POWER_TRUTH, result[POWER_NAME])]
    frame = law.sample(law.LAW_A, n, seed)
    if configuration == "mechanism_constant":
        result = _fit(frame, law.LAW_A, PLANS, treatment_learner=DummyClassifier(strategy="prior"))
        return [
            row(
                "contrast_x2__mechanism_constant",
                DIAGNOSTIC_ROLE,
                CONTRAST_TRUTH,
                result[CONTRAST_X2],
            )
        ]
    result = _fit(frame, law.LAW_A, PLANS, outcome_learner=DummyRegressor())
    estimate = result[CONTRAST_X2]
    rows = [row(f"contrast_x2__mechanism_correct__n_{n}", "positive", CONTRAST_TRUTH, estimate)]
    if n == REQUIREMENT_SIZES[1]:
        rows += [
            row(
                "contrast_x2__one_step", "positive", CONTRAST_TRUTH, estimate, "targeting_necessity"
            ),
            control_row(
                property_name="targeting_necessity",
                cell="contrast_x2__plug_in",
                truth=CONTRAST_TRUTH,
                estimate=common.plug_in(result, CONTRAST_X2),
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
                **common_row,
            ),
        ]
    return rows


def _band_rows(payload: tuple[int, int]) -> list[dict[str, Any]]:
    replicate, requested = payload
    frame = law.sample(law.LAW_A, BAND_N, _seed(JOINT, BAND_LABEL, replicate))
    result = _fit(frame, law.LAW_A, BAND_PLANS, reference=BAND_REFERENCE, simultaneous=True)
    return joint_coverage_rows(
        result,
        BAND_TRUTH,
        BAND_NAMES,
        label=BAND_LABEL,
        replicate=replicate,
        n=BAND_N,
        requested=requested,
        pointwise_critical=CRITICAL,
    )


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for family, label, configuration, n, replicates, law_name in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(
            ((_fit_set_rows, (family, label, configuration, r, n, requested, law_name)),)
            for r in range(requested)
        )
    band = BAND_REPLICATES if budget is None else budget
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
            calibration_controls(
                rows,
                STUDY,
                labels=CALIBRATION_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
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
    payloads = [
        ((family, label, configuration, r, n, draws, law_name),)
        for family, label, configuration, n, _, law_name in FIT_SETS
        for r in range(start, start + draws)
    ]
    outcomes = map_parallel(_probe_one, payloads, n_jobs=n_jobs)
    counts = {f"{family}/{label}": 0 for family, label, *_ in FIT_SETS}
    for (payload,), failed in zip(payloads, outcomes, strict=True):
        counts[f"{payload[0]}/{payload[1]}"] += int(failed)
    return counts


def _probe_one(payload: tuple[str, str, str, int, int, int, str]) -> bool:
    try:
        rows = _fit_set_rows(payload)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


# ---------------------------------------------------------------------------- verdicts

#: The column that publishes :math:`|\bar b| + 2\,\mathrm{MCSE}` of a requirement cell.
BIAS_BOUND_COLUMN = "absolute_bias_bound"
#: The column that publishes a cell's exact-law limit of its mean estimate.
LIMIT_COLUMN = "exact_limit"


def mechanism_term_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The full curve stays calibrated; the curve without the mechanism term under-reports."""
    margins = STUDY.margins
    family = summary["property"] == "mechanism_term_necessity"
    passes: list[bool] = []
    for index in summary.index[family]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == "mechanism_term_necessity") & (rows["cell"] == cell)]
        interval = ratio_intervals(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, "mechanism_term_necessity", cell),
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = interval.low
        summary.loc[index, "se_ratio_ci_upper"] = interval.high
        if cell.endswith("__full_eif"):
            passed = interval.within(*EFFICIENCY_RATIO_BAND)
        else:
            passed = interval.high < margins.calibration_se_ratio[0]
        summary.loc[index, "passed"] = bool(passed)
        passes.append(bool(passed))
    summary.loc[family, "property_passed"] = bool(passes) and all(passes)


def single_robustness_verdicts(summary: pd.DataFrame) -> None:
    """The two-size absolute-bias rule, and the published limit of the diagnostic cell."""
    family = summary["property"] == "single_robustness"
    for index in summary.index[family]:
        bound = abs(summary.loc[index, "bias"]) + 2.0 * summary.loc[index, "bias_se"]
        summary.loc[index, BIAS_BOUND_COLUMN] = float(bound)
    cells = {
        size: summary.index[
            family & (summary["cell"] == f"contrast_x2__mechanism_correct__n_{size}")
        ][0]
        for size in REQUIREMENT_SIZES
    }
    small, large = (summary.loc[cells[size], BIAS_BOUND_COLUMN] for size in REQUIREMENT_SIZES)
    verdict = bool(large < ABSOLUTE_BIAS_TOLERANCE and large < small)
    for index in cells.values():
        summary.loc[index, "passed"] = verdict
    diagnostic = family & (summary["cell"] == "contrast_x2__mechanism_constant")
    limit = common.constant_mechanism_limit(law.LAW_A, PLANS[X2]) - common.constant_mechanism_limit(
        law.LAW_A, PLANS[NATURAL]
    )
    summary.loc[diagnostic, LIMIT_COLUMN] = limit
    summary.loc[diagnostic, "passed"] = True
    summary.loc[family, "property_passed"] = verdict


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
            BIAS_BOUND_COLUMN,
            LIMIT_COLUMN,
        ),
        rate_labels=("contrast_x2",),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    mechanism_term_verdicts(summary, rows)
    single_robustness_verdicts(summary)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("contrast_x2",),
        arms=("one_step", "plug_in"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    return finish(summary, rates)


__all__ = [
    "FIT_SETS",
    "declared_cells",
    "failure_probe",
    "generate_property_rows",
    "summarize_properties",
]
