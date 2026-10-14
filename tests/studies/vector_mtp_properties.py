"""Property families of ``vector-mtp``.

Every cell reads one contrast against ``natural``: ``flag scaled`` on the point and two-node
laws, and ``full`` on the ordered law (:mod:`tests.studies.vector_mtp_common`).

==========================  =====================================================================
family                      cells, replications and size
==========================  =====================================================================
``interval_calibration``    six positives with correctly specified nuisances, 2,400 at n = 2,000
                            each, with the two derived controls of each.  Point ``TMLE``,
                            two-node ``LTMLE`` and the ordered law as one ``LTMLE`` node, each on
                            the density route (:class:`~tests.studies.vector_mtp_common.Oracle`)
                            and on the classifier route (the correctly specified
                            :class:`~tests.studies.vector_mtp_common.OracleLogOdds`)
``double_robustness``       the density route with a wrong outcome regression (the prior, and
                            the mean at the first node) and the oracle density, against the
                            same with the marginal oracle (``both_wrong``); 1,000 at n = 2,000.
                            Point ``TMLE`` on the ordered law under ``full``, two-node ``LTMLE``
                            under ``flag scaled``, and the ordered law as one ``LTMLE`` node
                            under ``full``.  ``full`` moves the flag, has a many-to-one
                            preimage, a non-unit Jacobian and an ordered read at once
``preimage_necessity``      two-node ``LTMLE`` under ``flag scaled``, wrong outcome, oracle
                            density: the declared fit against the same draw with the induced
                            density's preimage sum cut to the observed categorical cell
                            (``c = c'``); 1,000 at n = 2,000
==========================  =====================================================================

**Why the preimage cut is a control.**  Under ``flag scaled`` every value with flag 1 has a
preimage with flag 0 (the flag moves and the dose stays).  The cut drops that term, so the
ratio loses about half its mass.  With the outcome wrong, the targeting step carries the whole
correction, so the cut moves the estimate.  ``tests/unit/test_vector_mtp_design.py`` checks the
mutation against the closed-form ratio with the ``c = 0`` term removed.

**Sizes.**  Each calibration positive's pass probability at 2,400 replications is at least 0.90
under the SE-ratio, coverage and efficiency bands
(``tests/unit/test_vector_mtp_design.py``).  Each control's pass probability is computed from
its measured distance and the declared margin, at the declared replications.  The double-
robustness positives and the declared-preimage arm are sized the same way from their measured
bias.  No margin and no size was chosen to save time.

The failure rule is ``longitudinal-mtp``'s: a replication that raises is never redrawn, and the
property phase records no failure, so a raise stops the run.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.interventions import joint_density
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS, memory_capped_workers
from tests.studies import canonical_vector_mtp as study
from tests.studies import vector_mtp_common as common
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
DOUBLE_ROBUST_REPLICATES = 1_000
DOUBLE_ROBUST_N = 2_000
PREIMAGE_REPLICATES = 1_000
PREIMAGE_N = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
PREIMAGE_DISPLACEMENT = 0.10
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: Each property label: ``(law, plan, estimator, ratio route)``.
CELLS: dict[str, tuple[str, str, str, str]] = {
    "point_vector_density": ("point", "flag scaled", "point", "density"),
    "point_vector_classifier": ("point", "flag scaled", "point", "classifier"),
    "longitudinal_vector_density": ("long", "flag scaled", "longitudinal", "density"),
    "longitudinal_vector_classifier": ("long", "flag scaled", "longitudinal", "classifier"),
    "longitudinal_vector_ordered_density": ("ordered", "full", "longitudinal", "density"),
    "longitudinal_vector_ordered_classifier": ("ordered", "full", "longitudinal", "classifier"),
}
#: The double-robustness cells read the ordered law under ``full`` through point ``TMLE``.
ROBUSTNESS_CELLS: dict[str, tuple[str, str, str]] = {
    "point_vector_ordered_density": ("ordered", "full", "point"),
    "longitudinal_vector_density": ("long", "flag scaled", "longitudinal"),
    "longitudinal_vector_ordered_density": ("ordered", "full", "longitudinal"),
}
PREIMAGE_LABEL = "flag_scaled"


def estimand(estimator: str, plan: str) -> str:
    kind = "regimen" if estimator == "longitudinal" else "policy"
    return f"ate_{kind}[{plan} vs {study.REFERENCE}]"


def truth(law: str, plan: str) -> float:
    exact = common.TRUTHS[law]
    return exact(plan) - exact(study.REFERENCE)


#: The efficiency bound of each calibration label, pinned from
#: :func:`~tests.studies.vector_mtp_common.contrast_sd`;
#: ``tests/unit/test_vector_mtp_design.py`` recomputes them.
BOUNDS: dict[str, float] = {
    "point": 0.5961528334632183,
    "long": 0.9146507066003731,
    "ordered": 0.48801744352917514,
}
EFFICIENCY_SD: dict[str, float] = {label: BOUNDS[spec[0]] for label, spec in CELLS.items()}


# ---------------------------------------------------------------------------- the control


@contextmanager
def preimage_cut(active: bool) -> Iterator[None]:
    """The induced density with its preimage sum cut to the target's own cell, if ``active``.

    Each term of :func:`cleverly.interventions.joint_density._enumerate` is one candidate
    natural cell ``c`` and one piece path.  The cut keeps a term only where its cell is the
    categorical cell ``c'`` of the value it is evaluated at, so a plan that moves the flag
    loses every term whose natural flag differs from the assigned one.
    """
    if not active:
        yield
        return
    original = joint_density._enumerate

    def cut(
        layout: Any,
        branch: Any,
        target: Any,
        cells: Any,
        requests: Any,
        observed_range: Any,
    ) -> list[Any]:
        terms = original(layout, branch, target, cells, requests, observed_range)
        values = np.asarray(target, dtype=float)
        own = layout.cell_codes(values[:, layout.categorical], cells)
        return [
            joint_density._Term(
                term.cell, np.where(own == term.cell, term.weight, 0.0), term.factors
            )
            for term in terms
        ]

    joint_density._enumerate = cut
    try:
        yield
    finally:
        joint_density._enumerate = original


# ---------------------------------------------------------------------------- fits


def fit_cell(
    frame: pd.DataFrame,
    law: str,
    plan: str,
    estimator: str,
    *,
    ratio: Literal["density", "classifier"] = "density",
    **configuration: str,
) -> Any:
    """One property fit over ``(natural, plan)``."""
    labels = (study.REFERENCE, plan)
    if estimator == "point":
        return study.fit_point(frame, law, labels, ratio=ratio, **configuration)
    return study.fit_longitudinal(frame, law, labels, ratio=ratio, **configuration)


def calibration_fit(label: str, frame: pd.DataFrame) -> Any:
    law, plan, estimator, route = CELLS[label]
    if route == "classifier":
        return fit_cell(
            frame,
            law,
            plan,
            estimator,
            ratio="classifier",
            mechanism="log_odds",
            label=plan,
            pseudo="rich",
        )
    return fit_cell(frame, law, plan, estimator, pseudo="rich")


def robustness_fit(label: str, configuration: str, frame: pd.DataFrame) -> Any:
    law, plan, estimator = ROBUSTNESS_CELLS[label]
    mechanism = "oracle" if configuration == "outcome_wrong" else "marginal"
    return fit_cell(frame, law, plan, estimator, outcome="wrong", mechanism=mechanism)


def preimage_fit(frame: pd.DataFrame, *, cut: bool) -> Any:
    with preimage_cut(cut):
        return fit_cell(frame, "long", "flag scaled", "longitudinal", outcome="wrong")


# ---------------------------------------------------------------------------- the grid


class DeclaredLaw:
    """The law a declared cell reads, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        ``"point"``, ``"long"`` or ``"ordered"``.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def truth(self) -> dict[str, float]:
        plan = "full" if self.name == "ordered" else "flag scaled"
        value = truth(self.name, plan)
        return {estimand(kind, plan): value for kind in ("point", "longitudinal")}


#: Every fit set: ``(family, stream label, n, replicates)``.
FIT_SETS: tuple[tuple[str, str, int, int], ...] = (
    *(("interval_calibration", label, CALIBRATION_N, CALIBRATION_REPLICATES) for label in CELLS),
    *(
        (
            "double_robustness",
            f"{label}__{configuration}",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
        )
        for label in ROBUSTNESS_CELLS
        for configuration in ("outcome_wrong", "both_wrong")
    ),
    ("preimage_necessity", "paired", PREIMAGE_N, PREIMAGE_REPLICATES),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def _law_of(family: str, label: str) -> str:
    if family == "interval_calibration":
        return CELLS[label][0]
    if family == "double_robustness":
        return ROBUSTNESS_CELLS[label.rsplit("__", 1)[0]][0]
    return "long"


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(
        family: str, cell: str, role: str, law: str, n: int, reps: int, stream: str, name: str
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(law),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(family, stream, 0),
                role=role,
                estimand=name,
            )
        )

    for label, (law, plan, estimator, _) in CELLS.items():
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                law,
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                label,
                estimand(estimator, plan),
            )
    for label, (law, plan, estimator) in ROBUSTNESS_CELLS.items():
        for configuration in ("outcome_wrong", "both_wrong"):
            role = "control" if configuration == "both_wrong" else "positive"
            stream = f"{label}__{configuration}"
            add(
                "double_robustness",
                stream,
                role,
                law,
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                stream,
                estimand(estimator, plan),
            )
    name = estimand("longitudinal", "flag scaled")
    add(
        "preimage_necessity",
        f"{PREIMAGE_LABEL}__declared_preimage",
        "positive",
        "long",
        PREIMAGE_N,
        PREIMAGE_REPLICATES,
        "paired",
        name,
    )
    add(
        "preimage_necessity",
        f"{PREIMAGE_LABEL}__preimage_cut_control",
        "control",
        "long",
        PREIMAGE_N,
        PREIMAGE_REPLICATES,
        "paired",
        name,
    )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _fit_set_rows(payload: tuple[str, str, int, int, int]) -> list[dict[str, Any]]:
    family, label, replicate, n, requested = payload
    law = _law_of(family, label)
    frame = common.SAMPLERS[law](n, _seed(family, label, replicate))
    common_row: dict[str, Any] = {"replicate": replicate, "n": n, "requested": requested}

    def row(cell: str, role: str, value: float, estimate: Any) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            truth=value,
            estimate=estimate,
            alpha=STUDY.margins.alpha,
            **common_row,
        )

    if family == "interval_calibration":
        _, plan, estimator, _ = CELLS[label]
        result = calibration_fit(label, frame)
        name = estimand(estimator, plan)
        return [row(f"{label}__correctly_specified", "positive", truth(law, plan), result[name])]
    if family == "double_robustness":
        stem, configuration = label.rsplit("__", 1)
        _, plan, estimator = ROBUSTNESS_CELLS[stem]
        result = robustness_fit(stem, configuration, frame)
        role = "control" if configuration == "both_wrong" else "positive"
        return [row(label, role, truth(law, plan), result[estimand(estimator, plan)])]
    name = estimand("longitudinal", "flag scaled")
    value = truth("long", "flag scaled")
    declared = preimage_fit(frame, cut=False)
    dropped = preimage_fit(frame, cut=True)
    return [
        row(f"{PREIMAGE_LABEL}__declared_preimage", "positive", value, declared[name]),
        control_row(
            property_name=family,
            cell=f"{PREIMAGE_LABEL}__preimage_cut_control",
            truth=value,
            estimate=float(dropped[name].psi),
            standard_error=float(dropped[name].std_error),
            critical=CRITICAL,
            **common_row,
        ),
    ]


def fit_bytes(n: int) -> int:
    """A generous peak of one property worker at ``n`` rows, for the worker count.

    The design probe of ``tests/unit/test_vector_mtp_design.py`` measured a peak resident set
    of 0.24 GiB per worker at n = 2,000, the interpreter included, with the ordered law's point
    fit the largest.  This allows 0.5 GiB, or three one-dose oracle fits if that is more.
    """
    from tests.studies.oracle_density_bins import fit_bytes as one_dose

    return max(3 * one_dose(n), 2**29)


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for family, label, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(((family, label, r, n, requested),) for r in range(requested))
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    jobs = memory_capped_workers(n_jobs, fit_bytes(max(n for _, _, n, _ in FIT_SETS)))
    outcomes = map_parallel(_fit_set_rows, _payloads(budget), n_jobs=jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=tuple(CELLS),
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
    """Failed fits per fit set over streams ``start`` to ``start + draws - 1``."""
    payloads = [
        ((family, label, r, n, draws),)
        for family, label, n, _ in FIT_SETS
        for r in range(start, start + draws)
    ]
    outcomes = map_parallel(_probe_one, payloads, n_jobs=n_jobs)
    counts = {f"{family}/{label}": 0 for family, label, *_ in FIT_SETS}
    for (payload,), failed in zip(payloads, outcomes, strict=True):
        counts[f"{payload[0]}/{payload[1]}"] += int(failed)
    return counts


def _probe_one(payload: tuple[str, str, int, int, int]) -> bool:
    try:
        rows = _fit_set_rows(payload)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("preimage_displacement",),
        rate_labels=(),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="preimage_necessity",
        labels=(PREIMAGE_LABEL,),
        arms=("declared_preimage", "preimage_cut_control"),
        column="preimage_displacement",
        threshold=PREIMAGE_DISPLACEMENT,
    )
    return finish(summary, rates)
