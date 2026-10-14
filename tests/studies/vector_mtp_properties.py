"""Property families of ``vector-mtp``.

Every cell reads one contrast against ``natural course``: ``flag scaled`` on the point and
two-node laws, and ``full`` or ``crossed`` on the ordered law
(:mod:`tests.studies.vector_mtp_common`).

==========================  =====================================================================
family                      cells, replications and size
==========================  =====================================================================
``interval_calibration``    six positives with correctly specified nuisances, at n = 2,000, with
                            the two derived controls of each.  Point ``TMLE``, two-node
                            ``LTMLE`` and the ordered law as one ``LTMLE`` node, each on the
                            density route (:class:`~tests.studies.vector_mtp_common.Oracle`)
                            and on the classifier route (the correctly specified
                            :class:`~tests.studies.vector_mtp_common.OracleLogOdds`).  Each
                            positive has its own replications (:data:`CALIBRATION_REPLICATES`)
``double_robustness``       the density route in four nuisance configurations: ``both_correct``,
                            ``outcome_correct`` (the marginal oracle density), ``outcome_wrong``
                            (the prior, and the mean at the first node, with the oracle density)
                            and the ``both_wrong`` control; 2,000 at n = 2,000.  Point ``TMLE``
                            on the ordered law under ``full``, two-node ``LTMLE`` under ``flag
                            scaled``, and the ordered law as one ``LTMLE`` node under ``full``
``preimage_necessity``      two-node ``LTMLE`` under ``flag scaled``, wrong outcome, oracle
                            density: the declared fit against the same draw with the induced
                            density's preimage sum cut to the observed categorical cell
                            (``c = c'``); 1,000 at n = 2,000
``jacobian_necessity``      the ordered law as one ``LTMLE`` node under ``full``, wrong outcome,
                            oracle density: the declared fit against the same draw with the
                            Jacobian of every preimage dropped; 1,000 at n = 2,000
``image_necessity``         the same fit under ``crossed``: the declared fit against the same
                            draw with the flag map evaluated at the observed ``D1'`` instead of
                            the preimage's natural ``D1``; 1,000 at n = 2,000
==========================  =====================================================================

**What each control removes.**  Under ``flag scaled`` every value with flag 1 has a preimage
with flag 0 (the flag moves and the dose stays).  The preimage cut drops that term, so the ratio
loses about half its mass.  The Jacobian drop leaves every preimage and multiplies its density
by 1 instead of ``1/1.25``.  Under ``crossed`` a natural ``(0, D1)`` with ``2.4 <= D1 < 3``
keeps flag 0 and moves to ``D1' >= 3``.  The image defect reads ``D1'`` and sends it to flag 1.
With the outcome wrong, the targeting step carries the whole correction, so each defect moves
the estimate.  ``tests/unit/test_vector_mtp_design.py`` checks each mutation against the closed
form with the same defect (:func:`~tests.studies.vector_mtp_common.ordered_ratio`), and records
each control's exact-law shift and the probability that its pair passes.

**Sizes.**  The design test computes every number below before any run.  Each calibration
positive's replications make the product of the six positives' pass probabilities at least 0.90
at the SE ratios, coverages and efficiency ratios of fresh-stream probes.  The product stays at
least 0.90 with every probed number moved one Monte Carlo SE toward failing.  Each control's pass
probability is computed from its measured or exact-law distance and the declared margin.  The
double-robustness positives and each necessity pair's declared arm are sized the same way from
their measured bias, moved one Monte Carlo SE toward failing.  No margin and no size was chosen to save time.

**The defects patch** :mod:`cleverly.interventions.joint_density` **module-wide.**  A patch is
safe only because each property fit runs in its own worker process
(:func:`~cleverly.utils.parallel.map_parallel` uses process workers) and each fit is serial.  A
thread backend, or ``n_jobs > 1`` inside a fit, would leak the defect into concurrent fits.
:func:`density_defect` refuses to start while any other patch is active.

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
CALIBRATION_N = 2_000
DOUBLE_ROBUST_REPLICATES = 2_000
DOUBLE_ROBUST_N = 2_000
NECESSITY_REPLICATES = 1_000
NECESSITY_N = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
PREIMAGE_DISPLACEMENT = 0.10
#: The declared minimum displacement of ``jacobian_necessity`` and ``image_necessity``.
NECESSITY_DISPLACEMENT = 0.10
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
#: The replications of each calibration positive, sized by
#: ``tests/unit/test_vector_mtp_design.py`` from fresh-stream probes.
CALIBRATION_REPLICATES: dict[str, int] = {
    "point_vector_density": 24_000,
    "point_vector_classifier": 10_000,
    "longitudinal_vector_density": 9_600,
    "longitudinal_vector_classifier": 10_000,
    "longitudinal_vector_ordered_density": 19_200,
    "longitudinal_vector_ordered_classifier": 10_000,
}
#: The double-robustness cells read the ordered law under ``full`` through point ``TMLE``.
ROBUSTNESS_CELLS: dict[str, tuple[str, str, str]] = {
    "point_vector_ordered_density": ("ordered", "full", "point"),
    "longitudinal_vector_density": ("long", "flag scaled", "longitudinal"),
    "longitudinal_vector_ordered_density": ("ordered", "full", "longitudinal"),
}
#: Each double-robustness configuration: ``(outcome, mechanism)``.  A correct outcome reads
#: :class:`~tests.studies.vector_mtp_common.RichPseudo` at the two-node law's first node.
ROBUSTNESS_CONFIGURATIONS: dict[str, tuple[str, str]] = {
    "both_correct": ("correct", "oracle"),
    "outcome_correct": ("correct", "marginal"),
    "outcome_wrong": ("wrong", "oracle"),
    "both_wrong": ("wrong", "marginal"),
}
#: Each necessity family: ``(law, plan, label, (positive arm, control arm), defect)``.  Every
#: arm is ``LTMLE`` with the outcome wrong and the oracle density.
NECESSITY: dict[str, tuple[str, str, str, tuple[str, str], str]] = {
    "preimage_necessity": (
        "long",
        "flag scaled",
        "flag_scaled",
        ("declared_preimage", "preimage_cut_control"),
        "preimage_cut",
    ),
    "jacobian_necessity": (
        "ordered",
        "full",
        "ordered_full",
        ("declared_jacobian", "jacobian_drop_control"),
        "jacobian",
    ),
    "image_necessity": (
        "ordered",
        "crossed",
        "ordered_crossed",
        ("declared_image", "shifted_read_control"),
        "image",
    ),
}
PREIMAGE_LABEL = NECESSITY["preimage_necessity"][2]
#: The column each necessity family publishes its displacement in, and its threshold.
NECESSITY_COLUMNS: dict[str, tuple[str, float]] = {
    "preimage_necessity": ("preimage_displacement", PREIMAGE_DISPLACEMENT),
    "jacobian_necessity": ("necessity_displacement", NECESSITY_DISPLACEMENT),
    "image_necessity": ("necessity_displacement", NECESSITY_DISPLACEMENT),
}


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


# ---------------------------------------------------------------------------- the controls

#: The functions of :mod:`cleverly.interventions.joint_density` a defect replaces.
_PATCHED = ("_enumerate", "_image")
_ORIGINAL = {name: getattr(joint_density, name) for name in _PATCHED}


def _terms_with(terms: list[Any], weights: list[np.ndarray]) -> list[Any]:
    return [
        joint_density._Term(term.cell, weight, term.factors)
        for term, weight in zip(terms, weights, strict=True)
    ]


@contextmanager
def density_defect(defect: str | None) -> Iterator[None]:
    """The induced density with one deliberate defect, or unchanged if ``defect`` is ``None``.

    Each term of :func:`cleverly.interventions.joint_density._enumerate` is one candidate
    natural cell ``c`` and one piece path, weighted by its Jacobian where the categorical
    image of ``(c, beta)`` is the value's cell.

    * ``"preimage_cut"`` keeps a term only where its cell is the categorical cell ``c'`` of
      the value it is evaluated at, so a plan that moves the flag loses every term whose
      natural flag differs from the assigned one.
    * ``"jacobian"`` replaces each nonzero weight by 1: the inverse derivatives are dropped.
    * ``"image"`` evaluates each categorical map at the value's own doses instead of the
      preimage's natural doses.

    The patch is module-wide, so it refuses to start while another patch is active.
    """
    if defect is None:
        yield
        return
    for name in _PATCHED:
        if getattr(joint_density, name) is not _ORIGINAL[name]:
            raise RuntimeError(f"joint_density.{name} is already patched")
    enumerate_terms = _ORIGINAL["_enumerate"]
    image = _ORIGINAL["_image"]
    replaced: dict[str, Any] = {}
    if defect == "preimage_cut":

        def cut(layout: Any, branch: Any, target: Any, cells: Any, *rest: Any) -> list[Any]:
            terms = enumerate_terms(layout, branch, target, cells, *rest)
            values = np.asarray(target, dtype=float)
            own = layout.cell_codes(values[:, layout.categorical], cells)
            return _terms_with(
                terms, [np.where(own == term.cell, term.weight, 0.0) for term in terms]
            )

        replaced["_enumerate"] = cut
    elif defect == "jacobian":

        def unit(*arguments: Any) -> list[Any]:
            terms = enumerate_terms(*arguments)
            return _terms_with(terms, [np.where(term.weight > 0.0, 1.0, 0.0) for term in terms])

        replaced["_enumerate"] = unit
    elif defect == "image":
        targets: list[np.ndarray] = []

        def remember(layout: Any, branch: Any, target: Any, *rest: Any) -> list[Any]:
            targets.append(np.asarray(target, dtype=float))
            try:
                return enumerate_terms(layout, branch, target, *rest)
            finally:
                targets.pop()

        def at_value(
            layout: Any, branch: Any, cell_codes: Any, dose_values: Any, valid: Any
        ) -> Any:
            observed = targets[-1][:, list(layout.doses)]
            return image(layout, branch, cell_codes, observed, valid)

        replaced["_enumerate"] = remember
        replaced["_image"] = at_value
    else:
        raise ValueError(f"unknown defect {defect!r}")
    for name, function in replaced.items():
        setattr(joint_density, name, function)
    try:
        yield
    finally:
        for name in replaced:
            setattr(joint_density, name, _ORIGINAL[name])


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
    """One property fit over ``(natural course, plan)``."""
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
    outcome, mechanism = ROBUSTNESS_CONFIGURATIONS[configuration]
    pseudo = "rich" if outcome == "correct" else "least_squares"
    return fit_cell(
        frame, law, plan, estimator, outcome=outcome, mechanism=mechanism, pseudo=pseudo
    )


def necessity_fit(family: str, frame: pd.DataFrame, *, control: bool) -> Any:
    """One arm of a necessity pair: the declared fit, or the same fit with the defect."""
    law, plan, _, _, defect = NECESSITY[family]
    with density_defect(defect if control else None):
        return fit_cell(frame, law, plan, "longitudinal", outcome="wrong")


def preimage_fit(frame: pd.DataFrame, *, cut: bool) -> Any:
    return necessity_fit("preimage_necessity", frame, control=cut)


# ---------------------------------------------------------------------------- the grid


class DeclaredLaw:
    """The law a declared cell reads, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        ``"point"``, ``"long"`` or ``"ordered"``.
    plan : str
        The plan contrasted with ``natural course``.
    """

    def __init__(self, name: str, plan: str) -> None:
        self.name = name
        self.plan = plan

    def truth(self) -> dict[str, float]:
        value = truth(self.name, self.plan)
        return {estimand(kind, self.plan): value for kind in ("point", "longitudinal")}


#: Every fit set: ``(family, stream label, n, replicates)``.
FIT_SETS: tuple[tuple[str, str, int, int], ...] = (
    *(
        ("interval_calibration", label, CALIBRATION_N, CALIBRATION_REPLICATES[label])
        for label in CELLS
    ),
    *(
        (
            "double_robustness",
            f"{label}__{configuration}",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
        )
        for label in ROBUSTNESS_CELLS
        for configuration in ROBUSTNESS_CONFIGURATIONS
    ),
    *((family, "paired", NECESSITY_N, NECESSITY_REPLICATES) for family in NECESSITY),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def _law_of(family: str, label: str) -> str:
    if family == "interval_calibration":
        return CELLS[label][0]
    if family == "double_robustness":
        return ROBUSTNESS_CELLS[label.rsplit("__", 1)[0]][0]
    return NECESSITY[family][0]


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        law: DeclaredLaw,
        n: int,
        reps: int,
        stream: str,
        name: str,
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=law,
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
                DeclaredLaw(law, plan),
                CALIBRATION_N,
                CALIBRATION_REPLICATES[label],
                label,
                estimand(estimator, plan),
            )
    for label, (law, plan, estimator) in ROBUSTNESS_CELLS.items():
        for configuration in ROBUSTNESS_CONFIGURATIONS:
            role = "control" if configuration == "both_wrong" else "positive"
            stream = f"{label}__{configuration}"
            add(
                "double_robustness",
                stream,
                role,
                DeclaredLaw(law, plan),
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                stream,
                estimand(estimator, plan),
            )
    for family, (law, plan, label, arms, _) in NECESSITY.items():
        for arm, role in zip(arms, ("positive", "control"), strict=True):
            add(
                family,
                f"{label}__{arm}",
                role,
                DeclaredLaw(law, plan),
                NECESSITY_N,
                NECESSITY_REPLICATES,
                "paired",
                estimand("longitudinal", plan),
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
    _, plan, stem, (positive_arm, control_arm), _ = NECESSITY[family]
    name = estimand("longitudinal", plan)
    value = truth(law, plan)
    declared = necessity_fit(family, frame, control=False)
    defective = necessity_fit(family, frame, control=True)
    return [
        row(f"{stem}__{positive_arm}", "positive", value, declared[name]),
        control_row(
            property_name=family,
            cell=f"{stem}__{control_arm}",
            truth=value,
            estimate=float(defective[name].psi),
            standard_error=float(defective[name].std_error),
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
    columns = tuple(dict.fromkeys(column for column, _ in NECESSITY_COLUMNS.values()))
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=columns,
        rate_labels=(),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    for family, (_, _, label, arms, _) in NECESSITY.items():
        column, threshold = NECESSITY_COLUMNS[family]
        necessity_verdicts(
            summary,
            rows,
            family=family,
            labels=(label,),
            arms=arms,
            column=column,
            threshold=threshold,
        )
    return finish(summary, rates)
