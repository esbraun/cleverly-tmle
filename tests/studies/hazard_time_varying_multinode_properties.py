"""Property families of ``hazard-time-varying-multinode``.

Every fit is ``LTMLE`` with ``survival_model="hazard"`` and ``targeting="one_step"`` on law (M)
of :mod:`tests.studies.hazard_time_varying_multinode_law`: a binary treatment at three nodes
with a binary covariate :math:`L_k` before the treatment at nodes 2 and 3.  A cell label names
one estimate of one fit label of
:data:`~tests.studies.canonical_hazard_time_varying_multinode.FITS`: ``<fit>_contrast`` is always
against never in the risk by the last node, ``<fit>_rule`` and ``<fit>_policy`` the rule and the
policy against never, and ``<fit>_risk`` always's risk.  ``multinode_curve`` reads all 21
estimates of one fit under the simultaneous band.  Every truth is an exact sum over the law's
support and every efficiency bound is exact.

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``interval_calibration``        six arms (``multinode`` contrast, rule, policy and risk;
                                ``crossfit`` contrast and policy), each with the shrunken-SE
                                and noise controls; :data:`CALIBRATION_REPLICATES` fits per
                                fit label at n = 2,000
``double_robustness``           eleven arms on the five configurations' fits
                                (:data:`~tests.studies.canonical_hazard_time_varying_multinode.
                                DOUBLE_ROBUST_ARMS`); each configuration runs the largest count
                                among its arms, at least 1,200, at n = 2,000
``targeting_necessity``         ``multinode_contrast``: the targeted contrast of the
                                ``mechanism_correct`` fits, and the untargeted backward
                                plug-in of the same draws
``root_n_and_efficiency``       ``multinode_contrast`` and ``multinode_policy`` at n = 1,000,
                                2,000 and 8,000; 800 each.  The 1,000 rung carries no claim
``power``                       ``multinode_contrast`` against zero, on the ``multinode``
                                calibration fits
``simultaneous_coverage``       the band over every estimate of the ``multinode``
                                calibration fits, and its pointwise control
``iterated_mean_difference``    reported, no verdict: :data:`DIFFERENCE_REPLICATES` draws of
                                ``multinode`` fitted by the hazard fit and the iterated-mean
                                fit without and with ``isotonic=True``
==============================  ===========================================================

The wrong hazard and continuation learners omit ``L``; the wrong mechanism drops ``L_k`` from
the treatment and the retention models.  The rule and policy ``outcome_correct`` arms are where a
defect of the slot values shows: under a correct mechanism the walk repairs any initial fit that
is a function of :math:`(H_k, a)`, and on the static contrast the held slot, the current level and
the level plan coincide for every follower.

The failure rule is the registry's: a replication that raises is never redrawn, and the shared
harness refuses a cell that lost more than its bound.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.longitudinal import LTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_hazard_time_varying_multinode as study
from tests.studies import hazard_time_varying_multinode_law as law
from tests.studies import hazard_time_varying_properties as held
from tests.studies.evidence import property_verdicts
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
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
    reported_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import FAMILY as JOINT_FAMILY
from tests.studies.evidence.simultaneous import (
    joint_cells,
    joint_coverage_rows,
    joint_property_cells,
)

STUDY = study.STUDY
#: The replicates of a positive robustness arm whose exact limit is the truth, and the floor
#: of each wrong-learner arm's count under the rule of plan X70 rev 2.7.
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
DOUBLE_ROBUST_ARMS = study.DOUBLE_ROBUST_ARMS
#: The configurations that fit a wrong learner, each with its pilot label; their counts come
#: from the committed 800-draw pilots (the design test), provisionally the floor.  The four
#: pilots of plan X70 section 5.
WRONG_LEARNER_ARMS: dict[str, str] = {
    configuration: f"wrong_learner_{configuration}"
    for configuration in (
        "outcome_correct",
        "mechanism_correct",
        "continuation_wrong",
        "both_wrong",
    )
}
WRONG_LEARNER_REPLICATES: dict[str, int] = dict.fromkeys(WRONG_LEARNER_ARMS, 1_200)
NECESSITY_CONFIGURATION = "mechanism_correct"
NECESSITY_N = DOUBLE_ROBUST_N
RATE_REPLICATES = 800
RATE_SIZES = study.RATE_SIZES
CALIBRATION_N = 2_000
#: The arm-pooled rule over the six calibration arms, provisional until the calibration pilots
#: are committed (the design test).
CALIBRATION_REPLICATES = 4_000
POWER_REPLICATES = CALIBRATION_REPLICATES
POWER_N = CALIBRATION_N
DIFFERENCE_REPLICATES = 800
DIFFERENCE_N = 2_000

EFFICIENCY_RATIO_BAND = held.EFFICIENCY_RATIO_BAND
SHRUNKEN_SE_FACTOR = held.SHRUNKEN_SE_FACTOR
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
TARGETING_DISPLACEMENT = held.TARGETING_DISPLACEMENT
IC_GAP_SIZES = held.IC_GAP_SIZES
IC_GAP_DRAWS = held.IC_GAP_DRAWS
IC_GAP_BOUND = held.IC_GAP_BOUND
IC_GAP_FALL = held.IC_GAP_FALL
CONFIGURATIONS = tuple(study.CONFIGURATIONS)
DIFFERENCE_FAMILY = "iterated_mean_difference"
CONTRAST = study.CONTRAST_LABEL
FITS = study.FITS
INFLUENCE_COLUMNS = held.INFLUENCE_COLUMNS
CONVERGENCE_COLUMNS = held.CONVERGENCE_COLUMNS
CONVERGENCE_SUMMARY = held.CONVERGENCE_SUMMARY
STALLED_SCORE = held.STALLED_SCORE
NONCONVERGED_SUSPECT_SHARE = held.NONCONVERGED_SUSPECT_SHARE

#: Fit label -> ``(cell label, reported name)`` of each arm.
COEFFICIENTS = {
    fit_label: tuple((f"{fit_label}_{arm}", name) for arm, name in study.ARMS.items())
    for fit_label in FITS
}
ESTIMAND = {label: name for pairs in COEFFICIENTS.values() for label, name in pairs}
FIT_OF = {label: fit_label for fit_label, pairs in COEFFICIENTS.items() for label, _ in pairs}
CALIBRATION_LABELS = study.CALIBRATION_LABELS
CONTROLLED_LABELS = study.CONTROLLED_LABELS
JOINT_LABELS = study.JOINT_LABELS
JOINT_FIT = {label: label.removesuffix("_curve") for label in JOINT_LABELS}


def target(label: str) -> law.Target:
    """The linear target a cell label reads."""
    return study.target_of(ESTIMAND[label])


def _exact() -> tuple[dict[str, float], dict[str, float]]:
    truth: dict[str, float] = {}
    bound: dict[str, float] = {}
    for label in FIT_OF:
        aim = target(label)
        truth[label], bound[label] = aim.value(law.LAW), law.efficiency_sd(law.LAW, aim)
    return truth, bound


TRUTHS, EFFICIENCY_SD = _exact()


def iterated_fit(
    label: str,
    frame: pd.DataFrame,
    *,
    isotonic: bool = False,
    regression_correct: bool = True,
) -> Any:
    """The iterated-mean fit of the same regimens on one fit label's sample.

    Each node's regression is the cell mean over every column but the earlier ``L`` (over no
    ``L`` when ``regression_correct`` is false); the mechanism learners are the hazard fit's
    correct ones.
    """
    mechanism = study.learners("both_correct")
    del mechanism["outcome_learner"], mechanism["pseudo_learner"]
    return LTMLE(
        study.regimens(),
        reference=study.REFERENCE,
        n_folds=FITS[label],
        simultaneous=False,
        survival_model="iterated_mean",
        isotonic=isotonic,
        targeting="iterative",
        max_iter=100,
        tol=study.TOL,
        random_state=0,
        outcome_learner=law.continuation_learner(regression_correct),
        pseudo_learner=law.continuation_learner(regression_correct),
        **mechanism,
    ).fit(frame, **law.LAW.columns())


def largest_decrease(result: Any) -> tuple[float, float]:
    """The largest decrease of a fit's risk curves between nodes, and its standard error."""
    best: tuple[float, float] | None = None
    for label in study.LABELS:
        names = [study.risk_name(label, t) for t in range(1, study.LAST + 1)]
        for before_name, after_name in itertools.pairwise(names):
            before, after = result[before_name], result[after_name]
            drop = float(before.psi - after.psi)
            curve = np.asarray(before.influence_curve) - np.asarray(after.influence_curve)
            error = float(np.std(curve) / np.sqrt(curve.size))
            if best is None or drop > best[0]:
                best = (drop, error)
    assert best is not None
    return best


def draw(fit_label: str, n: int, seed: int) -> pd.DataFrame:
    del fit_label
    return law.LAW.draw(n, seed)


fits_differ_reading = held.fits_differ_reading


# ------------------------------------------------------------------ the grid


def configuration_replicates(configuration: str) -> int:
    """The replicates of one configuration's robustness fits: the largest of its arms."""
    return WRONG_LEARNER_REPLICATES.get(configuration, DOUBLE_ROBUST_REPLICATES)


#: Each configuration with the cell labels of its robustness arms.
CONFIGURATION_ARMS: dict[str, tuple[str, ...]] = {
    configuration: tuple(
        label
        for label, arm_configuration in DOUBLE_ROBUST_ARMS
        if arm_configuration == configuration
    )
    for configuration in CONFIGURATIONS
}

#: Every fit set: ``(family, stream label, configuration, n, replicates, fit label)``.  The
#: necessity cells ride on the ``mechanism_correct`` robustness fits and the power cell on the
#: ``multinode`` calibration fits.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        ("interval_calibration", fit, "both_correct", CALIBRATION_N, CALIBRATION_REPLICATES, fit)
        for fit in FITS
    ),
    *(
        (
            "double_robustness",
            configuration,
            configuration,
            DOUBLE_ROBUST_N,
            configuration_replicates(configuration),
            "multinode",
        )
        for configuration in CONFIGURATIONS
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "multinode")
        for size in RATE_SIZES
    ),
    *(
        (DIFFERENCE_FAMILY, fit, "both_correct", DIFFERENCE_N, DIFFERENCE_REPLICATES, fit)
        for fit in study.DIFFERENCE_FITS
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


class DeclaredEstimates:
    """The law a fit label's cells read, with its exact truths.

    Parameters
    ----------
    fit_label : str
        A key of :data:`FITS`.
    """

    def __init__(self, fit_label: str) -> None:
        self.fit_label = fit_label
        self.name = f"hazard_time_varying_multinode__{fit_label}"

    def truth(self) -> dict[str, float]:
        return {ESTIMAND[label]: TRUTHS[label] for label, _ in COEFFICIENTS[self.fit_label]}


class DeclaredDifference:
    """The zero every ``iterated_mean_difference`` difference row is recorded against."""

    name = "hazard_time_varying_multinode__iterated_mean_difference"

    def truth(self) -> dict[str, float]:
        return {"paired_difference": 0.0, "largest_decrease": 0.0}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        dgp: Any,
        estimand: str,
        n: int,
        reps: int,
        stream: str,
        draws_of: str | None = None,
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=dgp,
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(draws_of or family, stream, 0),
                role=role,
                estimand=estimand,
            )
        )

    multinode = DeclaredEstimates("multinode")
    for label in CALIBRATION_LABELS:
        fit = FIT_OF[label]
        arms = [("correctly_specified", "positive")]
        if label in CONTROLLED_LABELS:
            arms += [("shrunken_se_control", "control"), ("noise_control", "control")]
        for arm, role in arms:
            add(
                "interval_calibration",
                f"{label}__{arm}",
                role,
                DeclaredEstimates(fit),
                ESTIMAND[label],
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                fit,
            )
    for label, configuration in DOUBLE_ROBUST_ARMS:
        add(
            "double_robustness",
            f"{label}__{configuration}",
            "control" if configuration == "both_wrong" else "positive",
            multinode,
            ESTIMAND[label],
            DOUBLE_ROBUST_N,
            configuration_replicates(configuration),
            configuration,
        )
    for arm, role in (("targeted", "positive"), ("untargeted", "control")):
        add(
            "targeting_necessity",
            f"{CONTRAST}__{arm}",
            role,
            multinode,
            ESTIMAND[CONTRAST],
            NECESSITY_N,
            configuration_replicates(NECESSITY_CONFIGURATION),
            NECESSITY_CONFIGURATION,
            draws_of="double_robustness",
        )
    for size in RATE_SIZES:
        for label in study.LADDER_LABELS:
            add(
                "root_n_and_efficiency",
                f"{label}__n_{size}",
                "control" if size == min(RATE_SIZES) else "positive",
                multinode,
                ESTIMAND[label],
                size,
                RATE_REPLICATES,
                f"n_{size}",
            )
    add(
        "power",
        f"{CONTRAST}__alternative",
        "positive",
        multinode,
        ESTIMAND[CONTRAST],
        POWER_N,
        POWER_REPLICATES,
        "multinode",
        draws_of="interval_calibration",
    )
    for joint in JOINT_LABELS:
        cells.extend(
            joint_property_cells(
                joint,
                n=CALIBRATION_N,
                replicates=CALIBRATION_REPLICATES,
                seed=_seed("interval_calibration", JOINT_FIT[joint], 0),
            )
        )
    difference = DeclaredDifference()
    for fit in study.DIFFERENCE_FITS:
        for label in study.DIFFERENCE_LABELS:
            for arm in study.DIFFERENCE_ARMS:
                paired = arm in ("paired_difference", "isotonic_difference")
                add(
                    DIFFERENCE_FAMILY,
                    f"{label}__{arm}",
                    DIAGNOSTIC_ROLE,
                    difference if paired else DeclaredEstimates(fit),
                    "paired_difference" if paired else ESTIMAND[label],
                    DIFFERENCE_N,
                    DIFFERENCE_REPLICATES,
                    fit,
                )
        add(
            DIFFERENCE_FAMILY,
            f"{fit}__iterated_curve_decrease",
            DIAGNOSTIC_ROLE,
            difference,
            "largest_decrease",
            DIFFERENCE_N,
            DIFFERENCE_REPLICATES,
            fit,
        )
    return tuple(cells)


# ------------------------------------------------------------------ rows


DERIVED_CELLS = calibration_derived(CONTROLLED_LABELS)

_Payload = tuple[str, str, str, int, int, int, str]
_Key = tuple[_Payload]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    family, stream, _, replicate, n, _, label = key[0]
    return draw(label, n, _seed(family, stream, replicate))


def cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the fit set's rows and derived controls fill."""
    family, _, configuration, _, n, _, label = key[0]
    if family == "interval_calibration":
        names: list[str] = []
        for cell_label, _ in COEFFICIENTS[label]:
            if cell_label not in CALIBRATION_LABELS:
                continue
            name = f"{family}/{cell_label}__correctly_specified"
            names += [name, *DERIVED_CELLS.get(name, ())]
        joint = f"{label}_curve"
        if joint in JOINT_LABELS:
            names += [f"{JOINT_FAMILY}/{cell}" for cell in joint_cells(joint)]
        if label == "multinode":
            names.append(f"power/{CONTRAST}__alternative")
        return names
    if family == "double_robustness":
        names = [f"{family}/{cell}__{configuration}" for cell in CONFIGURATION_ARMS[configuration]]
        if configuration == NECESSITY_CONFIGURATION:
            names += [
                f"targeting_necessity/{CONTRAST}__{arm}" for arm in ("targeted", "untargeted")
            ]
        return names
    if family == "root_n_and_efficiency":
        return [f"{family}/{cell_label}__n_{n}" for cell_label in study.LADDER_LABELS]
    if family == DIFFERENCE_FAMILY:
        return [
            *(
                f"{family}/{cell_label}__{arm}"
                for cell_label in study.DIFFERENCE_LABELS
                for arm in study.DIFFERENCE_ARMS
            ),
            f"{family}/{label}__iterated_curve_decrease",
        ]
    raise ValueError(f"no fit set of family {family!r}")


def cell_of(key: _Key) -> str:
    return f"{key[0][0]}/{key[0][1]}"


def replicate_of(key: _Key) -> int:
    return key[0][3]


def fit_set_rows(payload: _Payload) -> list[dict[str, Any]]:
    """The replicate rows of one fit of one fit set, drawn and fitted in one call."""
    return fit_drawn(payload, _draw((payload,)))


def _paired_row(cell: str, hazard: Any, other: float, common_row: dict[str, Any]) -> dict[str, Any]:
    low, high = hazard.ci
    covered = int(low <= other <= high)
    return {
        "property": DIFFERENCE_FAMILY,
        "cell": cell,
        "role": DIAGNOSTIC_ROLE,
        **common_row,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": float(hazard.psi) - other,
        "std_error": float(hazard.std_error),
        "covered": covered,
        "rejected": 1 - covered,
    }


def _reading_row(
    cell: str, value: float, error: float, common_row: dict[str, Any]
) -> dict[str, Any]:
    """A reported reading recorded against zero: ``rejected`` marks a positive value."""
    positive = int(value > 0.0)
    return {
        "property": DIFFERENCE_FAMILY,
        "cell": cell,
        "role": DIAGNOSTIC_ROLE,
        **common_row,
        "failed_replicates": 0,
        "truth": 0.0,
        "estimate": value,
        "std_error": error,
        "covered": 1 - positive,
        "rejected": positive,
    }


class FitRecords:
    """The :class:`~tests.studies.canonical_hazard_time_varying.FitRecord` of every fit a key runs."""

    def __init__(self) -> None:
        self.records: list[study.FitRecord] = []

    def hazard(self, label: str, frame: pd.DataFrame, **settings: Any) -> Any:
        result, record = study.fit_recorded(label, frame, **settings)
        self.records.append(record)
        return result

    def untargeted(self, label: str, frame: pd.DataFrame, **settings: Any) -> Any:
        """The untargeted backward pass: its warnings are recorded; it has no walk to converge."""
        result, record = study.recorded(lambda: study.untargeted_fit(label, frame, **settings))
        self.records.append(study.FitRecord(True, record.unexpected_warnings))
        return result

    def iterated(self, label: str, frame: pd.DataFrame, **settings: Any) -> Any:
        """An iterated-mean fit: its warnings are recorded, its convergence is not the subject's."""
        result, record = study.recorded(lambda: iterated_fit(label, frame, **settings))
        self.records.append(study.FitRecord(True, record.unexpected_warnings))
        return result

    def columns(self) -> dict[str, float]:
        def extreme(values: list[float], pick: Any) -> float:
            finite = [value for value in values if np.isfinite(value)]
            return float(pick(finite)) if finite else float("nan")

        hazards = [r for r in self.records if np.isfinite(r.smallest_kish)]
        return {
            "converged": int(all(record.converged for record in self.records)),
            "final_score": max((r.final_score for r in self.records), default=0.0),
            "unexpected_warnings": sum(len(r.unexpected_warnings) for r in self.records),
            "smallest_kish": extreme([r.smallest_kish for r in hazards], min),
            "largest_relative_sd": extreme([r.largest_relative_sd for r in hazards], max),
            "largest_edgeworth_loss": extreme([r.largest_edgeworth_loss for r in hazards], max),
        }


stalled_fit = held.stalled_fit


def _difference_rows(
    frame: pd.DataFrame,
    common_row: dict[str, Any],
    shared: dict[str, Any],
    label: str,
    records: FitRecords,
) -> list[dict[str, Any]]:
    hazard = records.hazard(label, frame)
    iterated = records.iterated(label, frame)
    isotonic = records.iterated(label, frame, isotonic=True)
    rows: list[dict[str, Any]] = []
    for cell_label in study.DIFFERENCE_LABELS:
        name = ESTIMAND[cell_label]
        for arm, result in (("hazard_fit", hazard), ("iterated_fit", iterated)):
            rows.append(
                replicate_row(
                    property_name=DIFFERENCE_FAMILY,
                    cell=f"{cell_label}__{arm}",
                    role=DIAGNOSTIC_ROLE,
                    truth=TRUTHS[cell_label],
                    estimate=result[name],
                    alpha=STUDY.margins.alpha,
                    **shared,
                )
            )
        for arm, other in (("paired_difference", iterated), ("isotonic_difference", isotonic)):
            rows.append(
                _paired_row(
                    f"{cell_label}__{arm}", hazard[name], float(other[name].psi), common_row
                )
            )
    drop, error = largest_decrease(iterated)
    rows.append(_reading_row(f"{label}__iterated_curve_decrease", drop, error, common_row))
    return rows


def fit_drawn(payload: _Payload, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The replicate rows of one fit of one fit set on its drawn sample.

    Every row carries :data:`CONVERGENCE_COLUMNS` of the fits the key ran.
    """
    records = FitRecords()
    rows = _fit_drawn(payload, frame, records)
    columns = records.columns()
    return [{**row, **columns} for row in rows]


def _fit_drawn(payload: _Payload, frame: pd.DataFrame, records: FitRecords) -> list[dict[str, Any]]:
    family, _, configuration, replicate, n, requested, label = payload
    common_row = {"replicate": replicate, "n": n, "requested_replicates": requested}
    shared: dict[str, Any] = {"replicate": replicate, "n": n, "requested": requested}

    def row(cell_label: str, cell: str, role: str, estimate: Any, name: str = "") -> dict[str, Any]:
        return replicate_row(
            property_name=name or family,
            cell=cell,
            role=role,
            truth=TRUTHS[cell_label],
            estimate=estimate,
            alpha=STUDY.margins.alpha,
            **shared,
        )

    if family == DIFFERENCE_FAMILY:
        return _difference_rows(frame, common_row, shared, label, records)
    if family == "interval_calibration":
        joint = f"{label}_curve"
        banded = joint in JOINT_LABELS
        result = records.hazard(label, frame, simultaneous=banded)
        rows = [
            row(cell_label, f"{cell_label}__correctly_specified", "positive", result[name])
            for cell_label, name in COEFFICIENTS[label]
            if cell_label in CALIBRATION_LABELS
        ]
        if banded:
            rows.extend(
                joint_coverage_rows(
                    result,
                    study.TRUTH["multinode"],
                    tuple(result.estimates),
                    label=joint,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    pointwise_critical=CRITICAL,
                )
            )
        if label == "multinode":
            rows.append(
                row(
                    CONTRAST,
                    f"{CONTRAST}__alternative",
                    "positive",
                    result[ESTIMAND[CONTRAST]],
                    "power",
                )
            )
        return rows
    result = records.hazard(label, frame, configuration=configuration)
    if family == "root_n_and_efficiency":
        role = "control" if n == min(RATE_SIZES) else "positive"
        return [
            row(cell_label, f"{cell_label}__n_{n}", role, result[ESTIMAND[cell_label]])
            for cell_label in study.LADDER_LABELS
        ]
    if family != "double_robustness":
        raise ValueError(f"no fit set of family {family!r}")
    role = "control" if configuration == "both_wrong" else "positive"
    rows = [
        row(cell_label, f"{cell_label}__{configuration}", role, result[ESTIMAND[cell_label]])
        for cell_label in CONFIGURATION_ARMS[configuration]
    ]
    if configuration == NECESSITY_CONFIGURATION:
        estimate = result[ESTIMAND[CONTRAST]]
        plug_in = records.untargeted(label, frame, configuration=configuration)
        rows += [
            row(CONTRAST, f"{CONTRAST}__targeted", "positive", estimate, "targeting_necessity"),
            control_row(
                property_name="targeting_necessity",
                cell=f"{CONTRAST}__untargeted",
                truth=TRUTHS[CONTRAST],
                estimate=float(plug_in[ESTIMAND[CONTRAST]].psi),
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
                **shared,
            ),
        ]
    return rows


def _payloads(
    fit_sets: Sequence[tuple[str, str, str, int, int, str]], budget: int | None
) -> list[_Key]:
    out: list[_Key] = []
    for family, stream, configuration, n, replicates, label in fit_sets:
        requested = replicates if budget is None else budget
        out.extend(
            ((family, stream, configuration, r, n, requested, label),) for r in range(requested)
        )
    return out


#: The share of a cell's requested replications that the accounting lets a run lose.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_drawn(key[0], frame)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = fit_properties(
        _payloads(FIT_SETS, budget),
        _fit_key,
        cells_of=cells_of,
        cell_of=cell_of,
        replicate_of=replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
        extra=stalled_fit,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CONTROLLED_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.reindex(columns=[*REPLICATE_COLUMNS, *CONVERGENCE_COLUMNS]).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def synthetic_scale(cell: PropertyCell) -> float:
    """The standard deviation of a synthetic cell's estimates for the property differential."""
    label = cell.cell.split("__", 1)[0]
    sd = EFFICIENCY_SD.get(label, 1.0)
    return float(sd / np.sqrt(cell.n))


TARGETING_NECESSITY = property_verdicts.Necessity(
    family="targeting_necessity",
    labels=(CONTRAST,),
    arms=("targeted", "untargeted"),
    column="targeting_displacement",
    threshold=TARGETING_DISPLACEMENT,
)


def difference_readings(summary: pd.DataFrame) -> None:
    """Publish the ``iterated_mean_difference`` rows with no verdict.

    Each paired row gains ``paired_sd_over_bound``: the empirical SD of the paired difference
    over the estimate's efficiency-bound standard error at the row's ``n``.
    """
    mask = summary["property"] == DIFFERENCE_FAMILY
    paired = ("__paired_difference", "__isotonic_difference")
    for index in summary.index[mask]:
        cell = str(summary.loc[index, "cell"])
        if cell.endswith(paired):
            label = cell.split("__", 1)[0]
            bound = EFFICIENCY_SD[label] / np.sqrt(float(summary.loc[index, "n"]))
            summary.loc[index, "paired_sd_over_bound"] = (
                float(summary.loc[index, "empirical_se"]) / bound
            )
    reported_verdicts(summary, families=(DIFFERENCE_FAMILY,))


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the study's declared rules, and the reported rows."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "paired_sd_over_bound", *CONVERGENCE_SUMMARY),
        rate_labels=study.LADDER_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(summary, rows, TARGETING_NECESSITY)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    difference_readings(summary)
    held.convergence_reading(summary, rows)
    return finish(summary, rates)


#: The failure accounting's declared rules.  ``iterated_mean_difference`` states no verdict.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    *property_verdicts.necessity_rules(TARGETING_NECESSITY),
)
