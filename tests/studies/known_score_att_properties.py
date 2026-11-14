r"""Property families of ``known-score-att`` (X35 plan section 5, rev 2.1), and verdicts.

The cells are fixed before any run:

=====================================  =========  =====  =====  ===============================
cell                                   role       n      R      rule
=====================================  =========  =====  =====  ===============================
``interval_calibration/<label>``       positive   2,000  K      shared calibration rule, with
                                                                the efficiency band and the
                                                                derived controls
``efficiency_gain/att, atc``           positive   2,000  K      99% upper bound of the paired
                                                                SD ratio below
                                                                :data:`GAIN_BOUNDARY`
``efficiency_gain/*_zero_gain``        control    2,000  K      the same upper bound at or
                                                                above :data:`GAIN_BOUNDARY`
``known_score_accuracy/<cell>``        positive   2,000  A      bias margin, coverage floor,
                                                                SE sanity band
``known_score_accuracy/s_att__``       control    2,000  A      bias interval beyond the margin
``declared_constant``
``root_n_and_efficiency/n_<size>``     positive   ladder A      the registry's root-n rule
=====================================  =========  =====  =====  ===============================

K is :data:`CALIBRATION_REPLICATES` and A is :data:`ACCURACY_REPLICATES`.

**Calibration.**  One Law S draw per replication fits the known-score TMLE in sample, the
same fit with ``conditional_form="arm_mean"``, and the known-score TMLE stacked over five
folds with ``q_bounds=(-8, 8)``; each fit reports the ATT and the ATC.  The six labels
``<estimand>_<form>_<splitting>`` read the shared calibration rule
(:func:`~tests.studies.evidence.property_verdicts.calibration_verdicts`) with the exact
efficiency bound of their own curve (:data:`EFFICIENCY_SD`): :math:`D_{KS}` for the
known-score labels and :math:`D_{AM}` for the arm-mean labels.  Both efficiency ratios must lie
in :data:`EFFICIENCY_RATIO_BAND`.  Each label carries the derived ``shrunken_se_control`` and
``noise_control``.

**Efficiency gain.**  ``efficiency_gain/<label>__known_score`` over
``efficiency_gain/<label>__arm_mean`` is the paired spread ratio of the two forms on one draw
(:func:`paired_sd_ratio_below_boundary`).  The ``att`` and ``atc`` pairs repeat the in-sample
calibration fits; the ``zero_gain`` pairs fit Law S0, where :math:`\tau` is constant, the
exact ratio is one and the known-score form gains nothing.  The arm-mean rows are published
under :data:`~tests.studies.evidence.property_verdicts.DIAGNOSTIC_ROLE` and carry no verdict.

The boundary is :data:`GAIN_BOUNDARY`, not one.  On Law S0 the two forms differ only by a term
of order :math:`1/n` (their estimates correlate at 0.9999 in the pilot), so the paired interval
is narrow enough to resolve that term at K, and a boundary at one would let the zero-gain
control pass or fail on it (the pilot's ATC pair reads a ratio of 0.9992).  At 0.98 the
control holds (reads its boundary) whenever the two forms' spreads agree within 2%, and the
positives must show a gain of at least 2% (the exact gains are 9.9% and 9.2%).  The control
guards the instrument against a known-score path that spreads less than the arm-mean path
where no gain exists, such as a skipped or mis-scaled targeting step on that path; the
calibration SE ratio sees the same defect, so the control is a check of the instrument.

**Accuracy.**  The known-score ATT and ATC with the wrong outcome regression, on Law S, on
Law B with two arms, and on three-arm Law B (both contrasts against arm 0), and the control:
the Law S ATT with the correct outcome regression and the declared constant mechanism
:data:`~tests.studies.known_score_att_law.CONTROL_CONSTANT`, whose plug-in tends to the ATE.

**Root n.**  The known-score ATT on Law S with the correct outcome regression at n = 500, 2,000
and 8,000; the shared rule reads each rung and the two slopes.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import known_score_att as study
from tests.studies import known_score_att_law as law
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    paired_spread_ratio_interval,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
SHRUNKEN_SE_FACTOR = 0.70

CALIBRATION = "interval_calibration"
GAIN = "efficiency_gain"
ACCURACY = "known_score_accuracy"
RATE = "root_n_and_efficiency"

CALIBRATION_N = 2_000
#: K: the larger of two sizes, each the smallest multiple of 400 at which the six calibration
#: positives pass together with probability at least the calibration instrument's share of the
#: study-wide allocation, :data:`tests.studies.repeated_sizing.SIZED_SHARE` (0.9837).  At target
#: (SE ratio one, nominal coverage, a product) that is 3,200.  The arm-pooled predictive rule
#: (:func:`tests.studies.repeated_sizing.calibration_replicates`) at the committed 8,000-draw
#: pilot, whose SE ratios are 0.990 and 0.992 (known score, in sample), 0.992 and 0.994
#: (stacked) and 0.985 and 0.997 (arm mean), with the exact curve correlations, raises it to
#: this count.  The design test recomputes both.
CALIBRATION_REPLICATES = 4_800
#: The zero-gain pairs run at K, so the control reads the instrument the positives read.
ZERO_GAIN_REPLICATES = CALIBRATION_REPLICATES
ACCURACY_N = 2_000
#: A: the smallest multiple of 200, at least the primaries' 1,000, at which the accuracy family
#: passes with probability at least its share of the property instrument,
#: ``repeated_sizing.family_share(2)`` (0.9918, two property families: accuracy and the root-n
#: ladder): the property bias rule
#: (:func:`tests.studies.repeated_sizing.property_bias_pass_probability`) at the 2,000-draw
#: pilot, times each positive's coverage floor at the one-sided 80% lower bound of its pilot
#: coverage and its SE band, times the control's discrimination.  The design test recomputes
#: it.
ACCURACY_REPLICATES = 1_400
RATE_SIZES = study.RATE_SIZES
#: The same rule over the three rungs' clauses and the two slopes.
RATE_REPLICATES = 1_600

CALIBRATION_LABELS = study.CALIBRATION_LABELS
GAIN_LABELS = study.GAIN_LABELS
#: Each calibration label's estimand and setting.
LABEL_SETTINGS: dict[str, tuple[str, str]] = {
    "att_known_score_in_sample": ("att", "s_known_score"),
    "atc_known_score_in_sample": ("atc", "s_known_score"),
    "att_known_score_stacked": ("att", "s_known_score_stacked"),
    "atc_known_score_stacked": ("atc", "s_known_score_stacked"),
    "att_arm_mean_in_sample": ("att", "s_arm_mean"),
    "atc_arm_mean_in_sample": ("atc", "s_arm_mean"),
}
#: The exact SD of each label's curve at the truth, ``D_KS`` or ``D_AM``
#: (:func:`tests.studies.known_score_att_law.efficiency_sd`; the design test pins it at 1e-6).
EFFICIENCY_SD = {
    "att_known_score_in_sample": 2.874545,
    "atc_known_score_in_sample": 2.898891,
    "att_known_score_stacked": 2.874545,
    "atc_known_score_stacked": 2.898891,
    "att_arm_mean_in_sample": 3.188869,
    "atc_arm_mean_in_sample": 3.192232,
}
#: The band both efficiency ratios of a calibration positive must lie in, the registry's.
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

#: The paired gate's boundary on the SD ratio known score over arm mean.
GAIN_BOUNDARY = 0.98
#: Each gain label's numerator and denominator settings and its estimand.
GAIN_SETTINGS: dict[str, tuple[str, str, str]] = {
    "att": ("att", "s_known_score", "s_arm_mean"),
    "atc": ("atc", "s_known_score", "s_arm_mean"),
    "att_zero_gain": ("att", "s0_known_score", "s0_arm_mean"),
    "atc_zero_gain": ("atc", "s0_known_score", "s0_arm_mean"),
}
#: The exact curve SDs of each gain label's two forms, ``(known score, arm mean)``.
GAIN_SD = {
    "att": (2.874545, 3.188869),
    "atc": (2.898891, 3.192232),
    "att_zero_gain": (2.541461, 2.541461),
    "atc_zero_gain": (2.566857, 2.566857),
}

#: ``cell -> (setting, reported estimand, role)`` of the accuracy family.
ACCURACY_SETTINGS: dict[str, tuple[str, str, str]] = {
    "s_att__q_wrong": ("s_wrong_known_score", "att", "positive"),
    "s_atc__q_wrong": ("s_wrong_known_score", "atc", "positive"),
    "b2_att__q_wrong": ("b2_wrong_known_score", "att", "positive"),
    "b2_atc__q_wrong": ("b2_wrong_known_score", "atc", "positive"),
    "b3_att_1__q_wrong": ("b3_wrong_known_score", "att[1.0 vs 0.0]", "positive"),
    "b3_att_2__q_wrong": ("b3_wrong_known_score", "att[2.0 vs 0.0]", "positive"),
    "b3_atc_1__q_wrong": ("b3_wrong_known_score", "atc[1.0 vs 0.0]", "positive"),
    "b3_atc_2__q_wrong": ("b3_wrong_known_score", "atc[2.0 vs 0.0]", "positive"),
    "s_att__declared_constant": ("s_declared_constant", "att", "control"),
}
#: The accuracy draws: each kind's setting and the cells its one fit fills.
ACCURACY_KINDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "accuracy_s": ("s_wrong_known_score", ("s_att__q_wrong", "s_atc__q_wrong")),
    "accuracy_b2": ("b2_wrong_known_score", ("b2_att__q_wrong", "b2_atc__q_wrong")),
    "accuracy_b3": (
        "b3_wrong_known_score",
        ("b3_att_1__q_wrong", "b3_att_2__q_wrong", "b3_atc_1__q_wrong", "b3_atc_2__q_wrong"),
    ),
    "control": ("s_declared_constant", ("s_att__declared_constant",)),
}

#: The summary columns this study adds.
EXTRA_COLUMNS = (
    "spread_ratio",
    "spread_ratio_ci_lower",
    "spread_ratio_ci_upper",
    "spread_ratio_boundary",
)

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def _seed(stream: str, replicate: int) -> int:
    """The seed of one replication of a named sample stream."""
    return stream_seed(STUDY, "property_sample", stream, replicate)


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: Any,
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


# ----------------------------------------------------------------------------- the kinds
def _calibration_draw(replicate: int) -> pd.DataFrame:
    return law.draw("S", CALIBRATION_N, _seed("calibration", replicate))


def _calibration(replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    fits = {
        name: law.fit(frame, name)
        for name in ("s_known_score", "s_arm_mean", "s_known_score_stacked")
    }
    truth = law.setting_truth("s_known_score")
    rows = [
        _row(
            CALIBRATION,
            f"{label}__correctly_specified",
            "positive",
            replicate,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            truth[estimand],
            fits[setting][estimand],
        )
        for label, (estimand, setting) in LABEL_SETTINGS.items()
    ]
    for label in ("att", "atc"):
        estimand, numerator, denominator = GAIN_SETTINGS[label]
        for form, setting, role in (
            ("known_score", numerator, "positive"),
            ("arm_mean", denominator, DIAGNOSTIC_ROLE),
        ):
            rows.append(
                _row(
                    GAIN,
                    f"{label}__{form}",
                    role,
                    replicate,
                    CALIBRATION_N,
                    CALIBRATION_REPLICATES,
                    truth[estimand],
                    fits[setting][estimand],
                )
            )
    return rows


def _zero_gain_draw(replicate: int) -> pd.DataFrame:
    return law.draw("S0", CALIBRATION_N, _seed("zero_gain", replicate))


def _zero_gain(replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    fits = {name: law.fit(frame, name) for name in ("s0_known_score", "s0_arm_mean")}
    truth = law.setting_truth("s0_known_score")
    rows = []
    for label in ("att_zero_gain", "atc_zero_gain"):
        estimand, numerator, denominator = GAIN_SETTINGS[label]
        for form, setting, role in (
            ("known_score", numerator, "control"),
            ("arm_mean", denominator, DIAGNOSTIC_ROLE),
        ):
            rows.append(
                _row(
                    GAIN,
                    f"{label}__{form}",
                    role,
                    replicate,
                    CALIBRATION_N,
                    ZERO_GAIN_REPLICATES,
                    truth[estimand],
                    fits[setting][estimand],
                )
            )
    return rows


def _accuracy_draw(payload: tuple[str, int]) -> pd.DataFrame:
    kind, replicate = payload
    setting = ACCURACY_KINDS[kind][0]
    return law.sample(law.SETTINGS[setting], ACCURACY_N, _seed(kind, replicate))


def _accuracy(payload: tuple[str, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    kind, replicate = payload
    setting, cells = ACCURACY_KINDS[kind]
    result = law.fit(frame, setting)
    truth = law.setting_truth(setting)
    rows = []
    for cell in cells:
        _, name, role = ACCURACY_SETTINGS[cell]
        rows.append(
            _row(
                ACCURACY,
                cell,
                role,
                replicate,
                ACCURACY_N,
                ACCURACY_REPLICATES,
                truth[name],
                result[name],
            )
        )
    return rows


def _rate_draw(payload: tuple[int, int]) -> pd.DataFrame:
    size, replicate = payload
    return law.draw("S", size, _seed(f"rate_n_{size}", replicate))


def _rate(payload: tuple[int, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    size, replicate = payload
    result = law.fit(frame, "s_known_score", estimands=("att",))
    return [
        _row(
            RATE,
            f"n_{size}",
            "positive",
            replicate,
            size,
            RATE_REPLICATES,
            law.truth("S")["att"],
            result["att"],
        )
    ]


#: Each payload kind's draw and fit.
_KINDS: dict[str, tuple[Any, Any]] = {
    "calibration": (_calibration_draw, _calibration),
    "zero_gain": (_zero_gain_draw, _zero_gain),
    "accuracy": (_accuracy_draw, _accuracy),
    "rate": (_rate_draw, _rate),
}


def _dispatch_draw(payload: tuple[str, Any]) -> Any:
    kind, argument = payload
    return _KINDS[kind][0](argument)


def _dispatch(payload: tuple[str, Any], drawn: Any) -> list[dict[str, Any]]:
    kind, argument = payload
    return _KINDS[kind][1](argument, drawn)


_Key = tuple[tuple[str, Any]]


def payloads(budget: int | None = None) -> list[_Key]:
    """Every replication key, in a fixed order; ``budget`` caps each cell for a smoke run."""

    def count(declared: int) -> int:
        return declared if budget is None else min(declared, budget)

    keys: list[tuple[str, Any]] = []
    keys += [("calibration", r) for r in range(count(CALIBRATION_REPLICATES))]
    keys += [("zero_gain", r) for r in range(count(ZERO_GAIN_REPLICATES))]
    keys += [
        ("accuracy", (kind, r))
        for kind in ACCURACY_KINDS
        for r in range(count(ACCURACY_REPLICATES))
    ]
    keys += [("rate", (size, r)) for size in RATE_SIZES for r in range(count(RATE_REPLICATES))]
    return [(key,) for key in keys]


def _cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows and derived controls fill."""
    kind, argument = key[0]
    if kind == "calibration":
        names = []
        for label in CALIBRATION_LABELS:
            positive = f"{CALIBRATION}/{label}__correctly_specified"
            names += [positive, *DERIVED_CELLS[positive]]
        names += [
            f"{GAIN}/{label}__{form}"
            for label in ("att", "atc")
            for form in ("known_score", "arm_mean")
        ]
        return names
    if kind == "zero_gain":
        return [
            f"{GAIN}/{label}__{form}"
            for label in ("att_zero_gain", "atc_zero_gain")
            for form in ("known_score", "arm_mean")
        ]
    if kind == "accuracy":
        return [f"{ACCURACY}/{cell}" for cell in ACCURACY_KINDS[argument[0]][1]]
    return [f"{RATE}/n_{argument[0]}"]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    kind, argument = key[0]
    return int(argument[1] if kind in ("accuracy", "rate") else argument)


# ----------------------------------------------------------------------- declared cells
class DeclaredLaw:
    """The law a declared cell reads, with its exact truths.

    Parameters
    ----------
    setting : str
        A key of :data:`tests.studies.known_score_att_law.SETTINGS`.
    """

    def __init__(self, setting: str) -> None:
        self.setting = setting
        self.name = f"known_score_att__{law.SETTINGS[setting].law}"

    def truth(self) -> dict[str, float]:
        return dict(law.setting_truth(self.setting))


def _cell(
    family: str,
    cell: str,
    setting: str,
    n: int,
    replicates: int,
    role: str,
    estimand: str,
    stream: str,
) -> PropertyCell:
    return PropertyCell(
        property=family,
        cell=cell,
        dgp=DeclaredLaw(setting),
        outcome_learner=lambda: None,
        treatment_learner=lambda: None,
        n=n,
        replicates=replicates,
        seed=_seed(stream, 0),
        role=role,
        estimand=estimand,
        fit_kwargs={"setting": setting},
    )


def _fitted_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, size and replicate count."""
    cells = [
        _cell(
            CALIBRATION,
            f"{label}__correctly_specified",
            setting,
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            "positive",
            estimand,
            "calibration",
        )
        for label, (estimand, setting) in LABEL_SETTINGS.items()
    ]
    for label, (estimand, numerator, denominator) in GAIN_SETTINGS.items():
        zero = label.endswith("zero_gain")
        for form, setting, role in (
            ("known_score", numerator, "control" if zero else "positive"),
            ("arm_mean", denominator, DIAGNOSTIC_ROLE),
        ):
            cells.append(
                _cell(
                    GAIN,
                    f"{label}__{form}",
                    setting,
                    CALIBRATION_N,
                    ZERO_GAIN_REPLICATES if zero else CALIBRATION_REPLICATES,
                    role,
                    estimand,
                    "zero_gain" if zero else "calibration",
                )
            )
    for kind, (setting, names) in ACCURACY_KINDS.items():
        for cell in names:
            _, estimand, role = ACCURACY_SETTINGS[cell]
            cells.append(
                _cell(
                    ACCURACY, cell, setting, ACCURACY_N, ACCURACY_REPLICATES, role, estimand, kind
                )
            )
    cells += [
        _cell(
            RATE,
            f"n_{size}",
            "s_known_score",
            size,
            RATE_REPLICATES,
            "positive",
            "att",
            f"rate_n_{size}",
        )
        for size in RATE_SIZES
    ]
    return tuple(cells)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, and the calibration controls derived from its rows."""
    fitted = _fitted_cells()
    by_name = {f"{cell.property}/{cell.cell}": cell for cell in fitted}
    derived = tuple(
        dataclasses.replace(
            by_name[parent], property=name.split("/")[0], cell=name.split("/")[1], role="control"
        )
        for parent, names in DERIVED_CELLS.items()
        for name in names
    )
    return fitted + derived


def synthetic_scale(cell: PropertyCell) -> float:
    """The SD of a cell's estimates at its exact curve SD, for the property differential."""
    if cell.property == CALIBRATION:
        return EFFICIENCY_SD[cell.cell.split("__", 1)[0]] / float(np.sqrt(cell.n))
    if cell.property == GAIN:
        label, form = cell.cell.split("__", 1)
        return GAIN_SD[label][0 if form == "known_score" else 1] / float(np.sqrt(cell.n))
    return float(cell.n) ** -0.5


# -------------------------------------------------------------------------- generation
def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a smoke run."""
    outcomes = fit_properties(
        payloads(budget),
        splat_drawn(_dispatch),
        cells_of=_cells_of,
        draw=splat_draw(_dispatch_draw),
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    if budget is not None:
        # A capped smoke run requests what it runs, so the summary reads it as complete.
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    controls = calibration_controls(
        rows,
        STUDY,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return (
        pd.concat([rows, controls], ignore_index=True)
        .loc[:, list(REPLICATE_COLUMNS)]
        .sort_values(["property", "cell", "replicate"], ignore_index=True)
    )


# ----------------------------------------------------------------------------- verdicts
def _gain_seed(record: Any, label: str) -> int:
    return stream_seed(record, GAIN, label)


def paired_sd_ratio_below_boundary(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The ``efficiency_gain`` verdicts: the paired SD ratio against :data:`GAIN_BOUNDARY`.

    For each label the ratio is SD(known score) / SD(arm mean) over the label's paired
    replications, with the 99% paired-bootstrap interval of
    :func:`~tests.studies.evidence.properties.paired_spread_ratio_interval` on the study's
    stream ``efficiency_gain/<label>``.  A positive passes when the upper end lies below the
    boundary, and a zero-gain control when the upper end reaches it.  Both rows of a label
    publish the interval; the arm-mean row is diagnostic and states no verdict.
    """
    margins = STUDY.margins
    selected = rows.loc[rows["property"] == GAIN]
    for label in GAIN_LABELS:
        numerator = selected.loc[selected["cell"] == f"{label}__known_score"]
        denominator = selected.loc[selected["cell"] == f"{label}__arm_mean"]
        if numerator.empty or denominator.empty:
            continue
        result = paired_spread_ratio_interval(
            numerator,
            denominator,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=_gain_seed(STUDY, label),
        )
        control = label.endswith("zero_gain")
        high = result.interval.high
        passed = bool(high >= GAIN_BOUNDARY) if control else bool(high < GAIN_BOUNDARY)
        for form in ("known_score", "arm_mean"):
            mask = (summary["property"] == GAIN) & (summary["cell"] == f"{label}__{form}")
            summary.loc[mask, "spread_ratio"] = result.ratio
            summary.loc[mask, "spread_ratio_ci_lower"] = result.interval.low
            summary.loc[mask, "spread_ratio_ci_upper"] = high
            summary.loc[mask, "spread_ratio_boundary"] = GAIN_BOUNDARY
            summary.loc[mask, "passed"] = passed if form == "known_score" else True


def _accuracy_verdicts(summary: pd.DataFrame) -> None:
    margins = STUDY.margins
    family = summary["property"] == ACCURACY
    positive = family & (summary["role"] == "positive")
    summary.loc[positive, "passed"] = (
        summary.loc[positive, "bias_equivalent"].astype(bool)
        & (summary.loc[positive, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[positive, "se_ratio"].between(*margins.se_ratio_sanity)
    )
    control = family & (summary["role"] == "control")
    summary.loc[control, "passed"] = summary.loc[control, "bias_discriminated"].astype(bool)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the calibration band, the accuracy rule and the paired gate."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=EXTRA_COLUMNS, efficiency_bounds=EFFICIENCY_SD
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    _accuracy_verdicts(summary)
    paired_sd_ratio_below_boundary(summary, rows)
    return finish(summary, rates)


def _gain_clause(label: str) -> pf.Clause:
    control = label.endswith("zero_gain")

    def boundary(ctx: pf.Context) -> float:
        return GAIN_BOUNDARY

    return pf.spread_ratio_clause(
        f"{label} spread ratio {'reaches' if control else 'under'} its boundary",
        cells=(f"{label}__known_score", f"{label}__arm_mean"),
        side="lower" if control else "upper",
        judge=pf.upper_at_or_above_judge(boundary) if control else pf.below_judge(boundary),
        seed=lambda ctx, row, label=label: _gain_seed(ctx.record, label),
        applies=pf.cells(f"{label}__known_score"),
    )


#: The failure accounting's declared rules.  The calibration rule carries this study's
#: efficiency band; ``root_n_and_efficiency`` and ``root_n_rate`` read the shared rules
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES: tuple[Any, ...] = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    property_verdicts.accuracy_rule(ACCURACY, shared="_accuracy_verdicts"),
    pf.FamilyRule(
        families=frozenset({GAIN}),
        shared="paired_sd_ratio_below_boundary",
        clauses=tuple(_gain_clause(label) for label in GAIN_LABELS),
    ),
)


def smoke_keys(draws: int) -> list[_Key]:
    """Keys of a smoke on the first ``draws`` replicates of every cell."""
    return payloads(draws)


def fit_keys(keys: list[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [_dispatch(key[0], _dispatch_draw(key[0])) for key in keys]


__all__ = [
    "CALIBRATION_REPLICATES",
    "DERIVED_CELLS",
    "FAILURE_BOUND",
    "GAIN_BOUNDARY",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "declared_cells",
    "generate_property_rows",
    "paired_sd_ratio_below_boundary",
    "summarize_properties",
]
