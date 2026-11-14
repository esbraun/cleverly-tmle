"""Property families of ``known-mechanism-sample-weights`` (Fix-16 plan section 7), and verdicts.

The cells are fixed before any run.  Each stream is one law at one size; one draw of a stream
is fitted once per fit, and every cell of the stream reads those fits, so the cells of one size
always share their draw.  The shared draws are the paired design of
``docs/development/method-benchmarking.md`` ("fixed shared seeds so methods see identical draws
and paired controls merge by replication index"): each control's paired difference reads them.

=========================================  ==========  ===========  ======================
cell                                       role        fit          rule
=========================================  ==========  ===========  ======================
``interval_calibration/<label>__correctly  positive    correct      shared calibration
_specified``                                                        rule, with the derived
                                                                    controls
``sample_bias/<label>``                    positive    correct      bias inside 0.25 SD
``target_reading_control/<label>``         control     ``target``   bias beyond 0.25 SD
``sample_misspecified/<label>``            positive    outcome      bias inside 0.25 SD
                                                       wrong
``tilt_misspecified/sw_n4000``             positive    tilt wrong   bias inside 0.25 SD
``both_misspecified/sw_n4000``             control     both wrong   bias beyond 0.25 SD
=========================================  ==========  ===========  ======================

The labels are ``sw_n1000``, ``sw_n4000``, ``sw_n16000`` (law SW, point TMLE) and ``lw_n4000``
(law LW, LTMLE).  The fits are named in :data:`FITS`
(:data:`tests.studies.known_sample_weights_law.SETTINGS` and
:data:`~tests.studies.known_sample_weights_law.LW_SETTINGS`).  On law SW the control and the
outcome-wrong positive use the main-terms outcome regression; on law LW they use the
intercept-only node-one regression.  Every point cell reads the ATE and the longitudinal cells
read the regimen contrast.

**Rules.**

- ``interval_calibration`` reads the shared calibration rule
  (:func:`~tests.studies.evidence.property_verdicts.calibration_verdicts`) with the derived
  ``shrunken_se_control`` and ``noise_control``; the noise arm adds one unit of the exact SD of
  the gated fit's curve (:data:`EFFICIENCY_SD`).  The study claims no efficiency band.
- Every other family reads
  :func:`~tests.studies.evidence.property_verdicts.bias_role_verdicts`: a positive's 99%
  interval of the standardized bias inside the 0.25 margin, a control's wholly beyond it.  Their
  rows publish the SE ratio and coverage too, ungated: with one nuisance wrong the shipped
  interval is not claimed.

``sample_misspecified`` is the study's direct witness of the tilt: with the outcome regression
wrong, only a correct tilted mechanism removes the bias, and a reverted or broken tilt shows as
bias there.  ``tilt_misspecified`` witnesses the other half of the double robustness, and
``both_misspecified`` shows that its wrong tilt sits on a term that matters.

Each control row also publishes, ungated, its exact-law bias (``exact_bias``,
:data:`EXACT_CONTROL_BIAS`) and the mean paired difference of the control and its partner on
the same draws over the control's SD (``paired_difference_over_sd``, :data:`CONTROL_PAIRS`): the
part of the control's bias that the partner's one correct nuisance removes (plan rev 2.1 item
1, review m5).

**Smoke gate.**  :func:`smoke_findings` fits the first draws of every stream in this process and
returns every finding; ``python -m tests.studies.known_sample_weights_properties smoke`` exits
non-zero on any.  A finding is a cell without its rows, a non-finite estimate or standard error,
a ``"sample"`` fit that reads its mechanism as known (point) or records a known factor
(LTMLE), or a ``"target"`` fit that does not read both as declared.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_known_sample_weights as study
from tests.studies import known_sample_weights_law as law
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    bias_role_rule,
    bias_role_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
SHRUNKEN_SE_FACTOR = 0.70

POINT_LABELS = study.POINT_LABELS
LONGITUDINAL_LABEL = study.LONGITUDINAL_LABEL
CALIBRATION_LABELS = study.CALIBRATION_LABELS
#: Each label's sample size.
SIZES: dict[str, int] = {
    **dict(zip(POINT_LABELS, study.POINT_SIZES, strict=True)),
    LONGITUDINAL_LABEL: study.LONGITUDINAL_N,
}
#: Each label's replications, the same for every cell of the label.  Set by the declaration
#: from the committed 8,000-draw pilot (``tests/studies/known_sample_weights_pilot.csv``) by the
#: rule the design test recomputes: the smallest common multiple of 400 at which the product of
#: every gated cell's pass probability is at least 0.95, each read at the measured SE ratio,
#: coverage, bias and per-draw SE-ratio spread, with the SE ratio's interval scaled on the log
#: scale, and averaged over the pilot's own noise.  See the design test for the values.
REPLICATES: dict[str, int] = {
    "sw_n1000": 4_000,
    "sw_n4000": 4_000,
    "sw_n16000": 4_000,
    LONGITUDINAL_LABEL: 4_000,
}

#: The exact SD of each gated fit's curve times root n, which sizes the noise arm
#: (:func:`~tests.studies.known_sample_weights_law.point_influence_sd` and
#: :func:`~tests.studies.known_sample_weights_law.longitudinal_influence_sd`; the design test
#: pins them).
EFFICIENCY_SD: dict[str, float] = {
    **dict.fromkeys(POINT_LABELS, 1.072893),
    LONGITUDINAL_LABEL: 4.1765,
}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

CALIBRATION = "interval_calibration"
BIAS_FAMILY = "sample_bias"
CONTROL_FAMILY = "target_reading_control"
OUTCOME_WRONG_FAMILY = "sample_misspecified"
TILT_WRONG_FAMILY = "tilt_misspecified"
BOTH_WRONG_FAMILY = "both_misspecified"
#: The gated bias families, read by role.
BIAS_FAMILIES = (
    BIAS_FAMILY,
    CONTROL_FAMILY,
    OUTCOME_WRONG_FAMILY,
    TILT_WRONG_FAMILY,
    BOTH_WRONG_FAMILY,
)
#: The label of the tilt-wrong pair.
TILT_LABEL = "sw_n4000"
#: The summary columns this study adds.
EXTRA_COLUMNS = ("exact_bias", "paired_difference_over_sd")

#: Each control's exact-law bias (the design test pins them against
#: :func:`~tests.studies.known_sample_weights_law.control_limit`,
#: :func:`~tests.studies.known_sample_weights_law.both_wrong_limit` and
#: :func:`~tests.studies.known_sample_weights_law.lw_control_limit`); the SW control's is the
#: same at every size.
EXACT_CONTROL_BIAS: dict[tuple[str, str], float] = {
    **{(CONTROL_FAMILY, label): -0.033047 for label in POINT_LABELS},
    (CONTROL_FAMILY, LONGITUDINAL_LABEL): 0.455021,
    (BOTH_WRONG_FAMILY, TILT_LABEL): -0.034797,
}
#: Each control's partner on the same draws: the fit with the same wrong nuisance and the
#: other one correct.
CONTROL_PAIRS: dict[tuple[str, str], str] = {
    **{(CONTROL_FAMILY, label): OUTCOME_WRONG_FAMILY for label in CALIBRATION_LABELS},
    (BOTH_WRONG_FAMILY, TILT_LABEL): OUTCOME_WRONG_FAMILY,
}

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


@dataclasses.dataclass(frozen=True)
class Cell:
    """One declared fitted cell: its family, label, role and the fit it reads.

    Parameters
    ----------
    family : str
        The property family.
    label : str
        The stream label.
    role : str
        ``positive`` or ``control``.
    fit : str
        A key of :data:`~tests.studies.known_sample_weights_law.SETTINGS` or
        :data:`~tests.studies.known_sample_weights_law.LW_SETTINGS`.
    """

    family: str
    label: str
    role: str
    fit: str

    @property
    def name(self) -> str:
        """The cell name."""
        return f"{self.label}__correctly_specified" if self.family == CALIBRATION else self.label

    @property
    def key(self) -> str:
        """``family/cell``."""
        return f"{self.family}/{self.name}"


#: Each label's fits: ``(correct, control, outcome wrong)``.
FITS: dict[str, tuple[str, str, str]] = {
    **dict.fromkeys(POINT_LABELS, ("sample_correct", "target_main", "sample_main")),
    LONGITUDINAL_LABEL: ("ltmle_sample", "ltmle_target_flat", "ltmle_sample_flat"),
}


def _cells() -> tuple[Cell, ...]:
    cells = []
    for label in CALIBRATION_LABELS:
        correct, control, outcome_wrong = FITS[label]
        cells += [
            Cell(CALIBRATION, label, "positive", correct),
            Cell(BIAS_FAMILY, label, "positive", correct),
            Cell(CONTROL_FAMILY, label, "control", control),
            Cell(OUTCOME_WRONG_FAMILY, label, "positive", outcome_wrong),
        ]
    cells += [
        Cell(TILT_WRONG_FAMILY, TILT_LABEL, "positive", "tilt_wrong"),
        Cell(BOTH_WRONG_FAMILY, TILT_LABEL, "control", "both_wrong"),
    ]
    return tuple(cells)


#: Every declared fitted cell.
CELLS = _cells()


def fits_of(label: str) -> tuple[str, ...]:
    """The fits one draw of a stream makes, in a fixed order."""
    return tuple(dict.fromkeys(cell.fit for cell in CELLS if cell.label == label))


def truth_of(label: str) -> float:
    """The exact truth the label's cells read."""
    if label == LONGITUDINAL_LABEL:
        return law.longitudinal_truth()[law.CONTRAST]
    return law.truth()["ate"]


def draw(label: str, seed: int) -> pd.DataFrame:
    """One draw of a stream."""
    if label == LONGITUDINAL_LABEL:
        return law.draw_longitudinal(SIZES[label], seed)
    return law.draw(SIZES[label], seed)


def fit_one(label: str, frame: pd.DataFrame, fit: str) -> Any:
    """One fit of one draw: the single result (point) or the longitudinal result."""
    return law.fit_named(frame, fit)


def estimate_of(label: str, result: Any) -> Any:
    """The estimate a label's cells read."""
    return result[law.CONTRAST] if label == LONGITUDINAL_LABEL else result["ate"]


# ---------------------------------------------------------------------------- rows

_Key = tuple[tuple[str, int, int]]


def sample_seed(label: str, replicate: int) -> int:
    """The seed of one replication of a stream; the streams are disjoint by label."""
    return stream_seed(STUDY, "property_sample", label, replicate)


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    label, replicate, _ = key[0]
    return draw(label, sample_seed(label, replicate))


def cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names one key's rows fill, with the derived controls."""
    label = key[0][0]
    names = [cell.key for cell in CELLS if cell.label == label]
    return [*names, *(d for name in names for d in DERIVED_CELLS.get(name, ()))]


def cell_of(key: _Key) -> str:
    return f"stream/{key[0][0]}"


def replicate_of(key: _Key) -> int:
    return key[0][1]


def stream_rows(
    label: str, frame: pd.DataFrame, *, replicate: int, requested: int
) -> list[dict[str, Any]]:
    """Every row of one draw of a stream: each fit once, every cell of the stream off it."""
    truth = truth_of(label)
    estimates = {fit: estimate_of(label, fit_one(label, frame, fit)) for fit in fits_of(label)}
    return [
        replicate_row(
            property_name=cell.family,
            cell=cell.name,
            role=cell.role,
            replicate=replicate,
            n=SIZES[label],
            requested=requested,
            truth=truth,
            estimate=estimates[cell.fit],
            alpha=STUDY.margins.alpha,
        )
        for cell in CELLS
        if cell.label == label
    ]


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    label, replicate, requested = key[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stream_rows(label, frame, replicate=replicate, requested=requested)


def payloads(budget: int | None = None, start: int = 0) -> list[_Key]:
    """Every replication key; ``budget`` caps each stream for a smoke run."""
    keys: list[_Key] = []
    for label in CALIBRATION_LABELS:
        requested = REPLICATES[label] if budget is None else min(budget, REPLICATES[label])
        keys += [((label, replicate, requested),) for replicate in range(start, start + requested)]
    return keys


class DeclaredLaw:
    """The law a declared cell reads: a stream, with the truth its rows carry.

    Parameters
    ----------
    label : str
        The stream label.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.name = f"{study.SLUG}__{label}"

    def truth(self) -> dict[str, float]:
        return {"ate": truth_of(self.label)}


def _fitted_cells() -> tuple[PropertyCell, ...]:
    return tuple(
        PropertyCell(
            property=cell.family,
            cell=cell.name,
            dgp=DeclaredLaw(cell.label),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=SIZES[cell.label],
            replicates=REPLICATES[cell.label],
            seed=sample_seed(cell.label, 0),
            role=cell.role,
            estimand="ate",
            fit_kwargs={"fit": cell.fit},
        )
        for cell in CELLS
    )


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


def with_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """The fitted rows and the calibration controls derived from them, each at its own n."""
    controls = [
        calibration_controls(
            rows,
            STUDY,
            labels=(label,),
            efficiency_bounds=EFFICIENCY_SD,
            calibration_n=SIZES[label],
            shrunken_se_factor=SHRUNKEN_SE_FACTOR,
            critical=CRITICAL,
        )
        for label in CALIBRATION_LABELS
    ]
    return pd.concat([rows, *controls], ignore_index=True)


def check_cells(rows: pd.DataFrame) -> None:
    """Refuse rows whose cells differ from the declaration."""
    found = set(zip(rows["property"], rows["cell"], strict=True))
    expected = {(family, cell) for family, names in study.PROPERTY_CELLS.items() for cell in names}
    if found != expected:
        raise RuntimeError(f"cells {sorted(found ^ expected)} differ from the declaration")


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each stream for a smoke run."""
    outcomes = fit_properties(
        payloads(budget),
        _fit_key,
        cells_of=cells_of,
        cell_of=cell_of,
        replicate_of=replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        seed_of=lambda key: sample_seed(key[0][0], key[0][1]),
        derived=DERIVED_CELLS,
    )
    rows = with_controls(pd.DataFrame([row for result in outcomes for row in result]))
    rows = rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )
    check_cells(rows)
    return rows


# ---------------------------------------------------------------------------- verdicts


def _cell_rows(rows: pd.DataFrame, family: str, cell: str) -> pd.DataFrame:
    return rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]


def control_columns(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish each control's exact-law bias and its paired difference with its partner."""
    for (family, label), partner in CONTROL_PAIRS.items():
        mask = (summary["property"] == family) & (summary["cell"] == label)
        if int(mask.sum()) != 1:
            raise ValueError(f"{family}/{label} is not one summary row")
        control = _cell_rows(rows, family, label).set_index("replicate")["estimate"]
        reporting = _cell_rows(rows, partner, label).set_index("replicate")["estimate"]
        shared = control.index.intersection(reporting.index)
        difference = control.loc[shared] - reporting.loc[shared]
        summary.loc[mask, "exact_bias"] = EXACT_CONTROL_BIAS[(family, label)]
        summary.loc[mask, "paired_difference_over_sd"] = float(difference.mean()) / float(
            control.std(ddof=1)
        )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The calibration rule and its controls, and the bias rule by role."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    control_columns(summary, rows)
    return finish(summary, rates)


#: The failure accounting's declared rules.  ``interval_calibration`` reads the shared rules
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (bias_role_rule(BIAS_FAMILIES),)


def gated_cells() -> tuple[str, ...]:
    """The ``family/cell`` names whose red verdict gates the study, derived controls included."""
    fitted = tuple(cell.key for cell in CELLS)
    return (*fitted, *(name for names in DERIVED_CELLS.values() for name in names))


# ---------------------------------------------------------------------------- smoke


def smoke_keys(draws: int, start: int = 0) -> list[_Key]:
    """Keys of a smoke on replicate ids ``start`` to ``start + draws - 1`` of every stream."""
    return payloads(draws, start)


def fit_keys(keys: Sequence[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [_fit_key(key, _draw(key)) for key in keys]


def _mechanism_findings(label: str, frame: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    """The reading each fit of one draw records, and its seconds."""
    findings: list[str] = []
    seconds: dict[str, float] = {}
    for fit in fits_of(label):
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_one(label, frame, fit)
        seconds[fit] = time.perf_counter() - start
        if fit in law.LW_SETTINGS:
            reading = law.LW_SETTINGS[fit][0]
            expected_factors = () if reading == "sample" else ("A1", "A2")
            if tuple(result.config.known_factors) != expected_factors:
                findings.append(
                    f"{label}/{fit}: records known factors {result.config.known_factors}, "
                    f"expected {expected_factors}"
                )
            continue
        expected = "known" if law.SETTINGS[fit].reading == "target" else "estimated"
        if result.config.treatment_mechanism != expected:
            findings.append(
                f"{label}/{fit}: treatment_mechanism {result.config.treatment_mechanism!r}, "
                f"expected {expected!r}"
            )
    return findings, seconds


def smoke_findings(draws: int = 2) -> tuple[list[str], dict[tuple[str, str], float]]:
    """The declared smoke gate: every finding over ``draws`` draws of every stream.

    Returns the findings (empty when the gate passes) and the mean seconds of each fit of each
    stream, which the runner brief's budget reads.
    """
    findings: list[str] = []
    seconds: dict[tuple[str, str], list[float]] = {}
    rows = pd.DataFrame([row for result in fit_keys(smoke_keys(draws)) for row in result])
    rows = with_controls(rows)
    for family, names in study.PROPERTY_CELLS.items():
        for name in names:
            cell = _cell_rows(rows, family, name)
            if len(cell) != draws:
                findings.append(f"{family}/{name}: {len(cell)} rows, expected {draws}")
                continue
            values = cell[["estimate", "std_error"]].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)) or not np.all(cell["std_error"] > 0.0):
                findings.append(f"{family}/{name}: a non-finite estimate or standard error")
    for label in CALIBRATION_LABELS:
        for replicate in range(draws):
            found, timed = _mechanism_findings(label, draw(label, sample_seed(label, replicate)))
            findings += found
            for fit, value in timed.items():
                seconds.setdefault((label, fit), []).append(value)
    return findings, {key: float(np.mean(values)) for key, values in seconds.items()}


def cpu_hours(seconds: dict[tuple[str, str], float]) -> float:
    """The fit CPU-hours of the declared run at the given seconds per fit."""
    total = sum(value * REPLICATES[label] for (label, _), value in seconds.items())
    primary = seconds[
        (study.POINT_LABELS[study.POINT_SIZES.index(study.PRIMARY_N)], "sample_correct")
    ]
    return (total + primary * study.PRIMARY_REPLICATES) / 3600.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The declared smoke gate of this study.")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--draws", type=int, default=2)
    arguments = parser.parse_args(argv)
    findings, seconds = smoke_findings(arguments.draws)
    for (label, fit), value in sorted(seconds.items()):
        print(f"{label} {fit}: {value:.3f} s per fit")
    print(f"fit CPU-hours of the declared run at these timings: {cpu_hours(seconds):.2f}")
    for finding in findings:
        print(f"FINDING: {finding}")
    print("smoke gate:", "FAILED" if findings else "passed")
    return 1 if findings else 0


__all__ = [
    "CELLS",
    "DERIVED_CELLS",
    "EFFICIENCY_SD",
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "REPLICATES",
    "declared_cells",
    "gated_cells",
    "generate_property_rows",
    "smoke_findings",
    "summarize_properties",
]


if __name__ == "__main__":
    sys.exit(main())
