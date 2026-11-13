"""Property families of ``option-outcome-by-arm`` (Fix-8 plan section 5.3), and verdicts.

The cells are fixed before any run:

==========================================  =========  =====  =====  ==========================
cell                                        role       n      R      rule
==========================================  =========  =====  =====  ==========================
``interval_calibration/<label>``            positive   2,000  K      shared calibration rule,
                                                                     with the derived controls
``double_robustness/treatment_correct``     positive   2,000  1,600  bias equivalent, SE in band
``double_robustness/both_wrong``            control    2,000  1,600  bias beyond, SE in band
``pooled_outcome/<label>``                  reported   2,000  K      none
==========================================  =========  =====  =====  ==========================

The labels are ``in_sample_binary`` and ``stacked_binary`` (law ``LB``),
``in_sample_continuous`` (``LB-c``) and ``stacked_bounded`` (``LB-b``).  K is
:data:`CALIBRATION_REPLICATES`.  The two binary K cells read the same draws, and each
``pooled_outcome`` cell reads its K cell's draws, so the pooled fit and the per-arm fit differ
in the outcome regression alone.  Every cell reads the ATE.

**Rules.**

- ``interval_calibration`` reads the shared calibration rule
  (:func:`~tests.studies.evidence.property_verdicts.calibration_verdicts`) with the derived
  ``shrunken_se_control`` and ``noise_control``; the noise arm adds one unit of the exact
  efficiency bound of the cell's law (:data:`EFFICIENCY_SD`).  The study claims no efficiency
  band.
- ``double_robustness`` reads the shared union-model rule.
- ``pooled_outcome`` is reported, with
  :data:`~tests.studies.evidence.property_verdicts.DIAGNOSTIC_ROLE`.

The ``double_robustness`` and ``pooled_outcome`` rows carry their exact-law limits where the
outcome scale is fixed: ``exact_bias_over_sd`` and ``exact_se_ratio``
(:func:`tests.studies._option_by_arm_law.stacked_variance`).  The continuous pooled cell has
none, because its fluctuation runs on the sample range.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import _option_by_arm_law as law
from tests.studies import canonical_option_outcome_by_arm as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    ReplicationSpec,
    payload_cell,
    payload_replicate,
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
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
SHRUNKEN_SE_FACTOR = 0.70
TARGET = "ate"

CALIBRATION_N = 2_000
#: K's replications.  Plan 5.0's product floor for four cells is 2,800; the arm-pooled
#: predictive rule (:func:`tests.studies.repeated_sizing.calibration_replicates`) at the
#: committed 8,000-draw pilot raises it to this count.  The pilot SE ratios are 1.006 and 1.009
#: (binary, one draw stream, correlation 0.9988), 0.990 (continuous) and 0.991 (bounded); the
#: rule prices the spread between the laws.  The design test recomputes it.
CALIBRATION_REPLICATES = 6_400
ROBUSTNESS_N = 2_000
ROBUSTNESS_REPLICATES = 1_600

CALIBRATION_LABELS = study.CALIBRATION_LABELS
#: Each K label's per-arm setting, its pooled setting and its sample stream.
LABEL_SETTINGS: dict[str, tuple[str, str, str]] = {
    "in_sample_binary": ("binary_in_sample", "pooled_binary_in_sample", "lb_calibration"),
    "stacked_binary": ("binary_stacked", "pooled_binary_stacked", "lb_calibration"),
    "in_sample_continuous": (
        "continuous_in_sample",
        "pooled_continuous_in_sample",
        "lbc_calibration",
    ),
    "stacked_bounded": ("bounded_stacked", "pooled_bounded_stacked", "lbb_calibration"),
}
#: The exact efficiency bound of the ATE under each label's law, ``sqrt(E[D*^2])``, which sizes
#: the noise arm (:func:`tests.studies._option_by_arm_law.efficiency_sd`; the design test pins
#: it).
EFFICIENCY_SD = {
    "in_sample_binary": 0.979673,
    "stacked_binary": 0.979673,
    "in_sample_continuous": 2.288454,
    "stacked_bounded": 0.278148,
}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

#: The exact-law limits of the cells with a fixed outcome scale, pinned by the design test at
#: 1e-4: ``(bias over SD at n = 2,000, reported SE over SD)``.
EXACT_LIMITS: dict[tuple[str, str], tuple[float, float]] = {
    ("double_robustness", "treatment_correct"): (0.0, 1.0424),
    ("double_robustness", "both_wrong"): (3.4338, 1.0),
    ("pooled_outcome", "in_sample_binary"): (0.0, 1.0001),
    ("pooled_outcome", "stacked_binary"): (0.0, 1.0001),
    ("pooled_outcome", "stacked_bounded"): (0.0, 1.0010),
}

#: The families that state no verdict.
REPORTED_FAMILIES = ("pooled_outcome",)
#: The summary columns this study adds.
EXTRA_COLUMNS = ("exact_bias_over_sd", "exact_se_ratio")

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def _specs() -> list[ReplicationSpec]:
    """Every sampled cell, as ``(family, cell, n, R, setting, seed key)``.

    Every cell names its sample stream, and only the declared groups share one.
    """
    specs: list[ReplicationSpec] = []
    for label in CALIBRATION_LABELS:
        by_arm, pooled, stream = LABEL_SETTINGS[label]
        specs += [
            ReplicationSpec(
                "interval_calibration",
                f"{label}__correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                by_arm,
                seed_key=stream,
            ),
            ReplicationSpec(
                "pooled_outcome",
                label,
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                pooled,
                seed_key=stream,
            ),
        ]
    specs += [
        ReplicationSpec(
            "double_robustness",
            "treatment_correct",
            ROBUSTNESS_N,
            ROBUSTNESS_REPLICATES,
            "q_intercept",
            seed_key="lb_treatment_correct",
        ),
        ReplicationSpec(
            "double_robustness",
            "both_wrong",
            ROBUSTNESS_N,
            ROBUSTNESS_REPLICATES,
            "both_intercept",
            seed_key="lb_both_wrong",
        ),
    ]
    return specs


def _role(family: str, cell: str) -> str:
    if family in REPORTED_FAMILIES:
        return DIAGNOSTIC_ROLE
    if cell == "both_wrong":
        return "control"
    return "positive"


class DeclaredLaw:
    """The law a declared cell reads, with its exact truths.

    Parameters
    ----------
    label : str
        A key of :data:`tests.studies._option_by_arm_law.LAWS`.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.name = f"option_outcome_by_arm__{label}"

    def truth(self) -> dict[str, float]:
        return dict(law.truth(self.label))


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


def _fitted_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, size and replicate count."""
    return tuple(
        PropertyCell(
            property=spec.property,
            cell=spec.cell,
            dgp=DeclaredLaw(law.SETTINGS[spec.configuration].law),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=spec.n,
            replicates=spec.replicates,
            seed=sample_seed(spec, 0),
            role=_role(spec.property, spec.cell),
            estimand=TARGET,
            fit_kwargs={"setting": spec.configuration},
        )
        for spec in _specs()
    )


_Key = tuple[tuple[str, str, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    _, _, _, n, _, seed, setting = key[0]
    return law.draw(law.SETTINGS[setting].law, n, seed)


def _cells_of(key: _Key) -> list[str]:
    name = payload_cell(key)
    return [name, *DERIVED_CELLS.get(name, ())]


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, cell, replicate, n, requested, _, setting = key[0]
    result = law.fit(frame, setting)
    truth = law.truth(law.SETTINGS[setting].law)[TARGET]
    return [
        replicate_row(
            property_name=family,
            cell=cell,
            role=_role(family, cell),
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth,
            estimate=result[TARGET],
            alpha=STUDY.margins.alpha,
        )
    ]


def sample_seed(spec: ReplicationSpec, replicate: int) -> int:
    """The seed of one replication of a cell.

    The stream is the cell's ``seed_key`` alone, without its family, so the cells that share a
    key draw identical samples across families: the two binary K fits and their pooled fits,
    and each other K fit and its pooled fit.
    """
    return stream_seed(STUDY, "property_sample", spec.seed_key or spec.cell, replicate)


def payloads(budget: int | None = None) -> list[_Key]:
    """Every replication key; ``budget`` caps each cell for a smoke run."""
    keys: list[_Key] = []
    for spec in _specs():
        requested = spec.replicates if budget is None else min(budget, spec.replicates)
        keys += [
            (
                (
                    spec.property,
                    spec.cell,
                    replicate,
                    spec.n,
                    requested,
                    sample_seed(spec, replicate),
                    spec.configuration,
                ),
            )
            for replicate in range(requested)
        ]
    return keys


def with_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """The fitted rows and the calibration controls derived from them."""
    controls = calibration_controls(
        rows,
        STUDY,
        labels=CALIBRATION_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
        calibration_n=CALIBRATION_N,
        shrunken_se_factor=SHRUNKEN_SE_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each cell for a smoke run."""
    outcomes = fit_properties(
        payloads(budget),
        _fit_replication,
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = with_controls(pd.DataFrame([row for result in outcomes for row in result]))
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts of K and the union-model family; the pooled fits reported."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    for (family, cell), (bias, ratio) in EXACT_LIMITS.items():
        mask = (summary["property"] == family) & (summary["cell"] == cell)
        summary.loc[mask, "exact_bias_over_sd"] = bias
        summary.loc[mask, "exact_se_ratio"] = ratio
    # The reported rows state no verdict; ``passed`` is True for them, as the shared
    # diagnostics set it.
    summary.loc[summary["property"].isin(REPORTED_FAMILIES), "passed"] = True
    return finish(summary, rates)


#: The failure accounting's declared rules.  K and ``double_robustness`` read the shared rules
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES: tuple[Any, ...] = ()


def smoke_keys(draws: int) -> list[_Key]:
    """Keys of a smoke on the first ``draws`` replicates of every cell."""
    return payloads(draws)


def fit_keys(keys: list[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [_fit_replication(key, _draw(key)) for key in keys]


__all__ = [
    "CALIBRATION_REPLICATES",
    "DERIVED_CELLS",
    "EXACT_LIMITS",
    "FAILURE_BOUND",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
]
