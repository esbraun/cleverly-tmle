"""Property families of ``option-treatment-adjustment`` (Fix-8 plan section 5.4), and verdicts.

The cells are fixed before any run:

====================================  =========  ======  =====  ============================
cell                                  role       n       R      rule
====================================  =========  ======  =====  ============================
``interval_calibration/in_sample``    positive   2,000   K      shared calibration rule, with
``interval_calibration/stacked``                                the derived controls
``double_robustness/treatment_correct``  positive  2,000  1,600  bias equivalent, SE in band
``double_robustness/both_wrong``      control    2,000   1,600  bias beyond, SE in band
``instrument_slow_outcome/n16000``    control    16,000  1,600  bias beyond the margin
``instrument_slow_outcome_ladder/*``  reported   1,000,  1,600  none
                                                 4,000
``instrument_slow_outcome_rate``      reported   ladder         none (slope of log abs bias)
``instrument_reference/*``            reported   ladder  1,600  none
``instrument_parametric/n2000``       reported   2,000   2,400  none
====================================  =========  ======  =====  ============================

K is :data:`CALIBRATION_REPLICATES`.  The two K cells read the same draws, and so do the
``instrument_slow_outcome`` and ``instrument_reference`` fits of one size: those pairs differ in
the fit alone.  Every cell reads the ATE.

**Rules.**

- ``interval_calibration`` reads the shared calibration rule
  (:func:`~tests.studies.evidence.property_verdicts.calibration_verdicts`) with the derived
  ``shrunken_se_control`` and ``noise_control``; the noise arm adds one unit of the exact
  efficiency bound :data:`EFFICIENCY_SD`.  The study claims no efficiency band.
- ``double_robustness`` reads the shared union-model rule.
- ``instrument_slow_outcome`` reads
  :func:`~tests.studies.evidence.property_verdicts.bias_role_verdicts`: the 99% interval of the
  standardized bias lies wholly beyond the 0.25 margin.
- Every other family is reported, with :data:`~tests.studies.evidence.property_verdicts.DIAGNOSTIC_ROLE`.

Each instrument row's summary carries its exact-law limit: ``exact_bias_over_sd`` (the
targeting limit's ATE bias over the stacked-equation SD) and ``exact_se_ratio`` (the limit of
the reported SE over that SD).  ``instrument_slow_outcome_rate`` is the slope of the log
absolute bias on log n over the three sizes, with its 99% bootstrap interval: the constructed
error predicts -1/4, so a slope interval above -1/2 says the standardized bias grows.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import _option_adjustment_laws as laws
from tests.studies import canonical_option_treatment_adjustment as study
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
    bias_role_rule,
    bias_role_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    fitted_rate_row,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
SHRUNKEN_SE_FACTOR = 0.70
TARGET = "ate"

CALIBRATION_N = 2_000
#: K's replications.  Plan 5.0's product floor for two cells is 2,400; the arm-pooled
#: predictive rule (:func:`tests.studies.repeated_sizing.calibration_replicates`) at the
#: committed 8,000-draw pilot raises it to this count.  The two cells read the same draws
#: (pilot correlation of their estimates 0.9994), and their pilot SE ratios are 1.008 and
#: 1.013.  The design test recomputes it.
CALIBRATION_REPLICATES = 6_800
ROBUSTNESS_N = 2_000
ROBUSTNESS_REPLICATES = 1_600
LADDER = (1_000, 4_000, 16_000)
LADDER_REPLICATES = 1_600
PARAMETRIC_N = 2_000
PARAMETRIC_REPLICATES = 2_400

#: The exact efficiency bound of the ATE under ``LDA``, ``sqrt(E[D*^2])``, which sizes the noise
#: arm (:func:`tests.studies._option_adjustment_laws.efficiency_sd`; the design test pins it).
EFFICIENCY_SD = {"in_sample": 0.948643, "stacked": 0.948643}
CALIBRATION_LABELS = ("in_sample", "stacked")
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

#: Each instrument cell's exact-law limits at its size, pinned by the design test at 1e-4:
#: ``(bias over SD, reported SE over SD)``.
EXACT_LIMITS: dict[tuple[str, str], tuple[float, float]] = {
    ("instrument_slow_outcome_ladder", "n1000"): (-0.9517, 0.9148),
    ("instrument_slow_outcome_ladder", "n4000"): (-1.3488, 0.9143),
    ("instrument_slow_outcome", "n16000"): (-1.9096, 0.9143),
    ("instrument_reference", "n1000"): (-0.0001, 1.0038),
    ("instrument_reference", "n4000"): (-0.0002, 1.0018),
    ("instrument_reference", "n16000"): (-0.0003, 1.0008),
    ("instrument_parametric", "n2000"): (0.0, 0.9161),
}

#: The gated bias family and the families that state no verdict.
BIAS_FAMILIES = ("instrument_slow_outcome",)
REPORTED_FAMILIES = (
    "instrument_slow_outcome_ladder",
    "instrument_slow_outcome_rate",
    "instrument_reference",
    "instrument_parametric",
)
RATE_FAMILY = "instrument_slow_outcome_rate"
RATE_CELL = "bias_slope"
#: The summary columns this study adds.
EXTRA_COLUMNS = ("exact_bias_over_sd", "exact_se_ratio")

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def _ladder_family(n: int) -> str:
    return "instrument_slow_outcome" if n == max(LADDER) else "instrument_slow_outcome_ladder"


def _specs() -> list[ReplicationSpec]:
    """Every sampled cell, as ``(family, cell, n, R, setting, seed key)``.

    Every cell names its sample stream, and only the declared pairs share one.
    """
    specs = [
        ReplicationSpec(
            "interval_calibration",
            f"{label}__correctly_specified",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            f"lda_{label}",
            seed_key="lda_calibration",
        )
        for label in CALIBRATION_LABELS
    ]
    specs += [
        ReplicationSpec(
            "double_robustness",
            "treatment_correct",
            ROBUSTNESS_N,
            ROBUSTNESS_REPLICATES,
            "lda_q_intercept",
            seed_key="lda_treatment_correct",
        ),
        ReplicationSpec(
            "double_robustness",
            "both_wrong",
            ROBUSTNESS_N,
            ROBUSTNESS_REPLICATES,
            "lda_both_intercept",
            seed_key="lda_both_wrong",
        ),
    ]
    for n in sorted(LADDER, reverse=True):
        specs += [
            ReplicationSpec(
                _ladder_family(n), f"n{n}", n, LADDER_REPLICATES, "ldb_slow", seed_key=f"ldb_{n}"
            ),
            ReplicationSpec(
                "instrument_reference",
                f"n{n}",
                n,
                LADDER_REPLICATES,
                "ldb_slow_full",
                seed_key=f"ldb_{n}",
            ),
        ]
    specs.append(
        ReplicationSpec(
            "instrument_parametric",
            f"n{PARAMETRIC_N}",
            PARAMETRIC_N,
            PARAMETRIC_REPLICATES,
            "ldb_parametric",
            seed_key="ldb_parametric",
        )
    )
    return specs


def _role(family: str, cell: str) -> str:
    if family in REPORTED_FAMILIES:
        return DIAGNOSTIC_ROLE
    if family in BIAS_FAMILIES or cell == "both_wrong":
        return "control"
    return "positive"


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, size and replicate count."""
    return tuple(
        PropertyCell(
            property=spec.property,
            cell=spec.cell,
            dgp=laws.SETTINGS[spec.configuration].law,
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
    return laws.draw(laws.SETTINGS[setting].law, n, seed)


def _cells_of(key: _Key) -> list[str]:
    name = payload_cell(key)
    return [name, *DERIVED_CELLS.get(name, ())]


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    family, cell, replicate, n, requested, _, setting = key[0]
    result = laws.fit(frame, setting)
    truth = laws.truth(laws.SETTINGS[setting].law)[TARGET]
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
    key draw identical samples across families: the two K fits, and the constructed-error fit
    and its full-mechanism reference at one n.
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


def _ladder_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """The three sizes of the constructed-error fit, filed under one property for the slope."""
    ladder = rows.loc[
        rows["property"].isin(("instrument_slow_outcome", "instrument_slow_outcome_ladder"))
    ].copy()
    ladder["property"] = "instrument_slow_outcome_ladder"
    return ladder


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts of K and the union-model family, LR's bias rule, the rest reported."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    for (family, cell), (bias, ratio) in EXACT_LIMITS.items():
        mask = (summary["property"] == family) & (summary["cell"] == cell)
        summary.loc[mask, "exact_bias_over_sd"] = bias
        summary.loc[mask, "exact_se_ratio"] = ratio
    rates.append(
        fitted_rate_row(
            _ladder_rows(rows),
            STUDY,
            summary.columns,
            ladder_property="instrument_slow_outcome_ladder",
            property_name=RATE_FAMILY,
            cell=RATE_CELL,
            role=DIAGNOSTIC_ROLE,
            statistic="bias",
            seed_labels=(RATE_FAMILY, RATE_CELL),
            verdict=lambda fitted: True,
        )
    )
    # The reported rows state no verdict; ``passed`` is True for them, as the shared
    # diagnostics set it.
    summary.loc[summary["property"].isin(REPORTED_FAMILIES), "passed"] = True
    return finish(summary, rates)


#: The failure accounting's declared rules.  K and ``double_robustness`` read the shared rules
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`).
PROPERTY_RULES = (bias_role_rule(BIAS_FAMILIES),)


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
