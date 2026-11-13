"""Property families of ``option-ratio-trim`` (Fix-8 plan section 5.6), and verdicts.

The cells are fixed before any run:

=====================================  ==========  =====  =====  ==============================
cell                                   role        n      R      rule
=====================================  ==========  =====  =====  ==============================
``trim_point/<estimator>_<fold>``      positive    2,000  R      bias equivalent
``trim_necessity/<estimator>_trimmed``  control     2,000  R      bias beyond the margin
``trim_necessity/<est>_untrimmed``     positive    2,000  R      bias equivalent
``trim_reported/<cell>``               reported    2,000  R      none
=====================================  ==========  =====  =====  ==============================

R is :data:`REPLICATES`.  Every fit reads one draw of law ``LE`` per replicate (stream
``le``), so the trimmed and untrimmed fits, the two outcome regressions and the two
estimators differ on the same data in the setting alone.  Every cell reads
``E[Y^always]``.

* ``trim_point``: LTMLE and SDR, ``ratio_trim=0.95``, correct outcome regressions
  (``LinearRegression``, which contains every iterated regression of ``LE``) and correct
  mechanisms, in sample and cross-fitted with 5 folds.  The exact limit is the truth: the trim
  remainder ``sum_t P_0[(R~_t - R_t)(Q*_{t+1} - Q*_t)]`` vanishes when the outcome regressions
  are correct (:func:`tests.studies._option_trim_law.trim_remainder`).
* ``trim_necessity``: in sample, the W-blind outcome regression and correct mechanisms.
  Trimmed at 0.95 the fit keeps the remainder on the trimmed share, a control whose bias must
  lie wholly beyond the margin.  Untrimmed, double robustness removes it, a positive whose bias
  must lie wholly inside.
* ``trim_reported``: the fits trimmed at 0.99 (correct and W-blind), the untrimmed fits with
  correct outcome regressions, and the trimmed share of each draw's trimmed-at-0.95 fit,
  against its population limit (:func:`tests.studies._option_trim_law.trimmed_share`).

**Rules.**  ``trim_point`` and ``trim_necessity`` read
:func:`~tests.studies.evidence.property_verdicts.bias_role_verdicts`.  The trimmed fits report
no interval (``trimmed_ratio_plugin``), so no cell reads coverage or the SE ratio.  Their rows
carry the plug-in diagnostic SE, which the summary publishes as the plug-in SE ratio.
``trim_reported`` is reported, with
:data:`~tests.studies.evidence.property_verdicts.DIAGNOSTIC_ROLE`; the trimmed-share row
carries a unit placeholder SE (:func:`~tests.studies.evidence.properties.control_row`).

Every row of a bias or reported fit carries its exact-law limit (:data:`EXACT_BIAS`) in
``exact_bias``: the limit of the estimate minus the truth, from the weighted fit on the
quadrature law at the law's trim quantile (:func:`tests.studies._option_trim_law.exact_limit`).
It is in outcome units, not SD units: the trimmed fit's sampling SD is not its plug-in curve SD.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import _option_trim_law as law
from tests.studies import canonical_option_ratio_trim as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    payload_cell,
    payload_replicate,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    bias_role_rule,
    bias_role_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
N = 2_000
#: Every cell's replications: the smallest multiple of 400 at which the six bias positives
#: pass together with probability at least 0.95 at their pilot biases, and each control is
#: discriminated with probability above 0.9999 (the design test recomputes it).
REPLICATES = 1_600
STREAM = "le"

#: Each fit: the cells it writes as ``(family, cell)``.
FITS: dict[str, tuple[tuple[str, str], ...]] = {
    "ltmle_in_sample_right_trim95": (
        ("trim_point", "ltmle_in_sample"),
        ("trim_reported", "trimmed_share"),
    ),
    "ltmle_stacked_right_trim95": (("trim_point", "ltmle_stacked"),),
    "sdr_in_sample_right_trim95": (("trim_point", "sdr_in_sample"),),
    "sdr_stacked_right_trim95": (("trim_point", "sdr_stacked"),),
    "ltmle_in_sample_wrong_trim95": (("trim_necessity", "ltmle_trimmed"),),
    "ltmle_in_sample_wrong_untrimmed": (("trim_necessity", "ltmle_untrimmed"),),
    "sdr_in_sample_wrong_trim95": (("trim_necessity", "sdr_trimmed"),),
    "sdr_in_sample_wrong_untrimmed": (("trim_necessity", "sdr_untrimmed"),),
    "ltmle_in_sample_right_trim99": (("trim_reported", "ltmle_right_trim99"),),
    "sdr_in_sample_right_trim99": (("trim_reported", "sdr_right_trim99"),),
    "ltmle_in_sample_wrong_trim99": (("trim_reported", "ltmle_wrong_trim99"),),
    "sdr_in_sample_wrong_trim99": (("trim_reported", "sdr_wrong_trim99"),),
    "ltmle_in_sample_right_untrimmed": (("trim_reported", "ltmle_right_untrimmed"),),
    "sdr_in_sample_right_untrimmed": (("trim_reported", "sdr_right_untrimmed"),),
}
#: The cell whose row is the trimmed share, not an estimate of the truth.
SHARE_CELL = ("trim_reported", "trimmed_share")

#: The exact-law limit of each fitted cell's bias, pinned by the design test at 1e-6: the
#: weighted quadrature fit at the law's trim quantile minus the truth.  The correct-outcome
#: and untrimmed cells are 0 by construction; the design test shows each below 1e-8.
EXACT_BIAS: dict[tuple[str, str], float] = {
    ("trim_point", "ltmle_in_sample"): 0.0,
    ("trim_point", "ltmle_stacked"): 0.0,
    ("trim_point", "sdr_in_sample"): 0.0,
    ("trim_point", "sdr_stacked"): 0.0,
    ("trim_necessity", "ltmle_trimmed"): 0.067941,
    ("trim_necessity", "ltmle_untrimmed"): 0.0,
    ("trim_necessity", "sdr_trimmed"): 0.083213,
    ("trim_necessity", "sdr_untrimmed"): 0.0,
    ("trim_reported", "ltmle_right_trim99"): 0.0,
    ("trim_reported", "sdr_right_trim99"): 0.0,
    ("trim_reported", "ltmle_wrong_trim99"): 0.018398,
    ("trim_reported", "sdr_wrong_trim99"): 0.025028,
    ("trim_reported", "ltmle_right_untrimmed"): 0.0,
    ("trim_reported", "sdr_right_untrimmed"): 0.0,
}

#: The families read by the bias role rule.
BIAS_FAMILIES = ("trim_point", "trim_necessity")
#: The families that state no verdict.
REPORTED_FAMILIES = ("trim_reported",)
#: The summary columns this study adds.
EXTRA_COLUMNS = ("exact_bias",)

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def _role(family: str, cell: str) -> str:
    if family in REPORTED_FAMILIES:
        return DIAGNOSTIC_ROLE
    if cell.endswith("_trimmed"):
        return "control"
    return "positive"


def sample_seed(replicate: int) -> int:
    """The seed of one replication; every fit of the replicate reads the same draw."""
    return stream_seed(STUDY, "property_sample", STREAM, replicate)


class DeclaredLaw:
    """The law a declared cell reads, with its exact truths.

    Parameters
    ----------
    label : str
        The law, :data:`tests.studies._option_trim_law.LAW`.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.name = f"option_ratio_trim__{label}"

    def truth(self) -> dict[str, float]:
        return {law.ESTIMAND: law.truth(), "trimmed_share": law.trimmed_share(law.TRIM)}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, size and replicate count."""
    return tuple(
        PropertyCell(
            property=family,
            cell=cell,
            dgp=DeclaredLaw(law.LAW),
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=N,
            replicates=REPLICATES,
            seed=sample_seed(0),
            role=_role(family, cell),
            estimand="trimmed_share" if (family, cell) == SHARE_CELL else law.ESTIMAND,
            fit_kwargs={"setting": setting},
        )
        for setting, outputs in FITS.items()
        for family, cell in outputs
    )


_Key = tuple[tuple[str, str, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    _, _, _, n, _, seed, _ = key[0]
    return law.draw(n, seed)


def _cells_of(key: _Key) -> list[str]:
    return [f"{family}/{cell}" for family, cell in FITS[key[0][1]]]


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, setting, replicate, n, requested, _, _ = key[0]
    result = law.fit(frame, setting)
    rows = []
    for family, cell in FITS[setting]:
        if (family, cell) == SHARE_CELL:
            rows.append(
                control_row(
                    property_name=family,
                    cell=cell,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=law.trimmed_share(law.TRIM),
                    estimate=float(result.fits[law.REGIMEN].ratio_trim.trimmed_share),
                    standard_error=1.0,
                    critical=CRITICAL,
                    role=DIAGNOSTIC_ROLE,
                )
            )
            continue
        rows.append(
            replicate_row(
                property_name=family,
                cell=cell,
                role=_role(family, cell),
                replicate=replicate,
                n=n,
                requested=requested,
                truth=law.TRUTH,
                estimate=result.estimates[law.ESTIMAND],
                alpha=STUDY.margins.alpha,
            )
        )
    return rows


def payloads(budget: int | None = None) -> list[_Key]:
    """Every replication key; ``budget`` caps each fit for a smoke run.

    A key is ``(("fit", setting, replicate, n, requested, seed, stream),)``; its systematic
    prefix reads the fit ``fit/<setting>``.
    """
    requested = REPLICATES if budget is None else min(budget, REPLICATES)
    return [
        (("fit", setting, replicate, N, requested, sample_seed(replicate), STREAM),)
        for setting in FITS
        for replicate in range(requested)
    ]


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit for a smoke run."""
    outcomes = fit_properties(
        payloads(budget),
        _fit_replication,
        cells_of=_cells_of,
        cell_of=payload_cell,
        replicate_of=payload_replicate,
        n_jobs=n_jobs,
        draw=_draw,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The bias verdicts of the point and necessity families; the reported fits stated."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=())
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    for (family, cell), bias in EXACT_BIAS.items():
        mask = (summary["property"] == family) & (summary["cell"] == cell)
        summary.loc[mask, "exact_bias"] = bias
    # The reported rows state no verdict; ``passed`` is True for them, as the shared
    # diagnostics set it.
    summary.loc[summary["property"].isin(REPORTED_FAMILIES), "passed"] = True
    return finish(summary, rates)


#: The failure accounting's declared rules: both verdict families read
#: :func:`~tests.studies.evidence.property_verdicts.bias_role_rule`.
PROPERTY_RULES: tuple[Any, ...] = (bias_role_rule(BIAS_FAMILIES),)


def smoke_keys(draws: int) -> list[_Key]:
    """Keys of a smoke on the first ``draws`` replicates of every fit."""
    return payloads(draws)


def fit_keys(keys: list[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [_fit_replication(key, _draw(key)) for key in keys]


__all__ = [
    "EXACT_BIAS",
    "FAILURE_BOUND",
    "FITS",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "REPLICATES",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
]
