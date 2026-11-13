"""Property families of ``option-markov-order`` (Fix-8 plan section 5.7), and verdicts.

The cells are fixed before any run:

======================================  =========  =====  =====  =================================
cell                                    role       n      R      rule
======================================  =========  =====  =====  =================================
``interval_calibration/<label>``        positive   2,000  K      shared calibration rule, with
                                                                 the derived controls
``order_necessity/ate``                 control    2,000  1,600  bias beyond the margin
``order_double_robustness/ey_never``    positive   2,000  1,600  bias equivalent
``order_double_robustness/ey_never_     control    2,000  1,600  bias beyond the margin
blind_mechanisms``
``markov_reported/<cell>``              reported   2,000  K or   none
                                                          1,600
======================================  =========  =====  =====  =================================

The labels are ``order1_ey`` (order 1, ``EY(always)``), ``order1_ate`` (order 1, the ATE) and
``order2_ate`` (order 2, the ATE), all on law ``LF1``, where the treatment reads ``L_{t-1}`` and
the outcome reads ``L_3``, so order 1 is the smallest correct window.  K is
:data:`CALIBRATION_REPLICATES`.

One fit writes several cells, and the fits that share a sample stream share their draws:

- stream ``lf1_calibration`` (K draws of ``LF1``): the order-1 fit writes ``order1_ey`` and
  ``order1_ate``, the order-2 fit writes ``order2_ate``, and the full-history fit writes
  ``markov_reported/full_history_ey`` and ``full_history_ate``;
- stream ``lf1_necessity`` (1,600 draws of ``LF1``): the order-0 fit writes
  ``order_necessity/ate`` and ``markov_reported/order0_ey``;
- stream ``lf2_order0`` (1,600 draws of ``LF2``): the order-0 fit writes
  ``order_double_robustness/ey_never`` and ``markov_reported/lf2_order0_ey`` and
  ``lf2_order0_ate``;
- stream ``lf2_order0_blind`` (1,600 draws of ``LF2``): the order-0 fit with intercept-only
  treatment and censoring mechanisms writes ``order_double_robustness/ey_never_blind_mechanisms``.

**Rules.**

- ``interval_calibration`` reads the shared calibration rule
  (:func:`~tests.studies.evidence.property_verdicts.calibration_verdicts`) with the derived
  ``shrunken_se_control`` and ``noise_control``; the noise arm adds one unit of the exact
  efficiency bound (:data:`EFFICIENCY_SD`).  The study claims no efficiency band: the order
  restricts the model, and the row claims validity, not efficiency.
- ``order_necessity`` and ``order_double_robustness`` read
  :func:`~tests.studies.evidence.property_verdicts.bias_role_verdicts`: the order-0 fit on
  ``LF1`` drops ``L_{t-1}`` from the treatment mechanism and ``L_3`` from the outcome window,
  both sides wrong, so its ATE bias must lie wholly beyond the 0.25 margin.  On ``LF2`` the
  treatment mechanism reads only ``A_{t-1}``, which the arm block keeps, and the outcome leans
  on ``L_3``, which the order-0 window drops.  The order-0 fit has every mechanism right and the
  outcome window wrong, so its ``EY(never)`` bias must lie wholly inside the margin; with
  intercept-only mechanisms the same fit keeps the outcome error (exact 0.54 SD), so that
  control must lie wholly beyond it.  The pair shows that the mechanisms do the rescue.  It
  reads ``EY(never)`` because the ATE's control cannot reach the 0.45 SD floor inside the
  linear-probability range (0.42 SD at an ``L_3`` coefficient of 0.9).
- ``markov_reported`` is reported, with
  :data:`~tests.studies.evidence.property_verdicts.DIAGNOSTIC_ROLE`.

The bias and reported rows carry their exact-law limits (:data:`EXACT_LIMITS`) in
``exact_bias_over_sd``: the bias of the limit over the limit's plug-in curve SD at n = 2,000
(``sqrt(E[D^2] / n)`` of the limit fit's influence curve, not the estimator's sampling SD), from
the weighted fit on the quadrature law (:func:`tests.studies._option_markov_law.exact_limit`).

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import _option_markov_law as law
from tests.studies import canonical_option_markov_order as study
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
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
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
SHRUNKEN_SE_FACTOR = 0.70

CALIBRATION_N = 2_000
#: K's replications.  Plan 5.0's product floor for three cells is 2,800; the arm-pooled
#: predictive rule (:func:`tests.studies.repeated_sizing.calibration_replicates`) at the
#: committed 8,000-draw pilot raises it to this count.  The pilot SE ratios are 0.986
#: (order 1, ``EY(always)``), 0.992 (order 1, ATE) and 0.989 (order 2, ATE), on one draw
#: stream.  The design test recomputes both.
CALIBRATION_REPLICATES = 4_000
ROBUSTNESS_N = 2_000
ROBUSTNESS_REPLICATES = 1_600

CALIBRATION_LABELS = study.CALIBRATION_LABELS
#: The exact efficiency bound of each K label's estimand under ``LF1``, ``sqrt(E[D*^2])``,
#: which sizes the noise arm.  It is the order-1 limit's curve SD: at the truth the windowed
#: nuisances equal the full-history ones, so the curves agree (the design test pins both).
EFFICIENCY_SD = {
    "order1_ey": 1.239405,
    "order1_ate": 1.957708,
    "order2_ate": 1.957708,
}
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)


#: Each fit: its setting, sample stream, n and R, and the cells it writes as
#: ``(family, cell, estimand)``.
FITS: dict[str, tuple[str, int, int, tuple[tuple[str, str, str], ...]]] = {
    "lf1_order1": (
        "lf1_calibration",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        (
            ("interval_calibration", "order1_ey__correctly_specified", law.EY_ALWAYS),
            ("interval_calibration", "order1_ate__correctly_specified", law.ATE),
        ),
    ),
    "lf1_order2": (
        "lf1_calibration",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        (("interval_calibration", "order2_ate__correctly_specified", law.ATE),),
    ),
    "lf1_full": (
        "lf1_calibration",
        CALIBRATION_N,
        CALIBRATION_REPLICATES,
        (
            ("markov_reported", "full_history_ey", law.EY_ALWAYS),
            ("markov_reported", "full_history_ate", law.ATE),
        ),
    ),
    "lf1_order0": (
        "lf1_necessity",
        ROBUSTNESS_N,
        ROBUSTNESS_REPLICATES,
        (
            ("order_necessity", "ate", law.ATE),
            ("markov_reported", "order0_ey", law.EY_ALWAYS),
        ),
    ),
    "lf2_order0": (
        "lf2_order0",
        ROBUSTNESS_N,
        ROBUSTNESS_REPLICATES,
        (
            ("order_double_robustness", "ey_never", law.EY_NEVER),
            ("markov_reported", "lf2_order0_ey", law.EY_ALWAYS),
            ("markov_reported", "lf2_order0_ate", law.ATE),
        ),
    ),
    "lf2_order0_blind": (
        "lf2_order0_blind",
        ROBUSTNESS_N,
        ROBUSTNESS_REPLICATES,
        (("order_double_robustness", "ey_never_blind_mechanisms", law.EY_NEVER),),
    ),
}

#: The exact-law limits, pinned by the design test at 1e-4: the limit's bias over its plug-in
#: curve SD at n = 2,000 (:func:`tests.studies._option_markov_law.exact_limit`).
EXACT_LIMITS: dict[tuple[str, str], float] = {
    ("order_necessity", "ate"): 1.3664,
    ("order_double_robustness", "ey_never"): 0.0,
    ("order_double_robustness", "ey_never_blind_mechanisms"): 0.5372,
    ("markov_reported", "full_history_ey"): 0.0,
    ("markov_reported", "full_history_ate"): 0.0,
    ("markov_reported", "order0_ey"): 0.4566,
    ("markov_reported", "lf2_order0_ey"): 0.0,
    ("markov_reported", "lf2_order0_ate"): 0.0,
}

#: The families read by the bias role rule.
BIAS_FAMILIES = ("order_necessity", "order_double_robustness")
#: The families that state no verdict.
REPORTED_FAMILIES = ("markov_reported",)
#: The summary columns this study adds.
EXTRA_COLUMNS = ("exact_bias_over_sd",)

#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def _role(family: str, cell: str) -> str:
    if family in REPORTED_FAMILIES:
        return DIAGNOSTIC_ROLE
    if family == "order_necessity" or cell.endswith("_blind_mechanisms"):
        return "control"
    return "positive"


def sample_seed(stream: str, replicate: int) -> int:
    """The seed of one replication of a sample stream.

    The fits that share a stream draw identical samples: the order-1, order-2 and
    full-history fits of ``lf1_calibration`` read the same K draws.
    """
    return stream_seed(STUDY, "property_sample", stream, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, size and replicate count."""
    return tuple(
        PropertyCell(
            property=family,
            cell=cell,
            dgp=law.SETTINGS[setting].law,
            outcome_learner=lambda: None,
            treatment_learner=lambda: None,
            n=n,
            replicates=replicates,
            seed=sample_seed(stream, 0),
            role=_role(family, cell),
            estimand=estimand,
            fit_kwargs={"setting": setting},
        )
        for setting, (stream, n, replicates, outputs) in FITS.items()
        for family, cell, estimand in outputs
    )


_Key = tuple[tuple[str, str, int, int, int, int, str]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    _, setting, _, n, _, seed, _ = key[0]
    return law.draw(law.SETTINGS[setting].law, n, seed)


def _cells_of(key: _Key) -> list[str]:
    names = [f"{family}/{cell}" for family, cell, _ in FITS[key[0][1]][3]]
    return [expanded for name in names for expanded in (name, *DERIVED_CELLS.get(name, ()))]


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    _, setting, replicate, n, requested, _, _ = key[0]
    result = law.fit(frame, setting)
    truth = law.truth(law.SETTINGS[setting].law)
    return [
        replicate_row(
            property_name=family,
            cell=cell,
            role=_role(family, cell),
            replicate=replicate,
            n=n,
            requested=requested,
            truth=truth[estimand],
            estimate=result.estimates[estimand],
            alpha=STUDY.margins.alpha,
        )
        for family, cell, estimand in FITS[setting][3]
    ]


def payloads(budget: int | None = None) -> list[_Key]:
    """Every replication key; ``budget`` caps each fit for a smoke run.

    A key is ``(("fit", setting, replicate, n, requested, seed, stream),)``; its systematic
    prefix reads the fit ``fit/<setting>``.
    """
    keys: list[_Key] = []
    for setting, (stream, n, replicates, _) in FITS.items():
        requested = replicates if budget is None else min(budget, replicates)
        keys += [
            (("fit", setting, replicate, n, requested, sample_seed(stream, replicate), stream),)
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
    """Fit every property replication; ``budget`` caps each fit for a smoke run."""
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
    """The shared verdicts of K and the bias families; the reported fits stated."""
    summary, rates = apply_shared_verdicts(rows, STUDY, extra_columns=EXTRA_COLUMNS, rate_labels=())
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=None)
    bias_role_verdicts(summary, families=BIAS_FAMILIES)
    for (family, cell), bias in EXACT_LIMITS.items():
        mask = (summary["property"] == family) & (summary["cell"] == cell)
        summary.loc[mask, "exact_bias_over_sd"] = bias
    # The reported rows state no verdict; ``passed`` is True for them, as the shared
    # diagnostics set it.
    summary.loc[summary["property"].isin(REPORTED_FAMILIES), "passed"] = True
    return finish(summary, rates)


#: The failure accounting's declared rules.  K reads the shared rule
#: (:data:`~tests.studies.evidence.property_verdicts.FAMILY_RULES`); the bias families read
#: :func:`~tests.studies.evidence.property_verdicts.bias_role_rule`.
PROPERTY_RULES: tuple[Any, ...] = (bias_role_rule(BIAS_FAMILIES),)


def smoke_keys(draws: int) -> list[_Key]:
    """Keys of a smoke on the first ``draws`` replicates of every fit."""
    return payloads(draws)


def fit_keys(keys: list[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [_fit_replication(key, _draw(key)) for key in keys]


__all__ = [
    "CALIBRATION_REPLICATES",
    "DERIVED_CELLS",
    "EXACT_LIMITS",
    "FAILURE_BOUND",
    "FITS",
    "PROPERTY_RULES",
    "PROPERTY_WITNESS",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
]
