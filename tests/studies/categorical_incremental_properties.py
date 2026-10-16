r"""Property families of ``categorical-incremental`` (X43).

Family P fits point :class:`~cleverly.estimators.TMLE` with five folds and saturated
nuisances on law ``P`` of :mod:`tests.discrete_law_categorical_ipsi_point`, unless the table
names another setting.  Family L fits :class:`~cleverly.longitudinal.IPSI` with five folds on
law ``L`` or ``L0`` of :mod:`tests.discrete_law_categorical_ipsi_longitudinal`.  ``t_a`` is the
tilt ``{b: 2, c: 0.5}`` and ``contrast_t_a`` its contrast with the natural course.

==============================  ===============================================================
family                          cells
==============================  ===============================================================
``interval_calibration``        six labels (below), each with the two derived controls
``type_i_error``                ``contrast_t_a`` on law ``P0``, where every tilt has one mean
``power``                       ``contrast_t_a`` on law ``Pw`` (arm effects scaled by
                                :data:`~tests.discrete_law_categorical_ipsi_point.POWER_SCALE`)
``mechanism_term_necessity``    ``contrast_t_a`` on its own draws of law ``P``: the full curve;
                                the curve without the mechanism term (control); and the curve
                                whose mechanism term reads the first two levels only, the
                                binary blip of mutation M2 (control)
``single_robustness``           the mechanism saturated and the outcome regression constant, at
                                two sizes (positive); the mechanism constant and the outcome
                                regression saturated (diagnostic, no double robustness)
``in_sample_fit``               ``L``, ``contrast_t_a`` with one fold (reported, no verdict)
``censoring_complete_case``     ``L``, ``ey_regimen[T_a]`` fitted to the rows that never drop
                                out, without a censoring model (reported, no verdict)
==============================  ===============================================================

The calibration labels:

===============================  ===========================================================
label                            law and estimand
===============================  ===========================================================
``point_contrast_t_a``           ``P``, ``ate_ipsi[T_a vs natural course]``
``point_contrast_t_b``           ``P``, ``ate_ipsi[T_b vs natural course]``
``point_known_t_a``              ``P`` with :math:`g(a \mid W)` declared (no mechanism term)
``point_stratified_t_a_v1``      ``P`` with the baseline stratum ``V``; the contrast in
                                 stratum ``V = 1`` (the stratified softmax fluctuation)
``longitudinal_dropout_contrast_t_a``  ``L``, ``ate_regimen[T_a vs natural course]``
``longitudinal_contrast_t_a``    ``L0``, the same contrast without dropout
===============================  ===========================================================

The four point labels read one draw each replication, fitted three ways.  The two
longitudinal labels read one draw of ``L`` and one of ``L0``.

**Rules.**  ``interval_calibration``, ``type_i_error`` and ``power`` read the shared verdicts,
with :data:`EFFICIENCY_RATIO_BAND` around the exact influence-function SD.
``mechanism_term_necessity`` passes when the full curve's SE-ratio interval lies inside
:data:`EFFICIENCY_RATIO_BAND` and each control's lies wholly below the calibration band: the
design test computes the exact SD ratios, 0.61 and 0.78.  ``single_robustness``'s positive
pair passes when both sizes pass the shared bias-equivalence rule.  With the mechanism
correct the remainder ``E[(m - m_0)(D - D_0)/D]`` is second order whatever the outcome
regression is (``X43-plan.md`` 2.2).  The diagnostic cell publishes its bias beside the
exact limit of
:func:`~tests.discrete_law_categorical_ipsi_point.constant_mechanism_limit`.  The two
reported families publish their exact limits and state no verdict.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier, DummyRegressor

from tests import discrete_law_categorical_ipsi_longitudinal as long_law
from tests import discrete_law_categorical_ipsi_point as point
from tests.parallel import STUDY_JOBS
from tests.studies import categorical_incremental_common as common
from tests.studies.canonical_categorical_incremental import (
    POINT_CALIBRATION_N,
    REFERENCE,
    REQUIREMENT_SIZES,
    STUDY,
    longitudinal_eif_sd,
    longitudinal_truth,
)
from tests.studies.evidence import property_failures, property_verdicts
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
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

#: The replicates of every calibration cell and of the necessity family, by the arm-pooled
#: predictive rule (``tests/unit/test_categorical_incremental_design.py``).
CALIBRATION_REPLICATES = 2_800
#: The size of the longitudinal calibration and reported cells.  Provisional until the
#: family L pilot runs (stage 2).
LONGITUDINAL_N = 24_000
NULL_N = 4_000
NULL_REPLICATES = 2_000
REQUIREMENT_REPLICATES = 2_000
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

GATED = point.GATED
CONTRAST_T_B = f"ate_ipsi[T_b vs {REFERENCE}]"
STRATUM = 1
STRATIFIED = common.stratum_name(GATED, STRATUM)
LONG_CONTRAST = f"ate_regimen[T_a vs {REFERENCE}]"
LONG_MEAN = "ey_regimen[T_a]"

CALIBRATION_LABELS = (
    "point_contrast_t_a",
    "point_contrast_t_b",
    "point_known_t_a",
    "point_stratified_t_a_v1",
    "longitudinal_dropout_contrast_t_a",
    "longitudinal_contrast_t_a",
)
#: ``label -> (draw stream, estimand)``.
CALIBRATION: dict[str, tuple[str, str]] = {
    "point_contrast_t_a": ("law_p", GATED),
    "point_contrast_t_b": ("law_p", CONTRAST_T_B),
    "point_known_t_a": ("law_p", GATED),
    "point_stratified_t_a_v1": ("law_p", STRATIFIED),
    "longitudinal_dropout_contrast_t_a": ("law_l", LONG_CONTRAST),
    "longitudinal_contrast_t_a": ("law_l0", LONG_CONTRAST),
}
STRATUM_LAW = point.stratum_law(point.LAW_P, STRATUM)
TRUTH: dict[str, float] = {
    "point_contrast_t_a": point.truth(point.LAW_P, GATED),
    "point_contrast_t_b": point.truth(point.LAW_P, CONTRAST_T_B),
    "point_known_t_a": point.truth(point.LAW_P, GATED),
    "point_stratified_t_a_v1": point.truth(STRATUM_LAW, GATED),
    "longitudinal_dropout_contrast_t_a": longitudinal_truth(long_law.LAW_L, LONG_CONTRAST),
    "longitudinal_contrast_t_a": longitudinal_truth(long_law.LAW_L0, LONG_CONTRAST),
}
#: The stratum parameter's influence function is the stratum law's divided by ``P(V = 1)``.
STRATUM_SHARE = float(point.P_W[[w for w, v in enumerate(point.STRATA_OF_W) if v == STRATUM]].sum())
EFFICIENCY_SD: dict[str, float] = {
    "point_contrast_t_a": point.eif_sd(point.LAW_P, GATED),
    "point_contrast_t_b": point.eif_sd(point.LAW_P, CONTRAST_T_B),
    "point_known_t_a": point.dropped_sd(point.LAW_P, GATED),
    "point_stratified_t_a_v1": point.eif_sd(STRATUM_LAW, GATED) / np.sqrt(STRATUM_SHARE),
    "longitudinal_dropout_contrast_t_a": longitudinal_eif_sd(long_law.LAW_L, LONG_CONTRAST),
    "longitudinal_contrast_t_a": longitudinal_eif_sd(long_law.LAW_L0, LONG_CONTRAST),
}
NULL_TRUTH = point.truth(point.LAW_P0, GATED)
POWER_TRUTH = point.truth(point.LAW_PW, GATED)
CONTRAST_TRUTH = TRUTH["point_contrast_t_a"]
ONE_FOLD_TRUTH = TRUTH["longitudinal_dropout_contrast_t_a"]
COMPLETE_CASE_TRUTH = longitudinal_truth(long_law.LAW_L, LONG_MEAN)
COMPLETE_CASE_LIMIT = long_law.complete_case_limit(long_law.LAW_L, long_law.PLANS["T_a"])
_CONSTANT = point.constant_mechanism_limit(point.LAW_P)
CONSTANT_MECHANISM_LIMIT = _CONSTANT["T_a"] - _CONSTANT[REFERENCE]


def _size(label: str) -> int:
    return LONGITUDINAL_N if CALIBRATION[label][0] in {"law_l", "law_l0"} else POINT_CALIBRATION_N


# ---------------------------------------------------------------------------- the grid


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by its draw stream, with its exact truths.

    Parameters
    ----------
    name : str
        A draw stream: ``"law_p"``, ``"law_p0"``, ``"law_pw"``, ``"law_l"``, ``"law_l0"``.
    """

    name: str

    def truth(self) -> dict[str, float]:
        if self.name == "law_p0":
            return {GATED: NULL_TRUTH}
        if self.name == "law_pw":
            return {GATED: POWER_TRUTH}
        out = {
            CALIBRATION[label][1]: TRUTH[label]
            for label in CALIBRATION_LABELS
            if CALIBRATION[label][0] == self.name
        }
        if self.name == "law_p":
            out[GATED] = CONTRAST_TRUTH
        if self.name == "law_l":
            out[LONG_MEAN] = COMPLETE_CASE_TRUTH
        return out


#: Every fit set: ``(family, stream, configuration, n, replicates)``.
FIT_SETS: tuple[tuple[str, str, str, int, int], ...] = (
    ("interval_calibration", "law_p", "calibration", POINT_CALIBRATION_N, CALIBRATION_REPLICATES),
    ("interval_calibration", "law_l", "calibration", LONGITUDINAL_N, CALIBRATION_REPLICATES),
    ("interval_calibration", "law_l0", "calibration", LONGITUDINAL_N, CALIBRATION_REPLICATES),
    (
        "mechanism_term_necessity",
        "law_p",
        "necessity",
        POINT_CALIBRATION_N,
        CALIBRATION_REPLICATES,
    ),
    ("type_i_error", "law_p0", "null", NULL_N, NULL_REPLICATES),
    ("power", "law_pw", "power", NULL_N, NULL_REPLICATES),
    *(
        (
            "single_robustness",
            "law_p",
            f"outcome_constant__n_{size}",
            size,
            REQUIREMENT_REPLICATES,
        )
        for size in REQUIREMENT_SIZES
    ),
    (
        "single_robustness",
        "law_p",
        "mechanism_constant",
        REQUIREMENT_SIZES[0],
        REQUIREMENT_REPLICATES,
    ),
    ("censoring_complete_case", "law_l", "complete_case", LONGITUDINAL_N, REQUIREMENT_REPLICATES),
)

_Payload = tuple[str, str, str, int, int, int]


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def _stream_label(family: str, stream: str, configuration: str) -> str:
    """The stream a fit set draws: its law, and for the robustness sets its configuration."""
    return f"{stream}/{configuration}" if family == "single_robustness" else stream


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream.

    The two calibration controls of each label are derived from its rows, so they are not
    here.
    """
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        law_name: str,
        estimand: str,
        seed: int,
        *,
        n: int,
        replicates: int,
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
                seed=seed,
                role=role,
                estimand=estimand,
            )
        )

    for label in CALIBRATION_LABELS:
        stream, estimand = CALIBRATION[label]
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                stream,
                estimand,
                _seed("interval_calibration", stream, 0),
                n=_size(label),
                replicates=CALIBRATION_REPLICATES,
            )
    for cell, role in (
        ("point_contrast_t_a__full_eif", "positive"),
        ("point_contrast_t_a__mechanism_term_dropped", "control"),
        ("point_contrast_t_a__binary_blip", "control"),
    ):
        add(
            "mechanism_term_necessity",
            cell,
            role,
            "law_p",
            GATED,
            _seed("mechanism_term_necessity", "law_p", 0),
            n=POINT_CALIBRATION_N,
            replicates=CALIBRATION_REPLICATES,
        )
    add(
        "type_i_error",
        "point_contrast_t_a__sharp_null",
        "positive",
        "law_p0",
        GATED,
        _seed("type_i_error", "law_p0", 0),
        n=NULL_N,
        replicates=NULL_REPLICATES,
    )
    add(
        "power",
        "point_contrast_t_a__alternative",
        "positive",
        "law_pw",
        GATED,
        _seed("power", "law_pw", 0),
        n=NULL_N,
        replicates=NULL_REPLICATES,
    )
    for size in REQUIREMENT_SIZES:
        add(
            "single_robustness",
            f"point_contrast_t_a__mechanism_correct__n_{size}",
            "positive",
            "law_p",
            GATED,
            _seed("single_robustness", f"law_p/outcome_constant__n_{size}", 0),
            n=size,
            replicates=REQUIREMENT_REPLICATES,
        )
    add(
        "single_robustness",
        "point_contrast_t_a__mechanism_constant",
        DIAGNOSTIC_ROLE,
        "law_p",
        GATED,
        _seed("single_robustness", "law_p/mechanism_constant", 0),
        n=REQUIREMENT_SIZES[0],
        replicates=REQUIREMENT_REPLICATES,
    )
    add(
        "in_sample_fit",
        "longitudinal_dropout_contrast_t_a__one_fold",
        DIAGNOSTIC_ROLE,
        "law_l",
        LONG_CONTRAST,
        _seed("interval_calibration", "law_l", 0),
        n=LONGITUDINAL_N,
        replicates=CALIBRATION_REPLICATES,
    )
    add(
        "censoring_complete_case",
        "longitudinal_mean_t_a__complete_case",
        DIAGNOSTIC_ROLE,
        "law_l",
        LONG_MEAN,
        _seed("censoring_complete_case", "law_l", 0),
        n=LONGITUDINAL_N,
        replicates=REQUIREMENT_REPLICATES,
    )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _draw_fit_set(payload: _Payload) -> pd.DataFrame:
    """The sample one draw of one fit set fits, a function of the payload alone."""
    family, stream, configuration, replicate, n, _ = payload
    seed = _seed(family, _stream_label(family, stream, configuration), replicate)
    if stream == "law_l":
        return long_law.sample(long_law.LAW_L, n, seed)
    if stream == "law_l0":
        return long_law.sample(long_law.LAW_L0, n, seed)
    law = {"law_p": point.LAW_P, "law_p0": point.LAW_P0, "law_pw": point.LAW_PW}[stream]
    return point.sample(law, n, seed, strata=family == "interval_calibration")


def complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows that never drop out, without their censoring columns."""
    stay = np.ones(len(frame), dtype=bool)
    for t in range(1, long_law.LAW_L.n_times + 1):
        stay &= frame[f"C{t}"].to_numpy() == 1
    kept = frame.loc[stay].drop(columns=[f"C{t}" for t in range(1, long_law.LAW_L.n_times + 1)])
    return kept.reset_index(drop=True)


def _fit_set_drawn(payload: _Payload, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes."""
    family, stream, configuration, replicate, n, requested = payload
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
            alpha=STUDY.margins.alpha,
            **common_row,
        )

    if family == "interval_calibration" and stream == "law_p":
        plain = frame.drop(columns="V")
        crossfit = common.fit_point(plain, cross_fit=True)
        known = common.fit_point(plain, cross_fit=True, known=True)
        stratified = common.fit_point(frame, cross_fit=True, strata=True)
        return [
            row(
                "point_contrast_t_a__correctly_specified",
                "positive",
                TRUTH["point_contrast_t_a"],
                crossfit[GATED],
            ),
            row(
                "point_contrast_t_b__correctly_specified",
                "positive",
                TRUTH["point_contrast_t_b"],
                crossfit[CONTRAST_T_B],
            ),
            row(
                "point_known_t_a__correctly_specified",
                "positive",
                TRUTH["point_known_t_a"],
                known[GATED],
            ),
            row(
                "point_stratified_t_a_v1__correctly_specified",
                "positive",
                TRUTH["point_stratified_t_a_v1"],
                stratified[STRATIFIED],
            ),
        ]
    if family == "interval_calibration" and stream == "law_l":
        main = common.fit_longitudinal(frame, long_law.LAW_L, n_folds=common.N_FOLDS)
        one_fold = common.fit_longitudinal(frame, long_law.LAW_L, n_folds=1)
        return [
            row(
                "longitudinal_dropout_contrast_t_a__correctly_specified",
                "positive",
                TRUTH["longitudinal_dropout_contrast_t_a"],
                main[LONG_CONTRAST],
            ),
            row(
                "longitudinal_dropout_contrast_t_a__one_fold",
                DIAGNOSTIC_ROLE,
                ONE_FOLD_TRUTH,
                one_fold[LONG_CONTRAST],
                "in_sample_fit",
            ),
        ]
    if family == "interval_calibration":
        result = common.fit_longitudinal(frame, long_law.LAW_L0, n_folds=common.N_FOLDS)
        return [
            row(
                "longitudinal_contrast_t_a__correctly_specified",
                "positive",
                TRUTH["longitudinal_contrast_t_a"],
                result[LONG_CONTRAST],
            )
        ]
    if family == "mechanism_term_necessity":
        result = common.fit_point(frame, cross_fit=True)
        full = common.mechanism_terms(result, GATED)
        blip = common.mechanism_terms(result, GATED, levels=point.BLIP_LEVELS)
        psi = float(result[GATED].psi)
        return [
            row("point_contrast_t_a__full_eif", "positive", CONTRAST_TRUTH, result[GATED]),
            *(
                control_row(
                    property_name=family,
                    cell=f"point_contrast_t_a__{cell}",
                    truth=CONTRAST_TRUTH,
                    estimate=psi,
                    standard_error=common.standard_error_without(result, GATED, removed),
                    critical=CRITICAL,
                    **common_row,
                )
                for cell, removed in (
                    ("mechanism_term_dropped", full),
                    ("binary_blip", full - blip),
                )
            ),
        ]
    if family == "type_i_error":
        result = common.fit_point(frame, cross_fit=True)
        return [row("point_contrast_t_a__sharp_null", "positive", NULL_TRUTH, result[GATED])]
    if family == "power":
        result = common.fit_point(frame, cross_fit=True)
        return [row("point_contrast_t_a__alternative", "positive", POWER_TRUTH, result[GATED])]
    if family == "censoring_complete_case":
        result = common.fit_longitudinal(
            complete_cases(frame), long_law.LAW_L0, n_folds=common.N_FOLDS
        )
        return [
            row(
                "longitudinal_mean_t_a__complete_case",
                DIAGNOSTIC_ROLE,
                COMPLETE_CASE_TRUTH,
                result[LONG_MEAN],
            )
        ]
    if configuration == "mechanism_constant":
        result = common.fit_point(
            frame, cross_fit=True, treatment_learner=DummyClassifier(strategy="prior")
        )
        return [
            row(
                "point_contrast_t_a__mechanism_constant",
                DIAGNOSTIC_ROLE,
                CONTRAST_TRUTH,
                result[GATED],
            )
        ]
    result = common.fit_point(frame, cross_fit=True, outcome_learner=DummyRegressor())
    return [
        row(
            f"point_contrast_t_a__mechanism_correct__n_{n}",
            "positive",
            CONTRAST_TRUTH,
            result[GATED],
        )
    ]


#: The calibration positive arm of each label to the controls derived from it.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

_Key = tuple[str, _Payload]


def _payloads(budget: int | None) -> list[_Key]:
    out: list[_Key] = []
    for family, stream, configuration, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(
            ("fit_set", (family, stream, configuration, r, n, requested)) for r in range(requested)
        )
    return out


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    return _draw_fit_set(key[1])


def _run(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return list(_fit_set_drawn(key[1], frame))


def _cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows and derived controls fill."""
    family, stream, configuration, _, n, _ = key[1]
    calibration = "interval_calibration"
    if family == calibration:
        labels = [label for label in CALIBRATION_LABELS if CALIBRATION[label][0] == stream]
        names = [f"{calibration}/{label}__correctly_specified" for label in labels]
        names += [control for name in names for control in DERIVED_CELLS[name]]
        if stream == "law_l":
            names.append("in_sample_fit/longitudinal_dropout_contrast_t_a__one_fold")
        return names
    if family == "mechanism_term_necessity":
        return [
            f"{family}/point_contrast_t_a__{cell}"
            for cell in ("full_eif", "mechanism_term_dropped", "binary_blip")
        ]
    if family == "type_i_error":
        return [f"{family}/point_contrast_t_a__sharp_null"]
    if family == "power":
        return [f"{family}/point_contrast_t_a__alternative"]
    if family == "censoring_complete_case":
        return [f"{family}/longitudinal_mean_t_a__complete_case"]
    if configuration == "mechanism_constant":
        return [f"{family}/point_contrast_t_a__mechanism_constant"]
    return [f"{family}/point_contrast_t_a__mechanism_correct__n_{n}"]


def _cell_of(key: _Key) -> str:
    family, stream, configuration = key[1][:3]
    return f"{family}/{_stream_label(family, stream, configuration)}"


def _replicate_of(key: _Key) -> int:
    return key[1][3]


#: The share of a cell's requested replications that the accounting lets a run lose (RM39b2
#: rev 3 section 6): up to ``floor(0.01 R)`` lost replicates are judged by the worst case of
#: each clause; one more is ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = fit_properties(
        _payloads(budget),
        _run,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            *(
                calibration_controls(
                    rows,
                    STUDY,
                    labels=[label for label in CALIBRATION_LABELS if _size(label) == size],
                    efficiency_bounds=EFFICIENCY_SD,
                    calibration_n=size,
                    shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                    critical=CRITICAL,
                )
                for size in sorted({_size(label) for label in CALIBRATION_LABELS})
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------- verdicts

#: The column that publishes a cell's exact-law limit of its mean estimate.
LIMIT_COLUMN = "exact_limit"


def mechanism_term_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The full curve stays calibrated; each control curve under-reports."""
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
        if cell.endswith(property_verdicts.FULL_EIF_SUFFIX):
            passed = interval.within(*EFFICIENCY_RATIO_BAND)
        else:
            passed = interval.high < margins.calibration_se_ratio[0]
        summary.loc[index, "passed"] = bool(passed)
        passes.append(bool(passed))
    summary.loc[family, "property_passed"] = bool(passes) and all(passes)


def single_robustness_verdicts(summary: pd.DataFrame) -> None:
    """Bias equivalence at both sizes, and the published limit of the diagnostic cell."""
    family = summary["property"] == "single_robustness"
    positive = family & (summary["role"] == "positive")
    summary.loc[positive, "passed"] = summary.loc[positive, "bias_equivalent"].astype(bool)
    verdict = bool(positive.sum() == len(REQUIREMENT_SIZES)) and bool(
        summary.loc[positive, "bias_equivalent"].all()
    )
    diagnostic = family & (summary["cell"] == "point_contrast_t_a__mechanism_constant")
    summary.loc[diagnostic, LIMIT_COLUMN] = CONSTANT_MECHANISM_LIMIT
    summary.loc[diagnostic, "passed"] = True
    summary.loc[family, "property_passed"] = verdict


def reported_families(summary: pd.DataFrame) -> None:
    """The reported families state no verdict; each publishes its exact limit."""
    complete = summary["property"] == "censoring_complete_case"
    summary.loc[complete, LIMIT_COLUMN] = COMPLETE_CASE_LIMIT
    one_fold = summary["property"] == "in_sample_fit"
    summary.loc[one_fold, LIMIT_COLUMN] = ONE_FOLD_TRUTH
    for mask in (complete, one_fold):
        summary.loc[mask, "passed"] = True
        summary.loc[mask, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(LIMIT_COLUMN,),
        rate_labels=(),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    mechanism_term_verdicts(summary, rows)
    single_robustness_verdicts(summary)
    reported_families(summary)
    return finish(summary, rates)


__all__ = [
    "FIT_SETS",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
]


def _single_robustness_rule() -> property_failures.FamilyRule:
    """``single_robustness``: S1 bias equivalence on each positive cell, joint over the family.

    The diagnostic ``mechanism_constant`` cell publishes a limit and no verdict.
    """
    return property_failures.FamilyRule(
        families=frozenset({"single_robustness"}),
        shared="single_robustness_verdicts",
        clauses=(
            property_failures.bias_clause(
                "bias equivalent", kind="equivalent", applies=property_failures.roles("positive")
            ),
        ),
        group="family",
    )


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),
    property_verdicts.score_necessity_rule(
        "mechanism_term_necessity",
        efficiency_band=EFFICIENCY_RATIO_BAND,
        shared="mechanism_term_verdicts",
    ),
    _single_robustness_rule(),
)
