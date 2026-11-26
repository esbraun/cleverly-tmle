"""The ``variable-importance`` property cells: FWER, power, calibration and report-only readings.

Declared, not run (design rev 2, section 7). One key is one property sample of one law,
``stream_seed(STUDY, "property_sample", law, r)``; it fits both configurations on that sample,
and on the first :data:`~tests.studies.variable_importance.WITNESS_REPLICATES` samples of
``nulls_beside_signals`` it also refits each configuration with
``adjust_for_other_candidates=False`` (the adjustment witness, paired by replicate id). Every
fit passes the shape and identity gates of :mod:`tests.studies.variable_importance`, and a
failure stops the run.

=========================  ==========================================  =========  ==========
family                     cell                                        role       gate (99% CP)
=========================  ==========================================  =========  ==========
``fwer_calibration``       ``<complete null>__<config>__alpha<a>``     positive   inside
                                                                                  ``a (1 +/- d)``
``family_wise_error``      ``<mixed law>__<config>__alpha<a>``         positive   upper
                                                                                  ``<= a (1 + d)``
``family_wise_error``      ``<complete null>__...__unadjusted``,       control    lower
                           ``__shrunken_se``                                      ``> a (1 + d)``
``power``                  ``<law>__<config>__<row>``                  positive   lower ``>= 0.80``
``power``                  ``<law>__<config>__X4__one_sided``          control    upper ``< 0.80``
``interval_calibration``   ``X1_<config>``, ``X5_<config>``, with the  positive,  the shared
                           derived ``shrunken_se`` and ``noise``       control    calibration rule
                           controls
``false_discovery``        ``<law>__<config>__<bh|by|max_t>__q<q>``    diagnostic none
``correlation_error``      ``<law>__<config>``                         diagnostic none
``adjustment_witness``     ``<config>__<row>``                         diagnostic none
=========================  ==========================================  =========  ==========

An FWER row's ``rejected`` is ``1{V >= 1}`` on ``adjusted p <= alpha``. Its ``estimate`` is the
smallest adjusted p-value over the exact nulls, ``truth`` is ``alpha``, and ``std_error`` is a
unit placeholder that no verdict reads. The ``unadjusted`` control reads the raw p-values. The
``shrunken_se`` control divides each row's statistic by
:data:`~tests.studies.variable_importance.SHRUNKEN_FACTOR` and reruns the step-down on the
fit's own draws; the ``one_sided`` control replaces each p-value by the upper one-tail p-value
of the signed statistic and reruns it. Both read each row's statistic from ``wald_test()`` and
its ``residual_margin`` (review finding 8), and the run checks that the margin-reduced
statistic gives back the shipped p-value. They are computed from the key's own fit, so they
are ordinary cells of the key rather than ``DERIVED_CELLS``. A ``false_discovery`` row's
``estimate`` is the false discovery proportion, ``truth`` the reference ``q m0 / m``,
``covered`` is ``FDP <= q`` and ``rejected`` is ``FDP > 0``. A ``correlation_error`` row's
``estimate`` is the Frobenius distance between the estimated and the exact correlation.

``python -m tests.studies.variable_importance_properties --smoke`` fits the first
:data:`~tests.studies.variable_importance.SMOKE_REPLICATES` pilot-stream replicates of the two
smoke scenarios, prints the readings, ``identity gate: passed`` and the projected size of the
committed artifacts, and writes nothing. ``--pilot --raw DIR --jobs J`` runs the declared
pilot, checkpointed under ``DIR/checkpoints``, writes its rows under ``DIR`` and the readings
to :data:`~tests.studies.variable_importance.PILOT_PATH`, and prints the declared R.
"""

from __future__ import annotations

import argparse
import io
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from tests.parallel import STUDY_JOBS
from tests.studies import variable_importance as study
from tests.studies import variable_importance_law as law_module
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    reported_verdicts,
)
from tests.studies.evidence.replication import fit_properties, recording
from tests.studies.evidence.schema import reported_inference
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
PROPERTY_CELLS = study.PROPERTY_CELLS
#: The share of a cell's requested replications the accounting lets a run lose.
FAILURE_BOUND: Final[float] = 0.01
#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS
#: The run's property stream (design section 11).
STREAM = "property_sample"
CRITICAL = float(stats.norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: Each calibration positive arm's derived controls.
DERIVED_CELLS = calibration_derived(study.calibration_labels())
#: Above this projected size the runner stops before the run (runner brief).
ARTIFACT_CEILING_MB = study.ARTIFACT_CEILING_MB

#: ``(law, replicate, requested, seed, witness, configurations)``.
_Payload = tuple[str, int, int, int, bool, tuple[str, ...]]
_Key = tuple[_Payload]


# ------------------------------------------------------------------------ readings of one fit


def statistics(result: Any) -> dict[str, np.ndarray]:
    """Each row's Wald statistic, standard error and residual margin, in fit order.

    ``distance`` is the margin-reduced statistic ``max(0, |t| - margin / se)`` whose two-sided
    normal p-value is the shipped p-value; the check stops the run when it is not.
    """
    entries = study.fit_order(result)
    tests = [entry.estimate.wald_test() for entry in entries]
    signed = np.array([float(test.statistic) for test in tests])
    std_error = np.array([float(entry.estimate.std_error) for entry in entries])
    margin = np.array([float(entry.estimate.residual_margin) for entry in entries])
    pvalues = np.array([float(entry.estimate.pvalue) for entry in entries])
    distance = np.maximum(0.0, np.abs(signed) - margin / std_error)
    if not np.allclose(2.0 * stats.norm.sf(distance), pvalues, rtol=1e-9, atol=1e-300):
        raise AssertionError("the margin-reduced Wald statistic does not give the shipped p-value")
    return {
        "signed": signed,
        "distance": distance,
        "std_error": std_error,
        "margin": margin,
        "p": pvalues,
    }


def fit_correlation(result: Any) -> np.ndarray:
    """The result's correlation in fit order (the result stores it in entry order)."""
    names = [study.row_name(entry) for entry in result]
    position = {name: index for index, name in enumerate(names)}
    index = [position[name] for name in study.ESTIMANDS]
    return np.asarray(result.correlation)[np.ix_(index, index)]


def null_draws(result: Any) -> np.ndarray:
    """The fit's own absolute null draws, re-drawn from its seed in fit order."""
    factor = np.linalg.cholesky(fit_correlation(result))
    rng = np.random.default_rng(result.multiplier_seed)
    normals = rng.standard_normal((result.multiplier_draws, factor.shape[0]))
    return np.abs(normals @ factor.T)


def stepped(statistic: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """The step-down on fixed draws, floored at each statistic's own p-value and capped."""
    floor = 2.0 * stats.norm.sf(statistic)
    return np.minimum(np.maximum(law_module.step_down(statistic, draws), floor), 1.0)


def control_pvalues(result: Any, read: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The adjusted p-values of the ``shrunken_se`` and ``one_sided`` controls, in fit order."""
    draws = null_draws(result)
    one_tail = stats.norm.sf(np.sign(read["signed"]) * read["distance"])
    return {
        "shrunken_se": stepped(read["distance"] / study.SHRUNKEN_FACTOR, draws),
        "one_sided": stepped(stats.norm.isf(one_tail / 2.0), draws),
    }


def _row(
    family: str,
    cell: str,
    role: str,
    *,
    replicate: int,
    requested: int,
    truth: float,
    estimate: float,
    std_error: float = 1.0,
    covered: bool = False,
    rejected: bool,
) -> dict[str, Any]:
    return {
        "property": family,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": study.N,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": float(truth),
        "estimate": float(estimate),
        "std_error": float(std_error),
        "covered": int(covered),
        "rejected": int(rejected),
    }


def false_discovery(rejected: np.ndarray, nulls: np.ndarray) -> float:
    """``V / max(R, 1)``."""
    return float(np.sum(rejected & nulls) / max(int(np.sum(rejected)), 1))


def law_rows(
    law: str,
    configuration: str,
    result: Any,
    readings: dict[str, np.ndarray],
    *,
    replicate: int,
    requested: int,
) -> list[dict[str, Any]]:
    """Every row one fit of a law writes, in :func:`study.law_cells` order."""
    truth = study.truths(law)
    names = study.ESTIMANDS
    nulls = np.array([study.is_null(law, name) for name in names])
    read = statistics(result)
    controls = control_pvalues(result, read)
    entries = dict(zip(names, study.fit_order(result), strict=True))
    adjusted = {
        "max_t": readings["max_t"],
        "unadjusted": read["p"],
        "shrunken_se": controls["shrunken_se"],
    }
    exact = law_module.correlation(study.exact_covariance(law))
    rows: list[dict[str, Any]] = []
    common = {"replicate": replicate, "requested": requested}
    for family, cell, role in study.law_cells(law, configuration):
        if family in (study.FWER_CALIBRATION, study.FAMILY_WISE_ERROR):
            alpha = study.ALPHAS[
                [study.alpha_label(a) for a in study.ALPHAS].index(cell.split("__")[2])
            ]
            kind = cell.split("__")[3] if cell.count("__") == 3 else "max_t"
            values = adjusted[kind][nulls]
            rows.append(
                _row(
                    family,
                    cell,
                    role,
                    truth=alpha,
                    estimate=float(values.min()),
                    rejected=bool(np.any(values <= alpha)),
                    **common,
                )
            )
        elif family == study.POWER:
            label = cell.split("__")[2]
            name = next(item for item in names if study.row_label(item) == label)
            index = names.index(name)
            estimate = entries[name].estimate
            std_error, low, high = reported_inference(estimate)
            source = controls["one_sided"] if cell.endswith("__one_sided") else readings["max_t"]
            rows.append(
                _row(
                    family,
                    cell,
                    role,
                    truth=truth[name],
                    estimate=float(estimate.psi),
                    std_error=std_error,
                    covered=low <= truth[name] <= high,
                    rejected=bool(source[index] <= study.POWER_ALPHA),
                    **common,
                )
            )
        elif family == study.CALIBRATION:
            name = cell.split("_", 1)[0]
            estimate = entries[name].estimate
            std_error, low, high = reported_inference(estimate)
            rows.append(
                _row(
                    family,
                    cell,
                    role,
                    truth=truth[name],
                    estimate=float(estimate.psi),
                    std_error=std_error,
                    covered=low <= truth[name] <= high,
                    rejected=bool(read["p"][names.index(name)] < STUDY.margins.alpha),
                    **common,
                )
            )
        elif family == study.FALSE_DISCOVERY:
            _, _, procedure, level = cell.split("__")
            q = study.ALPHAS[[study.q_label(a) for a in study.ALPHAS].index(level)]
            proportion = false_discovery(readings[procedure] <= q, nulls)
            rows.append(
                _row(
                    family,
                    cell,
                    role,
                    truth=q * float(nulls.mean()),
                    estimate=proportion,
                    covered=proportion <= q,
                    rejected=proportion > 0.0,
                    **common,
                )
            )
        elif family == study.CORRELATION_ERROR:
            distance = float(np.linalg.norm(fit_correlation(result) - exact))
            rows.append(
                _row(family, cell, role, truth=0.0, estimate=distance, rejected=False, **common)
            )
        else:
            raise KeyError(family)
    return rows


def witness_rows(
    configuration: str,
    result: Any,
    readings: dict[str, np.ndarray],
    *,
    replicate: int,
    requested: int,
) -> list[dict[str, Any]]:
    """The adjustment witness's rows: each row read against its ``False`` truth."""
    truth = study.truths(study.WITNESS_LAW, adjust=False)
    entries = dict(zip(study.ESTIMANDS, study.fit_order(result), strict=True))
    rows = []
    for (family, cell, role), name in zip(
        study.witness_cells(configuration), study.ESTIMANDS, strict=True
    ):
        estimate = entries[name].estimate
        std_error, low, high = reported_inference(estimate)
        rows.append(
            _row(
                family,
                cell,
                role,
                replicate=replicate,
                requested=requested,
                truth=truth[name],
                estimate=float(estimate.psi),
                std_error=std_error,
                covered=low <= truth[name] <= high,
                rejected=bool(readings["max_t"][study.ESTIMANDS.index(name)] <= study.POWER_ALPHA),
            )
        )
    return rows


#: The upper-triangle positions of the correlation, stored once per replicate.
_UPPER = np.triu_indices(len(study.ESTIMANDS), k=1)


def replicate_record(
    law: str,
    configuration: str,
    adjust: bool,
    replicate: int,
    result: Any,
    readings: dict[str, np.ndarray],
) -> dict[str, Any]:
    """One row of :data:`~tests.studies.variable_importance.REPLICATES_FILE`."""
    read = statistics(result)
    entries = study.fit_order(result)
    record: dict[str, Any] = {
        "law": law,
        "configuration": configuration,
        "adjust_for_other_candidates": adjust,
        "replicate": replicate,
    }
    for index, name in enumerate(study.ESTIMANDS):
        label = study.row_label(name)
        record[f"psi_{label}"] = float(entries[index].estimate.psi)
        record[f"se_{label}"] = float(read["std_error"][index])
        record[f"margin_{label}"] = float(read["margin"][index])
        record[f"p_{label}"] = float(read["p"][index])
        record[f"maxt_{label}"] = float(readings["max_t"][index])
        record[f"bh_{label}"] = float(readings["bh"][index])
    correlation = fit_correlation(result)
    for i, j in zip(*_UPPER, strict=True):
        record[f"r_{study.row_label(study.ESTIMANDS[i])}_{study.row_label(study.ESTIMANDS[j])}"] = (
            float(correlation[i, j])
        )
    return record


# ------------------------------------------------------------------------ keys


def seed(law: str, replicate: int, *, pilot: bool = False) -> int:
    """A property sample's seed: the run stream, or the salted pilot stream."""
    return stream_seed(STUDY, study.PILOT_STREAM if pilot else STREAM, law, replicate)


def requested(law: str) -> int:
    """The declared property count of a law.

    Raises
    ------
    RuntimeError
        Before the declaration commit sets
        :data:`~tests.studies.variable_importance.DECLARED_REPLICATES`.
    """
    if law not in study.GATED_LAWS:
        return study.REPORTED_REPLICATES
    if study.DECLARED_REPLICATES is None:
        raise RuntimeError(
            f"{study.SLUG} is declared and has not run: the declared run waits for the "
            "declaration commit, which sets DECLARED_REPLICATES from the committed pilot"
        )
    return study.DECLARED_REPLICATES


def payloads(
    laws: Sequence[str] | None = None,
    budget: int | None = None,
    *,
    pilot: bool = False,
    configurations: tuple[str, ...] = study.CONFIGURATIONS,
) -> list[_Key]:
    """Every key, law by law; the witness rides on the first keys of its law (not the pilot)."""
    keys: list[_Key] = []
    for law in study.LAW_NAMES if laws is None else laws:
        total = study.PILOT_REPLICATES if pilot else requested(law)
        count = total if budget is None else min(budget, total)
        for replicate in range(count):
            witness = (
                not pilot
                and law == study.WITNESS_LAW
                and replicate < min(study.WITNESS_REPLICATES, count)
            )
            keys.append(
                (
                    (
                        law,
                        replicate,
                        count,
                        seed(law, replicate, pilot=pilot),
                        witness,
                        configurations,
                    ),
                )
            )
    return keys


def _draw(key: _Key) -> pd.DataFrame:
    law, _, _, sample_seed, _, _ = key[0]
    return law_module.LAWS[law].sample(study.N, sample_seed)


def _base_cells(key: _Key) -> list[str]:
    law, _, _, _, witness, configurations = key[0]
    names = []
    for configuration in configurations:
        names += [f"{family}/{cell}" for family, cell, _ in study.law_cells(law, configuration)]
        if witness:
            names += [f"{family}/{cell}" for family, cell, _ in study.witness_cells(configuration)]
    return names


def _cells_of(key: _Key) -> list[str]:
    names = _base_cells(key)
    return names + [control for name in names for control in DERIVED_CELLS.get(name, ())]


def _cell_of(key: _Key) -> str:
    return _base_cells(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[0][1]


def fit_replication(key: _Key, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[Any]]:
    """One key's rows and its per-fit records."""
    law, replicate, count, _, witness, configurations = key[0]
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for configuration in configurations:
        for adjust in (True, False) if witness else (True,):
            result = study.fit(frame, configuration, adjust=adjust)
            study.check_shape(result, study.N)
            readings = study.check_identity(result)
            if adjust:
                rows += law_rows(
                    law, configuration, result, readings, replicate=replicate, requested=count
                )
            else:
                rows += witness_rows(
                    configuration,
                    result,
                    readings,
                    replicate=replicate,
                    requested=min(study.WITNESS_REPLICATES, count),
                )
            records.append(
                replicate_record(law, configuration, adjust, replicate, result, readings)
            )
    return rows, records


def _fit_all(
    keys: list[_Key], n_jobs: int, *, checkpoint: Any = "ambient"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = fit_properties(
        keys,
        fit_replication,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        rows_at=0,
        derived=DERIVED_CELLS,
        seed_of=lambda key: key[0][3],
        checkpoint=checkpoint,
    )
    rows = pd.DataFrame([row for value in outcomes for row in value[0]])
    records = pd.DataFrame([record for value in outcomes for record in value[1]])
    return rows, records


def with_controls(rows: pd.DataFrame) -> pd.DataFrame:
    """The rows with the derived calibration controls appended (exact SD bounds)."""
    labels = tuple(
        label
        for label in study.calibration_labels()
        if (rows["cell"] == f"{label}__{study.CALIBRATION_SUFFIX}").any()
    )
    if not labels:
        return rows
    covariance = study.exact_covariance(study.WITNESS_LAW)

    def bound(label: str) -> float:
        index = study.ESTIMANDS.index(label.split("_", 1)[0])
        return float(np.sqrt(covariance[index, index]))

    bounds = {label: bound(label) for label in labels}
    controls = calibration_controls(
        rows,
        STUDY,
        labels=labels,
        efficiency_bounds=bounds,
        calibration_n=study.N,
        shrunken_se_factor=study.SHRUNKEN_FACTOR,
        critical=CRITICAL,
    )
    return pd.concat([rows, controls], ignore_index=True)


def _ordered(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


#: Significant digits of the replicates artifact: the estimates and p-values, and the
#: correlation (enough to rerun any adjustment to Monte Carlo precision, at less size).
DIGITS: Final[int] = 12
CORRELATION_DIGITS: Final[int] = 8


def formatted(records: pd.DataFrame) -> pd.DataFrame:
    """The replicates artifact with its floats written at :data:`DIGITS` significant digits.

    The correlation columns (``r_*``) take :data:`CORRELATION_DIGITS`.
    """
    out = records.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            spec = f".{CORRELATION_DIGITS if column.startswith('r_') else DIGITS}g"
            out[column] = out[column].map(lambda value, spec=spec: format(value, spec))
    return out


def generate_property_artifacts(
    *, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Fit every declared key; the rows, and the per-replicate artifact.

    Raises
    ------
    RuntimeError
        Before the declaration commit (:func:`requested`).
    """
    rows, records = _fit_all(payloads(), n_jobs)
    return _ordered(with_controls(rows)), {study.REPLICATES_FILE: formatted(records)}


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Fit every declared key's rows."""
    return generate_property_artifacts(n_jobs=n_jobs)[0]


# ------------------------------------------------------------------------ verdicts


def cell_alpha(cell: str) -> float:
    """The level of an FWER cell, from its ``alpha<a>`` part."""
    label = cell.split("__")[2]
    for alpha in study.ALPHAS:
        if study.alpha_label(alpha) == label:
            return alpha
    raise KeyError(cell)


def ceiling(cell: str) -> float:
    """``alpha (1 + delta)``."""
    return cell_alpha(cell) * (1.0 + study.DELTA)


def multiplicity_verdicts(summary: pd.DataFrame) -> None:
    """The FWER and power verdicts, cell by cell, on the 99% Clopper-Pearson interval."""
    families = (study.FWER_CALIBRATION, study.FAMILY_WISE_ERROR, study.POWER)
    for index in summary.index[summary["property"].isin(families)]:
        family = str(summary.loc[index, "property"])
        cell = str(summary.loc[index, "cell"])
        role = str(summary.loc[index, "role"])
        low = float(summary.loc[index, "rejection_ci_lower"])
        high = float(summary.loc[index, "rejection_ci_upper"])
        if role == DIAGNOSTIC_ROLE:
            passed = True
        elif family == study.FWER_CALIBRATION:
            alpha = cell_alpha(cell)
            passed = alpha * (1 - study.DELTA) <= low and high <= alpha * (1 + study.DELTA)
        elif family == study.FAMILY_WISE_ERROR:
            passed = high <= ceiling(cell) if role == "positive" else low > ceiling(cell)
        else:
            floor = property_verdicts.MINIMUM_POWER
            passed = low >= floor if role == "positive" else high < floor
        summary.loc[index, "passed"] = bool(passed)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The summary: FWER and power gated, calibration on the shared rule, the rest reported."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    multiplicity_verdicts(summary)
    calibration_verdicts(summary, margins=STUDY.margins, positive_suffix=study.CALIBRATION_SUFFIX)
    reported_verdicts(summary, families=study.REPORTED_FAMILIES)
    return finish(summary, rates)


def _family(*names: str) -> pf.Applies:
    def applies(ctx: pf.Context, row: pd.Series) -> bool:
        return str(row["property"]) in names

    return applies


def _both(first: pf.Applies, second: pf.Applies) -> pf.Applies:
    return lambda ctx, row: first(ctx, row) and second(ctx, row)


def _band(ctx: pf.Context, row: pd.Series, iv: Any) -> pf.Judged:
    alpha = cell_alpha(str(row["cell"]))
    low, high = alpha * (1 - study.DELTA), alpha * (1 + study.DELTA)
    return pf.Judged(min(iv.low - low, high - iv.high), bool(iv.within(low, high)))


def _under(ctx: pf.Context, row: pd.Series, iv: Any) -> pf.Judged:
    limit = ceiling(str(row["cell"]))
    return pf.Judged(limit - iv.high, bool(iv.high <= limit))


def _over(ctx: pf.Context, row: pd.Series, iv: Any) -> pf.Judged:
    limit = ceiling(str(row["cell"]))
    return pf.Judged(iv.low - limit, bool(iv.low > limit))


def multiplicity_rule() -> pf.FamilyRule:
    """The FWER and power families' clauses, all S3 (design section 7)."""
    power = property_verdicts.MINIMUM_POWER
    return pf.FamilyRule(
        families=frozenset({study.FWER_CALIBRATION, study.FAMILY_WISE_ERROR, study.POWER}),
        shared="multiplicity_verdicts",
        clauses=(
            pf.proportion_clause(
                "FWER inside the band",
                statistic="S3",
                judge=_band,
                applies=_both(_family(study.FWER_CALIBRATION), pf.roles("positive")),
            ),
            pf.proportion_clause(
                "FWER under the ceiling",
                statistic="S3",
                judge=_under,
                applies=_both(_family(study.FAMILY_WISE_ERROR), pf.roles("positive")),
            ),
            pf.proportion_clause(
                "control FWER above the ceiling",
                statistic="S3",
                judge=_over,
                applies=_both(_family(study.FAMILY_WISE_ERROR), pf.roles("control")),
            ),
            pf.proportion_clause(
                "rejection floor",
                statistic="S3",
                judge=pf.floor_judge(lambda ctx: power),
                applies=_both(_family(study.POWER), pf.roles("positive")),
            ),
            pf.proportion_clause(
                "one-sided rejection below the floor",
                statistic="S3",
                judge=pf.below_judge(lambda ctx: power),
                applies=_both(_family(study.POWER), pf.roles("control")),
            ),
        ),
    )


#: The failure accounting's declared rules.
PROPERTY_RULES = (
    multiplicity_rule(),
    property_verdicts.calibration_rule(efficiency_band=None),
)

#: The property differential's declared size: each cell capped at this many replicates, on
#: :func:`differential_cells` only (review finding 11).
DIFFERENTIAL_BUDGET: Final[int] = 200
#: The whole groups the differential reads: one complete null and one mixed law, in sample,
#: which hold every clause of both rules (band, ceiling, both controls, floor, one-sided,
#: calibration and its derived controls).
DIFFERENTIAL_LAWS: Final[tuple[str, ...]] = ("independent_null", "nulls_beside_signals")


def differential_cells() -> frozenset[str]:
    """The ``family/cell`` names of the differential's groups."""
    names = {
        f"{family}/{cell}"
        for law in DIFFERENTIAL_LAWS
        for family, cell, role in study.law_cells(law, "in_sample")
        if role != DIAGNOSTIC_ROLE
    }
    return frozenset(names | {c for name in names for c in DERIVED_CELLS.get(name, ())})


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=1e6, max_iter=2000)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every gated cell, each naming its law, its estimand and its stream root.

    Raises
    ------
    RuntimeError
        Before the declaration commit (:func:`requested`).
    """
    out = []
    for configuration in study.CONFIGURATIONS:
        for law in study.GATED_LAWS:
            for family, cell, role in study.law_cells(law, configuration):
                if role == DIAGNOSTIC_ROLE:
                    continue
                names = [f"{family}/{cell}", *DERIVED_CELLS.get(f"{family}/{cell}", ())]
                for name in names:
                    family_name, cell_name = name.split("/", 1)
                    out.append(
                        PropertyCell(
                            property=family_name,
                            cell=cell_name,
                            dgp=law_module.LAWS[law],
                            outcome_learner=_logistic,
                            treatment_learner=_logistic,
                            n=study.N,
                            replicates=requested(law),
                            seed=seed(law, 0),
                            role=role if name == names[0] else "control",
                            estimand="fwer"
                            if family in (study.FWER_CALIBRATION, study.FAMILY_WISE_ERROR)
                            else cell.split("__")[2 if family == study.POWER else 0],
                        )
                    )
    return tuple(out)


# ------------------------------------------------------------------------ pilot and smoke


#: Below this many replicates a calibration row reports no kappa (the smoke run).
KAPPA_MINIMUM: Final[int] = 50


def _kappa(group: pd.DataFrame, rng: np.random.Generator, draws: int = 2_000) -> float:
    """The SE ratio's spread times root R, by a bootstrap of the replicates."""
    estimates = group["estimate"].to_numpy(dtype=float)
    errors = group["std_error"].to_numpy(dtype=float)
    size = estimates.size
    index = rng.integers(0, size, size=(draws, size))
    ratios = errors[index].mean(axis=1) / estimates[index].std(axis=1, ddof=1)
    return float(ratios.std(ddof=1) * math.sqrt(size))


def pilot_readings(rows: pd.DataFrame) -> pd.DataFrame:
    """The pilot's planning readings (``kind``, ``law``, ``configuration``, ``alpha``, ``row``).

    ``fwer``: each gated law's largest FWER over the configurations at each level, which
    :func:`study.declared_replicates` reads beside the oracle; ``coverage``: each calibration
    row's coverage farthest from 0.95 over the configurations; ``se_ratio_kappa``: the largest
    bootstrap spread of a calibration row's SE ratio. The rest is reported only.
    """
    out: list[dict[str, Any]] = []

    def add(kind: str, value: float, **fields: Any) -> None:
        out.append(
            {
                "kind": kind,
                "law": "",
                "configuration": "",
                "alpha": math.nan,
                "row": "",
                "value": float(value),
                **fields,
            }
        )

    fwer = rows.loc[rows["property"].isin([study.FWER_CALIBRATION, study.FAMILY_WISE_ERROR])]
    for cell, group in fwer.groupby("cell", sort=True):
        parts = str(cell).split("__")
        law, configuration, alpha = parts[0], parts[1], cell_alpha(str(cell))
        kind = parts[3] if len(parts) == 4 else "fwer_by_configuration"
        add(
            kind,
            group["rejected"].mean(),
            law=law,
            configuration=configuration,
            alpha=alpha,
            replicates=len(group),
        )
    by_configuration = [item for item in out if item["kind"] == "fwer_by_configuration"]
    for law in study.LAW_NAMES:
        for alpha in study.ALPHAS:
            values = [
                item["value"]
                for item in by_configuration
                if item["law"] == law and item["alpha"] == alpha
            ]
            if values:
                kind = "fwer" if law in study.GATED_LAWS else "fwer_reported"
                add(kind, max(values), law=law, alpha=alpha)
    for cell, group in rows.loc[rows["property"] == study.POWER].groupby("cell", sort=True):
        parts = str(cell).split("__")
        kind = "one_sided" if len(parts) == 4 else "power"
        add(
            kind,
            group["rejected"].mean(),
            law=parts[0],
            configuration=parts[1],
            row=parts[2],
            replicates=len(group),
        )
    calibration = rows.loc[rows["property"] == study.CALIBRATION]
    rng = np.random.default_rng(stream_seed(STUDY, "pilot_kappa"))
    coverage: dict[str, list[float]] = {}
    kappas = []
    for cell, group in calibration.groupby("cell", sort=True):
        label = str(cell).split("__")[0]
        row, configuration = label.split("_", 1)
        value = float(group["covered"].mean())
        coverage.setdefault(row, []).append(value)
        kappa = _kappa(group, rng) if len(group) >= KAPPA_MINIMUM else math.nan
        if math.isfinite(kappa):
            kappas.append(kappa)
        add(
            "coverage_by_configuration",
            value,
            configuration=configuration,
            row=row,
            replicates=len(group),
        )
        add(
            "se_ratio",
            group["std_error"].mean() / group["estimate"].std(ddof=1),
            configuration=configuration,
            row=row,
        )
        add("se_ratio_kappa_by_configuration", kappa, configuration=configuration, row=row)
    for row, values in coverage.items():
        add("coverage", max(values, key=lambda value: abs(value - 0.95)), row=row)
    if kappas:
        add("se_ratio_kappa", max(kappas))
    discovery = rows.loc[rows["property"] == study.FALSE_DISCOVERY]
    for cell, group in discovery.groupby("cell", sort=True):
        parts = str(cell).split("__")
        add(
            f"fdr_{parts[2]}",
            group["estimate"].mean(),
            law=parts[0],
            configuration=parts[1],
            alpha=float(int(parts[3][1:]) / 100),
        )
    return pd.DataFrame(out)


def projected_megabytes(rows: pd.DataFrame, records: pd.DataFrame) -> dict[str, float]:
    """Each committed property file's projected size, scaled from these fits to the declared run.

    The rows are written through the published writer to a temporary file, so the Parquet
    size is measured, not guessed.
    """
    import tempfile

    from tests.studies.evidence.manifest import write_rows

    replicates = study.DECLARED_REPLICATES or study.PROVISIONAL_REPLICATES
    keys = replicates * len(study.GATED_LAWS) + study.REPORTED_REPLICATES
    configurations = len(study.CONFIGURATIONS)
    fits = (keys + study.WITNESS_REPLICATES) * configurations
    count = max(len(records), 1)
    text = io.StringIO()
    formatted(records).to_csv(text, index=False)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.parquet"
        write_rows(_ordered(with_controls(rows)), path)
        rows_bytes = path.stat().st_size
    return {
        study.REPLICATES_FILE: len(text.getvalue().encode()) / count * fits / 1e6,
        "property-replicates.parquet": rows_bytes / count * keys * configurations / 1e6,
    }


def smoke(replicates: int = study.SMOKE_REPLICATES) -> None:
    """Fit the smoke scenarios on the pilot stream, in process; print, write nothing."""
    keys: list[_Key] = []
    for scenario in study.SMOKE_SCENARIOS:
        law, configuration = study.split_scenario(scenario)
        keys += payloads([law], replicates, pilot=True, configurations=(configuration,))
    started = time.perf_counter()
    with recording(STUDY.implementation, witness=PROPERTY_WITNESS) as recorded:
        rows, records = _fit_all(keys, 1, checkpoint=None)
    seconds = time.perf_counter() - started
    print("identity gate: passed")
    print(f"fits: {len(records)}, {seconds / max(len(records), 1):.2f} s per fit")
    print(f"failure records: {sum(len(frame) for frame in recorded.frames)}")
    with pd.option_context(
        "display.width", 200, "display.max_rows", 200, "display.max_columns", 20
    ):
        print(pilot_readings(rows))
    sizes = projected_megabytes(rows, records)
    for name, value in sizes.items():
        verdict = "under" if value <= ARTIFACT_CEILING_MB else "OVER"
        print(
            f"projected {name}: {value:.1f} MB, {verdict} the {ARTIFACT_CEILING_MB:.0f} MB ceiling"
        )
    print(f"projected total: {sum(sizes.values()):.1f} MB")


def run_pilot(raw: Path, *, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """The declared pilot: every law at 800 on the salted stream, checkpointed under ``raw``."""
    raw.mkdir(parents=True, exist_ok=True)
    with recording(STUDY.implementation, witness=PROPERTY_WITNESS) as recorded:
        rows, records = _fit_all(payloads(pilot=True), n_jobs, checkpoint=raw / "checkpoints")
    rows.to_csv(raw / "pilot-rows.csv", index=False)
    formatted(records).to_csv(raw / "pilot-replicates.csv", index=False)
    print(f"failure records: {sum(len(frame) for frame in recorded.frames)}")
    readings = pilot_readings(rows)
    readings.to_csv(study.PILOT_PATH, index=False)
    return readings


def main(argv: Sequence[str] | None = None) -> None:
    """``--smoke``, or ``--pilot --raw DIR``."""
    parser = argparse.ArgumentParser(description=f"{study.SLUG} pilot and smoke")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--pilot", action="store_true")
    parser.add_argument("--raw", type=Path, help="the pilot's raw and checkpoint directory")
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    arguments = parser.parse_args(argv)
    if arguments.smoke:
        smoke()
        return
    if arguments.raw is None:
        parser.error("--pilot needs --raw DIR, outside the repository and outside /tmp")
    readings = run_pilot(arguments.raw, n_jobs=arguments.jobs)
    with pd.option_context(
        "display.width", 200, "display.max_rows", 400, "display.max_columns", 20
    ):
        print(readings)
    replicates, probability = study.declared_replicates(readings)
    print(f"declared R: {replicates} (family pass {probability:.3f})")


if __name__ == "__main__":
    main()
