"""The replay gates of ``canonical-ctmle-sequences`` (X16 plan, rev 2.2, section 3.4.2).

Quirk 2 of R ``ctmle`` drives R's cross-validation to the last candidate, so the two sides'
selected estimates are not one estimator.  The replay compares what both sides compute from
identical inputs: the full-sample construction and the C-TMLE1 final step, on R's own
propensity columns and R's own initial ``Q``.

R writes its coefficients on each reference row (``q_coef``, ``g_coef`` and the final-step
``*_coef`` columns).  :func:`replay_rows` redraws the replicate's sample, rebuilds R's arrays as
``expit(coef . [1, W])``, which the R runner asserts to ``1e-12`` against the arrays it used,
and runs the package on them through ``tests/canonical/ctmle_sequence_witness/harness.py``:

* ``binary_lasso``: :func:`~tests.canonical.ctmle_sequence_witness.harness.indexed_path` on
  the 21 rebuilt columns, and
  :func:`~tests.canonical.ctmle_sequence_witness.harness.final_options` from each final-step
  start R emitted;
* ``binary_learners``: the public ``learners`` fit on the four rebuilt array pairs, with the
  ``SL.glm`` coefficients as a fixed terminal
  (:func:`~tests.canonical.ctmle_sequence_witness.harness.public_fit`).

Both use ``penalty=True``, ``loss="loglik"``, ``alpha=0.995`` and the ``(0.025, 0.975)``
bounds, which match R's always-penalized construction criterion.  :func:`replay_summary`
applies the gate per scenario, and ``tests/unit/test_ctmle_sequences_design.py`` recomputes
it from the committed ``replay.csv``.

**The ``alpha`` bound.**  Both sides bound the initial ``Q`` and each accepted update into
``[1 - alpha, alpha]``.  The package also bounds every candidate's targeted ``Q``, and R does
not (``functions_general.R``, lines 185 to 201, and the glmnet path alike).  The ``clip`` flag
leaves out a replicate where that difference can move a compared value:

* ``binary_lasso``: R writes its own unbounded values, so the flag fires when any targeted
  ``Q`` of the package, on the path or in a final-step option, lies at or outside a bound.
  Where every value lies strictly inside, neither side's bound acts on a candidate.
* ``binary_learners``: R writes each candidate's estimate and loss from its own ``Qstars``,
  bounded the same way, so a bound on a counterfactual arm moves both sides alike.  The
  flag fires when a targeted ``Q`` of the package on an observed row, a row the fluctuation
  is fitted on, lies at or outside a bound.  There the package solves the score of the
  bounded fit, and R's ``glm`` solves the unbounded one, so the two ``epsilon`` differ.  When
  every observed row lies strictly inside, the package's score is the unbounded score, and
  both sides solve one equation.  R's construction scores stay unbounded: a bound that moved
  one of R's acceptance decisions shows as an accepted-index mismatch, which fails the gate.

R writes no targeted ``Q``, so the flag reads the package side.  The pre-flight
``tests/canonical/ctmle_sequences/preflight.py`` measures both flag rates.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import PropensityArrays
from cleverly.exceptions import PositivityWarning
from cleverly.utils.parallel import map_parallel
from tests.canonical.ctmle_sequence_witness import harness
from tests.parallel import STUDY_JOBS

#: The estimate gap (absolute) and the loss gap (relative) of the witness fixture's rows.
GLM_TOLERANCE = harness.GLM_TOLERANCE
#: A deciding score gap below this share of the round's best score flags a near tie.
MARGIN = harness.MARGIN
#: The fewest unflagged replicates each scenario's gate needs.
MIN_UNFLAGGED = 400
#: The fewest final-step comparisons, on unflagged replicates, of each scenario's gate.
MIN_FINAL = {"binary_lasso": 100, "binary_learners": 0}
#: The final-step starts R emits, by the prefix of their columns.
STARTS = ("final_r", "final_pkg")

REPLAY_COLUMNS = (
    "scenario",
    "replicate",
    "near_tie",
    "clip",
    "path_margin",
    "option_margin",
    "accepted_r",
    "accepted_package",
    "accepted_match",
    "est_gap",
    "loss_gap",
    "final_r_start",
    "final_r_index_r",
    "final_r_index_package",
    "final_pkg_start",
    "final_pkg_index_r",
    "final_pkg_index_package",
    "final_starts",
    "final_index_match",
    "final_est_gap",
    "final_loss_gap",
)

SUMMARY_COLUMNS = (
    "scenario",
    "replicates",
    "near_tie",
    "clip",
    "unflagged",
    "final_comparisons",
    "accepted_mismatches",
    "est_failures",
    "loss_failures",
    "final_index_mismatches",
    "final_est_failures",
    "final_loss_failures",
    "max_est_gap",
    "max_loss_gap",
    "max_final_est_gap",
    "max_final_loss_gap",
    "tolerance",
    "min_unflagged",
    "min_final",
    "passed",
)


def vector(value: Any) -> np.ndarray | None:
    """One vector column of an R row: ``%.17g`` values joined by ``;``, or missing.

    Parameters
    ----------
    value : Any
        The cell.  A one-value vector may arrive as a number.

    Returns
    -------
    ndarray of float or None
        The values, or ``None`` for a missing cell.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        return np.array([float(part) for part in value.split(";")], dtype=float)
    return np.array([float(value)], dtype=float)


def _coefficients(value: Any, width: int) -> np.ndarray:
    values = vector(value)
    if values is None or values.size % width:
        raise ValueError(f"a coefficient column holds {value!r}, not rows of {width} values")
    return values.reshape(-1, width)


def _columns(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    """One rebuilt array per coefficient row, ``expit(coef . [1, W])``."""
    return np.column_stack([harness.linear_logit(row, x) for row in coef])


def _relative(actual: np.ndarray, desired: np.ndarray) -> float:
    """The largest ``|actual - desired| / |desired|``, as ``assert_allclose(rtol=)`` reads it."""
    return float(np.max(np.abs(actual - desired) / np.abs(desired)))


def _absolute(actual: np.ndarray, desired: np.ndarray) -> float:
    return float(np.max(np.abs(actual - desired)))


def _indices(values: Iterable[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def _row_frame(row: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    from tests.studies.canonical_ctmle_sequences import COVARIATES, draw_scenario

    frame, _ = draw_scenario(str(row["scenario"]), int(row["n"]), int(row["replicate"]))
    frame = frame.reset_index(drop=True)
    return frame, frame.loc[:, list(COVARIATES)].to_numpy(dtype=float)


def _keywords(row: Mapping[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    from tests.studies.canonical_ctmle_sequences import COVARIATES, SELECTION_FOLDS

    width = len(COVARIATES) + 2
    q_coef = _coefficients(row["q_coef"], width)
    if q_coef.shape[0] != 1:
        raise ValueError("q_coef must hold one row: the intercept, A and the covariates")
    return {
        "covariates": COVARIATES,
        # Only the analysis-sample path is replayed, which no split reads.
        "split": np.arange(len(frame)) % SELECTION_FOLDS,
        "outcome_learner": harness.FixedLogit(tuple(float(v) for v in q_coef[0])),
    }


def replay_one(row: Mapping[str, Any]) -> dict[str, Any]:
    """Replay one R row on the package, and measure each gap.

    Parameters
    ----------
    row : Mapping
        One reference row with its vector columns.

    Returns
    -------
    dict
        One :data:`REPLAY_COLUMNS` row.
    """
    with warnings.catch_warnings():
        # Both laws put 8 to 11 percent of g outside the bounds by design (plan 3.4.1).
        warnings.simplefilter("ignore", PositivityWarning)
        return _replay_one(row)


def _replay_one(row: Mapping[str, Any]) -> dict[str, Any]:
    from tests.studies.canonical_ctmle_sequences import COVARIATES, PATH_LENGTHS

    scenario = str(row["scenario"])
    frame, x = _row_frame(row)
    y = frame["Y"].to_numpy(dtype=float)
    keywords = _keywords(row, frame)
    width = len(COVARIATES) + 1
    g_coef = _coefficients(row["g_coef"], width)
    if g_coef.shape[0] != PATH_LENGTHS[scenario]:
        raise ValueError(f"g_coef holds {g_coef.shape[0]} candidates, not {PATH_LENGTHS[scenario]}")
    columns = _columns(g_coef, x)
    if scenario == "binary_lasso":
        chooser = harness.selector(frame, **keywords)
        path = harness.indexed_path(chooser, columns)
        losses = path.likelihoods(y)
    elif scenario == "binary_learners":
        pairs = [PropensityArrays(full=column, cv=column) for column in columns[:, :-1].T]
        selection, chooser, path = harness.public_fit(
            frame,
            treatment_candidates=pairs,
            treatment_learner=harness.FixedLogit(tuple(float(v) for v in g_coef[-1])),
            **keywords,
        )
        losses = np.asarray(selection.train_loss, dtype=float)
    else:
        raise KeyError(scenario)
    r_accepted = vector(row["r_accepted"])
    accepted_r = () if r_accepted is None else tuple(int(v) - 1 for v in r_accepted)
    cand_est = vector(row["cand_est"])
    cand_loss = vector(row["cand_loss"])
    if cand_est is None or cand_loss is None:
        raise ValueError(f"{scenario} replicate {row['replicate']} carries no candidate output")
    out: dict[str, Any] = {
        "scenario": scenario,
        "replicate": int(row["replicate"]),
        "path_margin": harness.smallest_margin(path),
        "accepted_r": _indices(index + 1 for index in accepted_r),
        "accepted_package": _indices(index + 1 for index in path.accepted),
        "accepted_match": int(tuple(path.accepted) == accepted_r),
        "est_gap": _absolute(path.estimates, cand_est),
        "loss_gap": _relative(losses, cand_loss),
    }
    options: list[Any] = []
    margins = [math.inf]
    index_match = True
    est_gaps = [0.0]
    loss_gaps = [0.0]
    starts = 0
    for prefix in STARTS:
        start = row.get(f"{prefix}_start", math.nan)
        out[f"{prefix}_start"] = start
        out[f"{prefix}_index_r"] = math.nan
        out[f"{prefix}_index_package"] = math.nan
        if scenario != "binary_lasso" or start is None or math.isnan(float(start)):
            continue
        selected = int(start) - 1
        final_coef = _coefficients(row[f"{prefix}_coef"], width)
        index, candidate, scored = harness.final_options(
            chooser, path, columns, _columns(final_coef, x), selected
        )
        r_loss = vector(row[f"{prefix}_loss"])
        if r_loss is None or r_loss.size != len(scored):
            raise ValueError(f"{prefix}_loss does not hold one loss per remaining penalty")
        starts += 1
        options.extend(scored)
        margins.append(harness.option_margin(scored))
        r_index = int(row[f"{prefix}_index"]) - 1
        out[f"{prefix}_index_r"] = r_index + 1
        out[f"{prefix}_index_package"] = index + 1
        index_match = index_match and index == r_index
        est_gaps.append(abs(harness.ate(candidate.targeted) - float(row[f"{prefix}_est"])))
        loss_gaps.append(_relative(np.array([float(option.loss) for option in scored]), r_loss))
    out["option_margin"] = min(margins)
    out["near_tie"] = int(min(out["path_margin"], out["option_margin"]) < MARGIN)
    # The two scenarios read the bound differently (see the module docstring).
    lasso = scenario == "binary_lasso"
    out["clip"] = int(not harness.interior(path, options, arms=lasso))
    out["final_starts"] = starts
    out["final_index_match"] = int(index_match)
    out["final_est_gap"] = max(est_gaps)
    out["final_loss_gap"] = max(loss_gaps)
    return {column: out[column] for column in REPLAY_COLUMNS}


def replay_rows(rows: pd.DataFrame, *, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """``replay.csv``: one row per R fit, replayed on the package.

    Parameters
    ----------
    rows : DataFrame
        The assembled replicate rows of both implementations, the R rows with their vector
        columns.
    n_jobs : int
        Worker processes.

    Returns
    -------
    DataFrame
        :data:`REPLAY_COLUMNS`, sorted by scenario and replicate.
    """
    from tests.studies.canonical_ctmle_sequences import REFERENCE, REFERENCE_COLUMNS

    reference = rows.loc[rows["implementation"] == REFERENCE]
    missing = [column for column in REFERENCE_COLUMNS if column not in reference.columns]
    if missing:
        raise RuntimeError(f"the R rows carry no {missing} columns for the replay")
    keep = ["scenario", "replicate", "n", *REFERENCE_COLUMNS]
    payloads = [(record,) for record in reference.loc[:, keep].to_dict("records")]
    out = pd.DataFrame(map_parallel(replay_one, payloads, n_jobs=n_jobs), columns=REPLAY_COLUMNS)
    return out.sort_values(["scenario", "replicate"], ignore_index=True)


def replay_summary(compared: pd.DataFrame) -> pd.DataFrame:
    """``replay-summary.csv``: the gate of each scenario, from ``replay.csv``.

    On every replicate without a ``near_tie`` or ``clip`` flag, the accepted indices are
    identical, every candidate estimate is within :data:`GLM_TOLERANCE` and every loss within
    that share of R's loss, and at each final-step start the final index is identical and
    the estimate and every option loss are within the same tolerances.  The gate also needs
    :data:`MIN_UNFLAGGED` unflagged replicates and :data:`MIN_FINAL` final-step comparisons.

    Parameters
    ----------
    compared : DataFrame
        :func:`replay_rows`, or the committed ``replay.csv``.

    Returns
    -------
    DataFrame
        One :data:`SUMMARY_COLUMNS` row per scenario.
    """
    from tests.studies.canonical_ctmle_sequences import SCENARIOS

    out = []
    for scenario in SCENARIOS:
        rows = compared.loc[compared["scenario"] == scenario]
        flagged = (rows["near_tie"] != 0) | (rows["clip"] != 0)
        kept = rows.loc[~flagged]
        final = kept.loc[kept["final_starts"] > 0]

        def largest(frame: pd.DataFrame, column: str) -> float:
            return float(frame[column].max()) if len(frame) else math.nan

        counts = {
            "accepted_mismatches": int((kept["accepted_match"] == 0).sum()),
            "est_failures": int((kept["est_gap"] > GLM_TOLERANCE).sum()),
            "loss_failures": int((kept["loss_gap"] > GLM_TOLERANCE).sum()),
            "final_index_mismatches": int((final["final_index_match"] == 0).sum()),
            "final_est_failures": int((final["final_est_gap"] > GLM_TOLERANCE).sum()),
            "final_loss_failures": int((final["final_loss_gap"] > GLM_TOLERANCE).sum()),
        }
        comparisons = int(kept["final_starts"].sum())
        passed = (
            not any(counts.values())
            and len(kept) >= MIN_UNFLAGGED
            and comparisons >= MIN_FINAL[scenario]
        )
        out.append(
            {
                "scenario": scenario,
                "replicates": len(rows),
                "near_tie": int((rows["near_tie"] != 0).sum()),
                "clip": int((rows["clip"] != 0).sum()),
                "unflagged": len(kept),
                "final_comparisons": comparisons,
                **counts,
                "max_est_gap": largest(kept, "est_gap"),
                "max_loss_gap": largest(kept, "loss_gap"),
                "max_final_est_gap": largest(final, "final_est_gap"),
                "max_final_loss_gap": largest(final, "final_loss_gap"),
                "tolerance": GLM_TOLERANCE,
                "min_unflagged": MIN_UNFLAGGED,
                "min_final": MIN_FINAL[scenario],
                "passed": bool(passed),
            }
        )
    return pd.DataFrame(out, columns=SUMMARY_COLUMNS)


def unowned_red_gates(summary: pd.DataFrame, owner_ids: Sequence[str]) -> list[str]:
    """The red scenarios of ``replay-summary.csv`` that reach no owner.

    Under ``reporting`` a red replay gate is published and the run continues.  The red-cell
    ledger reads no ``scientific_failures`` hook, so it cannot route a red gate.  The gate
    reaches its owner, ``RED_CELL_OWNERS["replay"]``, when the roadmap's "Red-cell owners"
    table lists that ``id``, as the ledger requires of every owner of a red verdict row.

    Parameters
    ----------
    summary : DataFrame
        :func:`replay_summary`, or the committed ``replay-summary.csv``.
    owner_ids : sequence of str
        The ``id`` column of the roadmap's owner table.

    Returns
    -------
    list of str
        The red scenarios, or nothing when every gate passed or the owner is listed.
    """
    from tests.studies.canonical_ctmle_sequences import RED_CELL_OWNERS

    red = summary.loc[~summary["passed"].astype(bool), "scenario"]
    if red.empty or RED_CELL_OWNERS["replay"] in owner_ids:
        return []
    return [str(scenario) for scenario in red]
