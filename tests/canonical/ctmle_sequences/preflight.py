"""The Python pre-flight of the replay gates of ``canonical-ctmle-sequences`` (design review 2).

The replay (``tests/studies/ctmle_sequence_replay.py``) compares the package with R on R's own
arrays.  This pre-flight runs it without R, before the R smoke, to measure the flag rates and to
show that every unflagged gap is a rounding gap.  It stands in for R in two ways:

* R's propensity arrays are the package's own fits: the lasso path on each declared penalty,
  with the penalty 0 read at ``lambda_max * 1e-4`` as R's ``predict(s = 0)`` reads it, and for
  ``binary_learners`` one lasso per declared entry and a main-terms logistic terminal.
* R's targeting is the package's, with the bound on each candidate's targeted ``Q`` removed.
  Its ``epsilon`` is then the unbounded logistic MLE, as R's ``glm`` fits it.  The initial ``Q``
  and each accepted update stay bounded, as R bounds them.

R's written values follow ``run_ctmle_sequences.R``.  For ``binary_lasso`` they are unbounded.
For ``binary_learners`` they are R's ``Qstars`` bounded into ``[1 - alpha, alpha]``.  The flags
are the replay's own.  The pre-flight cannot see R's ``glm`` tolerance, so its unflagged gaps
are 0 where the code is right, while R's own gaps are not.  The R smoke measures those.

Run from the repository root::

    python -m tests.canonical.ctmle_sequences.preflight --replicates 200
"""

from __future__ import annotations

import argparse
import contextlib
import math
import warnings
from collections.abc import Iterator
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import PropensityArrays
from cleverly.estimators import ctmle as ctmle_module
from cleverly.exceptions import PositivityWarning
from cleverly.fluctuation import iterative
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import Submodel
from cleverly.learners._l1_logistic import default_penalty_grid, penalty_path
from cleverly.utils.bounds import expit, logit
from cleverly.utils.parallel import map_parallel
from tests.canonical.ctmle_sequence_witness import harness
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_ctmle_sequences as study
from tests.studies import ctmle_sequence_replay as replay

#: The largest unflagged gap the pre-flight accepts, from the X16 plan, 3.4.2.
SMOKE_TOLERANCE = 1e-7
COLUMNS = (
    "scenario",
    "replicate",
    "near_tie",
    "clip",
    "accepted_match",
    "est_gap",
    "loss_gap",
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
    "unflagged_share",
    "accepted_mismatches",
    "final_comparisons",
    "final_index_mismatches",
    "max_est_gap",
    "max_loss_gap",
    "max_final_est_gap",
    "max_final_loss_gap",
    "passed",
)


def _unbounded(
    fit: InitialFit, submodel: Submodel, epsilon: np.ndarray, alpha: float
) -> InitialFit:
    """:func:`~cleverly.fluctuation.iterative.apply_logistic` without its bound."""
    del alpha
    iterative.check_matching_arms(fit, submodel)
    return InitialFit(
        expit(logit(fit.observed) + submodel.observed @ epsilon),
        {
            level: expit(logit(values) + submodel.arms[level] @ epsilon)
            for level, values in fit.arms.items()
        },
    )


@contextlib.contextmanager
def r_targeting() -> Iterator[None]:
    """Fit and apply each candidate's fluctuation without the bound, as R does."""
    with (
        mock.patch.object(iterative, "apply_logistic", _unbounded),
        mock.patch.object(ctmle_module, "apply_logistic", _unbounded),
    ):
        yield


def _bounded(fit: InitialFit, y: np.ndarray) -> tuple[float, float]:
    """The estimate and the loss of one R candidate, its ``Qstar`` bounded as R's runner does."""
    low, high = harness.Q_INTERIOR
    arms = {level: np.clip(values, low, high) for level, values in fit.arms.items()}
    observed = np.clip(fit.observed, low, high)
    return float(np.mean(arms[1.0]) - np.mean(arms[0.0])), harness.negative_loglik(observed, y)


def _relative(actual: np.ndarray, desired: np.ndarray) -> float:
    return float(np.max(np.abs(actual - desired) / np.abs(desired)))


def _logistic_coef(x: np.ndarray, y: np.ndarray) -> tuple[float, ...]:
    model = LogisticRegression(C=1e6, max_iter=5000, tol=1e-12).fit(x, y)
    return (float(model.intercept_[0]), *(float(value) for value in model.coef_[0]))


def _columns(fits: list[Any], x: np.ndarray) -> np.ndarray:
    return np.column_stack([fit.predict_proba(x)[:, 1] for fit in fits])


def _lasso(frame: pd.DataFrame, x: np.ndarray, keywords: dict[str, Any]) -> dict[str, Any]:
    a = frame["A"].to_numpy()
    y = frame["Y"].to_numpy(dtype=float)
    grid = list(study.study_lambdas("binary_lasso"))
    clamp = float(default_penalty_grid(x, a)[0]) * 1e-4
    columns = _columns(penalty_path(x, a, None, [*grid, clamp]), x)
    chooser = harness.selector(frame, **keywords)
    path = harness.indexed_path(chooser, columns)
    with r_targeting():
        r_chooser = harness.selector(frame, **keywords)
        r_path = harness.indexed_path(r_chooser, columns)
    out: dict[str, Any] = {
        "accepted_match": int(path.accepted == r_path.accepted),
        "est_gap": float(np.max(np.abs(path.estimates - r_path.estimates))),
        "loss_gap": _relative(path.likelihoods(y), r_path.likelihoods(y)),
        "final_starts": 0,
        "final_index_match": 1,
        "final_est_gap": 0.0,
        "final_loss_gap": 0.0,
    }
    # The final step from the package's cross-validated index, as R's `final_pkg_*` columns.
    fitted = study.estimator("binary_lasso", frame).fit(
        frame, outcome="Y", treatment="A", covariates=list(study.COVARIATES)
    )
    start = int(fitted.single().extra["ctmle"].selected)
    options: list[Any] = []
    margin = math.inf
    if start < len(grid):
        penalties = [*grid, 0.0]
        final_columns = _columns(
            penalty_path(x, a, None, [value for value in penalties if value < penalties[start]]), x
        )
        index, candidate, options = harness.final_options(
            chooser, path, columns, final_columns, start
        )
        with r_targeting():
            r_index, r_candidate, r_options = harness.final_options(
                r_chooser, r_path, columns, final_columns, start
            )
        margin = harness.option_margin(options)
        out.update(
            final_starts=1,
            final_index_match=int(index == r_index),
            final_est_gap=abs(harness.ate(candidate.targeted) - harness.ate(r_candidate.targeted)),
            final_loss_gap=_relative(
                np.array([float(option.loss) for option in options]),
                np.array([float(option.loss) for option in r_options]),
            ),
        )
    out["near_tie"] = int(min(harness.smallest_margin(path), margin) < harness.MARGIN)
    out["clip"] = int(not harness.interior(path, options))
    return out


def _learners(frame: pd.DataFrame, x: np.ndarray, keywords: dict[str, Any]) -> dict[str, Any]:
    a = frame["A"].to_numpy()
    y = frame["Y"].to_numpy(dtype=float)
    fits = [penalty_path(x, a, None, [value])[0] for value in study.learner_lambdas()]
    pairs = [PropensityArrays(full=column, cv=column) for column in _columns(fits, x).T]
    terminal = harness.FixedLogit(_logistic_coef(x, a))
    selection, _, path = harness.public_fit(
        frame, treatment_candidates=pairs, treatment_learner=terminal, **keywords
    )
    with r_targeting():
        _, _, r_path = harness.public_fit(
            frame, treatment_candidates=pairs, treatment_learner=terminal, **keywords
        )
    written = np.array([_bounded(candidate.targeted, y) for candidate in r_path.candidates])
    return {
        "near_tie": int(harness.smallest_margin(path) < harness.MARGIN),
        "clip": int(not harness.interior(path, arms=False)),
        "accepted_match": int(path.accepted == r_path.accepted),
        "est_gap": float(np.max(np.abs(path.estimates - written[:, 0]))),
        "loss_gap": _relative(np.asarray(selection.train_loss, dtype=float), written[:, 1]),
        "final_starts": 0,
        "final_index_match": 1,
        "final_est_gap": 0.0,
        "final_loss_gap": 0.0,
    }


def preflight_one(scenario: str, replicate: int) -> dict[str, Any]:
    """One study replicate, replayed against the stand-in for R.

    Parameters
    ----------
    scenario : str
        ``binary_lasso`` or ``binary_learners``.
    replicate : int
        The study replicate, whose sample the study draws.

    Returns
    -------
    dict
        One :data:`COLUMNS` row.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PositivityWarning)
        frame, _ = study.draw_scenario(scenario, study.PRIMARY_N, replicate)
        frame = frame.reset_index(drop=True)
        x = frame.loc[:, list(study.COVARIATES)].to_numpy(dtype=float)
        keywords = {
            "covariates": study.COVARIATES,
            "split": np.arange(len(frame)) % study.SELECTION_FOLDS,
            "outcome_learner": harness.FixedLogit(
                _logistic_coef(
                    np.column_stack([frame["A"].to_numpy(), x]), frame["Y"].to_numpy(dtype=float)
                )
            ),
        }
        if scenario == "binary_lasso":
            out = _lasso(frame, x, keywords)
        elif scenario == "binary_learners":
            out = _learners(frame, x, keywords)
        else:
            raise KeyError(scenario)
    out.update(scenario=scenario, replicate=replicate)
    return {column: out[column] for column in COLUMNS}


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """The flag counts and the largest unflagged gaps of each scenario.

    A scenario passes when every unflagged gap is below :data:`SMOKE_TOLERANCE`, every
    unflagged accepted index and final index matches, and the unflagged share exceeds the
    share :data:`~tests.studies.ctmle_sequence_replay.MIN_UNFLAGGED` needs at the study's
    replicate count.

    Parameters
    ----------
    rows : DataFrame
        :data:`COLUMNS` rows.

    Returns
    -------
    DataFrame
        One :data:`SUMMARY_COLUMNS` row per scenario present.
    """
    out = []
    needed = replay.MIN_UNFLAGGED / study.PRIMARY_REPLICATES
    for scenario, group in rows.groupby("scenario", sort=True):
        kept = group.loc[(group["near_tie"] == 0) & (group["clip"] == 0)]
        final = kept.loc[kept["final_starts"] > 0]
        gaps = {
            "max_est_gap": float(kept["est_gap"].max()),
            "max_loss_gap": float(kept["loss_gap"].max()),
            "max_final_est_gap": float(final["final_est_gap"].max()) if len(final) else 0.0,
            "max_final_loss_gap": float(final["final_loss_gap"].max()) if len(final) else 0.0,
        }
        mismatches = int((kept["accepted_match"] == 0).sum())
        index_mismatches = int((final["final_index_match"] == 0).sum())
        share = len(kept) / len(group)
        out.append(
            {
                "scenario": scenario,
                "replicates": len(group),
                "near_tie": int(group["near_tie"].sum()),
                "clip": int(group["clip"].sum()),
                "unflagged": len(kept),
                "unflagged_share": share,
                "accepted_mismatches": mismatches,
                "final_comparisons": len(final),
                "final_index_mismatches": index_mismatches,
                **gaps,
                "passed": bool(
                    share > needed
                    and mismatches == 0
                    and index_mismatches == 0
                    and max(gaps.values()) < SMOKE_TOLERANCE
                ),
            }
        )
    return pd.DataFrame(out, columns=SUMMARY_COLUMNS)


def main(argv: list[str] | None = None) -> None:
    """Run the pre-flight and print its summary.

    Parameters
    ----------
    argv : list of str or None
        The command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--scenario", choices=study.SCENARIOS, action="append")
    parser.add_argument("--jobs", type=int, default=STUDY_JOBS)
    parser.add_argument("--output", help="also write the per-replicate rows to this CSV")
    arguments = parser.parse_args(argv)
    scenarios = arguments.scenario or list(study.SCENARIOS)
    payloads = [(s, r) for s in scenarios for r in range(arguments.replicates)]
    rows = pd.DataFrame(map_parallel(preflight_one, payloads, n_jobs=arguments.jobs))
    if arguments.output:
        rows.to_csv(arguments.output, index=False)
    print(summarize(rows).to_string(index=False))


if __name__ == "__main__":
    main()
