"""Registered evidence for the lasso and learners C-TMLE sequences, ``canonical-ctmle-sequences``.

The subject is ``CTMLE(strategy="lasso")`` and ``CTMLE(strategy="learners")``.  The comparator
is R ``ctmle`` 0.1.2 at ``18de559``: ``ctmleGlmnet(ctmletype = 1)`` for the lasso path, and
``build_gn_seq`` with ``ctmleGeneral(ctmletype = 1)`` for the declared candidates.  The image
is built from the unchanged ``tests/canonical/ctmle_selector`` context, and the runner
``tests/canonical/ctmle_sequences/run_ctmle_sequences.R`` is mounted beside the shared harness.

**The law** is the selector study's binary law, ``scenario_dgp("binary")`` on ``W1`` to ``W3``,
with the exact ATE from ``truth_for``.  This record draws its own datasets through
:func:`~tests.studies.evidence.seeds.draw_replicate`, so the samples are not the selector
study's.  Both scenarios sample that one law; the two names give each strategy its own seed
stream.

**The penalty grid** is read from ``tests/canonical/ctmle_sequences/lambdas.csv``, which both
sides read.  :mod:`tests.canonical.ctmle_sequences.design` writes it by :func:`penalty_design`:
twenty log-spaced penalties from ``top`` down to ``top / 100``, with ``top`` equal to
:data:`TOP_FRACTION` times the 1st percentile of ``lambda_max`` over :data:`DESIGN_DRAWS` design
draws of the law at the primary ``n``.  Every penalty then lies below ``lambda_max`` on almost
every replicate, so every candidate is a distinct mechanism.  :func:`learner_lambdas` are grid
entries 4, 8, 12 and 16.

**binary_lasso** fits ``strategy="lasso", lambdas=STUDY_LAMBDAS + (0,)`` in sample, with five
unstratified selection folds that :func:`assert_unstratified_selection` reads back.  R fits
``ctmleGlmnet`` on the same rows with the same split, carried in the ``selection_fold`` column.

**binary_learners** pairs on R's own split.  ``make_r_folds.R`` draws ``folds_R`` for every
replicate from the seed R sets before that replicate, and the result is committed as
``r_folds.csv.gz``.  The package fits ``strategy="learners"`` with ``selection_split=folds_R``
and one :class:`~cleverly.estimators.PropensityArrays` pair for each of
:func:`learner_lambdas`.  The private :class:`~cleverly.learners._l1_logistic.L1Logistic`
computes each pair: ``full`` on every row, and ``cv`` out of fold on ``folds_R``.  The terminal
is the main-terms logistic regression.  The declared differences: the package refits the
outcome inside each selection fold and R does not; R refits ``epsilon`` on every row inside
each fold (quirk 2); and the package refits its terminal under the split, while R reads the
``cv`` column of ``SL.glm``.

**The inference columns** of both implementations are the plain ``D*`` variance.  The package
reports its plug-in diagnostic through :func:`~tests.studies.evidence.schema.reported_inference`.
R reports ``var.psi`` of the type-1 final step for ``binary_lasso`` and
``varDstar_origscale[best_k] / n`` for ``binary_learners``.  R's ``initial_estimate`` is the
untargeted ``mean(q1 - q0)``, the nonzero targeting witness.

**Red cells.**  The record publishes under ``reporting``, declared before the run.
:data:`RED_CELL_OWNERS` names the owner of each kind of red row.  The ledger and the roadmap's
"Red-cell owners" table list a row only after the run shows it red, because
``tests/unit/test_red_cell_ledger.py`` refuses an owner with no red row.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly._typing import EstimandName
from cleverly.estimators import CTMLE, PropensityArrays
from cleverly.learners._l1_logistic import L1Logistic, default_penalty_grid
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ctmle_selector import (
    CTMLE_COMMIT,
    R_BASE_IMAGE,
    assert_unstratified_selection,
)
from tests.studies.canonical_cvtmle import STRATIFY_FOLDS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, stream_seed

PRIMARY_REPLICATES = 800
PRIMARY_N = 2000
SEED = 20261601
G_BOUNDS = (0.025, 0.975)
SELECTION_FOLDS = 5
COVARIATES = ("W1", "W2", "W3")
IMPLEMENTATION = "cleverly-ctmle-sequences"
REFERENCE = "r-ctmle"

#: Where the comparator's inputs live.  Both files are written in the declaration step and
#: committed with it, so this module imports without them and only a fit reads them.
ARTIFACTS = ROOT / "tests" / "canonical" / "ctmle_sequences"
LAMBDAS_FILE = ARTIFACTS / "lambdas.csv"
R_FOLDS_FILE = ARTIFACTS / "r_folds.csv.gz"

#: The grid rule.  ``top`` is this fraction of the 1st percentile of ``lambda_max`` over the
#: design draws, and the grid falls log-linearly from ``top`` to ``top * GRID_RATIO``.
TOP_FRACTION = 0.8
TOP_PERCENTILE = 1.0
GRID_RATIO = 0.01
GRID_SIZE = 20
DESIGN_DRAWS = 50
#: The label of the design draws' seed stream, so they share no sample with any replicate.
DESIGN_STREAM = "lambda-design"
#: The 1-based grid entries the ``binary_learners`` candidates use, least adaptive first.
LEARNER_ENTRIES = (4, 8, 12, 16)

#: The terminal and outcome learner of both scenarios: main-terms logistic, unpenalized.
LOGISTIC_C = 1e6
LOGISTIC_MAX_ITER = 2000

SCENARIO_ESTIMANDS: Mapping[str, tuple[EstimandName, ...]] = {
    "binary_lasso": ("ate",),
    "binary_learners": ("ate",),
}

PROPERTY_CELLS = {
    "double_robustness": (
        "lasso_outcome_correct",
        "lasso_treatment_correct",
        "learners_outcome_correct",
        "learners_treatment_correct",
        "both_wrong",
    ),
    "selector_necessity": ("lasso_collaborative", "learners_collaborative", "empty_control"),
    "multi_arm": ("lasso_treatment_correct", "empty_control"),
}

#: The owner of each kind of red row, declared before the run.  F18 owns the inference the
#: selector paths withhold.  ``X16-sequence-reading`` is a standing-record owner: it reads a red
#: paired row, or a red positive double-robustness cell, against quirks 2, 4 and 8 of the X16
#: plan and against the adapter rate in ``r-selection.csv``, before any other reading.
RED_CELL_OWNERS: Mapping[str, str] = {
    "truth/coverage": "F18",
    "truth/se_ratio": "F18",
    "property/selector_necessity": "F18",
    "paired": "X16-sequence-reading",
    "property/double_robustness/positive": "X16-sequence-reading",
}

#: The side artefacts: one row per R fit, and one descriptive paired comparison.
SELECTION_FILE = "r-selection.csv"
SELECTION_SUMMARY_FILE = "r-selection-summary.csv"
#: The columns the R runner writes after the published fourteen, read by
#: :func:`extra_artifacts` before the driver drops them.
REFERENCE_COLUMNS = ("best_k", "n_candidates", "adapter_used", "path_min_lambda")
#: The confidence level of the descriptive paired interval.
SUMMARY_CONFIDENCE = 0.99

STUDY = StudyRecord(
    name="C-TMLE candidate sequences",
    slug="canonical-ctmle-sequences",
    artifacts=ARTIFACTS,
    document="docs/technical-reference/method-evidence/c-tmle-candidate-sequences.md",
    anchor="c-tmle-candidate-sequences",
    scenarios=SCENARIO_ESTIMANDS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    # Reporting, declared before the run.  The selector-necessity cells of both selector
    # studies are red under F18, and quirk 2 of R ``ctmle`` can move the paired rows.  The
    # complete result is published, and RED_CELL_OWNERS names who reads each red row.
    publication_policy="reporting",
    extra_artifacts=(SELECTION_FILE, SELECTION_SUMMARY_FILE),
    modules=(
        "tests/studies/canonical_ctmle_sequences.py",
        "tests/studies/ctmle_sequence_properties.py",
        "tests/studies/canonical_ctmle_selector.py",
        "tests/studies/canonical_properties.py",
        "tests/studies/bounded_cv_laws.py",
        "tests/studies/canonical_cvtmle.py",
        "tests/studies/canonical_tmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/multi_arm_common.py",
        "tests/studies/multi_arm_properties.py",
        "tests/studies/point_study_helpers.py",
        "tests/conftest.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_ctmle_sequences",
    properties_module="tests.studies.ctmle_sequence_properties",
    property_cells=PROPERTY_CELLS,
)

REFERENCE_METADATA = {
    "ctmle_commit": CTMLE_COMMIT,
    "ctmle_version": "0.1.2",
    "r_base_image": R_BASE_IMAGE,
    "glmnet_control": "glmnet.control(fdev = 0, devmax = 1)",
    "binary_lasso": (
        "ctmleGlmnet(Y, A, W, Q = cbind(q0, q1), lambdas = c(STUDY_LAMBDAS, 0), ctmletype = 1, "
        "family = 'binomial', like_type = 'loglik', gbound = 0.025, PEN = FALSE, V = 5, "
        "folds = the package's selection folds, detailed = TRUE); std_error = sqrt(var.psi) of "
        "the type-1 final step"
    ),
    "binary_learners": (
        "build_gn_seq(A, W, SL.library = SL.cv1lasso..SL.cv4lasso at LEARNER_LAMBDAS and "
        "SL.glm, folds = folds_R), then set.seed(s_r) and ctmleGeneral(ctmletype = 1, "
        "gn_candidates, gn_candidates_cv, family = 'binomial', like_type = 'loglik', "
        "gbound = 0.025, PEN = FALSE, V = 5, folds = folds_R, detailed = TRUE); "
        "std_error = sqrt(varDstar_origscale[best_k] / n)"
    ),
    "type_1_adapter": (
        "ctmleGlmnet type 1 has no branch for a selected last penalty (quirk 7). The runner "
        "catches an error of ctmleGlmnet, rebuilds stage2_glmnet and cv_glmnet on the same "
        "Qinit, rethrows unless the rebuilt best_k is the last penalty, and then reports "
        "results.all$est[best_k] with varDstar_origscale[best_k] / n"
    ),
    "outcome_regression": "glm(Y ~ A + W1 + W2 + W3, family = binomial())",
    "red_cell_owners": dict(RED_CELL_OWNERS),
}

CONFIGURATION = {
    "cross_fit": False,
    "simultaneous_intervals": False,
    "selection_folds": SELECTION_FOLDS,
    "selection_inner_folds": 2,
    "penalty": False,
    "g_bounds": list(G_BOUNDS),
    "stratify_folds": STRATIFY_FOLDS,
    "outcome_learner": f"LogisticRegression(C={LOGISTIC_C:g}) on A, W1, W2, W3",
    "lambda_grid": (
        f"{GRID_SIZE} log-spaced penalties from top to top * {GRID_RATIO:g}, top = "
        f"{TOP_FRACTION} x the {TOP_PERCENTILE:g}st percentile of lambda_max over "
        f"{DESIGN_DRAWS} design draws of the binary law at n = {PRIMARY_N}; read from "
        "lambdas.csv; binary_lasso appends the penalty 0"
    ),
    "learner_lambdas": f"grid entries {', '.join(str(entry) for entry in LEARNER_ENTRIES)}",
    "binary_lasso_folds": (
        f"unstratified {SELECTION_FOLDS}-fold selection assignments drawn from the "
        "estimator's own seed and asserted; R reads the same assignment"
    ),
    "binary_learners_folds": (
        "folds_R from make_r_folds.R, committed as r_folds.csv.gz; selection_split = "
        "folds_R - 1; PropensityArrays cv columns fitted out of fold on folds_R"
    ),
    "declared_differences": (
        "the package refits the outcome inside each selection fold; R refits epsilon on every "
        "row inside each fold (quirk 2) and refits glmnet on every row (quirk 3); R clamps "
        "the penalty 0 to the path minimum in its construction and cross-validation (quirk 4); "
        "for binary_learners the package refits its terminal under the split, while R reads "
        "SL.glm's cv column"
    ),
    "inference": (
        "package: plug-in D* diagnostic (plugin_std_error); R: var.psi of the type-1 final "
        "step for binary_lasso and varDstar_origscale[best_k] / n for binary_learners"
    ),
    "failure_rule": (
        "a failed or dropped replication, a non-finite estimate, an R error outside the type-1 "
        "adapter, or a quirk-5 separation violation fails the run"
    ),
    "red_cell_owners": dict(RED_CELL_OWNERS),
    "comparison_scope": "binary ATE, two arms, complete outcomes, in-sample nuisances",
}


# --------------------------------------------------------------------------- the law


#: Both scenarios sample this one law.
LAW = "binary"


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw directly from a supplied seed for the manifest-seed audit."""
    from tests.studies.canonical_tmle import scenario_dgp, truth_for

    del scenario
    dgp = scenario_dgp(LAW)
    frame, _ = dgp.sample(n, seed=seed, backend="pandas")
    return frame, truth_for(dgp)


# --------------------------------------------------------------------------- the grid


def penalty_design(
    sample: Any,
    n: int,
    *,
    size: int,
    covariates: tuple[str, ...],
    label: str,
) -> tuple[float, ...]:
    """The declared grid rule: ``size`` log-spaced penalties from ``top`` to ``top / 100``.

    ``top`` is :data:`TOP_FRACTION` times the :data:`TOP_PERCENTILE` percentile of
    ``lambda_max`` over :data:`DESIGN_DRAWS` draws.  ``lambda_max`` is the first value of
    :func:`~cleverly.learners._l1_logistic.default_penalty_grid`, the smallest penalty at which
    every coefficient is zero.  Each draw takes its seed from this record's
    :data:`DESIGN_STREAM` stream under ``label``, so no draw is a replicate.

    Parameters
    ----------
    sample : callable
        ``sample(n, seed) -> frame`` for the law the grid is for.
    n : int
        The sample size of every draw.
    size : int
        The number of penalties.
    covariates : tuple of str
        The covariate columns the penalized mechanism reads.
    label : str
        The name of the law, which keeps two laws' design draws apart.

    Returns
    -------
    tuple of float
        The penalties, strictly decreasing and positive.
    """
    tops = []
    for draw in range(DESIGN_DRAWS):
        frame = sample(n, stream_seed(STUDY, DESIGN_STREAM, label, n, draw))
        x = frame.loc[:, list(covariates)].to_numpy(dtype=float)
        tops.append(float(default_penalty_grid(x, frame["A"].to_numpy())[0]))
    top = TOP_FRACTION * float(np.percentile(tops, TOP_PERCENTILE))
    grid = top * np.exp(np.linspace(0.0, math.log(GRID_RATIO), size))
    return tuple(float(value) for value in grid)


def primary_grid() -> tuple[float, ...]:
    """The primary grid, recomputed by the rule that wrote ``lambdas.csv``.

    Returns
    -------
    tuple of float
        :data:`GRID_SIZE` penalties.
    """

    def sample(n: int, seed: int) -> pd.DataFrame:
        return draw_from_seed("binary_lasso", n, seed)[0]

    return penalty_design(sample, PRIMARY_N, size=GRID_SIZE, covariates=COVARIATES, label=LAW)


def _missing(path: Path, what: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{path.relative_to(ROOT).as_posix()} is missing. It holds {what}, which the "
        "declaration step writes and commits; see tests/canonical/ctmle_sequences/README.md"
    )


@functools.cache
def study_lambdas() -> tuple[float, ...]:
    """``STUDY_LAMBDAS``: the committed primary grid, without the appended penalty 0.

    Returns
    -------
    tuple of float
        The :data:`GRID_SIZE` penalties of ``lambdas.csv``, in file order.

    Raises
    ------
    FileNotFoundError
        When ``lambdas.csv`` has not been written yet.
    ValueError
        When the file breaks the grid rule's shape.
    """
    if not LAMBDAS_FILE.exists():
        raise _missing(LAMBDAS_FILE, "the declared penalty grid")
    table = pd.read_csv(LAMBDAS_FILE, float_precision="round_trip")
    if list(table.columns) != ["entry", "lambda"] or len(table) != GRID_SIZE:
        raise ValueError(f"lambdas.csv must hold columns entry, lambda and {GRID_SIZE} rows")
    if not np.array_equal(table["entry"], np.arange(1, GRID_SIZE + 1)):
        raise ValueError("lambdas.csv entries must be 1 to the grid size, in order")
    values = table["lambda"].to_numpy(dtype=float)
    if not (np.all(values > 0.0) and np.all(np.diff(values) < 0.0)):
        raise ValueError("lambdas.csv must hold positive, strictly decreasing penalties")
    return tuple(float(value) for value in values)


def learner_lambdas() -> tuple[float, ...]:
    """``LEARNER_LAMBDAS``: grid entries :data:`LEARNER_ENTRIES` of ``lambdas.csv``.

    Returns
    -------
    tuple of float
        One penalty per declared candidate, least adaptive first.
    """
    grid = study_lambdas()
    return tuple(grid[entry - 1] for entry in LEARNER_ENTRIES)


def __getattr__(name: str) -> tuple[float, ...]:
    # The two grids keep the plan's names, and read the committed file only when used.
    if name == "STUDY_LAMBDAS":
        return study_lambdas()
    if name == "LEARNER_LAMBDAS":
        return learner_lambdas()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@functools.cache
def _r_folds_table() -> dict[int, np.ndarray]:
    if not R_FOLDS_FILE.exists():
        raise _missing(R_FOLDS_FILE, "R's selection folds for binary_learners")
    table = pd.read_csv(R_FOLDS_FILE)
    if list(table.columns) != ["replicate", "row", "fold"]:
        raise ValueError("r_folds.csv.gz must hold the columns replicate, row, fold")
    out: dict[int, np.ndarray] = {}
    for replicate, group in table.groupby("replicate", sort=True):
        ordered = group.sort_values("row")
        if not np.array_equal(ordered["row"], np.arange(len(ordered))):
            raise ValueError(f"r_folds.csv.gz replicate {replicate} does not list rows 0 to n-1")
        out[int(replicate)] = ordered["fold"].to_numpy(dtype=np.int64)
    return out


def r_folds(replicate: int, n: int) -> np.ndarray:
    """R's fold of each row of ``binary_learners`` replicate ``replicate``, as 1 to 5.

    Parameters
    ----------
    replicate : int
        The replicate index.
    n : int
        The sample size, which the committed folds must match.

    Returns
    -------
    ndarray of int
        One fold label per row, in sample order.

    Raises
    ------
    FileNotFoundError
        When ``r_folds.csv.gz`` has not been written yet.
    ValueError
        When the committed folds do not cover this replicate at this ``n``.
    """
    table = _r_folds_table()
    if replicate not in table:
        raise ValueError(f"r_folds.csv.gz has no replicate {replicate}")
    folds = table[replicate]
    if folds.size != n:
        raise ValueError(
            f"r_folds.csv.gz holds {folds.size} rows for replicate {replicate}, not n = {n}. "
            f"The folds are drawn at the declared n = {PRIMARY_N}"
        )
    if set(np.unique(folds).tolist()) != set(range(1, SELECTION_FOLDS + 1)):
        raise ValueError(f"r_folds.csv.gz replicate {replicate} does not use folds 1 to 5")
    return folds


# --------------------------------------------------------------------------- the fits


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER, solver="lbfgs")


def _settings() -> dict[str, Any]:
    """The estimator keywords both scenarios share."""
    return {
        "outcome_learner": _logistic(),
        "treatment_learner": _logistic(),
        "cross_fit": False,
        "selection_inner_folds": 2,
        "penalty": False,
        "estimands": ("ate",),
        "ctmle_estimand": "ate",
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        # The outcome is binary, so no ``q_bounds``: the scaler is already the identity.
        "stratify_folds": STRATIFY_FOLDS,
        "max_iter": 100,
        "tol": 1e-10,
        "random_state": 0,
    }


def propensity_arrays(frame: pd.DataFrame, folds: np.ndarray) -> list[PropensityArrays]:
    """The ``binary_learners`` candidates: one ``(full, cv)`` pair per learner penalty.

    ``full`` is :class:`~cleverly.learners._l1_logistic.L1Logistic` fitted on every row.
    ``cv`` predicts the rows of each fold from a fit on the other four, so each row's ``cv``
    value comes from a fit that never read that row.

    Parameters
    ----------
    frame : DataFrame
        The replicate's sample.
    folds : ndarray of int
        R's fold of each row, 1 to 5.

    Returns
    -------
    list of PropensityArrays
        One pair per entry of :func:`learner_lambdas`, least adaptive first.
    """
    x = frame.loc[:, list(COVARIATES)].to_numpy(dtype=float)
    a = frame["A"].to_numpy()
    pairs = []
    for penalty in learner_lambdas():
        full = L1Logistic(penalty=penalty).fit(x, a).predict_proba(x)[:, 1]
        cv = np.full(len(frame), np.nan)
        for fold in range(1, SELECTION_FOLDS + 1):
            held = folds == fold
            model = L1Logistic(penalty=penalty).fit(x[~held], a[~held])
            cv[held] = model.predict_proba(x[held])[:, 1]
        pairs.append(PropensityArrays(full=full, cv=cv))
    return pairs


def fit_cleverly(
    frame: pd.DataFrame, scenario: str, replicate: int, *, folds: np.ndarray | None = None
) -> Any:
    """Fit one scenario's subject on one replicate.

    Parameters
    ----------
    frame : DataFrame
        The replicate's sample.
    scenario : str
        ``"binary_lasso"`` or ``"binary_learners"``.
    replicate : int
        The replicate index, which selects R's folds for ``binary_learners``.
    folds : ndarray of int or None
        R's folds, 1 to 5.  ``None`` reads them from ``r_folds.csv.gz``.

    Returns
    -------
    Any
        The single-parameter result.
    """
    if scenario == "binary_lasso":
        estimator = CTMLE(
            strategy="lasso",
            lambdas=(*study_lambdas(), 0.0),
            selection_folds=SELECTION_FOLDS,
            **_settings(),
        )
    elif scenario == "binary_learners":
        split = r_folds(replicate, len(frame)) if folds is None else np.asarray(folds)
        estimator = CTMLE(
            strategy="learners",
            treatment_candidates=propensity_arrays(frame, split),
            selection_split=(split - 1).tolist(),
            **_settings(),
        )
    else:
        raise KeyError(scenario)
    result = estimator.fit(frame, outcome="Y", treatment="A", covariates=list(COVARIATES)).single()
    if scenario == "binary_lasso":
        assert_unstratified_selection(result)
    else:
        selection = result.extra["ctmle"].folds
        if not np.array_equal(selection.assignment, split - 1):
            raise RuntimeError("the learners fit did not score its candidates on R's folds")
    return result


def _rows_from_result(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    estimate = result["ate"]
    reference = float(truth["ate"])
    std_error, low, high = reported_inference(estimate)
    psi = float(estimate.psi)
    if not math.isfinite(psi):
        raise RuntimeError(f"{scenario} replicate {replicate} gave a non-finite estimate")
    return [
        {
            "implementation": IMPLEMENTATION,
            "scenario": scenario,
            "replicate": replicate,
            "n": n,
            "estimand": "ate",
            "truth": reference,
            "estimate": psi,
            "inference_estimate": psi,
            "std_error": std_error,
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": math.nan,
        }
    ]


def cleverly_rows(
    frame: pd.DataFrame,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario, replicate)
    return _rows_from_result(result, truth, scenario, replicate, len(frame))


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario, replicate)
    sample = frame.copy()
    # The split the subject scored its candidates on, taken off the fit.  For binary_lasso R
    # selects against it.  For binary_learners it is folds_R - 1, and R redraws folds_R from
    # its own seed and checks the draw against r_folds.csv.gz.
    sample.insert(0, "selection_fold", result.extra["ctmle"].folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in truth.items()},
    }
    return sample, truth_row, _rows_from_result(result, truth, scenario, replicate, len(frame))


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        ((scenario, replicate, n),)
        for scenario in STUDY.scenarios
        for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([truth for _, truth, _ in outcomes])
    rows = pd.DataFrame([row for _, _, records in outcomes for row in records])
    return samples, truths, rows.loc[:, list(REPLICATE_COLUMNS)]


# --------------------------------------------------------------------------- side artefacts


def selection_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per R fit: the selected index, the accepted candidates and the adapter flag.

    Parameters
    ----------
    rows : DataFrame
        The assembled replicate rows, with the reference's extra columns.

    Returns
    -------
    DataFrame
        ``scenario``, ``replicate`` and :data:`REFERENCE_COLUMNS`, sorted.
    """
    reference = rows.loc[rows["implementation"] == REFERENCE]
    missing = [column for column in REFERENCE_COLUMNS if column not in reference.columns]
    if missing:
        raise RuntimeError(f"the R rows carry no {missing} columns for {SELECTION_FILE}")
    out = reference.loc[:, ["scenario", "replicate", *REFERENCE_COLUMNS]].copy()
    out["best_k"] = out["best_k"].astype(int)
    out["n_candidates"] = out["n_candidates"].astype(int)
    out["adapter_used"] = out["adapter_used"].astype(int)
    if out.duplicated(["scenario", "replicate"]).any():
        raise RuntimeError("the R rows hold a replicate twice")
    if (out.loc[out["scenario"] == "binary_learners", "adapter_used"] != 0).any():
        raise RuntimeError("the type-1 adapter applies to binary_lasso only")
    return out.sort_values(["scenario", "replicate"], ignore_index=True)


def selection_summary(rows: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """The descriptive paired comparison over the ``binary_lasso`` replicates R fit directly.

    The replicates where R needed the type-1 adapter are left out, so the comparison reads
    only fits that ``ctmleGlmnet`` completed itself.  The interval is a Student interval at
    :data:`SUMMARY_CONFIDENCE` on the paired estimate difference, ``cleverly`` minus R.  It is
    descriptive and states no verdict.

    Parameters
    ----------
    rows : DataFrame
        The assembled replicate rows.
    selection : DataFrame
        :func:`selection_rows` of the same rows.

    Returns
    -------
    DataFrame
        One row.
    """
    from scipy.stats import t as student

    lasso = selection.loc[selection["scenario"] == "binary_lasso"]
    kept = lasso.loc[lasso["adapter_used"] == 0, "replicate"]
    subset = rows.loc[(rows["scenario"] == "binary_lasso") & rows["replicate"].isin(kept)]
    paired = subset.pivot(index="replicate", columns="implementation", values="estimate")
    paired = paired.dropna()
    difference = (paired[IMPLEMENTATION] - paired[REFERENCE]).to_numpy(dtype=float)
    count = int(difference.size)
    mean = float(np.mean(difference)) if count else math.nan
    sd = float(np.std(difference, ddof=1)) if count > 1 else math.nan
    half = (
        float(student.ppf(0.5 + SUMMARY_CONFIDENCE / 2.0, count - 1)) * sd / math.sqrt(count)
        if count > 1
        else math.nan
    )
    truth = subset.groupby("replicate")["truth"].first().reindex(paired.index)
    return pd.DataFrame(
        [
            {
                "scenario": "binary_lasso",
                "replicates": len(lasso),
                "adapter_replicates": int((lasso["adapter_used"] != 0).sum()),
                "compared_replicates": count,
                "confidence_level": SUMMARY_CONFIDENCE,
                "mean_difference": mean,
                "sd_difference": sd,
                "ci_lower": mean - half,
                "ci_upper": mean + half,
                "max_abs_difference": float(np.max(np.abs(difference))) if count else math.nan,
                "rmse_cleverly": float(np.sqrt(np.mean((paired[IMPLEMENTATION] - truth) ** 2))),
                "rmse_reference": float(np.sqrt(np.mean((paired[REFERENCE] - truth) ** 2))),
            }
        ]
    )


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``r-selection.csv`` and ``r-selection-summary.csv``, from the R rows' extra columns.

    Parameters
    ----------
    rows : DataFrame
        The assembled replicate rows of both implementations.

    Returns
    -------
    dict
        Each side artefact by its file name.
    """
    selection = selection_rows(rows)
    return {
        SELECTION_FILE: selection,
        SELECTION_SUMMARY_FILE: selection_summary(rows, selection),
    }
