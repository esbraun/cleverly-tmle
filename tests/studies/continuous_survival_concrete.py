r"""The ``concrete`` comparator phase of ``continuous-time-survival`` (roadmap X6 part f).

R ``concrete`` 1.0.5 (Chen et al.; CRAN, archived 2024-04-02) is the only other implementation
of the one-step continuous-time TMLE of Rytgaard and van der Laan (2024).  It tilts each arm's
hazard only for the units whose received arm is that arm, which is
``ContinuousSurvivalTMLE(tilt="factual")``, uses the Euclidean norm, fits every nuisance on all
rows, and stops when every target's :math:`|P_n D^*|` is at most :math:`\hat\sigma / (3 \sqrt n
\log n)`.  This phase pairs the two on the same draws of the registered law
(:mod:`tests.studies.continuous_survival_law`) in three cells, each at :data:`N` units and
:data:`REPLICATES` draws:

=====================  ===========================================  =================================
cell                   cleverly                                     ``concrete``
=====================  ===========================================  =================================
``p1_parity``          ``tilt="factual"``, ``cross_fit=False``,     the same models: Cox with
                       ``norm="euclidean"``, ``tol`` at concrete's  treatment interactions for both
                       rule, influence-curve variance; Cox with     causes, main-terms Cox for
                       treatment interactions for both causes       censoring, ``SL.glm`` for the
                       and main terms for censoring, both built     treatment
                       as concrete builds them
                       (:class:`ConcreteCoxHazard`); the main-terms
                       logistic for the treatment
``p2_correct``         the registered default fit                   a library holding the truth:
                       (:func:`tests.studies.continuous_survival.   Cox on the treatment, on main
                       engine`)                                     terms and with interactions for
                                                                    every hazard; ``SL.glm`` and
                                                                    ``SL.glmnet`` for the treatment
``p3_hazard_wrong``    the default targeting (``tilt="all"``) and   Cox on the treatment alone for
                       the P1 targeting (``tilt="factual"``), both  both causes, main-terms Cox for
                       with the Cox model on the treatment alone    censoring, ``SL.glm``
                       for both causes and the P1 censoring and
                       treatment models
=====================  ===========================================  =================================

**Readings, declared before any run** (X6 plan 2.6 and 5.6).

* ``p1_parity``: every estimate of cleverly's factual fit agrees with ``concrete``'s to
  :data:`PARITY_TOLERANCE` on every draw.  The agreement shows that cleverly's walk and stopping
  rule reproduce an independent implementation of the factual form.  The tolerance covers the
  three numerical forms the two implementations do not share: ``concrete`` computes each
  survival as :math:`\exp(-\Lambda)` where cleverly takes the product limit, accumulates each
  cause's risk with the survival after the jump where cleverly takes it before, and also stops
  on the all-cause survival's score.  ``max_abs_initial_difference`` in ``parity.csv`` reads
  the plug-ins before any step, so a reader can tell those forms from a difference of the walk.
* ``p3_hazard_wrong``: ``concrete``'s standardized bias on :data:`DECLARED_TARGET` lies above
  :data:`MARGIN` (its lower one-sided 95% bound), and cleverly's default fit lies inside it (its
  upper bound).  This is the factual-tilt mutation control seen from outside: with the event
  hazards wrong and the treatment model right only the all-unit tilt is doubly robust.
* ``p2_correct`` and every other row are reported as measured, with no verdict.  The X6 plan
  predicted both fits unbiased; ``concrete`` 1.0.5 takes each hazard's baseline from the
  treatment-only Cox model (``getHazEstimate.R``, ``BaseHazCox``) whatever model it selected,
  so its initial hazards are not those of the selected model, and the factual tilt does not
  remove the difference.  ``paired.csv`` reads the paired differences in squared error and
  coverage.

**The hazard construction.**  ``concrete`` multiplies the treatment-only model's Breslow baseline
at :math:`Z = 0` by :math:`\exp\{(x - m)^\top\hat\beta\}`, where :math:`\hat\beta` is the selected
model's and :math:`m` holds the column means of its design, except that ``survival::coxph``
leaves a column whose values all lie in :math:`\{-1, 0, 1\}` uncentered (``nocenter``).
:class:`ConcreteCoxHazard` builds the same hazard, so P1 compares targeting with targeting rather
than two baselines.  The registered study's Breslow baseline (``CoxHazard``) is what P2 and P3's
default fit use.

**Pin.**  The comparator is the CRAN archive tarball :data:`TARBALL_URL`, checked against
:data:`TARBALL_SHA256` in ``tests/canonical/concrete/Dockerfile`` before ``R CMD INSTALL``.  No
commit of ``imbroglio-dc/concrete`` matches it byte for byte (X6 ``sources.md`` section 6); the
nearest are :data:`NEAREST_COMMITS`.

**Seeds.**  Draw ``r`` of a cell reads ``stream_seed(STUDY, "concrete", cell, r)`` of the
registered record, a stream no other analysis of the record reads.  Each R group sets
``reference_seed(STUDY, RUNNER_NAME, "*", r)``; ``concrete`` draws only its cross-validation
folds from it.

Run the phase with ``python -m tests.canonical.continuous_survival_concrete.run_phase``; that
module's docstring gives the commands.
"""

from __future__ import annotations

import contextlib
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cleverly.survival import ContinuousSurvivalTMLE, CoxHazard
from cleverly.survival import engine as engine_module
from cleverly.survival._targeting import default_cutoff
from tests.studies import continuous_survival as registered
from tests.studies import continuous_survival_law as law
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import reference_seed, stream_seed

#: The record whose law, truths and seed entropy the phase reads.
STUDY = registered.STUDY
REPLICATES = 500
N = 1_000
CELLS = ("p1_parity", "p2_correct", "p3_hazard_wrong")

#: The bound on every paired estimate difference of ``p1_parity``.
PARITY_TOLERANCE = 2e-3
#: The standardized-bias margin of the registered record.
MARGIN = float(registered.MARGINS.standardized_bias)
#: The one-sided normal quantile of the P3 bounds.
ONE_SIDED = 1.6448536269514722
#: P3's declared target: the treated arm's risk of cause 1 at the middle time, the registered
#: record's double-robustness target.
DECLARED_TARGET = law.level_name("treated", "cause1", law.TIMES[1])
#: Every target both implementations report: each arm's risk of each cause at each time, and
#: each cause's risk difference.
ESTIMANDS: tuple[str, ...] = tuple(law.TRUTH)

TARBALL_URL = "https://cran.r-project.org/src/contrib/Archive/concrete/concrete_1.0.5.tar.gz"
#: Fetched from the CRAN archive and from cloud.r-project.org on 2026-11-19; both agree.
TARBALL_SHA256 = "4b7a67357bb0852d70322180f7be9e79a11c99321f73c01f34e720d8661c1416"
VERSION = "1.0.5"
#: ``94a86a4`` agrees with the tarball on 29 of 31 source files, ``ef91f89`` on 28 and holds the
#: version string; the tarball's ``R/formatArguments.R`` is in no commit.
NEAREST_COMMITS = ("94a86a4", "ef91f89")
IMAGE = "cleverly-concrete:1.0.5"
RUNNER = "continuous_survival_concrete/run_concrete.R"
RUNNER_NAME = "run_concrete.R"

SUBJECT = "cleverly"
SUBJECT_FACTUAL = "cleverly-factual"
REFERENCE = "concrete"

#: ``cell -> (implementation, configuration)`` of every cleverly fit the phase makes.
FITS: dict[str, tuple[tuple[str, str], ...]] = {
    "p1_parity": ((SUBJECT_FACTUAL, "parity"),),
    "p2_correct": ((SUBJECT, "default"),),
    "p3_hazard_wrong": ((SUBJECT, "hazard_wrong"), (SUBJECT_FACTUAL, "hazard_wrong_factual")),
}

#: The right-hand side of each Cox formula, as the R runner writes it.
FORMULAS = {
    "treatment": "Z",
    "main": "Z + X1 + X2 + X3 + X4",
    "interaction": "Z * (X1 + X2 + X3 + X4)",
}
#: ``cell -> (event hazards, censoring hazard, treatment library, folds)`` of ``concrete``.
#: The folds only reach the cross-validated selection, so a one-member library ignores them.
CONCRETE_CELLS: dict[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], int]] = {
    "p1_parity": (("interaction",), ("main",), ("SL.glm",), 2),
    "p2_correct": (
        ("treatment", "main", "interaction"),
        ("treatment", "main", "interaction"),
        ("SL.glm", "SL.glmnet"),
        10,
    ),
    "p3_hazard_wrong": (("treatment",), ("main",), ("SL.glm",), 2),
}

#: The columns a phase row carries beside the published schema.  ``ic_std_error`` is the
#: standard deviation of the influence curve over :math:`\sqrt n`; for ``concrete``'s risk
#: difference it reads the curves' difference, where the native ``std_error`` adds the two arms'
#: variances.  For cleverly it repeats ``std_error``.
EXTRA_COLUMNS = ("ic_std_error", "steps", "converged", "seconds")
ROW_COLUMNS = (*REPLICATE_COLUMNS, *EXTRA_COLUMNS)
SAMPLE_COLUMNS = ("scenario", "replicate", "row", "time", "event", "Z", *law.BASELINE)


# ---------------------------------------------------------------------------- the draws


def sample_seed(cell: str, replicate: int) -> int:
    """The seed of draw ``replicate`` of ``cell``."""
    if cell not in CELLS:
        raise KeyError(f"no cell {cell!r}")
    return stream_seed(STUDY, "concrete", cell, replicate)


def group_seed(replicate: int) -> int:
    """The seed the R harness sets before it fits replicate ``replicate``'s three cells."""
    return reference_seed(STUDY, RUNNER_NAME, "*", replicate)


def draw(cell: str, replicate: int, n: int = N) -> pd.DataFrame:
    """One draw of the registered base law."""
    return law.sample(n, sample_seed(cell, replicate))


def sample_rows(cell: str, replicate: int, frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` in the layout the R runner reads, one row per unit."""
    out = frame.loc[:, ["time", "event", "Z", *law.BASELINE]].copy()
    out.insert(0, "row", np.arange(len(frame)))
    out.insert(0, "replicate", replicate)
    out.insert(0, "scenario", cell)
    return out.loc[:, list(SAMPLE_COLUMNS)]


def truth_rows(cells: Sequence[str], replicates: int) -> pd.DataFrame:
    """Every target's truth for every cell and replicate, the runner's truth file."""
    return pd.DataFrame(
        [
            {"scenario": cell, "replicate": r, "estimand": name, "truth": float(law.TRUTH[name])}
            for cell in cells
            for r in range(replicates)
            for name in ESTIMANDS
        ]
    )


def seed_rows(replicates: int) -> pd.DataFrame:
    """The harness's seed file: one ``*`` row per replicate."""
    return pd.DataFrame(
        {
            "scenario": ["*"] * replicates,
            "replicate": list(range(replicates)),
            "seed": [group_seed(r) for r in range(replicates)],
        }
    )


# ------------------------------------------------------------------- the hazard learner


def _design(X: np.ndarray, terms: str) -> np.ndarray:
    """The Cox design of ``terms`` with the treatment in column 0, as ``CoxHazard`` builds it."""
    if terms == "treatment":
        return X[:, :1]
    if terms == "main":
        return X
    if terms == "interaction":
        return np.hstack([X, X[:, :1] * X[:, 1:]])
    raise ValueError(f"terms is 'treatment', 'main' or 'interaction'; got {terms!r}")


class ConcreteCoxHazard(BaseEstimator):  # type: ignore[misc]
    r"""The Cox hazard as ``concrete`` 1.0.5 predicts it.

    The coefficients are the maximum partial-likelihood fit of ``terms`` (``CoxHazard``, which
    agrees with ``survival::coxph`` on untied times).  The cumulative hazard of a row is the
    Breslow baseline of the treatment-only model at :math:`Z = 0` times
    :math:`\exp\{(x - m)^\top \hat\beta\}`, where :math:`m` holds the fitting rows' column means
    of the design, zero for a column whose values all lie in :math:`\{-1, 0, 1\}`.  That is
    ``basehaz(coxph(Surv(time, event == j) ~ Z), centered = TRUE)`` times
    ``predict(fit, type = "risk")`` in ``getHazFit`` and ``getHazSurvPred``.

    Parameters
    ----------
    terms : {"treatment", "main", "interaction"}
        The selected model's linear predictor, as :class:`~cleverly.survival.CoxHazard` reads it.
    """

    def __init__(self, terms: str = "main") -> None:
        self.terms = terms

    def fit(self, X: Any, time: Any, event: Any) -> ConcreteCoxHazard:
        """Fit the selected model and the treatment-only baseline on the same rows."""
        X = np.asarray(X, dtype=float)
        self.model_ = CoxHazard(self.terms).fit(X, time, event)  # type: ignore[arg-type]
        self.baseline_ = CoxHazard("treatment").fit(X, time, event)
        design = _design(X, self.terms)
        uncentered = np.all(np.isin(design, (-1.0, 0.0, 1.0)), axis=0)
        self.means_ = np.where(uncentered, 0.0, design.mean(axis=0))
        return self

    def cumulative_hazard(self, X: Any, times: Any) -> np.ndarray:
        """``(n, m)`` the cumulative hazard of each row at each time."""
        X = np.asarray(X, dtype=float)
        risk = np.exp((_design(X, self.terms) - self.means_) @ self.model_.coef_)
        base = self.baseline_.cumulative_hazard(np.zeros((1, X.shape[1])), times)[0]
        return np.asarray(risk[:, None] * base[None, :])


# ------------------------------------------------------------------------------ the fits


def logistic() -> Pipeline:
    """The unpenalized main-terms logistic of cleverly's default treatment library."""
    return Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(C=np.inf, max_iter=2000))]
    )


@contextlib.contextmanager
def logistic_only() -> Iterator[None]:
    """Fits inside see a treatment library of the main-terms logistic alone.

    ``cross_fit=False`` refuses a user treatment learner, and the default library's selection
    may pick a lasso member, which ``SL.glm`` never is.  A one-member library is selected
    trivially and refitted on every row, which is what ``concrete``'s one-member
    ``SuperLearner`` returns.
    """
    with mock.patch.object(
        engine_module, "_logistic_candidates", lambda random_state: [("logistic", logistic())]
    ):
        yield


def parity_cutoff(n: int) -> float:
    r"""``concrete``'s stopping bound on :math:`|P_n D^*| / \hat\sigma`, :math:`1 / (3 \sqrt n \log n)`."""
    return float(default_cutoff(n)) / 3.0


def engine(configuration: str, n: int = N) -> ContinuousSurvivalTMLE:
    """The cleverly fit of one configuration of :data:`FITS`."""
    if configuration == "default":
        return registered.engine(cross_fit=True)
    factual = configuration in ("parity", "hazard_wrong_factual")
    if configuration == "parity":
        hazards: tuple[Any, ...] = (ConcreteCoxHazard("interaction"),)
    elif configuration in ("hazard_wrong", "hazard_wrong_factual"):
        hazards = (ConcreteCoxHazard("treatment"),) if factual else (CoxHazard("treatment"),)
    else:
        raise KeyError(f"no configuration {configuration!r}")
    censoring = (ConcreteCoxHazard("main"),) if factual else (CoxHazard("main"),)
    settings: dict[str, Any] = (
        {
            "tilt": "factual",
            "norm": "euclidean",
            "tol": parity_cutoff(n),
            "variance": "influence_curve",
        }
        if factual
        else {}
    )
    return ContinuousSurvivalTMLE(
        law.TIMES,
        arms=law.ARMS,
        reference=law.REFERENCE,
        causes=law.CAUSES,
        cross_fit=False,
        hazard_library=hazards,
        censoring_library=censoring,
        simultaneous=False,
        random_state=0,
        **settings,
    )


def fit(frame: pd.DataFrame, configuration: str) -> Any:
    """One cleverly fit of one configuration."""
    subject = engine(configuration, len(frame))
    if configuration == "default":
        return registered.fit_with(frame, subject)
    with logistic_only():
        return registered.fit_with(frame, subject)


def result_rows(
    result: Any, implementation: str, cell: str, replicate: int, seconds: float
) -> list[dict[str, Any]]:
    """One row per target of :data:`ESTIMANDS`, in :data:`ROW_COLUMNS`."""
    record = result.record
    initial = result.initial_estimates
    rows = []
    for name in ESTIMANDS:
        estimate = result[name]
        low, high = estimate.ci
        truth = float(law.TRUTH[name])
        rows.append(
            {
                "implementation": implementation,
                "scenario": cell,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": truth,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= truth <= high),
                "initial_estimate": float(initial[name]),
                "ic_std_error": float(estimate.std_error),
                "steps": float(record.steps),
                "converged": float(record.converged),
                "seconds": seconds,
            }
        )
    return rows


def fit_replicate(cell: str, replicate: int, n: int = N) -> list[dict[str, Any]]:
    """Every cleverly fit of one draw of one cell."""
    frame = draw(cell, replicate, n)
    rows: list[dict[str, Any]] = []
    for implementation, configuration in FITS[cell]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            result = fit(frame, configuration)
            seconds = time.perf_counter() - start
        rows += result_rows(result, implementation, cell, replicate, seconds)
    return rows


# --------------------------------------------------------------------------- the readings


def read_reference(path: Any) -> pd.DataFrame:
    """The R runner's rows, every float exactly as written."""
    frame = pd.read_csv(path, float_precision="round_trip")
    missing = [column for column in ROW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"the concrete rows lack {missing}")
    if set(frame["implementation"]) != {REFERENCE}:
        raise ValueError("the concrete rows name another implementation")
    return frame.loc[:, list(ROW_COLUMNS)]


def _paired(rows: pd.DataFrame, cell: str, left: str, right: str) -> pd.DataFrame:
    """``left`` and ``right``'s rows of ``cell`` on the replicates both fitted."""
    selected = rows[rows["scenario"] == cell]
    a = selected[selected["implementation"] == left]
    b = selected[selected["implementation"] == right]
    return a.merge(b, on=["replicate", "estimand"], suffixes=("_left", "_right"))


def parity(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell and target: cleverly's factual fit against ``concrete`` on the same draws."""
    records = []
    for cell in ("p1_parity", "p3_hazard_wrong"):
        paired = _paired(rows, cell, SUBJECT_FACTUAL, REFERENCE)
        for name in ESTIMANDS:
            group = paired[paired["estimand"] == name]
            difference = group["estimate_left"] - group["estimate_right"]
            initial = group["initial_estimate_left"] - group["initial_estimate_right"]
            largest = float(np.max(np.abs(difference))) if len(group) else float("nan")
            records.append(
                {
                    "cell": cell,
                    "estimand": name,
                    "fits": len(group),
                    "max_abs_difference": largest,
                    "mean_difference": float(np.mean(difference)) if len(group) else np.nan,
                    "max_abs_initial_difference": (
                        float(np.max(np.abs(initial))) if len(group) else np.nan
                    ),
                    "mean_std_error_ratio": (
                        float(np.mean(group["ic_std_error_left"] / group["ic_std_error_right"]))
                        if len(group)
                        else np.nan
                    ),
                    "within_tolerance": bool(largest <= PARITY_TOLERANCE),
                }
            )
    return pd.DataFrame.from_records(records)


def performance(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell, implementation and target: bias, spread, RMSE, standard error and coverage."""
    records = []
    for (cell, implementation, name), group in rows.groupby(
        ["scenario", "implementation", "estimand"], sort=True
    ):
        estimate = group["estimate"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        fits = int(estimate.size)
        bias = float(np.mean(estimate) - truth)
        spread = float(np.std(estimate, ddof=1)) if fits > 1 else float("nan")
        standardized = bias / spread
        records.append(
            {
                "cell": cell,
                "implementation": implementation,
                "estimand": name,
                "fits": fits,
                "truth": truth,
                "bias": bias,
                "empirical_se": spread,
                "standardized_bias": standardized,
                # The delta-method standard error of a mean over a standard deviation.
                "standardized_bias_se": float(np.sqrt((1.0 + standardized**2 / 2.0) / fits)),
                "rmse": float(np.sqrt(np.mean((estimate - truth) ** 2))),
                "mean_std_error": float(group["std_error"].mean()),
                "coverage": float(group["covered"].mean()),
                "converged_share": float(group["converged"].mean()),
                "mean_steps": float(group["steps"].mean()),
                "mean_seconds": float(group["seconds"].mean()),
            }
        )
    return pd.DataFrame.from_records(records)


def paired_differences(rows: pd.DataFrame) -> pd.DataFrame:
    """Per cell and target, cleverly's default targeting minus ``concrete`` on the same draws."""
    records = []
    for cell in ("p2_correct", "p3_hazard_wrong"):
        paired = _paired(rows, cell, SUBJECT, REFERENCE)
        for name in ESTIMANDS:
            group = paired[paired["estimand"] == name]
            fits = len(group)
            squared = (group["estimate_left"] - group["truth_left"]) ** 2 - (
                group["estimate_right"] - group["truth_right"]
            ) ** 2
            covered = group["covered_left"] - group["covered_right"]
            records.append(
                {
                    "cell": cell,
                    "estimand": name,
                    "fits": fits,
                    "squared_error_difference": float(squared.mean()),
                    "squared_error_difference_se": float(squared.std(ddof=1) / np.sqrt(fits)),
                    "coverage_difference": float(covered.mean()),
                    "coverage_difference_se": float(covered.std(ddof=1) / np.sqrt(fits)),
                }
            )
    return pd.DataFrame.from_records(records)


def verdicts(parity_table: pd.DataFrame, performance_table: pd.DataFrame) -> pd.DataFrame:
    """The three declared readings, each with its statistic, threshold and verdict."""
    declared = parity_table[parity_table["cell"] == "p1_parity"]
    largest = float(declared["max_abs_difference"].max())

    def standardized(implementation: str) -> tuple[float, float]:
        row = performance_table[
            (performance_table["cell"] == "p3_hazard_wrong")
            & (performance_table["implementation"] == implementation)
            & (performance_table["estimand"] == DECLARED_TARGET)
        ].iloc[0]
        return abs(float(row["standardized_bias"])), float(row["standardized_bias_se"])

    reference, reference_se = standardized(REFERENCE)
    subject, subject_se = standardized(SUBJECT)
    detected = reference - ONE_SIDED * reference_se
    inside = subject + ONE_SIDED * subject_se
    return pd.DataFrame(
        [
            {
                "reading": "p1_estimates_agree",
                "cell": "p1_parity",
                "estimand": "every target",
                "statistic": largest,
                "threshold": PARITY_TOLERANCE,
                "verdict": "pass" if largest <= PARITY_TOLERANCE else "fail",
            },
            {
                "reading": "p3_concrete_bias_above_margin",
                "cell": "p3_hazard_wrong",
                "estimand": DECLARED_TARGET,
                "statistic": detected,
                "threshold": MARGIN,
                "verdict": "pass" if detected > MARGIN else "fail",
            },
            {
                "reading": "p3_cleverly_bias_within_margin",
                "cell": "p3_hazard_wrong",
                "estimand": DECLARED_TARGET,
                "statistic": inside,
                "threshold": MARGIN,
                "verdict": "pass" if inside < MARGIN else "fail",
            },
        ]
    )


#: The phase's tables, by file name, in the order the manifest hashes them.
TABLES = ("parity.csv", "performance.csv", "paired.csv", "verdicts.csv")


def tables(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every published table, a function of the committed rows alone."""
    parity_table = parity(rows)
    performance_table = performance(rows)
    return {
        "parity.csv": parity_table,
        "performance.csv": performance_table,
        "paired.csv": paired_differences(rows),
        "verdicts.csv": verdicts(parity_table, performance_table),
    }


def declaration() -> Mapping[str, Any]:
    """What the run was declared to be, recorded in the manifest and checked by the replay."""
    return {
        "study": STUDY.slug,
        "phase": "concrete",
        "cells": list(CELLS),
        "replicates": REPLICATES,
        "n": N,
        "estimands": list(ESTIMANDS),
        "parity_tolerance": PARITY_TOLERANCE,
        "margin": MARGIN,
        "declared_target": DECLARED_TARGET,
        "fits": {cell: [list(pair) for pair in pairs] for cell, pairs in FITS.items()},
        "concrete_cells": {
            cell: {
                "hazards": list(spec[0]),
                "censoring": list(spec[1]),
                "treatment": list(spec[2]),
                "folds": spec[3],
            }
            for cell, spec in CONCRETE_CELLS.items()
        },
        "formulas": dict(FORMULAS),
        "reference": {
            "implementation": REFERENCE,
            "version": VERSION,
            "tarball_url": TARBALL_URL,
            "tarball_sha256": TARBALL_SHA256,
            "nearest_commits": list(NEAREST_COMMITS),
            "image": IMAGE,
        },
    }


#: Where a run's artifacts are committed.
ARTIFACTS = ROOT / "tests" / "canonical" / "continuous_survival_concrete"
#: The phase's own modules, hashed in its manifest as a record of the run; no test gates them.
MODULES = (
    "tests/studies/continuous_survival_concrete.py",
    "tests/canonical/continuous_survival_concrete/run_phase.py",
    "tests/studies/continuous_survival.py",
    "tests/studies/continuous_survival_law.py",
)
#: The reference sources a run hashes, from the repository root; a replay gates each one.
REFERENCE_SOURCES = (
    "tests/canonical/concrete/Dockerfile",
    f"tests/canonical/{RUNNER}",
    "tests/canonical/study_harness.R",
)
