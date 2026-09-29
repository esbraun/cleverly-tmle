"""The laws, the fit, the truth harness and the oracle SE of the RM30 learned-rule studies.

The RM30 section of ``docs/roadmap.md`` declares everything here before any run: "The laws, and
the constants computed before the declaration", "The gated study", "Rules that both studies
follow" and "The fold-partition seeds".  Two registered studies share this module.
``learned-rule-cvtmle`` is gated, and ``learned-rule-cvtmle-boundary`` reports.

Every law draws ``W1 ~ U(-1, 1)`` and ``W2 ~ Bernoulli(0.5)``, a binary treatment with
``logit g0(1 | W) = 0.3 W1 - 0.2 W2``, and a binary outcome with
``logit Qbar0(a, W) = 0.2 + 0.5 W1 - 0.3 W2 + a b(W)``.  The laws differ in the blip ``b``.

The target of a fit is data-adaptive.  It is the average over the ten outer folds of the value
``Psi_d(P0) = E Qbar0(d(W), W)`` of the rule ``d_v`` learned on the training rows of fold ``v``.
Each replication therefore carries its own truth.  The harness refits each fold's outcome
learner on the rows the fit learned the rule from, and it refuses the replication unless the
refit gives the fit's own rule on every validation row (rule L4).  It then integrates the value
of the refit rule by the trapezoid rule on 4,001 points of ``W1`` for each ``W2``, with the known
``Qbar0``.  The same quadrature gives the oracle standard error.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.exceptions import ConvergenceWarning as CleverlyConvergenceWarning
from cleverly.interventions import LearnedRule
from cleverly.interventions.base import RegimeSet
from cleverly.learners._fitting import predict_mean
from cleverly.utils.parallel import map_parallel
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import replicate_seed, stream_seed

#: The outer folds of every fit.  At n = 2,000 each fold holds 200 rows, so the weights
#: ``1/V`` and ``n_v / n`` agree and the two variance forms coincide.
N_FOLDS = 10

#: The quadrature grid of ``W1`` on its support, the declared 4,001 points.
GRID = np.linspace(-1.0, 1.0, 4001)

#: The two values of ``W2``, each with probability one half.
W2_LEVELS = (0.0, 1.0)

#: The reported parameter of every fit.
ESTIMAND = "ey_learned_rule[learned rule]"

#: The outcome and the covariates, in the order the package builds the outcome design:
#: ``CausalData.treatment_design`` puts the treatment first, so a learner sees
#: ``(A, W1, W2)``.
COVARIATES = ("W1", "W2")

#: The two-sided 95% normal multiplier of every interval the study writes itself.
CRITICAL = 1.959963984540054

#: The blip ``b(W)`` of each declared law, on the logit scale, as a function of ``W1``.
BLIPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "non_exceptional": lambda w1: 0.1 + 1.0 * w1,
    "misspecified_limit": lambda w1: 0.8 * w1 + w1**2 - 0.3,
    "weak_blip": lambda w1: 0.15 * w1,
    "exceptional": lambda w1: 0.0 * w1,
}

#: The warning classes a declared run counts in its ``solver_warnings`` column (rule L4):
#: the scikit-learn solver warning of a learner, and the package's own targeting warning.
SOLVER_WARNINGS: tuple[type[Warning], ...] = (SklearnConvergenceWarning, CleverlyConvergenceWarning)

#: The column names a row carries beside the shared schema.  ``oracle_se`` is descriptive,
#: and no rule reads it.  ``rule_rows_checked`` counts the validation rows whose refit rule
#: equals the fit's rule, which is every row of a replication that completed.
#: ``solver_warnings`` counts the warnings of the fit and of its truth refits, and no rule
#: reads it either (rule L4).
HARNESS_COLUMNS = ("oracle_se", "rule_rows_checked", "solver_warnings")


# --------------------------------------------------------------------------- the law


def blip(law: str, w1: np.ndarray) -> np.ndarray:
    """The law's blip ``b(W)`` on the logit scale."""
    return np.asarray(BLIPS[law](np.asarray(w1, dtype=float)), dtype=float)


def g0(w1: np.ndarray, w2: np.ndarray | float) -> np.ndarray:
    """``P(A = 1 | W)``, from ``logit g0 = 0.3 W1 - 0.2 W2``."""
    return np.asarray(expit(0.3 * np.asarray(w1, dtype=float) - 0.2 * np.asarray(w2)), dtype=float)


def qbar0(a: np.ndarray | float, w1: np.ndarray, w2: np.ndarray | float, law: str) -> np.ndarray:
    """``P(Y = 1 | A = a, W)``, from ``logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + a b(W)``."""
    w1 = np.asarray(w1, dtype=float)
    return np.asarray(
        expit(0.2 + 0.5 * w1 - 0.3 * np.asarray(w2) + np.asarray(a) * blip(law, w1)), dtype=float
    )


def draw(law: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows from the named law, with columns ``W1, W2, A, Y``."""
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, g0(w1, w2)).astype(float)
    y = rng.binomial(1, qbar0(a, w1, w2, law)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


# ------------------------------------------------------------------------- the learners


def outcome_learner() -> Pipeline:
    """The declared outcome learner: every pairwise product of ``(A, W1, W2)``, logistic."""
    return Pipeline(
        [
            ("features", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("model", LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)),
        ]
    )


def drop_w1_learner() -> Pipeline:
    """The ``targeting_necessity`` outcome learner, which keeps ``A`` and ``W2`` only.

    Columns 0 and 2 of the design ``(A, W1, W2)``.  Its limit blip is nonzero at both values
    of ``W2``, so its limit rule treats every unit and C3 holds.
    """
    return Pipeline(
        [
            ("keep", ColumnTransformer([("keep", "passthrough", [0, 2])])),
            ("features", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("model", LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)),
        ]
    )


def forest_learner() -> RandomForestClassifier:
    """The ``fold_locality`` outcome learner.  Its fixed ``random_state`` makes a refit exact."""
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0)


def treatment_learner() -> LogisticRegression:
    """The declared treatment learner on ``(W1, W2)``, a correct parametric model."""
    return LogisticRegression(C=1e6, max_iter=1000)


# ------------------------------------------------------------------------------ the fit


class ValidationRowRule(TMLE):
    """The deliberate mutation of ``fold_locality``: learn each fold's rule on its own rows.

    Study-only.  The cross-fitted nuisances, the targeting and the variance are the fit's own.
    Only the rule changes: fold ``v``'s rule is the plug-in of the outcome learner fitted on
    fold ``v``'s *validation* rows, the rows that then evaluate it.
    """

    def _regimes(self, data: Any, estimates: Any) -> RegimeSet | None:
        assert self.learned_rule is not None
        design = np.asarray(data.treatment_design(), dtype=float)
        treat = np.zeros(data.n, dtype=bool)
        for _, test in estimates.folds:
            model = fit_on_rows(
                self.outcome_learner,
                design,
                np.asarray(data.outcome, dtype=float),
                np.asarray(data.weights, dtype=float),
                np.asarray(test),
                "classification",
                None,
            )
            treat[test] = _blip(model, np.asarray(data.covariates, dtype=float)[test]) > 0.0
        values = np.zeros((data.n, 2, 1), dtype=float)
        values[:, 1, 0] = treat
        values[:, 0, 0] = ~treat
        return RegimeSet((self.learned_rule.name,), values, 0.0)


def estimator(learner: Any, fold_seed: int, *, mutation: bool = False) -> TMLE:
    """The declared estimator, with the fold seed of rule L2."""
    kind = ValidationRowRule if mutation else TMLE
    return kind(
        learned_rule=LearnedRule(),
        n_folds=N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=treatment_learner(),
        random_state=fold_seed,
    )


@contextlib.contextmanager
def solver_warning_count() -> Iterator[list[int]]:
    """Count the solver warnings raised inside the block, in ``counter[0]``.

    Every warning is recorded, so a warning that a filter elsewhere would show once is
    counted each time it fires.  The other warnings are re-emitted unchanged.
    """
    counter = [0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield counter
    for message in caught:
        if issubclass(message.category, SOLVER_WARNINGS):
            counter[0] += 1
        else:
            warnings.warn_explicit(
                message.message, message.category, message.filename, message.lineno
            )


def fit(frame: pd.DataFrame, learner: Any, fold_seed: int, *, mutation: bool = False) -> Any:
    """Fit the declared estimator to one sample and return the single result."""
    return (
        estimator(learner, fold_seed, mutation=mutation)
        .fit(frame, outcome="Y", treatment="A", covariates=list(COVARIATES))
        .single()
    )


def untargeted_estimate(result: Any) -> float:
    """The same fit without its fluctuation: the fold average of the initial plug-ins.

    Each fold's plug-in is the mean of ``Qbar_v(d_i, W_i)`` over its validation rows, read off
    the out-of-fold outcome regression at the fit's own rule, and the folds are averaged with
    the weight ``1/V`` that the fit uses.
    """
    nuisance = result.nuisance
    control, treated = nuisance.outcome.levels
    scaler = nuisance.scaler
    q0 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[control]), dtype=float)
    q1 = np.asarray(scaler.unscale_levels(nuisance.outcome.arms[treated]), dtype=float)
    treat = rule_of(result)
    plug_in = np.where(treat, q1, q0)
    value = float(np.mean([np.mean(plug_in[test]) for _, test in nuisance.folds]))
    if not np.isfinite(value):
        raise RuntimeError(f"a non-finite untargeted estimate (rule L4): {value}")
    return value


def rule_of(result: Any) -> np.ndarray:
    """The rule the fit used at every row: ``True`` where it assigns the higher arm code."""
    return np.asarray(result.nuisance.regimes.values[:, 1, 0], dtype=float) > 0.5


# -------------------------------------------------------------------------- the harness


class HarnessMismatch(RuntimeError):
    """A truth refit whose rule differs from the fit's rule on a validation row (rule L4)."""


@dataclass(frozen=True)
class Truth:
    """What the harness computes for one fit.

    Parameters
    ----------
    truth : float
        ``(1/V) sum_v Psi_{d_v}(P0)``, the value of the rules the fit used.
    oracle_se : float
        ``sqrt(V^-2 sum_v Var_P0 D*(d_v, Qbar0, g0) / n_v)``, by the same quadrature.
    fold_values : tuple of float
        ``Psi_{d_v}(P0)`` for each fold.
    rows_checked : int
        The validation rows whose refit rule equals the fit's rule.
    """

    truth: float
    oracle_se: float
    fold_values: tuple[float, ...]
    rows_checked: int


def _blip(model: Any, covariates: np.ndarray) -> np.ndarray:
    """``Qbar(1, W) - Qbar(0, W)`` of a fitted outcome learner at the rows of ``covariates``.

    The design is ``(A, W1, W2)``, as the package builds it, and the prediction is the
    package's own ``predict_mean`` clipped to the unit interval, as the fit's is.
    """
    ones = np.ones((covariates.shape[0], 1))
    treated = np.clip(predict_mean(model, np.hstack([ones, covariates]), "classification"), 0, 1)
    control = np.clip(
        predict_mean(model, np.hstack([0.0 * ones, covariates]), "classification"), 0, 1
    )
    return np.asarray(treated - control, dtype=float)


def rule_value(treat: Mapping[float, np.ndarray], law: str) -> tuple[float, float]:
    """``Psi_d(P0)`` and ``Var_P0 D*(d, Qbar0, g0)`` of a rule given on the grid.

    ``treat[w2]`` is the rule at each grid point of ``W1`` for that ``W2``.  Both integrals
    use the trapezoid rule on :data:`GRID`, with the density ``1/2`` of ``W1`` and the
    probability ``1/2`` of each ``W2``.  With ``Q = Qbar0(d(W), W)`` and ``g = g0(d(W) | W)``,
    the variance is ``E[Q (1 - Q) / g] + E[Q^2] - Psi^2``.
    """
    value = 0.0
    second = 0.0
    for w2 in W2_LEVELS:
        rule = np.asarray(treat[w2], dtype=float)
        q = qbar0(rule, GRID, w2, law)
        treated = g0(GRID, w2)
        g = np.where(rule > 0.5, treated, 1.0 - treated)
        value += 0.5 * np.trapezoid(q, GRID) / 2.0
        second += 0.5 * np.trapezoid(q * (1.0 - q) / g + q**2, GRID) / 2.0
    return float(value), float(second - value**2)


def fixed_rule_grid(rule: Callable[[np.ndarray, float], np.ndarray]) -> dict[float, np.ndarray]:
    """A known rule ``rule(w1, w2)`` on the grid, as :func:`rule_value` reads it."""
    return {w2: np.asarray(rule(GRID, w2), dtype=bool) for w2 in W2_LEVELS}


def harness(
    result: Any,
    learner: Any,
    law: str,
    *,
    source: Literal["training", "validation"] = "training",
) -> Truth:
    """Refit each fold's rule, check it against the fit's rule, and integrate its value.

    ``source`` names the rows the fit learned each fold's rule on: the training complement for
    the declared estimator, and the validation rows for the ``fold_locality`` control.  The
    refit goes through the package's own ``fit_on_rows`` with the fit's own weights, so a
    deterministic learner reproduces the fitted model exactly.

    Raises
    ------
    HarnessMismatch
        When the refit rule differs from the fit's rule on any validation row.
    """
    data = result.data
    design = np.asarray(data.treatment_design(), dtype=float)
    covariates = np.asarray(data.covariates, dtype=float)
    outcome = np.asarray(data.outcome, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    fitted = rule_of(result)
    grid = {w2: np.column_stack([GRID, np.full_like(GRID, w2)]) for w2 in W2_LEVELS}
    values: list[float] = []
    variances: list[float] = []
    sizes: list[int] = []
    checked = 0
    for fold, (train, test) in enumerate(result.nuisance.folds):
        rows = np.asarray(train if source == "training" else test)
        model = fit_on_rows(learner, design, outcome, weights, rows, "classification", None)
        refit = _blip(model, covariates[test]) > 0.0
        disagree = int(np.sum(refit != fitted[test]))
        if disagree:
            raise HarnessMismatch(
                f"fold {fold}: the refit rule differs from the fit's rule on {disagree} of "
                f"{len(test)} validation rows"
            )
        checked += len(test)
        value, variance = rule_value(
            {w2: _blip(model, points) > 0.0 for w2, points in grid.items()}, law
        )
        values.append(value)
        variances.append(variance)
        sizes.append(len(test))
    folds = len(values)
    oracle = float(np.sqrt(sum(v / m for v, m in zip(variances, sizes, strict=True))) / folds)
    return Truth(
        truth=float(np.mean(values)),
        oracle_se=oracle,
        fold_values=tuple(values),
        rows_checked=checked,
    )


def measured(
    frame: pd.DataFrame,
    learner: Any,
    law: str,
    fold_seed: int,
    *,
    mutation: bool = False,
) -> tuple[Any, Truth, int]:
    """Fit one sample, run its harness, and count the solver warnings of both.

    Rule L4 stops the run on a raise, on a non-finite estimate or standard error, and on a
    harness mismatch.  Each of those raises here, so the replication is never replaced.
    """
    with solver_warning_count() as count:
        result = fit(frame, learner, fold_seed, mutation=mutation)
        truth = harness(result, learner, law, source="validation" if mutation else "training")
    estimate = result.estimates[ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or standard error (rule L4): {estimate}")
    return result, truth, count[0]


# ------------------------------------------------------------------ the primary rows


def primary_row(record: StudyRecord, scenario: str, replicate: int, n: int) -> dict[str, Any]:
    """One primary replication: the sample and fold seeds of ``record``, the fit, its truth."""
    sample_seed = replicate_seed(record, scenario, replicate)
    fold_seed = stream_seed(record, "fold_partition", "primary", scenario, replicate)
    frame = draw(scenario, n, sample_seed)
    result, truth, solver_warnings = measured(frame, outcome_learner(), scenario, fold_seed)
    estimate = result.estimates[ESTIMAND]
    std_error, low, high = reported_inference(estimate)
    return {
        "implementation": record.implementation,
        "scenario": scenario,
        "replicate": replicate,
        "n": n,
        "estimand": ESTIMAND,
        "truth": truth.truth,
        "estimate": float(estimate.psi),
        "inference_estimate": float(estimate.psi),
        "std_error": std_error,
        "ci_lower": low,
        "ci_upper": high,
        "inference_scale": "identity",
        "covered": int(low <= truth.truth <= high),
        "initial_estimate": untargeted_estimate(result),
        "oracle_se": truth.oracle_se,
        "rule_rows_checked": truth.rows_checked,
        "solver_warnings": solver_warnings,
    }


def _primary(payload: tuple[StudyRecord, str, int, int]) -> dict[str, Any]:
    record, scenario, replicate, n = payload
    return primary_row(record, scenario, replicate, n)


def draw_and_fit(record: StudyRecord, *, replicates: int, n: int, n_jobs: int) -> pd.DataFrame:
    """Every primary replication of ``record``, with the harness columns after the schema."""
    payloads = [
        ((record, scenario, replicate, n),)
        for scenario in record.scenarios
        for replicate in range(replicates)
    ]
    rows = pd.DataFrame(map_parallel(_primary, payloads, n_jobs=n_jobs))
    return rows.loc[:, [*REPLICATE_COLUMNS, *HARNESS_COLUMNS]]


def harness_artifact(rows: pd.DataFrame) -> pd.DataFrame:
    """The harness columns of the primary rows, keyed by scenario and replication."""
    return rows.loc[:, ["scenario", "replicate", "truth", *HARNESS_COLUMNS]].sort_values(
        ["scenario", "replicate"], ignore_index=True
    )


# -------------------------------------------------------------------------- the seeds


@dataclass(frozen=True)
class PropertyDraw:
    """One property cell's draw label, before rule L3 resolves it."""

    family: str
    label: str
    replicates: int
    law: str


def seed_labels(
    record: StudyRecord,
    primary_replicates: int,
    draws: tuple[PropertyDraw, ...],
) -> dict[tuple[str, str], tuple[Any, ...]]:
    """Apply rule L3 and return the suffix each property draw label resolves to.

    A sample seed is judged against every sample seed of the same law on the same record: the
    primary scenarios first, then the property cells in declaration order, then the replicates
    of a cell by index.  A colliding property cell moves its whole label to the same label
    followed by ``"retry"`` and the smallest counter ``j >= 1`` that leaves the set.  A
    collision inside a primary scenario stops the declaration.

    Returns
    -------
    dict
        ``(family, label) -> ()`` for a cell that keeps its label, and ``("retry", j)`` for one
        that moved.
    """
    taken: dict[str, set[int]] = {}
    for scenario in record.scenarios:
        seen = taken.setdefault(scenario, set())
        seeds = [replicate_seed(record, scenario, r) for r in range(primary_replicates)]
        if len(set(seeds)) != len(seeds) or set(seeds) & seen:
            raise RuntimeError(f"{record.slug}: primary seeds of {scenario} collide (rule L3)")
        seen.update(seeds)
    out: dict[tuple[str, str], tuple[Any, ...]] = {}
    for spec in draws:
        seen = taken.setdefault(spec.law, set())
        counter = 0
        while True:
            suffix: tuple[Any, ...] = () if counter == 0 else ("retry", counter)
            seeds = [
                stream_seed(record, "property_sample", spec.family, spec.label, r, *suffix)
                for r in range(spec.replicates)
            ]
            if len(set(seeds)) == len(seeds) and not set(seeds) & seen:
                break
            counter += 1
        seen.update(seeds)
        out[spec.family, spec.label] = suffix
    return out


def declared_sample_seeds(
    record: StudyRecord,
    primary_replicates: int,
    draws: tuple[PropertyDraw, ...],
    suffixes: Mapping[tuple[str, str], tuple[Any, ...]],
) -> set[int]:
    """Every sample seed a declared run of ``record`` draws, primary and property."""
    seeds = {
        replicate_seed(record, scenario, r)
        for scenario in record.scenarios
        for r in range(primary_replicates)
    }
    for spec in draws:
        suffix = suffixes[spec.family, spec.label]
        seeds.update(
            stream_seed(record, "property_sample", spec.family, spec.label, r, *suffix)
            for r in range(spec.replicates)
        )
    return seeds


def smoke_record(record: StudyRecord) -> StudyRecord:
    """The throwaway record of a smoke run (rule L9)."""
    from dataclasses import replace

    return replace(
        record,
        seed=stream_seed(record, "rm30", "smoke"),
        resampling_seed=stream_seed(record, "rm30", "smoke-resampling"),
    )
