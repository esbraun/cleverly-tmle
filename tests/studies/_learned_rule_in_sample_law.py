r"""The laws, fits and truth harness of the ``learned-rule-in-sample`` study (X11 PR F2, part (a)).

The study measures the in-sample learned-rule fit, ``cross_fit=False``: the plug-in rule
:math:`d_n` of the outcome regression fitted on every row, and the shipped in-sample regime
TMLE at :math:`d = d_n`. Its target is the value of that one rule,
:math:`\Psi_{d_n}(P_0) = E_0\{\bar Q_0(d_n(W), W)\}`, the full-sample data-adaptive target of
van der Laan and Luedtke (2015), Section 6, Theorem 5. It reads the RM30 laws, the treatment
learner and the truth harness of :mod:`tests.studies._learned_rule_law` without editing them.

``non_exceptional``
    The RM30 law with blip ``0.1 + W1``; the declared outcome model is correctly specified.
``misspecified_limit``
    The RM30 law with blip ``0.8 W1 + W1^2 - 0.3``; the declared outcome model is wrong, so
    the rule converges to the rule of its population projection (condition C3a with a
    misspecified limit), and the correct treatment model carries the remainder (C4a).

Three outcome learners. Each is a class, because the support key reads a learner's class
(:func:`cleverly._support.learner_family`) and reads every ``Pipeline`` as adaptive.

:class:`InteractionLogistic`
    The declared learner: RM30's logistic regression on every pairwise product of
    ``(A, W1, W2)``, a fixed dimension with nothing selected from the data, marked
    ``cleverly_learner_family = "parametric"``. Condition C5a holds for it.
:class:`DropW1Logistic`
    The ``targeting_necessity`` learner: RM30's ``drop_w1_learner`` (``A`` and ``W2`` only),
    marked parametric the same way. Its limit blip is nonzero at both values of ``W2``, so its
    rule settles and C3a holds; the fluctuation does all the adjusting.
:class:`ClippedInterpolatingTree`
    The control learner: ``DecisionTreeClassifier(min_samples_leaf=1, random_state=0)``, which
    interpolates a binary outcome, with ``predict_proba`` clipped to ``[0.01, 0.99]``. It
    carries no marker, so the key reads it adaptive and the fit raises
    :class:`~cleverly.exceptions.InSampleLearnerWarning`: C5a fails. ``q_bounds`` cannot clip it,
    because a binary outcome refuses ``q_bounds`` (plan review R1).

The truth of each replication is the value of the fit's own rule: the harness
:func:`tests.studies._learned_rule_law.harness` refits the outcome learner on every row (its
single-fold branch), requires the fit's rule on every row (rule L4), and integrates the rule's
value by the trapezoid rule on the RM30 grid.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from cleverly.estimators import TMLE
from cleverly.estimators._nuisance import fit_on_rows
from cleverly.exceptions import InSampleLearnerWarning
from cleverly.interventions import LearnedRule
from cleverly.learners.crossfit import make_folds
from tests.studies import _learned_rule_law as base

#: The reported parameter of every fit.
ESTIMAND = base.ESTIMAND

#: The scenarios, each a calibration cell and a primary scenario.
SCENARIOS = ("non_exceptional", "misspecified_limit")

#: The clipping of the interpolating control's probabilities (plan review R1).
CLIP = (0.01, 0.99)

#: The folds of the ``target_distinction`` diagnostic, as many as the RM30 cross-fitted fits.
DIAGNOSTIC_FOLDS = base.N_FOLDS

#: The columns a row carries beside the shared schema. ``oracle_se`` is descriptive.
#: ``rule_rows_checked`` counts the rows whose refit rule equals the fit's rule, every row of a
#: completed replication. ``solver_warnings`` counts the scikit-learn and ``cleverly``
#: convergence warnings of the fit and its truth refit. ``in_sample_warnings`` counts the fit's
#: :class:`~cleverly.exceptions.InSampleLearnerWarning`: 0 for the parametric learners, 1 for
#: the interpolating control. No rule reads any of them.
HARNESS_COLUMNS = ("oracle_se", "rule_rows_checked", "solver_warnings", "in_sample_warnings")


# ------------------------------------------------------------------------- the learners


class _Wrapped(ClassifierMixin, BaseEstimator):
    """A fixed scikit-learn pipeline behind a class of its own, which ``learner_family`` reads."""

    def _pipeline(self) -> Any:  # pragma: no cover - each subclass declares its own
        raise NotImplementedError

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Wrapped:
        options = {} if sample_weight is None else {"model__sample_weight": sample_weight}
        self.model_ = self._pipeline().fit(np.asarray(X, dtype=float), y, **options)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(np.asarray(X, dtype=float))


class InteractionLogistic(_Wrapped):
    """The declared outcome learner, RM30's ``outcome_learner()``, marked parametric.

    Every pairwise product of ``(A, W1, W2)``, then ``LogisticRegression(C=1e6, max_iter=5000,
    tol=1e-10)``: a fixed dimension and no selection, so condition C5a holds.
    """

    cleverly_learner_family: ClassVar[str] = "parametric"

    def _pipeline(self) -> Any:
        return base.outcome_learner()


def without_w1_main(design: Any) -> np.ndarray:
    """The design ``(A, W1, W2)`` without the ``W1`` main effect: ``(A, W2, A W1, A W2)``.

    The pairwise-product design of RM30's outcome learner less ``W1`` and ``W1 W2``. The
    treated arm keeps its ``W1`` slope through ``A W1``; the control arm loses it.
    """
    x = np.asarray(design, dtype=float)
    a, w1, w2 = x[:, 0], x[:, 1], x[:, 2]
    return np.column_stack([a, w2, a * w1, a * w2])


def without_w1_main_learner() -> Pipeline:
    """The ``targeting_necessity`` pipeline: :func:`without_w1_main`, then RM30's logistic."""
    return Pipeline(
        [
            ("features", FunctionTransformer(without_w1_main)),
            ("model", LogisticRegression(C=1e6, max_iter=5000, tol=1e-10)),
        ]
    )


class WithoutW1MainLogistic(_Wrapped):
    """The ``targeting_necessity`` outcome learner, marked parametric.

    :func:`without_w1_main_learner`: a fixed dimension, nothing selected, and wrong on every
    law, because the outcome logit carries ``0.5 W1`` in both arms. Its limit blip runs from
    about -0.3 to +0.35 on the probability scale on ``non_exceptional``, so its limit rule is
    far from exceptional, and its value is within 2e-5 of the optimal value
    (``test_learned_rule_in_sample_design``). The control arm's regression misses its ``W1``
    slope, so the untargeted plug-in is biased, and the fluctuation must remove that bias.
    """

    cleverly_learner_family: ClassVar[str] = "parametric"

    def _pipeline(self) -> Any:
        return without_w1_main_learner()


class DropW1Logistic(_Wrapped):
    """RM30's ``drop_w1_learner()``, marked parametric: the weak-limit diagnostic's learner.

    Columns ``A`` and ``W2`` of the design, their product, then the same logistic regression.
    It is the learner the plan names for the bias control. Its limit blip is 0.037 and 0.049
    on the probability scale at the two values of ``W2``, small beside its sampling error at
    n = 2,000, so the one rule flips between draws and the in-sample targeted arm carries an
    optimism bias (the design record, D2). It is reported, with no verdict.
    """

    cleverly_learner_family: ClassVar[str] = "parametric"

    def _pipeline(self) -> Any:
        return base.drop_w1_learner()


class ClippedInterpolatingTree(ClassifierMixin, BaseEstimator):
    """The control learner: a fully grown tree, its probabilities clipped to :data:`CLIP`.

    ``DecisionTreeClassifier(min_samples_leaf=1, random_state=0)`` interpolates a binary
    outcome, so its in-sample probabilities are 0 or 1 before the clip. The fixed
    ``random_state`` makes the truth refit exact. The class carries no
    ``cleverly_learner_family`` marker: it is adaptive, and C5a fails.

    Parameters
    ----------
    low : float
        The lower clip of the predicted probability.
    high : float
        The upper clip.
    """

    def __init__(self, low: float = CLIP[0], high: float = CLIP[1]) -> None:
        self.low = low
        self.high = high

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> ClippedInterpolatingTree:
        self.model_ = DecisionTreeClassifier(min_samples_leaf=1, random_state=0).fit(
            np.asarray(X, dtype=float), y, sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        treated = np.clip(
            self.model_.predict_proba(np.asarray(X, dtype=float))[:, 1], self.low, self.high
        )
        return np.column_stack([1.0 - treated, treated])


#: The learner of each kind of fit, and the learner families its support key must read.
LEARNERS: dict[str, type] = {
    "declared": InteractionLogistic,
    "without_w1_main": WithoutW1MainLogistic,
    "drop_w1": DropW1Logistic,
    "interpolating": ClippedInterpolatingTree,
}
FAMILIES: dict[str, dict[str, str]] = {
    "declared": {"outcome_learner_family": "parametric", "treatment_learner_family": "parametric"},
    "without_w1_main": {
        "outcome_learner_family": "parametric",
        "treatment_learner_family": "parametric",
    },
    "drop_w1": {"outcome_learner_family": "parametric", "treatment_learner_family": "parametric"},
    "interpolating": {
        "outcome_learner_family": "adaptive",
        "treatment_learner_family": "parametric",
    },
}


# ------------------------------------------------------------------------------ the fit


def estimator(learner: Any, seed: int) -> TMLE:
    """The declared in-sample estimator: one rule learned on every row, ``cross_fit=False``.

    ``seed`` is the fit's ``random_state``. The plug-in rule of a deterministic learner draws
    nothing from it; it fixes the record's split seed.
    """
    return TMLE(
        learned_rule=LearnedRule(),
        cross_fit=False,
        cv_evaluation=False,
        g_bounds="auto",
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        random_state=seed,
    )


def fit(frame: pd.DataFrame, learner: Any, seed: int) -> Any:
    """Fit the in-sample estimator to one sample and return the single result."""
    return (
        estimator(learner, seed)
        .fit(frame, outcome="Y", treatment="A", covariates=list(base.COVARIATES))
        .single()
    )


@dataclass
class Counts:
    """The warnings of one fit and its truth refit.

    Parameters
    ----------
    solver : int
        scikit-learn and ``cleverly`` convergence warnings (rule L4's count).
    in_sample : int
        :class:`~cleverly.exceptions.InSampleLearnerWarning`.
    """

    solver: int = 0
    in_sample: int = 0


@contextlib.contextmanager
def warning_counts() -> Iterator[Counts]:
    """Count the solver and in-sample-learner warnings raised inside the block.

    Every warning is recorded, so one that a filter elsewhere would show once is counted each
    time it fires. Every other warning is re-emitted unchanged.
    """
    counts = Counts()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield counts
    for message in caught:
        if issubclass(message.category, base.SOLVER_WARNINGS):
            counts.solver += 1
        elif issubclass(message.category, InSampleLearnerWarning):
            counts.in_sample += 1
        else:
            warnings.warn_explicit(
                message.message, message.category, message.filename, message.lineno
            )


def check_key(result: Any, kind: str) -> None:
    """Require the support key's learner families of ``kind`` and the in-sample scheme.

    A declared fit whose key reads another family would measure a configuration the study's
    support row does not name, so the replication stops (a design defect, not a fit failure).

    Raises
    ------
    RuntimeError
        When the key's scheme is not ``in_sample`` or a family differs from :data:`FAMILIES`.
    """
    key = result.config.support_key
    options = dict(key.options)
    found = {name: options.get(name) for name in FAMILIES[kind]}
    if key.splitting != ("in_sample",) or found != FAMILIES[kind]:
        raise RuntimeError(
            f"a {kind} fit keys {key.splitting} with {found}, not ('in_sample',) with "
            f"{FAMILIES[kind]}"
        )
    (split,) = result.extra["learned_rule"].splits
    if split.scheme != "in_sample" or split.fold_weights != (1.0,):
        raise RuntimeError(f"a {kind} fit records {split.scheme} with {split.fold_weights}")


def measured(frame: pd.DataFrame, law: str, kind: str, seed: int) -> tuple[Any, base.Truth, Counts]:
    """Fit one sample with the learner of ``kind``, run its harness, and count its warnings.

    Rule L4 stops the run on a non-finite estimate or standard error and on a harness
    mismatch, each a ``RuntimeError`` here; a recordable fitting error from ``cleverly`` or a
    numerical library is one lost replication (RM39b).
    """
    learner = LEARNERS[kind]
    with warning_counts() as counts:
        result = fit(frame, learner(), seed)
        truth = base.harness(result, learner(), law)
    check_key(result, kind)
    estimate = result.estimates[ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or standard error (rule L4): {estimate}")
    return result, truth, counts


def harness_columns(truth: base.Truth, counts: Counts) -> dict[str, Any]:
    """The harness columns of one fit."""
    return {
        "oracle_se": truth.oracle_se,
        "rule_rows_checked": truth.rows_checked,
        "solver_warnings": counts.solver,
        "in_sample_warnings": counts.in_sample,
    }


def untargeted_estimate(result: Any) -> float:
    """The fit without its fluctuation: the mean of the initial ``Qbar(d_n(W_i), W_i)``.

    With one pseudo-fold the fold average of :func:`tests.studies._learned_rule_law.
    untargeted_estimate` is the plain mean over every row; this reads it that way.
    """
    if len(list(result.nuisance.folds)) != 1:
        raise RuntimeError("an in-sample fit has one pseudo-fold")
    return base.untargeted_estimate(result)


# ---------------------------------------------------------------- the target distinction


@dataclass(frozen=True)
class FoldAverage:
    """The cross-fitted target of the same draw, beside the in-sample one.

    Parameters
    ----------
    truth : float
        ``(1/V) sum_v Psi_{d_v}(P0)``, with ``d_v`` the plug-in rule of the declared learner
        fitted on the training rows of fold ``v`` of :func:`diagnostic_folds`.
    grid_disagreement : float
        The mean, over the folds, of the share of the grid where ``d_v`` differs from the
        full-sample rule.
    """

    truth: float
    grid_disagreement: float


def diagnostic_folds(n: int, seed: int) -> Any:
    """The partition of the diagnostic: ``make_folds(n, 10, random_state=seed)``, unstratified."""
    return make_folds(n, DIAGNOSTIC_FOLDS, random_state=seed)


def grid_rule(model: Any) -> dict[float, np.ndarray]:
    """The plug-in rule of a fitted outcome model at every grid point."""
    return {
        w2: base._blip(model, np.column_stack([base.GRID, np.full_like(base.GRID, w2)])) > 0.0
        for w2 in base.W2_LEVELS
    }


def fold_average(result: Any, law: str, seed: int) -> FoldAverage:
    """The fold-average target of the cross-fitted rules of the same draw (no verdict).

    Fits the declared learner on each training complement of :func:`diagnostic_folds`, with
    the package's ``fit_on_rows``, integrates each rule's value on the RM30 grid, and averages
    with weight ``1/V``. It also refits the full-sample rule the same way, so the grid
    disagreement compares like with like.
    """
    data = result.data
    design = np.asarray(data.treatment_design(), dtype=float)
    outcome = np.asarray(data.outcome, dtype=float)
    weights = np.asarray(data.weights, dtype=float)

    def rule_on(rows: np.ndarray) -> dict[float, np.ndarray]:
        model = fit_on_rows(
            InteractionLogistic(), design, outcome, weights, rows, "classification", None
        )
        return grid_rule(model)

    full = rule_on(np.arange(data.n))
    values, shares = [], []
    for train, _ in diagnostic_folds(data.n, seed):
        rule = rule_on(np.asarray(train))
        values.append(base.rule_value(rule, law)[0])
        shares.append(
            float(np.mean(np.concatenate([rule[w2] != full[w2] for w2 in base.W2_LEVELS])))
        )
    return FoldAverage(truth=float(np.mean(values)), grid_disagreement=float(np.mean(shares)))


# ------------------------------------------------------------------ the limit rules


def projection(law: str, learner: Any = None, points: int = 4_001) -> Any:
    """The population projection of ``learner`` (RM30's outcome learner by default) on ``law``.

    Each grid point of ``(W1, W2)`` carries its trapezoid mass, each arm its propensity, and
    each outcome its probability under the law's ``Qbar0``, so the log-likelihood the learner
    maximises is the population one. The learner's ridge is removed. This is the
    ``learned-rule-compositions`` projection without weights. ``learner`` is a pipeline whose
    last step is named ``model``.
    """
    grid = np.linspace(-1.0, 1.0, points)
    mass = np.full(points, 2.0 / (points - 1))
    mass[[0, -1]] /= 2.0
    design, sample_weight, outcome = [], [], []
    for w2 in base.W2_LEVELS:
        treated = base.g0(grid, w2)
        for a in (0.0, 1.0):
            q = base.qbar0(a, grid, w2, law)
            for y, p_y in ((1.0, q), (0.0, 1.0 - q)):
                design.append(np.column_stack([np.full(points, a), grid, np.full(points, w2)]))
                sample_weight.append(mass * (treated if a else 1.0 - treated) * p_y)
                outcome.append(np.full(points, y))
    model = (base.outcome_learner() if learner is None else learner).set_params(
        model__C=1e12, model__tol=1e-13, model__max_iter=20_000
    )
    weights = np.concatenate(sample_weight)
    return model.fit(
        np.vstack(design), np.concatenate(outcome), model__sample_weight=weights / weights.mean()
    )


def population_rule(law: str, learner: Any = None) -> dict[float, np.ndarray]:
    """The limit rule: the plug-in rule of :func:`projection` on the grid."""
    return grid_rule(projection(law, learner))


def efficiency_bound(law: str) -> float:
    """The SD of the row curve ``D*(d_1, Qbar0, g0)`` at the declared learner's limit rule."""
    _, variance = base.rule_value(population_rule(law), law)
    return float(np.sqrt(variance))


def limit_blips(law: str, learner: Any = None) -> Mapping[float, np.ndarray]:
    """The blip of :func:`projection` at every grid point, by ``W2``."""
    model = projection(law, learner)
    return {
        w2: base._blip(model, np.column_stack([base.GRID, np.full_like(base.GRID, w2)]))
        for w2 in base.W2_LEVELS
    }
