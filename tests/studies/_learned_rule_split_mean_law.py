"""The fit, the truth harness and the controls of the ``learned-rule-split-mean`` study (X11 PR B).

The study measures the learned-rule value with ``repeats=R`` above 1, which reports the mean
over the ``R`` splits (X11 part (i)). It reuses the RM30 laws and learners of
:mod:`tests.studies._learned_rule_law` unchanged and adds what the mean over splits needs.

Each split ``r`` draws its own ten folds and learns its own fold rules, so it defines its own
target ``psi_tilde^(r)``, the fold average of the values of its rules. The reported target is
the mean of the ``R`` split targets. The truth of each replication is therefore rule L4 of the
RM30 studies applied to every split: the harness refits each fold's outcome learner on that
split's training rows, requires the split's rule on every validation row, integrates each
rule's value on the RM30 grid, averages over the folds of the split, and averages the ``R``
split values.

Two quantities sit beside the declared fit. :func:`independent_splits_se` is the deliberately
invalid standard error that treats the ``R`` splits as independent samples, the mean single-split
SE over ``sqrt(R)``. The splits reuse every row, so their errors are close to perfectly
correlated, and that SE is too small by about ``sqrt(R)``. :func:`median_estimate` is the median
of the ``R`` split estimates, which estimates no defined parameter. The study publishes it as a
descriptive column and no verdict reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule
from tests.studies import _learned_rule_law as base

#: The splits of every fit, the ``repeats`` argument.
SPLITS = 5

#: The outer folds of each split, as in the RM30 studies.
N_FOLDS = base.N_FOLDS

#: The reported parameter of every fit.
ESTIMAND = base.ESTIMAND

#: The RM30 laws the study draws, which are also its scenario names.
LAWS = ("non_exceptional", "misspecified_limit")

#: The relative tolerance of the guard that the reported point is the mean of the split points.
#: W-i1 of ``tests/unit/test_learned_rule_split_mean.py`` checks equality bit for bit; the
#: guard only stops a run that would measure a different estimator.
MEAN_TOLERANCE = 1e-12

#: The relative tolerance of the guard that the reported variance is at most the mean of the
#: split variances, which Jensen's inequality guarantees for ``n^-2 sum_i phibar_i^2``.
VARIANCE_TOLERANCE = 1e-12

#: The descriptive columns of what the mean over splits changes against one split, carried by
#: every primary row and every calibration-positive property row. ``first_split_estimate`` and
#: ``first_split_truth`` are the first split's own estimate and its own target (the per-split
#: harness value), and ``split_estimate_sd`` is the sample SD (ddof 1) of the ``R`` split
#: estimates within the draw. No rule reads them.
FIRST_SPLIT_COLUMNS = ("first_split_estimate", "first_split_truth", "split_estimate_sd")

#: The columns a property row carries beside the harness columns. ``median_estimate`` is the
#: median of the split estimates and ``split_se_mean`` the mean of the split standard errors.
#: They and :data:`FIRST_SPLIT_COLUMNS` are descriptive, and no rule reads them.
SPLIT_COLUMNS = ("median_estimate", "split_se_mean", *FIRST_SPLIT_COLUMNS)


def estimator(learner: Any, fold_seed: int) -> TMLE:
    """The declared estimator, ``R`` splits of ten folds, with the fold seed of rule L2."""
    return TMLE(
        learned_rule=LearnedRule(),
        n_folds=N_FOLDS,
        cv_evaluation=True,
        targeting_scheme="pooled",
        g_bounds="auto",
        repeats=SPLITS,
        outcome_learner=learner,
        treatment_learner=base.treatment_learner(),
        random_state=fold_seed,
    )


def fit(frame: pd.DataFrame, learner: Any, fold_seed: int) -> Any:
    """Fit the declared estimator to one sample and return the single result."""
    return (
        estimator(learner, fold_seed)
        .fit(frame, outcome="Y", treatment="A", covariates=list(base.COVARIATES))
        .single()
    )


def split_view(result: Any, split: int) -> SimpleNamespace:
    """Split ``split`` of ``result`` in the shape :func:`base.harness` reads.

    The harness reads the data, the folds and the realized rule. A split's folds and rule are
    its own nuisance's, ``result.repeats[split].nuisance``.
    """
    return SimpleNamespace(data=result.data, nuisance=result.repeats[split].nuisance)


@dataclass(frozen=True)
class Truth:
    """What the harness computes for one fit of ``R`` splits.

    Parameters
    ----------
    truth : float
        ``(1/R) sum_r psi_tilde^(r)``, the mean of the split targets.
    oracle_se : float
        The mean over the splits of each split's oracle SE (``base.Truth.oracle_se``).
        Descriptive; no rule reads it.
    split_values : tuple of float
        ``psi_tilde^(r)`` for each split.
    rows_checked : int
        The validation rows whose refit rule equals the split's rule, summed over the splits,
        so ``R * n`` for a replication that completed.
    """

    truth: float
    oracle_se: float
    split_values: tuple[float, ...]
    rows_checked: int


def harness(result: Any, learner: Any, law: str) -> Truth:
    """Apply rule L4 to every split of ``result`` and average the split targets.

    Raises
    ------
    tests.studies._learned_rule_law.HarnessMismatch
        When a refit rule differs from its split's rule on any validation row.
    """
    splits = [base.harness(split_view(result, r), learner, law) for r in range(len(result.repeats))]
    if len(splits) != SPLITS:
        raise RuntimeError(f"the fit holds {len(splits)} splits, not the declared {SPLITS}")
    values = tuple(split.truth for split in splits)
    return Truth(
        truth=float(np.mean(values)),
        oracle_se=float(np.mean([split.oracle_se for split in splits])),
        split_values=values,
        rows_checked=sum(split.rows_checked for split in splits),
    )


def split_estimates(result: Any) -> np.ndarray:
    """Each split's own point estimate, in the order of ``result.repeats``."""
    return np.array([float(repeat.psi[ESTIMAND]) for repeat in result.repeats])


def split_std_errors(result: Any) -> np.ndarray:
    """Each split's own standard error, the SE of a ``repeats=1`` fit at that split's seed."""
    return np.array([float(repeat.inference[ESTIMAND].se) for repeat in result.repeats])


def independent_splits_se(result: Any) -> float:
    """The deliberately invalid SE: the mean single-split SE over ``sqrt(R)``.

    It is the SE of a mean of ``R`` independent estimates, each with the split's SE. The
    splits share every row, so it is too small, and the control that reports it beside the
    fit's estimate must fall below the SE-ratio band.
    """
    errors = split_std_errors(result)
    return float(np.mean(errors) / np.sqrt(errors.size))


def untargeted_estimate(result: Any) -> float:
    """The mean over the splits of each split's fold average of its initial plug-ins."""
    return float(
        np.mean(
            [base.untargeted_estimate(split_view(result, r)) for r in range(len(result.repeats))]
        )
    )


def median_estimate(result: Any) -> float:
    """The median of the split estimates. Descriptive: it estimates no defined parameter."""
    return float(np.median(split_estimates(result)))


def measured(frame: pd.DataFrame, law: str, learner: Any, fold_seed: int) -> tuple[Any, Truth, int]:
    """Fit one sample, run its harness, and count the solver warnings of both (rule L4).

    Beside rule L4's stops, a reported point that is not the mean of the split points stops
    the replication, because the study would then measure another estimator.
    """
    with base.solver_warning_count() as count:
        result = fit(frame, learner, fold_seed)
        truth = harness(result, learner, law)
    estimate = result.estimates[ESTIMAND]
    if not (np.isfinite(estimate.psi) and np.isfinite(estimate.std_error)):
        raise RuntimeError(f"a non-finite estimate or standard error (rule L4): {estimate}")
    mean = float(np.mean(split_estimates(result)))
    if not np.isclose(float(estimate.psi), mean, rtol=MEAN_TOLERANCE, atol=0.0):
        raise RuntimeError(
            f"the reported point {float(estimate.psi)!r} is not the mean {mean!r} of the "
            "split points"
        )
    # Jensen: the variance of the mean curve, n^-2 sum_i phibar_i^2, is at most the mean of the
    # split variances n^-2 sum_i phi_ri^2. A larger one means the study measures another variance.
    variance = float(estimate.std_error) ** 2
    split_variance = float(np.mean(split_std_errors(result) ** 2))
    if variance > split_variance * (1.0 + VARIANCE_TOLERANCE):
        raise RuntimeError(
            f"the reported variance {variance!r} exceeds the mean {split_variance!r} of the "
            "split variances, which the mean of the split curves cannot do"
        )
    return result, truth, count[0]


def first_split_columns(result: Any, truth: Truth) -> dict[str, float]:
    """The descriptive :data:`FIRST_SPLIT_COLUMNS` of one fit and its harness truth."""
    estimates = split_estimates(result)
    return {
        "first_split_estimate": float(estimates[0]),
        "first_split_truth": float(truth.split_values[0]),
        "split_estimate_sd": float(np.std(estimates, ddof=1)),
    }


def split_columns(result: Any, truth: Truth) -> dict[str, float]:
    """The descriptive :data:`SPLIT_COLUMNS` of one fit and its harness truth."""
    return {
        "median_estimate": median_estimate(result),
        "split_se_mean": float(np.mean(split_std_errors(result))),
        **first_split_columns(result, truth),
    }


def population_rule(law: str, points: int = 4_001) -> dict[float, np.ndarray]:
    """The rule of the declared learner's population projection on ``law``, on the RM30 grid.

    Each grid point of ``(W1, W2)`` carries its trapezoid mass, each arm its propensity and each
    outcome its probability, so the weighted log-likelihood the learner maximises is the
    population one. The learner's ridge is removed. Under C3 every split's rules converge to
    this rule, so the mean curve converges to the curve of one split at it.
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
    model = base.outcome_learner().set_params(
        model__C=1e12, model__tol=1e-13, model__max_iter=20_000
    )
    weights = np.concatenate(sample_weight)
    model.fit(
        np.vstack(design), np.concatenate(outcome), model__sample_weight=weights / weights.mean()
    )
    return {
        w2: base._blip(model, np.column_stack([base.GRID, np.full_like(base.GRID, w2)])) > 0.0
        for w2 in base.W2_LEVELS
    }
