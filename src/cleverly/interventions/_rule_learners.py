r"""The blip and weighted-classification rule learners of a learned-rule fit.

Luedtke and van der Laan (2016, *International Journal of Biostatistics*), Sections 4.1 to
4.3, learn a treatment rule by a loss on a pseudo-outcome rather than by the plug-in of an
outcome regression.  This module holds the package's construction of the two losses for a
binary treatment at one time point.  Inside each outer training fold :math:`T_v` it

1. fits the outcome regression and the treatment mechanism by an inner cross-fit of
   :math:`T_v` into :data:`INNER_FOLDS` folds, with the fit's own learner templates.  Under
   ``id=`` the inner folds move whole clusters, so a :math:`T_v` with fewer than
   :data:`INNER_FOLDS` clusters gets one inner fold per cluster, with the ``UserWarning``
   of :func:`~cleverly.learners.crossfit.random_partition`;
2. computes the doubly robust blip pseudo-outcome of every row of :math:`T_v`,

   .. math::

       \tilde D_i = \frac{2A_i - 1}{g(A_i \mid W_i)}\{Y_i - \bar Q(A_i, W_i)\}
                    + \bar Q(1, W_i) - \bar Q(0, W_i),

   from the inner out-of-fold :math:`\bar Q` and :math:`g`, with :math:`g` truncated into
   the fit's resolved ``g_bounds`` (resolved at the full sample's effective size, the value
   ``result.config`` records) and :math:`Y` and :math:`\bar Q` on the fit's scaled
   :math:`[0, 1]` outcome;
3. fits the declared rule learner on :math:`T_v`: a regression of :math:`\tilde D` on
   :math:`W` for ``loss="blip"``, with the rule :math:`1\{\hat b_v(w) > 0\}`, or a
   classifier of :math:`1\{\tilde D > 0\}` on :math:`W` with sample weights
   :math:`|\tilde D|` for ``loss="weighted_classification"``, with the rule
   :math:`1\{\hat P_v(\tilde D > 0 \mid w) > 1/2\}`.  A ``SuperLearner`` rule learner
   scores its candidates on its own ``n_folds``, not on the fit's ``learner_folds``.

A tie assigns the lower arm code, as the plug-in rule does.  The pseudo-outcome is the
package's choice of the doubly robust blip pseudo-outcome: the source's Section 4.1 states
the blip losses with a known or estimated mechanism, and its Appendix A the doubly robust
versions.  The inner cross-fit is a rule-quality choice, not a validity condition: the
learned-rule value holds for any rule that is a function of :math:`T_v` with a limit
(van der Laan and Luedtke 2015, Theorem 6).  What it rules out is the outer out-of-fold
predictions, which at a row of :math:`T_v` come from a model trained on rows that include
fold :math:`v`'s validation rows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.causal_data import CausalData
    from ..estimators._nuisance import NuisanceEstimates
    from ..learners._fitting import Task
    from ..learners.crossfit import Folds
    from ..utils.bounds import OutcomeScaler

__all__ = [
    "INNER_FOLDS",
    "FoldRule",
    "InnerFit",
    "fit_fold_rule",
    "inner_seed",
    "inner_support_refusal",
    "pseudo_outcome",
    "rule_learner_treat",
]

#: The inner folds of the cross-fit inside each outer training fold.
INNER_FOLDS = 5

#: The inner nuisance fit: ``(data, folds, seed) -> NuisanceEstimates``, with the outcome
#: regression and the treatment mechanism of the fit's own learner templates.
InnerFit = Callable[["CausalData", "Folds", int], "NuisanceEstimates"]


def inner_seed(seed: int, fold: int) -> int:
    """Return the seed of the inner cross-fit of outer fold ``fold`` in the split ``seed``.

    The inner partition, and any learner that draws its own folds, read it.  The record
    keeps it, so a refit of one fold's rule reproduces the fit.

    Parameters
    ----------
    seed : int
        The split's seed.
    fold : int
        The outer fold, counted from 0.

    Returns
    -------
    int
        A seed in ``[0, 2**32 - 1]``.
    """
    state = np.random.SeedSequence([int(seed), int(fold)]).generate_state(1, dtype=np.uint32)
    return int(state[0])


def pseudo_outcome(
    nuisance: NuisanceEstimates,
    outcome: np.ndarray,
    treatment: np.ndarray,
    g_bounds: tuple[float, float],
) -> np.ndarray:
    r"""Return :math:`\tilde D_i` at every row of an inner cross-fit.

    Parameters
    ----------
    nuisance : NuisanceEstimates
        The inner cross-fitted nuisances, one prediction per row from a model that did
        not train on that row.
    outcome : numpy.ndarray
        The outcome on the fit's scaled :math:`[0, 1]` scale.
    treatment : numpy.ndarray
        The treatment codes, ``0.0`` or ``1.0``.
    g_bounds : tuple of float
        The fit's truncation bounds of the treatment mechanism.

    Returns
    -------
    numpy.ndarray
        The pseudo-outcome of each row.
    """
    control, treated = nuisance.outcome.levels
    q1 = np.asarray(nuisance.outcome.arms[treated], dtype=float)
    q0 = np.asarray(nuisance.outcome.arms[control], dtype=float)
    g = nuisance.bounded_propensity(g_bounds)
    g1 = np.asarray(g[:, nuisance.propensity.column_for(treated)], dtype=float)
    g0 = np.asarray(g[:, nuisance.propensity.column_for(control)], dtype=float)
    is_treated = np.asarray(treatment, dtype=float) == treated
    observed = np.where(is_treated, q1, q0)
    density = np.where(is_treated, g1, g0)
    sign = np.where(is_treated, 1.0, -1.0)
    return np.asarray(sign / density * (outcome - observed) + q1 - q0, dtype=float)


@dataclass(frozen=True)
class FoldRule:
    """The rule learner of one outer fold, fitted on its training rows.

    Parameters
    ----------
    loss : str
        ``"blip"`` or ``"weighted_classification"``.
    model : object or None
        The fitted learner, or ``None`` for a classifier whose training labels hold one
        value with positive weight.
    constant : bool
        The rule of a fold with no fitted classifier: ``True`` treats every row.
    seed : int
        The inner seed of the fold (:func:`inner_seed`).
    pseudo_outcome : numpy.ndarray
        :math:`\\tilde D` at each training row, in the order of the training rows.
    """

    loss: str
    model: Any
    constant: bool
    seed: int
    pseudo_outcome: np.ndarray

    def score(self, covariates: np.ndarray) -> np.ndarray:
        """Return the fitted blip, or the fitted probability of a positive pseudo-outcome.

        Parameters
        ----------
        covariates : numpy.ndarray
            The covariate rows, ``(m, p)``.

        Returns
        -------
        numpy.ndarray
            ``m`` values: :math:`\\hat b_v(w)` for ``loss="blip"``, and
            :math:`\\hat P_v(\\tilde D > 0 \\mid w)` for ``loss="weighted_classification"``.
        """
        from ..learners._fitting import predict_mean

        matrix = np.asarray(covariates, dtype=float)
        if self.model is None:
            return np.full(matrix.shape[0], 1.0 if self.constant else 0.0)
        task: Task = "regression" if self.loss == "blip" else "classification"
        return np.asarray(predict_mean(self.model, matrix, task), dtype=float)

    @property
    def threshold(self) -> float:
        """Return the threshold of the score: ``0`` for the blip, ``1/2`` for the classifier."""
        return 0.0 if self.loss == "blip" else 0.5

    def margin(self, covariates: np.ndarray) -> np.ndarray:
        """Return the score minus its threshold, :math:`\\hat b_v(w)` or :math:`\\hat P_v - 1/2`.

        Parameters
        ----------
        covariates : numpy.ndarray
            The covariate rows, ``(m, p)``.

        Returns
        -------
        numpy.ndarray
            ``m`` values.  The rule treats where the value is positive.
        """
        return self.score(covariates) - self.threshold

    def treat(self, covariates: np.ndarray) -> np.ndarray:
        """Return ``True`` where the rule assigns the higher arm code.

        Parameters
        ----------
        covariates : numpy.ndarray
            The covariate rows, ``(m, p)``.

        Returns
        -------
        numpy.ndarray
            ``m`` booleans.  A tie, a score exactly at the threshold, assigns the lower arm
            code.
        """
        return self.score(covariates) > self.threshold


def inner_support_refusal(
    inner: CausalData, folds: Folds, *, loss: str, fold: int | None
) -> str | None:
    """Return why an inner training complement cannot fit the inner nuisances, or ``None``.

    Every inner training complement needs a row in each arm, and for a binary outcome a
    row with each outcome class, as the outer split does.  The inner split reads no
    treatment or outcome, so the check runs after the draw and before any learner.

    Parameters
    ----------
    inner : CausalData
        The training rows of one outer fold.
    folds : Folds
        Their inner split.
    loss : str
        The loss of the rule learner, which the message names.
    fold : int or None
        The outer fold, counted from 0, which the message names when it is known.

    Returns
    -------
    str or None
        The message of the refusal, or ``None`` when every complement is supported.
    """
    from ..learners.crossfit import missing_training_support

    treatment = np.asarray(inner.treatment, dtype=float)
    support: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("treatment arm", treatment, np.unique(treatment))
    ]
    if inner.family == "binomial":
        outcome = np.asarray(inner.outcome, dtype=float)
        support.append(("outcome class", outcome, np.array([0.0, 1.0])))
    gap = missing_training_support(folds, support)
    if gap is None:
        return None
    inner_fold, name, missing = gap
    value = missing[0]
    if name == "treatment arm":
        lacks, model = f"treatment arm {inner.arm_label(value)}", "treatment mechanism"
    else:
        lacks, model = f"outcome {value:g}", "outcome regression"
    where = "an outer training fold" if fold is None else f"outer fold {fold}"
    return (
        f"LearnedRule(loss={loss!r}) learns each fold's rule from a pseudo-outcome whose "
        "outcome and treatment fits are cross-fitted inside the fold's training rows, and "
        f"inner fold {inner_fold} of {where} leaves a training complement with no row in "
        f"{lacks}, so the inner {model} cannot be fitted there. The inner split is drawn "
        "from the seed alone and reads no treatment or outcome. Fit the learned-rule value "
        "on a larger sample, with fewer outer folds, or with loss='plug_in'."
    )


def fit_fold_rule(
    data: CausalData,
    train: np.ndarray,
    *,
    loss: str,
    learner: Any,
    fit_inner: InnerFit,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    seed: int,
    fold: int | None = None,
) -> FoldRule:
    """Fit the rule learner of one outer fold on its training rows.

    Parameters
    ----------
    data : CausalData
        The prepared data of the fit.
    train : numpy.ndarray
        The outer training rows.
    loss : str
        ``"blip"`` or ``"weighted_classification"``.
    learner : object
        The declared rule learner, a template that is cloned before the fit.
    fit_inner : callable
        The inner nuisance fit (:data:`InnerFit`).
    scaler : OutcomeScaler
        The fit's outcome scaler.
    g_bounds : tuple of float
        The fit's truncation bounds of the treatment mechanism.
    seed : int
        The fold's inner seed (:func:`inner_seed`).
    fold : int or None
        The outer fold, counted from 0, which a refusal names.

    Returns
    -------
    FoldRule
        The fitted rule and the pseudo-outcomes it learned from.

    Raises
    ------
    DataError
        If an inner training complement lacks an arm, or for a binary outcome an outcome
        class (:func:`inner_support_refusal`).
    """
    from ..exceptions import DataError
    from ..learners._fitting import as_target, fit_learner
    from ..learners.crossfit import random_partition
    from ..learners.super_learner import resolve_learner

    rows = np.asarray(train)
    inner = data.subset(rows)
    folds = random_partition(inner.n, INNER_FOLDS, cluster=inner.cluster, seed=seed)
    refusal = inner_support_refusal(inner, folds, loss=loss, fold=fold)
    if refusal is not None:
        raise DataError(refusal)
    nuisance = fit_inner(inner, folds, seed)
    target = pseudo_outcome(
        nuisance,
        scaler.scale(np.asarray(inner.outcome, dtype=float)),
        np.asarray(inner.treatment, dtype=float),
        g_bounds,
    )
    covariates = np.asarray(inner.covariates, dtype=float)
    weights = np.asarray(inner.weights, dtype=float) if inner.is_weighted else None
    groups = inner.cluster
    task: Task = "regression" if loss == "blip" else "classification"
    model = resolve_learner(learner, task=task, random_state=seed)
    if loss == "blip":
        fitted = fit_learner(
            model,
            covariates,
            as_target(target, task),
            weights,
            groups=groups,
            warn_unweighted=False,
        )
        return FoldRule(loss, fitted, False, seed, target)
    labels = (target > 0.0).astype(float)
    sample_weight = np.abs(target) * (1.0 if weights is None else weights)
    positive = sample_weight > 0.0
    held = np.unique(labels[positive])
    if held.size < 2:
        # Every row with weight carries one label, so the weighted risk is minimised by
        # that label everywhere, and a classifier cannot be fitted to one class.
        constant = bool(held.size == 1 and held[0] > 0.5)
        return FoldRule(loss, None, constant, seed, target)
    fitted = fit_learner(
        model,
        covariates,
        as_target(labels, task),
        sample_weight,
        groups=groups,
        warn_unweighted=False,
    )
    return FoldRule(loss, fitted, False, seed, target)


def rule_learner_treat(
    data: CausalData,
    folds: Folds,
    *,
    loss: str,
    learner: Any,
    fit_inner: InnerFit,
    scaler: OutcomeScaler,
    g_bounds: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the rule of every row, from the rule learner of the row's outer fold.

    Parameters
    ----------
    data : CausalData
        The prepared data of the fit.
    folds : Folds
        The outer split.  A single fold (``cross_fit=False``) learns one rule on every row.
    loss : str
        ``"blip"`` or ``"weighted_classification"``.
    learner : object
        The declared rule learner.
    fit_inner : callable
        The inner nuisance fit (:data:`InnerFit`).
    scaler : OutcomeScaler
        The fit's outcome scaler.
    g_bounds : tuple of float
        The fit's truncation bounds of the treatment mechanism.
    seed : int
        The split's seed.  Fold ``v`` uses ``inner_seed(seed, v)``.

    Returns
    -------
    treat : numpy.ndarray
        One boolean per row: ``True`` where the rule of the row's fold assigns the higher
        arm code.
    margin : numpy.ndarray
        The score of the row's fold rule minus its threshold at the row
        (:meth:`FoldRule.margin`).
    fold_rules : numpy.ndarray
        ``(n, V)`` booleans: column ``v`` is the rule of fold ``v`` at every row, evaluated
        while the fold's fitted learner is in memory.  Its diagonal, the entry of each
        row's own fold, is ``treat``.
    """
    covariates = np.asarray(data.covariates, dtype=float)
    treat = np.zeros(data.n, dtype=bool)
    margin = np.zeros(data.n, dtype=float)
    fold_rules = np.zeros((data.n, folds.n_folds), dtype=bool)
    for fold, (train, test) in enumerate(folds):
        # One fold is the in-sample fit (cross_fit=False): its one rule is learned on every
        # row, and its inner cross-fit runs over all of them.
        rows_of_rule = np.arange(data.n) if folds.is_single else np.asarray(train)
        rule = fit_fold_rule(
            data,
            rows_of_rule,
            loss=loss,
            learner=learner,
            fit_inner=fit_inner,
            scaler=scaler,
            g_bounds=g_bounds,
            seed=inner_seed(seed, fold),
            fold=fold,
        )
        rows = np.asarray(test)
        treat[rows] = rule.treat(covariates[rows])
        margin[rows] = rule.margin(covariates[rows])
        fold_rules[:, fold] = rule.treat(covariates)
    return treat, margin, fold_rules
