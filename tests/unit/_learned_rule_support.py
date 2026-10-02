"""A compact law, learners and fits for the learned-rule witnesses.

The law is the ``non_exceptional`` law that the RM30 declaration names:
``W1 ~ U(-1, 1)``, ``W2 ~ Bernoulli(0.5)``, ``logit g0 = 0.3 W1 - 0.2 W2`` and
``logit Qbar0 = 0.2 + 0.5 W1 - 0.3 W2 + A (0.1 + W1)``.  Its blip changes sign at
``W1 = -0.1``, so the learned rule treats some rows and not others.

The learners are explicit and deterministic.  The outcome learner fits the interaction of
the treatment with each covariate, so it can learn the rule.  The misspecified one drops
``W1``, so its fit is wrong and the fluctuation moves: the targeting witnesses need a
nonzero ``epsilon``.
"""

from __future__ import annotations

import functools
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly.estimators import TMLE
from cleverly.interventions import LearnedRule

#: The covariates of the law, in the order the fits declare them.
COVARIATES = ("W1", "W2")

#: The reported parameter of a fit of the default rule.
NAME = "ey_learned_rule[learned rule]"

#: The fold-evaluated configuration the learned-rule value needs.
FOLD_EVALUATED: dict[str, Any] = {"cross_fit": True, "cv_evaluation": True}


def law_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Draw ``n`` rows of the ``non_exceptional`` law.

    Parameters
    ----------
    n : int
        Rows to draw.
    seed : int
        Seed of the draw.

    Returns
    -------
    pandas.DataFrame
        Columns ``W1``, ``W2``, ``A`` and ``Y``.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a = rng.binomial(1, expit(0.3 * w1 - 0.2 * w2)).astype(float)
    y = rng.binomial(1, expit(0.2 + 0.5 * w1 - 0.3 * w2 + a * (0.1 + w1))).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


def outcome_learner() -> Pipeline:
    """The outcome learner on the design ``(A, W1, W2)``, with every pairwise interaction."""
    return Pipeline(
        [
            ("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("fit", LogisticRegression(max_iter=5000)),
        ]
    )


def misspecified_learner() -> Pipeline:
    """An outcome learner that omits ``W1``, so its fit is wrong and ``epsilon`` is nonzero."""
    return Pipeline(
        [
            ("keep", ColumnTransformer([("keep", "passthrough", [0, 2])])),
            ("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("fit", LogisticRegression(max_iter=5000)),
        ]
    )


def treatment_learner() -> LogisticRegression:
    """The treatment learner, a correct logistic model on ``(W1, W2)``."""
    return LogisticRegression(max_iter=1000)


def learned_tmle(**overrides: Any) -> TMLE:
    """A fold-evaluated learned-rule estimator with explicit learners and five folds."""
    settings: dict[str, Any] = {
        "learned_rule": LearnedRule(),
        "outcome_learner": outcome_learner(),
        "treatment_learner": treatment_learner(),
        "n_folds": 5,
        "simultaneous": False,
        "random_state": 0,
        **FOLD_EVALUATED,
    }
    settings.update(overrides)
    return TMLE(**settings)


def fit(frame: Any = None, **overrides: Any) -> Any:
    """Fit :func:`learned_tmle` on ``frame``, which defaults to :func:`law_frame`."""
    frame = law_frame() if frame is None else frame
    return (
        learned_tmle(**overrides)
        .fit(frame, outcome="Y", treatment="A", covariates=COVARIATES)
        .single()
    )


@functools.cache
def planted() -> Any:
    """A misspecified outcome regression at unequal folds: n = 203, V = 5.

    The regression omits ``W1``, so the pooled ``epsilon`` is nonzero, and 203 rows make
    the five folds unequal, so the weights ``n / (V n_v)`` differ from one.  The fit is
    cached, and every caller only reads it.
    """
    return fit(law_frame(203, 1), outcome_learner=misspecified_learner())


def blip(model: Any, covariates: np.ndarray) -> np.ndarray:
    """The blip ``P(Y = 1 | A = 1, W) - P(Y = 1 | A = 0, W)`` of a fitted outcome model.

    Parameters
    ----------
    model : classifier
        A fitted outcome model on the design ``(A, W1, W2)``.
    covariates : numpy.ndarray
        The rows of ``(W1, W2)`` to evaluate the blip at.

    Returns
    -------
    numpy.ndarray
        The blip at each row.
    """
    rows = len(covariates)
    treated = np.column_stack([np.ones(rows), covariates])
    control = np.column_stack([np.zeros(rows), covariates])
    return model.predict_proba(treated)[:, 1] - model.predict_proba(control)[:, 1]


class RowSpy(ClassifierMixin, BaseEstimator):
    """The outcome learner of :func:`outcome_learner`, recording the rows of every fit.

    A fit is recorded by the ``W1`` values of its training rows, which are distinct on a
    continuous draw.  The record is a class attribute, because the fit clones the learner
    before each fold.
    """

    fits: ClassVar[list[np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RowSpy:
        design = np.asarray(X, dtype=float)
        RowSpy.fits.append(design[:, 1].copy())
        self.model_ = outcome_learner().fit(design, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> Any:
        return self.model_.predict_proba(np.asarray(X, dtype=float))


def validation_pieces(result: Any) -> dict[str, Any]:
    """The per-row pieces of a learned-rule fit that the longhand witnesses read.

    Parameters
    ----------
    result : TMLEResult
        A fold-evaluated learned-rule fit.

    Returns
    -------
    dict
        ``rule`` (the fitted rule of each row), ``h`` (the clever covariate at the
        observed arm), ``y`` (the scaled outcome), ``q_star_observed`` and ``q_star_rule``
        (the targeted regression at the observed arm and at the rule), ``tests`` (the
        validation rows of each fold) and ``epsilon``.
    """
    data = result.data
    nuisance = result.nuisance
    fluctuation = result.fluctuations["regime"]
    rule = np.asarray(nuisance.regimes.values[:, 1, 0], dtype=float)
    treatment = np.asarray(data.treatment, dtype=float)
    lower, upper = result.config.g_bounds
    g1 = np.clip(np.asarray(nuisance.propensity.arm(1.0), dtype=float), lower, upper)
    g_observed = np.where(treatment == 1.0, g1, 1.0 - g1)
    targeted = fluctuation.targeted
    q_star_rule = np.where(rule == 1.0, targeted.arms[1.0], targeted.arms[0.0])
    return {
        "rule": rule,
        "h": (treatment == rule).astype(float) / g_observed,
        "y": np.asarray(nuisance.scaler.scale(data.outcome), dtype=float),
        "q_star_observed": np.asarray(targeted.observed, dtype=float),
        "q_star_rule": np.asarray(q_star_rule, dtype=float),
        "tests": [np.asarray(test) for _, test in nuisance.folds],
        "epsilon": np.asarray(fluctuation.epsilon, dtype=float),
    }
