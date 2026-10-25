r"""A navigation offer whose benefit varies by patient, with the exact value of any offer policy.

A hospital can offer patients navigation at the discharge-home order.  Navigation helps patients
with high need and does slightly worse than usual care for patients with low need, so the office
wants a rule for who gets the offer.  The outcome is whether the patient stays home, without an
unplanned readmission, for 30 days.

=================  ==============================================================================
column             law
=================  ==============================================================================
``need``           Uniform(-1, 1), a need score from lowest (-1) to highest (1)
``lives_alone``    Bernoulli(0.5)
``offer``          Bernoulli, :math:`\operatorname{logit} g(W) = -0.1 + 0.5\,\text{need}
                   + 0.3\,\text{lives\_alone}`
``home_30``        Bernoulli, :math:`\operatorname{logit} \bar Q(a, W) = 0.8 - 0.6\,\text{need}
                   - 0.3\,\text{lives\_alone} + a\,(0.1 + \text{need})`
=================  ==============================================================================

**The rule condition.**  The effect of the offer on the logit scale, :math:`b(W) = 0.1 +
\text{need}`, is zero only at ``need = -0.1``, a set of probability zero.  So the best rule,
"offer when ``need > -0.1``", is unique, and a logistic outcome model with every pairwise product
of ``(offer, need, lives_alone)`` contains the true :math:`\bar Q`.  The law has the form of the
``non_exceptional`` law of the registered study ``learned-rule-cvtmle``: the same distribution of
the two covariates, a blip linear in the continuous one, and a logistic treatment mechanism.  The
coefficients differ.

**Truth.**  For an offer policy that offers navigation with probability :math:`p(W)`, the value is

.. math::

    \Psi_p = E_W\bigl[p(W)\, \bar Q(1, W) + \{1 - p(W)\}\, \bar Q(0, W)\bigr].

:func:`make_treatment_rules` evaluates it in closed form for the static plans and the best rule,
from :math:`\int \operatorname{expit}(c + s x)\,dx = \log(1 + e^{c + s x}) / s`.
:func:`treatment_rule_value` evaluates it for any policy by the trapezoid rule on a grid of
``need``.  ``tests/unit/test_datasets_treatment_rules.py`` compares both with Monte Carlo draws.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .._typing import Backend
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = ["learned_rule_truth", "make_treatment_rules", "treatment_rule_value"]

#: The coefficients of the law, each on the logit scale.
_LAW = {
    "lives_alone": 0.5,
    # the offer
    "offer_intercept": -0.1,
    "offer_need": 0.5,
    "offer_alone": 0.3,
    # staying home for 30 days, under usual care
    "home_intercept": 0.8,
    "home_need": -0.6,
    "home_alone": -0.3,
    # the effect of the offer: blip_intercept + blip_need * need
    "blip_intercept": 0.1,
    "blip_need": 1.0,
}

#: The need score at which the offer's effect changes sign.
BEST_RULE_CUTOFF = -_LAW["blip_intercept"] / _LAW["blip_need"]

#: The default grid of ``need`` for :func:`treatment_rule_value`.
_GRID_POINTS = 40_001


def _home(offer: Any, need: Any, alone: Any) -> Any:
    """``P(home_30 = 1 | offer, need, lives_alone)``."""
    return expit(
        _LAW["home_intercept"]
        + _LAW["home_need"] * need
        + _LAW["home_alone"] * alone
        + offer * (_LAW["blip_intercept"] + _LAW["blip_need"] * need)
    )


def _integral(offer: float, alone: float, low: float, high: float) -> float:
    """``int_low^high P(home_30 = 1 | offer, need, alone) d need``, in closed form.

    Both slopes of the law, ``home_need`` and ``home_need + blip_need``, are nonzero.
    """
    intercept = _LAW["home_intercept"] + _LAW["home_alone"] * alone
    slope = _LAW["home_need"]
    if offer:
        intercept += _LAW["blip_intercept"]
        slope += _LAW["blip_need"]
    return float(
        (np.logaddexp(0.0, intercept + slope * high) - np.logaddexp(0.0, intercept + slope * low))
        / slope
    )


def _threshold_value(cutoff: float) -> float:
    """The exact value of "offer when ``need > cutoff``", for a cutoff in ``[-1, 1]``."""
    total = 0.0
    for alone, mass in ((1.0, _LAW["lives_alone"]), (0.0, 1.0 - _LAW["lives_alone"])):
        inside = _integral(0.0, alone, -1.0, cutoff) + _integral(1.0, alone, cutoff, 1.0)
        total += mass * inside / 2.0
    return total


def treatment_rule_value(
    policy: Callable[[pd.DataFrame], Any], *, grid_points: int = _GRID_POINTS
) -> float:
    r"""Return the share who stay home for 30 days, had the offer followed ``policy``.

    Parameters
    ----------
    policy : callable
        Takes a frame with the columns ``need`` and ``lives_alone`` and returns, for each row,
        the probability of an offer: 0 or 1 for a rule, a value in between for a lottery.
    grid_points : int, default=40001
        The points of the trapezoid rule over ``need``, for each value of ``lives_alone``.

    Returns
    -------
    float
        :math:`\Psi_p`, the value of the policy. A policy that switches between 0 and 1 a few
        times over ``need`` has an error of about ``1e-5`` at the default grid.

    See Also
    --------
    cleverly.datasets.make_treatment_rules : The law, and the exact value of each static plan.

    Examples
    --------
    >>> from cleverly.datasets import treatment_rule_value
    >>> round(treatment_rule_value(lambda frame: frame["need"] > -0.1), 4)
    0.7197
    """
    if grid_points < 3:
        raise ValueError("grid_points must be at least 3")
    need = np.linspace(-1.0, 1.0, grid_points)
    total = 0.0
    for alone, mass in ((1.0, _LAW["lives_alone"]), (0.0, 1.0 - _LAW["lives_alone"])):
        points = pd.DataFrame({"need": need, "lives_alone": np.full(grid_points, alone)})
        offered = np.asarray(policy(points), dtype=float).reshape(-1)
        if offered.shape != need.shape:
            raise ValueError("policy must return one probability for each row")
        if np.any((offered < 0.0) | (offered > 1.0)) or not np.all(np.isfinite(offered)):
            raise ValueError("policy must return probabilities in [0, 1]")
        values = offered * _home(1.0, need, alone) + (1.0 - offered) * _home(0.0, need, alone)
        total += mass * float(np.trapezoid(values, need)) / 2.0
    return total


def make_treatment_rules(
    n: int = 2_000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Draw the navigation program whose offer helps high-need patients and not low-need ones.

    Parameters
    ----------
    n : int, default=2000
        The number of patients.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per patient, with the columns ``need``, ``lives_alone``, ``offer`` and
        ``home_30``.
    truth : dict of str to float
        The share who stay home for 30 days had every patient been offered navigation
        (``ey_all``), had none been (``ey_none``), and under the best rule (``ey_best_rule``),
        each exact; their difference ``ate`` (``ey_all - ey_none``); and ``best_rule_cutoff``,
        the need score above which the best rule offers navigation.

    See Also
    --------
    cleverly.datasets.treatment_rule_value : The exact value of any other offer policy.
    cleverly.datasets.learned_rule_truth : The true value of the rules a learned-rule fit used.

    Examples
    --------
    >>> from cleverly.datasets import make_treatment_rules
    >>> frame, truth = make_treatment_rules(n=5, seed=0)
    >>> list(frame.columns)
    ['need', 'lives_alone', 'offer', 'home_30']
    >>> round(truth["ey_best_rule"] - truth["ey_all"], 4)
    0.0434
    """
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    need = rng.uniform(-1.0, 1.0, n)
    alone = rng.binomial(1, _LAW["lives_alone"], n).astype(float)
    offer = rng.binomial(
        1,
        expit(_LAW["offer_intercept"] + _LAW["offer_need"] * need + _LAW["offer_alone"] * alone),
    ).astype(float)
    home = rng.binomial(1, _home(offer, need, alone)).astype(float)
    payload = {"need": need, "lives_alone": alone, "offer": offer, "home_30": home}
    truth = {
        "ey_all": _threshold_value(-1.0),
        "ey_none": _threshold_value(1.0),
        "ey_best_rule": _threshold_value(BEST_RULE_CUTOFF),
        "best_rule_cutoff": BEST_RULE_CUTOFF,
    }
    truth["ate"] = truth["ey_all"] - truth["ey_none"]
    return frame_from_dict(payload, backend=backend), truth


def learned_rule_truth(result: Any, outcome_learner: Any) -> float:
    """Return the true value of the rules that a learned-rule fit of these data used.

    A ``LearnedRuleValue`` fit learns one rule in each fold, from the outcome model fitted on the
    other folds, and reports the average of the values of those rules.  This function refits
    ``outcome_learner`` on each fold's training rows, checks that its rule matches the fit's rule
    on every row of the fold, and averages the exact values of the refitted rules.

    Parameters
    ----------
    result : EstimationResult
        A ``LearnedRuleValue`` fit of a :func:`make_treatment_rules` draw, adjusted for ``need``
        and ``lives_alone`` in that order.
    outcome_learner : estimator
        The outcome learner of the fit.  It must be deterministic, so that a refit reproduces the
        fitted model.

    Returns
    -------
    float
        The average over the folds of :func:`treatment_rule_value` of each fold's rule.

    Raises
    ------
    ValueError
        When the fit is not a ``LearnedRuleValue`` fit of ``home_30`` on ``offer``, adjusted
        for ``need`` and ``lives_alone``, or a refitted rule differs from the fit's rule on a
        row.

    See Also
    --------
    cleverly.LearnedRuleValue : The estimand of the fit.
    """
    from ..estimators._nuisance import fit_on_rows
    from ..learners._fitting import predict_mean

    if getattr(getattr(result, "config", None), "parameter_axis", None) != "learned_rule":
        raise ValueError("the fit must be a LearnedRuleValue fit")
    data = result.data
    if (data.outcome_name, data.treatment_name) != ("home_30", "offer"):
        raise ValueError("the fit must have the outcome home_30 and the treatment offer")
    if tuple(data.covariate_names) != ("need", "lives_alone"):
        raise ValueError("the fit must adjust for need and lives_alone, in that order")
    design = np.asarray(data.treatment_design(), dtype=float)
    covariates = np.asarray(data.covariates, dtype=float)
    outcome = np.asarray(data.outcome, dtype=float)
    weights = np.asarray(data.weights, dtype=float)
    fitted = np.asarray(result.nuisance.regimes.values[:, 1, 0], dtype=float) > 0.5

    def blip(model: Any, rows: np.ndarray) -> np.ndarray:
        ones = np.ones((rows.shape[0], 1))
        treated = predict_mean(model, np.hstack([ones, rows]), "classification")
        control = predict_mean(model, np.hstack([0.0 * ones, rows]), "classification")
        return np.asarray(np.clip(treated, 0, 1) - np.clip(control, 0, 1), dtype=float)

    values = []
    for fold, (train, test) in enumerate(result.nuisance.folds):
        model = fit_on_rows(
            outcome_learner, design, outcome, weights, np.asarray(train), "classification", None
        )
        if np.any((blip(model, covariates[test]) > 0.0) != fitted[test]):
            raise ValueError(f"the refitted rule of fold {fold} differs from the fit's rule")

        def rule(frame: pd.DataFrame, model: Any = model) -> np.ndarray:
            rows = frame[["need", "lives_alone"]].to_numpy(dtype=float)
            return (blip(model, rows) > 0.0).astype(float)

        values.append(treatment_rule_value(rule))
    return float(np.mean(values))
