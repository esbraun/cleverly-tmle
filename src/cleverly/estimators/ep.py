r"""The EP learner of a conditional contrast curve (P1 part (b)).

van der Laan, Carone and Luedtke (arXiv:2402.01972v2, a preprint; "EPL" below) build an
efficient plug-in (EP) learner of a conditional contrast in three steps.

1. **Nuisances.**  Cross-fitted outcome regressions :math:`\mu_n(a, W)`, the treatment
   mechanism :math:`\pi_n(a \mid W)` and, with missing outcomes, the observation mechanism
   :math:`\Pi_n(a, W)`, each row with its own out-of-fold fit.  They come from the shipped
   point-treatment nuisance fit (:class:`~cleverly.TMLE`), so the learners, folds, weights,
   clusters, truncation and outcome scaling are the ones every point fit uses.
2. **Debiasing** (EPL Algorithm 2, App C.1; Method 1).  One pooled logistic regression of the
   scaled outcome :math:`Y \in [0, 1]` on the features below, with offset
   :math:`\operatorname{logit}\mu_n(A_i, W_i)` and weight
   :math:`\Delta_i w_i / \{\pi_n(A_i \mid W_i)\,\Pi_n(A_i, W_i)\}`.  Its score
   :math:`\sum_i \Delta_i w_i x_i (Y_i - \mu^*_i) / (\pi_n \Pi_n)` is zero at the fit, which
   are EPL's equations (6), and :math:`\mu^*` stays inside :math:`(0, 1)`.

   - difference (CATE): :math:`x = (\mathbb 1\{A = t\} - \mathbb 1\{A = r\})\,\varphi_k(V)`,
     so :math:`\operatorname{logit}\mu^*(t, W) = \operatorname{logit}\mu_n(t, W) +
     \varphi_k^\top b` and :math:`\operatorname{logit}\mu^*(r, W) =
     \operatorname{logit}\mu_n(r, W) - \varphi_k^\top b`;
   - log ratio (log CRR): :math:`x = (\mathbb 1\{A \in \{t, r\}\}\,\varphi_k,
     \mathbb 1\{A = t\}\,\varphi_k)`, which span the two blocks :math:`H_1 = 1` and
     :math:`H_2 = -a` of EPL eq (3).

3. **Final stage** (EPL Algorithm 3).  On the difference scale, least squares of
   :math:`\mu^*(t, W) - \mu^*(r, W)` on the sieve basis (or any regressor the caller
   declares); on the log-ratio scale, the logistic regression of
   :math:`\mu^*_t / (\mu^*_t + \mu^*_r)` on the basis with weight :math:`\mu^*_t + \mu^*_r`.

The sieve size is fixed, read from a declared schedule ``k(n)``, or chosen by EPL Algorithm 4
(cross-validation of the one-step risk).  The cross-validation refits the nuisances by an
inner cross-fit of each training fold, so no nuisance and no curve that is evaluated on a
held-out fold saw that fold: EPL's own remedy for the leakage it notes after Algorithm 4.

The curve carries no interval.  EPL proves rates and the efficiency of the risk, not a limit
law for the curve (Section 5), so :meth:`cleverly.effect_curves.EffectCurveResult.predict`
refuses ``interval=True`` on an EP curve (R1 of the P1 plan).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.special import expit, logit

from .._typing import FloatArray, IntArray
from .._warnings import attach
from ..bases import Constant, Modifiers, Sieve, resolve
from ..data.causal_data import CausalData
from ..effect_modification import (
    EffectModification,
    effect_modification_ep_refusal,
    modifier_frame,
)
from ..exceptions import CapabilityError, DataError, Refusal, RefusalReason
from ..learners._fitting import clone_learner, fit_learner, seed_unset
from ..learners.parametric import logistic_newton

__all__ = [
    "EP_INTERVAL_REFUSAL",
    "EP_MEASURED",
    "INNER_FOLDS",
    "Debiasing",
    "DebiasingScores",
    "EPCurve",
    "EPEstimator",
    "EPFit",
    "SieveRecord",
    "cross_validated_risks",
    "curve_from_pieces",
    "debias",
    "debias_features",
    "ep_status",
    "final_stage",
    "nuisance_pieces",
    "one_step_risk",
    "refit_seed",
    "sieve_member",
    "size_refusal",
]

#: Inner folds of the cross-fit that refits the nuisances inside each training fold of the
#: sieve-size cross-validation (the shipped rule-learner count).
INNER_FOLDS = 5

#: R1: no interval for an EP curve.
EP_INTERVAL_REFUSAL = Refusal(
    RefusalReason.INFERENCE_GAP,
    "The EP learner reports the conditional effect curve without confidence intervals. Its "
    "source (van der Laan, Carone and Luedtke, arXiv:2402.01972v2, Section 5) proves rates "
    "and the efficiency of the risk, not a limit law for the curve, and an interval for a "
    "nonparametric conditional effect at a point or over a range needs a limit theorem of a "
    "new kind.",
    alternative=(
        "Declare a finite working model, ConditionalContrast(..., model=...), and fit "
        "method='tmle': its coefficients and its projection curve carry intervals."
    ),
)


# ---------------------------------------------------------------- the debiasing step


def debias_features(
    design: FloatArray, treatment: FloatArray, pair: tuple[float, float], scale: str
) -> FloatArray:
    """The debiasing features at each row's observed treatment.

    Parameters
    ----------
    design : ndarray
        ``(n, k)`` sieve basis :math:`\\varphi_k(V_i)`.
    treatment : ndarray
        ``(n,)`` arm codes.
    pair : tuple of float
        The arm codes ``(t, r)``.
    scale : {"difference", "log_ratio"}
        The contrast's scale.

    Returns
    -------
    ndarray
        ``(n, k)`` on the difference scale, ``(n, 2k)`` on the log-ratio scale.  Rows of an
        arm outside the pair carry 0.
    """
    phi = np.asarray(design, dtype=float)
    a = np.asarray(treatment, dtype=float)
    t, r = pair
    at_t = (a == t).astype(float)
    at_r = (a == r).astype(float)
    if scale == "difference":
        return phi * (at_t - at_r)[:, None]
    return np.column_stack([phi * (at_t + at_r)[:, None], phi * at_t[:, None]])


def _arm_increments(
    design: FloatArray, coefficients: FloatArray, scale: str
) -> tuple[FloatArray, FloatArray]:
    """The logit increments of arm ``t`` and arm ``r`` at every row."""
    phi = np.asarray(design, dtype=float)
    b = np.asarray(coefficients, dtype=float)
    k = phi.shape[1]
    if scale == "difference":
        step = phi @ b
        return step, -step
    common = phi @ b[:k]
    return common + phi @ b[k:], common


@dataclass(frozen=True)
class Debiasing:
    """One pooled debiasing regression and the targeted arm regressions it gives.

    Parameters
    ----------
    coefficients : ndarray
        The fitted coefficients :math:`b`, ``k`` on the difference scale and ``2k`` on the
        log-ratio scale.
    targeted_t : ndarray
        :math:`\\mu^*(t, W_i)` on the scaled outcome.
    targeted_r : ndarray
        :math:`\\mu^*(r, W_i)` on the scaled outcome.
    score_before : ndarray
        The weighted mean score of each feature at the initial fit.
    score_after : ndarray
        The same at the debiased fit; zero up to the solver's tolerance.
    """

    coefficients: FloatArray
    targeted_t: FloatArray
    targeted_r: FloatArray
    score_before: FloatArray
    score_after: FloatArray

    @property
    def contrast(self) -> FloatArray:
        """:math:`\\mu^*(t, W) - \\mu^*(r, W)` on the scaled outcome."""
        return self.targeted_t - self.targeted_r


def debias(
    *,
    y: FloatArray,
    treatment: FloatArray,
    observed: FloatArray,
    initial_observed: FloatArray,
    initial_t: FloatArray,
    initial_r: FloatArray,
    weight: FloatArray,
    design: FloatArray,
    pair: tuple[float, float],
    scale: str,
) -> Debiasing:
    """The pooled offset-logistic debiasing regression (EPL Algorithm 2, Method 1).

    Parameters
    ----------
    y : ndarray
        The scaled outcome in ``[0, 1]``; any value where ``observed`` is 0.
    treatment : ndarray
        Arm codes.
    observed : ndarray
        The outcome observation indicator :math:`\\Delta`.
    initial_observed : ndarray
        :math:`\\mu_n(A_i, W_i)`, inside ``(0, 1)``.
    initial_t : ndarray
        :math:`\\mu_n(t, W_i)`, inside ``(0, 1)``.
    initial_r : ndarray
        :math:`\\mu_n(r, W_i)`, inside ``(0, 1)``.
    weight : ndarray
        :math:`w_i / \\{\\pi_n(A_i \\mid W_i)\\,\\Pi_n(A_i, W_i)\\}`; the observation indicator
        is applied here.
    design : ndarray
        ``(n, k)`` sieve basis.
    pair : tuple of float
        The arm codes ``(t, r)``.
    scale : {"difference", "log_ratio"}
        The contrast's scale.

    Returns
    -------
    Debiasing
        The coefficients and the targeted arm regressions.
    """
    seen = np.asarray(observed, dtype=float) > 0.5
    x = debias_features(design, treatment, pair, scale)
    target = np.where(seen, np.asarray(y, dtype=float), 0.0)
    w = np.where(seen, np.asarray(weight, dtype=float), 0.0)
    offset = logit(np.asarray(initial_observed, dtype=float))
    total = float(np.sum(w))
    before = x.T @ (w * (target - np.asarray(initial_observed, dtype=float))) / total
    keep = np.flatnonzero(np.any(x != 0.0, axis=0))
    coefficients = np.zeros(x.shape[1])
    if keep.size:
        fitted, _ = logistic_newton(
            x[:, keep], target, w, label="EP debiasing regression", offset=offset
        )
        coefficients[keep] = fitted
    up_t, up_r = _arm_increments(design, coefficients, scale)
    targeted_t = expit(logit(np.asarray(initial_t, dtype=float)) + up_t)
    targeted_r = expit(logit(np.asarray(initial_r, dtype=float)) + up_r)
    after = x.T @ (w * (target - expit(offset + x @ coefficients))) / total
    return Debiasing(coefficients, targeted_t, targeted_r, before, after)


# ---------------------------------------------------------------- the final stage


@dataclass(frozen=True)
class EPCurve:
    """The fitted EP curve, which predicts the conditional contrast at new rows.

    Parameters
    ----------
    basis : Basis
        The sieve member the debiasing regression used, at the chosen size.
    modifiers : Modifiers
        The modifier schema.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    terms : tuple of str
        The basis terms.
    beta : ndarray or None
        The final stage's coefficients on the basis, on the scaled outcome, or ``None`` when a
        declared final learner fitted the curve.
    learner : Any
        The fitted final learner, or ``None`` for the default least squares.
    width : float
        ``b - a``, the outcome range the scaled contrast is multiplied by.
    clip : bool
        Whether predictions are clipped to ``[-(b - a), b - a]``: the default with
        ``model=None`` on the difference scale, where the target lies in that range.
    """

    basis: Any
    modifiers: Modifiers
    scale: str
    terms: tuple[str, ...]
    beta: FloatArray | None
    learner: Any
    width: float
    clip: bool

    def design(self, frame: Any) -> FloatArray:
        """The sieve basis at the rows of ``frame``, checked against the modifier schema.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.

        Returns
        -------
        ndarray
            ``(m, k)`` design.
        """
        return np.asarray(self.basis.evaluate(frame, self.modifiers), dtype=float)

    def raw(self, frame: Any) -> FloatArray:
        """The unclipped curve at the rows of ``frame``, on the outcome's own scale.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.

        Returns
        -------
        ndarray
            One prediction per row.
        """
        if self.learner is None:
            assert self.beta is not None
            value = self.design(frame) @ self.beta
        else:
            values = self.modifiers.values(frame, self.modifiers.columns)
            value = np.asarray(self.learner.predict(_learner_input(values)), dtype=float)
        if self.scale == "difference":
            return self.width * value
        return value

    def predict(self, frame: Any) -> FloatArray:
        """The curve at the rows of ``frame``, clipped when :attr:`clip` is set.

        Parameters
        ----------
        frame : dataframe or mapping
            Rows holding the modifier columns.

        Returns
        -------
        ndarray
            One prediction per row, on the outcome's own scale (a difference of means, or a
            log ratio).
        """
        value = self.raw(frame)
        if self.clip:
            return np.clip(value, -self.width, self.width)
        return value


def _learner_input(values: dict[str, np.ndarray]) -> Any:
    """The modifier frame a declared final learner reads, columns in declaration order."""
    import pandas as pd

    return pd.DataFrame({name: np.asarray(column) for name, column in values.items()})


def final_stage(
    debiased: Debiasing,
    design: FloatArray,
    weights: FloatArray,
    scale: str,
    *,
    learner: Any = None,
    values: dict[str, np.ndarray] | None = None,
) -> tuple[FloatArray | None, Any]:
    """Fit the final stage on the debiased arm regressions (EPL Algorithm 3).

    Parameters
    ----------
    debiased : Debiasing
        The debiasing step's targeted arm regressions.
    design : ndarray
        ``(n, k)`` sieve basis at the fit rows.
    weights : ndarray
        ``(n,)`` observation weights.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    learner : Any
        A declared final regressor, or ``None`` for the default.
    values : dict of str to ndarray or None
        The modifier columns a declared learner reads.

    Returns
    -------
    tuple
        The basis coefficients (or ``None``) and the fitted learner (or ``None``).
    """
    w = np.asarray(weights, dtype=float)
    if scale == "difference":
        contrast = debiased.contrast
        if learner is not None:
            assert values is not None
            # Unit weights are no weights. Passing them changes no fit, but it sends
            # scikit-learn's histogram booster through weighted-percentile binning, which
            # doubles its cost (tests/unit/test_learned_rule_ep_declared_design.py).
            unit = bool(np.all(w == 1.0))
            fitted = fit_learner(
                clone_learner(learner), _learner_input(values), contrast, None if unit else w
            )
            return None, fitted
        root = np.sqrt(w)
        beta = np.linalg.lstsq(design * root[:, None], contrast * root, rcond=None)[0]
        return np.asarray(beta, dtype=float), None
    total = debiased.targeted_t + debiased.targeted_r
    share = debiased.targeted_t / total
    beta, _ = logistic_newton(design, share, w * total, label="EP log-ratio final stage")
    return np.asarray(beta, dtype=float), None


def one_step_risk(
    prediction: FloatArray,
    d_t: FloatArray,
    d_r: FloatArray,
    scale: str,
    weights: FloatArray | None = None,
) -> float:
    """The one-step (doubly robust) risk of a curve, the loss EPL Algorithm 4 compares.

    Difference: :math:`P_n[\\theta^2 - 2\\theta(D_t - D_r)]`.  Log ratio:
    :math:`P_n[(D_t + D_r)\\log(1 + e^\\theta) - D_t\\theta]`.  ``D_a`` are the AIPW
    pseudo-outcomes of each arm, so the risk is EPL Theorem 1's one-step estimator at a fixed
    curve.

    Parameters
    ----------
    prediction : ndarray
        The curve at the rows, on the outcome's own scale.
    d_t : ndarray
        Pseudo-outcome of arm ``t``.
    d_r : ndarray
        Pseudo-outcome of arm ``r``.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    weights : ndarray or None
        Observation weights.

    Returns
    -------
    float
        The weighted mean loss.
    """
    theta = np.asarray(prediction, dtype=float)
    w = np.ones_like(theta) if weights is None else np.asarray(weights, dtype=float)
    if scale == "difference":
        loss = theta**2 - 2.0 * theta * (d_t - d_r)
    else:
        loss = (d_t + d_r) * np.logaddexp(0.0, theta) - d_t * theta
    return float(np.sum(w * loss) / np.sum(w))


#: One training fold's curves at a size, predicted at the fold's held-out modifier rows.
FoldPredictor = Callable[[int, dict[str, Any]], FloatArray]


def cross_validated_risks(
    splits: Sequence[tuple[int, IntArray, IntArray]],
    sizes: Sequence[int],
    fit_fold: Callable[[int, IntArray], FoldPredictor],
    frame: Mapping[str, Any],
    d_t: FloatArray,
    d_r: FloatArray,
    scale: str,
    weights: FloatArray,
) -> tuple[float, ...]:
    """The weighted cross-validated one-step risk of each sieve size (EPL Algorithm 4).

    Each split ``(key, train, test)`` refits its curves through ``fit_fold(key, train)``,
    which sees the training rows only, and scores each size's curve on ``test`` against the
    pseudo-outcomes there.  A size's risk is the weight-pooled mean of its held-out losses.
    Both the EP learner's size step and the EP rule learner's read this one loop.

    Parameters
    ----------
    splits : sequence of tuple
        ``(key, train, test)`` per training fold, with integer row indices.
    sizes : sequence of int
        The sieve sizes compared.
    fit_fold : callable
        ``fit_fold(key, train)`` returns ``predict(size, held)``, the curve of that size at the
        held-out modifier columns ``held``, on the scale the pseudo-outcomes are on.
    frame : mapping
        The modifier columns of every row.
    d_t : ndarray
        Pseudo-outcome of arm ``t`` at every row.
    d_r : ndarray
        Pseudo-outcome of arm ``r`` at every row.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    weights : ndarray
        The row weights.

    Returns
    -------
    tuple of float
        The risk of each size, in the order of ``sizes``.
    """
    totals = np.zeros(len(sizes))
    mass = 0.0
    for key, train, test in splits:
        predict = fit_fold(key, train)
        held = {name: np.asarray(column)[test] for name, column in frame.items()}
        held_weights = weights[test]
        total = float(np.sum(held_weights))
        for position, size in enumerate(sizes):
            totals[position] += (
                one_step_risk(predict(size, held), d_t[test], d_r[test], scale, held_weights)
                * total
            )
        mass += total
    return tuple(float(value / mass) for value in totals)


# ---------------------------------------------------------------- records and status


@dataclass(frozen=True)
class SieveRecord:
    """How the sieve size was set, and what the cross-validation read.

    Parameters
    ----------
    mode : {"fixed", "schedule", "cv"}
        The size rule.
    family : str
        The sieve family's basis, by repr.
    sizes : tuple of int
        The sizes tried; one for a fixed size or a schedule.
    cv_risks : tuple of float
        The cross-validated one-step risk at each size, or empty.
    chosen : int
        The size of the reported curve.
    inner_seeds : tuple of int
        The seed of each training fold's inner refit, or empty.
    inner_warnings : tuple of str
        The distinct warnings the inner refits raised.
    """

    mode: str
    family: str
    sizes: tuple[int, ...]
    cv_risks: tuple[float, ...]
    chosen: int
    inner_seeds: tuple[int, ...] = ()
    inner_warnings: tuple[str, ...] = ()


#: The EP configurations a registered study measured, by ``(scale, mode, target)``, each with
#: the study that measured it (P1 plan 2.6).  ``ep-cate`` measured the CATE curve at a declared
#: schedule and the EP coefficients of a declared model at a fixed size; ``ep-crr`` measured the
#: log-CRR curve at a declared schedule.  Each study fitted one outcome regression per arm, a
#: binary treatment, iid rows, complete outcomes and the default final stage, so a fit that
#: departs from any of these reads experimental.
EP_MEASURED: dict[tuple[str, str, str], str] = {
    ("difference", "schedule", "curve"): "ep-cate",
    ("difference", "fixed", "projection"): "ep-cate",
    ("log_ratio", "schedule", "curve"): "ep-crr",
}


def ep_status(
    *,
    scale: str,
    mode: str,
    target: str,
    declared_learner: bool,
    weighted: bool,
    clustered: bool,
    missing: bool,
    by_arm: bool,
    arms: int,
    known_mechanism: bool,
) -> tuple[str, str]:
    """The support label of an EP curve and its reason (P1 plan 2.6).

    A fit reads ``supported`` when its configuration is one of :data:`EP_MEASURED` and it
    departs from the measured composition in nothing.  Otherwise it reads ``experimental``,
    with every reason that applies.

    Parameters
    ----------
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    mode : {"fixed", "schedule", "cv"}
        The sieve size rule.
    target : {"curve", "projection"}
        ``"curve"`` for ``model=None``; ``"projection"`` when the estimand declared a model.
    declared_learner : bool
        Whether a declared final learner fitted the curve.
    weighted : bool
        Whether the fit has observation weights.
    clustered : bool
        Whether the fit has clusters.
    missing : bool
        Whether the fit has missing outcomes.
    by_arm : bool
        Whether the outcome regression is fitted per arm (``outcome_by_arm=True``).
    arms : int
        The number of treatment arms.
    known_mechanism : bool
        Whether the design declares the treatment probabilities.

    Returns
    -------
    tuple of str
        The label, ``"supported"`` or ``"experimental"``, and its reason.
    """
    reasons = []
    if mode == "cv":
        reasons.append(
            "The sieve size is chosen by cross-validation, which the EP learner's theorems do "
            "not cover (arXiv:2402.01972v2, Section 5.1)."
        )
    if declared_learner:
        reasons.append(
            "The theorems cover empirical risk minimization over a declared convex class, and "
            "the package cannot check that this final learner is one (arXiv:2402.01972v2, "
            "Section 7.1)."
        )
    extras = [
        name
        for name, present in (
            ("observation weights", weighted),
            ("clusters", clustered),
            ("missing outcomes", missing),
            ("more than two arms", arms > 2),
            ("declared treatment probabilities", known_mechanism),
        )
        if present
    ]
    if extras:
        reasons.append(
            f"The EP curve with {' and '.join(extras)} is implemented and not measured by a study."
        )
    if not by_arm:
        reasons.append(
            "The fit pools the arms in one outcome regression, and the registered EP studies "
            "measure one outcome regression per arm (outcome_by_arm=True)."
        )
    if mode != "cv" and not declared_learner and (scale, mode, target) not in EP_MEASURED:
        if scale == "log_ratio" and mode == "fixed":
            reasons.append(
                "No registered study measures the log-ratio EP curve at a fixed size, and at a "
                "fixed size its debiasing does not solve the projection's efficient score."
            )
        else:
            reasons.append(
                "No registered study measures the EP curve with model=None at a fixed size: the "
                "ep-cate study measures a fixed size with a declared model, and the curve at a "
                "declared schedule."
            )
    if reasons:
        return "experimental", " ".join(reasons)
    study = EP_MEASURED[(scale, mode, target)]
    if mode == "fixed":
        claim = (
            "The registered ep-cate study measures the EP coefficients of a declared model at a "
            "fixed size, the projection of the CATE onto the model's span; the curve has no "
            "interval."
        )
    else:
        claim = (
            f"The registered {study} study measures the EP curve at a declared sieve schedule "
            "against the oracle learner of arXiv:2402.01972v2, Theorem 5, whose conditions C1 "
            "to C6 the user must judge; the curve has no interval."
        )
    return "supported", claim


def sieve_member(family: Any, size: int, modifiers: Modifiers) -> Any:
    """The member of ``size`` terms per component of the sieve family ``family``.

    Parameters
    ----------
    family : Basis
        The sieve family's largest member.
    size : int
        The size; 0 is the constant.
    modifiers : Modifiers
        The schema the member's level sets are read from.

    Returns
    -------
    Basis
        The truncated, resolved basis.
    """
    if size == 0:
        basis: Any = Constant()
    elif int(size) == int(family.max_size):
        basis = family
    else:
        basis = family.truncate(int(size))
    return resolve(basis, modifiers)


def size_refusal(size: int, family: Any, n: int) -> CapabilityError | None:
    """The refusal of a declared sieve size above the family's largest member, or ``None``.

    Parameters
    ----------
    size : int
        The declared size at ``n`` rows.
    family : Basis
        The sieve family's largest member.
    n : int
        The rows the size is declared for.

    Returns
    -------
    CapabilityError or None
        The refusal, which names the size and the family.
    """
    largest = int(family.max_size)
    if size <= largest:
        return None
    return _refuse(
        f"The sieve declares {size} terms per component at n = {n}, and its family "
        f"{family!r} has at most {largest}. Fitting the {largest}-term member would replace the "
        "declared size, which is the size the EP learner's rate claims name.",
        alternative=(
            f"Declare a size of at most {largest}, or a larger family "
            "(Sieve.default(..., max_size=...))."
        ),
    )


# ---------------------------------------------------------------- the estimator


def _refuse(text: str, alternative: str | None = None) -> CapabilityError:
    return CapabilityError(text, reason=RefusalReason.INVALID_DECLARATION, alternative=alternative)


@dataclass(frozen=True)
class _Pieces:
    """The arrays one EP fit reads from a nuisance fit, on the scaled outcome."""

    y: FloatArray
    treatment: FloatArray
    observed: FloatArray
    initial_observed: FloatArray
    initial_t: FloatArray
    initial_r: FloatArray
    weight: FloatArray
    rows: FloatArray
    pair: tuple[float, float]


def nuisance_pieces(
    data: CausalData,
    nuisance: Any,
    *,
    pair: tuple[float, float],
    g_bounds: tuple[float, float],
    alpha: float,
    missingness_bound: float | None = None,
) -> _Pieces:
    """Read the debiasing inputs of one EP fit off cross-fitted nuisances.

    The part (b) curve reads the nuisances of its projection TMLE, and the ``loss="ep"`` rule
    learner of part (c) reads the inner cross-fit of each outer training fold; both go through
    this function, so the two cannot diverge.

    Parameters
    ----------
    data : CausalData
        The rows the nuisances were fitted on.
    nuisance : NuisanceEstimates
        Cross-fitted nuisances of those rows, one out-of-fold prediction per row.
    pair : tuple of float
        The arm codes ``(t, r)``.
    g_bounds : tuple of float
        The truncation bounds of the treatment mechanism.
    alpha : float
        The outcome shrinkage of the logit fluctuation (``TMLE(alpha=...)``), which keeps the
        offset finite.
    missingness_bound : float or None
        The lower bound of the observation mechanism, read when the nuisances carry one.

    Returns
    -------
    _Pieces
        The arrays of the debiasing regression, on the scaled outcome.
    """
    shrunk = nuisance.outcome.shrunk(alpha)
    treatment = np.asarray(data.treatment, dtype=float)
    observed = np.asarray(data.observed, dtype=float)
    scaled = nuisance.scaler.scale(np.where(observed > 0.5, np.asarray(data.outcome), 0.0))
    y = np.clip(np.where(observed > 0.5, scaled, 0.0), 0.0, 1.0)
    g = np.asarray(nuisance.propensity.bounded(tuple(g_bounds)), dtype=float)
    missingness = (
        None if missingness_bound is None else nuisance.bounded_missingness(missingness_bound)
    )
    weight = np.zeros(data.n)
    for arm in pair:
        column = nuisance.propensity.column_for(arm)
        at = treatment == arm
        density = g[:, column]
        if missingness is not None:
            m = np.asarray(missingness, dtype=float)
            density = density * (m[:, column] if m.ndim == 2 else m)
        weight[at] = 1.0 / density[at]
    rows = np.asarray(data.weights, dtype=float)
    return _Pieces(
        y=y,
        treatment=treatment,
        observed=observed,
        initial_observed=np.asarray(shrunk.observed, dtype=float),
        initial_t=np.asarray(shrunk.arms[pair[0]], dtype=float),
        initial_r=np.asarray(shrunk.arms[pair[1]], dtype=float),
        weight=rows * weight,
        rows=rows,
        pair=(float(pair[0]), float(pair[1])),
    )


def _pieces(fit: Any) -> _Pieces:
    """Read the debiasing inputs off a projection TMLE's nuisances and resolved bounds."""
    config = fit.config
    return nuisance_pieces(
        fit.data,
        fit.nuisance,
        pair=tuple(fit.nuisance.effect_modification.pair),
        g_bounds=tuple(config.g_bounds),
        alpha=config.alpha,
        missingness_bound=float(config.missingness_bound),
    )


def curve_from_pieces(
    pieces: _Pieces,
    design: FloatArray,
    *,
    basis: Any,
    modifiers: Modifiers,
    scale: str,
    width: float,
    clip: bool,
    final_learner: Any = None,
    values: dict[str, np.ndarray] | None = None,
) -> tuple[EPCurve, Debiasing]:
    """Debias and fit the final stage at one sieve member (EPL Algorithms 2 and 3).

    Parameters
    ----------
    pieces : _Pieces
        The debiasing inputs (:func:`nuisance_pieces`).
    design : ndarray
        ``(n, k)`` the member's basis at the rows.
    basis : Basis
        The member, which the curve evaluates at new rows.
    modifiers : Modifiers
        The modifier schema.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    width : float
        ``b - a``, the outcome range.
    clip : bool
        Whether the curve clips its predictions to ``[-(b - a), b - a]``.
    final_learner : Any
        A declared final regressor, or ``None`` for least squares on the basis.
    values : dict of str to ndarray or None
        The modifier columns a declared final learner reads.

    Returns
    -------
    tuple
        The curve and its debiasing regression.
    """
    debiased = debias(
        y=pieces.y,
        treatment=pieces.treatment,
        observed=pieces.observed,
        initial_observed=pieces.initial_observed,
        initial_t=pieces.initial_t,
        initial_r=pieces.initial_r,
        weight=pieces.weight,
        design=design,
        pair=pieces.pair,
        scale=scale,
    )
    beta, learner = final_stage(
        debiased,
        design,
        pieces.rows,
        scale,
        learner=final_learner,
        values=values if final_learner is not None else None,
    )
    curve = EPCurve(
        basis=basis,
        modifiers=modifiers,
        scale=scale,
        terms=tuple(basis.terms),
        beta=beta,
        learner=learner,
        width=float(width),
        clip=clip,
    )
    return curve, debiased


@dataclass(frozen=True)
class DebiasingScores:
    """The weighted mean score of each debiasing feature, before and after the regression.

    Parameters
    ----------
    features : tuple of str
        One name per debiasing feature: ``(A)*<term>`` on the difference scale, and
        ``pair*<term>`` then ``t*<term>`` on the log-ratio scale.
    before : ndarray
        The score at the initial outcome regression.
    after : ndarray
        The score at the debiased one; the regression solves these equations.
    backend : str
        The dataframe backend of the producing fit's data, which :meth:`to_frame` emits in.
    """

    features: tuple[str, ...]
    before: FloatArray
    after: FloatArray
    backend: str = "pandas"

    @classmethod
    def of(cls, fit: EPFit, backend: str = "pandas") -> DebiasingScores:
        """The report of one EP fit.

        Parameters
        ----------
        fit : EPFit
            The fit.
        backend : str, default "pandas"
            The dataframe backend of the fit's data.

        Returns
        -------
        DebiasingScores
            Its debiasing scores.
        """
        terms = fit.curve.terms
        if fit.curve.scale == "difference":
            names = tuple(f"(A)*{term}" for term in terms)
        else:
            names = (*(f"pair*{term}" for term in terms), *(f"t*{term}" for term in terms))
        return cls(names, fit.debiasing.score_before, fit.debiasing.score_after, backend)

    @property
    def max_after(self) -> float:
        """The largest absolute score after the debiasing regression."""
        return float(np.max(np.abs(self.after))) if self.after.size else 0.0

    def to_frame(self) -> Any:
        """One row per feature, in the producing fit's backend.

        Returns
        -------
        dataframe
            The scores before and after.
        """
        from ..utils.frames import frame_from_dict

        return frame_from_dict(
            {"feature": np.asarray(self.features), "before": self.before, "after": self.after},
            backend=self.backend,
        )

    def __repr__(self) -> str:
        return f"DebiasingScores(features={len(self.features)}, max_after={self.max_after:.3g})"


@dataclass(frozen=True, eq=False)
class EPFit:
    """One EP fit: its nuisance fit, its debiasing, its final stage and its sieve record.

    Parameters
    ----------
    data : CausalData
        The prepared data.
    nuisance_fit : TMLEResult
        The projection TMLE whose cross-fitted nuisances the EP fit read.  Its own targeted
        coefficients are not reported.
    estimator : EPEstimator
        The estimator, which :meth:`refit` reruns.
    curve : EPCurve
        The fitted curve.
    debiasing : Debiasing
        The debiasing regression at the chosen size.
    sieve : SieveRecord
        How the size was set.
    """

    data: CausalData
    nuisance_fit: Any
    estimator: EPEstimator
    curve: EPCurve
    debiasing: Debiasing
    sieve: SieveRecord

    @property
    def nuisance(self) -> Any:
        """The cross-fitted nuisances the EP fit read."""
        return self.nuisance_fit.nuisance

    @property
    def config(self) -> Any:
        """The resolved settings of the nuisance fit: bounds, folds, scaling."""
        return self.nuisance_fit.config

    @property
    def diagnostics(self) -> Any:
        """The nuisance fit's diagnostics: support and held-out nuisance fits."""
        return self.nuisance_fit.diagnostics

    def effect_curve(self) -> Any:
        """The fit as an :class:`~cleverly.effect_curves.EffectCurveResult`.

        Returns
        -------
        EffectCurveResult
            The curve, its in-sample predictions and its label.
        """
        return self.estimator._result(self)


class EPEstimator:
    """The EP learner of one conditional contrast, configured and refittable.

    Parameters
    ----------
    tmle_kwargs : dict
        The keyword arguments of the point nuisance fit, from the method.
    effect_modification : EffectModification
        The modifiers, scale and pair, with the sieve family's basis (the largest member).
    size : int, "cv", callable or tuple
        The sieve size rule, as :class:`cleverly.bases.Sieve` holds it.
    target : {"curve", "projection"}
        ``"curve"`` for ``model=None``; ``"projection"`` when the estimand declared a model.
    final_learner : Any
        A declared final regressor, or ``None``.
    identified_effect : Any
        The identified effect the fit came from.
    """

    def __init__(
        self,
        tmle_kwargs: dict[str, Any],
        effect_modification: EffectModification,
        size: Any,
        *,
        target: str,
        final_learner: Any = None,
        identified_effect: Any = None,
    ) -> None:
        self.tmle_kwargs = dict(tmle_kwargs)
        self.effect_modification = effect_modification
        self.size = size
        self.target = target
        self.final_learner = final_learner
        self.identified_effect = identified_effect

    # ------------------------------------------------------------ declarations

    @property
    def sieve(self) -> Sieve:
        """The sieve: the family's basis and the size rule."""
        return Sieve(self.effect_modification.basis, self.size)

    @property
    def mode(self) -> str:
        """``"fixed"``, ``"schedule"`` or ``"cv"``."""
        if isinstance(self.size, str):
            return "cv"
        if isinstance(self.size, int) and not isinstance(self.size, bool):
            return "fixed"
        return "schedule"

    def member(self, size: int) -> Any:
        """The sieve member of ``size`` terms per component, resolved against the modifiers.

        Parameters
        ----------
        size : int
            The size; 0 is the constant.

        Returns
        -------
        Basis
            The truncated basis.
        """
        return sieve_member(
            self.effect_modification.basis, size, self.effect_modification.modifiers
        )

    def sizes(self, n: int) -> tuple[int, ...]:
        """The sizes the fit tries at ``n`` rows.

        Parameters
        ----------
        n : int
            The sample size.

        Returns
        -------
        tuple of int
            ``1, ..., max_size`` under cross-validation; otherwise the declared size.
        """
        if self.mode == "cv":
            return tuple(range(1, int(self.effect_modification.basis.max_size) + 1))
        return (int(self.sieve.size_at(n)),)

    def _nuisance_kwargs(self, **changes: Any) -> dict[str, Any]:
        spec = self.effect_modification
        kwargs = {
            **self.tmle_kwargs,
            "estimands": (spec.target,),
            "reference": None,
            "effect_modification": EffectModification(
                Constant(),
                spec.modifiers,
                spec.scale,
                treatment=spec.treatment,
                reference=spec.reference,
            ),
        }
        kwargs.update(changes)
        return kwargs

    # ------------------------------------------------------------ fitting

    def check(self, data: CausalData) -> None:
        """Evaluate every sieve member on the rows, before any learner.

        Parameters
        ----------
        data : CausalData
            The prepared data.

        Raises
        ------
        CapabilityError
            If the data hold a missing treatment or baseline strata (R7-ep-mt,
            R7-ep-strata), a fixed or scheduled size exceeds the family's largest member,
            or a basis reads an undeclared domain or level (R3).
        DataError
            If a row breaks the modifier schema, or a member is collinear on the rows.
        """
        data_refusal = effect_modification_ep_refusal(data)
        if data_refusal is not None:
            raise CapabilityError.from_refusal(data_refusal)
        frame = modifier_frame(data, tuple(self.effect_modification.modifiers.columns))
        for size in self.sizes(data.n):
            refusal = size_refusal(size, self.effect_modification.basis, data.n)
            if refusal is not None:
                raise refusal
            design = np.asarray(
                self.member(size).evaluate(frame, self.effect_modification.modifiers), dtype=float
            )
            if np.linalg.matrix_rank(design) < design.shape[1]:
                raise DataError(
                    f"the sieve member of size {size} is collinear on the rows of the modifiers "
                    f"({design.shape[1]} terms, rank {np.linalg.matrix_rank(design)})"
                )

    def _nuisances(self, data: CausalData, **changes: Any) -> Any:
        from .tmle import TMLE

        return TMLE(**self._nuisance_kwargs(**changes)).fit(data).single()

    def _curve_at(self, nuisance_fit: Any, size: int) -> tuple[EPCurve, Debiasing]:
        """Debias and fit the final stage at ``size`` from the nuisances of ``nuisance_fit``."""
        data = nuisance_fit.data
        spec = self.effect_modification
        basis = self.member(size)
        values = modifier_frame(data, tuple(spec.modifiers.columns))
        design = np.asarray(basis.evaluate(values, spec.modifiers), dtype=float)
        lower, upper = nuisance_fit.nuisance.scaler.unscale_levels(np.array([0.0, 1.0]))
        curve, debiased = curve_from_pieces(
            _pieces(nuisance_fit),
            design,
            basis=basis,
            modifiers=spec.modifiers,
            scale=spec.scale,
            width=float(upper - lower),
            clip=self.target == "curve" and spec.scale == "difference",
            final_learner=seed_unset(self.final_learner, self.tmle_kwargs.get("random_state")),
            values=values,
        )
        return curve, debiased

    def _cross_validate(
        self, data: CausalData, outer: Any, sizes: tuple[int, ...]
    ) -> tuple[int, tuple[float, ...], tuple[int, ...], tuple[str, ...]]:
        """EPL Algorithm 4 with the nuisances refitted inside each training fold."""
        from ..effect_curves import _pseudo_outcomes

        spec = self.effect_modification
        config = outer.config
        pseudo = _pseudo_outcomes(
            outer.nuisance,
            data,
            tuple(outer.nuisance.effect_modification.pair),
            g_bounds=tuple(config.g_bounds),
            missingness_bound=float(config.missingness_bound),
        )
        _, _, d_t, d_r = pseudo
        assignment = np.asarray(outer.nuisance.folds.assignment)
        weights = np.asarray(data.weights, dtype=float)
        frame = modifier_frame(data, tuple(spec.modifiers.columns))
        seeds: list[int] = []
        messages: list[str] = []
        base = self.tmle_kwargs.get("random_state")
        splits = [
            (int(fold), np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold))
            for fold in np.unique(assignment)
        ]

        def fit_fold(fold: int, train: Any) -> FoldPredictor:
            seed = refit_seed(base, fold)
            seeds.append(seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                inner = self._nuisances(
                    data.subset(train), n_folds=INNER_FOLDS, random_state=seed, split_plan=None
                )
            messages.extend(str(item.message) for item in caught)
            return lambda size, held: self._curve_at(inner, size)[0].predict(held)

        risks = cross_validated_risks(splits, sizes, fit_fold, frame, d_t, d_r, spec.scale, weights)
        chosen = sizes[int(np.argmin(risks))]
        return chosen, risks, tuple(seeds), tuple(dict.fromkeys(messages))

    def fit(self, data: CausalData) -> EPFit:
        """Fit the nuisances, choose or read the size, debias and fit the final stage.

        Parameters
        ----------
        data : CausalData
            The prepared data.

        Returns
        -------
        EPFit
            The fit.
        """
        self.check(data)
        outer = self._nuisances(data)
        sizes = self.sizes(data.n)
        risks: tuple[float, ...] = ()
        seeds: tuple[int, ...] = ()
        messages: tuple[str, ...] = ()
        if self.mode == "cv":
            chosen, risks, seeds, messages = self._cross_validate(data, outer, sizes)
        else:
            chosen = sizes[0]
        curve, debiased = self._curve_at(outer, chosen)
        record = SieveRecord(
            mode=self.mode,
            family=repr(self.effect_modification.basis),
            sizes=sizes,
            cv_risks=risks,
            chosen=int(chosen),
            inner_seeds=seeds,
            inner_warnings=messages,
        )
        return EPFit(data, outer, self, curve, debiased, record)

    def refit(self, data: CausalData, *, random_state: int | None = None) -> EPFit:
        """Run the whole fit again on ``data``, nuisances and size selection included.

        Parameters
        ----------
        data : CausalData
            The prepared data, usually a subset of the fit's rows.
        random_state : int or None
            The seed of the refit; ``None`` keeps this estimator's own.

        Returns
        -------
        EPFit
            The refit.
        """
        estimator = copy.copy(self)
        estimator.tmle_kwargs = dict(self.tmle_kwargs)
        if random_state is not None:
            estimator.tmle_kwargs["random_state"] = random_state
        # A refit on other rows cannot reuse a split plan realised on these rows.
        estimator.tmle_kwargs["split_plan"] = None
        return estimator.fit(data)

    def with_basis(self, basis: Any) -> EPEstimator:
        """A copy whose sieve family is ``basis``, with the same size rule.

        Parameters
        ----------
        basis : Basis
            The new family.

        Returns
        -------
        EPEstimator
            The copy.
        """
        estimator = copy.copy(self)
        estimator.effect_modification = replace(self.effect_modification, basis=basis)
        return estimator

    # ------------------------------------------------------------ the result

    def _result(self, fit: EPFit) -> Any:
        from ..effect_curves import EffectCurveResult

        data = fit.data
        spec = self.effect_modification
        values = modifier_frame(data, tuple(spec.modifiers.columns))
        raw = fit.curve.raw(values)
        in_sample = fit.curve.predict(values)
        label, reason = ep_status(
            scale=spec.scale,
            mode=self.mode,
            target=self.target,
            declared_learner=self.final_learner is not None,
            weighted=bool(data.is_weighted),
            clustered=data.cluster is not None,
            missing=bool(np.any(np.asarray(data.observed) < 0.5)),
            by_arm=bool(self.tmle_kwargs.get("outcome_by_arm", False)),
            arms=int(data.n_arms),
            known_mechanism=data.known_treatment is not None,
        )
        stem = (
            "conditional average treatment effect"
            if spec.scale == "difference"
            else "log conditional risk ratio"
        )
        given = ", ".join(spec.modifiers.columns)
        if self.target == "projection":
            sentence = (
                f"the projection of the {stem} given {given} onto the declared basis "
                f"({', '.join(fit.curve.terms)}), estimated by the EP learner"
            )
        elif self.final_learner is not None:
            sentence = (
                f"the {stem} given {given}, approximated within the declared final learner's class"
            )
        else:
            sentence = (
                f"the {stem} given {given}, approximated within the span of the sieve at its "
                f"chosen size ({fit.sieve.chosen} "
                f"{'term' if fit.sieve.chosen == 1 else 'terms'} per component)"
            )
        lower, upper = fit.nuisance.scaler.unscale_levels(np.array([0.0, 1.0]))
        clipped = float(np.mean(raw != in_sample)) if fit.curve.clip else 0.0
        result = EffectCurveResult(
            estimand_name="cate" if spec.scale == "difference" else "log_crr",
            modifiers=spec.modifiers,
            model=fit.curve,
            scale=spec.scale,
            in_sample=in_sample,
            target_sentence=sentence,
            inference_status="no_interval",
            status=label,
            status_reason=reason,
            lower_bound=float(lower),
            upper_bound=float(upper),
            backend=str(data.backend),
            identified_effect=self.identified_effect,
            provenance=getattr(fit.nuisance_fit, "provenance", None),
            fit=fit,
            fitted_method="ep",
            sieve=fit.sieve,
            clipped_share=clipped,
        )
        # The outer nuisance fit's warnings (positivity, convergence, weighting) are the EP
        # curve's own; the inner refits' stay on the sieve record.
        attach(result, getattr(fit.nuisance_fit, "fit_warnings", ()))
        register = getattr(self.identified_effect, "_register_curve", None)
        if register is not None:
            register(result.in_sample)
        return result


def refit_seed(base: int | None, fold: int) -> int:
    """The seed of fold ``fold``'s inner refit in the sieve-size cross-validation.

    Parameters
    ----------
    base : int or None
        The seed of the fit whose folds are cross-validated.
    fold : int
        The cross-validation fold, counted from 0.

    Returns
    -------
    int
        A seed in ``[0, 2**32 - 1]``.
    """
    entropy = [0 if base is None else int(base), 1_000_003, fold]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])
