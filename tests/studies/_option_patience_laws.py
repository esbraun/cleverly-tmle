r"""Laws, learners, fits and exact limits of ``option-patience`` (X76 part (c)).

The study measures ``CTMLE(patience=p)``: the cross-validated scoring of a candidate path
stops after ``p`` candidates whose risk is not a new minimum (Fix-8 plan sections 2.3 and 5.5).
Four streams, each one law, fitted once per declared fit on every draw:

================  ===========================================================================
stream            law and fits
================  ===========================================================================
``lc1``           the ``canonical-ctmle-selector`` binary law (three covariates) with that
                  study's fit settings: greedy, ordered and discrete, each with
                  :data:`LC1_PATIENCE` and with the whole path
``lc2_lasso``     the ``canonical-ctmle-sequences`` lasso law with that study's grid and
                  settings, with :data:`LC2_PATIENCE` and with the whole path
``lc2_learners``  the ``canonical-ctmle-sequences`` learners law with that study's declared
                  candidates, on a selection split drawn with the sample
``lc3``           the late-confounder law below: ordered (the instruments first) and greedy,
                  each with :data:`LC3_PATIENCE` and with the whole path
================  ===========================================================================

**The late-confounder law** ``LC3``.  ``W1, Z1, Z2, Z3`` are iid N(0, 1),
``A ~ Bernoulli(expit(0.8 (Z1 + Z2 + Z3) + 0.6 W1))`` and
``Y ~ Bernoulli(expit(-0.3 + 0.5 A + W1))``.  ``W1`` is a confounder and the ``Z`` are
instruments.  The outcome regression is an unpenalized logistic regression on
``(A, Z1, Z2, Z3)``: it omits ``W1``, so it is wrong, and the treatment model must carry
``W1`` for the targeted fit to be consistent.  The ordered path puts the instruments first, so a
fit with patience 2 stops before ``W1`` whenever two instruments in a row fail to lower the
risk.

**g bounds on LC3** are :data:`LC3_G_BOUNDS`, declared rather than the sample-size default.
With the full (correct) logistic mechanism, the targeted fit solves the score equation at the
truncated mechanism ``g_t``, so its limit error is

.. math::

    \psi(Q^*) - \psi_0 = E[(Q^*_1 - Q_{0,1})(1 - g_0/g_{t})]
    - E[(Q^*_0 - Q_{0,0})(1 - (1 - g_0)/(1 - g_{t}))],

whatever path led to ``Q*``.  Both factors of each term are bounded, ``|Q* - Q0| <= 1`` for a
binary outcome, so :func:`lc3_truncation_bound` bounds the whole-path limit bias by
``E|1 - g0/g_t| + E|1 - (1 - g0)/(1 - g_t)|``, a one-dimensional integral over the
treatment index ``0.8 (Z1 + Z2 + Z3) + 0.6 W1 ~ N(0, 2.28)``.  Without truncation the limit
error is exactly zero (double robustness).  The design test reads the bound in units of the
whole-path SD at n = 2,000.

Every fit is in sample with five unstratified selection folds and two inner folds,
``penalty=False``, ``max_iter=100``, ``tol=1e-10`` and the ATE as the selection estimand.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from cleverly.estimators import CTMLE
from cleverly.estimators.ctmle import CTMLEStrategy
from tests.studies import canonical_ctmle_selector as selector
from tests.studies import canonical_ctmle_sequences as sequences
from tests.studies.canonical_cvtmle import STRATIFY_FOLDS
from tests.studies.canonical_tmle import scenario_dgp, truth_for

#: The sample size of every cell.
N = 2_000
#: The selection split of every fit.
SELECTION_FOLDS = 5
SELECTION_INNER_FOLDS = 2
MAX_ITER = 100
TOL = 1e-10
#: The ``g_bounds`` of each stream: the sibling studies' on LC1 and LC2, declared on LC3.
LC1_G_BOUNDS = selector.G_BOUNDS
LC2_G_BOUNDS = sequences.G_BOUNDS
LC3_G_BOUNDS = (0.01, 0.99)
#: The patience of each stream's patient fits.  ``LC1_PATIENCE`` is plan 5.5's ``p*``, the
#: smallest p in {1, 2} whose pilot stop share on the greedy path is at least 0.5; the design
#: test reads it off the committed pilot.
LC1_PATIENCE = 1
LC2_PATIENCE = 2
LC3_PATIENCE = 2

#: The covariates of each stream, in the column order every learner sees after ``A``.
LC1_COVARIATES = ("W1", "W2", "W3")
LC3_COVARIATES = ("W1", "Z1", "Z2", "Z3")
#: ``ordering=`` of the ordered LC3 fit: the instruments first, the confounder last.
LC3_ORDERING = ("Z1", "Z2", "Z3", "W1")
#: The LC3 confounder, which the outcome regression omits.
LC3_CONFOUNDER = "W1"
#: LC3's treatment index ``INSTRUMENT (Z1 + Z2 + Z3) + CONFOUNDER_EFFECT W1``.
LC3_INSTRUMENT = 0.8
LC3_CONFOUNDER_EFFECT = 0.6
#: LC3's outcome index ``(intercept, A, W1)``.
LC3_OUTCOME = (-0.3, 0.5, 1.0)
#: The Gauss-Hermite nodes of the exact integrals.
NODES = 80

#: The scenario names of the sibling laws.
LC1_SCENARIO = "binary"
LC2_SCENARIOS = {"lc2_lasso": "binary_lasso", "lc2_learners": "binary_learners"}
STREAMS = ("lc1", "lc2_lasso", "lc2_learners", "lc3")
#: The column of an ``lc2_learners`` sample that carries its selection fold, 1 to 5.
FOLD_COLUMN = "selection_fold"


# ---------------------------------------------------------------------------- LC3


def lc3_index_sd() -> float:
    """The SD of LC3's treatment index, ``sqrt(3 * 0.8^2 + 0.6^2)``."""
    return float(np.sqrt(3.0 * LC3_INSTRUMENT**2 + LC3_CONFOUNDER_EFFECT**2))


def lc3_qbar(a: float | np.ndarray, w1: np.ndarray) -> np.ndarray:
    """``E[Y | A = a, W1]`` under LC3."""
    intercept, effect, confounder = LC3_OUTCOME
    return np.asarray(expit(intercept + effect * np.asarray(a) + confounder * w1), dtype=float)


def _hermite(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    return x, w / w.sum()


@functools.cache
def lc3_truth(nodes: int = NODES) -> dict[str, float]:
    """The exact ATE and arm means of LC3, by a Gauss-Hermite rule over ``W1``.

    Parameters
    ----------
    nodes : int
        The number of nodes.

    Returns
    -------
    dict of str to float
        ``ate``, ``ey1`` and ``ey0``.
    """
    x, w = _hermite(nodes)
    ey1 = float(np.sum(w * lc3_qbar(1.0, x)))
    ey0 = float(np.sum(w * lc3_qbar(0.0, x)))
    return {"ate": ey1 - ey0, "ey1": ey1, "ey0": ey0}


def lc3_draw(n: int, seed: int) -> pd.DataFrame:
    """One LC3 sample of ``n`` rows."""
    rng = np.random.default_rng(seed)
    w1, z1, z2, z3 = (rng.standard_normal(n) for _ in range(4))
    g = expit(LC3_INSTRUMENT * (z1 + z2 + z3) + LC3_CONFOUNDER_EFFECT * w1)
    a = rng.binomial(1, g)
    y = rng.binomial(1, lc3_qbar(a.astype(float), w1))
    return pd.DataFrame({"W1": w1, "Z1": z1, "Z2": z2, "Z3": z3, "A": a, "Y": y})


def lc3_truncation_bound(bounds: tuple[float, float] = LC3_G_BOUNDS, nodes: int = 150) -> float:
    """The bound on the whole-path limit bias of LC3 that truncation at ``bounds`` leaves.

    ``E|1 - g0/g_t| + E|1 - (1 - g0)/(1 - g_t)|`` over the treatment index, which bounds the
    limit error of any targeted fit with a binary outcome that solves the ATE score equation at
    the truncated correct mechanism (module docstring).  Zero without truncation.

    Parameters
    ----------
    bounds : tuple of float
        The truncation bounds.
    nodes : int
        The Gauss-Hermite nodes over the standardized index.

    Returns
    -------
    float
        The bound, on the outcome's scale.
    """
    x, w = _hermite(nodes)
    g = expit(lc3_index_sd() * x)
    truncated = np.clip(g, *bounds)
    return float(
        np.sum(w * (np.abs(1.0 - g / truncated) + np.abs(1.0 - (1.0 - g) / (1.0 - truncated))))
    )


def lc3_truncated_share(bounds: tuple[float, float] = LC3_G_BOUNDS, nodes: int = 150) -> float:
    """The share of the LC3 population whose correct mechanism lies outside ``bounds``."""
    x, w = _hermite(nodes)
    g = expit(lc3_index_sd() * x)
    return float(np.sum(w * ((g < bounds[0]) | (g > bounds[1]))))


# ---------------------------------------------------------------------------- learners


def logistic() -> LogisticRegression:
    """The unpenalized logistic regression every stream fits."""
    return LogisticRegression(C=1e6, max_iter=2_000, solver="lbfgs")


def lc3_outcome_learner() -> Pipeline:
    """The LC3 outcome regression: logistic on ``(A, Z1, Z2, Z3)``, without ``W1``.

    The estimator hands the outcome learner ``[A, *covariates]`` in the declared covariate
    order, so ``W1`` is column 1 (``tests/unit/test_option_patience_witness.py`` reads the
    columns back).
    """
    keep = [0, *(1 + LC3_COVARIATES.index(name) for name in ("Z1", "Z2", "Z3"))]
    return Pipeline(
        [
            ("omit_confounder", ColumnTransformer([("keep", "passthrough", keep)])),
            ("logistic", logistic()),
        ]
    )


# ---------------------------------------------------------------------------- draws


def truth(stream: str) -> dict[str, float]:
    """The exact truths of a stream's law."""
    if stream == "lc1":
        return truth_for(scenario_dgp(LC1_SCENARIO))
    if stream in LC2_SCENARIOS:
        return truth_for(sequences.scenario_law(LC2_SCENARIOS[stream]))
    if stream == "lc3":
        return lc3_truth()
    raise KeyError(stream)


def selection_folds(n: int, seed: int) -> np.ndarray:
    """The ``lc2_learners`` selection split: balanced folds 1 to 5 from the sample's seed.

    Parameters
    ----------
    n : int
        The sample size.
    seed : int
        The sample's seed.

    Returns
    -------
    numpy.ndarray
        One fold label per row.
    """
    rng = np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(1,)))
    return rng.permutation(np.arange(n) % SELECTION_FOLDS) + 1


def draw(stream: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of a stream's law.  An ``lc2_learners`` sample carries its fold column."""
    if stream == "lc1":
        frame, _ = scenario_dgp(LC1_SCENARIO).sample(n, seed=seed, backend="pandas")
        return frame
    if stream in LC2_SCENARIOS:
        frame, _ = sequences.scenario_law(LC2_SCENARIOS[stream]).sample(
            n, seed=seed, backend="pandas"
        )
        if stream == "lc2_learners":
            frame = frame.assign(**{FOLD_COLUMN: selection_folds(n, seed)})
        return frame
    if stream == "lc3":
        return lc3_draw(n, seed)
    raise KeyError(stream)


# ---------------------------------------------------------------------------- fits


@dataclass(frozen=True)
class Fit:
    """One declared fit of a stream.

    Parameters
    ----------
    stream : str
        The stream whose draws it fits.
    strategy : str
        The C-TMLE strategy.
    patience : int or None
        ``patience=``; ``None`` scores the whole path.
    options : Mapping
        The strategy's own keywords.
    """

    stream: str
    strategy: CTMLEStrategy
    patience: int | None
    options: Mapping[str, Any] = field(default_factory=dict)


def _lc1_options(strategy: str) -> dict[str, Any]:
    if strategy == "ordered":
        return {"ordering": LC1_COVARIATES}
    if strategy == "discrete":
        return {"candidates": ((), ("W1",), ("W1", "W2"), LC1_COVARIATES)}
    return {}


#: The LC1 strategies, the selector study's three.
LC1_STRATEGIES: tuple[CTMLEStrategy, ...] = ("greedy", "ordered", "discrete")

#: Every declared fit, by name.
FITS: dict[str, Fit] = {
    **{
        f"lc1_{strategy}_{kind}": Fit("lc1", strategy, patience, _lc1_options(strategy))
        for strategy in LC1_STRATEGIES
        for kind, patience in (("patience", LC1_PATIENCE), ("whole", None))
    },
    **{
        f"{stream}_{kind}": Fit(stream, "lasso" if stream == "lc2_lasso" else "learners", patience)
        for stream in LC2_SCENARIOS
        for kind, patience in (("patience", LC2_PATIENCE), ("whole", None))
    },
    "lc3_ordered_patience": Fit("lc3", "ordered", LC3_PATIENCE, {"ordering": LC3_ORDERING}),
    "lc3_ordered_whole": Fit("lc3", "ordered", None, {"ordering": LC3_ORDERING}),
    "lc3_greedy_patience": Fit("lc3", "greedy", LC3_PATIENCE),
    "lc3_greedy_whole": Fit("lc3", "greedy", None),
}

#: The pilot's extra fits: the LC1 greedy path at each patience plan 5.5 chooses ``p*`` from.
PILOT_FITS: dict[str, Fit] = {
    f"lc1_greedy_patience_{p}": Fit("lc1", "greedy", p, _lc1_options("greedy")) for p in (1, 2)
}


def _shared(g_bounds: tuple[float, float]) -> dict[str, Any]:
    return {
        "cross_fit": False,
        "selection_folds": SELECTION_FOLDS,
        "selection_inner_folds": SELECTION_INNER_FOLDS,
        "penalty": False,
        "estimands": ("ate",),
        "ctmle_estimand": "ate",
        "simultaneous": False,
        "g_bounds": g_bounds,
        "stratify_folds": STRATIFY_FOLDS,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "random_state": 0,
    }


def estimator(frame: pd.DataFrame, name: str) -> CTMLE:
    """The unfitted estimator of one declared fit on one sample.

    Parameters
    ----------
    frame : pandas.DataFrame
        The sample, which the ``lc2_learners`` candidates are fitted on.
    name : str
        A key of :data:`FITS` or :data:`PILOT_FITS`.

    Returns
    -------
    CTMLE
        The estimator, with the declared settings.
    """
    spec = FITS[name] if name in FITS else PILOT_FITS[name]
    if spec.stream == "lc1":
        return CTMLE(
            strategy=spec.strategy,
            outcome_learner=logistic(),
            treatment_learner=logistic(),
            patience=spec.patience,
            **_shared(LC1_G_BOUNDS),
            **spec.options,
        )
    if spec.stream == "lc2_lasso":
        return CTMLE(
            strategy="lasso",
            lambdas=(*sequences.study_lambdas("binary_lasso"), 0.0),
            outcome_learner=logistic(),
            treatment_learner=logistic(),
            patience=spec.patience,
            **_shared(LC2_G_BOUNDS),
        )
    if spec.stream == "lc2_learners":
        folds = frame[FOLD_COLUMN].to_numpy(dtype=int)
        shared = _shared(LC2_G_BOUNDS)
        del shared["selection_folds"], shared["stratify_folds"]
        return CTMLE(
            strategy="learners",
            treatment_candidates=sequences.propensity_arrays(frame, folds),
            selection_split=(folds - 1).tolist(),
            selection_folds=SELECTION_FOLDS,
            outcome_learner=logistic(),
            treatment_learner=logistic(),
            patience=spec.patience,
            **shared,
        )
    if spec.stream == "lc3":
        return CTMLE(
            strategy=spec.strategy,
            outcome_learner=lc3_outcome_learner(),
            treatment_learner=logistic(),
            patience=spec.patience,
            **_shared(LC3_G_BOUNDS),
            **spec.options,
        )
    raise KeyError(spec.stream)


def covariates(stream: str) -> tuple[str, ...]:
    """The adjustment set of a stream's fits."""
    if stream == "lc1":
        return LC1_COVARIATES
    if stream in LC2_SCENARIOS:
        return sequences.COVARIATES
    if stream == "lc3":
        return LC3_COVARIATES
    raise KeyError(stream)


def fit(frame: pd.DataFrame, name: str) -> Any:
    """Fit one declared fit on one sample and return the single-parameter result."""
    stream = (FITS[name] if name in FITS else PILOT_FITS[name]).stream
    return (
        estimator(frame, name)
        .fit(frame, outcome="Y", treatment="A", covariates=list(covariates(stream)))
        .single()
    )


# ---------------------------------------------------------------------------- readings


def stopped(patient: Any, whole: Any) -> bool:
    """Whether the patience counter ended the path before the whole path's end.

    A stopped path is a prefix of the whole path (``CTMLESelection.stopped_at``), so the path
    ended early exactly when it is shorter than the whole path fitted on the same sample.
    """
    return len(patient.extra["ctmle"].path) < len(whole.extra["ctmle"].path)


def excludes_confounder(result: Any) -> bool:
    """Whether a fit's reported treatment model leaves out the LC3 confounder."""
    return LC3_CONFOUNDER not in tuple(result.extra["ctmle"].selected_covariates)


def ate(result: Any) -> float:
    """The fitted ATE."""
    return float(result["ate"].psi)


#: Fits that one draw of each stream needs, in fit order.
STREAM_FITS: dict[str, tuple[str, ...]] = {
    stream: tuple(name for name, spec in FITS.items() if spec.stream == stream)
    for stream in STREAMS
}


def fit_stream(frame: pd.DataFrame, stream: str) -> dict[str, Any]:
    """Every declared fit of one draw of a stream, by name."""
    return {name: fit(frame, name) for name in STREAM_FITS[stream]}


__all__ = [
    "FITS",
    "LC1_PATIENCE",
    "LC2_PATIENCE",
    "LC3_G_BOUNDS",
    "LC3_PATIENCE",
    "STREAMS",
    "STREAM_FITS",
    "N",
    "draw",
    "estimator",
    "excludes_confounder",
    "fit",
    "fit_stream",
    "lc3_truncation_bound",
    "lc3_truth",
    "stopped",
    "truth",
]
