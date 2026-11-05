r"""Declaration and driver of the ``conditional-projection`` study (P1 plan, section 5.1).

The study measures the projection TMLE of a :class:`cleverly.ConditionalContrast` with
``model=Linear(["W1", "W3"])``: interval calibration of the three coefficients on both
scales, under a known and an estimated mechanism, three arms, weights, clusters, missing
outcomes and the assessment of a fixed external score; a slow-nuisance ladder for the
product remainder; the size and power of the heterogeneity test; and controls that must
fail.  This module holds what the declaration fixes before any run:

- the laws, as samplers with the plan's coefficients;
- the exact truths, by Gauss--Legendre quadrature over ``U(-1, 1)^3`` (Gauss--Hermite over
  the cluster effect), whose error is far below any margin (:func:`quadrature_error`);
- the cells, with their law, ``n``, ``R``, family and the pass probability the plan
  computed (:data:`CELLS`);
- the learners and the fit of one replication (:func:`fit_cell`), through the study layer.

``python -m tests.studies.conditional_projection --smoke`` fits a few replications of the
cheap cells on seeds disjoint from the declared run and prints the coverage and the cost per
fit.  The registered run, its ``StudyRecord`` entry in
:func:`tests.studies.evidence.registry.registered`, its property module and its artifacts
land together: the fast tier reads committed artifacts for every registered study.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from cleverly import CausalStudy, ConditionalContrast, PointTreatment
from cleverly.bases import Linear, Modifiers

SLUG = "conditional-projection"

#: The declared run's seed, and the smoke run's: the two never share a replication.
SEED = 20261105
SMOKE_SEED = 77_000_000

#: The modifiers and the basis of every cell but ``hetero_trial``'s, which uses the same.
MODIFIERS = Modifiers(["W1", "W3"])
BASIS = Linear(["W1", "W3"])
TERMS = ("(intercept)", "W1", "W3")

#: Gauss--Legendre nodes per axis of ``W``, and Gauss--Hermite nodes of a cluster effect.
QUADRATURE_NODES = 48
HERMITE_NODES = 40

#: The slow-nuisance ladder's offset ``c n^{-1/3} h(W)``.
SLOW_CONSTANT = 5.0


# ----------------------------------------------------------------- the laws


def blip(w: np.ndarray) -> np.ndarray:
    """The arm-1 logit increment ``0.3 + 0.5 W1 + 0.4 sin(pi W3)``."""
    return 0.3 + 0.5 * w[..., 0] + 0.4 * np.sin(np.pi * w[..., 2])


def outcome_logit(w: np.ndarray, a: Any) -> np.ndarray:
    """``logit mu(a, W) = -0.3 + 0.6 W1 - 0.4 W2 + a (0.3 + 0.5 W1 + 0.4 sin(pi W3))``."""
    return -0.3 + 0.6 * w[..., 0] - 0.4 * w[..., 1] + np.asarray(a) * blip(w)


def propensity(w: np.ndarray) -> np.ndarray:
    """``logit pi = 0.4 W1 - 0.5 W2 + 0.3 W3``, in about ``[0.3, 0.7]``."""
    return expit(0.4 * w[..., 0] - 0.5 * w[..., 1] + 0.3 * w[..., 2])


def multi_propensity(w: np.ndarray) -> np.ndarray:
    """The three-arm mechanism: a softmax of ``(0, 0.3 W1 - 0.2 W2, 0.4 W1 + 0.3 W3)``."""
    scores = np.stack(
        [
            np.zeros(w.shape[:-1]),
            0.3 * w[..., 0] - 0.2 * w[..., 1],
            0.4 * w[..., 0] + 0.3 * w[..., 2],
        ],
        axis=-1,
    )
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


def observation(w: np.ndarray, a: Any) -> np.ndarray:
    """``logit Pi(a, W) = 1.5 + 0.6 W1 - 0.5 a``, in ``[0.60, 0.89]``."""
    return expit(1.5 + 0.6 * w[..., 0] - 0.5 * np.asarray(a))


def sampling_weight(w: np.ndarray) -> np.ndarray:
    """``w = 0.5 + 1.5 expit(2 W1)``, the ``weighted`` cell's sampling weights."""
    return 0.5 + 1.5 * expit(2.0 * w[..., 0])


def score(w: np.ndarray) -> np.ndarray:
    """The ``assess`` cell's declared, non-optimal score ``s(V) = 0.2 + 0.3 W1``."""
    return 0.2 + 0.3 * w[..., 0]


def hetero_mean(w: np.ndarray, a: Any, slope: float) -> np.ndarray:
    """``mu(a, W) = 0.4 + 0.1 W2 + a (0.2 + s W1)`` on the risk scale."""
    return 0.4 + 0.1 * w[..., 1] + np.asarray(a) * (0.2 + slope * w[..., 0])


def slow_offset(w: np.ndarray) -> np.ndarray:
    """``h(W) = W1^2 - 1/3 + W1 W3``, which breaks the level sets of both true indices."""
    return w[..., 0] ** 2 - 1.0 / 3.0 + w[..., 0] * w[..., 2]


def _frame(w: np.ndarray, a: np.ndarray, y: np.ndarray, **extra: Any) -> pd.DataFrame:
    return pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1], "W3": w[:, 2], **extra})


def sample(law: str, n: int, rng: np.random.Generator, *, slope: float = 0.0) -> pd.DataFrame:
    """Draw ``n`` rows of ``law``.

    Parameters
    ----------
    law : str
        A key of :data:`LAWS`.
    n : int
        Rows.
    rng : numpy.random.Generator
        The replication's generator.
    slope : float
        The ``hetero_trial`` slope ``s``.

    Returns
    -------
    pandas.DataFrame
        ``Y``, ``A``, ``W1``, ``W2``, ``W3`` and the law's own columns.
    """
    if law == "clustered":
        clusters = n // 5
        cluster = np.repeat(np.arange(clusters), 5)
        w = rng.uniform(-1.0, 1.0, (clusters * 5, 3))
        w[:, 0] = np.repeat(rng.uniform(-1.0, 1.0, clusters), 5)
        a = np.repeat(rng.binomial(1, expit(0.6 * w[::5, 0])), 5)
        u = np.repeat(rng.normal(0.0, 1.0, clusters), 5)
        y = rng.binomial(1, expit(outcome_logit(w, a) + u))
        return _frame(w, a, y, cluster=cluster)
    w = rng.uniform(-1.0, 1.0, (n, 3))
    if law == "hetero_trial":
        a = rng.binomial(1, 0.5, n)
        y = rng.binomial(1, hetero_mean(w, a, slope))
        return _frame(w, a, y, p0=np.full(n, 0.5), p1=np.full(n, 0.5))
    if law == "multi_arm":
        probabilities = multi_propensity(w)
        a = (rng.uniform(size=(n, 1)) > np.cumsum(probabilities, axis=1)).sum(axis=1)
        y = rng.binomial(1, expit(outcome_logit(w, a / 2.0)))
        return _frame(w, a, y)
    if law == "trial":
        a = rng.binomial(1, 0.5, n)
    elif law == "noweight_law":
        a = rng.binomial(1, expit(2.5 * w[:, 0]))
    else:
        a = rng.binomial(1, propensity(w))
    y = rng.binomial(1, expit(outcome_logit(w, a))).astype(float)
    extra: dict[str, Any] = {}
    if law == "trial":
        extra = {"p0": np.full(n, 0.5), "p1": np.full(n, 0.5)}
    if law == "missing":
        observed = rng.binomial(1, observation(w, a)).astype(float)
        y = np.where(observed == 1.0, y, np.nan)
        extra = {"Delta": observed}
    if law == "weighted":
        extra = {"weight": sampling_weight(w)}
    if law == "assess":
        extra = {"S": score(w)}
    return _frame(w, a, y, **extra)


#: Every law of the study.
LAWS: tuple[str, ...] = (
    "trial",
    "observational",
    "slow_both",
    "noweight_law",
    "crr",
    "multi_arm",
    "weighted",
    "clustered",
    "missing",
    "assess",
    "hetero_trial",
)


# --------------------------------------------------------------- the truths


def _grid() -> tuple[np.ndarray, np.ndarray]:
    """``U(-1, 1)^3`` quadrature nodes ``(m, 3)`` and their probability weights ``(m,)``."""
    nodes, weights = leggauss(QUADRATURE_NODES)
    mesh = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    mass = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1) / 8.0
    return mesh, mass


def arm_means(law: str, w: np.ndarray, *, slope: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``(mu_t(W), mu_r(W))`` of the law's contrasted pair, marginal over a cluster effect."""
    if law == "hetero_trial":
        return hetero_mean(w, 1, slope), hetero_mean(w, 0, slope)
    if law == "clustered":
        nodes, weights = hermegauss(HERMITE_NODES)
        weights = weights / weights.sum()

        def marginal(a: int) -> np.ndarray:
            eta = outcome_logit(w, a)[:, None] + nodes[None, :]
            return expit(eta) @ weights

        return marginal(1), marginal(0)
    return expit(outcome_logit(w, 1)), expit(outcome_logit(w, 0))


def projection(
    phi: np.ndarray, mass: np.ndarray, q_t: np.ndarray, q_r: np.ndarray, scale: str
) -> np.ndarray:
    """The projection of the contrast onto ``phi`` under the measure ``mass``."""
    gram = phi.T @ (mass[:, None] * phi)
    if scale == "difference":
        return np.linalg.solve(gram, phi.T @ (mass * (q_t - q_r)))
    beta = np.zeros(phi.shape[1])
    for _ in range(100):
        e = expit(phi @ beta)
        equation = phi.T @ (mass * ((q_t + q_r) * e - q_t))
        jacobian = phi.T @ ((mass * (q_t + q_r) * e * (1.0 - e))[:, None] * phi)
        step = np.linalg.solve(jacobian, equation)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-15:
            break
    return beta


def truth(law: str, scale: str = "difference", *, slope: float = 0.0) -> dict[str, float]:
    """The exact coefficients of a cell, keyed by the name the fit reports.

    The ``weighted`` truth is the projection of the weighted population, whose ``W`` law is
    tilted by the sampling weight; its arm regressions are unchanged because the weight is
    a function of ``W``.  The ``assess`` truth adds the score's risk difference against the
    constant ATE and its ``(1, s)`` calibration coefficients.

    Parameters
    ----------
    law : str
        A key of :data:`LAWS`.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    slope : float
        The ``hetero_trial`` slope.

    Returns
    -------
    dict of str to float
        The truth of each reported parameter.
    """
    w, mass = _grid()
    if law == "weighted":
        mass = mass * sampling_weight(w)
        mass = mass / mass.sum()
    q_t, q_r = arm_means(law, w, slope=slope)
    phi = np.column_stack([np.ones(len(w)), w[:, 0], w[:, 2]])
    stem = "cate_projection" if scale == "difference" else "log_crr_projection"
    beta = projection(phi, mass, q_t, q_r, scale)
    out = {f"{stem}[{term}]": float(value) for term, value in zip(TERMS, beta, strict=True)}
    if law == "assess":
        s = score(w)
        tau = q_t - q_r
        ate = float(mass @ tau)
        out["risk_difference"] = float(mass @ (s**2 - 2.0 * s * tau) + ate**2)
        calibration = projection(np.column_stack([np.ones(len(w)), s]), mass, q_t, q_r, scale)
        out["calibration[(intercept)]"] = float(calibration[0])
        out["calibration[S]"] = float(calibration[1])
    return out


def quadrature_error(law: str = "observational", scale: str = "difference") -> float:
    """The largest change in a truth when the nodes per axis drop from 48 to 32."""
    global QUADRATURE_NODES
    fine = truth(law, scale)
    saved = QUADRATURE_NODES
    QUADRATURE_NODES = 32
    try:
        coarse = truth(law, scale)
    finally:
        QUADRATURE_NODES = saved
    return max(abs(fine[name] - coarse[name]) for name in fine)


# ---------------------------------------------------------------- the cells


@dataclass(frozen=True)
class Cell:
    """One declared cell: its law, size, replications, family, and planned pass probability.

    Parameters
    ----------
    name : str
        The cell key.
    law : str
        A key of :data:`LAWS`.
    n : int
        Rows per replication.
    replicates : int
        Replications in the declared run.
    family : str
        The property family the cell reads.
    claim : str
        What a pass shows, or the control's must-fail statement.
    pass_probability : str
        The plan's computed pass probability (P1 plan, 5.0 and 5.1).
    scale : str
        The contrast's scale.
    slope : float
        The ``hetero_trial`` slope.
    """

    name: str
    law: str
    n: int
    replicates: int
    family: str
    claim: str
    pass_probability: str
    scale: str = "difference"
    slope: float = 0.0


CALIBRATION = "interval_calibration"

#: The cells of section 5.1, in its order.
CELLS: tuple[Cell, ...] = (
    Cell(
        "trial_2k",
        "trial",
        2_000,
        2_400,
        CALIBRATION,
        "CATE projection, known mechanism",
        "0.95 to 0.975",
    ),
    Cell("obs_1k", "observational", 1_000, 2_400, CALIBRATION, "CATE projection", "0.95 to 0.975"),
    Cell("obs_4k", "observational", 4_000, 2_400, CALIBRATION, "CATE projection", "0.95 to 0.975"),
    Cell(
        "crr_1k",
        "crr",
        1_000,
        2_400,
        CALIBRATION,
        "log-CRR projection",
        "0.95 to 0.975",
        "log_ratio",
    ),
    Cell(
        "crr_4k",
        "crr",
        4_000,
        2_400,
        CALIBRATION,
        "log-CRR projection",
        "0.95 to 0.975",
        "log_ratio",
    ),
    Cell(
        "multi_4k",
        "multi_arm",
        4_000,
        2_400,
        CALIBRATION,
        "pair (2, 0) of three arms",
        "0.95 to 0.975",
    ),
    Cell(
        "weighted_sl_2k",
        "weighted",
        2_000,
        2_400,
        CALIBRATION,
        "weights, ensemble learner",
        "0.95 to 0.975",
    ),
    Cell(
        "clustered_4k",
        "clustered",
        4_000,
        2_400,
        CALIBRATION,
        "the cluster as the unit",
        "0.95 to 0.975",
    ),
    Cell(
        "obs_missing_4k",
        "missing",
        4_000,
        2_400,
        CALIBRATION,
        "missing outcomes, difference",
        "0.95 to 0.975",
    ),
    Cell(
        "obs_missing_crr_4k",
        "missing",
        4_000,
        2_400,
        CALIBRATION,
        "missing outcomes, log ratio",
        "0.95 to 0.975",
        "log_ratio",
    ),
    Cell(
        "assess_score_2k",
        "assess",
        2_000,
        2_400,
        CALIBRATION,
        "risk difference and (1, s) calibration",
        "0.95 to 0.975",
    ),
    Cell(
        "slow_1k",
        "slow_both",
        1_000,
        2_400,
        "bias_ladder",
        "standardized bias inside +-0.25, and calibration",
        "0.998; 0.953",
    ),
    Cell(
        "slow_4k",
        "slow_both",
        4_000,
        2_400,
        "bias_ladder",
        "standardized bias inside +-0.25, and calibration",
        "1.000; 0.953",
    ),
    Cell(
        "slow_16k",
        "slow_both",
        16_000,
        2_400,
        "bias_ladder",
        "standardized bias inside +-0.25, and calibration",
        "1.000; 0.953",
    ),
    Cell(
        "null_2k", "hetero_trial", 2_000, 2_400, "type_i", "heterogeneity test size, df 2", "1.000"
    ),
    Cell(
        "power_4k",
        "hetero_trial",
        4_000,
        1_000,
        "power",
        "heterogeneity test power 0.932",
        "1.000",
        slope=0.10,
    ),
    Cell(
        "t_learner",
        "slow_both",
        4_000,
        500,
        "control",
        "beta = 0 must fail the bias bound",
        "P(fail) 1.000",
    ),
    Cell(
        "ipw_1k",
        "slow_both",
        1_000,
        500,
        "control",
        "IPW projection must fail the bias bound",
        "P(fail) 1.000",
    ),
    Cell(
        "ipw_4k",
        "slow_both",
        4_000,
        500,
        "control",
        "IPW projection must fail the bias bound",
        "P(fail) 1.000",
    ),
    Cell(
        "ipw_16k",
        "slow_both",
        16_000,
        500,
        "control",
        "IPW projection must fail the bias bound",
        "P(fail) 1.000",
    ),
    Cell(
        "iid_se",
        "clustered",
        4_000,
        2_400,
        "control",
        "the iid SE must fail the SE-ratio band",
        "P(fail) 1.000",
    ),
    Cell(
        "no_weight",
        "noweight_law",
        4_000,
        500,
        "control",
        "unit weights must fail the coverage floor",
        "P(fail) 1.000",
    ),
)

CELL_BY_NAME: Mapping[str, Cell] = {cell.name: cell for cell in CELLS}

#: The cells a smoke run fits: GLM learners, no ensemble, no 16k rung.
SMOKE_CELLS: tuple[str, ...] = (
    "trial_2k",
    "obs_1k",
    "crr_1k",
    "obs_missing_4k",
    "assess_score_2k",
    "null_2k",
)


# -------------------------------------------------------------- the learners


def spline_glm() -> Any:
    """The spline-GLM candidate of section 5.0: cubic splines, then a logistic GLM."""
    return make_pipeline(SplineTransformer(n_knots=5, degree=3), LogisticRegression(max_iter=2_000))


class SlowOffset(ClassifierMixin, BaseEstimator):
    """A declared slow learner: the correct GLM, then a logit offset ``c n^{-1/3} offset(W)``.

    ``target`` names which nuisance it fits, so the offset is ``h(W)`` for ``pi`` and
    ``h(W) + W1`` for the arm-1 outcome regression; the arm-0 regression carries none.  Its
    error is of exact order ``n^{-1/3}``, so the product of the two is ``n^{-2/3}``.
    """

    def __init__(
        self, target: str = "treatment", n: int = 1_000, columns: tuple[int, ...] = (0, 1, 2)
    ) -> None:
        self.target = target
        self.n = n
        self.columns = columns

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SlowOffset:
        self.model_ = clone(LogisticRegression(max_iter=2_000)).fit(
            np.asarray(X, dtype=float), y, sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def _offset(self, X: np.ndarray) -> np.ndarray:
        w = X[:, list(self.columns)]
        scale = SLOW_CONSTANT * float(self.n) ** (-1.0 / 3.0)
        if self.target == "treatment":
            return scale * slow_offset(w)
        arm = X[:, 0]
        return arm * scale * (slow_offset(w) + w[:, 0])

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p = np.clip(self.model_.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        p = expit(logit(p) + self._offset(X))
        return np.column_stack([1.0 - p, p])


def learners(cell: Cell) -> dict[str, Any]:
    """The nuisance learners and fit settings of ``cell``."""
    if cell.law == "slow_both":
        return {
            "outcome_learner": SlowOffset("outcome", cell.n, columns=(1, 2, 3)),
            "treatment_learner": SlowOffset("treatment", cell.n),
        }
    settings: dict[str, Any] = {
        "outcome_learner": spline_glm(),
        "treatment_learner": LogisticRegression(max_iter=2_000),
        "outcome_by_arm": True,
    }
    if cell.law == "missing":
        settings["missingness_learner"] = LogisticRegression(max_iter=2_000)
    return settings


def design(cell: Cell) -> PointTreatment:
    """The study design of ``cell``'s law."""
    roles: dict[str, Any] = {}
    if cell.law in ("trial", "hetero_trial"):
        roles["treatment_probabilities"] = {0: "p0", 1: "p1"}
    if cell.law == "missing":
        roles["missingness"] = "Delta"
    if cell.law == "weighted":
        roles["weights"] = "weight"
    if cell.law == "clustered":
        roles["cluster"] = "cluster"
    return PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), **roles)


def estimand(cell: Cell) -> ConditionalContrast:
    """The estimand of ``cell``: ``Linear(W1, W3)`` on the cell's scale and pair."""
    pair = {"treatment": 2, "reference": 0} if cell.law == "multi_arm" else {}
    return ConditionalContrast(MODIFIERS, scale=cell.scale, model=BASIS, **pair)  # type: ignore[arg-type]


def fit_cell(cell: Cell, frame: pd.DataFrame, *, random_state: int = 0) -> dict[str, Any]:
    """Fit one replication of ``cell`` and return its estimates and standard errors.

    Parameters
    ----------
    cell : Cell
        The cell.
    frame : pandas.DataFrame
        One replication drawn by :func:`sample`.
    random_state : int
        The fold seed.

    Returns
    -------
    dict
        ``{name: (estimate, std_error)}`` for each reported parameter, and for a
        heterogeneity cell ``"p_value"``.
    """
    study = CausalStudy(frame, design=design(cell))
    effect = study.identify(estimand(cell))
    settings = {**learners(cell), "random_state": random_state}
    if cell.law == "assess":
        assessment = effect.assess_score(
            "S", computed_from=["W1"], score_kind="external", **settings
        )
        out = {name: (e.psi, e.std_error) for name, e in assessment.estimates.items()}
    else:
        out = {}
    fit = effect.estimate(**settings)
    out.update({name: (e.psi, e.std_error) for name, e in fit.estimates.items()})
    if cell.law == "hetero_trial":
        out["p_value"] = fit.heterogeneity_test().p_value
    return out


def smoke(
    cells: tuple[str, ...] = SMOKE_CELLS, replicates: int = 3, *, n: int | None = None
) -> pd.DataFrame:
    """Fit a few replications of each cell on the smoke seeds; one row per parameter.

    Parameters
    ----------
    cells : tuple of str
        Cell names.
    replicates : int
        Replications per cell.
    n : int or None
        Rows per replication, in place of the cell's own.

    Returns
    -------
    pandas.DataFrame
        The estimate, its standard error, the truth, whether the 95% interval covers it,
        and the seconds per fit.
    """
    rows = []
    for name in cells:
        cell = CELL_BY_NAME[name]
        truths = truth(cell.law, cell.scale, slope=cell.slope)
        for replicate in range(replicates):
            rng = np.random.default_rng([SMOKE_SEED, CELLS.index(cell), replicate])
            frame = sample(cell.law, cell.n if n is None else n, rng, slope=cell.slope)
            start = time.perf_counter()
            estimates = fit_cell(cell, frame, random_state=replicate)
            seconds = time.perf_counter() - start
            for parameter, value in estimates.items():
                if parameter == "p_value":
                    rows.append(
                        {
                            "cell": name,
                            "replicate": replicate,
                            "parameter": parameter,
                            "estimate": value,
                            "seconds": seconds,
                        }
                    )
                    continue
                psi, se = value
                target = truths.get(parameter)
                rows.append(
                    {
                        "cell": name,
                        "replicate": replicate,
                        "parameter": parameter,
                        "estimate": psi,
                        "std_error": se,
                        "truth": target,
                        "covers": None
                        if target is None
                        else bool(abs(psi - target) <= 1.959964 * se),
                        "seconds": seconds,
                    }
                )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """Print a smoke run; the declared run is not started from here."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--cells", nargs="*", default=list(SMOKE_CELLS))
    arguments = parser.parse_args(argv)
    table = smoke(tuple(arguments.cells), arguments.replicates, n=arguments.n)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table)
        print(table.groupby("cell")[["covers", "seconds"]].mean(numeric_only=False))
    return 0


#: The declared numbers a reader checks against section 5.1 of the P1 plan.
DECLARATION: Mapping[str, Any] = {
    "slug": SLUG,
    "seed": SEED,
    "basis": TERMS,
    "cells": {cell.name: (cell.law, cell.n, cell.replicates, cell.family) for cell in CELLS},
    "comparator": None,
    "comparator_note": (
        "No comparator is run. The identity-MSM cross-check is a unit test, and hte3 has no "
        "projection estimator."
    ),
}

FitFunction = Callable[[Cell, pd.DataFrame], dict[str, Any]]

if __name__ == "__main__":  # pragma: no cover - the smoke entry point
    raise SystemExit(main())
