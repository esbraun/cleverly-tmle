r"""A longitudinal process with a known truth, and time-varying confounding in it.

The point of the process is the node :math:`L_2`: it is caused by :math:`A_1` and it
causes both :math:`A_2` and :math:`Y`.  A point-treatment analysis has no right answer
available -- adjusting for :math:`L_2` blocks the part of :math:`A_1`'s effect that runs
through it, and not adjusting for it leaves :math:`A_2` confounded -- so this is the
structure that separates a longitudinal estimator from a sequence of cross-sectional
ones.  ``tests/e2e/test_ltmle.py`` checks that separation numerically rather than
asserting it here.

The truth is computed by **quadrature, not simulation**.  Under an intervention the
treatment and censoring mechanisms drop out entirely, so what is left is an expectation
of the outcome regression over three independent standard normals, which
Gauss--Hermite evaluates to machine precision.  A Monte Carlo truth would carry an error
of its own into every coverage assertion built on it.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np

from .._typing import FloatArray
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = [
    "RULE_LABEL",
    "longitudinal_rule_truth",
    "longitudinal_truth",
    "make_longitudinal",
    "rule_arm_at_node_two",
]

#: The dynamic regimen :func:`make_longitudinal` ships a truth for: treat everybody at
#: the first node, then keep treating only those whose biomarker rose --
#: :math:`d_2 = 1\\{L_2 > 0\\}`.  "Start everyone, continue the responders", which is the
#: shape most dynamic rules in practice have.
#:
#: A *canonical* rule rather than one the caller passes, because a coverage study looks
#: its truth up by the name a fit reports, so the process and the study have to mean the
#: same regimen by it.
#:
#: Its truth is ``0.738``, and the threshold was chosen with one coincidence deliberately
#: avoided: ``0.5`` is ``sequential._FILLER``, the value a prediction leaking from a
#: censored row would carry, so a parameter whose truth sat there would let that bug pass.
RULE_LABEL = "treat then continue if l2 positive"


def rule_arm_at_node_two(l2: Any) -> Any:
    """``d_2 = 1{L_2 > 0}``, the second node of the regimen :data:`RULE_LABEL` names.

    Shared between the quadrature and the fit so the two cannot drift apart on the
    *threshold*, which is arithmetic.  What is deliberately not shared is the plumbing on
    either side: the quadrature applies this to a grid of latent draws and a fit applies
    it to a column of a dataframe, so the surrounding code is written twice and only the
    number in the middle is written once.
    """
    return (np.asarray(l2) > 0.0).astype(float)


#: Coefficients of the outcome regression, ``logit P(Y = 1 | A1, A2, L2, W)``.
_Y = {"intercept": -0.4, "a1": 0.5, "a2": 0.8, "l2": 0.4, "w1": 0.3, "w2": -0.2, "kink": 0.5}

#: ``L2 = l2_w1 * W1 + l2_a1 * A1 + noise``, the time-varying confounder ``A1`` moves.
_L2 = {"w1": 0.6, "a1": 0.9}


def _outcome_probability(
    w1: FloatArray,
    w2: FloatArray,
    l2: FloatArray,
    a1: FloatArray | float,
    a2: FloatArray | float,
) -> FloatArray:
    """``P(Y = 1 | ...)``, at a regimen's arms or at each unit's own.

    The ``tanh`` term keeps the regression off a linear index, so a ``glm`` nuisance
    learner here is misspecified rather than accidentally exact.
    """
    index = (
        _Y["intercept"]
        + _Y["a1"] * a1
        + _Y["a2"] * a2
        + _Y["l2"] * l2
        + _Y["w1"] * w1
        + _Y["w2"] * w2
        + _Y["kink"] * np.tanh(l2)
    )
    return expit(index)


@cache
def longitudinal_truth(a1: float, a2: float, nodes: int = 48) -> float:
    r"""``E[Y_{(a1, a2)}]``, the mean outcome had everyone followed that regimen.

    Under the intervention the mechanism is gone and :math:`L_2` is
    :math:`N(0.6 W_1 + 0.9 a_1, 1)`, so the parameter is an expectation of the outcome
    regression over three independent standard normals -- :math:`W_1`, :math:`W_2` and
    :math:`L_2`'s own noise.  A product Gauss--Hermite rule with ``nodes`` points per
    dimension evaluates it to well under ``1e-10``, which is what makes it usable as a
    *truth* in a coverage study rather than as a second estimate.
    """
    return _quadrature(a1, a2, nodes)


def longitudinal_rule_truth(
    a1: float,
    rule2: Callable[[Any], Any],
    *,
    split: float = 0.0,
    nodes: int = 48,
    panel: int = 160,
) -> float:
    r"""``E[Y_d]`` where the second node is a **rule** :math:`d_2(L_2)`, not a constant.

    ``rule2`` must be a step function of :math:`L_2` with its single jump at ``split``;
    it is handed an array and must return one arm per entry.

    **Why this is not** :func:`longitudinal_truth` **with a rule passed in.**  A product
    Gauss--Hermite rule converges spectrally on a smooth integrand and *algebraically* on
    a discontinuous one, and an indicator of :math:`L_2` is discontinuous in the very
    dimension one of those axes runs over.  Substituting a rule into the existing routine
    gives an answer that moves by ``1.7e-3`` between 48 and 64 nodes -- worse than the
    500,000-draw Monte Carlo it exists to avoid, and useless as a *truth* for a coverage
    study whose standard errors are near ``0.02``.

    So the :math:`L_2` axis is integrated as two Gauss--Legendre panels meeting at the
    jump, over :math:`\int \phi(l_2 - \mu)\, g(l_2, d_2)\, dl_2` with
    :math:`\mu = 0.6 W_1 + 0.9 a_1`.  Each panel's integrand is smooth, the endpoints are
    fixed rather than functions of :math:`\mu` -- the density carries :math:`\mu`, not the
    limits -- and the arm is constant *within* a panel, which is what removes the
    discontinuity rather than merely resolving it.  ``W_1`` and ``W_2`` stay
    Gauss--Hermite, since nothing here is discontinuous in them.

    The first node is a constant.  A rule there would put a second discontinuity on the
    ``W_1`` axis, needing the same treatment again; the exact law in
    ``tests/discrete_law_longitudinal.py`` carries the first-node case instead, where it
    is checked against a Gateaux derivative rather than an integral.
    """
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = points.reshape(-1, 1, 1)
    w2 = points.reshape(1, -1, 1)
    gauss = weights.reshape(-1, 1, 1) * weights.reshape(1, -1, 1)

    mean = _L2["w1"] * w1 + _L2["a1"] * a1
    # Wide enough that the normal density is negligible past it for every node of the
    # ``W1`` rule: |mean| stays under 8 even at the outermost node of a 64-point rule.
    edge = 20.0
    abscissa, quadrature = np.polynomial.legendre.leggauss(panel)
    total = 0.0
    for lower, upper in ((-edge, split), (split, edge)):
        half = 0.5 * (upper - lower)
        grid = (half * abscissa + 0.5 * (upper + lower)).reshape(1, 1, -1)
        density = np.exp(-0.5 * (grid - mean) ** 2) / np.sqrt(2.0 * np.pi)
        # Constant across the panel by construction, which is the whole point: the arm is
        # read once at an interior point rather than resolved node by node.
        arm = float(np.asarray(rule2(np.array([0.5 * (lower + upper)]))).reshape(-1)[0])
        integrand = density * _outcome_probability(w1, w2, grid, a1, arm)
        over_l2 = half * np.sum(integrand * quadrature.reshape(1, 1, -1), axis=2)
        total += float(np.sum(gauss[:, :, 0] * over_l2))
    return total


def _quadrature(a1: float, a2: float, nodes: int) -> float:
    """The three-dimensional Gauss--Hermite rule a *static* plan's truth is computed by.

    Under the intervention the treatment and censoring mechanisms drop out, so all that
    is left is an expectation of the outcome regression over :math:`W_1`, :math:`W_2` and
    :math:`L_2`'s own noise, every one of which the integrand is smooth in.
    """
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = points.reshape(-1, 1, 1)
    w2 = points.reshape(1, -1, 1)
    noise = points.reshape(1, 1, -1)
    mass = weights.reshape(-1, 1, 1) * weights.reshape(1, -1, 1) * weights.reshape(1, 1, -1)
    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + noise
    return float(np.sum(mass * _outcome_probability(w1, w2, l2, a1, a2)))


def make_longitudinal(
    n: int = 2000,
    *,
    seed: int | np.random.Generator | None = None,
    censoring: bool = True,
    cluster_size: int | None = None,
    backend: str = "pandas",
) -> tuple[Any, dict[str, float]]:
    """Two time points, a binary outcome and (optionally) monotone censoring.

    Returns ``(frame, truth)``.  The frame is wide -- a row per unit, a column per node
    -- with columns ``W1``, ``W2``, ``A1``, ``C1``, ``L2``, ``A2``, ``C2``, ``Y`` in time
    order, and the nodes after a unit's censoring time set to ``nan``, which is the form
    :class:`~cleverly.longitudinal.LongitudinalData` requires.

    ``truth`` holds ``ey_regimen[...]`` for the four static regimens, for the dynamic
    regimen :data:`RULE_LABEL`, and the contrasts of ``always`` and of that rule against
    ``never`` -- under the labels a fit reports them by, so a test can look up the truth
    with the name it read off the result.  A rule is in there because a coverage study
    keys into ``truth`` by reported name and so cannot supply its own.

    ``cluster_size`` adds an ``id`` column and, with it, an *unobserved* effect shared
    within each cluster that moves both treatment decisions and the outcome -- so the
    influence curves are correlated within a cluster and ignoring ``id=`` understates the
    standard error.  The same construction as
    :func:`~cleverly.datasets.make_clustered`, and for the same reason: an ``id`` column
    over independent rows makes a cluster-robust variance *equal* the plain one, which
    tests nothing.  The counterfactual means are unchanged -- the shared effect is
    marginalised over and enters neither ``L2`` nor the outcome regression's form -- so
    ``truth`` still holds.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)

    if cluster_size is None:
        ids = None
        shared = np.zeros(n)
    else:
        ids = np.arange(n) // cluster_size
        # Drawn per cluster and repeated, and deliberately not among the covariates.
        shared = rng.standard_normal(int(ids.max()) + 1)[ids]

    a1 = rng.binomial(1, expit(0.3 * w1 - 0.4 * w2 + 0.8 * shared)).astype(float)
    c1 = (
        rng.binomial(1, expit(2.2 + 0.3 * w1 - 0.3 * a1)).astype(float) if censoring else np.ones(n)
    )
    alive1 = c1 == 1.0

    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + rng.standard_normal(n)
    a2 = rng.binomial(1, expit(0.5 * l2 + 0.6 * a1 - 0.2 * w2 + 0.8 * shared)).astype(float)
    c2 = rng.binomial(1, expit(2.4 + 0.2 * l2)).astype(float) if censoring else np.ones(n)
    alive2 = alive1 & (c2 == 1.0)

    probability = _outcome_probability(w1, w2, l2, a1, a2)
    if cluster_size is not None:
        # Tilt the outcome by the same shared effect, so the residual -- and with it the
        # influence curve -- carries the within-cluster correlation.
        probability = expit(np.log(probability / (1.0 - probability)) + 0.8 * shared)
    y = rng.binomial(1, probability).astype(float)

    payload = {
        "W1": w1,
        "W2": w2,
        "A1": a1,
        "C1": c1,
        "L2": np.where(alive1, l2, np.nan),
        "A2": np.where(alive1, a2, np.nan),
        "C2": np.where(alive1, c2, np.nan),
        "Y": np.where(alive2, y, np.nan),
    }
    if not censoring:
        del payload["C1"]
        del payload["C2"]
    if ids is not None:
        payload["id"] = ids.astype(float)

    truth = {
        f"ey_regimen[{label}]": longitudinal_truth(plan[0], plan[1])
        for label, plan in (
            ("always", (1.0, 1.0)),
            ("never", (0.0, 0.0)),
            ("early", (1.0, 0.0)),
            ("late", (0.0, 1.0)),
        )
    }
    truth["ate_regimen[always vs never]"] = truth["ey_regimen[always]"] - truth["ey_regimen[never]"]
    truth[f"ey_regimen[{RULE_LABEL}]"] = longitudinal_rule_truth(1.0, rule_arm_at_node_two)
    truth[f"ate_regimen[{RULE_LABEL} vs never]"] = (
        truth[f"ey_regimen[{RULE_LABEL}]"] - truth["ey_regimen[never]"]
    )
    return frame_from_dict(payload, backend=backend), truth
