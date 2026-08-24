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

from .._typing import Backend, FloatArray
from ..utils.bounds import expit
from ..utils.frames import as_frame, column_array, frame_from_dict

__all__ = [
    "RULE_LABEL",
    "competing_truth",
    "longitudinal_rule_truth",
    "longitudinal_truth",
    "make_longitudinal",
    "make_longitudinal_competing",
    "make_longitudinal_survival",
    "make_longitudinal_weighted",
    "rule_arm_at_node_two",
    "survival_truth",
]

#: What separates a regimen from the horizon it is reported at, mirrored from
#: :data:`cleverly.longitudinal.estimator.HORIZON_INFIX` rather than imported, so a
#: dataset does not depend on an estimator: the two agreeing is exactly what a coverage
#: study needs, and ``test_datasets_longitudinal`` checks it against a real fit.
_HORIZON_INFIX = " @ t="

#: The dynamic regimen :func:`make_longitudinal` ships a truth for: treat everybody at
#: the first node, then keep treating only those whose biomarker rose --
#: :math:`d_2 = 1\\{L_2 > 0\\}`.  "Start everyone, continue the responders", which is the
#: shape most dynamic rules in practice have.
#:
#: A *canonical* rule rather than one the caller passes, because a coverage study looks
#: its truth up by the name a fit reports, so the process and the study have to mean the
#: same regimen by it.
#:
#: Its truth is ``0.740``, and one coincidence was deliberately avoided in getting there.
#: ``0.5`` is ``sequential._FILLER``, the value a prediction leaking from a censored row
#: carries, so a parameter whose truth sat there would let that bug pass -- and the first
#: draft's did, to twelve decimal places.  What was changed to move it was the **first
#: node's arm**, from ``0`` to ``1``, not the threshold: ``d_2`` still splits at ``0``,
#: and ``longitudinal_rule_truth(0.0, rule_arm_at_node_two)`` still comes to ``0.5``.
#: So a rule with ``d_1 = 0`` is not unusable -- ``tests/e2e/test_ltmle.py`` uses one,
#: because it puts the rule furthest from both constants -- it is unusable as a *truth*.
RULE_LABEL = "treat then continue if l2 positive"

#: How much of :math:`L_2`'s own noise is the cluster's rather than the unit's on a
#: clustered draw.  See :func:`_shared_within_clusters`.
_CLUSTER_RHO = 0.9


def _shared_within_clusters(
    rng: np.random.Generator, individual: FloatArray, ids: Any, arm: FloatArray
) -> FloatArray:
    r"""A standard normal with a per-cluster component and the *same* marginal.

    :math:`\sqrt{\rho}\,S_{c(i)} + \sqrt{1-\rho}\,E_i` with both terms standard normal and
    independent is standard normal for any :math:`\rho`.  Applied to :math:`L_2`'s own
    noise, that leaves :math:`L_2 \mid W_1, A_1` exactly the
    :math:`N(0.6W_1 + 0.9A_1, 1)` law :func:`longitudinal_truth`'s quadrature integrates
    over, so a clustered draw's ``truth`` is the *same number* as an unclustered one's --
    which is the property this construction is chosen for, and which the dataset tests
    assert directly.

    **Two independent per-cluster draws**, one used by the rows at :math:`A_1 = 1` and one
    by the rows at :math:`A_1 = 0`, because a *contrast* of regimens otherwise sees no
    clustering at all.  A single shared component moves both regimens' curves the same way,
    the difference cancels it, and the cluster-robust interval on the contrast comes out
    **narrower** than the independent one -- correctly, and it would leave the process
    unable to test clustering for every parameter it reports.  Two independent draws leave
    the two curves' cluster components independent, so the contrast's variance is their sum.
    Measured on ``ate_regimen[never vs always]``: ``0.92-1.00`` times the independent
    standard error with one draw, ``1.11-1.15`` with two.

    **Why not a hidden shared variable**, which is the obvious construction and is wrong
    twice over: entering the treatment mechanisms it confounds, so the declared
    counterfactual means stop being what an adjusted fit estimates; and entering the
    outcome on the logit scale it shifts them outright, since
    :math:`E_S[\text{expit}(\eta + \gamma S)] \neq \text{expit}(\eta)`.

    **And why** :math:`L_2` **rather than** :math:`W_2`, which also has a marginal worth
    preserving.  Sharing a baseline covariate reaches the influence curve only through the
    plug-in term :math:`\bar Q_1(W) - \psi`, and that term is small next to the weighted
    residuals: measured intracluster correlation was under ``0.002`` at every :math:`\rho`,
    and the cluster-robust standard error did not move.  :math:`L_2` is drawn *after* the
    first node, so the shared part lands in the node-one residual
    :math:`\bar Q_2(L_2, \cdot) - \bar Q_1(W)` -- mean zero given :math:`W`, but not given
    the cluster's draw -- which is a first-order component of the curve rather than a
    correction to it.

    The individual component is the noise already drawn rather than a fresh normal, which
    is what keeps an *unclustered* draw's random stream byte-for-byte what it was.
    """
    codes = np.asarray(ids).astype(int)
    treated, control = (rng.standard_normal(int(codes.max()) + 1)[codes] for _ in range(2))
    shared = np.where(np.asarray(arm) == 1.0, treated, control)
    return np.sqrt(_CLUSTER_RHO) * shared + np.sqrt(1.0 - _CLUSTER_RHO) * individual


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
    it is handed an array and must return one arm per entry.  That precondition is
    **checked, not trusted** -- see :func:`_check_step_rule`.  This routine's whole job is
    to be a *truth*, and a truth that is quietly wrong for an off-contract input is worse
    than no truth at all: the arm is read once per panel, so a rule jumping anywhere else
    would be integrated as though its jump were at ``split`` and would come back a
    perfectly plausible number.

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
        _check_step_rule(rule2, grid.reshape(-1), arm, (lower, upper), split)
        integrand = density * _outcome_probability(w1, w2, grid, a1, arm)
        over_l2 = half * np.sum(integrand * quadrature.reshape(1, 1, -1), axis=2)
        total += float(np.sum(gauss[:, :, 0] * over_l2))
    return total


def _check_step_rule(
    rule2: Callable[[Any], Any],
    grid: Any,
    arm: float,
    panel: tuple[float, float],
    split: float,
) -> None:
    """Refuse a rule that is not the step function the panel decomposition assumes.

    The rule is evaluated on the quadrature nodes this panel is about to sum over -- the
    same points, so the check costs one extra call per panel and asks exactly the question
    the integral needs answered: is the arm the constant that was read at the midpoint?

    Refusing by name rather than returning a number, because every accuracy claim in the
    longitudinal section rests on this function.  A rule whose jump is elsewhere does not
    make the quadrature fail; it makes it answer for a different regimen.
    """
    lower, upper = panel
    if arm not in (0.0, 1.0):
        raise ValueError(
            f"the rule returned {arm!r} at l2={0.5 * (lower + upper)!r}; a longitudinal "
            "fit takes a binary treatment at every node, so d_2 must return 0 or 1"
        )
    arms = np.asarray(rule2(np.asarray(grid)), dtype=float).reshape(-1)
    if arms.shape[0] != np.asarray(grid).shape[0]:
        raise ValueError(
            f"the rule returned {arms.shape[0]} arm(s) for {np.asarray(grid).shape[0]} "
            "values of l2; it must return one arm per entry"
        )
    disagree = arms != arm
    if disagree.any():
        witness = float(np.asarray(grid).reshape(-1)[np.argmax(disagree)])
        raise ValueError(
            f"the rule is not constant on ({lower:g}, {upper:g}): it returns {arm:g} at "
            f"l2={0.5 * (lower + upper):g} and {arms[np.argmax(disagree)]:g} at "
            f"l2={witness:g}. This routine integrates l2 as two Gauss-Legendre panels "
            f"meeting at split={split:g} and reads the arm once per panel, so it can only "
            "take a step function whose single jump is at split. A rule with its jump "
            "elsewhere needs split= set to it; a rule with two jumps needs a third panel, "
            "which is a change to this function rather than an argument to it"
        )


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
    backend: Backend | str | None = None,
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

    ``cluster_size`` adds an ``id`` column and, with it, a per-cluster component of ``L2``
    -- so the influence curves are correlated within a cluster and ignoring ``id=``
    understates the standard error.  There is an ``id`` column over independent rows
    otherwise, which makes a cluster-robust variance *equal* the plain one and tests
    nothing.  ``truth`` holds on a clustered draw, and for a reason rather than by
    assertion: the sharing preserves ``L2``'s standard normal noise marginal exactly, which is
    the law :func:`longitudinal_truth` integrates over.  See
    :func:`_shared_within_clusters`, which also says why the hidden-variable construction
    this replaced did not.

    Parameters
    ----------
    n : int
        Number of units.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    censoring : bool
        Whether to generate monotone censoring. False leaves every unit observed.
    cluster_size : int or None
        Rows per cluster. ``None`` leaves the rows independent.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        Wide-format observations, one row per unit and one column per node.
    truth : dict of str to float
        Regimen means and contrasts, under the labels a fit reports them by.

    See Also
    --------
    cleverly.LongitudinalTreatment : The design declaration for the nodes above.
    cleverly.longitudinal.LongitudinalResult : What a fit on this frame returns.
    cleverly.datasets.make_longitudinal_survival : The same shape with an event outcome.

    Examples
    --------
    >>> from cleverly.datasets import make_longitudinal
    >>> frame, truth = make_longitudinal(n=200, seed=0)
    >>> list(frame.columns)
    ['W1', 'W2', 'A1', 'C1', 'L2', 'A2', 'C2', 'Y']
    >>> round(truth["ate_regimen[always vs never]"], 3)
    0.362
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)

    ids = None if cluster_size is None else np.arange(n) // cluster_size

    a1 = rng.binomial(1, expit(0.3 * w1 - 0.4 * w2)).astype(float)
    c1 = (
        rng.binomial(1, expit(2.2 + 0.3 * w1 - 0.3 * a1)).astype(float) if censoring else np.ones(n)
    )
    alive1 = c1 == 1.0

    noise = rng.standard_normal(n)
    if ids is not None:
        noise = _shared_within_clusters(rng, noise, ids, a1)
    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + noise
    a2 = rng.binomial(1, expit(0.5 * l2 + 0.6 * a1 - 0.2 * w2)).astype(float)
    c2 = rng.binomial(1, expit(2.4 + 0.2 * l2)).astype(float) if censoring else np.ones(n)
    alive2 = alive1 & (c2 == 1.0)

    y = rng.binomial(1, _outcome_probability(w1, w2, l2, a1, a2)).astype(float)

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


#: Probability of being kept in the biased sample of :func:`make_longitudinal_weighted`,
#: as ``(low, high)`` -- the low one applying where ``W1`` is positive.  Selection depends
#: on a covariate that moves both treatment decisions and the outcome, so an unweighted
#: analysis of the retained rows answers for the wrong population; and both values are far
#: from zero, so the weights are a tilt rather than a positivity problem of their own.
_SELECTION = (0.3, 0.9)


def make_longitudinal_weighted(
    n: int = 2000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    r"""A *biased sample* of :func:`make_longitudinal`, with the design weights that undo it.

    Each unit is kept with a known probability :math:`\pi(W_1)` and carries
    :math:`w = 1/\pi`, which is the survey case: the sampling law is
    :math:`dP_S = \pi\,dP / E[\pi]`, so tilting it by :math:`w` gives
    :math:`dP_{S,w} = w\,dP_S/E_S[w] = dP` **exactly**.  The truth is therefore
    :func:`make_longitudinal`'s unchanged, and that is what makes this a real end-to-end
    check of the weighting rather than a restatement of it: a fit that ignored ``w`` would
    estimate the *selected* population's parameter and miss its nominal coverage, while one
    that applies it is estimating the parameter this ``truth`` names.

    ``n`` is the number of units drawn *before* selection, so the returned frame has about
    ``(pi_low + pi_high) / 2 * n`` rows.  Returned rather than resampled to a fixed size
    because the retained count is part of the experiment: forcing it would condition on
    something the design did not.

    Parameters
    ----------
    n : int
        Number of units in the population the sample is drawn from.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        A biased sample of :func:`make_longitudinal`, with its design weights.
    truth : dict of str to float
        The population regimen means, which the weights recover.
    """
    rng = np.random.default_rng(seed)
    frame, truth = make_longitudinal(n=n, seed=rng, backend=backend)
    low, high = _SELECTION
    source = as_frame(frame)
    keep_probability = np.where(column_array(source, "W1") > 0.0, low, high)
    selected = rng.random(n) < keep_probability
    # Subset the columns as numpy rather than the frame as a frame.  ``.loc`` and
    # ``reset_index`` are pandas-only, and using them here meant this generator forced
    # ``backend="pandas"`` above and then ignored what the caller asked for -- so on a
    # polars-only install it raised ``ImportError`` no matter what.  Every column ends up
    # numpy anyway one line later, so nothing is lost by masking it there.
    payload = {name: column_array(source, name)[selected] for name in source.columns}
    payload["w"] = 1.0 / keep_probability[selected]
    return frame_from_dict(payload, backend=backend), truth


#: Coefficients of the hazard at the first time point,
#: ``logit P(Y1 = 1 | W1, W2, A1, C1 = 1)``.
_H1 = {"intercept": -1.1, "a1": -0.7, "w1": 0.35, "w2": -0.25}

#: Coefficients of the hazard at the second, among the units still at risk there.  The
#: ``tanh`` term keeps the regression off a linear index, as ``_Y`` does, so a ``glm``
#: nuisance learner is misspecified here rather than accidentally exact.
_H2 = {
    "intercept": -1.15,
    "a1": -0.25,
    "a2": -0.8,
    "l2": 0.4,
    "w1": 0.3,
    "w2": -0.2,
    "kink": 0.5,
}


def _hazard_one(w1: FloatArray, w2: FloatArray, a1: FloatArray | float) -> FloatArray:
    """``P(Y1 = 1 | W1, W2, A1, C1 = 1)``, at a regimen's arm or at each unit's own."""
    return expit(_H1["intercept"] + _H1["a1"] * a1 + _H1["w1"] * w1 + _H1["w2"] * w2)


def _hazard_two(
    w1: FloatArray,
    w2: FloatArray,
    l2: FloatArray,
    a1: FloatArray | float,
    a2: FloatArray | float,
) -> FloatArray:
    """``P(Y2 = 1 | ...)`` among the units at risk entering the second node."""
    index = (
        _H2["intercept"]
        + _H2["a1"] * a1
        + _H2["a2"] * a2
        + _H2["l2"] * l2
        + _H2["w1"] * w1
        + _H2["w2"] * w2
        + _H2["kink"] * np.tanh(l2)
    )
    return expit(index)


@cache
def survival_truth(a1: float, a2: float, horizon: int, nodes: int = 48) -> float:
    r"""``P(Y_{k} = 1)`` under the static regimen ``(a1, a2)`` -- the cumulative risk.

    Under the intervention the treatment and censoring mechanisms drop out, so

    .. math::

        F(1) &= E\bigl[h_1(W, a_1)\bigr] \\
        F(2) &= E\bigl[h_1(W, a_1) + (1 - h_1(W, a_1))\, h_2(W, L_2, a_1, a_2)\bigr]

    over :math:`W_1`, :math:`W_2` and :math:`L_2`'s own noise, all standard normal and
    all independent, which a product Gauss--Hermite rule evaluates to well under
    ``1e-10``.

    **The second line rests on one property of the process, and it is worth naming
    because a plausible change to the process would silently break it.**  ``L2`` is drawn
    from :math:`(W, A_1)` alone and *not* from ``Y1``, so the law of ``L2`` among the
    units who survived the first node is the marginal law of ``L2``.  That is what lets
    the same three-dimensional rule integrate both terms.  Let ``L2`` depend on the event
    at the first node -- an entirely reasonable thing for a biomarker to do -- and
    :math:`E[h_2 \mid Y_1 = 0]` stops being :math:`E[h_2]`, the rule above quietly
    returns the wrong number, and every coverage assertion built on it inherits the
    error.
    """
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = points.reshape(-1, 1, 1)
    w2 = points.reshape(1, -1, 1)
    noise = points.reshape(1, 1, -1)
    mass = weights.reshape(-1, 1, 1) * weights.reshape(1, -1, 1) * weights.reshape(1, 1, -1)
    hazard1 = _hazard_one(w1, w2, a1)
    if horizon == 1:
        return float(np.sum(mass * hazard1))
    if horizon != 2:
        raise ValueError(f"horizon must be 1 or 2; got {horizon}")
    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + noise
    return float(np.sum(mass * (hazard1 + (1.0 - hazard1) * _hazard_two(w1, w2, l2, a1, a2))))


def make_longitudinal_survival(
    n: int = 2000,
    *,
    seed: int | np.random.Generator | None = None,
    censoring: bool = True,
    cluster_size: int | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Two time points, an **absorbing event at each**, and monotone censoring.

    The survival counterpart of :func:`make_longitudinal`, with the same time-varying
    confounding in it: ``L2`` is caused by ``A1`` and causes both ``A2`` and the hazard
    at the second node.  Returns ``(frame, truth)``.

    The frame is wide, with columns ``W1``, ``W2``, ``A1``, ``C1``, ``Y1``, ``L2``,
    ``A2``, ``C2``, ``Y2`` in time order.  A unit that is censored or that **has the
    event** has ``nan`` at every node after, which is what an absorbing outcome means and
    what :class:`~cleverly.longitudinal.LongitudinalData` requires -- so the missingness
    here has two causes rather than one, and a fit that conflated them would answer for a
    different population.

    ``truth`` holds ``risk_regimen[... @ t=k]`` for the four static regimens at both
    horizons, and the contrast of ``always`` against ``never`` at each -- under the names
    a fit reports them by, so a coverage study can look each up by the name it read off
    the result.  **Static plans only**: a rule's truth needs the two-panel treatment
    :func:`longitudinal_rule_truth` gives it, twice over, and a rule under a survival
    outcome is already checked exactly in ``tests/discrete_law_survival.py``.

    Parameters
    ----------
    n : int
        Number of units.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    censoring : bool
        Whether to generate monotone censoring. False leaves every unit observed.
    cluster_size : int or None
        Rows per cluster. ``None`` leaves the rows independent.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        Wide-format observations with one absorbing event node per time point.
    truth : dict of str to float
        Cumulative incidence by regimen and horizon, under a fit's own labels.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)

    ids = None if cluster_size is None else np.arange(n) // cluster_size

    a1 = rng.binomial(1, expit(0.3 * w1 - 0.4 * w2)).astype(float)
    c1 = (
        rng.binomial(1, expit(2.2 + 0.3 * w1 - 0.3 * a1)).astype(float) if censoring else np.ones(n)
    )
    observed1 = c1 == 1.0

    y1 = rng.binomial(1, _hazard_one(w1, w2, a1)).astype(float)
    at_risk2 = observed1 & (y1 == 0.0)

    # Drawn from (W, A1) and *not* from Y1 -- see ``survival_truth``, whose second line
    # is only right because the survivors' L2 law is the marginal one.  The clustered
    # draw shares part of the noise and leaves that law alone; see
    # ``_shared_within_clusters``.
    noise = rng.standard_normal(n)
    if ids is not None:
        noise = _shared_within_clusters(rng, noise, ids, a1)
    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + noise
    a2 = rng.binomial(1, expit(0.5 * l2 + 0.6 * a1 - 0.2 * w2)).astype(float)
    c2 = rng.binomial(1, expit(2.4 + 0.2 * l2)).astype(float) if censoring else np.ones(n)
    observed2 = at_risk2 & (c2 == 1.0)

    y2 = rng.binomial(1, _hazard_two(w1, w2, l2, a1, a2)).astype(float)

    payload = {
        "W1": w1,
        "W2": w2,
        "A1": a1,
        "C1": c1,
        "Y1": np.where(observed1, y1, np.nan),
        "L2": np.where(at_risk2, l2, np.nan),
        "A2": np.where(at_risk2, a2, np.nan),
        "C2": np.where(at_risk2, c2, np.nan),
        # Carried forward for a unit that already had the event, which is what a wide
        # survival frame looks like in practice and what the container accepts.
        "Y2": np.where(observed1 & (y1 == 1.0), 1.0, np.where(observed2, y2, np.nan)),
    }
    if not censoring:
        del payload["C1"]
        del payload["C2"]
    if ids is not None:
        payload["id"] = ids.astype(float)

    plans = (
        ("always", (1.0, 1.0)),
        ("never", (0.0, 0.0)),
        ("early", (1.0, 0.0)),
        ("late", (0.0, 1.0)),
    )
    truth = {
        f"risk_regimen[{label}{_HORIZON_INFIX}{horizon}]": survival_truth(plan[0], plan[1], horizon)
        for label, plan in plans
        for horizon in (1, 2)
    }
    for horizon in (1, 2):
        truth[f"ate_regimen[always vs never{_HORIZON_INFIX}{horizon}]"] = (
            truth[f"risk_regimen[always{_HORIZON_INFIX}{horizon}]"]
            - truth[f"risk_regimen[never{_HORIZON_INFIX}{horizon}]"]
        )
    return frame_from_dict(payload, backend=backend), truth


#: Log-odds that an event at the first node is a *relapse* rather than a death, given the
#: history.  Treatment pushes the split towards relapse while ``_H1`` lowers the all-cause
#: hazard, so the two causes' contrasts come out with opposite signs -- which is the shape
#: a competing-risks report exists to show, and the one a single-event fit cannot.
_SPLIT1 = {"intercept": 0.15, "a1": 1.1, "w1": 0.3, "w2": -0.2}

#: The same at the second node, among the units still at risk there.
_SPLIT2 = {"intercept": 0.1, "a1": 0.5, "a2": 0.9, "l2": -0.25, "w1": 0.2}


def _relapse_share_one(w1: FloatArray, w2: FloatArray, a1: FloatArray | float) -> FloatArray:
    """``P(cause = relapse | an event happened at the first node, history)``."""
    return expit(
        _SPLIT1["intercept"] + _SPLIT1["a1"] * a1 + _SPLIT1["w1"] * w1 + _SPLIT1["w2"] * w2
    )


def _relapse_share_two(
    w1: FloatArray,
    l2: FloatArray,
    a1: FloatArray | float,
    a2: FloatArray | float,
) -> FloatArray:
    """``P(cause = relapse | an event happened at the second node, history)``."""
    return expit(
        _SPLIT2["intercept"]
        + _SPLIT2["a1"] * a1
        + _SPLIT2["a2"] * a2
        + _SPLIT2["l2"] * l2
        + _SPLIT2["w1"] * w1
    )


@cache
def competing_truth(a1: float, a2: float, cause: str, horizon: int, nodes: int = 48) -> float:
    r"""``P(leave through ``cause`` by ``horizon``)`` under the static regimen ``(a1, a2)``.

    The cause-specific cumulative incidence.  Under the intervention the treatment and
    censoring mechanisms drop out, leaving

    .. math::

        F_j(1) &= E\bigl[h_1\, s_{1j}\bigr] \\
        F_j(2) &= E\bigl[h_1 s_{1j} + (1 - h_1)\, h_2\, s_{2j}\bigr]

    with :math:`h` the **all-cause** hazard and :math:`s_j` the share of events that are
    of cause :math:`j`.  Note which factor is which: the numerator is cause-specific and
    the survival factor :math:`1 - h_1` is all-cause, because a unit that left through the
    *other* cause is no more available to have this one than a unit that left through this
    one.  Writing :math:`1 - h_1 s_{1j}` there would be the cause's own survival, and is
    the mistake ``tests/discrete_law_competing.py`` exists to catch.

    Modelling the causes as an all-cause hazard times a share is what keeps this a
    quadrature rather than a simulation: the shares sum to one by construction, so the
    incidences and the event-free probability exhaust the mass exactly and no constraint
    has to be imposed on separately drawn hazards.

    **The same caveat as :func:`survival_truth`, and for the same reason.**  ``L2`` is
    drawn from :math:`(W, A_1)` and not from the event at the first node, so the law of
    ``L2`` among those still at risk is its marginal law, which is what lets one
    three-dimensional rule integrate both terms.  Let ``L2`` depend on the first node's
    event and the second line quietly returns the wrong number.
    """
    if cause not in ("relapse", "death"):
        raise ValueError(f"cause must be 'relapse' or 'death'; got {cause!r}")
    if horizon not in (1, 2):
        raise ValueError(f"horizon must be 1 or 2; got {horizon}")

    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    w1 = points.reshape(-1, 1, 1)
    w2 = points.reshape(1, -1, 1)
    noise = points.reshape(1, 1, -1)
    mass = weights.reshape(-1, 1, 1) * weights.reshape(1, -1, 1) * weights.reshape(1, 1, -1)

    hazard1 = _hazard_one(w1, w2, a1)
    share1 = _relapse_share_one(w1, w2, a1)
    if cause == "death":
        share1 = 1.0 - share1
    if horizon == 1:
        return float(np.sum(mass * hazard1 * share1))

    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + noise
    hazard2 = _hazard_two(w1, w2, l2, a1, a2)
    share2 = _relapse_share_two(w1, l2, a1, a2)
    if cause == "death":
        share2 = 1.0 - share2
    return float(np.sum(mass * (hazard1 * share1 + (1.0 - hazard1) * hazard2 * share2)))


def make_longitudinal_competing(
    n: int = 2000,
    *,
    seed: int | np.random.Generator | None = None,
    censoring: bool = True,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Two time points, **two competing absorbing causes** at each, monotone censoring.

    The competing-risks counterpart of :func:`make_longitudinal_survival`, sharing its
    hazards and its time-varying confounding: an event happens with the same all-cause
    probability, and a second draw decides which cause it was.  Returns ``(frame, truth)``.

    The frame is wide, with columns ``W1``, ``W2``, ``A1``, ``C1``, ``R1``, ``D1``, ``L2``,
    ``A2``, ``C2``, ``R2``, ``D2`` in time order -- one indicator per cause per node, which
    is the declaration :class:`~cleverly.longitudinal.LongitudinalData` takes.  A unit that
    is censored or that has **either** event has ``nan`` at every node after.

    ``truth`` holds ``cif_regimen[..., cause @ t=k]`` for the four static regimens at both
    causes and both horizons, and the contrast of ``always`` against ``never`` at each,
    under the names a fit reports them by -- so a coverage study looks each up by the name
    it read off the result.

    Parameters
    ----------
    n : int
        Number of units.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    censoring : bool
        Whether to generate monotone censoring. False leaves every unit observed.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        Wide-format observations with two competing event nodes per time point.
    truth : dict of str to float
        Cause-specific cumulative incidence by regimen, cause, and horizon.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)

    a1 = rng.binomial(1, expit(0.3 * w1 - 0.4 * w2)).astype(float)
    c1 = (
        rng.binomial(1, expit(2.2 + 0.3 * w1 - 0.3 * a1)).astype(float) if censoring else np.ones(n)
    )
    observed1 = c1 == 1.0

    event1 = rng.binomial(1, _hazard_one(w1, w2, a1)).astype(float)
    # One draw decides *whether*, a second *which*: the shares sum to one, so the causes
    # are exclusive by construction rather than by a rejection step.
    relapse1 = event1 * (rng.random(n) < _relapse_share_one(w1, w2, a1))
    death1 = event1 * (1.0 - relapse1)
    at_risk2 = observed1 & (event1 == 0.0)

    # Drawn from (W, A1) and *not* from the event -- see ``competing_truth``.
    l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + rng.standard_normal(n)
    a2 = rng.binomial(1, expit(0.5 * l2 + 0.6 * a1 - 0.2 * w2)).astype(float)
    c2 = rng.binomial(1, expit(2.4 + 0.2 * l2)).astype(float) if censoring else np.ones(n)
    observed2 = at_risk2 & (c2 == 1.0)

    event2 = rng.binomial(1, _hazard_two(w1, w2, l2, a1, a2)).astype(float)
    relapse2 = event2 * (rng.random(n) < _relapse_share_two(w1, l2, a1, a2))
    death2 = event2 * (1.0 - relapse2)

    def _carry(first: FloatArray, second: FloatArray) -> FloatArray:
        """A cause's column at the second node, absorbing.

        A unit that left through this cause at the first node carries its ``1`` forward;
        one that left through the *other* carries a ``0``, since it did not have this one
        and is not going to.  Both are what the container's absorbing rule accepts, and
        together they keep the two columns exclusive at every node.
        """
        return np.where(
            observed1 & (event1 == 1.0),
            first,
            np.where(observed2, second, np.nan),
        )

    payload = {
        "W1": w1,
        "W2": w2,
        "A1": a1,
        "C1": c1,
        "R1": np.where(observed1, relapse1, np.nan),
        "D1": np.where(observed1, death1, np.nan),
        "L2": np.where(at_risk2, l2, np.nan),
        "A2": np.where(at_risk2, a2, np.nan),
        "C2": np.where(at_risk2, c2, np.nan),
        "R2": _carry(relapse1, relapse2),
        "D2": _carry(death1, death2),
    }
    if not censoring:
        del payload["C1"]
        del payload["C2"]

    plans = (
        ("always", (1.0, 1.0)),
        ("never", (0.0, 0.0)),
        ("early", (1.0, 0.0)),
        ("late", (0.0, 1.0)),
    )
    truth = {
        f"cif_regimen[{label}, {cause}{_HORIZON_INFIX}{horizon}]": competing_truth(
            plan[0], plan[1], cause, horizon
        )
        for label, plan in plans
        for cause in ("relapse", "death")
        for horizon in (1, 2)
    }
    for cause in ("relapse", "death"):
        for horizon in (1, 2):
            truth[f"ate_regimen[always vs never, {cause}{_HORIZON_INFIX}{horizon}]"] = (
                truth[f"cif_regimen[always, {cause}{_HORIZON_INFIX}{horizon}]"]
                - truth[f"cif_regimen[never, {cause}{_HORIZON_INFIX}{horizon}]"]
            )
    return frame_from_dict(payload, backend=backend), truth
