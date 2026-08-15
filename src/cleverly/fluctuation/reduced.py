r"""The two extra clever covariates of doubly-robust nonparametric inference.

A plain TMLE solves one score equation per counterfactual mean,
:math:`P_n[1_a/g^*(a|W) \cdot \{Y - \bar Q^*(a, W)\}] = 0`, and that is enough for the
*point estimate* to be doubly robust.  It is not enough for the interval: the second-order
remainder is the product :math:`\|\hat g - g_0\| \cdot \|\hat{\bar Q} - \bar Q_0\|`, and
:math:`\sqrt n` times a product needs *both* factors shrinking.  van der Laan (2014) and
Benkeser, Carone, van der Laan & Gilbert (2017) buy an interval that survives one
inconsistent nuisance by solving two further equations, written here in the numbering the
software paper uses (Benkeser & Hejazi 2023, Observational Studies 9(2):43-78):

.. math::

    (9)  \quad & P_n\Bigl[\frac{Q_r(a, W)}{g^*(a|W)}\{1_a - g^*(a|W)\}\Bigr] = 0 \\
    (10) \quad & P_n\Bigl[1_a \frac{g_{r,2}(a|W)}{g_{r,1}(a|W)}
                          \{Y - \bar Q^*(a, W)\}\Bigr] = 0

with :math:`1_a = 1\{A = a\}` and the three reduced-dimension regressions of
:mod:`cleverly.estimators.reduced`.  This module builds the covariates; the alternation
that solves the pair beside equation (8) is
:func:`~cleverly.estimators.targeting.solve_with_reduction`.

**Neither of these can go in a registry, and the reason is the same for both: the group
stays** ``"mean"``.  The estimand is still :math:`E[Y^a]` and the report is still ``ey1``,
``ey0`` and ``ate`` -- a different estimator behind the same parameters, exactly as
:class:`~cleverly.CTMLE` is.  So ``register_submodel("mean", ...)`` would shadow the
builder every ordinary fit uses, and ``register_mechanism("mean", ...)`` would make
:func:`~cleverly.fluctuation.mechanism.needs_mechanism` true for the ``"mean"`` group and
divert *every* fit in the package into an alternation it has no reduced regressions for.
Two plain functions, called by name from the one solver that wants them, is what a group
that is already registered leaves available.

**Equation (10) is a second** :class:`~cleverly.fluctuation.submodel.Submodel` **in the
same group rather than two more columns on the first**, which is ``drtmle``'s
``Qsteps = 2`` -- a backfitting minimisation, "found to be more stable in simulations".
That choice is what keeps :attr:`~cleverly.fluctuation.submodel.Submodel.arm_columns`
mapping one arm to one column and
:meth:`~cleverly.fluctuation.submodel.Submodel.column_for` meaning what
:mod:`cleverly.sensitivity.omitted_variable` reads it as.  One wider submodel would have
made a column mean something new, and the Riesz representer that sensitivity analysis
builds would have been reading half a covariate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray
from ..utils.bounds import bound
from .submodel import Submodel

if TYPE_CHECKING:  # pragma: no cover - `estimators` imports `fluctuation`, not the reverse
    from ..estimators.reduced import ReducedSet

__all__ = ["reduced_mechanism_covariate", "reduced_outcome_submodel"]


def reduced_outcome_submodel(
    treatment: FloatArray,
    reduced: ReducedSet,
    *,
    bounds: tuple[float, float],
) -> Submodel:
    r"""Equation (10)'s covariate: :math:`1_a\,g_{r,2}(a|W)/g_{r,1}(a|W)`, one column per arm.

    Shaped exactly like :func:`~cleverly.fluctuation.submodel.mean_submodel`, and
    deliberately: the observed column carries the indicator and the counterfactual columns
    do not, because the update is applied at the counterfactual covariate while the score
    is taken at the observed one.  Reading ``arms[a]`` where ``observed`` was meant -- or
    the reverse -- is the mistake
    :func:`~cleverly.estimators.targeting.solve_with_reduction` inherits the shape of this
    builder to avoid.

    Parameters
    ----------
    treatment:
        Arm codes, length ``n``.
    reduced:
        The reduced-dimension regressions, whose :attr:`~cleverly.estimators.reduced
        .ReducedSet.arms` decide the column order.  Read off the object rather than passed
        beside it, so that a caller cannot hand this the arms of a different fit.
    bounds:
        Truncation for :math:`g_{r,1}`, which is the denominator here.

    Notes
    -----
    **The two reduced mechanisms are truncated at different times, and it is not an
    inconsistency.**  :math:`g_{r,2}`'s *target* was a quotient by :math:`\hat g`, so the
    bound that formed it is on record at fit time and cannot be moved afterwards --
    :func:`~cleverly.estimators.reduced.fit_reduced` says so, and it is why the part of a
    truncation curve that comes from that array is flat by construction.  :math:`g_{r,1}`
    is a *probability that this covariate divides by*, so it is bounded here, at targeting
    time, exactly as the propensity is in
    :func:`~cleverly.estimators.targeting.build_submodel`.  A sensitivity sweep therefore
    does move this denominator, and a reader is entitled to know which half of the extra
    equation it reached.

    :math:`g_{r,1}` is clipped column by column and **not** complemented across the arms:
    :math:`g_{r,1}(1|w)` and :math:`g_{r,1}(0|w)` condition on different designs --
    :math:`\hat{\bar Q}(1, W)` and :math:`\hat{\bar Q}(0, W)` -- so they are two
    regressions rather than a probability and its complement, and they do not sum to one
    before any truncation either.  :meth:`~cleverly.estimators.reduced.ReducedSet
    .bounded_gr1` is where that is written down.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    arms = reduced.arms
    if a.shape[0] != reduced.n:
        raise ValueError(
            f"the treatment has {a.shape[0]} rows and the reduced regressions {reduced.n}"
        )
    ratio = np.asarray(reduced.gr2, dtype=float) / reduced.bounded_gr1(bounds)
    zeros = np.zeros(a.shape[0])
    return Submodel(
        np.column_stack([(a == arm) * ratio[:, j] for j, arm in enumerate(arms)]),
        {
            arm: np.column_stack([ratio[:, j] if i == j else zeros for i in range(len(arms))])
            for j, arm in enumerate(arms)
        },
        tuple(f"h_dr{arm:g}" for arm in arms),
        # The same group as the equation it is solved beside. The label decides which
        # influence curve reads the fluctuation, and both of these target E[Y^a].
        "mean",
        {arm: j for j, arm in enumerate(arms)},
    )


def reduced_mechanism_covariate(
    reduced: ReducedSet,
    propensity: FloatArray,
    *,
    bounds: tuple[float, float],
) -> FloatArray:
    r"""Equation (9)'s covariate, one column per treatment arm.

    At two arms, :func:`~cleverly.fluctuation.mechanism.solve_mechanism` tilts the higher
    arm probability and both equations are expressed against its residual.  For the lower,

    .. math::

        1\{A = a_0\} - g^*(a_0|W) = -\bigl\{A - g^*(a_1|W)\bigr\},

    so its column is :math:`-Q_r(a_0, W)/g^*(a_0|W)`.  **The sign is the whole content of
    this function.**  Getting it backwards solves the lower arm's equation with the wrong
    sign, which moves ``ey0`` and ``ate`` and leaves ``ey1`` untouched -- a failure that
    reads as a problem with one estimand rather than with a covariate.

    Parameters
    ----------
    At more than two arms, R ``drtmle`` instead poses one binary equation per arm,
    :math:`P_n[Q_r(a,W)/g_a^*\{1_a-g_a^*\}]=0`.  The returned matrix is therefore simply
    :math:`Q_r/g^*` column by column.  Those independent fluctuations are not renormalised:
    renormalisation would reopen the scores they just solved.

    reduced:
        Supplies :math:`Q_r` and the arm order.
    propensity:
        The current upper-arm margin as ``(n,)`` for binary treatment, or the armwise
        ``(n, K)`` mechanism otherwise.  The **targeted** mechanism, not the initial one:
        this covariate reads the very mechanism it fluctuates, which is where
        it departs from :func:`~cleverly.fluctuation.mechanism.ipsi_mechanism_covariate`
        (that one reads only the targeted :math:`\bar Q^*`) and why the alternation rebuilds
        it after every mechanism step rather than once.
    bounds:
        Truncation for the mechanism in the denominator.  Binary treatment retains the
        complement convention; multiple arms are bounded column by column.

    Notes
    -----
    :math:`Q_r` is identically zero when :math:`\hat{\bar Q}` is right -- row by row, not
    merely on average -- so this covariate vanishes at the truth and the mechanism comes
    back untouched.  That is not a special case for
    :func:`~cleverly.fluctuation.mechanism.solve_mechanism`, which treats an all-zero
    covariate as an equation already solved, and it is why no exact-law Gateaux module can
    see whether this function is right.
    """
    arms = reduced.arms
    values = np.asarray(propensity, dtype=float)
    if values.ndim == 1:
        if len(arms) != 2:
            raise ValueError(f"a one-column reduced mechanism requires two arms; got {list(arms)}")
        g1 = bound(values.reshape(-1), float(bounds[0]), float(bounds[1]))
        if g1.shape[0] != reduced.n:
            raise ValueError(
                f"the mechanism has {g1.shape[0]} rows and the reduced regressions {reduced.n}"
            )
        qr = np.asarray(reduced.qr, dtype=float)
        mechanism = np.column_stack([1.0 - g1, g1])
        # +1 for the arm the tilt is on, -1 for the one whose residual is its negation.
        signs = np.array([-1.0, 1.0])
        return np.asarray(signs * qr / mechanism, dtype=float)

    if values.shape != (reduced.n, len(arms)):
        raise ValueError(f"the mechanism must be ({reduced.n}, {len(arms)}); got {values.shape}")
    mechanism = bound(values, float(bounds[0]), float(bounds[1]))
    return np.asarray(reduced.qr, dtype=float) / mechanism
