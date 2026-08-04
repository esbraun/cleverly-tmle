r"""Tier 1 of the coverage study: nuisance sequences handed to the estimator, not learned.

``docs/drtmle/validation-plan.md`` §5 asks for two tiers, and this is the first: *"a test-only
nuisance-injection interface handing the estimator* :math:`\hat Q = \bar Q_0 + n^{-\alpha} h_Q`
*while* :math:`\hat g \to g_1 \neq g_0`, *and the mirror"*.  It is **not an applied claim** and
must not be presented as one.  What it is, and what nothing else in this repository is, is the
only construction in which *"the intended asymptotic regime was entered"* is true **by
definition** -- which is what makes it the place to read a remainder off.

Two cells, both off the diagonal of the misspecification grid, because which nuisance is wrong
is the whole axis and one cell is an anecdote:

======================  ==========================================  ==========================
cell                    :math:`\hat Q`                              :math:`\hat g`
======================  ==========================================  ==========================
``q-drift``             :math:`\bar Q_0 + n^{-\alpha} h_a`          :math:`g_1 \neq g_0`, fixed
``g-drift``             :math:`\bar Q_0 + d_a`, fixed               :math:`g_0 + n^{-\alpha} h`
======================  ==========================================  ==========================

Three decisions here are load-bearing, and each is a way this could look right and be wrong.

**The base law is** :func:`~cleverly.datasets.linear_dgp`, **and it is chosen for overlap
rather than for difficulty.**  Every misspecification is *prescribed* here, so nothing is asked
of the process except that its own mechanism stay interior -- and a law whose propensity sits
near ``0.5`` puts both cells **inside** the supported contract by construction, so that a
coverage number from them is evidence about Theorem 1's estimator rather than about the
constrained rendering beside it (``docs/roadmap.md``'s item 25).
:func:`~cleverly.validation.correction_check`'s ``contract`` is what checks that rather than
assuming it, per cell and per size, and the study reports it.

**The outcome scaler is fixed by** :data:`Q_BOUNDS` **and is not recovered from the draw.**
The estimator maps ``Y`` onto ``[0, 1]`` before fitting :math:`\bar Q`, so an injected outcome
regression has to apply the same affine map -- and ``tests/conftest.py``'s
``OracleOutcomeContinuous`` recovers it by regressing the scaled outcome on the raw structural
mean, which is exact for a *binary* outcome and carries an :math:`O(n^{-1/2})` error from the
noise for a continuous one.  **That is the same order as the drift being injected**, so here the
support is declared instead: ``q_bounds=Q_BOUNDS`` on both estimators makes the map known in
advance and identical across draws, and a draw whose outcome leaves it raises from
:meth:`~cleverly.utils.bounds.OutcomeScaler.from_outcome` rather than being silently rescaled.

**The drift coefficient is the deliverable, not the rate.**  :math:`\alpha < 1/2` is not
sufficient: the remainder is an **inner product** rather than a norm, so

.. math::

    R_{2,a} = P_0\!\left[\frac{\hat g_a - g_{0,a}}{\hat g_a}(\hat Q_a - \bar Q_{0,a})\right]
            = n^{-\alpha} c_a + o(n^{-\alpha})

can have :math:`c_a = 0` with :math:`\|h_a\| > 0` -- and :math:`c_1 - c_0` can vanish in the ATE
with both arm coefficients nonzero.  So the shape is chosen so the coefficients are the design's
own constants rather than whatever falls out, with the arms given **opposite signs**, which
makes cancellation in the ATE impossible rather than merely unlikely.

**And there are two coefficients, not one.  That is C3b's repair and it is the whole of what
this module learned from C3a's pilot.**  The display above is the *plug-in* remainder, at the
initial regression; a fit's bias is the same expression at the **targeted** one, and targeting
solves :math:`P_0[w_a(\bar Q^*_a - \bar Q_{0,a})] = 0` with :math:`w_a = g_{0,a}/\hat g_a`.
Since the weight above is :math:`u_a = 1 - w_a`, the score removes precisely the part of the
injection the fluctuation can reach -- so a shape aligned with :math:`u_a` alone, which is what
this module used to inject, makes :math:`c_a` large and constrains **nothing** about what
survives.  Measured: the surviving coefficient was ``0.00092`` against a declared ``0.40``.

The repair is a second linear condition, not a smaller constant.  :math:`b_a =
P_0[v_a \cdot \text{shape}]` is a linear functional too (:func:`targeted_weight`), so the shape
is placed in the span of the two conditions' representers and solved for both declared values
at once (:func:`_shape_multipliers`).  :func:`drift_coefficients` and
:func:`targeted_coefficients` recompute the pair from the law by quadrature and
``tests/unit/test_drtmle_coverage.py`` asserts both, which is the verification §5 asks for in
place of inferring the regime from an :math:`L_2` rate.

**The two cells declare different targeted coefficients and the asymmetry is positivity.**
``q-drift`` perturbs the outcome regression, whose support is declared as :data:`Q_BOUNDS` and
wide on purpose, and holds :data:`Q_DRIFT_B` at ``0.20`` per arm.  ``g-drift`` perturbs a
**probability**, which must stay in ``(0, 1)`` at every row; since the fluctuation absorbs
92-95% there, a surviving ``0.40`` would need :math:`\hat g` to reach ``-0.16``, so
:data:`G_DRIFT_B_ATE` is ``0.10``.  The consequence is a scope statement rather than a defect:
``q-drift`` is the cell a coverage shortfall is claimed in and ``g-drift`` is where ``DRTMLE``
is checked to hold nominal under a drift.

What this module does **not** compute is :math:`P_0 \hat D` for the *doubly-robust* curve, and
so not ``R_remaining``.  The primary nuisances here are prescribed functions and integrate
exactly; the three reduced regressions are **fitted**, so evaluating their limit on new
covariates needs the fold-retained nuisance objects §5 puts in Tier 2.  Piece C2 built them as
``DRTMLE(evaluation=...)`` and ``benchmarks/drtmle_remainder.py`` is the arithmetic on top, so the
corrected remainder is available *here* too -- ``--evaluation-n`` is the knob, and item 13's rate
is C3's dispatch.  :func:`exact_remainder` is the *plug-in* remainder at the injected sequence and
:func:`exact_targeted_remainder` is the estimator's bias; the second is the regime-entry
evidence and the first says the injection is what it claims.

**This paragraph used to say the targeting step moves** :math:`\hat Q` **by**
:math:`O_p(n^{-1/2})`, *"smaller than the injected* :math:`n^{-\alpha}` *at every*
:math:`\alpha < 1/2` *and so leaves the drift's leading term where it was"*.  **That is false
and it is where the whole mis-sizing entered.**  :math:`\varepsilon` is not driven by sampling
noise here, it is driven by the injected bias -- the score equation has to remove it -- so the
step is :math:`O(n^{-\alpha})`, exactly the order of the injection, and it removes almost all
of it.  What is left is :func:`targeted_coefficients` and it is what the design now declares.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from scipy.optimize import brentq
from sklearn.base import BaseEstimator

from cleverly.datasets import DGP, linear_dgp
from cleverly.utils.bounds import OutcomeScaler, expit, logit

__all__ = [
    "ALPHA",
    "CELLS",
    "G_DRIFT_B_ATE",
    "G_DRIFT_C_ATE",
    "Q_BOUNDS",
    "Q_DRIFT_B",
    "Q_DRIFT_C",
    "InjectedMechanism",
    "InjectedOutcome",
    "base_law",
    "drift_coefficients",
    "exact_remainder",
    "exact_targeted_remainder",
    "free_shape",
    "nuisance_error",
    "plugin_weight",
    "population_epsilon",
    "targeted_coefficients",
    "targeted_weight",
]

#: The two off-diagonal cells, in the order every table reports them.
CELLS = ("q-drift", "g-drift")

#: The drift exponent.  ``0.25`` is the familiar bar for the *both-consistent* product
#: condition, which is **sufficient rather than necessary** here: in an off-diagonal cell the
#: misspecified nuisance's error is ``O(1)``, so the root-``n`` drift is ``n^(1/2 - alpha)``
#: and any ``alpha < 1/2`` grows.  It is reported as a knob rather than defended as a
#: threshold, and the reason not to push it smaller is on the other side of the ledger: the
#: appendix-B terms ``DRTMLE`` needs negligible are built out of the *same* primary nuisances,
#: so a badly enough estimated one degrades the corrected estimator too.
ALPHA = 0.25

#: The declared support of the outcome, passed as ``q_bounds=`` to both estimators so that the
#: injection's affine map is known rather than recovered -- see the module docstring.
#:
#: Wide on purpose.  ``linear_dgp``'s outcome has mean ``2.75`` and standard deviation ``1.75``
#: at the arm mixture the process draws, so ``n = 2,400`` reaches roughly ``[-3.5, 9.1]`` at
#: 3.6 standard deviations; this leaves better than two further standard deviations either
#: side.  It is not free -- a wide range compresses the scaled ``Qbar`` and so the logistic
#: fluctuation's leverage -- but it is applied identically to both estimators, so the
#: comparison the study makes is unaffected.
Q_BOUNDS = (-8.0, 14.0)

#: How wrong the ``q-drift`` cell's mechanism is: a constant shift of :math:`g_0`'s log odds.
#: A shift rather than a dropped term, so that :math:`g_1 - g_0` has **one sign everywhere** --
#: which is what makes the misspecification weight :math:`u_a` single-signed and the drift
#: coefficients below impossible to cancel.
G_LOGIT_SHIFT = 0.8

#: The ``q-drift`` cell's drift coefficients, **per arm and declared** rather than derived:
#: :math:`h_a` is normalised so that :math:`c_a` comes out at exactly these values, which is
#: what turns "choose :math:`h_a` analytically so :math:`|c_a| \ge c_{\min}`" into arithmetic
#: instead of a hope.  Opposite signs, so :math:`c_{ATE} = c_1 - c_0 = 0.30` is a **sum** of
#: magnitudes and cannot cancel.
#:
#: The magnitude is sized from the drift a coverage number can resolve rather than from taste,
#: and the sizing is a **prediction the pilot checks**.  ``bias / se ~ n^(1/2 - alpha)c/sigma``
#: with ``sigma_ATE`` measured at ``2.6`` on one injected fit (``se = 0.106`` at ``n = 600``),
#: so :math:`c_{ATE} = 0.40` puts the plain interval's shift at ``0.76``, ``0.91`` and ``1.08``
#: standard errors at ``600 / 1,200 / 2,400`` -- a ``TMLE`` coverage of about
#: ``0.87 / 0.86 / 0.81``, so a shortfall of ``0.08`` to ``0.14`` against 250 replicates' Monte
#: Carlo error of ``0.014``.  Comfortably clear of gate 2's predeclared ``0.05`` at every size
#: and not so large that the interval is a formality.  **Provisional until the pilot**, which
#: is the one point at which §5 permits it to move.
Q_DRIFT_C = {1.0: 0.20, 0.0: -0.20}

#: The ``q-drift`` cell's **targeted** coefficients: the ones the estimator's bias has, and
#: the ones a coverage shortfall is sized from.  Declared beside :data:`Q_DRIFT_C` rather than
#: instead of it, because §5's targeted-coefficient clause asks for **both** columns reported
#: -- the plug-in one says the injection is what it claims, and this one says the regime was
#: entered.
#:
#: **New with C3b, and the whole of Tier 1's repair.**  The design used to declare the first
#: alone and size a coverage gap off it; C3a's pilot measured no gap, and the decomposition in
#: ``benchmarks/drtmle_tier1_bias.py`` says why -- at the old shape the targeted coefficient
#: came out at ``0.00092`` against the declared ``0.40``, a factor of 436, because the
#: fluctuation's one free parameter per arm absorbed 98.6% of what was injected.
#:
#: Set at :data:`Q_DRIFT_C`'s own values, which re-applies the sizing at
#: ``docs/drtmle/coverage-study.md``'s *"the committed calculation"* to the quantity it was
#: always about: that arithmetic was right and was read against the wrong column, so
#: ``b_ATE = 0.40`` restores the ``0.08`` to ``0.14`` shortfall it predicted.  **Opposite
#: signs**, for :data:`Q_DRIFT_C`'s reason and with a sharper one behind it: the pilot's
#: targeted coefficients came out *both positive*, so ``b_ATE`` was a difference of magnitudes
#: where ``c_ATE`` was a sum.  The design's own no-cancellation property did not survive
#: targeting, and declaring ``b`` per arm is what restores it.
#:
#: **Setting it equal to** :data:`Q_DRIFT_C` **has a consequence worth knowing about, and it
#: is not a coincidence.**  :math:`b_a = c_a` means :math:`P_0[v_a h_a] = P_0[u_a h_a]`, and
#: since :math:`v_a - u_a = (1 - \kappa_a)w_a` with :math:`\kappa_a \neq 1`, that forces
#: :math:`P_0[w_a h_a] = 0` -- the injection is exactly **orthogonal to the fluctuation's own
#: score**.  So :math:`\varepsilon_a` is zero in the limit, the fluctuation absorbs nothing,
#: and the plug-in and targeted columns coincide.  Measured on real fits: a fitted
#: :math:`\varepsilon` of ``+0.00026 +/- 0.00183`` and an absorbed share of ``0.0000``.
Q_DRIFT_B = {1.0: 0.20, 0.0: -0.20}

#: The ``g-drift`` cell's target for the **ATE** coefficient.  One target rather than two,
#: and that is structural rather than a simplification: a binary treatment's mechanism has one
#: free function, since the estimator reads :math:`\hat g(1|W)` off a classifier and takes the
#: complement, so one perturbation determines both arms' coefficients and only their
#: combination can be set.  :func:`drift_coefficients` reports what :math:`c_1` and
#: :math:`c_0` came out at, and the test asserts both clear :data:`C_MIN`.
G_DRIFT_C_ATE = 0.40

#: The ``g-drift`` cell's **targeted** ATE coefficient -- :data:`Q_DRIFT_B`'s counterpart, and
#: one target rather than two for :data:`G_DRIFT_C_ATE`'s structural reason.  At the old shape
#: this came out at ``0.0259``, a factor of 15 below the declared plug-in one; the absorbed
#: share was 92.5% and 95.4% at the two arms.
#:
#: **A quarter of** :data:`Q_DRIFT_B`'s, and the asymmetry is *positivity* rather than a
#: preference.  The two cells inject into different objects and only one of them has a
#: declared support: ``q-drift`` moves the outcome regression, whose range is
#: :data:`Q_BOUNDS` and is wide on purpose, while ``g-drift`` moves a **probability**, which
#: has to stay in ``(0, 1)`` at every row of every draw.  Since the fluctuation absorbs 92-95%
#: of what is injected here, buying a surviving ``0.40`` costs a perturbation twice the
#: distance from :math:`g_0` to its nearest boundary -- measured, :math:`\hat g` reaches
#: ``-0.16`` at ``n = 200`` and :class:`InjectedMechanism` raises rather than clipping.
#:
#: Chosen by scanning the reachable values against **both** pre-flight conditions rather than
#: by taste.  At ``0.10`` the regime-entry column reads ``+0.0996 / +0.0994 / +0.0991`` at the
#: three study sizes -- within ``0.5%`` of declared and flat -- and :math:`\hat g` keeps a
#: margin of ``0.099`` at ``n = 200``.  At ``0.25`` the mechanism already leaves ``(0, 1)`` and
#: the column reads ``0.314`` against its own declared ``0.25``.
#:
#: **The consequence is a scope statement and it belongs on the face of the design.**  A drift
#: of ``0.10`` puts the plain interval's shift at ``0.19`` to ``0.27`` standard errors, so a
#: ``TMLE`` shortfall here is ``0.005`` to ``0.008`` -- real, and far below gate 2's
#: predeclared ``0.05``.  So ``g-drift`` is where ``DRTMLE`` is checked to hold nominal under a
#: drift and where the remainder is read off, and **`q-drift` is the cell a shortfall is
#: claimed in**.  That is the design note's *"Tier 1 may be a remainder anchor"* arriving in
#: one cell of two rather than in the tier, and it is a property of the estimand's setting
#: rather than of this instrument.
G_DRIFT_B_ATE = 0.10

#: The floor each realised coefficient has to clear for the cell to have entered the regime it
#: claims.  Not a tuning knob: it exists so that "the drift is nonzero" is an assertion with a
#: number behind it, at a value two orders below the targets above.
C_MIN = 0.02


#: How wrong the ``g-drift`` cell's outcome regression is, at arm 1, as a function of ``W``;
#: arm 0 gets :data:`G_DRIFT_ARM0_RATIO` times it.  Bounded through ``tanh`` so the injected
#: :math:`\hat Q` cannot leave :data:`Q_BOUNDS` however the covariates fall, and bounded
#: **away from zero** so the coefficients below cannot vanish through the error itself
#: disappearing.
def outcome_error(w: Any) -> Any:
    """``d(W) in [0.5, 1.5]``, the fixed outcome misspecification of the ``g-drift`` cell."""
    latent = np.asarray(w, dtype=float)
    return 1.0 + 0.5 * np.tanh(latent[:, 0])


#: Arm 0's share of :func:`outcome_error`.  Not ``1``: an error identical at both arms leaves
#: the *plug-in* contrast :math:`E[\hat Q_1 - \hat Q_0]` exactly right, which would make the
#: ATE cell's outcome regression wrong in a way no contrast could see -- a degenerate case
#: dressed as a general one.  Not ``0`` either, which would leave arm 0's mean undrifted.
G_DRIFT_ARM0_RATIO = 0.5


def base_law() -> DGP:
    """The law both cells are drawn from -- see the module docstring on why it is the easy one."""
    return linear_dgp()


def _mechanism(dgp: DGP, w: Any) -> Any:
    """``g_0(1 | W)``, at the latent matrix the process defines it on."""
    return np.asarray(dgp.propensity(np.asarray(w, dtype=float)), dtype=float)


def _wrong_mechanism(dgp: DGP, w: Any) -> Any:
    """``g_1(1 | W)``: the ``q-drift`` cell's prescribed limit, wrong by a log-odds shift."""
    return expit(logit(_mechanism(dgp, w)) + G_LOGIT_SHIFT)


def _arm(values: Any, arm: float) -> Any:
    """``P(A = arm | W)`` from the arm-1 column, by complement -- the binary path's own rule."""
    return values if arm == 1.0 else 1.0 - values


def _weight(dgp: DGP, w: Any, arm: float) -> Any:
    r""":math:`u_a = (\hat g_a - g_{0,a})/\hat g_a`, the ``q-drift`` cell's remainder weight.

    Single-signed at each arm because :data:`G_LOGIT_SHIFT` is a shift: positive at arm 1,
    negative at arm 0, everywhere.
    """
    wrong = _arm(_wrong_mechanism(dgp, w), arm)
    return (wrong - _arm(_mechanism(dgp, w), arm)) / wrong


# ------------------------------------------------------- what targeting does to the drift
#
# Everything from here to `targeted_coefficients` exists because of one finding: the column
# this module was built to report is *not* the quantity a fit's bias is, and the difference
# is the targeting step.  See `docs/drtmle/coverage-study.md`'s repair section.


def _limit_mechanism(cell: str, w: Any, arm: float) -> Any:
    r""":math:`\lim \hat g_a`: the wrong mechanism in ``q-drift``, the true one in ``g-drift``.

    The **limit** rather than the value at a finite ``n``, which is what "coefficient" means
    throughout this module -- :func:`drift_coefficients` says so in its own docstring and
    this has to agree with it or the two columns would be about different sequences.
    """
    dgp = base_law()
    values = _wrong_mechanism(dgp, w) if cell == "q-drift" else _mechanism(dgp, w)
    return _arm(values, arm)


def _limit_outcome(cell: str, w: Any, arm: float) -> Any:
    r""":math:`\lim \hat Q_a`, on the outcome's own scale.

    :math:`\bar Q_{0,a}` in ``q-drift``, where the injected drift vanishes; the **wrong**
    regression :math:`\bar Q_{0,a} + d_a` in ``g-drift``, where the outcome is the misspecified
    nuisance and its error does not shrink.  Getting this backwards would evaluate the
    fluctuation's direction at a regression no fit ever holds.
    """
    truth = np.asarray(base_law().outcome_mean(np.asarray(w, dtype=float), arm, None), dtype=float)
    if cell == "q-drift":
        return truth
    ratio = 1.0 if arm == 1.0 else G_DRIFT_ARM0_RATIO
    return truth + ratio * outcome_error(w)


def _score_weight(cell: str, w: Any, arm: float) -> Any:
    r""":math:`w_a = g_{0,a}/\hat g_a`, the weight the fluctuation's **score** is taken under.

    Not :func:`_weight`, and the pair is the whole of the repair: :math:`u_a = 1 - w_a` is the
    weight the *remainder* is an inner product against, and targeting drives the :math:`w_a`
    one to zero.  A design that makes only the first large has constrained nothing.
    """
    return _arm(_mechanism(base_law(), w), arm) / _limit_mechanism(cell, w, arm)


def _fluctuation_direction(cell: str, w: Any, arm: float) -> Any:
    r""":math:`S_a`, the direction the single :math:`\varepsilon_a` moves :math:`\bar Q_a` in.

    The ``mean`` submodel is logistic on the scaled outcome with the **counterfactual** clever
    covariate :math:`1/\hat g_a` (``cleverly.fluctuation.submodel``'s ``mean_submodel``), so
    :math:`\bar Q^*_a = \mathrm{expit}(\mathrm{logit}\,\hat Q^{sc}_a + \varepsilon_a/\hat g_a)`
    and its derivative at :math:`\varepsilon_a = 0` is
    :math:`\hat Q^{sc}_a(1 - \hat Q^{sc}_a)/\hat g_a`, times :attr:`OutcomeScaler.range` to
    come back to the outcome's own scale.

    One free parameter per arm, so what it can absorb is **one dimension** of the injection --
    which is why the repair is a second linear condition and not a smaller constant.
    """
    scaler = OutcomeScaler(*Q_BOUNDS)
    scaled = scaler.scale(_limit_outcome(cell, w, arm))
    return (Q_BOUNDS[1] - Q_BOUNDS[0]) * scaled * (1.0 - scaled) / _limit_mechanism(cell, w, arm)


@cache
def _absorbed(cell: str) -> tuple[float, ...]:
    r"""The one scalar per arm the targeted weight is built from, in the order ``(1.0, 0.0)``.

    It is a **different** scalar in the two cells, and the asymmetry is structural rather
    than untidy: which nuisance drifts decides which factor of the remainder's inner product
    carries the :math:`n^{-\alpha}` and so which of them the free shape is.

    ``q-drift`` returns :math:`\kappa_a = P_0[S_a]/P_0[w_a S_a]`.  The free shape is
    :math:`h_a` itself, so an :math:`\varepsilon_a` computed from it would be circular --
    :math:`\kappa_a` is what factors the fluctuation out of the shape, and it is why the
    repair below can be a linear solve at all.

    ``g-drift`` returns :math:`\varepsilon_a = -P_0[w_a d_a]/P_0[w_a S_a]` outright.  There the
    free shape is the *mechanism* perturbation and the outcome error :math:`d_a` is fixed, so
    the fluctuation's step is a constant of the design and nothing is circular.

    Cached beside :func:`_normalisers` and for its reason: integrations over ``2**18`` points
    that every coefficient needs.  Keyed by the cell alone, which is sound because every
    factor is a **limit** -- nothing here depends on ``n``, and an edit that made one depend
    on it would have to move the key.
    """
    dgp = base_law()
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")
    out = []
    for arm in (1.0, 0.0):
        denominator = dgp.expectation(
            lambda w, a=arm: _score_weight(cell, w, a) * _fluctuation_direction(cell, w, a)
        )
        if cell == "q-drift":
            numerator = dgp.expectation(lambda w, a=arm: _fluctuation_direction(cell, w, a))
            out.append(numerator / denominator)
        else:
            ratio = 1.0 if arm == 1.0 else G_DRIFT_ARM0_RATIO
            numerator = dgp.expectation(
                lambda w, a=arm, r=ratio: _score_weight(cell, w, a) * r * outcome_error(w)
            )
            out.append(-numerator / denominator)
    return tuple(out)


def plugin_weight(cell: str, w: Any, arm: float) -> Any:
    r"""What the cell's free shape multiplies in :math:`R_2(\hat Q)`, the **plug-in** remainder.

    :math:`u_a = (\hat g_a - g_{0,a})/\hat g_a` in ``q-drift``, where the free shape is the
    injected :math:`h_a`; the fixed outcome error :math:`d_a` in ``g-drift``, where the free
    shape is the mechanism perturbation.  In both cells
    :math:`c_a = P_0[\text{plugin} \cdot \text{shape}]`, which is the coefficient
    :func:`drift_coefficients` has always reported.
    """
    if cell == "q-drift":
        return _weight(base_law(), w, arm)
    if cell == "g-drift":
        ratio = 1.0 if arm == 1.0 else G_DRIFT_ARM0_RATIO
        return ratio * outcome_error(w)
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


def targeted_weight(cell: str, w: Any, arm: float) -> Any:
    r"""What the cell's free shape multiplies in :math:`R_2(\bar Q^*)`, the **estimator's bias**.

    The derivation is three lines.  The fluctuation's population score is
    :math:`P_0[w_a(\bar Q^*_a - \bar Q_{0,a})] = 0`; the bias is
    :math:`R_2(\bar Q^*_a) = P_0[u_a(\bar Q^*_a - \bar Q_{0,a})]` with :math:`u_a = 1 - w_a`,
    so the score kills the :math:`w_a` half.  Writing the offset as
    :math:`(\hat Q_a - \bar Q_{0,a}) + \varepsilon_a S_a` and eliminating
    :math:`\varepsilon_a` through the score leaves, per cell:

    - ``q-drift``: :math:`v_a = 1 - \kappa_a w_a`, against the shape :math:`h_a`;
    - ``g-drift``: :math:`r_a = d_a + \varepsilon_a S_a`, against the shape
      :math:`\tilde u_a = \lim n^{\alpha}u_a`, since there it is the *mechanism* that drifts.

    Either way :math:`b_a = P_0[\text{targeted} \cdot \text{shape}]` is a **linear functional
    of the free shape**, exactly as :math:`c_a` is -- which is what makes the repair a 2x2
    solve rather than a redesign, and what turns *"the drift survives targeting"* from a
    property to hope for into a condition a design can be built to satisfy.

    Neither weight is identically zero: :math:`v_a` vanishes only if :math:`w_a` is constant,
    i.e. only if :math:`\hat g_a \propto g_{0,a}`.  That is the numerical form of *"Tier 1 can
    be a demonstration"*, and ``tests/unit/test_drtmle_coverage.py`` **measures** it rather
    than asserting it -- the design note's live alternative was that no injection into a
    single nuisance produces a first-order shortfall at all.
    """
    scalar = dict(zip((1.0, 0.0), _absorbed(cell), strict=True))[arm]
    if cell == "q-drift":
        return 1.0 - scalar * _score_weight(cell, w, arm)
    if cell == "g-drift":
        return plugin_weight(cell, w, arm) + scalar * _fluctuation_direction(cell, w, arm)
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


def free_shape(cell: str, w: Any, arm: float) -> Any:
    r"""The cell's free shape, with its :math:`n^{-\alpha}` divided out.

    :math:`h_a` in ``q-drift``; :math:`\tilde u_a = \lim n^{\alpha}(\hat g_a - g_{0,a})/\hat g_a`
    in ``g-drift``, which is the mechanism perturbation over the limit mechanism and carries
    the arm-0 sign flip -- one free column and its complement, which is why that cell can set
    the ATE's coefficients and not each arm's.

    **Both coefficient functions read this**, so a change to the injection cannot move one
    column without moving the other.  That duplication used to sit between
    :func:`outcome_perturbation` and :func:`drift_coefficients`, where a test asserting the two
    agree could be satisfied by two edits made the same wrong way.
    """
    if cell == "q-drift":
        return _outcome_shape(cell, w, arm)
    if cell == "g-drift":
        g = _mechanism(base_law(), w)
        # The arm-0 mechanism moves by *minus* the arm-1 perturbation: one free column and
        # its complement.  Getting this sign wrong turns c_ATE from a sum of magnitudes into
        # a difference, which is the cancellation the design exists to make impossible.
        delta = _mechanism_shape(cell, w)
        return (delta if arm == 1.0 else -delta) / _arm(g, arm)
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


def _ate_representers(w: Any) -> tuple[Any, Any]:
    r"""``g-drift``'s two conditions, as functions the free shape is integrated against.

    That cell's mechanism has **one** free function :math:`\mu`, since the estimator reads
    :math:`\hat g(1|W)` off a classifier and takes the complement, so what can be set is one
    combination per condition rather than one per arm.  Writing
    :math:`\hat g - g_0 = n^{-\alpha}\mu g_0(1-g_0)` and summing the arms,

    .. math::

        c_{ATE} = P_0[\mu D], \quad D = (1-g_0)d_1 + g_0 d_0, \qquad
        b_{ATE} = P_0[\mu R], \quad R = (1-g_0)r_1 + g_0 r_0

    with :math:`d_a` and :math:`r_a` the two weights of :func:`plugin_weight` and
    :func:`targeted_weight`.  So the ATE's two coefficients are two linear conditions on one
    function, exactly as ``q-drift``'s are two on each arm's.
    """
    g = _mechanism(base_law(), w)
    return (
        (1.0 - g) * plugin_weight("g-drift", w, 1.0) + g * plugin_weight("g-drift", w, 0.0),
        (1.0 - g) * targeted_weight("g-drift", w, 1.0) + g * targeted_weight("g-drift", w, 0.0),
    )


@cache
def _shape_multipliers(cell: str) -> tuple[tuple[float, float], ...]:
    r"""The free shape's two multipliers, so that **both** coefficients are the declared ones.

    The design used to impose one condition -- align the shape with the plug-in weight and
    scale it until :math:`c` came out at its declared value -- and
    ``docs/drtmle/coverage-study.md``'s repair section is the written reason for imposing a
    second: a shape chosen only that way leaves :math:`b`, the coefficient a fit's bias
    actually has, unconstrained, and it came out 436 times smaller.

    So the shape is placed in the span of the two conditions' own representers,

    .. math::  \text{shape} = \lambda_1 \cdot \text{plug-in} + \lambda_2 \cdot \text{targeted}

    and :math:`(\lambda_1, \lambda_2)` solve a 2x2 system whose matrix is the **Gram** of that
    pair.  The old design is the :math:`\lambda_2 = 0` special case in ``q-drift``, where the
    plug-in representer *is* the shape it used, so this is a generalisation rather than a
    replacement -- and the span is the minimum-norm choice, which matters because the injected
    perturbation has to stay inside :data:`Q_BOUNDS` after scaling.

    Returned per arm in the order ``(1.0, 0.0)`` for ``q-drift`` and as a **one-tuple** for
    ``g-drift``, whose single free function can set only the ATE's pair.  Cached for
    :func:`_absorbed`'s reason: integrations over ``2**18`` points that every shape needs.  The
    key is the cell alone, which is sound because every factor is a limit and none depends on
    ``n``.
    """
    dgp = base_law()
    if cell == "q-drift":
        return tuple(
            _solve_conditions(
                lambda w, a=arm: plugin_weight(cell, w, a),
                lambda w, a=arm: targeted_weight(cell, w, a),
                Q_DRIFT_C[arm],
                Q_DRIFT_B[arm],
            )
            for arm in (1.0, 0.0)
        )
    if cell == "g-drift":
        del dgp
        return (
            _solve_conditions(
                lambda w: _ate_representers(w)[0],
                lambda w: _ate_representers(w)[1],
                G_DRIFT_C_ATE,
                G_DRIFT_B_ATE,
            ),
        )
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


def _solve_conditions(
    plugin: Callable[[Any], Any],
    targeted: Callable[[Any], Any],
    declared_c: float,
    declared_b: float,
) -> tuple[float, float]:
    """``(lambda_1, lambda_2)`` putting the shape's two coefficients at the declared pair.

    The matrix is symmetric because the basis *is* the pair of representers, so the solve is
    of a Gram matrix and is well conditioned exactly when the two conditions are not nearly
    the same condition.  ``tests/unit/test_drtmle_coverage.py`` measures that rather than
    assuming it -- a near-singular system would mean the fluctuation absorbs every direction
    the design can reach, which is the outcome under which Tier 1 is a remainder anchor and
    not a demonstration.
    """
    dgp = base_law()
    gram = np.array(
        [
            [
                dgp.expectation(lambda w: plugin(w) ** 2),
                dgp.expectation(lambda w: plugin(w) * targeted(w)),
            ],
            [
                dgp.expectation(lambda w: plugin(w) * targeted(w)),
                dgp.expectation(lambda w: targeted(w) ** 2),
            ],
        ]
    )
    solved = np.linalg.solve(gram, np.array([declared_c, declared_b]))
    return (float(solved[0]), float(solved[1]))


def _outcome_shape(cell: str, w: Any, arm: float) -> Any:
    r"""The outcome regression's error shape, with any :math:`n^{-\alpha}` divided out.

    ``q-drift``'s :math:`h_a` -- the **free** shape of that cell, and what the repair chooses;
    ``g-drift``'s fixed :math:`d_a`, which is that cell's misspecification and not free at all.
    """
    if cell == "q-drift":
        first, second = dict(zip((1.0, 0.0), _shape_multipliers(cell), strict=True))[arm]
        return first * plugin_weight(cell, w, arm) + second * targeted_weight(cell, w, arm)
    if cell == "g-drift":
        ratio = 1.0 if arm == 1.0 else G_DRIFT_ARM0_RATIO
        return ratio * outcome_error(w)
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


def _mechanism_shape(cell: str, w: Any) -> Any:
    r"""The arm-1 mechanism perturbation with :math:`n^{-\alpha}` divided out.

    Zero in ``q-drift``.  In ``g-drift`` it is the **free** shape,
    :math:`\mu(W)g_0(1|W)g_0(0|W)`: the :math:`g_0(1-g_0)` factor is what keeps :math:`\hat g`
    interior wherever :math:`g_0` nears a boundary -- the perturbation vanishes with the
    probability it perturbs -- and it is why that cell needs no clipping and so has no
    truncation active.  :math:`\mu` is the combination :func:`_ate_representers` names, chosen
    so both of the ATE's coefficients come out declared.
    """
    if cell == "q-drift":
        return np.zeros(np.asarray(w, dtype=float).shape[0])
    if cell == "g-drift":
        ((first, second),) = _shape_multipliers(cell)
        plugin, targeted = _ate_representers(w)
        g = _mechanism(base_law(), w)
        return (first * plugin + second * targeted) * g * (1.0 - g)
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


def outcome_perturbation(cell: str, n: int, w: Any, arm: float) -> Any:
    r""":math:`\hat Q_a - \bar Q_{0,a}` at ``n`` rows, on the **outcome's own** scale.

    ``q-drift``'s is the drifting :math:`n^{-\alpha}h_a`; ``g-drift``'s is the fixed
    :func:`outcome_error`, which does not shrink because in that cell the outcome regression is
    the *wrong* nuisance.  Which cell carries the :math:`n^{-\alpha}` is the whole difference
    between them, which is why it lives here rather than in :func:`_outcome_shape`.
    """
    scale = float(n) ** -ALPHA if cell == "q-drift" else 1.0
    return scale * _outcome_shape(cell, w, arm)


def mechanism_perturbation(cell: str, n: int, w: Any) -> Any:
    r""":math:`\hat g(1|W) - g_0(1|W)` at ``n`` rows.

    Zero in ``q-drift``, where the mechanism is the wrong nuisance and is handled by
    :func:`_wrong_mechanism` instead; the drifting :math:`n^{-\alpha}` shape in ``g-drift``.
    """
    return float(n) ** -ALPHA * _mechanism_shape(cell, w)


def injected_mechanism(cell: str, n: int, w: Any) -> Any:
    """``g-hat(1 | W)`` for a cell at a size: the wrong limit, or the truth plus a drift."""
    dgp = base_law()
    if cell == "q-drift":
        return _wrong_mechanism(dgp, w)
    return _mechanism(dgp, w) + mechanism_perturbation(cell, n, w)


def injected_outcome(cell: str, n: int, w: Any, arm: float) -> Any:
    """``Q-hat(a, W)`` for a cell at a size, on the outcome's own scale."""
    truth = np.asarray(base_law().outcome_mean(np.asarray(w, dtype=float), arm, None), dtype=float)
    return truth + outcome_perturbation(cell, n, w, arm)


# ------------------------------------------------------------------ the learners


class InjectedOutcome(BaseEstimator):
    """The outcome regression a cell prescribes, as a learner the estimator can be handed.

    Ignores its training rows entirely, exactly as ``tests/conftest.py``'s oracles do, and
    returns the prescribed function through the **declared** scaler rather than a recovered
    one -- see the module docstring on why that distinction is the same order as the drift.

    ``cell`` and ``n`` are stored under their own names so ``sklearn.base.clone`` reproduces
    the learner per fold; ``n`` is the *study's* sample size and not the training fold's,
    because the sequence is indexed by the size of the sample the estimator was handed.
    """

    def __init__(self, cell: str, n: int, q_bounds: tuple[float, float] = Q_BOUNDS) -> None:
        self.cell = cell
        self.n = n
        self.q_bounds = q_bounds

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> InjectedOutcome:
        del X, y, sample_weight
        return self

    def predict(self, X: Any) -> Any:
        design = np.asarray(X, dtype=float)
        arms, covariates = design[:, 0], design[:, 1:]
        scaler = OutcomeScaler(*self.q_bounds)
        one = scaler.scale(injected_outcome(self.cell, self.n, covariates, 1.0))
        zero = scaler.scale(injected_outcome(self.cell, self.n, covariates, 0.0))
        values = np.where(arms == 1.0, one, zero)
        if values.min() <= 0.0 or values.max() >= 1.0:
            # Clipping here would distort the drift by an unrecorded amount, which is the one
            # thing this module cannot afford: the injected perturbation *is* the measurement.
            raise ValueError(
                f"the injected outcome regression leaves (0, 1) on the scaled scale at "
                f"[{values.min():.4g}, {values.max():.4g}]; widen Q_BOUNDS rather than "
                "clipping, which would distort the drift the study reads"
            )
        return values


class InjectedMechanism(BaseEstimator):
    """The treatment mechanism a cell prescribes, as a classifier the estimator can be handed.

    Returns ``[1 - g, g]`` against ``classes_ = [0, 1]``, which is the shape
    :func:`~cleverly.estimators._nuisance.fit_nuisances` reads through
    ``predict_probabilities``.  One free column and its complement, because that is what a
    binary mechanism is here -- and it is why the ``g-drift`` cell can target the ATE's drift
    coefficient and not each arm's.
    """

    def __init__(self, cell: str, n: int) -> None:
        self.cell = cell
        self.n = n

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> InjectedMechanism:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> Any:
        values = injected_mechanism(self.cell, self.n, np.asarray(X, dtype=float))
        if values.min() <= 0.0 or values.max() >= 1.0:
            raise ValueError(
                f"the injected mechanism leaves (0, 1) at [{values.min():.4g}, "
                f"{values.max():.4g}]; the perturbation carries a g(1 - g) factor precisely "
                "so this cannot happen, so this is a design error rather than a hard draw"
            )
        return np.column_stack([1.0 - values, values])


# ------------------------------------------------------------------ what the law says


def drift_coefficients(cell: str) -> dict[str, float]:
    r"""The realised :math:`c_1`, :math:`c_0` and :math:`c_{ATE}` for a cell, by quadrature.

    :math:`c_a = P_0[(\hat g_a - g_{0,a})/\hat g_a \cdot h_a]` with the drifting factor
    :math:`n^{-\alpha}` divided out, so these are the constants
    :func:`exact_remainder` scales -- and the numbers ``Q_DRIFT_C`` and ``G_DRIFT_C_ATE``
    declare.  Evaluated at the *limit* mechanism in each cell, which is the wrong one in
    ``q-drift`` and the true one in ``g-drift``: that is what "coefficient" means, and reading
    it at a finite ``n``'s perturbed mechanism would fold the next order into it.

    Verified against the declaration rather than trusted: ``tests/unit/test_drtmle_coverage.py``
    asserts they agree and that each clears :data:`C_MIN`, which is §5's *"commit the
    coefficient calculation with the design"* made checkable.

    **It is not the coefficient a fit's bias has**, which is
    :func:`targeted_coefficients` -- this one is of the *plug-in* remainder, and at the design
    C3b replaced the two came out a factor of 436 apart.  They are one expression against two
    weights and both are now declared, which is what §5's targeted-coefficient clause asks for:
    the plug-in column says the injection is what it claims, and the targeted one says the
    regime was entered.
    """
    return _coefficients(cell, plugin_weight)


def targeted_coefficients(cell: str) -> dict[str, float]:
    r"""The realised :math:`b_1`, :math:`b_0` and :math:`b_{ATE}`: the **estimator's bias**.

    :math:`b_a = P_0[v_a \cdot \text{shape}_a]` against :func:`targeted_weight`, with the
    drifting :math:`n^{-\alpha}` divided out, so that
    :math:`\hat\psi - \psi_0 = (P_n - P_0)D^* + n^{-\alpha}b_{ATE} + o(n^{-\alpha})` and the
    first term is mean-zero across draws.

    **This is the column C3a's pilot found missing**, and its absence is the whole of what went
    wrong: the design normalised :func:`drift_coefficients` to ``0.40`` and sized a coverage
    shortfall from it, while the quantity a shortfall is made of came out at ``0.00092``
    because the fluctuation's one free parameter per arm absorbed 98.6% of the injection.  C3b
    makes it a *declared* coefficient rather than whatever falls out.  See
    ``docs/drtmle/coverage-study.md``'s repair section, and
    ``benchmarks/drtmle_tier1_bias.py`` for the measurement on real fits.
    """
    return _coefficients(cell, targeted_weight, key="b")


def _coefficients(cell: str, weight: Callable[..., Any], key: str = "c") -> dict[str, float]:
    """One quadrature, called at the two weights -- see the two functions above.

    Written once deliberately.  Two copies of this integral would be two chances for a
    coefficient and the column that reads it to disagree somewhere other than in the weight
    under test, which is the class of mistake this whole piece is a repair for.
    """
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")
    dgp = base_law()
    per_arm = {
        arm: dgp.expectation(lambda w, a=arm: weight(cell, w, a) * free_shape(cell, w, a))
        for arm in (1.0, 0.0)
    }
    return {
        f"{key}1": per_arm[1.0],
        f"{key}0": per_arm[0.0],
        f"{key}_ate": per_arm[1.0] - per_arm[0.0],
    }


def exact_remainder(cell: str, n: int) -> dict[str, float]:
    r"""The plug-in remainder at the injected sequence, integrated rather than simulated.

    .. math::

        R_{2,a} = P_0\!\left[\frac{\hat g_a - g_{0,a}}{\hat g_a}
                             (\hat Q_a - \bar Q_{0,a})\right],
        \qquad R_{2,ATE} = R_{2,1} - R_{2,0}

    Exact because both nuisances are prescribed functions of ``W`` -- §5's Tier-1 convention,
    with *"no model retention needed, because the nuisance sequence is prescribed"*.  This is
    the regime-entry quantity: :math:`n^{\alpha}R_{2,a} \to c_a`, which
    ``tests/unit/test_drtmle_coverage.py`` checks across sizes, and it is what the study
    reports in place of inferring the regime from a nuisance norm.

    It is **not** ``R_remaining``, the doubly-robust curve's remainder, which needs
    :math:`P_0\hat D` at the *fitted* reduced regressions and so the retained per-fold
    nuisances of piece C2.
    """
    dgp = base_law()
    out = {}
    for arm in (1.0, 0.0):

        def integrand(w: Any, arm: float = arm) -> Any:
            estimated = _arm(injected_mechanism(cell, n, w), arm)
            truth = _arm(_mechanism(dgp, w), arm)
            return (estimated - truth) / estimated * outcome_perturbation(cell, n, w, arm)

        out[f"r2_{int(arm)}"] = dgp.expectation(integrand)
    out["r2_ate"] = out["r2_1"] - out["r2_0"]
    return out


def population_epsilon(cell: str, n: int, arm: float) -> float:
    r"""The fluctuation's step at the **population** score, solved rather than linearised.

    :math:`\varepsilon_a` such that

    .. math::

        P_0\!\left[\frac{g_{0,a}}{\hat g_a}
                   \bigl(\bar Q^{sc}_{0,a}
                         - \mathrm{expit}(\mathrm{logit}\,\hat Q^{sc}_a
                                          + \varepsilon_a/\hat g_a)\bigr)\right] = 0,

    which is what ``cleverly``'s Newton solve converges to as ``n`` grows.  Solved exactly by a
    bracketed root find over the same Sobol rule rather than by the first-order expansion
    :func:`targeted_weight` factors :math:`\varepsilon_a` out with, because the two differ by
    the curvature of the logistic submodel -- measured at 20% of the step at ``n = 600``.

    So :func:`targeted_coefficients` is the **limit** and :func:`exact_targeted_remainder` is
    the prediction at a size, which is exactly the pair :func:`drift_coefficients` and
    :func:`exact_remainder` already are.
    """
    dgp = base_law()
    scaler = OutcomeScaler(*Q_BOUNDS)

    def score(epsilon: float) -> float:
        def integrand(w: Any) -> Any:
            ghat = _arm(injected_mechanism(cell, n, w), arm)
            initial = scaler.scale(injected_outcome(cell, n, w, arm))
            targeted = expit(logit(initial) + epsilon / ghat)
            truth = scaler.scale(
                np.asarray(base_law().outcome_mean(np.asarray(w, float), arm, None), float)
            )
            return _arm(_mechanism(dgp, w), arm) / ghat * (truth - targeted)

        return float(dgp.expectation(integrand))

    # The score is strictly decreasing in epsilon (the submodel is monotone and the weight is
    # positive), so any bracket that changes sign contains the one root.  Widening rather than
    # guessing: a design whose step needs more than this has left the regime it claims.
    bound = 1.0
    while score(-bound) * score(bound) > 0.0:
        bound *= 4.0
        if bound > 1024.0:
            raise ValueError(
                f"no population root for {cell!r} at n={n}, arm {arm} within +/-1024; the "
                "injected regression is too far from the truth for a one-parameter submodel"
            )
    return float(brentq(score, -bound, bound, xtol=1e-14, rtol=1e-15))


def committed_coefficient(cell: str) -> float:
    r"""What the pre-flight reads condition 1 against: :math:`b_{ATE}`, declared.

    Identical to :func:`targeted_coefficients`' ``b_ate`` **by construction** here, because the
    injected shape is solved to make it so -- which is exactly the difference between this tier
    and Tier 2, where the same name returns a *measured* constant because a fitted nuisance's
    bias is what it is.  The two tiers supply one name so the harness reads both without
    branching, and each says in its own docstring what kind of number it is returning.
    """
    return targeted_coefficients(cell)["b_ate"]


def exact_targeted_remainder(cell: str, n: int) -> dict[str, float]:
    r"""The **estimator's bias** at the injected sequence, integrated rather than simulated.

    .. math::

        R_{2,a}(\bar Q^*) = P_0\!\left[\frac{\hat g_a - g_{0,a}}{\hat g_a}
                                       (\bar Q^*_a - \bar Q_{0,a})\right]

    -- the same expression :func:`exact_remainder` integrates, at the **targeted** regression
    rather than the initial one, with :math:`\varepsilon_a` from :func:`population_epsilon`.
    :math:`n^{\alpha}R_{2,a}(\bar Q^*) \to b_a`.

    This is the column §5's targeted-coefficient clause requires be read against the declared
    coefficient, and the one a coverage shortfall is sized from:
    :math:`\hat\psi - \psi_0 = (P_n - P_0)D^* + R_2(\bar Q^*)`, whose first term is mean-zero
    across draws.  Reading :func:`exact_remainder` in its place is what C3's pilot cost.
    """
    dgp = base_law()
    scaler = OutcomeScaler(*Q_BOUNDS)
    out = {}
    for arm in (1.0, 0.0):
        epsilon = population_epsilon(cell, n, arm)

        def integrand(w: Any, arm: float = arm, epsilon: float = epsilon) -> Any:
            estimated = _arm(injected_mechanism(cell, n, w), arm)
            truth = _arm(_mechanism(dgp, w), arm)
            initial = scaler.scale(injected_outcome(cell, n, w, arm))
            targeted = scaler.unscale_level(expit(logit(initial) + epsilon / estimated))
            reference = np.asarray(dgp.outcome_mean(np.asarray(w, float), arm, None), float)
            return (estimated - truth) / estimated * (targeted - reference)

        out[f"r2_{int(arm)}"] = dgp.expectation(integrand)
    out["r2_ate"] = out["r2_1"] - out["r2_0"]
    return out


def nuisance_error(cell: str, n: int) -> dict[str, float]:
    r"""``L2(P_0)`` distance from each injected nuisance to the truth, per arm and pooled.

    The columns §5 calls *"verifying the regime was entered"*: the drifting nuisance's norm
    has to fall at :math:`n^{-\alpha}` and the misspecified one's has to stay **bounded away
    from zero**, and a study that reported only the first could not tell a shrinking product
    from a converging pair.  Both are exact here, which is the point of the tier.
    """
    dgp = base_law()
    outcome = {
        arm: np.sqrt(dgp.expectation(lambda w, arm=arm: outcome_perturbation(cell, n, w, arm) ** 2))
        for arm in (1.0, 0.0)
    }

    def mechanism(w: Any) -> Any:
        return (injected_mechanism(cell, n, w) - _mechanism(dgp, w)) ** 2

    return {
        "q_error_1": float(outcome[1.0]),
        "q_error_0": float(outcome[0.0]),
        "g_error": float(np.sqrt(dgp.expectation(mechanism))),
    }


def settings(cell: str, n: int) -> dict[str, Any]:
    """The estimator keywords a cell's fits share, injected nuisances included.

    ``reduced_outcome_learner`` and ``reduced_treatment_learner`` are named **explicitly** and
    that is not tidiness: ``DRTMLE``'s reduced regressions default to the primary
    *specification*, so leaving them off would hand this cell's injected learner to
    :math:`Q_r`, :math:`g_{r,1}` and :math:`g_{r,2}` -- which is the refusal that class's own
    docstring warns about (*"a learner instance built for classification cannot serve*
    :math:`Q_r`"), and which would make the reductions prescribed rather than fitted and the
    whole estimator a different object.
    """
    return {
        "outcome_learner": InjectedOutcome(cell, n),
        "treatment_learner": InjectedMechanism(cell, n),
        "q_bounds": Q_BOUNDS,
        "n_folds": 5,
        "learner_folds": 3,
        "simultaneous": False,
        "estimands": ("ate", "ey1", "ey0"),
    }


def summary_rows() -> list[list[str]]:
    """One row per cell: what the design committed to, so a run prints it beside its numbers.

    **Both** coefficient sets, because §5 asks a run to report both and to read the *targeted*
    one against the regime-entry column.  A table carrying only the plug-in pair is the table
    C3a's pilot was read off.
    """
    rows = []
    for cell in CELLS:
        plugin = drift_coefficients(cell)
        targeted = targeted_coefficients(cell)
        rows.append(
            [
                cell,
                f"{ALPHA:.2f}",
                f"{plugin['c1']:+.4f}",
                f"{plugin['c0']:+.4f}",
                f"{plugin['c_ate']:+.4f}",
                f"{targeted['b1']:+.4f}",
                f"{targeted['b0']:+.4f}",
                f"{targeted['b_ate']:+.4f}",
                f"{min(abs(targeted[key]) for key in ('b1', 'b0', 'b_ate')):.4f}",
            ]
        )
    return rows


SUMMARY_HEADERS: tuple[str, ...] = (
    "cell",
    "alpha",
    "c1",
    "c0",
    "c_ate",
    "b1",
    "b0",
    "b_ate",
    "min |b|",
)

#: What a caller reads a shape off, kept beside the functions so the two cannot drift.
SHAPES: dict[str, Callable[..., Any]] = {
    "outcome": outcome_perturbation,
    "mechanism": mechanism_perturbation,
}
