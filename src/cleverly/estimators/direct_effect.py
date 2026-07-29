r"""What ``intermediate=`` estimates, and the data structure that makes it a TMLE.

Passing ``intermediate=`` to :meth:`~cleverly.TMLE.fit` changes the *estimand*, not merely
the estimator, and it does so in a way that is easy to state loosely and hard to state
correctly.  "Controlled direct effect" names a family of parameters whose efficient
influence functions range from the two-term expression derived below to a sequence of
nested regressions spanning several likelihood factors.  Which one applies is decided
entirely by the observed-data structure.  This module writes down the structure this
package assumes, the parameter it therefore estimates, the efficient influence function of
that parameter, and -- explicitly -- the neighbouring problems it does **not** solve.

The estimand
------------

The observed data are :math:`O = (W, A, Z, \Delta, \Delta Y)` drawn i.i.d. from
:math:`P_0`, in a nonparametric model.  :math:`W` are baseline covariates realised before
:math:`A`; :math:`A \in \{0, 1\}` is the treatment; :math:`Z \in \{0, 1\}` is a variable
realised *after* :math:`A` and before :math:`Y`; :math:`\Delta` records whether :math:`Y`
was observed.  **Nothing else is measured between :math:`A` and :math:`Z`** -- that
sentence is the whole of the theory below, and :ref:`the scope section <cde-scope>` says
what happens when it is false.

For a fixed arm :math:`a` and a fixed level :math:`z`, define

.. math::

    \Psi_{a,z}(P) \;=\; E_{P}\Bigl[\, E_{P}\bigl(Y \mid A = a,\, Z = z,\, \Delta = 1,\,
        W\bigr) \Bigr] \;=\; E_P\bigl[\bar Q(a, z, W)\bigr],

the outer expectation being over the marginal law of :math:`W` under :math:`P`.  The
controlled direct effect at :math:`z` is the contrast

.. math:: \Psi^{\mathrm{CDE}}_z(P) = \Psi_{1,z}(P) - \Psi_{0,z}(P).

Note the subscript.  A controlled direct effect is a *different parameter for each*
:math:`z` -- "the effect of :math:`A` on :math:`Y` with :math:`Z` held at 0" and "... held
at 1" are two questions, and they have the same answer only when :math:`A` and :math:`Z`
do not interact in :math:`\bar Q`.  That is why :meth:`~cleverly.TMLE.fit` returns a
:class:`~cleverly.estimators.base.TMLEResultSet` rather than a single result: it fits both
levels and hands back one estimate per level.

Identification
--------------

:math:`\Psi_{a,z}(P_0) = E\bigl[Y(a, z)\bigr]` under

1. **Consistency**: :math:`Y = Y(A, Z)`, with no interference between units.
2. **No unmeasured confounding of** :math:`A`: :math:`Y(a, z) \perp A \mid W`.
3. **No unmeasured confounding of** :math:`Z`: :math:`Y(a, z) \perp Z \mid A, W`.
4. **Positivity**: :math:`g_a(W) = P(A = a \mid W)` and
   :math:`q_z(a, W) = P(Z = z \mid A = a, W)` are bounded away from zero, as is
   :math:`\pi_a(W) = P(\Delta = 1 \mid A = a, W)`.
5. **Missingness at random**: :math:`Y \perp \Delta \mid A, W`, and additionally
   :math:`\Delta \perp Z \mid A, W` -- see
   :meth:`~cleverly.data.causal_data.CausalData.missingness_design`, which states that
   assumption and why the missingness model deliberately excludes :math:`Z`.

Assumption 3 is the one that separates a controlled direct effect from an average
treatment effect, and it is the one this package previously never wrote down.  It says
that **the same baseline covariates that deconfound** :math:`A \to Y` **also deconfound**
:math:`Z \to Y`.  It is a strong requirement and it is not weakened by collecting more
data: a variable measured after :math:`A` cannot be used to satisfy it.  See below.

The efficient influence function
--------------------------------

Factorise the likelihood along the time ordering,

.. math::

    p(O) \;=\; p_W(W)\; g(A \mid W)\; q(Z \mid A, W)\; \pi(\Delta \mid A, W)\;
        p_Y(Y \mid A, Z, W, \Delta = 1),

which is an orthogonal decomposition of the tangent space of the nonparametric model into
five pieces.  Now read :math:`\Psi_{a,z}` off it: the parameter is an average of
:math:`\bar Q(a, z, \cdot)` -- a functional of :math:`p_Y` -- against the marginal
:math:`p_W`.  It does not depend on :math:`g`, on :math:`q`, or on :math:`\pi` at all.
Its canonical gradient therefore has components in exactly two of the five tangent spaces,
and both are standard:

.. math::

    D^*_{a,z}(O) \;=\;
      \underbrace{\frac{\mathbb 1\{A = a\}\, \mathbb 1\{Z = z\}\, \Delta}
                       {g_a(W)\, q_z(a, W)\, \pi_a(W)}
                  \bigl(Y - \bar Q(a, z, W)\bigr)}_{\text{score of } p_Y}
      \;+\; \underbrace{\bar Q(a, z, W) - \Psi_{a,z}}_{\text{score of } p_W}.

The weight on the residual is the Radon--Nikodym derivative of the target measure
:math:`dP_W` with respect to the measure the outcome data actually arrive under,
:math:`dP(A = a, Z = z, \Delta = 1, W)`; the indicators restrict the residual to the cell
where :math:`\bar Q(a, z, \cdot)` is identified.  This is what
:func:`~cleverly.fluctuation.submodel.mean_submodel` builds, and what
:func:`~cleverly.inference.influence.counterfactual_means` pairs with the residual.

Why there is no sequential regression here
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two-term form above is *not* the general shape of a controlled-direct-effect influence
function, and it is worth being precise about why it is the right one here rather than
merely asserting it.

Suppose a covariate :math:`L` were measured between :math:`A` and :math:`Z`, so that
:math:`O = (W, A, L, Z, Y)` and assumption 3 held given :math:`(A, W, L)` instead.  Then
:math:`E[Y(a,z)]` is a longitudinal g-formula parameter with two intervention nodes, its
identification runs through two nested regressions

.. math::

    \bar Q_2(a, z, W, L) = E[Y \mid A = a, Z = z, W, L],
    \qquad
    \bar Q_1(a, W) = E\bigl[\bar Q_2(a, z, W, L) \bigm| A = a, W\bigr],

and its efficient influence function carries a third term, one per intervention node:

.. math::

    D^*(O) = \frac{\mathbb 1\{A = a, Z = z\}}{g_a(W)\, q_z(a, W, L)}
                \bigl(Y - \bar Q_2\bigr)
           + \frac{\mathbb 1\{A = a\}}{g_a(W)}
                \bigl(\bar Q_2(a, z, W, L) - \bar Q_1(a, W)\bigr)
           + \bar Q_1(a, W) - \Psi .

Now set :math:`L = \emptyset`.  With no covariate between the two nodes,
:math:`\bar Q_2(a, z, W)` is already a function of :math:`W` alone, so the inner
conditional expectation defining :math:`\bar Q_1` integrates a :math:`W`-measurable
function against the conditional law of nothing:

.. math:: \bar Q_1(a, W) = E\bigl[\bar Q_2(a, z, W) \bigm| A = a, W\bigr]
          = \bar Q_2(a, z, W).

The middle term is therefore identically zero -- not small, not asymptotically
negligible, but zero at every :math:`P` and every :math:`O` -- and the third term
collapses onto the second of the two terms above.  The sequential regression is degenerate
because there is nothing for it to average over.  **One clever covariate suffices here
precisely because nothing is measured between** :math:`A` **and** :math:`Z`.

The corollary matters more than the theorem: the moment a real :math:`L` exists, the
estimator in this package is solving the wrong estimating equation, and no amount of
adding :math:`L` to ``covariates=`` repairs it.  See the scope section.

The submodel and the score it solves
------------------------------------

Targeting fluctuates the initial regression along

.. math:: \operatorname{logit} \bar Q^*_\epsilon(a, z, W)
          = \operatorname{logit} \bar Q^0(a, z, W) + \epsilon^\top h(a, z, W),
   \qquad
   h_a = \frac{\mathbb 1\{A = a\}\, \mathbb 1\{Z = z\}}{g_a(W)\, q_z(a, W)\, \pi_a(W)} ,

fitted by weighted binomial loss over the rows with :math:`\Delta = 1`.  The score of that
loss at :math:`\epsilon = 0` is :math:`\sum_{i: \Delta_i = 1} h_i (Y_i - \bar Q^0_i)`,
which is :math:`n` times the empirical mean of the first term of :math:`D^*`; the plug-in
:math:`\hat\Psi = P_n \bar Q^*(a, z, \cdot)` sets the empirical mean of the second term to
zero by construction.  Solving for :math:`\hat\epsilon` therefore gives
:math:`P_n D^*(\hat P) = 0`, which is the property the efficiency and double-robustness
arguments are built on.

Two implementation details follow from the display above rather than from convenience.
The indicator :math:`\mathbb 1\{Z_i = z\}` multiplies only the *observed* clever covariate
and not the counterfactual columns, because :math:`\bar Q^*(a, z, W_i)` is defined for
every unit while the residual exists only where :math:`Z_i = z`
(:mod:`cleverly.fluctuation.submodel` says the same thing about :math:`\Delta`).  And the
rows with :math:`Z_i \neq z` contribute an exact zero to the score rather than being
dropped, which is why the estimating equation is averaged over all :math:`n` rows.

Double robustness
-----------------

Expanding :math:`\Psi_{a,z}(\hat P) - \Psi_{a,z}(P_0) + P_0 D^*(\hat P)` gives the
second-order remainder

.. math::

    R_2 = \int \left(\frac{g_a\, q_z\, \pi_a}
                          {\hat g_a\, \hat q_z\, \hat\pi_a} - 1\right)
               \bigl(\bar Q_0 - \hat{\bar Q}\bigr)(a, z, w)\; dP_{W,0}(w),

a product of a mechanism error and an outcome-regression error.  The guarantee it buys is
therefore: **consistent if** :math:`\bar Q` **is right, or if the three-way product**
:math:`g\, q\, \pi` **is right.**

The obvious generalisation -- "consistent if any one of the four nuisances is right" -- is
false, for exactly the reason :mod:`cleverly.fluctuation.submodel` gives for the two-way
case: only the product appears in the estimating equation.  A correct propensity buys
nothing on its own when the intermediate mechanism is wrong, and errors in the three
mechanisms can cancel exactly.  Adding an intermediate variable adds a third factor to
that product, so it makes the mechanism half of the guarantee harder to earn, not easier.

Positivity, likewise, now has two ways to fail.  Overlap in :math:`g` can look immaculate
while :math:`q_z(a, W)` is near zero for a subpopulation -- and it is *asymmetric* in
:math:`z`, since the covariate divides by :math:`q_z` for the level being targeted and by
its complement at the other level.  ``res.sensitivity.positivity()`` reports the density
actually used at each level, and ``res.sensitivity.truncation_curve(mechanism=True)``
sweeps the bound applied to it.

.. _cde-scope:

What this is not
----------------

**Not a general longitudinal TMLE.**  The derivation above turns on the absence of any
covariate measured between :math:`A` and :math:`Z`.  If a variable :math:`L` exists that is
affected by :math:`A` and confounds :math:`Z \to Y` -- an *intermediate confounder*, the
usual reason a controlled direct effect is wanted in the first place -- then assumption 3
fails given :math:`(A, W)`, and the estimator here is inconsistent.  Adding :math:`L` to
``covariates=`` is not a repair and is strictly worse than leaving it out: :math:`L` is
post-treatment, so conditioning on it breaks assumption 2, the plug-in then averages
:math:`\bar Q(a, z, W, L)` over the *observed* marginal of :math:`L` rather than its
counterfactual law under :math:`A = a`, and the middle EIF term that would correct for
exactly this is missing from the estimating equation.  That data structure needs a
sequential estimator -- ``ltmle``, or a longitudinal ``tmle3`` specification -- and this
package does not implement one.  The package cannot detect the situation for you: whether
a column is baseline or post-treatment is knowledge about the study, not about the data.

**Not a natural direct or indirect effect.**  :math:`\Psi^{\mathrm{CDE}}_z` intervenes on
:math:`Z`, setting it to the same :math:`z` for everyone.  Natural (pure/total) direct and
indirect effects instead leave :math:`Z` at the level it would have taken under a
*different* arm, which requires a cross-world independence assumption that is not implied
by 1--5 and not testable even in a randomised trial.  Nothing here identifies them, and
:math:`\Psi^{\mathrm{CDE}}_1 - \Psi^{\mathrm{CDE}}_0` is an interaction contrast, not a
mediated effect: the two levels do not decompose the total effect into direct and indirect
parts.

**Not automatically a different answer at each level.**  The outcome design is
:math:`[A, W, Z]` in main effects (see
:meth:`~cleverly.data.causal_data.CausalData.treatment_design`); a controlled direct effect
varies with :math:`z` only through an :math:`A \times Z` interaction in :math:`\bar Q`.  A
learner that cannot represent one -- an additive ``"glm"``, for instance -- produces an
identical *initial* contrast at both levels, leaving the entire :math:`z`-dependence of the
answer to the targeting step.  That is not wrong, but it puts far more weight on the
mechanisms than a flexible library would; prefer a learner that can fit the interaction.

**Not collaboratively selected.**  :class:`~cleverly.estimators.ctmle.CTMLE` selects
covariates for :math:`g` against the targeted loss, and threads the intermediate level
through every candidate, so the selection is level-aware.  But :math:`q_z` itself is fitted
once and never enters the collaborative search, so the second denominator escapes that
machinery entirely.

State of the evidence
---------------------

The derivation above is an argument, not a machine-checked fact, and the distinction is
load-bearing in a library that elsewhere refuses to make it.  The average treatment effect
and its missing-outcome variant each have an exact proof in the suite: a finite-support law
a sample realises exactly (``tests/discrete_law.py``, ``tests/discrete_law_mar.py``), a
complex-step Gateaux derivative compared against the reported influence curve at machine
precision (``tests/unit/test_influence_gateaux.py``, ``...gateaux_mar.py``), and the
product form of :math:`R_2` checked against its closed form (``tests/unit/test_remainder.py``,
``...remainder_mar.py``).

**The controlled-direct-effect path has no such proof.**  Its verification is a
Monte-Carlo consistency check at a tolerance of 0.05 over eight replications
(``tests/e2e/test_missing_and_cde.py``) plus, per level, a coverage study that has to be
run deliberately (:class:`~cleverly.validation.simulation.CoverageStudy`, which now accepts
``intermediate_value=``).  Neither would distinguish the influence function above from a
subtly wrong one: as :mod:`cleverly.validation.score` notes, a wrong clever covariate used
*consistently* solves its own score equation to machine precision.  Constructing the
discrete-law counterpart -- support :math:`(w, a, z, \Delta y)`, the same complex-step
derivative, and negative controls against the total effect and against a
:math:`Z`-stratified functional -- is the outstanding item for this path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .._typing import FloatArray
from ..exceptions import DataError

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for type checkers
    from ..data.causal_data import CausalData
    from ._nuisance import NuisanceEstimates

__all__ = ["LEVELS", "check_level", "clever_covariate_inputs", "describe"]

#: The levels of the intermediate variable a controlled direct effect may be targeted at.
#: ``Z`` is validated binary on the way in (:func:`cleverly.data.validate.encode_binary`),
#: so these are the only two parameters the estimator can be asked for.
LEVELS: tuple[float, float] = (0.0, 1.0)


def check_level(value: float) -> float:
    """Validate a targeted intermediate level, returning it as a float.

    Kept as a function rather than an inline comparison because the failure it guards
    against is silent: the density accessor branches on ``value == 1.0`` and takes the
    complement otherwise, so an unrecognised level would quietly be treated as ``z = 0``
    and produce a plausible number for a parameter nobody asked for.
    """
    level = float(value)
    if level not in LEVELS:
        raise DataError(
            f"intermediate_value must be 0.0 or 1.0; got {value!r}. A controlled direct "
            "effect is defined per level of a binary intermediate, and Z is encoded to "
            "{0, 1} when the data is built."
        )
    return level


def clever_covariate_inputs(
    data: CausalData,
    nuisance: NuisanceEstimates,
    intermediate_value: float | None,
    lower: float,
) -> tuple[FloatArray | None, FloatArray | None]:
    r"""The two intermediate-specific arguments of the clever covariate.

    Returns ``(density, selection)``: the truncated :math:`q_z(a, W)` per arm, and the
    indicator :math:`\mathbb 1\{Z_i = z\}`.  Both are ``None`` when the data carries no
    intermediate variable, which is what the submodel builders expect for the ordinary
    point-treatment case.

    Raises rather than asserts on a missing level.  The check is the difference between
    estimating a controlled direct effect and estimating an average treatment effect with
    an extra covariate, so it has to survive ``python -O``.
    """
    if not data.has_intermediate:
        if intermediate_value is not None:
            raise DataError(
                f"intermediate_value={intermediate_value!r} was supplied but the data has "
                "no intermediate variable. Pass intermediate=<column> to fit(), or drop "
                "the level."
            )
        return None, None

    if intermediate_value is None:
        raise DataError(
            "the data carries an intermediate variable but no intermediate_value was "
            "supplied. A controlled direct effect is defined per level of Z, so the "
            "target is ambiguous without one; see cleverly.estimators.direct_effect."
        )
    level = check_level(intermediate_value)
    assert data.intermediate is not None  # implied by has_intermediate; for type checkers
    density = nuisance.intermediate_density(level, lower)
    selection = (data.intermediate == level).astype(float)
    return density, selection


def targeted_rows(data: CausalData, intermediate_value: float | None) -> FloatArray:
    """Boolean mask of the rows whose residual the clever covariate actually multiplies.

    A row contributes to the estimating equation only when its outcome was recorded *and*
    its intermediate took the level being targeted; every other row contributes an exact
    zero.  Diagnostics that summarise the weights the equation forms -- the effective
    sample size in :mod:`cleverly.sensitivity.positivity`, for one -- have to be taken
    over this set rather than over the complete cases, or they average in rows the
    mechanism never weighted.
    """
    mask = np.asarray(data.observed, dtype=bool)
    if not data.has_intermediate or intermediate_value is None:
        return mask
    assert data.intermediate is not None
    return mask & (data.intermediate == check_level(intermediate_value))


def describe(intermediate_value: float, name: str | None) -> str:
    """The estimand line reported in :meth:`~cleverly.estimators.base.TMLEResult.summary`.

    Lives beside the derivation so that the words a user reads and the parameter the code
    estimates cannot drift apart.
    """
    level = check_level(intermediate_value)
    return f"controlled direct effect at {name or 'Z'} = {level:.0f}"
