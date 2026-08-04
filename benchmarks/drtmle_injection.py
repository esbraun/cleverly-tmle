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
with both arm coefficients nonzero.  So :math:`h` is chosen **aligned with the misspecification
weight** and normalised so that the coefficients are the design's own constants
(:data:`Q_DRIFT_C`, :data:`G_DRIFT_C_ATE`) rather than whatever falls out: the arms are given
**opposite signs**, which makes cancellation in the ATE impossible rather than merely unlikely.
:func:`drift_coefficients` recomputes them from the law by quadrature and
``tests/unit/test_drtmle_coverage.py`` asserts the two agree, which is the verification §5 asks
for in place of inferring the regime from an :math:`L_2` rate.

What this module does **not** compute is :math:`P_0 \hat D` for the *doubly-robust* curve, and
so not ``R_remaining``.  The primary nuisances here are prescribed functions and integrate
exactly; the three reduced regressions are **fitted**, so evaluating their limit on new
covariates needs the fold-retained nuisance objects §5 puts in Tier 2.  Piece C2 built them as
``DRTMLE(evaluation=...)`` and ``benchmarks/drtmle_remainder.py`` is the arithmetic on top, so the
corrected remainder is available *here* too -- ``--evaluation-n`` is the knob, and item 13's rate
is C3's dispatch.  Everything here is about the *plug-in at the injected sequence*, which is
the regime-entry evidence -- and the targeting step moves :math:`\hat Q` by
:math:`O_p(n^{-1/2})`, which is smaller than the injected :math:`n^{-\alpha}` at every
:math:`\alpha < 1/2` and so leaves the drift's leading term where it was.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from cleverly.datasets import DGP, linear_dgp
from cleverly.utils.bounds import OutcomeScaler, expit, logit

__all__ = [
    "ALPHA",
    "CELLS",
    "G_DRIFT_C_ATE",
    "Q_BOUNDS",
    "Q_DRIFT_C",
    "InjectedMechanism",
    "InjectedOutcome",
    "base_law",
    "drift_coefficients",
    "exact_remainder",
    "nuisance_error",
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

#: The ``g-drift`` cell's target for the **ATE** coefficient.  One target rather than two,
#: and that is structural rather than a simplification: a binary treatment's mechanism has one
#: free function, since the estimator reads :math:`\hat g(1|W)` off a classifier and takes the
#: complement, so one perturbation determines both arms' coefficients and only their
#: combination can be set.  :func:`drift_coefficients` reports what :math:`c_1` and
#: :math:`c_0` came out at, and the test asserts both clear :data:`C_MIN`.
G_DRIFT_C_ATE = 0.40

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


@cache
def _normalisers(cell: str) -> tuple[float, ...]:
    r"""The quadrature the injected shapes are scaled by, so the coefficients are the declared.

    ``q-drift`` needs :math:`P_0[u_a^2]` per arm; ``g-drift`` needs
    :math:`P_0[d^2(g_0 + \rho g_1)]`, the one combination its single free perturbation can set.
    Both go through :meth:`~cleverly.datasets.DGP.expectation`, which is the **same Sobol rule**
    the truth is integrated with -- a second quadrature here would put a Monte Carlo error of
    its own between a coefficient and the coverage it explains.

    Cached because every injected learner needs it and the integration is over ``2**18``
    points; keyed by the cell, which is what the constants above are indexed by.
    """
    dgp = base_law()
    if cell == "q-drift":
        return tuple(
            dgp.expectation(lambda w, arm=arm: _weight(dgp, w, arm) ** 2) for arm in (1.0, 0.0)
        )
    if cell == "g-drift":

        def integrand(w: Any) -> Any:
            g = _mechanism(dgp, w)
            return outcome_error(w) ** 2 * ((1.0 - g) + G_DRIFT_ARM0_RATIO * g)

        return (dgp.expectation(integrand),)
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


def outcome_perturbation(cell: str, n: int, w: Any, arm: float) -> Any:
    r""":math:`\hat Q_a - \bar Q_{0,a}` at ``n`` rows, on the **outcome's own** scale.

    ``q-drift``'s is the drifting :math:`n^{-\alpha} h_a` with :math:`h_a` aligned with the
    misspecification weight and normalised to :data:`Q_DRIFT_C`; ``g-drift``'s is the fixed
    :func:`outcome_error`, which does not shrink because in that cell the outcome regression is
    the *wrong* nuisance.
    """
    if cell == "q-drift":
        norm = dict(zip((1.0, 0.0), _normalisers(cell), strict=True))[arm]
        shape = _weight(base_law(), w, arm)
        return float(n) ** -ALPHA * (Q_DRIFT_C[arm] / norm) * shape
    if cell == "g-drift":
        ratio = 1.0 if arm == 1.0 else G_DRIFT_ARM0_RATIO
        return ratio * outcome_error(w)
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


def mechanism_perturbation(cell: str, n: int, w: Any) -> Any:
    r""":math:`\hat g(1|W) - g_0(1|W)` at ``n`` rows.

    Zero in ``q-drift``, where the mechanism is the wrong nuisance and is handled by
    :func:`_wrong_mechanism` instead.  In ``g-drift`` it is
    :math:`n^{-\alpha}\lambda\,d(W)g_0(1|W)g_0(0|W)`: the :math:`g_0(1-g_0)` factor is what
    keeps :math:`\hat g` interior wherever :math:`g_0` is near a boundary -- the perturbation
    vanishes with the probability it perturbs -- and it is why this cell needs no clipping and
    so no truncation active.
    """
    if cell == "q-drift":
        return np.zeros(np.asarray(w, dtype=float).shape[0])
    if cell == "g-drift":
        (norm,) = _normalisers(cell)
        g = _mechanism(base_law(), w)
        return float(n) ** -ALPHA * (G_DRIFT_C_ATE / norm) * outcome_error(w) * g * (1.0 - g)
    raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")


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
    """
    dgp = base_law()
    per_arm = {}
    for arm in (1.0, 0.0):
        if cell == "q-drift":

            def integrand(w: Any, arm: float = arm) -> Any:
                # h_a with the n^-alpha divided out, times the weight it is aligned with.
                norm = dict(zip((1.0, 0.0), _normalisers(cell), strict=True))[arm]
                return _weight(dgp, w, arm) ** 2 * (Q_DRIFT_C[arm] / norm)

        elif cell == "g-drift":

            def integrand(w: Any, arm: float = arm) -> Any:
                (norm,) = _normalisers(cell)
                g = _mechanism(dgp, w)
                shape = (G_DRIFT_C_ATE / norm) * outcome_error(w) * g * (1.0 - g)
                # The arm-0 mechanism moves by *minus* the arm-1 perturbation: one free
                # column and its complement. Getting this sign wrong flips c_0 and turns the
                # ATE coefficient from a sum of magnitudes into a difference.
                delta = shape if arm == 1.0 else -shape
                ratio = 1.0 if arm == 1.0 else G_DRIFT_ARM0_RATIO
                return delta / _arm(g, arm) * ratio * outcome_error(w)

        else:
            raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")
        per_arm[arm] = dgp.expectation(integrand)
    return {
        "c1": per_arm[1.0],
        "c0": per_arm[0.0],
        "c_ate": per_arm[1.0] - per_arm[0.0],
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
    """One row per cell: what the design committed to, so a run prints it beside its numbers."""
    rows = []
    for cell in CELLS:
        realised = drift_coefficients(cell)
        rows.append(
            [
                cell,
                f"{ALPHA:.2f}",
                f"{realised['c1']:+.4f}",
                f"{realised['c0']:+.4f}",
                f"{realised['c_ate']:+.4f}",
                f"{min(abs(realised[key]) for key in ('c1', 'c0', 'c_ate')):.4f}",
            ]
        )
    return rows


SUMMARY_HEADERS: tuple[str, ...] = ("cell", "alpha", "c1", "c0", "c_ate", "min |c|")

#: What a caller reads a shape off, kept beside the functions so the two cannot drift.
SHAPES: dict[str, Callable[..., Any]] = {
    "outcome": outcome_perturbation,
    "mechanism": mechanism_perturbation,
}
