r"""The conditioning term of DR-TMLE's remainder is linear in the consistent nuisance's error.

Benkeser, Carone, van der Laan and Gilbert (2017), Appendix A, write the remainder that the
outcome-wrong branch leaves after equation (9) as :math:`R_{3,n} + R_{4,n} + M_{1,n}`, with

.. math::

    R_{3,n} = P_0\Bigl[\frac{\bar Q_{0n,r} - \bar Q_{0,r}}{g_0}\,(g_0 - g_n)\Bigr],
    \qquad
    \bar Q_{0n,r} = E_0[\,Y - \bar Q \mid A = 1,\ g_n(W),\ g_0(W)\,].

On a level set of :math:`(g_n, g_0)` the probability :math:`P(A = 1 \mid W) = g_0` is
constant, so :math:`\bar Q_{0n,r} = E_0[\bar Q_0 - \bar Q \mid g_n, g_0]`, and the tower rule
gives, with no approximation,

.. math::

    R_{3,n} = P_0\Bigl[\frac{F}{g_0}\,(g_0 - g_n)\Bigr],
    \qquad
    F = \bar Q_0 - \bar Q - \bar Q_{0,r}.

:math:`F` does not shrink when the outcome regression is wrong. It is zero only when
:math:`\bar Q_0 - \bar Q` is a function of :math:`g_0` (H1). So the term is **linear** in
:math:`g_n - g_0` unless the error keeps the level sets of :math:`g_0` (H2).
``docs/technical-reference/dr-tmle/theorem.md``, "The conditioning terms are linear in one
error", states the cases.

**The law.** Four cells with equal mass. The true mechanism ties two pairs of cells, and the
estimate splits each tie. That is what makes :math:`R_{3,n}` nonzero. It is the converse of
:data:`tests.unit.test_remainder_drtmle.TIED_G`, which ties the *estimate* where the truth is
untied and pins the coarse-reduction side instead.

**The reduction is handed in at its limit** :math:`\bar Q_{0,r} = E_0[Y - \bar Q \mid A, g_0]`.
That is the analogue of a consistent univariate reduction on a continuous law. On finite
support the exact reduction :math:`E[\cdot \mid \hat g_t]` would condition on every cell,
because :math:`\hat g_t` takes four distinct values, and it would remove the whole term. That
case is kept below as a negative case: it is an artifact of finite support.

**The library is in the path.** :math:`D^*` and its subtraction of :math:`D_A` both come from
:func:`cleverly.inference.influence.counterfactual_means`, with ``corrections=`` built by
:func:`cleverly.inference.influence.reduced_corrections` under ``guard=("Q",)``. That function
builds :math:`D_A` in :func:`~cleverly.inference.influence.reduced_correction_parts`. The
closed forms :math:`R_3` and :math:`R_4` are computed here from the law alone, with an
independently computed :math:`\bar Q_{0,r}`, never from the handed-in reduction. So a sign or
denominator error in the library's :math:`D_A` moves the leftover and not the closed form.
Two such source mutations were run against a hash-verified backup and each failed the
identity and linearity assertions.

**Why the direction has a** :math:`g_0`-**measurable part.** With :math:`h` alone the guard
removes nothing at first order, because :math:`\bar Q_{0,r}` is constant on each tie and
:math:`h` has mean zero there. The part :math:`\kappa(g_0 - 1/2)` makes :math:`P_0 D_A` of
first order, so a flipped or dropped guard fails at first order. It does not move
:math:`R_3`, because :math:`E[F \mid g_0] = 0`.

The sample realises the law exactly: each cell has ``ROWS`` rows, ``g_0 * ROWS`` of them
treated, and each row's outcome is its conditional mean. Every curve here is linear in
:math:`Y`, so a sample mean *is* a :math:`P_0` mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.estimators.reduced import ReducedSet
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import counterfactual_means, reduced_corrections

#: Rows per cell. Every mechanism value below times this is an integer.
ROWS = 10

#: Equal cell masses, and the true mechanism with two ties.
P = np.full(4, 0.25)
G0 = np.array([0.4, 0.4, 0.6, 0.6])

#: The true outcome regression, ``[cell, arm]``. Neither arm is a function of ``G0``.
Q0 = np.column_stack([[0.2, 0.6, 0.4, 0.3], [0.3, 0.7, 0.5, 0.2]])

#: The wrong outcome regression: a constant per arm.
QBAR = np.array([0.4, 0.5])

#: The tie-splitting direction and the weight of its ``g_0``-measurable part.
H = np.array([1.0, -1.0, 0.5, -0.5])
KAPPA = 2.0

#: The perturbation sizes. The leftover divided by ``t`` must agree at both.
STEPS = (1e-3, 1e-2)

#: Bounds that never bind on these mechanisms.
INERT = (1e-6, 1.0 - 1e-6)

#: The closed-form ``R_3 / t`` per estimand. An arithmetic pin, not evidence: the closed form
#: is linear by construction. Assertion (iv) on the library's leftover is the evidence.
R3_PER_STEP = {"ey1": 0.1875, "ey0": -0.135416666666667, "ate": 0.322916666666667}

#: Floors on ``|leftover / t|``, the quantity that comes through the library. Measured
#: 0.1878 and 0.1906 (arm 1), 0.1353 and 0.1340 (arm 0).
FLOORS = {"ey1": 0.15, "ey0": 0.10}

#: The linearity constant: ``|leftover/t - R_3/t| <= C t``. The gap is ``R_4 / t``, which
#: is ``O(t)``. Measured at arm 1: 2.9e-4 at ``t = 1e-3`` and 3.1e-3 at ``t = 1e-2``.
LINEARITY = 0.5

#: A named floor for a leftover that must vanish. Never an exact zero (brief: near-zero
#: comparisons).
ZERO = 1e-13


def _realised(g0: np.ndarray, q0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(cell, treatment, outcome)`` rows whose empirical law is the declared one."""
    cells, treatment, outcome = [], [], []
    for w, prob in enumerate(g0):
        treated = round(prob * ROWS)
        if not np.isclose(treated, prob * ROWS, rtol=0, atol=1e-12):
            raise ValueError(f"g_0 = {prob} times {ROWS} rows is not an integer")
        for a in (1.0,) * treated + (0.0,) * (ROWS - treated):
            cells.append(w)
            treatment.append(a)
            outcome.append(q0[w, int(a)])
    return np.array(cells), np.array(treatment), np.array(outcome)


def _conditional(values: np.ndarray, key: np.ndarray) -> np.ndarray:
    """``E[values | key]`` under the equal cell masses, per cell."""
    out = np.empty_like(values)
    for level in np.unique(np.round(key, 12)):
        mask = np.isclose(key, level, rtol=0, atol=1e-12)
        out[mask] = (P[mask] * values[mask]).sum() / P[mask].sum()
    return out


def _arm_mechanism(g: np.ndarray, arm: int) -> np.ndarray:
    """``P(A = arm | W)`` from the arm-1 mechanism."""
    return g if arm == 1 else 1.0 - g


def limit_reduction(q0: np.ndarray) -> np.ndarray:
    r"""``Q̄_{0,r}(a) = E_0[Y - Q̄ | A = a, g_0]`` per cell, ``[cell, arm]``.

    Conditioning on ``A = a`` leaves the law of ``W`` unchanged on a level set of
    ``g_0``, so this is ``E[Q̄_0 - Q̄ | g_0]``.
    """
    return np.column_stack([_conditional(q0[:, arm] - QBAR[arm], G0) for arm in (0, 1)])


def exact_reduction(q0: np.ndarray, g_hat: np.ndarray) -> np.ndarray:
    r"""``E_0[Y - Q̄ | A = a, ĝ]`` per cell: the finite-support artifact."""
    return np.column_stack([_conditional(q0[:, arm] - QBAR[arm], g_hat) for arm in (0, 1)])


def closed_forms(q0: np.ndarray, g_hat: np.ndarray, arm: int) -> tuple[float, float]:
    r"""``(R_3, R_4)`` for one arm, from the law alone.

    ``F`` uses its own ``E[· | g_0]``, computed here, and never the handed-in reduction.
    """
    resid = q0[:, arm] - QBAR[arm]
    f = resid - _conditional(resid, G0)
    g_true, g_est = _arm_mechanism(G0, arm), _arm_mechanism(g_hat, arm)
    r3 = float((P * f / g_true * (g_true - g_est)).sum())
    r4 = float((P * f * (g_true - g_est) ** 2 / (g_est * g_true)).sum())
    return r3, r4


def leftover(q0: np.ndarray, g_hat: np.ndarray, reduction: np.ndarray) -> dict[str, float]:
    r"""``R_2 - P_0 D_A`` per estimand, through the library.

    ``Ψ(Q̄) - ψ_0 + P_0 D*`` is the plain remainder ``R_2``. With ``D_A`` subtracted inside
    :func:`counterfactual_means`, the same sum is ``R_2 - P_0 D_A``: what equation (9)
    leaves when the reduction is ``reduction``.
    """
    cell, treatment, outcome = _realised(G0, q0)
    n = cell.size
    at = {arm: np.full(n, QBAR[arm]) for arm in (0, 1)}
    initial = InitialFit(
        observed=np.where(treatment == 1.0, at[1], at[0]), arms={1.0: at[1], 0.0: at[0]}
    )
    reduced = ReducedSet(
        qr=reduction[cell],
        gr1=np.full((n, 2), 0.5),
        gr2=np.zeros((n, 2)),
        arms=(0.0, 1.0),
        g_bounds=INERT,
    )
    corrections = reduced_corrections(
        outcome, initial, treatment, reduced, g_hat[cell], bounds=INERT, guard=("Q",)
    )
    means = counterfactual_means(
        outcome,
        initial,
        submodel_for("mean", treatment, g_hat[cell]),
        np.ones(n),
        corrections=corrections,
    )
    truth = {arm: float((P * q0[:, arm]).sum()) for arm in (0, 1)}
    left = {
        arm: means[float(arm)].psi - truth[arm] + float(np.mean(means[float(arm)].influence_curve))
        for arm in (0, 1)
    }
    return {"ey1": left[1], "ey0": left[0], "ate": left[1] - left[0]}


def estimate(step: float) -> np.ndarray:
    """``ĝ_t = g_0 + t (h + κ (g_0 - 1/2))``: splits each tie of ``g_0``."""
    g_hat = G0 + step * (H + KAPPA * (G0 - 0.5))
    assert np.unique(np.round(g_hat, 12)).size == 4
    return g_hat


def closed_by_estimand(q0: np.ndarray, g_hat: np.ndarray) -> dict[str, tuple[float, float]]:
    """``(R_3, R_4)`` for each estimand. The contrast's terms are differences."""
    one, zero = closed_forms(q0, g_hat, 1), closed_forms(q0, g_hat, 0)
    return {
        "ey1": one,
        "ey0": zero,
        "ate": (one[0] - zero[0], one[1] - zero[1]),
    }


class TestTheConditioningTermIsLinear:
    """The witness: a nonzero ``R_3`` that is linear in the split."""

    @pytest.mark.parametrize("step", STEPS)
    @pytest.mark.parametrize("name", ["ey1", "ey0", "ate"])
    def test_the_leftover_is_r3_plus_r4(self, step: float, name: str) -> None:
        """(i): the library's leftover equals the closed forms."""
        g_hat = estimate(step)
        got = leftover(Q0, g_hat, limit_reduction(Q0))[name]
        r3, r4 = closed_by_estimand(Q0, g_hat)[name]
        np.testing.assert_allclose(got, r3 + r4, rtol=1e-12, atol=0)

    @pytest.mark.parametrize("step", STEPS)
    @pytest.mark.parametrize("name", ["ey1", "ey0", "ate"])
    def test_the_closed_form_r3_is_pinned(self, step: float, name: str) -> None:
        """(ii): an arithmetic pin. ``κ`` does not move ``R_3``, since ``E[F | g_0] = 0``."""
        r3, _ = closed_by_estimand(Q0, estimate(step))[name]
        np.testing.assert_allclose(r3 / step, R3_PER_STEP[name], rtol=1e-10)

    @pytest.mark.parametrize("step", STEPS)
    @pytest.mark.parametrize("name", ["ey1", "ey0"])
    def test_the_leftover_does_not_vanish(self, step: float, name: str) -> None:
        """(iii): the floor sits on the library quantity, not on the closed form."""
        got = leftover(Q0, estimate(step), limit_reduction(Q0))[name]
        assert abs(got / step) >= FLOORS[name]

    @pytest.mark.parametrize("step", STEPS)
    @pytest.mark.parametrize("name", ["ey1", "ey0", "ate"])
    def test_the_leftover_is_first_order_in_the_split(self, step: float, name: str) -> None:
        """(iv): ``leftover / t`` tends to ``R_3 / t``, with a gap of order ``t``."""
        got = leftover(Q0, estimate(step), limit_reduction(Q0))[name]
        assert abs(got / step - R3_PER_STEP[name]) <= LINEARITY * step

    def test_the_guard_removes_a_first_order_amount(self) -> None:
        """Without ``κ`` the guard's removal would be second order, and blind to a sign.

        The leftover at ``κ = 2`` differs from ``R_2`` by ``P_0 D_A``, which is of order
        ``t``: 0.0249 t at arm 1 and -0.0126 t at arm 0.
        """
        step = STEPS[0]
        g_hat = estimate(step)
        zero_reduction = np.zeros_like(limit_reduction(Q0))
        plain = leftover(Q0, g_hat, zero_reduction)
        guarded = leftover(Q0, g_hat, limit_reduction(Q0))
        assert (plain["ey1"] - guarded["ey1"]) / step > 0.02
        assert (plain["ey0"] - guarded["ey0"]) / step < -0.01


class TestTheTermVanishesInTheListedCases:
    """Controls: each makes the leftover zero, so the witness above is not an artifact."""

    @pytest.mark.parametrize("step", STEPS)
    def test_h1_the_residual_is_a_function_of_g0(self, step: float) -> None:
        """H1: ``Q̄_0 - Q̄`` constant on each tie of ``g_0`` gives ``F ≡ 0``."""
        q0 = np.column_stack([[0.2, 0.2, 0.3, 0.3], [0.3, 0.3, 0.4, 0.4]])
        got = leftover(q0, estimate(step), limit_reduction(q0))
        for name in ("ey1", "ey0", "ate"):
            assert abs(got[name]) <= ZERO

    @pytest.mark.parametrize("step", STEPS)
    def test_h2_the_estimate_keeps_the_level_sets(self, step: float) -> None:
        """H2: ``ĝ = g_0 + t (g_0 - 1/2)`` is a function of ``g_0``."""
        g_hat = G0 + step * (G0 - 0.5)
        got = leftover(Q0, g_hat, limit_reduction(Q0))
        for name in ("ey1", "ey0", "ate"):
            assert abs(got[name]) <= ZERO

    @pytest.mark.parametrize("step", STEPS)
    def test_the_exact_reduction_on_finite_support_removes_everything(self, step: float) -> None:
        """The finite-support artifact: ``E[· | ĝ_t]`` conditions on every cell.

        ``ĝ_t`` takes four distinct values, so the exact reduction recovers the whole
        residual and ``R_4 = -R_3``. A continuous law has no such reduction. This is why the
        witness hands in the limit reduction.
        """
        g_hat = estimate(step)
        got = leftover(Q0, g_hat, exact_reduction(Q0, g_hat))
        for name in ("ey1", "ey0", "ate"):
            assert abs(got[name]) <= ZERO


#: The mirror law (arm 1 only): the outcome regression ties two pairs of cells, and the
#: mechanism is untied.
MIRROR_G0 = np.array([0.3, 0.6, 0.5, 0.7])
MIRROR_Q0 = np.array([0.4, 0.4, 0.6, 0.6])
MIRROR_K = np.array([1.0, -1.0, 0.5, -0.5])


def mirror_leftover(gbar: np.ndarray, step: float) -> float:
    r"""``R_2 - P_0 D_Y`` at arm 1, with ``D_Y`` from the library under ``guard=("g",)``.

    The outcome estimate is ``Q̄_n = Q̄_0 + s k``, which splits each tie of ``Q̄_0``. The
    mechanism is wrong (``ḡ``). The reductions are handed in at their limits
    ``g_{1,0,r} = E[A | Q̄_0]`` and ``g_{2,0,r} = E[(A - ḡ)/ḡ | Q̄_0]``.
    """
    q0 = np.column_stack([np.full(4, 0.5), MIRROR_Q0])
    cell, treatment, outcome = [], [], []
    for w, prob in enumerate(MIRROR_G0):
        treated = round(prob * ROWS)
        for a in (1.0,) * treated + (0.0,) * (ROWS - treated):
            cell.append(w)
            treatment.append(a)
            outcome.append(q0[w, int(a)])
    cells, a_obs, y = np.array(cell), np.array(treatment), np.array(outcome)
    n = cells.size
    q_hat = MIRROR_Q0 + step * MIRROR_K
    at = {0: np.full(n, 0.5), 1: q_hat[cells]}
    initial = InitialFit(
        observed=np.where(a_obs == 1.0, at[1], at[0]), arms={1.0: at[1], 0.0: at[0]}
    )
    ratio = (MIRROR_G0 - gbar) / gbar
    gr1 = _conditional(MIRROR_G0, MIRROR_Q0)
    gr2 = _conditional(ratio, MIRROR_Q0)
    reduced = ReducedSet(
        qr=np.zeros((n, 2)),
        gr1=np.column_stack([1.0 - gr1, gr1])[cells],
        gr2=np.column_stack([np.zeros(4), gr2])[cells],
        arms=(0.0, 1.0),
        g_bounds=INERT,
    )
    corrections = reduced_corrections(
        y, initial, a_obs, reduced, gbar[cells], bounds=INERT, guard=("g",)
    )
    means = counterfactual_means(
        y, initial, submodel_for("mean", a_obs, gbar[cells]), np.ones(n), corrections=corrections
    )
    arm = means[1.0]
    truth = float((P * MIRROR_Q0).sum())
    return arm.psi - truth + float(np.mean(arm.influence_curve))


class TestTheMirrorTermIsLinear:
    r"""Appendix B's ``R̃_{5,n}``: the same structure with the roles of the nuisances swapped.

    With ``r = (g_0 - ḡ)/ḡ`` and ``h = r - g_0 E[r | Q̄_0]/E[g_0 | Q̄_0]``, the leftover is
    ``P_0[h (Q̄_0 - Q̄_n)]``, linear in ``Q̄_n - Q̄_0``. It vanishes when ``1/ḡ - 1/g_0`` is a
    function of ``Q̄_0`` (mirror H1). Arm 1 only.
    """

    @pytest.mark.parametrize("step", STEPS)
    def test_a_wrong_mechanism_leaves_a_linear_term(self, step: float) -> None:
        """``leftover / s`` is constant and nonzero: ``h`` does not depend on ``Q̄_n``."""
        got = mirror_leftover(np.full(4, 0.5), step)
        np.testing.assert_allclose(got / step, 0.208333333333333, rtol=1e-10)

    @pytest.mark.parametrize("step", STEPS)
    def test_mirror_h1_makes_it_vanish(self, step: float) -> None:
        """``ḡ = g_0/(1 + g_0/2)`` makes ``1/ḡ - 1/g_0 = 1/2``, a constant."""
        got = mirror_leftover(MIRROR_G0 / (1.0 + 0.5 * MIRROR_G0), step)
        assert abs(got) <= ZERO
