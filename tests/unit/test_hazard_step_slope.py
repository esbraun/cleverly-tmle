"""The exact slope of the hazard walk loss along a move, which the bounded step takes the root of.

``step_bound=`` with ``step_search=True`` takes the length of each per-horizon step as the root of
the slope of :func:`~cleverly.longitudinal.hazard.walk_loss` along the move.  The slope is
written by hand through the clip of the logits (:func:`_clip_tangent`) and the derivative of each
person-node's term in its clipped logit (:class:`_WalkLine`).  These tests compare it with
Richardson-extrapolated differences of the loss itself, on crafted states small enough to read:

* state A holds softmax nodes: a logit released from the upper bound, the ratio-keeping shift
  with a free and a held second cause, two causes tied at the upper bound, a logit moving in from the lower bound, a full cell, an
  empty cell, a zero-weight unit and non-uniform weights;
* state B adds a bounded node, with a positive lower bound (so an outcome below 0 without an
  event), one logit held at the upper bound and one moving in from the lower bound;
* state C adds a full cell with a unit that is not at risk there, whose term is infinite outside
  the risk set.

Away from kinks the slope must match a central difference.  At length 0, where clipped logits sit
exactly on a bound, it must match the forward difference: the step moves forward, so the slope
is the right derivative.  Each component is shown to carry weight in the slope, and each has a
one-line mutation that the difference check catches.

No control drops the ``-inf`` mask at the end of :func:`_clip_tangent`.  That mutant is
equivalent: at an empty cell the shifted logit is already ``-inf``, which no clip branch counts,
and the term's derivative ``dN - h`` is 0 on counted units, so the empty cause's term is
identically zero.  The ``-inf`` hold of :meth:`HazardState.moved` is guarded by a control that can
fail: the slope's own moved state rebuilt without the hold, while the loss keeps it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import expit, xlogy

from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal.hazard import (
    HazardBounds,
    HazardInputs,
    HazardState,
    _clip_tangent,
    _quasi_terms,
    _WalkLine,
    direction,
    scores,
    walk_loss,
)
from tests.discrete_law_point_survival import CellMeans, CellProbabilities, PointLaw
from tests.unit._source_mutation import mutate
from tests.unit.test_hazard_survival_law import ArmOnly, _internals

#: The clip interval: ``alpha = expit(3)`` gives ``(-3, 3)``.
ALPHA = float(expit(3.0))
#: ``HazardInputs.bounds`` computes the same expression.
HIGH = float(np.log(ALPHA / (1.0 - ALPHA)))
LOW = -HIGH
NEG = -np.inf

#: Observed levels: units 0 to 3 and 6 to 8 at level 0, units 4 and 5 at level 1.
OBSERVED = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0])
N = OBSERVED.shape[0]
#: Unit 7 has weight 0.  The others are non-uniform.
WEIGHTS = np.array([1.3, 0.7, 1.1, 0.9, 1.2, 0.8, 1.0, 0.0, 1.05])
#: Events at node 1 (index 0): units 0 and 4 of cause 0, units 3 and 5 of cause 1.  At node 2
#: (index 1): unit 1 of cause 0.  The cells (node 1, level 1) and its two units form a full cell.
EVENTS = np.zeros((N, 2, 2))
EVENTS[0, 0, 0] = EVENTS[4, 0, 0] = EVENTS[1, 1, 0] = 1.0
EVENTS[3, 0, 1] = EVENTS[5, 0, 1] = 1.0
#: At risk at node 2: the units without an event at node 1.
AT_RISK = np.ones((N, 2), dtype=bool)
AT_RISK[[0, 3, 4, 5], 1] = False
EVENTS[~AT_RISK] = 0.0

#: Observed-level logits and moves, ``(n, T, J)``.  Each line names what it places.
OBSERVED_LOGITS = np.array(
    [
        [[HIGH, 0.5], [0.4, -0.2]],  # u0: top at the bound moving up (the shift); second free
        [[HIGH, 1.0], [0.3, NEG]],  # u1: released from the bound; node 2 beside the empty cause
        [[LOW, -1.0], [HIGH, NEG]],  # u2: moving in from the lower bound; node 2 held at the top
        [[HIGH, LOW], [0.1, 0.2]],  # u3: the shift pushes the second cause under the lower bound
        [[HIGH, 0.3], [0.0, 0.0]],  # u4: full cell, on the shift
        [[0.2, -0.4], [0.0, 0.0]],  # u5: full cell, free
        [[-0.5, 0.6], [-0.7, NEG]],  # u6: free; node 2 beside the empty cause
        [[0.9, -0.8], [0.1, NEG]],  # u7: weight 0
        [[HIGH, HIGH], [0.2, NEG]],  # u8: two causes tied at the top, both moving up
    ]
)
OBSERVED_MOVES = np.array(
    [
        [[0.8, -0.3], [0.5, 0.4]],
        [[-0.5, 0.2], [0.5, 0.9]],
        [[0.4, 0.1], [0.4, -0.2]],
        [[0.6, 0.2], [0.3, -0.6]],
        [[0.5, -0.2], [0.2, 0.1]],
        [[0.3, 0.7], [0.1, 0.2]],
        [[-0.4, 0.5], [-0.6, 0.3]],
        [[0.7, 0.6], [1.0, 1.0]],
        [[0.6, 0.2], [0.3, 0.1]],
    ]
)


def place(observed: np.ndarray, fill: float) -> np.ndarray:
    """``(n, T, L, J)`` with ``observed`` at each unit's level and ``fill`` at the other."""
    out = np.full((N, 2, 2, 2), fill)
    out[np.arange(N), :, OBSERVED] = observed
    return out


def inputs_of(full: np.ndarray, node_bounds: HazardBounds | None = None) -> HazardInputs:
    empty = np.zeros((2, 2, 2), dtype=bool)
    empty[1, 0, 1] = node_bounds is None
    return HazardInputs(
        AT_RISK,
        EVENTS,
        OBSERVED,
        np.ones((N, 2, 2)),
        np.full((1, N, 2), 0.5),
        WEIGHTS,
        ALPHA,
        empty,
        full,
        node_bounds,
    )


def state_a() -> tuple[HazardState, HazardInputs, np.ndarray]:
    full = np.zeros((2, 2), dtype=bool)
    full[0, 1] = True
    return (
        HazardState(place(OBSERVED_LOGITS, 0.0), full),
        inputs_of(full),
        place(OBSERVED_MOVES, 0.3),
    )


#: Node 2 bounded: cause 0 in ``[0.05, 0.55]``, cause 1 in ``[0, 0.4]``.
BOUNDS = HazardBounds(
    np.array([False, True]),
    np.array([[0.0, 0.0], [0.05, 0.0]]),
    np.array([[1.0, 1.0], [0.5, 0.4]]),
)


def state_b() -> tuple[HazardState, HazardInputs, np.ndarray]:
    full = np.zeros((2, 2), dtype=bool)
    full[0, 1] = True
    logits = OBSERVED_LOGITS.copy()
    moves = OBSERVED_MOVES.copy()
    # The bounded node holds independent per-cause logits, all finite.
    logits[:, 1] = [
        [0.4, -0.2],
        [HIGH, 0.3],  # u1: cause 0 at the upper bound, moving out (held)
        [LOW, -1.0],  # u2: cause 0 at the lower bound, moving in; no event, so y < 0
        [0.1, 0.2],
        [0.0, 0.0],
        [0.0, 0.0],
        [-0.7, 0.5],
        [0.1, 0.4],
        [0.2, -0.3],
    ]
    moves[:, 1] = [
        [0.5, 0.4],
        [0.6, -0.4],
        [0.5, 0.3],
        [0.3, -0.6],
        [0.2, 0.1],
        [0.1, 0.2],
        [-0.6, 0.3],
        [1.0, 1.0],
        [0.3, 0.1],
    ]
    return (
        HazardState(place(logits, 0.0), full, BOUNDS),
        inputs_of(full, BOUNDS),
        place(moves, 0.3),
    )


def state_c() -> tuple[HazardState, HazardInputs, np.ndarray]:
    # Units 4 and 5 are not at risk at node 2, where their level's cell is full: the no-event
    # probability there is 0, so their term is -inf outside the risk set.
    state, _, move = state_a()
    full = np.zeros((2, 2), dtype=bool)
    full[0, 1] = full[1, 1] = True
    return HazardState(state.logits, full), inputs_of(full), move


STATES = {"A": state_a, "B": state_b, "C": state_c}


def line_of(name: str) -> _WalkLine:
    state, inputs, move = STATES[name]()
    return _WalkLine.along(state, inputs, move)


# ------------------------------------------------------------------------- the reference


def central(line: _WalkLine, t: float, e: float = 1e-3) -> float:
    """Richardson-extrapolated central difference of the loss, error of order ``e**4``."""

    def d(h: float) -> float:
        return (line.loss(t + h) - line.loss(t - h)) / (2.0 * h)

    return (4.0 * d(e / 2.0) - d(e)) / 3.0


def forward(line: _WalkLine, e: float = 1e-3) -> float:
    """Twice Richardson-extrapolated forward difference at 0, error of order ``e**3``.

    A forward difference carries every power of ``e``, so the first extrapolation removes the
    ``e`` term (``2 D(e/2) - D(e)``) and the second the ``e**2`` term.
    """

    def d(h: float) -> float:
        return (line.loss(h) - line.loss(0.0)) / h

    def once(h: float) -> float:
        return 2.0 * d(h / 2.0) - d(h)

    return (4.0 * once(e / 2.0) - once(e)) / 3.0


def pattern(line: _WalkLine, t: float) -> tuple[bytes, ...]:
    """The clip's active set at length ``t``: top cause, shift, and each logit's bound side."""
    z = np.where(np.isneginf(line.logits), NEG, line.logits + t * line.step)
    finite = np.isfinite(z)
    top = np.max(z, axis=-1)
    ties = np.where(finite & (z == top[..., None]), line.step, NEG)
    index = np.argmax(ties, axis=-1)
    leader = np.take_along_axis(line.step, index[..., None], axis=-1)[..., 0]
    over = np.isfinite(top) & ((top > HIGH) | ((top == HIGH) & (leader > 0.0)))
    if line.bounded is not None:
        over = over & ~line.bounded[None, :]
    shifted = np.where(over[..., None], (z - top[..., None]) + HIGH, z)
    return tuple(
        np.asarray(a).tobytes() for a in (index, over, shifted <= LOW, shifted >= HIGH, finite)
    )


def summand(line: _WalkLine, t: float) -> np.ndarray:
    """``W G tau`` per unit, node and cause, as the slope sums it (before the minus and total)."""
    inputs = line.inputs
    rows, levels = inputs.rows, inputs.observed
    moved = line.state.moved(t * line.move, inputs.bounds)
    z = np.where(np.isneginf(line.logits), NEG, line.logits + t * line.step)
    tangent = _clip_tangent(z, line.step, inputs.bounds, line.bounded)
    grad = inputs.events - moved.hazards[rows, :, levels]
    if line.bounded is not None:
        k = line.bounded
        y = (inputs.events[:, k] - BOUNDS.lower[k][None]) / BOUNDS.scale[k][None]
        grad[:, k] = y - expit(moved.logits[rows, :, levels][:, k])
    weight = np.where(line.counted, line.base, 0.0)[..., None]
    return np.asarray(np.where(line.counted[..., None], weight * grad * tangent, 0.0))


LENGTHS = (0.02, 0.05, 0.09)


def assert_matches_the_loss(name: str) -> None:
    line = line_of(name)
    for t in LENGTHS:
        assert pattern(line, t - 1e-3) == pattern(line, t + 1e-3), (name, t)
        np.testing.assert_allclose(line.slope(t), central(line, t), rtol=1e-8, err_msg=name)
    assert pattern(line, 1e-12) == pattern(line, 1e-3), name
    np.testing.assert_allclose(line.slope(0.0), forward(line), rtol=1e-8, err_msg=name)


# --------------------------------------------------------------------------------- the checks


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_the_slope_is_the_derivative_of_the_walk_loss(name: str) -> None:
    """Checks 1 and 2: central differences away from kinks, the forward difference at 0.

    The active set of the clip is asserted equal at both ends of each window.  Logits move
    affinely in the length, so equal ends mean a constant active set between them, where the
    loss is smooth.  Measured relative gaps: 7e-13 to 1e-11 at the three lengths, 2.7e-12 and
    1.3e-11 at 0, against the gate of 1e-8.
    """
    assert_matches_the_loss(name)


def test_every_component_carries_weight_in_the_slope() -> None:
    """Check 3: each component contributes more than 1e-3 of the slope, so each is observed."""
    line = line_of("A")
    t = LENGTHS[0]
    terms = summand(line, t)
    whole = abs(float(terms.sum()))
    assert whole > 0.0
    np.testing.assert_allclose(-terms.sum() / line.total, line.slope(t), rtol=1e-14)
    rows, levels = line.inputs.rows, line.inputs.observed
    moved = line.state.moved(t * line.move, line.inputs.bounds)
    grad = line.inputs.events - moved.hazards[rows, :, levels]
    weight = np.where(line.counted, line.base, 0.0)[..., None]
    z = line.logits + t * line.step
    tangent = _clip_tangent(z, line.step, line.inputs.bounds, None)
    finite = np.isfinite(line.logits)
    # Held: the part of W G m that the clip removes.
    held = np.where(finite & (tangent == 0.0) & (line.step != 0.0), weight * grad * line.step, 0.0)
    # The shift: units 0, 3 and 4 at node 1.
    shift = terms[[0, 3, 4], 0].sum()
    # The full cell: units 4 and 5 at node 1.
    full = terms[[4, 5], 0].sum()
    # The weights: the change from unit weights, and the zero-weight unit's dropped term.
    unweighted = np.where(line.counted[..., None], grad * tangent, 0.0)
    reweighted = (terms - unweighted).sum()
    dropped = (grad * tangent)[7][AT_RISK[7]].sum()
    for label, part in {
        "held": held.sum(),
        "shift": shift,
        "full": full,
        "weights": reweighted,
        "zero weight": dropped,
    }.items():
        assert abs(float(part)) > 1e-3 * whole, label
    bounded = line_of("B")
    node = summand(bounded, t)[:, 1].sum()
    assert abs(float(node)) > 1e-3 * abs(float(summand(bounded, t).sum())), "bounded node"


def test_the_slope_at_zero_is_minus_the_score_norm_on_a_law() -> None:
    """Check 5: along the walk's first direction the slope at 0 is :math:`-\\lVert s \\rVert`.

    The weights have mean one, so the total weight is ``n``, and no logit is held at a bound
    against the move (asserted).  A reversed or unscaled slope fails it.  Measured gap 1.5e-14
    relative.
    """
    law = PointLaw("survival", 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            {"a0": 0, "a1": 1, "a2": 2},
            n_folds=1,
            survival_model="hazard",
            outcome_learner=ArmOnly(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
            targeting="iterative",
        ).fit(law.frame(), **law.fit_columns())
    inputs, state = _internals(result)
    for horizon in (1, 2):
        score, _ = scores(state, inputs, (horizon,))
        size = inputs.n_regimens * inputs.n_causes
        shape = (inputs.n_regimens, inputs.n_causes, 1)
        basis = [
            direction(state, inputs, (horizon,), np.eye(size)[m].reshape(shape))
            for m in range(size)
        ]
        current = score.reshape(-1)
        norm = float(np.linalg.norm(current))
        move = np.tensordot(current / norm, np.asarray(basis), axes=1)
        line = _WalkLine.along(state, inputs, move)
        tangent = _clip_tangent(line.logits, line.step, inputs.bounds, line.bounded)
        moving = line.counted[..., None] & np.isfinite(line.logits)
        np.testing.assert_array_equal(tangent[moving], line.step[moving])
        np.testing.assert_allclose(line.slope(0.0), -norm, rtol=1e-12)


def shipped_walk_loss(
    state: HazardState, inputs: HazardInputs, *, last_node: int | None = None
) -> float:
    """The body of ``walk_loss`` before it read its weights and terms through helpers."""
    rows, levels = inputs.rows, inputs.observed
    hazards = state.hazards[rows, :, levels]
    none = 1.0 - hazards.sum(axis=-1)
    terms = xlogy(inputs.events, hazards).sum(axis=-1) + xlogy(
        1.0 - inputs.events.sum(axis=-1), none
    )
    bounds = state.node_bounds
    if bounds is not None and bounds.active:
        terms[:, bounds.bounded] = _quasi_terms(
            inputs.events, state.logits[rows, :, levels], bounds
        )
    counted = inputs.at_risk & (inputs.weights != 0.0)[:, None]
    if last_node is not None:
        counted = counted & (np.arange(inputs.n_times) < last_node)[None, :]
    return -float(np.sum(np.where(counted, inputs.weights[:, None] * terms, 0.0)))


def test_the_walk_loss_is_bit_identical_to_the_shipped_body() -> None:
    """Check 6: ``walk_loss`` equals the shipped body with ``==``, under the warning filter.

    The repository turns ``RuntimeWarning`` into an error, so a split that multiplied a zeroed
    weight by state C's infinite term (``0 * -inf``) fails here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for name, build in STATES.items():
            state, inputs, move = build()
            for moved in (state, state.moved(0.05 * move, inputs.bounds)):
                for last in (None, 1, 2):
                    assert walk_loss(moved, inputs, last_node=last) == shipped_walk_loss(
                        moved, inputs, last_node=last
                    ), (name, last)
        line = line_of("C")
        assert not np.all(np.isfinite(hazard_module._walk_terms(line.state, line.inputs)))
        assert np.isfinite(line.mass()) and np.isfinite(line.slope(0.0))
    law = PointLaw("survival", 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            {"a0": 0, "a1": 1, "a2": 2},
            n_folds=1,
            survival_model="hazard",
            outcome_learner=ArmOnly(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
        ).fit(law.frame(), **law.fit_columns())
    inputs, state = _internals(result)
    for last in (None, 1, 2):
        assert walk_loss(state, inputs, last_node=last) == shipped_walk_loss(
            state, inputs, last_node=last
        )


# ------------------------------------------------------------------------ mutation controls


def tangent_edit(monkeypatch: pytest.MonkeyPatch, *edits: tuple[str, str]) -> None:
    mutate(monkeypatch, hazard_module, hazard_module, "_clip_tangent", list(edits))


def slope_edit(monkeypatch: pytest.MonkeyPatch, *edits: tuple[str, str]) -> None:
    mutate(monkeypatch, hazard_module, _WalkLine, "slope", list(edits))


def fails(name: str, *, at_zero: bool = False) -> bool:
    line = line_of(name)
    if at_zero:
        return not np.isclose(line.slope(0.0), forward(line), rtol=1e-8, atol=0.0)
    return any(
        not np.isclose(line.slope(t), central(line, t), rtol=1e-8, atol=0.0) for t in LENGTHS
    )


class TestMutationControls:
    """Check 4: each control breaks one line, and the difference check of 1 or 2 fails.

    Measured largest relative gaps under the mutants: clip ignored 1.8, shift without the top
    rate 0.55, left-sided tangent 1.7 (at 0), tie broken toward the smaller move 0.26 (at 0),
    bounded node without its scale 0.32, unit weights 0.28, full cell left out 0.35, slope state
    without the empty hold 2.2e-3.
    """

    def test_a_clip_that_is_ignored_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert not fails("A")
        tangent_edit(
            monkeypatch,
            (
                "return np.asarray(np.where(finite, tangent, 0.0), dtype=float)",
                "return np.asarray(np.where(finite, m, 0.0), dtype=float)",
            ),
        )
        assert fails("A")

    def test_a_shift_without_the_top_rate_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        tangent_edit(
            monkeypatch,
            ("rate = np.where(over[..., None], m - leader[..., None], m)", "rate = m"),
        )
        assert fails("A")

    def test_a_left_sided_tangent_fails_at_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert not fails("A", at_zero=True)
        tangent_edit(
            monkeypatch,
            (
                "((top > high) | ((top == high) & (leader > 0.0)))",
                "(top > high)",
            ),
            ("rising = (shifted == low) & (rate > 0.0)", "rising = shifted == low"),
            ("falling = (shifted == high) & (rate < 0.0)", "falling = shifted == high"),
        )
        assert fails("A", at_zero=True)

    def test_a_tie_broken_toward_the_smaller_move_fails_at_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Unit 8 holds two causes at the upper bound, moving up at 0.6 and 0.2.  The top is the
        # faster one, so the other falls at -0.4; the mutant keeps the slower one on top and
        # holds both.
        tangent_edit(
            monkeypatch,
            (
                "ties = np.where(finite & (z == top[..., None]), m, -np.inf)",
                "ties = np.where(finite & (z == top[..., None]), -m, -np.inf)",
            ),
        )
        assert fails("A", at_zero=True)

    def test_a_bounded_node_without_its_scale_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert not fails("B")
        slope_edit(
            monkeypatch,
            (
                "grad[:, k] = outcome - expit(inputs.at_observed(moved.logits)[:, k])",
                "grad[:, k] = (inputs.events - inputs.at_observed(moved.hazards))[:, k]",
            ),
        )
        assert fails("B")

    def test_unit_weights_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slope_edit(
            monkeypatch,
            ("summand = self.base[..., None] * grad * tangent", "summand = grad * tangent"),
        )
        assert fails("A")

    def test_a_slope_without_the_full_cell_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        slope_edit(
            monkeypatch,
            (
                "np.where(self.counted[..., None], summand, 0.0)",
                "np.where((self.counted & ~np.asarray(self.state.full)[:, inputs.observed].T)[..., None],"
                " summand, 0.0)",
            ),
        )
        assert fails("A")

    def test_a_slope_state_without_the_empty_hold_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The slope rebuilds its moved state with the empty cause back at the lower bound, while
        # the loss keeps the hold.  The other cause's hazard in the empty cell's group changes.
        slope_edit(
            monkeypatch,
            (
                "moved = self.state.moved(t * self.move, inputs.bounds)",
                "moved = HazardState(clip_logits(self.state.logits + t * self.move,"
                " inputs.bounds, self.bounded), self.state.full, self.state.node_bounds)",
            ),
        )
        assert fails("A")
