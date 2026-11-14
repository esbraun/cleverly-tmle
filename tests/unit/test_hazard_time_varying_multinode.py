"""Hazard targeting with a treatment at several nodes and covariates after baseline (X70 (b2)).

Law (M) of :mod:`tests.discrete_law_hazard_time_varying` is passed as data, one row per
support point weighted by its probability.  The (b1) backward pass runs with the multi-node
slot rule: node ``k`` is evaluated at each unit's observed history, so its slots are the
regimen's values of ``A_k`` there, with the hazards predicted at the observed earlier
treatments and ``A_k`` at each slot's level.

Two detection cases.  A defect of the panel (a term, a weight, an at-risk value, a soft event,
a clever covariate) shows under "wrong hazard and wrong continuation, correct mechanism", as on
the held design.  A defect of the initial slot values is repaired by the walk under a correct
mechanism, because any initial fit that is a function of ``(H_k, a)`` is, so it shows on the
other side of double robustness: correct (saturated) hazard and continuation learners with a
mechanism that omits ``L_k``.  Each mutation keeps a control on which its term vanishes.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from cleverly.exceptions import CapabilityError, MethodConfigurationError, RefusalReason
from cleverly.interventions import Incremental, Shift, Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen, LongitudinalData
from cleverly.longitudinal import estimator as estimator_module
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal import hazard_backward as backward
from cleverly.longitudinal.regimen import counterfactual_policy_density
from cleverly.methods import LTMLE_TILT_REFUSAL
from tests.discrete_law_hazard_time_varying import (
    CellMeans,
    CellProbabilities,
    MultiNodeLaw,
    NoL,
)
from tests.unit.test_hazard_time_varying import (
    ClippedLinear,
    Logistic,
    _curve_without_covariate_term,
    _draw,
    _drop,
    _escape,
    _hard_at_risk,
    _refuse,
    covariate_case,
    frozen_direction,
    mutated_panel,
    spied,
    tail_dropped,
)

TOL = 1e-10
EXACT = 1e-8
#: The floor a detected mutation must miss the truth by.
MISSED = 1e-3
STEP = 2e-2
LAW = MultiNodeLaw()

Density = Callable[[int, tuple[Any, ...]], np.ndarray]


# ------------------------------------------------------------------ the regimens


def _rule_node(node: int) -> Callable[[Any], np.ndarray]:
    """The rule of law (M) on a rule's history frame ``[W, L_1, ..., L_k]``."""
    if node == 1:
        return lambda h: np.asarray(h["W1"], dtype=int)
    return lambda h: (np.asarray(h[f"L{node}"], dtype=float) > 0.0).astype(int)


def _histories(law: MultiNodeLaw, frame: Any, node: int) -> list[tuple[Any, ...]]:
    """``H_k`` of every row of a policy frame, which carries the earlier treatments."""
    out = []
    w1 = np.asarray(frame["W1"], dtype=float).astype(int)
    w2 = np.asarray(frame["W2"], dtype=float).astype(int)
    arms = [np.asarray(frame[f"A{s}"], dtype=float).astype(int) for s in range(1, node)]
    covariates = [np.asarray(frame[f"L{s}"], dtype=float) for s in range(2, node + 1)]
    for i in range(len(w1)):
        history: list[Any] = [w1[i], w2[i], None]
        for s in range(node - 1):
            value = covariates[s][i]
            history.extend([arms[s][i], int(value) if value != 0 else None])
        out.append(tuple(history))
    return out


def _policy_node(law: MultiNodeLaw, node: int) -> Stochastic:
    def density(frame: Any) -> np.ndarray:
        return np.array([law.policy(node, history) for history in _histories(law, frame, node)])

    return Stochastic(density, "q", density_kind="known")


def regimens(law: MultiNodeLaw, *names: str) -> dict[str, Any]:
    """The regimens of law (M) by name: ``always``, ``never``, ``rule`` and ``policy``."""
    nodes = range(1, law.n_times + 1)
    every = {
        "never": 0,
        "always": 1,
        "rule": DynamicRegimen("rule", tuple(_rule_node(k) for k in nodes), rule_kind="known"),
        "policy": DynamicRegimen("policy", tuple(_policy_node(law, k) for k in nodes)),
    }
    return {name: every[name] for name in (names or every)}


def densities(law: MultiNodeLaw, *names: str) -> dict[str, Density]:
    """The law module's density of each named regimen."""
    every = {
        "never": law.static(0),
        "always": law.static(1),
        "rule": law.rule,
        "policy": law.policy,
    }
    return {name: every[name] for name in (names or every)}


STATIC = ("never", "always")
ALL = ("never", "always", "rule", "policy")


# ------------------------------------------------------------------ the fits


def fit(
    law: MultiNodeLaw,
    names: tuple[str, ...] = ALL,
    *,
    hazard: Any = None,
    continuation: Any = None,
    treatment: Any = None,
    censoring: Any = None,
    frame: Any = None,
    columns: dict[str, Any] | None = None,
    plans: dict[str, Any] | None = None,
    **settings: Any,
) -> Any:
    """A one-step hazard fit of law-as-data: wrong hazard and continuation, right mechanism.

    ``continuation=False`` passes no ``pseudo_learner``, as a baseline-only fit needs.
    ``plans`` replaces the named regimens.
    """
    if continuation is not False:
        settings["pseudo_learner"] = NoL(CellMeans()) if continuation is None else continuation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(
            regimens(law, *names) if plans is None else plans,
            n_folds=settings.pop("n_folds", 1),
            survival_model=settings.pop("survival_model", "hazard"),
            targeting=settings.pop("targeting", "one_step"),
            step_size=STEP,
            tol=settings.pop("tol", TOL),
            g_bounds=settings.pop("g_bounds", (1e-8, 1.0)),
            outcome_learner=NoL(Logistic()) if hazard is None else hazard,
            treatment_learner=CellProbabilities() if treatment is None else treatment,
            censoring_learner=CellMeans() if censoring is None else censoring,
            **settings,
        ).fit(
            law.frame() if frame is None else frame,
            **(law.columns() if columns is None else columns),
        )


def saturated_outcome() -> dict[str, Any]:
    """The other side of double robustness: saturated hazard and continuation, the mechanism
    without ``L_k``."""
    return {
        "hazard": CellMeans(),
        "continuation": CellMeans(),
        "treatment": NoL(CellProbabilities()),
        "censoring": NoL(CellMeans()),
    }


def estimate(result: Any, label: str, t: int, cause: str | None = None) -> float:
    if cause is None:
        return float(result.estimates[f"risk_regimen[{label} @ t={t}]"].psi)
    return float(result.estimates[f"cif_regimen[{label}, {cause} @ t={t}]"].psi)


def worst_miss(
    law: MultiNodeLaw, result: Any, names: tuple[str, ...], horizons: Any = None
) -> float:
    """The largest distance of an estimate from the enumerated truth."""
    causes = {1: None} if law.causes == 1 else {1: "relapse", 2: "death"}
    miss = 0.0
    for label, density in densities(law, *names).items():
        for j, cause in causes.items():
            for t in range(1, law.n_times + 1) if horizons is None else horizons:
                miss = max(miss, abs(estimate(result, label, t, cause) - law.truth(density, t, j)))
    return miss


@contextmanager
def patched(target: Any, name: str, value: Any) -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(target, name, value)
        yield


# ------------------------------------------------------------------ the law


def test_the_law_is_a_distribution_and_its_eif_is_centred() -> None:
    """The support masses sum to one, every ``g_k`` lies in ``[0.15, 0.85]`` and each EIF has mean
    zero, for each regimen."""
    assert LAW.probs.sum() == pytest.approx(1.0, abs=1e-14)
    probs = LAW.probs / LAW.probs.sum()
    for density in densities(LAW).values():
        assert abs(np.sum(probs * LAW.eif(density, 3))) < 1e-14
    frame = LAW.frame()
    for node in range(1, 4):
        g = LAW.true_g(frame, node)[:, 1]
        assert g.min() >= 0.15 and g.max() <= 0.85


# --------------------------------------------------------- law-as-data witnesses


@pytest.mark.parametrize("causes", [1, 2])
def test_a_wrong_hazard_and_continuation_with_the_right_mechanism_is_exact(causes: int) -> None:
    """Static arms, a rule and a known policy reading ``A_{k-1}`` return the truth everywhere."""
    law = MultiNodeLaw(causes=causes)
    result = fit(law)
    assert worst_miss(law, result, ALL) < EXACT
    suffix = ", relapse" if causes == 2 else ""
    for label in ALL:
        record = result.fits[f"{label}{suffix} @ t=3"].hazard
        assert record.converged and len(record.nodes) == law.n_times and record.n_iter > 0
    # The iterated-mean fit of the shipped path, saturated, agrees with the law module.
    reference = fit(
        law,
        hazard=CellMeans(),
        continuation=CellMeans(),
        survival_model="iterated_mean",
        targeting="iterative",
    )
    assert worst_miss(law, reference, ALL) < 1e-10
    # The untargeted plug-in misses, so the exact estimate is not an accident.
    with backward.untargeted():
        assert worst_miss(law, fit(law), ALL) > MISSED


def test_a_misspecified_hazard_reading_the_covariate_is_exact() -> None:
    """The covariate detection case of (b1): a hazard reading ``L_k`` but additive, a constant
    continuation, the right mechanism."""
    assert worst_miss(LAW, fit(LAW, **covariate_case()), ALL) < EXACT


def test_the_influence_curve_is_the_efficient_influence_function() -> None:
    """Saturated learners: no walk moves; the curve is the iterated-mean fit's and the EIF."""
    result = fit(LAW, hazard=CellMeans(), continuation=CellMeans())
    reference = fit(
        LAW,
        hazard=CellMeans(),
        continuation=CellMeans(),
        survival_model="iterated_mean",
        targeting="iterative",
    )
    frame = LAW.frame()
    weights = frame["wt"].to_numpy() / frame["wt"].mean()
    for label, density in densities(LAW).items():
        for t in range(1, LAW.n_times + 1):
            key = f"{label} @ t={t}"
            curve = result.fits[key].influence_curve_scaled
            assert result.fits[key].hazard.n_iter == 0
            np.testing.assert_allclose(
                curve, reference.fits[key].influence_curve_scaled, rtol=0.0, atol=1e-10
            )
            np.testing.assert_allclose(curve, LAW.eif(density, t) * weights, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("moves", [True, False])
def test_a_curve_without_its_covariate_term_misses_the_efficient_spread(moves: bool) -> None:
    """Mutation: the curve read from sub-node 0 only.  Control: ``L`` moves no hazard and no
    treatment, so the covariate term is zero row by row under static arms."""
    law = MultiNodeLaw(l_moves=moves)
    saturated = {"hazard": CellMeans(), "continuation": CellMeans()}
    full = fit(law, STATIC, **saturated)
    with patched(backward, "node_curve_terms", _curve_without_covariate_term()):
        mutated = fit(law, STATIC, **saturated)
    frame = law.frame()
    weights = frame["wt"].to_numpy() / frame["wt"].mean()
    probs = law.probs / law.probs.sum()
    for label, density in densities(law, *STATIC).items():
        key = f"{label} @ t=3"
        sd = np.sqrt(np.sum(probs * law.eif(density, 3) ** 2))
        curve = mutated.fits[key].influence_curve_scaled / weights
        if moves:
            assert abs(np.sqrt(np.sum(probs * curve**2)) / sd - 1.0) > 0.01
        else:
            np.testing.assert_allclose(
                mutated.fits[key].influence_curve_scaled,
                full.fits[key].influence_curve_scaled,
                rtol=0.0,
                atol=1e-10,
            )


# ------------------------------------------------------------- panel mutations


def test_mutation_no_continuation_term() -> None:
    """Sub-nodes after 0 leave the score and the direction; a saturated continuation controls it."""
    with (
        mutated_panel(lambda panel: _drop(panel, slice(1, None))),
        frozen_direction(slice(1, None)),
    ):
        assert worst_miss(LAW, fit(LAW, STATIC, **covariate_case()), STATIC) > MISSED
        control = fit(LAW, STATIC, hazard=Logistic(), continuation=CellMeans())
        assert worst_miss(LAW, control, STATIC) < EXACT


def test_mutation_no_hazard_term() -> None:
    """Sub-node 0 leaves the panel; a saturated hazard learner is the control."""
    with mutated_panel(lambda panel: _drop(panel, slice(0, 1))), frozen_direction(slice(0, 1)):
        assert worst_miss(LAW, fit(LAW, STATIC), STATIC) > MISSED
        assert worst_miss(LAW, fit(LAW, STATIC, hazard=CellMeans()), STATIC) < EXACT


def _with_retention(law: MultiNodeLaw, frame: pd.DataFrame) -> Callable[[Any], Any]:
    def change(panel: Any) -> Any:
        c = law.true_retention(frame, panel.node)[panel.rows]
        inverse = np.array(panel.inputs.inverse, copy=True)
        loss = np.array(panel.inputs.loss_weight, copy=True)
        inverse[:, 1:] *= c[:, None, None]
        loss[:, 1:] *= c[:, None]
        return replace(panel.inputs, inverse=inverse, loss_weight=loss)

    return change


@pytest.mark.parametrize("censored", [True, False])
def test_mutation_weight_without_its_own_node_retention(censored: bool) -> None:
    """The covariate term weighted by ``h_k c_k``; the control law has no dropout."""
    law = MultiNodeLaw(censored=censored)
    with mutated_panel(_with_retention(law, law.frame())):
        miss = worst_miss(law, fit(law, STATIC, **covariate_case()), STATIC)
    assert miss > MISSED if censored else miss < EXACT


@pytest.mark.parametrize("saturated", [False, True])
def test_mutation_stale_pseudo_outcome(saturated: bool) -> None:
    """Node ``k``'s soft events read node ``k + 1``'s initial curve; saturated fits control it."""
    last: list[Any] = []
    walk = backward.walk_curve
    pseudo = backward.pseudo_distribution

    def walk_spy(initial: Any, *args: Any, **kwargs: Any) -> Any:
        last[:] = [initial]
        return walk(initial, *args, **kwargs)

    def stale(state: Any, *args: Any) -> Any:
        return pseudo(last[0] if last else state, *args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backward, "walk_curve", walk_spy)
        patch.setattr(backward, "pseudo_distribution", stale)
        kw = {"hazard": CellMeans(), "continuation": CellMeans()} if saturated else covariate_case()
        miss = worst_miss(LAW, fit(LAW, STATIC, **kw), STATIC)
    assert miss < EXACT if saturated else miss > MISSED


@pytest.mark.parametrize("last_only", [False, True])
def test_mutation_hard_at_risk(last_only: bool) -> None:
    """``rho`` replaced by one at every later sub-node; events at the last node only control it."""
    law = MultiNodeLaw(last_only=last_only)
    with mutated_panel(_hard_at_risk):
        miss = worst_miss(law, fit(law, STATIC, **covariate_case()), STATIC)
    assert miss < EXACT if last_only else miss > MISSED


@pytest.mark.parametrize("n_times", [3, 2])
def test_mutation_continuation_kappa_without_its_tail(n_times: int) -> None:
    """The fit reports ``t = T`` only.  At ``T = 2`` node 1's one continuation sub-node is the
    horizon, where the tail is zero: the control."""
    law = MultiNodeLaw(n_times=n_times, levels=(2,) * n_times)
    with tail_dropped():
        result = fit(law, STATIC, horizons=[n_times], **covariate_case())
    miss = worst_miss(law, result, STATIC, horizons=[n_times])
    assert miss > MISSED if n_times == 3 else miss < EXACT


def _one_hot_rule(node: int) -> Callable[[Any], np.ndarray]:
    """The rule of law (M) at ``node`` as a one-hot density."""
    rule = _rule_node(node)
    return lambda h: np.eye(2)[rule(h)]


def _observed_slot(panel: Any) -> Any:
    q = np.zeros_like(panel.inputs.q)
    q[0, np.arange(panel.inputs.n), panel.inputs.observed] = 1.0
    return replace(panel.inputs, q=q)


@pytest.mark.parametrize("control", [False, True])
def test_mutation_observed_slot_at_a_policy_node(control: bool) -> None:
    """The pseudo-outcome and the node-1 plug-in read the observed level at every policy node,
    not the policy mixture.

    The control is the rule declared as a one-hot policy on a law where every unit follows the
    rule, so the observed level is the policy's slot for every unit.
    """
    law = MultiNodeLaw(follow=control)
    plans = None
    if control:
        nodes = range(1, law.n_times + 1)
        one_hot = tuple(Stochastic(_one_hot_rule(k), "q", density_kind="known") for k in nodes)
        plans = {"policy": DynamicRegimen("policy", one_hot)}
    with spied("node_slots") as calls, mutated_panel(_observed_slot):
        result = fit(law, ("policy",), plans=plans)
    if control:
        density = law.rule
        miss = max(abs(estimate(result, "policy", t) - law.truth(density, t)) for t in (1, 2, 3))
    else:
        miss = worst_miss(law, result, ("policy",))
    if control:
        # The resolver turns a one-hot density into the rule it equals, so every node holds one
        # slot and the mutation touches nothing: a no-touch check of the observed slot.
        assert all(entry.n_slots == 1 for entry in calls)
        assert miss < EXACT
    else:
        assert {entry.node for entry in calls if entry.n_slots == 2} == {1, 2, 3}
        assert miss > MISSED


# --------------------------------------------------------------- weights


def test_the_weight_is_the_iterated_mean_clever_covariate() -> None:
    """``h_k`` equals the iterated-mean clever covariate of the same plan at every node, with
    the mechanism fitted on the regimens' plans alone (no level plan)."""
    hazard = fit(LAW)
    reference = fit(LAW, survival_model="iterated_mean", targeting="iterative")
    for label in ALL:
        for t in (1, 3):
            ours = hazard.fits[f"{label} @ t={t}"].hazard_clever
            for step in reference.fits[f"{label} @ t={t}"].steps:
                np.testing.assert_allclose(
                    ours[:, step.time - 1], step.clever, rtol=1e-12, atol=1e-12
                )


# --------------------------------------------------------------- the slots


def _held(law: MultiNodeLaw) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The data of ``law`` as a held design: ``A1`` held over every node."""
    frame = law.frame().drop(columns=[f"A{k}" for k in range(2, law.n_times + 1)])
    columns = law.columns()
    columns["treatment"] = "A1"
    return frame, columns


HELD_LAW = MultiNodeLaw(copies=frozenset({2, 3}))


def _cell_means() -> dict[str, Any]:
    return {"hazard": CellMeans(), "continuation": CellMeans(), "censoring": CellMeans()}


def test_the_multinode_branch_reduces_to_the_held_one() -> None:
    """Every later treatment copies ``A_1``: the multi-node fit is the held fit of the same data.

    Saturated learners throughout, so every later treatment factor is exactly one and the
    repeated treatment columns of the multi-node design change no prediction.
    """
    multinode = fit(HELD_LAW, STATIC, **_cell_means())
    frame, columns = _held(HELD_LAW)
    held = fit(HELD_LAW, STATIC, frame=frame, columns=columns, **_cell_means())
    for label in STATIC:
        for t in range(1, HELD_LAW.n_times + 1):
            key = f"{label} @ t={t}"
            assert multinode.fits[key].psi_scaled == pytest.approx(
                held.fits[key].psi_scaled, abs=1e-10
            )
            np.testing.assert_allclose(
                multinode.fits[key].influence_curve_scaled,
                held.fits[key].influence_curve_scaled,
                rtol=0.0,
                atol=1e-10,
            )


def _held_slot_at_later_nodes() -> Callable[..., Any]:
    """Mutation: at ``k >= 2`` the held branch runs: one slot, the observed ``A_1`` code."""
    original = backward.node_slots

    def mutant(data: Any, plan: Any, node: int, initial: Any, slots: Any = None) -> Any:
        out = original(data, plan, node, initial, slots)
        if node == 1:
            return out
        code = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
        return backward.NodeSlots(
            node,
            np.asarray(initial.hazards[np.arange(data.n), node - 1, code])[:, None, :],
            np.ones((data.n, 1)),
            code[:, None],
            out.n_keys,
            np.zeros(data.n, dtype=np.int64),
            out.design,
            (out.design,),
        )

    return mutant


def test_mutation_the_held_slot_at_later_nodes_misses() -> None:
    """Under the rule and the policy a follower's ``A_k`` differs from ``A_1``; the held slot
    reads the hazard at ``A_1``.  The held reduction is the control."""
    with patched(backward, "node_slots", _held_slot_at_later_nodes()):
        mutated = fit(LAW, ("rule", "policy"), **saturated_outcome())
        control = fit(HELD_LAW, STATIC, **_cell_means())
    miss = worst_miss(LAW, mutated, ("rule", "policy"))
    assert miss > MISSED, miss
    assert worst_miss(HELD_LAW, control, STATIC) < EXACT
    # Without the mutation the same learners are exact.
    assert worst_miss(LAW, fit(LAW, ("rule", "policy"), **saturated_outcome()), ALL[2:]) < EXACT


def _hand_design(frame: pd.DataFrame, node: int, level: int | None = None) -> np.ndarray:
    """``[W1, W2, L_2, ..., L_k, A_1, ..., A_k]`` from the columns, ``A_k`` set to ``level``.

    Built from the frame, not from :meth:`LongitudinalData.history_design`; ``level=None``
    keeps the observed ``A_k``.
    """
    columns = [frame["W1"].to_numpy(), frame["W2"].to_numpy()]
    columns += [np.nan_to_num(frame[f"L{s}"].to_numpy()) for s in range(2, node + 1)]
    columns += [np.nan_to_num(frame[f"A{s}"].to_numpy()) for s in range(1, node)]
    current = np.nan_to_num(frame[f"A{node}"].to_numpy())
    columns.append(current if level is None else np.full(len(frame), float(level)))
    return np.column_stack(columns).astype(float)


def test_the_slot_hazards_are_the_current_level_predictions() -> None:
    """``hazards[:, k - 1, a]`` is the learner at the observed ``A_{<k}`` with ``A_k = a``, and
    ``observed`` the learner at the fit design."""
    frame = LAW.frame()
    data = LongitudinalData.from_frame(frame, **LAW.columns())
    at_risk, events = hazard_module.event_indicators(data)
    with spied("fit_hazards") as calls:
        fit(LAW, STATIC, hazard=Logistic())
    (initial,) = calls
    assert initial.hazards.shape == (data.n, 3, 2, 1)
    assert initial.observed is not None
    for node in range(1, 4):
        rows = at_risk[:, node - 1]
        fitted = _hand_design(frame, node)
        np.testing.assert_array_equal(fitted, data.history_design(node, include_current=True))
        learner = Logistic().fit(
            fitted[rows], events[rows, node - 1, 0], sample_weight=data.weights[rows]
        )
        for level in (0, 1):
            expected = learner.predict_proba(_hand_design(frame, node, level))[:, 1]
            np.testing.assert_allclose(
                initial.hazards[:, node - 1, level, 0], expected, rtol=0.0, atol=1e-12
            )
        np.testing.assert_allclose(
            initial.observed[:, node - 1, 0],
            learner.predict_proba(fitted)[:, 1],
            rtol=0.0,
            atol=1e-12,
        )


def _level_plan_designs(data: Any, time: int) -> dict[str, np.ndarray]:
    """Mutation: every node of the prediction design set to the level, the level plan of (a)."""
    out = {}
    for code in range(len(data.treatment_levels[time - 1])):
        values = np.full((data.n, data.n_times), float(code))
        out[hazard_module.current_level_label(code)] = data.history_design(
            time, treatment=values, include_current=True
        )
    return out


class CurrentOnly(CellMeans):
    """Cell means over ``W``, the last ``L`` column and the current treatment only."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Any:
        return super().fit(self._kept(X), y, sample_weight=sample_weight)

    def predict(self, X: Any) -> Any:
        return super().predict(self._kept(X))

    def predict_proba(self, X: Any) -> Any:
        return super().predict_proba(self._kept(X))

    @staticmethod
    def _kept(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        covariates = np.flatnonzero(np.any(matrix < 0.0, axis=0))
        keep = [0, 1, *covariates[-1:], matrix.shape[1] - 1]
        return matrix[:, keep]


@pytest.mark.parametrize("prefix", [True, False])
def test_mutation_level_plan_predictions(prefix: bool) -> None:
    """The slot hazards predicted at the level plan, every node at ``a``.  The hazard reads
    ``A_{k-1}``, so the rule and the policy miss; the control hazard does not read it, and a
    learner that drops it makes the mutation a no-op."""
    law = MultiNodeLaw(prefix_hazard=prefix)
    learners = saturated_outcome()
    if not prefix:
        learners["hazard"] = CurrentOnly()
    with patched(hazard_module, "_current_level_designs", _level_plan_designs):
        miss = worst_miss(law, fit(law, ("rule", "policy"), **learners), ("rule", "policy"))
    assert miss > MISSED if prefix else miss < EXACT


def _policy_at_level_paths() -> Callable[..., Any]:
    """Mutation: at a policy node ``k >= 2`` the policy is evaluated with every earlier
    treatment set to the slot's level, the paths of (a)'s ``counterfactual_policy_density``."""
    original = backward.node_slots

    def mutant(data: Any, plan: Any, node: int, initial: Any, slots: Any = None) -> Any:
        out = original(data, plan, node, initial, slots)
        if node == 1 or not plan.is_policy_node(node):
            return out
        q = np.zeros_like(out.q)
        for code in range(out.n_slots):
            path = np.full((data.n, data.n_times), float(code))
            q[:, code] = counterfactual_policy_density(plan.regimen, data, node, path)[:, code]
        return replace(out, q=q / q.sum(axis=1, keepdims=True))

    return mutant


@pytest.mark.parametrize("prefix", [True, False])
def test_mutation_policy_at_a_counterfactual_prefix(prefix: bool) -> None:
    """The policy read at the level path, not the observed history; a policy reading ``L_k``
    alone is the control."""
    law = MultiNodeLaw(policy_prefix=prefix)
    with patched(backward, "node_slots", _policy_at_level_paths()):
        miss = worst_miss(law, fit(law, ("policy",), **saturated_outcome()), ("policy",))
    assert miss > MISSED if prefix else miss < EXACT


def test_unequal_level_counts_key_each_node_by_its_own_levels() -> None:
    """A three-level node 2 under a known policy over its three levels is exact, and its panel
    keys three levels.  Mutation: the key count of node 1 (the held branch's) drops the third
    level's held cell."""
    law = MultiNodeLaw(levels=(2, 3, 2))
    assert worst_miss(law, fit(law), ALL) < EXACT
    held = MultiNodeLaw(levels=(2, 3, 2), no_event=frozenset({(2, 2)}))
    with spied("node_panel") as panels:
        result = fit(held, ("policy",))
    assert worst_miss(held, result, ("policy",)) < EXACT
    node2 = next(panel for panel in panels if panel.node == 2)
    assert node2.inputs.empty.shape[1] == 3
    assert node2.inputs.empty[0, 2, 0]
    original = backward.node_slots

    def short(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        return replace(out, n_keys=len(args[0].treatment_levels[0]))

    with spied("node_panel") as panels, patched(backward, "node_slots", short):
        try:
            fit(held, ("policy",))
        except (IndexError, ValueError):
            return
    node2 = next(panel for panel in panels if panel.node == 2)
    assert not (node2.inputs.empty.shape[1] == 3 and node2.inputs.empty[0, 2, 0])


def test_a_held_cell_is_keyed_by_the_node_treatment() -> None:
    """No event at node 3 under ``A_3 = 1``: the node-3 cell of level 1 is held at zero, the
    walks converge and the fit is exact.  Mutation: keys from the observed ``A_1``."""
    law = MultiNodeLaw(no_event=frozenset({(3, 1)}))
    with spied("node_panel") as panels:
        result = fit(law)
    assert worst_miss(law, result, ALL) < EXACT
    assert all(result.fits[f"{label} @ t=3"].hazard.converged for label in ALL)
    # A static ``never`` arm has no follower at level 1, so its panel holds nothing there.
    held = [panel for panel in panels if panel.node == 3 and (panel.inputs.cells == 1).any()]
    assert len(held) == len(ALL) - 1 and all(panel.inputs.empty[0, 1, 0] for panel in held)
    original = backward.node_slots

    def keyed_by_first(data: Any, plan: Any, node: int, initial: Any, slots: Any = None) -> Any:
        out = original(data, plan, node, initial, slots)
        code = np.asarray(np.nan_to_num(data.treatment[:, 0], nan=0.0), dtype=np.int64)
        return replace(out, keys=np.repeat(code[:, None], out.n_slots, axis=1))

    with spied("node_panel") as panels, patched(backward, "node_slots", keyed_by_first):
        fit(law, ("policy",))
    node3 = next(panel for panel in panels if panel.node == 3)
    assert not node3.inputs.empty[0, 1, 0]


# --------------------------------------------------------------- the window


def _window(law: MultiNodeLaw) -> LongitudinalData:
    return replace(LongitudinalData.from_frame(law.frame(), **law.columns()), markov_order=0)


def _old_window_arm_start(self: LongitudinalData, time: int) -> int:
    """Mutation: the window's arms start at its covariates, which drops ``A_{k-1}``."""
    if self.markov_order is None:
        return 1
    return self.window_start(time)


def test_a_markov_window_keeps_the_previous_arm_and_the_fit_exact() -> None:
    """Law (M) is order-0 Markov.  ``markov_order=0`` is exact with saturated windowed hazard
    and continuation and a mechanism without ``L_k``, and with the right mechanism and the
    detection-case learners."""
    data = _window(LAW)
    assert data.history_design(3, include_current=True).shape[1] == 5  # W1, W2, L3, A2, A3
    outcome = fit(LAW, frame=data, columns={}, **saturated_outcome())
    assert worst_miss(LAW, outcome, ALL) < EXACT
    assert worst_miss(LAW, fit(LAW, frame=data, columns={}), ALL) < EXACT


@pytest.mark.parametrize("prefix", [True, False])
def test_mutation_a_window_that_drops_the_previous_arm(prefix: bool) -> None:
    """The arms start at ``window_start``.  The hazard and the policy read ``A_{k-1}``, so the
    windowed hazard pools over it; the control reads neither."""
    law = MultiNodeLaw(prefix_hazard=prefix, policy_prefix=prefix)
    data = _window(law)
    with patched(LongitudinalData, "window_arm_start", _old_window_arm_start):
        assert data.history_design(3, include_current=True).shape[1] == 4  # no A2
        miss = worst_miss(law, fit(law, frame=data, columns={}, **saturated_outcome()), ALL)
    assert miss > MISSED if prefix else miss < EXACT


# ------------------------------------------------------------- declared factors


def _declared(law: MultiNodeLaw, frame: pd.DataFrame) -> dict[str, Any]:
    treat = np.column_stack([law.true_g(frame, k)[:, 1] for k in range(1, law.n_times + 1)])
    retain = np.column_stack([law.true_retention(frame, k) for k in range(1, law.n_times + 1)])
    columns = law.columns()
    columns.update(
        treatment_probabilities=treat,
        censoring_probabilities=retain,
        # The law as data weights its support points by their mass, so the declared
        # probabilities are the weighted population's.
        mechanism_law="target",
    )
    return columns


def _second_factor_read_as_one(law: MultiNodeLaw, frame: pd.DataFrame) -> Callable[..., Any]:
    """Mutation: the declared node-2 treatment factor read as one in the cumulative."""
    original = backward.clever_weights
    g = law.true_g(frame, 2)
    observed = np.asarray(np.nan_to_num(frame["A2"].to_numpy(), nan=0.0), dtype=np.int64)
    factor = g[np.arange(len(frame)), observed]

    def mutant(*args: Any, **kwargs: Any) -> Any:
        weight = np.array(original(*args, **kwargs), copy=True)
        weight[:, 1:] *= factor[:, None]
        return weight

    return mutant


@pytest.mark.parametrize("control", [False, True])
def test_declared_factors_fit_and_their_second_node_is_read(control: bool) -> None:
    """True ``g_k`` and retention declared at every node (H26 does not apply here) give the
    truth.  Mutation: the node-2 factor read as one.  Control: node 2 copies ``A_1``, so that
    factor is one on every follower of a static arm."""
    law = MultiNodeLaw(copies=frozenset({2})) if control else LAW
    names = STATIC if control else ALL
    frame = law.frame()
    columns = _declared(law, frame)
    learners = {
        "treatment": _refuse()["treatment_learner"],
        "censoring": _refuse()["censoring_learner"],
    }
    assert worst_miss(law, fit(law, names, columns=columns, **learners), names) < EXACT
    with patched(backward, "clever_weights", _second_factor_read_as_one(law, frame)):
        miss = worst_miss(law, fit(law, names, columns=columns, **learners), names)
    assert miss < EXACT if control else miss > MISSED


# --------------------------------------------------------------- L1 alone


def _policy_l1(frame: Any) -> np.ndarray:
    p = 0.3 + 0.4 * np.asarray(frame["W2"], dtype=float)
    return np.column_stack([1.0 - p, p])


def _l1_regimens(n_times: int) -> dict[str, Any]:
    rule = DynamicRegimen(
        "rule",
        tuple(lambda h: np.asarray(h["W2"], dtype=int) for _ in range(n_times)),
        rule_kind="known",
    )
    node = Stochastic(_policy_l1, "q", density_kind="known")
    return {
        "never": 0,
        "always": 1,
        "rule": rule,
        "policy": DynamicRegimen("policy", (node,) * n_times),
    }


def test_a_covariate_at_node_one_alone_runs_the_path_engine() -> None:
    """``W2`` as a node-1 block on a multi-node design gives the fit with ``W2`` declared
    baseline, through part (a)'s path engine, under static arms, a rule and a policy reading
    it.  H23 and H26 still refuse such a design; a block at node 2 routes to the backward pass."""
    frame = LAW.frame().drop(columns=["L2", "L3"])
    base = LAW.columns(l1=True)
    base["baseline"], base["time_varying"] = ["W1", "W2"], [[], [], []]
    first = LAW.columns(l1=True)
    settings: dict[str, Any] = {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
    }

    def path_fit(columns: dict[str, Any], **extra: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LTMLE(
                _l1_regimens(3),
                n_folds=1,
                survival_model="hazard",
                targeting="iterative",
                tol=TOL,
                g_bounds=(1e-8, 1.0),
                censoring_learner=CellMeans(),
                **settings,
                **extra,
            ).fit(frame, **columns)

    with spied("node_panel") as panels:
        a, b = path_fit(base), path_fit(first)
    assert not panels
    for name, value in a.estimates.items():
        assert b.estimates[name].psi == pytest.approx(value.psi, abs=1e-12)
    data = replace(LongitudinalData.from_frame(frame, **first), markov_order=0)
    with pytest.raises(
        CapabilityError, match=re.escape(hazard_module.HAZARD_MARKOV_ORDER_REFUSAL.message())
    ):
        LTMLE(_l1_regimens(3), survival_model="hazard", **_refuse()).fit(data)
    treat = np.full((len(frame), 3), 0.5)
    with pytest.raises(CapabilityError, match="declares the treatment probabilities of 'A2'"):
        LTMLE(_l1_regimens(3), survival_model="hazard", **_refuse()).fit(
            frame, treatment_probabilities=treat, mechanism_law="target", **first
        )
    with spied("node_panel") as panels:
        fit(LAW, STATIC)
    assert {panel.node for panel in panels} == {1, 2, 3}


# ----------------------------------------------------------- statistical witnesses


def _gaussian_frame(n: int, seed: int) -> pd.DataFrame:
    """Law (M) with ``L_k`` Gaussian AR(1) given ``(W1, A_{k-1}, L_{k-1})``, ``T = 3``."""
    rng = np.random.default_rng(seed)
    w1 = (rng.random(n) < 0.4).astype(float)
    w2 = (rng.random(n) < np.where(w1 == 1, 0.55, 0.3)).astype(float)
    out: dict[str, np.ndarray] = {"W1": w1, "W2": w2}
    alive = np.ones(n, dtype=bool)
    l, previous = np.zeros(n), np.zeros(n)
    for k in range(1, 4):
        if k > 1:
            l = np.where(alive, 0.5 * l + 0.6 * w1 - 0.5 * previous + rng.normal(size=n), np.nan)
            out[f"L{k}"] = l
            l = np.nan_to_num(l)
        lin = -0.2 + 0.6 * w1 - 0.4 * w2 if k == 1 else -0.4 + 0.5 * w1 + 0.8 * previous + 0.6 * l
        a = (rng.random(n) < expit(lin)).astype(float)
        out[f"A{k}"] = np.where(alive, a, np.nan)
        c = rng.random(n) < expit(1.8 - 0.4 * w2 + 0.3 * a - 0.8 * l)
        out[f"C{k}"] = np.where(alive, c, np.nan)
        alive = alive & c
        y = rng.random(n) < _gaussian_hazard(k, w1, previous, a, l)
        out[f"Y{k}"] = np.where(alive, y, np.nan)
        alive = alive & ~y
        previous = np.where(np.isnan(out[f"A{k}"]), 0.0, a)
    return pd.DataFrame(out)


def _gaussian_hazard(k: int, w1: Any, previous: Any, a: Any, l: Any) -> Any:
    prefix = 0.6 * previous if k > 1 else 0.0
    return expit(-1.6 + 0.2 * k - 0.6 * a + 0.5 * w1 + 0.9 * l + prefix)


def _smooth_policy(k: int, l: float) -> float:
    return float(expit(-0.5 + l)) if k > 1 else float(expit(-0.5))


def _gaussian_truth(regimen: str, t: int, nodes: int = 40) -> float:
    """Nested Gauss-Hermite quadrature of the risk at horizon ``t``."""
    x, wq = np.polynomial.hermite_e.hermegauss(nodes)
    wq = wq / wq.sum()

    def q_of(k: int, l: float) -> np.ndarray:
        if regimen == "policy":
            p = _smooth_policy(k, l)
            return np.array([1.0 - p, p])
        return np.eye(2)[1 if regimen == "always" else 0]

    def q(k: int, w1: float, previous: float, l: float) -> float:
        total = 0.0
        for a, qa in enumerate(q_of(k, l)):
            if qa == 0.0:
                continue
            lam = float(_gaussian_hazard(k, w1, previous, a, l))
            value = lam
            if k < t:
                mean = 0.5 * l + 0.6 * w1 - 0.5 * a
                later = sum(
                    weight * q(k + 1, w1, a, mean + node)
                    for node, weight in zip(x, wq, strict=True)
                )
                value += (1.0 - lam) * later
            total += qa * value
        return total

    return sum(p1 * q(1, w1, 0.0, 0.0) for w1, p1 in ((0.0, 0.6), (1.0, 0.4)))


def _smooth_policy_node(k: int) -> Stochastic:
    def density(frame: Any) -> np.ndarray:
        l = np.asarray(frame[f"L{k}"], dtype=float) if k > 1 else np.zeros(len(frame))
        p = np.array([_smooth_policy(k, value) for value in l])
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, "q", density_kind="known")


def test_a_continuous_covariate_after_baseline_covers_the_truth() -> None:
    """Gaussian ``L_k``: logistic mechanism (correct), wrong hazard and continuation, ``T = 3``,
    static arms and a smooth known policy."""
    frame = _gaussian_frame(20_000, 7)
    for regimen in ("always", "policy"):
        assert abs(_gaussian_truth(regimen, 3, 20) - _gaussian_truth(regimen, 3, 30)) < 1e-10
    columns = {
        "treatment": ["A1", "A2", "A3"],
        "baseline": ["W1", "W2"],
        "censoring": ["C1", "C2", "C3"],
        "time_varying": [[], ["L2"], ["L3"]],
        "outcome": ["Y1", "Y2", "Y3"],
    }
    regimens = {
        "never": 0,
        "always": 1,
        "policy": DynamicRegimen("policy", tuple(_smooth_policy_node(k) for k in (1, 2, 3))),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(
            regimens,
            n_folds=1,
            survival_model="hazard",
            targeting="one_step",
            step_size=STEP,
            tol=TOL,
            outcome_learner=NoL(Logistic()),
            pseudo_learner=ClippedLinear(),
            treatment_learner=Logistic(),
            censoring_learner=Logistic(),
        ).fit(frame, **columns)
    for label in regimens:
        for t in (1, 2, 3):
            value = result.estimates[f"risk_regimen[{label} @ t={t}]"]
            assert abs(value.psi - _gaussian_truth(label, t, 20)) < 4.0 * value.std_error
        record = result.fits[f"{label} @ t=3"].hazard
        assert all(node.relative_score <= TOL for node in record.nodes)


def test_cross_fitting_covers_the_truth_and_solves_every_node() -> None:
    """Five folds on draws: within four standard errors, and every pooled node walk solved."""
    frame = _draw(LAW, 20_000, 3)
    result = fit(
        LAW,
        frame=frame,
        columns={**LAW.columns(), "weights": None},
        n_folds=5,
        continuation=CellMeans(),
        g_bounds=(0.01, 1.0),
        random_state=0,
    )
    for label, density in densities(LAW).items():
        for t in range(1, LAW.n_times + 1):
            value = result.estimates[f"risk_regimen[{label} @ t={t}]"]
            assert abs(value.psi - LAW.truth(density, t)) < 4.0 * value.std_error
        record = result.fits[f"{label} @ t=3"].hazard
        assert record.converged
        assert all(node.relative_score <= TOL for node in record.nodes)


# ------------------------------------------------------------------ refusals


def _mtp_regimen(node: int) -> DynamicRegimen:
    plan: list[Any] = [0, 0, 0]
    plan[node - 1] = Shift(1.0, cap=1.0)
    return DynamicRegimen("shifted", tuple(plan))


def _tilt_regimen(*nodes: int) -> DynamicRegimen:
    plan: list[Any] = [0, 0, 0]
    for node in nodes:
        plan[node - 1] = Incremental(2.0)
    return DynamicRegimen("tilted", tuple(plan))


@pytest.mark.parametrize("node", [1, 2])
@pytest.mark.parametrize("kind", ["mtp", "tilt"])
def test_h27_a_policy_or_a_tilt_at_any_node_is_refused(kind: str, node: int) -> None:
    """A modified treatment policy, and an odds tilt of a declared mechanism, at node 1 or 2."""
    frame = LAW.frame()
    columns = _declared(LAW, frame) if kind == "tilt" else LAW.columns()
    regimen = _mtp_regimen(node) if kind == "mtp" else _tilt_regimen(node)
    what = (
        "a modified treatment policy"
        if kind == "mtp"
        else ("an odds tilt of the declared treatment mechanism")
    )
    expected = backward.hazard_time_varying_regimen_refusal(regimen.label, kind, node)
    with pytest.raises(CapabilityError, match=_escape(expected.message())) as raised:
        LTMLE(
            {"never": 0, regimen.label: regimen},
            survival_model="hazard",
            targeting="one_step",
            pseudo_learner=_refuse()["outcome_learner"],
            **_refuse(),
        ).fit(frame, **columns)
    assert raised.value.refusal.reason is RefusalReason.API_REDIRECTION
    assert raised.value.refusal.roadmap is None
    assert f"holds {what} at node {node}" in str(raised.value)
    assert "survival_model='iterated_mean', which fits this regimen" in str(raised.value)


@pytest.mark.parametrize("with_mtp", [False, True])
def test_a_tilt_of_an_estimated_mechanism_meets_the_shipped_tilt_refusal(with_mtp: bool) -> None:
    """No declared factor at the tilt node: ``_resolve_tilts`` refuses, not H27, whose route
    would refuse the same regimen."""
    plan: list[Any] = [0, Incremental(2.0), Shift(1.0, cap=1.0) if with_mtp else 0]
    regimen = DynamicRegimen("tilted", tuple(plan))
    with pytest.raises(CapabilityError, match=_escape(LTMLE_TILT_REFUSAL.message())) as raised:
        LTMLE(
            {"never": 0, "tilted": regimen},
            survival_model="hazard",
            targeting="one_step",
            pseudo_learner=_refuse()["outcome_learner"],
            **_refuse(),
        ).fit(LAW.frame(), **LAW.columns())
    assert raised.value.refusal == LTMLE_TILT_REFUSAL


def _continuous_second_node() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = LAW.frame()
    rng = np.random.default_rng(0)
    frame["A2"] = np.where(frame["A2"].isna(), np.nan, rng.normal(size=len(frame)))
    return frame, {**LAW.columns(), "continuous_treatment": ["A2"]}


def test_h28_comes_before_h22_and_h22_before_h27() -> None:
    frame, columns = _continuous_second_node()
    with pytest.raises(MethodConfigurationError, match="needs targeting='one_step'"):
        LTMLE(STATIC_REGIMENS, survival_model="hazard", targeting="iterative", **_refuse()).fit(
            frame, **columns
        )
    shifted = DynamicRegimen("shifted", (Shift(1.0, cap=1.0), 0, 0))
    with pytest.raises(CapabilityError, match=r"node 2 \('A2'\) is continuous"):
        LTMLE(
            {"never": 0, "shifted": shifted},
            survival_model="hazard",
            targeting="one_step",
            **_refuse(),
        ).fit(frame, **columns)


STATIC_REGIMENS = {"never": 0, "always": 1}


def test_h23_still_refuses_a_markov_order_without_later_covariates() -> None:
    """``markov_order=`` is fitted with covariates after baseline and refused on the path engine."""
    frame = LAW.frame().drop(columns=["L2", "L3"])
    columns = LAW.columns()
    columns["time_varying"] = [[], [], []]
    data = replace(LongitudinalData.from_frame(frame, **columns), markov_order=0)
    with pytest.raises(CapabilityError, match="markov_order= restricts"):
        LTMLE(STATIC_REGIMENS, survival_model="hazard", **_refuse()).fit(data)


def test_the_nuisance_report_scores_each_hazard_at_the_observed_history() -> None:
    """A (b2) fit's hazard rows read ``observed``: the learner at each unit's own history."""
    from tests.unit._capability_sweep_support import fit_hazard_time_varying_multinode
    from tests.unit.test_hazard_multinode import _observed_rho

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_hazard_time_varying_multinode()
    data = result.data
    _, events = hazard_module.event_indicators(data)
    expected = _observed_rho(result)
    rows = [row for row in result.diagnostics.nuisance_models().rows if row.role == "hazard"]
    assert len(rows) == data.n_times
    for row in rows:
        mask, rho = expected[row.time, 0]
        p = np.clip(rho[mask], 1e-12, 1.0 - 1e-12)
        y = events[mask, row.time - 1, 0]
        w = data.weights[mask]
        log_loss = -np.average(y * np.log(p) + (1.0 - y) * np.log(1.0 - p), weights=w)
        assert row.loss == pytest.approx(log_loss, rel=0.0, abs=1e-12)


def test_the_estimator_routes_a_multinode_design_with_later_covariates_past_the_paths() -> None:
    """Neither the path preflight nor the level plans run: the backward pass gets no level plan
    and the mechanism is fitted on the regimens' plans alone."""
    seen: list[Any] = []
    original_paths = estimator_module.path_slots
    original_pass = estimator_module.target_time_varying

    def paths(*args: Any, **kwargs: Any) -> Any:
        seen.append("paths")
        return original_paths(*args, **kwargs)

    def backward_pass(data: Any, plans: Any, levels: Any, mechanism: Any, **kwargs: Any) -> Any:
        seen.append(tuple(levels))
        assert all(not label.startswith("\x00") for node in mechanism.treatment for label in node)
        return original_pass(data, plans, levels, mechanism, **kwargs)

    with (
        patched(estimator_module, "path_slots", paths),
        patched(estimator_module, "target_time_varying", backward_pass),
    ):
        fit(LAW, STATIC)
    assert seen == [()]
