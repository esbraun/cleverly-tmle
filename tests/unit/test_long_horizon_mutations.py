"""Witnesses and mutation controls of the long-horizon recursion, on the exact law.

Every component of the K-node recursion that can vanish at the truth has a nonzero witness here,
or a mutation that must move a stated quantity by more than a named floor.  Each check fits
``LTMLE`` or ``SDR`` on the weighted exact frame of :mod:`tests.discrete_law_longitudinal_horizon`
with ``n_folds=1``, so an unmutated fit with a known mechanism returns the truth to the
exact-frame noise of about ``1e-12``.

A floor of ``1e-3`` applies to an estimate move unless a check names another.  Where no declared
learner reaches five times the floor at one node, the check asserts at one tenth of the exact
reading recorded below, never below ``1e-8``.  The recorded readings come from the same fits;
``tests/unit/test_long_horizon_design.py`` holds the design-time readings on the other laws.

The pooled cross-fitted update, the fold-targeted carry (M3), the trajectory-level folds and the
SDR constructions (M7a, M7b) run at K = 5 on a draw; the SDR partial weights run at K = 5 on the
frame.  The two-node versions live in
``tests/unit/test_pooled_longitudinal_targeting.py`` and ``tests/unit/test_sdr_construction.py``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from cleverly.longitudinal import sequential
from cleverly.longitudinal.sequential import Mechanism
from tests import discrete_law_longitudinal_horizon as H
from tests import longitudinal_horizon_plans as P
from tests.studies import long_horizon_common as C
from tests.studies import long_horizon_mutations as M
from tests.unit import test_pooled_longitudinal_targeting as T

SETTINGS = [(3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3)]
SETTING_IDS = [f"K{K}m{m}" for K, m in SETTINGS]

#: The witness plans of the targeting rows, per arm count.
PLANS = {2: ("never", "always", "respond", "coin"), 3: ("low", "high", "respond", "mix")}

#: The exact-frame noise: an unmutated fit with a known mechanism returns the truth to this.
EXACT = 1e-10
#: The estimate-move floor of every mutation that names no other.
FLOOR = 1e-3


def _frame(K: int, m: int, law: H.Law = H.L_R) -> pd.DataFrame:
    return H.weighted_frame(K, m, law)


def _weight(frame: pd.DataFrame) -> np.ndarray:
    return frame["w"].to_numpy(dtype=float)


def _top(m: int) -> str:
    return "high" if m == 3 else "always"


def _assert_moves(shift: float, record: float) -> None:
    """``|shift| > FLOOR`` where the record reaches five times it, else one tenth of the record."""
    floor = FLOOR if abs(record) >= 5 * FLOOR else max(abs(record) / 10, 1e-8)
    assert abs(shift) > floor, (shift, record)
    assert shift == pytest.approx(record, rel=1e-2, abs=1e-6)


# ----------------------------------------------------------------------------- (a) per node
#: The per-node rung at (K, m, plan, node) where D1, the W-blind GLM, misses five times a floor.
#: D2 is the intercept and D3 the fixed-offset GLM.  Every pair meets five times all three floors
#: with its rung on L_R, so no pair falls back to one tenth of a record.
RUNGS = {
    (3, 3, "low", 1): "D3",
    (3, 3, "respond", 2): "D2",
    (3, 3, "mix", 3): "D2",
    (4, 2, "never", 1): "D3",
    (4, 2, "never", 2): "D2",
    (4, 2, "respond", 4): "D2",
    (4, 3, "low", 1): "D3",
    (4, 3, "respond", 2): "D2",
    (4, 3, "mix", 4): "D2",
    (5, 3, "low", 1): "D3",
    (5, 3, "respond", 2): "D2",
    (5, 3, "mix", 5): "D2",
}


def _rung(name: str, plan: str, K: int, m: int) -> Any:
    policy = tuple(sorted(H.policy_nodes(plan, K)))
    return {
        "D1": C.ColumnSubsetGLM("W", m, K, policy),
        "D2": C.InterceptOnly(),
        "D3": C.FixedOffsetLearner(),
    }[name]


def _pattern(plan: str, K: int, m: int, learners: dict[int, Any]) -> C.NodePattern:
    return C.NodePattern(learners, None, m, K, tuple(sorted(H.policy_nodes(plan, K))))


TARGETING_CASES = [(K, m, plan) for K, m in SETTINGS for plan in PLANS[m]]


@pytest.mark.parametrize(
    ("K", "m", "plan"), TARGETING_CASES, ids=[f"K{K}m{m}-{p}" for K, m, p in TARGETING_CASES]
)
def test_each_node_is_targeted_with_a_wrong_outcome_there(K: int, m: int, plan: str) -> None:
    """Per node: the truth, ``|epsilon_t| > 1e-3``, a skip moves ``> 1e-4``, its score ``> 1e-6``.

    ``NodePattern`` fits the node's rung at node ``t`` and the declared GLM elsewhere, with the
    known mechanism.  One plan per fit: a policy plan's outcome widths collide with a label's.
    """
    frame = _frame(K, m)
    truth = H.truth(K, m, plan)
    for t in range(1, K + 1):
        rung = RUNGS.get((K, m, plan, t), "D1")
        learner = _pattern(plan, K, m, {t: _rung(rung, plan, K, m)})
        base = C.fit(frame, K, m, (plan,), weights="w", outcome_learner=learner)
        with M.skip_update({t}):
            skipped = C.fit(frame, K, m, (plan,), weights="w", outcome_learner=learner)
        where = (t, rung)
        assert M.estimate(base, plan) == pytest.approx(truth, abs=1e-8), where
        assert abs(M.epsilon(base, plan, t)) > 5e-3, where
        assert abs(M.estimate(skipped, plan) - M.estimate(base, plan)) > 5e-4, where
        assert abs(M.node_score(skipped, plan, t)) > 5e-6, where
        # The unmutated node solves its score, so the skip is what broke it.
        assert abs(M.node_score(base, plan, t)) < 1e-9, where


def test_the_glm_leaves_the_low_plan_almost_untargeted_at_node_one() -> None:
    """Why ``low`` at node 1 needs D3: with D1 (= D2 there) ``|epsilon_1|`` is ``1.8e-3``."""
    K, m, plan = 5, 3, "low"
    frame = _frame(K, m)
    readings = {}
    for rung in ("D1", "D2", "D3"):
        learner = _pattern(plan, K, m, {1: _rung(rung, plan, K, m)})
        readings[rung] = abs(
            M.epsilon(C.fit(frame, K, m, (plan,), weights="w", outcome_learner=learner), plan, 1)
        )
    assert readings["D1"] == pytest.approx(readings["D2"], rel=1e-8)
    assert readings["D1"] < 5e-3 < readings["D3"]


# ----------------------------------------------------------------------------- (b) every node
@pytest.mark.parametrize(("K", "m"), SETTINGS, ids=SETTING_IDS)
def test_the_glm_curve_is_the_oracle_curve_at_its_own_regressions(K: int, m: int) -> None:
    """The declared GLM everywhere: truth to ``1e-8``; the curve is ``curve()`` at the fit's
    regressions to ``1e-10`` and departs from the EIF by more than ``1e-3``."""
    frame = _frame(K, m)
    weight = _weight(frame)
    result = C.fit(frame, K, m, PLANS[m], weights="w")
    for plan in PLANS[m]:
        assert M.estimate(result, plan) == pytest.approx(H.truth(K, m, plan), abs=1e-8)
        scale = max(1.0, float(np.max(np.abs(H.eif(K, m, plan)))))
        assert M.curve_departure(result, plan, K, m, H.L_R, weight) < EXACT * scale
        assert M.eif_departure(result, plan, K, m, H.L_R, weight) > 1e-3


# ----------------------------------------------------------------------------- (c), (d) censoring
#: The shift of ``c_t = 1`` at each node, static top plan on L_C (``mutg_*_g4_glm.log``).
CENSORING_SHIFTS = {
    (5, 3): (-8.977e-3, -1.511e-2, -1.730e-2, -1.817e-2, -1.998e-2),
    (5, 2): (-8.438e-3, -1.406e-2, -1.603e-2, -1.680e-2, -1.848e-2),
    (3, 3): (-9.978e-3, -1.614e-2, -1.816e-2),
}


@pytest.mark.parametrize(("K", "m"), sorted(CENSORING_SHIFTS), ids=lambda v: str(v))
def test_each_censoring_factor_moves_the_estimate(K: int, m: int) -> None:
    """``c_t = 1`` at node ``t`` moves the estimate and breaks the curve at the fit's regressions."""
    law = H.L_C
    frame = _frame(K, m, law)
    weight = _weight(frame)
    plan = _top(m)
    base = C.fit(frame, K, m, (plan,), weights="w", g_bounds=(1e-12, 1.0))
    assert M.estimate(base, plan) == pytest.approx(H.truth(K, m, plan), abs=EXACT)
    for t, record in enumerate(CENSORING_SHIFTS[K, m], start=1):
        mutated = frame.copy()
        mutated.loc[mutated[f"A{t}"].notna(), f"pc{t}"] = 1.0
        result = C.fit(mutated, K, m, (plan,), weights="w", g_bounds=(1e-12, 1.0))
        _assert_moves(M.estimate(result, plan) - M.estimate(base, plan), record)
        assert M.curve_departure(result, plan, K, m, law, weight) > 1e-3


def _path_masks(frame: pd.DataFrame, K: int, top: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The oracle's at-risk and follower sets of the static plan ``top``, from the columns."""
    at_risk, following = [], []
    running = np.ones(len(frame), dtype=bool)
    for t in range(1, K + 1):
        at_risk.append(running.copy())
        arm = frame[f"A{t}"].to_numpy(dtype=float)
        stays = frame[f"C{t}"].to_numpy(dtype=float)
        running = running & (arm == top) & (stays == 1.0)
        following.append(running.copy())
    return at_risk, following


@pytest.mark.parametrize("m", [2, 3])
def test_a_censored_unit_does_not_follow_the_plan(m: int) -> None:
    """The masks are the oracle path sets; keeping censored units as followers moves the estimate."""
    K, law = 5, H.L_C
    frame = _frame(K, m, law)
    plan = _top(m)
    base = C.fit(frame, K, m, (plan,), weights="w", g_bounds=(1e-12, 1.0))
    at_risk, following = _path_masks(frame, K, m - 1)
    for step in base.fits[plan].steps:
        np.testing.assert_array_equal(step.at_risk, at_risk[step.time - 1])
        np.testing.assert_array_equal(step.trained_on, following[step.time - 1])
    with M.censored_units_follow():
        mutated = C.fit(frame, K, m, (plan,), weights="w", g_bounds=(1e-12, 1.0))
    assert abs(M.estimate(mutated, plan) - M.estimate(base, plan)) > 5 * FLOOR


# ----------------------------------------------------------------------------- (e), (g) plan inputs
def _per_node_truth(name: str, K: int, m: int, t: int) -> float:
    return H.truth(K, m, H.plan(name, K, m, nodes=(t,)))


@pytest.mark.parametrize(("K", "m"), SETTINGS, ids=SETTING_IDS)
def test_a_lagged_rule_reads_a_different_follower_set(K: int, m: int) -> None:
    """The rule reads ``L_{t-1}`` instead of ``L_t``: every node at once and one node at a time.

    The shift is the law-free truth difference, so each per-node fit is held to the mutated
    plan's truth, and each per-node shift to more than one tenth of its own value.
    """
    frame = _frame(K, m)
    every = C.fit(frame, K, m, ("respond", "lagged"), weights="w")
    shift = M.estimate(every, "lagged") - M.estimate(every, "respond")
    assert shift == pytest.approx(H.truth(K, m, "lagged") - H.truth(K, m, "respond"), abs=1e-8)
    assert abs(shift) > 5 * FLOOR
    for t in range(2, K + 1):
        regimens = {
            "respond": P.regimen("respond", K, m),
            "lagged": P.regimen("lagged", K, m, (t,)),
        }
        result = C.fit(frame, K, m, (), weights="w", regimens=regimens)
        truth = _per_node_truth("lagged", K, m, t)
        assert M.estimate(result, "lagged") == pytest.approx(truth, abs=1e-8)
        record = truth - H.truth(K, m, "respond")
        assert abs(M.estimate(result, "lagged") - M.estimate(result, "respond")) > abs(record) / 10
        assert abs(record) > FLOOR


@pytest.mark.parametrize("K", [3, 4, 5])
def test_a_policy_without_its_earlier_arm_is_a_different_policy(K: int) -> None:
    """``mix`` rotates by ``L_t + A_{t-1}``; ``mix_noprev`` drops ``A_{t-1}`` at the mutated nodes."""
    m = 3
    frame = _frame(K, m)
    every = C.fit(frame, K, m, ("mix", "mix_noprev"), weights="w")
    shift = M.estimate(every, "mix_noprev") - M.estimate(every, "mix")
    assert shift == pytest.approx(H.truth(K, m, "mix_noprev") - H.truth(K, m, "mix"), abs=1e-8)
    assert abs(shift) > 5 * FLOOR
    for t in range(2, K + 1):
        regimens = {
            "mix": P.regimen("mix", K, m),
            "mix_noprev": P.regimen("mix_noprev", K, m, (t,)),
        }
        reached = frame.loc[frame[f"A{t}"].notna()]
        declared = np.asarray(regimens["mix"].plan[t - 1].density_fn(reached), dtype=float)
        mutated = np.asarray(regimens["mix_noprev"].plan[t - 1].density_fn(reached), dtype=float)
        assert float(np.max(np.abs(declared - mutated))) == pytest.approx(0.25)
        result = C.fit(frame, K, m, (), weights="w", regimens=regimens)
        truth = _per_node_truth("mix_noprev", K, m, t)
        assert M.estimate(result, "mix_noprev") == pytest.approx(truth, abs=1e-8)
        record = truth - H.truth(K, m, "mix")
        assert abs(M.estimate(result, "mix_noprev") - M.estimate(result, "mix")) > abs(record) / 10
        assert abs(record) > FLOOR


# ----------------------------------------------------------------------------- (f) the third arm
#: Per-node shifts of the two-arm complement with the W-blind GLM (rung O2), on L_R and L_C.
THIRD_ARM_SHIFTS = {
    (3, "LR"): (-2.70e-3, -1.50e-3, -1.28e-3),
    (3, "LC"): (-3.10e-3, -1.81e-3, -2.04e-3),
    (4, "LR"): (-2.09e-3, -2.11e-3, -1.53e-3, -1.17e-3),
    (4, "LC"): (-2.59e-3, -2.49e-3, -1.82e-3, -2.04e-3),
    (5, "LR"): (-1.82e-3, -1.66e-3, -2.13e-3, -1.46e-3, -1.12e-3),
    (5, "LC"): (-2.42e-3, -2.11e-3, -2.46e-3, -1.77e-3, -2.09e-3),
}
#: The every-node shift with the same rung.
THIRD_ARM_EVERY = {
    (3, "LR"): -5.582e-3,
    (3, "LC"): -6.978e-3,
    (4, "LR"): -7.161e-3,
    (4, "LC"): -9.065e-3,
    (5, "LR"): -8.521e-3,
    (5, "LC"): -1.101e-2,
}
LAWS = {"LR": H.L_R, "LC": H.L_C}


@pytest.mark.parametrize(("K", "law_name"), sorted(THIRD_ARM_EVERY), ids=lambda v: str(v))
def test_a_two_arm_complement_is_wrong_at_three_levels(K: int, law_name: str) -> None:
    """``P(A_t = 1)`` read as ``1 - P(A_t = 2)``: both arms on the learner route, plan ``mix``."""
    m, plan, law = 3, "mix", LAWS[law_name]
    frame = _frame(K, m, law)
    weight = _weight(frame)
    settings: dict[str, Any] = {
        "weights": "w",
        "outcome_learner": C.outcome_rung("O2", plan, K, m),
        "probabilities": "censoring",
        "g_bounds": (1e-12, 1.0),
    }

    def run(nodes: tuple[int, ...]) -> Any:
        learner = C.HorizonComplementMechanism(K, m, law, nodes)
        return C.fit(frame, K, m, (plan,), treatment_learner=learner, **settings)

    base = run(())
    assert M.estimate(base, plan) == pytest.approx(H.truth(K, m, plan), abs=EXACT)
    every = M.estimate(run(tuple(range(1, K + 1))), plan) - M.estimate(base, plan)
    _assert_moves(every, THIRD_ARM_EVERY[K, law_name])
    for t, record in enumerate(THIRD_ARM_SHIFTS[K, law_name], start=1):
        result = run((t,))
        _assert_moves(M.estimate(result, plan) - M.estimate(base, plan), record)
        assert M.curve_departure(result, plan, K, m, law, weight) > 1e-3


def test_the_known_mechanism_learner_is_the_law() -> None:
    """``KnownHorizonMechanism`` returns the law's arm probabilities at every node."""
    for K, m in ((5, 2), (5, 3)):
        frame = _frame(K, m)
        result = C.fit(
            frame,
            K,
            m,
            (_top(m),),
            weights="w",
            probabilities="censoring",
            treatment_learner=C.KnownHorizonMechanism(K, m, H.L_R),
        )
        assert M.estimate(result, _top(m)) == pytest.approx(H.truth(K, m, _top(m)), abs=EXACT)


# ----------------------------------------------------------------------------- (h) M7
def _coefficients(result: Any, frame: pd.DataFrame, plan: str, K: int, m: int) -> float:
    """The largest coefficient of ``OracleOffsetGLM`` refit on each node's own rows."""
    weight = _weight(frame)
    policy = H.policy_nodes(plan, K)
    largest = 0.0
    for step in result.fits[plan].steps:
        rows = step.trained_on
        design = C._design(frame, step.time, m, policy)
        model = C.OracleOffsetGLM(plan, K, m, H.L_R).fit(
            design[rows], step.pseudo_outcome[rows], weight[rows]
        )
        largest = max(largest, float(np.max(np.abs(model.coef_))))
    return largest


@pytest.mark.parametrize(("K", "m"), [(3, 2), (3, 3), (5, 3)], ids=["K3m2", "K3m3", "K5m3"])
@pytest.mark.parametrize("mechanism", ["known", "constant"])
def test_a_correct_outcome_regression_fits_to_zero(K: int, m: int, mechanism: str) -> None:
    """``OracleOffsetGLM``: every coefficient and every ``epsilon_t`` is 0, and the truth is exact."""
    frame = _frame(K, m)
    if mechanism == "constant":
        frame = C.wrong_mechanism(frame, K, m, "constant")
    for plan in PLANS[m]:
        learner = C.OracleOffsetGLM(plan, K, m, H.L_R)
        result = C.fit(frame, K, m, (plan,), weights="w", outcome_learner=learner)
        assert M.estimate(result, plan) == pytest.approx(H.truth(K, m, plan), abs=EXACT)
        assert max(abs(M.epsilon(result, plan, t)) for t in range(1, K + 1)) < 1e-8
        assert _coefficients(result, frame, plan, K, m) < 1e-8


#: K = 5, m = 3, the constant mechanism: the shift of each mutation at each node.
AT_RISK_SHIFTS: dict[str, tuple[float, ...]] = {"respond": (1.4e-2, 1.5e-2, 2.4e-2, 3.6e-2, 3.6e-2)}
NO_RULE_CHECK_SHIFTS: dict[str, tuple[float, ...]] = {
    "low": (1.4e-2, 1.8e-2, 2.7e-2, 3.1e-2, 1.4e-2),
    "respond": (-9.2e-3, 2.0e-3, -8.9e-4, 2.7e-3, 1.2e-2),
}
ROW_MUTATIONS: dict[str, tuple[Callable[[int], Any], dict[str, tuple[float, ...]]]] = {
    "at_risk_rows": (M.fit_on_at_risk, AT_RISK_SHIFTS),
    "no_rule_check": (M.rule_check_dropped, NO_RULE_CHECK_SHIFTS),
}


@pytest.mark.parametrize("mutation", sorted(ROW_MUTATIONS))
def test_a_correct_outcome_regression_on_the_wrong_rows_is_biased(mutation: str) -> None:
    """M7 under the constant mechanism, where the estimate rests on the outcome regression.

    With the known mechanism the targeted estimate is exact whatever the regression, so the row
    defects show only here.
    """
    K, m = 5, 3
    frame = C.wrong_mechanism(_frame(K, m), K, m, "constant")
    patch, records = ROW_MUTATIONS[mutation]
    for plan, shifts in records.items():
        learner = C.OracleOffsetGLM(plan, K, m, H.L_R)
        base = M.estimate(C.fit(frame, K, m, (plan,), weights="w", outcome_learner=learner), plan)
        for t, record in enumerate(shifts, start=1):
            with patch(t):
                result = C.fit(frame, K, m, (plan,), weights="w", outcome_learner=learner)
            shift = M.estimate(result, plan) - base
            floor = FLOOR if abs(record) >= 5 * FLOOR else abs(record) / 10
            assert abs(shift) > floor, (plan, t, shift)
            assert shift == pytest.approx(record, rel=0.1), (plan, t)


# ----------------------------------------------------------------------------- (i) pseudo-outcome
@pytest.mark.parametrize(("K", "m"), SETTINGS, ids=SETTING_IDS)
def test_every_node_regresses_the_targeted_value(K: int, m: int) -> None:
    """Every node regressing the untargeted next value moves the estimate (intercept learner)."""
    frame = _frame(K, m)
    for plan in PLANS[m]:
        with M.untargeted_pseudo_outcome(range(2, K + 1)):
            result = C.fit(frame, K, m, (plan,), weights="w", outcome_learner=C.InterceptOnly())
        assert abs(M.estimate(result, plan) - H.truth(K, m, plan)) > 5 * FLOOR, plan


#: Node ``t`` alone regresses node ``t + 1``'s untargeted value, D1 at ``t + 1``: shift per ``t``.
PSEUDO_SHIFTS: dict[tuple[int, int], dict[str, tuple[float, ...]]] = {
    (3, 2): {"never": (-1.42e-3, -9.72e-3), "respond": (-6.10e-3, -1.47e-3)},
    (3, 3): {"respond": (-2.35e-4, 5.34e-3), "mix": (3.82e-3, 3.93e-4)},
    (4, 3): {"respond": (3.46e-4, 5.51e-3, 8.50e-3), "mix": (4.06e-3, 3.68e-3, 6.66e-5)},
    (5, 2): {"respond": (-8.05e-3, -4.88e-3, -2.30e-3, 1.92e-3)},
    (5, 3): {
        "low": (-1.54e-3, -2.43e-3, -2.27e-3, -1.54e-2),
        "mix": (4.03e-3, 3.89e-3, 3.69e-3, -1.54e-4),
    },
}


@pytest.mark.parametrize(("K", "m"), sorted(PSEUDO_SHIFTS), ids=lambda v: str(v))
def test_each_node_regresses_the_targeted_value(K: int, m: int) -> None:
    """Per node: the untargeted next value at node ``t`` alone, the W-blind GLM at ``t + 1``."""
    frame = _frame(K, m)
    for plan, shifts in PSEUDO_SHIFTS[K, m].items():
        for t, record in enumerate(shifts, start=1):
            learner = _pattern(plan, K, m, {t + 1: _rung("D1", plan, K, m)})
            base = C.fit(frame, K, m, (plan,), weights="w", outcome_learner=learner)
            with M.untargeted_pseudo_outcome({t + 1}):
                result = C.fit(frame, K, m, (plan,), weights="w", outcome_learner=learner)
            _assert_moves(M.estimate(result, plan) - M.estimate(base, plan), record)


# ----------------------------------------------------------------------------- (j) paired contrast
@pytest.mark.parametrize(("K", "m"), SETTINGS, ids=SETTING_IDS)
def test_a_policy_and_its_last_node_stop_share_their_followers(K: int, m: int) -> None:
    """Inflation at least 1.3, and the reported contrast curve is the paired difference."""
    first, second = ("mix", "mixlast") if m == 3 else ("coin", "coinlast")
    reading = H.contrast(K, m, first, second)
    assert reading["inflation"] >= 1.3
    frame = _frame(K, m)
    weight = _weight(frame)
    result = C.fit(
        frame,
        K,
        m,
        (second, first),
        weights="w",
        outcome_learner=C.CellMeans(),
        pseudo_learner=C.CellMeans(),
    )
    curve = C.per_unit_curve(result, f"ate_regimen[{first} vs {second}]", weight)
    paired = H.eif(K, m, first) - H.eif(K, m, second)
    scale = max(1.0, float(np.max(np.abs(paired))))
    np.testing.assert_allclose(curve, paired, rtol=0, atol=EXACT * scale)
    p = weight / weight.sum()
    variance = float(np.sum(p * curve**2))
    assert variance == pytest.approx(reading["sigma"] ** 2, rel=1e-8)
    sum_of_levels = sum(float(np.sum(p * H.eif(K, m, name) ** 2)) for name in (first, second))
    assert np.sqrt(sum_of_levels / variance) == pytest.approx(reading["inflation"], rel=1e-8)


# ----------------------------------------------------------------------------- (k) pooled update
def _relative_scores(result: Any) -> list[float]:
    scores = []
    for fit in result.fits.values():
        weights = np.asarray(fit.obs_weights, dtype=float)
        for step in fit.steps:
            multiplier = weights * step.clever
            score = float(np.mean(multiplier * (step.pseudo_outcome - step.targeted)))
            scores.append(abs(score) / float(np.mean(np.abs(multiplier))))
    return scores


POOLED_PLANS = ("low", "high", "respond")


@pytest.fixture(scope="module")
def draw() -> pd.DataFrame:
    frame = H.draw(5, 3, 1500, 11, H.L_R)
    frame["v"] = np.random.default_rng(12).uniform(0.5, 2.0, len(frame))
    return frame


def _pooled(frame: pd.DataFrame, **settings: Any) -> Any:
    return C.fit(
        frame,
        5,
        3,
        POOLED_PLANS,
        probabilities=False,
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        outcome_learner=C.InterceptOnly(),
        **{"n_folds": 3, **settings},
    )


def _moved(result: Any, base: Any) -> float:
    return max(abs(M.estimate(result, plan) - M.estimate(base, plan)) for plan in POOLED_PLANS)


def test_the_pooled_update_solves_every_node_at_five_nodes(draw: pd.DataFrame) -> None:
    """Three outer folds, five nodes, three levels: one solved epsilon per node, each far from 0."""
    base = _pooled(draw)
    epsilons = [
        abs(float(step.fluctuation.epsilon[0])) for f in base.fits.values() for step in f.steps
    ]
    assert min(epsilons) > 1e-2
    assert max(_relative_scores(base)) < 1e-9


def test_the_four_pooled_mutations_move_the_estimate_at_five_nodes(
    draw: pd.DataFrame,
) -> None:
    """M1, M1b (loss and observation weights) and M2 of the two-node module, at K = 5."""
    base = _pooled(draw)
    weighted = _pooled(draw, weights="v")
    real = sequential.solve_fluctuation
    training = np.asarray(base.folds.assignment != 0)
    observation = np.asarray(weighted.data.weights, dtype=float)

    def narrowed(
        outcome: Any, initial: Any, submodel: Any, weights: Any, rows: Any, **kw: Any
    ) -> Any:
        return real(outcome, initial, submodel, weights, rows & training, **kw)

    def unweighted(
        outcome: Any, initial: Any, submodel: Any, weights: Any, rows: Any, **kw: Any
    ) -> Any:
        return real(outcome, initial, submodel, np.where(weights > 0, 1.0, 0.0), rows, **kw)

    def untilted(
        outcome: Any, initial: Any, submodel: Any, weights: Any, rows: Any, **kw: Any
    ) -> Any:
        return real(outcome, initial, submodel, weights / observation, rows, **kw)

    for patched, reference, settings in (
        (narrowed, base, {}),
        (unweighted, base, {}),
        (untilted, weighted, {"weights": "v"}),
    ):
        with mock.patch.object(sequential, "solve_fluctuation", patched):
            mutated = _pooled(draw, **settings)
        assert max(_relative_scores(mutated)) > 1e-4
        assert _moved(mutated, reference) > 1e-4
    # M2: the in-sample mechanism read in place of the out-of-fold one.
    in_sample = _pooled(draw, n_folds=1).mechanism
    cumulative = Mechanism.cumulative_with_unbounded

    def in_sample_cumulative(self: Mechanism, data: Any, plan: Any, bounds: Any) -> Any:
        return cumulative(in_sample, data, plan, bounds)

    with mock.patch.object(Mechanism, "cumulative_with_unbounded", in_sample_cumulative):
        mutated = _pooled(draw)
    assert _moved(mutated, base) > 1e-4


# ----------------------------------------------------------------------------- M3, leakage
#: The estimate floor of a construction mutation, as in the two-node modules.
CAUGHT = 1e-4


def _linear_pooled(draw: pd.DataFrame, plans: tuple[str, ...], **settings: Any) -> Any:
    return C.fit(
        draw,
        5,
        3,
        plans,
        probabilities=False,
        treatment_learner=LogisticRegression(max_iter=1000),
        censoring_learner=LogisticRegression(max_iter=1000),
        **{"n_folds": 3, "learner_folds": 2, **settings},
    )


@pytest.fixture(scope="module")
def stitched(draw: pd.DataFrame) -> Any:
    """Linear regressions, so the longhand is a least-squares solve; the draw is misspecified."""
    return _linear_pooled(
        draw,
        ("low", "respond"),
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
        tol=1e-10,
    )


def test_the_stitched_initial_is_the_untargeted_fold_recursion_at_five_nodes(
    stitched: Any,
) -> None:
    """T3 at K = 5 and three levels: every node's stitch equals the K-node longhand to 1e-12."""
    T._assert_epsilon_witness(stitched)
    T._assert_stitched_longhand(stitched)


def test_m3_the_fold_targeted_carry_is_a_different_initial_fit_at_five_nodes(
    stitched: Any,
) -> None:
    """M3: fold-targeted predictions carried backward move node 1's stitched fit.

    Recorded: 8.4e-2 (``low``) and 1.4e-1 (``respond``) against the 1e-6 floor of the two-node
    module.  The unmutated stitch is the longhand above, so M3 fails it and the shipped fit passes.
    """
    for fit in stitched.fits.values():
        carried, _ = T._longhand_recursion(stitched, fit, fold_targeted=True)
        assert float(np.max(np.abs(fit.steps[0].initial - carried[1]))) > 1e-3


def _nearest_neighbour_recursion(
    result: Any, fit: Any, assignment: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Stitched held-out predictions of a 1-NN recursion, with node ``t`` split by ``assignment[t]``.

    Fold ``k`` of node ``t`` trains on the followers outside ``assignment[t] == k`` and regresses
    fold ``k``'s own prediction from node ``t + 1``, whose training rows were outside
    ``assignment[t + 1] == k``.  One assignment for every node is the shipped split.
    """
    data = result.data
    initial = {t: np.full(data.n, np.nan) for t in range(1, fit.horizon + 1)}
    folds = range(result.folds.n_folds)
    carried = {k: T._longhand_seed(result) for k in folds}
    for t in range(fit.horizon, 0, -1):
        at_risk, following = T._longhand_masks(result, fit, t)
        design = data.covariate_history(t)
        for k in folds:
            held = assignment[t] == k
            target = T._longhand_target(result, fit, carried[k], t)
            rows = following & ~held
            model = KNeighborsRegressor(n_neighbors=1).fit(design[rows], target[rows])
            prediction = np.where(at_risk, model.predict(design), 0.5)
            initial[t][held] = prediction[held]
            carried[k] = prediction
    return initial


def test_trajectory_level_folds_do_not_leak_a_later_pseudo_outcome_at_five_nodes(
    draw: pd.DataFrame,
) -> None:
    """The rigged learner of ``test_crossfit_leakage`` at K = 5, in two arrays.

    A 1-NN regression reproduces a training row's target exactly, so an in-sample value is
    visible as an equality and an out-of-fold one as an inequality, with no tolerance and no
    seed sensitivity.  Under the shipped split every node's stitched fit is the longhand with one
    assignment for all nodes, array for array.  Under per-node folds (the assignment shifted by
    the node) a unit held out at node ``t`` was in training at node ``t + 1``, and nodes 1 to
    K - 1 differ (recorded: 1.0 at every one of them, 0.0 at node K, whose split is shared).
    """
    result = _linear_pooled(
        draw,
        ("low", "respond"),
        outcome_learner=KNeighborsRegressor(n_neighbors=1),
        pseudo_learner=KNeighborsRegressor(n_neighbors=1),
    )
    assignment = np.asarray(result.folds.assignment)
    shipped = dict.fromkeys(range(1, 6), assignment)
    per_node = {t: (assignment + (5 - t)) % 3 for t in range(1, 6)}
    for fit in result.fits.values():
        honest = _nearest_neighbour_recursion(result, fit, shipped)
        leaky = _nearest_neighbour_recursion(result, fit, per_node)
        for step in fit.steps:
            np.testing.assert_array_equal(step.initial, honest[step.time])
            moved = float(np.max(np.abs(leaky[step.time] - honest[step.time])))
            assert (moved > 0.0) == (step.time < 5), (fit.regimen.label, step.time, moved)


# ----------------------------------------------------------------------------- (m), (n) SDR
def _pattern_frame(wrong_nodes: tuple[int, ...]) -> pd.DataFrame:
    """The K = 5, m = 3 frame on L_R with the constant mechanism (M1) at ``wrong_nodes``."""
    K, m = 5, 3
    frame = _frame(K, m)
    wrong = C.wrong_mechanism(frame, K, m, "constant")
    for t in wrong_nodes:
        columns = [f"g{t}_{k}" for k in range(m)] + [f"pc{t}"]
        frame[columns] = wrong[columns]
    return frame


def _pattern_learners(
    plan: str, oracle_nodes: tuple[int, ...], link: str
) -> tuple[C.NodePattern, C.NodePattern]:
    """``OracleOffsetGLM`` at ``oracle_nodes`` and the intercept (O3) elsewhere."""
    K, m = 5, 3
    outcome = {t: C.OracleOffsetGLM(plan, K, m, H.L_R) for t in oracle_nodes}
    pseudo = {t: C.OracleOffsetGLM(plan, K, m, H.L_R, link=link) for t in oracle_nodes}
    default = C.InterceptOnly()
    return C.NodePattern(outcome, default, m, K), C.NodePattern(pseudo, default, m, K)


def _longhand_sdr(
    frame: pd.DataFrame, plan: str, outcome: Any, pseudo: Any, *, start: str = "node"
) -> float:
    """The SDR estimate of a label or rule plan at K = 5, m = 3, from the frame's columns.

    Node 5 fits ``outcome`` to ``Y`` and node ``t < 5`` fits ``pseudo`` to the transform
    ``phi_{t+1}``, on the followers, with the frame's weights.  ``phi_t`` is
    ``m_t + sum_{s >= t} W_{t,s} (m_{s+1} - m_s)`` with ``W_{t,s} = 1 / prod_{k=t}^{s} p_k`` on
    the followers through ``s`` and ``p_k`` the frame's treatment and retention columns at the
    plan's arm.  ``start="first"`` is the mutation: ``prod_{k=1}^{s}``.
    """
    K, m = 5, 3
    weight = _weight(frame)
    n = len(frame)
    factors, at_risk, following = [], [], []
    running = np.ones(n, dtype=bool)
    for t in range(1, K + 1):
        assigned = C.plan_density(frame, plan, K, m, t).argmax(axis=1)
        arm = np.nan_to_num(frame[f"A{t}"].to_numpy(dtype=float), nan=-1.0).astype(int)
        g = np.nan_to_num(frame[[f"g{t}_{k}" for k in range(m)]].to_numpy(dtype=float))
        retained = np.nan_to_num(frame[f"pc{t}"].to_numpy(dtype=float))
        factors.append(g[np.arange(n), assigned] * retained)
        at_risk.append(running.copy())
        stays = np.nan_to_num(frame[f"C{t}"].to_numpy(dtype=float)) == 1.0
        running = running & (arm == assigned) & stays
        following.append(running.copy())
    y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
    fitted: dict[int, np.ndarray] = {}
    target = y
    for t in range(K, 0, -1):
        rows = following[t - 1]
        design = C._design(frame, t, m, ())
        model = clone(outcome if t == K else pseudo).fit(
            design[rows], target[rows], sample_weight=weight[rows]
        )
        fitted[t] = np.asarray(model.predict(design), dtype=float)
        phi = fitted[t].copy()
        first = 1 if start == "first" else t
        for s in range(t, K + 1):
            path = np.prod(np.column_stack(factors[first - 1 : s]), axis=1)
            later = fitted[s + 1] if s < K else y
            ratio = np.where(following[s - 1], 1.0 / np.where(path > 0, path, 1.0), 0.0)
            phi = phi + ratio * (later - fitted[s])
        target = np.where(at_risk[t - 1], phi, 0.0)
    return float(np.sum(weight * target) / np.sum(weight))


@pytest.mark.parametrize("plan", ["low", "respond"])
def test_the_sdr_partial_weights_start_at_their_own_node(plan: str) -> None:
    """``P_odd`` at K = 5, m = 3: the engine is the longhand; weights from node 1 are not.

    With the mechanism right at every node, or the regression right at every node, the weights
    from node 1 return the same estimate.  ``P_odd`` relies on each node's own weight.
    """
    frame = _pattern_frame((2, 4))
    outcome, pseudo = _pattern_learners(plan, (2, 4), "identity")
    result = C.fit(
        frame,
        5,
        3,
        (plan,),
        estimator="SDR",
        weights="w",
        outcome_learner=outcome,
        pseudo_learner=pseudo,
    )
    engine = M.estimate(result, plan)
    assert engine == pytest.approx(H.truth(5, 3, plan), abs=EXACT)
    assert engine == pytest.approx(_longhand_sdr(frame, plan, outcome, pseudo), abs=EXACT)
    assert abs(_longhand_sdr(frame, plan, outcome, pseudo, start="first") - engine) > 1e-4


def _pattern_contrast(
    oracle_nodes: tuple[int, ...], wrong_nodes: tuple[int, ...], estimator: str = "SDR"
) -> float:
    """``respond`` minus ``low`` with per-node nuisances, minus the truth difference.

    ``oracle_nodes`` fit ``OracleOffsetGLM`` and the rest the intercept (O3); ``wrong_nodes``
    carry the constant mechanism (M1) and the rest the law's.
    """
    frame = _pattern_frame(wrong_nodes)
    link = "identity" if estimator == "SDR" else "logit"
    estimates = {}
    for plan in ("respond", "low"):
        outcome, pseudo = _pattern_learners(plan, oracle_nodes, link)
        result = C.fit(
            frame,
            5,
            3,
            (plan,),
            estimator=estimator,
            weights="w",
            outcome_learner=outcome,
            pseudo_learner=pseudo,
        )
        estimates[plan] = M.estimate(result, plan)
    truth = H.truth(5, 3, "respond") - H.truth(5, 3, "low")
    return estimates["respond"] - estimates["low"] - truth


def test_sdr_is_exact_with_one_right_nuisance_at_each_node() -> None:
    """``P_odd`` and ``P_even`` for the pair (O3, M1); three breaks move the estimate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert abs(_pattern_contrast((2, 4), (2, 4))) < EXACT
        assert abs(_pattern_contrast((1, 3, 5), (1, 3, 5))) < EXACT
        assert _pattern_contrast((2, 4), (2, 3, 4)) == pytest.approx(0.0339, abs=1e-4)
        assert _pattern_contrast((2, 4), (1, 2, 3, 4, 5)) == pytest.approx(0.2203, abs=1e-4)
        assert _pattern_contrast((2, 4), (2, 4), "LTMLE") == pytest.approx(0.0616, abs=1e-4)


# ----------------------------------------------------------------------------- M7a, M7b
def _sdr_longhand(result: Any, fit: Any, construction: str = "training") -> tuple[float, Any]:
    """The cross-fitted SDR estimate of a label or rule plan at K = 5, m = 3, by hand.

    Linear regressions on the history design, the plan's follower masks from the raw arrays and
    each fold's mechanism refit on its training rows.  Partial weight ``W_{t,s}`` divides the
    followers through ``s`` by ``prod_{k=t}^{s} p_k``, bounded as a product.  ``"training"`` is
    the shipped recursion: fold ``k`` composes every transform from its own fits.
    ``"out_of_fold"`` is M7b: the same, with the out-of-fold mechanism at the training rows.
    ``"stitched"`` is M7a, Step 2 as printed: each row reads the fits of the fold holding it out.
    """
    K = 5
    data = result.data
    n = data.n
    assignment = np.asarray(result.folds.assignment)
    n_folds = result.folds.n_folds
    label = fit.regimen.label
    lower, upper = result.config.g_bounds
    masks = [T._longhand_masks(result, fit, t) for t in range(1, K + 1)]
    y = np.nan_to_num(np.asarray(data.outcome, dtype=float))

    def factors(mechanism: Mechanism) -> list[np.ndarray]:
        return [
            np.asarray(mechanism.treatment[t][label]) * np.asarray(mechanism.censoring[t][label])
            for t in range(K)
        ]

    out_of_fold = factors(result.mechanism)
    per_fold = [factors(T.refit_fold_mechanism(result, k)) for k in range(n_folds)]

    def weight(p: list[np.ndarray], t: int, s: int) -> np.ndarray:
        path = np.prod(np.column_stack(p[t - 1 : s]), axis=1)
        return np.where(masks[s - 1][1], 1.0 / np.clip(path, lower, upper), 0.0)

    def transform(fitted: dict[int, np.ndarray], p: list[np.ndarray], t: int) -> np.ndarray:
        phi = fitted[t].copy()
        for s in range(t, K + 1):
            later = fitted[s + 1] if s < K else y
            phi = phi + weight(p, t, s) * (later - fitted[s])
        return phi

    def regression(target: np.ndarray, t: int, training: np.ndarray) -> np.ndarray:
        at_risk, following = masks[t - 1]
        design = data.covariate_history(t)
        model = LinearRegression().fit(design[following & training], target[following & training])
        return np.where(at_risk, model.predict(design), 0.0)

    first = np.zeros(n)
    if construction == "stitched":
        mixed = [np.zeros(n) for _ in range(K)]
        for k in range(n_folds):
            for t in range(K):
                mixed[t][assignment == k] = per_fold[k][t][assignment == k]
        fitted: dict[int, np.ndarray] = {}
        target = y
        for t in range(K, 0, -1):
            fitted[t] = np.zeros(n)
            for k in range(n_folds):
                held = assignment == k
                fitted[t][held] = regression(target, t, ~held)[held]
            phi = transform(fitted, mixed, t)
            target = np.where(masks[t - 1][0], phi, 0.0)
        first = phi
    else:
        for k in range(n_folds):
            held = assignment == k
            p = per_fold[k] if construction == "training" else out_of_fold
            fitted = {}
            target = y
            for t in range(K, 0, -1):
                fitted[t] = regression(target, t, ~held)
                phi = transform(fitted, p, t)
                target = np.where(masks[t - 1][0], phi, 0.0)
            first[held] = phi[held]
    return float(first.mean()), first


@pytest.fixture(scope="module")
def sdr_fit(draw: pd.DataFrame) -> Any:
    return _linear_pooled(
        draw,
        ("low", "respond"),
        estimator="SDR",
        outcome_learner=LinearRegression(),
        pseudo_learner=LinearRegression(),
    )


def test_the_sdr_engine_is_the_k_node_longhand_with_censoring(sdr_fit: Any) -> None:
    """T5 at K = 5, three levels and censoring: estimate and per-unit transform to 1e-12."""
    for label, fit in sdr_fit.fits.items():
        psi, first = _sdr_longhand(sdr_fit, fit)
        assert M.estimate(sdr_fit, label) == pytest.approx(psi, abs=1e-12)
        curve = np.asarray(sdr_fit[f"ey_regimen[{label}]"].influence_curve, dtype=float)
        np.testing.assert_allclose(curve, first - psi, rtol=0, atol=1e-12)


def test_m7a_and_m7b_move_the_sdr_estimate_at_five_nodes(sdr_fit: Any) -> None:
    """M7a (stitched Step 2) and M7b (out-of-fold mechanism at training rows) each exceed 1e-4.

    Recorded on this draw: M7a -3.4e-2 (``low``) and -4.3e-2 (``respond``); M7b -5.6e-4 and
    +2.5e-3.
    """
    for label, fit in sdr_fit.fits.items():
        engine = M.estimate(sdr_fit, label)
        for construction in ("stitched", "out_of_fold"):
            psi, _ = _sdr_longhand(sdr_fit, fit, construction)
            assert abs(psi - engine) > CAUGHT, (label, construction)
