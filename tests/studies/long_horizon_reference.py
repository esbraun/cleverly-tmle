"""The comparator inputs of the three long-horizon studies, built from a seed alone (X59).

A reference-only run (``tests/canonical/regenerate.py --reference-only``) re-draws each paired
replicate from its seed and hands R everything the comparator needs, so the R side re-implements
no rule of the law or the plans.  Nothing here fits ``cleverly``.

Every column is computed from :mod:`tests.discrete_law_longitudinal_horizon`, the oracle the
committed truths come from, on the history each unit observed:

``abar__<plan>__<t>`` (S1)
    The arm a deterministic plan assigns at node ``t``.  A unit that never reached ``t`` takes
    arm 0, which no estimate reads: ``ltmle`` follows a unit only while it is uncensored.
``gA<t>``, ``gC<t>`` (S1)
    ``ltmle``'s numeric ``gform``: ``P(A_t = 1 | H_t)`` and ``P(C_t = 1 | H_t, A_t)``, filled
    with :data:`FILLER` after censoring, where nothing downstream reads them.
``ratio__<plan>__<t>`` (S2, S3)
    The per-node ratio ``q_t(A_t | H_t) 1{C_t = 1} / (g_t(A_t | H_t) c_t)``, zero where the unit
    leaves the plan or the observation at ``t``; ``cleverly`` divides by its running product.
``shift__<plan>__<t>__<copy>`` (S2, S3)
    The arm each copy of a unit takes at ``t``: one copy for a deterministic plan, and
    :data:`COPIES` for a plan with a known stochastic node, whose probabilities are all
    multiples of one quarter (``tests/canonical/lmtp_policy_adapter.R``).
``policy__<plan>`` (S2, S3)
    The plan's stochastic nodes as a bitmask (bit ``t - 1`` for node ``t``): the plan-stratified
    learner keeps those arms in its design and stratifies the rest.
``fold``
    The realized outer fold: 0 at one fold, else :func:`cleverly.learners.crossfit.random_partition`
    at ``random_state=0``, as every committed fit drew it (``long_horizon_common.fit``).
``design__*``
    The scenario's ``K``, ``m``, estimator and declared mutation, as constants.

The identity gate of the smoke (``tests/canonical/x59_smoke.py``) refits the first replicates and
checks that the written folds and ratios are the ones the committed fits read.

A declared mutation (``X59-plan.md`` section 4.1, M2) is an input: ``CLEVERLY_X59_MUTATION`` set
while the inputs are written changes the archive, so its resume key, and never a checkpoint of
the unmutated run.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from cleverly.learners.crossfit import random_partition
from tests import discrete_law_longitudinal_horizon as law_
from tests.studies import long_horizon_common as common
from tests.studies import long_horizon_primary as primary
from tests.studies.evidence.parity import ReferenceSample
from tests.studies.evidence.registry import StudyRecord
from tests.studies.evidence.seeds import replicate_seed
from tests.studies.fractional_glm import QuasiBinomialGLM

#: Copies per unit for a plan with a known stochastic node.
COPIES = 4
#: The value of a probability column a unit never reaches.
FILLER = 0.5
#: The outer fold seed of every committed fit (``long_horizon_common.fit``).
FOLD_SEED = 0
#: The range every outcome regression must stay inside on both sides (M3): ``cleverly``'s
#: ``[1 - 0.9995, 0.9995]`` clip.  A prediction outside it is a design defect, not a bound.
Q_BOUNDS = (5e-4, 0.9995)

#: The mutation environment variable, and the declared mutations (M2).
MUTATION_ENV = "CLEVERLY_X59_MUTATION"
MUTATIONS = {
    "none": 0,
    # The node-2 mechanism scaled by 1.05: S1's gform column gA2, S2 and S3's node-2 ratio.
    "gform_node2": 1,
    # The node-K outcome regression without W: S1's Qform at Y, the lmtp learner at node K.
    "qform_node_k": 2,
}
#: The comparator families.
COMPARATORS = ("ltmle", "lmtp_tmle", "lmtp_sdr")
#: The R ``lmtp`` outcome learner (``long_horizon_lmtp_adapter.R``): the declared plan-stratified
#: regression, or the pooled diagnostic of the follower-mapping check.
LEARNER_ENV = "CLEVERLY_X59_LEARNER"
LEARNERS = ("stratified", "pooled")
#: A study module's ``REFERENCE_ENVIRONMENT``: what the driver keys, forwards and records
#: (``tests/canonical/regenerate.py``, ``reference_environment``).  ``ltmle`` reads no learner.
LTMLE_ENVIRONMENT = {MUTATION_ENV: "none"}
LMTP_ENVIRONMENT = {MUTATION_ENV: "none", LEARNER_ENV: "stratified"}


def mutation() -> str:
    """The declared mutation this process writes, from :data:`MUTATION_ENV`."""
    name = os.environ.get(MUTATION_ENV, "none") or "none"
    if name not in MUTATIONS:
        raise ValueError(f"{MUTATION_ENV}={name!r} is not a declared mutation: {sorted(MUTATIONS)}")
    return name


# ----------------------------------------------------------------------------- observed histories
def _codes(frame: pd.DataFrame, name: str) -> np.ndarray:
    return np.nan_to_num(frame[name].to_numpy(float)).astype(int)


def reached(frame: pd.DataFrame, t: int) -> np.ndarray:
    """Whether each unit reached node ``t`` (its ``A_t`` is observed)."""
    return frame[f"A{t}"].notna().to_numpy()


def density(frame: pd.DataFrame, plan: str, K: int, m: int, t: int) -> np.ndarray:
    """``(n, m)`` intervention density of ``plan`` at node ``t`` on each unit's observed history.

    The oracle states a plan on its history tree; each unit is mapped to its node-``t`` history
    by :func:`tests.discrete_law_longitudinal_horizon.history_index`.  A unit that never
    reached ``t`` reads the history with its missing entries at zero.
    """
    tree = law_.histories(K, m)
    covariates = {s: _codes(frame, f"L{s}") for s in range(2, t + 1)}
    arms = {s: _codes(frame, f"A{s}") for s in range(1, t)}
    index = law_.history_index(t, m, _codes(frame, "W"), covariates, arms)
    return np.asarray(law_.plan(plan, K, m)(t, tree)[index], dtype=float)


def assigned(frame: pd.DataFrame, plan: str, K: int, m: int) -> np.ndarray:
    """``(n, K)`` arm of a deterministic plan at each node; refuses a stochastic node."""
    out = np.zeros((len(frame), K), dtype=int)
    for t in range(1, K + 1):
        q = density(frame, plan, K, m, t)
        if not np.all((q == 0.0) | (q == 1.0)):
            raise ValueError(f"plan {plan!r} is stochastic at node {t}")
        out[:, t - 1] = np.argmax(q, axis=1)
    return out


def ratios(frame: pd.DataFrame, plan: str, K: int, m: int) -> np.ndarray:
    """``(n, K)`` per-node ratio at the observed arm; zero off the plan or the observation."""
    out = np.zeros((len(frame), K))
    for t in range(1, K + 1):
        here = reached(frame, t)
        arm = _codes(frame, f"A{t}")
        q = density(frame, plan, K, m, t)[np.arange(len(frame)), arm]
        g = np.column_stack([frame[f"g{t}_{k}"].to_numpy(float) for k in range(m)])
        g_observed = g[np.arange(len(frame)), arm]
        stays = np.nan_to_num(frame[f"C{t}"].to_numpy(float)) == 1.0
        keep = here & stays & (q > 0.0)
        safe = np.where(keep, g_observed * frame[f"pc{t}"].to_numpy(float), 1.0)
        out[:, t - 1] = np.where(keep, q / safe, 0.0)
    return out


def allocate(probabilities: np.ndarray, copies: int) -> np.ndarray:
    """``(n, copies)`` arm of each copy: ``p * copies`` copies per arm, lowest arm first."""
    counts = probabilities * copies
    if np.max(np.abs(counts - np.round(counts))) > 1e-12:
        raise ValueError(f"a plan probability is not a multiple of 1/{copies}")
    cumulative = np.cumsum(np.round(counts), axis=1)
    if np.any(cumulative[:, -1] != copies):
        raise ValueError(f"a plan row allocates other than {copies} copies")
    slots = np.arange(copies)
    return np.argmax(cumulative[:, None, :] > slots[None, :, None], axis=2)


def policy_mask(plan: str, K: int) -> int:
    """The plan's known stochastic nodes as a bitmask, bit ``t - 1`` for node ``t``."""
    return sum(1 << (t - 1) for t in law_.policy_nodes(plan, K))


def folds(n: int, n_folds: int) -> np.ndarray:
    """The realized outer fold of each unit, as the committed fits drew it."""
    if n_folds <= 1:
        return np.zeros(n, dtype=int)
    return np.asarray(random_partition(n, n_folds, seed=FOLD_SEED).assignment, dtype=int)


# ----------------------------------------------------------------------------- the plug-in
def _reached_density(frame: pd.DataFrame, plan: str, K: int, m: int, t: int) -> np.ndarray:
    """:func:`tests.studies.long_horizon_common.plan_density` on the units that reached ``t``.

    A rule reads ``L_t``, which is missing after censoring, so the density is stated on the
    reached rows only; the others are zero, and the regression masks them out either way.
    """
    out = np.zeros((len(frame), m))
    here = reached(frame, t)
    if here.any():
        rows = frame.loc[here].reset_index(drop=True)
        out[here] = common.plan_density(rows, plan, K, m, t)
    return out


def plugin(frame: pd.DataFrame, plan: str, K: int, m: int) -> tuple[float, float, float]:
    """The untargeted sequential regression, and the range of the predictions it carries.

    The arithmetic of :func:`tests.studies.long_horizon_common.untargeted_plugin` with the
    declared GLM (``tests/unit/test_x59_long_horizon_reference.py`` holds the two equal), plus
    the smallest and largest carried prediction over the units at risk, for the Q-bound check.
    """
    n = len(frame)
    policy = law_.policy_nodes(plan, K)
    follow = np.ones(n, dtype=bool)
    risk: dict[int, np.ndarray] = {}
    fol: dict[int, np.ndarray] = {}
    dens: dict[int, np.ndarray] = {}
    for t in range(1, K + 1):
        observed = frame[f"A{t}"].notna().to_numpy() & follow
        risk[t] = observed
        dens[t] = _reached_density(frame, plan, K, m, t)
        arm = _codes(frame, f"A{t}")
        positive = dens[t][np.arange(n), arm] > 0
        fol[t] = observed & positive & (np.nan_to_num(frame[f"C{t}"].to_numpy(float)) == 1.0)
        follow = fol[t]
    target = np.nan_to_num(frame["Y"].to_numpy(float))
    low, high = np.inf, -np.inf
    for t in range(K, 0, -1):
        rows = fol[t]
        X = common._design(frame, t, m, policy)
        model = clone(QuasiBinomialGLM()).fit(X[rows], target[rows])
        if t in policy:
            by_arm = np.column_stack(
                [
                    common._predict(model, common._design(frame, t, m, policy, arm=a))
                    for a in range(m)
                ]
            )
            carried = np.sum(dens[t] * by_arm, axis=1)
            spread = by_arm[risk[t]]
        else:
            carried = common._predict(model, X)
            spread = carried[risk[t]]
        if spread.size:
            low, high = min(low, float(spread.min())), max(high, float(spread.max()))
        target = np.where(risk[t], carried, 0.0)
    return float(np.mean(target)), low, high


def plugins(frame: pd.DataFrame, scenario: primary.Scenario) -> dict[str, float]:
    """``cleverly``'s untargeted plug-in of every Wald estimand of a plan-mean scenario.

    Refuses a scenario whose carried predictions leave :data:`Q_BOUNDS`: the comparison would
    then read two clips, not two regressions.
    """
    values: dict[str, float] = {}
    for plan in scenario.plans:
        value, low, high = plugin(frame, plan, scenario.K, scenario.m)
        if low < Q_BOUNDS[0] or high > Q_BOUNDS[1]:
            raise ValueError(
                f"{scenario.name} plan {plan}: an outcome regression reaches [{low:.3g}, "
                f"{high:.3g}], outside {Q_BOUNDS}"
            )
        values[primary.mean_name(plan)] = value
    for plan in scenario.plans[1:]:
        values[primary.contrast_name(plan, scenario.reference)] = (
            values[primary.mean_name(plan)] - values[primary.mean_name(scenario.reference)]
        )
    for first, second in scenario.paired:
        values[primary.contrast_name(first, second)] = (
            values[primary.mean_name(first)] - values[primary.mean_name(second)]
        )
    return values


# ----------------------------------------------------------------------------- the samples
def _data_columns(frame: pd.DataFrame, K: int) -> pd.DataFrame:
    """The observed nodes as nullable integers, in time order."""
    names = ["W"]
    for t in range(1, K + 1):
        if t > 1:
            names.append(f"L{t}")
        names += [f"A{t}", f"C{t}"]
    names.append("Y")
    return pd.DataFrame({name: frame[name].astype("Int64") for name in names})


def ltmle_columns(frame: pd.DataFrame, scenario: primary.Scenario) -> dict[str, Any]:
    """S1's ``gform`` and ``abar`` columns, and the duration of each MSM plan."""
    K, m = scenario.K, scenario.m
    if m != 2:
        raise ValueError("R ltmle takes binary nodes only")
    name = mutation()
    out: dict[str, Any] = {}
    for t in range(1, K + 1):
        g = frame[f"g{t}_1"].to_numpy(float)
        if name == "gform_node2" and t == 2:
            g = np.minimum(1.05 * g, 1.0 - 1e-6)
        out[f"gA{t}"] = np.where(np.isnan(g), FILLER, g)
        c = frame[f"pc{t}"].to_numpy(float)
        out[f"gC{t}"] = np.where(np.isnan(c), FILLER, c)
    for plan in scenario.plans:
        if policy_mask(plan, K):
            # ltmle takes no stochastic plan; its cells are unpaired (reference_scenarios).
            continue
        arms = assigned(frame, plan, K, m)
        for t in range(1, K + 1):
            out[f"abar__{plan}__{t}"] = arms[:, t - 1]
        if scenario.msm:
            out[f"msmdesign__{plan}"] = np.full(len(frame), float(plan.removeprefix("dur")))
    return out


def lmtp_columns(frame: pd.DataFrame, scenario: primary.Scenario) -> dict[str, Any]:
    """S2 and S3's ratio, shift and policy columns."""
    K, m = scenario.K, scenario.m
    name = mutation()
    out: dict[str, Any] = {}
    for plan in scenario.plans:
        ratio = ratios(frame, plan, K, m)
        if name == "gform_node2":
            ratio[:, 1] = ratio[:, 1] / 1.05
        mask = policy_mask(plan, K)
        copies = COPIES if mask else 1
        for t in range(1, K + 1):
            out[f"ratio__{plan}__{t}"] = ratio[:, t - 1]
            allocation = allocate(density(frame, plan, K, m, t), copies)
            for copy in range(copies):
                out[f"shift__{plan}__{t}__{copy + 1}"] = allocation[:, copy]
        out[f"policy__{plan}"] = np.full(len(frame), mask)
    return out


def sample(
    record: StudyRecord,
    scenarios: Sequence[primary.Scenario],
    comparator: str,
    scenario_name: str,
    n: int,
    replicate: int,
    *,
    parity_scenarios: frozenset[str] = frozenset(),
) -> ReferenceSample:
    """One replicate's comparator input, from the study's own seed stream.

    ``parity_scenarios`` names the scenarios whose G-C and witness rows need the untargeted
    plug-in; it is computed only there, because it costs ``K`` GLM fits per plan.
    """
    if comparator not in COMPARATORS:
        raise ValueError(f"unknown comparator {comparator!r}")
    scenario = next(s for s in scenarios if s.name == scenario_name)
    seed = replicate_seed(record, scenario_name, replicate)
    frame = law_.draw(scenario.K, scenario.m, n, seed, scenario.law)
    truths = primary.truths(scenario)
    columns: dict[str, Any] = dict(_data_columns(frame, scenario.K))
    columns["fold"] = folds(n, scenario.n_folds)
    if comparator == "ltmle":
        columns.update(ltmle_columns(frame, scenario))
    else:
        columns.update(lmtp_columns(frame, scenario))
    constants = {
        "design__K": scenario.K,
        "design__m": scenario.m,
        "design__sdr": int(comparator == "lmtp_sdr"),
        "design__msm": int(scenario.msm),
        "design__mutation": MUTATIONS[mutation()],
    }
    for key, value in constants.items():
        columns[key] = np.full(n, value)
    values = plugins(frame, scenario) if scenario_name in parity_scenarios else {}
    return ReferenceSample(pd.DataFrame(columns), truths, values)


# ----------------------------------------------------------------------------- declarations
def wald_cells(scenarios: Sequence[primary.Scenario]) -> dict[str, tuple[str, ...]]:
    """Scenario -> its Wald estimands, the cells a comparator can report."""
    return {scenario.name: primary.wald_estimands(scenario) for scenario in scenarios}


def reported_cells(
    scenarios: Sequence[primary.Scenario],
    unpaired_plans: Mapping[str, Sequence[str]] | None = None,
    unpaired_scenarios: Sequence[str] = (),
) -> dict[str, tuple[str, ...]]:
    """``reference_scenarios``: each scenario's Wald estimands minus the unpaired ones.

    A scenario in ``unpaired_scenarios`` is left out whole.  ``unpaired_plans`` maps a scenario
    to the plans the comparator cannot fit; every estimand that reads one of them is dropped.
    """
    blocked = dict(unpaired_plans or {})
    out: dict[str, tuple[str, ...]] = {}
    for scenario in scenarios:
        if scenario.name in unpaired_scenarios:
            continue
        drop = set(blocked.get(scenario.name, ()))
        kept = tuple(
            name
            for name in primary.wald_estimands(scenario)
            if not any(
                f"[{plan}]" in name or f"[{plan} " in name or f" {plan}]" in name for plan in drop
            )
        )
        out[scenario.name] = kept
    return out
