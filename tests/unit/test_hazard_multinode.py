"""Hazard targeting with a treatment decided at several nodes, checked on finite laws (X70 (a)).

Each law below is a finite support passed as weighted data: every support point once, with
its probability as its observation weight (plan section 4).  So the empirical law is the law,
and a fit whose mechanism is right and whose hazard is wrong must return the g-formula to the
tolerance of its score.  The truth is the path enumeration of the law's own tables in this
file and never calls the estimator.

The base law has a binary baseline covariate ``W``, a binary treatment ``A_k`` at each of
three nodes that depends on ``(W, A_{k-1})``, dropout on ``(W, A_k, A_{k-1})``, and an event
hazard on ``(W, A_k, A_{k-1})``.  The wrong hazard learner is saturated in ``(W, A_k)`` and
drops ``A_{k-1}``, so its plug-in misses and the walk has work to do; every witness checks
that the initial plug-in misses.  The regimens are "always", "never", a rule ``d(W)``, a
known policy whose probability of treating depends on the previous arm, and a one-hot policy
"stay on the previous arm", which the shipped plan resolution would collapse at the observed
arms.

Each mutation control breaks one component of the construction (plan 4.3) and must move the
estimate, so an exact witness cannot pass because the component vanishes at the truth.
"""

from __future__ import annotations

import itertools
import re
import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import ModifiedPolicy, Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.longitudinal import hazard as hazard_module
from tests.discrete_law_point_survival import CellMeans, CellProbabilities

#: The first one-step step and the relative-score tolerance of the exact witnesses.
STEP = 2e-2
TOL = 1e-11
#: How close an exact estimate must be to the truth.
EXACT = 1e-8
#: How far a mutation, or a wrong initial plug-in, must move the estimate.
MISSED = 1e-3


def _g1(w: int) -> float:
    """``P(A_1 = 1 | W)``."""
    return (0.3, 0.6)[w]


def _g(node: int, w: int, previous: int) -> float:
    """``P(A_k = 1 | W, A_{k-1})`` at a node after the first."""
    return ((0.25, 0.4), (0.6, 0.9))[w][previous] - 0.05 * (node - 2)


def _retained(node: int, w: int, a: int, previous: int) -> float:
    """``P(C_k = 1 | W, A_k, A_{k-1})``."""
    return 0.9 - 0.1 * ((w + a + 2 * previous + node) % 3)


def _hazard(node: int, w: int, a: int, previous: int) -> float:
    """``P(Y_k = 1 | at risk, W, A_k, A_{k-1})``."""
    return 0.1 + 0.08 * a + 0.12 * previous - 0.05 * a * previous + 0.05 * w + 0.03 * node


def _split(node: int, w: int, a: int, previous: int) -> tuple[float, float]:
    """The two cause-specific hazards at risk, ``(relapse, death)``."""
    total = _hazard(node, w, a, previous)
    share = 0.3 + 0.2 * ((w + a + previous + node) % 3)
    return (total * share, total * (1.0 - share))


@dataclass(frozen=True)
class PathLaw:
    """A binary treatment at each node, censoring and a survival outcome, as weighted data.

    Parameters
    ----------
    n_times : int
        The nodes.
    causes : int
        One cause, or two coded ``R`` and ``D`` in the frame.
    g1, g, retained, hazard : callable
        The tables of the law; each later-node table reads ``(node, w, a, previous)``.
    """

    n_times: int = 3
    causes: int = 1
    g1: Callable[[int], float] = _g1
    g: Callable[[int, int, int], float] = _g
    retained: Callable[[int, int, int, int], float] = _retained
    hazard: Callable[[int, int, int, int], float] = _hazard
    support: tuple[tuple[Any, ...], ...] = field(init=False)
    masses: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        points: list[tuple[tuple[Any, ...], float]] = []
        for w in (0, 1):
            self._walk((w,), 0.5, 1, 0, points)
        object.__setattr__(self, "support", tuple(point for point, _ in points))
        object.__setattr__(self, "masses", np.array([mass for _, mass in points]))
        assert abs(self.masses.sum() - 1.0) < 1e-12

    def hazards(self, node: int, w: int, a: int, previous: int) -> tuple[float, ...]:
        """The cause-specific hazards at node ``k`` given at risk."""
        if self.causes == 1:
            return (self.hazard(node, w, a, previous),)
        return _split(node, w, a, previous)

    def treat(self, node: int, w: int, previous: int) -> float:
        """``P(A_k = 1 | W, A_{k-1})``."""
        return self.g1(w) if node == 1 else self.g(node, w, previous)

    def _walk(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        previous: int,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        width = 1 + 3 * self.n_times
        if node > self.n_times:
            out.append((prefix, mass))
            return
        w = prefix[0]
        p = self.treat(node, w, previous)
        for a, pa in ((1, p), (0, 1.0 - p)):
            if pa <= 0.0:
                continue
            c = self.retained(node, w, a, previous)
            here = mass * pa
            if c < 1.0:
                point = (*prefix, a, 0)
                out.append((point + (None,) * (width - len(point)), here * (1.0 - c)))
            lam = self.hazards(node, w, a, previous)
            for code, share in enumerate(lam, start=1):
                if share > 0.0:
                    point = (*prefix, a, 1, code)
                    out.append((point + (None,) * (width - len(point)), here * c * share))
            rest = here * c * (1.0 - sum(lam))
            if rest > 0.0:
                self._walk((*prefix, a, 1, 0), rest, node + 1, a, out)

    # ---------------------------------------------------------------- the sample

    def frame(self) -> pd.DataFrame:
        """The law as data: one row per support point, with its probability in ``wt``."""

        def column(i: int) -> np.ndarray:
            return np.array(
                [np.nan if point[i] is None else float(point[i]) for point in self.support]
            )

        out: dict[str, np.ndarray] = {"W": column(0)}
        for node in range(1, self.n_times + 1):
            base = 1 + 3 * (node - 1)
            out[f"A{node}"] = column(base)
            out[f"C{node}"] = column(base + 1)
            code = column(base + 2)
            if self.causes == 2:
                out[f"R{node}"] = np.where(np.isnan(code), np.nan, (code == 1).astype(float))
                out[f"D{node}"] = np.where(np.isnan(code), np.nan, (code == 2).astype(float))
            else:
                out[f"Y{node}"] = code
        frame = pd.DataFrame(out)
        frame["wt"] = self.masses * len(self.support)
        return frame

    def columns(self) -> dict[str, Any]:
        """The ``fit`` keywords."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "treatment": [f"A{k}" for k in nodes],
            "baseline": ["W"],
            "censoring": [f"C{k}" for k in nodes],
            "weights": "wt",
        }
        if self.causes == 2:
            out["outcome"] = {
                "relapse": [f"R{k}" for k in nodes],
                "death": [f"D{k}" for k in nodes],
            }
        else:
            out["outcome"] = [f"Y{k}" for k in nodes]
        return out

    # ---------------------------------------------------------------- the truth

    def truth(self, regimen: Callable[[int, int, int], float], horizon: int, cause: int = 1) -> float:
        """``E sum_u q(u | W) F_j(t | u, W)`` by enumerating the paths.

        ``regimen(node, w, previous)`` is the probability of treating at the node.
        """
        total = 0.0
        for w in (0, 1):
            for path in itertools.product((0, 1), repeat=horizon):
                mass = 0.5
                previous = 0
                for node, a in enumerate(path, start=1):
                    p = regimen(node, w, previous)
                    mass *= p if a == 1 else 1.0 - p
                    previous = a
                if mass == 0.0:
                    continue
                survival, incidence, previous = 1.0, 0.0, 0
                for node, a in enumerate(path, start=1):
                    lam = self.hazards(node, w, a, previous)
                    incidence += survival * lam[cause - 1]
                    survival *= 1.0 - sum(lam)
                    previous = a
                total += mass * incidence
        return total


# ------------------------------------------------------------------- the regimens


def _always(node: int, w: int, previous: int) -> float:
    return 1.0


def _never(node: int, w: int, previous: int) -> float:
    return 0.0


def _rule_truth(node: int, w: int, previous: int) -> float:
    """``d(W)``: treat when ``W = 1`` at the odd nodes, and when ``W = 0`` at the even ones."""
    return float((w + node) % 2 == 0)


def _policy_truth(node: int, w: int, previous: int) -> float:
    """The known policy: treat with probability ``0.3 + 0.4 A_{k-1} + 0.1 W``."""
    return 0.3 + 0.4 * previous + 0.1 * w


def _stay_truth(node: int, w: int, previous: int) -> float:
    """Treat at node 1 with probability ``0.3 + 0.1 W``, then stay on the previous arm."""
    return 0.3 + 0.1 * w if node == 1 else float(previous)


def _rule_node(node: int) -> Callable[[Any], Any]:
    def assign(h: Any) -> Any:
        return ((np.asarray(h["W"], dtype=float) + node) % 2 == 0).astype(int)

    return assign


def _policy_density(node: int, n_times: int) -> Stochastic:
    def density(h: Any) -> np.ndarray:
        previous = (
            np.zeros(len(h)) if node == 1 else np.asarray(h[f"A{node - 1}"], dtype=float)
        )
        p = 0.3 + 0.4 * previous + 0.1 * np.asarray(h["W"], dtype=float)
        return np.column_stack([1.0 - p, p])

    return Stochastic(density, f"q{node}", density_kind="known")


def _stay_density(node: int) -> Stochastic:
    def density(h: Any) -> np.ndarray:
        if node == 1:
            p = 0.3 + 0.1 * np.asarray(h["W"], dtype=float)
            return np.column_stack([1.0 - p, p])
        previous = np.asarray(h[f"A{node - 1}"], dtype=float)
        return np.column_stack([1.0 - previous, previous])

    return Stochastic(density, f"stay{node}", density_kind="known")


def regimens(n_times: int = 3) -> dict[str, Any]:
    """The five regimens of the base law, by label."""
    return {
        "always": (1,) * n_times,
        "never": (0,) * n_times,
        "rule": DynamicRegimen(
            "rule", tuple(_rule_node(k) for k in range(1, n_times + 1)), rule_kind="known"
        ),
        "policy": DynamicRegimen(
            "policy", tuple(_policy_density(k, n_times) for k in range(1, n_times + 1))
        ),
        "stay": DynamicRegimen("stay", tuple(_stay_density(k) for k in range(1, n_times + 1))),
    }


TRUTHS: dict[str, Callable[[int, int, int], float]] = {
    "always": _always,
    "never": _never,
    "rule": _rule_truth,
    "policy": _policy_truth,
    "stay": _stay_truth,
}


# -------------------------------------------------------------------- the learners


class WrongHazard(CellMeans):
    """Saturated in ``(W, A_k)``: the first and the last design column, dropping ``A_{k-1}``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WrongHazard:
        super().fit(_ends(X), y, sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(_ends(X))


def _ends(X: Any) -> np.ndarray:
    matrix = np.asarray(X, dtype=float)
    return matrix[:, [0, -1]]


def ltmle(chosen: dict[str, Any], **settings: Any) -> LTMLE:
    """A hazard ``LTMLE`` with the correct mechanism and the wrong hazard learner."""
    options: dict[str, Any] = {
        "n_folds": 1,
        "survival_model": "hazard",
        "targeting": "one_step",
        "step_size": STEP,
        "tol": TOL,
        "g_bounds": (1e-6, 1.0),
        "outcome_learner": WrongHazard(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
    }
    options.update(settings)
    return LTMLE(chosen, **options)


#: The regimen sets of the exact witnesses.  A set that holds always, never and a mixture of
#: the two ("stay") has nearly collinear targets, which the diagonal metric of the one-step
#: walk crosses in many thousands of steps, so "stay" is fitted beside "always" alone.
SETS = (("always", "never", "rule", "policy"), ("always", "stay"))

LAW = PathLaw()
COMPETING = PathLaw(causes=2)


def fit(law: PathLaw, labels: tuple[str, ...], **settings: Any) -> Any:
    """The hazard fit of ``labels`` on ``law``, with the sparse-curve warning muted."""
    chosen = {label: regimens(law.n_times)[label] for label in labels}
    frame = settings.pop("frame", None)
    columns = {**law.columns(), **settings.pop("columns", {})}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*read their influence curve from few rows")
        return ltmle(chosen, **settings).fit(law.frame() if frame is None else frame, **columns)


def risk(result: Any, label: str, horizon: int, cause: str | None = None) -> float:
    key = f"{label} @ t={horizon}" if cause is None else f"{label}, {cause} @ t={horizon}"
    return float(result.fits[key].psi_scaled)


def miss(result: Any, law: PathLaw, labels: tuple[str, ...], cause: int = 1) -> float:
    """The largest distance of an estimate from the truth, over the labels and horizons."""
    name = None if law.causes == 1 else ("relapse", "death")[cause - 1]
    return max(
        abs(risk(result, label, t, name) - law.truth(TRUTHS[label], t, cause))
        for label in labels
        for t in range(1, law.n_times + 1)
    )


@contextmanager
def captured(name: str) -> Iterator[list[tuple[tuple[Any, ...], Any]]]:
    """Record the arguments and the return value of ``hazard_module.<name>``."""
    original = getattr(hazard_module, name)
    calls: list[tuple[tuple[Any, ...], Any]] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        calls.append((args, out))
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hazard_module, name, spy)
        from cleverly.longitudinal import estimator as estimator_module

        if hasattr(estimator_module, name):
            patch.setattr(estimator_module, name, spy)
        yield calls


# --------------------------------------------------------------------------- witnesses


@pytest.mark.parametrize("labels", SETS)
@pytest.mark.parametrize("targeting", ["one_step", "iterative"])
def test_a_wrong_hazard_with_the_true_mechanism_gives_the_truth(
    labels: tuple[str, ...], targeting: str
) -> None:
    """Every regimen's curve equals the enumerated g-formula, on both algorithms."""
    result = fit(LAW, labels, targeting=targeting, max_iter=200)
    assert miss(result, LAW, labels) < EXACT
    record = result.fits[f"{labels[0]} @ t=3"].hazard
    assert record.converged and record.n_iter > 0
    # The wrong learner's plug-in misses, so the walk did the work.
    first = np.asarray(record.initial_levels)
    truths = [LAW.truth(TRUTHS[label], 3) for label in labels]
    if targeting == "one_step":
        start = first.reshape(len(labels), 1, 3)[:, 0, 2]
    else:
        start = first.reshape(len(labels), 1)[:, 0]
    assert np.min(np.abs(start - truths)) > MISSED


def test_each_competing_cause_is_exact() -> None:
    labels = SETS[0]
    result = fit(COMPETING, labels)
    for cause in (1, 2):
        assert miss(result, COMPETING, labels, cause) < EXACT


def test_the_curves_are_monotone_and_the_estimates_experimental() -> None:
    """A one-step curve is non-decreasing, and no support row covers the design yet."""
    result = fit(LAW, SETS[0])
    for label in SETS[0]:
        curve = [risk(result, label, t) for t in (1, 2, 3)]
        assert all(later >= earlier for earlier, later in itertools.pairwise(curve))
    labels = {estimate.support.label for estimate in result.estimates.values()}
    assert labels == {"experimental"}


def test_the_copies_of_one_hazard_stay_bit_equal_after_the_walk() -> None:
    with captured("walk_curve") as calls:
        fit(LAW, SETS[0])
    ((args, (state, _)),) = calls
    inputs = args[1]
    assert inputs.multinode and inputs.n_levels > len(SETS[0])
    shared = inputs.share_copies(state.hazards)
    assert np.array_equal(shared, state.hazards)
    # The copies are genuine: at node 1 some prefix class holds slots of two regimens.
    owner = np.argmax(inputs.q > 0.0, axis=0)  # (n, U)
    ids = inputs.prefix[:, 0, :]
    same = ids[:, :, None] == ids[:, None, :]
    assert np.any(same & (owner[:, :, None] != owner[:, None, :]))


@pytest.mark.parametrize("targeting", ["one_step", "iterative"])
def test_the_influence_curve_is_the_iterated_mean_curve(targeting: str) -> None:
    """With saturated hazards and the true mechanism, each curve is the efficient one.

    The iterated-mean fit of the same regimens, saturated as well, reports the efficient
    influence function at the truth, so the hazard fit's curve equals it row by row, for
    every regimen and every contrast.
    """
    labels = SETS[0]
    hazard = fit(LAW, labels, targeting=targeting, outcome_learner=CellMeans(), max_iter=200)
    iterated = fit(
        LAW,
        labels,
        survival_model="iterated_mean",
        targeting="iterative",
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
    )
    curves, expected = hazard.influence_curves, iterated.influence_curves
    assert set(curves) == set(expected)
    assert any(name.startswith("ate_regimen[") for name in curves)
    for name, curve in curves.items():
        assert np.ptp(curve) > 0.1
        np.testing.assert_allclose(curve, expected[name], rtol=0.0, atol=1e-10)


def test_an_influence_curve_without_its_plug_in_term_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutation: the curve drops :math:`\\sum_u q_r(u) F_j(t \\mid u) - \\psi`."""
    labels = ("policy",)
    exact = fit(LAW, labels, outcome_learner=CellMeans()).influence_curves

    def no_plug_in(state: Any, inputs: Any, horizons: Any) -> Any:
        terms, _ = hazard_module._score_terms(state, inputs, horizons)
        index = [t - 1 for t in horizons]
        levels = state.levels(inputs.q, inputs.weights)[:, :, index]
        return levels, terms * inputs.weights

    monkeypatch.setattr(hazard_module, "influence", no_plug_in)
    mutated = fit(LAW, labels, outcome_learner=CellMeans()).influence_curves
    name = "risk_regimen[policy @ t=2]"
    assert np.max(np.abs(mutated[name] - exact[name])) > MISSED


# ------------------------------------------------------------------- mutation controls


def _from_plan_values(
    data: Any, plans: Any
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], tuple[int, ...]]:
    """The mutation: each slot built from ``Plan.values`` and the plan's evaluated policy.

    ``Plan.values`` holds code zero off the risk set and the collapsed arms of a one-hot
    policy, read at the observed earlier arm; a policy node reads ``Plan.policy_at``, which is
    zero off the risk set.
    """
    paths, masses, owners, draws = [], [], [], []
    for r, plan in enumerate(plans):
        current = [(np.array(plan.values, dtype=float), np.ones(data.n))]
        for time in range(1, data.n_times + 1):
            if not plan.is_policy_node(time):
                continue
            density = plan.policy_at(time)
            grown = []
            for codes, mass in current:
                for code in range(density.shape[1]):
                    branch = mass * density[:, code]
                    if np.any(branch > 0.0):
                        extended = codes.copy()
                        extended[:, time - 1] = float(code)
                        grown.append((extended, branch))
            current = grown
        for codes, mass in current:
            paths.append(codes)
            masses.append(mass)
            owners.append(r)
        draws.append(0)
    return paths, masses, owners, tuple(draws)


def test_mutation_slots_from_plan_values_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Slots read from ``Plan.values`` miss for "always" and for the one-hot "stay"."""
    monkeypatch.setattr(hazard_module, "_enumerate_paths", _from_plan_values)
    for labels in (("always",), ("stay",)):
        assert miss(fit(LAW, labels), LAW, labels) > MISSED


def test_mutation_copies_updated_apart_diverge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each slot moved by its own term only: the copies of one prefix part.

    No miss against the truth is asserted, since path by path the copies can each solve their
    own score.  The reported estimate is no longer the plug-in of one law: it differs from the
    plug-in recomputed from one copy per prefix.
    """
    monkeypatch.setattr(hazard_module.HazardInputs, "prefix_sum", lambda self, step: step)
    labels = ("always", "stay")
    with captured("walk_curve") as calls, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit(LAW, labels)
    ((args, (state, _)),) = calls
    inputs = args[1]
    shared = hazard_module.HazardState(
        inputs.share_copies(state.logits), state.full, state.node_bounds
    )
    assert not np.array_equal(inputs.share_copies(state.hazards), state.hazards)
    recomputed = shared.levels(inputs.q, inputs.weights)
    reported = np.array([[risk(result, label, t) for t in (1, 2, 3)] for label in labels])
    assert np.max(np.abs(recomputed[:, 0, :] - reported)) > 1e-6


def test_mutation_double_counted_slot_mass_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """"always" also carries mass on the duplicate of its path that "stay" owns."""
    original = hazard_module.path_slots

    def doubled(data: Any, plans: Any) -> Any:
        slots = original(data, plans)
        q = slots.q.copy()
        ones = [
            u
            for u, plan in enumerate(slots.plans)
            if slots.owner[u] == 1 and np.all(plan.values == 1.0)
        ]
        (duplicate,) = ones
        own = int(np.flatnonzero(slots.owner == 0)[0])
        q[0, :, duplicate] = q[0, :, own]
        return replace(slots, q=q)

    labels = ("always", "stay")
    exact = fit(LAW, labels)
    assert miss(exact, LAW, labels) < EXACT
    monkeypatch.setattr(hazard_module, "path_slots", doubled)
    mutated = fit(LAW, labels)
    assert abs(risk(mutated, "always", 3) - LAW.truth(_always, 3)) > MISSED


def _node_one(original: Callable[..., Any]) -> Callable[..., Any]:
    """The mutation: the path weight keeps the treatment factor of node 1 alone."""

    def cumulative(self: Any, data: Any, plan: Any, bounds: tuple[float, float]) -> Any:
        if not plan.label.startswith("\x00hazard-path"):
            return original(self, data, plan, bounds)
        running = np.ones(data.n)
        columns = []
        for time in range(1, data.n_times + 1):
            if time == 1:
                running = running * self.treatment[0][plan.label]
            running = running * self.censoring[time - 1][plan.label]
            columns.append(running)
        raw = np.column_stack(columns)
        return raw, np.clip(raw, *bounds)

    return cumulative


def _copy(node: int, w: int, previous: int) -> float:
    """A later treatment that copies the previous arm."""
    return float(previous)


COPY = PathLaw(g=_copy)


def test_mutation_node_one_weight_misses_and_its_control_agrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """:math:`\\overline{gc}_k` replaced by :math:`g_1 c_1 \\cdots c_k`.

    On the base law the dropped factors are not one and the estimate misses.  On a law whose
    later treatment copies the previous arm, every dropped factor is one along the always and
    never paths, and the mutation agrees with the exact fit.
    """
    from cleverly.longitudinal.sequential import Mechanism

    labels = ("always", "never")
    exact_copy = fit(COPY, labels)
    assert miss(exact_copy, COPY, labels) < EXACT
    monkeypatch.setattr(
        Mechanism, "cumulative_with_unbounded", _node_one(Mechanism.cumulative_with_unbounded)
    )
    assert miss(fit(LAW, labels), LAW, labels) > MISSED
    control = fit(COPY, labels)
    for label in labels:
        for t in (1, 2, 3):
            assert risk(control, label, t) == pytest.approx(risk(exact_copy, label, t), abs=EXACT)


def test_mutation_a_deviated_residual_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit whose observed prefix no slot carries contributes its residual at slot 0."""
    original = hazard_module.path_slots

    def leaking(data: Any, plans: Any) -> Any:
        slots = original(data, plans)
        match = slots.observed_match.copy()
        at_risk, _ = hazard_module.event_indicators(data)
        deviated = at_risk & ~match.any(axis=2)
        assert deviated.any()
        match[:, :, 0] |= deviated
        return replace(slots, observed_match=match)

    monkeypatch.setattr(hazard_module, "path_slots", leaking)
    # A static plan and a rule each have deviated units; the policy puts mass on every path.
    for labels in (("always",), ("rule",)):
        assert miss(fit(LAW, labels), LAW, labels) > MISSED


# ------------------------------------------------------------------------ held cells


def _no_event_on_two_arms(node: int, w: int, a: int, previous: int) -> float:
    """The base hazard, but no event at node 2 after treating at nodes 1 and 2."""
    if node == 2 and a == 1 and previous == 1:
        return 0.0
    return _hazard(node, w, a, previous)


EMPTY = PathLaw(hazard=_no_event_on_two_arms)


def _treat_w(h: Any) -> Any:
    return np.asarray(h["W"], dtype=int)


def _treat_w_truth(node: int, w: int, previous: int) -> float:
    return float(w)


def test_an_empty_prefix_cell_holds_every_copy() -> None:
    """The cell (node 2, prefix (1, 1)) has followers and no event.

    Its hazard sits in the slot of "always" and in the slot of a rule that treats when
    ``W = 1``.  Both copies are held at zero, stay bit-equal, and the estimates are exact.
    """
    chosen = {
        "always": (1, 1, 1),
        "treat W": DynamicRegimen("treat W", (_treat_w,) * 3, rule_kind="known"),
    }
    with captured("walk_curve") as calls, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*read their influence curve from few rows")
        result = ltmle(chosen).fit(EMPTY.frame(), **EMPTY.columns())
    ((args, (state, _)),) = calls
    inputs = args[1]
    paths = np.stack([plan.values for plan in []], axis=0) if False else None
    del paths
    held = np.zeros(inputs.at_risk.shape + (inputs.n_levels,), dtype=bool)
    empty, _ = inputs.held_cells()
    held = empty[:, 1, :, 0]  # node 2, cause 1
    assert held[:, 0].all()  # every unit's "always" slot
    w = EMPTY.frame()["W"].to_numpy()
    assert np.array_equal(held[:, 1], w == 1)  # the rule's slot where it treats twice
    assert np.all(state.hazards[:, 1, 0, 0] == 0.0)
    assert np.all(state.hazards[w == 1, 1, 1, 0] == 0.0)
    assert np.array_equal(inputs.share_copies(state.hazards), state.hazards)
    for t in (1, 2, 3):
        assert risk(result, "always", t) == pytest.approx(EMPTY.truth(_always, t), abs=EXACT)
        assert risk(result, "treat W", t) == pytest.approx(
            EMPTY.truth(_treat_w_truth, t), abs=EXACT
        )
    assert result.fits["always @ t=3"].hazard_constant_nodes == (2,)


# ------------------------------------------------------------------------- reductions


HELD = PathLaw(g=_copy)


def test_a_treatment_that_never_changes_reduces_to_the_held_fit() -> None:
    """Declared at three nodes, a treatment that copies node 1 gives the held fit's curve."""
    frame = HELD.frame()
    frame["A"] = frame["A1"]
    multi = fit(HELD, ("always", "never"), frame=frame)
    columns = {**HELD.columns(), "treatment": "A"}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*read their influence curve from few rows")
        held = ltmle({"always": 1, "never": 0}).fit(frame, **columns)
    for label in ("always", "never"):
        for t in (1, 2, 3):
            assert risk(multi, label, t) == pytest.approx(risk(held, label, t), abs=1e-10)


# ---------------------------------------------------------------------------- settings


@pytest.mark.parametrize(
    "settings",
    [
        {"iterate": "best"},
        {"node_bounds": {2: (0.0, 0.6)}},
        {"targeting": "iterative", "step_bound": 0.5, "tol": 1e-9, "max_iter": 500},
        {"stopping": "scaled_var"},
    ],
    ids=["best iterate", "node bounds", "step bound", "declared rule"],
)
def test_the_settings_of_the_walk_carry_over(settings: dict[str, Any]) -> None:
    labels = SETS[0]
    result = fit(LAW, labels, **settings)
    record = result.fits["always @ t=3"].hazard
    assert record.converged
    if "stopping" in settings:
        # The declared rule stops at its threshold, short of the exact solve.
        assert record.rule_ratio is not None and record.rule_ratio <= 1.0
        return
    assert miss(result, LAW, labels) < EXACT
    if "iterate" in settings:
        assert record.best_iterate is not None
    if "node_bounds" in settings:
        assert record.share_clipped and record.share_clipped[0][0] == 2


def test_clusters_sum_the_curve_within_a_cluster() -> None:
    labels = ("policy",)
    frame = LAW.frame()
    frame["cl"] = np.arange(len(frame)) // 2
    plain = fit(LAW, labels)
    clustered = fit(LAW, labels, frame=frame, columns={"id": "cl"})
    name = "risk_regimen[policy @ t=3]"
    assert clustered.estimates[name].psi == pytest.approx(plain.estimates[name].psi, abs=1e-12)
    curve = plain.influence_curves[name]
    sums = np.bincount(frame["cl"].to_numpy(), weights=curve)
    n = len(frame)
    expected = np.sqrt(np.sum(sums**2) / n**2 * (len(sums) / (len(sums) - 1)))
    assert np.sqrt(clustered.estimates[name].variance) == pytest.approx(expected, rel=1e-6)


def test_the_path_hazards_are_cross_fitted() -> None:
    """Each held-out row's path hazard is the training folds' fit at the path's design."""
    with captured("fit_hazards") as calls:
        result = fit(LAW, ("policy",), n_folds=3, random_state=0)
    assert result.fits["policy @ t=3"].hazard.converged
    ((args, initial),) = calls
    data, slots = args[0], args[1]
    at_risk, events = hazard_module.event_indicators(data)
    assignment = result.folds.assignment
    design = data.history_design(2, include_current=True)
    expected = np.empty((data.n, len(slots)))
    for fold in np.unique(assignment):
        train = at_risk[:, 1] & (assignment != fold)
        learner = WrongHazard().fit(design[train], events[train, 1, 0], data.weights[train])
        held = assignment == fold
        for u, plan in enumerate(slots):
            at = data.history_design(2, treatment=plan.values, include_current=True)
            expected[held, u] = learner.predict(at[held])
    np.testing.assert_allclose(initial.hazards[:, 1, :, 0], expected, rtol=0.0, atol=1e-12)


def test_a_replay_at_the_fitted_bound_is_bit_identical() -> None:
    from cleverly.longitudinal.estimator import _fitted_replay_matches, _refit_bound

    result = fit(LAW, SETS[0])
    replay = _refit_bound(result, result.repeats[0], result.config.g_bounds)
    assert _fitted_replay_matches(result, result.repeats[0], replay)
    frame = result.diagnostics.truncation_curve(bounds=[0.01, 0.05])
    assert len(frame) > 0


# ------------------------------------------------------------------------ working model


TWO = PathLaw(n_times=2)
PATHS = {"00": (0, 0), "01": (0, 1), "10": (1, 0), "11": (1, 1)}


def _msm_design(modifier: bool) -> Callable[[Any, int, Any], np.ndarray]:
    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        w = np.asarray(frame["W"], dtype=float)
        dose = float(sum(PATHS[str(label)]))
        columns = [np.ones_like(w), np.full_like(w, dose), np.full_like(w, float(horizon))]
        if modifier:
            columns.append(dose * w)
        return np.column_stack(columns)

    return design


def _msm(modifier: bool) -> Any:
    from cleverly.msm import MSM

    terms = ("1", "dose", "t", "dose x W") if modifier else ("1", "dose", "t")
    return MSM(design=_msm_design(modifier), terms=terms, design_kind="known", link="logit")


def _path_incidence(law: PathLaw, w: int, path: tuple[int, ...], horizon: int) -> float:
    survival, incidence, previous = 1.0, 0.0, 0
    for node, a in enumerate(path[:horizon], start=1):
        lam = law.hazards(node, w, a, previous)
        incidence += survival * lam[0]
        survival *= 1.0 - sum(lam)
        previous = a
    return incidence


def _exact_beta(law: PathLaw, modifier: bool) -> np.ndarray:
    """The projection of the law's cell risks onto the logistic working model, by Newton."""
    design = _msm_design(modifier)
    cells = [(label, t) for t in (1, 2) for label in PATHS]
    frame = pd.DataFrame({"W": [0.0, 1.0]})
    phi = np.stack([design(label, t, frame) for label, t in cells], axis=1)  # (2, C, p)
    psi = np.array(
        [[_path_incidence(law, w, PATHS[label], t) for label, t in cells] for w in (0, 1)]
    )
    beta = np.zeros(phi.shape[2])
    for _ in range(60):
        mean = 1.0 / (1.0 + np.exp(-np.einsum("wcp,p->wc", phi, beta)))
        score = sum(0.5 * phi[w].T @ (psi[w] - mean[w]) for w in (0, 1))
        jacobian = sum(0.5 * (phi[w].T * (mean[w] * (1.0 - mean[w]))) @ phi[w] for w in (0, 1))
        beta = beta + np.linalg.solve(jacobian, score)
    return beta


def _msm_fit(modifier: bool) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*read their influence curve from few rows")
        return ltmle(PATHS, msm=_msm(modifier), tol=1e-12).fit(TWO.frame(), **TWO.columns())


def test_a_working_model_over_the_paths_gives_the_exact_projection() -> None:
    """``msm=`` over the four static paths of two nodes: a dose term and a modifier."""
    truth = _exact_beta(TWO, True)
    fit_ = _msm_fit(True).msm_fits[0]
    assert fit_.converged
    np.testing.assert_allclose(fit_.beta, truth, rtol=0.0, atol=EXACT)
    plug_in = fit_.hazard.initial_levels
    assert np.max(np.abs(plug_in - truth)) > MISSED


def test_mutation_targeting_the_cells_misses_with_a_modifier_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cell scores solve the coefficient scores only when the design is free of ``W``."""
    walk = hazard_module.walk_curve

    def cells(*args: Any, combination: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    monkeypatch.setattr(hazard_module, "walk_curve", cells)
    mutated = _msm_fit(True).msm_fits[0].beta
    assert abs(mutated[3] - _exact_beta(TWO, True)[3]) > MISSED
    control = _msm_fit(False).msm_fits[0].beta
    np.testing.assert_allclose(control, _exact_beta(TWO, False), rtol=0.0, atol=EXACT)


class _Untouched:
    """A learner that fails the test if the fit reaches it."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("H17 must refuse before any learner")

    def predict_proba(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("H17 must refuse before any learner")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


def _coin(node: int) -> Stochastic:
    def density(history: pd.DataFrame) -> np.ndarray:
        return np.full((len(history), 2), 0.5)

    return Stochastic(density, f"coin {node}", density_kind="known")


def test_h17_a_policy_at_many_nodes_is_refused_before_any_learner() -> None:
    """A fair coin at every node gives ``2**k`` paths after ``k`` nodes, until the cap."""
    n, n_times = 2_000, 10
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"W": rng.normal(size=n)})
    alive = np.ones(n, dtype=bool)
    for k in range(1, n_times + 1):
        frame[f"A{k}"] = np.where(alive, rng.integers(0, 2, n), np.nan)
        frame[f"C{k}"] = np.where(alive, 1.0, np.nan)
        event = rng.random(n) < 0.05
        frame[f"Y{k}"] = np.where(alive, event, np.nan)
        alive &= ~event
    nodes = range(1, n_times + 1)
    coin = DynamicRegimen("coin", tuple(_coin(k) for k in nodes))
    learners = {
        name: _Untouched() for name in ("outcome_learner", "treatment_learner", "censoring_learner")
    }
    estimator = LTMLE({"coin": coin}, survival_model="hazard", n_folds=1, **learners)
    with pytest.raises(CapabilityError) as raised:
        estimator.fit(
            frame,
            treatment=[f"A{k}" for k in nodes],
            outcome=[f"Y{k}" for k in nodes],
            censoring=[f"C{k}" for k in nodes],
            baseline=["W"],
        )
    refusal = raised.value.refusal
    assert refusal.reason is RefusalReason.API_REDIRECTION
    found = re.search(
        r"at (\d+) nodes, which gives (\d+) treatment paths .* (\d+) units, (\d+) nodes, "
        r"(\d+) paths, (\d+) causes and (\d+) design columns give a predicted peak of "
        r"([\d.]+) GB, above the limit of 2 GB\.",
        str(raised.value),
    )
    assert found is not None, str(raised.value)
    drawn, paths, units, times, slots, causes, columns = map(int, found.groups()[:7])
    assert (units, times, causes, paths, slots) == (n, n_times, 1, 2**drawn, 2**drawn)
    limit = hazard_module.PATH_PEAK_LIMIT
    peak = hazard_module.path_peak(units, times, slots, causes, columns)
    assert peak > limit >= hazard_module.path_peak(units, times, slots // 2, causes, columns)
    assert float(found.group(8)) == pytest.approx(peak / 1024.0**3, abs=0.05)


#: A record, not a run: the traced peak (MB) of whole one-step fits at ``iterate="best"`` of a
#: known policy with full support at every node, ``(n, T, J, p, U, peak)``, from which
#: :data:`~cleverly.longitudinal.hazard.PATH_STATE_BYTES` and ``PATH_DESIGN_BYTES`` were set.
PEAKS = (
    (20_000, 4, 1, 5, 16, 367.7),
    (20_000, 5, 1, 6, 32, 874.0),
    (20_000, 4, 2, 5, 16, 522.6),
    (20_000, 4, 1, 44, 16, 362.2),
    (40_000, 4, 1, 5, 16, 711.9),
)


@pytest.mark.parametrize(("n", "n_times", "causes", "columns", "paths", "peak"), PEAKS)
def test_h17_the_predicted_peak_covers_each_measured_fit(
    n: int, n_times: int, causes: int, columns: int, paths: int, peak: float
) -> None:
    """The prediction stays above each recorded peak and within twice it."""
    predicted = hazard_module.path_peak(n, n_times, paths, causes, columns) / 1e6
    assert peak <= predicted <= 2.0 * peak
