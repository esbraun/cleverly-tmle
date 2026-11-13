"""A modified treatment policy on the hazard path, checked exactly on finite laws (X70 part (c)).

Each law below is a finite support whose conditional probabilities are multiples of one
quarter, laid out in exact proportions, so the empirical law equals the law and a fit with the
correct mechanism and a wrong hazard learner must return the g-formula exactly.  The truth is
computed from the law's tables in this file and never calls the estimator.

The dose ``A`` takes the values ``0, 1, 2``.  A fit reads it as three levels (a categorical
node), as a dose on the unit grid (a continuous node, with the binned density reading the
law's own bins through :func:`tests.longitudinal_mtp.exact_bins`), or beside a binary flag
``F`` as a vector with a dose (a mixed node).  The policy "up" moves the dose one step up and
keeps the top, ``d(a) = min(a + 1, 2)``.

The wrong hazard learner is an unpenalised logistic regression additive in its design: the
law's hazard has interactions, so its plug-in misses the truth and the walk has work to do.
Every witness checks that the initial plug-in misses, so an exact estimate is not an accident
of a correct initial fit.  Each mutation control breaks one component the witness reads and
must move the estimate.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.interventions import JointPolicy, ModifiedPolicy, Piece, Randomizer, Shift
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from tests.discrete_law_point_survival import CellMeans, CellProbabilities
from tests.longitudinal_mtp import exact_bins

pytestmark = pytest.mark.usefixtures("exact_grid_law")

INF = np.inf
#: The first one-step step and the relative-score tolerance of the shipped exact witnesses.
STEP = 2e-2
TOL = 1e-10
EXACT = 1e-8
#: How far a mutation, or a wrong initial plug-in, must move the estimate.
MISSED = 1e-3

#: ``P(A = a | W, F)``, indexed ``[(w + f) % 2, a]``.
DOSE = np.array([[0.25, 0.25, 0.50], [0.50, 0.25, 0.25]])
#: The one-cause hazard by node, indexed ``[node - 1][w][a]``; ``F = 1`` adds a quarter.
HAZARD = (
    ((0.25, 0.50, 0.50), (0.50, 0.75, 0.25)),
    ((0.50, 0.25, 0.75), (0.25, 0.50, 0.25)),
    ((0.25, 0.25, 0.50), (0.50, 0.75, 0.25)),
)
#: The two-cause split ``(P(cause 1), P(cause 2))`` at risk, by ``(w + 2a + f + node) % 3``.
PAIRS = ((0.25, 0.25), (0.50, 0.25), (0.25, 0.50))


@dataclass(frozen=True)
class GridLaw:
    """A held baseline dose ``A``, an optional flag ``F``, and a survival outcome.

    Parameters
    ----------
    n_times : int
        The nodes.  A vector treatment is declared one node at a time, so a held vector is
        not a design the container reads: a law with ``flag`` has one node.
    flag : bool
        Whether the treatment is the vector ``(F, A)``.
    causes : int
        One cause, or two coded ``R`` and ``D`` in the frame.
    censored : bool
        Whether units drop out, with a retention that depends on ``(W, F, A)``.
    empty : Callable or None
        ``(node, w, f, a) -> bool``: the cells at which cause 2 never occurs.  Its mass
        moves to cause 1.
    """

    n_times: int = 2
    flag: bool = False
    causes: int = 1
    censored: bool = True
    empty: Callable[[int, int, int, int], bool] | None = None
    support: tuple[tuple[Any, ...], ...] = field(init=False)
    counts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        assert not self.flag or self.n_times == 1
        points: list[tuple[tuple[Any, ...], float]] = []
        for w in (0, 1):
            for f in (0, 1) if self.flag else (0,):
                pf = (0.25 if f == w else 0.75) if self.flag else 1.0
                for a in range(3):
                    mass = 0.5 * pf * DOSE[(w + f) % 2, a]
                    self._walk((w, f, a), mass, 1, points)
        support = tuple(point for point, _ in points)
        masses = np.array([mass for _, mass in points])
        n = 2 * 4 ** (1 + int(self.flag) + 2 * self.n_times)
        counts = np.rint(masses * n)
        assert np.max(np.abs(masses * n - counts)) < 1e-9
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "counts", counts.astype(int))

    # ---------------------------------------------------------------- the tables

    def retained(self, node: int, w: int, f: int, a: int) -> float:
        """``P(C_k = 1 | W, F, A)``."""
        return 0.75 - 0.25 * ((w + f + a + node) % 2) if self.censored else 1.0

    def hazards(self, node: int, w: int, f: int, a: int) -> tuple[float, ...]:
        """The cause-specific hazards at node ``k`` given at risk."""
        if self.causes == 1:
            return (min(HAZARD[node - 1][w][a] + 0.25 * f, 0.75),)
        first, second = PAIRS[(w + 2 * a + f + node) % 3]
        if self.empty is not None and self.empty(node, w, f, a):
            return (first + second, 0.0)
        return (first, second)

    def _walk(
        self,
        prefix: tuple[Any, ...],
        mass: float,
        node: int,
        out: list[tuple[tuple[Any, ...], float]],
    ) -> None:
        width = 3 + 2 * self.n_times
        if node > self.n_times:
            out.append((prefix, mass))
            return
        w, f, a = prefix[:3]
        c = self.retained(node, w, f, a)
        if c < 1.0:
            out.append(((*prefix, 0) + (None,) * (width - len(prefix) - 1), mass * (1.0 - c)))
        lam = self.hazards(node, w, f, a)
        for code, share in enumerate(lam, start=1):
            if share > 0.0:
                point = (*prefix, 1, code)
                out.append((point + (None,) * (width - len(point)), mass * c * share))
        self._walk((*prefix, 1, 0), mass * c * (1.0 - sum(lam)), node + 1, out)

    # ---------------------------------------------------------------- the sample

    def frame(self) -> pd.DataFrame:
        """The exact sample, one row per unit."""
        cells = np.repeat(np.arange(len(self.support)), self.counts)

        def column(i: int) -> np.ndarray:
            raw = [np.nan if point[i] is None else float(point[i]) for point in self.support]
            return np.array(raw)[cells]

        out: dict[str, np.ndarray] = {"W": column(0), "A": column(2)}
        if self.flag:
            out["F"] = column(1)
        for node in range(1, self.n_times + 1):
            out[f"C{node}"] = column(1 + 2 * node)
            code = column(2 + 2 * node)
            if self.causes == 2:
                out[f"R{node}"] = np.where(np.isnan(code), np.nan, (code == 1).astype(float))
                out[f"D{node}"] = np.where(np.isnan(code), np.nan, (code == 2).astype(float))
            else:
                out[f"Y{node}"] = code
        return pd.DataFrame(out)

    def columns(self, reading: str) -> dict[str, Any]:
        """The ``fit`` keywords for a ``"categorical"``, ``"continuous"`` or ``"mixed"`` node."""
        nodes = range(1, self.n_times + 1)
        out: dict[str, Any] = {
            "baseline": ["W"],
            "censoring": [f"C{k}" for k in nodes],
            "treatment": [["F", "A"]] if reading == "mixed" else "A",
        }
        if reading != "categorical":
            out["continuous_treatment"] = ["A"]
        if self.causes == 2:
            out["outcome"] = {
                "relapse": [f"R{k}" for k in nodes],
                "death": [f"D{k}" for k in nodes],
            }
        else:
            out["outcome"] = [f"Y{k}" for k in nodes]
        return out

    # ---------------------------------------------------------------- the truth

    def incidence(self, w: int, f: int, a: int, horizon: int, cause: int = 1) -> float:
        """``F_j(t | a, f, w)`` from the hazard tables."""
        survival, total = 1.0, 0.0
        for node in range(1, horizon + 1):
            lam = self.hazards(node, w, f, a)
            total += survival * lam[cause - 1]
            survival *= 1.0 - sum(lam)
        return total

    def truth(
        self, branches: tuple[tuple[float, Callable[[int], int]], ...], horizon: int, cause: int = 1
    ) -> float:
        """``E sum_b p_b F_j(t | d_b(A), F, W)`` over the law of ``(W, F, A)``."""
        psi = 0.0
        for w in (0, 1):
            for f in (0, 1) if self.flag else (0,):
                pf = (0.25 if f == w else 0.75) if self.flag else 1.0
                for a in range(3):
                    mass = 0.5 * pf * DOSE[(w + f) % 2, a]
                    psi += mass * sum(
                        p * self.incidence(w, f, move(a), horizon, cause) for p, move in branches
                    )
        return psi


def up(a: int) -> int:
    return min(a + 1, 2)


def stay(a: int) -> int:
    return a


UP = ((1.0, up),)
RANDOM = ((0.25, up), (0.75, stay))
NATURAL = ((1.0, stay),)


def _up_apply(a: Any, h: Any) -> Any:
    return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)


def _pieces_up() -> tuple[Piece, ...]:
    return (
        Piece(-INF, 1.5, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
        Piece(1.5, INF),
    )


def policy(reading: str, kind: str) -> Any:
    """The policy ``kind`` ("up", "random" or "natural") on a node of ``reading``."""
    if reading == "mixed":
        if kind == "natural":
            return JointPolicy(kind, components={}, policy_kind="known")
        return JointPolicy(kind, components={"A": policy("continuous", kind)}, policy_kind="known")
    if kind == "natural":
        return ModifiedPolicy("natural", pieces=(Piece(-INF, INF),), policy_kind="known")
    if kind == "up":
        if reading == "categorical":
            return ModifiedPolicy("up", apply=_up_apply, policy_kind="known")
        return Shift(1.0, cap=2.0)
    return ModifiedPolicy(
        "random",
        pieces={"up": _pieces_up(), "stay": (Piece(-INF, INF),)},
        randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
        policy_kind="known",
    )


BRANCHES = {"up": UP, "random": RANDOM, "natural": NATURAL}


class Wrong(LogisticRegression):  # type: ignore[misc]
    """A wrong hazard learner: unpenalised logistic, additive in its design."""

    def __init__(self) -> None:
        super().__init__(C=np.inf, tol=1e-12, max_iter=10_000)


def fit(
    law: GridLaw,
    regimens: dict[str, Any],
    reading: str,
    *,
    learner: Any = None,
    targeting: str = "one_step",
    **settings: Any,
) -> Any:
    """A hazard fit of ``regimens`` on the exact sample of ``law``."""
    with warnings.catch_warnings(), exact_bins():
        warnings.simplefilter("ignore")
        return LTMLE(
            regimens,
            n_folds=settings.pop("n_folds", 1),
            survival_model=settings.pop("survival_model", "hazard"),
            targeting=targeting,
            step_size=STEP,
            tol=TOL,
            g_bounds=(1e-8, 1.0),
            outcome_learner=learner if learner is not None else Wrong(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
            **settings,
        ).fit(law.frame(), **law.columns(reading))


def risk(result: Any, label: str, horizon: int, cause: str | None = None) -> float:
    """The reported risk, or the cause incidence of a two-cause law."""
    if cause is None:
        return float(result.estimates[f"risk_regimen[{label} @ t={horizon}]"].psi)
    return float(result.estimates[f"cif_regimen[{label}, {cause} @ t={horizon}]"].psi)


def initial_level(result: Any, label: str, horizon: int) -> float:
    """The plug-in of the initial hazards, read from the walk's record."""
    record = result.fits[f"{label} @ t={horizon}"].hazard
    index = record.targets.index((label, None, horizon))  # one cause
    return float(record.initial_levels[index])


@contextmanager
def captured(name: str) -> Iterator[list[Any]]:
    """Record every return value of ``hazard_module.<name>`` while the block runs."""
    original = getattr(hazard_module, name)
    calls: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        out = original(*args, **kwargs)
        calls.append(out)
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hazard_module, name, spy)
        from cleverly.longitudinal import estimator as estimator_module

        if hasattr(estimator_module, name):
            patch.setattr(estimator_module, name, spy)
        yield calls


def _g(law: GridLaw, frame: pd.DataFrame) -> np.ndarray:
    """``(n,)`` the true ``g(A | W, F)`` of every row of the exact sample."""
    w = frame["W"].to_numpy(dtype=int)
    f = frame["F"].to_numpy(dtype=int) if law.flag else np.zeros_like(w)
    return np.asarray(DOSE[(w + f) % 2, frame["A"].to_numpy(dtype=int)], dtype=float)


# ------------------------------------------------------------------ exact witnesses

#: Every reading, policy and ratio route the witnesses cover.
CASES = [
    ("categorical", "up", "density"),
    ("categorical", "random", "density"),
    *[
        (reading, kind, ratio)
        for reading in ("continuous", "mixed")
        for kind in ("up", "random")
        for ratio in ("density", "classifier")
    ],
]


def _law(reading: str, **settings: Any) -> GridLaw:
    return GridLaw(flag=True, n_times=1, **settings) if reading == "mixed" else GridLaw(**settings)


@pytest.mark.parametrize("targeting", ["one_step", "iterative"])
@pytest.mark.parametrize(("reading", "kind", "ratio"), CASES)
def test_a_policy_on_a_wrong_hazard_is_exact(
    reading: str, kind: str, ratio: str, targeting: str
) -> None:
    """The weighted walk returns ``E sum_b p_b F(t | d_b(A), W)`` exactly (4.2 witnesses).

    The categorical reading is the three-level witness, the continuous reading the
    Gaussian-location witness in its exact form, and the mixed reading the vector witness,
    each with a deterministic policy and a two-branch randomizer, on both ratio routes.
    """
    law = _law(reading)
    result = fit(law, {kind: policy(reading, kind)}, reading, targeting=targeting, ratio=ratio)
    for t in range(1, law.n_times + 1):
        truth = law.truth(BRANCHES[kind], t)
        assert risk(result, kind, t) == pytest.approx(truth, abs=EXACT)
        assert abs(initial_level(result, kind, t) - truth) > MISSED
        assert result.fits[f"{kind} @ t={t}"].hazard.converged


# ------------------------------------------------------------------ the weight

def _drop(law: GridLaw, part: str) -> Callable[..., Any]:
    """An ``mtp_inputs`` that drops the density ratio, or ``1 / G_k``, from the weight."""
    original = hazard_module.mtp_inputs
    g = _g(law, law.frame())

    def mutated(data: Any, plan: Any, slots: Any, cumulative: Any, *args: Any, **kw: Any) -> Any:
        inputs = original(data, plan, slots, cumulative, *args, **kw)
        # The cumulative holds g G_k at a categorical node and G_k elsewhere.
        retention = cumulative / g[:, None] if slots is None else cumulative
        numerator = plan.cumulative_numerator(data)
        ratio = numerator / g[:, None] if slots is None else numerator
        kept = 1.0 / retention if part == "ratio" else ratio
        return replace(
            inputs,
            inverse=np.repeat(kept[:, :, None], inputs.n_levels, axis=2),
            loss_weight=inputs.weights[:, None] * kept,
        )

    return mutated


@pytest.mark.parametrize("reading", ["categorical", "continuous"])
@pytest.mark.parametrize("part", ["ratio", "retention"])
def test_dropping_a_factor_of_the_weight_misses(
    reading: str, part: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the ratio, or without ``1 / G_k``, the walk solves the wrong scores.

    Each control law makes the dropped factor one: the natural course for the ratio, and a
    law without dropout for the retention.  There the mutated fit stays exact, so the miss
    is the dropped factor's and not the mutation's.
    """
    law = _law(reading)
    control = _law(reading, censored=False) if part == "retention" else law
    kind_control = "up" if part == "retention" else "natural"
    from cleverly.longitudinal import estimator as estimator_module

    monkeypatch.setattr(estimator_module, "mtp_inputs", _drop(law, part))
    missed = fit(law, {"up": policy(reading, "up")}, reading)
    assert abs(risk(missed, "up", 2) - law.truth(UP, 2)) > MISSED
    monkeypatch.setattr(estimator_module, "mtp_inputs", _drop(control, part))
    kept = fit(control, {kind_control: policy(reading, kind_control)}, reading)
    truth = control.truth(BRANCHES[kind_control], 2)
    assert risk(kept, kind_control, 2) == pytest.approx(truth, abs=EXACT)


@pytest.mark.parametrize(
    ("reading", "ratio"),
    [
        ("categorical", "density"),
        ("continuous", "density"),
        ("continuous", "classifier"),
        ("mixed", "density"),
        ("mixed", "classifier"),
    ],
)
def test_the_weight_is_the_iterated_mean_clever_weight(reading: str, ratio: str) -> None:
    """``omega / w`` is the iterated-mean fit's clever covariate of the same plan, node by node.

    This ties the classifier ratio to the validated iterated-mean route as well as the
    density one.
    """
    law = _law(reading)
    with captured("mtp_inputs") as calls:
        fit(law, {"up": policy(reading, "up")}, reading, ratio=ratio)
    (inputs,) = calls
    iterated = fit(
        law,
        {"up": policy(reading, "up")},
        reading,
        ratio=ratio,
        survival_model="iterated_mean",
        targeting="iterative",
        pseudo_learner=CellMeans(),
    )
    weights = inputs.weights
    for step in iterated.fits[f"up @ t={law.n_times}"].steps:
        k = step.time - 1
        rows = step.trained_on & inputs.at_risk[:, k]
        assert np.count_nonzero(rows) > 0
        assert np.array_equal(step.trained_on, inputs.at_risk[:, k])
        np.testing.assert_allclose(
            inputs.inverse_observed()[rows, k], step.clever[rows], rtol=1e-12, atol=0.0
        )
        np.testing.assert_allclose(
            inputs.loss_weight[rows, k], weights[rows] * step.clever[rows], rtol=1e-12, atol=0.0
        )
    # The ratio is not one on this law, so the agreement is not a tautology of the weights.
    assert np.ptp(inputs.inverse_observed()[inputs.at_risk]) > 0.1


def test_the_loss_is_weighted_by_omega() -> None:
    """The walk loss's derivative along each covariate is minus ``n`` times its weighted score.

    The weighted submodel puts :math:`\\omega` in the loss, so
    :math:`d/d\\epsilon\\, L(\\eta + \\epsilon \\kappa) = -\\sum_i \\sum_k \\omega_{ik} \\kappa
    (dN - \\lambda)`.  The mutation weights the loss by the observation weight alone, and the
    identity fails by more than the derivative's error.
    """
    law = _law("continuous")
    with captured("mtp_inputs") as calls, captured("fit_hazards") as initial:
        fit(law, {"up": policy("continuous", "up")}, "continuous")
    (inputs,) = calls
    empty, full = inputs.held_cells()
    state = hazard_module.HazardState.from_hazards(
        initial[0].slots["up"], inputs.bounds, empty, full
    )
    horizons = (1, 2)

    def derivative(at: Any) -> np.ndarray:
        out = []
        for m in range(len(horizons)):
            unit = np.zeros((1, 1, len(horizons)))
            unit[0, 0, m] = 1.0
            move = hazard_module.direction(state, at, horizons, unit)
            eps = 1e-6
            ahead = hazard_module.walk_loss(state.moved(eps * move, at.bounds), at)
            behind = hazard_module.walk_loss(state.moved(-eps * move, at.bounds), at)
            out.append((ahead - behind) / (2.0 * eps))
        return np.array(out)

    score, _ = hazard_module.scores(state, inputs, horizons)
    expected = -inputs.n * score.reshape(-1)
    assert np.max(np.abs(expected)) > 1.0
    np.testing.assert_allclose(derivative(inputs), expected, rtol=1e-5)
    unweighted = replace(inputs, loss_weight=np.repeat(inputs.weights[:, None], 2, axis=1))
    gap = np.abs(derivative(unweighted) - expected)
    assert np.max(gap) > 1e-2 * np.max(np.abs(expected))


# ------------------------------------------------------------------ the slots

def test_each_slot_is_predicted_at_its_own_design() -> None:
    """A dose slot reads the shifted dose; a mixed slot splices the shifted components.

    At a mixed node :meth:`history_design` ignores an assignment, so the splice of
    :meth:`component_design` at the branch's values is what moves the slot.
    """
    law = _law("continuous")
    result = fit(law, {"up": policy("continuous", "up")}, "continuous")
    data = result.data
    (plan,) = result.repeats[0].replay_recipe.plans
    slots = hazard_module.mtp_slots(data, plan)
    assert slots is not None and slots.n_slots == 2
    shifted = np.minimum(law.frame()["A"].to_numpy() + 1.0, 2.0)
    np.testing.assert_array_equal(slots.values[1], shifted)
    for time in (1, 2):
        design = hazard_module.slot_design(data, time, slots, 1)
        expected = data.history_design(
            time, treatment=np.repeat(shifted[:, None], 2, axis=1), include_current=True
        )
        np.testing.assert_array_equal(design, expected)
        assert not np.array_equal(design, data.history_design(time, include_current=True))

    law = _law("mixed")
    result = fit(law, {"up": policy("mixed", "up")}, "mixed")
    data = result.data
    (plan,) = result.repeats[0].replay_recipe.plans
    slots = hazard_module.mtp_slots(data, plan)
    assert slots is not None
    frame = law.frame()
    branch = np.column_stack([frame["F"], np.minimum(frame["A"] + 1.0, 2.0)])
    np.testing.assert_array_equal(slots.values[1], branch)
    design = hazard_module.slot_design(data, 1, slots, 1)
    covariates = data.window_covariates(1)
    np.testing.assert_array_equal(
        design, np.hstack([covariates, data.component_design(1, values=branch)])
    )
    ignored = data.history_design(
        1, treatment=np.asarray(branch[:, 1:2], dtype=float), include_current=True
    )
    assert not np.array_equal(design, ignored)
    # The observed slot is the fit's own design.
    np.testing.assert_array_equal(
        hazard_module.slot_design(data, 1, slots, 0), data.history_design(1, include_current=True)
    )


def test_a_mixed_slot_at_the_observed_components_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Predicting the shifted slots at the observed components is the natural course's plug-in.

    The control is the natural joint policy, whose slots are the observed components, so the
    mutation changes nothing there and the fit stays exact.
    """
    law = _law("mixed")

    def observed(data: Any, time: int, slots: Any, slot: int) -> np.ndarray:
        return np.asarray(data.history_design(time, include_current=True))

    monkeypatch.setattr(hazard_module, "slot_design", observed)
    missed = fit(law, {"up": policy("mixed", "up")}, "mixed")
    assert abs(risk(missed, "up", 1) - law.truth(UP, 1)) > MISSED
    kept = fit(law, {"natural": policy("mixed", "natural")}, "mixed")
    assert risk(kept, "natural", 1) == pytest.approx(law.truth(NATURAL, 1), abs=EXACT)


def test_the_randomizer_mixes_its_branch_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plug-in weighs each branch slot by its probability.

    The mutation puts all the mass of the plug-in on the first branch and misses the
    randomizer's truth.
    """
    law = _law("continuous")
    original = hazard_module.mtp_slots

    def first_branch(data: Any, plan: Any) -> Any:
        slots = original(data, plan)
        probabilities = np.zeros_like(slots.probabilities)
        probabilities[:, 0] = 1.0
        return replace(slots, probabilities=probabilities)

    exact = fit(law, {"random": policy("continuous", "random")}, "continuous")
    assert risk(exact, "random", 2) == pytest.approx(law.truth(RANDOM, 2), abs=EXACT)
    from cleverly.longitudinal import estimator as estimator_module

    monkeypatch.setattr(estimator_module, "mtp_slots", first_branch)
    missed = fit(law, {"random": policy("continuous", "random")}, "continuous")
    assert abs(risk(missed, "random", 2) - law.truth(RANDOM, 2)) > MISSED


# ------------------------------------------------------------------ one walk per policy

def test_each_policy_has_its_own_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two policies and a static arm: two weighted walks and the shared walk.

    Each estimate is the one its regimen gets alone, and each is exact.  The mutation gives
    the second policy the first policy's weight, as one walk over both would, and misses.
    """
    law = _law("categorical")
    regimens = {"up": policy("categorical", "up"), "random": policy("categorical", "random")}
    together = fit(law, {**regimens, "a0": 0}, "categorical")
    records = {label: together.fits[f"{label} @ t=2"].hazard for label in ("up", "random", "a0")}
    assert len({id(record) for record in records.values()}) == 3
    for label, record in records.items():
        assert {target[0] for target in record.targets} == {label}
    for label, branches in (("up", UP), ("random", RANDOM)):
        alone = fit(law, {label: regimens[label]}, "categorical")
        for t in (1, 2):
            assert risk(together, label, t) == risk(alone, label, t)
            assert risk(together, label, t) == pytest.approx(law.truth(branches, t), abs=EXACT)
    static = ((1.0, lambda a: 0),)
    assert risk(together, "a0", 2) == pytest.approx(law.truth(static, 2), abs=EXACT)

    original = hazard_module.mtp_inputs
    first: list[Any] = []

    def shared(*args: Any, **kwargs: Any) -> Any:
        inputs = original(*args, **kwargs)
        if not first:
            first.append(inputs)
            return inputs
        return replace(inputs, inverse=first[0].inverse, loss_weight=first[0].loss_weight)

    from cleverly.longitudinal import estimator as estimator_module

    monkeypatch.setattr(estimator_module, "mtp_inputs", shared)
    mutated = fit(law, regimens, "categorical")
    assert abs(risk(mutated, "random", 2) - law.truth(RANDOM, 2)) > MISSED


# ------------------------------------------------------------------ held cells

def _final_states(result_calls: list[Any]) -> list[Any]:
    return [state for reports in result_calls for state, _ in reports.values()]


@pytest.mark.parametrize("targeting", ["one_step", "iterative"])
def test_a_held_level_cell_holds_every_unit_at_that_level(targeting: str) -> None:
    """A level with no death at node 1: every unit's hazard there stays exactly zero."""
    law = GridLaw(causes=2, empty=lambda node, w, f, a: node == 1 and a == 2)
    with captured("_target") as calls, captured("mtp_inputs") as inputs:
        result = fit(law, {"up": policy("categorical", "up")}, "categorical", targeting=targeting)
    assert inputs[0].empty[0, 2, 1] and not inputs[0].empty[0, 1, 1]
    for state in _final_states(calls):
        assert np.all(state.hazards[:, 0, 2, 1] == 0.0)
        assert np.all(state.hazards[:, 0, 1, 1] > 0.0)
    for cause, code in (("relapse", 1), ("death", 2)):
        for t in (1, 2):
            truth = law.truth(UP, t, code)
            assert risk(result, "up", t, cause) == pytest.approx(truth, abs=EXACT)


def test_a_held_dose_cell_holds_every_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """No death at node 1 at any dose: every slot copy there is held, and the walk converges.

    Without the hold the clipped logit leaves a death score the walk cannot solve: on this
    law its relative score settles near ``2.8e-4`` by step 1,000 and stays there, where the
    held walk converges in under 200 steps.
    """
    law = GridLaw(causes=2, empty=lambda node, w, f, a: node == 1)
    with captured("_target") as calls:
        result = fit(law, {"up": policy("continuous", "up")}, "continuous")
    for state in _final_states(calls):
        assert np.all(state.hazards[:, 0, :, 1] == 0.0)
    record = result.fits["up, death @ t=2"].hazard
    assert record.converged and record.n_iter < 200
    for cause, code in (("relapse", 1), ("death", 2)):
        truth = law.truth(UP, 2, code)
        assert risk(result, "up", 2, cause) == pytest.approx(truth, abs=EXACT)

    original = hazard_module._held_cells

    def unheld(*args: Any, **kwargs: Any) -> Any:
        empty, full, excluded = original(*args, **kwargs)
        return np.zeros_like(empty), full, excluded

    walk = hazard_module.walk_curve

    def capped(*args: Any, **kwargs: Any) -> Any:
        return walk(*args, **{**kwargs, "max_steps": 1_500})

    monkeypatch.setattr(hazard_module, "_held_cells", unheld)
    monkeypatch.setattr(hazard_module, "walk_curve", capped)
    mutated = fit(law, {"up": policy("continuous", "up")}, "continuous")
    record = mutated.fits["up, death @ t=2"].hazard
    assert not record.converged
    assert record.n_iter == 1_500
    assert record.relative_score > 1e-4


def test_a_held_mixed_cell_holds_the_slots_of_its_categorical_values() -> None:
    """No death at node 1 with ``F = 1``: those slots are held, the ``F = 0`` slots are not."""
    law = GridLaw(flag=True, n_times=1, causes=2, empty=lambda node, w, f, a: f == 1)
    with captured("_target") as calls:
        result = fit(law, {"up": policy("mixed", "up")}, "mixed")
    flagged = law.frame()["F"].to_numpy() == 1.0
    for state in _final_states(calls):
        assert np.all(state.hazards[flagged, 0, :, 1] == 0.0)
        assert np.all(state.hazards[~flagged, 0, :, 1] > 0.0)
    for cause, code in (("relapse", 1), ("death", 2)):
        truth = law.truth(UP, 1, code)
        assert risk(result, "up", 1, cause) == pytest.approx(truth, abs=EXACT)


# ------------------------------------------------------------------ cross-fitting

def test_the_slot_hazards_are_cross_fitted() -> None:
    """Each held-out row's shifted-slot hazard is the training folds' fit at its shifted dose.

    The in-sample fit predicts something else at the same rows, so the check can tell the
    two apart, and the weighted walk still solves its score.
    """
    law = _law("continuous")
    with captured("fit_hazards") as calls:
        result = fit(law, {"up": policy("continuous", "up")}, "continuous", n_folds=3)
    assert result.fits["up @ t=2"].hazard.converged
    data = result.data
    (plan,) = result.repeats[0].replay_recipe.plans
    slots = hazard_module.mtp_slots(data, plan)
    assert slots is not None
    at_risk, events = hazard_module.event_indicators(data)
    design = data.history_design(1, include_current=True)
    shifted = hazard_module.slot_design(data, 1, slots, 1)
    assignment = result.folds.assignment
    expected = np.empty(data.n)
    for fold in np.unique(assignment):
        train = at_risk[:, 0] & (assignment != fold)
        learner = Wrong().fit(design[train], events[train, 0, 0])
        held = assignment == fold
        expected[held] = learner.predict_proba(shifted[held])[:, 1]
    got = calls[0].slots["up"][:, 0, 1, 0]
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-7)
    in_sample = Wrong().fit(design[at_risk[:, 0]], events[at_risk[:, 0], 0, 0])
    assert np.max(np.abs(in_sample.predict_proba(shifted)[:, 1] - got)) > 1e-4


# ------------------------------------------------------------------ monotone curves

@pytest.mark.parametrize("reading", ["categorical", "continuous"])
def test_each_policy_walk_reports_a_monotone_curve(reading: str) -> None:
    """On a sample that is not the law, each cause's curve is non-decreasing in floating point."""
    law = GridLaw(n_times=3, causes=2)
    frame = law.frame()
    rows = np.random.default_rng(4).choice(len(frame), size=3_000, replace=False)
    with warnings.catch_warnings(), exact_bins():
        warnings.simplefilter("ignore")
        result = LTMLE(
            {"up": policy(reading, "up"), "random": policy(reading, "random")},
            n_folds=1,
            survival_model="hazard",
            targeting="one_step",
            step_size=STEP,
            # Monotonicity holds at every step of the walk, so a loose tolerance is enough.
            tol=1e-4,
            outcome_learner=Wrong(),
            treatment_learner=CellProbabilities(),
            censoring_learner=CellMeans(),
        ).fit(frame.iloc[rows].reset_index(drop=True), **law.columns(reading))
    for label in ("up", "random"):
        for cause in ("relapse", "death"):
            curve = [risk(result, label, t, cause) for t in (1, 2, 3)]
            assert all(later >= earlier for earlier, later in zip(curve, curve[1:], strict=False))
