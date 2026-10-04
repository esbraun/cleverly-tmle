"""Point-treatment modified treatment policies against numerically differentiated ones.

Every policy class beyond the additive shift is evaluated by ``src/`` -- Equation (3) of
Díaz, Williams, Hoffman and Schenck (2023) on a binned density, or the discrete formula on
a categorical treatment -- and its mean and influence curve are compared with the longhand
law of ``tests/discrete_law_policy_point.py``, which differentiates the parameter by a
complex step.  As in ``tests/unit/test_influence_gateaux_shift.py`` the nuisances are the
true ones and the targeting step is not run, so the test is about the clever covariate and
the plug-in rather than about the Newton solver.

The mutation controls monkeypatch one component at a time -- the inverse, the Jacobian,
the absolute value of a decreasing piece, a covariate-dependent boundary, a piece's
membership indicator, the randomizer integration -- and assert that the Gateaux check
then fails by more than ``1e-4``.  Each mutation runs on a law where its component is
nonzero, which is the CLAUDE.md rule for terms that vanish at the truth.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import tests.discrete_law_policy_point as law
from cleverly.data import CausalData
from cleverly.exceptions import CapabilityError, DataError
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import submodel_for
from cleverly.inference.influence import policy_means
from cleverly.interventions import (
    ModifiedPolicy,
    Piece,
    Piecewise,
    PolicySet,
    Randomizer,
    RiskRatioTilt,
    Scale,
    Shift,
)
from cleverly.interventions import policy as policy_module
from cleverly.learners.crossfit import Folds
from cleverly.learners.density import ConditionalDensity
from tests.discrete_law_longitudinal import CellMeans

INF = np.inf


def _w(h: Any) -> np.ndarray:
    return np.asarray(h["W"], dtype=float)


#: The estimator-side declaration of every policy of the law, by grid and label.
DECLARED: dict[str, dict[str, Any]] = {
    "unit": {
        "natural course": Shift(0.0, cap=None),
        "decreasing": ModifiedPolicy(
            "decreasing",
            pieces=(
                Piece(
                    -INF,
                    INF,
                    lambda a, h: 3.0 - a,
                    lambda b, h: 3.0 - b,
                    lambda b, h: -1.0,
                ),
            ),
            policy_kind="known",
        ),
        "boundary": ModifiedPolicy(
            "boundary",
            pieces=(
                Piece(
                    -INF,
                    lambda h: 1.5 + 0.5 * _w(h),
                    lambda a, h: a + 1.0,
                    lambda b, h: b - 1.0,
                    lambda b, h: 1.0,
                ),
                Piece(lambda h: 1.5 + 0.5 * _w(h), INF),
            ),
            policy_kind="known",
        ),
        "randomized": ModifiedPolicy(
            "randomized",
            pieces={
                "down": (
                    Piece(-INF, 2.5),
                    Piece(2.5, INF, lambda a, h: a - 1.0, lambda b, h: b + 1.0, lambda b, h: 1.0),
                ),
                "stay": (Piece(-INF, INF),),
            },
            randomizer=Randomizer(("down", "stay"), (0.25, 0.75)),
            policy_kind="known",
        ),
    },
    "multiplicative": {
        "natural course": Scale(1.0, cap=None),
        "x2": Scale(2.0, cap=8.0, name="x2"),
    },
    "piecewise": {
        "natural course": Shift(0.0, cap=None),
        "piecewise": Piecewise(
            (
                (0.0, 1.5, Scale(2.0, None)),
                (1.5, 7.0, Shift(2.0, None)),
                (7.0, INF, Shift(0.0, None)),
            ),
            name="piecewise",
        ),
    },
    "binary": {
        "natural course": RiskRatioTilt(1.0),
        "rr 0.5": RiskRatioTilt(0.5),
        "rr 4": RiskRatioTilt(4.0, name="rr 4"),
        "treat W=1": ModifiedPolicy(
            "treat W=1",
            apply=lambda a, h: np.where(_w(h) == 1.0, 1.0, np.asarray(a, dtype=float)),
            policy_kind="known",
        ),
    },
    "three level": {
        "natural course": Shift(0.0, cap=None),
        "drop above 2": ModifiedPolicy(
            "drop above 2",
            apply=lambda a, h: np.where(np.asarray(a, dtype=float) > 2.0, a - 1.0, a),
            policy_kind="known",
        ),
    },
}

DISCRETE = {"binary", "three level"}


def _data(grid_name: str) -> tuple[CausalData, np.ndarray, np.ndarray, np.ndarray]:
    grid = law.GRIDS[grid_name]
    frame = grid.frame()
    covariate = frame["W"].to_numpy().astype(int)
    dose = frame["A"].to_numpy(dtype=float)
    outcome = frame["Y"].to_numpy(dtype=float)
    kind = "discrete" if grid_name in DISCRETE else "continuous"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_arrays(
            outcome,
            dose,
            covariate.reshape(-1, 1).astype(float),
            covariate_names=["W"],
            treatment_kind=kind,
        )
    return data, covariate, dose, outcome


def _support_index(grid: law.Grid, values: np.ndarray, discrete: bool) -> np.ndarray:
    if discrete:
        return np.asarray(values, dtype=np.int64)
    doses = np.asarray(grid.doses)
    index = np.searchsorted(doses, values)
    assert np.all(doses[index] == values), "a policy left the law's support"
    return index


def _evaluate(grid_name: str, labels: tuple[str, ...]) -> PolicySet:
    data, covariate, _, _ = _data(grid_name)
    grid = law.GRIDS[grid_name]
    policies = tuple(DECLARED[grid_name][label] for label in labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if grid_name in DISCRETE:
            return PolicySet.evaluate(policies, data, propensity=grid.g[covariate])
        density = ConditionalDensity(grid.bins(grid.g[covariate]), grid.edges)
        return PolicySet.evaluate(policies, data, density)


def _means(grid_name: str, labels: tuple[str, ...], policy_set: PolicySet | None = None) -> Any:
    grid = law.GRIDS[grid_name]
    discrete = grid_name in DISCRETE
    data, covariate, dose, outcome = _data(grid_name)
    policy_set = _evaluate(grid_name, labels) if policy_set is None else policy_set
    observed_index = _support_index(grid, dose if not discrete else data.treatment, discrete)
    initial = InitialFit(
        grid.q[covariate, observed_index],
        {
            float(c): grid.q[covariate, _support_index(grid, policy_set.shifted[:, c], discrete)]
            for c in range(policy_set.n_components)
        },
    )
    propensity = grid.g[covariate] if discrete else np.zeros((dose.size, 0))
    submodel = submodel_for(
        "mtp",
        np.asarray(data.treatment, dtype=float),
        propensity,
        arms=data.arm_codes if discrete else (),
        policies=policy_set.design_at(propensity if discrete else None),
    )
    return policy_means(outcome, initial, submodel, np.ones(dose.size), mixing=policy_set.mixing)


def _name(grid_name: str) -> str:
    return "ey_rr_tilt" if grid_name == "binary" else "ey_policy"


CASES = [
    (grid_name, label)
    for grid_name, policies in law.POLICIES.items()
    for label in policies
    if not (grid_name == "binary" and label == "treat W=1")
] + [("binary", "treat W=1")]


@pytest.mark.parametrize(("grid_name", "label"), CASES)
def test_the_mean_is_the_law(grid_name: str, label: str) -> None:
    labels = ("natural course", label) if label != "natural course" else (label,)
    means = _means(grid_name, labels)
    code = float(labels.index(label))
    expected = law.truth(grid_name, f"{_name(grid_name)}[{label}]")
    assert means[code].psi == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(("grid_name", "label"), CASES)
def test_the_curve_is_the_gateaux_derivative(grid_name: str, label: str) -> None:
    labels = ("natural course", label) if label != "natural course" else (label,)
    means = _means(grid_name, labels)
    code = float(labels.index(label))
    reported = np.asarray(means[code].influence_curve)[law.GRIDS[grid_name].first_row_of()]
    expected = law.eif(grid_name, f"{_name(grid_name)}[{label}]")
    np.testing.assert_allclose(reported, expected, atol=1e-10, rtol=0)


def test_the_policies_differ_from_the_natural_course_and_from_each_other() -> None:
    """Nonzero witnesses: every policy moves its mean by more than 0.01."""
    for grid_name, policies in law.POLICIES.items():
        name = _name(grid_name)
        base = law.truth(grid_name, f"{name}[natural course]")
        for label in policies:
            if label == "natural course":
                continue
            assert abs(law.truth(grid_name, f"{name}[{label}]") - base) > 0.01, (grid_name, label)


def test_the_ratio_differs_from_one_on_some_cell_of_every_policy() -> None:
    for grid_name, policies in law.POLICIES.items():
        for label in policies:
            if label == "natural course":
                continue
            ratios = _evaluate(grid_name, (label,)).policy_ratio[:, 0]
            # A randomized policy moves one unit in four, so its ratio moves less.
            floor = 0.3 if label == "randomized" else 0.5
            assert np.max(np.abs(ratios - 1.0)) > floor, (grid_name, label)


# ------------------------------------------------------------------ mutations


def _gateaux_gap(grid_name: str, label: str) -> float:
    labels = ("natural course", label)
    means = _means(grid_name, labels)
    reported = np.asarray(means[1.0].influence_curve)[law.GRIDS[grid_name].first_row_of()]
    expected = law.eif(grid_name, f"{_name(grid_name)}[{label}]")
    return float(np.max(np.abs(reported - expected)))


def _mutated_piece_ratio(*, inverse: bool = True, absolute: bool = True) -> Any:
    """``_PieceBranch.ratio`` with the inverse or the absolute value of ``b'`` removed."""

    def ratio(self: Any, values: Any, frame: Any, density: Any) -> Any:
        b = np.asarray(values, dtype=float).reshape(-1)
        denominator = density(b)
        numerator = np.zeros(b.size)
        for piece in self.pieces:
            if piece.map is None:
                continue
            source = policy_module._per_row(policy_module._call(piece.inverse, b, frame()), b.size)
            if not inverse:
                source = b
            slope = policy_module._per_row(
                policy_module._call(piece.derivative, b, frame()), b.size
            )
            inside = policy_module._member(source, *self.bounds(piece, frame, b.size), piece.closed)
            jacobian = np.abs(slope) if absolute else slope
            numerator = numerator + np.where(
                inside, density(np.where(inside, source, b)) * jacobian, 0.0
            )
        safe = np.where(denominator > 0.0, denominator, 1.0)
        covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
        for piece in self.pieces:
            if piece.map is None:
                covariate = covariate + policy_module._member(
                    b, *self.bounds(piece, frame, b.size), piece.closed
                )
        return covariate

    return ratio


class TestMutations:
    """Each mutation breaks one component and the Gateaux check must see it."""

    def test_m1_dropping_the_inverse_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _gateaux_gap("unit", "decreasing") < 1e-10
        monkeypatch.setattr(
            policy_module._PieceBranch, "ratio", _mutated_piece_ratio(inverse=False)
        )
        assert _gateaux_gap("unit", "decreasing") > 1e-4

    def test_m2_dropping_the_jacobian_of_a_scale_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_jacobian(self: Any, values: Any, frame: Any, density: Any) -> Any:
            a = np.asarray(values, dtype=float)
            factor = float(self.scale.factor)
            numerator = density(a / factor)
            denominator = density(a)
            safe = np.where(denominator > 0.0, denominator, 1.0)
            covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
            if self.scale.cap is None:
                return covariate
            cap = float(self.scale.cap)
            return covariate * (a <= cap) + (a * factor > cap)

        assert _gateaux_gap("multiplicative", "x2") < 1e-10
        monkeypatch.setattr(policy_module._ScaleBranch, "ratio", no_jacobian)
        assert _gateaux_gap("multiplicative", "x2") > 1e-4

    def test_m2_dropping_the_jacobian_of_a_piecewise_scale_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = policy_module._PiecewiseBranch.ratio

        def no_jacobian(self: Any, values: Any, frame: Any, density: Any) -> Any:
            mutated = Piecewise(
                tuple(
                    (low, high, Shift(0.0, None) if isinstance(m, Scale) and m.factor == 1.0 else m)
                    for low, high, m in self.policy.pieces
                ),
                name=self.policy.name,
            )
            twin = policy_module._PiecewiseBranch(mutated)
            b = np.asarray(values, dtype=float)
            # Equation (3) with every scaled piece's Jacobian set to one.
            out = original(twin, b, frame, density)
            for low, high, m in self.policy.pieces:
                if isinstance(m, Scale) and m.factor != 1.0:
                    source = b / m.factor
                    inside = policy_module._member(source, low, high, self.policy.closed)
                    denominator = density(b)
                    safe = np.where(denominator > 0.0, denominator, 1.0)
                    extra = np.where(inside, density(source) * (1.0 - 1.0 / m.factor), 0.0)
                    out = out + np.where(denominator > 0.0, extra / safe, 0.0)
            return out

        assert _gateaux_gap("piecewise", "piecewise") < 1e-10
        monkeypatch.setattr(policy_module._PiecewiseBranch, "ratio", no_jacobian)
        assert _gateaux_gap("piecewise", "piecewise") > 1e-4

    def test_m3_dropping_a_piece_indicator_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_indicator(self: Any, values: Any, frame: Any, density: Any) -> Any:
            # Equation (3) with the scale piece's membership indicator dropped.
            b = np.asarray(values, dtype=float)
            denominator = density(b)
            numerator = np.zeros(b.size)
            for low, high, m in self.policy.pieces:
                if policy_module._is_identity_map(m):
                    continue
                if isinstance(m, Shift):
                    source, jacobian = b - m.delta, 1.0
                    inside = policy_module._member(source, low, high, self.policy.closed)
                else:
                    # The scale piece forgets its interval: 8 / 2 = 4 lies outside [0, 1.5)
                    # and is counted anyway.
                    source, jacobian = b / m.factor, 1.0 / m.factor
                    inside = np.isfinite(source)
                numerator = numerator + np.where(inside, density(source) * jacobian, 0.0)
            safe = np.where(denominator > 0.0, denominator, 1.0)
            covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
            for low, high, m in self.policy.pieces:
                if policy_module._is_identity_map(m):
                    covariate = covariate + policy_module._member(b, low, high, self.policy.closed)
            return covariate

        assert _gateaux_gap("piecewise", "piecewise") < 1e-10
        monkeypatch.setattr(policy_module._PiecewiseBranch, "ratio", no_indicator)
        assert _gateaux_gap("piecewise", "piecewise") > 1e-4

    def test_m12_dropping_the_absolute_value_of_a_decreasing_piece_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _gateaux_gap("unit", "decreasing") < 1e-10
        monkeypatch.setattr(
            policy_module._PieceBranch, "ratio", _mutated_piece_ratio(absolute=False)
        )
        assert _gateaux_gap("unit", "decreasing") > 1e-4

    def test_m13_a_boundary_read_at_one_covariate_value_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = policy_module._PieceBranch.bounds

        def fixed_h(self: Any, piece: Any, frame: Any, n: int) -> Any:
            lower, upper = original(self, piece, frame, n)
            if callable(piece.upper):
                upper = np.full(n, upper[0])
            if callable(piece.lower):
                lower = np.full(n, lower[0])
            return lower, upper

        assert _gateaux_gap("unit", "boundary") < 1e-10
        monkeypatch.setattr(policy_module._PieceBranch, "bounds", fixed_h)
        assert _gateaux_gap("unit", "boundary") > 1e-4

    def test_m8_the_first_randomizer_value_alone_fails(self) -> None:
        policy_set = _evaluate("unit", ("natural course", "randomized"))
        assert policy_set.mixing is not None
        first_only = policy_set.mixing.copy()
        first_only[1] = 0.0
        first_only[1, 1] = 1.0
        from dataclasses import replace

        mutated = replace(policy_set, mixing=first_only)
        means = _means("unit", ("natural course", "randomized"), mutated)
        reported = np.asarray(means[1.0].influence_curve)[law.UNIT.first_row_of()]
        expected = law.eif("unit", "ey_policy[randomized]")
        assert float(np.max(np.abs(reported - expected))) > 1e-4


# ------------------------------------------------------------------ shipped Shift arithmetic


def _shipped_ratio(density: ConditionalDensity, values: np.ndarray, delta: float, cap: Any) -> Any:
    """The f71548ce ``_ratio`` of ``interventions/shift.py``, verbatim."""
    a = np.asarray(values, dtype=float).reshape(-1)
    numerator = density.density_at(a - delta)
    denominator = density.density_at(a)
    safe = np.where(denominator > 0.0, denominator, 1.0)
    covariate = np.where(denominator > 0.0, numerator / safe, 0.0)
    if cap is not None:
        cap = float(cap)
        covariate = covariate * (a <= cap).astype(float)
        covariate = covariate + (a > cap - delta).astype(float)
    return np.asarray(covariate, dtype=float)


def _shipped_apply(values: np.ndarray, delta: float, cap: Any) -> Any:
    """The f71548ce ``Shift.apply``, verbatim."""
    a = np.asarray(values, dtype=float).reshape(-1)
    moved = np.asarray(a + delta, dtype=float)
    if cap is None:
        return moved
    held = np.asarray(moved > float(cap), dtype=bool)
    return np.asarray(np.where(held, a, moved), dtype=float)


@pytest.mark.parametrize("cap", [None, 3.0, 2.0])
def test_t1c_the_shift_arrays_are_the_shipped_ones(cap: Any) -> None:
    """Bit identity of the ``Shift`` path, with a float-edge row and a zero-density row.

    Row 0 sits where ``a + delta > cap`` and ``a > cap - delta`` disagree by one rounding:
    ``0.1 + 0.6 > 0.7`` fails and ``0.1 > 0.7 - 0.6`` holds.  Row 1's observed dose has
    zero estimated density, so its covariate is the identity term alone.
    """
    delta = 0.6
    edge_cap = 0.7 if cap is not None else None
    rng = np.random.default_rng(3)
    treatment = np.concatenate([[0.1, 1.5], rng.uniform(0.0, 3.0, 60)])
    edges = np.linspace(-0.5, 3.5, 9)
    probabilities = rng.dirichlet(np.ones(8), size=treatment.size)
    probabilities[1, np.digitize(1.5, edges) - 1] = 0.0
    probabilities[1] /= probabilities[1].sum()
    density = ConditionalDensity(probabilities, edges)
    shifts = (Shift(0.0, cap=None), Shift(delta, cap=edge_cap, name="edge"), Shift(1.0, cap=cap))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_arrays(
            np.arange(treatment.size) % 2,
            treatment,
            np.arange(treatment.size, dtype=float)[:, None],
            treatment_kind="continuous",
        )
        evaluated = PolicySet.evaluate(shifts, data, density)
    if edge_cap is not None:
        assert (0.1 + delta > edge_cap) != (edge_cap - delta < 0.1)
    assert density.density_at(treatment)[1] == 0.0
    shifted = np.column_stack([_shipped_apply(treatment, s.delta, s.cap) for s in shifts])
    ratio = np.column_stack([_shipped_ratio(density, treatment, s.delta, s.cap) for s in shifts])
    ratio_at = np.stack(
        [
            np.column_stack(
                [_shipped_ratio(density, shifted[:, j], s.delta, s.cap) for s in shifts]
            )
            for j in range(len(shifts))
        ],
        axis=1,
    )
    assert np.array_equal(evaluated.shifted, shifted)
    assert np.array_equal(evaluated.ratio, ratio)
    assert np.array_equal(evaluated.ratio_at, ratio_at)
    assert evaluated.mixing is None


# ------------------------------------------------------------------ classifier route


def test_t4_a_saturated_classifier_returns_the_ratio_and_agrees_with_the_density() -> None:
    """Section 5.4 on the exact law: odds of the stacked label are g^d / g at every cell."""
    data, _, _, _ = _data("multiplicative")
    policies = tuple(DECLARED["multiplicative"][label] for label in ("natural course", "x2"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_classifier = PolicySet.evaluate_by_classifier(
            policies, data, CellMeans(), Folds.single(data.n)
        )
    by_density = _evaluate("multiplicative", ("natural course", "x2"))
    np.testing.assert_allclose(by_classifier.ratio, by_density.ratio, atol=1e-12, rtol=0)
    np.testing.assert_allclose(by_classifier.ratio_at, by_density.ratio_at, atol=1e-12, rtol=0)
    assert np.max(np.abs(by_density.ratio[:, 1] - 1.0)) > 0.5


def test_t4_the_classifier_folds_split_units_not_stacked_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit's two copies train together, and a held-out unit trains in neither copy."""
    from cleverly.learners import density_ratio

    data, _, dose, _ = _data("unit")
    from cleverly.learners.crossfit import make_folds

    folds = make_folds(data.n, 3, random_state=1)
    seen: list[np.ndarray] = []
    original = density_ratio.fit_learner

    def record(learner: Any, design: Any, target: Any, weight: Any, **kwargs: Any) -> Any:
        seen.append(np.asarray(kwargs["groups"]))
        return original(learner, design, target, weight, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", record)
    density_ratio.classifier_ratio(
        CellMeans(),
        data.covariates,
        dose,
        ((1.0, dose + 0.0),),
        data.weights,
        folds,
        evaluate_at=[dose],
    )
    assert len(seen) == 3
    for groups, (train, test) in zip(seen, folds, strict=True):
        units, counts = np.unique(groups, return_counts=True)
        assert set(units.tolist()) == set(np.asarray(train).tolist())
        assert np.all(counts == 2)
        assert not set(units.tolist()) & set(np.asarray(test).tolist())


# ------------------------------------------------------------------ remainder


def test_t5_the_remainder_is_second_order_with_a_nonzero_coefficient() -> None:
    """Psi(P_eps) - Psi(P) - eps P D is O(eps^2), and the eps^2 coefficient is not zero."""
    grid_name, label = "multiplicative", "x2"
    means = _means(grid_name, ("natural course", label))
    curve = np.asarray(means[1.0].influence_curve)[law.MULTIPLICATIVE.first_row_of()]
    probs = law.MULTIPLICATIVE.probs
    direction = np.zeros_like(probs)
    direction[0, 1, 1] = 1.0
    direction[2, 3, 0] = 1.0
    direction = direction / direction.sum() - probs
    pd = float(np.sum(direction.reshape(-1) * _per_cell(curve, law.MULTIPLICATIVE)))
    base = float(np.real(law.functional(probs, grid_name, f"ey_policy[{label}]")))
    coefficients = []
    for eps in (1e-2, 5e-3, 2.5e-3):
        moved = float(
            np.real(law.functional(probs + eps * direction, grid_name, f"ey_policy[{label}]"))
        )
        coefficients.append((moved - base - eps * pd) / eps**2)
    assert abs(coefficients[0]) > 1e-4
    assert abs(coefficients[-1] - coefficients[-2]) < 0.05 * abs(coefficients[-1])


def _per_cell(curve: np.ndarray, grid: law.Grid) -> np.ndarray:
    """The curve at each support point, laid out as the ``(3, K, 2)`` cell array."""
    out = np.zeros((3, len(grid.doses), 2))
    for value, (w, k, y) in zip(curve, grid.support, strict=True):
        out[w, k, y] = value
    return out.reshape(-1)


# ------------------------------------------------------------------ refusals


class TestRefusalsBeforeAnyLearner:
    """Each refusal of a declared point policy, with its exact text."""

    def test_r2_pieces_that_do_not_partition_the_doses(self) -> None:
        data, covariate, _, _ = _data("unit")
        gap = ModifiedPolicy(
            "gap",
            pieces=(
                Piece(-INF, 1.5, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
                Piece(2.5, INF),
            ),
            policy_kind="known",
        )
        density = ConditionalDensity(law.UNIT.bins(law.UNIT.g[covariate]), law.UNIT.edges)
        with pytest.raises(
            DataError, match=r"does not assign .* observed doses to exactly one piece"
        ):
            PolicySet.evaluate((gap,), data, density)

    def test_r3_a_threshold_piece_on_a_continuous_dose(self) -> None:
        data, covariate, _, _ = _data("unit")
        floor = ModifiedPolicy(
            "floor",
            pieces=(
                Piece(-INF, INF, lambda a, h: np.minimum(a, 1.0), lambda b, h: b, lambda b, h: 1.0),
            ),
            policy_kind="known",
        )
        density = ConditionalDensity(law.UNIT.bins(law.UNIT.g[covariate]), law.UNIT.edges)
        with pytest.raises(CapabilityError, match="not pathwise differentiable"):
            PolicySet.evaluate((floor,), data, density)

    def test_r3_apply_without_pieces_on_a_continuous_dose(self) -> None:
        data, covariate, _, _ = _data("unit")
        undeclared = ModifiedPolicy("apply", apply=lambda a, h: a + 1.0, policy_kind="known")
        density = ConditionalDensity(law.UNIT.bins(law.UNIT.g[covariate]), law.UNIT.edges)
        with pytest.raises(CapabilityError, match="declares apply= on a continuous treatment"):
            PolicySet.evaluate((undeclared,), data, density)

    def test_r4_an_inverse_that_does_not_invert(self) -> None:
        data, covariate, _, _ = _data("unit")
        wrong = ModifiedPolicy(
            "wrong",
            pieces=(
                Piece(-INF, INF, lambda a, h: a + 1.0, lambda b, h: b - 0.5, lambda b, h: 1.0),
            ),
            policy_kind="known",
        )
        density = ConditionalDensity(law.UNIT.bins(law.UNIT.g[covariate]), law.UNIT.edges)
        with pytest.raises(DataError, match=r"inverse\(apply\(a\)\) differs from a"):
            PolicySet.evaluate((wrong,), data, density)

    @pytest.mark.parametrize(
        ("kind", "match"),
        [(None, "inspect a closure"), ("estimated", "declared as estimated from the sample")],
    )
    def test_r5_r6_the_declaration(self, kind: Any, match: str) -> None:
        from cleverly.estimators import TMLE

        policy = ModifiedPolicy(
            "p",
            pieces=(
                Piece(-INF, INF, lambda a, h: a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
            ),
            policy_kind=kind,
        )
        with pytest.raises(CapabilityError, match=match):
            TMLE(policies=[policy])

    def test_r9_a_continuous_randomizer(self) -> None:
        with pytest.raises(CapabilityError, match="two-point randomizer and is accepted"):
            ModifiedPolicy("u", apply=lambda a, h, e: a, randomizer=object(), policy_kind="known")  # type: ignore[arg-type]

    def test_r11_a_map_to_a_value_that_is_not_a_level(self) -> None:
        data, covariate, _, _ = _data("three level")
        with pytest.raises(DataError, match="that are not levels of the node"):
            PolicySet.evaluate(
                (Shift(1.0, cap=None),), data, propensity=law.THREE_LEVEL.g[covariate]
            )

    def test_r11_an_arithmetic_policy_on_text_labels(self) -> None:
        grid = law.BINARY
        frame = grid.frame(lambda k: ("no", "yes")[k])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"])
        covariate = frame["W"].to_numpy().astype(int)
        with pytest.raises(DataError, match="adds, scales or splits a dose"):
            PolicySet.evaluate((Shift(1.0, cap=None),), data, propensity=grid.g[covariate])

    def test_a_risk_ratio_tilt_beside_another_policy(self) -> None:
        from cleverly.estimators import TMLE

        with pytest.raises(CapabilityError, match="reports its own estimands"):
            TMLE(policies=[RiskRatioTilt(1.0), Shift(0.0, cap=None)])
