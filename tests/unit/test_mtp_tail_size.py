"""The tail sizes of the boundaries law, the change-of-measure identity, and its exact ratio.

``tests/studies/mtp_tail_size_common.py`` integrates the moments of the boundaries law's
efficient influence function by quadrature, and the diagonal form of the policy-weighted
variance with the true regressions.  The pins below are the order-40 values; the order-80
check and the whole table are in ``tests/diagnostics/fix5_heavy_ratios/run.py`` and its
``reading.csv`` (``tests/unit/test_heavy_ratio_reading.py``).

The law-B ``OracleRatio`` of ``tests/studies/mtp_boundaries_common.py`` gives the exact ratio
on the classifier route.  Its stacked column order is pinned here against ``cleverly``'s
classifier route with a spy, and its ratio against the closed form.
"""

from __future__ import annotations

import warnings
from collections.abc import Collection
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from tests.studies import canonical_mtp_boundaries as study
from tests.studies import canonical_mtp_pricing as pricing
from tests.studies import mtp_boundaries_common as law
from tests.studies import mtp_boundaries_properties as boundaries_properties
from tests.studies import mtp_pricing_common as law_p
from tests.studies import mtp_pricing_properties as pricing_properties
from tests.studies import mtp_tail_size_common as tail
from tests.studies.longitudinal_sdr_properties import WALD_GATED_STEP, WALD_REGIME_FACTOR

#: The order-40 quadrature of each plan's tail size, read off ``law_b_moments``.
PINNED_KAPPA = {("p4", "ate"): 3260.86, ("p4", "ey"): 2764.97}


# ------------------------------------------------------------------------ the tail rule


@pytest.mark.parametrize(
    ("kappa", "n_kappa"), [(1.0, 0), (1.0625, 1), (5.0, 64), (3260.859, 52158)]
)
def test_the_tail_size_is_the_ceiling_of_the_rule(kappa: float, n_kappa: int) -> None:
    assert tail.tail_size(kappa) == n_kappa


@pytest.mark.parametrize("n_kappa", [1, 199, 200, 201, 52158, 44224])
def test_the_gated_size_is_the_least_step_at_or_above_twice_the_tail_size(n_kappa: int) -> None:
    gated = tail.gated_size(n_kappa)
    assert gated % WALD_GATED_STEP == 0
    assert gated >= WALD_REGIME_FACTOR * n_kappa
    assert gated - WALD_GATED_STEP < WALD_REGIME_FACTOR * n_kappa


def test_classify_splits_by_twice_the_tail_size() -> None:
    sizes = {"a": 4_000, "b": 104_400, "c": 5_000, "d": 400}
    gated, below, missing = tail.classify(sizes, {"a": 52_158, "b": 52_158, "c": None, "d": 200})
    assert gated == {"b", "d"}
    assert below == {"a"}
    assert missing == {"c"}


# ------------------------------------------------------------------ law B, by quadrature


@pytest.mark.parametrize(("plan", "quantity"), sorted(PINNED_KAPPA))
def test_the_p4_tail_sizes(plan: str, quantity: str) -> None:
    moments = tail.law_b_moments(plan, quantity)  # type: ignore[arg-type]
    assert moments.mass == pytest.approx(1.0, abs=1e-10)
    assert abs(moments.mean) < 1e-10
    assert moments.kappa == pytest.approx(PINNED_KAPPA[plan, quantity], abs=0.01)


def test_the_second_inverse_branch_carries_the_tail() -> None:
    """Witness: dropping the second branch of the ``p4`` ratio moves kappa by over 10%."""
    shipped = tail.law_b_moments("p4", "ate")
    mutated = tail.law_b_moments("p4", "ate", mutation="one_branch")
    assert abs(mutated.kappa / shipped.kappa - 1.0) > 0.10


def test_the_cumulative_product_carries_the_tail() -> None:
    """Witness: the first node's ratio alone in the last term puts kappa below 1,000."""
    assert tail.law_b_moments("p4", "ate", mutation="first_ratio_only").kappa < 1_000


# ---------------------------------------------- the change of measure (policy variance)


@pytest.mark.parametrize("right", [None, "natural"])
def test_the_diagonal_form_equals_the_second_moment_of_the_curve(right: str | None) -> None:
    """Witness 1 on ``p4``: with the true regressions the diagonal form is exact."""
    form = tail.diagonal_variance("p4", right)
    moments = tail.moments_covariance("p4", right)
    assert form == pytest.approx(moments, rel=1e-8)


def test_the_diagonal_form_gives_the_contrast_variance() -> None:
    ey = tail.diagonal_variance("p4")
    cross = tail.diagonal_variance("p4", "natural")
    natural = tail.moments_covariance("natural")
    assert ey == pytest.approx(tail.law_b_moments("p4", "ey").sd ** 2, rel=1e-8)
    assert ey + natural - 2 * cross == pytest.approx(
        tail.law_b_moments("p4", "ate").sd ** 2, rel=1e-8
    )


@pytest.mark.parametrize("shift", ["observed_dose", "no_ratio"])
def test_a_lost_shift_moves_the_form_by_more_than_half(shift: str) -> None:
    """Witness 2: reading ``S_t`` at ``A_t``, or dropping ``r(A^d)``, moves ``p4`` by over 50%."""
    shipped = tail.diagonal_variance("p4")
    mutated = tail.diagonal_variance("p4", shift=shift)  # type: ignore[arg-type]
    assert abs(mutated / shipped - 1.0) > 0.5


# ----------------------------------------------------------------- the law-B oracle ratio


class Spy(BaseEstimator):
    """Records each stacked design the classifier route hands a learner."""

    seen: ClassVar[list[np.ndarray]] = []

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Spy:
        del sample_weight
        x, label = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        type(self).seen.append(x[label == 0.0])
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.full((len(X), 2), 0.5)


def _fit(frame: Any, treatment_learner: Any, labels: tuple[str, ...] = ("natural", "p4")) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.fit(
            frame,
            "B",
            labels,
            outcome="wrong",
            ratio="classifier",
            treatment_learner=treatment_learner,
        )


def test_the_stacked_columns_are_the_dose_then_the_history() -> None:
    frame = law.sample_b(400, 3)
    Spy.seen.clear()
    _fit(frame, Spy())
    by_width = {block.shape[1]: block for block in Spy.seen}
    node1 = frame[["A1", "W1", "W2"]].to_numpy(dtype=float)
    node2 = frame[["A2", "W1", "W2", "L2", "A1"]].to_numpy(dtype=float)
    np.testing.assert_array_equal(by_width[3], node1)
    np.testing.assert_array_equal(by_width[5], node2)


@pytest.mark.parametrize("plan", ["p1", "p2", "p3", "p4"])
def test_the_oracle_ratio_is_the_closed_form(plan: str) -> None:
    frame = law.sample_b(2_000, 5)
    result = _fit(frame, law.OracleRatio(), ("natural", plan))
    fit = next(fit for fit in result.fits.values() if fit.regimen.label == plan)
    w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
    a1, l2, a2 = frame["A1"].to_numpy(), frame["L2"].to_numpy(), frame["A2"].to_numpy()
    u = law.upper(w2)
    expected = np.column_stack(
        [
            law.node_ratio(plan, 1, a1, law.mean1(w1, w2), u),
            law.node_ratio(plan, 2, a2, law.mean2(w1, w2, a1, l2), u),
        ]
    )
    np.testing.assert_allclose(fit.node_ratio, expected, rtol=1e-12, atol=1e-12)


def test_the_one_branch_mutation_moves_the_estimate() -> None:
    """The ``branch_necessity`` defect moves ``ate[p4]`` by more than 3 SE, wrong outcome."""
    frame = law.sample_b(2_000, 7)
    name = "ate_regimen[p4 vs natural]"
    shipped = _fit(frame, law.OracleRatio())[name]
    mutated = _fit(frame, law.OracleRatio(one_branch=True))[name]
    assert abs(mutated.psi - shipped.psi) > 3 * shipped.std_error


#: The floor of witness 3a for each plan (plan rev 3.3a).  ``p4`` moves by 2.7% and ``p1`` by
#: 0.33%; both floors sit at least five orders above the 1e-8 tolerance of witness 1.
POLICY_RESIDUAL_FLOOR = {"p1": 0.001, "p4": 0.01}


@pytest.mark.parametrize("plan", sorted(POLICY_RESIDUAL_FLOOR))
def test_a_residual_at_the_policy_dose_moves_the_exact_form(plan: str) -> None:
    """Witness 3a: the mutated recursion misses ``Var D`` upward by more than the floor.

    The shipped form meets it to 1e-8 (witness 1).  The mutation takes each node's residual
    against its regression at the policy dose, the closed-form issue of Tran et al. (2023).
    """
    shipped = tail.diagonal_variance(plan)
    mutated = tail.diagonal_variance(plan, shift="policy_residual")
    assert mutated / shipped - 1.0 > POLICY_RESIDUAL_FLOOR[plan]


# --------------------------------------------- the tail-size rule in the source studies
#
# Every Wald-verdicted cell of mtp-boundaries and mtp-pricing (each primary truth row, whose
# ``lmtp`` row follows its ``cleverly`` row, and each calibration positive) sits in exactly
# one of its study's ``BELOW_TAIL_SIZE``, ``WALD_GATED_CELLS`` and ``TAIL_SIZE_NOT_COMPUTED``.
# A cell's tail size is the larger of its quantity's efficient influence function's and its
# fit's own curve's, read from the committed diagnostic files.  The necessity and
# double-robustness positives read bias and displacement only, and no control is a validity
# claim, so neither enters.

DIAGNOSTIC = Path(__file__).resolve().parents[1] / "diagnostics" / "fix5_heavy_ratios"


@dataclass(frozen=True)
class TailCell:
    """One Wald-verdicted cell with its n and its tail size.

    Parameters
    ----------
    name : str
        ``scenario/estimand`` for a truth row, ``interval_calibration/<cell>`` otherwise.
    n : int
        Its declared size.
    n_kappa : int
        Its tail size, or a lower bound of it when ``exact`` is false.
    exact : bool
        False when only the efficient influence function's tail size is computed: the fit's
        own curve reads a learned ratio whose limit has no closed form.
    """

    name: str
    n: int
    n_kappa: int
    exact: bool = True


def _bernoulli_tail(p: float) -> int:
    """The tail size of ``Y - p`` for a binary ``Y``: the natural plan's influence function."""
    return tail.tail_size((1.0 - 3.0 * p * (1.0 - p)) / (p * (1.0 - p)))


@cache
def _law_b_eif() -> dict[tuple[str, str], int]:
    frame = pd.read_csv(DIAGNOSTIC / "quadrature.csv")
    frame = frame[frame.order == 40]
    out = {(r.plan, r.quantity): int(r.n_kappa) for r in frame.itertuples()}
    for label_law in ("B", "B_atom"):
        out[f"natural/{label_law}", "ey"] = _bernoulli_tail(law.truth("natural", label_law))
    return out


@cache
def _law_b_own() -> dict[tuple[str, str, str, str], int]:
    frame = pd.read_csv(DIAGNOSTIC / "law_b_own_moments.csv")
    frame = frame[frame.order == 40]
    return {
        (r.law, r.learners, r.estimator, r.quantity): int(r.n_kappa) for r in frame.itertuples()
    }


def _boundaries_eif(label: str, quantity: str, label_law: str) -> int:
    if label == law.REFERENCE:
        return _law_b_eif()[f"natural/{label_law}", "ey"]
    return _law_b_eif()[label, quantity]


def boundaries_cells() -> list[TailCell]:
    """Every Wald cell of mtp-boundaries with its tail size."""
    own = _law_b_own()
    cells = []
    for scenario, (label_law, estimator, labels) in study.SCENARIOS.items():
        for label in labels:
            for quantity in ("ey", "ate"):
                if quantity == "ate" and label == law.REFERENCE:
                    continue
                name = (
                    f"ey_regimen[{label}]"
                    if quantity == "ey"
                    else f"ate_regimen[{label} vs {law.REFERENCE}]"
                )
                n_kappa = max(
                    _boundaries_eif(label, quantity, label_law),
                    own[label_law, "primary", estimator, f"{quantity}[{label}]"],
                )
                cells.append(TailCell(f"{scenario}/{name}", study.PRIMARY_N, n_kappa))
    sizes = {
        f"{cell.property}/{cell.cell}": cell.n
        for cell in boundaries_properties.declared_cells()
        if cell.property == "interval_calibration" and cell.role == "positive"
    }
    for label, (plan, estimator) in boundaries_properties.CELLS.items():
        name = f"interval_calibration/{label}__correctly_specified"
        n_kappa = max(
            _law_b_eif()[plan, "ate"], own[law.LAW_OF[plan], "correct", estimator, f"ate[{plan}]"]
        )
        cells.append(TailCell(name, sizes[name], n_kappa))
    return cells


@cache
def _law_p() -> dict[tuple[int, str, str, str], int]:
    frame = pd.read_csv(DIAGNOSTIC / "law_p_moments.csv")
    return {
        (int(r.tau), r.learners, r.estimator, r.quantity): int(r.n_kappa)
        for r in frame.itertuples()
    }


def pricing_cells() -> list[TailCell]:
    """Every Wald cell of mtp-pricing with its tail size.

    The property learners' ``LTMLE`` own curve, with the exact ratio and a correctly specified
    outcome model, stands in for the efficient influence function (law P has no closed form).
    """
    own = _law_p()
    cells = []
    for scenario, (tau, estimator) in pricing.SCENARIOS.items():
        for label in law_p.LABELS:
            for quantity in ("ey", "ate"):
                if quantity == "ate" and label == law_p.REFERENCE:
                    continue
                name = (
                    f"ey_regimen[{label}]"
                    if quantity == "ey"
                    else f"ate_regimen[{label} vs {law_p.REFERENCE}]"
                )
                key = f"{quantity}[{label}]"
                n_kappa = max(
                    own[tau, "property", "ltmle", key], own[tau, "primary", estimator, key]
                )
                cells.append(TailCell(f"{scenario}/{name}", pricing.PRIMARY_N, n_kappa))
    sizes = {
        f"{cell.property}/{cell.cell}": cell.n
        for cell in pricing_properties.declared_cells()
        if cell.property == "interval_calibration" and cell.role == "positive"
    }
    for label, (tau, route, estimator, folds, revenue) in pricing_properties.CELLS.items():
        name = f"interval_calibration/{label}__correctly_specified"
        learners = "revenue" if revenue else "property"
        eif = own[tau, learners, "ltmle", f"ey[{pricing_properties.PLAN}]"]
        exact = (route, estimator, folds) == ("oracle", "ltmle", 1)
        cells.append(TailCell(name, sizes[name], eif, exact))
    return cells


def tail_size_problems(
    cells: list[TailCell],
    below: Collection[str],
    gated: Collection[str],
    not_computed: Collection[str],
) -> list[str]:
    """What breaks check 1 of the tail-size rule; empty when the declaration holds.

    The three classes partition the Wald cells.  A cell with a computed tail size is gated
    exactly when its n is at least ``WALD_REGIME_FACTOR`` times it.  A cell with only a lower
    bound is below when its n is under twice the bound, and not computed otherwise.  The gated
    twins of check 2 belong to mtp-policy-variance, which declares them.
    """
    problems: list[str] = []
    names = {cell.name for cell in cells}
    classes = (set(below), set(gated), set(not_computed))
    if sum(len(group) for group in classes) != len(set().union(*classes)):
        problems.append("check 1: a cell sits in two classes")
    if set().union(*classes) != names:
        missing = sorted(names - set().union(*classes))
        extra = sorted(set().union(*classes) - names)
        problems.append(f"check 1: the classes miss {missing} and hold {extra}")
    for cell in cells:
        need = WALD_REGIME_FACTOR * cell.n_kappa
        if cell.n < need:
            expected = below
        elif cell.exact:
            expected = gated
        else:
            expected = not_computed
        if cell.name not in expected:
            problems.append(f"check 1: {cell.name} has n {cell.n} against {need}, misclassified")
    return problems


STUDIES = {
    "boundaries": (boundaries_cells, study, boundaries_properties),
    "pricing": (pricing_cells, pricing, pricing_properties),
}


def _declared(name: str) -> tuple[list[TailCell], set[str], set[str], set[str]]:
    cells, truth_module, property_module = STUDIES[name]
    below = set(truth_module.BELOW_TAIL_SIZE) | set(property_module.BELOW_TAIL_SIZE)
    gated = set(truth_module.WALD_GATED_CELLS) | set(property_module.WALD_GATED_CELLS)
    missing = set(truth_module.TAIL_SIZE_NOT_COMPUTED) | set(property_module.TAIL_SIZE_NOT_COMPUTED)
    return cells(), below, gated, missing


@pytest.mark.parametrize("name", sorted(STUDIES))
def test_each_wald_cell_gates_only_at_twice_its_tail_size(name: str) -> None:
    assert tail_size_problems(*_declared(name)) == []


@pytest.mark.parametrize("name", sorted(STUDIES))
def test_each_module_holds_its_own_cells(name: str) -> None:
    """The truth rows' classes live in the study module, the calibration cells' in its properties."""
    _, truth_module, property_module = STUDIES[name]
    for module, own in ((truth_module, False), (property_module, True)):
        for group in (
            module.BELOW_TAIL_SIZE,
            module.WALD_GATED_CELLS,
            module.TAIL_SIZE_NOT_COMPUTED,
        ):
            assert all(key.startswith("interval_calibration/") == own for key in group)


def test_gating_a_below_cell_fails() -> None:
    cells, below, gated, missing = _declared("boundaries")
    key = "boundaries_ltmle/ate_regimen[p4 vs natural]"
    problems = tail_size_problems(cells, below - {key}, gated | {key}, missing)
    assert any(problem.startswith(f"check 1: {key} has n 5000") for problem in problems)


def test_an_undeclared_cell_fails() -> None:
    cells, below, gated, missing = _declared("pricing")
    extra = TailCell("pricing_ltmle_tau7/ey_regimen[price_up]", 5_000, 10)
    problems = tail_size_problems([*cells, extra], below, gated, missing)
    assert any("pricing_ltmle_tau7" in problem for problem in problems)


def test_a_learned_ratio_cell_is_not_gated_on_its_proxy() -> None:
    """Without the own curve a gated reading on the proxy alone fails."""
    cells, below, gated, missing = _declared("pricing")
    key = next(iter(missing))
    problems = tail_size_problems(cells, below, gated | {key}, missing - {key})
    assert any(problem.startswith(f"check 1: {key}") for problem in problems)
