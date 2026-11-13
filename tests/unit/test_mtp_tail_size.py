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
from typing import Any, ClassVar

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from tests.studies import canonical_mtp_boundaries as study
from tests.studies import mtp_boundaries_common as law
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
