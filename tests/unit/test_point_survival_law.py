"""A held baseline treatment over survival nodes, checked exactly on a finite law.

``tests/discrete_law_point_survival.py`` realises each law exactly, so a saturated fit is
the oracle and every assertion is an identity.  The estimate equals the g-formula, the
influence curve equals the Gateaux derivative of the g-formula, and the held fit equals the
shipped fit on copied treatment columns.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE
from cleverly.validation.longitudinal import LONGITUDINAL_HELD_DECISION
from tests.discrete_law_point_survival import (
    CellMeans,
    CellProbabilities,
    PointLaw,
    static,
    survival_law,
)

REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
HORIZONS = (1, 2, 3)
#: Bit-exact identities hold on the platform that computed them.  CI runs on Linux, where a
#: different BLAS can move the last bits, so CI compares at a relative 1e-12.
EXACT = os.environ.get("CI") is None
#: An absolute floor for comparisons of values that may be zero.
FLOOR = 1e-12


def _learners() -> dict[str, object]:
    return {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "censoring_learner": CellMeans(),
    }


def _fit(law: PointLaw, *, held: bool = True, regimens: dict | None = None) -> object:
    return LTMLE(regimens or REGIMENS, n_folds=1, **_learners()).fit(
        law.frame(), **law.fit_columns(held=held)
    )


def _keys(law: PointLaw, label: str) -> list[tuple[str, int, int]]:
    if law.causes == 2:
        return [
            (f"cif_regimen[{label}, {name} @ t={h}]", h, code)
            for h in HORIZONS
            for code, name in ((1, "relapse"), (2, "death"))
        ]
    return [(f"risk_regimen[{label} @ t={h}]", h, 1) for h in HORIZONS]


def _same(left: np.ndarray, right: np.ndarray) -> None:
    if EXACT:
        assert np.array_equal(left, right)
    else:
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=FLOOR)


LAWS = {
    "one cause": survival_law(),
    "two causes": survival_law(causes=2),
    "with L2": survival_law(with_l2=True),
}


@pytest.fixture(scope="module", params=list(LAWS))
def fitted(request: pytest.FixtureRequest) -> tuple[PointLaw, object]:
    law = LAWS[request.param]
    return law, _fit(law)


def test_the_estimate_is_the_g_formula(fitted: tuple[PointLaw, object]) -> None:
    law, result = fitted
    for label, arm in REGIMENS.items():
        for name, horizon, cause in _keys(law, label):
            truth = law.functional(law.probs, static(arm), horizon, cause)
            assert result[name].psi == pytest.approx(truth, rel=1e-12, abs=FLOOR)  # type: ignore[index]


def test_the_influence_curve_is_the_gateaux_derivative(fitted: tuple[PointLaw, object]) -> None:
    law, result = fitted
    rows = law.first_row_of()
    for label, arm in REGIMENS.items():
        for name, horizon, cause in _keys(law, label):
            eif = law.eif(static(arm), horizon, cause)
            np.testing.assert_allclose(
                result[name].influence_curve[rows],  # type: ignore[index]
                eif,
                rtol=1e-10,
                atol=1e-12,
            )


def test_the_held_fit_equals_the_copied_column_fit(fitted: tuple[PointLaw, object]) -> None:
    """At one fold with cell-mean learners the copied fit's later g_t is exactly 1."""
    law, held = fitted
    copied = _fit(law, held=False)
    assert set(held.estimates) == set(copied.estimates)  # type: ignore[attr-defined]
    for name in held.estimates:  # type: ignore[attr-defined]
        _same(np.array([held[name].psi]), np.array([copied[name].psi]))  # type: ignore[index]
        _same(held[name].influence_curve, copied[name].influence_curve)  # type: ignore[index]
    for key, fit in held.fits.items():  # type: ignore[attr-defined]
        other = copied.fits[key]  # type: ignore[attr-defined]
        for step, twin in zip(fit.steps, other.steps, strict=True):
            # Off the followers the copied fit holds fallback values and the held one 1.
            followers = step.trained_on
            assert np.array_equal(followers, twin.trained_on)
            column = step.time - 1
            _same(fit.cumulative[followers, column], other.cumulative[followers, column])


def test_the_held_mechanism_has_factor_one_and_no_model(fitted: tuple[PointLaw, object]) -> None:
    law, result = fitted
    mechanism = result.mechanism  # type: ignore[attr-defined]
    for node in range(2, law.n_times + 1):
        for label in REGIMENS:
            assert np.array_equal(mechanism.treatment[node - 1][label], np.ones(law.n))
        assert mechanism.treatment_observed[node - 1].shape == (law.n, 0)
        assert mechanism.treatment_diagnostics[node - 1] == ()
    omissions = result.diagnostics.nuisance_models().omissions  # type: ignore[attr-defined]
    held = sorted(item.time for item in omissions if item.reason == LONGITUDINAL_HELD_DECISION)
    assert held == list(range(2, law.n_times + 1))


def test_the_masks_close_the_recursion(fitted: tuple[PointLaw, object]) -> None:
    """``at_risk(t + 1) == following(t) & event-free at t`` on the held design."""
    law, result = fitted
    data = result.data  # type: ignore[attr-defined]
    for fit in result.fits.values():  # type: ignore[attr-defined]
        if fit.horizon != law.n_times:
            continue
        for step, later in zip(fit.steps[:-1], fit.steps[1:], strict=True):
            assert later.time == step.time + 1
            expected = step.trained_on & data.event_free_through(step.time)
            assert np.array_equal(later.at_risk, expected)


def test_one_cause_declared_as_a_mapping_equals_survival() -> None:
    law = survival_law()
    single = _fit(law)
    columns = law.fit_columns()
    columns["outcome"] = {"event": columns["outcome"]}
    mapped = LTMLE(REGIMENS, n_folds=1, **_learners()).fit(law.frame(), **columns)
    for label in REGIMENS:
        for horizon in HORIZONS:
            left = single[f"risk_regimen[{label} @ t={horizon}]"]  # type: ignore[index]
            right = mapped[f"cif_regimen[{label}, event @ t={horizon}]"]  # type: ignore[index]
            _same(np.array([left.psi]), np.array([right.psi]))
            _same(left.influence_curve, right.influence_curve)


def test_a_multi_node_plan_on_a_held_design_is_refused_before_any_learner() -> None:
    class Refuse(CellMeans):
        def fit(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("a learner ran")

    law = survival_law()
    with pytest.raises(ValueError, match="declares 3 nodes, but this design has one treatment"):
        LTMLE(
            {"switch": (0, 1, 1)},
            n_folds=1,
            outcome_learner=Refuse(),
            treatment_learner=Refuse(),
            censoring_learner=Refuse(),
        ).fit(law.frame(), **law.fit_columns())


def test_a_repeated_static_plan_reads_as_its_one_arm() -> None:
    law = survival_law()
    one = _fit(law, regimens={"a1": 1})
    repeated = _fit(law, regimens={"a1": (1, 1, 1)})
    for name in one.estimates:  # type: ignore[attr-defined]
        assert one[name].psi == repeated[name].psi  # type: ignore[index]


def test_a_saturated_working_model_reproduces_the_held_risks() -> None:
    """One coefficient per arm and horizon: the projection is the risks themselves."""
    from cleverly.msm import MSM

    law = survival_law()
    cells = [(label, horizon) for label in REGIMENS for horizon in HORIZONS]

    def design(label: object, horizon: int, frame: object) -> np.ndarray:
        n = len(frame)  # type: ignore[arg-type]
        return np.column_stack([np.full(n, float((label, horizon) == cell)) for cell in cells])

    msm = MSM(design=design, terms=tuple(f"{a}_t{h}" for a, h in cells), design_kind="known")
    fit = LTMLE(REGIMENS, n_folds=1, msm=msm, **_learners()).fit(law.frame(), **law.fit_columns())
    for (label, horizon), term in zip(cells, msm.terms, strict=True):
        truth = law.functional(law.probs, static(REGIMENS[label]), horizon)
        assert fit[f"msm_regimen[{term}]"].psi == pytest.approx(truth, rel=1e-10, abs=FLOOR)
