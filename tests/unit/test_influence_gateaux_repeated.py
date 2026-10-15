r"""The repeated-outcome influence curve, against a numerically differentiated functional.

``tests/discrete_law_repeated.py`` holds two laws with an outcome at each node that is not
absorbing, a sample that realises each one exactly, and the g-formula of each horizon
written on the cell probabilities.  Its complex-step Gateaux derivative *is* the efficient
influence function.  Nothing the library supplies enters it.

With the saturated learner at every node the fit holds the exact nuisances, so the point
estimate is the truth and the influence curve is the EIF, at every regimen and horizon.
The design tests below show that the two mistakes this outcome invites move the answer on
these laws: the absorbing reading, and a history that omits the earlier outcome.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import numpy as np
import pytest

from cleverly.inference.multiplier import simultaneous_bands
from cleverly.longitudinal import LTMLE, Repeated

from .. import discrete_law_repeated as laws
from ..studies.canonical_ltmle import declared_regimens

#: Truncation wide enough never to bind: every mechanism lies in [0.25, 0.75].
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)

COLUMNS: dict[str, Any] = {
    "outcome": Repeated(["Y1", "Y2"]),
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}

LAWS = (laws.BINARY, laws.CONTINUOUS)

#: Multiplier draws of the banded fit and of the bands rebuilt beside it.
MULTIPLIERS = 2000


def oracle_fit(law: laws.RepeatedLaw, frame: Any = None, **overrides: Any) -> Any:
    """A fit of the law's exact sample with the saturated learner at every node."""
    settings: dict[str, Any] = {
        "reference": laws.REGIMEN_REFERENCE,
        "outcome_learner": laws.CellMeans(),
        "pseudo_learner": laws.CellMeans(),
        "treatment_learner": laws.CellMeans(),
        "censoring_learner": laws.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    regimens = settings.pop("regimens", declared_regimens(law.regimen_spec()))
    return LTMLE(regimens, **settings).fit(law.frame() if frame is None else frame, **COLUMNS)


@pytest.fixture(scope="module", params=LAWS, ids=[law.name for law in LAWS])
def law_and_fit(request: pytest.FixtureRequest) -> tuple[laws.RepeatedLaw, Any]:
    law = request.param
    return law, oracle_fit(law)


# ---------------------------------------------------------------- the law itself


@pytest.mark.parametrize("law", LAWS, ids=[law.name for law in LAWS])
def test_the_two_enumerations_of_the_truth_agree(law: laws.RepeatedLaw) -> None:
    """The cell g-formula and the counterfactual composition give the same numbers."""
    for name in law.names:
        assert law.truth[name] == pytest.approx(law.counterfactual_value(name), abs=1e-14)


def test_censoring_reads_the_history_and_never_the_current_outcome() -> None:
    """``C_t`` is a function of ``(H_t, A_t)``: its tables have no ``Y_t`` axis.

    The witness is structural: ``C1`` is indexed ``[w, a1]`` and ``C2`` is indexed
    ``[w, a1, high(Y1), l2, a2]``.  Each conditional probability of ``C2 = 1`` given the
    history is then the same whatever ``Y2`` turns out to be, which the sample confirms.
    """
    assert laws.C1.shape == (2, 2)
    assert laws.C2.shape == (2, 2, 2, 2, 2)
    law = laws.BINARY
    for w, a1, y1, l2, a2 in itertools.product(*(range(2),) * 5):
        reached = law._mass(law.probs, w=w, a1=a1, c1=1, y1=y1, l2=l2, a2=a2)
        kept = law._mass(law.probs, w=w, a1=a1, c1=1, y1=y1, l2=l2, a2=a2, c2=1)
        assert kept / reached == pytest.approx(laws.C2[w, a1, law.high(y1), l2, a2], abs=0)


def test_the_outcome_returns() -> None:
    """``P(Y2 = 0 | Y1 = 1, .) >= 0.3`` in every cell: the outcome is not absorbing."""
    law = laws.BINARY
    for w, a1, l2, a2 in itertools.product(*(range(2),) * 4):
        assert law.y2_table(w, a1, 1, l2, a2)[0] >= 0.3


def test_the_absorbing_reading_misses_the_truth() -> None:
    """``E[max(Y1, Y2)^d]`` differs from ``E[Y2^d]`` by the declared gap at ``h = 2``."""
    law = laws.BINARY
    for regimen in law.regimens:
        name = f"ey_regimen[{regimen} @ t=2]"
        gap = abs(law.functional(law.probs, name, absorbing=True) - law.truth[name])
        assert gap >= laws.ABSORBING_GAP, regimen


def test_a_history_without_the_earlier_outcome_misses_the_truth() -> None:
    """The g-formula on ``(W, A1, L2)`` misses ``E[Y2^d]`` by the declared gap."""
    law = laws.BINARY
    for regimen in ("never", "always"):
        name = f"ey_regimen[{regimen} @ t=2]"
        assert abs(law.history_free_value(name) - law.truth[name]) >= laws.HISTORY_GAP


def test_the_rule_is_a_parameter_no_static_plan_reaches() -> None:
    """``treat_if_high`` reads ``Y1``, and its truth at ``h = 2`` is its own."""
    for law in LAWS:
        rule = law.truth["ey_regimen[treat_if_high @ t=2]"]
        for label in ("never", "always"):
            assert abs(rule - law.truth[f"ey_regimen[{label} @ t=2]"]) > 1e-2


def test_the_continuous_law_spans_a_wider_first_outcome() -> None:
    """``Y1`` leaves ``[0, 1]`` while ``Y2`` stays inside it: the scaler witness needs both."""
    law = laws.CONTINUOUS
    assert min(law.y1_values) < 0.0 and max(law.y1_values) > 1.0
    assert min(law.y2_values) >= 0.0 and max(law.y2_values) <= 1.0


# ---------------------------------------------------------------- the fit


def test_every_reported_parameter_has_an_oracle_and_no_more(
    law_and_fit: tuple[laws.RepeatedLaw, Any],
) -> None:
    law, fit = law_and_fit
    assert set(fit) == set(law.names)
    assert fit.data.family == law.family


def test_point_estimate_is_the_g_formula(law_and_fit: tuple[laws.RepeatedLaw, Any]) -> None:
    """With exact nuisances the estimate is the truth at every regimen and horizon."""
    law, fit = law_and_fit
    for name in law.names:
        assert fit.psi(name) == pytest.approx(law.truth[name], abs=1e-12), name


def test_influence_curve_is_the_gateaux_derivative(
    law_and_fit: tuple[laws.RepeatedLaw, Any],
) -> None:
    law, fit = law_and_fit
    rows = law.first_row_of()
    for name in law.names:
        reported = fit.influence_curves[name][rows]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-11, rtol=0, err_msg=name)


def test_targeting_had_nothing_to_do(law_and_fit: tuple[laws.RepeatedLaw, Any]) -> None:
    """An exact initial fit solves every score equation, so ``epsilon`` is zero."""
    _, fit = law_and_fit
    for regimen_fit in fit.fits.values():
        for step in regimen_fit.steps:
            assert step.fluctuation.converged
            assert abs(float(step.fluctuation.epsilon[0])) < 1e-8


# ---------------------------------------------------------------- W3: the joint band


@pytest.fixture(scope="module")
def banded() -> Any:
    return oracle_fit(laws.BINARY, simultaneous=True, n_multiplier=MULTIPLIERS, random_state=0)


def test_the_horizons_are_correlated(banded: Any) -> None:
    """The two horizons of one plan share units, so their curves correlate."""
    for regimen in laws.BINARY.regimens:
        first = banded.influence_curves[f"ey_regimen[{regimen} @ t=1]"]
        second = banded.influence_curves[f"ey_regimen[{regimen} @ t=2]"]
        # The binary law gives correlations of -0.12, -0.13 and -0.29.
        assert abs(np.corrcoef(first, second)[0, 1]) > 0.1, regimen


def test_the_band_is_built_from_the_stacked_curves(banded: Any) -> None:
    """The fit's band is the multiplier band over every parameter's own curve.

    A band built from curves made independent across horizons has a different critical
    value, so this test fails when the band ignores the cross-horizon covariance.
    """
    estimates = {name: banded.estimates[name] for name in banded.estimates}
    joint = simultaneous_bands(estimates, alpha=0.05, n_replicates=MULTIPLIERS, random_state=0)
    assert banded.simultaneous.critical_value == pytest.approx(joint.critical_value, abs=0)
    rng = np.random.default_rng(1)
    order = rng.permutation(laws.N)
    shuffled = {
        name: (
            dataclasses.replace(estimate, influence_curve=estimate.influence_curve[order])
            if "@ t=2" in name
            else estimate
        )
        for name, estimate in estimates.items()
    }
    independent = simultaneous_bands(shuffled, alpha=0.05, n_replicates=MULTIPLIERS, random_state=0)
    assert abs(independent.critical_value - joint.critical_value) > 1e-3


def test_the_static_plans_read_the_earlier_outcome() -> None:
    """The M3 witness: a static plan's ``h = 2`` mean needs ``Y1`` in the history.

    No plan here reads ``Y1``, so a history without it still fits.  It then converges to
    :meth:`~tests.discrete_law_repeated.RepeatedLaw.history_free_value`, which misses the
    truth by at least :data:`~tests.discrete_law_repeated.HISTORY_GAP`.
    """
    law = laws.BINARY
    fit = oracle_fit(law, regimens={"never": 0, "always": 1})
    for regimen in ("never", "always"):
        name = f"ey_regimen[{regimen} @ t=2]"
        assert fit.psi(name) == pytest.approx(law.truth[name], abs=1e-12)


# ---------------------------------------------------------------- E4: the incremental functional

DELTAS = (0.5, 1.0, 2.0)


@pytest.mark.parametrize("law", LAWS, ids=[law.name for law in LAWS])
def test_e4_the_tilt_functional_equals_forward_composition(law: laws.RepeatedLaw) -> None:
    """Kim et al. (2021, Theorem 3.1) on the cells equals the tilted conditionals composed.

    The two enumerations share only the tables.  The cell form reads the observed law with
    censoring in place, and the composition sets every censoring node to one.
    """
    for delta in DELTAS:
        for horizon in laws.HORIZONS:
            cells = law.tilt_functional(law.probs, delta, horizon)
            composed = law.tilt_counterfactual_value(delta, horizon)
            assert cells == pytest.approx(composed, abs=1e-14), (delta, horizon)


def test_e4_the_tilt_functional_moves_with_the_tilt() -> None:
    """A tilt functional that ignored ``delta`` would pass the agreement above at ``delta = 1``."""
    for law in LAWS:
        for horizon in laws.HORIZONS:
            values = [law.tilt_functional(law.probs, delta, horizon) for delta in DELTAS]
            assert abs(values[0] - values[2]) > 1e-2, (law.name, horizon)


def test_e4_the_absorbing_tilt_functional_misses_the_truth_at_the_second_horizon() -> None:
    """The return-to-zero precondition: for every ``delta`` the absorbing reading differs.

    At ``h = 1`` there is no earlier outcome to absorb, so the two functionals are equal.  The
    law-side value cannot fail under a change to ``cleverly``: it guards the law against a
    constant that makes the outcome absorbing.
    """
    law = laws.BINARY
    for delta in DELTAS:
        first = law.tilt_functional(law.probs, delta, 1, absorbing=True)
        assert first == pytest.approx(law.tilt_functional(law.probs, delta, 1), abs=1e-14)
        gap = law.tilt_functional(law.probs, delta, 2, absorbing=True) - law.tilt_functional(
            law.probs, delta, 2
        )
        assert gap >= laws.ABSORBING_GAP, delta


@pytest.mark.parametrize("law", LAWS, ids=[law.name for law in LAWS])
def test_e4_the_tilt_influence_function_is_centered_and_nonzero(law: laws.RepeatedLaw) -> None:
    """The complex-step derivative of the tilt functional has mean zero and a spread."""
    for delta in (0.5, 2.0):
        for horizon in laws.HORIZONS:
            curve = law.tilt_eif(delta, horizon)
            assert abs(float(np.sum(law.probs * curve))) < 1e-12
            assert float(np.sqrt(np.sum(law.probs * curve**2))) > 0.3
