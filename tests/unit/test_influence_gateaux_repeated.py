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
from cleverly.interventions import Incremental
from cleverly.longitudinal import IPSI, LTMLE, SDR, Repeated
from cleverly.longitudinal.estimator import HORIZON_INFIX

from .. import discrete_law_repeated as laws
from ..studies.canonical_ltmle import declared_regimens
from .test_repeated_law_values import NON_SEQUENTIAL_GAP as SDR_NON_SEQUENTIAL_GAP
from .test_repeated_law_values import PLUG_IN_GAP as SDR_PLUG_IN_GAP

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


# ---------------------------------------------------------------- IPSI: E5 to E8

#: The tilts of the IPSI fits, the natural course first so that it is the reference.
TILTS = (1.0, 0.5, 2.0)
LABELS = {1.0: "natural course", 0.5: "odds x0.5", 2.0: "odds x2"}
CONTRASTED = (0.5, 2.0)


class Baseline(laws.CellMeans):
    """The saturated learner on the baseline covariate ``W`` alone, the first design column.

    Every design of a repeated fit starts with ``W``.  A mechanism that reads ``W`` alone is
    exact at node 1, where ``H_1 = W``, and wrong at node 2, where the history holds ``A_1``,
    ``Y_1`` and ``L_2``.
    """

    #: The design columns kept besides ``W``: none.
    arm = False

    def _select(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        columns = [0, matrix.shape[1] - 1] if self.arm and matrix.shape[1] > 1 else [0]
        return matrix[:, columns]

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> Baseline:
        super().fit(self._select(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(self._select(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(self._select(X))


class BaselineAndArm(Baseline):
    """``W`` and the arm of the current node, which is the last column of a tilted design.

    A regression that omits the history but keeps the arm separates the two arms of a node,
    so the mechanism term has a blip to multiply.  It is exact at node 1 and wrong at node 2.
    """

    arm = True


def ipsi_fit(
    law: laws.RepeatedLaw,
    *,
    frame: Any = None,
    outcome: Any = None,
    treatment: Any = None,
    censoring: Any = None,
    tilts: tuple[float, ...] = TILTS,
    roles: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """``IPSI`` on the law's exact sample, with a saturated learner unless one is handed in."""
    settings: dict[str, Any] = {
        "outcome_learner": outcome or laws.CellMeans(),
        "pseudo_learner": outcome or laws.CellMeans(),
        "treatment_learner": treatment or laws.CellMeans(),
        "censoring_learner": censoring or laws.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    plans = [Incremental(delta) for delta in tilts]
    return IPSI(plans, **settings).fit(
        law.frame() if frame is None else frame, **({**COLUMNS, **(roles or {})})
    )


def _mean(delta: float, horizon: int) -> str:
    return f"ey_regimen[{LABELS[delta]} @ t={horizon}]"


def _contrast(delta: float, horizon: int) -> str:
    return f"ate_regimen[{LABELS[delta]} vs {LABELS[1.0]} @ t={horizon}]"


def check_e5(law: laws.RepeatedLaw, fit: Any) -> None:
    """E5: the estimate is the tilt functional, and the curve its complex-step derivative.

    The derivative is taken of :meth:`~tests.discrete_law_repeated.RepeatedLaw.tilt_functional`
    on the cell probabilities, so nothing the library supplies enters it.
    """
    rows = law.first_row_of()
    for horizon in laws.HORIZONS:
        eif = {delta: law.tilt_eif(delta, horizon) for delta in TILTS}
        for delta in TILTS:
            truth = law.tilt_functional(law.probs, delta, horizon)
            assert fit.psi(_mean(delta, horizon)) == pytest.approx(truth, abs=1e-12), (
                f"estimate: delta={delta}, h={horizon}"
            )
            np.testing.assert_allclose(
                fit.influence_curves[_mean(delta, horizon)][rows],
                eif[delta],
                atol=1e-11,
                rtol=0,
                err_msg=f"curve: delta={delta}, h={horizon}",
            )
        for delta in CONTRASTED:
            name = _contrast(delta, horizon)
            assert fit.psi(name) == pytest.approx(
                law.tilt_functional(law.probs, delta, horizon)
                - law.tilt_functional(law.probs, 1.0, horizon),
                abs=1e-12,
            ), f"estimate: {name}"
            np.testing.assert_allclose(
                fit.influence_curves[name][rows],
                eif[delta] - eif[1.0],
                atol=1e-11,
                rtol=0,
                err_msg=f"curve: {name}",
            )


@pytest.fixture(scope="module", params=LAWS, ids=[law.name for law in LAWS])
def ipsi_law_and_fit(request: pytest.FixtureRequest) -> tuple[laws.RepeatedLaw, Any]:
    law = request.param
    return law, ipsi_fit(law)


def test_e5_saturated_m_and_estimated_mechanisms_give_the_tilt_functional_and_its_derivative(
    ipsi_law_and_fit: tuple[laws.RepeatedLaw, Any],
) -> None:
    """``IPSI`` with saturated ``m`` and estimated ``pi`` and ``omega`` is exact.

    The mechanism term carries 1% to 3% of the curve variance on the X32a law, so a curve
    without it fails the derivative check.  The mutation controls N3 and N9 run this check.
    """
    law, fit = ipsi_law_and_fit
    assert set(fit) == {
        *(_mean(delta, h) for delta in TILTS for h in laws.HORIZONS),
        *(_contrast(delta, h) for delta in CONTRASTED for h in laws.HORIZONS),
    }
    check_e5(law, fit)


def test_e5_the_mechanism_term_is_nonzero_at_every_tilt_node(
    ipsi_law_and_fit: tuple[laws.RepeatedLaw, Any],
) -> None:
    """The term that N3 drops is present: a nonzero column at each node of a tilted plan."""
    _, fit = ipsi_law_and_fit
    for delta in CONTRASTED:
        term = fit.fits[f"{LABELS[delta]}{HORIZON_INFIX}2"].mechanism_term
        assert term.shape[1] == 2
        assert np.max(np.abs(term[:, 0])) > 1e-3 and np.max(np.abs(term[:, 1])) > 1e-3


#: ``E6``: the ``W``-only plug-in ``P(m-bar_1)`` misses the tilt functional at ``h = 2`` by
#: these amounts, so the one-step correction carries the whole gap.
PLUG_IN_GAP = {
    "binary": {0.5: +0.047090, 2.0: -0.029297},
    "continuous": {0.5: -0.005218, 2.0: +0.009470},
}

#: ``E7``: saturated ``m`` with ``W``-only ``pi`` and ``omega``, and with ``W``-only ``pi``
#: alone, misses the tilt functional at ``h = 2``.
NOT_DOUBLY_ROBUST_BIAS = {
    "binary": {0.5: (+0.000803, +0.001453), 2.0: (-0.000907, -0.000312)},
    "continuous": {0.5: (-0.000766, -0.001078), 2.0: (+0.001467, +0.001107)},
}


def known_treatment(law: laws.RepeatedLaw) -> tuple[Any, dict[str, Any]]:
    """The law's frame with ``P(A_t = a | H_t)`` columns, and the roles that declare them.

    A row that does not reach node ``t`` holds missing values there, as the fit expects.
    """
    frame = law.frame()
    high = np.array([-1 if np.isnan(y) else law.high(law.y1_values.index(y)) for y in frame["Y1"]])
    node_2 = (high >= 0) & frame["L2"].notna().to_numpy()
    w, a1, l2 = (frame[name].fillna(0).to_numpy().astype(int) for name in ("W", "A1", "L2"))
    g1 = laws.G1[w]
    g2 = np.full(len(frame), np.nan)
    g2[node_2] = laws.G2[w[node_2], a1[node_2], high[node_2], l2[node_2]]
    frame = frame.assign(g1_1=g1, g1_0=1.0 - g1, g2_1=g2, g2_0=1.0 - g2)
    roles = {
        "treatment_probabilities": {
            "A1": {0.0: "g1_0", 1.0: "g1_1"},
            "A2": {0.0: "g2_0", 1.0: "g2_1"},
        }
    }
    return frame, roles


def check_e6(law: laws.RepeatedLaw, fit: Any) -> None:
    """E6: with ``W``-only ``m`` and consistent mechanisms the estimate is the tilt functional.

    The curve is the law-side curve of the same one-step estimator.
    """
    rows = law.first_row_of()
    for delta in TILTS:
        for horizon in laws.HORIZONS:
            truth = law.tilt_functional(law.probs, delta, horizon)
            assert fit.psi(_mean(delta, horizon)) == pytest.approx(truth, abs=1e-12), (
                f"estimate: delta={delta}, h={horizon}"
            )
            _, curve = law.ipsi_population(delta, horizon, key="W")
            np.testing.assert_allclose(
                fit.influence_curves[_mean(delta, horizon)][rows],
                curve,
                atol=1e-11,
                rtol=0,
                err_msg=f"curve: delta={delta}, h={horizon}",
            )


@pytest.mark.parametrize("law", LAWS, ids=[law.name for law in LAWS])
def test_e6_w_only_m_with_estimated_mechanisms_is_exact_and_its_plug_in_is_not(
    law: laws.RepeatedLaw,
) -> None:
    """Single robustness: a wrong ``m`` with consistent ``pi`` and ``omega`` still gives the truth.

    The regression reads ``W`` and the arm of the node, which is saturated at node 1 and
    wrong at node 2.  The plug-in of that regression misses the functional at ``h = 2`` by a
    declared nonzero gap, and the one-step estimate does not.  With ``omega`` replaced by one,
    the same one-step misses (the precondition of the mutation control N9).
    """
    check_e6(law, ipsi_fit(law, outcome=BaselineAndArm()))
    for delta, gap in PLUG_IN_GAP[law.name].items():
        truth = law.tilt_functional(law.probs, delta, 2)
        plug_in, _ = law.ipsi_population(delta, 2, key="W", correction=False)
        assert plug_in - truth == pytest.approx(gap, abs=6e-7)
        assert abs(gap) > 1e-3
        without, _ = law.ipsi_population(delta, 2, key="W", omega=False)
        assert abs(without - truth) > 1e-3


@pytest.mark.parametrize("law", LAWS, ids=[law.name for law in LAWS])
def test_e6_a_declared_treatment_mechanism_removes_the_mechanism_term(
    law: laws.RepeatedLaw,
) -> None:
    """With ``pi`` declared known the term is absent, and the estimate is still the truth."""
    frame, roles = known_treatment(law)
    estimated = ipsi_fit(law, outcome=BaselineAndArm())
    # A declared mechanism may not be truncated, and the natural course has cumulative factor 1.
    known = ipsi_fit(law, outcome=BaselineAndArm(), frame=frame, roles=roles, g_bounds=(1e-8, 1.0))
    for delta in CONTRASTED:
        fit = known.fits[f"{LABELS[delta]}{HORIZON_INFIX}2"]
        assert not np.any(fit.mechanism_term)
        assert np.any(estimated.fits[f"{LABELS[delta]}{HORIZON_INFIX}2"].mechanism_term)
        for horizon in laws.HORIZONS:
            truth = law.tilt_functional(law.probs, delta, horizon)
            assert known.psi(_mean(delta, horizon)) == pytest.approx(truth, abs=1e-12)


@pytest.mark.parametrize("law", LAWS, ids=[law.name for law in LAWS])
def test_e7_ipsi_is_not_doubly_robust(law: laws.RepeatedLaw) -> None:
    """Saturated ``m`` with ``W``-only ``pi`` and ``omega`` reaches its limit, which is not the truth.

    The limit is the cell-mean form of the one-step estimator.  The control separates the
    cause: a ``W``-only ``omega`` with a saturated ``pi`` is exact, so the bias comes from the
    treatment mechanism and not from ``m`` or ``omega``.
    """
    wrong = ipsi_fit(law, treatment=Baseline(), censoring=Baseline())
    only_pi = ipsi_fit(law, treatment=Baseline())
    only_omega = ipsi_fit(law, censoring=Baseline())
    rows = law.first_row_of()
    for delta, (bias, pi_bias) in NOT_DOUBLY_ROBUST_BIAS[law.name].items():
        truth = law.tilt_functional(law.probs, delta, 2)
        limit, curve = law.ipsi_population(delta, 2, key="full", pi_key=["W"], omega_key=["W"])
        pi_limit, _ = law.ipsi_population(delta, 2, key="full", pi_key=["W"])
        assert limit - truth == pytest.approx(bias, abs=6e-7)
        assert pi_limit - truth == pytest.approx(pi_bias, abs=6e-7)
        assert min(abs(bias), abs(pi_bias)) > 2e-4
        assert wrong.psi(_mean(delta, 2)) == pytest.approx(limit, abs=1e-11)
        np.testing.assert_allclose(
            wrong.influence_curves[_mean(delta, 2)][rows], curve, atol=1e-10, rtol=0
        )
        assert only_pi.psi(_mean(delta, 2)) == pytest.approx(pi_limit, abs=1e-11)
        assert only_omega.psi(_mean(delta, 2)) == pytest.approx(truth, abs=1e-12)
    # At h = 1 the history is W, so a W-only treatment mechanism is saturated and exact.
    for delta in CONTRASTED:
        truth = law.tilt_functional(law.probs, delta, 1)
        assert wrong.psi(_mean(delta, 1)) == pytest.approx(truth, abs=1e-12)


@pytest.fixture(scope="module")
def ipsi_banded() -> Any:
    return ipsi_fit(laws.BINARY, simultaneous=True, n_multiplier=MULTIPLIERS, random_state=0)


def check_e8(fit: Any) -> None:
    """E8: the fit's critical value is the multiplier band over the stacked curves.

    Rebuilt beside the fit with the same seed.  A band built from curves made independent
    across horizons has a different critical value, so a change in the covariance fails here.
    """
    estimates = {name: fit.estimates[name] for name in fit.estimates}
    joint = simultaneous_bands(estimates, alpha=0.05, n_replicates=MULTIPLIERS, random_state=0)
    assert fit.simultaneous.critical_value == pytest.approx(joint.critical_value, abs=0), (
        "critical value"
    )
    order = np.random.default_rng(1).permutation(laws.N)
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


def test_e8_the_ipsi_horizons_are_correlated(ipsi_banded: Any) -> None:
    """The horizons of one tilt share units, and the correlation is not zero."""
    for delta in CONTRASTED:
        first = ipsi_banded.influence_curves[_mean(delta, 1)]
        second = ipsi_banded.influence_curves[_mean(delta, 2)]
        assert abs(np.corrcoef(first, second)[0, 1]) > 0.05, delta


def test_e8_the_ipsi_band_is_built_from_the_stacked_curves(ipsi_banded: Any) -> None:
    check_e8(ipsi_banded)


# ---------------------------------------------------------------- SDR: E1 to E3, E2b and E8

#: The plans of the exact witnesses, as the laws declare them.
SDR_PLANS = ("never", "always", "treat_if_high")


class NodeOne(laws.CellMeans):
    """The saturated learner, except at node 1, where it keeps the columns ``keep`` only.

    The mechanisms of a repeated fit see a design whose width names the node: the treatment
    design holds 1 column at node 1 (``W``) and 4 at node 2, and the censoring design holds 2
    (``W``, ``A_1``) and 5.  ``width`` is the node-1 width and ``keep`` the columns that
    remain there, so an empty ``keep`` is an intercept.
    """

    width: int = 1
    keep: tuple[int, ...] = ()

    def _select(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        if matrix.shape[1] != self.width:
            return matrix
        return matrix[:, list(self.keep)] if self.keep else np.zeros((matrix.shape[0], 1))

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> NodeOne:
        super().fit(self._select(X), y, sample_weight=sample_weight)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return super().predict(self._select(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return super().predict_proba(self._select(X))


class InterceptAtNodeOne(NodeOne):
    """A treatment mechanism that is an intercept at node 1 and saturated at node 2."""

    width = 1
    keep = ()


class ArmAtNodeOne(NodeOne):
    """A censoring mechanism that reads ``A_1`` alone at node 1 and is saturated at node 2."""

    width = 2
    keep = (1,)


def sdr_fit(
    law: laws.RepeatedLaw,
    *,
    frame: Any = None,
    outcome: Any = None,
    treatment: Any = None,
    censoring: Any = None,
    roles: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """``SDR`` on the law's exact sample, with a saturated learner unless one is handed in."""
    settings: dict[str, Any] = {
        "reference": laws.REGIMEN_REFERENCE,
        "outcome_learner": outcome or laws.CellMeans(),
        "pseudo_learner": outcome or laws.CellMeans(),
        "treatment_learner": treatment or laws.CellMeans(),
        "censoring_learner": censoring or laws.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    regimens = settings.pop("regimens", declared_regimens(law.regimen_spec()))
    return SDR(regimens, **settings).fit(
        law.frame() if frame is None else frame, **({**COLUMNS, **(roles or {})})
    )


def known_mechanisms(law: laws.RepeatedLaw) -> tuple[Any, dict[str, Any]]:
    """The law's frame with its mechanism columns, and the roles that declare them.

    ``g{t}_{a}`` is :math:`P(A_t = a \\mid H_t)` and ``r{t}`` is :math:`P(C_t = 1 \\mid H_t,
    A_t)` at the arm the row holds.  A row that does not reach node ``t`` holds missing values
    there, as the fit expects.  The tables are the ones the law samples from.
    """
    frame = law.frame()
    high = np.array([-1 if np.isnan(y) else law.high(law.y1_values.index(y)) for y in frame["Y1"]])
    node_2 = (high >= 0) & frame["L2"].notna().to_numpy()
    w, a1, l2, a2 = (
        frame[name].fillna(0).to_numpy().astype(int) for name in ("W", "A1", "L2", "A2")
    )
    g1 = laws.G1[w]
    g2 = np.full(len(frame), np.nan)
    g2[node_2] = laws.G2[w[node_2], a1[node_2], high[node_2], l2[node_2]]
    r2 = np.full(len(frame), np.nan)
    r2[node_2] = laws.C2[w[node_2], a1[node_2], high[node_2], l2[node_2], a2[node_2]]
    frame = frame.assign(g1_1=g1, g1_0=1.0 - g1, g2_1=g2, g2_0=1.0 - g2, r1=laws.C1[w, a1], r2=r2)
    roles = {
        "treatment_probabilities": {
            "A1": {0.0: "g1_0", 1.0: "g1_1"},
            "A2": {0.0: "g2_0", 1.0: "g2_1"},
        },
        "censoring_probabilities": {"C1": "r1", "C2": "r2"},
    }
    return frame, roles


def _sdr_mean(plan: str, horizon: int) -> str:
    return f"ey_regimen[{plan} @ t={horizon}]"


#: A reported cell of an ``SDR`` fit: a plan and a horizon.
Cell = tuple[str, int]
SDR_CELLS: tuple[Cell, ...] = tuple((plan, h) for plan in SDR_PLANS for h in laws.HORIZONS)


def check_e1(law: laws.RepeatedLaw, fit: Any, names: Any = None) -> None:
    """E1: the estimate is the g-formula, and the curve its complex-step Gateaux derivative.

    The derivative is taken of the g-formula on the cell probabilities, so nothing the library
    supplies enters it.  The check runs over every reported mean and contrast, or over
    ``names``.
    """
    rows = law.first_row_of()
    for name in law.names if names is None else names:
        assert fit.psi(name) == pytest.approx(law.truth[name], abs=1e-12), f"estimate: {name}"
        np.testing.assert_allclose(
            fit.influence_curves[name][rows],
            law.eif(name),
            atol=1e-11,
            rtol=0,
            err_msg=f"curve: {name}",
        )


def check_e2(law: laws.RepeatedLaw, fit: Any, cells: Any = SDR_CELLS) -> None:
    """E2: with ``W``-only regressions and known mechanisms the estimate is the truth.

    The correction carries the whole gap that the ``W``-only plug-in leaves at ``h = 2``.
    """
    for plan, horizon in cells:
        name = _sdr_mean(plan, horizon)
        assert fit.psi(name) == pytest.approx(law.truth[name], abs=1e-12), f"estimate: {name}"


def check_e2b(law: laws.RepeatedLaw, fit: Any) -> None:
    """E2b: a wrong node-2 regression with the true ratio, and a wrong node-1 ratio.

    The node-1 regression is saturated in ``W`` and so correct for its pseudo-outcome.
    Sequential double robustness (Díaz et al. Lemma 4) makes the estimate exact.
    """
    check_e2(law, fit)


def check_e3(law: laws.RepeatedLaw, fit: Any, cells: Any = SDR_CELLS) -> None:
    """E3: the curve is an independent law-side transform ``phi_1(m-bar, r) - psi``.

    The transform is written on the support points from the known tables, at the nuisances of
    E2, for each cell and for its contrast against the reference.
    """
    rows = law.first_row_of()
    for plan, horizon in cells:
        _, curve = law.sdr_population(plan, horizon, key_1=["W"], key_2=["W"])
        name = _sdr_mean(plan, horizon)
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], curve, atol=1e-11, rtol=0, err_msg=f"curve: {name}"
        )
        if plan != laws.REGIMEN_REFERENCE:
            _, reference = law.sdr_population(
                laws.REGIMEN_REFERENCE, horizon, key_1=["W"], key_2=["W"]
            )
            name = f"ate_regimen[{plan} vs {laws.REGIMEN_REFERENCE} @ t={horizon}]"
            np.testing.assert_allclose(
                fit.influence_curves[name][rows],
                curve - reference,
                atol=1e-11,
                rtol=0,
                err_msg=f"curve: {name}",
            )


@pytest.fixture(scope="module", params=LAWS, ids=[law.name for law in LAWS])
def sdr_law_and_fit(request: pytest.FixtureRequest) -> tuple[laws.RepeatedLaw, Any]:
    law = request.param
    return law, sdr_fit(law)


def test_e1_saturated_regressions_and_estimated_mechanisms_give_the_g_formula_and_its_derivative(
    sdr_law_and_fit: tuple[laws.RepeatedLaw, Any],
) -> None:
    """``SDR`` with saturated regressions and estimated mechanisms is exact at every horizon.

    The point check alone is vacuous for the correction, because a saturated regression
    leaves nothing to correct in sample.  The derivative check is not: the curve is the
    transform, and a wrong correction term moves it.  E2 and E3 carry the correction.
    """
    law, fit = sdr_law_and_fit
    assert set(fit) == set(law.names)
    check_e1(law, fit)


@pytest.fixture(scope="module", params=LAWS, ids=[law.name for law in LAWS])
def sdr_known_fit(request: pytest.FixtureRequest) -> tuple[laws.RepeatedLaw, Any]:
    """``W``-only regressions at both nodes, with the law's mechanisms declared known."""
    law = request.param
    frame, roles = known_mechanisms(law)
    return law, sdr_fit(law, frame=frame, roles=roles, outcome=Baseline())


def test_e2_w_only_regressions_with_known_mechanisms_give_the_truth_and_the_plug_in_does_not(
    sdr_known_fit: tuple[laws.RepeatedLaw, Any],
) -> None:
    """The correction carries the whole gap that the ``W``-only plug-in leaves at ``h = 2``.

    The precondition is a declared nonzero gap per plan.  At ``h = 1`` the node-1 regression
    on ``W`` is saturated, so the gap is zero by construction and E3 carries that horizon.
    """
    law, fit = sdr_known_fit
    check_e2(law, fit)
    for plan, declared in zip(SDR_PLANS, SDR_PLUG_IN_GAP[law.name], strict=True):
        plug_in, _ = law.sdr_population(plan, 2, key_1=["W"], key_2=["W"], scale=0.0)
        gap = plug_in - law.truth[_sdr_mean(plan, 2)]
        assert gap == pytest.approx(declared, abs=6e-7), plan
        assert abs(gap) > 1e-3, plan
        assert abs(fit.psi(_sdr_mean(plan, 2)) - plug_in) > 1e-3, plan


def test_e3_the_curve_is_the_law_side_transform(
    sdr_known_fit: tuple[laws.RepeatedLaw, Any],
) -> None:
    """The curve under E2 is the transform ``phi_1 - psi``, which is not the efficient curve."""
    law, fit = sdr_known_fit
    check_e3(law, fit)
    rows = law.first_row_of()
    for plan in SDR_PLANS:
        reported = fit.influence_curves[_sdr_mean(plan, 2)][rows]
        assert np.max(np.abs(reported - law.eif(_sdr_mean(plan, 2)))) > 1e-2, plan


@pytest.mark.parametrize("law", LAWS, ids=[law.name for law in LAWS])
def test_e2b_sequential_double_robustness_at_a_mixed_pattern(law: laws.RepeatedLaw) -> None:
    """Node 2 wrong with the true ratio, node 1 right with a wrong ratio: still the truth.

    The node-2 regression reads ``W`` alone.  The node-1 mechanisms are an intercept
    (treatment) and the arm alone (censoring), which are wrong, and the node-1 regression is
    saturated in ``W``.  The estimator that regresses node 1 on ``m_2`` and not on
    ``phi_2`` misses by a declared nonzero amount at the same nuisances.
    """
    fit = sdr_fit(
        law,
        outcome=Baseline(),
        treatment=InterceptAtNodeOne(),
        censoring=ArmAtNodeOne(),
    )
    check_e2b(law, fit)
    options: dict[str, Any] = {"key_1": ["W"], "key_2": ["W"], "ratio_1": "intercept"}
    for plan, declared in zip(SDR_PLANS, SDR_NON_SEQUENTIAL_GAP[law.name], strict=True):
        sequential, curve = law.sdr_population(plan, 2, **options)
        control, _ = law.sdr_population(plan, 2, sequential=False, **options)
        assert fit.psi(_sdr_mean(plan, 2)) == pytest.approx(sequential, abs=1e-12), plan
        # The curve is the transform at the wrong node-1 ratio, so the fit holds that ratio.
        np.testing.assert_allclose(
            fit.influence_curves[_sdr_mean(plan, 2)][law.first_row_of()],
            curve,
            atol=1e-11,
            rtol=0,
            err_msg=f"curve: {plan}",
        )
        assert control - law.truth[_sdr_mean(plan, 2)] == pytest.approx(declared, abs=6e-7), plan
        assert abs(declared) > 1e-4, plan
        assert abs(fit.psi(_sdr_mean(plan, 2)) - control) > 1e-4, plan


def test_e2b_the_mixed_pattern_has_a_wrong_ratio_at_node_one() -> None:
    """The precondition: the node-1 ratio of the mixed pattern differs from the tabled one."""
    law = laws.BINARY
    wrong, _ = law.sdr_population("never", 1, key_1=["W"], key_2=["W"], ratio_1="intercept")
    plug_in, _ = law.sdr_population("never", 1, key_1=["W"], key_2=["W"], scale=0.0)
    # The correction at h = 1 vanishes in sample whatever the ratio, because the node-1
    # regression is saturated in W: both are the truth.  The ratios differ in the curve.
    _, wrong_curve = law.sdr_population("never", 1, key_1=["W"], key_2=["W"], ratio_1="intercept")
    _, true_curve = law.sdr_population("never", 1, key_1=["W"], key_2=["W"])
    assert wrong == pytest.approx(plug_in, abs=1e-12)
    assert np.max(np.abs(wrong_curve - true_curve)) > 1e-2


@pytest.fixture(scope="module")
def sdr_banded() -> Any:
    return sdr_fit(laws.BINARY, simultaneous=True, n_multiplier=MULTIPLIERS, random_state=0)


def test_e8_the_sdr_horizons_are_correlated(sdr_banded: Any) -> None:
    """The horizons of one plan share units, and the correlation is not zero."""
    for plan in SDR_PLANS:
        first = sdr_banded.influence_curves[_sdr_mean(plan, 1)]
        second = sdr_banded.influence_curves[_sdr_mean(plan, 2)]
        assert abs(np.corrcoef(first, second)[0, 1]) > 0.05, plan


def test_e8_the_sdr_band_is_built_from_the_stacked_curves(sdr_banded: Any) -> None:
    check_e8(sdr_banded)
