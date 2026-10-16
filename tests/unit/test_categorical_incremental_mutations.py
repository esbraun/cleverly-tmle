r"""Deliberate-mutation controls of the categorical incremental fit.

An exact-law check is blind to a term that vanishes at the truth.  Each control here breaks one
line of the shipped source, through :func:`tests.unit._source_mutation.mutate`, and asks one
witness to fail.  The unmutated source passes every witness first, so a failure is the
mutation's and not the witness's.

A witness is a **gap**: a distance between what the fit reports and what the oracle or the
estimator's own equations say it must report.  The shipped source leaves every gap below its
``TOLERANCE``.  A mutated source must run to the end and leave the gap of the named witness above
``MARGIN``, so a mutant that crashes, or that moves a value by less than a rounding error, fails
the control and does not pass it.

=====  ==========================================================  =========================
Id     The broken line                                             The witness that fails
=====  ==========================================================  =========================
M1     the mechanism term of the influence curve is dropped        C1 curve
M2     the mechanism term sees the indicators of two of three      C1 curve
       levels; the mechanism covariate of the third is zero        C1t identity
M3     the covariate at the received level uses the largest        C1 curve, C1t identity
       factor instead of that level's own
M4     one level's factor is left out of the normaliser :math:`D`  C1 estimate
M5     the mechanism update is one-versus-rest, one shared         C1t simplex
       :math:`\epsilon`, and unnormalised
M5a    the mechanism update is one-versus-rest, one                C1t shape
       :math:`\epsilon` per arm (armwise), and unnormalised
M5b    the softmax solver drops its row normalisation              C1t simplex
M6     the mechanism covariate is not centred at :math:`m`         C1t identity, covariate
M9     a known mechanism keeps the mechanism term                  C1k curve
M10    a mapping key is resolved by declaration order, not label   C1 curve
M11    the two-level branch ignores the factor of the first level  C5 equality
=====  ==========================================================  =========================

Why M5 is read at the simplex and the shape and not at the C1t identity.  The identity
"bias equals the remainder at the targeted pair" holds for every pair that solves the two
estimating equations, by whichever route, and the tilt :math:`q = g f / (g^\top f)` does not
change when a row of :math:`g` is rescaled.  The armwise update has one coefficient per arm and
solves a strictly larger system; on the three-cell law it saturates :math:`g` (identity gap
2e-14, bias moved from -9.1e-3 to -2e-14).  The shared one-versus-rest update leaves rows off
the simplex by 1.1e-3 and the identity at 1e-10.  What the softmax solver adds that the identity
cannot see is the simplex and the single :math:`\epsilon` shared by the arms, and those are the
witnesses.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import cleverly.estimators.targeting as targeting_module
import cleverly.fluctuation.mechanism as mechanism_module
import cleverly.fluctuation.submodel as submodel_module
import cleverly.inference.influence as influence_module
import cleverly.interventions.incremental as incremental_module
import cleverly.targets.base as targets_module
from cleverly.data import CausalData
from cleverly.fluctuation.iterative import InitialFit
from cleverly.interventions import Incremental, IPSISet
from tests import discrete_law_categorical_incremental as law
from tests import discrete_law_multi as base
from tests.unit import _categorical_incremental_support as sup
from tests.unit._source_mutation import mutate

NAMES = law.MEANS + law.CONTRASTS
#: A mutated fit must leave the gap of its witness at least this large: a wrong value and not a
#: rounding error.
MARGIN = 1e-4

SOFTMAX_RETURN = "    return solve_multinomial_mechanism(treatment, tilt.propensity, covariate, weights, tol=tol)\n"

#: ``solve_mechanism`` once over the ``K`` stacked responses: the coefficient is shared by the
#: arms and the rows of the result are not normalised.
_SHARED_ONE_VERSUS_REST = """    import dataclasses

    codes = np.rint(np.asarray(treatment, dtype=float).reshape(-1))
    k = tilt.n_arms
    stacked = solve_mechanism(
        np.concatenate([(codes == column).astype(float) for column in range(k)]),
        np.concatenate([tilt.propensity[:, column] for column in range(k)]),
        np.concatenate([covariate[:, column, :] for column in range(k)]),
        np.tile(np.asarray(weights, dtype=float).reshape(-1), k),
        tol=tol,
    )
    return dataclasses.replace(
        stacked, propensity=stacked.propensity.reshape(k, codes.size).T
    )
"""

#: ``solve_mechanism`` once per arm, with ``1(A = a)`` as the response, as the armwise DR-TMLE
#: mechanism does: a coefficient per arm and unnormalised rows.
_ARMWISE = """    import dataclasses

    codes = np.rint(np.asarray(treatment, dtype=float).reshape(-1))
    solved = [
        solve_mechanism(
            (codes == column).astype(float),
            tilt.propensity[:, column],
            covariate[:, column, :],
            weights,
            tol=tol,
        )
        for column in range(tilt.n_arms)
    ]
    return dataclasses.replace(
        solved[0],
        propensity=np.column_stack([fit.propensity for fit in solved]),
        epsilon=np.concatenate([fit.epsilon for fit in solved]),
    )
"""


def c1_curve() -> float:
    """C1: the largest distance of an estimate or a curve of the exact fit from the oracle."""
    fit = sup.exact_fit()
    gap = 0.0
    for name in NAMES:
        reported = sup.at_support(fit.estimates[name].influence_curve)
        gap = max(
            gap,
            abs(fit.estimates[name].psi - law.truth_of(name)),
            float(np.max(np.abs(reported - law.eif_of(name)))),
        )
    return gap


def c1_estimate() -> float:
    """C1, the estimates alone."""
    fit = sup.exact_fit()
    return max(abs(fit.estimates[name].psi - law.truth_of(name)) for name in NAMES)


def c1k_curve() -> float:
    """C1k: a declared known mechanism has the curve without the mechanism term."""
    fit = sup.exact_fit(known=True)
    gap = 0.0
    for name in law.MEANS:
        tilt = law.TILTS[name[len("ey_ipsi[") : -1]]
        expected = law.eif(tilt) - law.mechanism_term(tilt)
        reported = sup.at_support(fit.estimates[name].influence_curve)
        gap = max(gap, float(np.max(np.abs(reported - expected))))
    return gap


def c1t_simplex() -> float:
    """C1t: the targeted mechanism ``g*`` is a simplex, to the largest row-sum error."""
    g_star, _ = sup.targeted(sup.weak_fit(), False)
    return float(np.max(np.abs(g_star.sum(axis=1) - 1.0)))


def c1t_identity() -> float:
    """C1t: the weak-learner fit is biased by exactly the remainder, to the largest gap."""
    fit = sup.weak_fit()
    everyone = np.ones(fit.data.n, dtype=bool)
    gap = 0.0
    for name in ("T_a", "T_b"):
        bias = fit.estimates[f"ey_ipsi[{name}]"].psi - law.truth_of(f"ey_ipsi[{name}]")
        closed = sup.remainder_at(fit, law.TILTS[name], False, everyone)
        gap = max(gap, abs(bias - closed))
    return gap


def c1t_shape() -> float:
    """C1t: the mechanism step has one coefficient per tilt, shared by the arms."""
    fit = sup.weak_fit()
    width = len(fit.fluctuations["ipsi"].mechanism.epsilon)
    return float(abs(width - fit.nuisance.incremental.n_tilts))


def covariate_is_centred() -> float:
    """The mechanism covariate has mean zero under the mechanism it is built at."""
    fit = sup.exact_fit()
    tilts = fit.nuisance.incremental
    arms = np.column_stack([fit.nuisance.outcome.arms[float(c)] for c in range(tilts.n_arms)])
    initial = InitialFit(
        observed=arms[np.arange(arms.shape[0]), np.rint(fit.data.treatment).astype(int)],
        arms={float(c): arms[:, c] for c in range(tilts.n_arms)},
    )
    covariate = mechanism_module.ipsi_mechanism_covariate(initial, tilts)
    expected = np.einsum("ij,ijr->ir", np.asarray(tilts.propensity), covariate)
    assert np.max(np.abs(covariate)) > 0.05, "a zero covariate hides a missing centring"
    return float(np.max(np.abs(expected)))


def c5_equality() -> float:
    """C5: ``{0: 2, 1: 4}`` on two levels is the odds multiplier ``2``."""
    frame = sup.frame().assign(B=lambda f: (f["A"] == "high").astype(float))

    data = CausalData.from_frame(frame, outcome="Y", treatment="B", covariates=["W"])
    treated = np.array([0.55, 0.30, 0.15])[frame["W"].to_numpy().astype(int)]
    g = np.column_stack([1.0 - treated, treated])
    reference = IPSISet.evaluate((Incremental(2.0, name="t"),), data, g)
    other = IPSISet.evaluate((Incremental({0.0: 2.0, 1.0: 4.0}, name="t"),), data, g)
    return max(
        float(np.max(np.abs(other.values - reference.values))),
        float(np.max(np.abs(other.weights - reference.weights))),
    )


#: ``(witness, TOLERANCE)``: the shipped source leaves the gap at or below the tolerance.
WITNESSES: dict[str, tuple[Callable[[], float], float]] = {
    "c1_curve": (c1_curve, 1e-10),
    "c1_estimate": (c1_estimate, 1e-10),
    "c1k_curve": (c1k_curve, 1e-12),
    "c1t_simplex": (c1t_simplex, 1e-12),
    "c1t_identity": (c1t_identity, 1e-8),
    "c1t_shape": (c1t_shape, 0.0),
    "covariate_is_centred": (covariate_is_centred, 1e-12),
    "c5_equality": (c5_equality, 1e-12),
}


@pytest.mark.parametrize("witness", WITNESSES)
def test_every_witness_passes_on_the_shipped_source(witness: str) -> None:
    gap, tolerance = WITNESSES[witness][0](), WITNESSES[witness][1]
    assert gap <= tolerance, f"{witness}: gap {gap:.3e}"


def _fails(*names: str, margin: float = MARGIN) -> None:
    """Each named witness runs on the broken source and leaves a gap above ``margin``.

    An exception from the broken source is not a failure of the witness: it propagates, and
    the control fails with it.  A mutant that crashes shows nothing about the value it moves.
    """
    for name in names:
        gap = WITNESSES[name][0]()
        assert gap > margin, f"{name}: gap {gap:.3e} does not clear {margin:.0e}"


class TestTheMutationsAreSeen:
    def test_m1_dropping_the_mechanism_term(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            influence_module,
            influence_module,
            "_ipsi_term_general",
            [("* (at_observed - mixture))", "* (at_observed - mixture) * 0.0)")],
        )
        _fails("c1_curve")

    def test_m2_the_term_sees_two_of_three_levels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            influence_module,
            influence_module,
            "_ipsi_term_general",
            [("* (at_observed - mixture))", "* (at_observed - mixture) * (np.rint(a) < 2))")],
        )
        _fails("c1_curve")

    def test_m2_the_covariate_of_the_third_level_is_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            mechanism_module,
            mechanism_module,
            "_covariate_general",
            [
                (
                    "    return np.asarray(np.asarray(incremental.weights, dtype=float) * centred",
                    "    centred[:, 2, :] = 0.0\n"
                    "    return np.asarray(np.asarray(incremental.weights, dtype=float) * centred",
                )
            ],
        )
        _fails("c1t_identity")

    def test_m3_the_outcome_covariate_at_the_received_level_uses_the_largest_factor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            submodel_module,
            submodel_module,
            "ipsi_submodel",
            [
                (
                    'observed = np.einsum("ij,ijr->ir", indicator, weights)',
                    "observed = weights.max(axis=1)",
                )
            ],
        )
        monkeypatch.setitem(
            submodel_module.SUBMODEL_BUILDERS, "ipsi", submodel_module.ipsi_submodel
        )
        _fails("c1_curve", "c1t_identity")

    def test_m3_the_mechanism_term_at_the_received_level_uses_the_largest_factor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            incremental_module,
            incremental_module.IPSISet,
            "observed",
            [
                (
                    'np.einsum("ij,ijr->ir", indicator, self.weights)',
                    "self.weights.max(axis=1)",
                )
            ],
        )
        _fails("c1_curve")

    def test_m4_one_level_is_left_out_of_the_normaliser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            incremental_module,
            incremental_module,
            "_tilt_general",
            [("d = g @ f", "d = g[:, :2] @ f[:2] + g[:, 2]")],
        )
        _fails("c1_estimate")

    def test_m5_the_update_is_one_versus_rest_with_one_shared_coefficient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One logistic over the ``K`` stacked responses ``1(A = a)``: no row normalisation."""
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_solve_tilt_mechanism",
            [(SOFTMAX_RETURN, _SHARED_ONE_VERSUS_REST)],
        )
        _fails("c1t_simplex")

    def test_m5a_the_update_is_armwise_with_a_coefficient_per_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pattern of ``solve_armwise_bounded_mechanism``: one logistic per arm."""
        mutate(
            monkeypatch,
            targeting_module,
            targeting_module,
            "_solve_tilt_mechanism",
            [(SOFTMAX_RETURN, _ARMWISE)],
        )
        _fails("c1t_shape")

    def test_m5b_the_softmax_solver_drops_its_row_normalisation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rows leave the simplex by 5.0e-5.  The margin is 1e-5, not ``MARGIN``.

        This mutant rescales each row of ``g*`` and nothing else: ``q`` and the estimate do not
        move (gap 4e-11 on the C1t identity), so the simplex is the only witness, and its
        shipped value is 1e-16.  The gap is five orders above the tolerance of the witness.
        """
        mutate(
            monkeypatch,
            mechanism_module,
            targeting_module,
            "solve_multinomial_mechanism",
            [
                (
                    "return np.asarray(p / p.sum(axis=1, keepdims=True), dtype=float)",
                    "return np.asarray(np.exp(log_g + h @ epsilon), dtype=float)",
                )
            ],
        )
        _fails("c1t_simplex", margin=1e-5)

    def test_m6_the_mechanism_covariate_is_not_centred(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            mechanism_module,
            mechanism_module,
            "_covariate_general",
            [("predictions[:, :, None] - mixture[:, None, :]", "predictions[:, :, None]")],
        )
        _fails("c1t_identity", "covariate_is_centred")

    def test_m9_a_known_mechanism_keeps_the_mechanism_term(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            influence_module,
            targets_module,
            "ipsi_means",
            [("if incremental.mechanism_known", "if False and incremental.mechanism_known")],
        )
        _fails("c1k_curve")

    def test_m10_a_key_is_resolved_by_declaration_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        declared = base.LABELS
        mutate(
            monkeypatch,
            incremental_module,
            incremental_module.Incremental,
            "factors",
            [
                (
                    "position = {label: index for index, label in reversed(list(enumerate(labels)))}",
                    f"position = {{label: index for index, label in enumerate({declared!r})}}",
                )
            ],
        )
        _fails("c1_curve")

    def test_m11_the_two_level_branch_ignores_the_first_factor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            incremental_module,
            incremental_module,
            "_tilt_binary",
            [("delta = entry[1] / entry[0]", "delta = entry[1]")],
        )
        _fails("c5_equality")


def test_the_mutated_modules_are_restored() -> None:
    """``monkeypatch`` put the shipped functions back after each control."""
    for witness, tolerance in WITNESSES.values():
        assert witness() <= tolerance
