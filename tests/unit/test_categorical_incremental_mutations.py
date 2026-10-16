r"""Deliberate-mutation controls of the categorical incremental fit.

An exact-law check is blind to a term that vanishes at the truth.  Each control here breaks one
line of the shipped source, through :func:`tests.unit._source_mutation.mutate`, and asks one
witness to fail.  The unmutated source passes every witness first, so a failure is the
mutation's and not the witness's.

=====  ==========================================================  =====================
Id     The broken line                                             The witness that fails
=====  ==========================================================  =====================
M1     the mechanism term of the influence curve is dropped        C1 curve
M2     the mechanism term sees the indicators of two of three      C1 curve
       levels; the mechanism covariate of the third is zero        C1t bias
M3     the covariate at the received level uses the largest        C1 curve, C1t bias
       factor instead of that level's own
M4     one level's factor is left out of the normaliser :math:`D`  C1 estimate
M5     the mechanism update is one-versus-rest and unnormalised    C1t bias and simplex
M6     the mechanism covariate is not centred at :math:`m`         C1t bias, covariate
M9     a known mechanism keeps the mechanism term                  C1k curve
M10    a mapping key is resolved by declaration order, not label   C1 curve
M11    the two-level branch ignores the factor of the first level  C5 equality
=====  ==========================================================  =====================
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
from tests.unit._declaration_support import assert_every_witness_fails
from tests.unit._source_mutation import mutate

NAMES = law.MEANS + law.CONTRASTS
#: A bias of this size is a failure of the targeting step, not a rounding error.
BIAS = 1e-6


def c1_curve() -> None:
    """C1: every estimate and every curve of the exact fit equals the oracle."""
    fit = sup.exact_fit()
    for name in NAMES:
        assert fit.estimates[name].psi == pytest.approx(law.truth_of(name), abs=1e-10), name
        reported = sup.at_support(fit.estimates[name].influence_curve)
        np.testing.assert_allclose(reported, law.eif_of(name), atol=1e-10, rtol=0, err_msg=name)


def c1_estimate() -> None:
    """C1, the estimates alone."""
    fit = sup.exact_fit()
    for name in NAMES:
        assert fit.estimates[name].psi == pytest.approx(law.truth_of(name), abs=1e-10), name


def c1k_curve() -> None:
    """C1k: a declared known mechanism has the curve without the mechanism term."""
    fit = sup.exact_fit(known=True)
    for name in law.MEANS:
        tilt = law.TILTS[name[len("ey_ipsi[") : -1]]
        expected = law.eif(tilt) - law.mechanism_term(tilt)
        reported = sup.at_support(fit.estimates[name].influence_curve)
        np.testing.assert_allclose(reported, expected, atol=1e-12, rtol=0, err_msg=name)


def c1t_bias() -> None:
    """C1t: the weak-learner fit is biased by exactly the remainder, and ``g*`` is a simplex."""
    fit = sup.weak_fit()
    g_star, _ = sup.targeted(fit, False)
    np.testing.assert_allclose(g_star.sum(axis=1), 1.0, atol=1e-12, rtol=0)
    everyone = np.ones(fit.data.n, dtype=bool)
    for name in ("T_a", "T_b"):
        bias = fit.estimates[f"ey_ipsi[{name}]"].psi - law.truth_of(f"ey_ipsi[{name}]")
        closed = sup.remainder_at(fit, law.TILTS[name], False, everyone)
        assert abs(bias - closed) < BIAS, f"{name}: bias {bias:.3e}, remainder {closed:.3e}"


def covariate_is_centred() -> None:
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
    np.testing.assert_allclose(expected, 0.0, atol=1e-12, rtol=0)
    assert np.max(np.abs(covariate)) > 0.05, "a zero covariate hides a missing centring"


def c5_equality() -> None:
    """C5: ``{0: 2, 1: 4}`` on two levels is the odds multiplier ``2``."""
    frame = sup.frame().assign(B=lambda f: (f["A"] == "high").astype(float))

    data = CausalData.from_frame(frame, outcome="Y", treatment="B", covariates=["W"])
    treated = np.array([0.55, 0.30, 0.15])[frame["W"].to_numpy().astype(int)]
    g = np.column_stack([1.0 - treated, treated])
    reference = IPSISet.evaluate((Incremental(2.0, name="t"),), data, g)
    other = IPSISet.evaluate((Incremental({0.0: 2.0, 1.0: 4.0}, name="t"),), data, g)
    np.testing.assert_allclose(other.values, reference.values, atol=1e-12, rtol=0)
    np.testing.assert_allclose(other.weights, reference.weights, atol=1e-12, rtol=0)


WITNESSES: dict[str, Callable[[], None]] = {
    "c1_curve": c1_curve,
    "c1_estimate": c1_estimate,
    "c1k_curve": c1k_curve,
    "c1t_bias": c1t_bias,
    "covariate_is_centred": covariate_is_centred,
    "c5_equality": c5_equality,
}


@pytest.mark.parametrize("witness", WITNESSES)
def test_every_witness_passes_on_the_shipped_source(witness: str) -> None:
    WITNESSES[witness]()


def _fails(*names: str) -> None:
    assert_every_witness_fails(WITNESSES[name] for name in names)


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
        _fails("c1t_bias")

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
        _fails("c1_curve", "c1t_bias")

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

    def test_m5_the_mechanism_update_is_unnormalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
        _fails("c1t_bias")

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
        _fails("c1t_bias", "covariate_is_centred")

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
    for witness in WITNESSES.values():
        witness()
