r"""Deliberate-mutation controls of the categorical incremental fit over time.

An exact-law check is blind to a term that vanishes at the truth.  Each control here breaks one
line of the shipped source, through :func:`tests.unit._source_mutation.mutate`, and asks one
witness to fail.  The unmutated source passes every witness first, so a failure is the
mutation's and not the witness's.

=====  ==========================================================  ===========================
Id     The broken line                                             The witness that fails
=====  ==========================================================  ===========================
M1     the mechanism term of a node is dropped                     C2 curve
M2     the term and the factor read two of three levels            C2 curve
M3     the factor at the received level is the largest factor      C2 curve
M4     one level's factor is left out of the normaliser :math:`D`  C2 curve
M7     node 2 reads node 1's factors                               C2 curve
M8     the censoring factor is dropped from :math:`W_t`            C2 curve
M9     a known mechanism keeps the mechanism term                  known curve
M10    a mapping key is resolved by declaration order              C2 curve
M11    the two-level branch ignores the factor of the first level  C5 equality
=====  ==========================================================  ===========================
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import cleverly.interventions.incremental as incremental_module
from tests import discrete_law_longitudinal_categorical_incremental as law
from tests.unit import _categorical_ipsi_support as sup
from tests.unit._source_mutation import mutate

#: ``cleverly.longitudinal.ipsi`` as an attribute of its package is the ``ipsi`` function.
ipsi_module = importlib.import_module("cleverly.longitudinal.ipsi")
regimen_module = importlib.import_module("cleverly.longitudinal.regimen")

NAMES = law.names(law.PLANS, law.REFERENCE)


def _fit_end() -> Any:
    return sup.fit_ipsi(law.end_frame(), law.PLANS)


def c2_curve() -> None:
    """C2: every estimate and curve of the censored three-level fit equals the oracle."""
    fit = _fit_end()
    for name, truth, eif, rows in sup.expected(NAMES, law.end_function, law.PROBS, law.COUNTS):
        assert fit.psi(name) == pytest.approx(truth, abs=1e-10), name
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], eif, atol=1e-8, rtol=0, err_msg=name
        )


def c2_estimate() -> None:
    """C2, the estimates alone."""
    fit = _fit_end()
    for name, truth, _, _ in sup.expected(NAMES, law.end_function, law.PROBS, law.COUNTS):
        assert fit.psi(name) == pytest.approx(truth, abs=1e-10), name


def known_curve() -> None:
    """A declared mechanism at both nodes: the curve has no mechanism term."""
    fit = sup.fit_ipsi(
        sup.known_frame(), law.PLANS, sup.UNCENSORED_COLUMNS, declared=sup.KNOWN_DECLARED
    )
    probs, counts = law.UNCENSORED_PROBS, law.UNCENSORED_COUNTS
    rows = law.first_rows(counts)
    keep = probs > 0
    for name in NAMES:

        def at(p: Any, name: str = name) -> Any:
            return law.end_function(name, base=probs, known=(1, 2))(p)

        assert fit.psi(name) == pytest.approx(float(at(probs)), abs=1e-10), name
        np.testing.assert_allclose(
            fit.influence_curves[name][rows[keep]],
            law.eif(at, probs)[keep],
            atol=1e-8,
            rtol=0,
            err_msg=name,
        )


def c5_equality() -> None:
    """C5: ``{0: 2, 1: 4}`` and ``{0: 0.5}`` on two levels are the odds multiplier ``2``."""
    reference = sup.binary_fit(2.0)
    for declaration in ({0.0: 2.0, 1.0: 4.0}, {0.0: 0.5}):
        other = sup.binary_fit(declaration)
        for name in ("ey_regimen[tilted]", "ate_regimen[tilted vs natural]"):
            assert other.psi(name) == pytest.approx(reference.psi(name), abs=1e-12)
            np.testing.assert_allclose(
                other.influence_curves[name],
                reference.influence_curves[name],
                atol=1e-12,
                rtol=0,
            )


WITNESSES: dict[str, Callable[[], None]] = {
    "c2_curve": c2_curve,
    "c2_estimate": c2_estimate,
    "known_curve": known_curve,
    "c5_equality": c5_equality,
}


@pytest.mark.parametrize("witness", WITNESSES)
def test_every_witness_passes_on_the_shipped_source(witness: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        WITNESSES[witness]()


def _fails(*names: str) -> None:
    """Each named witness fails: an assertion or a refusal raised by the broken source."""
    for name in names:
        with pytest.raises(Exception), warnings.catch_warnings():  # noqa: B017
            warnings.simplefilter("ignore")
            WITNESSES[name]()


class TestTheMutationsAreSeen:
    def test_m1_dropping_the_mechanism_term(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            ipsi_module,
            ipsi_module,
            "_recursion",
            [
                (
                    "term = np.where(at_risk, weights.weights[time - 1] * derivative, 0.0)",
                    "term = np.where(at_risk, weights.weights[time - 1] * derivative * 0.0, 0.0)",
                )
            ],
        )
        _fails("c2_curve")

    def test_m2_the_term_reads_two_of_three_levels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            ipsi_module,
            ipsi_module,
            "_mechanism_term",
            [("if g.shape[1] == 2:", "if True:")],
        )
        _fails("c2_curve")

    def test_m2_the_factor_reads_two_of_three_levels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mutate(
            monkeypatch,
            ipsi_module,
            ipsi_module,
            "_tilt_factor",
            [("if g.shape[1] == 2:", "if True:")],
        )
        _fails("c2_curve")

    def test_m3_the_term_at_the_received_level_uses_the_largest_factor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            ipsi_module,
            ipsi_module,
            "mechanism_term_general",
            [("factors[code] / (g @ factors)", "factors.max() / (g @ factors)")],
        )
        _fails("c2_curve")

    def test_m3_the_factor_at_the_received_level_uses_the_largest_factor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            ipsi_module,
            ipsi_module,
            "tilt_factor_general",
            [("factors[arm.astype(np.int64)] / (g @ factors)", "factors.max() / (g @ factors)")],
        )
        _fails("c2_curve")

    def test_m4_one_level_is_left_out_of_the_normaliser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            ipsi_module,
            ipsi_module,
            "tilt_factor_general",
            [("/ (g @ factors)", "/ (g[:, :2] @ factors[:2] + g[:, 2])")],
        )
        _fails("c2_curve")

    def test_m4_the_tilted_density_leaves_a_level_out_of_its_normaliser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            regimen_module,
            regimen_module,
            "tilted_density_general",
            [("(g @ factors)", "(g[:, 1:] @ factors[1:] + g[:, 0])")],
        )
        _fails("c2_curve")

    def test_m7_node_two_reads_the_factors_of_node_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            regimen_module,
            regimen_module.Plan,
            "tilt_factors",
            [("self.node_factors[time - 1]", "self.node_factors[0]")],
        )
        _fails("c2_curve", "c2_estimate")

    def test_m8_the_censoring_factor_is_dropped_from_the_weight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            ipsi_module,
            ipsi_module,
            "_weights",
            [("limited = bounded[:, time - 1]", "limited = np.ones(data.n)")],
        )
        _fails("c2_curve")

    def test_m9_a_known_mechanism_keeps_the_mechanism_term(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            ipsi_module,
            ipsi_module,
            "_recursion",
            [("if plan.is_tilt_node(time) and time not in known:", "if plan.is_tilt_node(time):")],
        )
        _fails("known_curve")

    def test_m10_a_key_is_resolved_by_declaration_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mutate(
            monkeypatch,
            incremental_module,
            incremental_module.Incremental,
            "factors",
            [
                (
                    "position = {label: index for index, label in reversed(list(enumerate(labels)))}",
                    f"position = {{label: index for index, label in enumerate({law.ARM_LABELS!r})}}",
                )
            ],
        )
        _fails("c2_curve")

    @pytest.mark.parametrize(
        ("owner", "name", "old", "new"),
        [
            (
                ipsi_module,
                "_tilt_factor",
                "tilt_factor_binary(arm, g[:, 1], factors[1] / factors[0])",
                "tilt_factor_binary(arm, g[:, 1], factors[1])",
            ),
            (
                ipsi_module,
                "_mechanism_term",
                "mechanism_term_binary(arm, g[:, 1], by_arm, factors[1] / factors[0])",
                "mechanism_term_binary(arm, g[:, 1], by_arm, factors[1])",
            ),
            (
                regimen_module.Plan,
                "with_tilts",
                "tilted_density_binary(g[:, 1], factors[1] / factors[0])",
                "tilted_density_binary(g[:, 1], factors[1])",
            ),
        ],
        ids=["factor", "term", "density"],
    )
    def test_m11_the_two_level_branch_ignores_the_first_factor(
        self, monkeypatch: pytest.MonkeyPatch, owner: Any, name: str, old: str, new: str
    ) -> None:
        mutate(
            monkeypatch,
            regimen_module if owner is regimen_module.Plan else ipsi_module,
            owner,
            name,
            [(old, new)],
        )
        _fails("c5_equality")


def test_the_mutated_modules_are_restored() -> None:
    """``monkeypatch`` put the shipped functions back after each control."""
    for witness in WITNESSES.values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            witness()
