r"""Deliberate-mutation controls of the categorical incremental fit over time.

An exact-law check is blind to a term that vanishes at the truth.  Each control here breaks one
line of the shipped source, through :func:`tests.unit._source_mutation.mutate`, and asks one
witness to fail.  The unmutated source passes every witness first, so a failure is the
mutation's and not the witness's.

A witness is a **gap**: the largest distance between what the fit reports and what the oracle
says it must report.  The shipped source has a gap below ``ESTIMATE_TOLERANCE`` and ``CURVE_TOLERANCE``.  A mutated source must
run to the end and leave a gap above ``MARGIN``, so a mutant that crashes, or that moves a value
by less than a rounding error, fails the control and does not pass it.

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
from typing import Any, NamedTuple

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


#: The shipped fit agrees with the oracle to this distance in an estimate and in a curve.
ESTIMATE_TOLERANCE = 1e-10
CURVE_TOLERANCE = 1e-8
#: A mutated fit must leave a gap of at least this size: a wrong value and not a rounding error.
MARGIN = 1e-4


class Gap(NamedTuple):
    """The largest distance of a fit from the oracle, in estimates and in curves."""

    estimate: float
    curve: float

    def widen(self, estimate: float, curve: np.ndarray | float) -> Gap:
        """This gap, or the one a further estimate and curve reach if it is wider."""
        return Gap(
            max(self.estimate, estimate),
            max(self.curve, float(np.max(np.abs(curve)))),
        )


def c2_curve() -> Gap:
    """C2: the distance of the censored three-level fit from the oracle."""
    fit = _fit_end()
    gap = Gap(0.0, 0.0)
    for name, truth, eif, rows in sup.expected(NAMES, law.end_function, law.PROBS, law.COUNTS):
        gap = gap.widen(abs(fit.psi(name) - truth), fit.influence_curves[name][rows] - eif)
    return gap


def c2_estimate() -> Gap:
    """C2, the estimates alone."""
    fit = _fit_end()
    gap = Gap(0.0, 0.0)
    for name, truth, _, _ in sup.expected(NAMES, law.end_function, law.PROBS, law.COUNTS):
        gap = gap.widen(abs(fit.psi(name) - truth), 0.0)
    return gap


def known_curve() -> Gap:
    """A declared mechanism at both nodes: the curve has no mechanism term."""
    fit = sup.fit_ipsi(
        sup.known_frame(), law.PLANS, sup.UNCENSORED_COLUMNS, declared=sup.KNOWN_DECLARED
    )
    probs, counts = law.UNCENSORED_PROBS, law.UNCENSORED_COUNTS
    rows = law.first_rows(counts)
    keep = probs > 0
    gap = Gap(0.0, 0.0)
    for name in NAMES:

        def at(p: Any, name: str = name) -> Any:
            return law.end_function(name, base=probs, known=(1, 2))(p)

        gap = gap.widen(
            abs(fit.psi(name) - float(at(probs))),
            fit.influence_curves[name][rows[keep]] - law.eif(at, probs)[keep],
        )
    return gap


def c5_equality() -> Gap:
    """C5: ``{0: 2, 1: 4}`` and ``{0: 0.5}`` on two levels are the odds multiplier ``2``."""
    reference = sup.binary_fit(2.0)
    gap = Gap(0.0, 0.0)
    for declaration in ({0.0: 2.0, 1.0: 4.0}, {0.0: 0.5}):
        other = sup.binary_fit(declaration)
        for name in ("ey_regimen[tilted]", "ate_regimen[tilted vs natural]"):
            gap = gap.widen(
                abs(other.psi(name) - reference.psi(name)),
                other.influence_curves[name] - reference.influence_curves[name],
            )
    return gap


WITNESSES: dict[str, Callable[[], Gap]] = {
    "c2_curve": c2_curve,
    "c2_estimate": c2_estimate,
    "known_curve": known_curve,
    "c5_equality": c5_equality,
}


def _run(name: str) -> Gap:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WITNESSES[name]()


@pytest.mark.parametrize("witness", WITNESSES)
def test_every_witness_passes_on_the_shipped_source(witness: str) -> None:
    gap = _run(witness)
    assert gap.estimate <= ESTIMATE_TOLERANCE
    assert gap.curve <= CURVE_TOLERANCE


def _fails(*names: str) -> None:
    """Each named witness runs on the broken source and leaves a gap above ``MARGIN``.

    An exception from the broken source is not a failure of the witness: it propagates, and
    the control fails with it.  A mutant that crashes shows nothing about the value it moves.
    ``c2_estimate`` measures estimates alone, so its gap is the estimate gap.
    """
    for name in names:
        gap = _run(name)
        assert max(gap) > MARGIN, f"{name}: estimate gap {gap.estimate:.3e}, curve {gap.curve:.3e}"


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
            "_extend_tilt",
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
            [
                (
                    "factors = self.node_factors[time - 1] if self.node_factors else None",
                    "factors = (self.node_factors[0] or self.node_factors[time - 1]) "
                    "if self.node_factors else None",
                )
            ],
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
                "_extend_tilt",
                "tilt_product_binary(tilt, arm, g[:, 1], factors[1] / factors[0])",
                "tilt_product_binary(tilt, arm, g[:, 1], factors[1])",
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
    for name in WITNESSES:
        gap = _run(name)
        assert gap.estimate <= ESTIMATE_TOLERANCE
        assert gap.curve <= CURVE_TOLERANCE
