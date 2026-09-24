r"""The standard error of the omitted-variable bound, on the exact finite laws (RM22).

The bound's one-sided limits read the curve of :math:`\hat\nu^2` through ``psi_max_bias``.
For the ATT and the ATC, the representer and :math:`m(O, \alpha)` both divide by the share
:math:`p` of the arm that the parameter conditions on, so the estimate is a weighted mean over
:math:`\hat p^2`, and its curve carries :math:`-2 \nu^2 w (1\{A = c\} - p) / p`.  The term sums
to zero over the rows, so a check of a mean or of a point cannot see it.  Every witness here
compares curves row by row.

Each comparator is a complex-step Gateaux derivative of a functional that
:mod:`tests.discrete_law` or :mod:`tests.discrete_law_multi` writes from the cell
probabilities.  Neither module imports the package, and no comparator reads the curve under
test.  The fits are the oracle fits of :mod:`tests.unit._exact_sensitivity_support`: the sample
realises the law, targeting moves nothing, and each reported curve is the curve at the law.

:class:`TestTheWitnessHasTeeth` commits the mutations that the RM22 plan predicts.  Each one
replaces the helper that computes the term, or the arm a parameter conditions on, and
measures how far the curve then moves from the derivative.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from scipy import stats

import cleverly.sensitivity.omitted_variable as ov
from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity._parameters import ArmParameter
from tests import discrete_law as law
from tests import discrete_law_multi as multi
from tests.unit import _exact_sensitivity_support as support

#: The tolerance of every row comparison.  The plan's probe measured at most 6.0e-14.
ATOL = 1e-11

#: The weight function of the weighted two-arm fit.  It tilts on the treatment and the
#: outcome, so it moves ``g``, ``Q`` and the arm shares together.
WEIGHTING = "treatment_and_outcome"

#: The large strength of the RM22 contract, ``c_Y = 0.5``, ``c_D = 0.3`` and ``rho = 1``.
CF_Y, CF_D = 0.5, 0.3

#: :math:`\sqrt{c_Y c_D / (1 - c_D)}` at that strength, written here rather than read from
#: the package.
STRENGTH = float(np.sqrt(CF_Y * CF_D / (1.0 - CF_D)))

#: The smallest deviation from the derivative that each committed mutation produced when the
#: RM22 delivery measured it, rounded down.  Each is at least ten orders of magnitude above
#: :data:`ATOL`.  M1 drops the term, M2 flips its sign, M3 uses an unweighted share, M4 drops
#: the weights, M5 makes the ATT condition on the reference, and M6 adds the term to every
#: parameter.  The minima measured were 9.541, 19.08, 3.977, 3.638, 6.864 and 3.700.
M1_FLOOR = 9.5
M2_FLOOR = 19.0
M3_FLOOR = 3.9
M4_FLOOR = 3.6
M5_FLOOR = 6.8
M6_FLOOR = 3.6

#: How far the lower-bound curve of the ATT and the ATC moves without the term (0.6202 and
#: 0.4766 measured), and how far the clustered standard error moves (0.00174 and 0.00159).
BOUND_CURVE_FLOOR = 0.4
CLUSTERED_GAP_FLOOR = 1e-3

#: The cluster of each row of the two-arm sample: 50 clusters of 20 rows, by row index, so
#: each cluster mixes support points.
CLUSTER = np.arange(law.N) % 50


@dataclass(frozen=True)
class Case:
    """One parameter of one oracle fit, named in the law's own arm indices."""

    law: str
    weighted: bool
    oracle: str

    @property
    def reported(self) -> str:
        """The name the fit reports this parameter under."""
        return self.oracle if self.law == "binary" else multi.reported_name(self.oracle)

    @property
    def conditional(self) -> bool:
        """Whether the parameter conditions on an arm, which is when the term applies."""
        return self.oracle.startswith(("att", "atc"))

    def __str__(self) -> str:
        weighting = "weighted" if self.weighted else "unweighted"
        return f"{self.law}-{weighting}-{self.reported}"


CONDITIONAL: tuple[Case, ...] = (
    *(Case("binary", weighted, name) for weighted in (False, True) for name in ("att", "atc")),
    *(
        Case("multi", weighted, f"{stem}[{arm} vs 0]")
        for weighted in (False, True)
        for stem in ("att", "atc")
        for arm in (1, 2)
    ),
)

UNCONDITIONAL: tuple[Case, ...] = (
    *(
        Case("binary", weighted, name)
        for weighted in (False, True)
        for name in ("ey1", "ey0", "ate")
    ),
    *(
        Case("multi", weighted, name)
        for weighted in (False, True)
        for name in ("ey[0]", "ey[1]", "ey[2]", "ate[1 vs 0]", "ate[2 vs 0]")
    ),
)


@pytest.fixture(scope="module")
def fits() -> dict[tuple[str, bool], Any]:
    """The four oracle fits: each law, with and without weights."""
    return {
        ("binary", False): support.binary_oracle_fit()[0],
        ("binary", True): support.binary_oracle_fit(WEIGHTING)[0],
        ("multi", False): support.multi_oracle_fit(),
        ("multi", True): support.multi_oracle_fit(weighted=True),
    }


@pytest.fixture(scope="module")
def clustered_fit() -> Any:
    """The unweighted two-arm fit, with the rows grouped into :data:`CLUSTER`."""
    return support.binary_oracle_fit(estimands=("ate", "att", "atc"), cluster=CLUSTER)[0]


def _tilt(case: Case) -> Callable[[Any], Any]:
    """The law a weighted fit estimates, as a function of the sampling law."""
    if not case.weighted:
        return lambda probs: probs
    if case.law == "binary":
        cells = law.cell_weights(law.WEIGHT_FUNCTIONS[WEIGHTING])
        return lambda probs: law.tilt(probs, cells)
    return support.multi_tilt


@functools.cache
def exact_nu2_curve(case: Case) -> np.ndarray:
    """The Gateaux derivative of :math:`\\nu^2` at every support point of the case's law."""
    tilt = _tilt(case)
    if case.law == "binary":
        return law.gateaux_eif(lambda probs: law.riesz_second_moment(tilt(probs), case.oracle))
    return support.multi_gateaux_eif(
        lambda probs: multi.riesz_second_moment(tilt(probs), case.oracle)
    )


def nu2_at_the_law(case: Case) -> float:
    """:math:`\\nu^2` of the law the case's fit estimates."""
    module = law if case.law == "binary" else multi
    return float(module.riesz_second_moment(_tilt(case)(module.PROBS), case.oracle))


def first_rows(case: Case) -> np.ndarray:
    """A representative row of each support point of the case's sample."""
    return law.first_row_of() if case.law == "binary" else support.multi_first_row_of()


def reported_nu2_curve(fits: dict[tuple[str, bool], Any], case: Case) -> np.ndarray:
    """The curve of :math:`\\hat\\nu^2` the fit reports, at one row of each support point."""
    elements = ov.sensitivity_elements(fits[(case.law, case.weighted)], case.reported)
    assert elements.psi_nu2 is not None
    return np.asarray(elements.psi_nu2)[first_rows(case)]


def deviation(fits: dict[tuple[str, bool], Any], case: Case) -> float:
    """The largest distance, over the support points, between the curve and the derivative."""
    return float(np.max(np.abs(reported_nu2_curve(fits, case) - exact_nu2_curve(case))))


@functools.cache
def exact_lower_bound_curve(name: str) -> np.ndarray:
    r"""The Gateaux derivative of :math:`\theta - s \sqrt{\sigma^2 \nu^2}` on the two-arm law."""

    def lower(probs: Any) -> Any:
        spread = law.residual_variance(probs) * law.riesz_second_moment(probs, name)
        return law.functional(probs, name) - STRENGTH * np.sqrt(spread)

    return law.gateaux_eif(lower)


def reported_lower_bound_curve(result: Any, name: str) -> np.ndarray:
    """The curve the lower limit reads: the estimate's curve minus the strength times the bias's."""
    elements = ov.sensitivity_elements(result, name)
    assert elements.psi_max_bias is not None
    return np.asarray(result[name].influence_curve) - STRENGTH * np.asarray(elements.psi_max_bias)


def reported_lower_standard_error(result: Any, name: str) -> float:
    """The standard error of the lower limit, read back off the limit the bound reports."""
    bounds = ov.omitted_variable_bounds(result, name, cf_y=CF_Y, cf_d=CF_D, rho=1.0)
    return float((bounds.lower - bounds.ci_lower) / stats.norm.ppf(0.95))


class TestTheConditionalCurve:
    @pytest.mark.parametrize("case", CONDITIONAL, ids=str)
    def test_the_curve_is_the_gateaux_derivative(
        self, fits: dict[tuple[str, bool], Any], case: Case
    ) -> None:
        """Row by row, including the derivative through the conditioning share."""
        np.testing.assert_allclose(
            reported_nu2_curve(fits, case), exact_nu2_curve(case), atol=ATOL, rtol=0
        )

    @pytest.mark.parametrize("case", CONDITIONAL, ids=str)
    def test_the_term_is_invisible_to_a_mean(
        self, fits: dict[tuple[str, bool], Any], case: Case, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why the comparison above is row by row: the term moves no mean and no point.

        The share is the weighted share, so the weighted rows of the term sum to zero
        exactly, and the point :math:`\\hat\\nu^2` is :math:`\\nu^2` of the law with or
        without it.  The term is still large on each row.
        """
        result = fits[(case.law, case.weighted)]
        with_term = ov.sensitivity_elements(result, case.reported)
        monkeypatch.setattr(
            ov, "_conditioning_share_influence", lambda nu2, indicator, share, weights: 0.0
        )
        without_term = ov.sensitivity_elements(result, case.reported)
        assert with_term.psi_nu2 is not None and without_term.psi_nu2 is not None
        term = np.asarray(with_term.psi_nu2) - np.asarray(without_term.psi_nu2)
        assert abs(float(term.mean())) < 1e-12
        assert with_term.nu2 == without_term.nu2
        assert with_term.nu2 == pytest.approx(nu2_at_the_law(case), abs=1e-11)
        assert float(np.max(np.abs(term))) > 1.0


class TestTheUnconditionalCurve:
    @pytest.mark.parametrize("case", UNCONDITIONAL, ids=str)
    def test_the_curve_is_the_gateaux_derivative(
        self, fits: dict[tuple[str, bool], Any], case: Case
    ) -> None:
        """The ATE, the arm means and each ``ate[...]`` contrast, which RM22 must not move."""
        np.testing.assert_allclose(
            reported_nu2_curve(fits, case), exact_nu2_curve(case), atol=ATOL, rtol=0
        )

    def test_the_helper_runs_once_for_each_conditional_parameter_and_never_otherwise(
        self, fits: dict[tuple[str, bool], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[float] = []
        original = ov._conditioning_share_influence

        def spy(nu2: float, indicator: Any, share: float, weights: Any) -> Any:
            calls.append(nu2)
            return original(nu2, indicator, share, weights)

        monkeypatch.setattr(ov, "_conditioning_share_influence", spy)
        for case in UNCONDITIONAL:
            ov.sensitivity_elements(fits[(case.law, case.weighted)], case.reported)
        assert calls == []
        for case in CONDITIONAL:
            ov.sensitivity_elements(fits[(case.law, case.weighted)], case.reported)
        assert len(calls) == len(CONDITIONAL)


class TestTheBoundCurve:
    """The whole lower bound, :math:`\\theta - s \\sqrt{\\sigma^2 \\nu^2}`, on the two-arm law."""

    @pytest.mark.parametrize("name", ["ate", "att", "atc"])
    def test_the_lower_bound_curve_is_the_gateaux_derivative(
        self, fits: dict[tuple[str, bool], Any], name: str
    ) -> None:
        result = fits[("binary", False)]
        reported = reported_lower_bound_curve(result, name)[law.first_row_of()]
        np.testing.assert_allclose(reported, exact_lower_bound_curve(name), atol=ATOL, rtol=0)

    @pytest.mark.parametrize("name", ["ate", "att", "atc"])
    def test_the_bound_is_the_functional(
        self, fits: dict[tuple[str, bool], Any], name: str
    ) -> None:
        """The premise: at the law, the reported bound is the bound of the law."""
        bounds = ov.omitted_variable_bounds(
            fits[("binary", False)], name, cf_y=CF_Y, cf_d=CF_D, rho=1.0
        )
        spread = law.residual_variance(law.PROBS) * law.riesz_second_moment(law.PROBS, name)
        expected = law.functional(law.PROBS, name) - STRENGTH * np.sqrt(spread)
        assert bounds.confounding_strength == pytest.approx(STRENGTH, abs=1e-15)
        assert bounds.lower == pytest.approx(float(expected), abs=1e-12)


class TestTheClusteredStandardError:
    """The limit's standard error is the cluster sum of the exact curve."""

    def test_the_fit_keeps_its_inference(self, clustered_fit: Any) -> None:
        assert clustered_fit.inference_status == "influence_curve"

    @pytest.mark.parametrize("name", ["att", "atc", "ate"])
    def test_the_standard_error_sums_the_exact_curve(self, clustered_fit: Any, name: str) -> None:
        exact_rows = exact_lower_bound_curve(name)[law.cell_of_row()]
        clustered = float(np.sqrt(influence_variance(exact_rows, CLUSTER)))
        row_level = float(np.sqrt(influence_variance(exact_rows, None)))
        assert reported_lower_standard_error(clustered_fit, name) == pytest.approx(
            clustered, abs=1e-12
        )
        # Not vacuous: the cluster sum is not the row-level variance on this sample.
        assert abs(clustered - row_level) > 0.01


def _share_mutation(
    monkeypatch: pytest.MonkeyPatch, replacement: Callable[..., Any]
) -> Callable[..., Any]:
    """Replace the helper by ``replacement(original, nu2, indicator, share, weights)``."""
    original = ov._conditioning_share_influence
    monkeypatch.setattr(
        ov,
        "_conditioning_share_influence",
        lambda nu2, indicator, share, weights: replacement(
            original, nu2, indicator, share, weights
        ),
    )
    return original


def _conditioning_mutation(
    monkeypatch: pytest.MonkeyPatch, arm_of: Callable[[ArmParameter, Any], Any]
) -> None:
    """Replace ``ArmParameter.conditions_on`` by ``arm_of(parameter, original_arm)``."""
    original = ArmParameter.conditions_on.fget
    assert original is not None
    monkeypatch.setattr(
        ArmParameter, "conditions_on", property(lambda self: arm_of(self, original(self)))
    )


def _deviations(fits: dict[tuple[str, bool], Any], cases: tuple[Case, ...]) -> dict[str, float]:
    return {str(case): deviation(fits, case) for case in cases}


def _assert_every_case_fails(
    fits: dict[tuple[str, bool], Any], cases: tuple[Case, ...], floor: float
) -> dict[str, float]:
    """Each case misses the derivative by more than ``floor``, and the witness fails it."""
    measured = _deviations(fits, cases)
    assert min(measured.values()) > floor, measured
    for case in cases:
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                reported_nu2_curve(fits, case), exact_nu2_curve(case), atol=ATOL, rtol=0
            )
    return measured


def _assert_every_case_passes(fits: dict[tuple[str, bool], Any], cases: tuple[Case, ...]) -> None:
    measured = _deviations(fits, cases)
    assert max(measured.values()) < ATOL, measured


class TestTheWitnessHasTeeth:
    """The committed mutations of the RM22 plan, each measured against its prediction.

    A floor is the smallest deviation that the plan's probe measured for that mutation, and
    at least 1.0.  The witness tolerance is 1e-11, so each floor is ten orders of magnitude
    above it.
    """

    def test_m0_the_helper_rewrapped_changes_nothing(
        self, fits: dict[tuple[str, bool], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _share_mutation(monkeypatch, lambda original, *arguments: original(*arguments))
        _assert_every_case_passes(fits, CONDITIONAL + UNCONDITIONAL)

    def test_m1_a_helper_that_returns_zeros_fails_every_conditional_witness(
        self,
        fits: dict[tuple[str, bool], Any],
        clustered_fit: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _share_mutation(monkeypatch, lambda original, nu2, indicator, share, weights: 0.0)
        _assert_every_case_fails(fits, CONDITIONAL, floor=M1_FLOOR)
        _assert_every_case_passes(fits, UNCONDITIONAL)
        result = fits[("binary", False)]
        for name in ("att", "atc"):
            moved = reported_lower_bound_curve(result, name)[law.first_row_of()]
            assert np.max(np.abs(moved - exact_lower_bound_curve(name))) > BOUND_CURVE_FLOOR
            exact_rows = exact_lower_bound_curve(name)[law.cell_of_row()]
            clustered = float(np.sqrt(influence_variance(exact_rows, CLUSTER)))
            gap = abs(reported_lower_standard_error(clustered_fit, name) - clustered)
            assert gap > CLUSTERED_GAP_FLOOR

    def test_m2_a_flipped_sign_fails_every_conditional_witness(
        self, fits: dict[tuple[str, bool], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _share_mutation(monkeypatch, lambda original, *arguments: -original(*arguments))
        _assert_every_case_fails(fits, CONDITIONAL, floor=M2_FLOOR)

    def test_m3_an_unweighted_share_fails_the_weighted_witnesses_only(
        self, fits: dict[tuple[str, bool], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _share_mutation(
            monkeypatch,
            lambda original, nu2, indicator, share, weights: original(
                nu2, indicator, float(np.mean(indicator)), weights
            ),
        )
        weighted = tuple(case for case in CONDITIONAL if case.weighted)
        _assert_every_case_fails(fits, weighted, floor=M3_FLOOR)
        _assert_every_case_passes(fits, tuple(case for case in CONDITIONAL if not case.weighted))

    def test_m4_dropped_weights_fail_the_weighted_witnesses_only(
        self, fits: dict[tuple[str, bool], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _share_mutation(
            monkeypatch,
            lambda original, nu2, indicator, share, weights: original(
                nu2, indicator, share, np.ones_like(weights)
            ),
        )
        weighted = tuple(case for case in CONDITIONAL if case.weighted)
        _assert_every_case_fails(fits, weighted, floor=M4_FLOOR)
        _assert_every_case_passes(fits, tuple(case for case in CONDITIONAL if not case.weighted))

    def test_m5_an_att_that_conditions_on_the_reference_fails_the_att_witnesses(
        self, fits: dict[tuple[str, bool], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _conditioning_mutation(
            monkeypatch,
            lambda parameter, arm: parameter.versus if parameter.group == "att" else arm,
        )
        att = tuple(case for case in CONDITIONAL if case.oracle.startswith("att"))
        _assert_every_case_fails(fits, att, floor=M5_FLOOR)
        _assert_every_case_passes(fits, tuple(case for case in CONDITIONAL if case not in att))

    def test_m6_a_term_on_every_parameter_fails_the_unconditional_witnesses(
        self, fits: dict[tuple[str, bool], Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _conditioning_mutation(
            monkeypatch, lambda parameter, arm: parameter.arm if arm is None else arm
        )
        _assert_every_case_fails(fits, UNCONDITIONAL, floor=M6_FLOOR)
        _assert_every_case_passes(fits, CONDITIONAL)
