"""Every E-value request on a controlled-direct-effect fit refuses (roadmap row RM21).

A controlled direct effect is identified under two assumptions of no unmeasured confounding: one
for the treatment and the outcome, and one for the intermediate variable and the outcome. This
package has no implemented E-value bound for its fitted target and confounding model. So
:func:`~cleverly.sensitivity.evalue._select_evalue` refuses every request on such a fit, before it
resolves the estimand, and the capability row reads ``unavailable``. F25 in ``docs/roadmap.md``
holds the missing result.

Before the change the Gaussian branch and the reported-ratio branches answered on that fit, and
a default request on a binary fit without ``rr`` reached the odds-ratio fallback. One table,
:data:`REQUESTS`, drives every witness, so a branch the table names is checked on each side.

*   :class:`TestAFitWithoutAnIntermediateKeepsEachBranch` is the control. Each request keeps its
    branch on the same law fitted without ``intermediate=``, and the rows reach all five branches.
*   :class:`TestAControlledDirectEffectRefusesEveryBranch` pins the refusal on each request at
    both levels, through the capability row, the free function, and the facade.
*   :class:`TestTheRefusalBlocksANumber` is the nonzero witness. Each blocked branch computes a
    finite E-value above 1 on the same fit, so the refusal withholds a number.
*   :class:`TestTheMutationsFailTheWitness` rebuilds ``_select_evalue`` from its own source with
    the check moved, and asserts the exact set of requests that leak at each position.
"""

from __future__ import annotations
import __future__

import importlib
import inspect
import typing
import warnings
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import pytest

from cleverly import CausalStudy, ControlledDirectEffect, PointTreatment
from cleverly.assessment import AssessmentStatus
from cleverly.datasets import make_multi_arm, make_shift_dose
from cleverly.estimators import direct_effect
from cleverly.estimators.direct_effect import LEVELS, declares_intermediate
from cleverly.exceptions import CapabilityError, DataError, PositivityWarning
from cleverly.interventions import Shift
from cleverly.sensitivity._parameters import arm_parameter_keys
from cleverly.sensitivity.evalue import (
    _DERIVED_RR,
    _DIRECT_EFFECT_REFUSAL,
    _Branch,
    _evalue_from_selection,
    _EValueRefusal,
    _EValueSelection,
)
from tests.conftest import FAST_KWARGS, IN_SAMPLE
from tests.unit._direct_effect_support import (
    COVARIATES,
    binary_cde_frame,
    cde_frame,
    column_only,
    fit_cde,
    fit_without_intermediate,
    level_only,
)

pytestmark = pytest.mark.xdist_group("evalue_direct_effect")

#: Imported by name, because ``cleverly.sensitivity`` rebinds ``evalue`` to the function. The
#: mutation tests replace attributes of this module, and every call below reads them from it.
evalue_module = importlib.import_module("cleverly.sensitivity.evalue")
derived_module = importlib.import_module("cleverly.sensitivity._derived")

#: The text of the cached-retarget rule in ``_risk_ratio_refusal``, which a direct call meets.
DERIVED_TEXT = (
    "cached-nuisance retargeting does not pass the fitted intermediate intervention level"
)


#: Each law: its frame and the estimands its fit reports. ``odds`` reports no ``rr``, so its
#: default request meets the derivation, or the odds-ratio fallback when that refuses.
LAWS: dict[str, tuple[Callable[[], Any], tuple[str, ...]]] = {
    "gaussian": (cde_frame, ("ate", "att", "atc", "ey1", "ey0")),
    "ratio": (binary_cde_frame, ("ate", "rr", "or", "ey1", "ey0")),
    "odds": (binary_cde_frame, ("ate", "or", "ey1", "ey0")),
}


def as_fitted(result: Any) -> Any:
    """The fit as it stands."""
    return result


def cv_evaluated(result: Any) -> Any:
    """The fit read as CV-evaluated, so the exact retarget refuses and the fallbacks answer.

    The edit changes the configuration and refits nothing. The nuisances and estimates stay
    those of the in-sample fit, so the result is a routing witness only. The two requests that
    use it reach ``reported_or`` and ``fixed_baseline_ate`` by the same selection as a real
    CV-evaluated fit, and their E-values describe the in-sample estimates.
    """
    return replace(result, config=replace(result.config, cv_evaluation=True))


class Request(NamedTuple):
    """One E-value request, and the branch it takes on the fit without an intermediate."""

    name: str
    law: str
    edit: Callable[[Any], Any]
    estimand: str | None
    branch: str


REQUESTS = (
    Request("gaussian-default", "gaussian", as_fitted, None, "gaussian_difference"),
    Request("gaussian-ate", "gaussian", as_fitted, "ate", "gaussian_difference"),
    Request("gaussian-att", "gaussian", as_fitted, "att", "gaussian_difference"),
    Request("gaussian-atc", "gaussian", as_fitted, "atc", "gaussian_difference"),
    Request("ratio-default", "ratio", as_fitted, None, "reported_rr"),
    Request("ratio-rr", "ratio", as_fitted, "rr", "reported_rr"),
    Request("ratio-or", "ratio", as_fitted, "or", "reported_or"),
    Request("ratio-ate", "ratio", as_fitted, "ate", "derived_rr"),
    Request("odds-default", "odds", as_fitted, None, "derived_rr"),
    Request("odds-cv-default", "odds", cv_evaluated, None, "reported_or"),
    Request("odds-cv-ate", "odds", cv_evaluated, "ate", "fixed_baseline_ate"),
)

#: The two requests that no branch answers even without an intermediate variable: a level, and
#: a multi-arm default. Each one refused later with another status before RM21.
LATE_REQUESTS = (("ey1", "gaussian", "ey1"), ("multi-arm-default", "multi", None))

EVERY_NAME = frozenset({case.name for case in REQUESTS} | {name for name, *_ in LATE_REQUESTS})


def ids(case: Request) -> str:
    return case.name


@pytest.fixture(scope="module")
def cde_fits() -> dict[str, Any]:
    """Each law fitted with ``intermediate="Z"``, keyed by law and then by level."""
    fits = {law: fit_cde(frame(), estimands) for law, (frame, estimands) in LAWS.items()}
    frame, _ = make_multi_arm(n=400, seed=3)
    frame["Z"] = np.random.default_rng(0).integers(0, 2, len(frame))
    with warnings.catch_warnings():
        # Three arms at n = 400 truncate some propensities. The refusal reads no mechanism.
        warnings.simplefilter("ignore", PositivityWarning)
        fits["multi"] = fit_cde(frame)
    return fits


@pytest.fixture(scope="module")
def plain_fits() -> dict[str, Any]:
    """Each law fitted without the intermediate variable."""
    return {
        law: fit_without_intermediate(frame(), estimands)
        for law, (frame, estimands) in LAWS.items()
    }


def raised(call: Callable[[], Any]) -> str | None:
    """The refusal a call raises, or ``None`` when it answers.

    No ``pytest.raises``: a call that answers must fail as an ``AssertionError``, which is
    what :func:`leaks` counts. ``pytest.raises`` fails with another exception type.
    """
    try:
        call()
    except CapabilityError as error:
        return str(error)
    return None


def evalue_row(result: Any, estimand: str | None) -> Any:
    """The capability row the facade publishes, or the one ``assess()`` builds for a request."""
    facade = result.sensitivity
    if estimand is None:
        return facade.capability("evalue")
    return facade._capability_for_arguments("evalue", {"estimand": estimand})


def facade_refusal(row: Any, estimand: str | None) -> str:
    """The message the facade raises for a refused row.

    A default request meets ``_require``, which prefixes the kind, the operation and the
    status. An explicit estimand skips ``_require`` and raises the selection's reason alone.
    """
    if estimand is not None:
        return str(row.reason)
    return f"sensitivity {row.operation!r} is {row.status.value}: {row.reason}"


def assert_refused(result: Any, estimand: str | None) -> None:
    """The row, the free function, and the facade each refuse with the RM21 reason.

    Each surface reads a fresh copy, because the facade caches its selection on the result.
    """
    row = evalue_row(replace(result), estimand)
    assert not row.available
    assert row.status is AssessmentStatus.UNAVAILABLE
    assert row.reason == _DIRECT_EFFECT_REFUSAL
    assert row.requires_arguments == ()
    assert raised(lambda: evalue_module.evalue(replace(result), estimand)) == (
        _DIRECT_EFFECT_REFUSAL
    )
    facade = replace(result).sensitivity
    call = facade.evalue if estimand is None else partial(facade.evalue, estimand)
    assert raised(call) == facade_refusal(row, estimand)


def assert_keeps(result: Any, estimand: str | None, branch: str) -> None:
    """A fit without an intermediate variable takes its branch and reports an E-value."""
    assert not declares_intermediate(result)
    assert evalue_module._select_evalue(result, estimand).branch == branch
    row = evalue_row(replace(result), estimand)
    assert row.available
    assert row.status is AssessmentStatus.PASSED
    report = evalue_module.evalue(replace(result), estimand)
    assert np.isfinite(report.point)
    assert report.point > 1.0


def derived_helper_refuses(result: Any) -> bool:
    """Whether a direct call of ``_derived_risk_ratio`` meets its own intermediate rule."""
    return DERIVED_TEXT in (raised(lambda: derived_module._derived_risk_ratio(result, "ate")) or "")


class TestAFitWithoutAnIntermediateKeepsEachBranch:
    """Witness 3, the control: the refusal reads the fit class, not the branch."""

    @pytest.mark.parametrize("case", REQUESTS, ids=ids)
    def test_the_request_keeps_its_branch(self, plain_fits: dict[str, Any], case: Request) -> None:
        assert_keeps(case.edit(plain_fits[case.law]), case.estimand, case.branch)

    def test_the_controls_reach_every_branch(self) -> None:
        assert {case.branch for case in REQUESTS} == set(typing.get_args(_Branch))


class TestAControlledDirectEffectRefusesEveryBranch:
    """Witness 1: each request, at each level, refuses before any branch."""

    @pytest.mark.parametrize("case", REQUESTS, ids=ids)
    @pytest.mark.parametrize("level", LEVELS)
    def test_the_request_refuses(
        self, cde_fits: dict[str, Any], level: float, case: Request
    ) -> None:
        assert_refused(case.edit(cde_fits[case.law][level]), case.estimand)

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_level_refuses_before_its_not_applicable_check(
        self, cde_fits: dict[str, Any], plain_fits: dict[str, Any], level: float
    ) -> None:
        assert_refused(cde_fits["gaussian"][level], "ey1")
        with pytest.raises(_EValueRefusal) as control:
            evalue_module._select_evalue(plain_fits["gaussian"], "ey1")
        assert control.value.status is AssessmentStatus.NOT_APPLICABLE

    @pytest.mark.parametrize("level", LEVELS)
    def test_a_multi_arm_fit_does_not_defer(self, cde_fits: dict[str, Any], level: float) -> None:
        result = cde_fits["multi"][level]
        contrasts = [name for name in result.estimates if name.startswith("ate[")]
        assert len(contrasts) == 2
        for estimand in (None, *contrasts):
            assert_refused(result, estimand)

    def test_the_assessment_reports_the_refusal(self, cde_fits: dict[str, Any]) -> None:
        item = replace(cde_fits["ratio"][0.0]).assess().sensitivity["evalue"]
        assert item.status is AssessmentStatus.UNAVAILABLE
        assert item.detail == _DIRECT_EFFECT_REFUSAL
        assert dict(item.arguments) == {}

    def test_a_continuous_dose_with_an_intermediate_is_not_applicable(self) -> None:
        """The continuous-treatment check runs first, so this fit keeps its older refusal.

        A shift of a continuous dose supports ``intermediate=``. The E-value needs a discrete
        arm contrast, so the row reads ``not_applicable`` before the RM21 check can run. The
        request is refused either way.
        """
        frame, _ = make_shift_dose(n=400, seed=3)
        frame["Z"] = np.random.default_rng(0).integers(0, 2, len(frame))
        with warnings.catch_warnings():
            # The uncapped shift extrapolates on a few rows. The refusal reads no mechanism.
            warnings.simplefilter("ignore", PositivityWarning)
            result = fit_cde(frame, shifts=[Shift(0.5, cap=None)])[0.0]
        assert result.data.is_continuous_treatment
        assert declares_intermediate(result)
        row = result.sensitivity.capability("evalue")
        assert not row.available
        assert row.status is AssessmentStatus.NOT_APPLICABLE
        assert row.reason == "an E-value requires a discrete arm contrast"

    def test_no_branch_computes(
        self, monkeypatch: pytest.MonkeyPatch, cde_fits: dict[str, Any]
    ) -> None:
        """The refusal comes before the estimand resolves, and before every branch."""

        def computed(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("computed")

        for name in (
            "arm_parameter_keys",
            "_default_estimand",
            "_risk_ratio_refusal",
            "_baseline_mean",
            "_standardising_sd",
            "_derived_risk_ratio",
        ):
            monkeypatch.setattr(evalue_module, name, computed)
        for case in REQUESTS:
            assert_refused(case.edit(cde_fits[case.law][0.0]), case.estimand)

    def test_a_level_without_an_intermediate_column_refuses(
        self, plain_fits: dict[str, Any]
    ) -> None:
        for estimand in (None, "ate"):
            assert_refused(level_only(plain_fits["ratio"]), estimand)

    def test_an_intermediate_column_without_a_level_refuses(
        self, plain_fits: dict[str, Any]
    ) -> None:
        for estimand in (None, "ate"):
            assert_refused(column_only(plain_fits["ratio"]), estimand)

    def test_the_study_entry_refuses(self) -> None:
        design = PointTreatment(
            outcome="Y", treatment="A", adjustment=tuple(COVARIATES), intermediate="Z"
        )
        result = CausalStudy(cde_frame(), design=design).estimate(
            ControlledDirectEffect(intermediate=0.0), **FAST_KWARGS, **IN_SAMPLE
        )
        assert declares_intermediate(result)
        for estimand in (None, "ate"):
            assert_refused(result, estimand)

    def test_the_derived_helper_keeps_its_own_boundary(self, cde_fits: dict[str, Any]) -> None:
        """A reported CDE ratio exists, but the cached retarget omits its fixed level."""
        result = cde_fits["ratio"][0.0]
        assert {"rr", "or"} <= result.estimates.keys()
        assert arm_parameter_keys(result)["rr"].estimand == "rr"
        assert derived_helper_refuses(replace(result))


class TestARequestedParameterGetsItsOwnRefusal:
    """The selector names the condition that makes each explicit request ineligible."""

    def test_a_level_has_no_reference_arm(self, plain_fits: dict[str, Any]) -> None:
        result = plain_fits["gaussian"]
        with pytest.raises(_EValueRefusal) as caught:
            evalue_module._select_evalue(result, "ey1")
        assert caught.value.status is AssessmentStatus.NOT_APPLICABLE
        assert str(caught.value) == (
            "an E-value needs a two-arm contrast; target 'ey1' has no reference arm"
        )

    @pytest.mark.parametrize(
        "key_changes,reason",
        [
            ({"axis": "shift"}, "an E-value needs an arm contrast, not the 'shift' axis"),
            (
                {"stratum": ("W", 1)},
                "this package requires an unconditioned marginal arm contrast; this contrast "
                "is conditional on a baseline stratum",
            ),
        ],
    )
    def test_an_axis_or_stratum_refusal_names_its_condition(
        self, plain_fits: dict[str, Any], key_changes: dict[str, Any], reason: str
    ) -> None:
        result = plain_fits["ratio"]
        keys = arm_parameter_keys(result)
        keys["rr"] = replace(keys["rr"], **key_changes)
        modified = replace(result, parameter_keys=keys)
        row = evalue_row(modified, "rr")
        assert row.status is AssessmentStatus.NOT_APPLICABLE
        assert row.reason == reason
        assert raised(lambda: evalue_module.evalue(modified, "rr")) == reason


#: The requests whose branch reads a stored estimate. ``derived_rr`` is left out, because its
#: helper refuses the fit by itself.
BLOCKED = tuple(
    case for case in REQUESTS if case.estimand is not None and case.branch != _DERIVED_RR
)


class TestTheRefusalBlocksANumber:
    """The nonzero witness: without the refusal, each blocked branch reports an E-value."""

    def test_the_blocked_requests_cover_every_other_branch(self) -> None:
        assert {case.branch for case in BLOCKED} == set(typing.get_args(_Branch)) - {_DERIVED_RR}

    @pytest.mark.parametrize("case", BLOCKED, ids=ids)
    @pytest.mark.parametrize("level", LEVELS)
    def test_the_branch_computes_on_the_fit(
        self, cde_fits: dict[str, Any], level: float, case: Request
    ) -> None:
        result = case.edit(cde_fits[case.law][level])
        selection = _EValueSelection(case.estimand, case.branch)  # type: ignore[arg-type]
        report = _evalue_from_selection(replace(result), selection)
        assert np.isfinite(report.point)
        assert report.point > 1.0

    def test_both_gaussian_levels_divide_by_one_observed_scale(
        self, cde_fits: dict[str, Any], plain_fits: dict[str, Any]
    ) -> None:
        """The standardized branch ignores the intervention on the intermediate variable.

        Both levels, and the fit without an intermediate variable, divide by the standard
        deviation of the observed outcome. Neither counterfactual mean has that scale.
        """
        selection = _EValueSelection("ate", "gaussian_difference")
        fits = [cde_fits["gaussian"][level] for level in LEVELS] + [plain_fits["gaussian"]]
        scales = {_evalue_from_selection(fit, selection).note.split(". ")[0] for fit in fits}
        assert len(scales) == 1
        assert scales.pop().startswith("Standardised by sd(Y) = ")


#: The module to import from; every surface that refuses a fit with an intermediate variable.
SURFACES = (
    "cleverly.sensitivity.evalue",
    "cleverly.sensitivity._derived",
    "cleverly.sensitivity.omitted_variable",
    "cleverly.sensitivity._simulated_confounding_request",
)


class TestOnePredicate:
    def test_the_predicate_reads_either_half(
        self, cde_fits: dict[str, Any], plain_fits: dict[str, Any]
    ) -> None:
        assert declares_intermediate(cde_fits["ratio"][0.0])
        assert not declares_intermediate(plain_fits["ratio"])
        assert declares_intermediate(level_only(plain_fits["ratio"]))
        assert declares_intermediate(column_only(plain_fits["ratio"]))

    @pytest.mark.parametrize("module", SURFACES)
    def test_every_sensitivity_surface_reads_one_predicate(self, module: str) -> None:
        surface = importlib.import_module(module)
        assert surface.declares_intermediate is direct_effect.declares_intermediate


#: The two lines of the check, which the mutations below move as one block.
CHECK = (
    "    if declares_intermediate(result):\n",
    "        raise _EValueRefusal(AssessmentStatus.UNAVAILABLE, _DIRECT_EFFECT_REFUSAL)\n",
)
#: The line after the Gaussian branch, where the contract's mutation puts the check.
BELOW_THE_GAUSSIAN_BRANCH = '    if key.estimand in {"att", "atc"}:\n'
#: The fixed-baseline branch, which the check stood above before RM21.
ABOVE_THE_FIXED_BASELINE = "    baseline = _baseline_mean(result, source, keys)\n"
#: The status refusal, the first check after the estimand resolves.
BESIDE_THE_STATUS_REFUSAL = "    if not result.estimates[source].supplies_inference:\n"


def rebuilt(anchor: str | None) -> Callable[..., Any]:
    """``_select_evalue`` compiled again from its own source, with the check moved above anchor.

    ``None`` compiles the source unchanged, which is the control. The mutant differs from the
    shipped function only in the position of the check. The annotations flag is needed because
    the source names ``TMLEResult``, which the module imports for type checking only.
    """
    lines = inspect.getsource(evalue_module._select_evalue).splitlines(keepends=True)
    starts = [at for at in range(len(lines) - 1) if tuple(lines[at : at + 2]) == CHECK]
    assert len(starts) == 1
    moved = list(lines)
    if anchor is not None:
        del moved[starts[0] : starts[0] + 2]
        assert moved.count(anchor) == 1
        at = moved.index(anchor)
        moved[at:at] = CHECK
        assert moved != lines
        assert sorted(moved) == sorted(lines)
    namespace = dict(vars(evalue_module))
    code = compile(
        "".join(moved),
        "<rebuilt _select_evalue>",
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)
    return typing.cast("Callable[..., Any]", namespace["_select_evalue"])


def leaks(cde_fits: dict[str, Any]) -> set[str]:
    """The requests at ``z = 0`` that do not refuse with the RM21 reason."""
    cases = [(case.name, case.edit(cde_fits[case.law][0.0]), case.estimand) for case in REQUESTS]
    cases += [(name, cde_fits[law][0.0], estimand) for name, law, estimand in LATE_REQUESTS]
    failed = set()
    for name, result, estimand in cases:
        try:
            assert_refused(result, estimand)
        except AssertionError:
            failed.add(name)
    return failed


#: The requests that return before the check when it stands below the Gaussian branch.
BELOW_THE_GAUSSIAN_LEAKS = frozenset(
    {
        "gaussian-default",
        "gaussian-ate",
        "gaussian-att",
        "gaussian-atc",
        "ratio-default",
        "ratio-rr",
        "ratio-or",
        "ey1",
        "multi-arm-default",
    }
)


class TestTheMutationsFailTheWitness:
    """Witness 2: each position of the check leaks an exact set of requests."""

    def test_m0_the_rebuilt_function_refuses_and_keeps(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cde_fits: dict[str, Any],
        plain_fits: dict[str, Any],
    ) -> None:
        monkeypatch.setattr(evalue_module, "_select_evalue", rebuilt(None))
        assert leaks(cde_fits) == set()
        for case in REQUESTS:
            assert_keeps(case.edit(plain_fits[case.law]), case.estimand, case.branch)

    def test_m1_the_check_below_the_gaussian_branch_leaks(
        self, monkeypatch: pytest.MonkeyPatch, cde_fits: dict[str, Any]
    ) -> None:
        """The contract's mutation. The leaked number is the one the blocked branch computes."""
        monkeypatch.setattr(evalue_module, "_select_evalue", rebuilt(BELOW_THE_GAUSSIAN_BRANCH))
        assert leaks(cde_fits) == BELOW_THE_GAUSSIAN_LEAKS
        result = cde_fits["gaussian"][0.0]
        blocked = _evalue_from_selection(result, _EValueSelection("ate", "gaussian_difference"))
        assert evalue_module.evalue(replace(result), "ate").point == blocked.point

    def test_m2_the_check_at_its_place_before_rm21_leaks(
        self, monkeypatch: pytest.MonkeyPatch, cde_fits: dict[str, Any]
    ) -> None:
        """The odds-ratio fallback also returns before the check."""
        monkeypatch.setattr(evalue_module, "_select_evalue", rebuilt(ABOVE_THE_FIXED_BASELINE))
        assert leaks(cde_fits) == BELOW_THE_GAUSSIAN_LEAKS | {"odds-default", "odds-cv-default"}

    def test_m3_the_check_after_the_estimand_resolves_leaks(
        self, monkeypatch: pytest.MonkeyPatch, cde_fits: dict[str, Any]
    ) -> None:
        """A level reads ``not_applicable``, and a multi-arm default reads ``deferred``."""
        monkeypatch.setattr(evalue_module, "_select_evalue", rebuilt(BESIDE_THE_STATUS_REFUSAL))
        assert leaks(cde_fits) == {"ey1", "multi-arm-default"}

    def test_m4_a_predicate_that_reads_no_fit_leaks_every_request(
        self, monkeypatch: pytest.MonkeyPatch, cde_fits: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(evalue_module, "declares_intermediate", lambda result: False)
        assert leaks(cde_fits) == EVERY_NAME

    def test_m5_the_derived_rule_removed_fails_its_own_witness(
        self, monkeypatch: pytest.MonkeyPatch, cde_fits: dict[str, Any]
    ) -> None:
        """The E-value refuses first, so only the direct call of the helper sees the rule.

        Without the rule the helper retargets, and ``retarget`` receives no level of the
        intermediate variable. So the witness above meets a ``DataError`` in place of its
        refusal.
        """
        monkeypatch.setattr(derived_module, "declares_intermediate", lambda result: False)
        assert leaks(cde_fits) == set()
        with pytest.raises(DataError, match="no intermediate_value was supplied"):
            derived_helper_refuses(replace(cde_fits["ratio"][0.0]))
