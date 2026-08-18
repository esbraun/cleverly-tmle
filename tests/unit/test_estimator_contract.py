r"""What every estimator owes the subsystems around it, and what each of them refuses.

This module enforces the public contract directly: every subsystem keyed to
:class:`~cleverly.estimators.base.TMLEResult` has to be *either* reused
deliberately *or* refused by name, and "an ``AttributeError`` from a subsystem that was
never taught about a new result type is not a refusal; nor is a replicate loop's blanket
``except Exception`` turning a missing method into 'the fit is too unstable to bootstrap'".
Until now that was enforced by care.  ``LTMLE`` is the case it was written about, and there
will be another: a variant that cannot go through :meth:`TMLE._nuisances` needs its own
result object, and then every row of the matrix below has to be answered again.

**The roster is discovered, never written down.**  A hand-listed set is a list that a new
estimator is not on, which is the failure this module exists to catch -- so the estimators
come from ``cleverly.__all__`` and the report classes from a walk of the package.  What is
written down is the *exceptions*, each with a reason, and each checked for liveness in the
other direction: a refusal whose subsystem has since learned the result type, or an
exclusion naming a class that no longer exists, is a failure rather than a comment, because
a dead exemption reads as load-bearing.

Three things are checked, and they fail on disjoint mistakes:

* **the roster** -- an exported class with a public ``fit`` is either the point-treatment
  estimator or one of :data:`SEPARATE_ESTIMATORS`, with the reason it could not be the
  first;
* **the shared surface** -- every (result, subsystem) cell of :data:`SURFACE` either works
  or raises the declared exception with a message that says what the derivation would need.
  A cell answered by ``AttributeError`` fails whichever way it is declared;
* **the backend promise** -- narwhals returns results in the backend the caller passed in.
  That promise was once broken in six ``to_frame``\ s at once because
  ``emit_frame``'s ``data=`` defaulted to ``None`` and nothing ever passed it.  That is a
  property of a *class*, so it is checked over every class in the package that has a
  ``to_frame`` at all, statically, and then confirmed on a real polars fit.

The static half is what makes the dynamic half more than a spot check: a report class added
tomorrow and not yet reachable from any fit here still has to declare how it routes the
backend.

**Each of the four was watched to fail before it was left passing**, which is the only way
a structural check earns its place -- one that cannot fail is a comment with a runtime.
Renaming the ``SuperLearner`` exemption turned the roster test *and* the liveness test red;
declaring ``LongitudinalResult.save`` supported turned its cell red, and the leftover row
turned the coverage test red with it; deleting ``ScoreCheck``'s row turned the route test
red; and changing ``ScoreCheck.to_frame`` in ``src`` to pass ``backend=None`` -- the exact
shape of the six-``to_frame`` bug -- turned the polars sweep red, which is the one that
matters, since that mutation leaves every other test in the suite green.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from collections.abc import Mapping
from typing import Any

import polars as pl
import pytest

import cleverly
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.estimators import TMLE
from cleverly.longitudinal import LTMLE
from cleverly.methods import CollaborativeTMLEMethod, DRTMLEMethod, TMLEMethod
from tests.conftest import assert_estimate_coherent

# --------------------------------------------------------------------------- the roster

#: Exported classes with a public ``fit`` that answer no causal parameter at all, and so
#: owe the subsystems below nothing.  Kept separate from :data:`SEPARATE_ESTIMATORS`
#: because the two say different things: this one says "not an estimator", that one says
#: "an estimator whose parameter a ``Target`` cannot express".
NON_ESTIMATORS: dict[str, str] = {
    "SuperLearner": (
        "a nuisance learner, with scikit-learn's fit/predict rather than cleverly's. It "
        "produces a prediction, not a result object, so there is no estimate for a "
        "sensitivity analysis or an influence curve to be about"
    ),
}

#: Exported classes with a public ``fit`` that are deliberately *not* the point-treatment
#: estimator, and why each could not be.  A row here is a claim that the parameter cannot
#: reuse the point-treatment result contract.
SEPARATE_ESTIMATORS: dict[str, str] = {}


def _exported_fit_classes() -> dict[str, type]:
    """Every class ``cleverly`` exports that has a public ``fit``, read off ``__all__``."""
    found: dict[str, type] = {}
    for name in cleverly.__all__:
        obj = getattr(cleverly, name)
        if not inspect.isclass(obj):
            continue
        fit = inspect.getattr_static(obj, "fit", None)
        if fit is not None and (
            inspect.isfunction(fit) or isinstance(fit, (staticmethod, classmethod))
        ):
            found[name] = obj
    return found


FIT_CLASSES = _exported_fit_classes()


def test_the_beginner_facing_root_is_pinned() -> None:
    expected = {
        "ATC",
        "ATE",
        "ATT",
        "BackdoorMeanContrast",
        "CapabilityError",
        "CausalResult",
        "CausalStudy",
        "CleverlyError",
        "CollaborativeTMLEMethod",
        "ControlledDirectEffect",
        "ConvergenceWarning",
        "CounterfactualMean",
        "CrossFitting",
        "DRTMLEMethod",
        "DataError",
        "Estimand",
        "EstimationMethod",
        "ExplicitAdjustmentProvider",
        "IdentificationProvider",
        "IdentifiedEffect",
        "IncrementalEffect",
        "IncrementalMean",
        "Inference",
        "LongitudinalTreatment",
        "MSMProjection",
        "MethodAvailability",
        "MethodConfigurationError",
        "ModelSpec",
        "ModifiedTreatmentPolicy",
        "ModifiedTreatmentPolicyEffect",
        "NaturalCourseMean",
        "NotFittedError",
        "OddsRatio",
        "ParameterEstimate",
        "ParameterKey",
        "PointTreatment",
        "PopulationAttributableFraction",
        "PopulationAttributableRisk",
        "PositivityWarning",
        "Provenance",
        "RegimeContrast",
        "RegimeMean",
        "RiskRatio",
        "Runtime",
        "SuperLearner",
        "TMLEMethod",
        "Targeting",
        "VariableImportanceEntry",
        "VariableImportanceResult",
        "WeightingWarning",
        "__version__",
        "load",
        "variable_importance",
    }
    assert set(cleverly.__all__) == expected


def test_the_roster_is_not_empty() -> None:
    # A discovery that silently found nothing would make every test below vacuous, which
    # is the one way a structural check fails open.
    assert {"SuperLearner": cleverly.SuperLearner} == FIT_CLASSES
    assert all(not hasattr(cleverly, name) for name in ("TMLE", "CTMLE", "DRTMLE", "LTMLE"))


def test_public_estimation_methods_are_typed_instead_of_fit_constructors() -> None:
    methods = (TMLEMethod(), CollaborativeTMLEMethod(), DRTMLEMethod())
    assert {method.name for method in methods} == {"tmle", "collaborative_tmle", "drtmle"}
    assert all(dataclasses.is_dataclass(method) for method in methods)


def test_every_fitting_class_is_a_tmle_or_says_why_not() -> None:
    excused = set(SEPARATE_ESTIMATORS) | set(NON_ESTIMATORS)
    stray = sorted(
        name
        for name, obj in FIT_CLASSES.items()
        if name not in excused and not issubclass(obj, TMLE)
    )
    assert stray == [], (
        f"exported fit-classes that are neither a TMLE subclass nor a declared separate "
        f"estimator: {stray}. Either override TMLE._nuisances -- which is what CTMLE does, "
        f"and why every influence curve, sensitivity analysis and diagnostic keeps working "
        f"untouched -- or add a SEPARATE_ESTIMATORS entry saying what the parameter needs "
        f"that a Target cannot express, and answer every row of SURFACE for its result"
    )


@pytest.mark.parametrize("excused", [SEPARATE_ESTIMATORS, NON_ESTIMATORS], ids=["separate", "not"])
def test_exemption_entries_are_live(excused: dict[str, str]) -> None:
    # A dead exemption reads as load-bearing: the next reader takes it for a standing
    # decision rather than for a row nobody removed.
    for name in excused:
        assert name in FIT_CLASSES, f"stale exemption: {name} is not an exported fit-class"
        assert not issubclass(FIT_CLASSES[name], TMLE), (
            f"{name} is a TMLE subclass now, so it inherits the shared surface; "
            f"remove its exemption"
        )


# ------------------------------------------------------------------- the shared surface


def _two_parameters(result: Any) -> list[str]:
    """The first two parameter names a result reports, whichever way it indexes them.

    A longitudinal result *is* a mapping over its names; a point-treatment one holds them
    in ``estimates``.  The difference is the whole reason this module exists, so it is read
    here rather than assumed away.
    """
    names = list(result) if isinstance(result, Mapping) else list(result.estimates)
    assert len(names) >= 2, f"{type(result).__name__} reports {names}; a contrast needs two"
    return names[:2]


@dataclasses.dataclass(frozen=True)
class Refusal:
    """A cell a result declines to answer, and the shape of the decline."""

    #: What is raised.  ``NotImplementedError`` for a subsystem that could exist and does
    #: not; ``ValueError`` for a report this *fit* has no content for.
    error: type[BaseException]
    #: A fragment of the message, so a refusal cannot be quietly reworded into silence.
    says: str
    #: Why the subsystem was not taught this result type.
    because: str


#: The subsystems every result is asked about, and what each result answers.  A callable
#: takes the result and exercises the cell; the value is either ``...`` for "this works" or
#: a :class:`Refusal`.  Nothing here asserts a *number* -- that is what the oracle laws are
#: for.  What it asserts is that the cell was answered on purpose.
SURFACE: dict[str, dict[str, Any]] = {
    "to_frame": {
        "call": lambda r, tmp: r.to_frame(),
        "TMLEResult": ...,
        "LongitudinalResult": ...,
    },
    "summary": {
        "call": lambda r, tmp: r.summary(),
        "TMLEResult": ...,
        "LongitudinalResult": ...,
    },
    "influence_curves": {
        "call": lambda r, tmp: r.influence_curves,
        "TMLEResult": ...,
        "LongitudinalResult": ...,
    },
    "covariance": {
        "call": lambda r, tmp: r.covariance(),
        "TMLEResult": ...,
        "LongitudinalResult": ...,
    },
    "contrast": {
        "call": lambda r, tmp: r.contrast(lambda p: p[0] - p[1], _two_parameters(r)),
        "TMLEResult": ...,
        "LongitudinalResult": ...,
    },
    "sensitivity": {
        "call": lambda r, tmp: r.sensitivity,
        "TMLEResult": ...,
        "LongitudinalResult": Refusal(
            NotImplementedError,
            "not available on a longitudinal fit",
            "every analysis in the suite re-targets against cached nuisances, and g_bounds "
            "enters the pseudo-outcome of every earlier node through the recursion -- so "
            "there is no retarget that re-solves the fluctuation alone. diagnostics() "
            "reports the leverage instead",
        ),
    },
    "validation": {
        "call": lambda r, tmp: r.validation,
        "TMLEResult": ...,
        "LongitudinalResult": Refusal(
            NotImplementedError,
            "not available on a longitudinal fit",
            "the suite reads result.repeats and result.estimator, which a longitudinal "
            "result does not carry; the score equations it would check are already "
            "reported per node by diagnostics()",
        ),
    },
    "save": {
        "call": lambda r, tmp: r.save(tmp / "result.joblib"),
        "TMLEResult": ...,
        "LongitudinalResult": ...,
    },
    "diagnostics": {
        "call": lambda r, tmp: r.diagnostics(),
        "TMLEResult": Refusal(
            AttributeError,
            "diagnostics",
            "the point-treatment fit has no nodes to report per, and what a longitudinal "
            "diagnostics() gives -- cumulative weight and effective n per node -- is "
            "reported here by result.sensitivity.positivity(). This is the one cell "
            "answered by an AttributeError on purpose, because the name was never part of "
            "the point-treatment surface rather than removed from it",
        ),
        "LongitudinalResult": ...,
    },
    "coefficients": {
        "call": lambda r, tmp: r.coefficients(),
        "TMLEResult": Refusal(
            ValueError,
            "no working model",
            "a coefficient is a parameter of an msm=, and this fit declared none; the "
            "refusal is about the fit rather than about the result class",
        ),
        "LongitudinalResult": Refusal(
            ValueError,
            "no working model",
            "as on the point-treatment path: msm= is what makes beta a parameter",
        ),
    },
    "curve": {
        "call": lambda r, tmp: r.curve(),
        "TMLEResult": Refusal(
            AttributeError,
            "curve",
            "a curve is indexed by a horizon, which a point-treatment fit does not have. "
            "Declared, like diagnostics, so that adding one here is a decision",
        ),
        "LongitudinalResult": Refusal(
            ValueError,
            "one end-of-study outcome",
            "this fixture's fit reports a number rather than a curve; outcome=[...] is "
            "what makes the report a cumulative risk at every horizon",
        ),
    },
}

#: The one exception type that is *never* an acceptable answer for a cell declared to work,
#: and is acceptable for a refusal only where :data:`SURFACE` says so in writing.
NOT_A_REFUSAL = AttributeError


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    """One fit per result class, in polars, shared by every test below.

    Small and parametric on purpose: nothing here reads an estimate, so the fit exists only
    to have a result object with real arrays behind it.
    """
    point_frame, _ = make_linear_ate(n=400, seed=0)
    point = (
        TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            n_folds=2,
            learner_folds=2,
            random_state=0,
            simultaneous=False,
        )
        .fit(pl.from_pandas(point_frame), outcome="Y", treatment="A")
        .single()
    )
    long_frame, _ = make_longitudinal(n=400, seed=11)
    longitudinal = LTMLE(
        {"always": 1, "never": 0},
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        n_folds=2,
        learner_folds=2,
        random_state=0,
    ).fit(
        pl.from_pandas(long_frame),
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    return {"TMLEResult": point, "LongitudinalResult": longitudinal}


def test_the_surface_covers_every_result_the_roster_produces(results) -> None:
    # Both directions, so a result class added without rows fails here rather than being
    # swept over silently, and a row for a class nobody produces fails too.
    produced = set(results)
    for subsystem, row in SURFACE.items():
        declared = set(row) - {"call"}
        assert declared == produced, (
            f"SURFACE['{subsystem}'] answers for {sorted(declared)} but the roster "
            f"produces {sorted(produced)}"
        )


@pytest.mark.parametrize("subsystem", sorted(SURFACE))
@pytest.mark.parametrize("result_name", ["TMLEResult", "LongitudinalResult"])
def test_every_cell_is_answered_on_purpose(
    results, tmp_path, result_name: str, subsystem: str
) -> None:
    row = SURFACE[subsystem]
    result = results[result_name]
    assert type(result).__name__ == result_name
    expected = row[result_name]

    if expected is ...:
        # Supported: it must not raise at all.  A cell that started refusing is as much a
        # silent change as one that started working.
        row["call"](result, tmp_path)
        return

    with pytest.raises(expected.error) as raised:
        row["call"](result, tmp_path)
    message = str(raised.value)
    assert expected.says in message, (
        f"{result_name}.{subsystem} refuses with a message that no longer contains "
        f"{expected.says!r}: {message!r}"
    )
    if expected.error is not NOT_A_REFUSAL:
        # A subsystem that was never taught the result type raises AttributeError, and that
        # is not a refusal.  Where the declared error
        # is something else, an AttributeError reaching here would have failed the raises
        # block above -- this pins the remaining half, that the message says something.
        assert len(message) > 80, (
            f"{result_name}.{subsystem} refuses in {len(message)} characters. A refusal "
            f"says what the derivation would need, so that the reader knows whether they "
            f"are looking at 'not written yet' or at 'wrong by construction'"
        )


# ------------------------------------------------------------------ the backend promise


def _package_classes() -> dict[str, type]:
    """Every class the package defines, from a walk rather than from a list."""
    modules = [cleverly]
    for found in pkgutil.walk_packages(cleverly.__path__, "cleverly."):
        modules.append(importlib.import_module(found.name))
    classes: dict[str, type] = {}
    for module in modules:
        for obj in vars(module).values():
            if inspect.isclass(obj) and obj.__module__.startswith("cleverly"):
                classes[obj.__qualname__] = obj
    return classes


#: How a class that emits a frame gets hold of the caller's backend.  Every class with a
#: ``to_frame`` has to be in exactly one of these, and the name of the route is the claim:
#:
#: * ``field`` -- it carries ``backend: str | None`` and passes it to ``emit_frame``: a
#:   name, not a frame, because holding the input
#:   frame pinned it in memory for the life of every result and could not be restored by
#:   ``load()``;
#: * ``container`` -- it holds the :class:`~cleverly.data.CausalData` and emits through
#:   ``data.frame_like``, which reads the same stored name;
#: * ``argument`` -- the backend is a parameter of ``to_frame`` because the object was not
#:   built from a fit at all.
BACKEND_ROUTES: dict[str, str] = {
    "CausalResult": "protocol",
    "CausalData": "field",
    "CorrectionCheck": "field",
    "CVTargeting": "field",
    "NuisanceDiagnostics": "field",
    "PositivityReport": "field",
    "RefutationResult": "field",
    "ScoreCheck": "field",
    "SupportReport": "field",
    "TMLEResult": "container",
    "TMLEResultSet": "container",
    "VariableImportanceResult": "field",
    "LongitudinalResult": "container",
    "CTMLESelection": "argument",
    "StudyResult": "argument",
}


def test_every_frame_emitting_class_declares_how_it_finds_the_backend() -> None:
    emitters = {name for name, obj in _package_classes().items() if "to_frame" in dir(obj)}
    assert emitters, "the package walk found no frame-emitting classes at all"

    unrouted = sorted(emitters - set(BACKEND_ROUTES))
    assert unrouted == [], (
        f"classes with a to_frame and no declared backend route: {unrouted}. Results are "
        f"returned in the backend the caller passed in; the field defaulting to None and "
        f"nothing ever passing it is how six to_frames came to return pandas for a polars "
        f"fit. Add the field, or add a row saying which route this one takes"
    )
    stale = sorted(set(BACKEND_ROUTES) - emitters)
    assert stale == [], f"BACKEND_ROUTES rows for classes with no to_frame: {stale}"


@pytest.mark.parametrize(
    "name", sorted(n for n, route in BACKEND_ROUTES.items() if route == "field")
)
def test_the_field_route_really_carries_a_backend_field(name: str) -> None:
    obj = _package_classes()[name]
    fields = {f.name for f in dataclasses.fields(obj)} if dataclasses.is_dataclass(obj) else set()
    assert "backend" in fields or hasattr(obj, "backend"), (
        f"{name} is declared to route the backend through a field it does not have"
    )


#: What a polars fit is asked to emit.  Reachability is the point: the static test above
#: says every emitter declares a route, and this says the routes that a fit actually walks
#: arrive at polars.  A cell here that the fixture cannot reach belongs in the static test,
#: not in a skip.
POLARS_EMITTERS: dict[str, list[Any]] = {
    "TMLEResult": [
        lambda r: r.to_frame(),
        lambda r: r.influence_frame(),
        lambda r: r.sensitivity.positivity().to_frame(),
        # SupportReport is deliberately absent: it reports overlap for *declared regimes*
        # and refuses on an arm-indexed fit, so reaching it would need a second fixture
        # fit. Its route is pinned statically above, which is the division of labour this
        # module is built on -- a class declares its route, a fit confirms the ones it
        # walks.
        lambda r: r.validation.score_check().to_frame(),
        lambda r: r.validation.nuisance().to_frame(),
        lambda r: r.validation.refute().to_frame(),
    ],
    "LongitudinalResult": [
        lambda r: r.to_frame(),
        lambda r: r.diagnostics(),
    ],
}


@pytest.mark.parametrize("result_name", sorted(POLARS_EMITTERS))
def test_a_polars_fit_emits_polars_everywhere_it_is_reachable(results, result_name: str) -> None:
    result = results[result_name]
    wrong = []
    for index, emit in enumerate(POLARS_EMITTERS[result_name]):
        frame = emit(result)
        if not isinstance(frame, pl.DataFrame):
            wrong.append((index, type(frame).__name__))
    assert wrong == [], (
        f"{result_name} emitters returning something other than polars for a polars fit: "
        f"{wrong}. The promise is about the *library*, not the dtype backend"
    )


# ------------------------------------------------------------ the estimates themselves


def _reported(result: Any) -> dict[str, Any]:
    return dict(result) if isinstance(result, Mapping) else dict(result.estimates)


@pytest.mark.parametrize("result_name", ["TMLEResult", "LongitudinalResult"])
def test_every_reported_estimate_is_internally_coherent(results, result_name: str) -> None:
    """The variance is the curve's, and the interval and the p-value agree about the null.

    Swept over *every* parameter both results report rather than spot-checked on one,
    because the mistakes this catches -- a curve that drifted from its variance, a scale
    confusion between the two null branches -- are per-parameter and would sit under a
    check of ``ate`` alone.
    """
    reported = _reported(results[result_name])
    assert reported, f"{result_name} reported no parameters at all"
    for estimate in reported.values():
        assert_estimate_coherent(estimate)


def test_the_coherence_check_can_fail(results) -> None:
    """The negative control, without which the sweep above is a comment with a runtime.

    Scaling the curve leaves ``psi``, ``variance``, ``ci`` and ``pvalue`` all exactly as
    they were and mutually consistent -- it is *only* the recomputation from the returned
    state that sees it, which is the whole argument for making that recomputation.
    """
    estimate = next(iter(_reported(results["TMLEResult"]).values()))
    drifted = dataclasses.replace(estimate, influence_curve=estimate.influence_curve * 1.5)
    with pytest.raises(AssertionError, match="is not the variance of the curve"):
        assert_estimate_coherent(drifted)
    # ...and it is genuinely only that clause: everything else about the drifted estimate
    # still passes, which is why the recomputation is not redundant with the rest.
    assert_estimate_coherent(drifted, variance_from_curve=False)
