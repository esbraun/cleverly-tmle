r"""T1 to T5, T14 and T18: the IPSI estimator on exactly realised laws.

Each law below is realised exactly in its sample, so a saturated learner returns the law's
own nuisances.  Kennedy's estimator then reports the odds-tilt functional, and its curve is
the efficient influence function.  The oracle is :mod:`tests.discrete_law_longitudinal_ipsi`.
It states each parameter as a g-formula over finite-support masses, and it reads the
treatment mechanism off the perturbed law.  So its complex-step derivative carries the
mechanism term at every tilt node, and a fit that dropped or mis-weighted the term fails the
curve comparison.

The witness tests at the end assert that each mechanism term holds at least a tenth of the
influence function's variance, and that every arm difference the term multiplies is at
least 0.05.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.exceptions import CapabilityError, PositivityWarning
from cleverly.interventions import Incremental, Stochastic
from cleverly.longitudinal import IPSI, LTMLE, DynamicRegimen
from cleverly.msm import MSM

from .. import discrete_law_competing as competing
from .. import discrete_law_longitudinal as binary
from .. import discrete_law_longitudinal_ipsi as law
from .. import discrete_law_survival as survival
from ..longitudinal_ipsi import COLUMNS, known_treatment, regimens

POINT = 1e-12
CURVE = 1e-10
NO_TRUNCATION = (1e-8, 1.0)
CellMeans = binary.CellMeans

THREE_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2", "A3"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"], ["L3"]],
}
EVENT_COLUMNS: dict[str, Any] = {
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
TILTS = {label: plan for label, plan in law.PLANS.items() if all(n[0] == "tilt" for n in plan)}


def fit_ipsi(
    frame: Any,
    plans: Any,
    columns: dict[str, Any] = COLUMNS,
    *,
    declared: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """An in-sample fit with the saturated learner at every node.

    ``declared`` holds keywords of ``fit``: weights and known mechanisms.
    """
    settings: dict[str, Any] = {
        "reference": law.REFERENCE,
        "outcome_learner": CellMeans(),
        "pseudo_learner": CellMeans(),
        "treatment_learner": CellMeans(),
        "censoring_learner": CellMeans() if "censoring" in columns else None,
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    settings.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return IPSI(regimens(plans), **settings).fit(frame, **columns, **(declared or {}))


Expected = list[tuple[str, float, np.ndarray, np.ndarray]]


def _expected(
    names: tuple[str, ...], function: Callable[[str], Any], probs: np.ndarray, counts: np.ndarray
) -> Expected:
    rows = law.first_rows(counts)
    keep = probs > 0
    out = []
    for name in names:
        at = function(name)
        out.append((name, float(at(probs)), law.eif(at, probs)[keep], rows[keep]))
    return out


CASES: dict[str, Callable[[], tuple[Any, Expected]]] = {}


def _case(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    def register(builder: Callable[[], Any]) -> Callable[[], Any]:
        CASES[name] = builder
        return builder

    return register


@_case("base")
def _base() -> Any:
    names = law.names(law.PLANS, law.REFERENCE)
    return fit_ipsi(binary.frame(), law.PLANS), _expected(
        names, law.binary_function, binary.PROBS, binary.COUNTS
    )


@_case("tilt")
def _tilt() -> Any:
    names = law.names(law.PLANS, law.REFERENCE)
    return fit_ipsi(law.tilt_frame(), law.PLANS), _expected(
        names,
        lambda name: law.binary_function(name, base=law.TILT_PROBS),
        law.TILT_PROBS,
        law.TILT_COUNTS,
    )


@_case("three")
def _three() -> Any:
    names = law.names(law.THREE_PLANS, law.REFERENCE)
    return fit_ipsi(law.three_frame(), law.THREE_PLANS, THREE_COLUMNS), _expected(
        names, law.three_function, law.THREE_PROBS, law.THREE_COUNTS
    )


@_case("survival")
def _survival() -> Any:
    names = law.names(law.EVENT_PLANS, law.REFERENCE, survival.HORIZONS)
    columns = {"outcome": ["Y1", "Y2"], **EVENT_COLUMNS}
    fit = fit_ipsi(survival.frame(), law.EVENT_PLANS, columns, horizons=list(survival.HORIZONS))
    return fit, _expected(names, law.survival_function, survival.PROBS, survival.COUNTS)


@_case("competing")
def _competing() -> Any:
    columns = {"outcome": competing.outcome_columns(), **EVENT_COLUMNS}
    fit = fit_ipsi(competing.frame(), law.EVENT_PLANS, columns, horizons=list(competing.HORIZONS))
    return fit, _expected(
        law.competing_names(), law.competing_function, competing.PROBS, competing.COUNTS
    )


@_case("positivity")
def _positivity() -> Any:
    names = law.names(TILTS, law.REFERENCE)
    fit = fit_ipsi(law.positivity_frame(), TILTS, g_bounds=(0.05, 1.0))
    return fit, _expected(
        names,
        lambda name: law.binary_function(name, plans=TILTS, base=law.POSITIVITY_PROBS),
        law.POSITIVITY_PROBS,
        law.POSITIVITY_COUNTS,
    )


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request: Any) -> tuple[str, Any, Expected]:
    fit, expected = CASES[request.param]()
    return request.param, fit, expected


def test_every_reported_parameter_is_checked(case: Any) -> None:
    _, fit, expected = case
    assert fit.config.estimator == "ipsi"
    assert set(fit.estimates) == {name for name, *_ in expected}


def test_the_estimate_is_the_tilt_functional(case: Any) -> None:
    """T1 to T4 and T18: the estimate is the g-formula under the tilt, to ``1e-12``."""
    _, fit, expected = case
    for name, truth, _, _ in expected:
        assert fit.psi(name) == pytest.approx(truth, abs=POINT), name


def test_the_curve_is_the_gateaux_derivative(case: Any) -> None:
    """The curve is the efficient influence function, mechanism terms included."""
    _, fit, expected = case
    for name, _, eif, rows in expected:
        np.testing.assert_allclose(
            fit.influence_curves[name][rows], eif, atol=CURVE, rtol=0, err_msg=name
        )


def test_no_node_fits_a_fluctuation(case: Any) -> None:
    _, fit, _ = case
    for regimen_fit in fit.fits.values():
        for step in regimen_fit.steps:
            assert step.fluctuation is None
            np.testing.assert_array_equal(step.targeted, step.initial)


def test_the_estimate_is_the_mean_of_phi(case: Any) -> None:
    _, fit, _ = case
    for regimen_fit in fit.fits.values():
        first = regimen_fit.steps[0]
        mean = float(np.average(first.transform, weights=fit.data.weights))
        assert regimen_fit.psi_scaled == pytest.approx(mean, abs=1e-14)


# ------------------------------------------------------------------ T1r reductions


def test_the_natural_course_without_dropout_is_the_sample_mean() -> None:
    """T1r: at ``delta = 1`` on an uncensored law the estimate is the mean of ``Y``."""
    frame = law.three_frame()
    fit = fit_ipsi(frame, {"natural": law.THREE_PLANS["natural"]}, THREE_COLUMNS)
    y = frame["Y"].to_numpy(dtype=float)
    assert fit.psi("ey_regimen[natural]") == pytest.approx(float(np.mean(y)), abs=POINT)
    np.testing.assert_allclose(
        fit.influence_curves["ey_regimen[natural]"], y - np.mean(y), atol=POINT, rtol=0
    )


def test_a_unit_tilt_beside_a_label_is_the_label_then_natural_course() -> None:
    """T1r: a tilt node with ``delta = 1`` follows the observed mechanism there."""
    plans = {
        "natural": law.PLANS["natural"],
        "always_then_one": (law.arm(1), law.tilt(1.0)),
    }
    fit = fit_ipsi(law.tilt_frame(), plans)
    name = "ey_regimen[always_then_one]"
    truth = law.binary_function(name, plans=plans, base=law.TILT_PROBS)(law.TILT_PROBS)
    assert fit.psi(name) == pytest.approx(float(truth), abs=POINT)


# ------------------------------------------------------------------ witnesses


def _tilt_function(plan: Any, known: tuple[int, ...] = ()) -> Callable[[Any], Any]:
    return lambda p: law.functional_binary(p, plan, base=law.TILT_PROBS, known=known)


def _three_function(plan: Any, known: tuple[int, ...] = ()) -> Callable[[Any], Any]:
    return lambda p: law.functional_three(p, plan, known=known)


@pytest.mark.parametrize("delta", [0.5, 2.0])
@pytest.mark.parametrize("node", [1, 2])
def test_each_mechanism_term_holds_a_tenth_of_the_variance_on_the_tilt_law(
    delta: float, node: int
) -> None:
    """The M1, M5, M6 and M15 witness: the term a mutation drops is not small."""
    plan = (law.tilt(delta), law.tilt(delta))
    share = law.mechanism_share(_tilt_function(plan), _tilt_function(plan, (node,)), law.TILT_PROBS)
    assert share >= 0.10


@pytest.mark.parametrize("delta", [0.5, 2.0])
@pytest.mark.parametrize("node", [1, 2, 3])
def test_each_mechanism_term_holds_a_tenth_of_the_variance_on_the_three_node_law(
    delta: float, node: int
) -> None:
    plan = (law.tilt(delta),) * 3
    share = law.mechanism_share(
        _three_function(plan), _three_function(plan, (node,)), law.THREE_PROBS
    )
    assert share >= 0.10


@pytest.mark.parametrize("delta", [0.5, 2.0])
def test_every_arm_difference_is_at_least_five_hundredths(delta: float) -> None:
    for gaps in law.arm_gaps_binary(law.TILT_PROBS, (law.tilt(delta),) * 2).values():
        assert gaps.min() >= 0.05
    for gaps in law.arm_gaps_three(law.THREE_PROBS, (law.tilt(delta),) * 3).values():
        assert gaps.min() >= 0.05


def test_the_tilts_move_the_parameter() -> None:
    """``psi(delta)`` leaves ``psi(1)``, and the per-node plan differs from both constants."""
    truth = law.TILT_TRUTH
    natural = truth["ey_regimen[natural]"]
    for label in ("half", "double"):
        assert abs(truth[f"ey_regimen[{label}]"] - natural) > 0.01
    for label in ("half", "double"):
        assert abs(truth["ey_regimen[per_node]"] - truth[f"ey_regimen[{label}]"]) > 0.01
    three = {
        label: float(law.functional_three(law.THREE_PROBS, plan))
        for label, plan in law.THREE_PLANS.items()
    }
    assert abs(three["half"] - three["natural"]) > 0.01
    assert abs(three["double"] - three["natural"]) > 0.01


# ------------------------------------------------------------------ T5 weights


def _weight(w: int, a1: int, c1: int, l2: Any, a2: Any, c2: Any, y: Any) -> float:
    """A weight that reads the treatment, so the weighted mechanism differs (M18 witness)."""
    return 1.0 + a1 + 2.0 * (a2 or 0) + 0.5 * w


def test_the_weighted_fit_is_the_oracle_of_the_weighted_law() -> None:
    """T5: the weights reach the mechanism, the regressions and the mean."""
    cells = binary.cell_weights(_weight)
    frame = law.tilt_frame()
    frame["weight"] = np.repeat(cells, law.TILT_COUNTS)
    fit = fit_ipsi(frame, law.PLANS, declared={"weights": "weight"})
    rows = law.first_rows(law.TILT_COUNTS)
    keep = law.TILT_PROBS > 0
    for name in law.NAMES:
        at = law.binary_function(name, base=law.TILT_PROBS, weights=cells)
        assert fit.psi(name) == pytest.approx(float(at(law.TILT_PROBS)), abs=POINT), name
        np.testing.assert_allclose(
            fit.influence_curves[name][rows[keep]],
            law.eif(at, law.TILT_PROBS)[keep],
            atol=CURVE,
            rtol=0,
            err_msg=name,
        )


def test_the_weights_move_the_treatment_mechanism() -> None:
    """The M18 witness: weighted and unweighted :math:`\\pi_2` differ by more than 0.05."""
    cells = binary.cell_weights(_weight)
    weighted = law.weighted(law.TILT_PROBS, cells)
    gaps = []
    for w, a1, l2 in np.ndindex(2, 2, 2):
        history = {"w": w, "a1": a1, "c1": 1, "l2": l2}
        pi = [
            law._binary_mass(p, **history, a2=1) / law._binary_mass(p, **history)
            for p in (law.TILT_PROBS, weighted)
        ]
        gaps.append(abs(pi[0] - pi[1]))
    assert max(gaps) > 0.05


# ------------------------------------------------------------------ T10 remainder


def test_the_taylor_remainder_is_second_order_and_nonzero() -> None:
    """T10: along a path toward another law the remainder shrinks as the square of the step.

    The remainder over the step squared is the same number at three steps, and that number
    is not zero.  So the curve is the derivative, and the functional is not linear in the
    law.
    """
    probs = law.TILT_PROBS
    other = probs * (1.0 + 0.5 * np.sin(np.arange(probs.size)))
    direction = other / other.sum() - probs
    function = law.binary_function("ey_regimen[double]", base=probs)
    slope = float(np.nansum(direction * law.eif(function, probs)))
    ratios = [
        (float(function(probs + step * direction)) - float(function(probs)) - step * slope)
        / step**2
        for step in (1e-2, 5e-3, 2.5e-3)
    ]
    assert abs(ratios[-1]) > 1e-3
    np.testing.assert_allclose(ratios, ratios[-1], rtol=0.05)


def test_a_correct_mechanism_makes_any_regression_exact() -> None:
    """T10: with the mechanisms exact, a constant regression leaves no error."""
    wrong = DummyRegressor(strategy="constant", constant=0.3)
    fit = fit_ipsi(law.tilt_frame(), law.PLANS, outcome_learner=wrong, pseudo_learner=wrong)
    for name in law.NAMES:
        assert fit.psi(name) == pytest.approx(law.TILT_TRUTH[name], abs=POINT), name


def test_a_correct_regression_does_not_rescue_a_wrong_mechanism() -> None:
    r"""T10: the tilt is not doubly robust, and a label plan on the same fit is.

    The treatment learner predicts the marginal share, so every tilt node reads a wrong
    :math:`\pi`.  The label plan's error is zero to ``1e-12`` and each tilt's error
    exceeds ``1e-3`` (Kennedy 2019, Section 3.3).
    """
    plans = {label: law.PLANS[label] for label in ("natural", "half", "double", "always")}
    fit = fit_ipsi(law.tilt_frame(), plans, treatment_learner=DummyClassifier(strategy="prior"))

    def truth(label: str) -> float:
        return float(law.functional_binary(law.TILT_PROBS, plans[label], base=law.TILT_PROBS))

    assert fit.psi("ey_regimen[always]") == pytest.approx(truth("always"), abs=POINT)
    for label in ("half", "double"):
        assert abs(fit.psi(f"ey_regimen[{label}]") - truth(label)) > 1e-3, label


# ------------------------------------------------------------------ T11 working model

MSM_PLANS = {label: law.PLANS[label] for label in ("half", "natural", "double")}
LOG_DELTA = {"half": np.log(0.5), "natural": 0.0, "double": np.log(2.0)}


def _msm(columns: Callable[[str], list[float]], terms: tuple[str, ...]) -> MSM:
    def build(label: Any, horizon: int, frame: Any) -> np.ndarray:
        del horizon
        return np.tile(np.asarray(columns(label), dtype=float), (len(frame), 1))

    return MSM(design=build, terms=terms, link="identity", design_kind="known")


def test_the_log_delta_projection_is_the_gateaux_derivative() -> None:
    """T11: a model linear in ``log(delta)`` is a fixed linear map of the three plan means.

    With uniform weights and a design that reads no covariate, the coefficients are
    ``(X'X)^{-1} X' psi``.  So the oracle applies that map to the plan functionals, and the
    complex-step derivative of the map is the curve of each coefficient.
    """
    fit = fit_ipsi(
        law.tilt_frame(),
        MSM_PLANS,
        msm=_msm(lambda label: [1.0, LOG_DELTA[label]], ("1", "log_delta")),
        reference=None,
    )
    design = np.array([[1.0, LOG_DELTA[label]] for label in MSM_PLANS])
    projection = np.linalg.solve(design.T @ design, design.T)
    plans = [
        law.binary_function(f"ey_regimen[{label}]", plans=MSM_PLANS, base=law.TILT_PROBS)
        for label in MSM_PLANS
    ]
    rows = law.first_rows(law.TILT_COUNTS)
    keep = law.TILT_PROBS > 0
    for term, weights in zip(("1", "log_delta"), projection, strict=True):

        def coefficient(p: Any, weights: np.ndarray = weights) -> Any:
            return sum(w * plan(p) for w, plan in zip(weights, plans, strict=True))

        name = f"msm_regimen[{term}]"
        assert fit.psi(name) == pytest.approx(float(coefficient(law.TILT_PROBS)), abs=POINT)
        np.testing.assert_allclose(
            fit.influence_curves[name][rows[keep]],
            law.eif(coefficient, law.TILT_PROBS)[keep],
            atol=CURVE,
            rtol=0,
            err_msg=name,
        )
    assert abs(fit.psi("msm_regimen[log_delta]")) > 0.01


def test_a_saturated_working_model_is_the_plan_means() -> None:
    """T11: one indicator per plan returns each plan's mean and its curve."""
    labels = tuple(MSM_PLANS)
    fit = fit_ipsi(
        law.tilt_frame(),
        MSM_PLANS,
        msm=_msm(lambda label: [float(label == other) for other in labels], labels),
        reference=None,
    )
    plain = fit_ipsi(law.tilt_frame(), MSM_PLANS)
    for label in labels:
        mean = f"ey_regimen[{label}]"
        assert fit.psi(f"msm_regimen[{label}]") == pytest.approx(plain.psi(mean), abs=POINT)
        np.testing.assert_allclose(
            fit.influence_curves[f"msm_regimen[{label}]"],
            plain.influence_curves[mean],
            atol=POINT,
            rtol=0,
        )


# ------------------------------------------------------------------ T14 known mechanism


def test_a_known_mechanism_drops_the_mechanism_term() -> None:
    """T14: with :math:`\\pi` declared, the curve is the known-mechanism Gateaux derivative."""
    frame = law.tilt_frame()
    declared = known_treatment(frame, law.TILT_TABLES)
    fitted = fit_ipsi(
        frame,
        law.PLANS,
        treatment_learner=None,
        declared={"treatment_probabilities": declared},
    )
    rows = law.first_rows(law.TILT_COUNTS)
    keep = law.TILT_PROBS > 0
    for name in law.NAMES:
        at = law.binary_function(name, base=law.TILT_PROBS, known=(1, 2))
        assert fitted.psi(name) == pytest.approx(float(at(law.TILT_PROBS)), abs=POINT)
        np.testing.assert_allclose(
            fitted.influence_curves[name][rows[keep]],
            law.eif(at, law.TILT_PROBS)[keep],
            atol=CURVE,
            rtol=0,
            err_msg=name,
        )
    for regimen_fit in fitted.fits.values():
        assert regimen_fit.mechanism_term is not None
        np.testing.assert_array_equal(regimen_fit.mechanism_term, 0.0)


def _stochastic_tilt(delta: float, node: int) -> Stochastic:
    """The hand-built known policy :math:`q_t` of a tilt on the tilt law's own :math:`\\pi_t`."""

    def density(frame: Any) -> np.ndarray:
        w = np.asarray(frame["W"], dtype=int)
        if node == 1:
            pi = law.TILT_TABLES["G1"][w]
        else:
            a1 = np.asarray(frame["A1"], dtype=float).astype(int)
            l2 = np.asarray(frame["L2"], dtype=int)
            pi = law.TILT_TABLES["G2"][w, a1, l2]
        q = delta * pi / (delta * pi + 1.0 - pi)
        return np.column_stack([1.0 - q, q])

    return Stochastic(density, f"tilt{node}", density_kind="known")


def test_ltmle_fits_a_known_tilt_as_the_known_stochastic_policy() -> None:
    """T14: the translated plan is the hand-built known-policy fit, bit for bit."""
    frame = law.tilt_frame()
    declared = known_treatment(frame, law.TILT_TABLES)
    settings: dict[str, Any] = {
        "reference": "natural",
        "outcome_learner": CellMeans(),
        "pseudo_learner": CellMeans(),
        "censoring_learner": CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
    }
    hand = {
        "natural": DynamicRegimen("natural", (_stochastic_tilt(1.0, 1), _stochastic_tilt(1.0, 2))),
        "per_node": DynamicRegimen(
            "per_node", (_stochastic_tilt(2.0, 1), _stochastic_tilt(0.5, 2))
        ),
    }
    plans = {label: law.PLANS[label] for label in hand}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        translated = LTMLE(regimens(plans), **settings).fit(
            frame, **COLUMNS, treatment_probabilities=declared
        )
        built = LTMLE(hand, **settings).fit(frame, **COLUMNS, treatment_probabilities=declared)
    for name in law.names(plans, "natural"):
        assert translated.psi(name) == pytest.approx(built.psi(name), abs=POINT)
        np.testing.assert_allclose(
            translated.influence_curves[name], built.influence_curves[name], atol=POINT, rtol=0
        )
        assert translated.psi(name) == pytest.approx(law.TILT_TRUTH[name], abs=1e-10)


def test_a_known_tilt_needs_no_positivity_and_ltmle_refuses_the_bound() -> None:
    """T14 on the binding law: IPSI keeps the cancelled form, translated LTMLE refuses.

    The translated plan divides by the bounded declared mechanism.  Where ``g_bounds``
    moves it, the shipped preflight refuses the fit before any learner, rather than
    estimate a truncated functional.
    """
    frame = law.positivity_frame()
    declared = {"treatment_probabilities": known_treatment(frame, law.POSITIVITY_TABLES)}
    fitted = fit_ipsi(frame, TILTS, g_bounds=(0.05, 1.0), treatment_learner=None, declared=declared)
    rows = law.first_rows(law.POSITIVITY_COUNTS)
    keep = law.POSITIVITY_PROBS > 0
    for name in law.names(TILTS, law.REFERENCE):
        at = law.binary_function(name, plans=TILTS, base=law.POSITIVITY_PROBS, known=(1, 2))
        assert fitted.psi(name) == pytest.approx(float(at(law.POSITIVITY_PROBS)), abs=POINT)
        np.testing.assert_allclose(
            fitted.influence_curves[name][rows[keep]],
            law.eif(at, law.POSITIVITY_PROBS)[keep],
            atol=CURVE,
            rtol=0,
            err_msg=name,
        )
    with pytest.raises(CapabilityError, match="leaves the truncation bounds"):
        LTMLE(
            regimens(TILTS),
            outcome_learner=CellMeans(),
            censoring_learner=CellMeans(),
            n_folds=1,
            g_bounds=(0.05, 1.0),
        ).fit(frame, **COLUMNS, **declared)


def test_ltmle_refuses_a_tilt_with_an_estimated_mechanism() -> None:
    """R1 at the plan: an estimated :math:`\\pi` has no fluctuation over time."""
    with pytest.raises(CapabilityError, match="LTMLE fits no mechanism fluctuation over time"):
        LTMLE(
            {"double": Incremental(2.0)},
            outcome_learner=CellMeans(),
            treatment_learner=CellMeans(),
            censoring_learner=CellMeans(),
            n_folds=1,
        ).fit(law.tilt_frame(), **COLUMNS)


# ------------------------------------------------------------------ T18 no positivity


def test_the_tilt_needs_no_positivity_where_a_label_does() -> None:
    """T18: the tilt plans fit exactly (the ``positivity`` case); a label plan warns.

    The ``always`` plan divides by :math:`g_2 = 1/32` in cell ``[0, 0, 0]``, which the bound
    ``0.05`` replaces, so the shipped truncation warning fires.  A tilt never divides by
    :math:`\\pi_t`, so the tilt plans of the same fit are exact.
    """
    plans = {"natural": law.PLANS["natural"], "always": law.PLANS["always"]}
    with pytest.warns(PositivityWarning, match="always at t=2"):
        IPSI(
            regimens(plans),
            outcome_learner=CellMeans(),
            pseudo_learner=CellMeans(),
            treatment_learner=CellMeans(),
            censoring_learner=CellMeans(),
            n_folds=1,
            g_bounds=(0.05, 1.0),
            simultaneous=False,
        ).fit(law.positivity_frame(), **COLUMNS)
