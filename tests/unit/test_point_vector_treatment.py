"""A point vector treatment: a flag and a dose fitted together, against the mixed-law oracle.

The treatment is ``A = (F, D)``, a binary flag and a dose.  Two readings exist.

- With ``continuous_treatment=["D"]`` the dose is continuous, and the one supported fit is
  ``TMLE(policies=[JointPolicy(...)], ratio="classifier")``.  The stacked classifier of
  Díaz, Williams, Hoffman and Schenck (2023, Section 5.4) reads both components beside the
  covariates.
- Without it, both columns are categorical and the treatment is one categorical treatment
  over the observed tuples, read by every categorical estimand and by the discrete formula
  of a modified treatment policy.

The oracle is the mixed law of ``tests/discrete_law_policy_point.py``.  Its frame realises
the law by integer counts, and the learners are saturated, so every exact fit equals the
g-formula and its complex-step Gateaux derivative.  W8 is the targeting witness, where the
outcome regression is misspecified and the fluctuation moves.  The mutation controls are in
``tests/unit/test_point_vector_mutations.py``.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tests.discrete_law_policy_point as law
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import JointPolicy, ModifiedPolicy, Piece, Randomizer, Shift
from cleverly.targets import TARGETS
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_longitudinal_multivalue import CellProbabilities

pytestmark = pytest.mark.xdist_group("point_vector_treatment")

#: What every one-column reader says at a vector treatment with a continuous component.
ONE_COLUMN = "reads a treatment of one column. Such a treatment has no arms"

EXACT = 1e-10
INF = np.inf
COLUMNS: dict[str, Any] = {"outcome": "Y", "treatment": ["F", "D"], "covariates": ["W"]}
#: The policies of W1: every mixed-law plan the classifier route fits.
LABELS = ("natural", "dose up", "flag gated", "dose gated", "random")


# ------------------------------------------------------------------ policies
#
# Module-level functions, so a fitted result pickles.


def set_flag(a: Any, h: Any) -> np.ndarray:
    """Set the flag to one."""
    return np.ones(len(a))


def flag_boundary(h: Any) -> np.ndarray:
    """Doses below 1.5 move where the natural flag is one; none move where it is zero."""
    return np.where(np.asarray(h["F"], dtype=float) == 1.0, 1.5, -INF)


def dose_up(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + 1.0


def dose_down(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - 1.0


def unit_slope(b: Any, h: Any) -> np.ndarray:
    return np.ones(len(b))


def tuple_dose_up(a: Any, h: Any) -> list[tuple[Any, float]]:
    """The categorical reading of "dose up": a map of tuple labels to tuple labels."""
    return [(f, min(float(d) + 1.0, 2.0)) for f, d in a]


UP = Shift(1.0, cap=2.0)
UP_IF_FLAGGED = ModifiedPolicy(
    "up if flagged",
    pieces=(
        Piece(-INF, flag_boundary, dose_up, dose_down, unit_slope),
        Piece(flag_boundary, INF),
    ),
    policy_kind="known",
)
SET_FLAG = ModifiedPolicy("set flag", apply=set_flag, policy_kind="known")


def joint(label: str) -> JointPolicy:
    """The declaration of a mixed-law plan, by its oracle label."""
    if label == "natural":
        return JointPolicy("natural", components={}, policy_kind="known")
    if label == "dose up":
        return JointPolicy("dose up", components={"D": UP}, policy_kind="known")
    if label == "flag gated":
        return JointPolicy(
            "flag gated", components={"F": SET_FLAG, "D": UP_IF_FLAGGED}, policy_kind="known"
        )
    if label == "dose gated":
        return JointPolicy("dose gated", components={"D": UP_IF_FLAGGED}, policy_kind="known")
    if label == "random":
        return JointPolicy(
            "random",
            components={"up": {"D": UP}, "stay": {}},
            randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
            policy_kind="known",
        )
    raise KeyError(label)


# ------------------------------------------------------------------ fits


def vector_data(frame: pd.DataFrame | None = None, **columns: Any) -> CausalData:
    """The mixed reading: ``D`` is a continuous component."""
    declared = {**COLUMNS, **columns}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalData.from_frame(
            law.mixed_frame() if frame is None else frame, continuous_treatment=["D"], **declared
        )


def categorical_data(frame: pd.DataFrame | None = None, **columns: Any) -> CausalData:
    """The all-categorical reading: the treatment levels are the observed tuples."""
    declared = {**COLUMNS, **columns}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalData.from_frame(law.mixed_frame() if frame is None else frame, **declared)


def settings(**overrides: Any) -> dict[str, Any]:
    """Saturated learners, one fold, one fit per parameter."""
    return {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellMeans(),
        "cross_fit": False,
        "ratio": "classifier",
        "simultaneous": False,
        **overrides,
    }


def fit(
    labels: Sequence[str] = LABELS,
    data: CausalData | None = None,
    policies: Sequence[object] | None = None,
    **overrides: Any,
) -> Any:
    declared = [joint(label) for label in labels] if policies is None else list(policies)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(policies=declared, **settings(**overrides))
            .fit(vector_data() if data is None else data)
            .single()
        )


def deviation(result: Any, labels: Sequence[str], truth: Sequence[str] | None = None) -> float:
    """The largest error of an estimate or of its influence curve against the oracle."""
    rows = law.mixed_first_row_of()
    worst = 0.0
    for label, oracle in zip(labels, labels if truth is None else truth, strict=True):
        estimate = result.estimates[f"ey_policy[{label}]"]
        worst = max(worst, abs(estimate.psi - law.mixed_truth(oracle)))
        curve = np.asarray(estimate.influence_curve)[rows]
        worst = max(worst, float(np.max(np.abs(curve - law.mixed_eif(oracle)))))
    return worst


def misspecified_settings(**overrides: Any) -> dict[str, Any]:
    """A misspecified outcome regression, so the fluctuation has work to do."""
    from sklearn.linear_model import LogisticRegression

    return settings(outcome_learner=LogisticRegression(), **overrides)


# ------------------------------------------------------------------ storage


def test_a_dose_component_makes_a_vector_treatment() -> None:
    frame = law.mixed_frame()
    data = vector_data(frame)
    assert data.treatment_kind == "vector"
    assert data.is_vector_treatment
    assert data.treatment_name == "F+D"
    assert data.component_names == ("F", "D")
    assert np.all(np.isnan(data.treatment))
    assert data.treatment_components is not None
    np.testing.assert_array_equal(data.treatment_components, frame[["F", "D"]].to_numpy())
    # The flag enters as its drop-first indicator and the dose raw.
    np.testing.assert_array_equal(data.component_design(), frame[["F", "D"]].to_numpy())
    restored = data.to_frame()
    np.testing.assert_array_equal(restored[["F", "D"]].to_numpy(), frame[["F", "D"]].to_numpy())


#: Each one-column reader, called on the data.
READERS: dict[str, Any] = {
    "n_arms": lambda data: data.n_arms,
    "arm_codes": lambda data: data.arm_codes,
    "arm_label": lambda data: data.arm_label(0.0),
    "is_continuous_treatment": lambda data: data.is_continuous_treatment,
    "is_binary_treatment": lambda data: data.is_binary_treatment,
    "treatment_block": lambda data: data.treatment_block(np.zeros(data.n)),
    "with_known_mechanism": lambda data: data.with_known_mechanism(np.full((data.n, 2), 0.5)),
}


@pytest.mark.parametrize("reader", sorted(READERS))
def test_a_one_column_reader_refuses_a_vector_treatment(reader: str) -> None:
    data = vector_data()
    with pytest.raises(CapabilityError, match=rf"vector treatment.*{ONE_COLUMN}"):
        READERS[reader](data)


def test_an_all_categorical_vector_is_one_categorical_treatment_over_its_tuples() -> None:
    data = categorical_data()
    assert data.treatment_kind == "discrete"
    assert data.treatment_name == "F+D"
    assert data.component_names == ("F", "D")
    assert data.treatment_levels == (
        (0, 0.0),
        (0, 1.0),
        (0, 2.0),
        (1, 0.0),
        (1, 1.0),
        (1, 2.0),
    )
    assert data.treatment_components is None


# ------------------------------------------------------------------ W1, W4, W11


def test_w1_every_policy_mean_and_influence_curve_is_exact() -> None:
    result = fit()
    assert deviation(result, LABELS) < EXACT
    rows = law.mixed_first_row_of()
    contrast = result.estimates["ate_policy[flag gated vs natural]"]
    truth = law.mixed_truth("flag gated") - law.mixed_truth("natural")
    assert contrast.psi == pytest.approx(truth, abs=EXACT)
    eif = law.mixed_eif("flag gated") - law.mixed_eif("natural")
    np.testing.assert_allclose(np.asarray(contrast.influence_curve)[rows], eif, atol=EXACT)


def test_w1_the_policies_are_not_degenerate() -> None:
    """The plans differ pairwise, and each moves the mean away from the natural course."""
    truths = {label: law.mixed_truth(label) for label in LABELS}
    values = list(truths.values())
    for i, first in enumerate(values):
        for second in values[i + 1 :]:
            assert abs(first - second) > 1e-4
    for label in LABELS[1:]:
        assert abs(truths[label] - truths["natural"]) > 0.003


def test_w4_the_dose_map_reads_the_natural_flag_after_the_flag_moves() -> None:
    """The flag-gated plan reads the natural flag; its shifted-flag reading is far away."""
    natural = law.mixed_truth("flag gated")
    shifted = law.mixed_truth("flag gated (shifted flag)")
    assert abs(natural - shifted) >= 1e-2
    result = fit(("natural", "flag gated"))
    assert result.psi("ey_policy[flag gated]") == pytest.approx(natural, abs=EXACT)


def test_w11_a_known_randomizer_mixes_its_branches() -> None:
    """The randomized plan is exact (inside W1) and is the mixture of its branch means."""
    result = fit(("natural", "dose up", "random"))
    mixture = 0.25 * result.psi("ey_policy[dose up]") + 0.75 * result.psi("ey_policy[natural]")
    assert result.psi("ey_policy[random]") == pytest.approx(mixture, abs=EXACT)
    assert deviation(result, ("random",)) < EXACT


def test_l6_the_classifier_reads_the_components_beside_the_covariates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each stacked design is ``[flag indicator, dose, W]``, never a joint level code."""
    from cleverly.learners import density_ratio

    designs: list[np.ndarray] = []
    original = density_ratio.fit_learner

    def record(learner: Any, design: Any, target: Any, weight: Any, **kwargs: Any) -> Any:
        designs.append(np.asarray(design))
        return original(learner, design, target, weight, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", record)
    frame = law.mixed_frame()
    fit(("natural",))
    assert len(designs) == 1
    natural = frame[["F", "D", "W"]].to_numpy()
    np.testing.assert_array_equal(designs[0], np.vstack([natural, natural]))


# ------------------------------------------------------------------ W2, W12: all-categorical


def _categorical_declarations() -> dict[str, object]:
    return {
        "natural": joint("natural"),
        "joint dose up": JointPolicy(
            "joint dose up",
            components={"D": ModifiedPolicy("d", apply=_label_up, policy_kind="known")},
            policy_kind="known",
        ),
        "tuple dose up": ModifiedPolicy("tuple dose up", apply=tuple_dose_up, policy_kind="known"),
        "flag gated": JointPolicy(
            "flag gated",
            components={
                "F": ModifiedPolicy("f", apply=set_flag, policy_kind="known"),
                "D": ModifiedPolicy("d", apply=_label_up_if_flagged, policy_kind="known"),
            },
            policy_kind="known",
        ),
    }


def _label_up(a: Any, h: Any) -> np.ndarray:
    return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)


def _label_up_if_flagged(a: Any, h: Any) -> np.ndarray:
    dose = np.asarray(a, dtype=float)
    return np.where(np.asarray(h["F"], dtype=float) == 1.0, np.minimum(dose + 1.0, 2.0), dose)


#: The oracle label of each all-categorical declaration.
CATEGORICAL_TRUTH = {
    "natural": "natural",
    "joint dose up": "dose up",
    "tuple dose up": "dose up",
    "flag gated": "flag gated",
}


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_w2_policies_on_an_all_categorical_vector_are_exact(ratio: str) -> None:
    """A joint policy and a tuple map both reach the discrete formula, on either route."""
    declared = _categorical_declarations()
    learner = CellProbabilities() if ratio == "density" else CellMeans()
    result = fit(
        tuple(declared),
        data=categorical_data(),
        policies=list(declared.values()),
        ratio=ratio,
        treatment_learner=learner,
        g_bounds=(1e-6, 1.0 - 1e-6),
    )
    assert deviation(result, tuple(declared), tuple(CATEGORICAL_TRUTH.values())) < EXACT


def test_w2_a_static_tuple_level_is_the_static_intervention() -> None:
    result = _categorical_default_fit(categorical_data())
    rows = law.mixed_first_row_of()
    estimate = result.estimates["ey[(1, 2.0)]"]
    assert estimate.psi == pytest.approx(law.mixed_truth("static (1, 2)"), abs=EXACT)
    np.testing.assert_allclose(
        np.asarray(estimate.influence_curve)[rows], law.mixed_eif("static (1, 2)"), atol=EXACT
    )


def _categorical_default_fit(data: CausalData, **overrides: Any) -> Any:
    configuration = {
        "outcome_learner": CellMeans(),
        "treatment_learner": CellProbabilities(),
        "cross_fit": False,
        "g_bounds": (1e-6, 1.0 - 1e-6),
        "simultaneous": False,
        **overrides,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(**configuration).fit(data).single()


def test_w12_an_all_categorical_vector_equals_a_precombined_column() -> None:
    """Every default estimand equals the fit on one column coded ``3 f + d``."""
    frame = law.mixed_frame().assign(A=lambda f: 3 * f["F"] + f["D"])
    vector = _categorical_default_fit(categorical_data(frame))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_data = CausalData.from_frame(frame, outcome="Y", treatment="A", covariates=["W"])
    combined = _categorical_default_fit(combined_data)
    assert len(vector.estimates) == len(combined.estimates) == 21
    for (left, first), (right, second) in zip(
        vector.estimates.items(), combined.estimates.items(), strict=True
    ):
        assert left.split("[")[0] == right.split("[")[0]
        assert first.psi == pytest.approx(second.psi, abs=EXACT)
        np.testing.assert_allclose(first.influence_curve, second.influence_curve, atol=EXACT)


def test_tuple_level_names_parse_in_every_accessor() -> None:
    """A tuple label is a name: psi, ratio, contrast and a tuple ``reference=`` read it."""
    result = _categorical_default_fit(categorical_data())
    low, high = "ey[(0, 0.0)]", "ey[(1, 2.0)]"
    assert result.psi(high) - result.psi(low) == pytest.approx(
        result.psi("ate[(1, 2.0) vs (0, 0.0)]"), abs=EXACT
    )
    assert result.ratio(high, low).psi == pytest.approx(
        result.psi("rr[(1, 2.0) vs (0, 0.0)]"), abs=EXACT
    )
    difference = result.contrast(lambda v: v[0] - v[1], [high, low])
    assert difference.psi == pytest.approx(result.psi("ate[(1, 2.0) vs (0, 0.0)]"), abs=EXACT)
    referenced = _categorical_default_fit(categorical_data(), reference=(1, 2.0))
    assert referenced.psi("ate[(0, 0.0) vs (1, 2.0)]") == pytest.approx(
        result.psi(low) - result.psi(high), abs=EXACT
    )


@pytest.mark.parametrize("reading", ["categorical", "mixed"])
def test_a_vector_result_round_trips_through_dumps(reading: str) -> None:
    if reading == "categorical":
        result = _categorical_default_fit(categorical_data())
    else:
        result = fit(("natural", "flag gated"))
    restored = loads(dumps(result))
    assert restored.data.component_names == ("F", "D")
    assert restored.data.treatment_kind == result.data.treatment_kind
    for name, estimate in result.estimates.items():
        assert restored.psi(name) == estimate.psi
    if reading == "mixed":
        np.testing.assert_array_equal(
            restored.data.treatment_components, result.data.treatment_components
        )
        with pytest.raises(CapabilityError, match=ONE_COLUMN):
            restored.data.arm_codes  # noqa: B018


# ------------------------------------------------------------------ W8


def test_w8_targeting_on_the_exact_law_with_a_misspecified_outcome_reaches_the_truth() -> None:
    """Double robustness at the exact ratio: a misspecified ``Qbar`` and a moving fluctuation.

    The ratio is exact, so the targeted estimate equals the truth only when the clever
    covariate reads the right components at the observed and at the assigned values.  The
    mutation controls of the classifier label block and of the counterfactual design use it.
    """
    result = fit(**misspecified_settings())
    for label in LABELS:
        assert result.psi(f"ey_policy[{label}]") == pytest.approx(law.mixed_truth(label), abs=EXACT)
    epsilon = np.asarray(result.fluctuations["mtp"].epsilon, dtype=float)
    assert np.max(np.abs(epsilon)) > 1e-3
    assert result.fluctuations["mtp"].converged


@pytest.mark.parametrize("folds", [1, 3])
def test_w8_the_targeted_influence_curve_has_mean_zero_on_a_sample(folds: int) -> None:
    from sklearn.linear_model import LogisticRegression

    data = vector_data(law.mixed_sample(900, 11))
    result = fit(
        ("natural", "dose up", "flag gated", "random"),
        data=data,
        **misspecified_settings(
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=folds > 1,
            n_folds=max(folds, 2),
            random_state=5,
        ),
    )
    epsilon = np.asarray(result.fluctuations["mtp"].epsilon, dtype=float)
    assert np.max(np.abs(epsilon)) > 1e-3
    for estimate in result.estimates.values():
        assert abs(float(np.mean(estimate.influence_curve))) < 1e-10


# ------------------------------------------------------------------ H1: every reader


def _arguments(
    result: Any, where: str, tmp_path: Path
) -> dict[str, tuple[tuple[Any, ...], dict[str, Any]]]:
    names = list(result.estimates)
    return {
        "capability": (("support" if where == "diagnostics" else "evalue",), {}),
        "psi": ((names[0],), {}),
        "contrast": ((lambda v: v[1] - v[0], names[:2]), {}),
        "ratio": ((names[1], names[0]), {}),
        "save": ((tmp_path / "result.joblib",), {}),
    }


def _outcomes(subject: Any, arguments: dict[str, Any]) -> dict[str, type[BaseException] | None]:
    """Call every public member of ``subject``: ``None`` when it returns, else the error."""
    outcomes: dict[str, type[BaseException] | None] = {}
    owner = type(subject)
    for name in sorted(dir(owner)):
        if name.startswith("_"):
            continue
        static = inspect.getattr_static(owner, name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = getattr(subject, name)
                if (
                    callable(value)
                    and not isinstance(static, property)
                    and not inspect.isclass(value)
                ):
                    args, kwargs = arguments.get(name, ((), {}))
                    value = value(*args, **kwargs)
                    if callable(getattr(value, "summary", None)):
                        value.summary()
        except Exception as error:
            outcomes[name] = type(error)
        else:
            outcomes[name] = None
    return outcomes


def _sample_fits() -> tuple[Any, Any]:
    """A vector fit and a one-column dose fit of the same route, on the same sample."""
    from sklearn.linear_model import LogisticRegression

    frame = law.mixed_sample(500, 7)
    learners = {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": LogisticRegression(),
    }
    vector = fit(
        ("natural", "dose up", "random"), data=vector_data(frame), n_bootstrap=5, **learners
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dose = CausalData.from_frame(
            frame, outcome="Y", treatment="D", covariates=["W", "F"], treatment_kind="continuous"
        )
    scalar = fit(data=dose, policies=[Shift(0.0, cap=None), UP], n_bootstrap=5, **learners)
    return vector, scalar


def test_h1_every_reader_of_a_vector_result_returns_or_refuses(tmp_path: Path) -> None:
    """Each public member returns a value or raises ``CapabilityError``.

    A member that raises another error must raise the same one on a one-column dose fit of
    the same route: such an error belongs to the parameter axis, not to the vector.
    """
    vector, scalar = _sample_fits()
    subjects = {
        "result": (vector, scalar),
        "diagnostics": (vector.diagnostics, scalar.diagnostics),
        "sensitivity": (vector.sensitivity, scalar.sensitivity),
    }
    other: dict[str, type[BaseException]] = {}
    for where, (subject, reference) in subjects.items():
        found = _outcomes(subject, _arguments(vector, where, tmp_path))
        expected = _outcomes(reference, _arguments(scalar, where, tmp_path))
        for name, error in found.items():
            if error is None or issubclass(error, CapabilityError):
                continue
            assert expected.get(name) is error, f"{where}.{name} raised {error.__name__}"
            other[f"{where}.{name}"] = error
    assert set(other) <= {
        "result.ate",
        "result.estimate",
        "result.coefficients",
        "result.repeat_spread",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert vector.diagnostics.support() is not None
        assert vector.diagnostics.score_equations() is not None
        assert len(vector.diagnostics.truncation_curve()) > 0
        with pytest.raises(CapabilityError, match="requires a discrete arm contrast"):
            vector.sensitivity.evalue()
    assert vector.bootstrap is not None
    assert vector.bootstrap.n_failed == 0
    for draws in vector.bootstrap.draws.values():
        assert np.all(np.isfinite(draws))


def test_h1_retargeting_cached_nuisances_reproduces_the_fit() -> None:
    result = fit(("natural", "flag gated"), **misspecified_settings())
    estimator = TMLE(policies=[joint("natural"), joint("flag gated")], **misspecified_settings())
    estimates, _ = estimator.retarget(
        result.data, result.nuisance, estimands=result.config.estimands
    )
    for name, estimate in result.estimates.items():
        assert estimates[name].psi == pytest.approx(estimate.psi, abs=1e-12)


# ------------------------------------------------------------------ registry-wide refusal


def _registered_estimators() -> list[type]:
    import cleverly.estimators as estimators

    found = []
    for name in estimators.__all__:
        candidate = getattr(estimators, name)
        if inspect.isclass(candidate) and issubclass(candidate, TMLE):
            found.append(candidate)
    return found


def test_every_estimator_and_estimand_but_the_policy_route_refuses_as_a_dose_does() -> None:
    """Every registered estimator class and every target refuses with a dose's reason.

    A vector with a continuous component has no arms, as a continuous dose has none.  Each
    request off the policy route must raise the error type that the same request raises
    on the one-column dose ``D`` with ``F`` among the covariates, and the vector's message
    must name the vector.  The one exception is an estimand that is not a policy mean or
    contrast beside ``policies=``, which the vector refuses by name.  ``variable_importance``
    and the array function ``tmle()`` take no ``CausalData`` and are outside this sweep.
    """
    data = vector_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dose = CausalData.from_frame(
            law.mixed_frame(),
            outcome="Y",
            treatment="D",
            covariates=["W", "F"],
            treatment_kind="continuous",
        )
    estimators = _registered_estimators()
    assert {item.__name__ for item in estimators} >= {"TMLE", "CTMLE", "DRTMLE"}
    learners = {"outcome_learner": CellMeans(), "treatment_learner": CellMeans()}

    def attempt(build: Any, target: CausalData) -> BaseException:
        with pytest.raises(Exception) as caught, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(target).fit(target)
        return caught.value

    def policies(target: CausalData) -> list[object]:
        return [joint("dose up")] if target is data else [UP]

    builds: list[Any] = []
    for estimator in estimators:
        builds.append(lambda target, estimator=estimator: estimator(**learners, cross_fit=False))
        if estimator is not TMLE:
            builds.append(
                lambda target, estimator=estimator: estimator(
                    **learners, policies=policies(target), ratio="classifier", cross_fit=False
                )
            )
    for target_name in TARGETS:
        builds.append(
            lambda target, name=target_name: TMLE(**learners, estimands=[name], cross_fit=False)
        )
    for build in builds:
        vector_error, dose_error = attempt(build, data), attempt(build, dose)
        assert type(vector_error) is type(dose_error), (vector_error, dose_error)
        assert isinstance(vector_error, (CapabilityError, DataError))
        # The estimator's own reason, word for word, or the dose's reason naming the vector.
        assert str(vector_error) == str(dose_error) or "'F+D'" in str(vector_error)
    for target_name in TARGETS:
        if target_name in ("ey_policy", "ate_policy"):
            continue
        with pytest.raises(CapabilityError, match="A policy fit reports"):
            fit(("dose up",), estimands=[target_name])


# ------------------------------------------------------------------ subset


def test_a_reversed_subset_with_repeats_keeps_each_row_with_its_components() -> None:
    frame = law.mixed_frame()
    index = np.concatenate([np.arange(len(frame))[::-1], [0, 0, 5, 200]])
    for build in (vector_data, categorical_data):
        data = build(frame)
        part = data.subset(index)
        assert part.n == index.size
        np.testing.assert_array_equal(part.outcome, data.outcome[index])
        np.testing.assert_array_equal(part.covariates, data.covariates[index])
        np.testing.assert_array_equal(part.component_design(), data.component_design()[index])
        if data.treatment_components is not None:
            np.testing.assert_array_equal(
                part.treatment_components, data.treatment_components[index]
            )
        else:
            np.testing.assert_array_equal(part.treatment, data.treatment[index])
            assert part.treatment_levels == data.treatment_levels
        rebuilt = build(frame.iloc[index].reset_index(drop=True))
        np.testing.assert_array_equal(part.component_design(), rebuilt.component_design())


def test_a_fit_on_a_subset_equals_the_fit_on_the_same_rows() -> None:
    frame = law.mixed_frame()
    index = np.concatenate([np.arange(len(frame))[::-1], np.arange(0, len(frame), 2)])
    by_subset = fit(("natural", "flag gated"), data=vector_data(frame).subset(index))
    by_rows = fit(
        ("natural", "flag gated"), data=vector_data(frame.iloc[index].reset_index(drop=True))
    )
    for name, estimate in by_rows.estimates.items():
        assert by_subset.psi(name) == pytest.approx(estimate.psi, abs=EXACT)


# ------------------------------------------------------------------ compositions


def test_integer_weights_equal_the_expanded_sample() -> None:
    frame = law.mixed_sample(600, 3)
    weights = 1.0 + (np.arange(len(frame)) % 3 == 0)
    expanded = pd.concat([frame, frame[weights == 2.0]], ignore_index=True)
    labels = ("natural", "flag gated", "random")
    by_weight = fit(labels, data=vector_data(frame.assign(w=weights), weights="w"))
    by_copy = fit(labels, data=vector_data(expanded))
    for label in labels:
        name = f"ey_policy[{label}]"
        assert by_weight.psi(name) == pytest.approx(by_copy.psi(name), abs=1e-10)


def _recorded_groups(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    from cleverly.learners import density_ratio

    seen: list[np.ndarray] = []
    original = density_ratio.fit_learner

    def record(learner: Any, design: Any, target: Any, weight: Any, **kwargs: Any) -> Any:
        seen.append(np.asarray(kwargs["groups"]))
        return original(learner, design, target, weight, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", record)
    return seen


@pytest.mark.parametrize("folds", [1, 3])
def test_a_clustered_fit_keeps_whole_clusters(folds: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stacked classifier groups a cluster's rows and their copies."""
    from sklearn.linear_model import LogisticRegression

    seen = _recorded_groups(monkeypatch)
    frame = law.mixed_sample(900, 4).assign(cluster=lambda f: np.arange(len(f)) // 3)
    result = fit(
        ("natural", "flag gated"),
        data=vector_data(frame, id="cluster"),
        treatment_learner=LogisticRegression(),
        cross_fit=folds > 1,
        n_folds=max(folds, 2),
        random_state=1,
    )
    assert np.isfinite(result.psi("ate_policy[flag gated vs natural]"))
    clusters = result.data.cluster
    assert seen
    for groups in seen:
        assert set(np.unique(groups)) <= set(np.unique(clusters))
        if folds > 1:
            held = set(np.unique(clusters)) - set(np.unique(groups))
            assert held
            # A held-out cluster is held out whole: none of its rows trains.
            assert not np.isin(clusters[np.isin(clusters, list(held))], groups).any()


def test_cross_fitting_splits_units_not_stacked_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A unit's two copies train together, and the held-out units partition the sample."""
    from sklearn.linear_model import LogisticRegression

    seen = _recorded_groups(monkeypatch)
    result = fit(
        ("natural", "flag gated"),
        data=vector_data(law.mixed_sample(600, 2)),
        treatment_learner=LogisticRegression(),
        cross_fit=True,
        n_folds=3,
        random_state=3,
    )
    n = result.data.n
    # Two plans and three folds: one classifier per plan and fold.
    assert len(seen) == 2 * 3
    for plan in range(2):
        held_out = []
        for groups in seen[3 * plan : 3 * plan + 3]:
            units, counts = np.unique(groups, return_counts=True)
            assert np.all(counts == 2)
            held_out.append(np.setdiff1d(np.arange(n), units))
        merged = np.concatenate(held_out)
        assert merged.size == n
        np.testing.assert_array_equal(np.sort(merged), np.arange(n))


# ------------------------------------------------------------------ refusals


def test_a_known_mechanism_of_a_dose_vector_is_refused() -> None:
    frame = law.mixed_frame().assign(p=0.5)
    with pytest.raises(CapabilityError) as caught, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CausalData.from_frame(
            frame, continuous_treatment=["D"], treatment_probabilities=["p", "p"], **COLUMNS
        )
    assert str(caught.value) == (
        "treatment 'F+D' is a vector with a continuous component. treatment_probabilities= "
        "gives arm probabilities, and a continuous treatment has a conditional density, not "
        "arms. A known density is not supported; fit without treatment_probabilities=."
    )


def test_a_missing_treatment_of_a_dose_vector_is_refused_as_a_dose_is() -> None:
    frame = law.mixed_frame().assign(observed=1.0)
    with pytest.raises(DataError) as caught:
        CausalData.from_frame(
            frame, continuous_treatment=["D"], treatment_delta="observed", **COLUMNS
        )
    assert str(caught.value) == (
        "a missing treatment is supported for an arm-coded treatment only; F+D is a vector "
        "with a continuous component, and a conditional density fitted on the recorded rows "
        "has no composite indicator to divide by. Drop observed= or the rows with an "
        "unrecorded treatment."
    )


def test_a_policy_on_the_whole_vector_is_refused() -> None:
    with pytest.raises(CapabilityError) as caught:
        fit(policies=[UP])
    assert "is declared on the vector treatment 'F+D' as a whole" in str(caught.value)
    assert "declare JointPolicy(name, components={column: policy, ...}" in str(caught.value)


def test_an_estimand_beside_the_policy_means_is_refused() -> None:
    with pytest.raises(CapabilityError) as caught:
        fit(("natural", "dose up"), estimands=["ey_policy", "ate"])
    assert str(caught.value) == (
        "treatment 'F+D' is a vector with a continuous component, and estimands= asks for "
        "['ate']. A policy fit reports ['ate_policy', 'ey_policy'], the policy means and "
        "their contrasts; every other registered estimand names an arm, and this treatment "
        "has none"
    )


@pytest.mark.parametrize(
    ("keyword", "message"),
    [
        ({"treatment": ["F"]}, "A vector treatment names two or more columns"),
        ({"treatment": ["F", "F"]}, "repeats a column"),
        ({"treatment": "D"}, "is one column"),
    ],
)
def test_a_malformed_vector_declaration_is_a_data_error(
    keyword: dict[str, Any], message: str
) -> None:
    frame = law.mixed_frame()
    with pytest.raises(DataError, match=message):
        CausalData.from_frame(
            frame, outcome="Y", covariates=["W"], continuous_treatment=["D"], **keyword
        )


def test_w12_a_missing_tuple_equals_a_missing_precombined_column() -> None:
    """An all-categorical vector declared missing at random is one categorical treatment.

    Every seventh row records neither component.  The fit equals the fit on the column
    ``3 f + d`` with the same rows missing and the same ``treatment_delta=``.
    """
    frame = law.mixed_frame().assign(A=lambda f: 3 * f["F"] + f["D"])
    unrecorded = np.arange(len(frame)) % 7 == 0
    frame.loc[unrecorded, ["F", "D", "A"]] = np.nan
    frame = frame.assign(recorded=(~unrecorded).astype(float))
    vector_data_ = categorical_data(frame, treatment_delta="recorded")
    assert vector_data_.treatment_kind == "discrete"
    assert np.isnan(vector_data_.treatment[unrecorded]).all()
    vector = _categorical_default_fit(vector_data_)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        combined_data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=["W"], treatment_delta="recorded"
        )
    combined = _categorical_default_fit(combined_data)
    assert len(vector.estimates) == len(combined.estimates) > 0
    for (left, first), (right, second) in zip(
        vector.estimates.items(), combined.estimates.items(), strict=True
    ):
        assert left.split("[")[0] == right.split("[")[0]
        assert first.psi == pytest.approx(second.psi, abs=EXACT)
        np.testing.assert_allclose(first.influence_curve, second.influence_curve, atol=EXACT)


def test_a_recorded_row_with_a_missing_component_is_a_data_error() -> None:
    frame = law.mixed_frame().assign(recorded=1.0)
    frame.loc[0, "D"] = np.nan
    with pytest.raises(
        DataError, match=r"'D' has 1 missing value\(s\) on rows flagged as recorded"
    ):
        categorical_data(frame, treatment_delta="recorded")


# ------------------------------------------------------------------ fingerprint (review MEDIUM-1)


def test_the_data_fingerprint_reads_every_vector_component() -> None:
    """Two vector datasets that differ in one component fingerprint differently.

    On (a2) the treatment slot holds ``NaN`` alone, so a digest of it could not tell two
    samples apart; the ``SplitPlan`` source binding and the provenance read this digest.
    """
    from cleverly.provenance import data_fingerprint, fingerprint_array

    frame = law.mixed_sample(200, 4)
    dose = frame.copy()
    dose.loc[dose.index[0], "D"] = 7.0
    flag = frame.copy()
    flag.loc[flag.index[0], "F"] = 1.0 - flag.loc[flag.index[0], "F"]
    base = vector_data(frame)
    digests = {
        "base": data_fingerprint(base),
        "dose": data_fingerprint(vector_data(dose)),
        "flag": data_fingerprint(vector_data(flag)),
        "all categorical": data_fingerprint(categorical_data(frame)),
    }
    assert len(set(digests.values())) == len(digests), digests
    # A copy that takes the dose sample's components is that sample.
    moved = base.with_treatment(np.asarray(vector_data(dose).treatment_components))
    assert data_fingerprint(moved) == digests["dose"]
    renamed = vector_data(frame.rename(columns={"F": "E"}), treatment=["E", "D"])
    assert data_fingerprint(renamed) != digests["base"]
    scalar = CausalData.from_frame(frame, outcome="Y", treatment="F", covariates=["W", "D"])
    assert data_fingerprint(scalar) == fingerprint_array(
        scalar.outcome, scalar.treatment, scalar.covariates
    )
