"""The kernel-smoothed dose-response curve through ``CausalStudy`` (X123 part (b)).

``study.identify(DoseResponseCurve(curve)).estimate(...)`` fits
:class:`~cleverly.estimators.DoseResponseTMLE` on the study's data.  The route adds no
estimand and no estimation step, so the registered dose-response study covers it exactly where
it covers the direct fit.  These tests check that carry-over: the route equals the direct fit
bit for bit, every forwarded method field reaches the engine (each with a nonzero witness), and
every field the curve does not read is refused by name before any learner.  They also check the
two declarations a dose-response fit used to ignore (a treatment-mechanism subset and a
declared conditional dose support), the identification record, the design and method
refusals, the protocol stamp, the planned-fit preview and the round trip.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly
from cleverly import (
    CausalStudy,
    CounterfactualMean,
    CrossFitting,
    DoseResponseCurve,
    Inference,
    LongitudinalTreatment,
    LongPanel,
    ModelSpec,
    MSMProjection,
    PointTreatment,
    Runtime,
    Targeting,
    TimeToEvent,
    TMLEMethod,
    TrialWithExternalData,
)
from cleverly.data import CausalData
from cleverly.dose_response import DoseResponse
from cleverly.estimators import DoseResponseTMLE
from cleverly.exceptions import (
    CapabilityError,
    DataError,
    MethodConfigurationError,
    RefusalReason,
)
from cleverly.interventions import Support
from cleverly.learners.crossfit import SplitPlan, random_partition
from cleverly.methods import (
    DOSE_RESPONSE_FIELDS,
    DOSE_RESPONSE_FOLD_FIELDS,
    DOSE_RESPONSE_GAP_FIELDS,
)
from cleverly.msm import MSM
from cleverly.study import _dose_response_design_refusal
from tests.unit import _planned_fits_support as counting
from tests.unit._declaration_support import assert_refused_before_any_call
from tests.unit._natural_course_support import NeverFit

pytestmark = pytest.mark.xdist_group("dose_response_study")

COLUMNS = {"outcome": "Y", "treatment": "A", "covariates": ["W1", "W2"]}
DESIGN = {"outcome": "Y", "treatment": "A", "adjustment": ("W1", "W2")}


def _frame(n: int = 400, seed: int = 3, *, gaussian: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    a = rng.normal(2.5 + 0.6 * w1, 1.2)
    if gaussian:
        y = np.clip(0.2 + 0.4 * a - 0.05 * a * a + 0.6 * w1 + rng.normal(0.0, 0.8, n), -2.9, 4.9)
    else:
        y = rng.binomial(1, expit(-1.0 + 0.5 * a - 0.08 * a * a + 0.4 * w1 + 0.3 * w2))
    weights = rng.uniform(0.5, 1.5, size=n)
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2, "wt": weights})


def _curve(floor: float = 0.0, doses: tuple[float, ...] = (1.5, 2.5, 3.5)) -> DoseResponse:
    return DoseResponse(doses=doses, bandwidths=(0.75,), support=(0.5, 4.5), support_floor=floor)


def _learners(gaussian: bool = False) -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression() if gaussian else LogisticRegression(max_iter=1000),
        "treatment_learner": LogisticRegression(max_iter=1000),
    }


#: Method-field keywords of ``estimate`` -> the engine keyword each one becomes.
_ENGINE_NAMES = {"n_multiplier": "n_replicates"}
#: The fold and bin settings both routes share, so the method's 10-fold default is not compared.
_BASE = {"n_folds": 2, "density_bins": 8, "random_state": 0}


def _study(frame: pd.DataFrame, **design: Any) -> CausalStudy:
    fields = {**DESIGN, "treatment_kind": "continuous", **design}
    return CausalStudy(frame, design=PointTreatment(**fields))


def _route(
    frame: pd.DataFrame, curve: DoseResponse, design: dict[str, Any] | None = None, **settings: Any
) -> Any:
    gaussian = frame["Y"].nunique() > 2
    effect = _study(frame, **(design or {})).identify(DoseResponseCurve(curve))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(**{**_learners(gaussian), **_BASE, **settings})


def _direct(
    frame: pd.DataFrame, curve: DoseResponse, columns: dict[str, Any] | None = None, **settings: Any
) -> Any:
    gaussian = frame["Y"].nunique() > 2
    options = {**_learners(gaussian), **_BASE, **settings}
    options = {_ENGINE_NAMES.get(name, name): value for name, value in options.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DoseResponseTMLE(curve, **options).fit(frame, **{**COLUMNS, **(columns or {})})


def _facts(result: Any) -> tuple[Any, ...]:
    """Every number and label a fit reports, for an exact comparison."""
    rows = []
    for name, estimate in result.estimates.items():
        rows.append(
            (
                name,
                estimate.psi,
                estimate.variance,
                tuple(estimate.influence_curve),
                estimate.ci,
                estimate.support.label,
                estimate.support.reason,
            )
        )
    band = result.band
    return (
        tuple(rows),
        band.critical_value,
        band.kind,
        band.reference,
        band.n_replicates,
        band.support.label if band.support is not None else None,
        band.support.reason if band.support is not None else None,
        result.pair_support,
        result.nuisance_record.cross_fitted,
        int(result.nuisance_record.folds.max()) + 1,
        result.provenance.data_fingerprint,
        result.provenance.fold_fingerprint,
    )


def _labels(result: Any) -> set[str]:
    return {estimate.support.label for estimate in result.estimates.values()}


# ------------------------------------------------------------------ route equivalence


EQUIVALENT: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any], bool]] = {
    "binary": ({}, {}, {}, False),
    "binary strata": ({"strata": ("W2",)}, {"strata": ["W2"]}, {}, False),
    "continuous linear": ({}, {}, {"fluctuation": "linear", "q_bounds": (-3.0, 5.0)}, True),
    "in sample": ({}, {}, {"cross_fit": False}, False),
    "weighted": ({"weights": "wt"}, {"weights": "wt"}, {}, False),
}


@pytest.mark.parametrize("name", sorted(EQUIVALENT))
def test_the_route_equals_the_direct_fit(name: str) -> None:
    design, columns, settings, gaussian = EQUIVALENT[name]
    frame = _frame(gaussian=gaussian)
    routed = _route(frame, _curve(), design, **settings)
    direct = _direct(frame, _curve(), columns, **settings)
    assert _facts(routed) == _facts(direct)


# ------------------------------------------------------------------ forwarded fields


def test_the_measured_configuration_reads_supported() -> None:
    """The baseline of the label witnesses below: a floor of zero, every label supported."""
    result = _route(_frame(), _curve())
    assert _labels(result) == {"supported"}
    assert result.band.support.label == "supported"


#: field -> (value, data kind); each moves something the default fit reports.
FORWARDED: dict[str, tuple[Any, bool]] = {
    "alpha": (0.1, False),
    "n_multiplier": (2_000, False),
    "multiplier_kind": ("rademacher", False),
    "band_reference": ("t", False),
    "n_folds": (3, False),
    "density_bins": (6, False),
    "random_state": (1, False),
    "cross_fit": (False, False),
    "fluctuation": ("linear", True),
    "q_bounds": ((-4.0, 6.0), True),
}


@pytest.mark.parametrize("field", sorted(FORWARDED))
def test_each_read_field_reaches_the_engine(field: str) -> None:
    value, gaussian = FORWARDED[field]
    frame = _frame(gaussian=gaussian)
    base = {"q_bounds": (-3.0, 5.0)} if gaussian else {}
    changed = {**base, field: value}
    routed = _route(frame, _curve(), **changed)
    assert _facts(routed) == _facts(_direct(frame, _curve(), **changed))
    # The nonzero witness: the default would have reported something else.
    assert _facts(routed) != _facts(_route(frame, _curve(), **base))


@pytest.mark.parametrize(
    ("field", "value"),
    [("n_multiplier", 2_000), ("multiplier_kind", "rademacher"), ("band_reference", "t")],
)
def test_a_band_the_study_did_not_measure_reads_experimental(field: str, value: Any) -> None:
    result = _route(_frame(), _curve(), **{field: value})
    assert result.band.support.label == "experimental"
    assert _labels(result) == {"supported"}
    # The band itself moves, not only its label: a label read from the settings would flip
    # even if the settings never reached the multiplier draw.
    band, default = result.band, _route(_frame(), _curve()).band
    attribute = {"n_multiplier": "n_replicates", "multiplier_kind": "kind"}.get(field, "reference")
    assert getattr(band, attribute) == value
    assert getattr(default, attribute) != value
    if field == "band_reference":
        # The t reference keeps the max-t quantile and widens each row on its own t margin.
        assert band.critical_value == default.critical_value
        assert band.critical_values != default.critical_values
        assert band.bands != default.bands
    else:
        assert band.critical_value != default.critical_value


def test_the_run_label_reaches_the_provenance() -> None:
    assert _route(_frame(), _curve(), run_id="trial-run").provenance.run_id == "trial-run"
    assert _route(_frame(), _curve()).provenance.run_id is None


def test_learner_folds_reach_the_super_learner() -> None:
    """The default library's candidate fits move with ``learner_folds``, counted unfitted."""
    effect = _study(_frame()).identify(DoseResponseCurve(_curve()))

    def candidates(folds: int) -> int:
        plan = effect.planned_fits(
            outcome_learner=None,
            treatment_learner=LogisticRegression(max_iter=1000),
            learner_folds=folds,
            **_BASE,
        )
        return sum(row.candidate_fits for row in plan.rows if row.role == "outcome_learner")

    assert candidates(3) != candidates(5)


# ------------------------------------------------------------------ labels


def test_a_pair_below_the_floor_reads_experimental_through_the_route() -> None:
    result = _route(_frame(), _curve(floor=1e9))
    assert _labels(result) == {"experimental"}
    assert all(row.below_floor for row in result.pair_support)
    direct = _direct(_frame(), _curve(floor=1e9))
    assert _facts(result) == _facts(direct)


# ------------------------------------------------------------------ restricted mechanism


@pytest.mark.parametrize(
    ("subset", "label"),
    [
        (None, "supported"),
        (("W1", "W2"), "supported"),
        (("W1",), "experimental"),
        ((), "experimental"),
    ],
    ids=["undeclared", "every column", "proper subset", "empty"],
)
def test_a_treatment_mechanism_subset_reads_experimental(subset: Any, label: str) -> None:
    design = {} if subset is None else {"treatment_adjustment": subset}
    result = _route(_frame(), _curve(), design)
    assert _labels(result) == {label}
    assert result.band.support.label == label
    if label == "experimental":
        assert "treatment_adjustment=" in next(iter(result.estimates.values())).support.reason
        restrictions = result.identified_effect.identification.model_restrictions
        assert len(restrictions) == 1
    else:
        assert result.identified_effect.identification.model_restrictions == ()


def test_the_direct_fit_reads_the_subset_too() -> None:
    frame = _frame()
    data = _study(frame, treatment_adjustment=("W1",)).data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = DoseResponseTMLE(_curve(), **_learners(), n_folds=2, density_bins=8).fit(data)
    assert _labels(direct) == {"experimental"}


# ------------------------------------------------------------------ declared dose support


def _support_frame(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """A dose uniform on ``[W1, W1 + 3]``, with ``W1`` uniform on ``[0, 1]``."""
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    a = w1 + 3.0 * rng.uniform(size=n)
    y = rng.binomial(1, expit(-1.0 + 0.4 * a + 0.3 * w1))
    return pd.DataFrame({"Y": y, "A": a, "W1": w1, "W2": w2})


#: The support the law obeys.  The kernel of dose 1 (``[0.6, 1.4]``) leaves it on rows with
#: ``W1 > 0.6``, and the kernel of dose 3 (``[2.6, 3.4]``) on rows with ``W1 < 0.4``.
_CUTTING = Support(lambda h: np.asarray(h["W1"]), lambda h: np.asarray(h["W1"]) + 3.0)


def _support_curve(doses: tuple[float, ...]) -> DoseResponse:
    return DoseResponse(doses=doses, bandwidths=(0.4,), support=(0.5, 3.5), support_floor=0.0)


def test_a_covering_support_reaches_the_engine_as_declared() -> None:
    """A support that covers every kernel is accepted and reaches the density as declared.

    The density route reads the declared support, so the study fit equals the engine fit on
    the same declared container, and differs from the fit without the declaration.
    """
    frame = _support_frame()
    curve = _support_curve((2.0,))
    covered = _route(frame, curve, {"treatment_support": _CUTTING})
    declared = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], treatment_kind="continuous"
    ).with_dose_declarations(_CUTTING)
    options = {
        _ENGINE_NAMES.get(name, name): value
        for name, value in {**_learners(False), **_BASE}.items()
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = DoseResponseTMLE(curve, **options).fit(declared)
    assert _facts(covered)[:-2] == _facts(engine)[:-2]
    assert _facts(covered)[:-2] != _facts(_route(frame, curve))[:-2]


@pytest.mark.parametrize(
    ("doses", "names"),
    [
        ((2.0, 3.0), ["dose_response[h=0.4,a=3]"]),
        ((1.0, 2.0, 3.0), ["dose_response[h=0.4,a=1]", "dose_response[h=0.4,a=3]"]),
    ],
    ids=["one pair", "two pairs"],
)
def test_a_support_that_cuts_a_kernel_is_refused_before_any_learner(
    doses: tuple[float, ...], names: list[str]
) -> None:
    frame = _support_frame()
    w1 = frame["W1"].to_numpy()
    cut = (w1 < 0.4) | ((w1 > 0.6) if 1.0 in doses else False)
    study = _study(frame, treatment_support=_CUTTING)
    raised = assert_refused_before_any_call(
        lambda: study.identify(DoseResponseCurve(_support_curve(doses))),
        None,
        "",
        f"{int(cut.sum())} rows have a declared dose support that excludes doses inside the "
        f"kernel of pair(s) {', '.join(names)}",
        "positivity fails by declaration",
    )
    assert raised.refusal.reason == RefusalReason.INVALID_DECLARATION
    # The direct engine reads the declaration on the study's data the same way.
    with pytest.raises(CapabilityError, match="excludes doses inside the kernel"):
        DoseResponseTMLE(
            _support_curve(doses), outcome_learner=NeverFit(), treatment_learner=NeverFit()
        ).fit(study.data)


def test_a_kernel_is_checked_only_inside_the_dose_range() -> None:
    """The kernel of dose 0.6 at bandwidth 0.4 reaches 0.2, below the range's 0.5.

    Rows with ``W1 > 0.2`` cannot receive doses in ``[0.2, 0.5)``, but the smoothed curve never
    reads them, so the support covers the pair's kernel inside the range and the curve is
    identified.  Checking the untruncated kernel would refuse it.
    """
    frame = _support_frame()
    support = Support(lambda h: np.minimum(np.asarray(h["W1"]), 0.5), lambda h: h["W1"] + 3.0)
    curve = _support_curve((0.6, 2.0))
    assert (frame["W1"] > 0.2).any()
    effect = _study(frame, treatment_support=support).identify(DoseResponseCurve(curve))
    assert effect.functional.interventions == curve.pairs


def test_an_observed_dose_outside_its_support_is_refused_first() -> None:
    frame = _support_frame()
    narrow = Support(lambda h: np.asarray(h["W1"]) + 0.2, lambda h: np.asarray(h["W1"]) + 3.0)
    study = _study(frame, treatment_support=narrow)
    assert_refused_before_any_call(
        lambda: study.identify(DoseResponseCurve(_support_curve((2.0,)))),
        None,
        "",
        "the declared support of node 'A' excludes",
        error=DataError,
    )


def test_a_continuous_msm_with_a_declared_support_is_refused_before_any_learner() -> None:
    """Review finding 7: an MSM never reads a declared support, and it never ignores one.

    The point TMLE refuses ``treatment_support=`` without ``policies=`` before any learner,
    covering support or not, so a continuous MSM cannot average the outcome regression over
    doses a declaration excludes.
    """
    effect = _study(_support_frame(), treatment_support=Support(-1.0, 5.0)).identify(
        MSMProjection(MSM.linear(doses=[0.5, 2.0, 3.5]))
    )
    assert_refused_before_any_call(
        lambda: effect.estimate(
            outcome_learner=NeverFit(), treatment_learner=NeverFit(), n_folds=2
        ),
        None,
        "",
        "treatment_support= and dose_atoms= declare the dose a modified treatment policy moves",
        error=DataError,
    )


# ------------------------------------------------------------------ randomized design


def test_a_randomized_dose_fits_through_the_study() -> None:
    frame = _frame()
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=(), randomized=True, treatment_kind="continuous"
        ),
    )
    effect = study.identify(DoseResponseCurve(_curve()))
    text = str(effect.summary())
    assert "overlap" in text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = effect.estimate(**_learners(), **_BASE)
    assert set(result.estimates) == set(_curve().names)
    assert _labels(result) <= {"supported", "experimental"}
    direct = DoseResponseTMLE(_curve(), **_learners(), **_BASE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _facts(result) == _facts(direct.fit(study.data))


# ------------------------------------------------------------------ identification


def test_the_identified_functional_and_record() -> None:
    curve = _curve()
    effect = _study(_frame()).identify(DoseResponseCurve(curve))
    functional = effect.functional
    assert (functional.target, functional.axis) == ("dose_response", "dose_response")
    assert functional.interventions == curve.pairs
    assert "for each declared pair (h, a0) in (0.75, 1.5), (0.75, 2.5), (0.75, 3.5)" in (
        functional.expression
    )
    statements = effect.identification.assumptions
    assert statements[0].startswith("consistency: Y = Y^a when A = a, for every dose a")
    assert statements[3].startswith("positivity over the dose interval:")
    assert len(statements) == 6
    assert effect.identification.required_nuisances == ("outcome_regression", "treatment_mechanism")
    text = " ".join(str(effect.summary()).split())
    assert "every dose near each declared dose inside the declared range" in text
    stratified = _study(_frame(), strata=("W2",)).identify(DoseResponseCurve(curve))
    assert stratified.identification.assumptions[-1].startswith("each stratum's curve")
    assert len(stratified.identification.assumptions) == 7


def test_the_estimand_refuses_a_bare_declaration() -> None:
    with pytest.raises(DataError, match=r"takes a cleverly\.dose_response\.DoseResponse"):
        DoseResponseCurve((1.0, 2.0))  # type: ignore[arg-type]
    assert DoseResponseCurve(_curve()).name == "dose_response"
    assert cleverly.DoseResponseCurve is DoseResponseCurve


def test_a_fixed_dose_mean_points_at_the_curve() -> None:
    with pytest.raises(CapabilityError) as raised:
        _study(_frame()).identify(CounterfactualMean())
    assert "DoseResponseCurve(DoseResponse(" in str(raised.value)


# ------------------------------------------------------------------ methods


def _effect() -> Any:
    return _study(_frame()).identify(DoseResponseCurve(_curve()))


def test_tmle_alone_fits_the_curve() -> None:
    records = {item.name: item for item in _effect().available_methods()}
    assert [name for name, item in records.items() if item.available] == ["tmle"]
    for name in ("collaborative_tmle", "drtmle"):
        refusal = records[name].refusal
        assert refusal.reason == RefusalReason.IMPLEMENTATION_GAP
        assert refusal.roadmap == "F5"
    assert records["adaptive_iptw"].refusal.reason == RefusalReason.INVALID_DECLARATION
    with pytest.raises(CapabilityError):
        _effect().estimate("drtmle")


def _group_fields() -> list[tuple[str, str]]:
    groups = {
        "models": ModelSpec,
        "cross_fitting": CrossFitting,
        "targeting": Targeting,
        "inference": Inference,
        "runtime": Runtime,
    }
    return [(group, item.name) for group, cls in groups.items() for item in dataclasses.fields(cls)]


#: A non-default value for every field the curve does not read.
_UNREAD_VALUES: dict[str, Any] = {
    "missingness_learner": LogisticRegression(),
    "intermediate_learner": LogisticRegression(),
    "pseudo_learner": LinearRegression(),
    "censoring_learner": LogisticRegression(),
    "competing_learner": LogisticRegression(),
    "ratio": "classifier",
    "screen_treatment": True,
    "min_retain": 2,
    "hazard_pooling": "pooled",
    "outcome_by_arm": True,
    "survival_model": "cox",
    "repeats": 3,
    "stratify_by": "treatment",
    "targeting_scheme": "fold",
    "fold_evaluation": True,
    "split_plan": SplitPlan.from_folds([random_partition(400, 2, seed=0)]),
    "algorithm": "one_step",
    "g_bounds": (0.05, 0.95),
    "nuisance_bound": 0.05,
    "submodel_alpha": 0.99,
    "target_weights": True,
    "step_size": 0.01,
    "max_iter": 5,
    "tol": 1e-6,
    "stopping": "scaled_var",
    "step_bound": 0.5,
    "step_search": False,
    "iterate": "best",
    "conditional_form": "contrast",
    "isotonic": True,
    "ratio_trim": 0.01,
    "node_bounds": (0.01, 0.99),
    "n_bootstrap": 50,
    "bootstrap_resampling": "cluster",
    "band_scale": "identity",
    "bootstrap_interval": "bca",
    "variance": "policy_weighted",
    "cluster_variance": "jackknife",
    "hazard_library": (LogisticRegression(),),
    "censoring_library": (LogisticRegression(),),
    "norm": "euclidean",
    "max_memory_bytes": 2**30,
}


#: Fields that are valid only beside another changed field, which is then refused first.
_COMPANIONS: dict[str, dict[str, Any]] = {
    "step_search": {"step_bound": 0.5},
    "bootstrap_interval": {"n_bootstrap": 50},
}


def test_every_method_field_is_classified() -> None:
    """The read, gap, fold and does-not-apply sets partition every field of the five groups."""
    every = set(_group_fields())
    assert every >= DOSE_RESPONSE_FIELDS
    assert set(DOSE_RESPONSE_GAP_FIELDS) <= every
    assert every >= DOSE_RESPONSE_FOLD_FIELDS
    assert not DOSE_RESPONSE_FIELDS & set(DOSE_RESPONSE_GAP_FIELDS)
    assert not DOSE_RESPONSE_FOLD_FIELDS & (DOSE_RESPONSE_FIELDS | set(DOSE_RESPONSE_GAP_FIELDS))
    unread = every - DOSE_RESPONSE_FIELDS
    assert {name for _, name in unread} == set(_UNREAD_VALUES)


def _code(group: str, name: str) -> RefusalReason:
    if (group, name) in DOSE_RESPONSE_GAP_FIELDS:
        return RefusalReason.IMPLEMENTATION_GAP
    if (group, name) in DOSE_RESPONSE_FOLD_FIELDS:
        return RefusalReason.INFERENCE_GAP
    return RefusalReason.INVALID_DECLARATION


@pytest.mark.parametrize(
    ("group", "name"), sorted(key for key in _group_fields() if key not in DOSE_RESPONSE_FIELDS)
)
def test_an_unread_field_is_refused_before_any_learner(group: str, name: str) -> None:
    effect = _effect()
    changed = {**_COMPANIONS.get(name, {}), name: _UNREAD_VALUES[name]}
    NeverFit.calls = 0
    with pytest.raises((MethodConfigurationError, CapabilityError)) as raised:
        # A stratified split is refused when the group is built, for every estimand.
        settings = {group: dataclasses.replace(getattr(TMLEMethod(), group), **changed)}
        method = dataclasses.replace(TMLEMethod(), **settings)
        effect.estimate(method, outcome_learner=NeverFit(), treatment_learner=NeverFit())
    assert NeverFit.calls == 0
    if name == "ratio":
        # The study's own check names a ratio on a target with no policy first.
        assert raised.value.refusal.reason == RefusalReason.INVALID_DECLARATION
        return
    if name == "stratify_by":
        assert raised.value.refusal.roadmap == "F32"
        return
    assert raised.value.refusal.reason == _code(group, name)
    if (group, name) not in DOSE_RESPONSE_FOLD_FIELDS:
        first = next(item for item in _UNREAD_VALUES if item in changed)
        assert f"{group}.{first}" in str(raised.value)


def test_simultaneous_false_is_refused_and_true_is_accepted() -> None:
    effect = _effect()
    NeverFit.calls = 0
    with pytest.raises(MethodConfigurationError, match="always reports its band"):
        effect.estimate(
            simultaneous=False, outcome_learner=NeverFit(), treatment_learner=NeverFit()
        )
    plan = effect.planned_fits(simultaneous=True, **_learners(), **_BASE)
    assert plan.rows


# ------------------------------------------------------------------ design refusals


def _discrete_study() -> CausalStudy:
    frame = _frame()
    frame["A"] = (frame["A"] > 2.5).astype(int)
    return CausalStudy(frame, design=PointTreatment(**DESIGN))


def _vector_study() -> CausalStudy:
    frame = _frame()
    frame["F"] = (frame["W1"] > 0).astype(int)
    design = PointTreatment(
        outcome="Y", treatment=["F", "A"], adjustment=("W1", "W2"), continuous_treatment=["A"]
    )
    return CausalStudy(frame, design=design)


@pytest.mark.parametrize(
    ("build", "fragment", "reason"),
    [
        (_discrete_study, "is declared discrete", RefusalReason.INVALID_DECLARATION),
        (_vector_study, "is a vector", RefusalReason.INVALID_DECLARATION),
    ],
    ids=["discrete", "vector"],
)
def test_a_design_without_one_continuous_dose_is_refused(
    build: Any, fragment: str, reason: RefusalReason
) -> None:
    study = build()
    raised = assert_refused_before_any_call(
        lambda: study.identify(DoseResponseCurve(_curve())), None, "", fragment
    )
    assert raised.refusal.reason == reason


def test_the_other_designs_are_refused_by_class() -> None:
    trial = _dose_response_design_refusal(object.__new__(TrialWithExternalData), None)
    assert trial is not None and trial.reason == RefusalReason.INVALID_DECLARATION
    for cls in (LongitudinalTreatment, TimeToEvent, LongPanel):
        refusal = _dose_response_design_refusal(object.__new__(cls), None)
        assert refusal is not None
        assert refusal.reason == RefusalReason.IMPLEMENTATION_GAP
        assert refusal.roadmap == "X95"
        assert cls.__name__ in refusal.condition


def test_a_longitudinal_study_refuses_the_curve() -> None:
    from cleverly.datasets import make_longitudinal

    frame, _ = make_longitudinal(n=200, seed=0, censoring=False)
    study = CausalStudy(
        frame,
        design=LongitudinalTreatment(
            outcome="Y", treatment=("A1", "A2"), baseline=("W1", "W2"), time_varying=((), ("L2",))
        ),
    )
    with pytest.raises(CapabilityError, match="LongitudinalTreatment design is not implemented"):
        study.identify(DoseResponseCurve(_curve()))


@pytest.mark.parametrize(
    ("change", "design", "fragment", "error"),
    [
        ("missing outcome", {"missingness": "R"}, "a missing outcome", CapabilityError),
        (
            "missing treatment",
            {"treatment_missingness": "R"},
            "the law of A given W is not identified",
            DataError,
        ),
        ("intermediate", {"intermediate": "Z"}, "an intermediate variable", CapabilityError),
        ("atoms", {"dose_atoms": (0.0,)}, "declared dose atoms", CapabilityError),
        (
            "declared mechanism",
            {"treatment_probabilities": {1.0: "g"}},
            "A known density is not supported",
            CapabilityError,
        ),
        ("cluster", {"cluster": "C"}, "does not yet support clustered rows", CapabilityError),
        ("19 doses", {}, "the dose takes only 19 distinct values", CapabilityError),
        ("range", {}, "observed dose range", DataError),
    ],
)
def test_the_engine_refusals_now_raise_at_identify(
    change: str, design: dict[str, Any], fragment: str, error: type[Exception]
) -> None:
    frame = _frame()
    curve = _curve()
    rows = np.arange(len(frame))
    if change == "missing outcome":
        frame["R"] = (rows % 7 != 0).astype(int)
        frame.loc[frame["R"] == 0, "Y"] = np.nan
    elif change == "missing treatment":
        frame["R"] = (rows % 7 != 0).astype(int)
        frame.loc[frame["R"] == 0, "A"] = np.nan
    elif change == "intermediate":
        frame["Z"] = (frame["W1"] > 0).astype(int)
    elif change == "atoms":
        frame.loc[:39, "A"] = 0.0
    elif change == "declared mechanism":
        frame["g"] = 0.3
    elif change == "cluster":
        frame["C"] = rows // 4
    elif change == "19 doses":
        # One value short of the 20 distinct doses a density needs.
        frame["A"] = np.linspace(0.3, 4.7, 19)[rows % 19]
    else:
        curve = DoseResponse(doses=(2.0,), bandwidths=(0.5,), support=(-10.0, 10.0))
    assert_refused_before_any_call(
        lambda: _study(frame, **design).identify(DoseResponseCurve(curve)),
        None,
        "",
        fragment,
        error=error,
    )


def test_the_trial_design_refuses_the_curve_end_to_end() -> None:
    from tests.unit.test_atmle_workflow import _frame as trial_frame

    frame = trial_frame(external_treated=0.3)
    design = TrialWithExternalData(outcome="Y", treatment="A", adjustment=("W",), source="S")
    study = CausalStudy(frame, design=design)
    raised = assert_refused_before_any_call(
        lambda: study.identify(DoseResponseCurve(_curve())), None, "", "it declares no dose"
    )
    assert raised.refusal.reason == RefusalReason.INVALID_DECLARATION


# ------------------------------------------------------------------ protocol and provenance


def test_the_protocol_fingerprint_is_stamped() -> None:
    from tests.unit.summary_fixtures.setup import protocol

    frame = _frame()
    first, second = protocol(), protocol(horizon="60 days after discharge")
    results = []
    for declared in (first, second):
        study = CausalStudy(
            frame,
            design=PointTreatment(**DESIGN, treatment_kind="continuous"),
            protocol=declared,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results.append(
                study.identify(DoseResponseCurve(_curve())).estimate(**_learners(), **_BASE)
            )
    assert results[0].provenance.protocol_fingerprint == first.fingerprint
    assert results[0].provenance.protocol_fingerprint != results[1].provenance.protocol_fingerprint
    plain = _route(frame, _curve())
    assert plain.provenance.protocol_fingerprint is None
    assert plain.provenance.data_fingerprint == _direct(frame, _curve()).provenance.data_fingerprint
    assert "No study protocol was recorded." in str(plain.summary(detail="full"))
    assert "No study protocol was recorded." not in str(results[0].summary(detail="full"))


def test_the_result_carries_the_question_and_its_keys() -> None:
    result = _route(_frame(), _curve(), run_id="keys")
    assert result.identified_effect.estimand == DoseResponseCurve(_curve())
    assert isinstance(result.method, TMLEMethod)
    key = result.parameter_keys["dose_response[h=0.75,a=2.5]"]
    assert (key.estimand, key.axis, key.value, key.bandwidth) == (
        "dose_response",
        "dose",
        2.5,
        0.75,
    )
    assert set(result.parameter_keys) == set(result.estimates)
    stratified = _route(_frame(), _curve(), {"strata": ("W2",)})
    levels = {key.stratum for key in stratified.parameter_keys.values()}
    assert levels == {(0.0,), (1.0,)}


# ------------------------------------------------------------------ planned fits


@pytest.mark.parametrize("ensemble", [False, True], ids=["bare", "super learner"])
def test_the_preview_counts_the_fits_the_route_makes(ensemble: bool) -> None:
    effect = _effect()

    def settings() -> dict[str, Any]:
        outcome = (
            counting.ensemble("outcome_learner")
            if ensemble
            else counting.logistic("outcome_learner")
        )
        return {
            "outcome_learner": outcome,
            "treatment_learner": counting.logistic("treatment_learner"),
            **_BASE,
        }

    found = counting.witness(
        lambda: effect.planned_fits(**settings()), lambda: effect.estimate(**settings())
    )
    assert found.plan.exact
    counting.check(
        found,
        plain={"outcome_learner": "LogisticRegression", "treatment_learner": "LogisticRegression"},
    )


# ------------------------------------------------------------------ round trip


def test_save_and_load_keep_the_study_record(tmp_path: Any) -> None:
    result = _route(_frame(), _curve(), run_id="saved")
    path = tmp_path / "curve.cleverly"
    result.save(path)
    loaded = cleverly.load(path)
    assert loaded.parameter_keys == result.parameter_keys
    assert loaded.provenance == result.provenance
    assert loaded.method.name == result.method.name
    assert loaded.method.models.density_bins == result.method.models.density_bins
    assert loaded.identified_effect.functional == result.identified_effect.functional
    assert _facts(loaded)[:-2] == _facts(result)[:-2]
    assert str(loaded.summary()) == str(result.summary())

    def statuses(fit: Any) -> dict[str, Any]:
        rows = {row.operation: row.status for row in fit.diagnostics.capabilities}
        rows.update({row.operation: row.status for row in fit.sensitivity.capabilities})
        return rows

    assert statuses(loaded) == statuses(result)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checked = result.validate()
        reloaded = loaded.validate()
    assert [(item.name, item.status) for item in checked.items] == [
        (item.name, item.status) for item in reloaded.items
    ]
