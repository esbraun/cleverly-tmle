"""The sequential baselines through ``CausalStudy``: availability, methods and diagnostics.

``method="gcomp"`` and ``method="ipw"`` fit a longitudinal regimen mean or contrast.  Both
refuse a point functional and a working model over regimens.  IPW fits no outcome regression,
so its nuisance report omits the outcome role.
"""

from __future__ import annotations

from typing import Any

import pytest

from cleverly import (
    CausalStudy,
    GComputationMethod,
    IPWMethod,
    LongitudinalTreatment,
    MSMProjection,
    PointTreatment,
    RegimeContrast,
    RegimeMean,
    Targeting,
)
from cleverly.assessment import (
    LONGITUDINAL_NO_OUTCOME_REGRESSION,
    AssessmentStatus,
)
from cleverly.datasets import make_linear_ate, make_longitudinal
from cleverly.exceptions import CapabilityError, MethodConfigurationError
from cleverly.longitudinal import SequentialGComputation, SequentialIPW
from cleverly.longitudinal.estimator import BASELINE_MSM_REFUSAL, baseline_cross_fit_refusal
from cleverly.methods import BASELINE_POINT_REFUSAL, baseline_no_fluctuation_refusal
from cleverly.msm import MSM

PAIR = {"always": 1, "never": 0}
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ("A1", "A2"),
    "baseline": ("W1", "W2"),
    "time_varying": ((), ("L2",)),
    "censoring": ("C1", "C2"),
}


@pytest.fixture(scope="module")
def study() -> CausalStudy:
    frame, _ = make_longitudinal(n=400, seed=12)
    return CausalStudy(frame, design=LongitudinalTreatment(**COLUMNS))


@pytest.mark.parametrize(
    ("method", "engine"), [("gcomp", SequentialGComputation), ("ipw", SequentialIPW)]
)
def test_study_estimate_equals_the_engine(study: CausalStudy, method: str, engine: type) -> None:
    result = study.identify(RegimeContrast(PAIR, reference="never")).estimate(
        method=method, random_state=3, simultaneous=False
    )
    direct = engine(PAIR, reference="never", random_state=3, simultaneous=False).fit(study.data)
    assert result.config.estimator == method
    assert result.fitted_method == method
    assert result.config.n_folds == 1
    assert "ate_regimen[always vs never]" in result.estimates
    for name, estimate in result.estimates.items():
        assert estimate.psi == pytest.approx(direct.estimates[name].psi, abs=1e-12)
    assert result.parameter_keys.keys() == result.estimates.keys()


def test_study_assess_reports_the_baseline_rows(study: CausalStudy) -> None:
    effect = study.identify(RegimeMean(PAIR))
    gcomp = effect.estimate(method="gcomp", random_state=3, simultaneous=False)
    ipw = effect.estimate(method="ipw", random_state=3, simultaneous=False)
    for result in (gcomp, ipw):
        rows = {row.operation: row for row in result.diagnostics.capabilities}
        assert rows["score_equations"].status is AssessmentStatus.NOT_APPLICABLE
        assert rows["support"].available and rows["nuisance_models"].available
        assert result.assess() is not None
    rows = {row.operation: row for row in gcomp.diagnostics.capabilities}
    assert rows["truncation_curve"].status is AssessmentStatus.UNAVAILABLE
    rows = {row.operation: row for row in ipw.diagnostics.capabilities}
    assert rows["truncation_curve"].available


def test_ipw_normalization_reaches_the_engine_and_the_summary(study: CausalStudy) -> None:
    effect = study.identify(RegimeMean(PAIR))
    hajek = effect.estimate(method="ipw", random_state=3, simultaneous=False)
    ht = effect.estimate(
        method=IPWMethod(normalization="horvitz_thompson"), random_state=3, simultaneous=False
    )
    assert hajek.config.normalization == "hajek"
    assert ht.config.normalization == "horvitz_thompson"
    assert "Hajek" in hajek.summary()
    assert "Horvitz" in ht.summary()


def test_ipw_nuisance_report_omits_the_outcome_regression(study: CausalStudy) -> None:
    effect = study.identify(RegimeMean(PAIR))
    ipw = effect.estimate(method="ipw", random_state=3, simultaneous=False)
    report = ipw.diagnostics.nuisance_models()
    assert {row.role for row in report.rows} == {"treatment", "censoring"}
    assert [(o.role, o.time, o.reason) for o in report.omissions] == [
        ("outcome", None, LONGITUDINAL_NO_OUTCOME_REGRESSION)
    ]
    gcomp = effect.estimate(method="gcomp", random_state=3, simultaneous=False)
    roles = {row.role for row in gcomp.diagnostics.nuisance_models().rows}
    assert {"outcome", "treatment", "censoring"} <= roles


def test_baselines_are_unavailable_on_a_point_functional() -> None:
    from cleverly import ATE

    frame, _ = make_linear_ate(n=120, seed=19)
    effect = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2"))
    ).identify(ATE())
    methods = {record.name: record for record in effect.available_methods()}
    for name in ("gcomp", "ipw"):
        assert not methods[name].available
        assert methods[name].refusal == BASELINE_POINT_REFUSAL
        with pytest.raises(CapabilityError, match="point-treatment functional"):
            effect.estimate(method=name)


def test_baselines_are_unavailable_on_a_working_model(study: CausalStudy) -> None:
    effect = study.identify(MSMProjection(MSM.linear(), regimens=PAIR))
    methods = {record.name: record for record in effect.available_methods()}
    for name in ("gcomp", "ipw"):
        assert not methods[name].available
        assert methods[name].refusal == BASELINE_MSM_REFUSAL
    assert methods["sdr"].available


def test_baselines_are_available_on_a_longitudinal_functional(study: CausalStudy) -> None:
    methods = {r.name: r for r in study.identify(RegimeMean(PAIR)).available_methods()}
    assert methods["gcomp"].available and methods["gcomp"].reason is None
    assert methods["ipw"].available and methods["ipw"].reason is None


@pytest.mark.parametrize("method_class", [GComputationMethod, IPWMethod])
def test_baseline_methods_default_to_in_sample(method_class: type) -> None:
    method = method_class()
    assert not method.cross_fitting.enabled
    assert method.estimator_kwargs(longitudinal=True)["n_folds"] == 1
    with pytest.raises(MethodConfigurationError) as raised:
        method.estimator_kwargs()
    assert raised.value.refusal == BASELINE_POINT_REFUSAL


@pytest.mark.parametrize("method_class", [GComputationMethod, IPWMethod])
@pytest.mark.parametrize(
    ("field", "value", "engine"),
    [
        ("submodel_alpha", 0.9, "alpha"),
        ("max_iter", 3, "max_iter"),
        ("tol", 1e-6, "tol"),
        ("stopping", "scaled_var", "stopping"),
        ("step_bound", 0.5, "step_bound"),
        ("iterate", "best", "iterate"),
    ],
)
def test_baseline_methods_refuse_a_fluctuation_setting(
    method_class: type, field: str, value: Any, engine: str
) -> None:
    method = method_class(targeting=Targeting(**{field: value}))
    with pytest.raises(MethodConfigurationError) as raised:
        method.estimator_kwargs(longitudinal=True)
    assert raised.value.refusal == baseline_no_fluctuation_refusal(engine)


def test_baseline_methods_translate_to_engine_settings() -> None:
    SequentialGComputation(PAIR, **GComputationMethod().estimator_kwargs(longitudinal=True))
    kwargs = IPWMethod(normalization="horvitz_thompson").estimator_kwargs(longitudinal=True)
    assert SequentialIPW(PAIR, **kwargs).normalization == "horvitz_thompson"


def test_cross_fitting_enabled_reaches_the_engine_refusal(study: CausalStudy) -> None:
    from cleverly import CrossFitting

    effect = study.identify(RegimeMean(PAIR))
    method = GComputationMethod(cross_fitting=CrossFitting(n_folds=3))
    with pytest.raises(CapabilityError) as raised:
        effect.estimate(method=method)
    assert raised.value.refusal == baseline_cross_fit_refusal("SequentialGComputation")
