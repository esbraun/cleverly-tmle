"""Vector treatments through the CausalStudy spine, and the ``ratio`` method setting.

``PointTreatment(treatment=[...], continuous_treatment=[...])`` declares a point vector
treatment, and ``LongitudinalTreatment`` declares vector nodes.  The spine fits both on the
route that ``ModelSpec.ratio`` names.  The witnesses follow the template of
``test_t11_a_study_identifies_and_estimates_a_contrast_of_policies``: identify, estimate
with saturated learners, and compare with the exact-law oracle.

- Point, all categorical (a1): one categorical treatment over the observed tuples.  A joint
  policy and a static tuple level each equal the oracle.
- Point with a dose (a2): a contrast of joint policies equals the oracle on both routes.
- Longitudinal mixed nodes: a contrast of joint-policy regimens equals the oracle on both
  routes.
- Routing: on a sample with misspecified learners the two routes give different numbers,
  and the study's number equals the direct fit's on the route it names.  A categorical
  point policy, a point vector and a mixed panel each carry the witness.

The refusals run before any learner, which a learner that raises on ``fit`` checks.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import tests.discrete_law_longitudinal_mtp as long_law
import tests.discrete_law_policy_point as law
from cleverly import (
    ATE,
    CausalStudy,
    CollaborativeTMLEMethod,
    CounterfactualMean,
    DRTMLEMethod,
    IPSIMethod,
    LongitudinalTreatment,
    ModelSpec,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    MSMProjection,
    PointTreatment,
    RegimeContrast,
    TMLEMethod,
)
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, CleverlyError, DataError, MethodConfigurationError
from cleverly.longitudinal import LTMLE
from cleverly.methods import SHORTCUTS
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_longitudinal_multivalue import CellProbabilities

from .. import longitudinal_mtp as mtp
from . import test_point_vector_treatment as point_tests
from .test_longitudinal_mixed_node import NeverFit

pytestmark = pytest.mark.xdist_group("vector_study_spine")

EXACT = 1e-10
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
NEVER = {"outcome_learner": NeverFit(), "treatment_learner": NeverFit()}


def point_design(**fields: Any) -> PointTreatment:
    return PointTreatment(outcome="Y", treatment=["F", "D"], adjustment=("W",), **fields)


def mixed_design() -> LongitudinalTreatment:
    return LongitudinalTreatment(
        outcome="Y",
        treatment=(("F1", "D1"), ("F2", "D2")),
        baseline=("W",),
        time_varying=((), ("L2",)),
        continuous_treatment=("D1", "D2"),
    )


def _estimate(identified: Any, **overrides: Any) -> Any:
    """An exact fit: the bins of every density are the oracle's support points."""
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return identified.estimate(**overrides)


def _sample_estimate(identified: Any, **overrides: Any) -> Any:
    """A sample fit, with the default bins that the direct fit beside it uses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return identified.estimate(**overrides)


# ------------------------------------------------------------------ point (a1)


def test_a_point_categorical_vector_identifies_and_estimates_through_the_spine() -> None:
    """The (a1) reading: a joint-policy contrast and a static tuple mean equal the oracle."""
    design = point_design()
    assert design.support == "discrete"
    assert design.treatment == ("F", "D") and design.treatment_name == "F+D"
    study = CausalStudy(law.mixed_frame(), design=design)
    declared = point_tests._categorical_declarations()
    identified = study.identify(
        ModifiedTreatmentPolicyEffect(
            policies=[declared["natural"], declared["flag gated"]], reference="natural"
        )
    )
    rows = law.mixed_first_row_of()
    for ratio, learner in (("classifier", CellMeans()), ("density", CellProbabilities())):
        result = _estimate(
            identified,
            outcome_learner=CellMeans(),
            treatment_learner=learner,
            cross_fit=False,
            ratio=ratio,
            g_bounds=(1e-6, 1.0 - 1e-6),
            simultaneous=False,
        )
        estimate = result.estimates["ate_policy[flag gated vs natural]"]
        truth = law.mixed_truth("flag gated") - law.mixed_truth("natural")
        assert estimate.psi == pytest.approx(truth, abs=EXACT)
        eif = law.mixed_eif("flag gated") - law.mixed_eif("natural")
        np.testing.assert_allclose(np.asarray(estimate.influence_curve)[rows], eif, atol=EXACT)
    static = study.identify(CounterfactualMean(treatment=(1, 2.0)))
    result = _estimate(
        static,
        outcome_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        cross_fit=False,
        g_bounds=(1e-6, 1.0 - 1e-6),
        simultaneous=False,
    )
    assert result.psi("ey[(1, 2.0)]") == pytest.approx(law.mixed_truth("static (1, 2)"), abs=EXACT)


# ------------------------------------------------------------------ point (a2)


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_a_point_dose_vector_identifies_and_estimates_through_the_spine(ratio: str) -> None:
    design = point_design(continuous_treatment=["D"])
    assert design.support == "vector"
    study = CausalStudy(law.mixed_frame(), design=design)
    identified = study.identify(
        ModifiedTreatmentPolicyEffect(
            policies=[point_tests.joint("natural"), point_tests.joint("flag gated")],
            reference="natural",
        )
    )
    available = {row.name: row for row in identified.available_methods()}
    assert available["tmle"].available
    for name in ("collaborative_tmle", "drtmle", "sdr", "ipsi"):
        assert not available[name].available
    assert "'F+D' is a vector with a continuous component" in str(
        available["collaborative_tmle"].reason
    )
    assert "DRTMLE and policies= are not combined" in str(available["drtmle"].reason)
    result = _estimate(
        identified,
        outcome_learner=CellMeans(),
        treatment_learner=CellMeans() if ratio == "classifier" else CellProbabilities(),
        cross_fit=False,
        ratio=ratio,
        g_bounds=NO_TRUNCATION,
        simultaneous=False,
    )
    estimate = result.estimates["ate_policy[flag gated vs natural]"]
    truth = law.mixed_truth("flag gated") - law.mixed_truth("natural")
    assert estimate.psi == pytest.approx(truth, abs=EXACT)
    eif = law.mixed_eif("flag gated") - law.mixed_eif("natural")
    rows = law.mixed_first_row_of()
    np.testing.assert_allclose(np.asarray(estimate.influence_curve)[rows], eif, atol=EXACT)


def test_a_prepared_vector_container_passes_the_design_check() -> None:
    design = point_design(continuous_treatment=["D"])
    data = point_tests.vector_data()
    assert design.prepare(data) is data
    with pytest.raises(DataError, match="continuous_treatment"):
        point_design().prepare(data)
    with pytest.raises(DataError, match="continuous_treatment=\\('D',\\)"):
        point_design(continuous_treatment=["D"]).prepare(point_tests.categorical_data())


def test_a_point_vector_design_refuses_as_the_container_does() -> None:
    with pytest.raises(DataError, match="treatment_kind='continuous' describes a treatment"):
        point_design(treatment_kind="continuous")
    with pytest.raises(DataError, match="names 1 column"):
        PointTreatment(outcome="Y", treatment=["F"], adjustment=("W",))
    with pytest.raises(DataError, match="not treatment columns"):
        point_design(continuous_treatment=["W"])
    with pytest.raises(DataError, match="is one column"):
        PointTreatment(outcome="Y", treatment="D", adjustment=("W",), continuous_treatment=["D"])
    with pytest.raises(DataError, match="different columns"):
        PointTreatment(outcome="Y", treatment=["F", "Y"], adjustment=("W",))


@pytest.mark.parametrize(
    ("estimand", "reason"),
    [
        (ATE(), "it is indexed by arm, and such a treatment has no arms"),
        (MSMProjection(model="linear"), "a vector has no one dose axis"),
    ],
    ids=["arm", "msm"],
)
def test_an_estimand_off_the_policy_axis_is_refused_at_a_dose_vector(
    estimand: Any, reason: str
) -> None:
    study = CausalStudy(law.mixed_frame(), design=point_design(continuous_treatment=["D"]))
    with pytest.raises(CapabilityError) as caught:
        study.identify(estimand)
    assert str(caught.value).startswith(
        "treatment 'F+D' is a vector with a continuous component, and "
    )
    assert reason in str(caught.value)
    assert "policies=[JointPolicy(...), ...]" in str(caught.value)


def test_the_axis_readers_fail_closed_on_an_unknown_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A support no reader names raises instead of taking the arm branch."""
    from cleverly import study as study_module

    design = point_design(continuous_treatment=["D"])
    assert study_module._design_has_arms(design) is False
    assert study_module._design_has_arms(point_design()) is True
    monkeypatch.setattr(PointTreatment, "support", property(lambda self: "matrix"))
    with pytest.raises(CleverlyError, match="treatment support 'matrix' has no reading"):
        study_module._design_has_arms(design)


# ------------------------------------------------------------------ longitudinal mixed


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_mixed_nodes_identify_and_estimate_through_the_spine(ratio: str) -> None:
    study = CausalStudy(long_law.mixed_frame(), design=mixed_design())
    identified = study.identify(
        RegimeContrast(mtp.mixed_regimens(("natural", "flag gated")), reference="natural")
    )
    available = {row.name: row for row in identified.available_methods()}
    assert available["tmle"].available and available["sdr"].available
    assert not available["ipsi"].available
    assert "IPSI estimates no density ratio" in str(available["ipsi"].reason)
    result = _estimate(
        identified,
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        g_bounds=(1e-8, 1.0),
        cross_fit=False,
        ratio=ratio,
        simultaneous=False,
    )
    name = "ate_regimen[flag gated vs natural]"
    truth = long_law.mixed_truth("flag gated") - long_law.mixed_truth("natural")
    assert result.psi(name) == pytest.approx(truth, abs=EXACT)
    rows = long_law.mixed_first_row_of()
    eif = long_law.mixed_eif("flag gated") - long_law.mixed_eif("natural")
    np.testing.assert_allclose(result.influence_curves[name][rows], eif, atol=EXACT)


# ------------------------------------------------------------------ routing


def _sample_learners() -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": LogisticRegression(),
        "cross_fit": False,
        "simultaneous": False,
    }


def test_a_categorical_point_policy_takes_the_route_ratio_names() -> None:
    """The study fit equals the direct fit on its route, and the routes differ."""
    frame = law.mixed_sample(400, 11)
    design = PointTreatment(outcome="Y", treatment="F", adjustment=("W", "D"))
    flag = point_tests.SET_FLAG
    identified = CausalStudy(frame, design=design).identify(
        ModifiedTreatmentPolicy(policies=[flag])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_frame(frame, outcome="Y", treatment="F", covariates=["W", "D"])
        direct = {
            ratio: TMLE(policies=[flag], ratio=ratio, **_sample_learners()).fit(data).single()
            for ratio in ("classifier", "density")
        }
    name = "ey_policy[set flag]"
    assert abs(direct["classifier"].psi(name) - direct["density"].psi(name)) > 1e-4
    for ratio in ("classifier", "density"):
        study = _sample_estimate(identified, ratio=ratio, **_sample_learners())
        assert study.psi(name) == pytest.approx(direct[ratio].psi(name), abs=1e-12)


def test_a_point_dose_vector_takes_the_route_ratio_names() -> None:
    frame = law.mixed_sample(400, 12)
    policies = [point_tests.joint("natural"), point_tests.joint("dose up")]
    identified = CausalStudy(frame, design=point_design(continuous_treatment=["D"])).identify(
        ModifiedTreatmentPolicyEffect(policies=policies, reference="natural")
    )
    data = point_tests.vector_data(frame)
    name = "ate_policy[dose up vs natural]"
    psis = {}
    for ratio in ("classifier", "density"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = (
                TMLE(policies=policies, ratio=ratio, density_bins=5, **_sample_learners())
                .fit(data)
                .single()
            )
        study = _sample_estimate(identified, ratio=ratio, density_bins=5, **_sample_learners())
        assert study.psi(name) == pytest.approx(direct.psi(name), abs=1e-12)
        psis[ratio] = direct.psi(name)
    assert abs(psis["classifier"] - psis["density"]) > 1e-4


def test_mixed_nodes_take_the_route_and_the_bins_the_method_names() -> None:
    """Before this row, ``density_bins`` was a point-only setting that a panel refused."""
    frame = long_law.mixed_frame()
    regimens = mtp.mixed_regimens(("natural", "dose up"))
    identified = CausalStudy(frame, design=mixed_design()).identify(
        RegimeContrast(regimens, reference="natural")
    )
    from sklearn.linear_model import LinearRegression

    # Misspecified regressions, so the targeting step reads the ratio and the routes differ.
    settings = {
        "outcome_learner": LogisticRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(),
        "simultaneous": False,
    }
    name = "ate_regimen[dose up vs natural]"
    psis = {}
    for ratio, bins in (("classifier", None), ("density", 4)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            direct = LTMLE(
                regimens,
                reference="natural",
                ratio=ratio,
                density_bins=bins,
                n_folds=1,
                **settings,
            ).fit(frame, **mtp.MIXED_COLUMNS)
        study = _sample_estimate(
            identified, ratio=ratio, density_bins=bins, cross_fit=False, **settings
        )
        assert study.psi(name) == pytest.approx(direct.psi(name), abs=1e-12)
        psis[(ratio, bins)] = direct.psi(name)
    assert abs(psis[("classifier", None)] - psis[("density", 4)]) > 1e-4
    # The bins reach the engine: two bins is the engine's own refusal.
    with pytest.raises(ValueError, match="density_bins must be at least 3; got 2"):
        _sample_estimate(identified, density_bins=2, cross_fit=False, **settings)


# ------------------------------------------------------------------ refusals


def test_ratio_classifier_is_refused_where_no_policy_ratio_is_estimated() -> None:
    """Each refusal runs before any learner: the learners raise on ``fit``."""
    frame = law.mixed_frame()
    ate = CausalStudy(frame, design=PointTreatment("Y", "F", adjustment=("W",))).identify(ATE())
    with pytest.raises(MethodConfigurationError, match=r"ATE declares no policy\. Drop ratio="):
        ate.estimate(ratio="classifier", **NEVER)
    known = frame.assign(p0=lambda f: 0.5, p1=lambda f: 0.5)
    design = PointTreatment("Y", "F", adjustment=("W",), treatment_probabilities={0: "p0", 1: "p1"})
    policy = CausalStudy(known, design=design).identify(
        ModifiedTreatmentPolicy(policies=[point_tests.SET_FLAG])
    )
    with pytest.raises(MethodConfigurationError, match="treatment_probabilities= declares"):
        policy.estimate(ratio="classifier", **NEVER)
    panel = long_law.mixed_frame()
    categorical = LongitudinalTreatment(
        outcome="Y", treatment=("F1", "F2"), baseline=("W",), time_varying=((), ("L2",))
    )
    static = CausalStudy(panel, design=categorical).identify(
        RegimeContrast({"one": (1, 1), "zero": (0, 0)}, reference="zero")
    )
    with pytest.raises(MethodConfigurationError, match="every node of this design is categ"):
        static.estimate(ratio="classifier", pseudo_learner=NeverFit(), **NEVER)


@pytest.mark.parametrize(
    ("method", "match"),
    [
        (CollaborativeTMLEMethod, "collaborative TMLE fits no policies="),
        (DRTMLEMethod, "DR-TMLE fits no policies="),
        (IPSIMethod, "IPSI estimates none"),
    ],
)
def test_a_method_that_fits_no_policy_ratio_refuses_ratio_classifier(
    method: Any, match: str
) -> None:
    with pytest.raises(MethodConfigurationError, match=match):
        method(models=ModelSpec(ratio="classifier"))
    with pytest.raises(MethodConfigurationError, match=match):
        method().with_overrides(ratio="classifier")


def test_ipsi_refuses_density_bins() -> None:
    with pytest.raises(MethodConfigurationError, match="IPSI fits no density"):
        IPSIMethod().with_overrides(density_bins=5)


def test_ratio_is_validated_at_construction() -> None:
    with pytest.raises(MethodConfigurationError, match="ratio must be 'density' or 'classifier'"):
        ModelSpec(ratio="odds")  # type: ignore[arg-type]


def test_the_ratio_and_bins_shortcuts_reach_both_engines() -> None:
    """``ratio`` and ``density_bins`` set ``ModelSpec`` and reach both translations."""
    assert SHORTCUTS["models"]["ratio"] == "ratio"
    assert SHORTCUTS["models"]["density_bins"] == "density_bins"
    method = TMLEMethod().with_overrides(ratio="classifier", density_bins=7)
    assert method.models.ratio == "classifier" and method.models.density_bins == 7
    for longitudinal in (False, True):
        kwargs = method.estimator_kwargs(longitudinal=longitudinal)
        assert kwargs["ratio"] == "classifier"
        assert kwargs["density_bins"] == 7
