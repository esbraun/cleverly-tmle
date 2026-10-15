"""The declared support of each continuous component of a vector treatment (X40 step H).

The law: a flag ``F ~ Bernoulli(1/2)`` and a dose ``D | F ~ U(0, 1 + F)``, so the support of
the dose is ``[0, 1 + F]`` given the flag.  The chain-rule density orders the flag first, so
the declared support of ``D`` reads ``F``.  Díaz et al. (2023), Assumption 1, asks that the
assigned vector lie in the support of the vector, so an assigned dose is checked against the
support at the flag the same branch assigns.

- Witness (i): "flag on" sets ``F = 1`` and shifts the dose up to 2.  It passes, because
  every assigned dose lies in ``[0, 2]``.  Read at the natural flag it would fail.
- Witness (ii): "flag off" sets ``F = 0`` and keeps the dose.  It is refused, because a dose
  above 1 at ``F = 0`` is off the support.  Read at the natural flag it would pass.
- Mutation control M: the check read at the natural flag reverses both witnesses.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tests.discrete_law_policy_point as point_law
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import JointPolicy, ModifiedPolicy, Shift, Support
from cleverly.interventions import policy as policy_module
from cleverly.longitudinal import LTMLE, DynamicRegimen, LongitudinalData
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_longitudinal_multivalue import CellProbabilities

from .. import discrete_law_longitudinal_mtp as panel_law
from .. import longitudinal_mtp as mtp
from ..dose_declarations import NeverFit

REACHED = "a learner was fitted before the refusal"
REFUSED = "is not identified under the declared support"
NEVER = {"outcome_learner": NeverFit(), "treatment_learner": NeverFit()}


def flag_one(a: Any, h: Any) -> np.ndarray:
    return np.ones(len(a))


def flag_zero(a: Any, h: Any) -> np.ndarray:
    return np.zeros(len(a))


def dose_upper(h: Any) -> np.ndarray:
    """``1 + F``: the top of the dose support given the flag."""
    return 1.0 + np.asarray(h["F"], dtype=float)


SUPPORT = Support(0.0, dose_upper)
SET_ON = ModifiedPolicy("set on", apply=flag_one, policy_kind="known")
SET_OFF = ModifiedPolicy("set off", apply=flag_zero, policy_kind="known")
NATURAL = JointPolicy("natural", components={}, policy_kind="known")
FLAG_ON = JointPolicy(
    "flag on", components={"F": SET_ON, "D": Shift(0.5, cap=2.0)}, policy_kind="known"
)
FLAG_OFF = JointPolicy("flag off", components={"F": SET_OFF}, policy_kind="known")


def frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    f = rng.binomial(1, 0.5, n).astype(float)
    d = rng.uniform(0.0, 1.0 + f)
    y = rng.normal(d + w, 1.0)
    return pd.DataFrame({"W": w, "F": f, "D": d, "Y": y})


def point_fit(policies: list[Any], data: pd.DataFrame | None = None, **declared: Any) -> Any:
    return TMLE(policies=policies, cross_fit=False, ratio="classifier", **NEVER).fit(
        frame() if data is None else data,
        outcome="Y",
        treatment=["F", "D"],
        covariates=["W"],
        continuous_treatment=["D"],
        **declared,
    )


def natural_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation M: read each component support at the natural categorical components."""
    real = policy_module.check_component_supports

    def mutated(policy: Any, components: Any, natural: Any, assigned: Any, *args: Any, **kw: Any):
        categorical = np.array([not component.continuous for component in components])
        moved = [np.where(categorical, natural, values) for values in assigned]
        return real(policy, components, natural, moved, *args, **kw)

    monkeypatch.setattr(policy_module, "check_component_supports", mutated)


# ------------------------------------------------------------------ the point vector


def test_witness_i_a_dose_inside_the_support_at_the_assigned_flag_passes() -> None:
    """ "flag on" passes the check and reaches the learners."""
    with pytest.raises(AssertionError, match=REACHED):
        point_fit([NATURAL, FLAG_ON], treatment_support={"D": SUPPORT})


def test_witness_ii_a_dose_outside_the_support_at_the_assigned_flag_is_refused() -> None:
    with pytest.raises(DataError, match=REFUSED) as caught:
        point_fit([NATURAL, FLAG_OFF], treatment_support={"D": SUPPORT})
    assert "at node 'D'" in str(caught.value)
    assert "the policy 'flag off'" in str(caught.value)
    # The control: without the declaration the same fit reaches the learners.
    with pytest.raises(AssertionError, match=REACHED):
        point_fit([NATURAL, FLAG_OFF])


def test_mutation_m_the_natural_flag_reverses_both_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    natural_flag(monkeypatch)
    with pytest.raises(DataError, match=REFUSED):
        point_fit([NATURAL, FLAG_ON], treatment_support={"D": SUPPORT})
    with pytest.raises(AssertionError, match=REACHED):
        point_fit([NATURAL, FLAG_OFF], treatment_support={"D": SUPPORT})


def test_an_observed_dose_outside_its_component_support_is_refused() -> None:
    """R-a2: the observed dose is checked at the natural flag."""
    with pytest.raises(DataError, match="observed dose"):
        point_fit([NATURAL], treatment_support={"D": Support(0.0, 1.0)})


def test_the_declaration_is_a_mapping_of_continuous_components() -> None:
    columns = {"outcome": "Y", "treatment": ["F", "D"], "covariates": ["W"]}
    data = CausalData.from_frame(frame(), continuous_treatment=["D"], **columns)
    declared = data.with_dose_declarations({"D": SUPPORT})
    assert declared.component_supports == {"D": SUPPORT}
    assert data.component_supports == {}
    with pytest.raises(DataError, match="maps each continuous component"):
        data.with_dose_declarations(SUPPORT)
    with pytest.raises(DataError, match="not a continuous component"):
        data.with_dose_declarations({"F": SUPPORT})
    with pytest.raises(DataError, match="a declared support is Support"):
        data.with_dose_declarations({"D": (0.0, 2.0)})
    with pytest.raises(CapabilityError, match="declares atoms for the vector treatment"):
        data.with_dose_declarations(None, (1.0,))


def _exact_point(**declared: Any) -> Any:
    labels = ("natural", "dose up", "flag gated")
    policies = [
        JointPolicy("natural", components={}, policy_kind="known"),
        JointPolicy("dose up", components={"D": Shift(1.0, cap=2.0)}, policy_kind="known"),
        JointPolicy(
            "flag gated",
            components={"F": SET_ON, "D": Shift(1.0, cap=2.0)},
            policy_kind="known",
        ),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                policies=policies,
                outcome_learner=CellMeans(),
                treatment_learner=CellMeans(),
                cross_fit=False,
                ratio="classifier",
                simultaneous=False,
            )
            .fit(
                point_law.mixed_frame(),
                outcome="Y",
                treatment=["F", "D"],
                covariates=["W"],
                continuous_treatment=["D"],
                **declared,
            )
            .single()
        )
    return [result.estimates[f"ey_policy[{label}]"].psi for label in labels]


def test_a_support_that_does_not_bind_leaves_the_estimate_bit_identical() -> None:
    assert _exact_point(treatment_support={"D": Support(-1.0, 3.0)}) == _exact_point()


# ------------------------------------------------------------------ the mixed node


def _panel_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    f1 = rng.binomial(1, 0.5, n).astype(float)
    d1 = rng.uniform(0.0, 1.0 + f1)
    l2 = rng.normal(d1, 1.0)
    f2 = rng.binomial(1, 0.5, n).astype(float)
    d2 = rng.uniform(0.0, 1.0 + f2)
    y = rng.binomial(1, 0.5, n).astype(float)
    return pd.DataFrame({"W": w, "F1": f1, "D1": d1, "L2": l2, "F2": f2, "D2": d2, "Y": y})


def _node_upper(h: Any) -> np.ndarray:
    return 1.0 + np.asarray(h["F2"], dtype=float)


def _second(name: str, components: dict[str, Any]) -> JointPolicy:
    return JointPolicy(name, components=components, policy_kind="known")


def _panel_fit(second: JointPolicy, **declared: Any) -> Any:
    first = JointPolicy("natural", components={}, policy_kind="known")
    regimens = {
        "natural": DynamicRegimen("natural", (first, first)),
        second.name: DynamicRegimen(second.name, (first, second)),
    }
    never = {**NEVER, "pseudo_learner": NeverFit(), "censoring_learner": NeverFit()}
    return LTMLE(regimens, n_folds=1, ratio="classifier", **never).fit(
        _panel_frame(),
        outcome="Y",
        treatment=[["F1", "D1"], ["F2", "D2"]],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        **declared,
    )


PANEL_SUPPORT = {"D1": None, "D2": Support(0.0, _node_upper)}
PANEL_ON = _second("flag on", {"F2": SET_ON, "D2": Shift(0.5, cap=2.0)})
PANEL_OFF = _second("flag off", {"F2": SET_OFF})


def test_the_mixed_node_checks_each_branch_at_its_assigned_flag() -> None:
    with pytest.raises(AssertionError, match=REACHED):
        _panel_fit(PANEL_ON, continuous_treatment=PANEL_SUPPORT)
    with pytest.raises(DataError, match=REFUSED) as caught:
        _panel_fit(PANEL_OFF, continuous_treatment=PANEL_SUPPORT)
    assert "regimen 'flag off'" in str(caught.value) and "at node 'D2'" in str(caught.value)
    with pytest.raises(AssertionError, match=REACHED):
        _panel_fit(PANEL_OFF, continuous_treatment=["D1", "D2"])


def test_mutation_m_at_the_mixed_node(monkeypatch: pytest.MonkeyPatch) -> None:
    natural_flag(monkeypatch)
    with pytest.raises(DataError, match=REFUSED):
        _panel_fit(PANEL_ON, continuous_treatment=PANEL_SUPPORT)
    with pytest.raises(AssertionError, match=REACHED):
        _panel_fit(PANEL_OFF, continuous_treatment=PANEL_SUPPORT)


def test_the_mixed_node_declarations() -> None:
    columns = {
        "outcome": "Y",
        "treatment": [["F1", "D1"], ["F2", "D2"]],
        "baseline": ["W"],
        "time_varying": [[], ["L2"]],
    }
    data = LongitudinalData.from_frame(
        _panel_frame(), continuous_treatment=PANEL_SUPPORT, **columns
    )
    assert data.component_supports(2) == {"D2": PANEL_SUPPORT["D2"]}
    assert data.component_supports(1) == {} and data.node_support(2) is None
    with pytest.raises(DataError, match="categorical component of a vector node"):
        data.with_dose_declarations({"F2": Support(0.0, 1.0)}, {})
    with pytest.raises(CapabilityError, match="a component of a vector node"):
        data.with_dose_declarations({}, {"D2": (0.5,)})


def test_a_support_that_does_not_bind_leaves_the_mixed_node_bit_identical() -> None:
    labels = ("natural", "dose up", "flag gated")

    def fit(continuous: Any) -> list[float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = LTMLE(
                mtp.mixed_regimens(labels),
                reference="natural",
                outcome_learner=CellMeans(),
                pseudo_learner=CellMeans(),
                treatment_learner=CellProbabilities(),
                censoring_learner=CellMeans(),
                n_folds=1,
                g_bounds=(1e-8, 1.0),
                simultaneous=False,
                ratio="classifier",
            ).fit(
                panel_law.mixed_frame("end"),
                **{**mtp.MIXED_COLUMNS, "continuous_treatment": continuous},
            )
        return [float(result.psi(f"ey_regimen[{label}]")) for label in labels]

    wide = Support(-1.0, 3.0)
    assert fit({"D1": wide, "D2": wide}) == fit(["D1", "D2"])


# ------------------------------------------------------------------ the study spine


def test_the_point_design_carries_the_component_supports() -> None:
    from cleverly import CausalStudy, PointTreatment

    def design(**declared: Any) -> PointTreatment:
        return PointTreatment(
            outcome="Y",
            treatment=["F", "D"],
            adjustment=("W",),
            continuous_treatment=["D"],
            **declared,
        )

    study = CausalStudy(frame(), design=design(treatment_support={"D": SUPPORT}))
    assert study.data.component_supports == {"D": SUPPORT}
    CausalStudy(study.data, design=design(treatment_support={"D": SUPPORT}))
    with pytest.raises(DataError, match="treatment_support="):
        CausalStudy(study.data, design=design())


def test_the_sequence_design_carries_the_component_supports() -> None:
    from cleverly import CausalStudy, LongitudinalTreatment

    def design(continuous: Any) -> LongitudinalTreatment:
        return LongitudinalTreatment(
            outcome="Y",
            treatment=[["F1", "D1"], ["F2", "D2"]],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=continuous,
        )

    study = CausalStudy(_panel_frame(), design=design(PANEL_SUPPORT))
    assert study.data.component_supports(2) == {"D2": PANEL_SUPPORT["D2"]}
    CausalStudy(study.data, design=design(PANEL_SUPPORT))
    with pytest.raises(DataError, match="declared dose supports"):
        CausalStudy(study.data, design=design(["D1", "D2"]))
