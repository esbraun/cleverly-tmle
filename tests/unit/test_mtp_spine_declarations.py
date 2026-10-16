"""The study spine carries the declared support and atoms of a dose (X40 step H, item 3).

``PointTreatment``, ``TimeToEvent`` and ``LongPanel`` take ``treatment_support`` and
``dose_atoms``; ``LongitudinalTreatment`` takes the mapping form of ``continuous_treatment``
and ``dose_atoms``, as ``LTMLE.fit`` does.  Each witness pairs a spine fit with the direct
fit of the same declaration: a refusal reads the same text, and a passing fit reports the same
estimate bit for bit.  The nonzero control is the same spine fit without the declaration,
which reaches the learners (R-a1) or meets R-b4 (an undeclared atom).
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import (
    CausalStudy,
    LongitudinalTreatment,
    LongPanel,
    ModifiedTreatmentPolicy,
    PointTreatment,
    RegimeMean,
    TimeToEvent,
)
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.interventions import ModifiedPolicy, Shift, Support
from cleverly.longitudinal import LTMLE, DynamicRegimen

from .. import dose_declarations as law
from .. import longitudinal_mtp as mtp

NATURAL = Shift(0.0, cap=None, name="natural")
#: ``a + 0.5`` held at 1.5: on a row with ``W = 0`` it assigns up to 1.5, above ``u(h) = 1``.
BINDING = Shift(0.5, cap=1.5, name="binding")
TO_ATOM = ModifiedPolicy("to atom", pieces=mtp.TO_ATOM_PIECES, policy_kind="known")
NEVER = {"outcome_learner": law.NeverFit(), "treatment_learner": law.NeverFit()}


def _point_design(**declared: Any) -> PointTreatment:
    return PointTreatment(
        outcome="Y", treatment="A", adjustment=("W",), treatment_kind="continuous", **declared
    )


def _point_estimate(frame: pd.DataFrame, policies: list[Any], design: Any, **fit: Any) -> Any:
    study = CausalStudy(frame, design=design)
    return study.identify(ModifiedTreatmentPolicy(policies=policies)).estimate(
        simultaneous=False, **fit
    )


# ------------------------------------------------------------------ the point design


def test_the_point_design_refuses_a_policy_outside_its_declared_support() -> None:
    """R-a1 through the spine, with the text of the direct fit, before any learner."""
    frame = law.point_frame()
    support = Support(0.0, law.upper)
    with pytest.raises(DataError) as direct:
        TMLE(policies=[NATURAL, BINDING], cross_fit=False, **NEVER).fit(
            frame, outcome="Y", treatment="A", covariates=["W"], treatment_support=support
        )
    assert "is not identified under the declared support" in str(direct.value)
    with pytest.raises(DataError, match=f"^{re.escape(str(direct.value))}$"):
        _point_estimate(
            frame,
            [NATURAL, BINDING],
            _point_design(treatment_support=support),
            cross_fit=False,
            **NEVER,
        )
    # The control: without the declaration the same spine fit reaches the learners.
    with (
        pytest.raises(AssertionError, match="a learner was fitted before the refusal"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        _point_estimate(frame, [NATURAL, BINDING], _point_design(), cross_fit=False, **NEVER)


def _jump_direct() -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(policies=[NATURAL, TO_ATOM], simultaneous=False, **law.point_learners())
            .fit(
                law.jump_frame(800),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                dose_atoms=(law.JUMP_ATOM,),
            )
            .single()
        )
    return float(result.psi("ey_policy[to atom]"))


def test_the_point_design_declares_atoms_and_equals_the_direct_fit() -> None:
    """A spine fit with ``dose_atoms`` reports the direct fit's estimate bit for bit."""
    frame = law.jump_frame(800)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _point_estimate(
            frame,
            [NATURAL, TO_ATOM],
            _point_design(dose_atoms=(law.JUMP_ATOM,)),
            **law.point_learners(),
        )
    assert result.data.dose_atoms == (law.JUMP_ATOM,)
    assert float(result.psi("ey_policy[to atom]")) == _jump_direct()
    # The control: without the atoms the recorded point mass is refused (R-b4).
    with pytest.raises(DataError, match="needs that mass declared"):
        _point_estimate(frame, [NATURAL, TO_ATOM], _point_design(), **law.point_learners())


def test_a_prepared_point_container_must_carry_the_design_declarations() -> None:
    support = Support(0.0, law.upper)
    study = CausalStudy(law.point_frame(), design=_point_design(treatment_support=support))
    assert study.data.treatment_support == support
    CausalStudy(study.data, design=_point_design(treatment_support=support))
    with pytest.raises(DataError, match="treatment_support="):
        CausalStudy(study.data, design=_point_design())
    with pytest.raises(DataError, match="dose_atoms="):
        CausalStudy(study.data, design=_point_design(treatment_support=support, dose_atoms=(1.0,)))


# ------------------------------------------------------------------ the sequence designs


def _panel_design(**declared: Any) -> LongitudinalTreatment:
    return LongitudinalTreatment(
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        **declared,
    )


def _panel_estimate(frame: pd.DataFrame, regimens: dict[str, Any], design: Any, **fit: Any) -> Any:
    """The spine fit of ``regimens``; one fold is the spine's in-sample fit."""
    if fit.pop("n_folds", None) == 1:
        fit["cross_fit"] = False
    study = CausalStudy(frame, design=design)
    return study.identify(RegimeMean(regimens)).estimate(**fit)


def test_the_sequence_design_takes_the_mapping_form_of_continuous_treatment() -> None:
    """R-a1 through ``LongitudinalTreatment(continuous_treatment={...})``, before any learner."""
    frame = law.panel_frame()
    support = Support(0.0, law.upper)
    regimens = {"natural": NATURAL, "up at 2": DynamicRegimen("up at 2", (NATURAL, BINDING))}
    never = {**NEVER, "pseudo_learner": law.NeverFit()}
    with pytest.raises(DataError) as direct:
        LTMLE(regimens, n_folds=1, **never).fit(
            frame, continuous_treatment={"A1": None, "A2": support}, **law.PANEL
        )
    assert "at node 'A2'" in str(direct.value)
    design = _panel_design(continuous_treatment={"A1": None, "A2": support})
    assert design.continuous_treatment == (("A1", None), ("A2", support))
    with pytest.raises(DataError, match=f"^{re.escape(str(direct.value))}$"):
        _panel_estimate(frame, regimens, design, n_folds=1, **never)
    with (
        pytest.raises(AssertionError, match="a learner was fitted before the refusal"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        _panel_estimate(
            frame,
            regimens,
            _panel_design(continuous_treatment=["A1", "A2"]),
            n_folds=1,
            **never,
        )


def test_the_sequence_design_declares_atoms_and_equals_the_direct_fit() -> None:
    frame = law.jump_panel(800)
    regimens = {"natural": NATURAL, "to atom": DynamicRegimen("to atom", (NATURAL, mtp.TO_ATOM))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = LTMLE(regimens, **law.panel_learners()).fit(
            frame,
            continuous_treatment=["A1", "A2"],
            dose_atoms={"A2": (law.JUMP_ATOM,)},
            **law.PANEL,
        )
        design = _panel_design(
            continuous_treatment=["A1", "A2"], dose_atoms={"A2": [law.JUMP_ATOM]}
        )
        spine = _panel_estimate(frame, regimens, design, **law.panel_learners())
    assert spine.data.node_atoms(2) == (law.JUMP_ATOM,)
    assert float(spine.psi("ey_regimen[to atom]")) == float(direct.psi("ey_regimen[to atom]"))
    with pytest.raises(DataError, match="needs that mass declared"):
        _panel_estimate(
            frame,
            regimens,
            _panel_design(continuous_treatment=["A1", "A2"]),
            **law.panel_learners(),
        )


def test_a_prepared_sequence_container_must_carry_the_design_declarations() -> None:
    support = Support(0.0, law.upper)
    design = _panel_design(continuous_treatment={"A1": None, "A2": support})
    study = CausalStudy(law.panel_frame(), design=design)
    assert study.data.node_support(2) == support and study.data.node_support(1) is None
    CausalStudy(study.data, design=design)
    with pytest.raises(DataError, match="declared dose supports"):
        CausalStudy(study.data, design=_panel_design(continuous_treatment=["A1", "A2"]))
    with pytest.raises(DataError, match="dose_atoms="):
        CausalStudy(
            study.data,
            design=_panel_design(
                continuous_treatment={"A1": None, "A2": support}, dose_atoms={"A1": (0.5,)}
            ),
        )


def _survival_frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """One row per unit: ``W``, a dose ``A ~ U(0, 1 + W)``, a time on 1 to 3 and an event."""
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 2, n).astype(float)
    w[0] = 1.0
    a = rng.uniform(0.0, 1.0 + w)
    time = rng.integers(1, 4, n).astype(float)
    event = rng.binomial(1, 0.6, n)
    return pd.DataFrame({"W": w, "A": a, "ftime": time, "ftype": event})


def test_the_time_to_event_design_declares_the_dose() -> None:
    support = Support(0.0, law.upper)
    design = TimeToEvent(
        time="ftime",
        event="ftype",
        treatment="A",
        baseline=["W"],
        continuous_treatment=True,
        treatment_support=support,
    )
    study = CausalStudy(_survival_frame(), design=design)
    assert all(study.data.node_support(t) == support for t in range(1, study.data.n_times + 1))
    CausalStudy(study.data, design=design)
    with pytest.raises(DataError, match="treatment_support="):
        CausalStudy(
            study.data,
            design=TimeToEvent(
                time="ftime",
                event="ftype",
                treatment="A",
                baseline=["W"],
                continuous_treatment=True,
            ),
        )


def test_the_long_panel_design_declares_the_dose_at_every_node() -> None:
    frame = law.panel_frame()
    rows = []
    for t, (dose, cov) in enumerate((("A1", None), ("A2", "L2")), start=1):
        rows.append(
            pd.DataFrame(
                {
                    "unit": np.arange(len(frame)),
                    "t": t,
                    "W": frame["W"],
                    "L": np.random.default_rng(t).normal(size=len(frame))
                    if cov is None
                    else frame[cov],
                    "A": frame[dose],
                    "Y": frame["Y"],
                }
            )
        )
    panel = pd.concat(rows, ignore_index=True)
    support = Support(0.0, law.upper)
    design = LongPanel(
        unit="unit",
        time="t",
        treatment="A",
        baseline=["W"],
        time_varying=["L"],
        outcome="Y",
        continuous_treatment=True,
        treatment_support=support,
    )
    study = CausalStudy(panel, design=design)
    assert study.data.node_support(1) == support and study.data.node_support(2) == support
    CausalStudy(study.data, design=design)
    with pytest.raises(DataError, match="treatment_support="):
        CausalStudy(
            study.data,
            design=LongPanel(
                unit="unit",
                time="t",
                treatment="A",
                baseline=["W"],
                time_varying=["L"],
                outcome="Y",
                continuous_treatment=True,
            ),
        )
