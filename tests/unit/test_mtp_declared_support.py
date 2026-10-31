"""The declared conditional support of a continuous dose (X40 part (a)).

A declared support ``Support(lower, upper)`` states the support of ``A_t`` given the history,
which Díaz et al. (2023), Assumption 1, needs the assigned dose to stay inside.  The fit
checks it before any learner: R-a2 refuses an observed dose outside it, and R-a1 refuses a
policy that assigns a row at risk a dose outside it.  T-a1 to T-a5 pin both refusals, an
``h``-dependent bound, the support report and the bit-identity of a non-binding declaration.
M-a1 and M-a2 are deliberate mutations, applied by monkeypatch, that each witness must catch.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pytest

import cleverly.interventions.policy as policy_module
from cleverly.estimators import TMLE
from cleverly.exceptions import DataError
from cleverly.interventions import ModifiedPolicy, Piece, Shift, Support
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.validation.refute import refute, refute_refusal

from .. import dose_declarations as law

INF = np.inf
NATURAL = Shift(0.0, cap=None)
#: ``a + 0.5`` held at 1.5: on a row with ``W = 0`` it assigns up to 1.5, above ``u(h) = 1``.
BINDING = Shift(0.5, cap=1.5, name="binding")


def _point_fit(frame: Any, policies: list[Any], **declared: Any) -> Any:
    learners = {"outcome_learner": law.NeverFit(), "treatment_learner": law.NeverFit()}
    return TMLE(policies=policies, cross_fit=False, **learners).fit(
        frame, outcome="Y", treatment="A", covariates=["W"], **declared
    )


def _panel_fit(frame: Any, regimens: dict[str, Any], support: Any) -> Any:
    learners = {
        "outcome_learner": law.NeverFit(),
        "pseudo_learner": law.NeverFit(),
        "treatment_learner": law.NeverFit(),
    }
    return LTMLE(regimens, reference="natural", n_folds=1, **learners).fit(
        frame, continuous_treatment={"A1": None, "A2": support}, **law.PANEL
    )


def _outside(frame: Any, column: str, cap: float, delta: float) -> int:
    """Rows whose assigned dose ``a + delta`` (held at ``cap``) exceeds ``u(h)``."""
    a = frame[column].to_numpy()
    assigned = np.where(a + delta <= cap, a + delta, a)
    return int(np.sum(assigned > law.upper(frame)))


def _observed_range_excursions(frame: Any, column: str, cap: float, delta: float) -> int:
    """What the observed-range check (``_warn_outside_support``) counts: doses above the max."""
    a = frame[column].to_numpy()
    assigned = np.where(a + delta <= cap, a + delta, a)
    return int(np.sum(assigned > a.max()))


# ------------------------------------------------------------------ T-a1, T-a2


def test_t_a1_a_policy_outside_the_declared_support_is_refused_before_any_learner() -> None:
    frame = law.point_frame()
    count = _outside(frame, "A", 1.5, 0.5)
    assert count > 0
    assert count != _observed_range_excursions(frame, "A", 1.5, 0.5)
    first = int(np.flatnonzero(_assigned(frame, "A", 1.5, 0.5) > law.upper(frame))[0])
    dose = float(_assigned(frame, "A", 1.5, 0.5)[first])
    expected = (
        "regimen 'binding' is not identified under the declared support: at node 'A' the "
        f"policy 'binding' assigns {count} row(s) at risk a dose outside the declared support "
        f"(first: dose {dose:g} against [0, 1]). Díaz et al. (2023), Assumption 1, needs the "
        "assigned dose inside the support of A_t given the history. Cap the policy at the "
        "support boundary, for example Shift(delta, cap=...) or a Piece bound that reads the "
        "frame, or remove the regimen"
    )
    with pytest.raises(DataError, match=re.escape(expected) + "$"):
        _point_fit(frame, [NATURAL, BINDING], treatment_support=Support(0.0, law.upper))


def test_t_a1_the_longitudinal_refusal_names_the_regimen_and_the_node() -> None:
    frame = law.panel_frame()
    count = _outside(frame, "A2", 1.5, 0.5)
    assert count > 0
    assigned = _assigned(frame, "A2", 1.5, 0.5)
    first = int(np.flatnonzero(assigned > law.upper(frame))[0])
    expected = (
        "regimen 'up at 2' is not identified under the declared support: at node 'A2' the "
        f"policy 'binding' assigns {count} row(s) at risk a dose outside the declared support "
        f"(first: dose {float(assigned[first]):g} against [0, 1])."
    )
    regimens = {"natural": NATURAL, "up at 2": DynamicRegimen("up at 2", (NATURAL, BINDING))}
    with pytest.raises(DataError, match=re.escape(expected)):
        _panel_fit(frame, regimens, Support(0.0, law.upper))


def _assigned(frame: Any, column: str, cap: float, delta: float) -> np.ndarray:
    a = frame[column].to_numpy()
    return np.where(a + delta <= cap, a + delta, a)


def test_t_a2_a_declared_support_that_excludes_an_observed_dose_is_refused() -> None:
    frame = law.point_frame()
    a = frame["A"].to_numpy()
    narrow = Support(0.0, 0.9)
    count = int(np.sum(a > 0.9))
    first = int(np.flatnonzero(a > 0.9)[0])
    expected = (
        f"the declared support of node 'A' excludes {count} observed dose(s) (first: "
        f"{a[first]:g} against [0, 0.9]). The declaration contradicts the data; widen it or "
        "correct the dose"
    )
    with pytest.raises(DataError, match=re.escape(expected) + "$"):
        _point_fit(frame, [NATURAL, BINDING], treatment_support=narrow)
    with pytest.raises(DataError, match="the declared support of node 'A2' excludes "):
        _panel_fit(
            law.panel_frame(),
            {"natural": NATURAL, "up at 2": DynamicRegimen("up at 2", (NATURAL, NATURAL))},
            narrow,
        )


# ------------------------------------------------------------------ T-a3 and M-a1

#: ``a + 0.5`` below ``u(h) - 0.5``, the identity above: inside ``u(h)`` on every row.
INSIDE = ModifiedPolicy(
    "inside",
    pieces=(
        Piece(
            -INF,
            lambda h: law.upper(h) - 0.5,
            lambda a, h: a + 0.5,
            lambda b, h: b - 0.5,
            lambda b, h: 1.0,
        ),
        Piece(lambda h: law.upper(h) - 0.5, INF),
    ),
    policy_kind="known",
)
#: The same map with the boundary at ``max_h u(h) - 0.5 = 1.5``, which reads no history: it
#: stays inside ``max_h u(h)`` but leaves ``u(h)`` on the rows with ``W = 0``.
UNIFORM = ModifiedPolicy(
    "uniform",
    pieces=(
        Piece(-INF, 1.5, lambda a, h: a + 0.5, lambda b, h: b - 0.5, lambda b, h: 1.0),
        Piece(1.5, INF),
    ),
    policy_kind="known",
)


def _t_a3() -> tuple[bool, bool]:
    """Whether ``INSIDE`` passes the checks and ``UNIFORM`` is refused by R-a1."""
    frame = law.point_frame()
    support = {"treatment_support": Support(0.0, law.upper)}
    with pytest.raises(AssertionError, match="fitted before the refusal"):
        _point_fit(frame, [NATURAL, INSIDE], **support)
    inside_passes = True
    try:
        _point_fit(frame, [NATURAL, UNIFORM], **support)
    except DataError as error:
        uniform_refused = "not identified under the declared support" in str(error)
    except AssertionError:
        uniform_refused = False
    return inside_passes, uniform_refused


def test_t_a3_an_h_dependent_support_is_read_on_every_row() -> None:
    assert law.point_frame()["W"].iloc[0] == 1.0
    assert _t_a3() == (True, True)


def test_m_a1_a_support_read_at_the_first_row_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    original = Support.bounds

    def first_row(self: Support, frame: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = original(self, frame, n)
        return np.full(n, lower[0]), np.full(n, upper[0])

    monkeypatch.setattr(Support, "bounds", first_row)
    assert _t_a3() != (True, True)


# ------------------------------------------------------------------ M-a2


def test_m_a2_a_check_of_the_observed_dose_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the check reading the observed dose, T-a1's policy is no longer refused."""
    original = policy_module._assigned_outside_support

    def observed(policy: Any, a: Any, frame: Any, rows: Any, support: Any) -> Any:
        return original(NATURAL, a, frame, rows, support)

    monkeypatch.setattr(policy_module, "_assigned_outside_support", observed)
    with pytest.raises(AssertionError, match="fitted before the refusal"):
        _point_fit(law.point_frame(), [NATURAL, BINDING], treatment_support=Support(0.0, law.upper))


# ------------------------------------------------------------------ T-a4, T-a5


def _fit(frame: Any, ratio: str, **declared: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TMLE(policies=[NATURAL, INSIDE], ratio=ratio, **law.point_learners()).fit(
            frame, outcome="Y", treatment="A", covariates=["W"], **declared
        )


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_a4_the_support_report_after_a_passing_check(ratio: str) -> None:
    result = _fit(law.point_frame(), ratio, treatment_support=Support(0.0, law.upper))
    report = result.single().diagnostics.support()["inside"]
    assert report.outside_declared_support == 0
    if ratio == "density":
        assert isinstance(report.unsupported, int)
    else:
        assert report.unsupported is None
    facts = report.summary().facts
    assert facts["Rows sent outside the declared support"] == "0"
    zero = f"{report.unsupported:,}" if ratio == "density" else "not measured"
    assert facts["Rows sent to zero density"] == zero
    undeclared = _fit(law.point_frame(), ratio).single().diagnostics.support()["inside"]
    assert undeclared.outside_declared_support is None


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_a5_a_non_binding_declaration_leaves_the_estimate_bit_identical(ratio: str) -> None:
    frame = law.point_frame()
    plain = _fit(frame, ratio).single()["ey_policy[inside]"]
    declared = _fit(frame, ratio, treatment_support=Support(-10.0, 10.0)).single()
    assert declared["ey_policy[inside]"].psi == plain.psi
    assert np.array_equal(declared["ey_policy[inside]"].influence_curve, plain.influence_curve)


def test_t_a5_the_longitudinal_estimate_is_bit_identical() -> None:
    frame = law.panel_frame()
    regimens = {"natural": NATURAL, "up at 2": DynamicRegimen("up at 2", (NATURAL, INSIDE))}

    def fit(columns: Any) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LTMLE(regimens, reference="natural", **law.panel_learners()).fit(
                frame, continuous_treatment=columns, **law.PANEL
            )

    plain = fit(["A1", "A2"])
    declared = fit({"A1": Support(-10.0, 10.0), "A2": Support(0.0, law.upper)})
    for name in plain.estimates:
        assert declared[name].psi == plain[name].psi, name
        assert np.array_equal(declared[name].influence_curve, plain[name].influence_curve)


# ------------------------------------------------------------------ declarations


def test_a_support_refuses_nan_and_an_empty_interval() -> None:
    with pytest.raises(DataError, match="must not be NaN"):
        Support(float("nan"), 1.0)
    with pytest.raises(DataError, match="needs lower < upper"):
        Support(1.0, 1.0)


def test_a_support_without_a_policy_is_refused() -> None:
    with pytest.raises(DataError):
        TMLE(**law.point_learners()).fit(
            law.point_frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            treatment_support=Support(0.0, 2.0),
        )


def test_a_history_dependent_support_takes_the_band_rank_point_placebo() -> None:
    """X82 part J: the placebo moves each dose's position within its own band (was #428's
    refusal), so every placebo dose meets the declared support the refit checks."""
    contrast = "ate_policy[inside vs natural course]"
    frame = law.point_frame()
    declared = _fit(frame, "density", treatment_support=Support(0.0, law.upper)).single()
    assert refute_refusal(declared, estimand=contrast, tests=("placebo",)) is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = refute(declared, estimand=contrast, tests=("placebo",), n_replicates=3)
    assert len(report["placebo"].values) == 3
    assert "keeps each unit's declared band" in report["placebo"].detail
    fixed = _fit(frame, "density", treatment_support=Support(-10.0, 10.0)).single()
    assert refute_refusal(fixed, estimand=contrast, tests=("placebo",)) is None
    plain = _fit(frame, "density").single()
    assert refute_refusal(plain, estimand=contrast, tests=("placebo",)) is None
