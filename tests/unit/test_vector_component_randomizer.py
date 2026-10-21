"""A randomizer on a component and a risk-ratio tilt of a binary component, at a vector.

A randomizer on a component's :class:`~cleverly.interventions.ModifiedPolicy` draws
independently of the other components and of the data.  The joint law of the draws is then
the product law, a finite partition, so the declaration is the joint randomizer over the
product of the components' draws with product probabilities.  A
:class:`~cleverly.interventions.RiskRatioTilt` of a binary component is ``lmtp``'s ``ipsi()``:
the two-point randomizer that keeps the label with probability ``delta`` (or ``1 / delta``)
and otherwise sets it to 0 (or to 1).  It enters the product the same way.

The oracles are the point mixed law of ``tests/discrete_law_policy_point.py`` and the
two-node mixed law of ``tests/discrete_law_longitudinal_mtp.py``, whose tilt plans are
written as branches with their probabilities and share no code with ``src/``.  Every exact
fit equals the g-formula and its complex-step Gateaux derivative, on both ratio routes and
on the all-categorical reading.  The mutation controls swap the keep and set probabilities,
drop the product weight, and swap the label the tilt sets.  Each one moves a witness by more
than ``CAUGHT``; the measured deviations are in each docstring.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import tests.discrete_law_longitudinal_mtp as long_law
import tests.discrete_law_policy_point as law
from cleverly.exceptions import DataError
from cleverly.interventions import JointPolicy, ModifiedPolicy, Piece, Randomizer, RiskRatioTilt
from cleverly.interventions import policy as policy_module
from cleverly.longitudinal import DynamicRegimen
from tests.discrete_law_longitudinal import CellMeans
from tests.discrete_law_longitudinal_multivalue import CellProbabilities

from .. import longitudinal_mtp as mtp
from . import test_longitudinal_mixed_node as long_tests
from . import test_point_vector_treatment as point_tests
from ._source_mutation import mutate

#: The mixed laws put every dose on a grid, so R-b4 would refuse each fit; the
#: fixture turns that refusal off and leaves every other check on (X40 step H).
pytestmark = [
    pytest.mark.xdist_group("vector_component_randomizer"),
    pytest.mark.usefixtures("exact_grid_law"),
]

EXACT = 1e-10
CAUGHT = 1e-4
INF = np.inf
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)
TILTS = ("flag tilt down", "flag tilt up", "flag tilt up, random dose")


# ------------------------------------------------------------------ declarations
#
# Module-level functions, so a fitted result pickles.


def dose_up(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + 1.0


def dose_down(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - 1.0


def unit_slope(b: Any, h: Any) -> np.ndarray:
    return np.ones(len(b))


def label_up_or_stay(a: Any, h: Any, value: object) -> np.ndarray:
    """The label map of the randomized dose: up to the top at ``"up"``, unchanged at ``"stay"``."""
    dose = np.asarray(a, dtype=float)
    return np.minimum(dose + 1.0, 2.0) if value == "up" else dose


def random_dose(top: float = 2.0) -> ModifiedPolicy:
    """The dose randomizer of the ``"random"`` plan, declared on the component.

    ``"up"`` raises the dose by one below ``top - 0.5`` and holds the top; ``"stay"`` is the
    identity.  The probabilities are those of the joint randomizer of ``"random"``.
    """
    return ModifiedPolicy(
        "random dose",
        pieces={
            "up": (Piece(-INF, top - 0.5, dose_up, dose_down, unit_slope), Piece(top - 0.5, INF)),
            "stay": (Piece(-INF, INF),),
        },
        randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
        policy_kind="known",
    )


def random_label() -> ModifiedPolicy:
    """The dose randomizer as a label map, for the categorical reading.

    A continuous dose refuses ``apply=``, so the ``(a2)`` reading takes :func:`random_dose`.
    """
    return ModifiedPolicy(
        "random label",
        apply=label_up_or_stay,
        randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
        policy_kind="known",
    )


def point_policy(label: str, *, dose: Any = None) -> JointPolicy:
    """A point declaration, by its oracle label; ``dose`` overrides the dose randomizer."""
    randomized = random_dose() if dose is None else dose
    if label == "random":
        return JointPolicy(label, components={"D": randomized}, policy_kind="known")
    if label == "flag tilt down":
        return JointPolicy(label, components={"F": RiskRatioTilt(0.25)}, policy_kind="known")
    if label == "flag tilt up":
        return JointPolicy(label, components={"F": RiskRatioTilt(4.0)}, policy_kind="known")
    if label == "flag tilt up, random dose":
        return JointPolicy(
            label, components={"F": RiskRatioTilt(4.0), "D": randomized}, policy_kind="known"
        )
    raise KeyError(label)


def through_the_joint_randomizer() -> JointPolicy:
    """``"flag tilt up, random dose"`` with the dose draw on the joint policy instead."""
    return JointPolicy(
        "flag tilt up, random dose",
        components={
            "up": {"F": RiskRatioTilt(4.0), "D": point_tests.UP},
            "stay": {"F": RiskRatioTilt(4.0)},
        },
        randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
        policy_kind="known",
    )


LABELS = ("random", *TILTS)


def point_fit(ratio: str, policies: list[object], **overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "ratio": ratio,
        "treatment_learner": CellProbabilities() if ratio == "density" else CellMeans(),
        "g_bounds": NO_TRUNCATION,
        **overrides,
    }
    with mtp.exact_bins():
        return point_tests.fit(
            tuple(policy.name for policy in policies),  # type: ignore[attr-defined]
            policies=policies,
            **settings,
        )


def point_error(result: Any, labels: tuple[str, ...] = LABELS) -> float:
    return point_tests.deviation(result, labels)


def long_plans(**dose: Any) -> dict[str, Any]:
    plans: dict[str, Any] = {"natural": mtp.mixed_regimens(("natural",))["natural"]}
    nodes = []
    for time in (1, 2):
        randomized = random_dose() if not dose else dose[f"D{time}"]
        nodes.append(
            JointPolicy(
                "flag tilt up, random dose",
                components={f"F{time}": RiskRatioTilt(4.0), f"D{time}": randomized},
                policy_kind="known",
            )
        )
    plans["flag tilt up, random dose"] = DynamicRegimen("flag tilt up, random dose", tuple(nodes))
    return plans


# ------------------------------------------------------------------ witnesses


def test_the_tilt_plans_are_not_degenerate() -> None:
    """Each tilt moves the mean from the natural course, from the other tilts and from a set.

    The dose draw moves the tilt-up mean by 0.014.  ``"random"`` alone sits 0.004 from the
    natural course, so its witness rests on the influence curve as well as the mean.
    """
    truths = {label: law.mixed_truth(label) for label in (*TILTS, "natural")}
    values = list(truths.values())
    assert min(abs(a - b) for i, a in enumerate(values) for b in values[i + 1 :]) > 1e-2
    # A tilt is not the deterministic set: keeping the label with probability 1/4 moves the
    # mean by 0.0078.
    set_flag = law._mean_under(
        *_joint_and_qbar(), ((1.0, law._mixed_map(lambda k: law._index(1, law.MIXED_CELLS[k][1]))),)
    )
    assert abs(truths["flag tilt up"] - float(np.real(set_flag))) > 5e-3


def _joint_and_qbar() -> tuple[np.ndarray, np.ndarray]:
    cells = np.asarray(law.MIXED_PROBS).reshape(2, 6, 2)
    joint = cells[:, :, 0] + cells[:, :, 1]
    return joint, cells[:, :, 1] / joint


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_a_component_randomizer_and_a_tilt_are_exact_at_a_point(ratio: str) -> None:
    """Every plan equals the oracle.  Measured: at most 1e-14 (classifier), 1e-11 (density)."""
    result = point_fit(ratio, [point_policy(label) for label in LABELS])
    assert point_error(result) < EXACT


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_a_component_randomizer_equals_the_joint_randomizer(ratio: str) -> None:
    """Declared on the component or on the joint policy, the plan is one plan."""
    on_component = point_fit(ratio, [point_policy(TILTS[2])])
    on_joint = point_fit(ratio, [through_the_joint_randomizer()])
    name = f"ey_policy[{TILTS[2]}]"
    assert on_component.psi(name) == pytest.approx(on_joint.psi(name), abs=EXACT)
    np.testing.assert_allclose(
        on_component.estimates[name].influence_curve,
        on_joint.estimates[name].influence_curve,
        atol=EXACT,
    )
    assert point_error(on_joint, (TILTS[2],)) < EXACT


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_a_tilt_and_a_component_randomizer_are_exact_on_the_categorical_reading(
    ratio: str,
) -> None:
    """The all-categorical reading takes the same declarations by the discrete formula."""
    policies = [point_policy(label, dose=random_label()) for label in LABELS]
    result = point_tests.fit(
        LABELS,
        data=point_tests.categorical_data(),
        policies=policies,
        ratio=ratio,
        treatment_learner=CellProbabilities() if ratio == "density" else CellMeans(),
        g_bounds=(1e-6, 1.0 - 1e-6),
    )
    assert point_error(result) < EXACT


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_a_tilt_and_a_component_randomizer_are_exact_at_mixed_nodes(ratio: str) -> None:
    """Each node draws its tilt and its dose independently.  Measured: at most 1e-11."""
    labels = ("natural", "flag tilt up, random dose")
    with mtp.exact_bins():
        result = long_tests.fit(labels, regimens=long_plans(), ratio=ratio)
    assert long_tests.deviation(result, labels) < EXACT
    truth = long_law.mixed_truth("flag tilt up, random dose", "end", 2)
    assert abs(truth - long_law.mixed_truth("random", "end", 2)) > 1e-2


# ------------------------------------------------------------------ refusals


def test_a_tilt_on_a_dose_is_refused_with_the_scalar_reason() -> None:
    tilt = JointPolicy("tilt dose", components={"D": RiskRatioTilt(0.5)}, policy_kind="known")
    with pytest.raises(DataError) as caught:
        point_tests.fit(("tilt dose",), policies=[tilt])
    assert "RiskRatioTilt sets a binary treatment coded 0 and 1, but component 'D' is " in str(
        caught.value
    )
    assert "continuous. It is lmtp's ipsi()" in str(caught.value)


def test_a_tilt_on_a_three_level_component_is_refused_with_the_scalar_reason() -> None:
    tilt = JointPolicy("tilt dose", components={"D": RiskRatioTilt(0.5)}, policy_kind="known")
    with pytest.raises(DataError, match=r"but component 'D' has levels \[0\.0, 1\.0, 2\.0\]"):
        point_tests.fit(("tilt dose",), data=point_tests.categorical_data(), policies=[tilt])


# ------------------------------------------------------------------ mutation controls


def _gap() -> float:
    gap = max(
        point_error(point_fit("classifier", [point_policy(label) for label in LABELS])),
        point_error(point_fit("density", [point_policy(label) for label in LABELS])),
    )
    assert gap > CAUGHT
    return gap


def test_mt1_swapped_keep_and_set_probabilities_are_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tilt keeps with ``1 - delta``.  Measured: 0.80 on both routes (curve)."""
    mutate(
        monkeypatch,
        policy_module,
        policy_module,
        "_component_draws",
        [
            (
                "return ((keep, None), (1.0 - keep, constant))",
                "return ((1.0 - keep, None), (keep, constant))",
            )
        ],
    )
    _gap()


def test_mt2_a_dropped_product_weight_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each branch keeps the joint probability only.  Measured: 5.8 on both routes (curve)."""
    mutate(
        monkeypatch,
        policy_module,
        policy_module,
        "joint_branches",
        [("                weight *= p", "                weight *= 1.0")],
    )
    _gap()


def test_mt3_a_swapped_label_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tilt down sets the flag to 1, the tilt up to 0.  Measured: 2.0 on both routes (curve)."""
    mutate(
        monkeypatch,
        policy_module,
        policy_module,
        "_component_draws",
        [
            (
                "(delta, 0.0) if delta < 1.0 else (1.0 / delta, 1.0)",
                "(delta, 1.0) if delta < 1.0 else (1.0 / delta, 0.0)",
            )
        ],
    )
    _gap()


def test_the_unmutated_controls_are_exact() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert (
            point_error(point_fit("classifier", [point_policy(label) for label in LABELS])) < EXACT
        )
