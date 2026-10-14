"""``classifier_ratio`` on an ``(n, d)`` treatment block, and the ``JointPolicy`` declaration.

W6: a scalar call is bit-identical to the same call on an ``(n, 1)`` block, and its stacked
design is the ``[dose, history]`` matrix the scalar route has always built.  W7: on a
two-column block the folds still split units, not stacked rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import (
    JointPolicy,
    Randomizer,
    Shift,
)
from cleverly.interventions.policy import joint_branches, policy_branches
from cleverly.learners import density_ratio
from cleverly.learners.crossfit import Folds, make_folds


def _inputs(seed: int, n: int = 120) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    history = rng.normal(size=(n, 2))
    dose = rng.normal(size=n) + history[:, 0]
    return {
        "history": history,
        "dose": dose,
        "branches": ((0.3, dose + 0.5), (0.7, dose * 0.8)),
        "weights": rng.uniform(0.5, 1.5, size=n),
        "mask": rng.uniform(size=n) > 0.1,
        "groups": np.repeat(np.arange(n // 2), 2),
    }


def _call(inputs: dict[str, Any], folds: Folds, column: bool) -> list[np.ndarray]:
    def shape(values: np.ndarray) -> np.ndarray:
        return values.reshape(-1, 1) if column else values

    return density_ratio.classifier_ratio(
        LogisticRegression(max_iter=1000),
        inputs["history"],
        shape(inputs["dose"]),
        tuple((p, shape(d)) for p, d in inputs["branches"]),
        inputs["weights"],
        folds,
        evaluate_at=[shape(inputs["dose"]), shape(inputs["branches"][0][1])],
        fit_mask=inputs["mask"],
        groups=inputs["groups"],
    )


@pytest.mark.parametrize("n_folds", [1, 3])
def test_w6_a_scalar_call_equals_its_one_column_block_bit_for_bit(n_folds: int) -> None:
    inputs = _inputs(3)
    n = len(inputs["dose"])
    folds = Folds.single(n) if n_folds == 1 else make_folds(n, n_folds, random_state=2)
    scalar = _call(inputs, folds, column=False)
    block = _call(inputs, folds, column=True)
    for left, right in zip(scalar, block, strict=True):
        assert np.array_equal(left, right)
        assert left.shape == (n,)


def test_w6_the_scalar_design_is_the_dose_beside_the_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stacked design of a scalar call, row for row, as the scalar route built it."""
    inputs = _inputs(5)
    n = len(inputs["dose"])
    seen: list[np.ndarray] = []
    original = density_ratio.fit_learner

    def record(learner: Any, design: Any, target: Any, weight: Any, **kwargs: Any) -> Any:
        seen.append(np.asarray(design))
        return original(learner, design, target, weight, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", record)
    _call(inputs, Folds.single(n), column=False)
    keep = inputs["mask"]
    h = inputs["history"][keep]
    blocks = [inputs["dose"], *(d for _, d in inputs["branches"])]
    expected = np.vstack([np.column_stack([dose[keep], h]) for dose in blocks])
    (design,) = seen
    assert np.array_equal(design, expected)


def test_w7_the_folds_split_units_on_a_two_column_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit's stacked copies train together, and a held-out unit trains in none."""
    inputs = _inputs(7)
    n = len(inputs["dose"])
    flag = (inputs["history"][:, 1] > 0).astype(float)
    observed = np.column_stack([flag, inputs["dose"]])
    assigned = np.column_stack([np.ones(n), inputs["dose"] + 0.5])
    folds = make_folds(n, 3, random_state=1)
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    original = density_ratio.fit_learner

    def record(learner: Any, design: Any, target: Any, weight: Any, **kwargs: Any) -> Any:
        seen.append((np.asarray(design), np.asarray(kwargs["groups"])))
        return original(learner, design, target, weight, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", record)
    density_ratio.classifier_ratio(
        LogisticRegression(max_iter=1000),
        inputs["history"],
        observed,
        ((1.0, assigned),),
        inputs["weights"],
        folds,
        evaluate_at=[observed],
    )
    assert len(seen) == 3
    for (design, groups), (train, test) in zip(seen, folds, strict=True):
        assert design.shape[1] == 2 + inputs["history"].shape[1]
        units, counts = np.unique(groups, return_counts=True)
        assert set(units.tolist()) == set(np.asarray(train).tolist())
        assert np.all(counts == 2)
        assert not set(units.tolist()) & set(np.asarray(test).tolist())


def test_a_block_of_another_width_is_refused() -> None:
    inputs = _inputs(9)
    n = len(inputs["dose"])
    observed = np.column_stack([inputs["dose"], inputs["dose"]])
    with pytest.raises(DataError, match="every block of the stack holds the same columns"):
        density_ratio.classifier_ratio(
            LogisticRegression(),
            inputs["history"],
            observed,
            ((1.0, inputs["dose"]),),
            inputs["weights"],
            Folds.single(n),
            evaluate_at=[observed],
        )
    with pytest.raises(DataError, match=r"reads \(n,\) or \(n, d\)"):
        density_ratio.classifier_ratio(
            LogisticRegression(),
            inputs["history"],
            inputs["dose"][:-1],
            ((1.0, inputs["dose"]),),
            inputs["weights"],
            Folds.single(n),
            evaluate_at=[inputs["dose"]],
        )


# ------------------------------------------------------------------ JointPolicy


def test_a_joint_policy_holds_its_components_and_its_randomizer_branches() -> None:
    up = Shift(1.0, cap=2.0)
    plain = JointPolicy("up", components={"D1": up}, policy_kind="known")
    assert joint_branches(plain) == ((1.0, {"D1": up}),)
    drawn = JointPolicy(
        "random",
        components={"up": {"D1": up}, "stay": {}},
        randomizer=Randomizer(("up", "stay"), (0.25, 0.75)),
        policy_kind="known",
    )
    assert joint_branches(drawn) == ((0.25, {"D1": up}), (0.75, {}))


def test_a_joint_policy_refuses_what_a_component_cannot_hold() -> None:
    """A tilt and a randomized component are refused in ``test_longitudinal_mixed_node``."""
    up = Shift(1.0, cap=2.0)
    with pytest.raises(DataError, match="holds a joint policy"):
        JointPolicy("j", components={"D1": JointPolicy("i", components={})}, policy_kind="known")
    with pytest.raises(DataError, match="A component takes"):
        JointPolicy("j", components={"D1": 1.0}, policy_kind="known")
    with pytest.raises(DataError, match="maps each of its values"):
        JointPolicy(
            "j",
            components={"up": {"D1": up}},
            randomizer=Randomizer(("up", "stay"), (0.5, 0.5)),
            policy_kind="known",
        )


def test_a_joint_policy_has_no_scalar_branches() -> None:
    joint = JointPolicy("j", components={"D1": Shift(1.0, cap=2.0)}, policy_kind="known")
    with pytest.raises(CapabilityError, match="applies at a vector node of a longitudinal fit"):
        policy_branches(joint)
