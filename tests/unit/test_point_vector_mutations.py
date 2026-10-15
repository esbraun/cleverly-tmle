"""Mutation controls of the point vector treatment with a dose component.

Each control breaks one line of the shipped source through ``monkeypatch`` and asks an exact
witness of :mod:`tests.unit.test_point_vector_treatment` to move by more than ``CAUGHT``.
Two witnesses answer:

- W1, the policy means and influence curves of the saturated fit on the mixed law;
- W8, the targeted estimates on the same law with a misspecified outcome regression.  The
  ratio is exact there, so the targeted estimate is the truth only when the clever
  covariate and the counterfactual outcome predictions read both components correctly.

The measured deviations at the time of writing are in each docstring.  They are far above
``CAUGHT``, and the controls assert only the bound.
"""

from __future__ import annotations

import importlib

import pytest

import tests.discrete_law_policy_point as law
from cleverly.estimators import _nuisance
from cleverly.interventions import PolicySet
from cleverly.interventions import policy as policy_module

from ._source_mutation import mutate
from .test_point_vector_treatment import (
    EXACT,
    LABELS,
    deviation,
    fit,
    misspecified_settings,
)

pytestmark = pytest.mark.xdist_group("point_vector_mutations")

CAUGHT = 1e-4
#: The module, which the function ``cleverly.estimators.tmle`` shadows as an attribute.
tmle_module = importlib.import_module("cleverly.estimators.tmle")

#: The classifier's observed block and each branch's assigned block.
OBSERVED = ("observed = data.component_design()\n", "observed = data.component_design(){}\n")
ASSIGNED = (
    "data.component_design(shifted[:, c]) for c in range(count)]",
    "data.component_design(shifted[:, c]){} for c in range(count)]",
)
#: A branch's assigned values with the flag put back to its natural value.
NATURAL_FLAG = "data.component_design(np.column_stack([natural[:, 0], shifted[:, {}, 1]]))"


def _both(cut: str) -> list[tuple[str, str]]:
    """Cut the same columns from the observed and the assigned block of the classifier."""
    return [(old, new.format(cut)) for old, new in (OBSERVED, ASSIGNED)]


def _targeted_error() -> float:
    """The largest error of W8's targeted estimates on the exact law."""
    result = fit(**misspecified_settings())
    return max(abs(result.psi(f"ey_policy[{label}]") - law.mixed_truth(label)) for label in LABELS)


def _caught() -> float:
    """W1 and W8 under the mutation in force: one of them must fail."""
    gap = max(deviation(fit(), LABELS), _targeted_error())
    assert gap > CAUGHT
    return gap


def test_the_unmutated_witnesses_are_exact() -> None:
    """The baseline each control departs from."""
    assert deviation(fit(), LABELS) < EXACT
    assert _targeted_error() < EXACT


def test_m1_a_classifier_without_the_flag_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stacked classifier reads the dose only.  Measured: W1 1.50, W8 0.038."""
    mutate(monkeypatch, policy_module, PolicySet, "_evaluate_vector", _both("[:, 1:]"))
    _caught()


def test_m2_a_classifier_without_the_dose_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stacked classifier reads the flag only.  Measured: W1 1.50, W8 0.051."""
    mutate(monkeypatch, policy_module, PolicySet, "_evaluate_vector", _both("[:, :1]"))
    _caught()


def test_m3_a_dose_map_that_reads_the_assigned_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dose map reads the flag the policy assigns.  Measured: W1 1.05, W8 0.047.

    W4 is the reason this is visible: the flag-gated policy moves the flag, and its
    shifted-flag reading differs from the natural-flag one by 0.047 in the parameter.
    """
    natural = "other.name: labels[other.name] for other in components if not other.continuous"
    assigned = (
        "other.name: other.labels(assigned[:, names.index(other.name)]) "
        "for other in components if not other.continuous"
    )
    mutate(monkeypatch, policy_module, policy_module, "_joint_branch", [(natural, assigned)])
    _caught()
    result = fit(("natural", "flag gated"))
    shifted = law.mixed_truth("flag gated (shifted flag)")
    assert result.psi("ey_policy[flag gated]") == pytest.approx(shifted, abs=EXACT)


def test_m4_an_outcome_prediction_with_the_natural_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counterfactual outcome design keeps the natural flag.  Measured: W1 0.47, W8 0.038."""
    old = "code: policy_set.shifted[:, index]\n"
    new = (
        "code: np.column_stack([data.treatment_components[:, 0], "
        "policy_set.shifted[:, index][:, 1]])\n"
    )
    mutate(monkeypatch, _nuisance, _nuisance, "fit_nuisances", [(old, new)])
    # The estimator holds its own reference to the function.
    monkeypatch.setattr(tmle_module, "fit_nuisances", _nuisance.fit_nuisances)
    _caught()


def test_m7_a_classifier_label_block_with_the_natural_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The label-1 block keeps the natural flag.  Measured: W1 1.00, W8 0.0029."""
    edit = ("((1.0, assigned[c]),)", f"((1.0, {NATURAL_FLAG.format('c')}),)")
    mutate(monkeypatch, policy_module, PolicySet, "_evaluate_vector", [edit])
    _caught()


def test_m7_a_ratio_evaluated_at_the_natural_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ratio at each branch is read with the natural flag.  Measured: W1 7e-16, W8 0.0059.

    ``ratio_at`` enters the targeting step only, so W1, whose outcome regression is already
    exact, cannot see it.  W8 is the witness.
    """
    edit = (
        "evaluate_at=[observed, *assigned],",
        f"evaluate_at=[observed, *({NATURAL_FLAG.format('s')} for s in range(count))],",
    )
    mutate(monkeypatch, policy_module, PolicySet, "_evaluate_vector", [edit])
    _caught()
