"""Mutation controls and the targeting witness of the longitudinal mixed node.

Each control breaks one line of the shipped source through ``monkeypatch`` and asks the exact
witness of :mod:`tests.unit.test_longitudinal_mixed_node` to move by more than ``CAUGHT``.
An exact-law check is blind to a term that vanishes at the truth, so each control names the
component it guards.  The measured deviations at the time of writing are in each docstring;
they are far above ``CAUGHT``, and the controls assert only the bound.

W8 is the targeting witness on a sample, where the outcome regression is misspecified and the
coefficients are nonzero.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from cleverly.interventions import policy as policy_module
from cleverly.longitudinal import LTMLE, sequential
from cleverly.longitudinal import data as data_module
from cleverly.longitudinal.data import LongitudinalData

from .. import discrete_law_longitudinal_mtp as law
from .. import longitudinal_mtp as mtp
from ._source_mutation import mutate
from .test_longitudinal_mixed_node import COLUMNS, EXACT, LABELS, deviation, fit, settings

pytestmark = pytest.mark.xdist_group("longitudinal_mixed_node_mutations")

CAUGHT = 1e-4

#: The classifier's observed block and each branch's assigned block.
OBSERVED = (
    "observed = data.component_design(time)\n",
    "observed = data.component_design(time){}\n",
)
ASSIGNED = ("data.component_design(time, target))", "data.component_design(time, target){})")


def _both(cut: str) -> list[tuple[str, str]]:
    """Cut the same columns from the observed and the assigned block of the classifier."""
    return [(old, new.format(cut)) for old, new in (OBSERVED, ASSIGNED)]


def _caught(design: str = "end") -> float:
    """The exact witness under the mutation in force, which must fail."""
    result = fit(LABELS, design)
    gap = deviation(result, LABELS, design)
    assert gap > CAUGHT
    return gap


def test_the_unmutated_witness_is_exact() -> None:
    """The baseline each control departs from."""
    for design in ("end", "censored"):
        assert deviation(fit(LABELS, design), LABELS, design) < EXACT


def test_m1_a_classifier_without_the_flag_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stacked classifier reads the dose only.  Measured deviation: 8.45."""
    mutate(monkeypatch, sequential, sequential, "_mixed_node_ratios", _both("[:, 1:]"))
    _caught()


def test_m2_a_classifier_without_the_dose_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stacked classifier reads the flag only.  Measured deviation: 8.45."""
    mutate(monkeypatch, sequential, sequential, "_mixed_node_ratios", _both("[:, :1]"))
    _caught()


def test_m3_a_dose_map_that_reads_the_assigned_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dose map reads the flag the policy assigns.  Measured deviation: 9.43.

    W4 is the reason this is visible: the flag-gated plan moves the flag, and its
    shifted-flag reading differs from the natural-flag one by 0.028 in the parameter.
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
    assert result.psi("ey_regimen[flag gated]") == pytest.approx(shifted, abs=EXACT)


def test_m5_a_history_without_the_earlier_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node-2 mechanism history drops the node-1 flag.  Measured deviation: 3.00."""
    edit = (
        "columns.append(self.component_design(t))",
        "columns.append(self.component_design(t)[:, 1:] if t < time else self.component_design(t))",
    )
    mutate(monkeypatch, data_module, LongitudinalData, "history_design", [edit])
    _caught()


def test_m6_a_kind_read_off_the_first_component_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kind reads the first component only, so the flag-first nodes read as categorical.

    On the grid law the categorical reading of the same tuples is itself exact, so the
    estimate cannot catch this mutation.  The witness is the node kind, which W3 asserts.
    """
    edit = (
        "declared = [name in continuous for name in block]",
        "declared = [block[0] in continuous]",
    )
    mutate(monkeypatch, data_module, data_module, "_node_kind_of", [edit])
    result = fit(("natural", "dose up"))
    assert [result.data.node_kind(time) for time in (1, 2)] != ["mixed", "mixed"]


def test_m7_a_classifier_label_block_with_the_natural_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The label-1 block keeps the natural flag.  Measured deviation: 7.66."""
    natural = (
        "data.component_design(time, np.column_stack([data.node_values(time)[:, 0], "
        "target[:, 1]])))"
    )
    mutate(monkeypatch, sequential, sequential, "_mixed_node_ratios", [(ASSIGNED[0], natural)])
    _caught()


def test_m7_an_outcome_prediction_with_the_natural_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prediction at a branch keeps the natural flag.  Measured deviation: 2.11."""
    old = "targets[int(arm)]  # type: ignore[arg-type]\n                    if current"
    new = (
        "np.column_stack([data.node_values(node)[:, 0], targets[int(arm)][:, 1]])\n"
        "                    if current"
    )
    mutate(monkeypatch, sequential, sequential, "outcome_design", [(old, new)])
    _caught()


def test_m8_a_censoring_design_without_the_dose_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The censoring model drops the current dose.  Measured deviation: 4.10 (censored).

    The end-of-study reading fits no censoring model, so only the censored reading sees it.
    """
    edit = (
        "columns.append(self.component_design(t))",
        "columns.append(self.component_design(t)[:, :1] if t == time else "
        "self.component_design(t))",
    )
    mutate(monkeypatch, data_module, LongitudinalData, "history_design", [edit])
    _caught("censored")


def test_m9_an_outcome_design_without_the_earlier_flag_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node-2 outcome regression drops the node-1 flag.  Measured deviation: 3.32."""
    old = "            continue\n        if data.is_continuous_node(node):"
    new = (
        "            if node < time:\n"
        "                blocks[-1] = blocks[-1][:, 1:]\n"
        "            continue\n"
        "        if data.is_continuous_node(node):"
    )
    mutate(monkeypatch, sequential, sequential, "outcome_design", [(old, new)])
    _caught()


# ------------------------------------------------------------------ W8


def _misspecified(folds: int) -> Any:
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LogisticRegression

    from ..studies.canonical_ltmle import QuasiBinomialGLM

    labels = ("natural", "dose up", "flag gated", "random")
    configuration = settings(
        g_bounds=(0.01, 1.0),
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=DummyRegressor(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=folds,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(mtp.mixed_regimens(labels), reference="natural", **configuration).fit(
            law.mixed_sample(900, 11), **COLUMNS
        )
    return result, labels[1:]


@pytest.mark.parametrize("folds", [1, 5])
def test_w8_a_misspecified_fit_targets_and_solves_its_score(folds: int) -> None:
    """Nonzero coefficients at both mixed nodes, and ``P_n D* = 0`` to ``1e-10``.

    The treatment learner is logistic.  A saturated classifier fitted out of fold on 900
    rows meets cells that hold the assigned block only, and its ratio is then the odds
    guard, which is a learner failure and not a property of the node.
    """
    result, labels = _misspecified(folds)
    for label in labels:
        steps = result.fits[label].steps
        assert all(abs(float(step.fluctuation.epsilon[0])) > 1e-3 for step in steps), label
        assert all(step.fluctuation.converged for step in steps), label
        curve = result.influence_curves[f"ey_regimen[{label}]"]
        assert abs(float(np.mean(curve))) < 1e-10, label
