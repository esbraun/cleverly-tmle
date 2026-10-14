"""Mutation controls of the density route at a vector treatment with a continuous component.

Each control breaks the shipped source through ``monkeypatch`` (``_source_mutation.mutate``)
and asks an exact witness of :mod:`tests.unit.test_vector_density_route` to move by more
than ``CAUGHT``.  The witnesses are chosen so the broken term does not vanish at the truth
(plan review rev 3, H3):

- the geometric law's "ordered" plan moves dose 1, and dose 2's law reads dose 1, so dose
  2's frame, conditioning design and Jacobian all read the preimage of dose 1 (M10, M12,
  M13, M15);
- the point mixed law's W4 plan reads the natural flag in the dose map, with the flag moved
  (M11, M16); W17 holds the top dose by a cap while the flag moves (M14); W18's flag map
  reads the natural dose, with the dose moved (M17);
- W8-density targets a misspecified outcome regression, so ``ratio_at`` enters (M18);
- the out-of-fold provenance witness of the counterfactual predictions (B1).

The measured deviation at the time of writing is in each docstring: the largest error of an
estimate or of its influence curve against the oracle, or, for B1, of a prediction.
"""

from __future__ import annotations

import numpy as np
import pytest

import tests.discrete_law_policy_point as point_law
from cleverly.interventions import PolicySet, joint_density
from cleverly.interventions import policy as policy_module
from cleverly.learners import density as density_module

from . import test_vector_density_route as route
from ._source_mutation import mutate

pytestmark = pytest.mark.xdist_group("vector_density_mutations")

CAUGHT = 1e-4
_ENUMERATE = (joint_density, joint_density, "_enumerate")


def _geo_ordered() -> float:
    """W15: the "ordered" plan at the point reading of the geometric law."""
    result = route.fit_geo_point(("natural", "ordered"))
    return route.geo_deviation(result, ("natural", "ordered"), longitudinal=False)


def _geo_ordered_long() -> float:
    """W15 at the one-node longitudinal reading of the geometric law."""
    result = route.fit_geo_long(("natural", "ordered"))
    return route.geo_deviation(result, ("natural", "ordered"), longitudinal=True)


def _point(labels: tuple[str, ...]) -> float:
    return route.point_deviation(route.fit_point(labels), labels)


def _targeted_error() -> float:
    """W8-density: the targeted error on the exact point law, outcome misspecified."""
    from sklearn.linear_model import LogisticRegression

    labels = tuple(route.POINT_TRUTH)
    result = route.fit_point(labels, outcome_learner=LogisticRegression())
    return max(
        abs(result.psi(f"ey_policy[{label}]") - point_law.mixed_truth(route.POINT_TRUTH[label]))
        for label in labels
    )


def test_the_witnesses_are_exact_without_a_mutation() -> None:
    assert _geo_ordered() < route.EXACT
    assert _geo_ordered_long() < route.EXACT
    assert _point(("flag gated", "flag set, dose up", "flag reads dose")) < route.EXACT
    assert _targeted_error() < route.EXACT


# ------------------------------------------------------------------ M10, M12, M13, M15


@pytest.mark.parametrize("dose", [0, 1])
def test_m10_a_jacobian_dropped_from_one_dose_is_caught(
    monkeypatch: pytest.MonkeyPatch, dose: int
) -> None:
    """``|d beta_k / d b_k|`` is replaced by one for dose ``k``.  Measured: dose 1 7.9, dose 2 3.9."""
    edit = (
        "jacobian * np.abs(np.where(inside, slope, 0.0)),",
        f"jacobian * (np.abs(np.where(inside, slope, 0.0)) if index != layout.doses[{dose}] "
        "else inside.astype(float)),",
    )
    mutate(monkeypatch, *_ENUMERATE, [edit])
    assert _geo_ordered() > CAUGHT


def test_m12_dose_two_conditioned_on_the_value_instead_of_the_preimage_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dose 2's density reads ``b_1`` in place of ``beta_1``.  Measured: 5.25 on both readings."""
    edit = (
        "position, cell_codes, dose_values, dose_values[:, position], cell=cell",
        "position, cell_codes, "
        "np.where(np.arange(len(doses)) < position, target[:, doses], dose_values), "
        "dose_values[:, position], cell=cell",
    )
    mutate(monkeypatch, *_ENUMERATE, [edit])
    assert _geo_ordered() > CAUGHT
    assert _geo_ordered_long() > CAUGHT


def test_m13_a_dose_map_that_reads_the_assigned_earlier_dose_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dose 2's assign map reads dose 1 after the policy moved it.  Measured: 0.71.

    The ordered-map analogue of M3: the assigned values then follow the shifted-dose-1
    reading, whose truth differs by ``0.039``.
    """
    edit = (
        "readable.update({other: labels[other] for other in earlier})",
        "readable.update({other: assigned[:, names.index(other)] for other in earlier})",
    )
    mutate(monkeypatch, policy_module, policy_module, "_joint_branch", [edit])
    assert _geo_ordered() > CAUGHT
    # The saturated plug-in reads the assigned values, so it lands on the shifted reading.
    result = route.fit_geo_point(("ordered",))
    shifted = route.geo.truth("ordered (shifted dose 1)")
    assert result.psi("ey_policy[ordered]") == pytest.approx(shifted, abs=route.EXACT)


def test_m15_the_ordered_inverse_at_the_value_instead_of_the_preimage_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dose 2's pieces, inverse and derivative read ``b_1`` in place of ``beta_1``.  Measured: 3.9 on both readings."""
    edit = (
        "readable.update({names[e]: betas[e] for e in earlier})",
        "readable.update({names[e]: target[:, e] for e in earlier})",
    )
    mutate(monkeypatch, *_ENUMERATE, [edit])
    assert _geo_ordered() > CAUGHT
    assert _geo_ordered_long() > CAUGHT


# ------------------------------------------------------------------ M11, M14, M16, M17


def test_m11_the_preimage_sum_cut_to_the_value_cell_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the candidate cell ``c = c'`` is summed.  Measured: 1.0 on W16."""
    edit = (
        "hit = valid & np.all(image == target_codes, axis=1)",
        "hit = valid & np.all(image == target_codes, axis=1) "
        "& np.all(cell_codes == target_codes, axis=1)",
    )
    mutate(monkeypatch, *_ENUMERATE, [edit])
    assert _point(("flag gated",)) > CAUGHT


def test_m14_the_identity_piece_shortcut_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """An identity piece reads the value's own cell, a per-component ratio of one.  Measured: 0.25.

    This is the scalar route's shortcut (plan review rev 3, H1): on W17 the cap holds the
    top dose while the flag moves, and the dose's law depends on the flag.
    """
    edits = [
        (
            "            grown: list[tuple[dict[int, FloatArray], BoolArray, FloatArray]] = []\n",
            "            grown: list[tuple[dict[int, FloatArray], BoolArray, FloatArray]] = []\n"
            "            flags: list[bool] = []\n",
        ),
        (
            "                    grown.append(\n",
            "                    flags.append(piece.identity)\n                    grown.append(\n",
        ),
        (
            "        for betas, valid, jacobian in paths:\n            dose_values",
            "        for path_index, (betas, valid, jacobian) in enumerate(paths):\n"
            "            dose_values",
        ),
        (
            "position, cell_codes, dose_values, dose_values[:, position], cell=cell",
            "position, "
            "target_codes if flags[path_index] else cell_codes, "
            "target[:, doses] if flags[path_index] else dose_values, "
            "(target[:, doses] if flags[path_index] else dose_values)[:, position], "
            "cell=None if flags[path_index] else cell",
        ),
    ]
    mutate(monkeypatch, *_ENUMERATE, edits)
    assert _point(("flag set, dose up",)) > CAUGHT


def test_m16_dose_maps_read_the_value_cell_instead_of_the_candidate_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dose map's frame holds the value's flag, not the candidate cell.  Measured: 1.0."""
    edit = (
        "readable = dict(cell_labels)",
        "readable = {names[i]: layout.labels(i, target_codes[:, j]) "
        "for j, i in enumerate(categorical)}",
    )
    mutate(monkeypatch, *_ENUMERATE, [edit])
    assert _point(("flag gated",)) > CAUGHT


def test_m17_a_categorical_image_at_the_value_instead_of_the_preimage_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag map reads ``b`` in place of ``beta``.  Measured: 1.5 on W18."""
    edit = (
        "image = _image(layout, branch, cell_codes, dose_values, valid)",
        "image = _image(layout, branch, cell_codes, target[:, doses], valid)",
    )
    mutate(monkeypatch, *_ENUMERATE, [edit])
    assert _point(("flag reads dose",)) > CAUGHT


# ------------------------------------------------------------------ M18, B1


def test_m18_ratio_at_read_at_the_observed_value_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ``ratio_at`` column holds the ratio at the observed value.  Measured: W8 0.050, W13 0.

    W13 cannot see it: with an exact outcome regression the fluctuation does not move.
    """
    edit = (
        "return np.asarray(full, dtype=float)",
        "return np.asarray(np.concatenate([full[:, :1], np.repeat(full[:, :1], "
        "full.shape[2], axis=1)], axis=1), dtype=float)",
    )
    mutate(monkeypatch, policy_module, PolicySet, "design_at", [edit])
    assert _point(tuple(route.POINT_TRUTH)) < route.EXACT
    assert _targeted_error() > CAUGHT


def _provenance_gap() -> float:
    """B1: the largest gap of a prediction at the observed design from the stored density."""
    from sklearn.linear_model import LogisticRegression

    from cleverly.learners import Folds
    from cleverly.learners.density import fit_conditional_density

    rng = np.random.default_rng(3)
    w = rng.normal(size=(300, 2))
    a = w[:, 0] + rng.normal(size=300)
    density, _, predicted = fit_conditional_density(
        LogisticRegression(),
        w,
        a,
        np.ones(300),
        Folds(np.arange(300) % 3, 3),
        n_bins=8,
        predict_designs={"same": (w, a)},
    )
    return float(np.max(np.abs(predicted["same"] - density.density_at(a))))


def test_b1_counterfactual_predictions_by_a_model_of_every_row_are_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each fold predicts the designs with a model fitted on every row.  Measured: 0.11."""
    assert _provenance_gap() == 0.0
    edit = (
        "                predicted_at(model, test),",
        "                predicted_at(fit_on(np.arange(a.size)), test),",
    )
    mutate(monkeypatch, density_module, density_module, "fit_conditional_density", [edit])
    assert _provenance_gap() > CAUGHT
