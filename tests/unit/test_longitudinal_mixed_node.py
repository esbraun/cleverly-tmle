"""A longitudinal vector node with a continuous component, on the classifier route.

The oracle is the mixed reading of :mod:`tests.discrete_law_longitudinal_mtp`: two nodes,
each a binary flag and a dose on the grid ``{0, 1, 2}``, with a plan of
:class:`~cleverly.interventions.JointPolicy` nodes.  A saturated learner on a frame that
realises the law exactly makes every comparison exact (the conditions are in the oracle's
docstring), so the estimate and the influence curve must equal the g-formula and its
complex-step Gateaux derivative to rounding.

W3: both nodes mixed, every plan.  W4: the policy moves the flag and gates the dose on the
natural flag, and the shifted-flag reading differs by more than ``1e-2``.  W5: the later
node's history holds both components of the earlier one.  W9: censoring whose retention reads
both components.  W10: with an identity flag map, a joint policy equals the four-node fit of
the same columns.  W11: a known randomizer at the joint-policy level.  The composition rows
of plan section 10.6 and the refusals follow.  The mutation controls are in
``tests/unit/test_longitudinal_mixed_node_mutations.py``.
"""

from __future__ import annotations

import ast
import contextlib
import warnings
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import (
    Incremental,
    JointPolicy,
    ModifiedPolicy,
    Piece,
    Randomizer,
    RiskRatioTilt,
    Scale,
    Shift,
)
from cleverly.longitudinal import IPSI, LTMLE, SDR, DynamicRegimen, LongitudinalData

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_mtp as law
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import longitudinal_mtp as mtp

NO_TRUNCATION = (1e-8, 1.0)
EXACT = 1e-10
LABELS = ("natural", "dose up", "flag gated", "dose gated", "random")
COLUMNS = mtp.MIXED_COLUMNS
INF = np.inf


def settings(**overrides: Any) -> dict[str, Any]:
    """Saturated learners, one fold, no truncation and the classifier route."""
    return {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": multivalue.CellProbabilities(),
        "censoring_learner": binary_law.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        "simultaneous": False,
        "ratio": "classifier",
        **overrides,
    }


def fit(
    labels: tuple[str, ...] = LABELS,
    design: str = "end",
    *,
    frame: Any = None,
    regimens: Any = None,
    columns: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """A fit of ``labels`` on one reading of the mixed law, the first as reference."""
    declared = dict(COLUMNS if columns is None else columns)
    if design == "censored":
        declared["censoring"] = ["C1", "C2"]
    if design == "survival":
        declared["outcome"] = ["Y1", "Y2"]
    plans = mtp.mixed_regimens(labels) if regimens is None else regimens
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(plans, reference=labels[0], **settings(**overrides)).fit(
            law.mixed_frame(design) if frame is None else frame, **declared
        )


def names(labels: tuple[str, ...], design: str = "end") -> tuple[tuple[str, str, int], ...]:
    """``(reported name, plan label, horizon)`` of every regimen mean of a fit."""
    if design == "survival":
        return tuple(
            (f"risk_regimen[{label} @ t={horizon}]", label, horizon)
            for label in labels
            for horizon in (1, 2)
        )
    return tuple((f"ey_regimen[{label}]", label, 2) for label in labels)


def deviation(result: Any, labels: tuple[str, ...], design: str = "end") -> float:
    """The largest error of any regimen mean or curve against the oracle."""
    rows = law.mixed_first_row_of(design)
    worst = 0.0
    for name, label, horizon in names(labels, design):
        worst = max(worst, abs(result.psi(name) - law.mixed_truth(label, design, horizon)))
        curve = result.influence_curves[name][rows]
        worst = max(worst, float(np.max(np.abs(curve - law.mixed_eif(label, design, horizon)))))
    return worst


@cache
def _exact_fit(design: str = "end") -> Any:
    return fit(design=design)


# ------------------------------------------------------------------ W3, W4, W11


def test_w3_both_mixed_nodes_equal_the_g_formula_and_its_gateaux_derivative() -> None:
    """Every plan, including the W11 randomizer, at both mixed nodes."""
    result = _exact_fit()
    assert [result.data.node_kind(time) for time in (1, 2)] == ["mixed", "mixed"]
    assert deviation(result, LABELS) < EXACT
    # The classifier route fits no density, and the treatment factor is one at a mixed node.
    assert not result.mechanism.densities
    for label in LABELS[1:]:
        name = f"ate_regimen[{label} vs natural]"
        truth = law.mixed_truth(label) - law.mixed_truth("natural")
        assert result.psi(name) == pytest.approx(truth, abs=EXACT)


def test_w3_the_plans_are_not_degenerate() -> None:
    """Every plan moves the parameter, and the ratios differ from one at each node."""
    means = {label: law.mixed_truth(label) for label in (*LABELS, "flag gated (shifted flag)")}
    ordered = sorted(means.values())
    assert min(b - a for a, b in pairwise(ordered)) > 1e-4
    for label in LABELS[1:]:
        assert abs(means[label] - means["natural"]) > 0.005, label
    numerator = _exact_fit().fits["flag gated"].cumulative_numerator
    first = numerator[:, 0]
    second = numerator[:, 1] / np.where(first > 0, first, 1.0)
    assert np.max(np.abs(first - 1.0)) > 0.5
    assert np.max(np.abs(second - 1.0)) > 0.5


def test_w4_the_dose_map_reads_the_natural_flag_after_the_flag_moves() -> None:
    """The policy sets the flag to one and gates the dose on the natural flag.

    The shifted-flag reading (M3) is another parameter, by more than ``1e-2`` on this law.
    """
    natural_read = law.mixed_truth("flag gated")
    shifted_read = law.mixed_truth("flag gated (shifted flag)")
    assert abs(natural_read - shifted_read) > 1e-2
    result = _exact_fit()
    assert result.psi("ey_regimen[flag gated]") == pytest.approx(natural_read, abs=EXACT)
    assert abs(result.psi("ey_regimen[flag gated]") - shifted_read) > 1e-2


def test_w11_a_known_randomizer_at_the_joint_policy_level() -> None:
    """The randomizer of the joint policy weights its branches by the known law."""
    result = _exact_fit()
    plan = result.fits["random"]
    assert deviation(result, ("random",)) < EXACT
    assert abs(law.mixed_truth("random") - law.mixed_truth("dose up")) > 0.01
    assert plan.policy[0] is not None and plan.policy[0].shape[1] == 2


# ------------------------------------------------------------------ W5


def test_w5_the_later_history_holds_both_components_of_the_earlier_node() -> None:
    """Each dose enters as itself and the flag as its own indicator, never as a joint code."""
    data = _exact_fit().data
    frame = law.mixed_frame()
    expected = np.column_stack([frame["W"], frame["L2"], frame["F1"] == 1.0, frame["D1"]]).astype(
        float
    )
    np.testing.assert_array_equal(data.history_design(2), expected)
    current = data.history_design(2, include_current=True)
    np.testing.assert_array_equal(current[:, 4:], np.column_stack([frame["F2"], frame["D2"]]))
    assert data.component_names(1) == ("F1", "D1")
    assert data.component_levels[0] == ((0, 1), ())


# ------------------------------------------------------------------ W9 censoring, survival


def test_w9_censoring_that_reads_both_components_of_the_mixed_node() -> None:
    labels = ("natural", "dose up", "flag gated", "random")
    result = fit(labels, "censored")
    assert deviation(result, labels, "censored") < EXACT
    assert result.data.has_censoring


def test_w9_the_earlier_node_enters_the_later_regression_directly() -> None:
    """The law's outcome reads both node-1 components at fixed later values, so M9 can fail."""
    q = np.array(
        [
            [law._mixed_q(w, k1, l2, k2) for k1 in range(6)]
            for w in range(2)
            for l2 in range(2)
            for k2 in range(6)
        ]
    )
    flag_effect = np.abs(q[:, 3:] - q[:, :3])
    dose_effect = np.abs(np.diff(q[:, :3], axis=1))
    assert flag_effect.max() > 0.2 and dose_effect.max() > 0.2


def test_a_survival_outcome_at_mixed_nodes_is_the_cumulative_risk() -> None:
    """The scalar classifier path carries the event nodes unchanged."""
    labels = ("natural", "dose up", "flag gated", "random")
    result = fit(labels, "survival")
    assert deviation(result, labels, "survival") < EXACT
    assert (
        abs(law.mixed_truth("dose up", "survival", 1) - law.mixed_truth("natural", "survival", 1))
        > 0.01
    )


def test_a_one_cause_competing_fit_at_mixed_nodes_is_the_survival_fit() -> None:
    """The competing-risks path reads the same nodes: one cause is the survival curve."""
    labels = ("natural", "flag gated")
    survival = fit(labels, "survival")
    columns = {**COLUMNS, "outcome": {"event": ["Y1", "Y2"]}}
    competing = fit(labels, frame=law.mixed_frame("survival"), columns=columns)
    for horizon in (1, 2):
        for label in labels:
            single = survival.psi(f"risk_regimen[{label} @ t={horizon}]")
            cause = competing.psi(f"cif_regimen[{label}, event @ t={horizon}]")
            assert cause == pytest.approx(single, abs=1e-12)


# ------------------------------------------------------------------ W10


def _four_node_fit(labels: tuple[str, ...]) -> Any:
    """The same columns as four nodes: flag, dose, flag, dose."""
    natural = Shift(0.0, cap=None)
    plans = {
        "natural": DynamicRegimen("natural", (natural,) * 4),
        "dose gated": DynamicRegimen(
            "dose gated",
            (natural, mtp._up_if_flagged("F1"), natural, mtp._up_if_flagged("F2")),
        ),
    }
    columns = {
        "outcome": "Y",
        "treatment": ["F1", "D1", "F2", "D2"],
        "baseline": ["W"],
        "time_varying": [[], [], ["L2"], []],
        "continuous_treatment": ["D1", "D2"],
    }
    return fit(labels, regimens={label: plans[label] for label in labels}, columns=columns)


def test_w10_an_identity_flag_map_equals_the_four_node_fit() -> None:
    """Two consecutive nodes read the intervened flag, and a joint policy the natural one.

    With an identity flag map the two agree, which ties the mixed path to the shipped
    scalar classifier path.
    """
    labels = ("natural", "dose gated")
    joint = _exact_fit()
    split = _four_node_fit(labels)
    for label in labels:
        name = f"ey_regimen[{label}]"
        assert joint.psi(name) == pytest.approx(split.psi(name), abs=EXACT)
        np.testing.assert_allclose(
            joint.influence_curves[name], split.influence_curves[name], atol=EXACT, rtol=0
        )
    assert abs(law.mixed_truth("dose gated") - law.mixed_truth("natural")) > 0.005


# ------------------------------------------------------------------ compositions (10.6)


def test_integer_weights_equal_the_expanded_sample() -> None:
    frame = law.mixed_sample(600, 3)
    weights = 1.0 + (np.arange(len(frame)) % 3 == 0)
    expanded = pd.concat([frame, frame[weights == 2.0]], ignore_index=True)
    labels = ("natural", "flag gated", "random")
    by_weight = fit(labels, frame=frame.assign(w=weights), columns={**COLUMNS, "weights": "w"})
    by_copy = fit(labels, frame=expanded)
    for label in labels:
        name = f"ey_regimen[{label}]"
        assert by_weight.psi(name) == pytest.approx(by_copy.psi(name), abs=1e-10)


@pytest.mark.parametrize("folds", [1, 3])
def test_a_clustered_fit_keeps_whole_clusters(folds: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stacked classifier groups a cluster's rows and their copies in its inner folds."""
    from cleverly.learners import density_ratio

    seen: list[np.ndarray] = []
    original = density_ratio.fit_learner

    def record(learner: Any, design: Any, target: Any, weight: Any, **kwargs: Any) -> Any:
        seen.append(np.asarray(kwargs["groups"]))
        return original(learner, design, target, weight, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", record)
    frame = law.mixed_sample(900, 4).assign(cluster=lambda f: np.arange(len(f)) // 3)
    result = fit(
        ("natural", "flag gated"),
        frame=frame,
        columns={**COLUMNS, "id": "cluster"},
        n_folds=folds,
        random_state=1,
    )
    assert np.isfinite(result.psi("ate_regimen[flag gated vs natural]"))
    clusters = result.data.cluster
    assert seen and all(set(np.unique(groups)) <= set(np.unique(clusters)) for groups in seen)
    if folds > 1:
        for train, test in result.folds:
            assert not set(clusters[train]) & set(clusters[test])


def test_cross_fitting_splits_units_not_stacked_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """W7 at a mixed node: a unit's copies train together and a held-out unit in neither."""
    from cleverly.learners import density_ratio

    seen: list[np.ndarray] = []
    original = density_ratio.fit_learner

    def record(learner: Any, design: Any, target: Any, weight: Any, **kwargs: Any) -> Any:
        seen.append(np.asarray(kwargs["groups"]))
        return original(learner, design, target, weight, **kwargs)

    monkeypatch.setattr(density_ratio, "fit_learner", record)
    result = fit(
        ("natural", "flag gated"), frame=law.mixed_sample(600, 2), n_folds=3, random_state=3
    )
    folds = list(result.folds)
    # Two nodes, two plans, three folds: one classifier per plan, node and fold.
    assert len(seen) == 2 * 2 * 3
    for index, groups in enumerate(seen):
        train, test = folds[index % 3]
        units, counts = np.unique(groups, return_counts=True)
        assert set(units.tolist()) <= set(np.asarray(train).tolist())
        assert np.all(counts == 2)
        assert not set(units.tolist()) & set(np.asarray(test).tolist())


def test_a_working_model_over_mixed_plans_reproduces_the_separate_fits() -> None:
    """A saturated ``msm=`` over the joint-policy plans equals the per-plan fits."""
    from cleverly.msm import MSM

    cells = ("natural", "dose up", "flag gated")
    model = MSM(
        design=lambda label, horizon, w: np.column_stack(
            [np.full(len(w), float(label == item)) for item in cells]
        ),
        terms=cells,
        design_kind="known",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        projected = LTMLE(mtp.mixed_regimens(cells), msm=model, **settings()).fit(
            law.mixed_frame(), **COLUMNS
        )
    separate = _exact_fit()
    for label in cells:
        assert projected.psi(f"msm_regimen[{label}]") == pytest.approx(
            separate.psi(f"ey_regimen[{label}]"), abs=EXACT
        )
        np.testing.assert_allclose(
            projected.influence_curves[f"msm_regimen[{label}]"],
            separate.influence_curves[f"ey_regimen[{label}]"],
            atol=EXACT,
        )


def test_the_truncation_replay_reproduces_the_fit() -> None:
    """The bound reaches no factor of a mixed node: its ratio sits in the numerator."""
    from cleverly.longitudinal.estimator import longitudinal_truncation_curve

    result = fit(("natural", "flag gated"), frame=law.mixed_sample(800, 9), g_bounds=(0.01, 1.0))
    curve = longitudinal_truncation_curve(
        result, [(0.01, 1.0), (0.2, 1.0)], estimands=["ey_regimen[flag gated]"]
    ).set_index("is_fitted_bound")
    assert curve.loc[True, "delta_from_fitted"] == 0.0
    assert abs(curve.loc[False, "delta_from_fitted"]) < 1e-12


def test_a_bootstrap_refits_mixed_nodes() -> None:
    result = fit(
        ("natural", "flag gated"), frame=law.mixed_sample(500, 5), n_bootstrap=3, random_state=4
    )
    assert result.bootstrap is not None
    assert result.bootstrap.n_failed == 0
    for draws in result.bootstrap.draws.values():
        assert np.all(np.isfinite(draws))


def test_a_prepared_container_passes_the_design_check() -> None:
    from cleverly import LongitudinalTreatment

    design = LongitudinalTreatment(
        outcome="Y",
        treatment=(("F1", "D1"), ("F2", "D2")),
        baseline=("W",),
        time_varying=((), ("L2",)),
        continuous_treatment=("D1", "D2"),
    )
    data = design.prepare(law.mixed_frame())
    assert design.prepare(data) is data
    assert data.continuous_columns == ("D1", "D2")
    wrong = LongitudinalTreatment(
        outcome="Y",
        treatment=(("F1", "D1"), ("F2", "D2")),
        baseline=("W",),
        time_varying=((), ("L2",)),
        continuous_treatment=("D1",),
    )
    with pytest.raises(DataError, match="continuous_treatment"):
        wrong.prepare(data)


# ------------------------------------------------------------------ the subset rule


def test_a_reversed_subset_with_repeats_keeps_each_row_with_its_components() -> None:
    data = LongitudinalData.from_frame(
        law.mixed_frame("censored"),
        censoring=["C1", "C2"],
        **COLUMNS,
    )
    index = np.concatenate([np.arange(data.n)[::-1], np.arange(0, data.n, 7)])
    copy = data.subset(index)
    for time in (1, 2):
        np.testing.assert_array_equal(copy.node_values(time), data.node_values(time)[index])
        np.testing.assert_array_equal(copy.node_present(time), data.node_present(time)[index])
    assert copy.component_levels == data.component_levels
    assert copy.node_kinds == data.node_kinds
    assert np.isnan(copy.node_values(2)).any()


# ------------------------------------------------------------------ fail-closed readers


def test_one_column_readers_raise_at_a_mixed_node() -> None:
    data = _exact_fit().data
    with pytest.raises(CapabilityError, match="read node_kind"):
        data.is_continuous_node(1)
    with pytest.raises(CapabilityError, match="read node_kind"):
        data.node_codes(2)
    assert data.node_values(1).shape == (data.n, 2)


def test_hazard_targeting_refuses_a_vector_with_a_dose_by_name() -> None:
    """``refuse_hazard_design`` reads the node kind, so a vector gets the H11 refusal."""
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "W": rng.integers(0, 2, n).astype(float),
            "F": rng.integers(0, 2, n).astype(float),
            "D": rng.uniform(0.0, 2.0, n),
            "C1": 1.0,
            "Y1": 0.0,
        }
    )
    policy = JointPolicy("up", components={"D": Shift(0.5, cap=None)}, policy_kind="known")
    with pytest.raises(CapabilityError, match="treatment 'F\\+D' is a vector with a dose"):
        LTMLE({"up": policy}, n_folds=1, survival_model="hazard").fit(
            frame,
            treatment=[["F", "D"]],
            continuous_treatment=["D"],
            outcome=["Y1"],
            baseline=["W"],
            censoring=["C1"],
        )


#: The functions allowed to read ``.treatment[...]`` or call ``is_continuous_node``, each
#: because it dispatches on the node kind first or reads another object's ``treatment``.
_DISPATCHING: dict[tuple[str, str], str] = {
    ("data/causal_data.py", "CausalData.treated_fraction"): "point CausalData",
    ("data/causal_data.py", "CausalData.subset"): "point CausalData",
    ("data/known_mechanism.py", "KnownNodeMechanisms.treatment_at"): "declared matrices",
    ("estimators/_nuisance.py", "_retained_covariates"): "point CausalData",
    ("estimators/ctmle.py", "CTMLE._derivative_nuisances"): "point CausalData",
    ("estimators/ctmle.py", "_Selector._intercept_propensity.proportions"): "point CausalData",
    ("estimators/ctmle.py", "_Selector._fluctuate"): "point CausalData",
    ("estimators/ctmle.py", "_Selector._lasso_propensities"): "point CausalData",
    ("estimators/ctmle.py", "_Selector._ordering"): "point CausalData",
    ("longitudinal/data.py", "LongitudinalData.node_present"): "presence of every kind",
    ("longitudinal/data.py", "LongitudinalData.node_codes"): "raises at a mixed node",
    ("longitudinal/data.py", "LongitudinalData.node_values"): "dispatches on the kind",
    ("longitudinal/data.py", "LongitudinalData.subset"): "indexes the slot by row",
    ("longitudinal/data.py", "LongitudinalData.followed_through"): "compares presence codes",
    ("longitudinal/data.py", "LongitudinalData.observed_labels"): "mixed branch first",
    ("longitudinal/data.py", "LongitudinalData.policy_frame"): "mixed branch first",
    ("longitudinal/data.py", "LongitudinalData.history_design"): "mixed branch first",
    ("longitudinal/data.py", "LongitudinalData.encode_assignment"): "mixed branch first",
    ("longitudinal/estimator.py", "_run_hazard"): "after refuse_hazard_design",
    ("longitudinal/gformula.py", "Layout.build"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "ModelBook.models"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator.__init__"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator._modified"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator._set_treatment"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator._tilted"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_support_rows"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "fit_models"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "natural_nodes"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "observed_table"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "write_treatment"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "has_set_dose"): "is_continuous_node raises at a mixed node",
    ("longitudinal/hazard.py", "hazard_inputs"): "after refuse_hazard_design",
    ("longitudinal/regimen.py", "_resolve_mtp_node"): "mixed branch first",
    ("longitudinal/sdr.py", "node_factors"): "fitted factors",
    ("longitudinal/sequential.py", "Mechanism.cumulative_with_unbounded"): "fitted factors",
    ("longitudinal/sequential.py", "outcome_design"): "mixed branch first",
    ("longitudinal/sequential.py", "node_ratios"): "fitted factors",
    ("study.py", "LongPanel._check_prepared"): (
        "a panel container only: from_long_panel refuses a vector node with a dose"
    ),
    ("study.py", "TimeToEvent._check_prepared"): "one baseline column",
    ("validation/longitudinal.py", "_hazard_nuisances"): "a hazard fit only",
    ("validation/longitudinal.py", "_treatment_report"): "mixed branch first",
}


#: The functions that index ``.treatment_levels[...]``, each with why it cannot read a
#: mixed node's entry.  The entry raises ``CapabilityError`` on any read, so a reader missed
#: here fails closed; the list makes each one a reviewed decision (review LOW-6).
_LEVEL_READERS: dict[tuple[str, str], str] = {
    ("data/causal_data.py", "CausalData.arm_label"): "point CausalData; refuses a vector first",
    ("longitudinal/data.py", "LongitudinalData._declare_known"): "a mixed node is refused first",
    ("longitudinal/data.py", "LongitudinalData.encode_assignment"): "mixed branch first",
    ("longitudinal/data.py", "LongitudinalData.history_design"): "mixed branch first",
    ("longitudinal/data.py", "LongitudinalData.observed_labels"): "mixed branch first",
    ("longitudinal/data.py", "LongitudinalData.policy_frame"): "mixed branch first",
    (
        "longitudinal/estimator.py",
        "ParametricGFormula._refuse_design",
    ): "a tilt at a mixed node is refused first",
    ("longitudinal/gformula.py", "Layout.build"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator._code_of"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator._codes_of"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator._level_array"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator._modified"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "_Simulator._policy_density"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "fit_models"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/gformula.py", "write_treatment"): "after GFORMULA_MIXED_NODE_REFUSAL",
    ("longitudinal/hazard.py", "level_plans"): "after refuse_hazard_design",
    ("longitudinal/hazard.py", "regimen_weights"): "after refuse_hazard_design",
    ("longitudinal/regimen.py", "DynamicRegimen.policy_density"): "categorical policy node",
    ("longitudinal/regimen.py", "_categorical_carry"): "categorical policy node",
    ("longitudinal/regimen.py", "_resolve_mtp_node"): "mixed branch first",
    ("longitudinal/regimen.py", "_resolve_plan"): "a tilt at a mixed node is refused first",
    ("longitudinal/sequential.py", "_categorical_node"): "categorical node",
    ("longitudinal/sequential.py", "_check_policy_arm_support"): "categorical node guard",
    ("longitudinal/sequential.py", "_rule_hint"): "a rule at a mixed node is refused",
    ("longitudinal/sequential.py", "outcome_design"): "mixed branch first",
    ("longitudinal/sequential.py", "preflight_mechanism_support"): "categorical branch",
    ("study.py", "BackdoorMeanContrast.expression"): "a point design's own levels",
    ("study.py", "BackdoorMeanContrast.reference_arm"): "a point design's own levels",
    ("study.py", "BackdoorMeanContrast.selected_arm"): "a point design's own levels",
    (
        "validation/longitudinal.py",
        "_ipsi_node_row",
    ): "a tilt node of a fitted plan; a mixed one is refused",
    ("validation/longitudinal.py", "_longitudinal_stagewise"): "skips a mixed node",
}


class _Readers(ast.NodeVisitor):
    """Record each function of ``module`` that reads ``.treatment[...]`` or calls the check."""

    def __init__(
        self, module: str, found: set[tuple[str, str]], attribute: str = "treatment"
    ) -> None:
        self.module = module
        self.found = found
        self.attribute = attribute
        self.stack: list[str] = []

    def visit_FunctionDef(self, node: Any) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Attribute) and node.value.attr == self.attribute:
            self.found.add((self.module, ".".join(self.stack)))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            self.attribute == "treatment"
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "is_continuous_node"
        ):
            self.found.add((self.module, ".".join(self.stack)))
        self.generic_visit(node)


def _one_column_reads(attribute: str = "treatment") -> set[tuple[str, str]]:
    root = Path(__file__).resolve().parents[2] / "src" / "cleverly"
    found: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        _Readers(module, found, attribute).visit(ast.parse(path.read_text(encoding="utf-8")))
    return found


def test_every_one_column_treatment_read_sits_in_a_dispatching_function() -> None:
    """A new reader of ``.treatment[...]`` or ``is_continuous_node`` must be reviewed here.

    A mixed node's slot holds ``0.0`` on present rows, a valid level code, so a reader that
    did not dispatch on the node kind would read level zero or a zero dose and continue.
    """
    found = _one_column_reads()
    assert found - set(_DISPATCHING) == set(), "unreviewed one-column treatment readers"
    assert set(_DISPATCHING) - found == set(), "stale entries in the allow-list"


def test_a_long_panel_holds_no_mixed_node() -> None:
    """The premise of the ``LongPanel._check_prepared`` entry: a panel node is never mixed."""
    from .test_long_panel import build, long_panel

    frame = long_panel(20)
    frame["dose"] = np.linspace(0.0, 1.0, len(frame))
    with pytest.raises(DataError, match="a vector node is categorical"):
        build(frame, treatment=["offer", "dose"], continuous_treatment=True)


def test_every_level_set_read_sits_in_a_reviewed_function() -> None:
    """A new reader of ``.treatment_levels[...]`` must be reviewed here (review LOW-6)."""
    found = _one_column_reads("treatment_levels")
    assert found - set(_LEVEL_READERS) == set(), "unreviewed treatment_levels readers"
    assert set(_LEVEL_READERS) - found == set(), "stale entries in the allow-list"


def test_a_mixed_node_has_no_level_set_to_read() -> None:
    """Every read of a mixed node's ``treatment_levels`` entry refuses (review LOW-6).

    It held ``()``, so a reader that skipped the node-kind check read zero levels and went
    on.  The entry pickles, compares and prints.
    """
    import pickle

    data = _exact_fit().data
    entry = data.treatment_levels[0]
    for read in (len, list, bool, lambda levels: levels[0], lambda levels: 0 in levels):
        with pytest.raises(CapabilityError, match="read node_kind"):
            read(entry)
    assert pickle.loads(pickle.dumps(data)).treatment_levels == data.treatment_levels
    assert "mixed node" in repr(entry)


def test_every_reader_of_a_mixed_node_result_returns_or_refuses() -> None:
    """No public member of a mixed-node result fails with an unplanned error.

    A member returns, or raises ``CapabilityError`` or the ``ValueError`` of a request the
    parameter axis does not carry.  The sweep found the stagewise report reading a mixed
    node's level set, and ``benchmark()`` reading an arm attribute the panel lacks.
    """
    result = _exact_fit()
    subjects = {"result": result, "diagnostics": result.diagnostics}
    subjects["sensitivity"] = result.sensitivity
    for where, subject in subjects.items():
        for name in sorted(dir(type(subject))):
            if name.startswith("_") or name == "save":
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    value = getattr(subject, name)
                    if callable(value) and not isinstance(value, type):
                        value()
            except (CapabilityError, ValueError):
                continue
            except TypeError as error:
                assert "argument" in str(error), f"{where}.{name}: {error}"


# ------------------------------------------------------------------ refusals


class NeverFit:
    """A learner that raises if fitted, the witness that a refusal precedes every learner."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> NeverFit:
        return self

    def fit(self, *args: Any, **kwargs: Any) -> NeverFit:
        raise AssertionError("a learner was fitted before the refusal")

    def predict(self, x: Any) -> Any:
        raise AssertionError("a learner predicted before the refusal")


def _refused(regimens: Any, *, frame: Any = None, ratio: str = "classifier", **columns: Any) -> Any:
    learners = {
        "outcome_learner": NeverFit(),
        "pseudo_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LTMLE(regimens, **settings(ratio=ratio, **learners)).fit(
            law.mixed_frame() if frame is None else frame, **{**COLUMNS, **columns}
        )


def _plan(node1: Any, node2: Any) -> dict[str, Any]:
    return {"p": DynamicRegimen("p", (node1, node2))}


def _joint(label: str, time: int) -> JointPolicy:
    return mtp._joint(label, time)


@pytest.mark.parametrize(
    "node",
    [
        Shift(1.0, cap=2.0),
        Scale(2.0, cap=2.0),
        ModifiedPolicy("map", apply=lambda a, h: a, policy_kind="known"),
        RiskRatioTilt(0.5),
        1.0,
        DynamicRegimen("rule", (lambda h: h["W"], 0.0), rule_kind="known").plan[0],
    ],
    ids=["shift", "scale", "apply", "tilt", "label", "rule"],
)
def test_a_whole_node_declaration_at_a_mixed_node(node: Any) -> None:
    regimens = {"p": DynamicRegimen("p", (node, _joint("natural", 2)), rule_kind="known")}
    with pytest.raises(CapabilityError, match="takes a JointPolicy, one policy per component"):
        _refused(regimens)


@pytest.mark.parametrize("engine", [LTMLE, SDR, IPSI], ids=["ltmle", "sdr", "ipsi"])
@pytest.mark.parametrize("whole_plan", [False, True], ids=["node", "plan"])
@pytest.mark.parametrize(
    "tilt", [Incremental(2.0), Incremental({"high": 2.0})], ids=["scalar", "mapping"]
)
def test_an_odds_tilt_at_a_mixed_node(engine: Any, whole_plan: bool, tilt: Incremental) -> None:
    """An odds tilt multiplies the odds of one categorical node; a mixed node is not one.

    The refusal precedes every learner and precedes the read of the node's levels, so a
    mapping whose key is no level of the node meets this refusal and not the key refusal.  It
    holds on every engine that reads a tilt, for a tilt declared at the node and for a
    one-entry plan that tilts every node.
    """
    regimens = {
        "p": tilt
        if whole_plan
        else DynamicRegimen("p", (tilt, _joint("natural", 2)), rule_kind="known")
    }
    learners = {
        "outcome_learner": NeverFit(),
        "pseudo_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "censoring_learner": NeverFit(),
    }
    options = settings(**learners)
    if engine is not LTMLE:
        options.pop("simultaneous")
    with (
        pytest.raises(CapabilityError, match="is not one categorical node") as refusal,
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        engine(regimens, **options).fit(law.mixed_frame(), **COLUMNS)
    assert "inside a JointPolicy" in str(refusal.value)
    assert "'F1+D1'" in str(refusal.value)


def test_a_static_regimen_at_a_mixed_node() -> None:
    with pytest.raises(CapabilityError, match="takes a JointPolicy"):
        _refused({"static": 1.0})


def test_a_joint_policy_at_a_node_of_one_column() -> None:
    columns = {"treatment": ["F1", ["F2", "D2"]], "continuous_treatment": ["D2"]}
    with pytest.raises(CapabilityError, match="which is one column"):
        _refused(_plan(_joint("natural", 1), _joint("natural", 2)), **columns)


def test_a_component_that_is_not_a_column_of_the_node() -> None:
    wrong = JointPolicy("wrong", components={"D2": Shift(1.0, cap=2.0)}, policy_kind="known")
    with pytest.raises(DataError, match=r"declares component\(s\) \['D2'\]"):
        _refused(_plan(wrong, _joint("natural", 2)))


def test_a_label_map_on_a_dose_component() -> None:
    mapped = JointPolicy(
        "apply",
        components={"D1": ModifiedPolicy("d", apply=lambda a, h: a, policy_kind="known")},
        policy_kind="known",
    )
    with pytest.raises(CapabilityError, match="declares apply= on a continuous treatment"):
        _refused(_plan(mapped, _joint("natural", 2)))


def test_a_dose_map_that_collapses_an_interval_is_checked_per_component() -> None:
    floor = ModifiedPolicy(
        "floor",
        pieces=(
            Piece(-INF, INF, lambda a, h: np.minimum(a, 1.0), lambda b, h: b, lambda b, h: 1.0),
        ),
        policy_kind="known",
    )
    collapsing = JointPolicy("floor", components={"D1": floor}, policy_kind="known")
    with pytest.raises(CapabilityError, match="not pathwise differentiable"):
        _refused(_plan(collapsing, _joint("natural", 2)))


def test_a_dose_map_that_reads_a_later_dose_names_the_order_that_resolves_it() -> None:
    """A map that reads a dose after it in the declared order is a declaration error.

    Any acyclic read graph has an order that makes the map triangular, so the refusal names
    it (plan rev 3.1, M6 option B).
    """
    frame = law.mixed_frame().assign(E1=lambda f: f["D1"] + f["W"])
    reads = ModifiedPolicy(
        "reads E1",
        pieces=(
            Piece(
                -INF,
                lambda h: np.asarray(h["E1"], dtype=float) + 10.0,
                lambda a, h: a + 1.0,
                lambda b, h: b - 1.0,
                lambda b, h: 1.0,
            ),
            Piece(lambda h: np.asarray(h["E1"], dtype=float) + 10.0, INF),
        ),
        policy_kind="known",
    )
    triangular = JointPolicy("triangular", components={"D1": reads}, policy_kind="known")
    with pytest.raises(DataError, match=r"'D1' reads \['E1'\].*order=\('E1', 'D1'\)"):
        _refused(
            _plan(triangular, _joint("natural", 2)),
            frame=frame,
            treatment=[["F1", "D1", "E1"], ["F2", "D2"]],
            continuous_treatment=["D1", "E1", "D2"],
        )


def test_a_label_off_the_levels_of_a_flag() -> None:
    off = JointPolicy(
        "off",
        components={"F1": ModifiedPolicy("f", apply=lambda a, h: a + 5, policy_kind="known")},
        policy_kind="known",
    )
    with pytest.raises(DataError, match="not levels of the component"):
        _refused(_plan(off, _joint("natural", 2)))


def test_an_undeclared_joint_policy_and_an_undeclared_component() -> None:
    with pytest.raises(CapabilityError, match="inspect a closure"):
        _refused(_plan(JointPolicy("u", components={}), _joint("natural", 2)))
    undeclared = JointPolicy(
        "c",
        components={"F1": ModifiedPolicy("f", apply=lambda a, h: a)},
        policy_kind="known",
    )
    with pytest.raises(CapabilityError, match="inspect a closure"):
        _refused(_plan(undeclared, _joint("natural", 2)))


def test_a_tilt_on_a_dose_component_is_refused_with_the_scalar_reason() -> None:
    """A component randomizer and a tilt are declarations; a tilt must set a 0/1 label."""
    randomized = ModifiedPolicy(
        "r",
        pieces={"a": (Piece(-INF, INF),), "b": (Piece(-INF, INF),)},
        randomizer=Randomizer(("a", "b"), (0.5, 0.5)),
        policy_kind="known",
    )
    JointPolicy("j", components={"D1": randomized}, policy_kind="known")
    tilt = JointPolicy("tilt dose", components={"D1": RiskRatioTilt(0.5)}, policy_kind="known")
    with pytest.raises(
        DataError,
        match="RiskRatioTilt sets a binary treatment coded 0 and 1, "
        "but component 'D1' is continuous",
    ):
        _refused(_plan(tilt, _joint("natural", 2)))


def test_a_known_mechanism_at_a_mixed_node() -> None:
    probabilities = np.full((law.MIXED_N["end"], 2), 0.5)
    with pytest.raises(CapabilityError, match="vector node with a dose has no levels"):
        _refused(mtp.mixed_regimens(("natural",)), treatment_probabilities=probabilities)


def test_a_joint_policy_on_a_point_treatment_is_refused() -> None:
    from cleverly.data import CausalData
    from cleverly.estimators import TMLE

    frame = law.mixed_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="D1", covariates=["W"], treatment_kind="continuous"
        )
    joint = JointPolicy("j", components={"D1": Shift(1.0, cap=2.0)}, policy_kind="known")
    with (
        pytest.raises(
            CapabilityError, match="applies at a vector node of a longitudinal fit, or at a point"
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        TMLE(policies=[joint], ratio="classifier", cross_fit=False).fit(data)


# ------------------------------------------------------------------ all-categorical vectors


def test_a_joint_policy_on_a_categorical_vector_node_is_its_tuple_map() -> None:
    """One declaration for both node kinds: on two binary components it is the tuple map."""
    from .test_influence_gateaux_longitudinal_mtp import deviation as vector_deviation
    from .test_influence_gateaux_longitudinal_mtp import fit as vector_fit

    def joint(time: int) -> JointPolicy:
        first, second = f"A{time}a", f"A{time}b"
        return JointPolicy(
            "vector",
            components={
                second: ModifiedPolicy(
                    "b",
                    apply=lambda a, h, first=first: np.where(
                        np.asarray(h[first], dtype=float) == 1, 0, np.asarray(a)
                    ),
                    policy_kind="known",
                )
            },
            policy_kind="known",
        )

    labels = ("natural", "vector")
    regimens = {
        "natural": mtp.NATURAL,
        "vector": DynamicRegimen("vector", (joint(1), joint(2))),
    }
    result = vector_fit(labels, reading="vector", regimens=regimens)
    assert vector_deviation(result, labels) < EXACT


# ------------------------------------------------------------------ one training fold


def _fold_mechanism(data: LongitudinalData, ratio: str, rows: Any = None) -> Any:
    from cleverly.learners import Folds
    from cleverly.longitudinal.regimen import resolve_plans, resolve_regimens
    from cleverly.longitudinal.sequential import fit_mechanism

    spec = {
        label: DynamicRegimen(label, (_joint(label, 1), _joint(label, 2)), rule_kind="known")
        for label in ("dose up", "flag gated")
    }
    plans = resolve_plans(resolve_regimens(spec, data.n_times), data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit_mechanism(
            data,
            plans,
            treatment_learner=multivalue.CellProbabilities(),
            censoring_learner=binary_law.CellMeans(),
            folds=Folds.single(data.n),
            ratio=ratio,
            density_bins=4,
            fit_rows=rows,
        )


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_a_restricted_mechanism_trains_a_mixed_node_on_its_fold_rows(ratio: str) -> None:
    """``fit_rows`` reaches every learner of a mixed node, as it does at a scalar node.

    The witness is the same fit on the training rows alone.  The saturated learners make
    each prediction a function of the training rows only, so the restricted fit equals the
    subset fit at those rows: the ratio numerator and the bounded treatment factor, on both
    routes.  The density route's bin edges are read from every row at risk, as at a scalar
    node, so they equal the unrestricted fit's; on this dose grid the subset's are the same.
    """
    data = LongitudinalData.from_frame(law.mixed_sample(900, 11), **COLUMNS)
    rows = np.arange(data.n) % 3 != 0
    restricted = _fold_mechanism(data, ratio, rows)
    alone = _fold_mechanism(data.subset(np.flatnonzero(rows)), ratio)
    everyone = _fold_mechanism(data, ratio)
    moved = 0.0
    for label in ("dose up", "flag gated"):
        for time in (1, 2):
            np.testing.assert_allclose(
                restricted.policy_numerators[label][time][rows],
                alone.policy_numerators[label][time],
                rtol=0,
                atol=EXACT,
            )
            np.testing.assert_allclose(
                restricted.treatment[time - 1][label][rows],
                alone.treatment[time - 1][label],
                rtol=0,
                atol=EXACT,
            )
            moved = max(
                moved,
                float(
                    np.max(
                        np.abs(
                            restricted.policy_numerators[label][time]
                            - everyone.policy_numerators[label][time]
                        )
                    )
                ),
            )
    # Nondegeneracy: the restriction moves the fit.
    assert moved > 1e-4
    if ratio == "density":
        for key, density in restricted.densities.items():
            np.testing.assert_array_equal(density.edges, everyone.densities[key].edges)


# ------------------------------------------------------------------ SDR (review MEDIUM-2)


def _sdr_fit(ratio: str, *, copies: int = 1, monkeypatch: Any = None) -> Any:
    """SDR at both mixed nodes of the exact law, with ``copies`` copies of the frame.

    With two copies, fold ``k`` holds copy ``k``, so each training complement is one exact
    copy of the law: the training-fold mechanism (``fit_rows``) and the out-of-fold
    predictions are then exact too.
    """
    from cleverly.learners import Folds

    frame = pd.concat([law.mixed_frame("end")] * copies, ignore_index=True)
    configuration = settings(ratio=ratio, n_folds=copies)
    configuration.pop("simultaneous")
    if copies > 1:
        size = len(frame) // copies
        assignment = np.repeat(np.arange(copies), size)
        monkeypatch.setattr(SDR, "_folds", lambda self, data: Folds(assignment, copies))
    exact = mtp.exact_bins() if ratio == "density" else contextlib.nullcontext()
    with warnings.catch_warnings(), exact:
        warnings.simplefilter("ignore")
        return SDR(mtp.mixed_regimens(LABELS), reference=LABELS[0], **configuration).fit(
            frame, **COLUMNS
        )


def sdr_deviation(result: Any, copies: int = 1) -> float:
    """The largest error of an SDR mean, or of its curve on any copy of the exact frame."""
    size = len(law.mixed_frame("end"))
    rows = law.mixed_first_row_of("end")
    worst = 0.0
    for name, label, horizon in names(LABELS):
        worst = max(worst, abs(result.psi(name) - law.mixed_truth(label, "end", horizon)))
        eif = law.mixed_eif(label, "end", horizon)
        for copy in range(copies):
            curve = result.influence_curves[name][copy * size + rows]
            worst = max(worst, float(np.max(np.abs(curve - eif))))
    return worst


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_sdr_at_both_mixed_nodes_equals_the_g_formula_and_its_gateaux_derivative(
    ratio: str,
) -> None:
    """SDR reads the same mixed-node ratio as LTMLE, on both routes, in sample."""
    result = _sdr_fit(ratio)
    assert [result.data.node_kind(time) for time in (1, 2)] == ["mixed", "mixed"]
    assert sdr_deviation(result) < EXACT


@pytest.mark.parametrize("ratio", ["classifier", "density"])
def test_cross_fitted_sdr_at_mixed_nodes_is_exact_when_each_fold_is_the_law(
    ratio: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two folds, each one exact copy: the training-fold recursion reads ``fit_rows``."""
    result = _sdr_fit(ratio, copies=2, monkeypatch=monkeypatch)
    assert sdr_deviation(result, copies=2) < EXACT
