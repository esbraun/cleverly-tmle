"""The density route at a vector treatment with a continuous component, against exact laws.

The node's law factors by the chain rule, and a :class:`~cleverly.interventions.JointPolicy`
induces the density of :mod:`cleverly.interventions.joint_density`: a sum over candidate
categorical cells and piece paths of the preimage density times the declared inverse
derivatives.  Three coarse laws realise their law by integer counts, and saturated learners
on ``exact_bins()`` grids make every exact fit equal the g-formula and its complex-step
Gateaux derivative.

- The point mixed law of ``tests/discrete_law_policy_point.py`` and the two-node mixed law of
  ``tests/discrete_law_longitudinal_mtp.py``: a flag and a dose on ``{0, 1, 2}``.  W13 runs
  every plan of the classifier witnesses on the density route, including the W4 plan that
  sets the flag and gates the dose on the natural flag.  W16 is its many-to-one flag map.
  W17 moves the flag and holds the top dose by a cap.  W18 is a flag map that reads the
  natural dose, with the dose moved.
- The geometric law of ``tests/discrete_law_vector_density.py``: a flag and two doses on
  ``{1, 2, 4, 8}``, read as a point treatment and as one longitudinal node.  W15 doubles
  dose 1 and doubles dose 2 where the natural dose 1 is small, so dose 2's frame, its
  conditioning and its Jacobian all read the preimage of dose 1.
- A law with ``D`` independent of ``F`` given ``W``, for the M1 witness: a flag-only policy
  at a vector node equals the categorical fit of the flag, with the bound binding.

W14 compares the two ratio routes where no bound binds.  W8-density targets a misspecified
outcome regression.  The mutation controls are in
``tests/unit/test_vector_density_mutations.py``.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tests.discrete_law_policy_point as point_law
import tests.discrete_law_vector_density as geo
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, DataError, LongitudinalError, PositivityWarning
from cleverly.interventions import (
    JointPolicy,
    ModifiedPolicy,
    Piece,
    Randomizer,
    Scale,
    Shift,
)
from cleverly.longitudinal import LTMLE, SDR

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_mtp as long_law
from .. import longitudinal_mtp as mtp
from ..discrete_law_longitudinal_multivalue import CellProbabilities
from . import test_longitudinal_mixed_node as long_tests
from . import test_point_vector_treatment as point_tests

#: The mixed laws put every dose on a grid, so R-b4 would refuse each fit; the
#: fixture turns that refusal off and leaves every other check on (X40 step H).
pytestmark = [
    pytest.mark.xdist_group("vector_density_route"),
    pytest.mark.usefixtures("exact_grid_law"),
]

EXACT = 1e-10
INF = np.inf
#: No bound binds on any of these laws: the smallest cell probability is far above it.
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)


class _Grid:
    """The geometric grid, which ``exact_bins`` reads by component name (L2)."""

    edges = geo.EDGES


GRIDS = {"D1": _Grid, "D2": _Grid}


# ------------------------------------------------------------------ declarations
#
# Module-level functions, so a fitted result pickles.


def set_flag(a: Any, h: Any) -> np.ndarray:
    return np.ones(len(a))


def flag_if_dosed(a: Any, h: Any) -> np.ndarray:
    """W18: the flag goes to one where the natural dose is at least one."""
    return np.where(np.asarray(h["D"], dtype=float) >= 1.0, 1, np.asarray(a))


def double(a: Any, h: Any) -> np.ndarray:
    return 2.0 * np.asarray(a, dtype=float)


def halve(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) / 2.0


def half_slope(b: Any, h: Any) -> np.ndarray:
    return np.full(len(b), 0.5)


def small_d1(h: Any) -> np.ndarray:
    """Dose 2 doubles below 6 where the natural dose 1 is at most 2, and nowhere else."""
    return np.where(np.asarray(h["D1"], dtype=float) <= 2.0, 6.0, -INF)


def small_d2(h: Any) -> np.ndarray:
    """Dose 1 doubles below 6 where the natural dose 2 is at most 2, and nowhere else."""
    return np.where(np.asarray(h["D2"], dtype=float) <= 2.0, 6.0, -INF)


def flagged(h: Any) -> np.ndarray:
    """Dose 1 doubles below 6 where the natural flag is one."""
    return np.where(np.asarray(h["F"], dtype=float) == 1.0, 6.0, -INF)


def nudge(a: Any, h: Any) -> np.ndarray:
    """A change of dose that stays inside its bin of the geometric grid."""
    return np.asarray(a, dtype=float) + 0.01


def unnudge(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - 0.01


def unit_slope(b: Any, h: Any) -> np.ndarray:
    return np.ones(len(b))


def flag_if_large_d1(a: Any, h: Any) -> np.ndarray:
    """The flag goes to one where the natural dose 1 is at least 4."""
    return np.where(np.asarray(h["D1"], dtype=float) >= 4.0, 1, np.asarray(a))


def _doubling(name: str, boundary: Any) -> ModifiedPolicy:
    return ModifiedPolicy(
        name,
        pieces=(Piece(-INF, boundary, double, halve, half_slope), Piece(boundary, INF)),
        policy_kind="known",
    )


DOUBLE = Scale(2.0, cap=8.0)


def point_joint(label: str) -> JointPolicy:
    """A point mixed-law plan: the classifier witnesses' plans and W17, W18."""
    if label == "flag set, dose up":
        return JointPolicy(
            label,
            components={
                "F": ModifiedPolicy("flag", apply=set_flag, policy_kind="known"),
                "D": Shift(1.0, cap=2.0),
            },
            policy_kind="known",
        )
    if label == "flag reads dose":
        return JointPolicy(
            label,
            components={
                "F": ModifiedPolicy("flag", apply=flag_if_dosed, policy_kind="known"),
                "D": Shift(1.0, cap=2.0),
            },
            policy_kind="known",
        )
    return point_tests.joint(label)


def geo_joint(label: str) -> JointPolicy:
    """A geometric-law plan, by its oracle label."""
    if label == "natural":
        return JointPolicy(label, components={}, policy_kind="known")
    if label == "ordered":
        return JointPolicy(
            label,
            components={"D1": DOUBLE, "D2": _doubling("d2 if small d1", small_d1)},
            policy_kind="known",
        )
    if label == "reverse ordered":
        return JointPolicy(
            label,
            components={
                "D1": _doubling("d1 if small d2", small_d2),
                "D2": _doubling("d2 if small", 3.0),
            },
            order=("D2", "D1"),
            policy_kind="known",
        )
    if label == "full":
        return JointPolicy(label, components=_full_components(), policy_kind="known")
    if label == "scale both":
        return JointPolicy(label, components={"D1": DOUBLE, "D2": DOUBLE}, policy_kind="known")
    if label == "random":
        return JointPolicy(
            label,
            components={
                "full": _full_components(),
                "scale": {"D1": DOUBLE, "D2": DOUBLE},
            },
            randomizer=Randomizer(("full", "scale"), (0.5, 0.5)),
            policy_kind="known",
        )
    raise KeyError(label)


def _full_components() -> dict[str, Any]:
    return {
        "F": ModifiedPolicy("flag", apply=flag_if_large_d1, policy_kind="known"),
        "D1": _doubling("d1 if flagged", flagged),
        "D2": _doubling("d2 if small d1", small_d1),
    }


GEO_LABELS = ("natural", "ordered", "reverse ordered", "full", "scale both", "random")
#: The point mixed-law plans and the oracle plan each one's truth reads.
POINT_TRUTH = {
    **{label: label for label in point_tests.LABELS},
    "flag set, dose up": "flag gated (shifted flag)",
    "flag reads dose": "flag reads dose",
}
LONG_TRUTH = {
    **{label: label for label in long_tests.LABELS},
    "flag set, dose up": "flag gated (shifted flag)",
    "flag reads dose": "flag reads dose",
}


# ------------------------------------------------------------------ fits


def density_settings(**overrides: Any) -> dict[str, Any]:
    """Saturated learners, one fold, no truncation, the density route."""
    return {
        "outcome_learner": binary_law.CellMeans(),
        "treatment_learner": CellProbabilities(),
        "cross_fit": False,
        "ratio": "density",
        "simultaneous": False,
        "g_bounds": NO_TRUNCATION,
        **overrides,
    }


def fit_point(
    labels: Sequence[str] = tuple(POINT_TRUTH),
    data: CausalData | None = None,
    policies: Sequence[object] | None = None,
    **overrides: Any,
) -> Any:
    """A density-route fit of the point mixed law."""
    declared = [point_joint(label) for label in labels] if policies is None else list(policies)
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return (
            TMLE(policies=declared, **density_settings(**overrides))
            .fit(point_tests.vector_data() if data is None else data)
            .single()
        )


def point_deviation(result: Any, labels: Sequence[str]) -> float:
    rows = point_law.mixed_first_row_of()
    worst = 0.0
    for label in labels:
        oracle = POINT_TRUTH[label]
        estimate = result.estimates[f"ey_policy[{label}]"]
        worst = max(worst, abs(estimate.psi - point_law.mixed_truth(oracle)))
        curve = np.asarray(estimate.influence_curve)[rows]
        worst = max(worst, float(np.max(np.abs(curve - point_law.mixed_eif(oracle)))))
    return worst


def long_regimens(labels: Sequence[str]) -> dict[str, Any]:
    from cleverly.longitudinal import DynamicRegimen

    return {
        label: DynamicRegimen(label, (mtp._joint(label, 1), mtp._joint(label, 2)))
        for label in labels
    }


def fit_long(
    labels: Sequence[str] = tuple(LONG_TRUTH), design: str = "end", **overrides: Any
) -> Any:
    """A density-route fit of the two-node mixed law."""
    with mtp.exact_bins():
        return long_tests.fit(
            tuple(labels), design, regimens=long_regimens(labels), ratio="density", **overrides
        )


def long_deviation(result: Any, labels: Sequence[str], design: str = "end") -> float:
    rows = long_law.mixed_first_row_of(design)
    worst = 0.0
    for name, label, horizon in long_tests.names(tuple(labels), design):
        oracle = LONG_TRUTH[label]
        worst = max(worst, abs(result.psi(name) - long_law.mixed_truth(oracle, design, horizon)))
        curve = result.influence_curves[name][rows]
        eif = long_law.mixed_eif(oracle, design, horizon)
        worst = max(worst, float(np.max(np.abs(curve - eif))))
    return worst


GEO_COLUMNS: dict[str, Any] = {"outcome": "Y", "treatment": ["F", "D1", "D2"], "covariates": ["W"]}
GEO_LONG_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": [["F", "D1", "D2"]],
    "baseline": ["W"],
    "continuous_treatment": ["D1", "D2"],
}


def geo_data(frame: pd.DataFrame | None = None) -> CausalData:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalData.from_frame(
            geo.frame() if frame is None else frame,
            continuous_treatment=["D1", "D2"],
            **GEO_COLUMNS,
        )


def fit_geo_point(labels: Sequence[str] = GEO_LABELS, **overrides: Any) -> Any:
    declared = [geo_joint(label) for label in labels]
    with warnings.catch_warnings(), mtp.exact_bins(GRIDS):
        warnings.simplefilter("ignore")
        return TMLE(policies=declared, **density_settings(**overrides)).fit(geo_data()).single()


def fit_geo_long(labels: Sequence[str] = GEO_LABELS, **overrides: Any) -> Any:
    from cleverly.longitudinal import DynamicRegimen

    settings = {**long_tests.settings(), "ratio": "density", **overrides}
    regimens = {label: DynamicRegimen(label, (geo_joint(label),)) for label in labels}
    with warnings.catch_warnings(), mtp.exact_bins(GRIDS):
        warnings.simplefilter("ignore")
        return LTMLE(regimens, reference=labels[0], **settings).fit(geo.frame(), **GEO_LONG_COLUMNS)


#: The geometric law's oracle label of each declared plan.
GEO_TRUTH = {label: label for label in GEO_LABELS}


def geo_deviation(result: Any, labels: Sequence[str], *, longitudinal: bool) -> float:
    rows = geo.first_row_of()
    worst = 0.0
    for label in labels:
        oracle = GEO_TRUTH[label]
        if longitudinal:
            name = f"ey_regimen[{label}]"
            psi, curve = result.psi(name), result.influence_curves[name][rows]
        else:
            estimate = result.estimates[f"ey_policy[{label}]"]
            psi, curve = estimate.psi, np.asarray(estimate.influence_curve)[rows]
        worst = max(worst, abs(psi - geo.truth(oracle)))
        worst = max(worst, float(np.max(np.abs(curve - geo.eif(oracle)))))
    return worst


# ------------------------------------------------------------------ W13, W16, W17, W18


def test_w13_every_point_policy_on_the_density_route_is_exact() -> None:
    """The classifier witnesses' plans, W4 among them, and W17 and W18."""
    result = fit_point()
    assert point_deviation(result, tuple(POINT_TRUTH)) < EXACT


@pytest.mark.parametrize("design", long_law.MIXED_DESIGNS)
def test_w13_every_longitudinal_plan_on_the_density_route_is_exact(design: str) -> None:
    result = fit_long(design=design)
    assert [result.data.node_kind(time) for time in (1, 2)] == ["mixed", "mixed"]
    assert sorted(result.mechanism.densities) == [(1, "D1"), (2, "D2")]
    assert long_deviation(result, tuple(LONG_TRUTH), design) < EXACT


def test_w13_the_new_plans_are_not_degenerate() -> None:
    """W17 and W18 move the mean on both laws, and the flag move is visible.

    Each differs from the natural course by more than ``5e-3``, and from "dose up", the plan
    with the same dose map and no flag move, by more than ``5e-3``.
    """
    for law, truth in ((point_law, POINT_TRUTH), (long_law, LONG_TRUTH)):
        natural = law.mixed_truth("natural")
        dose_up = law.mixed_truth("dose up")
        for label in ("flag set, dose up", "flag reads dose"):
            value = law.mixed_truth(truth[label])
            assert abs(value - natural) > 5e-3, (law.__name__, label)
            assert abs(value - dose_up) > 5e-3, (law.__name__, label)


def test_w16_a_many_to_one_flag_map_sums_two_candidate_cells() -> None:
    """The W4 flag map sends both flags to one, so two cells reach the same value."""
    result = fit_point(("flag gated",))
    parts = result.nuisance.policies.joint
    assert parts is not None and len(parts.cells) == 2
    induced = parts.induced[:, 0, 0, :] * parts.cell_probabilities
    both = (induced[:, 0] > 0.0) & (induced[:, 1] > 0.0)
    assert both.mean() > 0.25


def test_w17_a_cap_held_dose_reads_the_candidate_cell() -> None:
    """The cap holds the top dose, an identity piece, while the flag moves.

    The dose's law depends on the flag, so the identity piece contributes
    ``g(0, b) / g(1, b)``, not one (plan review rev 3, H1).
    """
    g = point_law.MIXED_G
    assert not np.allclose(
        g[:, :3] / g[:, :3].sum(axis=1, keepdims=True),
        g[:, 3:] / g[:, 3:].sum(axis=1, keepdims=True),
    )
    result = fit_point(("flag set, dose up",))
    assert point_deviation(result, ("flag set, dose up",)) < EXACT
    data = result.data
    held = (np.asarray(data.treatment_components)[:, 1] == 2.0) & (
        np.asarray(data.treatment_components)[:, 0] == 1.0
    )
    ratio = result.nuisance.policies.ratio[:, 0]
    # A held row's ratio is not one: the flag's preimage cell reads the dose there.
    assert np.max(np.abs(ratio[held] - 1.0)) > 0.1


# ------------------------------------------------------------------ W15


def test_w15_the_ordered_plans_are_not_degenerate() -> None:
    truths = {label: geo.truth(label) for label in geo.POLICIES}
    assert abs(truths["ordered"] - truths["ordered (shifted dose 1)"]) >= 1e-2
    assert abs(truths["full"] - truths["full (shifted flag)"]) >= 1e-2
    deterministic = [label for label in truths if label != "random"]
    for first, second in itertools.combinations(deterministic, 2):
        assert abs(truths[first] - truths[second]) > 1e-3, (first, second)


def test_w15_every_geometric_plan_is_exact_at_a_point_treatment() -> None:
    result = fit_geo_point()
    assert geo_deviation(result, GEO_LABELS, longitudinal=False) < EXACT


def test_w15_every_geometric_plan_is_exact_at_a_longitudinal_node() -> None:
    result = fit_geo_long()
    assert result.data.node_kind(1) == "mixed"
    assert sorted(result.mechanism.densities) == [(1, "D1"), (1, "D2")]
    assert geo_deviation(result, GEO_LABELS, longitudinal=True) < EXACT


def test_w15_the_ordered_plan_reads_the_preimage_of_dose_one() -> None:
    """Dose 2's law depends on dose 1, and the plan moves dose 1 on most rows."""
    assert any(
        not np.isclose(geo.g2(w, f, 0, 0), geo.g2(w, f, 1, 0)) for w in range(2) for f in range(2)
    )
    data = geo_data()
    moved = np.asarray(data.treatment_components)[:, 1] < 8.0
    assert moved.mean() > 0.5


# ------------------------------------------------------------------ W14


def test_w14_the_density_and_classifier_routes_agree_where_no_bound_binds() -> None:
    """Both routes are exact on the point and the longitudinal mixed law, so they agree.

    No bound binds: the smallest categorical probability of either law is ``1 / 4``, far
    above ``1e-8``.  Where a bound binds the routes differ by design, because the classifier
    route cannot split the categorical factor out of its ratio (plan rev 3.1, M1).
    """
    labels = point_tests.LABELS
    density = fit_point(labels)
    classifier = fit_point(labels, ratio="classifier", treatment_learner=binary_law.CellMeans())
    assert float(np.min(density.nuisance.policies.joint.cell_probabilities)) >= 0.25
    for name, estimate in density.estimates.items():
        other = classifier.estimates[name]
        assert estimate.psi == pytest.approx(other.psi, abs=EXACT)
        np.testing.assert_allclose(estimate.influence_curve, other.influence_curve, atol=EXACT)
    long_density = fit_long(long_tests.LABELS)
    long_classifier = long_tests.fit(long_tests.LABELS)
    for name in long_density.estimates:
        assert long_density.psi(name) == pytest.approx(long_classifier.psi(name), abs=EXACT)
        np.testing.assert_allclose(
            long_density.influence_curves[name], long_classifier.influence_curves[name], atol=EXACT
        )


# ------------------------------------------------------------------ W8-density


def _misspecified_point(**overrides: Any) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    return {"outcome_learner": LogisticRegression(), **overrides}


def test_w8_density_targeting_on_the_exact_law_reaches_the_truth() -> None:
    """The ratio is exact and the outcome regression is not, so the fluctuation carries it.

    The targeted estimate equals the truth only when the clever covariate is right at the
    observed value and at every assigned value: ``ratio_at`` enters here, and nowhere in W13.
    """
    labels = tuple(POINT_TRUTH)
    result = fit_point(labels, **_misspecified_point())
    for label in labels:
        truth = point_law.mixed_truth(POINT_TRUTH[label])
        assert result.psi(f"ey_policy[{label}]") == pytest.approx(truth, abs=EXACT)
    epsilon = np.asarray(result.fluctuations["mtp"].epsilon, dtype=float)
    assert np.max(np.abs(epsilon)) > 1e-3
    assert result.fluctuations["mtp"].converged


def test_w8_density_targeting_on_the_geometric_law_reaches_the_truth() -> None:
    """The same at two doses, where dose 2's ratio reads the preimage of dose 1."""
    from sklearn.linear_model import LogisticRegression

    labels = ("natural", "ordered", "full")
    result = fit_geo_point(labels, outcome_learner=LogisticRegression())
    for label in labels:
        assert result.psi(f"ey_policy[{label}]") == pytest.approx(geo.truth(label), abs=EXACT)
    epsilon = np.asarray(result.fluctuations["mtp"].epsilon, dtype=float)
    assert np.max(np.abs(epsilon)) > 1e-3


@pytest.mark.parametrize("folds", [1, 3])
def test_w8_density_the_targeted_curve_has_mean_zero_on_a_point_sample(folds: int) -> None:
    from sklearn.linear_model import LogisticRegression

    data = point_tests.vector_data(point_law.mixed_sample(900, 11))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                policies=[point_joint(label) for label in ("natural", "flag gated", "random")],
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                ratio="density",
                simultaneous=False,
                cross_fit=folds > 1,
                n_folds=max(folds, 2),
                random_state=5,
            )
            .fit(data)
            .single()
        )
    epsilon = np.asarray(result.fluctuations["mtp"].epsilon, dtype=float)
    assert np.max(np.abs(epsilon)) > 1e-3
    for estimate in result.estimates.values():
        assert abs(float(np.mean(estimate.influence_curve))) < 1e-10


@pytest.mark.parametrize("folds", [1, 5])
def test_w8_density_a_longitudinal_fit_targets_and_solves_its_score(folds: int) -> None:
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LogisticRegression

    from ..studies.canonical_ltmle import QuasiBinomialGLM

    labels = ("natural", "dose up", "flag gated", "random")
    configuration = long_tests.settings(
        g_bounds=(0.01, 1.0),
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=DummyRegressor(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=folds,
        random_state=0,
        ratio="density",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(mtp.mixed_regimens(labels), reference="natural", **configuration).fit(
            long_law.mixed_wide_sample(900, 11), **long_tests.COLUMNS
        )
    for label in labels[1:]:
        steps = result.fits[label].steps
        assert all(abs(float(step.fluctuation.epsilon[0])) > 1e-3 for step in steps), label
        assert all(step.fluctuation.converged for step in steps), label
        curve = result.influence_curves[f"ey_regimen[{label}]"]
        assert abs(float(np.mean(curve))) < 1e-10, label


# ------------------------------------------------------------------ M1: the bound


#: ``(w, f, d, y)`` of a law with ``D`` independent of ``F`` given ``W``.
_M1_SUPPORT = tuple(itertools.product(range(2), range(2), range(3), range(2)))
_M1_P_F = (0.25, 0.75)
_M1_G_D = ((0.25, 0.5, 0.25), (0.5, 0.25, 0.25))
_M1_Q = (0.25, 0.5, 0.75)


def _m1_q(w: int, f: int, d: int) -> float:
    return _M1_Q[(w + 2 * f + d + f * d) % 3]


def _m1_frame() -> pd.DataFrame:
    rows = []
    for w, f, d, y in _M1_SUPPORT:
        p_f = _M1_P_F[w] if f else 1.0 - _M1_P_F[w]
        q = _m1_q(w, f, d) if y else 1.0 - _m1_q(w, f, d)
        count = 0.5 * p_f * _M1_G_D[w][d] * q * 256
        assert count == round(count) and count > 0
        rows.extend([(float(w), float(f), float(d), float(y))] * round(count))
    return pd.DataFrame(rows, columns=["W", "F", "D", "Y"])


SET_FLAG_ONLY = JointPolicy(
    "set flag",
    components={"F": ModifiedPolicy("flag", apply=set_flag, policy_kind="known")},
    policy_kind="known",
)


def test_m1_a_flag_only_policy_equals_the_categorical_fit_with_the_bound_binding() -> None:
    """The categorical factor is bounded as a categorical treatment's mechanism is.

    With ``D`` independent of ``F`` given ``W``, the ratio of a flag-only policy at the
    vector ``(F, D)`` is the discrete ratio of ``F`` given ``W``, and the categorical fit of
    ``F`` with ``D`` among the covariates estimates the same parameter.  ``g_bounds`` binds:
    ``P(F = 1 | W = 0) = 1 / 4`` is raised to ``0.3``.  The bounded fits agree, and differ
    from the unbounded fit, so the bound reaches the vector ratio.
    """
    from sklearn.linear_model import LogisticRegression

    frame = _m1_frame()
    settings = {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": CellProbabilities(),
        "cross_fit": False,
        "simultaneous": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vector = CausalData.from_frame(
            frame, outcome="Y", treatment=["F", "D"], covariates=["W"], continuous_treatment=["D"]
        )
        flag = CausalData.from_frame(frame, outcome="Y", treatment="F", covariates=["W", "D"])
        categorical_policy = ModifiedPolicy("set flag", apply=set_flag, policy_kind="known")
        results = {}
        for bounds in ((0.3, 0.7), NO_TRUNCATION):
            results[("vector", bounds)] = (
                TMLE(policies=[SET_FLAG_ONLY], ratio="density", g_bounds=bounds, **settings)
                .fit(vector)
                .single()
            )
            results[("flag", bounds)] = (
                TMLE(policies=[categorical_policy], g_bounds=bounds, **settings).fit(flag).single()
            )
    name = "ey_policy[set flag]"
    bound = (0.3, 0.7)
    vector_fit, flag_fit = results[("vector", bound)], results[("flag", bound)]
    assert vector_fit.psi(name) == pytest.approx(flag_fit.psi(name), abs=1e-9)
    np.testing.assert_allclose(
        vector_fit.estimates[name].influence_curve,
        flag_fit.estimates[name].influence_curve,
        atol=1e-9,
    )
    moved = abs(vector_fit.psi(name) - results[("vector", NO_TRUNCATION)].psi(name))
    assert moved > 1e-4


def test_m1_a_longitudinal_flag_only_policy_equals_the_categorical_node_with_the_bound() -> None:
    """At one longitudinal node, the categorical factor joins the bounded running product."""
    from cleverly.longitudinal import DynamicRegimen

    frame = _m1_frame()
    bound = (0.3, 1.0)
    common = {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": CellProbabilities(),
        "n_folds": 1,
        "simultaneous": False,
    }
    flag_policy = ModifiedPolicy("set flag", apply=set_flag, policy_kind="known")
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        fits = {}
        for bounds in (bound, (1e-8, 1.0)):
            fits[("vector", bounds)] = LTMLE(
                {"set flag": DynamicRegimen("set flag", (SET_FLAG_ONLY,))},
                ratio="density",
                g_bounds=bounds,
                **common,
            ).fit(
                frame,
                outcome="Y",
                treatment=[["F", "D"]],
                baseline=["W"],
                continuous_treatment=["D"],
            )
            fits[("flag", bounds)] = LTMLE(
                {"set flag": DynamicRegimen("set flag", (flag_policy,))},
                g_bounds=bounds,
                **common,
            ).fit(frame, outcome="Y", treatment=["F"], baseline=["W", "D"])
    name = "ey_regimen[set flag]"
    vector_fit, flag_fit = fits[("vector", bound)], fits[("flag", bound)]
    assert vector_fit.psi(name) == pytest.approx(flag_fit.psi(name), abs=1e-9)
    np.testing.assert_allclose(
        vector_fit.influence_curves[name], flag_fit.influence_curves[name], atol=1e-9
    )
    unbounded = fits[("vector", (1e-8, 1.0))].influence_curves[name]
    assert float(np.max(np.abs(vector_fit.influence_curves[name] - unbounded))) > 1e-4


# ------------------------------------------------------------------ B1: storage and provenance


def test_b1_the_joint_parts_store_values_per_row_and_round_trip() -> None:
    """``dumps``/``loads``, ``subset`` and ``retarget`` carry the factors of the joint law.

    The parts hold ``(n, E, B, K)`` factor values and ``(n, E)`` observed values, never an
    ``(n, B)`` matrix per counterfactual design.
    """
    from sklearn.linear_model import LogisticRegression

    labels = ("natural", "flag gated", "random")
    result = fit_point(labels, outcome_learner=LogisticRegression())
    policies = result.nuisance.policies
    parts = policies.joint
    n = result.data.n
    count = policies.n_components
    assert parts.induced.shape == (n, count + 1, count, len(parts.cells))
    assert parts.observed.shape == (n, count + 1)
    restored = loads(dumps(result))
    for name, estimate in result.estimates.items():
        assert restored.psi(name) == estimate.psi
    again = restored.nuisance.policies.joint
    np.testing.assert_array_equal(again.induced, parts.induced)
    np.testing.assert_array_equal(again.cell_probabilities, parts.cell_probabilities)
    np.testing.assert_array_equal(
        again.densities["D"].bin_probabilities, parts.densities["D"].bin_probabilities
    )
    index = np.concatenate([np.arange(n)[::-1], np.arange(0, n, 5)])
    part = policies.subset(index)
    np.testing.assert_array_equal(part.joint.induced, parts.induced[index])
    np.testing.assert_array_equal(part.joint.targets, parts.targets[index])
    bound = (0.3, 0.7)
    np.testing.assert_array_equal(
        part.design_at(None, bound), policies.design_at(None, bound)[index]
    )
    estimator = TMLE(
        policies=[point_joint(label) for label in labels],
        **density_settings(outcome_learner=LogisticRegression()),
    )
    estimates, _ = estimator.retarget(
        result.data, result.nuisance, estimands=result.config.estimands
    )
    for name, estimate in result.estimates.items():
        assert estimates[name].psi == pytest.approx(estimate.psi, abs=1e-12)


def test_b1_a_bound_moves_the_point_ratio_through_the_categorical_factor() -> None:
    """``design_at`` bounds the categorical factor, in the numerator and the denominator."""
    result = fit_point(("flag gated",))
    policies = result.nuisance.policies
    np.testing.assert_array_equal(policies.design_at(None), policies.design)
    np.testing.assert_allclose(policies.design_at(None, NO_TRUNCATION), policies.design)
    # ``P(F = 1 | W) = 1 / 2`` on this law, so the bound binds at ``0.6``.
    assert np.max(np.abs(policies.design_at(None, (0.6, 0.9)) - policies.design)) > 0.1


def test_b1_each_counterfactual_prediction_comes_from_the_held_out_fold() -> None:
    """At the observed design a counterfactual prediction is the out-of-fold density.

    Row ``i`` of every design is predicted by the model of the fold that held ``i`` out, so a
    design equal to the observed one returns the stored out-of-fold density bit for bit.  A
    model of every row would not: the nonzero gap below is that model's distance.
    """
    from sklearn.linear_model import LogisticRegression

    from cleverly.learners import Folds
    from cleverly.learners.density import fit_conditional_density

    rng = np.random.default_rng(3)
    w = rng.normal(size=(300, 2))
    a = w[:, 0] + rng.normal(size=300)
    folds = Folds(np.arange(300) % 3, 3)
    weights = np.ones(300)
    density, _, predicted = fit_conditional_density(
        LogisticRegression(),
        w,
        a,
        weights,
        folds,
        n_bins=8,
        predict_designs={"same": (w, a), "pair": (w[::-1], np.column_stack([a, a + 0.5]))},
    )
    np.testing.assert_array_equal(predicted["same"], density.density_at(a))
    assert predicted["pair"].shape == (300, 2)
    everyone, _ = fit_conditional_density(
        LogisticRegression(), w, a, weights, Folds.single(300), n_bins=8
    )
    assert float(np.max(np.abs(everyone.density_at(a) - predicted["same"]))) > 1e-3


def test_b1_the_joint_observed_factor_is_the_product_of_the_stored_densities() -> None:
    """With three folds, the evaluator's observed factor reads the same out-of-fold models."""
    from sklearn.linear_model import LogisticRegression

    data = geo_data(geo.sample(600, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                policies=[geo_joint("ordered")],
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                ratio="density",
                cross_fit=True,
                n_folds=3,
                density_bins=4,
                random_state=1,
            )
            .fit(data)
            .single()
        )
    parts = result.nuisance.policies.joint
    natural = np.asarray(data.treatment_components)
    product = parts.densities["D1"].density_at(natural[:, 1]) * parts.densities["D2"].density_at(
        natural[:, 2]
    )
    np.testing.assert_array_equal(parts.observed[:, 0], product)


# ------------------------------------------------------------------ H2, M7, order


def on_eighths(h: Any) -> np.ndarray:
    return np.isclose(np.asarray(h["D1"], dtype=float) * 8.0 % 1.0, 0.0)


def upper_at_eighths(h: Any) -> np.ndarray:
    """The identity piece ends at 5 on an observed ``D1`` and at 6 off the eighths."""
    return np.where(on_eighths(h), 5.0, 6.0)


def lower_at_eighths(h: Any) -> np.ndarray:
    """The moved piece starts at 5 on an observed ``D1`` and at 4 off the eighths."""
    return np.where(on_eighths(h), 5.0, 4.0)


def up_one(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + 1.0


def down_one(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - 1.0


def unit(b: Any, h: Any) -> np.ndarray:
    return np.ones(len(b))


def _overlap_data() -> CausalData:
    rng = np.random.default_rng(8)
    n = 240
    frame = pd.DataFrame(
        {
            "W": rng.integers(0, 2, n).astype(float),
            "F": rng.integers(0, 2, n).astype(float),
            "D1": rng.integers(0, 64, n) / 8.0,
            "D2": rng.uniform(0.0, 8.0, n),
            "Y": rng.integers(0, 2, n).astype(float),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CausalData.from_frame(frame, continuous_treatment=["D1", "D2"], **GEO_COLUMNS)


def test_h2_pieces_that_overlap_at_a_preimage_of_dose_one_are_refused() -> None:
    """The pieces partition ``D2`` at every observed ``D1`` and overlap at ``D1 - 0.3``.

    The check of the observed frames passes.  The density formula reads dose 2's pieces at
    the preimage of dose 1, where both pieces hold ``[4, 6)``; the fit refuses there.
    """
    from sklearn.linear_model import LogisticRegression

    pieces = ModifiedPolicy(
        "overlap off the eighths",
        pieces=(
            Piece(-INF, upper_at_eighths),
            Piece(lower_at_eighths, INF, up_one, down_one, unit),
        ),
        policy_kind="known",
    )
    policy = JointPolicy(
        "overlap", components={"D1": Shift(0.3, cap=None), "D2": pieces}, policy_kind="known"
    )
    learners = {
        "outcome_learner": LogisticRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The classifier route reads the observed frames only, and accepts the pieces.
        TMLE(policies=[policy], ratio="classifier", **learners).fit(_overlap_data())
        with pytest.raises(DataError, match=r"do not partition the dose at \d+ preimage"):
            TMLE(policies=[policy], ratio="density", **learners).fit(_overlap_data())


def reads_d2_by_get(h: Any) -> np.ndarray:
    """A bound that reads ``D2`` through ``get``, with a default."""
    return np.where(np.asarray(h.get("D2", 0.0), dtype=float) <= 2.0, 6.0, -INF)


def test_m7_a_read_through_get_is_refused_and_names_the_order() -> None:
    """``h.get("D2", 0)`` is a read: the frame proxy refuses it instead of the default."""
    policy = JointPolicy(
        "get",
        components={"D1": _doubling("d1 by get", reads_d2_by_get), "D2": DOUBLE},
        policy_kind="known",
    )
    with pytest.raises(DataError, match=r"'D1' reads \['D2'\].*order=\('D2', 'D1'\)"):
        _fit_geo_policy(policy)
    ordered = JointPolicy(
        "get",
        components={"D1": _doubling("d1 by get", reads_d2_by_get), "D2": DOUBLE},
        order=("D2", "D1"),
        policy_kind="known",
    )
    _fit_geo_policy(ordered)


def reads_d2_and_catches(h: Any) -> np.ndarray:
    """A bound that reads ``D2`` and swallows every exception, falling back to 6."""
    try:
        d2 = np.asarray(h["D2"], dtype=float)
    except Exception:
        return np.full(len(h), 6.0)
    return np.where(d2 <= 2.0, 6.0, -INF)


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_m7_a_map_that_catches_its_refused_read_is_still_refused(ratio: str) -> None:
    """The proxy records the read, and the read is refused after the map returns (LOW-7).

    Before, ``except Exception`` swallowed the refusal and the map silently used its
    fallback on every row.
    """
    policy = JointPolicy(
        "catches",
        components={"D1": _doubling("d1 catches", reads_d2_and_catches), "D2": DOUBLE},
        policy_kind="known",
    )
    with pytest.raises(DataError, match=r"'D1' reads \['D2'\].*order=\('D2', 'D1'\)"):
        _fit_geo_policy(policy, ratio=ratio)


def test_m7_the_frame_proxy_refuses_every_access_by_name() -> None:
    from cleverly.interventions.policy import ForbiddenRead, GuardedFrame

    frame = GuardedFrame(pd.DataFrame({"W": [0.0], "D2": [1.0]}), frozenset({"D2"}))
    np.testing.assert_array_equal(frame["W"], [0.0])
    for access in (
        lambda: frame["D2"],
        lambda: frame.get("D2", 0.0),
        lambda: "D2" in frame,
        lambda: frame.D2,
        lambda: frame[["W", "D2"]],
    ):
        with pytest.raises(ForbiddenRead, match="'D2'"):
            access()
    assert frame.violations == ["D2"] * 5
    assert issubclass(ForbiddenRead, DataError)


def reads_d1(h: Any) -> np.ndarray:
    return np.where(np.asarray(h["D1"], dtype=float) <= 2.0, 6.0, -INF)


def reads_d2(h: Any) -> np.ndarray:
    return np.where(np.asarray(h["D2"], dtype=float) <= 2.0, 6.0, -INF)


def test_a_cycle_of_dose_reads_is_refused_with_its_reason() -> None:
    policy = JointPolicy(
        "cycle",
        components={"D1": _doubling("d1", reads_d2), "D2": _doubling("d2", reads_d1)},
        policy_kind="known",
    )
    with pytest.raises(DataError, match=r"read each other in a cycle.*no checkable declaration"):
        _fit_geo_policy(policy)


def _fit_geo_policy(
    policy: JointPolicy, frame: pd.DataFrame | None = None, **overrides: Any
) -> Any:
    with warnings.catch_warnings(), mtp.exact_bins(GRIDS):
        warnings.simplefilter("ignore")
        return (
            TMLE(policies=[policy], **density_settings(**overrides)).fit(geo_data(frame)).single()
        )


# ------------------------------------------------------------------ L3, L5


def test_l3_the_resolution_warning_reads_each_dose() -> None:
    """Dose 2 moves across the grid; dose 1 moves within its bins and is reported."""
    policy = JointPolicy(
        "tiny", components={"D1": Shift(0.01, cap=None), "D2": DOUBLE}, policy_kind="known"
    )
    with mtp.exact_bins(GRIDS), pytest.warns(UserWarning, match="across a bin edge"):
        TMLE(policies=[policy], **density_settings()).fit(geo_data()).single()


def test_l3_the_resolution_warning_counts_only_the_rows_a_gated_map_moves() -> None:
    """A gated map is read on the rows it moves, and the warning names the dose.

    ``d1 if flagged`` doubles dose 1 below 6 where the flag is one: 40.6% of the rows. Each
    doubling crosses an edge of the geometric grid, so the moved rows all cross. Counted over
    every row the share is 0.406, below the 0.5 threshold, which is the misfire this pins.
    The warning changes no number: the fit with it silenced is the same fit.
    """
    gated = JointPolicy(
        "gated", components={"D1": _doubling("d1 if flagged", flagged)}, policy_kind="known"
    )
    frame = geo.frame()
    moved = (frame["F"] == 1.0) & (frame["D1"] < 6.0)
    assert moved.mean() == pytest.approx(0.40625)
    with mtp.exact_bins(GRIDS), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loud = TMLE(policies=[gated], **density_settings()).fit(geo_data()).single()
    assert not [w for w in caught if "across a bin edge" in str(w.message)]
    with mtp.exact_bins(GRIDS), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quiet = TMLE(policies=[gated], **density_settings()).fit(geo_data()).single()
    for name, estimate in loud.estimates.items():
        assert estimate.psi == quiet.estimates[name].psi
        np.testing.assert_array_equal(
            estimate.influence_curve, quiet.estimates[name].influence_curve
        )
    # A gated map that moves its rows within a bin still warns, over those rows only.
    tiny = JointPolicy(
        "tiny gated",
        components={
            "D1": ModifiedPolicy(
                "nudge",
                pieces=(Piece(-INF, flagged, nudge, unnudge, unit_slope), Piece(flagged, INF)),
                policy_kind="known",
            )
        },
        policy_kind="known",
    )
    pattern = rf"changes 'D1' on {int(moved.sum())} rows, and moves only 0\.0% of them"
    with mtp.exact_bins(GRIDS), pytest.warns(UserWarning, match=pattern):
        TMLE(policies=[tiny], **density_settings()).fit(geo_data()).single()


def test_l5_a_value_of_estimated_density_zero_warns_on_both_routes() -> None:
    from cleverly.longitudinal import DynamicRegimen

    away = JointPolicy("away", components={"D1": Shift(20.0, cap=None)}, policy_kind="known")
    pattern = "estimated density of the node is zero"
    with mtp.exact_bins(GRIDS), pytest.warns(PositivityWarning, match=pattern):
        TMLE(policies=[away], **density_settings()).fit(geo_data()).single()
    settings = {**long_tests.settings(), "ratio": "density"}
    with mtp.exact_bins(GRIDS), pytest.warns(PositivityWarning, match=pattern):
        LTMLE({"away": DynamicRegimen("away", (away,))}, **settings).fit(
            geo.frame(), **GEO_LONG_COLUMNS
        )


# ------------------------------------------------------------------ longitudinal replay


def test_the_longitudinal_truncation_replay_reaches_the_categorical_factor() -> None:
    """The fitted bound reproduces the fit; a binding bound moves it through ``g_C``.

    The outcome regression is misspecified, so the fluctuation, and with it the bound,
    moves the estimate.  On the classifier route the replay moves nothing, because the
    whole ratio sits in the numerator there.
    """
    from sklearn.dummy import DummyRegressor

    from cleverly.longitudinal.estimator import longitudinal_truncation_curve

    from ..studies.canonical_ltmle import QuasiBinomialGLM

    frame = long_law.mixed_sample(800, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = long_tests.fit(
            ("natural", "flag gated"),
            frame=frame,
            g_bounds=(0.01, 1.0),
            ratio="density",
            outcome_learner=QuasiBinomialGLM(),
            pseudo_learner=DummyRegressor(),
        )
        curve = longitudinal_truncation_curve(
            result, [(0.01, 1.0), (0.45, 1.0)], estimands=["ey_regimen[flag gated]"]
        ).set_index("is_fitted_bound")
    assert curve.loc[True, "delta_from_fitted"] == 0.0
    assert abs(curve.loc[False, "delta_from_fitted"]) > 1e-4


def test_a_longitudinal_density_result_round_trips_and_bootstraps() -> None:
    frame = long_law.mixed_sample(500, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = long_tests.fit(
            ("natural", "dose up", "random"),
            frame=frame,
            ratio="density",
            n_bootstrap=3,
            random_state=4,
        )
    restored = loads(dumps(result))
    for name in result.estimates:
        assert restored.psi(name) == result.psi(name)
    assert sorted(restored.mechanism.densities) == [(1, "D1"), (2, "D2")]
    assert result.bootstrap is not None and result.bootstrap.n_failed == 0
    for draws in result.bootstrap.draws.values():
        assert np.all(np.isfinite(draws))
    # Each mixed node reports the log loss of its joint law: the categorical factor times
    # every dose's density at the observed value.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = result.diagnostics.nuisance_models()
    treatment = [row for row in report.rows if row.role == "treatment"]
    assert [row.time for row in treatment] == [1, 2]
    for row in treatment:
        assert row.model.kind == "conditional density"
        assert np.isfinite(row.loss) and row.loss > 0.0


# ------------------------------------------------------------------ M4: memory


def _two_doses() -> tuple[Any, Any, Any]:
    from sklearn.linear_model import LogisticRegression

    from cleverly.learners import Folds
    from cleverly.learners.density import _grid, _table_size

    rng = np.random.default_rng(4)
    w = rng.normal(size=(400, 2))
    doses = [rng.normal(size=400), rng.normal(size=400)]
    learner = LogisticRegression()
    tables = []
    for k, dose in enumerate(doses):
        grid = _grid(dose, None, 10, None, None)
        records, columns, _ = _table_size(learner, w.shape[1] + k, grid, None)
        tables.append(records * columns * 8)
    components = [(w.shape[1] + k, dose) for k, dose in enumerate(doses)]
    return learner, components, (tables, Folds(np.arange(400) % 2, 2))


def test_m4_the_preflight_counts_the_earlier_doses_and_the_parallel_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node's peak is the earlier doses' bin probabilities plus one table per worker.

    Each table fits alone in the memory below, and the second dose's table beside the first
    dose's retained ``(n, B)`` probabilities does not.  The same tables with two folds fitted
    at once do not fit either.  A check of one table at a time passes both.
    """
    from cleverly.exceptions import MethodConfigurationError
    from cleverly.learners import density as density_module
    from cleverly.learners.density import preflight_conditional_densities

    learner, components, (tables, folds) = _two_doses()
    retained = 400 * 10 * 8
    free = max(tables) + retained // 2
    monkeypatch.setattr(density_module, "available_memory", lambda: free)
    single = density_module.Folds.single(400)
    with pytest.raises(MethodConfigurationError, match="already held"):
        preflight_conditional_densities(learner, components, single, n_bins=10, names=["A", "B"])
    preflight_conditional_densities(learner, components[:1], single, n_bins=10)
    monkeypatch.setattr(density_module, "available_memory", lambda: max(tables) + retained * 2)
    preflight_conditional_densities(learner, components, single, n_bins=10)
    preflight_conditional_densities(learner, components, folds, n_bins=10, n_jobs=1)
    with pytest.raises(MethodConfigurationError, match="2 fold fit"):
        preflight_conditional_densities(learner, components, folds, n_bins=10, n_jobs=2)


def test_m4_a_vector_fit_refuses_before_any_learner(monkeypatch: pytest.MonkeyPatch) -> None:
    from cleverly.estimators import _nuisance
    from cleverly.exceptions import MethodConfigurationError
    from cleverly.learners import density as density_module

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner was fitted before the refusal")

    monkeypatch.setattr(density_module, "available_memory", lambda: 1024)
    monkeypatch.setattr(_nuisance, "cross_fit_predictions", refuse)
    monkeypatch.setattr(density_module, "fit_learner", refuse)
    with pytest.raises(MethodConfigurationError, match="pooled hazard design"):
        _fit_geo_policy(geo_joint("full"))


# ------------------------------------------------------------------ cell support (review HIGH-1)


class _NoFit:
    """A learner that fails if any fit runs, so a refusal is shown to come first."""

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    def set_params(self, **params: Any) -> _NoFit:
        return self

    def fit(self, *args: Any, **kwargs: Any) -> _NoFit:
        raise AssertionError("a learner ran before the cell-support refusal")

    def predict_proba(self, x: Any) -> Any:
        raise AssertionError("a learner ran before the cell-support refusal")


def _lone_flag(frame: pd.DataFrame, flag: str) -> pd.DataFrame:
    """The frame with ``flag = 1`` on exactly one row, which every split leaves out once."""
    out = frame.copy()
    out[flag] = 0.0
    out.loc[out.index[0], flag] = 1.0
    return out


def test_a_point_cell_one_unit_holds_is_refused_before_any_learner() -> None:
    """The density route's g_C needs every cell in each training complement.

    Without the refusal the fit returned, and the ratio at the lone row was exactly zero:
    the complement's multinomial gave the cell probability zero (review HIGH-1).
    """
    frame = _lone_flag(point_law.mixed_sample(400, 3), "F")
    data = point_tests.vector_data(frame)
    estimator = TMLE(
        policies=[point_joint("dose up")],
        outcome_learner=_NoFit(),
        treatment_learner=_NoFit(),
        ratio="density",
        cross_fit=True,
        n_folds=5,
        random_state=0,
    )
    with pytest.raises(DataError) as caught:
        estimator.fit(data)
    message = str(caught.value)
    assert "needs each cell of the categorical components of 'F+D'" in message
    assert "cell (F=1) appears in 1 row(s)" in message
    assert "ratio='classifier'" in message


def test_a_point_cell_that_one_training_fold_lacks_is_refused() -> None:
    """Two rows of a cell in one fold leave that fold's complement without the cell."""
    from cleverly import SplitPlan
    from cleverly.learners import random_partition

    drawn = random_partition(400, 4, seed=0)
    first = np.flatnonzero(np.asarray(drawn.assignment) == 0)[:2]
    frame = point_law.mixed_sample(400, 3)
    frame["F"] = 0.0
    frame.loc[frame.index[first], "F"] = 1.0
    data = point_tests.vector_data(frame)
    plan = SplitPlan.from_folds([drawn])
    estimator = TMLE(
        policies=[point_joint("dose up")],
        outcome_learner=_NoFit(),
        treatment_learner=_NoFit(),
        ratio="density",
        split_plan=plan,
    )
    with pytest.raises(DataError, match=r"contains no row in cell \(F=1\)"):
        estimator.fit(data)


def test_the_classifier_route_fits_no_cell_probabilities_and_is_not_refused() -> None:
    """The cell check belongs to the density route; the classifier route still fits."""
    from sklearn.linear_model import LogisticRegression

    frame = _lone_flag(point_law.mixed_sample(400, 3), "F")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TMLE(
            policies=[point_joint("dose up")],
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            ratio="classifier",
            cross_fit=True,
            n_folds=5,
            random_state=0,
        ).fit(point_tests.vector_data(frame))
    assert np.isfinite(result.single().psi("ey_policy[dose up]"))


@pytest.mark.parametrize("estimator", [LTMLE, SDR])
def test_a_longitudinal_mixed_cell_a_training_fold_lacks_is_refused(estimator: Any) -> None:
    """At a mixed node the cells of the categorical components are levels of g_C."""
    frame = _lone_flag(long_law.mixed_sample(400, 3), "F2")
    configuration = long_tests.settings(
        outcome_learner=_NoFit(),
        pseudo_learner=_NoFit(),
        treatment_learner=_NoFit(),
        censoring_learner=_NoFit(),
        n_folds=5,
        random_state=0,
        ratio="density",
    )
    with pytest.raises(LongitudinalError) as caught:
        estimator(mtp.mixed_regimens(("natural", "dose up")), **configuration).fit(
            frame, **long_tests.COLUMNS
        )
    message = str(caught.value)
    assert "is missing level(s) ['cell (F2=1.0)']" in message
    assert "ratio='classifier'" in message


# ------------------------------------------------------------------ design memory (review HIGH-2)


def _design_heavy_node(n: int = 1200, width: int = 12, copies: int = 1) -> tuple[Any, ...]:
    """A node of six cells and two capped doses, under three policies that set the flag.

    Every policy moves both doses with a cap, so dose 2's density is asked for at many
    counterfactual designs: candidate cell times the pieces of dose 1.  ``copies`` repeats
    each policy under a new name, which asks for the same designs again.
    """
    from cleverly.interventions.joint_density import JointBranch, JointEvaluation, NodeLayout
    from cleverly.interventions.policy import JointComponent, joint_assignments, joint_branches

    rng = np.random.default_rng(0)
    natural = np.column_stack(
        [rng.integers(0, 2, n), rng.integers(0, 3, n), rng.normal(size=n), rng.normal(size=n)]
    ).astype(float)
    components = (
        JointComponent("F", (0, 1)),
        JointComponent("G", (0, 1, 2)),
        JointComponent("D1"),
        JointComponent("D2"),
    )

    def frame_with(extra: Any) -> pd.DataFrame:
        return pd.DataFrame({key: np.asarray(value) for key, value in extra.items()})

    policies = [
        JointPolicy(
            f"p{i}-{copy}",
            components={
                "F": SET_FLAG,
                "D1": Shift(0.2 * (i + 1), cap=1.5),
                "D2": Shift(0.1 * (i + 1), cap=1.5),
            },
            policy_kind="known",
        )
        for i in range(3)
        for copy in range(copies)
    ]
    everyone = np.ones(n, dtype=bool)
    layout = NodeLayout(
        components, natural, everyone, frame_with, rng.normal(size=(n, width)), "node"
    )
    branches, assigned = [], []
    for policy in policies:
        ((_, mapping),) = joint_branches(policy)
        branches.append(JointBranch(policy.name, mapping, policy.order))
        ((_, values),) = joint_assignments(
            policy, components, natural, frame_with, everyone, where="node"
        )
        assigned.append(values)
    every = tuple(range(len(branches)))
    evaluations = [JointEvaluation(natural, every)] + [
        JointEvaluation(values, every) for values in assigned
    ]
    return layout, branches, evaluations


SET_FLAG = ModifiedPolicy("flag", apply=set_flag, policy_kind="known")


class _Stop(Exception):
    pass


def _record_preflight(monkeypatch: pytest.MonkeyPatch, **settings: Any) -> dict[str, Any]:
    """Run the evaluator up to its preflight and return what the preflight is asked."""
    from sklearn.linear_model import LogisticRegression

    from cleverly.interventions import joint_density as jd
    from cleverly.learners import Folds

    recorded: dict[str, Any] = {}

    def record(*args: Any, **kwargs: Any) -> None:
        recorded.update(args=args, kwargs=kwargs)
        raise _Stop

    monkeypatch.setattr(jd, "preflight_conditional_densities", record)
    layout, branches, evaluations = _design_heavy_node(**settings)
    with pytest.raises(_Stop):
        jd.evaluate_joint_density(
            layout,
            branches,
            evaluations,
            learner=LogisticRegression(),
            weights=np.ones(layout.n),
            folds=Folds(np.arange(layout.n) % 2, 2),
            n_bins=8,
        )
    monkeypatch.undo()
    return recorded


def test_identical_counterfactual_designs_are_stored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second copy of every policy asks for the same designs and adds none.

    Before the review a design was keyed by a fresh serial per term, so each copy stored
    every design again: 1,069 MiB at n = 20,000 in the review's probe.
    """
    once = _record_preflight(monkeypatch)["kwargs"]["predictions"]
    twice = _record_preflight(monkeypatch, copies=2)["kwargs"]["predictions"]
    for single, double in zip(once, twice, strict=True):
        # ``shipped`` is the stored columns plus the values, and ``kept`` the values.
        # The stored columns do not grow; only the values asked at them do.
        assert double.shipped - double.kept == single.shipped - single.kept
        assert double.kept > single.kept


def test_the_preflight_refuses_designs_the_tables_alone_would_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counterfactual designs count toward the peak, and the refusal names a remedy.

    The memory below fits every long table beside the earlier doses' bin probabilities,
    and does not fit the designs and values of the doses besides.  The same check without
    the designs passes, which is what the preflight did before the review.
    """
    from cleverly.exceptions import MethodConfigurationError
    from cleverly.learners import density as density_module
    from cleverly.learners.density import _grid, _table_size, preflight_conditional_densities

    recorded = _record_preflight(monkeypatch)
    learner, components, folds = recorded["args"]
    kwargs = dict(recorded["kwargs"])
    predictions = kwargs.pop("predictions")
    tables, retained = [], 0
    for n_covariates, dose in components:
        grid = _grid(np.asarray(dose), None, 8, None, None)
        records, columns, _ = _table_size(learner, n_covariates, grid, None)
        tables.append(records * columns * 8 + retained)
        retained += dose.size * grid.n_total * 8
    designs = max(cost.peak(1) for cost in predictions)
    assert designs > 0
    free = max(tables) + designs // 2
    monkeypatch.setattr(density_module, "available_memory", lambda: free)
    preflight_conditional_densities(learner, components, folds, **kwargs)
    with pytest.raises(MethodConfigurationError) as caught:
        preflight_conditional_densities(
            learner, components, folds, predictions=predictions, **kwargs
        )
    message = str(caught.value)
    assert "of counterfactual designs" in message
    assert "declare fewer policies or pieces at the node" in message


def test_a_design_heavy_fit_refuses_before_any_learner(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluator itself refuses, before the categorical factor or a hazard is fitted."""
    from sklearn.linear_model import LogisticRegression

    from cleverly.estimators import _nuisance
    from cleverly.exceptions import MethodConfigurationError
    from cleverly.interventions import joint_density as jd
    from cleverly.learners import Folds
    from cleverly.learners import density as density_module

    recorded = _record_preflight(monkeypatch)
    designs = max(cost.peak(1) for cost in recorded["kwargs"]["predictions"])

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a learner was fitted before the refusal")

    monkeypatch.setattr(density_module, "available_memory", lambda: designs - 1)
    monkeypatch.setattr(_nuisance, "cross_fit_predictions", refuse)
    monkeypatch.setattr(density_module, "fit_learner", refuse)
    layout, branches, evaluations = _design_heavy_node()
    with pytest.raises(MethodConfigurationError, match="counterfactual designs"):
        jd.evaluate_joint_density(
            layout,
            branches,
            evaluations,
            learner=LogisticRegression(),
            weights=np.ones(layout.n),
            folds=Folds(np.arange(layout.n) % 2, 2),
            n_bins=8,
        )


def test_parallel_workers_each_count_a_copy_of_the_designs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With two fold fits at once, each worker holds the stored columns and the values."""
    predictions = _record_preflight(monkeypatch)["kwargs"]["predictions"]
    for cost in predictions:
        assert cost.shipped > 0
        assert cost.peak(2) == cost.held + 2 * cost.shipped + 2 * cost.rows
        assert cost.peak(1) == cost.held + cost.rows


# ------------------------------------------------------------------ coverage (review LOW-2)


def on_grid_d1(h: Any) -> np.ndarray:
    """3 where the natural dose 1 is a grid dose, else 1.5: a gap off the grid."""
    d1 = np.asarray(h["D1"], dtype=float)
    return np.where(np.isin(d1, geo.DOSES), 3.0, 1.5)


GAP_OFF_THE_GRID = JointPolicy(
    "gap off the grid",
    components={
        "D1": Shift(0.5, cap=None),
        "D2": ModifiedPolicy(
            "gap", pieces=(Piece(-INF, on_grid_d1), Piece(3.0, INF)), policy_kind="known"
        ),
    },
    policy_kind="known",
)


def test_a_piece_gap_a_counterfactual_frame_reaches_is_refused() -> None:
    """Dose 2's pieces partition its range at every observed dose 1, and not off the grid.

    Dose 1's shift reads dose 2's density at ``beta_1 = b_1 - 0.5``, which is off the grid,
    and there ``[1.5, 3)`` holds no piece while dose 2 takes the value 2.  Without the check
    the fit returned, with that mass dropped from the induced density.
    """
    with pytest.raises(DataError, match=r"no piece holds the observed dose 2 on \d+ frame"):
        _fit_geo_policy(GAP_OFF_THE_GRID)


def test_the_same_pieces_without_the_gap_fit() -> None:
    """Closing the gap with a third identity piece fits: the check reads the gap alone."""
    closed = JointPolicy(
        "closed",
        components={
            "D1": Shift(0.5, cap=None),
            "D2": ModifiedPolicy(
                "closed",
                pieces=(Piece(-INF, on_grid_d1), Piece(on_grid_d1, 3.0), Piece(3.0, INF)),
                policy_kind="known",
            ),
        },
        policy_kind="known",
    )
    result = _fit_geo_policy(closed)
    assert np.isfinite(result.psi("ey_policy[closed]"))


# ------------------------------------------------------------------ monotonicity (review LOW-5)


def double_on_grid(a: Any, h: Any) -> np.ndarray:
    """Doubles dose 2 where dose 1 is a grid dose, and is flat off the grid."""
    on = np.isin(np.asarray(h["D1"], dtype=float), geo.DOSES)
    return np.where(on, 2.0 * np.asarray(a, dtype=float), 3.0)


def test_a_map_flat_only_on_a_counterfactual_frame_names_the_conservative_reading() -> None:
    """The monotonicity check reads counterfactual frames over the observed dose range.

    Dose 1's shift reads dose 2's map at ``beta_1`` off the grid, where the map is flat.
    The refusal says which frames it read and that the reading is conservative.
    """
    policy = JointPolicy(
        "flat off the grid",
        components={
            "D1": Shift(0.5, cap=None),
            "D2": ModifiedPolicy(
                "flat",
                pieces=(Piece(-INF, 6.0, double_on_grid, halve, half_slope), Piece(6.0, INF)),
                policy_kind="known",
            ),
        },
        policy_kind="known",
    )
    with pytest.raises(CapabilityError) as caught:
        _fit_geo_policy(policy)
    message = str(caught.value)
    assert "is not strictly monotone there" in message
    assert "preimages of the earlier doses" in message
    assert "the check is conservative there" in message


# ------------------------------------------------------------------ the zero contract (review LOW-3)


def test_the_lookup_is_exactly_zero_outside_the_outer_edges() -> None:
    """The contract that lets the declaration checks skip frames outside the bin range."""
    from cleverly.learners.density import _lookup

    edges = np.array([0.0, 1.0, 3.0])
    probabilities = np.array([[0.25, 0.75], [0.5, 0.5]])
    values = np.array([[-1e-12, 0.0, 3.0, 3.0 + 1e-12], [np.nan, 0.5, 2.0, np.inf]])
    found = _lookup(probabilities, edges, values)
    assert found[0, 0] == 0.0 and found[0, 3] == 0.0 and found[1, 0] == 0.0
    assert found[1, 3] == 0.0
    assert np.all(found[:, 1:3] > 0.0)


def test_the_checks_skip_only_frames_outside_the_fitted_bin_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The range that ``_inside`` reads is each dose's outer bin edges, as the lookup's.

    The observed minimum and maximum lie strictly inside the padded outer edges, so a frame
    between them would have been skipped although its density is not zero.
    """
    from cleverly.interventions import joint_density as jd

    seen: list[Any] = []
    original = jd._enumerate

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(args[5])
        return original(*args, **kwargs)

    monkeypatch.setattr(jd, "_enumerate", spy)
    from sklearn.linear_model import LogisticRegression

    data = geo_data(geo.sample(400, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = (
            TMLE(
                policies=[geo_joint("ordered")],
                outcome_learner=LogisticRegression(),
                treatment_learner=LogisticRegression(max_iter=1000),
                ratio="density",
                cross_fit=False,
                density_bins=4,
            )
            .fit(data)
            .single()
        )
    densities = result.nuisance.policies.joint.densities
    expected = [
        (float(densities[name].edges[0]), float(densities[name].edges[-1])) for name in ("D1", "D2")
    ]
    assert seen and all(list(item) == expected for item in seen)
    natural = np.asarray(data.treatment_components, dtype=float)
    assert expected[0][0] < natural[:, 1].min()
