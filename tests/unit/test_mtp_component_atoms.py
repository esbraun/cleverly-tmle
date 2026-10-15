"""Declared atoms at a dose of a vector treatment or a mixed node (X40 review M4).

A dose with declared atoms has a mixed factor in the joint density: :math:`p_{k,j}` at an
atom and :math:`(1 - \\sum_j p_{k,j}) f_k` elsewhere, against Lebesgue measure plus counting
measure on the atoms (``cleverly.interventions.joint_density``).  The oracle is the geometric
law of ``tests/discrete_law_vector_atoms.py``, read as a point treatment and as one mixed
node, with saturated learners on ``exact_bins()`` grids: every exact fit equals the g-formula
and its complex-step Gateaux derivative.

- T-b1 analog: every plan is exact on both ratio routes for ``TMLE``, ``LTMLE`` and ``SDR``,
  and the witnesses are nonzero.  Every grid dose declared an atom is exact for every plan
  of the geometric law.  A collapse to an atom is exact on the classifier route.  On the
  density route it is exact at the last dose, and refused at an earlier one.
- M-b1 analog at the last dose: dropping the interval's mass, or reading the binned density
  as a mass, breaks the collapse's exactness.
- M-b1 and M-b2 analogs: dropping the atom paths, or reading the continuous density at an
  atom, moves the curve by more than ``CAUGHT``.
- R-b3: an atom sent off the atoms is refused on the observed frame, and on a counterfactual
  frame of the density route that no observed row holds.  Without the check, that fit runs.
- R-b2: each declared atom needs rows overall, per training fold, and on the density route
  per categorical cell that records it.
- T-b6 analog: a vector without atoms is bit-identical to the fit before M4.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tests.discrete_law_vector_atoms as law
import tests.discrete_law_vector_density as geo
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import JointPolicy, ModifiedPolicy, Piece, Randomizer
from cleverly.interventions import joint_density as joint_module
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen, LongitudinalData

from .. import longitudinal_mtp as mtp
from ..dose_declarations import NeverFit
from . import test_longitudinal_mixed_node as long_tests
from . import test_vector_density_route as route

pytestmark = [
    pytest.mark.xdist_group("mtp_component_atoms"),
    pytest.mark.usefixtures("exact_grid_law"),
]

EXACT = 1e-10
CAUGHT = 1e-3
INF = np.inf
REACHED = "a learner was fitted before the refusal"


# ------------------------------------------------------------------ the plans


def plus_one(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + 1.0


def minus_one(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - 1.0


def unit(b: Any, h: Any) -> np.ndarray:
    return np.ones(len(b))


def plus_half(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + 0.5


def minus_half(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - 0.5


def up_if_unflagged(h: Any) -> np.ndarray:
    return np.where(np.asarray(h["F"], dtype=float) == 0.0, 1.5, -INF)


def small_d1(h: Any) -> np.ndarray:
    return np.where(np.asarray(h["D1"], dtype=float) <= 2.0, 3.0, -INF)


def flag_if_atom(a: Any, h: Any) -> np.ndarray:
    return np.where(np.asarray(h["D1"], dtype=float) == 1.0, 1, np.asarray(a))


def unflagged_at_one(h: Any) -> np.ndarray:
    """``1.5`` where the natural flag is 0 and ``W`` is 1: a frame no observed row holds."""
    f, w = np.asarray(h["F"], dtype=float), np.asarray(h["W"], dtype=float)
    return np.where((f == 0.0) & (w == 1.0), 1.5, -INF)


def _known(name: str, pieces: Any) -> ModifiedPolicy:
    return ModifiedPolicy(name, pieces=pieces, policy_kind="known")


SMOOTH_D1 = Piece(3.0, 6.0, route.double, route.halve, route.half_slope)
D1_HELD = (Piece(-INF, 3.0), SMOOTH_D1, Piece(6.0, INF))
D1_UP = (Piece(-INF, 1.5, plus_one, minus_one, unit), Piece(1.5, 3.0), SMOOTH_D1, Piece(6.0, INF))
D1_UP_IF = (
    Piece(-INF, up_if_unflagged, plus_one, minus_one, unit),
    Piece(up_if_unflagged, 3.0),
    SMOOTH_D1,
    Piece(6.0, INF),
)
D2_HELD = (Piece(-INF, 3.0, route.double, route.halve, route.half_slope), Piece(3.0, INF))
D2_IF = (Piece(-INF, small_d1, route.double, route.halve, route.half_slope), Piece(small_d1, INF))
FLAG = ModifiedPolicy("flag", apply=route.set_flag, policy_kind="known")
COLLAPSE_D2 = (Piece(-INF, 1.5), Piece(1.5, 6.0, to_atom=8.0), Piece(6.0, INF))


def _components(label: str) -> dict[str, Any]:
    if label == "natural":
        return {}
    if label == "atoms held":
        return {"D1": _known("d1 held", D1_HELD), "D2": _known("d2 held", D2_HELD)}
    if label == "atom up":
        return {"D1": _known("d1 up", D1_UP)}
    if label == "flag up, atom up":
        return {"F": FLAG, "D1": _known("d1 up if unflagged", D1_UP_IF)}
    if label == "flag reads atom":
        return {
            "F": ModifiedPolicy("flag", apply=flag_if_atom, policy_kind="known"),
            "D1": _known("d1 held", D1_HELD),
            "D2": _known("d2 if small d1", D2_IF),
        }
    if label == "collapse to atom":
        pieces = (Piece(-INF, 1.5), Piece(1.5, 6.0, to_atom=1.0), Piece(6.0, INF))
        return {"D1": _known("collapse", pieces)}
    if label == "collapse last dose":
        return {"D2": _known("collapse last", COLLAPSE_D2)}
    raise KeyError(label)


def joint(label: str) -> JointPolicy:
    """The plan of the oracle label ``label``."""
    if label == "random":
        return JointPolicy(
            label,
            components={"a": _components("flag up, atom up"), "b": _components("atoms held")},
            randomizer=Randomizer(("a", "b"), (0.3, 0.7)),
            policy_kind="known",
        )
    return JointPolicy(label, components=_components(label), policy_kind="known")


# ------------------------------------------------------------------ fits


def _point(
    labels: Sequence[str],
    ratio: str,
    *,
    atoms: Any = law.ATOMS,
    plans: Any = joint,
    frame: pd.DataFrame | None = None,
) -> Any:
    with warnings.catch_warnings(), mtp.exact_bins(route.GRIDS):
        warnings.simplefilter("ignore")
        return (
            TMLE(policies=[plans(label) for label in labels], **route.density_settings(ratio=ratio))
            .fit(route.geo_data(frame), dose_atoms=atoms)
            .single()
        )


def _long(
    estimator: type,
    labels: Sequence[str],
    ratio: str,
    *,
    atoms: Any = law.ATOMS,
    plans: Any = joint,
    frame: pd.DataFrame | None = None,
) -> Any:
    settings = {**long_tests.settings(), "ratio": ratio}
    if estimator is SDR:
        settings.pop("simultaneous", None)
    regimens = {label: DynamicRegimen(label, (plans(label),)) for label in labels}
    with warnings.catch_warnings(), mtp.exact_bins(route.GRIDS):
        warnings.simplefilter("ignore")
        return estimator(regimens, reference=labels[0], **settings).fit(
            geo.frame() if frame is None else frame, dose_atoms=atoms, **route.GEO_LONG_COLUMNS
        )


def _gaps(result: Any, labels: Sequence[str], *, longitudinal: bool) -> tuple[float, float]:
    """The largest gap of the estimate and of the curve to the oracle, over ``labels``."""
    rows = law.first_row_of()
    psi = curve = 0.0
    for label in labels:
        if longitudinal:
            name = f"ey_regimen[{label}]"
            estimate, reported = result.psi(name), result.influence_curves[name][rows]
        else:
            fitted = result.estimates[f"ey_policy[{label}]"]
            estimate, reported = fitted.psi, np.asarray(fitted.influence_curve)[rows]
        psi = max(psi, abs(estimate - law.truth(label)))
        curve = max(curve, float(np.max(np.abs(reported - law.eif(label)))))
    return psi, curve


def _fit(estimator: Any, labels: Sequence[str], ratio: str, **kwargs: Any) -> tuple[float, float]:
    if estimator is TMLE:
        return _gaps(_point(labels, ratio, **kwargs), labels, longitudinal=False)
    return _gaps(_long(estimator, labels, ratio, **kwargs), labels, longitudinal=True)


ESTIMATORS = pytest.mark.parametrize("estimator", [TMLE, LTMLE, SDR], ids=["tmle", "ltmle", "sdr"])
RATIOS = pytest.mark.parametrize("ratio", ["density", "classifier"])


# ------------------------------------------------------------------ T-b1


def test_t_b1_the_component_witnesses_are_nonzero() -> None:
    """The atom ratio exceeds 2, a flag-dependent branch moves mass, and each plan moves θ."""
    assert law.atom_up_ratio().max() > 2.0
    truths = {label: law.truth(label) for label in (*law.LABELS, "collapse to atom")}
    assert min(abs(a - b) for a, b in _pairs(list(truths.values()))) > 1e-3
    # "flag up, atom up" moves the atom only on the natural flag 0, so it is not "atom up".
    assert abs(truths["flag up, atom up"] - truths["atom up"]) > 0.01


def _pairs(values: list[float]) -> list[tuple[float, float]]:
    return [(a, b) for i, a in enumerate(values) for b in values[i + 1 :]]


@ESTIMATORS
@RATIOS
def test_t_b1_every_plan_with_component_atoms_is_exact(estimator: Any, ratio: str) -> None:
    psi, curve = _fit(estimator, law.LABELS, ratio)
    assert psi < EXACT
    assert curve < EXACT


@pytest.mark.parametrize("estimator", [TMLE, LTMLE], ids=["tmle", "ltmle"])
@RATIOS
def test_t_b1_every_grid_dose_declared_an_atom_is_exact(estimator: Any, ratio: str) -> None:
    """Dose 1's factor is then the discrete formula, so every geometric plan is exact."""
    labels = route.GEO_LABELS
    result = (
        _point(labels, ratio, atoms=law.ALL_ATOMS, plans=route.geo_joint)
        if estimator is TMLE
        else _long(estimator, labels, ratio, atoms=law.ALL_ATOMS, plans=route.geo_joint)
    )
    assert route.geo_deviation(result, labels, longitudinal=estimator is not TMLE) < EXACT


@pytest.mark.parametrize("estimator", [TMLE, LTMLE], ids=["tmle", "ltmle"])
def test_t_b1_a_collapse_to_an_atom_is_exact_on_the_classifier_route(estimator: Any) -> None:
    psi, curve = _fit(estimator, ("natural", "collapse to atom"), "classifier")
    assert psi < EXACT
    assert curve < EXACT


def test_a_collapse_to_an_atom_is_refused_on_the_density_route() -> None:
    text = joint_module.collapse_refusal("collapse", 1.0)
    with pytest.raises(CapabilityError, match=re.escape(text) + "$"):
        _point(("natural", "collapse to atom"), "density")
    with pytest.raises(CapabilityError, match=re.escape(text) + "$"):
        _long(LTMLE, ("natural", "collapse to atom"), "density")


COLLAPSE_LAST = ("natural", "collapse last dose")


@ESTIMATORS
@RATIOS
def test_t_b1_a_collapse_at_the_last_dose_is_exact_on_both_routes(
    estimator: Any, ratio: str
) -> None:
    """The mass of ``[1.5, 6)`` at dose 2's atom 8 is a finite sum of binned densities."""
    psi, curve = _fit(estimator, COLLAPSE_LAST, ratio)
    assert psi < EXACT
    assert curve < EXACT
    # The nonzero witness: the collapse moves the parameter.
    assert abs(law.truth("collapse last dose") - law.truth("natural")) > 1e-3


def _drop_collapse_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: the atom of the last dose receives no mass from the interval."""
    monkeypatch.setattr(joint_module, "_collapse_paths", lambda *args, **kwargs: [])


def _density_as_mass(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: each bin adds its density, not its density times the overlap."""
    original = joint_module._collapse_paths

    def mutated(*args: Any, **kwargs: Any) -> Any:
        return [
            (betas, reach, np.where(reach, 1.0, 0.0))
            for betas, reach, _ in original(*args, **kwargs)
        ]

    monkeypatch.setattr(joint_module, "_collapse_paths", mutated)


@pytest.mark.parametrize(
    "mutation", [_drop_collapse_paths, _density_as_mass], ids=["no mass", "density as mass"]
)
@pytest.mark.parametrize("estimator", [TMLE, LTMLE], ids=["tmle", "ltmle"])
def test_m_b1_the_last_dose_collapse_mutations_fail_t_b1(
    monkeypatch: pytest.MonkeyPatch, mutation: Any, estimator: Any
) -> None:
    mutation(monkeypatch)
    psi, curve = _fit(estimator, COLLAPSE_LAST, "density")
    assert max(psi, curve) > CAUGHT


def flag_if_high_d2(a: Any, h: Any) -> np.ndarray:
    return np.where(np.asarray(h["D2"], dtype=float) >= 8.0, 1, np.asarray(a))


def test_a_last_dose_collapse_that_another_map_reads_is_refused() -> None:
    """A categorical map that reads the collapsed dose varies over the interval."""
    policy = JointPolicy(
        "flag reads the collapse",
        components={
            "F": ModifiedPolicy("flag", apply=flag_if_high_d2, policy_kind="known"),
            "D2": _known("collapse last", COLLAPSE_D2),
        },
        policy_kind="known",
    )
    with pytest.raises(CapabilityError, match="a categorical map reads"):
        _point(("natural", "flag reads the collapse"), "density", plans=_with(policy))


def test_a_collapse_at_the_last_column_but_not_the_last_in_order_is_refused() -> None:
    policy = JointPolicy(
        "collapse first in order",
        components={"D2": _known("collapse last", COLLAPSE_D2)},
        order=("D2", "D1"),
        policy_kind="known",
    )
    text = joint_module.collapse_refusal("collapse last", 8.0)
    with pytest.raises(CapabilityError, match=re.escape(text) + "$"):
        _point(("natural", "collapse first in order"), "density", plans=_with(policy))


def _with(policy: JointPolicy) -> Any:
    def plans(label: str) -> JointPolicy:
        return policy if label == policy.name else joint(label)

    return plans


# ------------------------------------------------------------------ M-b1, M-b2


def _drop_atom_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-b1: an atom value of a dose is reached by no path, so its induced mass is zero."""
    monkeypatch.setattr(joint_module, "_atom_paths", lambda *args, **kwargs: [])


def _continuous_at_atoms(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-b2: the factor reads the continuous density at an atom instead of ``p_{k,j}``."""

    def mutated(*args: Any, **kwargs: Any) -> Any:
        return dict(args[6])

    monkeypatch.setattr(joint_module, "_mixed_factor", mutated)


@pytest.mark.parametrize("mutation", [_drop_atom_paths, _continuous_at_atoms], ids=["M-b1", "M-b2"])
@pytest.mark.parametrize("estimator", [TMLE, LTMLE], ids=["tmle", "ltmle"])
def test_m_b1_m_b2_the_component_atom_mutations_fail_t_b1(
    monkeypatch: pytest.MonkeyPatch, mutation: Any, estimator: Any
) -> None:
    labels = ("natural", "atom up")
    psi, curve = _fit(estimator, labels, "density")
    assert max(psi, curve) < EXACT
    mutation(monkeypatch)
    psi, curve = _fit(estimator, labels, "density")
    assert max(psi, curve) > CAUGHT


# ------------------------------------------------------------------ R-b3


def _never(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "cross_fit": False,
        **overrides,
    }


def _never_point(policies: list[Any], ratio: str, frame: pd.DataFrame, atoms: Any) -> Any:
    return TMLE(policies=policies, ratio=ratio, **_never()).fit(
        route.geo_data(frame), dose_atoms=atoms
    )


def _never_long(policies: list[Any], ratio: str, frame: pd.DataFrame, atoms: Any) -> Any:
    regimens = {policy.name: DynamicRegimen(policy.name, (policy,)) for policy in policies}
    never = {**_never(), "pseudo_learner": NeverFit(), "censoring_learner": NeverFit()}
    never.pop("cross_fit")
    return LTMLE(regimens, reference=policies[0].name, n_folds=1, ratio=ratio, **never).fit(
        frame, dose_atoms=atoms, **route.GEO_LONG_COLUMNS
    )


NATURAL = JointPolicy("natural", components={}, policy_kind="known")
OFF_THE_ATOMS = JointPolicy(
    "half up",
    components={
        "D1": _known("d1 half up", (Piece(-INF, 1.5, plus_half, minus_half, unit), Piece(1.5, INF)))
    },
    policy_kind="known",
)


def _rb3(where: str, note: str = "") -> str:
    from cleverly.interventions.policy import atom_image_refusal

    return atom_image_refusal("d1 half up", where, 1.0, 1.5, describe="d1 half up", note=note)


@RATIOS
@pytest.mark.parametrize("fit", [_never_point, _never_long], ids=["point", "node"])
def test_r_b3_an_atom_sent_off_the_atoms_is_refused_on_the_observed_frame(
    ratio: str, fit: Any
) -> None:
    with pytest.raises(CapabilityError, match=re.escape("moves the declared atom 1 to 1.5")):
        fit([NATURAL, OFF_THE_ATOMS], ratio, geo.frame(), law.ATOMS)
    # The control: with recorded mass at 1.5 declared an atom, 1.5 is a valid image.
    frame = geo.frame()
    frame.loc[np.flatnonzero(frame["D1"] == 2.0)[::4], "D1"] = 1.5
    with pytest.raises(AssertionError, match=REACHED):
        fit([NATURAL, OFF_THE_ATOMS], ratio, frame, {"D1": (1.0, 1.5, 2.0)})


def _flagged_at_one() -> pd.DataFrame:
    """The geometric frame with ``F = 1`` wherever ``W = 1``: no row holds ``(W, F) = (1, 0)``."""
    frame = geo.frame()
    frame["F"] = np.where(frame["W"] == 1.0, 1.0, frame["F"])
    return frame


#: The flag is set, and dose 1 sends its atom 1 to 1.5 only on the natural flag 0 at
#: ``W = 1``.  The observed frame never holds it; the density route's candidate cell
#: ``F = 0`` does, at every row with ``W = 1``.
COUNTERFACTUAL = JointPolicy(
    "flag up, half up",
    components={
        "F": FLAG,
        "D1": _known(
            "d1 half up",
            (
                Piece(-INF, unflagged_at_one, plus_half, minus_half, unit),
                Piece(unflagged_at_one, INF),
            ),
        ),
    },
    policy_kind="known",
)


@pytest.mark.parametrize("fit", [_never_point, _never_long], ids=["point", "node"])
def test_r_b3_a_counterfactual_frame_that_moves_the_atom_off_the_atoms_is_refused(
    fit: Any,
) -> None:
    frame = _flagged_at_one()
    with pytest.raises(CapabilityError, match=re.escape(joint_module._COUNTERFACTUAL_ATOMS)):
        fit([NATURAL, COUNTERFACTUAL], "density", frame, law.ATOMS)
    # The classifier route reads observed frames only, and every observed image is an atom.
    with pytest.raises(AssertionError, match=REACHED):
        fit([NATURAL, COUNTERFACTUAL], "classifier", frame, law.ATOMS)


def test_r_b3_the_counterfactual_check_is_the_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: with the check off, the density route reaches the learners."""
    original = joint_module._atom_paths

    def unchecked(*args: Any) -> Any:
        values = list(args)
        values[7] = np.zeros_like(values[7])
        return original(*values)

    monkeypatch.setattr(joint_module, "_atom_paths", unchecked)
    with pytest.raises(AssertionError, match=REACHED):
        _never_point([NATURAL, COUNTERFACTUAL], "density", _flagged_at_one(), law.ATOMS)


# ------------------------------------------------------------------ R-b2

HOLD = JointPolicy("atoms held", components=_components("atoms held"), policy_kind="known")


@RATIOS
def test_r_b2_a_declared_atom_with_no_row_is_refused(ratio: str) -> None:
    with pytest.raises(DataError, match=r"^dose_atoms declares 3 at node 'D2', but no row"):
        _never_point([NATURAL, HOLD], ratio, geo.frame(), {"D2": (3.0, 8.0)})
    with pytest.raises(DataError, match=r"^dose_atoms declares 3 at node 'D2', but no row"):
        _never_long([NATURAL, HOLD], ratio, geo.frame(), {"D2": (3.0, 8.0)})


def _one_row_at(column: str, value: float, where: Any = None) -> pd.DataFrame:
    frame = geo.frame()
    rows = np.flatnonzero(np.ones(len(frame), dtype=bool) if where is None else where(frame))
    frame.loc[rows[0], column] = value
    return frame


@RATIOS
def test_r_b2_a_declared_atom_missing_from_a_training_fold_is_refused(ratio: str) -> None:
    frame = _one_row_at("D2", 3.0)
    pattern = r"^dose_atoms declares 3 at node 'D2', but the training rows of outer fold [12]"
    with pytest.raises(DataError, match=pattern):
        TMLE(policies=[NATURAL, HOLD], ratio=ratio, **_never(cross_fit=True, n_folds=2)).fit(
            route.geo_data(frame), dose_atoms={"D2": (3.0, 8.0)}
        )
    regimens = {p.name: DynamicRegimen(p.name, (p,)) for p in (NATURAL, HOLD)}
    never = {
        "outcome_learner": NeverFit(),
        "treatment_learner": NeverFit(),
        "pseudo_learner": NeverFit(),
        "censoring_learner": NeverFit(),
    }
    with pytest.raises(DataError, match=pattern):
        LTMLE(regimens, reference="natural", n_folds=2, ratio=ratio, **never).fit(
            frame, dose_atoms={"D2": (3.0, 8.0)}, **route.GEO_LONG_COLUMNS
        )


def _atom_once_in_flag_zero() -> pd.DataFrame:
    """The atom 1 of dose 1 on one row of the cell ``F = 0``, and on many of ``F = 1``."""
    frame = geo.frame()
    at = np.flatnonzero((frame["F"] == 0.0) & (frame["D1"] == 1.0))
    frame.loc[at[1:], "D1"] = 2.0
    return frame


def test_r_b2_the_density_route_needs_the_atom_in_each_cell_of_each_fold() -> None:
    frame = _atom_once_in_flag_zero()
    folds = _never(cross_fit=True, n_folds=2)
    pattern = (
        r"^dose_atoms declares 1 at component 'D1', but the training rows of outer fold [12] "
        r"hold no row at risk with that dose in the cell \(F=0(\.0)?\)"
    )
    with pytest.raises(DataError, match=pattern):
        TMLE(policies=[NATURAL, HOLD], ratio="density", **folds).fit(
            route.geo_data(frame), dose_atoms=law.ATOMS
        )
    # The classifier route divides by no probability of the atom within a cell.
    with pytest.raises(AssertionError, match=REACHED):
        TMLE(policies=[NATURAL, HOLD], ratio="classifier", **folds).fit(
            route.geo_data(frame), dose_atoms=law.ATOMS
        )


# ------------------------------------------------------------------ declarations, designs


def test_the_designs_carry_an_indicator_for_each_component_atom() -> None:
    data = route.geo_data().with_dose_declarations(None, law.ATOMS)
    assert data.component_atoms == {"D1": (1.0, 2.0), "D2": (8.0,)}
    rows = np.array([[1.0, 2.0, 8.0], [0.0, 4.0, 1.0]])
    values = np.resize(rows, (data.n, 3))
    design = data.component_design(values)
    # F as one indicator; D1, its atoms 1 and 2; D2, its atom 8.
    expected = np.resize(
        np.array([[1.0, 2.0, 0.0, 1.0, 8.0, 1.0], [0.0, 4.0, 0.0, 0.0, 1.0, 0.0]]), (data.n, 6)
    )
    np.testing.assert_array_equal(design, expected)
    plain = route.geo_data().component_design(values)
    np.testing.assert_array_equal(plain, expected[:, [0, 1, 4]])
    node = LongitudinalData.from_frame(geo.frame(), dose_atoms=law.ATOMS, **route.GEO_LONG_COLUMNS)
    assert node.component_atoms(1) == law.ATOMS
    np.testing.assert_array_equal(node.component_design(1, values), expected)


def test_the_point_design_declares_component_atoms() -> None:
    from cleverly import CausalStudy, PointTreatment

    def design(**declared: Any) -> PointTreatment:
        return PointTreatment(
            outcome="Y",
            treatment=["F", "D1", "D2"],
            adjustment=("W",),
            continuous_treatment=["D1", "D2"],
            **declared,
        )

    study = CausalStudy(geo.frame(), design=design(dose_atoms=law.ATOMS))
    assert study.data.component_atoms == law.ATOMS
    CausalStudy(study.data, design=design(dose_atoms={"D2": (8.0,), "D1": (2.0, 1.0)}))
    with pytest.raises(DataError, match="dose_atoms="):
        CausalStudy(study.data, design=design())


def test_a_scalar_treatment_refuses_atoms_by_component() -> None:
    data = CausalData.from_frame(
        geo.frame(), outcome="Y", treatment="D1", covariates=["W"], treatment_kind="continuous"
    )
    with pytest.raises(DataError):
        data.with_dose_declarations(None, {"D1": (1.0,)})


# ------------------------------------------------------------------ T-b6

#: ``[psi, sum of squared curve, first curve entry, last curve entry]`` of each fit on
#: ``geo.sample(3000, 5)`` with no atom declared, recorded on the commit before M4.
PINS: dict[str, list[float]] = {
    "TMLE:density:ordered": [0.6795959341905007, 1630.71023404607, 0.20677032878069634, 0.2833688503160503],
    "TMLE:density:full": [0.6521542386800482, 1432.4819687041706, 0.23421202429114885, 0.3108105458265028],
    "TMLE:density:random": [0.6736423173455979, 1117.7813408640945, 0.2127239456255991, 0.28932246716095306],
    "LTMLE:density:ordered": [0.6795959341905392, 1630.710862579851, 0.2067720521432973, 0.28336856905855967],
    "LTMLE:density:full": [0.6521542386800482, 1432.478172475145, 0.23420939768358817, 0.31080872428291473],
    "LTMLE:density:random": [0.6736423173456257, 1117.780194835638, 0.21272397079219563, 0.2893215845884334],
    "SDR:density:ordered": [0.6795959341905515, 1630.7105856832231, 0.20676770217308482, 0.2833670287724114],
    "SDR:density:full": [0.6521542386800482, 1432.4781724751451, 0.23420939768358817, 0.31080872428291473],
    "SDR:density:random": [0.6736423173456285, 1117.7922117481398, 0.21272131901800784, 0.2893206456173344],
    "TMLE:classifier:ordered": [0.6795959341905007, 1630.7102340460697, 0.20677032878069634, 0.2833688503160503],
    "TMLE:classifier:full": [0.6521542386800482, 1432.4819687041709, 0.23421202429114885, 0.3108105458265028],
    "TMLE:classifier:random": [0.6736423173455979, 1117.7813408640945, 0.2127239456255991, 0.28932246716095306],
    "LTMLE:classifier:ordered": [0.6795959341905392, 1630.7108625798508, 0.2067720521432973, 0.28336856905855967],
    "LTMLE:classifier:full": [0.6521542386800482, 1432.478172475145, 0.23420939768358817, 0.31080872428291473],
    "LTMLE:classifier:random": [0.6736423173456257, 1117.7801948356378, 0.21272397079219563, 0.2893215845884334],
    "SDR:classifier:ordered": [0.6795959341905515, 1630.7105856832227, 0.20676770217308482, 0.2833670287724114],
    "SDR:classifier:full": [0.6521542386800482, 1432.478172475145, 0.23420939768358817, 0.31080872428291473],
    "SDR:classifier:random": [0.6736423173456285, 1117.7922117481398, 0.21272131901800784, 0.2893206456173344],
}  # fmt: skip


@ESTIMATORS
@RATIOS
def test_t_b6_a_vector_without_atoms_is_bit_identical(estimator: Any, ratio: str) -> None:
    labels = ("ordered", "full", "random")
    frame = geo.sample(3000, 5)
    if estimator is TMLE:
        result = _point(labels, ratio, atoms=(), plans=route.geo_joint, frame=frame)
        curves = {
            label: np.asarray(result.estimates[f"ey_policy[{label}]"].influence_curve)
            for label in labels
        }
        psis = {label: result.estimates[f"ey_policy[{label}]"].psi for label in labels}
    else:
        result = _long(estimator, labels, ratio, atoms={}, plans=route.geo_joint, frame=frame)
        curves = {label: result.influence_curves[f"ey_regimen[{label}]"] for label in labels}
        psis = {label: result.psi(f"ey_regimen[{label}]") for label in labels}
    for label in labels:
        curve = np.asarray(curves[label])
        got = [psis[label], float(np.sum(curve**2)), float(curve[0]), float(curve[-1])]
        np.testing.assert_allclose(
            got, PINS[f"{estimator.__name__}:{ratio}:{label}"], rtol=1e-12, atol=0.0
        )
