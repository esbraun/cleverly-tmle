"""Declared atoms of a continuous dose: the mixed ratio and its refusals (X40 part (b)).

A dose with a recorded point mass, such as a list price, has no density at that dose.  The
fit takes the policy ratio with respect to Lebesgue measure plus counting measure on the
declared atoms (``X40-plan.md`` section 2.3).  At an atom :math:`a_k` the ratio is

.. math::

    r(a_k, h) = \\frac{(1 - \\sum_j p_j) \\sum_{C \\to a_k} \\int_C f
                + \\sum_{j: d(a_j) = a_k} p_j}{p_k},

and elsewhere it is Equation (3) of Díaz et al. (2023) on the continuous part.  The ratio
exists only when every collapse target and every atom image is a declared atom.

The witnesses are T-b1 to T-b9 and T-s1 of the plan.  T-b1 holds the estimate and the curve
of LTMLE, SDR and the point TMLE to the exact functional and its Gateaux derivative on the
unit grid law with the dose 3 declared an atom.  The exact grid laws need the
``exact_grid_law`` fixture: every grid dose is a point mass, which R-b4 refuses.  M-b1 to
M-b6 are deliberate mutations, applied by monkeypatch, that each witness must catch.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cleverly
import cleverly.interventions.policy as policy_module
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import ModifiedPolicy, Piece, Piecewise, Randomizer, Scale, Shift
from cleverly.interventions.policy import (
    _atom_rule_flags,
    branch_ratio,
    lazy_frame,
    lowered_ratio,
    policy_branches,
)
from cleverly.learners.density import ConditionalDensity, MixedDensity
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen, LongitudinalData

from .. import discrete_law_longitudinal as binary_law
from .. import discrete_law_longitudinal_mtp as law
from .. import discrete_law_longitudinal_multivalue as multivalue
from .. import discrete_law_policy_point as point_law
from .. import dose_declarations as doses
from .. import longitudinal_mtp as mtp

EXACT = 1e-10
CAUGHT = 1e-4
INF = np.inf
SRC = Path(cleverly.__file__).resolve().parent
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}
ATOM_LABELS = ("to atom", "hold atom", "atom random")
NATURAL = Shift(0.0, cap=None)


def _exact(estimator: type, ratio: str) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "outcome_learner": binary_law.CellMeans(),
        "pseudo_learner": binary_law.CellMeans(),
        "treatment_learner": multivalue.CellProbabilities(),
        "n_folds": 1,
        "g_bounds": (1e-8, 1.0),
        "random_state": 2,
        "ratio": ratio,
    }
    if estimator is LTMLE:
        settings["simultaneous"] = False
    return settings


def _panel_fit(estimator: type, ratio: str, labels: tuple[str, ...] = ATOM_LABELS) -> Any:
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return estimator(
            mtp.atom_regimens(("natural", *labels)), reference="natural", **_exact(estimator, ratio)
        ).fit(law.frame(), dose_atoms=mtp.ATOMS, **COLUMNS)


POINT_POLICIES = {
    "natural course": Shift(0.0, cap=None, name="natural course"),
    "to atom": ModifiedPolicy("to atom", pieces=mtp.TO_ATOM_PIECES, policy_kind="known"),
    "hold atom": Shift(1.0, cap=2.5, name="hold atom"),
    "atom random": ModifiedPolicy(
        "atom random",
        pieces={"to": mtp.TO_ATOM_PIECES, "hold": mtp.HOLD_ATOM_PIECES},
        randomizer=Randomizer(("to", "hold"), (0.4, 0.6)),
        policy_kind="known",
    ),
}


def _point_fit(ratio: str, labels: tuple[str, ...] = ATOM_LABELS) -> Any:
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        return (
            TMLE(
                policies=[POINT_POLICIES[label] for label in ("natural course", *labels)],
                ratio=ratio,
                outcome_learner=binary_law.CellMeans(),
                treatment_learner=multivalue.CellProbabilities(),
                cross_fit=False,
                g_bounds=(1e-8, 1.0 - 1e-8),
            )
            .fit(
                point_law.UNIT.frame(),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                dose_atoms=(point_law.ATOM_DOSE,),
            )
            .single()
        )


def _panel_gaps(result: Any, labels: tuple[str, ...] = ATOM_LABELS) -> tuple[float, float]:
    rows = law.first_row_of()
    psi = curve = 0.0
    for label in labels:
        name = f"ey_regimen[{label}]"
        psi = max(psi, abs(result.psi(name) - law.truth(name)))
        reported = np.asarray(result[name].influence_curve)[rows]
        curve = max(curve, float(np.max(np.abs(reported - law.eif(name)))))
    return psi, curve


def _point_gaps(result: Any, labels: tuple[str, ...] = ATOM_LABELS) -> tuple[float, float]:
    rows = point_law.UNIT.first_row_of()
    psi = curve = 0.0
    for label in labels:
        estimate = result[f"ey_policy[{label}]"]
        psi = max(psi, abs(estimate.psi - point_law.atom_truth(label)))
        reported = np.asarray(estimate.influence_curve)[rows]
        curve = max(curve, float(np.max(np.abs(reported - point_law.atom_eif(label)))))
    return psi, curve


# ------------------------------------------------------------------ T-b1


def test_t_b1_the_witnesses_are_nonzero() -> None:
    """The atom's ratio exceeds 2, its preimage holds 20% of the mass, and holding moves θ."""
    assert law.atom_ratio_cells().max() > 2.0
    assert law.preimage_mass() >= 0.2
    gap = law.truth("ey_regimen[to atom]") - law.truth("ey_regimen[hold atom]")
    assert abs(gap) > 0.01
    g = point_law.UNIT.g
    assert np.max((g[:, 2] + g[:, 3]) / g[:, 3]) > 2.0
    assert float(np.sum(point_law.UNIT.probs[:, 2, :])) >= 0.2
    assert abs(point_law.atom_truth("to atom") - point_law.atom_truth("hold atom")) > 0.01


@pytest.mark.usefixtures("exact_grid_law")
@pytest.mark.parametrize("estimator", [LTMLE, SDR], ids=["ltmle", "sdr"])
@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_b1_the_longitudinal_mixed_law_is_exact(estimator: type, ratio: str) -> None:
    psi, curve = _panel_gaps(_panel_fit(estimator, ratio))
    assert psi < EXACT
    assert curve < EXACT


@pytest.mark.usefixtures("exact_grid_law")
@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_b1_the_point_mixed_law_is_exact(ratio: str) -> None:
    psi, curve = _point_gaps(_point_fit(ratio))
    assert psi < EXACT
    assert curve < EXACT


@pytest.mark.usefixtures("exact_grid_law")
def test_t_b1_the_density_route_fits_a_mixed_density() -> None:
    fit = _point_fit("density", ("to atom",))
    support = fit.diagnostics.support()["to atom"]
    assert support.atom_mass is not None
    assert set(support.atom_mass) == {point_law.ATOM_DOSE}


# ------------------------------------------------------------------ M-b1, M-b2


def _drop_preimage(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-b1: the preimage integral of every collapse is dropped from the atom's mass."""
    monkeypatch.setattr(MixedDensity, "mass_between", lambda self, lower, upper: np.zeros(self.n))


def _continuous_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-b2: the atom ratio divides by the continuous density instead of ``p_k(h)``."""
    original = policy_module.mixed_ratio

    def mutated(branch: Any, values: Any, frame: Any, density: Any) -> Any:
        ratio = np.asarray(original(branch, values, frame, density), dtype=float)
        a = np.asarray(values, dtype=float).reshape(-1)
        which = policy_module._atom_index(a, density.atoms)
        at = which >= 0
        rows = np.arange(a.size)
        p = np.asarray(density.atom_probabilities)[rows, np.maximum(which, 0)]
        f = np.asarray(density.density_at(np.where(at, a + 1e-9, a)), dtype=float)
        safe = np.where(f > 0.0, f, 1.0)
        return np.where(at, np.where(f > 0.0, ratio * p / safe, 0.0), ratio)

    monkeypatch.setattr(policy_module, "mixed_ratio", mutated)


@pytest.mark.usefixtures("exact_grid_law")
@pytest.mark.parametrize(
    "mutation", [_drop_preimage, _continuous_denominator], ids=["M-b1", "M-b2"]
)
def test_m_b1_m_b2_the_atom_mass_mutations_fail_t_b1(
    monkeypatch: pytest.MonkeyPatch, mutation: Any
) -> None:
    mutation(monkeypatch)
    psi, curve = _panel_gaps(_panel_fit(LTMLE, "density", ("to atom",)), ("to atom",))
    assert max(psi, curve) > CAUGHT
    psi, curve = _point_gaps(_point_fit("density", ("to atom",)), ("to atom",))
    assert max(psi, curve) > CAUGHT


# ------------------------------------------------------------------ T-b2, T-b3


def _never(**overrides: Any) -> dict[str, Any]:
    return {
        "outcome_learner": doses.NeverFit(),
        "treatment_learner": doses.NeverFit(),
        "cross_fit": False,
        **overrides,
    }


def _point_refusal(frame: Any, policies: list[Any], ratio: str, **declared: Any) -> Any:
    return TMLE(policies=policies, ratio=ratio, **_never()).fit(
        frame, outcome="Y", treatment="A", covariates=["W"], **declared
    )


def _rb1(name: str, lower: float, upper: float, value: float, where: str = "the treatment") -> str:
    return (
        f"policy {name!r} at {where} maps the interval [{lower:g}, {upper:g}) to the single "
        f"dose {value:g} on a continuous treatment. The natural dose has no recorded mass at "
        f"{value:g}, so the law of the assigned dose has an atom that the observed law lacks, "
        f"and no density ratio exists. The mean then contains the mean under the static dose "
        f"{value:g}, which is not pathwise differentiable (Díaz et al. 2023, Section 4, journal "
        f"page 850), so no root-n interval exists. If the dose has a recorded point mass at "
        f"{value:g}, declare it with dose_atoms and the piece with Piece({lower:g}, {upper:g}, "
        f"to_atom={value:g}). Otherwise use a piecewise map that moves every dose, or collapse "
        "the dose into categories and declare the policy on a categorical node"
    )


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_b2_a_collapse_to_an_undeclared_dose_is_refused(ratio: str) -> None:
    frame = doses.point_frame()
    to_one = ModifiedPolicy(
        "to one",
        pieces=(Piece(-INF, 0.5), Piece(0.5, 1.0, to_atom=1.0), Piece(1.0, INF)),
        policy_kind="known",
    )
    with pytest.raises(CapabilityError, match=re.escape(_rb1("to one", 0.5, 1, 1)) + "$"):
        _point_refusal(frame, [NATURAL, to_one], ratio)
    # A map that is flat on its piece: Scale(0) written as a ModifiedPolicy.
    flat = ModifiedPolicy(
        "flat",
        pieces=(
            Piece(-INF, 0.5, lambda a, h: 0.0 * a + 1.0, lambda b, h: b - 1.0, lambda b, h: 1.0),
            Piece(0.5, INF),
        ),
        policy_kind="known",
    )
    a = frame["A"].to_numpy()
    low = float(a.min())
    with pytest.raises(CapabilityError, match=re.escape(_rb1("flat", low, 0.5, 1)) + "$"):
        _point_refusal(frame, [NATURAL, flat], ratio)


def test_t_b2_scale_zero_names_the_atom_declaration() -> None:
    with pytest.raises(
        DataError, match=re.escape("declare it with dose_atoms and the piece with Piece(")
    ):
        Scale(0.0, cap=None)


def test_t_b2_the_longitudinal_refusal_names_the_node() -> None:
    frame = doses.panel_frame()
    to_one = ModifiedPolicy(
        "to one",
        pieces=(Piece(-INF, 0.5), Piece(0.5, 1.0, to_atom=1.0), Piece(1.0, INF)),
        policy_kind="known",
    )
    regimens = {"natural": NATURAL, "to one": DynamicRegimen("to one", (NATURAL, to_one))}
    with pytest.raises(
        CapabilityError, match=r"maps the interval \[0\.5, 1\) to the single dose 1"
    ):
        LTMLE(regimens, reference="natural", n_folds=1, **_panel_never()).fit(
            frame, continuous_treatment=["A1", "A2"], **doses.PANEL
        )


def _panel_never() -> dict[str, Any]:
    return {
        "outcome_learner": doses.NeverFit(),
        "pseudo_learner": doses.NeverFit(),
        "treatment_learner": doses.NeverFit(),
    }


def _rb2(value: float, node: str, where_none: str) -> str:
    return (
        f"dose_atoms declares {value:g} at node {node!r}, but {where_none}. A declared atom "
        "needs recorded mass: the ratio at the atom divides by "
        f"P(A = {value:g} | H), which the data estimate only from rows that received it"
    )


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_b3_a_declared_atom_with_no_row_is_refused(ratio: str) -> None:
    frame = doses.point_frame(atom=0.5)
    policy = ModifiedPolicy("hold", pieces=mtp.HOLD_ATOM_PIECES, policy_kind="known")
    text = _rb2(5, "A", "no row at risk at that node holds that dose")
    with pytest.raises(DataError, match=re.escape(text) + "$"):
        _point_refusal(frame, [NATURAL, policy], ratio, dose_atoms=(0.5, 5.0))


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_b3_a_declared_atom_missing_from_a_training_fold_is_refused(ratio: str) -> None:
    frame = doses.panel_frame(atom=0.5)
    a2 = frame["A2"].to_numpy().copy()
    # One more atom on a single row: the training rows of the fold that holds it out have none.
    a2[int(np.flatnonzero(a2 != 0.5)[0])] = 1.75
    frame["A2"] = a2
    regimens = {
        "natural": NATURAL,
        "hold": DynamicRegimen(
            "hold",
            (NATURAL, ModifiedPolicy("hold", pieces=(Piece(-INF, INF),), policy_kind="known")),
        ),
    }
    text = _rb2(1.75, "A2", "the training rows of outer fold ")
    with pytest.raises(DataError, match=re.escape(text.split("fold ")[0])):
        LTMLE(regimens, reference="natural", n_folds=2, ratio=ratio, **_panel_never()).fit(
            frame,
            continuous_treatment=["A1", "A2"],
            dose_atoms={"A2": (0.5, 1.75)},
            **doses.PANEL,
        )


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_b3_the_point_fold_check_precedes_every_learner(ratio: str) -> None:
    frame = doses.point_frame(atom=0.5)
    a = frame["A"].to_numpy().copy()
    a[int(np.flatnonzero(a != 0.5)[0])] = 1.75
    frame["A"] = a
    hold = ModifiedPolicy("hold", pieces=(Piece(-INF, INF),), policy_kind="known")
    text = _rb2(1.75, "A", "the training rows of outer fold 9 hold no row at risk with that dose")
    pattern = re.escape(text).replace("fold\\ 9", "fold\\ [12]") + "$"
    with pytest.raises(DataError, match=pattern):
        TMLE(policies=[hold], ratio=ratio, **_never(cross_fit=True, n_folds=2)).fit(
            frame, outcome="Y", treatment="A", covariates=["W"], dose_atoms=(0.5, 1.75)
        )


# ------------------------------------------------------------------ T-b4, M-b3, M-b4

TO_ATOM_POINT = ModifiedPolicy("to atom", pieces=mtp.TO_ATOM_PIECES, policy_kind="known")


def test_t_b4_the_law_jumps_at_the_atom() -> None:
    for w in (0.0, 1.0):
        at = float(doses.jump_regression(doses.JUMP_ATOM, w))
        beside = float(doses.jump_regression(doses.JUMP_ATOM - 1e-9, w))
        assert abs(at - beside) > 0.05


def test_t_b4_the_designs_read_the_assigned_dose() -> None:
    frame = doses.jump_frame(200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=["W"], treatment_kind="continuous"
        ).with_dose_declarations(dose_atoms=(doses.JUMP_ATOM,))
    n = len(frame)
    block = data.treatment_block(np.full(n, doses.JUMP_ATOM))
    np.testing.assert_array_equal(block[:, 1], np.ones(n))
    block = data.treatment_block(np.full(n, 1.0))
    np.testing.assert_array_equal(block[:, 1], np.zeros(n))
    panel = LongitudinalData.from_frame(
        doses.jump_panel(200),
        continuous_treatment=["A1", "A2"],
        dose_atoms={"A2": (doses.JUMP_ATOM,)},
        **doses.PANEL,
    )
    np.testing.assert_array_equal(panel.dose_block(2, np.full(200, 3.0))[:, 1], np.ones(200))
    np.testing.assert_array_equal(panel.dose_block(2, np.full(200, 1.0))[:, 1], np.zeros(200))
    assert panel.dose_block(1, np.full(200, 3.0)).shape == (200, 1)


def _jump_point() -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            TMLE(policies=[NATURAL, TO_ATOM_POINT], **doses.point_learners())
            .fit(
                doses.jump_frame(),
                outcome="Y",
                treatment="A",
                covariates=["W"],
                dose_atoms=(doses.JUMP_ATOM,),
            )
            .single()
        )


def _plug_in(result: Any) -> float:
    """The initial regression's mean at the assigned dose of "to atom": the plug-in.

    The targeted estimate is doubly robust, so a design error the correct ratio repairs moves
    it at second order only.  The plug-in reads the design at first order.
    """
    return float(np.mean(result.nuisance.outcome.arms[1.0]))


def _jump_panel() -> float:
    regimens = {"natural": NATURAL, "to atom": DynamicRegimen("to atom", (NATURAL, mtp.TO_ATOM))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = LTMLE(regimens, reference="natural", **doses.panel_learners()).fit(
            doses.jump_panel(),
            continuous_treatment=["A1", "A2"],
            dose_atoms={"A2": (doses.JUMP_ATOM,)},
            **doses.PANEL,
        )
    return float(result.psi("ey_regimen[to atom]"))


def test_t_b4_the_point_estimate_reads_the_jump() -> None:
    estimate = _jump_point()["ey_policy[to atom]"]
    assert abs(estimate.psi - doses.jump_truth()) < 4.0 * estimate.std_error


def _drop_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-b3: the atom indicator is dropped from every design."""
    point = CausalData.treatment_block

    def point_block(self: Any, codes: Any) -> Any:
        return point(self, codes)[:, :1] if self.dose_atoms else point(self, codes)

    monkeypatch.setattr(CausalData, "treatment_block", point_block)
    monkeypatch.setattr(
        LongitudinalData,
        "dose_block",
        lambda self, time, dose: np.asarray(dose, dtype=float).reshape(-1, 1),
    )


def _observed_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-b4: the atom indicator reads the observed dose in a counterfactual design."""
    point = CausalData.treatment_block

    def point_block(self: Any, codes: Any) -> Any:
        block = point(self, codes)
        if self.dose_atoms:
            block[:, 1:] = point(self, self.treatment)[:, 1:]
        return block

    panel = LongitudinalData.dose_block

    def panel_block(self: Any, time: int, dose: Any) -> Any:
        block = panel(self, time, dose)
        observed = np.nan_to_num(self.node_codes(time), nan=0.0)
        block[:, 1:] = panel(self, time, observed)[:, 1:]
        return block

    monkeypatch.setattr(CausalData, "treatment_block", point_block)
    monkeypatch.setattr(LongitudinalData, "dose_block", panel_block)


@pytest.mark.parametrize("mutation", [_drop_indicator, _observed_indicator], ids=["M-b3", "M-b4"])
def test_m_b3_m_b4_the_indicator_mutations_move_the_estimate(
    monkeypatch: pytest.MonkeyPatch, mutation: Any
) -> None:
    point, panel = _plug_in(_jump_point()), _jump_panel()
    mutation(monkeypatch)
    assert abs(_plug_in(_jump_point()) - point) > CAUGHT
    assert abs(_jump_panel() - panel) > CAUGHT


# ------------------------------------------------------------------ T-b5, M-b5


def _rb3(name: str, atom: float, image: float, branch: str) -> str:
    return (
        f"policy {name!r} at the treatment moves the declared atom {atom:g} to {image:g} "
        f"({branch}). The natural dose has no recorded mass at {image:g}, so every row that "
        f"holds the dose {atom:g} receives a dose where the observed law has no mass. The law "
        "of the assigned dose then has an atom that the observed law lacks, and no density "
        "ratio exists: this is the failure of a collapse (Díaz et al. 2023, Section 4, journal "
        "page 850) reached from the atom. Hold the atom in place (a cap or a piece that keeps "
        f"it), send it to another declared atom, or, if the dose has recorded mass at {image:g}, "
        "declare it in dose_atoms"
    )


@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_b5_a_branch_that_moves_the_atom_off_the_atoms_is_refused(ratio: str) -> None:
    frame = doses.jump_frame(600)
    atoms = {"dose_atoms": (doses.JUMP_ATOM,)}
    shift = Shift(0.5, cap=None, name="half")
    text = _rb3("half", 3, 3.5, "Shift(delta=0.5, cap=None)")
    with pytest.raises(CapabilityError, match=re.escape(text) + "$"):
        _point_refusal(frame, [NATURAL, shift], ratio, **atoms)
    scale = Scale(0.9, cap=None, name="tenth")
    text = _rb3("tenth", 3, 2.7, "Scale(factor=0.9, cap=None)")
    with pytest.raises(CapabilityError, match=re.escape(text) + "$"):
        _point_refusal(frame, [NATURAL, scale], ratio, **atoms)
    # Scale(0.9) holds an atom at 0, and a capped Shift holds the atom 3: both pass the checks.
    zero = doses.jump_frame(600)
    zero["A"] = np.where(zero["A"] == doses.JUMP_ATOM, 0.0, zero["A"])
    with pytest.raises(AssertionError, match="fitted before the refusal"):
        _point_refusal(zero, [NATURAL, scale], ratio, dose_atoms=(0.0,))
    with pytest.raises(AssertionError, match="fitted before the refusal"):
        _point_refusal(frame, [NATURAL, Shift(0.5, cap=2.9, name="held")], ratio, **atoms)


def test_m_b5_without_the_atom_image_check_the_ratio_loses_the_atom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid ratio has mean one under the observed law.  Moving the atom to a dose with no
    mass drops the atom's rows from the ratio, so its mean falls by the atom's share."""
    monkeypatch.setattr(policy_module, "_check_atom_images", lambda *args, **kwargs: None)
    shift = Shift(0.5, cap=None, name="half")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TMLE(policies=[NATURAL, shift], **doses.point_learners()).fit(
            doses.jump_frame(),
            outcome="Y",
            treatment="A",
            covariates=["W"],
            dose_atoms=(doses.JUMP_ATOM,),
        )
    mean_ratio = result.single().diagnostics.support()["half"].mean_ratio
    assert abs(mean_ratio - 1.0) > 0.1
    assert np.isfinite(result.single()["ey_policy[half]"].psi)


# ------------------------------------------------------------------ T-b6


def _frame(n: int, seed: int) -> Any:
    import pandas as pd

    rng = np.random.default_rng(seed)
    return pd.DataFrame({"W": rng.normal(size=n)})


def _density(frame: Any) -> Any:
    w = np.asarray(frame["W"], dtype=float)

    def at(values: Any) -> np.ndarray:
        b = np.asarray(values, dtype=float).reshape(-1)
        return np.exp(-0.5 * (b - 0.5 * w) ** 2) / np.sqrt(2.0 * np.pi)

    return at


LOWERED_CASES = {
    "shift": Shift(0.7, cap=1.5),
    "shift uncapped": Shift(-0.4, cap=None),
    "scale": Scale(1.3, cap=2.0),
    "scale below one": Scale(0.6, cap=None),
    "piecewise": Piecewise(
        ((-INF, 0.0, Shift(0.5, None)), (0.0, 1.0, Scale(1.5, None)), (1.0, INF, Shift(0.0, None)))
    ),
    "modified": ModifiedPolicy(
        "modified",
        pieces=(
            Piece(
                -INF,
                lambda h: np.asarray(h["W"]),
                lambda a, h: 2.0 * a - 1.0,
                lambda b, h: (b + 1.0) / 2.0,
                lambda b, h: 0.5,
            ),
            Piece(lambda h: np.asarray(h["W"]), INF),
        ),
        policy_kind="known",
    ),
    "decreasing": ModifiedPolicy(
        "decreasing",
        pieces=(Piece(-INF, INF, lambda a, h: 3.0 - a, lambda b, h: 3.0 - b, lambda b, h: -1.0),),
        policy_kind="known",
    ),
}


@pytest.mark.parametrize("label", sorted(LOWERED_CASES))
def test_t_b6_the_lowered_form_equals_each_closed_form(label: str) -> None:
    n = 10_000
    frame_value = _frame(n, 7)
    frame = lazy_frame(lambda: frame_value)
    density = _density(frame_value)
    values = np.random.default_rng(8).normal(scale=2.0, size=n)
    for _, branch in policy_branches(LOWERED_CASES[label]):
        closed = branch.ratio(values, frame, density)
        lowered = lowered_ratio(branch.lowered(frame, n), values, density)
        np.testing.assert_allclose(lowered, closed, rtol=1e-12, atol=1e-12)
        assert np.max(np.abs(closed - 1.0)) > 0.1


# ------------------------------------------------------------------ T-b7


def _point_data() -> tuple[CausalData, np.ndarray]:
    frame = point_law.UNIT.frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=["W"], treatment_kind="continuous"
        ).with_dose_declarations(dose_atoms=(point_law.ATOM_DOSE,))
    return data, frame["W"].to_numpy().astype(int)


def _mixed_law_density(covariate: np.ndarray) -> MixedDensity:
    g = point_law.UNIT.g[covariate]
    p = g[:, point_law.ATOM_INDEX : point_law.ATOM_INDEX + 1]
    continuous = g.copy()
    continuous[:, point_law.ATOM_INDEX] = 0.0
    continuous = continuous / (1.0 - p)
    return MixedDensity(
        (point_law.ATOM_DOSE,),
        p,
        ConditionalDensity(point_law.UNIT.bins(continuous), point_law.UNIT.edges),
    )


@pytest.mark.parametrize("label", ATOM_LABELS)
def test_t_b7_the_mixed_ratio_is_the_discrete_ratio_at_the_ratio_layer(label: str) -> None:
    data, covariate = _point_data()
    density = _mixed_law_density(covariate)
    observed = np.searchsorted(point_law.UNIT.doses, data.treatment)
    g = point_law.UNIT.g[covariate]
    frame = lazy_frame(lambda: data.frame_like({"W": data.covariates[:, 0]}))
    for (probability, mapping), (_, branch) in zip(
        point_law.ATOM_POLICIES[label],
        policy_branches(POINT_POLICIES[label]),
        strict=True,
    ):
        assigned = np.zeros_like(g)
        for w in range(3):
            for k, target in enumerate(mapping[w]):
                rows = covariate == w
                assigned[rows, target] += g[rows, k]
        expected = (
            assigned[np.arange(len(observed)), observed] / g[np.arange(len(observed)), observed]
        )
        ratio = branch_ratio(branch, data.treatment, frame, density)
        np.testing.assert_allclose(ratio, expected, rtol=0, atol=1e-12)
        del probability


# ------------------------------------------------------------------ T-b8, M-b6


WITNESSES = {
    "A": (doses.one_atom, 2_000, 320, "the dose 6.37 on "),
    "B": (doses.grid(0.02), 1_000, 320, "density bins"),
    "C": (doses.grid(0.1), 1_000, None, "only "),
    "D": (doses.list_prices, 2_000, None, "(dispersion z = "),
}


def _witness(rule: str) -> np.ndarray:
    law_of, n, _, _ = WITNESSES[rule]
    return law_of(n, np.random.default_rng(4008))


@pytest.mark.parametrize("rule", sorted(WITNESSES))
def test_t_b8_each_rule_has_a_law_on_which_it_alone_fires(rule: str) -> None:
    _, _, bins, _ = WITNESSES[rule]
    flags = _atom_rule_flags(_witness(rule), atoms=(), bins=bins)
    assert flags == {other: other == rule for other in "ABCD"}


def _undeclared(rule: str) -> Any:
    _, _, bins, _ = WITNESSES[rule]
    ratio = "classifier" if bins is None else "density"
    policy = Shift(0.1, cap=9.0, name="up")
    return TMLE(policies=[NATURAL, policy], ratio=ratio, density_bins=bins, **_never()).fit(
        doses.dose_frame(_witness(rule), 1), outcome="Y", treatment="A", covariates=["W"]
    )


@pytest.mark.parametrize("rule", sorted(WITNESSES))
def test_t_b8_each_witness_is_refused_with_its_finding(rule: str) -> None:
    finding = WITNESSES[rule][3]
    with pytest.raises(DataError, match=r"^node 'A' records ") as caught:
        _undeclared(rule)
    message = str(caught.value)
    assert finding in message
    assert "Declare dose_atoms=(" in message
    if rule == "C":
        assert "distinct doses, and the dose " in message
    if rule == "B":
        assert "distinct doses against 320 density bins" in message


def test_t_b8_the_nulls_pass() -> None:
    for n in (2_000, 100_000):
        cents = doses.cents(n, np.random.default_rng(11))
        assert not any(_atom_rule_flags(cents, atoms=(), bins=None).values()), n
        assert not any(_atom_rule_flags(cents, atoms=(), bins=40).values()), n
    smooth = doses.smooth_dose(300, np.random.default_rng(12))
    assert not any(_atom_rule_flags(smooth, atoms=(), bins=320).values())


def test_t_b8_a_node_where_every_plan_is_the_natural_course_is_not_checked() -> None:
    frame = doses.dose_frame(_witness("A"), 1)
    with pytest.raises(AssertionError, match="fitted before the refusal"):
        TMLE(policies=[NATURAL], **_never()).fit(
            frame, outcome="Y", treatment="A", covariates=["W"]
        )


def _rule_off(monkeypatch: pytest.MonkeyPatch, rule: str) -> None:
    original = policy_module._atom_rules

    def without(doses: Any, *, atoms: Any, bins: Any) -> Any:
        return original(doses, atoms=atoms, bins=bins, off=(rule,))

    monkeypatch.setattr(policy_module, "_atom_rules", without)


@pytest.mark.parametrize("rule", sorted(WITNESSES))
def test_m_b6_without_its_rule_the_witness_reaches_the_learners(
    monkeypatch: pytest.MonkeyPatch, rule: str
) -> None:
    _rule_off(monkeypatch, rule)
    with pytest.raises(AssertionError, match="fitted before the refusal"):
        _undeclared(rule)


#: The rules whose witness has a declared-atom oracle a fast fit can reach: (A) declares
#: its one atom, and (C) declares the grid's 60 recorded values as atoms.
ORACLES = {
    "A": (
        ModifiedPolicy(
            "up",
            pieces=(
                Piece(-INF, 6.0, lambda a, h: a + 0.3, lambda b, h: b - 0.3, lambda b, h: 1.0),
                Piece(6.0, INF),
            ),
            policy_kind="known",
        ),
        "density",
    ),
    # The cap holds 8.1, whose upper neighbour 8.2 has no row in this sample.
    "C": (Shift(0.1, cap=8.0, name="up"), "classifier"),
}


@pytest.mark.parametrize("rule", sorted(ORACLES))
def test_m_b6_without_its_rule_the_fit_misses_its_declared_atom_oracle(
    monkeypatch: pytest.MonkeyPatch, rule: str
) -> None:
    policy, ratio = ORACLES[rule]
    dose = _witness(rule)
    frame = doses.dose_frame(dose, 1, jump=1.0, atom=6.37)
    atoms = (6.37,) if rule == "A" else tuple(np.unique(dose))

    def fit(**declared: Any) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = TMLE(policies=[NATURAL, policy], ratio=ratio, **doses.point_learners()).fit(
                frame, outcome="Y", treatment="A", covariates=["W"], **declared
            )
        return float(result.single()["ey_policy[up]"].psi)

    oracle = fit(dose_atoms=atoms)
    _rule_off(monkeypatch, rule)
    assert abs(fit() - oracle) > CAUGHT


# ------------------------------------------------------------------ T-b9


@pytest.mark.parametrize("estimator", [LTMLE, SDR], ids=["ltmle", "sdr"])
@pytest.mark.parametrize("ratio", ["density", "classifier"])
def test_t_b9_a_grid_declared_as_atoms_is_the_discrete_formula(estimator: type, ratio: str) -> None:
    """No fixture: an all-atom node has no undeclared dose, so R-b4 passes it."""
    labels = ("up at 2",)
    with warnings.catch_warnings(), mtp.exact_bins():
        warnings.simplefilter("ignore")
        result = estimator(
            mtp.regimens(("natural", *labels)), reference="natural", **_exact(estimator, ratio)
        ).fit(law.frame(), dose_atoms={"A2": law.UNIT_GRID.doses}, **COLUMNS)
    rows = law.first_row_of()
    name = "ey_regimen[up at 2]"
    assert abs(result.psi(name) - law.truth(name)) < EXACT
    reported = np.asarray(result[name].influence_curve)[rows]
    np.testing.assert_allclose(reported, law.eif(name), rtol=0, atol=EXACT)


# ------------------------------------------------------------------ T-s1


def _public_objects() -> list[Any]:
    import importlib

    seen: list[Any] = []
    modules = [cleverly]
    for name in ("estimators", "interventions", "longitudinal", "data", "learners"):
        modules.append(importlib.import_module(f"cleverly.{name}"))
    for module in modules:
        for name in getattr(module, "__all__", ()):
            seen.append(getattr(module, name))
    return seen


def test_t_s1_no_public_object_reaches_the_seam() -> None:
    pattern = re.compile("atom_check", re.IGNORECASE)
    for obj in _public_objects():
        targets = [obj]
        if inspect.isclass(obj):
            targets.extend(
                member
                for _, member in inspect.getmembers(obj)
                if inspect.isfunction(member) or inspect.ismethod(member)
            )
            if dataclasses.is_dataclass(obj):
                for field in dataclasses.fields(obj):
                    assert not pattern.search(field.name), (obj, field.name)
        for target in targets:
            if not callable(target):
                continue
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            for parameter in signature.parameters:
                assert not pattern.search(parameter), (obj, target, parameter)


def test_t_s1_one_reader_in_src() -> None:
    loads = stores = 0
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "_ATOM_CHECK" not in text:
            continue
        assert path.name == "policy.py", path
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Name) and node.id == "_ATOM_CHECK":
                if isinstance(node.ctx, ast.Load):
                    loads += 1
                else:
                    stores += 1
            if isinstance(node, (ast.ImportFrom)):
                assert all(alias.name != "_ATOM_CHECK" for alias in node.names)
        assert "environ" not in text.split("_ATOM_CHECK")[1][:200]
    assert (loads, stores) == (1, 1)


def test_t_s1_with_the_default_every_exact_grid_law_is_refused() -> None:
    with pytest.raises(DataError, match=r"^node 'A2' records .*Declare dose_atoms=\{'A2': \("):
        LTMLE(mtp.regimens(("natural", "up at 2")), reference="natural", **_panel_never()).fit(
            law.frame(), **COLUMNS
        )
    with pytest.raises(DataError, match=r"^node 'A' records .*Declare dose_atoms=\("):
        TMLE(policies=[NATURAL, Shift(1.0, cap=3.0)], **_never()).fit(
            point_law.UNIT.frame(), outcome="Y", treatment="A", covariates=["W"]
        )


# ------------------------------------------------------------------ declarations


def test_a_component_of_a_vector_node_cannot_declare_atoms() -> None:
    frame = law.frame(vector=True)
    frame["D2"] = frame["A2b"].astype(float) + 0.5
    with pytest.raises(CapabilityError, match="a component of a vector node"):
        LongitudinalData.from_frame(
            frame,
            outcome="Y",
            treatment=["A1a", ["A2a", "D2"]],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            continuous_treatment=["D2"],
            dose_atoms={"D2": (0.5,)},
        )


def test_atoms_on_a_categorical_node_are_refused() -> None:
    with pytest.raises(DataError, match="which is not a continuous treatment node"):
        LongitudinalData.from_frame(
            law.frame(), continuous_treatment=["A1"], dose_atoms={"A2": (1.0,)}, **_columns()
        )
    with pytest.raises(DataError, match="holds distinct finite doses"):
        LongitudinalData.from_frame(
            law.frame(), continuous_treatment=["A2"], dose_atoms={"A2": (1.0, 1.0)}, **_columns()
        )


def _columns() -> dict[str, Any]:
    return {key: value for key, value in COLUMNS.items() if key != "continuous_treatment"}


def test_a_to_atom_piece_takes_no_map() -> None:
    with pytest.raises(DataError):
        Piece(0.0, 1.0, lambda a, h: a, lambda b, h: b, lambda b, h: 1.0, to_atom=1.0)
    with pytest.raises(DataError):
        Piece(0.0, 1.0, to_atom=float("inf"))


# ------------------------------------------------------------------ m1: the classifier's atom indicator


def _smooth_mixed_law(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """``(w, dose, assigned, atom_rows, truth)``: an atom of 40% at 3 and a normal elsewhere.

    The assigned dose sends the continuous interval ``[1.5, 2.5)`` to the atom, so the ratio
    at the atom is ``(p + (1 - p) P(1.5 <= A < 2.5)) / p`` with ``p = 0.4``, which is above 1
    on every history.  The learner below is logistic in the dose and the history, so only an
    indicator column can give its odds a jump at 3.
    """
    from scipy import stats

    rng = np.random.default_rng(2)
    p = 0.4
    w = rng.normal(size=n)
    at_atom = rng.uniform(size=n) < p
    dose = np.where(at_atom, 3.0, rng.normal(size=n) + 1.5)
    assigned = np.where((dose >= 1.5) & (dose < 2.5), 3.0, dose)
    inside = stats.norm.cdf(1.0) - stats.norm.cdf(0.0)
    truth = (p + (1.0 - p) * inside) / p
    return w, dose, assigned, at_atom, truth


def _atom_ratio_at_the_atom(*, patched: bool, monkeypatch: pytest.MonkeyPatch) -> float:
    from sklearn.linear_model import LogisticRegression

    from cleverly.learners import density_ratio
    from cleverly.learners.crossfit import Folds

    w, dose, assigned, at_atom, _ = _smooth_mixed_law(6000)
    if patched:
        monkeypatch.setattr(density_ratio, "_with_atoms", lambda block, atoms: block)
    (ratio,) = density_ratio.classifier_ratio(
        LogisticRegression(max_iter=2000),
        w.reshape(-1, 1),
        dose,
        [(1.0, assigned)],
        np.ones(dose.size),
        Folds.single(dose.size),
        evaluate_at=[dose],
        atoms=(3.0,),
    )
    return float(np.mean(ratio[at_atom]))


def test_m1_the_classifier_indicator_lets_a_smooth_learner_see_the_atom_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The atom ratio of a logistic learner is the true ratio within 10% with the indicator."""
    truth = _smooth_mixed_law(6000)[4]
    assert truth > 1.2
    estimate = _atom_ratio_at_the_atom(patched=False, monkeypatch=monkeypatch)
    assert estimate == pytest.approx(truth, rel=0.1)


def test_m1_dropping_the_indicator_moves_the_atom_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """M-c-ind: without the column the same learner misses the atom ratio by more than CAUGHT."""
    truth = _smooth_mixed_law(6000)[4]
    estimate = _atom_ratio_at_the_atom(patched=True, monkeypatch=monkeypatch)
    assert abs(estimate - truth) > 0.1 > CAUGHT
