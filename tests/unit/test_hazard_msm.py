"""A working model over the hazard curve, checked exactly on a finite law (X70 part (d)).

``tests/discrete_law_point_survival.py`` realises each law exactly, so a saturated hazard fit
is the oracle and the empirical projection of the true cell risks is the parameter.  The
parameter here is computed from the law by a Newton solve written in this file, in complex
arithmetic, so the same code gives its efficient influence function by complex-step
derivatives.  Nothing in it calls the estimator or :func:`cleverly.msm.solve_projection`.

The wrong hazard learner keeps ``W`` but is additive in ``(W, A)``: the law's hazard has an
interaction, so the plug-in coefficient of the ``a x W`` term is wrong and not zero.  A
learner with no ``W`` would give that coefficient exactly zero under static regimens and a
uniform projection weight, which targeting could not be seen to correct.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.exceptions import ConvergenceWarning, WeightingWarning
from cleverly.interventions import ModifiedPolicy
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal import hazard as hazard_module
from cleverly.longitudinal.estimator import (
    _fitted_replay_matches,
    _refit_bound,
)
from cleverly.longitudinal.hazard import MSMTargets
from cleverly.msm import MSM
from tests.discrete_law_point_survival import CellMeans, CellProbabilities, PointLaw

REGIMENS = {"a0": 0, "a1": 1, "a2": 2}
DOSE = {"a0": 0.0, "a1": 1.0, "a2": 2.0}
LAW = PointLaw("survival", 2)
COMPETING = PointLaw("survival", 2, causes=2)
#: The first one-step step, as in ``test_hazard_survival_law.py``.
STEP = 2e-2
#: The shipped wrong-hazard witnesses pair ``abs=1e-8`` with ``tol=1e-10`` on risks.  A
#: coefficient on the logit scale is a risk score times :math:`M^{-1}`, which amplifies the
#: walk's residual score about a hundredfold here, so the coefficient witnesses walk to 1e-12.
TOL = 1e-12
WRONG = 1e-8


class Additive(LogisticRegression):  # type: ignore[misc]
    """A wrong hazard learner: unpenalised logistic, additive in ``W`` and the arm dummies."""

    def __init__(self) -> None:
        super().__init__(C=np.inf, tol=1e-12, max_iter=10_000)


def _design(modifier: bool) -> Callable[[Any, int, Any], np.ndarray]:
    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        w = np.asarray(frame["W"], dtype=float)
        a = DOSE[str(label)]
        columns = [np.ones_like(w), np.full_like(w, a), np.full_like(w, float(horizon))]
        if modifier:
            columns.append(a * w)
        return np.column_stack(columns)

    return design


def _msm(modifier: bool = True, link: str = "logit") -> MSM:
    terms = ("1", "a", "t", "aW") if modifier else ("1", "a", "t")
    return MSM(design=_design(modifier), terms=terms, design_kind="known", link=link)


def _saturated() -> MSM:
    cells = [(label, t) for t in (1, 2) for label in REGIMENS]

    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        n = len(frame)
        return np.column_stack([np.full(n, float((str(label), horizon) == cell)) for cell in cells])

    return MSM(
        design=design,
        terms=tuple(f"{label}@{t}" for label, t in cells),
        design_kind="known",
        link="identity",
    )


def _fit(
    law: PointLaw,
    model: MSM | None,
    *,
    learner: Any = None,
    frame: pd.DataFrame | None = None,
    **columns: Any,
) -> Any:
    return LTMLE(
        REGIMENS,
        n_folds=1,
        survival_model="hazard",
        targeting="one_step",
        step_size=STEP,
        tol=TOL,
        msm=model,
        outcome_learner=learner if learner is not None else CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(law.frame() if frame is None else frame, **law.fit_columns(), **columns)


# ------------------------------------------------------------------- the exact parameter


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def _exact_beta(
    law: PointLaw, probs: Any, model: MSM, cause: int = 1, *, link: str = "logit"
) -> Any:
    """The projection of the law's cell risks onto ``model``, by Newton's method.

    Every operation is a sum, a product, a quotient or an exponential, so the value is
    analytic in ``probs`` and complex probabilities give complex-step derivatives.
    """
    labels = list(REGIMENS)
    cells = [(label, t) for t in range(1, law.n_times + 1) for label in labels]
    frame = pd.DataFrame({"W": [0.0, 1.0]})
    design = np.stack([model.design(label, t, frame) for label, t in cells], axis=1)  # (2, C, p)
    total = law._mass(probs, {})
    share = [law._mass(probs, {"W": w}) / total for w in (0, 1)]
    psi = np.array(
        [
            [
                law._regression(probs, {"W": w, "A": REGIMENS[label]}, 1, t, cause)
                for label, t in cells
            ]
            for w in (0, 1)
        ]
    )
    beta: Any = np.zeros(design.shape[2], dtype=psi.dtype)
    for _ in range(60):
        eta = np.einsum("wcp,p->wc", design, beta)
        mean = _expit(eta) if link == "logit" else eta
        slope = mean * (1.0 - mean) if link == "logit" else np.ones_like(eta)
        score = sum(share[w] * design[w].T @ (psi[w] - mean[w]) for w in (0, 1))
        jacobian = sum(share[w] * (design[w].T * slope[w]) @ design[w] for w in (0, 1))
        beta = beta + np.linalg.solve(jacobian, score)
    return beta


def _exact_eif(law: PointLaw, model: MSM, cause: int = 1, *, step: float = 1e-30) -> np.ndarray:
    """``(support, p)`` efficient influence function of the projection, by complex step."""
    base = law.probs.astype(complex)
    out = []
    for point in range(len(law.support)):
        mass = np.zeros_like(base)
        mass[point] = 1.0
        perturbed = (1.0 - 1j * step) * base + 1j * step * mass
        out.append(np.imag(_exact_beta(law, perturbed, model, cause)) / step)
    return np.array(out)


def _beta(result: Any, cause: int = 0) -> np.ndarray:
    return np.asarray(result.msm_fits[cause].beta, dtype=float)


# ----------------------------------------------------------------------------- witnesses


@pytest.mark.parametrize("modifier", [True, False])
def test_a_wrong_hazard_with_the_true_mechanism_gives_the_exact_projection(modifier: bool) -> None:
    """Double robustness on the coefficients: the plug-in misses, the targeted fit does not."""
    model = _msm(modifier)
    truth = _exact_beta(LAW, LAW.probs, model).real
    result = _fit(LAW, model, learner=Additive())
    fit = result.msm_fits[0]
    assert fit.converged
    assert fit.hazard is not None and fit.hazard.n_iter > 0
    np.testing.assert_allclose(fit.beta, truth, rtol=0.0, atol=WRONG)
    # The wrong learner's plug-in coefficients, which the walk had to move.
    plug_in = fit.hazard.initial_levels
    assert np.max(np.abs(plug_in - truth)) > 1e-3
    if modifier:
        assert abs(plug_in[3]) > 1e-3 and abs(plug_in[3] - truth[3]) > 1e-3


def test_each_competing_cause_gets_its_own_exact_projection() -> None:
    model = _msm()
    result = _fit(COMPETING, model, learner=Additive())
    assert [fit.cause for fit in result.msm_fits] == ["relapse", "death"]
    for index, code in enumerate((1, 2)):
        truth = _exact_beta(COMPETING, COMPETING.probs, model, code).real
        np.testing.assert_allclose(_beta(result, index), truth, rtol=0.0, atol=WRONG)
    assert result.msm_fits[0].hazard is result.msm_fits[1].hazard


def test_mutation_targeting_the_cells_misses_with_a_modifier_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cell scores solve the coefficient scores only when ``h phi`` is free of ``V``."""
    walk = hazard_module.walk_curve

    def cells(*args: Any, combination: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    monkeypatch.setattr(hazard_module, "walk_curve", cells)
    with_modifier = _msm(True)
    mutated = _beta(_fit(LAW, with_modifier, learner=Additive()))
    truth = _exact_beta(LAW, LAW.probs, with_modifier).real
    assert abs(mutated[3] - truth[3]) > 1e-3
    # Control: without the modifier the same mutation solves the coefficient scores.
    free = _msm(False)
    control = _beta(_fit(LAW, free, learner=Additive()))
    np.testing.assert_allclose(control, _exact_beta(LAW, LAW.probs, free).real, atol=WRONG)


def test_the_walk_targets_the_coefficient_scores() -> None:
    result = _fit(LAW, _msm(), learner=Additive())
    record = result.msm_fits[0].hazard
    assert record.targets == (("1", None), ("a", None), ("t", None), ("aW", None))
    assert record.score.shape == (4,)
    assert record.relative_score <= TOL


def test_the_influence_curve_is_the_exact_efficient_influence_function() -> None:
    """Saturated hazards on law-as-data: the walk does not move, and the curve is exact."""
    model = _msm()
    result = _fit(LAW, model)
    fit = result.msm_fits[0]
    assert fit.hazard is not None and fit.hazard.n_iter == 0
    np.testing.assert_allclose(fit.beta, _exact_beta(LAW, LAW.probs, model).real, atol=1e-12)
    eif = _exact_eif(LAW, model)
    rows = LAW.first_row_of()
    np.testing.assert_allclose(fit.influence_curves[rows], eif, rtol=0.0, atol=1e-10)
    # Mutation: without M^{-1} the curve's SD moves by more than 10 per cent.
    unscaled = fit.influence_curves @ fit.projection.jacobian.T
    sd, mutated = fit.influence_curves.std(axis=0), unscaled.std(axis=0)
    assert np.all(np.abs(mutated / sd - 1.0) > 0.10)


def test_a_saturated_design_reproduces_the_cell_estimates() -> None:
    cells = _fit(LAW, None)
    projected = _fit(LAW, _saturated())
    fit = projected.msm_fits[0]
    for column, term in enumerate(fit.model.terms):
        label, t = term.split("@")
        expected = cells[f"risk_regimen[{label} @ t={t}]"]
        assert fit.beta[column] == pytest.approx(expected.psi, abs=1e-10)
        np.testing.assert_allclose(
            fit.influence_curves[:, column], expected.influence_curve, atol=1e-10
        )


def test_a_weighted_fit_gives_the_projection_of_the_tilted_law() -> None:
    frame = LAW.frame()
    weights = 1 + frame["W"].to_numpy(dtype=int) + (frame["A"].to_numpy() == 2).astype(int)
    weighted = frame.assign(wt=weights.astype(float))
    model = _msm()
    with pytest.warns(WeightingWarning):
        oracle = _fit(LAW, model, frame=weighted, weights="wt")
    with pytest.warns(WeightingWarning):
        wrong = _fit(LAW, model, learner=Additive(), frame=weighted, weights="wt")
    unweighted = _beta(_fit(LAW, model))
    assert np.max(np.abs(_beta(oracle) - unweighted)) > 1e-3
    np.testing.assert_allclose(_beta(wrong), _beta(oracle), rtol=0.0, atol=WRONG)
    # Mutation: dropping w from the bracket divides each row of the curve by its weight.
    fit = oracle.msm_fits[0]
    scale = weights / weights.mean()
    dropped = fit.influence_curves / scale[:, None]
    assert np.max(np.abs(dropped.std(axis=0) / fit.influence_curves.std(axis=0) - 1.0)) > 0.01


def test_a_walk_stopped_short_is_not_converged(monkeypatch: pytest.MonkeyPatch) -> None:
    walk = hazard_module.walk_curve

    def one_step(*args: Any, **kwargs: Any) -> Any:
        return walk(*args, **{**kwargs, "max_steps": 1})

    monkeypatch.setattr(hazard_module, "walk_curve", one_step)
    with pytest.warns(ConvergenceWarning):
        result = _fit(LAW, _msm(), learner=Additive())
    fit = result.msm_fits[0]
    assert fit.hazard is not None and not fit.hazard.converged
    assert fit.projection.converged
    assert not fit.converged


def test_a_replay_at_the_fitted_bound_is_bit_identical() -> None:
    result = _fit(LAW, _msm(), learner=Additive())
    replay = _refit_bound(result, result.repeats[0], result.config.g_bounds)
    assert _fitted_replay_matches(result, result.repeats[0], replay)
    assert replay.msm_fits[0].hazard is not None


def test_the_cell_map_files_each_cell_at_its_regimen_and_horizon() -> None:
    """A nonzero witness for :meth:`MSMTargets.coefficients` and :meth:`MSMTargets.combine`."""
    result = _fit(LAW, _msm())
    model = result.msm
    targets = MSMTargets.of(model, list(REGIMENS), (1, 2))
    rng = np.random.default_rng(0)
    gamma = rng.normal(size=(1, model.n_terms))
    coefficients = targets.coefficients(gamma)
    primitive = rng.normal(size=(len(REGIMENS), 1, 2, model.n))
    # <coefficients, primitive> summed over cells equals <gamma, combine(primitive)>.
    left = np.einsum("nrjh,rjhn->", coefficients, primitive)
    right = np.einsum("jm,jmn->", gamma, targets.combine(primitive))
    assert left == pytest.approx(right, rel=1e-12)
    expected = np.zeros_like(coefficients)
    for k, cell in enumerate(model.cells):
        r, h = list(REGIMENS).index(cell.label), cell.horizon - 1
        expected[:, r, 0, h] += model.weighted_design[:, k] @ gamma[0]
    np.testing.assert_array_equal(coefficients, expected)


def test_the_isotonic_redirection_fits_with_a_modified_treatment_policy() -> None:
    """H5's alternative: msm= with isotonic=False on the iterated mean takes an MTP cell."""

    def up(a: Any, h: Any) -> Any:
        return np.minimum(np.asarray(a, dtype=float) + 1.0, 2.0)

    def design(label: Any, horizon: int, frame: Any) -> np.ndarray:
        return np.ones((len(frame), 1))

    model = MSM(design=design, terms=("intercept",), design_kind="known")
    result = LTMLE(
        {"up": ModifiedPolicy("up", apply=up, policy_kind="known"), "a1": 1},
        n_folds=1,
        msm=model,
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
    ).fit(LAW.frame(), **LAW.fit_columns())
    assert np.isfinite(result.msm_fits[0].beta).all()


def test_the_full_metric_changes_the_path_and_not_the_solution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diagonal metric walks thousands of steps to the same coefficients."""
    full = _fit(LAW, _msm(), learner=Additive())
    monkeypatch.setattr(
        hazard_module, "_full_metric", lambda curves, sigma: np.diag(sigma.reshape(-1))
    )
    diagonal = _fit(LAW, _msm(), learner=Additive())
    np.testing.assert_allclose(_beta(diagonal), _beta(full), rtol=0.0, atol=WRONG)
    assert diagonal.msm_fits[0].hazard.n_iter > 10 * full.msm_fits[0].hazard.n_iter


@pytest.mark.parametrize(
    "settings",
    [{"n_folds": 2}, {"stopping": "scaled_var"}, {"iterate": "best"}],
    ids=["cross-fitted", "declared rule", "best iterate"],
)
def test_the_settings_of_the_walk_carry_over(settings: dict[str, Any]) -> None:
    options = {"n_folds": 1, **settings}
    result = LTMLE(
        REGIMENS,
        survival_model="hazard",
        targeting="one_step",
        step_size=STEP,
        msm=_msm(),
        outcome_learner=Additive(),
        treatment_learner=CellProbabilities(),
        censoring_learner=CellMeans(),
        random_state=0,
        **options,
    ).fit(LAW.frame(), **LAW.fit_columns())
    fit = result.msm_fits[0]
    assert fit.converged
    assert np.isfinite(fit.beta).all() and np.isfinite(fit.influence_curves).all()
    record = fit.hazard
    if "stopping" in settings:
        assert record.rule_ratio is not None and record.rule_ratio <= 1.0
        assert record.thresholds is not None and record.thresholds.shape == (4,)
    if "iterate" in settings:
        assert record.best_iterate is not None
    if "n_folds" in settings:
        # Cross-fitted nuisances are not the law's, so the check is statistical: within one
        # standard error of the projection.
        se = fit.influence_curves.std(axis=0) / np.sqrt(LAW.n)
        assert np.all(np.abs(fit.beta - _exact_beta(LAW, LAW.probs, _msm()).real) < se)


def test_the_score_diagnostics_read_one_row_per_coefficient() -> None:
    result = _fit(COMPETING, _msm(), learner=Additive())
    table = result.diagnostics.score_equations().to_frame()
    assert len(table) == 8
    assert set(table["cause"]) == {"relapse", "death"}
    assert table["passed"].all()
