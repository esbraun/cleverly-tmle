"""Pre-run design checks of the registered study ``canonical-ctmle-sequences``.

Each check reads a law, a declaration or a declaration-step input, never a committed result,
so it holds before the run (X16 plan, rev 2.2, sections 3.4.1 and 3.4.3).

The population quantities of the two primary laws are computed on a scrambled Sobol rule over
``W`` (``DGP.quadrature``), with ``A`` and ``Y`` integrated exactly: each fit weights the rows
``(W, a)`` by ``P(A = a | W)`` and reads ``E[Y | A, W]`` in place of ``Y``.  The plan's
calibration used one draw of 200,000 rows.  A draw that size moves each limit by about
0.1 SD at ``n = 2,000``, which is the size of the plan's tightest bound, so the rule replaces
the draw.  The values agree with the plan's table within that noise.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from cleverly.datasets import DGP
from cleverly.estimators import TMLE
from cleverly.learners._l1_logistic import penalty_path
from cleverly.utils.bounds import expit, logit
from tests.canonical.ctmle_sequence_witness import harness
from tests.canonical.ctmle_sequences import power_probe, regenerate
from tests.studies import bounded_cv_laws, multi_arm_common
from tests.studies import canonical_ctmle_sequences as study
from tests.studies import ctmle_sequence_properties as properties
from tests.studies.canonical_tmle import truth_for
from tests.studies.evidence.seeds import stream_seed

#: The Sobol points of the population quantities.  Doubling them moves no limit below by more
#: than 0.003 SD.
POINTS = 2**14
#: The sample size at which every primary-law SD is read.
DESIGN_N = study.PRIMARY_N
MARGIN = study.STUDY.margins.standardized_bias


# --------------------------------------------------------------- population limits


def _population(law: DGP) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    w = np.asarray(law.quadrature(POINTS), dtype=float)[:, : len(study.COVARIATES)]
    g0 = np.asarray(law.propensity(w), dtype=float)
    means = {arm: np.asarray(law.outcome_mean(w, arm, None), dtype=float) for arm in (0.0, 1.0)}
    return w, g0, means


def _arms(w: np.ndarray, g0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rows ``(W, a)`` for both arms, with weight ``P(A = a | W)``."""
    return np.vstack([w, w]), np.repeat([1, 0], len(w)), np.concatenate([g0, 1.0 - g0])


def _outcome_glm(
    w: np.ndarray, g0: np.ndarray, means: dict[float, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The main-terms outcome GLM's population limit, at both arms."""
    designs, labels, weights = [], [], []
    for arm in (1.0, 0.0):
        share = g0 if arm == 1.0 else 1.0 - g0
        for y in (1.0, 0.0):
            designs.append(np.column_stack([np.full(len(w), arm), w]))
            labels.append(np.full(len(w), y))
            weights.append(share * (means[arm] if y == 1.0 else 1.0 - means[arm]))
    weight = np.concatenate(weights)
    model = LogisticRegression(C=1e10, max_iter=10_000, tol=1e-12).fit(
        np.vstack(designs),
        np.concatenate(labels),
        sample_weight=weight * weight.size / weight.sum(),
    )
    arm = {a: model.predict_proba(np.column_stack([np.full(len(w), a), w]))[:, 1] for a in (0, 1)}
    return arm[0], arm[1]


def _lasso(w: np.ndarray, g0: np.ndarray, lambdas: Sequence[float]) -> list[np.ndarray]:
    x, a, weight = _arms(w, g0)
    return [fit.predict_proba(w)[:, 1] for fit in penalty_path(x, a, weight, list(lambdas))]


def _main_terms_g(w: np.ndarray, g0: np.ndarray) -> np.ndarray:
    x, a, weight = _arms(w, g0)
    model = LogisticRegression(C=1e10, max_iter=10_000, tol=1e-12)
    model.fit(x, a, sample_weight=weight * weight.size / weight.sum())
    return np.asarray(model.predict_proba(w)[:, 1], dtype=float)


def _fluctuated(
    g0: np.ndarray,
    means: dict[float, np.ndarray],
    q0: np.ndarray,
    q1: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The limit of one logistic fluctuation of ``(q0, q1)`` along ``g``, at both arms.

    ``g`` is bounded already.
    """
    arms = (
        (1.0, g0, q1, np.column_stack([np.zeros_like(g), 1.0 / g])),
        (0.0, 1.0 - g0, q0, np.column_stack([1.0 / (1.0 - g), np.zeros_like(g)])),
    )
    epsilon = np.zeros(2)
    for _ in range(100):
        gradient, hessian = np.zeros(2), np.zeros((2, 2))
        for arm, share, q, h in arms:
            p = expit(logit(q) + h @ epsilon)
            gradient += h.T @ (share * (means[arm] - p))
            hessian += (h * (share * p * (1.0 - p))[:, None]).T @ h
        step = np.linalg.solve(hessian, gradient)
        epsilon += step
        if np.max(np.abs(step)) < 1e-13:
            break
    return expit(logit(q0) + epsilon[0] / (1.0 - g)), expit(logit(q1) + epsilon[1] / g)


def _tmle_limit(
    g0: np.ndarray,
    means: dict[float, np.ndarray],
    q0: np.ndarray,
    q1: np.ndarray,
    g: np.ndarray,
) -> tuple[float, float]:
    """One logistic fluctuation of ``(q0, q1)`` along ``g``: the limit and ``Var(D*)``."""
    g = np.clip(g, *study.G_BOUNDS)
    q0s, q1s = _fluctuated(g0, means, q0, q1, g)
    psi = float(np.mean(q1s - q0s))
    centred = q1s - q0s - psi
    residual = g0 / g**2 * (means[1.0] * (1.0 - means[1.0]) + (means[1.0] - q1s) ** 2) + (
        1.0 - g0
    ) / (1.0 - g) ** 2 * (means[0.0] * (1.0 - means[0.0]) + (means[0.0] - q0s) ** 2)
    cross = (
        2.0 * centred * (g0 / g * (means[1.0] - q1s) - (1.0 - g0) / (1.0 - g) * (means[0.0] - q0s))
    )
    return psi, float(np.mean(residual + cross + centred**2))


def _columns(scenario: str, w: np.ndarray, g0: np.ndarray) -> list[np.ndarray]:
    """The population limit of each candidate mechanism, in path order."""
    if scenario == "binary_lasso":
        return _lasso(w, g0, (*study.study_lambdas(scenario), 0.0))
    return [*_lasso(w, g0, study.learner_lambdas()), _main_terms_g(w, g0)]


def _limits(scenario: str) -> tuple[np.ndarray, float]:
    """Each candidate's single-fluctuation limit bias, and the GLM plug-in bias, in SD units.

    The SD is that of the path's last candidate at ``n = 2,000``.
    """
    law = study.scenario_law(scenario)
    truth = truth_for(law)["ate"]
    w, g0, means = _population(law)
    q0, q1 = _outcome_glm(w, g0, means)
    fits = [_tmle_limit(g0, means, q0, q1, g) for g in _columns(scenario, w, g0)]
    sd = math.sqrt(fits[-1][1] / DESIGN_N)
    limits = np.array([(psi - truth) / sd for psi, _ in fits])
    return limits, (float(np.mean(q1 - q0)) - truth) / sd


@pytest.fixture(scope="module")
def lasso_limits() -> tuple[np.ndarray, float]:
    return _limits("binary_lasso")


@pytest.fixture(scope="module")
def learners_limits() -> tuple[np.ndarray, float]:
    return _limits("binary_learners")


class TestTheLassoLaw:
    """``x16_lasso_law``: a weak, late confounder, so the final step matters (plan 3.4.1)."""

    def test_the_candidate_limits_spread(self, lasso_limits: tuple[np.ndarray, float]) -> None:
        limits, _ = lasso_limits
        assert limits.size == study.PATH_LENGTHS["binary_lasso"]
        assert float(np.max(limits) - np.min(limits)) >= 1.0, limits

    def test_the_plug_in_is_biased(self, lasso_limits: tuple[np.ndarray, float]) -> None:
        _, plugin = lasso_limits
        assert plugin >= 0.75

    def test_w3_enters_the_path_late(self) -> None:
        """On the design draws, ``W3`` has no coefficient before grid index 3 (0-based)."""
        law = study.x16_lasso_law()
        grid = list(study.study_lambdas("binary_lasso"))
        entries = []
        for draw in range(20):
            seed = stream_seed(study.STUDY, study.DESIGN_STREAM, law.name, DESIGN_N, draw)
            frame, _ = law.sample(DESIGN_N, seed=seed, backend="pandas")
            x = frame.loc[:, list(study.COVARIATES)].to_numpy(dtype=float)
            fits = penalty_path(x, frame["A"].to_numpy(), None, grid)
            # Column 2 is W3, and coefficient column 1 is the treated class.
            entries.append(next((k for k, fit in enumerate(fits) if fit.coef[2, 1] != 0.0), 99))
        assert sum(entry >= 3 for entry in entries) >= 18, entries


class TestTheLearnersLaw:
    """``x16_learners_law``: the early entries leave ``W3`` out and the late ones carry it."""

    def test_the_early_entries_are_biased(self, learners_limits: tuple[np.ndarray, float]) -> None:
        limits, _ = learners_limits
        assert limits.size == study.PATH_LENGTHS["binary_learners"]
        assert study.LEARNER_ENTRIES[:2] == (2, 4)
        assert limits[0] >= 3.0 and limits[1] >= 1.5, limits

    def test_the_late_entries_and_the_terminal_are_not(
        self, learners_limits: tuple[np.ndarray, float]
    ) -> None:
        limits, _ = learners_limits
        assert study.LEARNER_ENTRIES[2:] == (16, 19)
        assert np.all(np.abs(limits[2:]) <= 0.1), limits


@pytest.mark.parametrize("scenario", study.SCENARIOS)
def test_the_bounds_bind_on_a_declared_share(scenario: str) -> None:
    """The share of ``g`` outside ``[0.025, 0.975]`` lies in ``[0.05, 0.15]`` on both laws."""
    _, g0, _ = _population(study.scenario_law(scenario))
    low, high = study.G_BOUNDS
    share = float(np.mean((g0 < low) | (g0 > high)))
    assert 0.05 <= share <= 0.15, share


def _share_at_alpha(values: Sequence[np.ndarray]) -> float:
    """The share of ``Q`` values at or outside R's ``[1 - alpha, alpha]``."""
    stacked = np.concatenate([np.ravel(value) for value in values])
    low, high = harness.Q_INTERIOR
    return float(np.mean((stacked <= low) | (stacked >= high)))


@pytest.mark.parametrize("scenario", study.SCENARIOS)
def test_the_share_of_q_at_the_alpha_bounds(scenario: str) -> None:
    """Where R's ``alpha`` bound acts on each law, at the population limits (review 2, B1).

    Both sides bound the initial ``Q`` and each accepted update, and only the package bounds a
    candidate's targeted ``Q``.  The initial outcome fit lies inside the bounds on both laws,
    so the bound acts only through the targeting step.  On the lasso law no candidate's
    targeted limit reaches a bound, so the replay's ``clip`` flag fires only by sampling.  On
    the learners law the late candidates reach the low bound on a small share of ``W``, which
    binds on most samples of 2,000 rows.  R therefore writes its learners candidates bounded.
    """
    w, g0, means = _population(study.scenario_law(scenario))
    q0, q1 = _outcome_glm(w, g0, means)
    assert _share_at_alpha([q0, q1]) == 0.0
    targeted = [
        _share_at_alpha(_fluctuated(g0, means, q0, q1, np.clip(g, *study.G_BOUNDS)))
        for g in _columns(scenario, w, g0)
    ]
    if scenario == "binary_lasso":
        assert max(targeted) == 0.0, targeted
    else:
        # A nonzero witness: the bound binds, on at most 0.1 percent of the arm values.
        assert 0.0 < max(targeted) <= 1e-3, targeted


@pytest.mark.parametrize("scenario", study.SCENARIOS)
def test_the_law_matches_the_plan(scenario: str) -> None:
    """The ATE agrees with the plan's 4,000,000-draw Monte Carlo value within its error."""
    expected = {"binary_lasso": 0.08373, "binary_learners": 0.06623}[scenario]
    assert truth_for(study.scenario_law(scenario))["ate"] == pytest.approx(expected, abs=5e-4)


# --------------------------------------------------------------- the outcome-correct law

#: The size at which the outcome learner's limit is read, and the largest gap from the true
#: mean that a correctly specified learner may leave there.  At ``n = 100,000`` the sampling
#: error of each fitted mean is about ``0.003`` at worst, so ``0.005`` binds only when the
#: learner cannot represent the mean.
LIMIT_N = 100_000
LIMIT_TOLERANCE = 0.005

#: The witness size and the cells' size: a bias is read at ``WITNESS_N`` and stated in SDs at
#: ``CELL_N``.
WITNESS_N = 200_000
CELL_N = 700


def test_the_outcome_learner_is_correct_for_the_outcome_correct_law() -> None:
    law = properties.outcome_correct_dgp()
    frame, _ = law.sample(LIMIT_N, seed=20261601, backend="pandas")
    columns = ["A", *law.covariate_names]
    model = properties.correct_outcome_learner().fit(
        frame[columns].to_numpy(float), frame["Y"].to_numpy(float)
    )
    w = np.asarray(law.quadrature(POINTS), dtype=float)[:, : len(law.covariate_names)]
    gaps = []
    for arm in (0.0, 1.0):
        design = np.column_stack([np.full(len(w), arm), w])
        fitted = model.predict(design)
        gaps.append(float(np.max(np.abs(fitted - law.outcome_mean(w, arm, None)))))
    assert max(gaps) <= LIMIT_TOLERANCE, gaps


def _constant_q_bias(law: DGP) -> float:
    """TMLE with a constant ``Q`` and main-terms ``g``: its bias in SDs at ``n = 700``.

    This is the estimate a selector reports when it ignores the outcome learner and ends at
    the main-terms mechanism (plan 3.4.3, B2).  No main-terms mechanism reads ``f``, so the
    targeting cannot remove the confounding ``f`` carries.
    """
    frame, _ = law.sample(WITNESS_N, seed=20261618, backend="pandas")
    estimate = (
        TMLE(
            outcome_learner=DummyRegressor(),
            treatment_learner=LogisticRegression(C=1e10, max_iter=5000),
            cross_fit=False,
            estimands=("ate",),
            g_bounds=study.G_BOUNDS,
            q_bounds=bounded_cv_laws.Q_BOUNDS,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=list(law.covariate_names))
        .single()["ate"]
    )
    sd_at_cell = estimate.std_error * math.sqrt(WITNESS_N / CELL_N)
    return (estimate.psi - law.truth()["ate"]) / sd_at_cell


def test_the_main_terms_projection_is_biased_without_the_outcome_learner() -> None:
    """A nonzero witness: a constant ``Q`` with main-terms ``g`` misses by 1 SD or more.

    The control drops ``f`` from ``g``.  The same witness then lies inside the margin, so the
    witness measures the interaction in ``g`` and not the constant outcome fit.
    """
    assert _constant_q_bias(properties.outcome_correct_dgp()) >= 1.0
    assert abs(_constant_q_bias(properties.outcome_correct_dgp(0.0))) < MARGIN


def test_the_outcome_correct_law_is_the_declared_law() -> None:
    law = properties.outcome_correct_dgp()
    assert law.family == "beta" and law.concentration == 20.0
    assert law.truth()["ate"] == pytest.approx(0.10521, abs=5e-4)
    w = np.asarray(law.quadrature(POINTS), dtype=float)[:, :4]
    g = np.asarray(law.propensity(w), dtype=float)
    assert g.min() >= 0.05 and g.max() <= 0.95
    # The control law differs only in g.
    control = properties.outcome_correct_dgp(0.0)
    for arm in (0.0, 1.0):
        np.testing.assert_array_equal(
            law.outcome_mean(w, arm, None), control.outcome_mean(w, arm, None)
        )
    assert not np.allclose(law.propensity(w), control.propensity(w))


def test_the_multi_arm_law_reports_its_truncation_share() -> None:
    """The share of units the declared ``g_bounds`` move on the multi-arm law: none."""
    process = multi_arm_common.law()
    frame, _ = process.sample(20_000, seed=4242, backend="pandas")
    latent = frame[["W1", "W2", "W3"]].to_numpy()[:, : process.n_latent]
    probabilities = process.probabilities(latent)
    low, high = multi_arm_common.G_BOUNDS
    share = float(((probabilities < low) | (probabilities > high)).any(axis=1).mean())
    assert share == 0.0


#: The sample size at which a candidate's limit bias is read, and the factor by which index 0
#: must clear the standardized-bias margin at the necessity cells' size.  At this seed, along
#: the nested path, index 0 lies about 4.8 standard deviations from the truth, index 1 about
#: 0.14 and the terminal about 0.17.
BIAS_N = 200_000
BIAS_FACTOR = 4.0
#: The first run's declared candidates, ``W2`` then ``W1``: the deliberate-mutation control.
FIRST_RUN_COLUMNS = ((1,), (0,))


@functools.cache
def _path_biases(columns: tuple[tuple[int, ...], ...]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """The TMLE bias at each index of the path the package builds, in SD at the cell's ``n``.

    The fit is the ``learners_collaborative`` cell's own estimator in sample at
    :data:`BIAS_N`, with the declared candidates at ``columns``.  Each index's estimate is its
    targeted fit along the built path, so an index fluctuates the targeted fit of the index
    before it.  A standalone fit of each candidate misses that: index 1 of the first run's
    declaration reads ``W1`` alone and lies about 0.08 SD from the truth standalone, but about
    4.0 SD along the path.  The SD at the cell's ``n`` is the reported candidate's plug-in
    standard error at :data:`BIAS_N`, rescaled by the square root of the sizes.

    Returns
    -------
    tuple
        The bias of each index, and the selection's ``treatment_risk_increases``.
    """
    law = bounded_cv_laws.instrument_dgp()
    (cell,) = [
        cell for cell in properties.declared_cells() if cell.cell == "learners_collaborative"
    ]
    assert cell.dgp.name == law.name and cell.outcome_learner is DummyRegressor
    frame, _ = law.sample(BIAS_N, seed=20261616, backend="pandas")
    estimator = properties._estimator(cell)()
    estimator.cross_fit = False
    estimator.treatment_candidates = [properties._main_terms(entry) for entry in columns]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with harness._captured(trace=True) as capture:
            result = estimator.fit(
                frame, outcome="Y", treatment="A", covariates=list(law.covariate_names)
            ).single()
    sd_at_cell = result["ate"].plugin_std_error * np.sqrt(BIAS_N / cell.n)
    truth = law.truth()["ate"]
    biases = tuple(
        (harness.ate(candidate.targeted) - truth) / sd_at_cell
        for candidate in capture.paths[0].candidates
    )
    return biases, result.extra["ctmle"].treatment_risk_increases


def test_the_declared_learners_are_biased_only_at_index_0_along_the_built_path() -> None:
    """A nonzero witness: a selector that never leaves index 0 fails ``learners_collaborative``.

    Index 0 reads the instrument alone and omits the confounder, so its bias lies far outside
    the standardized-bias margin.  Index 1 adds the confounder and the terminal reads every
    covariate, so both lie inside it along the path, and the treatment loss decreases.
    """
    biases, increases = _path_biases(properties.LEARNER_COLUMNS)
    assert abs(biases[0]) > BIAS_FACTOR * MARGIN
    assert all(abs(bias) < MARGIN for bias in biases[1:]), biases
    assert increases == ()


def test_the_first_run_declaration_is_biased_at_index_1_along_the_path() -> None:
    """The mutation control: ``W2`` then ``W1`` breaks the order, and index 1 inherits index 0.

    The path test above fails on this declaration, and the package's order warning names
    index 1.
    """
    biases, increases = _path_biases(FIRST_RUN_COLUMNS)
    assert abs(biases[1]) > BIAS_FACTOR * MARGIN
    assert increases == (1,)


# --------------------------------------------------------------- the declaration


def test_the_declared_cells_and_seeds() -> None:
    cells = properties.declared_cells()
    declared = {(cell.property, cell.cell) for cell in cells}
    expected = {(family, cell) for family, names in study.PROPERTY_CELLS.items() for cell in names}
    assert declared == expected
    assert all(16_100 <= cell.seed <= 16_400 for cell in cells)
    assert len({cell.seed for cell in cells if cell.property == "double_robustness"}) == 6
    budgets = {(cell.property, cell.cell): (cell.n, cell.replicates) for cell in cells}
    for name in (
        "lasso_outcome_correct",
        "learners_outcome_correct",
        "both_wrong_lasso",
        "both_wrong_learners",
    ):
        assert budgets["double_robustness", name] == (700, 1_200)
    for name in ("lasso_treatment_correct", "learners_treatment_correct"):
        assert budgets["double_robustness", name] == (2_000, 1_200)
    for name in ("lasso_collaborative", "learners_collaborative", "empty_control"):
        assert budgets["selector_necessity", name] == (1_500, 800)
    for name in ("lasso_treatment_correct", "empty_control"):
        assert budgets["multi_arm", name] == (2_000, 1_200)
    necessity = {cell.seed for cell in cells if cell.property == "selector_necessity"}
    assert len(necessity) == 1, "the necessity cells must share one seed to pair"
    two_arm = [cell for cell in cells if cell.property != "multi_arm"]
    assert all(cell.dgp.family == "beta" for cell in two_arm)


def test_each_control_fits_the_strategy_under_test() -> None:
    """The both-wrong controls fit lasso and learners, with a constant outcome (plan 3.4.3)."""
    cells = {cell.cell: cell for cell in properties.declared_cells()}
    for name, strategy in (("both_wrong_lasso", "lasso"), ("both_wrong_learners", "learners")):
        cell = cells[name]
        assert cell.role == "control" and cell.outcome_learner is DummyRegressor
        assert cell.dgp is properties.outcome_correct_dgp()
        assert properties._estimator(cell)().strategy == strategy
    for name in ("lasso_outcome_correct", "learners_outcome_correct"):
        assert cells[name].outcome_learner is properties.correct_outcome_learner


def test_each_property_grid_follows_the_declared_rule() -> None:
    lasso = [
        cell for cell in properties.declared_cells() if properties._strategy(cell.cell) == "lasso"
    ]
    assert len(lasso) == 5
    for cell in lasso:
        grid = np.asarray(properties.property_grid(cell))
        assert grid.size == properties.PROPERTY_GRID_SIZE
        assert grid[-1] == 0.0
        assert np.all(np.diff(grid) < 0.0)
        assert grid[-2] == pytest.approx(grid[0] * study.GRID_RATIO, rel=1e-12)


@pytest.mark.parametrize("scenario", study.SCENARIOS)
def test_the_primary_grid_rule(scenario: str) -> None:
    grid = np.asarray(study.primary_grid(scenario))
    assert grid.size == study.GRID_SIZE
    assert np.all(np.diff(grid) < 0.0)
    assert grid[-1] == pytest.approx(grid[0] * study.GRID_RATIO, rel=1e-12)


def test_the_learner_entries() -> None:
    assert study.LEARNER_ENTRIES == (2, 4, 16, 19)
    if not study.LAMBDAS_FILE.exists():
        pytest.skip("lambdas.csv is written and committed in the declaration step")
    grid = study.study_lambdas("binary_learners")
    assert study.learner_lambdas() == tuple(grid[entry - 1] for entry in (2, 4, 16, 19))


def test_the_committed_grids_are_the_rule_s_output() -> None:
    """``lambdas.csv`` holds one grid per scenario, each by the rule on that scenario's law."""
    if not study.LAMBDAS_FILE.exists():
        pytest.skip("lambdas.csv is written and committed in the declaration step")
    # The default pandas parser can miss a 17-digit value by about 1e-13. The tolerance is
    # a few ulps: the rule's lambda_max is a sum whose rounding differs between machines.
    committed = pd.read_csv(study.LAMBDAS_FILE, float_precision="round_trip")
    assert list(committed.columns) == ["scenario", "entry", "lambda"]
    for scenario in study.SCENARIOS:
        rows = committed.loc[committed["scenario"] == scenario]
        np.testing.assert_allclose(
            rows["lambda"].to_numpy(dtype=float),
            study.primary_grid(scenario),
            rtol=1e-15,
            atol=0.0,
        )


def test_the_study_imports_without_its_declaration_inputs() -> None:
    """The record and the grid rule need neither committed input; a fit names the missing one."""
    assert study.STUDY.slug == "canonical-ctmle-sequences"
    if study.LAMBDAS_FILE.exists():
        pytest.skip("lambdas.csv is committed")
    with pytest.raises(FileNotFoundError, match=r"lambdas\.csv"):
        study.study_lambdas("binary_lasso")


def test_the_paired_comparison_is_descriptive() -> None:
    assert study.STUDY.property_artifacts == (study.PROPERTY_SELECTION_FILE,)
    assert study.STUDY.paired_role == "descriptive"
    assert study.STUDY.extra_artifacts == study.REFERENCE_SIDE_ARTIFACTS


# --------------------------------------------------------------- the R runner and its rows


def test_the_r_runner_writes_the_columns_the_study_reads() -> None:
    """The R row's extra columns are :data:`REFERENCE_COLUMNS`, the replay's included."""
    from tests.canonical import runner_source
    from tests.studies.evidence.schema import REPLICATE_COLUMNS
    from tests.unit.test_canonical_runner_parity import published_keys

    runner = study.ARTIFACTS / "run_ctmle_sequences.R"
    keys = published_keys(runner_source(runner))
    assert keys[len(REPLICATE_COLUMNS) :] == study.REFERENCE_COLUMNS
    assert len(study.FOLD_PATH_COLUMNS) == study.SELECTION_FOLDS


def test_the_r_runner_reads_the_declared_entries_and_quirk_text() -> None:
    from tests.canonical import runner_source

    source = runner_source(study.ARTIFACTS / "run_ctmle_sequences.R")
    entries = ", ".join(f"{entry}L" for entry in study.LEARNER_ENTRIES)
    assert f"LEARNER_ENTRIES <- c({entries})" in source
    assert f'QUIRK_7_TEXT <- "{study.QUIRK_7_TEXT}"' in source
    assert 'COVARIATES <- c("W1", "W2", "W3", "W4")' in source


def _reference_row(scenario: str, replicate: int, **values: object) -> dict[str, object]:
    lasso = scenario == "binary_lasso"
    row: dict[str, object] = {
        "implementation": study.REFERENCE,
        "scenario": scenario,
        "replicate": replicate,
        "truth": 0.1,
        "estimate": 0.12 + 0.01 * replicate,
        "best_k": 21 if lasso else 5,
        "n_candidates": 3,
        "adapter_used": int(lasso),
        "path_min_lambda": 1e-5 if lasso else np.nan,
        **{
            column: (1e-5 + index * 1e-7) if lasso else np.nan
            for index, column in enumerate(study.FOLD_PATH_COLUMNS)
        },
        "adapter_error_matched": int(lasso) if lasso else np.nan,
        "rebuild_best_k": 21 if lasso else np.nan,
        "rebuild_agrees": np.nan,
        "r_accepted": "3;21" if lasso else "5",
    }
    for column in study.REFERENCE_COLUMNS:
        row.setdefault(column, np.nan)
    row.update(values)
    return row


def _package_row(scenario: str, replicate: int, selected: int) -> dict[str, object]:
    return {
        "implementation": study.IMPLEMENTATION,
        "scenario": scenario,
        "replicate": replicate,
        "truth": 0.1,
        "estimate": 0.1 + 0.01 * replicate,
        "package_selected": selected,
        "package_final": 21 if selected < 21 else np.nan,
        "package_accepted": "4;21",
    }


def _rows(adapter: Sequence[int]) -> pd.DataFrame:
    rows = []
    for replicate, used in enumerate(adapter):
        best_k = 21 if used else 5
        rows.append(
            _reference_row(
                "binary_lasso",
                replicate,
                best_k=best_k,
                adapter_used=used,
                adapter_error_matched=used,
                rebuild_best_k=best_k,
                rebuild_agrees=np.nan if used else 1,
            )
        )
        rows.append(_package_row("binary_lasso", replicate, 7))
    rows.append(_reference_row("binary_learners", 0))
    rows.append(_package_row("binary_learners", 0, 5))
    return pd.DataFrame(rows)


def test_r_selection_keeps_every_path_minimum_and_both_selections() -> None:
    selection = study.selection_rows(_rows((1,)))
    assert list(selection.columns) == [
        "scenario",
        "replicate",
        *study.SELECTION_REFERENCE_COLUMNS,
        *study.PACKAGE_COLUMNS,
    ]
    lasso = selection.loc[selection["scenario"] == "binary_lasso"].iloc[0]
    assert lasso["path_min_lambda_fold_5"] == pytest.approx(1e-5 + 4e-7, rel=0, abs=1e-18)
    assert lasso["package_selected"] == 7 and lasso["r_accepted"] == "3;21"


def test_r_selection_refuses_an_adapter_without_the_quirk_7_message() -> None:
    rows = _rows((1,))
    rows.loc[rows["implementation"] == study.REFERENCE, "adapter_error_matched"] = 0
    with pytest.raises(RuntimeError, match="quirk-7"):
        study.selection_rows(rows)


def test_r_selection_refuses_a_rebuild_disagreement() -> None:
    rows = _rows((0,))
    lasso = (rows["implementation"] == study.REFERENCE) & (rows["scenario"] == "binary_lasso")
    rows.loc[lasso, "rebuild_agrees"] = 0
    with pytest.raises(RuntimeError, match="rebuild"):
        study.selection_rows(rows)


def test_the_selection_summary_pairs_the_replicates_r_fit_directly() -> None:
    rows = _rows((0, 1, 0))
    summary = study.selection_summary(rows, study.selection_rows(rows))
    lasso = summary.loc[summary["scenario"] == "binary_lasso"].iloc[0]
    assert lasso["adapter_replicates"] == 1 and lasso["compared_replicates"] == 2
    assert lasso["r_selected_last"] == 1 and lasso["package_selected_last"] == 0
    assert lasso["mean_difference"] == pytest.approx(-0.02)
    assert list(summary["scenario"]) == list(study.SCENARIOS)


def test_the_selection_summary_survives_an_adapter_on_every_replicate() -> None:
    """No replicate is compared, so the pivot has no columns and every statistic is NaN."""
    rows = _rows((1, 1, 1))
    summary = study.selection_summary(rows, study.selection_rows(rows))
    lasso = summary.loc[summary["scenario"] == "binary_lasso"].iloc[0]
    assert lasso["adapter_replicates"] == 3 and lasso["compared_replicates"] == 0
    assert np.isnan(lasso["mean_difference"]) and np.isnan(lasso["rmse_cleverly"])


def test_accepted_indices_read_the_blocks_of_n_steps() -> None:
    assert study.accepted_indices((1, 1, 2, 3, 3, 3)) == (1, 2, 5)
    with pytest.raises(RuntimeError, match="do not count"):
        study.accepted_indices((1, 3, 3))


# --------------------------------------------------------------- pass probability and power

#: The inputs of plan 3.4.3 that the law pilot does not rerun: each cell's bias in its own SD
#: and its replicate count.  A ``discriminated`` control must lie outside the margin.  The
#: keys of ``power_probe.PROPERTY_PILOT_CELLS`` hold the property pilot's measurements, which
#: :func:`test_the_declared_gates_hold_the_property_pilot` reads back from
#: ``property-pilot.csv``.  The keys of ``study.STAND_INS`` hold stand-ins from other studies.
DECLARED_GATES = {
    "double_robustness/lasso_outcome_correct": (-0.042, 1_200, "equivalent"),
    "double_robustness/learners_outcome_correct": (-0.007, 1_200, "equivalent"),
    "double_robustness/lasso_treatment_correct": (-0.042, 1_200, "equivalent"),
    "double_robustness/learners_treatment_correct": (-0.080, 1_200, "equivalent"),
    "double_robustness/both_wrong_lasso": (4.48, 1_200, "discriminated"),
    "double_robustness/both_wrong_learners": (4.511, 1_200, "discriminated"),
    "selector_necessity/empty_control": (5.81, 800, "discriminated"),
    "multi_arm/lasso_treatment_correct": (-0.041, 1_200, "equivalent"),
    "multi_arm/empty_control": (2.761, 1_200, "discriminated"),
}
#: The declared expected red of the necessity family: ``learners_collaborative``'s bias clause
#: (design review S3), by its pilot bias and replicates.  It is not a gate, so its pass
#: probability enters no product.  The family's discriminating gate is the RMSE bound.
DECLARED_EXPECTED_RED = {
    "selector_necessity/learners_collaborative": (0.315, 800),
}
#: The rounding of a measured bias in :data:`DECLARED_GATES`.
DECLARED_DIGITS = 3
#: The mutations each truth gate must catch, by the pilot row that measures them.  ``no
#: sequence`` is absent: the replay gates and the structural invariants catch it (plan 3.4.1).
MUTATIONS = (
    ("binary_lasso", "no_targeting"),
    ("binary_learners", "no_targeting"),
    ("binary_lasso", "no_final_step"),
    ("binary_learners", "stuck_at_0"),
)
#: The B2 mutation (a selector that ignores the outcome learner), caught by the
#: outcome-correct cells at 1,200 replicates.
B2_MUTATION = 3.70


def pass_probability(bias: float, replicates: int, *, rule: str = "equivalent") -> float:
    """The normal approximation of plan 3.4.3 to a bias gate's pass probability.

    Parameters
    ----------
    bias : float
        The bias in the estimator's own SD.
    replicates : int
        The cell's replicates.
    rule : str
        ``"equivalent"`` for a gate inside the margin, ``"discriminated"`` for a control
        outside it.

    Returns
    -------
    float
        The probability that the bias interval passes its rule.
    """
    s = 1.0 / math.sqrt(replicates)
    if rule == "discriminated":
        return float(norm.cdf((abs(bias) - (MARGIN + 2.576 * s)) / s))
    t = MARGIN - 2.576 * s
    return float(norm.cdf((t - bias) / s) - norm.cdf((-t - bias) / s))


def test_the_pass_probability_reproduces_the_plan() -> None:
    """The formula against two values the plan prints: 0.99998 and the 0.9988 power."""
    assert pass_probability(-0.011, 800) == pytest.approx(0.99998, abs=5e-6)
    assert 1.0 - pass_probability(0.266, 800) == pytest.approx(0.9988, abs=2e-4)


@pytest.fixture(scope="module")
def probe() -> pd.DataFrame:
    if not power_probe.PROBE_FILE.exists():
        pytest.skip("power-probe.csv is written and committed in the declaration step")
    return pd.read_csv(power_probe.PROBE_FILE).set_index(["scenario", "estimator"])


def test_the_probe_ran_on_the_declared_budget(probe: pd.DataFrame) -> None:
    assert set(probe.index) == {
        (scenario, name) for scenario in study.SCENARIOS for name in power_probe.ESTIMATORS
    }
    assert (probe["replicates"] == power_probe.PILOT_REPLICATES).all()


def test_every_gate_passes_when_correct(probe: pd.DataFrame) -> None:
    probabilities = {
        f"package bias/{scenario}": pass_probability(
            float(probe.loc[(scenario, "ctmle"), "bias_sd"]), study.PRIMARY_REPLICATES
        )
        for scenario in study.SCENARIOS
    }
    probabilities.update(
        {
            name: pass_probability(bias, replicates, rule=rule)
            for name, (bias, replicates, rule) in DECLARED_GATES.items()
        }
    )
    assert all(value >= 0.99 for value in probabilities.values()), probabilities
    assert math.prod(probabilities.values()) >= 0.99, probabilities


def test_every_mutation_has_power(probe: pd.DataFrame) -> None:
    power = {
        f"{scenario}/{name}": 1.0
        - pass_probability(float(probe.loc[(scenario, name), "bias_sd"]), study.PRIMARY_REPLICATES)
        for scenario, name in MUTATIONS
    }
    power["B2"] = 1.0 - pass_probability(B2_MUTATION, properties.ROBUST_REPLICATES)
    assert all(value >= 0.8 for value in power.values()), power


def test_no_stand_in_bias_survives_into_the_run() -> None:
    """The study's manifest exists only after the run, by which time every stand-in is replaced."""
    assert set(DECLARED_GATES) >= study.STAND_INS
    assert study.STAND_INS.isdisjoint(power_probe.PROPERTY_PILOT_CELLS)
    if (study.ARTIFACTS / "manifest.json").exists():
        assert not study.STAND_INS, f"these gates hold stand-in biases: {sorted(study.STAND_INS)}"


@pytest.fixture(scope="module")
def property_pilot() -> pd.DataFrame:
    if not power_probe.PROPERTY_PILOT_FILE.exists():
        pytest.skip("property-pilot.csv is written and committed in the declaration step")
    frame = pd.read_csv(power_probe.PROPERTY_PILOT_FILE)
    return frame.set_index(frame["property"] + "/" + frame["cell"])


def test_the_property_pilot_ran_on_the_declared_budget(property_pilot: pd.DataFrame) -> None:
    assert list(property_pilot.index) == list(power_probe.PROPERTY_PILOT_CELLS)
    assert (property_pilot["replicates"] == power_probe.PROPERTY_PILOT_REPLICATES).all()
    assert (property_pilot["failed_replicates"] == 0).all()
    cells = {f"{cell.property}/{cell.cell}": cell for cell in power_probe.pilot_cells()}
    for key, row in property_pilot.iterrows():
        assert row["role"] == cells[key].role and row["n"] == cells[key].n


def test_the_declared_gates_hold_the_property_pilot(property_pilot: pd.DataFrame) -> None:
    """Each measured bias, rounded, with a Monte Carlo SE below a tenth of its gate margin.

    The gate margin of an equivalent cell is :data:`MARGIN`.  A discriminated control's is
    its bias's distance outside the margin.
    """
    for key, row in property_pilot.iterrows():
        if key in DECLARED_EXPECTED_RED:
            bias, _ = DECLARED_EXPECTED_RED[key]
            assert bias == round(float(row["bias_sd"]), DECLARED_DIGITS), key
            continue
        bias, _, rule = DECLARED_GATES[key]
        assert bias == round(float(row["bias_sd"]), DECLARED_DIGITS), key
        gate_margin = MARGIN if rule == "equivalent" else abs(bias) - MARGIN
        assert row["bias_sd_mc_se"] < gate_margin / 10, key


def test_the_expected_red_bias_clause_is_expected_to_fail() -> None:
    """The nested declaration keeps the S3 expected red: its clause passes with probability < 0.05."""
    for key, (bias, replicates) in DECLARED_EXPECTED_RED.items():
        assert pass_probability(bias, replicates) < 0.05, key


def test_the_pilot_seeds_miss_every_registered_replicate() -> None:
    registered = power_probe.registered_seeds()
    for cell in power_probe.pilot_cells():
        seeds = power_probe.property_pilot_seeds(cell, 50)
        assert len(set(seeds)) == 50 and registered.isdisjoint(seeds)


def test_the_standardized_bias_se_is_the_normal_delta_method() -> None:
    errors = np.random.default_rng(0).normal(0.5, 1.0, 200_000)
    expected = math.sqrt((1.0 + 0.5**2 / 2.0) / errors.size)
    assert power_probe.standardized_bias_se(errors) == pytest.approx(expected, rel=0.02)


@pytest.mark.parametrize(
    "arguments, refused",
    [
        ([], True),
        (["--replicates", str(study.PRIMARY_REPLICATES)], True),
        (["--skip-r", "--jobs", "4"], True),
        (["--replicates", "10", "--primary-only"], False),
        (["--replicates=10"], False),
        (["--primary-only"], False),
    ],
)
def test_a_full_run_refuses_a_stand_in(arguments: list[str], refused: bool) -> None:
    stand_ins = frozenset({"multi_arm/empty_control"})
    if refused:
        with pytest.raises(SystemExit, match="multi_arm/empty_control"):
            regenerate.refuse_stand_ins(arguments, stand_ins)
    else:
        regenerate.refuse_stand_ins(arguments, stand_ins)
    regenerate.refuse_stand_ins(arguments, frozenset())


def test_the_expected_red_rows_are_declared_rows() -> None:
    declared = {f"{cell.property}/{cell.cell}" for cell in properties.declared_cells()}
    assert set(study.EXPECTED_RED_PROPERTIES) <= declared
    assert set(study.EXPECTED_JOINT_CLAUSE_RED) <= set(study.EXPECTED_RED_PROPERTIES)
    truth = {
        f"{study.IMPLEMENTATION}/{scenario}/{estimand}"
        for scenario, estimands in study.STUDY.scenarios.items()
        for estimand in estimands
    }
    assert set(study.EXPECTED_RED_TRUTH) <= truth
    assert set(DECLARED_EXPECTED_RED).isdisjoint(DECLARED_GATES)
    assert set(DECLARED_EXPECTED_RED) <= set(study.EXPECTED_RED_PROPERTIES)


def test_the_committed_red_set_is_the_expected_red_set() -> None:
    """The run's red rows are the declared expected reds, each failing as declared.

    ``learners_collaborative`` fails its own bias rule, and the family's other two cells fail
    the joint clause only.  The RMSE bound of each collaborative cell passes, since it is the
    family's discriminating gate.
    """
    if not study.STUDY.artifact("properties.csv").exists():
        pytest.skip("the registered run has not been committed yet")
    from tests.studies.evidence import red_cells

    tables = red_cells.frames(study.STUDY)
    red = {row.key.key: row.fails_by for row in red_cells.red_rows(study.STUDY, tables)}
    assert set(red) == set(study.EXPECTED_RED)
    for key in study.EXPECTED_JOINT_CLAUSE_RED:
        assert red[key] == "the family's joint clause"
    assert red["selector_necessity/learners_collaborative"] == "its own rule"
    summary = tables["properties"].set_index(["property", "cell"])
    for cell in ("lasso_collaborative", "learners_collaborative"):
        bound = float(summary.loc[("selector_necessity", cell), "rmse_ratio_upper"])
        assert bound <= properties.SELECTOR_RMSE_RATIO
