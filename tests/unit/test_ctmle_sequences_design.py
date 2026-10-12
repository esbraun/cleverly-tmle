"""Pre-run design checks of the registered study ``canonical-ctmle-sequences``.

Each check reads a law or a declaration, never a committed result, so it holds before the run.
The outcome-correct law has two claims that the double-robustness cells rest on: the declared
outcome learner is correct for it, and no main-terms logistic mechanism is.  A cell labelled
"outcome correct" on a law that broke either claim would measure something else and still
publish under its label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from tests.studies import canonical_ctmle_sequences as study
from tests.studies import ctmle_sequence_properties as properties
from tests.studies import multi_arm_common
from tests.studies.fractional_glm import QuasiBinomialGLM

#: The size at which the outcome learner's limit is read, and the largest gap from the true
#: mean that a correctly specified learner may leave there.  At ``n = 100,000`` the sampling
#: error of each fitted mean is about ``0.003`` at worst, so ``0.005`` binds only when the
#: learner cannot represent the mean.
LIMIT_N = 100_000
LIMIT_TOLERANCE = 0.005

#: Quadrature points for the population quantities.
POINTS = 2**14


def _quadrature() -> np.ndarray:
    return np.asarray(properties.outcome_correct_dgp().quadrature(POINTS), dtype=float)


def test_the_outcome_learner_is_correct_for_the_outcome_correct_law() -> None:
    law = properties.outcome_correct_dgp()
    frame, _ = law.sample(LIMIT_N, seed=20261601, backend="pandas")
    columns = ["A", *law.covariate_names]
    model = QuasiBinomialGLM().fit(frame[columns].to_numpy(float), frame["Y"].to_numpy(float))
    w = _quadrature()
    gaps = []
    for arm in (0.0, 1.0):
        design = np.column_stack([np.full(len(w), arm), w[:, : len(law.covariate_names)]])
        fitted = model.predict(design)
        gaps.append(float(np.max(np.abs(fitted - law.outcome_mean(w, arm, None)))))
    assert max(gaps) <= LIMIT_TOLERANCE, gaps


def _projection_kl(propensity: np.ndarray, covariates: np.ndarray) -> float:
    """``E[KL(g0 || g*)]`` with ``g*`` the main-terms logistic projection of ``g0``."""
    stacked = np.vstack([covariates, covariates])
    labels = np.concatenate([np.ones(len(covariates)), np.zeros(len(covariates))])
    weights = np.concatenate([propensity, 1.0 - propensity])
    projection = LogisticRegression(C=1e10, max_iter=5000, tol=1e-12)
    projection.fit(stacked, labels, sample_weight=weights)
    fitted = projection.predict_proba(covariates)[:, 1]
    return float(
        np.mean(
            propensity * np.log(propensity / fitted)
            + (1.0 - propensity) * np.log((1.0 - propensity) / (1.0 - fitted))
        )
    )


def test_no_main_terms_logistic_mechanism_is_correct_for_the_outcome_correct_law() -> None:
    """The whole lasso path is misspecified: the projection's KL gap clears the floor."""
    law = properties.outcome_correct_dgp()
    w = _quadrature()
    covariates = w[:, : len(law.covariate_names)]
    gap = _projection_kl(np.asarray(law.propensity(w), dtype=float), covariates)
    assert gap > properties.MINIMUM_PROJECTION_KL, gap


def test_the_kl_floor_detects_a_main_terms_propensity() -> None:
    """A mutation control: the bounded linear law's own propensity has no gap."""
    from tests.studies import bounded_cv_laws

    law = bounded_cv_laws.linear_dgp()
    w = _quadrature()
    covariates = w[:, : len(law.covariate_names)]
    gap = _projection_kl(np.asarray(law.propensity(w), dtype=float), covariates)
    assert gap < properties.MINIMUM_PROJECTION_KL / 100, gap


def test_the_outcome_correct_law_keeps_the_bounded_linear_outcome() -> None:
    from tests.studies import bounded_cv_laws

    law = properties.outcome_correct_dgp()
    twin = bounded_cv_laws.linear_dgp()
    w = _quadrature()
    for arm in (0.0, 1.0):
        np.testing.assert_array_equal(
            law.outcome_mean(w, arm, None), twin.outcome_mean(w, arm, None)
        )
    assert law.truth()["ate"] == pytest.approx(twin.truth()["ate"], abs=1e-12)
    assert not np.allclose(law.propensity(w), twin.propensity(w))


def test_the_multi_arm_law_reports_its_truncation_share() -> None:
    """The share of units the declared ``g_bounds`` move on the multi-arm law: none."""
    process = multi_arm_common.law()
    frame, _ = process.sample(20_000, seed=4242, backend="pandas")
    latent = frame[["W1", "W2", "W3"]].to_numpy()[:, : process.n_latent]
    probabilities = process.probabilities(latent)
    low, high = multi_arm_common.G_BOUNDS
    share = float(((probabilities < low) | (probabilities > high)).any(axis=1).mean())
    assert share == 0.0


def test_the_declared_cells_and_seeds() -> None:
    cells = properties.declared_cells()
    declared = {(cell.property, cell.cell) for cell in cells}
    expected = {(family, cell) for family, names in study.PROPERTY_CELLS.items() for cell in names}
    assert declared == expected
    assert all(16_100 <= cell.seed <= 16_400 for cell in cells)
    budgets = {(cell.property, cell.cell): (cell.n, cell.replicates) for cell in cells}
    for name in ("lasso_outcome_correct", "learners_outcome_correct", "both_wrong"):
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


def test_each_property_grid_follows_the_declared_rule() -> None:
    lasso = [cell for cell in properties.declared_cells() if cell.cell.startswith("lasso_")]
    assert len(lasso) == 4
    for cell in lasso:
        grid = np.asarray(properties.property_grid(cell))
        assert grid.size == properties.PROPERTY_GRID_SIZE
        assert grid[-1] == 0.0
        assert np.all(np.diff(grid) < 0.0)
        assert grid[-2] == pytest.approx(grid[0] * study.GRID_RATIO, rel=1e-12)


def test_the_primary_grid_rule() -> None:
    grid = np.asarray(study.primary_grid())
    assert grid.size == study.GRID_SIZE
    assert np.all(np.diff(grid) < 0.0)
    assert grid[-1] == pytest.approx(grid[0] * study.GRID_RATIO, rel=1e-12)
    assert study.LEARNER_ENTRIES == (4, 8, 12, 16)


def test_the_committed_grid_is_the_rule_s_output() -> None:
    """``lambdas.csv`` is written in the declaration step; before it, there is nothing to read."""
    if not study.LAMBDAS_FILE.exists():
        pytest.skip("lambdas.csv is written and committed in the declaration step")
    committed = pd.read_csv(study.LAMBDAS_FILE)["lambda"].to_numpy(dtype=float)
    np.testing.assert_allclose(committed, study.primary_grid(), rtol=1e-15, atol=0.0)


def test_the_study_imports_without_its_declaration_inputs() -> None:
    """The record and the grid rule need neither committed input; a fit names the missing one."""
    assert study.STUDY.slug == "canonical-ctmle-sequences"
    if study.LAMBDAS_FILE.exists():
        pytest.skip("lambdas.csv is committed")
    with pytest.raises(FileNotFoundError, match=r"lambdas\.csv"):
        study.study_lambdas()
