"""The R ``ctmle`` candidate-sequence witness (plan X16, section 3.2).

``tests/canonical/ctmle_sequence_witness`` holds R ``ctmle`` and ``glmnet`` outputs on one
fixture sample. The rows here compare the package with them: the L1 solver with ``glmnet``,
and the indexed construction, the fluctuation and the C-TMLE1 final step with
``ctmleGeneral`` and ``ctmleGlmnet`` on R's own propensity arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.inference import influence_variance
from cleverly.learners import _l1_logistic as l1
from tests.canonical.ctmle_sequence_witness import harness
from tests.canonical.ctmle_sequence_witness.regenerate import (
    FINAL_MOVE,
    HERE,
    OUTPUTS,
    SOURCES,
    matrix,
    read,
    read_facts,
)

MANIFEST = HERE / "manifest.json"

pytestmark = pytest.mark.skipif(
    not MANIFEST.exists(),
    reason="the witness outputs are not committed yet; the local thread runs regenerate.py",
)


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inputs() -> pd.DataFrame:
    return read(HERE / "inputs.csv")


@pytest.fixture(scope="module")
def grid() -> np.ndarray:
    return read(HERE / "grid.csv")["lambda"].to_numpy(float)


@pytest.fixture(scope="module")
def facts() -> dict[str, str]:
    return read_facts(HERE)


@pytest.fixture(scope="module")
def chooser(inputs: pd.DataFrame) -> Any:
    return harness.selector(inputs)


@pytest.fixture(scope="module")
def x(inputs: pd.DataFrame) -> np.ndarray:
    return inputs[list(harness.COVARIATES)].to_numpy(float)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTheFixture:
    def test_every_file_matches_its_recorded_hash(self, manifest: dict[str, Any]) -> None:
        names = ("inputs.csv", "grid.csv", *OUTPUTS, *SOURCES)
        assert {name: _sha256(HERE / name) for name in names} == {
            name: manifest["sha256"][name] for name in names
        }

    def test_the_seed_redraws_the_inputs(
        self, manifest: dict[str, Any], inputs: pd.DataFrame, grid: np.ndarray
    ) -> None:
        frame, drawn = harness.draw_inputs(int(manifest["seed"]))
        frame = frame.reset_index(drop=True)
        # The draw and the grid are exact. ``Q0W`` and ``Q1W`` are a refit, so they agree to
        # the rounding of the machine's linear algebra only.
        fitted = ["Q0W", "Q1W"]
        pd.testing.assert_frame_equal(
            frame.drop(columns=fitted),
            inputs.drop(columns=fitted),
            check_dtype=False,
            check_exact=True,
        )
        np.testing.assert_allclose(frame[fitted], inputs[fitted], rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(drawn, grid)

    def test_the_initial_fit_is_interior(self, inputs: pd.DataFrame) -> None:
        q = inputs[["Q0W", "Q1W"]].to_numpy()
        low, high = harness.Q_RANGE
        assert low <= q.min() and q.max() <= high


# ---------------------------------------------------------------- the solver rows


def _glmnet_fit(row: pd.Series, x: np.ndarray, w: np.ndarray | None) -> l1.L1Fit:
    """A binary ``glmnet`` solution, restated on the package's standardized scale."""
    beta = row[list(harness.COVARIATES)].to_numpy(float)
    center, scale = l1._standardization(x, l1._normalized(np.ones(len(x)) if w is None else w))
    intercept = float(row["a0"]) + float(beta @ center)
    coef = np.column_stack([np.zeros(beta.size), beta * scale])
    return l1.L1Fit(
        penalty=float(row["lambda"]),
        intercept=np.array([0.0, intercept]),
        coef=coef,
        center=center,
        scale=scale,
        constant=None,
        kkt=np.nan,
    )


class TestTheSolver:
    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_binary(
        self,
        inputs: pd.DataFrame,
        x: np.ndarray,
        grid: np.ndarray,
        manifest: dict[str, Any],
        weighted: bool,
    ) -> None:
        a = inputs["A"].to_numpy(float)
        w = inputs["w"].to_numpy(float) if weighted else None
        name = "weighted" if weighted else "tight"
        fits = l1.penalty_path(x, a, w, grid)
        # (a) The package meets its own tolerance.
        assert max(fit.kkt for fit in fits) <= l1.KKT_TOLERANCE
        # (b) The package residual reads glmnet's coefficients as near-optimal on the
        # same objective.
        coefficients = read(HERE / f"coef_{name}.csv")
        residuals = [
            l1.kkt_residual(x, a, w, _glmnet_fit(row, x, w)) for _, row in coefficients.iterrows()
        ]
        assert max(residuals) <= 1e-6
        # (c) The predictions agree within ten times glmnet's measured self-gap.
        package = np.column_stack([fit.predict_proba(x)[:, 1] for fit in fits])
        tolerance = manifest["prediction_tolerance"]["weighted" if weighted else "binary"]
        assert np.max(np.abs(package - matrix(HERE, f"g_{name}.csv"))) <= tolerance

    def test_multinomial(
        self, inputs: pd.DataFrame, x: np.ndarray, grid: np.ndarray, manifest: dict[str, Any]
    ) -> None:
        a3 = inputs["A3"].to_numpy()
        fits = l1.penalty_path(x, a3, None, grid)
        assert max(fit.kkt for fit in fits) <= l1.KKT_TOLERANCE
        r = read(HERE / "g_multinomial.csv")
        classes = np.unique(a3)
        package = np.concatenate([fit.predict_proba(x) for fit in fits], axis=1)
        reference = np.column_stack(
            [r[f"k{k + 1:02d}_a{c}"] for k in range(len(grid)) for c in classes]
        )
        assert (
            np.max(np.abs(package - reference)) <= manifest["prediction_tolerance"]["multinomial"]
        )

    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_lambda_max(
        self, inputs: pd.DataFrame, x: np.ndarray, facts: dict[str, str], weighted: bool
    ) -> None:
        a = inputs["A"].to_numpy(float)
        w = inputs["w"].to_numpy(float) if weighted else None
        key = "lambda_max_weighted" if weighted else "lambda_max"
        top = l1.default_penalty_grid(x, a, w)[0]
        assert top == pytest.approx(float(facts[key]), rel=1e-10)


# ------------------------------------------------------------ the selection rows


@pytest.fixture(scope="module")
def internal() -> np.ndarray:
    return matrix(HERE, "g_internal.csv")


@pytest.fixture(scope="module")
def lasso_path(chooser: Any, internal: np.ndarray) -> harness.Path:
    return harness.indexed_path(chooser, internal)


class TestCtmleGeneral:
    @pytest.fixture(scope="class")
    def compared(self, chooser: Any, internal: np.ndarray) -> tuple[harness.Path, pd.DataFrame]:
        columns = np.column_stack([internal, matrix(HERE, "g_glm.csv")])
        return harness.indexed_path(chooser, columns), read(HERE / "general_path.csv")

    def test_the_accepted_indices_are_exact(self, compared: Any) -> None:
        path, r = compared
        assert path.accepted == tuple(int(i) - 1 for i in r.loc[r["accepted"] == 1, "index"])

    def test_estimates_and_losses(self, compared: Any, inputs: pd.DataFrame) -> None:
        path, r = compared
        y = inputs["Y"].to_numpy(float)
        np.testing.assert_allclose(path.estimates, r["est"], rtol=0.0, atol=harness.GLM_TOLERANCE)
        np.testing.assert_allclose(path.likelihoods(y), r["likelihood"], rtol=harness.GLM_TOLERANCE)


class TestCtmleGlmnet:
    def test_the_accepted_penalties_are_exact(self, lasso_path: harness.Path) -> None:
        # R marks acceptance at full precision (``grid %in% candidates$lambdas``), but
        # ``write.csv`` prints the ``lambda`` column to 15 digits. The test compares the
        # positions R marks; the printed penalty identifies each row only.
        r = read(HERE / "glmnet_path.csv")
        assert lasso_path.accepted == tuple(int(i) - 1 for i in r.loc[r["accepted"] == 1, "index"])

    def test_estimates_and_losses(self, lasso_path: harness.Path, inputs: pd.DataFrame) -> None:
        r = read(HERE / "glmnet_path.csv")
        y = inputs["Y"].to_numpy(float)
        np.testing.assert_allclose(
            lasso_path.estimates, r["est"], rtol=0.0, atol=harness.GLM_TOLERANCE
        )
        np.testing.assert_allclose(
            lasso_path.likelihoods(y), r["likelihood"], rtol=harness.GLM_TOLERANCE
        )

    def test_the_final_step(
        self,
        chooser: Any,
        lasso_path: harness.Path,
        internal: np.ndarray,
        facts: dict[str, str],
    ) -> None:
        selected = int(facts["glmnet_best_k"]) - 1
        _, estimate = harness.final_step(
            chooser, lasso_path, internal, matrix(HERE, "g_final_cols.csv"), selected
        )
        assert estimate == pytest.approx(float(facts["glmnet_est"]), abs=harness.GLM_TOLERANCE)

    def test_the_plugin_variance_is_r_var_psi(
        self,
        chooser: Any,
        lasso_path: harness.Path,
        internal: np.ndarray,
        facts: dict[str, str],
    ) -> None:
        """``plugin_std_error`` squared equals ``var.psi`` of the type-1 final step.

        Both sides divide ``var(D*)`` with the ``n - 1`` denominator by ``n``
        (``ctmle_glmnet.R``, line 316, and :func:`~cleverly.inference.influence_variance`).
        The measured relative gap is about ``2e-8``. A factor ``n / (n - 1)`` on either side
        would move the ratio by ``1e-3``, far outside the ``glm`` tolerance.
        """
        selected = int(facts["glmnet_best_k"]) - 1
        _, candidate = harness.final_candidate(
            chooser, lasso_path, internal, matrix(HERE, "g_final_cols.csv"), selected
        )
        curve = chooser.influence(candidate.targeted, candidate.submodel, chooser.all_rows)
        variance = influence_variance(curve)
        expected = float(facts["glmnet_var_psi"])
        assert variance == pytest.approx(expected, rel=harness.GLM_TOLERANCE)
        n = len(curve)
        assert variance * n / (n - 1) != pytest.approx(expected, rel=harness.GLM_TOLERANCE)


class TestTheWitnessesBind:
    """W1 to W3 on R's own path: each fails when its component is removed."""

    def test_w1_two_acceptances_and_a_jump(self) -> None:
        r = read(HERE / "glmnet_path.csv")
        accepted = r.loc[r["accepted"] == 1, "index"].to_numpy() - 1
        assert accepted.size >= 2
        assert np.any(np.diff(np.concatenate([[-1], accepted])) > 1)

    def test_w2_the_stop_binds_before_the_end(self, lasso_path: harness.Path) -> None:
        r = read(HERE / "glmnet_path.csv")
        accepted = r.loc[r["accepted"] == 1, "index"].to_numpy() - 1
        assert lasso_path.stopped
        assert accepted[-1] < len(r) - 1

    def test_w3_the_final_step_binds(
        self,
        chooser: Any,
        lasso_path: harness.Path,
        internal: np.ndarray,
        facts: dict[str, str],
    ) -> None:
        r = read(HERE / "glmnet_path.csv")
        selected = int(facts["glmnet_best_k"]) - 1
        assert abs(float(facts["glmnet_est"]) - float(r["est"].iloc[selected])) > FINAL_MOVE
        final, _ = harness.final_step(
            chooser, lasso_path, internal, matrix(HERE, "g_final_cols.csv"), selected
        )
        assert final not in (selected + 1, internal.shape[1] - 1)

    def test_the_decisions_have_margin(self, lasso_path: harness.Path) -> None:
        assert harness.smallest_margin(lasso_path) > harness.MARGIN
        assert harness.interior(lasso_path)
