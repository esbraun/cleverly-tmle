r"""C-TMLE0 against R ``ctmle`` 0.1.2 ``ctmletype = 3``, on R's own arrays.

``tests/canonical/ctmle_derivative_witness`` holds R's initial outcome regression, the two
mechanisms and the outcome for four blocks, with the coefficients, the targeted fit, the
curve ``Dstar``, the estimate and ``var.psi`` of a tight-tolerance pass of the type-3 code.
The fixture records that this pass and ``ctmleGlmnet(ctmletype = 3)`` differ only by the
``glm`` tolerance, and that ``ctmleGeneral`` and ``fluctuation = "linear"`` equal
``ctmleGlmnet`` bit for bit.

The seam runs on exactly those arrays, at ``alpha = 0.995`` and ``g_bounds = (0.025,
0.975)``.  Block A uses ``glmnet`` at the two declared penalties, once with a binary and
once with a continuous outcome.  Block B's partner crosses ``gbound`` on rows where the
mechanism does not, so it fails when the partner is left unbounded.  Block C's mechanism is
clipped at both bounds, so it fails when the ratio reads the unbounded mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleverly.estimators._ctmle_derivative import DerivativeUpdate, derivative_update
from cleverly.estimators.targeting import TargetingSpec
from cleverly.fluctuation.iterative import InitialFit
from cleverly.fluctuation.submodel import mean_submodel
from cleverly.inference.influence import counterfactual_means
from tests.canonical.ctmle_derivative_witness.regenerate import (
    ALPHA,
    BLOCKS,
    GBOUND,
    OUTPUTS,
    _sha256,
)

WITNESS = Path(__file__).parents[1] / "canonical" / "ctmle_derivative_witness"
BOUNDS = (GBOUND, 1.0 - GBOUND)
SPEC = TargetingSpec(alpha=ALPHA)


def _summary() -> pd.DataFrame:
    return pd.read_csv(WITNESS / "summary.csv", float_precision="round_trip").set_index("block")


def _rows(block: str) -> pd.DataFrame:
    return pd.read_csv(WITNESS / f"rows-{block}.csv", float_precision="round_trip")


def _mechanism(values: np.ndarray) -> np.ndarray:
    g1 = np.clip(values, *BOUNDS)
    return np.column_stack([1.0 - g1, g1])


def _seam(rows: pd.DataFrame) -> tuple[DerivativeUpdate, InitialFit, np.ndarray]:
    a = rows["A"].to_numpy(dtype=float)
    mechanism = _mechanism(rows["g1W"].to_numpy())
    partner = _mechanism(rows["g1WPrev"].to_numpy())
    initial = InitialFit(
        rows["QAW"].to_numpy(), {0.0: rows["Q0W"].to_numpy(), 1.0: rows["Q1W"].to_numpy()}
    )
    base = mean_submodel(a, mechanism, arms=(0.0, 1.0))
    n = len(rows)
    result = derivative_update(
        rows["Ystar"].to_numpy(),
        initial,
        base,
        mechanism,
        partner,
        a,
        np.ones(n, dtype=bool),
        np.ones(n),
        SPEC,
    )
    means = counterfactual_means(rows["Ystar"].to_numpy(), result.targeted, base, np.ones(n))
    curve = means[1.0].influence_curve - means[0.0].influence_curve
    return result, result.targeted, curve


def test_the_manifest_records_the_committed_bytes() -> None:
    manifest = json.loads((WITNESS / "manifest.json").read_text(encoding="utf-8"))
    for name in ("inputs.csv", "penalties.csv", "witness.R", *OUTPUTS):
        assert manifest["sha256"][name] == _sha256(WITNESS / name), name
    assert manifest["versions"]["ctmle_version"] == "0.1.2"
    assert manifest["versions"]["bound_source"] == "tmle"


@pytest.mark.parametrize("block", BLOCKS)
def test_the_seam_reproduces_r_type_3(block: str) -> None:
    rows = _rows(block)
    expected = _summary().loc[block]
    result, targeted, curve = _seam(rows)
    scale = float(expected["ab_high"] - expected["ab_low"])
    n = len(rows)

    assert result.names == ("h0", "h1", "dh0", "dh1")
    np.testing.assert_allclose(
        result.epsilon,
        expected[["H0W", "H1W", "ddH0W", "ddH1W"]].to_numpy(dtype=float),
        rtol=1e-6,
    )
    np.testing.assert_allclose(targeted.observed, rows["Qstar_AW"], atol=1e-6)
    np.testing.assert_allclose(targeted.arms[0.0], rows["Qstar_0W"], atol=1e-6)
    np.testing.assert_allclose(targeted.arms[1.0], rows["Qstar_1W"], atol=1e-6)
    np.testing.assert_allclose(curve, rows["Dstar"], atol=1e-6)
    estimate = (np.mean(targeted.arms[1.0]) - np.mean(targeted.arms[0.0])) * scale
    assert estimate == pytest.approx(float(expected["est"]), abs=1e-6)
    # R's var() divides by n - 1; the package's standard error divides by n.
    std_error = float(np.sqrt(np.mean(curve**2) / n)) * scale
    assert std_error**2 * n / (n - 1) == pytest.approx(float(expected["var_psi"]), abs=1e-6)


@pytest.mark.parametrize("block", BLOCKS)
def test_the_copy_of_the_type_3_code_is_ctmle_glmnet(block: str) -> None:
    # witness.R stops unless the default pass of its copy equals ctmleGlmnet, ctmleGeneral and
    # fluctuation = "linear" bit for bit. The recorded values show the first identity.
    expected = _summary().loc[block]
    assert float(expected["package_est"]) == float(expected["verbatim_est"])
    assert float(expected["package_var_psi"]) == float(expected["verbatim_var_psi"])
