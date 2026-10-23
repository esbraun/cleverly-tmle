"""W2: ``StructuralNestedMean`` against the committed EconML ``DynamicDML`` fixture.

``tests/canonical/econml_dynamic_dml/parity_fixture.json`` holds the inputs of seven cases, the
cross-fitting folds, and EconML 0.17.0's coefficients and covariance.  This module refits each
case with ``StructuralNestedMean`` on the same folds and the same nuisance models.  It never
imports EconML.

The fixture supports four claims.

* The coefficients agree for every node type, with and without baseline features.
* The covariance agrees where every node is scalar, after the factor ``n / (n - 1)``.  EconML
  divides the score variance by ``n``, and ``cleverly`` uses the sample variance of the curve.
* On a categorical node, EconML fills the off-diagonal Jacobian blocks in the transposed order.
  ``StructuralNestedMean`` matches the correct sandwich, and it matches EconML when the same
  transposition is applied to its source.
* The contrast error of a categorical case differs from EconML's by about ten percent, and a
  bootstrap of the final stage with the residuals held fixed sides with ``cleverly``.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from cleverly.learners import GLM
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import StructuralNestedMean
from cleverly.longitudinal import structural as module
from tests.canonical.econml_dynamic_dml.generate_parity_fixture import LinearProbability

from ._source_mutation import mutate

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "tests" / "canonical" / "econml_dynamic_dml"
FIXTURE = DIRECTORY / "parity_fixture.json"
WHEEL_SHA256 = "c66ef0164fa807025a445e60ab9144cd0e1d0112bf2b232da9a5fd35d1b75045"

POINT = 1e-8
COVARIANCE = 1e-10

#: The cases whose nodes are all scalar: a binary indicator or a dose.
SCALAR_CASES = (
    "binary_scalar",
    "binary_features",
    "binary_logistic",
    "continuous",
    "continuous_features",
)
CATEGORICAL_CASES = ("categorical", "categorical_features")
CONTRAST_CASES = ("binary_scalar", "binary_logistic", "continuous")


@cache
def _fixture() -> dict[str, Any]:
    parsed: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return parsed


def _fit(name: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fit one case on the fixture's folds with the nuisance models EconML used."""
    case = _fixture()["cases"][name]
    spec = case["spec"]
    folds = Folds(np.asarray(case["folds"], dtype=int), 2)
    monkeypatch.setattr(StructuralNestedMean, "_folds", lambda self, data, seed=None: folds)
    logistic = spec["nuisance"] == "logistic"
    regimens = None
    if spec["contrast"] is not None:
        regimens = {"lo": [0.0, 0.0], "hi": list(spec["contrast"])}
    features = {1: ["W0"], 2: ["W0"]} if spec["features"] else None
    continuous = ["A1", "A2"] if spec["kind"] == "continuous" else []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StructuralNestedMean(
            regimens,
            blip_features=features,
            outcome_learner=LinearRegression(),
            treatment_learner=GLM(tol=1e-13) if logistic else LinearProbability(),
            n_folds=2,
            simultaneous=False,
        ).fit(
            pd.DataFrame(case["columns"]),
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W0"],
            time_varying=[["L1"], ["L2"]],
            family="gaussian",
            continuous_treatment=continuous,
        )


def _psi(result: Any) -> np.ndarray:
    """The coefficients in EconML's order: node, coding component, feature."""
    return np.concatenate([np.asarray(result.blip(node)).T.ravel() for node in (1, 2)])


def _covariance(result: Any, n: int) -> np.ndarray:
    """The coefficient covariance on EconML's variance scale, ``1 / n`` times the curve variance."""
    names = [name for name in result if name.startswith("blip[")]
    return np.asarray(result.covariance(names)) * (n - 1) / n


def _econml(name: str) -> dict[str, Any]:
    record: dict[str, Any] = _fixture()["cases"][name]["econml"]
    return record


def _n(name: str) -> int:
    return len(_fixture()["cases"][name]["folds"])


def test_the_fixture_records_the_files_that_made_it() -> None:
    fixture = _fixture()
    assert fixture["econml_version"] == "0.17.0"
    assert fixture["wheel_sha256"] == WHEEL_SHA256
    environment = fixture["environment"]
    assert set(environment) == {"python", "route"}
    assert environment["route"] != "unrecorded", "the fixture names the route that wrote it"
    for key, file in (
        ("generator_sha256", "generate_parity_fixture.py"),
        ("dockerfile_sha256", "Dockerfile"),
        ("requirements_sha256", "requirements.txt"),
    ):
        digest = hashlib.sha256((DIRECTORY / file).read_bytes()).hexdigest()
        assert fixture[key] == digest, f"{file} changed since the fixture was written"
    requirements = (DIRECTORY / "requirements.txt").read_text(encoding="utf-8")
    pinned = requirements.split("econml==0.17.0 \\\n", 1)[1].split("\n", 1)[0].strip()
    assert pinned == f"--hash=sha256:{WHEEL_SHA256}"
    assert "econml" not in (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("name", [*SCALAR_CASES, *CATEGORICAL_CASES])
def test_w2_coefficients_equal_econml_for_every_node_type(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = np.asarray(_econml(name)["coef"], dtype=float).reshape(-1)
    assert np.max(np.abs(_psi(_fit(name, monkeypatch)) - expected)) < POINT
    assert np.max(np.abs(expected)) > 0.1, "a witness that vanishes would pass for any estimator"


@pytest.mark.parametrize("name", SCALAR_CASES)
def test_w2_covariance_equals_econml_when_every_node_is_scalar(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _fit(name, monkeypatch)
    expected = np.asarray(_econml(name)["cov"], dtype=float)
    assert np.max(np.abs(_covariance(result, _n(name)) - expected)) < COVARIANCE
    assert np.max(np.abs(expected)) > 1e-3


@pytest.mark.parametrize("name", CONTRAST_CASES)
def test_w2_contrast_equals_econml_ate_inference_with_constant_blips(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``phi = 1`` the contrast is a fixed sum of coefficients, as EconML treats it."""
    result = _fit(name, monkeypatch)
    ate = _econml(name)["ate"]
    contrast = result["ate_regimen[hi vs lo]"]
    scale = np.sqrt((_n(name) - 1) / _n(name))
    assert abs(contrast.psi - ate["point"]) < POINT
    assert abs(contrast.std_error * scale - ate["stderr"]) < 1e-9


@pytest.mark.parametrize("name", CATEGORICAL_CASES)
def test_w2_a_categorical_node_differs_from_econml_by_the_transposed_blocks(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EconML's covariance differs on a vector coding, and cleverly keeps the correct blocks."""
    result = _fit(name, monkeypatch)
    ours = _covariance(result, _n(name))
    econml = np.asarray(_econml(name)["cov"], dtype=float)
    assert np.max(np.abs(ours - econml)) > 1e-3 * np.max(np.abs(econml))
    sandwiches = _econml(name)["sandwiches"]
    if sandwiches is None:
        return
    assert np.max(np.abs(ours - np.asarray(sandwiches["correct"]))) < COVARIANCE
    transposed = np.asarray(sandwiches["block_transposed"])
    assert np.max(np.abs(econml - transposed)) < COVARIANCE, "EconML fills the transposed blocks"
    assert np.max(np.abs(ours - transposed)) > 1e-3 * np.max(np.abs(econml))


def test_w2_the_categorical_contrast_error_follows_the_correct_sandwich(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "categorical"
    result = _fit(name, monkeypatch)
    record = _econml(name)
    scale = np.sqrt((_n(name) - 1) / _n(name))
    ours = result["ate_regimen[hi vs lo]"].std_error * scale
    assert abs(result["ate_regimen[hi vs lo]"].psi - record["ate"]["point"]) < POINT
    boot = record["bootstrap_contrast_se"]
    assert abs(ours / boot - 1.0) < 0.02, "the fixed-residual bootstrap agrees with cleverly"
    assert abs(record["ate"]["stderr"] / boot - 1.0) > 0.05, "EconML's error is outside the band"


def test_w2_applying_the_transposition_to_cleverly_reproduces_the_econml_covariance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mutation M4 of the source gives EconML's covariance on the categorical case."""
    name = "categorical"
    clean = _covariance(_fit(name, monkeypatch), _n(name))
    mutate(
        monkeypatch,
        module,
        module,
        "_jacobian",
        [
            (
                "block = (own * w[:, None]).T @ z_res[(t, j)] / n",
                "block = ((own * w[:, None]).T @ z_res[(t, j)] / n)"
                ".T if j > t else (own * w[:, None]).T @ z_res[(t, j)] / n",
            )
        ],
    )
    broken = _covariance(_fit(name, monkeypatch), _n(name))
    econml = np.asarray(_econml(name)["cov"], dtype=float)
    assert np.max(np.abs(broken - econml)) < COVARIANCE
    assert np.max(np.abs(clean - econml)) > 1e-3 * np.max(np.abs(econml))
