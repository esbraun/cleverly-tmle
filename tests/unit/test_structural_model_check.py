"""The size of ``StructuralNestedMeanResult.check_model`` on a correct structural model.

The structural and the nonparametric contrasts nearly cancel, so their difference has a far
smaller variance than either estimator.  On a cross-fitted fit, each estimator also carries a
second-order remainder that depends on the fold split.  The curves do not carry it.  It is
negligible against either estimator but not against the difference.  At 2 folds a check that
read the curves of one split rejected a correct model at 14 % to 18 % at the 5 % level.

The check now averages the difference over several splits and adds the between-split
covariance.  The size tests run the shipped check at 2 folds on a correct model.  The mean Wald
statistic must stay under the one-sided 0.999 upper limit of a mean of chi-squared draws.  The
same limit applies to the mean of each gap against the mean reported covariance, which removes
the co-movement of a replicate's gap and its own covariance.  The one-split statistic of the
same fits must exceed that limit, so the instrument detects the defect that it guards against.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.learners import GLM
from cleverly.longitudinal import StructuralNestedMean
from cleverly.longitudinal import structural as module

from ._source_mutation import mutate

REGIMENS = {"r00": [0, 0], "r11": [1, 1], "r10": [1, 0], "r01": [0, 1]}
# The blip of node 1 varies with W and that of node 2 with W and A1: the model is correct.
FEATURES = {1: ["W"], 2: ["W", "A1"]}
Q_BOUNDS = (-8.0, 12.0)
DF = 3
N = 500
REPLICATES = 60
# Three splits estimate the covariance with fewer degrees of freedom than the default five, so
# the size limit is harder to meet, and the fixture costs three fits of each estimator.
SPLITS = 3

pytestmark = pytest.mark.xdist_group("structural_model_check")


def _expit(x: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-x))


def _draw(n: int, seed: int) -> pd.DataFrame:
    """Binary W, A1, L2 and A2 and a Gaussian outcome whose blips are linear in the features."""
    rng = np.random.default_rng(seed)
    w = rng.binomial(1, 0.5, n)
    a1 = rng.binomial(1, _expit(-0.3 + 0.8 * w))
    l2 = rng.binomial(1, _expit(-0.2 + 0.7 * w + 0.9 * a1))
    a2 = rng.binomial(1, _expit(-0.4 + 0.8 * l2 + 0.5 * a1 - 0.6 * w))
    y = (
        a1 * (1.0 + 0.5 * w)
        + a2 * (0.5 + 0.4 * w + 0.6 * a1)
        + 0.7 * l2
        + 0.3 * w
        + rng.normal(size=n)
    )
    return pd.DataFrame({"W": w * 1.0, "A1": a1, "L2": l2 * 1.0, "A2": a2, "Y": y})


def _fit(frame: pd.DataFrame, n_folds: int, seed: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return StructuralNestedMean(
            REGIMENS,
            blip_features=FEATURES,
            outcome_learner=GLM(family="gaussian", interactions=2),
            treatment_learner=GLM(interactions=2),
            n_folds=n_folds,
            simultaneous=False,
            random_state=seed,
        ).fit(
            frame,
            outcome="Y",
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            family="gaussian",
        )


def _limit(replicates: int) -> float:
    """The one-sided 0.999 upper limit of the mean of ``replicates`` chi-squared draws."""
    return DF + float(stats.norm.ppf(0.999)) * np.sqrt(2.0 * DF / replicates)


def _wald(gap: Any, covariance: Any) -> float:
    return float(gap @ np.linalg.solve(covariance, gap))


def _run(monkeypatch: pytest.MonkeyPatch, n_folds: int) -> list[dict[str, Any]]:
    """Run the check on each replicate and record what ``_split_covariance`` received."""
    shipped = module._split_covariance
    calls: list[dict[str, Any]] = []

    def record(differences: Any, curves: Sequence[Any], cluster: Any) -> Any:
        covariance = shipped(differences, curves, cluster)
        calls.append({"differences": differences, "curves": list(curves), "cluster": cluster})
        return covariance

    monkeypatch.setattr(module, "_split_covariance", record)
    out = []
    for seed in range(REPLICATES):
        check = _fit(_draw(N, seed), n_folds, seed).check_model(q_bounds=Q_BOUNDS, n_splits=SPLITS)
        assert check.tested and check.df == DF
        out.append({"check": check, **calls[-1]})
    monkeypatch.setattr(module, "_split_covariance", shipped)
    return out


@pytest.fixture(scope="module")
def two_folds() -> list[dict[str, Any]]:
    monkeypatch = pytest.MonkeyPatch()
    try:
        return _run(monkeypatch, n_folds=2)
    finally:
        monkeypatch.undo()


def _calibration(gaps: Sequence[Any], covariances: Sequence[Any]) -> float:
    """The mean of each gap against the mean reported covariance: ``df`` when calibrated."""
    average = np.mean(covariances, axis=0)
    return float(np.mean([_wald(gap, average) for gap in gaps]))


def test_the_check_holds_its_size_at_two_folds(two_folds: list[dict[str, Any]]) -> None:
    checks = [row["check"] for row in two_folds]
    assert np.mean([check.statistic for check in checks]) <= _limit(REPLICATES)
    calibration = _calibration(
        [check.difference for check in checks], [check.covariance for check in checks]
    )
    assert calibration <= _limit(REPLICATES)
    for row in two_folds:
        check = row["check"]
        assert check.n_splits == SPLITS
        assert np.allclose(check.difference, row["differences"].mean(axis=0), atol=1e-12)
        assert check.statistic == pytest.approx(_wald(check.difference, check.covariance))


def test_the_one_split_check_fails_the_size_limit(two_folds: list[dict[str, Any]]) -> None:
    """The check before the fix: the fit's own split and the covariance of its own curve."""
    gaps = [row["differences"][0] for row in two_folds]
    covariances = [
        module._split_covariance(row["differences"][:1], row["curves"][:1], row["cluster"])
        for row in two_folds
    ]
    assert _calibration(gaps, covariances) > _limit(REPLICATES)


def _two_splits() -> tuple[Any, list[Any], Any]:
    """Two splits written out: the cross product of the two curves plus half the spread.

    The curves share a component, so the cross term is nonzero.  The two differences
    differ, so the between-split term is nonzero.
    """
    rng = np.random.default_rng(0)
    common = rng.normal(size=(40, 2))
    curves = [common + 0.5 * rng.normal(size=(40, 2)) for _ in range(2)]
    differences = np.array([[0.3, -0.1], [-0.2, 0.4]])
    centered = [curve - curve.mean(axis=0) for curve in curves]
    cross = (centered[0].T @ centered[1] + centered[1].T @ centered[0]) / 2.0 / 39 / 40
    spread = np.outer(differences[0] - differences[1], differences[0] - differences[1]) / 2.0
    return differences, curves, cross + spread / 2.0


def test_the_split_covariance_is_the_cross_split_curve_covariance_plus_the_split_term() -> None:
    differences, curves, expected = _two_splits()
    assert np.allclose(module._split_covariance(differences, curves, None), expected)


@pytest.mark.parametrize(
    "edit",
    [
        ("covariance = first_order + between / count", "covariance = first_order"),
        ("first_order = (total - own) / (count * (count - 1))", "first_order = own / count"),
    ],
    ids=["no-between-split-term", "own-curve-covariance"],
)
def test_a_broken_split_covariance_fails_the_two_split_identity(
    monkeypatch: pytest.MonkeyPatch, edit: tuple[str, str]
) -> None:
    mutate(monkeypatch, module, module, "_split_covariance", [edit])
    differences, curves, expected = _two_splits()
    assert not np.allclose(module._split_covariance(differences, curves, None), expected)


def test_an_in_sample_check_reads_one_split_and_n_splits_is_validated() -> None:
    result = _fit(_draw(N, 0), n_folds=1, seed=0)
    check = result.check_model()
    assert check.n_splits == 1
    with pytest.raises(ValueError, match="n_splits"):
        result.check_model(n_splits=1)


def test_a_seeded_check_is_reproducible() -> None:
    result = _fit(_draw(N, 1), n_folds=2, seed=1)
    first = result.check_model(q_bounds=Q_BOUNDS, n_splits=3)
    second = result.check_model(q_bounds=Q_BOUNDS, n_splits=3)
    assert first.n_splits == 3
    assert np.array_equal(first.covariance, second.covariance)
    assert np.array_equal(first.ltmle.folds.assignment, result.folds.assignment)
