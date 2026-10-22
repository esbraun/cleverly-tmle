"""The point laws of X82 part F (a) to (d) for the simulated common cause.

(a) A multi-arm treatment swaps the two arms of a declared pair in the upper latent tail.
(b) A clustered fit declares the level of the latent: one value per row, per cluster, or a
mix with a declared intra-cluster correlation.
(c) A categorical benchmark covariate calibrates as one block of its indicators, on an
unpenalized fit.
(d) Estimated observation weights stay fixed on their rows in every cell.

Each law has a witness that reads the perturbed data or the calibration directly, and a
mutation control that shows the witness can fail.
"""

from __future__ import annotations

import importlib
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from cleverly import ATE, CausalStudy, CounterfactualMean, CrossFitting, PointTreatment, TMLEMethod
from cleverly.datasets import make_multi_arm
from cleverly.exceptions import CapabilityError
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity.simulated_confounding import (
    SEPARATION_METHOD,
    _binary_calibration,
    _block_continuous_calibration,
    _continuous_calibration,
    _draw_latent,
    swapped_mechanism,
)
from tests.unit._confounding_support import forbid_draw_and_refit

#: The module, which the function ``cleverly.sensitivity.simulated_confounding`` shadows.
surface_module = importlib.import_module("cleverly.sensitivity.simulated_confounding")

_IN_SAMPLE = TMLEMethod(cross_fitting=CrossFitting(enabled=False))


def _estimate(study: CausalStudy, target: Any, *, binary: bool = False) -> Any:
    return study.identify(target).estimate(
        method=_IN_SAMPLE,
        outcome_learner=LogisticRegression(max_iter=1000) if binary else LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        random_state=5,
    )


def _capture_refits(monkeypatch: pytest.MonkeyPatch, result: Any) -> list[Any]:
    """Record the data of every refit the surface runs, and run it."""
    seen: list[Any] = []
    estimator = result.estimator
    original = estimator.refit

    def refit(data: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(data)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(estimator, "refit", refit)
    return seen


def _tail(latent: np.ndarray, strength: float) -> np.ndarray:
    """The published tail mask, restated without the production helper."""
    return latent >= NormalDist().inv_cdf(1.0 - strength)


# ------------------------------------------------------------------ (a) multi-arm


@pytest.fixture(scope="module")
def arms() -> Any:
    frame, _ = make_multi_arm(n=400, seed=13)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    return _estimate(study, ATE(reference="low"))


def _contrast(result: Any) -> str:
    return next(name for name in result.estimates if name.startswith("ate[high"))


GRID = ConfounderStrengthGrid(treatment=(0.0, 0.15, 0.3), outcome=(0.0, 0.2))


def test_the_pair_order_does_not_change_the_swap(arms: Any) -> None:
    """The swap reads the tail mask only, so naming ``(a, b)`` or ``(b, a)`` is one law."""
    alias = _contrast(arms)
    forward = simulated_confounding(
        arms, alias, grid=GRID, random_state=4, arm_pair=("high", "low")
    )
    backward = simulated_confounding(
        arms, alias, grid=GRID, random_state=4, arm_pair=("low", "high")
    )
    assert forward.arm_pair == ("high", "low")
    assert backward.arm_pair == ("low", "high")
    assert [cell.estimate for cell in forward.cells] == [cell.estimate for cell in backward.cells]
    moved = [cell for cell in forward.cells if cell.treatment_strength != 0.0]
    assert all(cell.estimate != forward.original_estimate for cell in moved)
    # The association reads the indicator of the first arm, so it changes sign.
    for left, right in zip(forward.cells, backward.cells, strict=True):
        assert left.induced_treatment_association == pytest.approx(
            -right.induced_treatment_association, abs=1e-12
        )


def test_a_one_way_swap_breaks_the_pair_order_witness(
    arms: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: moving arm ``a`` to ``b`` alone makes the order matter."""

    def one_way(values: np.ndarray, mask: np.ndarray, pair: tuple[float, float]) -> np.ndarray:
        out = values.copy()
        out[mask & (values == pair[0])] = pair[1]
        return out

    monkeypatch.setattr(surface_module, "_swap_arms", one_way)
    alias = _contrast(arms)
    forward = simulated_confounding(
        arms, alias, grid=GRID, random_state=4, arm_pair=("high", "low")
    )
    backward = simulated_confounding(
        arms, alias, grid=GRID, random_state=4, arm_pair=("low", "high")
    )
    gaps = [
        abs(left.estimate - right.estimate)
        for left, right in zip(forward.cells, backward.cells, strict=True)
    ]
    assert max(gaps) > 1e-3


def test_the_swap_keeps_the_third_arm_and_exchanges_the_pair(
    arms: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture_refits(monkeypatch, arms)
    surface = simulated_confounding(
        arms,
        _contrast(arms),
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0,)),
        random_state=4,
    )
    data = arms.data
    latent = np.random.default_rng(surface.latent_seed).normal(size=data.n)
    high = float(data.treatment_levels.index("high"))
    low = float(data.treatment_levels.index("low"))
    mid = float(data.treatment_levels.index("medium"))
    (replacement,) = seen
    original = data.treatment
    swapped = replacement.treatment
    tail = _tail(latent, 0.3)
    third = original == mid
    assert np.array_equal(swapped[third], original[third])
    expected = original.copy()
    expected[tail & (original == high)] = low
    expected[tail & (original == low)] = high
    assert np.array_equal(swapped, expected)
    assert np.any(swapped != original)


def test_a_cycle_through_every_arm_moves_the_third_arm(
    arms: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation control: a ``K``-cycle in the tail fails the third-arm witness."""

    def cycle(values: np.ndarray, mask: np.ndarray, pair: tuple[float, float]) -> np.ndarray:
        out = values.copy()
        out[mask] = (values[mask] + 1.0) % 3.0
        return out

    monkeypatch.setattr(surface_module, "_swap_arms", cycle)
    seen = _capture_refits(monkeypatch, arms)
    simulated_confounding(
        arms,
        _contrast(arms),
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0,)),
        random_state=4,
    )
    mid = float(arms.data.treatment_levels.index("medium"))
    third = arms.data.treatment == mid
    assert np.any(seen[0].treatment[third] != arms.data.treatment[third])


def test_an_arm_mean_needs_its_pair_and_a_pair_must_match_the_contrast(arms: Any) -> None:
    alias = _contrast(arms)
    with pytest.raises(ValueError, match="disagrees with the contrast"):
        simulated_confounding(arms, alias, grid=GRID, arm_pair=("medium", "low"))
    with pytest.raises(ValueError, match="names two different arms"):
        simulated_confounding(arms, alias, grid=GRID, arm_pair=("high", "high"))
    with pytest.raises(ValueError, match="names two different arms"):
        simulated_confounding(arms, alias, grid=GRID, arm_pair=("high", "absent"))


def test_an_arm_mean_of_a_multi_arm_fit_swaps_its_declared_pair() -> None:
    frame, _ = make_multi_arm(n=400, seed=13)
    study = CausalStudy(
        frame,
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3")),
    )
    result = _estimate(study, CounterfactualMean(treatment="high"))
    (alias,) = result.estimates
    with pytest.raises(ValueError, match="pass arm_pair="):
        simulated_confounding(result, alias, grid=GRID)
    surface = simulated_confounding(
        result, alias, grid=GRID, random_state=4, arm_pair=("high", "medium")
    )
    assert surface.complete
    assert surface.arm_pair == ("high", "medium")


def test_the_swapped_mechanism_mixes_the_pair_and_keeps_the_rest() -> None:
    """The known mechanism of the swap, ``(1 - s) g_a + s g_b``, on a fixed matrix."""
    declared = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    out = swapped_mechanism(declared, 0.25, (0, 2))
    assert out[:, 0] == pytest.approx(0.75 * declared[:, 0] + 0.25 * declared[:, 2])
    assert out[:, 2] == pytest.approx(0.75 * declared[:, 2] + 0.25 * declared[:, 0])
    assert np.array_equal(out[:, 1], declared[:, 1])
    assert out.sum(axis=1) == pytest.approx(1.0)


def test_a_known_multi_arm_mechanism_is_carried_into_each_swapped_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(3)
    n = 360
    w = rng.normal(size=n)
    probabilities = np.column_stack((np.full(n, 0.3), np.full(n, 0.3), np.full(n, 0.4)))
    codes = np.array([rng.choice(3, p=row) for row in probabilities])
    labels = np.array(["a", "b", "c"])[codes]
    frame = pd.DataFrame(
        {
            "W": w,
            "A": labels,
            "Y": w + codes + rng.normal(size=n),
            "pa": probabilities[:, 0],
            "pb": probabilities[:, 1],
            "pc": probabilities[:, 2],
        }
    )
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            treatment_probabilities={"a": "pa", "b": "pb", "c": "pc"},
        ),
    )
    result = _estimate(study, ATE(reference="a"))
    alias = next(name for name in result.estimates if name.startswith("ate[c"))
    seen = _capture_refits(monkeypatch, result)
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,)),
        random_state=4,
    )
    assert surface.known_mechanism_carried
    (replacement,) = seen
    carried = np.asarray(replacement.known_treatment.values)
    first, second = (result.data.treatment_levels.index(arm) for arm in ("c", "a"))
    expected = probabilities.copy()
    expected[:, first] = 0.8 * probabilities[:, first] + 0.2 * probabilities[:, second]
    expected[:, second] = 0.8 * probabilities[:, second] + 0.2 * probabilities[:, first]
    assert carried == pytest.approx(expected, abs=1e-12)


# ------------------------------------------------------------------ (b) latent level


def _clustered_frame(*, constant_treatment: bool, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(40), 8)
    n = clusters.size
    w = rng.normal(size=n)
    if constant_treatment:
        a = rng.binomial(1, 0.5, size=40)[clusters]
    else:
        a = rng.binomial(1, 1 / (1 + np.exp(-w)))
    y = w + a + rng.normal(size=40)[clusters] + rng.normal(size=n)
    return pd.DataFrame({"W": w, "A": a, "Y": y, "cl": clusters})


def _clustered_fit(*, constant_treatment: bool) -> Any:
    study = CausalStudy(
        _clustered_frame(constant_treatment=constant_treatment),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",), cluster="cl"),
    )
    return _estimate(study, ATE())


def test_a_cluster_latent_is_constant_within_each_cluster() -> None:
    cluster = np.repeat(np.arange(5), 4)
    latent = _draw_latent(cluster.size, 11, cluster, "cluster", None)
    for code in range(5):
        assert np.unique(latent[cluster == code]).size == 1
    assert np.unique(latent).size == 5


def test_a_mixed_latent_is_the_declared_composition() -> None:
    cluster = np.repeat(np.arange(5), 4)
    latent = _draw_latent(cluster.size, 11, cluster, "mixed", 0.36)
    generator = np.random.default_rng(11)
    shared = generator.normal(size=5)[cluster]
    expected = 0.6 * shared + 0.8 * generator.normal(size=cluster.size)
    assert latent == pytest.approx(expected, abs=1e-15)


def test_a_row_latent_keeps_the_unclustered_stream() -> None:
    assert np.array_equal(
        _draw_latent(30, 11, None, "row", None), np.random.default_rng(11).normal(size=30)
    )


def test_a_cluster_latent_read_by_row_fails_the_constancy_witness() -> None:
    """Mutation control: a draw indexed by row is not constant within a cluster."""
    cluster = np.repeat(np.arange(5), 4)
    latent = np.random.default_rng(11).normal(size=cluster.size)
    assert any(np.unique(latent[cluster == code]).size > 1 for code in range(5))


def test_a_clustered_surface_reads_the_latent_at_its_declared_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _clustered_fit(constant_treatment=False)
    seen = _capture_refits(monkeypatch, result)
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0, 0.5)),
        random_state=4,
        latent_level="cluster",
    )
    (replacement,) = seen
    implied = (result.data.outcome - replacement.outcome) / 0.5
    for code in np.unique(result.data.cluster):
        rows = result.data.cluster == code
        assert np.ptp(implied[rows]) == pytest.approx(0.0, abs=1e-12)
    assert surface.latent_level == "cluster"
    assert surface.n_clusters == 40
    assert "latent level: cluster over 40 clusters" in surface.summary()


def test_a_clustered_fit_must_declare_its_latent_level() -> None:
    result = _clustered_fit(constant_treatment=False)
    grid = ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,))
    with pytest.raises(ValueError, match="declare latent_level="):
        simulated_confounding(result, grid=grid)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        simulated_confounding(result, grid=grid, latent_level="mixed")
    with pytest.raises(ValueError, match="sets the correlation of latent_level='mixed'"):
        simulated_confounding(result, grid=grid, latent_level="row", latent_rho=0.3)
    with pytest.raises(ValueError, match="must be one of"):
        simulated_confounding(result, grid=grid, latent_level="unit")


def test_an_unclustered_fit_takes_the_row_level_only(arms: Any) -> None:
    alias = _contrast(arms)
    with pytest.raises(ValueError, match="needs a clustered fit"):
        simulated_confounding(arms, alias, grid=GRID, latent_level="cluster")


@pytest.mark.parametrize(("level", "rho"), [("row", None), ("mixed", 0.5)])
def test_a_row_varying_latent_cannot_move_a_cluster_level_treatment(
    monkeypatch: pytest.MonkeyPatch, level: str, rho: float | None
) -> None:
    """The refusal fires before any draw, and only when a cell moves the treatment."""
    result = _clustered_fit(constant_treatment=True)
    forbid_draw_and_refit(monkeypatch, result.estimator)
    with pytest.raises(CapabilityError, match="constant within every cluster"):
        simulated_confounding(
            result,
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.1), outcome=(0.0,)),
            latent_level=level,  # type: ignore[arg-type]
            latent_rho=rho,
        )


@pytest.mark.parametrize(("level", "rho"), [("row", None), ("mixed", 0.5)])
def test_an_outcome_only_row_latent_runs_on_a_cluster_level_treatment(
    monkeypatch: pytest.MonkeyPatch, level: str, rho: float | None
) -> None:
    """An outcome-only grid changes no treatment, so a row-varying latent is well posed."""
    result = _clustered_fit(constant_treatment=True)
    seen = _capture_refits(monkeypatch, result)
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0, 0.3)),
        random_state=4,
        latent_level=level,  # type: ignore[arg-type]
        latent_rho=rho,
    )
    assert surface.complete
    assert seen
    for replacement in seen:
        assert np.array_equal(replacement.treatment, result.data.treatment)


def test_a_cluster_latent_flips_whole_clusters(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _clustered_fit(constant_treatment=True)
    seen = _capture_refits(monkeypatch, result)
    simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.3), outcome=(0.0,)),
        random_state=4,
        latent_level="cluster",
    )
    (replacement,) = seen
    flipped = replacement.treatment != result.data.treatment
    assert np.any(flipped)
    for code in np.unique(result.data.cluster):
        rows = result.data.cluster == code
        assert np.unique(flipped[rows]).size == 1


# ------------------------------------------------------------------ (c) categorical block


def _categorical_frame(levels: tuple[str, str, str], seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 500
    w = rng.normal(size=n)
    g = rng.integers(0, 3, size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-(0.4 * w + 0.9 * (g == 1) - 0.7 * (g == 2)))))
    y = w + a + 0.8 * (g == 2) - 0.5 * (g == 1) + rng.normal(size=n)
    return pd.DataFrame({"W": w, "G": np.array(levels)[g], "A": a, "Y": y})


def _categorical_fit(levels: tuple[str, str, str]) -> Any:
    study = CausalStudy(
        _categorical_frame(levels),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=("W", "G")),
    )
    return _estimate(study, ATE())


def _block_strengths(result: Any) -> dict[str, float]:
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=("G",),
        random_state=4,
    )
    return {row.role: row.strength for row in surface.calibrations}


def test_a_block_calibration_does_not_depend_on_the_reference_level() -> None:
    """``"a"`` sorts first and is dropped; renamed ``"z"`` it sorts last and ``"b"`` drops."""
    first = _categorical_fit(("a", "b", "c"))
    second = _categorical_fit(("z", "b", "c"))
    assert first.data.encodings[0].dropped_level != second.data.encodings[0].dropped_level
    left, right = _block_strengths(first), _block_strengths(second)
    assert left["treatment"] > 0.0
    assert left["outcome"] > 0.0
    assert left["treatment"] == pytest.approx(right["treatment"], rel=1e-8)
    assert left["outcome"] == pytest.approx(right["outcome"], rel=1e-8)


def _penalized_block_fraction(result: Any) -> float:
    data = result.data
    design = StandardScaler().fit_transform(data.covariates)
    block = [data.covariate_names.index(name) for name in data.encodings[0].generated]
    model = LogisticRegression(max_iter=1000).fit(design, data.treatment)
    removed = design.copy()
    removed[:, block] = 0.0
    return float(np.mean(model.predict(removed) != model.predict(design)))


def test_a_penalized_block_fit_depends_on_the_reference_level() -> None:
    """Mutation control: the default L2 fit gives a different block strength per reference."""
    first = _categorical_fit(("a", "b", "c"))
    second = _categorical_fit(("z", "b", "c"))
    assert _penalized_block_fraction(first) != pytest.approx(
        _penalized_block_fraction(second), rel=1e-6
    )


def test_a_one_column_block_is_the_absolute_numeric_strength() -> None:
    rng = np.random.default_rng(2)
    column = rng.binomial(1, 0.4, size=200).astype(float)
    target = 0.7 * column + rng.normal(size=200)
    weights = np.ones(200)
    block = _block_continuous_calibration(column[:, None], target, weights)
    numeric = _continuous_calibration(column, target, weights)
    assert block == pytest.approx(abs(numeric), rel=1e-10)
    assert block > 0.0


def test_a_separated_calibration_reports_nan_and_says_why() -> None:
    rng = np.random.default_rng(8)
    design = rng.normal(size=(120, 2))
    target = (design[:, 0] > 0.0).astype(float)
    strength, method = _binary_calibration(design, target, [0], np.ones(120))
    assert np.isnan(strength)
    assert method == SEPARATION_METHOD
    overlap = (design[:, 0] + rng.normal(size=120) > 0.0).astype(float)
    strength, method = _binary_calibration(design, overlap, [0], np.ones(120))
    assert np.isfinite(strength)
    assert method == "unpenalized logistic class-prediction change fraction"


def test_the_summary_counts_the_separated_calibrations() -> None:
    rng = np.random.default_rng(8)
    n = 300
    w = rng.normal(size=n)
    v = rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n)
    frame = pd.DataFrame({"W": w, "V": v + 5.0 * a, "A": a, "Y": w + a + rng.normal(size=n)})
    frame.loc[frame["A"] == 1, "V"] = np.abs(frame.loc[frame["A"] == 1, "V"]) + 10.0
    frame.loc[frame["A"] == 0, "V"] = -np.abs(frame.loc[frame["A"] == 0, "V"])
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=("W", "V"))
    )
    result = study.identify(ATE()).estimate(
        method=_IN_SAMPLE,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        random_state=5,
        g_bounds=(0.01, 0.99),
    )
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        benchmark_covariates=("V",),
        random_state=4,
    )
    rows = {row.role: row for row in surface.calibrations}
    assert np.isnan(rows["treatment"].strength)
    assert "calibration: 1 of 2 strengths are nan" in surface.summary()


def test_an_indicator_is_refused_with_its_logical_covariate() -> None:
    result = _categorical_fit(("a", "b", "c"))
    indicator = result.data.encodings[0].generated[0]
    with pytest.raises(CapabilityError, match=f"{indicator!r} is one indicator of 'G'"):
        simulated_confounding(
            result,
            grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
            benchmark_covariates=(indicator,),
        )


# ------------------------------------------------------------------ (d) estimated weights


def _weighted_frame(seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    w = rng.binomial(1, 0.45, size=n).astype(float)
    a = rng.binomial(1, 0.35 + 0.3 * w)
    y = w + a * (1.0 + 2.0 * w) + rng.normal(size=n)
    return pd.DataFrame({"W": w, "A": a, "Y": y, "wt": np.where(w > 0, 3.0, 0.5)})


def _weighted_fit(estimated: bool) -> Any:
    study = CausalStudy(
        _weighted_frame(),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W",),
            weights="wt",
            weights_estimated=estimated,
        ),
    )
    return study.identify(ATE()).estimate(
        method=_IN_SAMPLE,
        outcome_learner=make_pipeline(
            PolynomialFeatures(2, interaction_only=True), LinearRegression()
        ),
        treatment_learner=LogisticRegression(C=np.inf, max_iter=1000),
        random_state=5,
    )


def _plug_in(data: Any, weighted: bool) -> float:
    """The saturated plug-in ATE of ``data``, with or without its row weights."""
    w = data.covariates[:, data.covariate_names.index("W")]
    a, y = data.treatment, data.outcome
    cell = {
        (arm, level): float(np.mean(y[(a == arm) & (w == level)]))
        for arm in (0.0, 1.0)
        for level in (0.0, 1.0)
    }
    contrast = np.array([cell[1.0, level] - cell[0.0, level] for level in w])
    weights = data.weights if weighted else np.ones(data.n)
    return float(np.average(contrast, weights=weights))


def test_estimated_weights_stay_on_their_rows_and_the_cell_is_the_weighted_plug_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _weighted_fit(estimated=True)
    assert result.data.declares_estimated_weights
    # Zero strength: a refit of the unperturbed data reproduces the weighted estimate.
    again = result.estimator.refit(result.data, random_state=result.estimator.random_state)
    assert again["ate"].psi == pytest.approx(result["ate"].psi, rel=1e-12)
    assert result["ate"].psi == pytest.approx(_plug_in(result.data, weighted=True), rel=1e-8)
    seen = _capture_refits(monkeypatch, result)
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.4)),
        random_state=4,
    )
    assert surface.complete
    assert surface.weights_estimated
    assert any("declared estimated" in line for line in surface.population_lines())
    for replacement, cell in zip(seen, surface.cells[1:], strict=True):
        assert np.array_equal(replacement.weights, result.data.weights)
        assert cell.estimate == pytest.approx(_plug_in(replacement, weighted=True), rel=1e-8)
        # Mutation control: the same calculation without the weights is another number,
        # so a refit that dropped them would fail the equality above.
        assert abs(cell.estimate - _plug_in(replacement, weighted=False)) > 1e-3


def test_fixed_weights_report_no_estimated_weight_line() -> None:
    result = _weighted_fit(estimated=False)
    surface = simulated_confounding(
        result,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,)),
        random_state=4,
    )
    assert not surface.weights_estimated
    assert not any("declared estimated" in line for line in surface.population_lines())
