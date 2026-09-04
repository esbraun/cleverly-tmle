"""Population contracts for simulated common causes, with independent refit witnesses."""

from __future__ import annotations

import importlib
from dataclasses import replace
from functools import cache
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATC,
    ATE,
    ATT,
    AssessmentStatus,
    CausalStudy,
    CounterfactualMean,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    OddsRatio,
    PointTreatment,
    RiskRatio,
)
from cleverly.assessment import INTERPRETERS
from cleverly.estimators import CTMLE, DRTMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Shift
from cleverly.sensitivity import (
    ConfounderStrengthGrid,
    SimulatedConfoundingResult,
    simulated_confounding,
)
from cleverly.sensitivity._simulated_confounding_request import (
    _eligible_binary_parameter_names,
    _validate_request,
    _zero_delta_policy_means,
)
from cleverly.targets.base import parameter_name
from tests.unit._confounding_support import (
    _STRATEGY_OVERRIDES,
    _strategy_method,
    alias_for,
    confounding_estimate,
    confounding_study,
    with_estimator,
    with_functional,
    with_key,
)
from tests.unit._confounding_support import (
    baseline_mask as _mask,
)
from tests.unit._confounding_support import (
    forbid_draw_and_refit as _forbid_draw_and_refit,
)
from tests.unit._confounding_support import (
    replacement as _replacement,
)


@cache
def _fit_population(
    target: str = "att",
    *,
    method: str = "tmle",
    repeats: int = 1,
    reference: int = 0,
    strata: bool = True,
) -> Any:
    continuous = target in {"ey_shift", "ate_shift"}
    binary = target in {"rr", "or", "ey", "ey1", "ey0"}
    policies = (Shift(0.0, cap=10.0, name="natural"), Shift(0.4, cap=10.0, name="up"))
    targets = {
        "ate": ATE(reference=reference),
        "att": ATT(reference=reference),
        "atc": ATC(reference=reference),
        "rr": RiskRatio(reference=reference),
        "or": OddsRatio(reference=reference),
        "ey": CounterfactualMean(),
        "ey1": CounterfactualMean(treatment=1),
        "ey0": CounterfactualMean(treatment=0),
        "ey_shift": ModifiedTreatmentPolicy(shifts=policies),
        "ate_shift": ModifiedTreatmentPolicyEffect(shifts=policies),
    }
    configured: Any = method
    if method in _STRATEGY_OVERRIDES:
        configured = _strategy_method(method, selection_estimand=target)
    return confounding_estimate(
        confounding_study(
            law="population", continuous=continuous, binary=binary, strata=strata, weighted=True
        ),
        targets[target],
        method=configured,
        binary=binary,
        repeats=repeats,
    )


def _alias(result: Any, target: str, stratum: tuple[str, ...] | None) -> str:
    return alias_for(result, target, stratum, value="up" if target == "ey_shift" else None)


def _correlation(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    return float(
        np.average(x * y, weights=w)
        / np.sqrt(np.average(x * x, weights=w) * np.average(y * y, weights=w))
    )


@cache
def _fit_string_arms(target: str) -> tuple[Any, np.ndarray]:
    """Fit a conditional effect on string arm labels, and return the labels beside it.

    Every other fixture here labels the arms ``0`` and ``1``, where the position of a
    label in ``treatment_levels`` equals the label. A level lookup replaced by a cast
    then reads the same, and only a non-numeric label separates the two.

    Returns
    -------
    tuple of (TMLEResult, ndarray)
        The fitted result, and the original string arm of every row.
    """
    rng = np.random.default_rng(912)
    n = 180
    w = rng.normal(size=n)
    v = np.where(np.arange(n) % 3 == 0, "small", "large")
    coded = rng.binomial(1, 1 / (1 + np.exp(-0.8 * w + 0.9 * (v == "small"))))
    arms = np.where(coded == 1, "active", "control")
    y = 0.4 + (1.2 + 1.8 * w) * coded + 0.5 * w + rng.normal(scale=0.3, size=n)
    frame = pd.DataFrame({"W": w, "V": v, "A": arms, "Y": y})
    estimand = ATT(reference="control") if target == "att" else ATC(reference="control")
    result = (
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W", "V"),
                strata=("V",),
                treatment_kind="discrete",
            ),
        )
        .identify(estimand)
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=12,
            simultaneous=False,
        )
    )
    return result, arms


def _fit_collapsing_att(extra: float) -> Any:
    """Fit a marginal ATT whose treated arm sits mostly in the published flip tail.

    The surface flips the upper 0.32 tail of its own latent vector. A treated arm equal
    to that tail, plus ``extra`` of the remaining rows, leaves exactly ``extra`` of the
    rows treated after the flip. The conditioning share therefore moves from
    ``tail + extra`` to ``extra``, and one parameter places the surface on either side of
    the collapse rule.

    Parameters
    ----------
    extra : float
        Share of the non-tail rows that is treated before the perturbation.

    Returns
    -------
    TMLEResult
        Fitted ATT whose conditioning population the published flip rebuilds.
    """
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    n = 200
    latent = np.random.default_rng(module._latent_child_seed(31)).normal(size=n)
    tail = latent >= NormalDist().inv_cdf(1 - 0.32)
    rng = np.random.default_rng(77)
    treated = tail.copy()
    treated[rng.permutation(np.flatnonzero(~tail))[: round(extra * n)]] = True
    w = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "A": treated.astype(int),
            "W": w,
            "Y": treated * (1 + w) + rng.normal(size=n),
        }
    )
    return (
        CausalStudy(
            frame,
            design=PointTreatment(outcome="Y", treatment="A", adjustment=("W",)),
        )
        .identify(ATT())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            learner_folds=2,
            simultaneous=False,
            random_state=12,
        )
    )


@pytest.mark.parametrize("target", ["ate", "att", "atc", "ey", "ey1", "ey0", "rr", "or"])
def test_binary_stratum_cells_equal_complete_weighted_refits(target: str) -> None:
    result = _fit_population(target)
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17)),
        random_state=31,
    )
    cell = surface.cells[-1]
    replacement = _replacement(result, surface, 0.22, 0.17)
    manual = result.estimator.refit(replacement, random_state=surface.root_seed)
    assert cell.failure is None
    assert cell.estimate == manual[alias].psi
    assert cell.displacement == manual[alias].inference_value - result[alias].inference_value
    assert abs(cell.displacement) > 1e-5
    assert surface.stratum == ("small",)
    assert surface.strata_names == ("V",)
    assert surface.refit_population == "full_fitted_population"
    np.testing.assert_array_equal(replacement.strata, result.data.strata)
    np.testing.assert_array_equal(replacement.weights, result.data.weights)
    assert replacement.n == result.data.n


@pytest.mark.parametrize("target", ["att", "atc"])
@pytest.mark.parametrize("reference", [0, 1])
@pytest.mark.parametrize("stratum", [None, ("small",)])
def test_conditional_population_moves_with_perturbed_treatment(
    target: str,
    reference: int,
    stratum: tuple[str, ...] | None,
) -> None:
    result = _fit_population(target, reference=reference)
    alias = _alias(result, target, stratum)
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.32), outcome=(0.0,)),
        random_state=31,
    )
    replacement = _replacement(result, surface, 0.32, 0.0)
    manual = result.estimator.refit(replacement, random_state=surface.root_seed)
    baseline = _mask(result, stratum)
    conditioning = 1 - reference if target == "att" else reference
    old = baseline & (result.data.treatment == conditioning)
    moved = baseline & (replacement.treatment == conditioning)
    assert np.count_nonzero(old != moved) > 10
    # The targeted outcome contrast is held fixed here: only membership can change.
    targeted = manual.fluctuations[target].targeted
    contrast = targeted.arms[float(1 - reference)] - targeted.arms[float(reference)]
    current = np.average(contrast[moved], weights=replacement.weights[moved])
    frozen = np.average(contrast[old], weights=replacement.weights[old])
    assert abs(current - frozen) > 1e-3
    assert manual.nuisance.scaler.unscale_difference(current) == pytest.approx(manual[alias].psi)
    assert abs(manual.nuisance.scaler.unscale_difference(frozen) - manual[alias].psi) > 1e-3
    assert surface.population == "perturbed_treatment_group"
    assert surface.conditioning_arm == conditioning
    assert surface.stratum == stratum
    assert surface.strata_names == (() if stratum is None else ("V",))
    # Both ends of the population claim, not just the moving one: the anchor reports the
    # unperturbed conditioning share, and the perturbed cell reports its own.
    anchor_fraction = surface.cells[0].target_population_fraction
    assert anchor_fraction == pytest.approx(
        np.average(
            result.data.treatment[baseline] == conditioning, weights=result.data.weights[baseline]
        )
    )
    assert surface.cells[-1].target_population_fraction == pytest.approx(
        np.average(
            replacement.treatment[baseline] == conditioning, weights=replacement.weights[baseline]
        )
    )
    assert abs(anchor_fraction - surface.cells[-1].target_population_fraction) > 0.05
    assert surface.cells[-1].estimate == manual[alias].psi
    assert surface.cells[0].estimate == result[alias].psi
    assert surface.cells[0].displacement == 0.0


@pytest.mark.parametrize(
    ("method", "target"),
    [
        ("greedy", "ate"),
        ("ordered", "ate"),
        ("discrete", "ate"),
        ("oat", "ate"),
        # The conditional arm means and the two ratios, which the marginal C-TMLE grid
        # already covers and the stratified grid did not.
        ("greedy", "ey"),
        ("greedy", "ey1"),
        ("greedy", "rr"),
        ("greedy", "or"),
    ],
)
def test_collaborative_strata_keep_full_weighted_selection_refit(method: str, target: str) -> None:
    result = _fit_population(target, method=method)
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17)),
        random_state=31,
    )
    manual = result.estimator.refit(_replacement(result, surface, 0.22, 0.17), random_state=31)
    assert type(result.estimator) is CTMLE
    assert surface.cells[-1].estimate == manual[alias].psi
    assert abs(surface.cells[-1].displacement) > 1e-4
    assert surface.population == "baseline"
    assert surface.stratum == ("small",)
    assert surface.cells[-1].target_population_fraction == 1.0
    assert surface.movement_scale == (
        "log_ratio" if target in {"rr", "or"} else "estimate_difference"
    )


@pytest.mark.parametrize("target", ["ey_shift", "ate_shift"])
def test_continuous_policy_strata_equal_complete_refits(target: str) -> None:
    result = _fit_population(target)
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.4), outcome=(0.0, 0.17)),
        random_state=31,
    )
    manual = result.estimator.refit(_replacement(result, surface, 0.4, 0.17), random_state=31)
    assert surface.cells[-1].failure is None
    assert surface.cells[-1].estimate == manual[alias].psi
    assert abs(surface.cells[-1].displacement) > 1e-5
    assert surface.stratum == ("small",)


def test_strata_share_full_latent_draw_and_global_calibration_but_local_association() -> None:
    result = _fit_population("att")
    kwargs = {
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.32), outcome=(0.0,)),
        "benchmark_covariates": ("W",),
        "random_state": 31,
    }
    surfaces = [
        simulated_confounding(result, _alias(result, "att", stratum), **kwargs)
        for stratum in [None, ("small",), ("large",)]
    ]
    pooled, small, large = surfaces
    assert pooled.latent_seed == small.latent_seed == large.latent_seed
    assert pooled.calibrations == small.calibrations == large.calibrations
    assert small.calibration_population == "full_fitted_population"
    latent = np.random.default_rng(small.latent_seed).normal(size=result.data.n)
    data = _replacement(result, small, 0.32, 0.0)
    for surface in surfaces:
        mask = _mask(result, surface.stratum)
        expected = _correlation(data.treatment[mask], latent[mask], data.weights[mask])
        assert surface.cells[-1].induced_treatment_association == pytest.approx(expected)
    assert (
        abs(
            small.cells[-1].induced_treatment_association
            - pooled.cells[-1].induced_treatment_association
        )
        > 0.03
    )
    assert small.association_population == "selected_baseline_stratum"
    assert pooled.association_population == "full_fitted_population"


def test_weighted_repeated_population_surface_cache_and_persistence() -> None:
    result = _fit_population("att", repeats=3)
    alias = _alias(result, "att", ("small",))
    kwargs = {
        "estimand": alias,
        "grid": ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0,)),
        "benchmark_covariates": ("W",),
        "random_state": 32,
    }
    surface = result.sensitivity.simulated_confounding(**kwargs)
    manual = result.estimator.refit(_replacement(result, surface, 0.22, 0.0), random_state=32)
    assert surface.n_repeats == 3
    assert surface.cells[-1].estimate == manual[alias].psi
    assert surface.cells[-1].estimate == float(
        np.median([draw.psi[alias] for draw in manual.repeats])
    )
    assert abs(surface.cells[-1].estimate - manual.repeats[0].psi[alias]) > 1e-5
    assert result.sensitivity.simulated_confounding(**kwargs) is surface
    assert loads(dumps(result)).sensitivity.simulated_confounding(**kwargs) == surface
    frame = surface.to_frame()
    assert list(frame["target_population_fraction"]) == [
        cell.target_population_fraction for cell in surface.cells
    ]
    assert set(frame["association_population"]) == {"selected_baseline_stratum"}
    assert set(surface.calibration_frame()["calibration_population"]) == {"full_fitted_population"}
    summary = surface.summary()
    assert "target population: perturbed_treatment_group" in summary
    assert "conditioning arm: 1" in summary
    assert "baseline stratum: ('small',)" in summary
    assert "strata columns: ('V',)" in summary
    assert "association population: selected_baseline_stratum" in summary


#: The two refusals the tampering table separates. Matching the exception type alone let a
#: guard that fired for an unrelated reason stand in for the one under test.
_BASELINE_REFUSAL = "simulated_confounding found inconsistent baseline-stratum metadata"
_PARAMETER_REFUSAL = "simulated_confounding found inconsistent registered binary parameter metadata"


def _tampered(result: Any, change: str, alias: str) -> Any:
    """Corrupt one provenance field of a fitted result and return the copy."""
    data = result.data
    identified = result.identified_effect
    study = identified._study
    if change == "key":
        return with_key(result, alias, stratum=("absent",))
    if change == "levels":
        return replace(result, data=replace(data, strata_levels=(("absent",), ("large",))))
    if change == "names":
        return replace(result, data=replace(data, strata_names=("wrong",)))
    if change == "study":
        return replace(result, identified_effect=replace(identified, _study=None))
    if change == "design":
        return replace(
            result,
            identified_effect=replace(
                identified, _study=SimpleNamespace(design=object(), data=data)
            ),
        )
    if change == "levels-order":
        # Names and codes still agree; only the order of the level list moves, which is
        # what turns the requested stratum into the code of the other one.
        return replace(result, data=replace(data, strata_levels=(("large",), ("small",))))
    if change == "strata-values":
        return replace(result, data=replace(data, strata=1 - data.strata))
    if change in {"strata-codes", "strata-shape"}:
        strata = data.strata * 2 if change == "strata-codes" else np.append(data.strata, 0)
        # The study repeats the same corrupt column, so the row-by-row comparison agrees
        # and only the code-range or the shape check can refuse.
        return replace(
            result,
            data=replace(data, strata=strata),
            identified_effect=replace(
                identified,
                _study=SimpleNamespace(design=study.design, data=replace(data, strata=strata)),
            ),
        )
    if change == "empty-stratum":
        weights = data.weights.copy()
        weights[data.strata == data.strata_levels.index(("small",))] = 0.0
        return replace(result, data=replace(data, weights=weights))
    if change == "typed":
        return replace(result, identified_effect=replace(identified, estimand=ATC()))
    if change == "functional":
        return with_functional(result, target="atc")
    # ``TMLE.__init__`` leaves ``estimands`` unresolved, so ``None`` is the shape an
    # estimator that never fitted carries. The guard must name it, not raise ``TypeError``
    # out of ``tuple(None)``.
    return with_estimator(result, estimands=None if change == "replay-unresolved" else ("atc",))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("key", _BASELINE_REFUSAL),
        ("levels", _BASELINE_REFUSAL),
        ("names", _BASELINE_REFUSAL),
        ("study", _BASELINE_REFUSAL),
        ("design", _BASELINE_REFUSAL),
        ("levels-order", _BASELINE_REFUSAL),
        ("strata-values", _BASELINE_REFUSAL),
        ("strata-codes", _BASELINE_REFUSAL),
        ("strata-shape", _BASELINE_REFUSAL),
        ("empty-stratum", _BASELINE_REFUSAL),
        ("typed", _PARAMETER_REFUSAL),
        ("functional", _PARAMETER_REFUSAL),
        ("replay", _PARAMETER_REFUSAL),
        ("replay-unresolved", _PARAMETER_REFUSAL),
    ],
)
def test_population_provenance_tampering_precedes_draws_and_refits(
    change: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each corrupted field reaches a guard, and the surface refuses before it draws.

    ``match=`` separates the two blocks and not the guards inside them. Ten rows corrupt
    the baseline-population coherence chain, which raises ``_BASELINE_REFUSAL`` from a
    single ``or``, and four corrupt the registered-parameter chain, which raises
    ``_PARAMETER_REFUSAL`` the same way. What separates the rows is the corruption. Each
    row leaves every other conjunct satisfied, so only the conjunct it targets can refuse.
    Deleting the shape check fails ``[strata-shape]`` alone. Deleting the code-range check
    fails ``[strata-codes]`` alone.

    ``[replay]`` and ``[replay-unresolved]`` are the exception. They reach the same
    ``target not in replay_targets`` conjunct, and they separate the shape of the refusal
    rather than its site: an unresolved ``estimands`` must reach the named
    ``CapabilityError`` and not raise ``TypeError`` on the way.

    ``_forbid_draw_and_refit`` pins that every refusal comes before the latent draw and
    the refit, so no row reaches its message through a later failure.
    """
    result = _fit_population("att")
    alias = _alias(result, "att", ("small",))
    tampered = _tampered(result, change, alias)
    _forbid_draw_and_refit(monkeypatch, tampered.estimator)
    with pytest.raises(CapabilityError, match=message):
        simulated_confounding(
            tampered, alias, grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        )


def test_a_validated_request_compares_and_hashes_without_its_mask() -> None:
    """``_ValidatedRequest.baseline_mask`` carries ``compare=False``, and this is why.

    The field holds an ``ndarray``. Inside the generated ``__eq__`` the mask compares
    elementwise, so the chain of per-field ``==`` yields an array and ``bool()`` of it
    raises. The generated ``__hash__`` hashes the same field and raises ``TypeError``.
    Without the marker both statements below fail, so the marker is not a comment.
    """
    result = _fit_population("att")
    alias = _alias(result, "att", ("small",))
    request = _validate_request(
        result, alias, ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)), ()
    )
    same = replace(request, baseline_mask=request.baseline_mask.copy())
    other = replace(request, conditioning_code=1.0 - request.conditioning_code)

    assert request.baseline_mask.shape == (result.data.n,)
    assert request.baseline_mask.any()
    assert isinstance(hash(request), int)
    assert request == request
    # The mask is excluded, not merely tolerated: a distinct array of the same values
    # still compares equal, and the fields that do identify the request still decide.
    assert same.baseline_mask is not request.baseline_mask
    assert same == request and hash(same) == hash(request)
    assert other != request


def test_a_faithful_study_surrogate_still_resolves_the_baseline_stratum() -> None:
    """The control for the tampering table: substitution alone refuses nothing.

    Three rows above replace ``IdentifiedEffect._study`` with a stand-in. If the stand-in
    were what the guard rejected, those rows would witness the substitution and not the
    corrupt field they carry.
    """
    result = _fit_population("att")
    alias = _alias(result, "att", ("small",))
    identified = result.identified_effect
    forged = replace(
        result,
        identified_effect=replace(
            identified,
            _study=SimpleNamespace(design=identified._study.design, data=result.data),
        ),
    )
    surface = simulated_confounding(
        forged,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,)),
        random_state=31,
    )
    assert surface.stratum == ("small",)
    assert surface.cells[0].estimate == result[alias].psi
    assert surface.cells[0].target_population_fraction == pytest.approx(
        np.average(
            result.data.treatment[_mask(result, ("small",))] == 1,
            weights=result.data.weights[_mask(result, ("small",))],
        )
    )


@pytest.mark.parametrize("target", ["att", "atc"])
def test_empty_perturbed_conditioning_population_retains_a_failed_cell(target: str) -> None:
    """The total collapse fails its own cell, and the assessment row still reports it.

    The conditioning arm of the perturbed cell empties, so its refit raises ``DataError``
    and the cell is not a successful cell. It is also the only cell that collapsed. A
    minimum taken over ``successful_cells`` therefore reads the anchor against itself, and
    it withholds the population advice in the one case where the collapse is total. The
    row would then report a positive minimum on a surface whose own frame shows ``0.0``.
    """
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    n = 160
    latent = np.random.default_rng(module._latent_child_seed(31)).normal(size=n)
    tail = latent >= NormalDist().inv_cdf(0.68)
    small = np.arange(n) % 2 == 0
    a = tail.astype(int) if target == "att" else (~tail).astype(int)
    # Outside the selected stratum both arms survive the same published flip.
    a[~small] = np.arange(n)[~small] % 3 == 0
    rng = np.random.default_rng(932)
    w = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "A": a,
            "W": w,
            "V": np.where(small, "small", "large"),
            "Y": a * (1 + w) + rng.normal(size=n),
        }
    )
    result = (
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W", "V"),
                strata=("V",),
            ),
        )
        .identify(ATT() if target == "att" else ATC())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            learner_folds=2,
            simultaneous=False,
            random_state=12,
        )
    )
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.32), outcome=(0.0,)),
        random_state=31,
    )
    assert surface.cells[0].failure is None
    # The anchor is the nonzero control for the ``None`` below: the surface reports an
    # association wherever the treatment of the cell still varies.
    assert surface.cells[0].induced_treatment_association is not None
    assert surface.cells[0].target_population_fraction > 0.0
    cell = surface.cells[-1]
    assert cell.target_population_fraction == 0.0
    assert cell.failure is not None
    # The retained reason, not merely some reason: the arm that defines the conditioning
    # population is empty, and a constant treatment has no correlation to report.
    assert cell.failure.error_type == "DataError"
    assert "no positive-weight observations from treatment arm" in cell.failure.message
    assert cell.induced_treatment_association is None
    assert cell.estimate is None and cell.displacement is None
    assert not surface.complete

    item = INTERPRETERS["simulated_confounding"](surface, None)
    anchor = surface.cells[0].target_population_fraction
    assert surface.successful_cells == (surface.cells[0],)
    assert item.status is AssessmentStatus.WARNING
    assert f"minimum target population fraction 0 against anchor {anchor:.4g}" in item.detail
    assert any("keeps under half its unperturbed share" in step for step in item.next_steps)


@pytest.mark.parametrize("method", ["greedy", "oat", "drtmle"])
@pytest.mark.parametrize("target", ["att", "atc"])
def test_unsupported_conditional_population_estimators_refuse_before_draws(
    target: str,
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fit_population(target, strata=False)
    estimator = DRTMLE() if method == "drtmle" else CTMLE(strategy=method)
    result = replace(result, estimator=estimator)
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(module, "_latent_child_seed", lambda *_: pytest.fail("drew before refusal"))
    monkeypatch.setattr(
        estimator, "refit", lambda *_args, **_kwargs: pytest.fail("refit before refusal")
    )
    with pytest.raises(CapabilityError, match="exact ordinary TMLE"):
        simulated_confounding(
            result, target, grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        )


def test_stratified_drtmle_refuses_before_draws(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _fit_population("ate")
    estimator = DRTMLE()
    result = replace(result, estimator=estimator)
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(module, "_latent_child_seed", lambda *_: pytest.fail("drew before refusal"))
    monkeypatch.setattr(
        estimator, "refit", lambda *_args, **_kwargs: pytest.fail("refit before refusal")
    )
    with pytest.raises(CapabilityError, match="stratified reduced-regression targeting"):
        simulated_confounding(
            result,
            _alias(result, "ate", ("small",)),
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,)),
        )


def test_unavailable_policy_alias_excludes_conditional_natural_course_means() -> None:
    result = _fit_population("ey_shift")
    supported = [alias for alias, key in result.parameter_keys.items() if key.value == "up"]
    natural = [alias for alias, key in result.parameter_keys.items() if key.value == "natural"]
    assert len(natural) == 3 and len(supported) == 3
    with pytest.raises(ValueError, match="unavailable") as refusal:
        simulated_confounding(
            result, "unknown", grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )
    assert all(alias in str(refusal.value) for alias in supported)
    assert all(alias not in str(refusal.value) for alias in natural)


@pytest.mark.parametrize(("target", "conditioning"), [("att", "active"), ("atc", "control")])
def test_a_string_arm_conditions_through_the_fitted_treatment_levels(
    target: str, conditioning: str
) -> None:
    """The conditioning arm is a label, and its code is its position in the fitted levels.

    Every other fixture labels the arms ``0`` and ``1``, so the position of a label equals
    the label and a cast reads the same as a lookup. Here ``"active"`` sits at position
    zero and ``"control"`` at position one, so a cast raises and a swapped lookup reports
    the other arm's share.
    """
    result, arms = _fit_string_arms(target)
    alias = _alias(result, target, ("small",))
    surface = simulated_confounding(
        result,
        alias,
        grid=ConfounderStrengthGrid(treatment=(0.0, 0.32), outcome=(0.0,)),
        random_state=31,
    )
    small = _mask(result, ("small",))
    other = "control" if conditioning == "active" else "active"

    assert result.data.treatment_levels == ("active", "control")
    assert surface.population == "perturbed_treatment_group"
    assert surface.conditioning_arm == conditioning
    assert surface.cells[0].target_population_fraction == pytest.approx(
        float(np.mean(arms[small] == conditioning))
    )
    assert surface.cells[0].target_population_fraction != pytest.approx(
        float(np.mean(arms[small] == other))
    )
    assert surface.cells[-1].failure is None
    moved = surface.cells[-1].target_population_fraction
    assert abs(moved - surface.cells[0].target_population_fraction) > 0.05


def test_the_eligible_alias_set_drops_what_the_replay_guard_would_refuse() -> None:
    """The advertised set and the guard read one predicate, so they cannot disagree.

    The facade substitutes a sole eligible alias, and the capability row asks for an
    explicit estimand only when several remain. An eligible set that kept an alias the
    guard refuses moved the refusal from the row to the call.
    """
    conditional = _fit_population("att")
    marginal_mean = _fit_population("ey1")
    collaborative = replace(conditional, estimator=CTMLE(strategy="greedy"))
    reduced = replace(marginal_mean, estimator=DRTMLE())

    assert _eligible_binary_parameter_names(conditional) == tuple(conditional.estimates)
    assert _eligible_binary_parameter_names(marginal_mean) == tuple(marginal_mean.estimates)
    # C-TMLE leaves the mean family, so no ATT alias survives.  DR-TMLE keeps the mean
    # family but cannot replay a requested stratum, so only the marginal alias survives.
    assert _eligible_binary_parameter_names(collaborative) == ()
    assert _eligible_binary_parameter_names(reduced) == ("ey1",)
    assert len(marginal_mean.estimates) == 3

    row = "simulated_confounding"
    assert conditional.sensitivity.capability(row).requires_arguments == ("grid", "estimand")
    assert collaborative.sensitivity.capability(row).requires_arguments == ("grid",)
    assert reduced.sensitivity.capability(row).requires_arguments == ("grid",)


def test_the_facade_refuses_a_conditional_alias_it_never_advertised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facade still reaches the guard, and the guard names the estimator boundary."""
    result = replace(_fit_population("att", strata=False), estimator=CTMLE(strategy="greedy"))
    _forbid_draw_and_refit(monkeypatch, result.estimator)
    assert _eligible_binary_parameter_names(result) == ()
    with pytest.raises(CapabilityError, match="ATT and ATC under exact ordinary TMLE"):
        result.sensitivity.simulated_confounding(
            grid=ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0,))
        )


def test_the_natural_course_mean_is_withheld_from_its_canonical_name_alone() -> None:
    """The refused policy mean is named twice, and either name must withhold it.

    ``_zero_delta_policy_means`` reads the declared interventions for the canonical alias
    and the stored parameter keys for every stratified alias. A result whose keys omit the
    canonical alias leaves only the first half to withhold it.
    """
    result = _fit_population("ey_shift")
    canonical = parameter_name("ey_shift", arm="natural")
    stripped = {
        alias: key
        for alias, key in result.parameter_keys.items()
        if not (key.estimand == "ey_shift" and key.value == "natural" and key.stratum is None)
    }
    forged = replace(result, parameter_keys=stripped)
    offered = parameter_name("ey_shift", arm="up")

    assert canonical in result.estimates
    assert canonical not in stripped
    assert canonical in _zero_delta_policy_means(forged)
    with pytest.raises(ValueError, match="unavailable") as refusal:
        simulated_confounding(
            forged, "unknown", grid=ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
        )
    assert repr(canonical) not in str(refusal.value)
    assert repr(offered) in str(refusal.value)


def test_the_assessment_row_renders_a_real_conditional_population_surface() -> None:
    """A real surface reaches the interpreter, and the row names the requested stratum.

    The interpreter was exercised on a hand-built stand-in only. A field renamed on the
    surface, or a stratum the row never read, left that stand-in unchanged.
    """
    result = _fit_population("att")
    alias = _alias(result, "att", ("small",))
    battery = result.assess(
        include_refits=True,
        arguments={
            "simulated_confounding": {
                "grid": ConfounderStrengthGrid(treatment=(0.0, 0.32), outcome=(0.0,)),
                "estimand": alias,
            }
        },
    )
    item = battery.sensitivity["simulated_confounding"]
    small = _mask(result, ("small",))
    anchor = np.average(result.data.treatment[small] == 1, weights=result.data.weights[small])

    assert type(item._report) is SimulatedConfoundingResult
    assert item._report.stratum == ("small",)
    assert item.status is AssessmentStatus.COMPLETED
    assert "target population: perturbed_treatment_group" in item.detail
    assert "conditioning arm: 1" in item.detail
    assert "baseline stratum: ('small',)" in item.detail
    assert "strata columns: ('V',)" in item.detail
    assert "association population: selected_baseline_stratum" in item.detail
    assert f"against anchor {anchor:.4g}" in item.detail


@pytest.mark.parametrize(
    ("extra", "status", "steps"),
    [(0.30, AssessmentStatus.WARNING, 1), (0.36, AssessmentStatus.COMPLETED, 0)],
)
def test_the_assessment_row_warns_when_the_conditioning_group_halves(
    extra: float, status: AssessmentStatus, steps: int
) -> None:
    """The collapse rule is pinned from both sides of its one-half threshold.

    Both fits sit within 0.03 of the threshold, so a rule that used another constant
    reports the wrong status for one of them.

    The grid also declares the anchor last, and the status alone cannot witness that. The
    anchor holds the largest fraction on the surface, so reading a different cell can only
    lower the divisor, and a rule that read ``cells[0]`` weakens the threshold and never
    strengthens it. It flips the ``0.30`` row to ``completed`` and leaves the ``0.36`` row
    ``completed`` for the wrong reason. Each row therefore checks the anchor the detail
    line printed, which the mutation moves on both rows.
    """
    result = _fit_collapsing_att(extra)
    surface = simulated_confounding(
        result,
        "att",
        grid=ConfounderStrengthGrid(treatment=(0.32, 0.0), outcome=(0.0,)),
        random_state=31,
    )
    perturbed, anchor = surface.cells
    item = INTERPRETERS["simulated_confounding"](surface, None)

    assert surface.complete
    assert surface.population == "perturbed_treatment_group"
    assert (anchor.treatment_strength, anchor.outcome_strength) == (0.0, 0.0)
    assert anchor.target_population_fraction == pytest.approx(
        float(np.mean(result.data.treatment == 1.0))
    )
    assert perturbed.target_population_fraction == pytest.approx(extra)
    ratio = perturbed.target_population_fraction / anchor.target_population_fraction
    assert 0.47 < ratio < 0.53
    assert item.status is status
    assert len(item.next_steps) == steps
    if steps:
        assert "keeps under half its unperturbed share" in item.next_steps[0]
    # The divisor the rule used, read back off the row it printed. This is what both rows
    # witness about cell selection; the status witnesses it on the collapsing row alone.
    assert (
        f"minimum target population fraction {perturbed.target_population_fraction:.4g} "
        f"against anchor {anchor.target_population_fraction:.4g}"
    ) in item.detail
    assert f"against anchor {perturbed.target_population_fraction:.4g}" not in item.detail


def test_the_reading_guard_names_the_population_channel_only_where_it_exists() -> None:
    """A conditional surface reads its movement differently from a baseline surface.

    An ATT cell rebuilds its own population, so a near-zero association does not say the
    movement is misclassification. The baseline sentence would contradict the population
    fields printed above it in the same summary.
    """
    conditional = _fit_population("att")
    marginal = _fit_population("ate")
    grid = ConfounderStrengthGrid(treatment=(0.0,), outcome=(0.0,))
    att = simulated_confounding(
        conditional, _alias(conditional, "att", ("small",)), grid=grid, random_state=31
    )
    ate = simulated_confounding(
        marginal, _alias(marginal, "ate", ("small",)), grid=grid, random_state=31
    )
    misclassification = (
        "An association near zero says the treatment axis moved the estimate by "
        "misclassification and not by confounding."
    )
    conditional_summary = att.summary()
    baseline_summary = ate.summary()

    assert att.population == "perturbed_treatment_group"
    assert ate.population == "baseline"
    assert "ATT and ATC membership follows each cell's perturbed treatment." in conditional_summary
    assert (
        "An association near zero says the treatment axis opened no confounding path."
        in conditional_summary
    )
    assert "rebuilds its ATT or ATC population from the perturbed treatment" in conditional_summary
    assert "Read target_population_fraction beside the movement." in conditional_summary
    assert misclassification not in conditional_summary

    assert misclassification in baseline_summary
    assert "rebuilds its ATT or ATC population" not in baseline_summary
    assert "ATT and ATC membership follows" not in baseline_summary
    assert "Read target_population_fraction beside the movement." not in baseline_summary
