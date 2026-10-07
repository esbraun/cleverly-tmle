"""Nonzero witnesses for composition across inference scales and bootstrap boundaries."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest

from cleverly.inference import Transform, make_estimate
from cleverly.inference.bootstrap import BootstrapResult
from cleverly.inference.results import (
    derived_bootstrap,
    ratio_contrast,
    ratio_function,
    smooth_contrast,
)

CURVE = np.array([-0.2, 0.1, -0.1, 0.2])


def _identity(values: np.ndarray) -> float:
    return float(values[0])


def _identity_gradient(values: np.ndarray) -> np.ndarray:
    return np.ones(1)


@pytest.mark.parametrize("value", [0.4, 2.5])
def test_identity_ratio_contrast_recovers_the_reported_scale(value: float) -> None:
    source = make_estimate("rr", value, CURVE, n=4, scale="ratio", log_psi=np.log(value))
    logged = smooth_contrast(
        {"rr": source}, _identity, ["rr"], n=4, scale="ratio", gradient=_identity_gradient
    )
    np.testing.assert_allclose(logged.influence_curve, source.influence_curve, atol=1e-15)
    assert logged.variance == pytest.approx(source.variance, rel=1e-14)
    assert logged.ci == pytest.approx(source.ci, rel=1e-14)
    plain = smooth_contrast(
        {"rr": source}, _identity, ["rr"], n=4, scale="level", gradient=_identity_gradient
    )
    np.testing.assert_allclose(plain.influence_curve, value * CURVE, atol=1e-15)


def test_a_mixed_ratio_and_level_contrast_uses_each_reported_scale() -> None:
    ratio = make_estimate("or", 2.5, CURVE, n=4, scale="ratio", log_psi=np.log(2.5))
    level = make_estimate("ey", 0.4, CURVE[::-1], n=4, scale="level")
    derived = smooth_contrast(
        {"or": ratio, "ey": level},
        lambda values: float(values[0] - values[1]),
        ["or", "ey"],
        n=4,
        gradient=lambda values: np.array([1.0, -1.0]),
    )
    assert derived.psi == pytest.approx(2.1)
    np.testing.assert_allclose(derived.influence_curve, 2.5 * CURVE - CURVE[::-1])


@pytest.mark.parametrize("kind", ["rr", "or"])
@pytest.mark.parametrize("complement", [False, True])
def test_ratio_contrast_recovers_transformed_level_curves(
    kind: Literal["rr", "or"], complement: bool
) -> None:
    a, b = 0.6, 0.3
    curve_b = np.array([-0.1, -0.3, 0.2, 0.2])
    # Construct both transformed inputs independently of smooth_contrast: one curve
    # is on the log scale and the other is on the logit scale.
    top = replace(
        make_estimate("a", a, CURVE / a, n=4, scale="level"),
        transform=Transform.log(),
    )
    bottom = replace(
        make_estimate("b", b, curve_b / (b * (1.0 - b)), n=4, scale="level"),
        transform=Transform.logit(),
    )
    derived = ratio_contrast(
        {"a": top, "b": bottom}, "a", "b", kind=kind, complement=complement, n=4, name="r"
    )
    curve_a = CURVE
    if complement:
        a, b = 1.0 - a, 1.0 - b
        curve_a, curve_b = -curve_a, -curve_b
    if kind == "rr":
        expected_value = a / b
        expected_curve = curve_a / a - curve_b / b
    else:
        expected_value = (a / (1.0 - a)) / (b / (1.0 - b))
        expected_curve = curve_a / (a * (1.0 - a)) - curve_b / (b * (1.0 - b))
    assert derived.psi == pytest.approx(expected_value, rel=1e-14)
    np.testing.assert_allclose(derived.influence_curve, expected_curve, atol=1e-15)
    assert derived.variance == pytest.approx(np.var(expected_curve, ddof=1) / 4, rel=1e-14)


@pytest.mark.parametrize(
    "transform",
    [
        Transform.log(),
        Transform.logit(),
        Transform("negative log", lambda x: -np.log(x), lambda y: np.exp(-y), lambda x: -1 / x),
    ],
)
def test_a_transformed_input_is_mapped_back_before_the_next_contrast(transform: Transform) -> None:
    source = make_estimate("ey", 0.4, CURVE, n=4, scale="level")
    transformed = smooth_contrast(
        {"ey": source},
        _identity,
        ["ey"],
        n=4,
        scale="level",
        gradient=_identity_gradient,
        transform=transform,
    )
    plain = smooth_contrast(
        {"transformed": transformed},
        _identity,
        ["transformed"],
        n=4,
        scale="level",
        gradient=_identity_gradient,
    )
    np.testing.assert_allclose(plain.influence_curve, CURVE, atol=1e-15)
    assert plain.variance == pytest.approx(source.variance, rel=1e-14)


@pytest.mark.parametrize("slope", [0.0, np.inf, np.nan])
def test_a_noninvertible_input_scale_refuses_composition(slope: float) -> None:
    source = replace(
        make_estimate("bad", 0.4, CURVE, n=4),
        transform=Transform("invalid slope", _identity_scalar, _identity_scalar, lambda x: slope),
    )
    with pytest.raises(ValueError, match="finite nonzero slope"):
        smooth_contrast({"bad": source}, _identity, ["bad"], n=4, gradient=_identity_gradient)


def _identity_scalar(value: float) -> float:
    return value


def test_a_nonfinite_transformed_input_refuses_composition() -> None:
    source = replace(
        make_estimate("bad", 0.4, CURVE, n=4),
        transform=Transform("invalid value", lambda x: np.inf, _identity_scalar, lambda x: 1.0),
    )
    with pytest.raises(ValueError, match="finite transformed value"):
        smooth_contrast({"bad": source}, _identity, ["bad"], n=4, gradient=_identity_gradient)


@pytest.mark.parametrize("transform", [Transform.log(), Transform.logit()])
def test_an_interior_transformed_estimate_can_be_displayed_without_a_default_test(
    transform: Transform,
) -> None:
    source = make_estimate("ey", 0.4, CURVE, n=4, scale="level")
    estimate = smooth_contrast(
        {"ey": source}, _identity, ["ey"], n=4, transform=transform, gradient=_identity_gradient
    )
    assert "p=nan" in repr(estimate)
    assert np.isnan(estimate.to_dict()["p_value"])
    assert estimate.wald_test(null=0.4).pvalue == 1.0
    with pytest.raises(ValueError, match="default null 0"):
        _ = estimate.pvalue


@pytest.mark.parametrize(
    ("kind", "complement", "invalid"),
    [
        ("rr", False, (0.4, 0.0)),
        ("rr", False, (0.0, 0.3)),
        ("rr", False, (-0.1, 0.3)),
        ("rr", True, (0.4, 1.0)),
        ("or", False, (1.0, 0.3)),
        ("or", False, (0.4, 1.0)),
        ("or", False, (0.4, 0.0)),
        ("or", True, (0.0, 0.3)),
    ],
)
def test_invalid_ratio_draws_are_dropped_and_counted(
    kind: str, complement: bool, invalid: tuple[float, float]
) -> None:
    # The two failed refits are absent from the stored arrays, as run_bootstrap stores
    # only successful refits. One nonfinite input and one domain-invalid ratio add two
    # failures for this derived parameter. The two interior draws remain aligned.
    draws = {
        "a": np.array([0.4, invalid[0], np.nan, 0.6]),
        "b": np.array([0.2, invalid[1], 0.3, 0.3]),
    }
    bootstrap = BootstrapResult(draws=draws, n_requested=6, n_failed=2, resampling="iid")
    estimates = {
        name: make_estimate(name, 0.4, CURVE, n=4).with_bootstrap(bootstrap.summary(name))
        for name in draws
    }
    function = ratio_function(kind, complement)  # type: ignore[arg-type]
    summary = derived_bootstrap(estimates, bootstrap, ["a", "b"], function, alpha=0.05)
    assert summary is not None
    assert summary.n_failed == 4
    assert summary.n_replicates == 2
    expected = np.array(
        {
            ("rr", False): (2.0, 2.0),
            ("rr", True): (0.75, 4.0 / 7.0),
            ("or", False): (8.0 / 3.0, 3.5),
            ("or", True): (3.0 / 8.0, 2.0 / 7.0),
        }[kind, complement]
    )
    np.testing.assert_allclose(summary.draws, expected, atol=1e-15, rtol=1e-14)
    np.testing.assert_allclose(summary.ci, np.quantile(expected, [0.025, 0.975]))


def test_derived_bootstrap_does_not_swallow_a_user_callable_error() -> None:
    bootstrap = BootstrapResult(
        draws={"a": np.array([0.4, 0.6])}, n_requested=2, n_failed=0, resampling="iid"
    )
    estimate = make_estimate("a", 0.4, CURVE, n=4).with_bootstrap(bootstrap.summary("a"))

    def broken(values: np.ndarray) -> float:
        raise ValueError("the user's contrast is broken")

    with pytest.raises(ValueError, match="the user's contrast is broken"):
        derived_bootstrap({"a": estimate}, bootstrap, ["a"], broken, alpha=0.05)


def test_a_nonfinite_callback_value_counts_as_a_derived_failure() -> None:
    bootstrap = BootstrapResult(
        draws={"a": np.array([0.4, 0.6, 0.8])}, n_requested=4, n_failed=1, resampling="iid"
    )
    estimate = make_estimate("a", 0.4, CURVE, n=4).with_bootstrap(bootstrap.summary("a"))
    summary = derived_bootstrap(
        {"a": estimate},
        bootstrap,
        ["a"],
        lambda values: float(values[0]) if values[0] < 0.7 else float("inf"),
        alpha=0.05,
    )
    assert summary is not None
    assert summary.n_failed == 2
    assert summary.n_replicates == 2
    np.testing.assert_array_equal(summary.draws, [0.4, 0.6])
