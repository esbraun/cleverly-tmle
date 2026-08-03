r"""Every estimand's influence curve in one traversal -- the plan's headline hypothesis.

The package computes each estimand separately.  ``counterfactual_means`` builds
``1{A=a}/g_a (Y - Q*) + Q*_a - psi_a`` per arm, ``att_estimate`` and ``atc_estimate``
build the conditional curves, ``ratio_estimates`` delta-methods the risk and odds ratios
out of the two arm means, and each of those is its own set of full-length numpy
expressions over the same six arrays.  The hypothesis worth testing is that computing them
together -- one pass over the rows, every accumulator updated from the values already in
registers -- is materially cheaper.

**The profile says to expect a negative result, and the benchmark runs it anyway.**
Measured on a cached-nuisance ``retarget`` at ``n = 20,000``, going from one estimand to
seven moves the whole post-nuisance step from 11.5 ms to 28.6 ms -- and almost all of that
is the *fluctuation*, not the curves: ``att`` and ``atc`` are separate target groups and
each pays its own Newton solve.  ``counterfactual_means`` and ``_conditional_effects``
together are 2-4 ms of the 28.6.  So a fused kernel is optimising something like a tenth of
the step, and even a perfect one cannot return more than that.

That is a reason to *measure* rather than a reason to skip: "we fused it and it did not
matter" is a usable finding, "we assumed it would not matter" is not.  The kernel is also
the natural place to check the identity ``IC_ate == IC_ey1 - IC_ey0`` holds in the fused
path, which is the exact-identity gate the repository prefers over a tolerance.

The comparison is over the whole estimand set as a mapping, so an implementation that
computes six of seven correctly fails rather than scoring six-sevenths.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..fixtures import Regime, make_influence
from ..implementations.numba_parallel import PARALLEL_AVAILABLE, pjit, prange
from ..implementations.numba_serial import njit
from ..validation import compare_mapping
from . import KernelSpec, register

__all__ = ["build", "numba_estimands", "numba_estimands_parallel", "numpy_estimands"]

#: Every estimand the fused kernel produces, in report order.  ``n_estimands`` selects a
#: prefix, so the sweep at 1 / 3 / 5 / 7 is a prefix of one list rather than four lists.
ESTIMANDS = ("ey1", "ey0", "ate", "att", "atc", "rr", "or")


def build(
    n: int = 100_000,
    n_estimands: int = 7,
    regime: Regime = "good",
    seed: int = 20260803,
) -> dict[str, Any]:
    if not 1 <= n_estimands <= len(ESTIMANDS):
        raise ValueError(f"n_estimands must be in 1..{len(ESTIMANDS)}; got {n_estimands}")
    fixture = make_influence(n, n_arms=2, regime=regime, seed=seed)
    return {"fixture": fixture, "estimands": ESTIMANDS[:n_estimands]}


# --------------------------------------------------------------------------- numpy


def numpy_estimands(inputs: dict[str, Any]) -> dict[str, Any]:
    """Each estimand built from its own full-length expressions, as the package does.

    Transcribed from :mod:`cleverly.inference.influence`: ``counterfactual_means`` for the
    arm means, the ATT and ATC conditional curves, and the delta-method log-ratio curves
    for ``rr`` and ``or``.  The plug-in and the curve are separate passes there and are
    kept separate here.
    """
    fixture = inputs["fixture"]
    wanted = inputs["estimands"]
    y = fixture.outcome
    q = fixture.targeted_arms
    q_observed = fixture.targeted_observed
    g = fixture.propensity
    indicator = fixture.treatment_indicator
    weights = fixture.weights
    mask = fixture.observed
    n = fixture.n

    residual = np.zeros_like(y)
    residual[mask] = y[mask] - q_observed[mask]

    out: dict[str, Any] = {}
    curves: dict[str, np.ndarray] = {}
    psi: dict[str, float] = {}

    for arm, name in ((1, "ey1"), (0, "ey0")):
        plug_in = float(np.average(q[:, arm], weights=weights))
        curve = weights * (indicator[:, arm] / g[:, arm] * residual + q[:, arm] - plug_in)
        psi[name] = plug_in
        curves[name] = curve

    if "ate" in wanted:
        psi["ate"] = psi["ey1"] - psi["ey0"]
        curves["ate"] = curves["ey1"] - curves["ey0"]

    share = fixture.arm_fractions
    if "att" in wanted:
        # E[Y^1 - Y^0 | A = 1]: the odds g_0/g_1 in place of the second inverse weight,
        # and the treated share as the denominator.
        treated = indicator[:, 1]
        odds = g[:, 0] / g[:, 1]
        plug_in = float(np.mean(treated * (q[:, 1] - q[:, 0])) / share[1])
        curves["att"] = (
            weights
            * (treated * residual - indicator[:, 0] * odds * residual
               + treated * (q[:, 1] - q[:, 0] - plug_in))
            / share[1]
        )
        psi["att"] = plug_in
    if "atc" in wanted:
        control = indicator[:, 0]
        odds = g[:, 1] / g[:, 0]
        plug_in = float(np.mean(control * (q[:, 1] - q[:, 0])) / share[0])
        curves["atc"] = (
            weights
            * (indicator[:, 1] * odds * residual - control * residual
               + control * (q[:, 1] - q[:, 0] - plug_in))
            / share[0]
        )
        psi["atc"] = plug_in
    if "rr" in wanted:
        psi["rr"] = psi["ey1"] / psi["ey0"]
        curves["rr"] = curves["ey1"] / psi["ey1"] - curves["ey0"] / psi["ey0"]
    if "or" in wanted:
        psi["or"] = (psi["ey1"] / (1.0 - psi["ey1"])) / (psi["ey0"] / (1.0 - psi["ey0"]))
        curves["or"] = curves["ey1"] / (psi["ey1"] * (1.0 - psi["ey1"])) - curves["ey0"] / (
            psi["ey0"] * (1.0 - psi["ey0"])
        )

    for name in wanted:
        out[f"psi_{name}"] = np.array([psi[name]])
        out[f"ic_{name}"] = curves[name]
        out[f"se_{name}"] = np.array([np.sqrt((curves[name] ** 2).mean() / n)])
    return out


# --------------------------------------------------------------------------- numba


@njit()
def _moments_serial(y, q, q_observed, g, indicator, weights, mask):
    """The plug-ins, in one pass.  The curves need them, so they come first."""
    rows = y.shape[0]
    total_weight = 0.0
    psi1 = 0.0
    psi0 = 0.0
    att_numerator = 0.0
    atc_numerator = 0.0
    treated_share = 0.0
    control_share = 0.0
    for i in range(rows):
        w = weights[i]
        total_weight += w
        psi1 += w * q[i, 1]
        psi0 += w * q[i, 0]
        contrast = q[i, 1] - q[i, 0]
        att_numerator += indicator[i, 1] * contrast
        atc_numerator += indicator[i, 0] * contrast
        treated_share += indicator[i, 1]
        control_share += indicator[i, 0]
    return (
        psi1 / total_weight,
        psi0 / total_weight,
        att_numerator / treated_share,
        atc_numerator / control_share,
        treated_share / rows,
        control_share / rows,
    )


@njit()
def _curves_serial(
    y, q, q_observed, g, indicator, weights, mask,
    psi1, psi0, psi_att, psi_atc, share1, share0, out,
):
    """Every requested curve, one row at a time.

    ``out`` is ``(7, n)`` in :data:`ESTIMANDS` order.  The shared quantities -- the
    residual, the contrast, the two inverse weights -- are computed once per row and
    consumed by all seven, which is the whole hypothesis this kernel tests.
    """
    rows = y.shape[0]
    or1 = psi1 * (1.0 - psi1)
    or0 = psi0 * (1.0 - psi0)
    for i in range(rows):
        w = weights[i]
        residual = 0.0
        if mask[i]:
            residual = y[i] - q_observed[i]
        d1 = indicator[i, 1]
        d0 = indicator[i, 0]
        g1 = g[i, 1]
        g0 = g[i, 0]
        contrast = q[i, 1] - q[i, 0]
        ic1 = w * (d1 / g1 * residual + q[i, 1] - psi1)
        ic0 = w * (d0 / g0 * residual + q[i, 0] - psi0)
        out[0, i] = ic1
        out[1, i] = ic0
        out[2, i] = ic1 - ic0
        out[3, i] = w * (
            d1 * residual - d0 * (g0 / g1) * residual + d1 * (contrast - psi_att)
        ) / share1
        out[4, i] = w * (
            d1 * (g1 / g0) * residual - d0 * residual + d0 * (contrast - psi_atc)
        ) / share0
        out[5, i] = ic1 / psi1 - ic0 / psi0
        out[6, i] = ic1 / or1 - ic0 / or0


@pjit()
def _curves_parallel(
    y, q, q_observed, g, indicator, weights, mask,
    psi1, psi0, psi_att, psi_atc, share1, share0, out,
):
    """The same, with ``prange`` over rows.

    Rows are genuinely independent here -- every output slot is written by exactly one
    iteration -- so there is no reduction and no thread-local buffer.  That makes this the
    cleanest parallel kernel in the package and the one whose scaling curve is the purest
    statement about memory bandwidth: there is nothing else left to limit it.
    """
    rows = y.shape[0]
    or1 = psi1 * (1.0 - psi1)
    or0 = psi0 * (1.0 - psi0)
    for i in prange(rows):
        w = weights[i]
        residual = 0.0
        if mask[i]:
            residual = y[i] - q_observed[i]
        d1 = indicator[i, 1]
        d0 = indicator[i, 0]
        g1 = g[i, 1]
        g0 = g[i, 0]
        contrast = q[i, 1] - q[i, 0]
        ic1 = w * (d1 / g1 * residual + q[i, 1] - psi1)
        ic0 = w * (d0 / g0 * residual + q[i, 0] - psi0)
        out[0, i] = ic1
        out[1, i] = ic0
        out[2, i] = ic1 - ic0
        out[3, i] = w * (
            d1 * residual - d0 * (g0 / g1) * residual + d1 * (contrast - psi_att)
        ) / share1
        out[4, i] = w * (
            d1 * (g1 / g0) * residual - d0 * residual + d0 * (contrast - psi_atc)
        ) / share0
        out[5, i] = ic1 / psi1 - ic0 / psi0
        out[6, i] = ic1 / or1 - ic0 / or0


def _run(inputs: dict[str, Any], curves_kernel: Any) -> dict[str, Any]:
    fixture = inputs["fixture"]
    wanted = inputs["estimands"]
    args = (
        fixture.outcome,
        fixture.targeted_arms,
        fixture.targeted_observed,
        fixture.propensity,
        fixture.treatment_indicator,
        fixture.weights,
        fixture.observed,
    )
    psi1, psi0, psi_att, psi_atc, share1, share0 = _moments_serial(*args)
    out = np.empty((len(ESTIMANDS), fixture.n))
    curves_kernel(*args, psi1, psi0, psi_att, psi_atc, share1, share0, out)

    psi = {
        "ey1": psi1,
        "ey0": psi0,
        "ate": psi1 - psi0,
        "att": psi_att,
        "atc": psi_atc,
        "rr": psi1 / psi0,
        "or": (psi1 / (1.0 - psi1)) / (psi0 / (1.0 - psi0)),
    }
    result: dict[str, Any] = {}
    for index, name in enumerate(ESTIMANDS):
        if name not in wanted:
            continue
        curve = out[index]
        result[f"psi_{name}"] = np.array([psi[name]])
        result[f"ic_{name}"] = curve
        result[f"se_{name}"] = np.array([np.sqrt((curve**2).mean() / fixture.n)])
    return result


def numba_estimands(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _curves_serial)


def numba_estimands_parallel(inputs: dict[str, Any]) -> dict[str, Any]:
    return _run(inputs, _curves_parallel)


_IMPLEMENTATIONS: dict[str, Any] = {"numpy": numpy_estimands}
if PARALLEL_AVAILABLE:
    _IMPLEMENTATIONS["numba"] = numba_estimands
    _IMPLEMENTATIONS["numba_parallel"] = numba_estimands_parallel

register(
    KernelSpec(
        name="fused_influence_curves",
        estimator="tmle",
        build=build,
        implementations=_IMPLEMENTATIONS,
        compare=compare_mapping,
        tolerance=(1e-11, 1e-12),
        parallel_axis="rows",
        note=(
            "the plan's headline hypothesis; the profile puts the curves at about a tenth "
            "of a cached-nuisance retarget, so the ceiling is low before anything is fused"
        ),
        dimensions={
            "n": 100_000,
            "n_estimands": 7,
            "regime": "good",
            "seed": 20260803,
        },
        amortise=True,
    )
)
