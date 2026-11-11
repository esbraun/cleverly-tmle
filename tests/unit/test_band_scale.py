"""The logit band scale, the bounds it reads, and the joint rows that judge it.

``band-finite-sample-plan.md`` section 6.  A band over a probability built on the identity scale
loses coverage when the reported standard error moves with the estimate (the binomial
mean-variance coupling); ``band_scale="logit"`` builds it on the logit of the estimate's position
in its range.  These tests pin the endpoint law, the transform and its slope by mutation, the
invariance that lets every scale share one critical value, the Mammen fill, the bounds every
builtin target declares, the boundary refusal, and the joint rows of the study harness.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model
from scipy.special import expit, logit

import cleverly
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError
from cleverly.inference import delta, multiplier
from cleverly.inference.influence import make_estimate, median_estimates, parameter_bounds
from cleverly.inference.multiplier import (
    median_bands,
    multiplier_critical_value,
    out_of_range,
    simultaneous_bands,
)
from cleverly.targets import TARGETS
from tests.studies.evidence import simultaneous as joint
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.registry import ROOT

N = 400


def _curves(seed: int = 0, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    curves = np.column_stack(
        [base[:, 0], 0.6 * base[:, 0] + 0.8 * base[:, 1], 0.5 * base[:, 1] + base[:, 2]]
    )
    return curves - curves.mean(axis=0)


def _estimates(psis: tuple[float, float, float] = (0.85, 0.4, 0.3)) -> dict[str, Any]:
    """A level near 0.85, a difference near 0.4 and an unbounded level."""
    curves = _curves()
    return {
        "ey1": make_estimate("ey1", psis[0], 0.3 * curves[:, 0], n=N, bounds=(0.0, 1.0)),
        "ate": make_estimate("ate", psis[1], 0.4 * curves[:, 1], n=N, bounds=(-1.0, 1.0)),
        "level": make_estimate("level", psis[2], 2.0 * curves[:, 2], n=N),
    }


def _expected(estimates: dict[str, Any], critical: float) -> dict[str, tuple[float, float]]:
    """The endpoint law, written from the formulas rather than the package's transform."""
    p, p_se = estimates["ey1"].psi, estimates["ey1"].std_error
    d, d_se = estimates["ate"].psi, estimates["ate"].std_error
    level, level_se = estimates["level"].psi, estimates["level"].std_error
    p_half = critical * p_se / (p * (1.0 - p))
    d_half = critical * d_se / (1.0 - d**2)
    return {
        "ey1": (float(expit(logit(p) - p_half)), float(expit(logit(p) + p_half))),
        "ate": (float(np.tanh(np.arctanh(d) - d_half)), float(np.tanh(np.arctanh(d) + d_half))),
        "level": (level - critical * level_se, level + critical * level_se),
    }


def _band(estimates: dict[str, Any], band_scale: str = "logit", kind: str = "mammen") -> Any:
    return simultaneous_bands(
        estimates,
        n_replicates=2_000,
        kind=kind,  # type: ignore[arg-type]
        random_state=0,
        band_scale=band_scale,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- the endpoint law


class TestTheEndpointLaw:
    def test_the_limits_are_the_logit_and_atanh_bands(self) -> None:
        estimates = _estimates()
        bands = _band(estimates)
        expected = _expected(estimates, bands.critical_value)
        for name in estimates:
            np.testing.assert_allclose(bands.bands[name], expected[name], rtol=0, atol=1e-12)
        assert bands.scales == {"ey1": "logit", "ate": "range_logit", "level": "identity"}
        assert bands.band_scale == "logit"
        # Far from 0.5 and 0 the transformed band is asymmetric, so the witness has power.
        for name in ("ey1", "ate"):
            low, high = bands.bands[name]
            psi = estimates[name].psi
            assert abs((high - psi) - (psi - low)) > 1e-3

    def test_the_identity_band_keeps_the_symmetric_limits(self) -> None:
        estimates = _estimates()
        bands = _band(estimates, band_scale="identity")
        for name, estimate in estimates.items():
            half = bands.critical_value * estimate.std_error
            np.testing.assert_allclose(
                bands.bands[name], (estimate.psi - half, estimate.psi + half), atol=1e-12
            )
        assert set(bands.scales.values()) == {"identity"}
        assert "scale" in bands.to_frame().columns

    @pytest.mark.parametrize("bounds", [(0.0, 1.0), (-1.0, 1.0)])
    def test_a_resolver_patched_to_identity_fails_the_endpoint_law(
        self, bounds: tuple[float, float]
    ) -> None:
        """Mutation: dropping the transform for one range moves that range's limits."""
        estimates = _estimates()
        real = multiplier._range_transform

        def mutant(estimate: Any, band_scale: str) -> Any:
            return None if estimate.bounds == bounds else real(estimate, band_scale)

        with patch.object(multiplier, "_range_transform", mutant):
            bands = _band(estimates)
        expected = _expected(estimates, bands.critical_value)
        name = "ey1" if bounds == (0.0, 1.0) else "ate"
        assert not np.allclose(bands.bands[name], expected[name], rtol=0, atol=1e-6)

    def test_a_wrong_slope_fails_the_endpoint_law_and_the_row_identity(self) -> None:
        """Mutation: the delta method without the division by the slope."""
        estimates = _estimates()
        with patch.object(delta.Transform, "slope", lambda self, value, step=1e-6: 1.0):
            bands = _band(estimates)
        expected = _expected(estimates, bands.critical_value)
        assert not np.allclose(bands.bands["ey1"], expected["ey1"], rtol=0, atol=1e-6)
        assert not np.allclose(bands.bands["ate"], expected["ate"], rtol=0, atol=1e-6)
        # The mutant's band is too narrow.  The joint row reads the statistic on the correct
        # slope, so a truth just outside the mutant's limit is outside the band but inside the
        # critical value.
        truth = {name: estimate.psi for name, estimate in estimates.items()}
        truth["ate"] = bands.bands["ate"][1] + 1e-6
        with pytest.raises(AssertionError, match="not read on the scale"):
            joint.joint_coverage_rows(
                _Result(estimates, bands),
                truth,
                list(estimates),
                label="witness",
                replicate=0,
                n=N,
                requested=1,
                pointwise_critical=1.959963984540054,
            )


# --------------------------------------------------------------- the critical value


class TestTheCriticalValue:
    @pytest.mark.parametrize("kind", ["rademacher", "mammen"])
    def test_rescaling_a_column_with_its_standard_error_leaves_c(self, kind: str) -> None:
        curves = _curves(3)
        se = curves.std(axis=0) / np.sqrt(N)
        factors = np.array([2.5, -0.3, 7.0])
        settings: dict[str, Any] = {"n": N, "n_replicates": 3_000, "kind": kind, "random_state": 1}
        base = multiplier_critical_value(curves, se, **settings)
        scaled = multiplier_critical_value(curves * factors, se * np.abs(factors), **settings)
        assert scaled == pytest.approx(base, rel=1e-12, abs=0)
        # Negative control: rescaling the curves alone moves the critical value.
        alone = multiplier_critical_value(curves * factors, se, **settings)
        assert abs(alone - base) > 1e-3

    def test_c_does_not_depend_on_the_band_scale(self) -> None:
        estimates = _estimates()
        for kind in ("rademacher", "mammen"):
            logit_band = _band(estimates, "logit", kind)
            identity_band = _band(estimates, "identity", kind)
            assert logit_band.critical_value == identity_band.critical_value

    def test_a_sparse_column_gives_a_finite_mammen_c(self) -> None:
        """B1: a column carried by three units still gives a finite fixed-SE critical value.

        A multinomial bootstrap-t that restudentizes resamples the units, and a resample that
        draws none of the three has a zero standard error and an infinite statistic, which the
        fixed-SE multiplier never forms.
        """
        rng = np.random.default_rng(5)
        n = 200
        curves = rng.normal(size=(n, 3))
        sparse = np.zeros(n)
        sparse[:3] = [4.0, -3.0, -1.0]
        curves[:, 2] = sparse - sparse.mean()
        curves -= curves.mean(axis=0)
        se = np.sqrt((curves**2).mean(axis=0) / n)
        settings: dict[str, Any] = {"n": n, "n_replicates": 4_000, "random_state": 0}
        mammen = multiplier_critical_value(curves, se, kind="mammen", **settings)
        rademacher = multiplier_critical_value(curves, se, kind="rademacher", **settings)
        assert np.isfinite(mammen)
        assert abs(mammen - rademacher) < 0.2
        # The mutant: resample the units and restudentize.  A resample that draws none of the
        # three carrying units has a zero standard error in that column, so its statistic is
        # infinite, and that happens far more often than alpha / 10.
        picked = rng.integers(0, n, size=(2_000, n))
        empty = ~np.isin(picked, np.arange(3)).any(axis=1)
        assert float(empty.mean()) > 0.05 / 10

    def test_the_mammen_threshold_keeps_the_moments(self) -> None:
        """The integer threshold moves the moments by at most 5.2e-10."""
        p = int(multiplier._MAMMEN_THRESHOLD) / 2**32
        low, high = multiplier._MAMMEN_LOW, multiplier._MAMMEN_HIGH
        moments = [p * low**k + (1.0 - p) * high**k for k in (1, 2, 3)]
        np.testing.assert_allclose(moments, [0.0, 1.0, 1.0], rtol=0, atol=1e-9)
        # A probability off by 1e-6 fails the same check.
        off = p + 1e-6
        shifted = [off * low**k + (1.0 - off) * high**k for k in (1, 2, 3)]
        assert not np.allclose(shifted, [0.0, 1.0, 1.0], rtol=0, atol=1e-9)

    def test_the_mammen_draws_match_their_moments(self) -> None:
        draws = multiplier._multipliers(np.random.default_rng(9), (1_000, 10_000), "mammen")
        draws = draws.reshape(-1)
        assert set(np.unique(draws)) == {multiplier._MAMMEN_LOW, multiplier._MAMMEN_HIGH}
        size = draws.size
        for k, target in ((1, 0.0), (2, 1.0), (3, 1.0)):
            values = draws**k
            mcse = float(values.std()) / math.sqrt(size)
            assert abs(float(values.mean()) - target) < 5 * mcse


# ------------------------------------------------------------------- the bounds


#: Every builtin target's declaration (plan 3.4).  A new target fails until it is listed.
DECLARED = {
    **dict.fromkeys(
        (
            "ey",
            "ey1",
            "ey0",
            "ey_obs",
            "ey_regime",
            "ey_learned_rule",
            "ey_optimal_rule",
            "ey_known_regime",
            "ey_ipsi",
            "ey_policy",
            "ey_rr_tilt",
        ),
        "mean",
    ),
    **dict.fromkeys(
        (
            "ate",
            "att",
            "atc",
            "par",
            "ate_regime",
            "ate_learned_rule",
            "ate_ipsi",
            "ate_policy",
            "ate_rr_tilt",
        ),
        "difference",
    ),
    **dict.fromkeys(
        ("paf", "rr", "or", "cate_projection", "log_crr_projection", "msm"),
        None,
    ),
}


def _binary_frame(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    a = rng.binomial(1, expit(0.4 * w))
    y = rng.binomial(1, expit(-0.2 + 2.0 * a + 0.5 * w))
    return pd.DataFrame({"W": w, "A": a, "Y": y})


def _tmle(**settings: Any) -> TMLE:
    chosen: dict[str, Any] = {
        "outcome_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "treatment_learner": sklearn.linear_model.LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "cross_fit": False,
        "random_state": 0,
        **settings,
    }
    return TMLE(**chosen)


class TestTheBounds:
    def test_every_builtin_target_declares_its_kind(self) -> None:
        assert {name: target.bounded for name, target in TARGETS.items()} == DECLARED

    def test_the_bounds_come_from_the_declaration_and_the_family(self) -> None:
        assert parameter_bounds("mean", "binomial") == (0.0, 1.0)
        assert parameter_bounds("difference", "binomial") == (-1.0, 1.0)
        assert parameter_bounds("mean", "gaussian") is None
        assert parameter_bounds(None, "binomial") is None

    def test_a_binary_fit_carries_the_bounds_and_a_continuous_one_does_not(self) -> None:
        frame = _binary_frame()
        binary = _tmle(simultaneous=False).fit(frame, outcome="Y", treatment="A").single()
        assert binary["ey1"].bounds == (0.0, 1.0)
        assert binary["ate"].bounds == (-1.0, 1.0)
        continuous = frame.assign(Y=frame["Y"] + 0.1 * frame["W"])
        fitted = (
            _tmle(
                simultaneous=False,
                outcome_learner=sklearn.linear_model.LinearRegression(),
            )
            .fit(continuous, outcome="Y", treatment="A")
            .single()
        )
        assert fitted["ey1"].bounds is None
        assert fitted["ate"].bounds is None

    def test_the_bounds_survive_the_median_and_a_save(self, tmp_path: Any) -> None:
        estimates = _estimates()
        combined = median_estimates([estimates, _estimates((0.84, 0.41, 0.31))])
        assert combined["ey1"].bounds == (0.0, 1.0)
        assert combined["level"].bounds is None
        frame = _binary_frame()
        result = _tmle(band_scale="logit").fit(frame, outcome="Y", treatment="A").single()
        path = tmp_path / "fit.cleverly"
        result.save(path)
        loaded = cleverly.load(path)
        assert loaded["ey1"].bounds == (0.0, 1.0)
        assert loaded.simultaneous.scales == result.simultaneous.scales

    def test_a_binary_tmle_fit_builds_the_logit_band(self) -> None:
        frame = _binary_frame()
        result = (
            _tmle(band_scale="logit", multiplier_kind="mammen")
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        bands = result.simultaneous
        assert bands.scales["ey1"] == "logit"
        assert bands.scales["ate"] == "range_logit"
        p, se = result["ey1"].psi, result["ey1"].std_error
        half = bands.critical_value * se / (p * (1.0 - p))
        np.testing.assert_allclose(
            bands.bands["ey1"], (expit(logit(p) - half), expit(logit(p) + half)), atol=1e-12
        )


# ------------------------------------------------------------- the boundary rule


class TestTheBoundaryRule:
    @pytest.mark.parametrize("psi", [0.0, 1.0, 1.02, -0.01])
    def test_an_estimate_on_or_outside_its_range_is_refused(self, psi: float) -> None:
        estimates = _estimates((psi, 0.4, 0.3))
        assert out_of_range(estimates, "logit") == ("ey1",)
        with pytest.raises(CapabilityError, match="'ey1'"):
            _band(estimates)
        # The identity band is built.
        assert out_of_range(estimates, "identity") == ()
        assert _band(estimates, "identity").bands["ey1"][0] < psi

    def test_a_zero_standard_error_at_a_bound_is_a_point_band(self) -> None:
        estimates = _estimates()
        estimates["edge"] = make_estimate("edge", 1.0, np.zeros(N), n=N, bounds=(0.0, 1.0))
        assert out_of_range(estimates, "logit") == ()
        assert _band(estimates).bands["edge"] == (1.0, 1.0)

    def test_the_repeated_band_refuses_too(self) -> None:
        estimates = _estimates((1.0, 0.4, 0.3))
        with pytest.raises(CapabilityError, match="'ey1'"):
            median_bands(
                [estimates, estimates],
                estimates,
                n_replicates=500,
                kind="mammen",
                random_state=0,
                band_scale="logit",
            )

    def test_the_repeated_band_reads_the_transformed_limits(self) -> None:
        draws = [_estimates(), _estimates((0.84, 0.41, 0.31))]
        combined = median_estimates(draws)
        bands = median_bands(
            draws, combined, n_replicates=500, kind="mammen", random_state=0, band_scale="logit"
        )
        expected = _expected(combined, bands.critical_value)
        np.testing.assert_allclose(bands.bands["ey1"], expected["ey1"], atol=1e-12)
        np.testing.assert_allclose(bands.bands["ate"], expected["ate"], atol=1e-12)

    def test_a_fit_skips_the_band_and_says_why(self) -> None:
        # Every treated unit has the event, so the treated mean is 1 with a positive SE from
        # the targeting step on a well-specified fit; the refusal path is read directly.
        frame = _binary_frame()
        fit = _tmle(band_scale="logit").fit(frame, outcome="Y", treatment="A")
        result = fit.single()
        moved = dict(result.estimates)
        assert out_of_range(moved, "logit") == ()
        moved["ey1"] = make_estimate(
            "ey1", 1.0, result["ey1"].influence_curve, n=result["ey1"].n, bounds=(0.0, 1.0)
        )
        assert out_of_range(moved, "logit") == ("ey1",)
        from cleverly._inference_status import NO_LOGIT_RANGE_BANDS

        assert "{names}" in NO_LOGIT_RANGE_BANDS
        assert "ey1" in NO_LOGIT_RANGE_BANDS.format(names="ey1")


# ------------------------------------------------------------------- joint rows


class _Result:
    """The two attributes a joint row reads: the estimates and the band."""

    def __init__(self, estimates: dict[str, Any], bands: Any) -> None:
        self._estimates = estimates
        self.simultaneous = bands

    def __getitem__(self, name: str) -> Any:
        return self._estimates[name]

    def __iter__(self) -> Any:
        return iter(self._estimates)


def _scale_rows(estimates: dict[str, Any], truth: dict[str, float], bands: Any) -> list[dict]:
    return joint.scale_coverage_rows(
        _Result(estimates, bands),
        truth,
        list(estimates),
        label="witness",
        replicate=0,
        n=N,
        requested=1,
        pointwise_critical=1.959963984540054,
        options=joint.BandOptions("logit", "mammen", 2_000, 0),
        reporting={"shipped": joint.BandOptions("identity", "rademacher", 2_000, 0)},
    )


class TestTheJointRows:
    def test_each_row_reads_its_own_scale(self) -> None:
        estimates = _estimates()
        bands = _band(estimates)
        truth = {"ey1": 0.80, "ate": 0.35, "level": 0.25}
        rows = {row["cell"].split("__")[1]: row for row in _scale_rows(estimates, truth, bands)}
        assert set(rows) == {
            joint.BAND,
            joint.CONTROL,
            joint.SCALE_CONTROL,
            "shipped",
            joint.REFUSED,
        }
        for name in (joint.BAND, joint.CONTROL, joint.SCALE_CONTROL, "shipped"):
            row = rows[name]
            assert row["covered"] == int(row["estimate"] <= row["std_error"])
        # The band and its identity control share the critical value and differ in scale.
        assert rows[joint.BAND]["std_error"] == rows[joint.SCALE_CONTROL]["std_error"]
        assert rows[joint.BAND]["estimate"] != rows[joint.SCALE_CONTROL]["estimate"]
        assert rows[joint.SCALE_CONTROL]["role"] == joint.SCALE_ROLE
        assert rows[joint.REFUSED]["covered"] == 0

    def test_a_refused_band_is_not_covered_and_keeps_the_replicate(self) -> None:
        estimates = _estimates((1.0, 0.4, 0.3))
        truth = {"ey1": 0.99, "ate": 0.4, "level": 0.3}
        rows = _scale_rows(estimates, truth, None)
        assert len(rows) == 5
        by_cell = {row["cell"].split("__")[1]: row for row in rows}
        band = by_cell[joint.BAND]
        assert band["covered"] == 0
        assert band["estimate"] == band["std_error"] + joint.REFUSED_MARGIN
        assert band["std_error"] == by_cell[joint.SCALE_CONTROL]["std_error"]
        assert by_cell[joint.REFUSED]["covered"] == 1
        assert by_cell[joint.REFUSED]["estimate"] == 1.0
        assert set(rows[0]) == set(joint._row(**_ROW))

    def test_a_band_built_over_an_out_of_range_estimate_is_an_error(self) -> None:
        estimates = _estimates((1.0, 0.4, 0.3))
        bands = _band(estimates, "identity")
        with pytest.raises(AssertionError, match="out-of-range"):
            _scale_rows(estimates, {"ey1": 0.99, "ate": 0.4, "level": 0.3}, bands)

    def test_a_band_with_other_options_is_an_error(self) -> None:
        estimates = _estimates()
        bands = _band(estimates, kind="rademacher")
        with pytest.raises(AssertionError, match="not the declared"):
            _scale_rows(estimates, {"ey1": 0.8, "ate": 0.35, "level": 0.25}, bands)


_ROW = {
    "cell": "c",
    "role": "positive",
    "replicate": 0,
    "n": 1,
    "requested": 1,
    "statistic": 0.0,
    "critical": 1.0,
    "covered": True,
}


def test_every_committed_band_row_is_its_statistic_inside_its_critical_value() -> None:
    """Every ``*__simultaneous_band`` row satisfies ``covered == (estimate <= std_error)``.

    Restricted to that suffix: a ``targeting_curves`` row means something else.
    """
    read = 0
    for path in sorted((ROOT / "tests" / "canonical").glob("*/property-replicates.parquet")):
        rows = read_rows(path)
        band = rows.loc[rows["cell"].astype(str).str.endswith(f"__{joint.BAND}")]
        if band.empty:
            continue
        read += 1
        identity = band["covered"] == (band["estimate"] <= band["std_error"]).astype(int)
        assert identity.all(), path.parent.name
    assert read >= 20
