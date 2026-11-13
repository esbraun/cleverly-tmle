"""The ``hazard-time-varying`` declaration: truths, bounds, limits, sizes, pilot readings, seeds.

Each truth is checked by an independent route: a forward sum over the covariate paths written
here, and a fit on each law's exact sample.  Each efficiency bound is checked against the
influence curve of that exact-sample fit, which at the true nuisances is the efficient
influence function.  The robustness limits and the wrong learners are checked on the exact sample
and against a limit computed from the law alone, and the replicate counts against the shared
sizing rules.  The pilot readings come from the committed pilot artifact, and the tests that read
it skip until it is committed.  Nothing here reads a study run.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import norm

import cleverly.longitudinal.estimator as estimator_module
from cleverly import _support
from cleverly._support_rows import LONGITUDINAL_ROWS
from cleverly.utils.bounds import DEFAULT_LTMLE_G_BOUNDS
from tests.studies import canonical_hazard_time_varying as study
from tests.studies import hazard_time_varying_law as law
from tests.studies import hazard_time_varying_pilot as pilot
from tests.studies import hazard_time_varying_properties as properties
from tests.studies import mtp_study_tools
from tests.studies import repeated_sizing as sizing
from tests.studies.evidence.property_verdicts import design_power, power_cell_pass_probability
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

MARGINS = study.STUDY.margins
CONTRAST = study.CONTRAST_LABEL
#: How close an exact-sample fit with a correct nuisance comes to the truth.  The walks stop at
#: ``tol=1e-10`` on the relative score and the logistic fits are solved to ``1e-10``.
EXACT = 1e-7
#: The exact-law gaps, in efficiency SEs at n = 2,000, of the plug-in whose hazard and
#: continuation omit ``L`` and of the both-wrong targeted fit, on ``one_cause_contrast``.
WRONG_GAP = 4.225
BOTH_WRONG_GAP = 4.216


def _quiet(function: Any, *args: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def _scale(label: str, n: int) -> float:
    return properties.EFFICIENCY_SD[label] / np.sqrt(n)


def _exact_fit(fit_label: str, configuration: str = "both_correct") -> Any:
    """The fit of one fit label's law on its exact sample, each outcome weighted by its mass."""
    frame = study.law_of(fit_label).exact_frame()
    return _quiet(study.fit, fit_label, frame, configuration=configuration, weights="wt")


@pytest.fixture(scope="module")
def exact_fits() -> dict[str, Any]:
    """The ``one_cause`` exact-sample fit of every robustness configuration."""
    return {c: _exact_fit("one_cause", c) for c in properties.CONFIGURATIONS}


# ---------------------------------------------------------------------------- independent routes


def _forward_truth(current: law.Law, regimen: Any, horizon: int, cause: int) -> float:
    """``E Q_1(W, d(A))`` by a forward sum over the covariate paths, apart from the law module.

    Each path of ``L_2, ..., L_t`` carries its probability given ``(W1, a)``, and along it the
    cause's incidence accumulates as survival times the cause's hazard at each node.
    """
    cov = current.covariate
    total = 0.0
    for w1, w2 in law.STRATA:
        g = current.g(w1, w2)
        for natural_level in range(current.arms):
            a = regimen(natural_level)
            for path in itertools.product((1, -1), repeat=horizon - 1):
                weight = 1.0
                previous = 0.0
                for value in path:
                    p = float(
                        expit(cov.intercept + cov.w1 * w1 + cov.arm[a] + cov.previous * previous)
                    )
                    weight *= p if value == 1 else 1.0 - p
                    previous = 1.0 if value == 1 else 0.0
                survival, incidence = 1.0, 0.0
                for node in range(1, horizon + 1):
                    flag = 0.0 if node == 1 else (1.0 if path[node - 2] == 1 else 0.0)
                    taken, hazards = 0.0, []
                    for factor in current.hazards:
                        eta = (
                            factor.intercept
                            + factor.trend * (node - 1)
                            + factor.arm[a]
                            + factor.w1 * w1
                            + factor.w2 * w2
                            + factor.l * flag
                        )
                        hazards.append(float(expit(eta)) * (1.0 - taken))
                        taken += hazards[-1]
                    incidence += survival * hazards[cause - 1]
                    survival *= 1.0 - taken
                total += law.MASS * g[natural_level] * weight * incidence
    return total


# ---------------------------------------------------------------------------- truths


class TestTheTruths:
    @pytest.mark.parametrize("law_name", list(law.LAWS))
    def test_the_backward_truth_is_the_forward_sum(self, law_name: str) -> None:
        current = law.LAWS[law_name]
        for label, regimen in study.maps(law_name).items():
            for cause in range(1, current.causes + 1):
                for horizon in range(1, current.n_times + 1):
                    assert current.truth(regimen, horizon, cause) == pytest.approx(
                        _forward_truth(current, regimen, horizon, cause), abs=1e-14
                    ), (label, cause, horizon)

    @pytest.mark.parametrize("fit_label", study.SCENARIOS)
    def test_the_exact_sample_fit_returns_every_truth(self, fit_label: str) -> None:
        result = _exact_fit(fit_label)
        for name, value in study.truths(study.FITS[fit_label][0]).items():
            assert result[name].psi == pytest.approx(value, abs=EXACT), name

    def test_the_support_is_the_law(self) -> None:
        for current in law.LAWS.values():
            assert current.probs.sum() == pytest.approx(1.0, abs=1e-12)
            assert np.all(current.probs > 0.0)
            assert len(set(current.support)) == len(current.support)

    def test_a_large_draw_matches_the_support(self) -> None:
        """Every outcome column's mean on 200,000 draws lies within 4 SE of its exact mean."""
        for current in law.LAWS.values():
            exact = current.exact_frame()
            frame = current.draw(200_000, stream_seed(study.STUDY, "design_draw", current.name, 0))
            assert list(frame.columns) == [name for name in exact.columns if name != "wt"]
            for name in frame.columns:
                observed = frame[name].notna()
                kept = exact[name].notna()
                assert observed.mean() == pytest.approx(
                    exact.loc[kept, "wt"].sum(), abs=4.0 * np.sqrt(0.25 / len(frame))
                ), name
                mean = float(np.average(exact.loc[kept, name], weights=exact.loc[kept, "wt"]))
                values = frame.loc[observed, name].to_numpy(dtype=float)
                spread = values.std() / np.sqrt(values.size)
                assert abs(values.mean() - mean) < 4.0 * spread + 1e-12, name

    def test_the_declared_truths(self) -> None:
        np.testing.assert_allclose(
            [
                properties.TRUTHS[f"{fit}_{arm}"]
                for fit in study.SCENARIOS
                for arm in properties.ARMS
            ],
            [-0.458239, 0.192996, -0.322846, 0.124564, -0.162513, 0.137545],
            atol=5e-6,
        )
        for arm in properties.ARMS:
            assert properties.TRUTHS[f"crossfit_{arm}"] == properties.TRUTHS[f"one_cause_{arm}"]
        for scenario in study.SCENARIOS:
            assert study.TRUTH[scenario] == study.truths(study.FITS[scenario][0])

    def test_the_curves_rise(self) -> None:
        for name, current in law.LAWS.items():
            for regimen in study.maps(name).values():
                for cause in range(1, current.causes + 1):
                    values = [
                        current.truth(regimen, t, cause) for t in range(1, current.n_times + 1)
                    ]
                    assert np.all(np.diff(values) > 0.0)

    def test_the_bounds_never_bind(self) -> None:
        """Positivity over every stratum, level and covariate path.

        The cumulative bound reads :math:`g(a \\mid W) \\prod_k c_k`, whose smallest value is
        above twice the default bound of 0.01 on every law.
        """
        expected = {
            "one_cause": {"g_min": 0.2891, "retained_min": 0.1361, "cumulative_min": 0.0394},
            "competing": {"g_min": 0.2891, "retained_min": 0.2230, "cumulative_min": 0.0644},
            "three_arm": {"g_min": 0.2237, "retained_min": 0.1361, "cumulative_min": 0.0314},
        }
        for name, values in expected.items():
            reading = law.positivity(law.LAWS[name])
            assert reading["cumulative_min"] > 2.0 * DEFAULT_LTMLE_G_BOUNDS[0]
            for key, value in values.items():
                assert reading[key] == pytest.approx(value, abs=1e-4), (name, key)


# ---------------------------------------------------------------------------- bounds


class TestTheEfficiencyBounds:
    @pytest.mark.parametrize("fit_label", study.SCENARIOS)
    def test_the_exact_fit_influence_curve_is_the_efficient_one(self, fit_label: str) -> None:
        """At the true nuisances the fit's influence curve is the efficient influence function.

        On the exact sample each row's curve is the support point's value times its weight over
        the mean weight, to the walk's tolerance.
        """
        current = study.law_of(fit_label)
        result = _exact_fit(fit_label)
        scale = current.probs * current.probs.size
        for label, name in properties.COEFFICIENTS[fit_label]:
            curve = np.asarray(result[name].influence_curve)
            np.testing.assert_allclose(
                curve, properties.target(label).curve(current) * scale, atol=1e-5
            )

    def test_the_efficient_influence_function_is_centred(self) -> None:
        for label in properties.TRUTHS:
            current = study.law_of(properties.FIT_OF[label])
            curve = properties.target(label).curve(current)
            assert float(np.sum(current.probs * curve)) == pytest.approx(0.0, abs=1e-12)

    def test_the_declared_bounds(self) -> None:
        np.testing.assert_allclose(
            [
                properties.EFFICIENCY_SD[f"{fit}_{arm}"]
                for fit in study.SCENARIOS
                for arm in properties.ARMS
            ],
            [1.06334, 0.55052, 1.02828, 0.45688, 0.43240, 0.44021],
            atol=5e-5,
        )

    def test_the_vectorized_curve_is_the_support_curve(self) -> None:
        for fit_label in study.SCENARIOS:
            current = study.law_of(fit_label)
            frame = current.exact_frame().drop(columns="wt")
            for label, _ in properties.COEFFICIENTS[fit_label]:
                aim = properties.target(label)
                np.testing.assert_allclose(
                    current.frame_curves(aim, frame), aim.curve(current), atol=1e-14
                )


# ---------------------------------------------------------------------------- the learners


class TestTheLearners:
    def test_last_l_keeps_the_current_covariate_and_no_l_drops_every_one(self) -> None:
        design = np.array([[0.0, 2.0, 1.0, -1.0, 1.0, 0.0], [1.0, 3.0, -1.0, 1.0, 0.0, 1.0]])
        assert list(law.LastL(law.CellMeans())._kept(design)) == [0, 1, 3, 4, 5]
        assert list(law.NoL(law.CellMeans())._kept(design)) == [0, 1, 4, 5]
        first = np.array([[0.0, 2.0, 1.0], [1.0, 3.0, 0.0]])
        assert list(law.LastL(law.CellMeans())._kept(first)) == [0, 1, 2]

    def test_the_untargeted_pass_of_the_correct_learners_is_the_truth(self) -> None:
        """The hazard and continuation learners are correctly specified on every node.

        With the walks skipped only the initial fits remain, so the plug-in reaches the truth on
        the exact sample only when each node's design keeps ``L_k``: a learner that kept an
        earlier ``L`` column instead would miss it.
        """
        for fit_label in study.SCENARIOS:
            current = study.law_of(fit_label)
            result = _quiet(
                study.untargeted_fit,
                fit_label,
                current.exact_frame(),
                configuration="both_correct",
                weights="wt",
            )
            for label, name in properties.COEFFICIENTS[fit_label]:
                assert result[name].psi == pytest.approx(properties.TRUTHS[label], abs=EXACT)


# ---------------------------------------------------------------------------- robustness


class TestTheRobustnessCells:
    def test_every_positive_limit_is_the_truth(self, exact_fits: dict[str, Any]) -> None:
        for configuration, result in exact_fits.items():
            if configuration == "both_wrong":
                continue
            for label, name in properties.COEFFICIENTS["one_cause"]:
                assert result[name].psi == pytest.approx(properties.TRUTHS[label], abs=EXACT), (
                    configuration,
                    label,
                )

    def test_the_both_wrong_control_is_discriminated(self, exact_fits: dict[str, Any]) -> None:
        result = exact_fits["both_wrong"]
        gap = result[properties.ESTIMAND[CONTRAST]].psi - properties.TRUTHS[CONTRAST]
        n = properties.DOUBLE_ROBUST_N
        assert gap / _scale(CONTRAST, n) == pytest.approx(BOTH_WRONG_GAP, abs=1e-3)
        for spread in (_scale(CONTRAST, n), 1.5 * _scale(CONTRAST, n)):
            probability = sizing.bias_discriminated_probability(
                MARGINS, gap / spread, properties.DOUBLE_ROBUST_REPLICATES
            )
            assert probability >= sizing.discrimination_floor(2)


# ---------------------------------------------------------------------------- wrong learner


PILOT = pd.read_csv(pilot.OUTPUT) if pilot.OUTPUT.exists() else pd.DataFrame()


def _pilot(label: str) -> pd.DataFrame:
    return PILOT.loc[PILOT["label"] == label] if not PILOT.empty else PILOT


CALIBRATION_PILOTED = not PILOT.empty and PILOT["label"].str.startswith("calibration_").any()

#: The pass probability each positive bias arm needs (design review d).
POSITIVE_PASS = 0.99
#: Every positive arm whose verdict reads a bias interval, with its replicates.
POSITIVE_ARMS: dict[str, int] = {
    cell.cell: cell.replicates
    for cell in properties.declared_cells()
    if cell.role == "positive"
    and (
        cell.property in ("double_robustness", "targeting_necessity")
        or (cell.property == "root_n_and_efficiency" and cell.n != min(properties.RATE_SIZES))
    )
}


def _tolerated_bias(replicates: int) -> float:
    """The largest |standardized bias| at which a positive arm passes with ``POSITIVE_PASS``."""
    z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
    return MARGINS.standardized_bias - (z + float(norm.ppf(POSITIVE_PASS))) / np.sqrt(replicates)


def _wrong_learner_errors() -> np.ndarray:
    """The pilot's targeted contrast errors with the wrong learners at n = 2,000.

    They pool ``fits_differ_mechanism_correct``, whose hazard fit is the same wrong-learner fit
    on its own seeds (plan X70 rev 2.7).
    """
    name = properties.ESTIMAND[CONTRAST]
    rows = _pilot("wrong_learner_one_cause")
    paired = _pilot("fits_differ_mechanism_correct")
    if rows.empty or paired.empty:
        pytest.skip("the wrong-learner pilot has not been committed")
    rows = rows.loc[rows["estimand"] == name]
    paired = paired.loc[paired["estimand"] == name]
    assert rows["refusal"].fillna("").eq("").all()
    assert len(rows) == pilot.WRONG_LEARNER_DRAWS
    assert len(paired) == pilot.MECHANISM_DIFFER_DRAWS
    values = np.concatenate(
        [rows["wrong_targeted"].to_numpy(dtype=float), paired["estimate"].to_numpy(dtype=float)]
    )
    return values - properties.TRUTHS[CONTRAST]


class TestTheWrongLearners:
    def test_the_wrong_learner_rule(self) -> None:
        """The plug-in contrast limit lies at least 3 efficiency SEs from the truth at n = 2,000."""
        limit = law.wrong_plug_in_limit(law.ONE_CAUSE, properties.target(CONTRAST))
        bias = (limit - properties.TRUTHS[CONTRAST]) / _scale(CONTRAST, 2_000)
        assert abs(bias) >= 3.0
        assert bias == pytest.approx(WRONG_GAP, abs=1e-3)

    def test_the_estimator_plug_in_is_the_limit(self) -> None:
        """The untargeted backward pass on the exact sample is the limit computed from the law."""
        result = _quiet(
            study.untargeted_fit,
            "one_cause",
            law.ONE_CAUSE.exact_frame(),
            configuration="mechanism_correct",
            weights="wt",
        )
        for label, name in properties.COEFFICIENTS["one_cause"]:
            limit = law.wrong_plug_in_limit(law.ONE_CAUSE, properties.target(label))
            assert result[name].psi == pytest.approx(limit, abs=1e-8)

    def test_the_dropout_is_what_the_wrong_learners_miss(self) -> None:
        """The control: with drop-out free of ``L_k`` the plug-in without ``L`` misses by little.

        The wrong learners' gap comes from the drop-out reading ``L_k``; on the same law with
        that coefficient set to zero, only the non-collapsibility of the logistic hazard over
        ``L`` is left, below one efficiency SE.
        """
        from dataclasses import replace

        free = replace(law.ONE_CAUSE, dropout=replace(law.DROPOUT, l=0.0))
        aim = properties.target(CONTRAST)
        gap = (law.wrong_plug_in_limit(free, aim) - aim.value(free)) / (
            law.efficiency_sd(free, aim) / np.sqrt(2_000)
        )
        assert abs(gap) < 1.0

    def test_the_pilot_plug_in_varies(self) -> None:
        rows = _pilot("wrong_learner_one_cause")
        if rows.empty:
            pytest.skip("the pilot has not been committed")
        rows = rows.loc[rows["estimand"] == properties.ESTIMAND[CONTRAST]]
        assert rows["refusal"].fillna("").eq("").all()
        values = rows["wrong_plug_in"].to_numpy(dtype=float)
        assert np.var(values, ddof=1) > 0.0
        limit = law.wrong_plug_in_limit(law.ONE_CAUSE, properties.target(CONTRAST))
        error = np.std(values, ddof=1) / np.sqrt(values.size)
        assert abs(values.mean() - limit) < 3.0 * error


# ---------------------------------------------------------------------------- sizes


class TestTheSizes:
    def test_the_power_cell_has_power(self) -> None:
        power = design_power(
            properties.TRUTHS[CONTRAST], properties.EFFICIENCY_SD[CONTRAST], properties.POWER_N
        )
        assert power >= 0.80
        assert power_cell_pass_probability(power, properties.POWER_REPLICATES) >= 0.99

    def test_the_derived_controls_leave_the_band(self) -> None:
        replicates = properties.CALIBRATION_REPLICATES
        assert (
            sizing.calibration_control_probability(
                MARGINS, replicates, properties.SHRUNKEN_SE_FACTOR
            )
            > 0.999
        )
        z = float(norm.ppf(0.5 + MARGINS.confidence_level / 2.0))
        noise = np.sqrt(2.0)
        spread = noise / np.sqrt(2.0 * (replicates - 1))
        upper = properties.EFFICIENCY_RATIO_BAND[1]
        assert norm.cdf((noise - z * spread - upper) / spread) > 0.999

    def test_the_untargeted_control_is_discriminated(self) -> None:
        for spread in (1.0, 1.5):
            probability = sizing.bias_discriminated_probability(
                MARGINS, WRONG_GAP / spread, properties.WRONG_LEARNER_REPLICATES
            )
            assert probability >= sizing.discrimination_floor(2)

    @pytest.mark.skipif(CALIBRATION_PILOTED, reason="the committed calibration pilot sizes R")
    def test_the_calibration_replicates_at_a_calibrated_pilot(self) -> None:
        """The arm-pooled rule at a pilot of the declared size that reads a ratio of one."""
        arms = sum(len(properties.COEFFICIENTS[label]) for label in properties.CALIBRATION_FITS)
        assert arms == 8
        replicates, cell, family = mtp_study_tools.smallest_replicates(
            1.0, sizing.CALIBRATION_PILOT_DRAWS, arms
        )
        assert replicates == properties.CALIBRATION_REPLICATES
        assert cell >= mtp_study_tools.CELL_TARGET and family >= mtp_study_tools.FAMILY_TARGET

    @staticmethod
    def _calibration_rows(fit_label: str) -> pd.DataFrame:
        rows = _pilot(f"calibration_{fit_label}")
        if rows.empty:
            pytest.skip(f"the calibration pilot of {fit_label} has not been committed")
        return rows

    def test_the_calibration_replicates_follow_the_pilot(self) -> None:
        ratios = []
        for fit_label in properties.CALIBRATION_FITS:
            rows = self._calibration_rows(fit_label)
            assert rows["refusal"].fillna("").eq("").all()
            for _, name in properties.COEFFICIENTS[fit_label]:
                arm = rows.loc[rows["estimand"] == name]
                assert len(arm) == sizing.CALIBRATION_PILOT_DRAWS
                ratios.append(float(arm["std_error"].mean() / arm["estimate"].std(ddof=1)))
        reading = mtp_study_tools.replicates_from_pilot(ratios, sizing.CALIBRATION_PILOT_DRAWS)
        assert not reading["suspect"]
        assert all(reading["screened"])
        assert reading["R"] == properties.CALIBRATION_REPLICATES

    def test_every_positive_arm_passes_at_its_exact_limit(self) -> None:
        """Each positive bias arm passes with probability at least 0.99 at standardized bias 0."""
        for cell, replicates in POSITIVE_ARMS.items():
            probability = sizing.bias_equivalent_probability(MARGINS, 0.0, replicates)
            assert probability >= POSITIVE_PASS, (cell, probability)
        assert _tolerated_bias(properties.DOUBLE_ROBUST_REPLICATES) == pytest.approx(
            0.1085, abs=1e-3
        )

    def test_the_wrong_learner_arms_are_sized_by_the_pilot_rule(self) -> None:
        """The arms that fit the wrong learners take the pre-committed size (plan X70 rev 2.7).

        The arm's replicates are the larger of
        :data:`~tests.studies.hazard_time_varying_properties.DOUBLE_ROBUST_REPLICATES` and the
        smallest multiple of 400 that passes with :data:`POSITIVE_PASS` at the pilot's
        |standardized bias| plus one of its standard errors, with no cap; a reading that needs
        more than 4,000 stops the declaration for a diagnosis.  The reading pools the
        ``fits_differ_mechanism_correct`` draws, the same hazard fit on its own seeds.
        """
        error = _wrong_learner_errors()
        standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
        needed = int(properties.DOUBLE_ROBUST_REPLICATES)
        while sizing.bias_equivalent_probability(MARGINS, standardized, needed) < POSITIVE_PASS:
            needed += 400
        assert needed <= 4_000, (standardized, needed)
        assert POSITIVE_ARMS[f"{CONTRAST}__mechanism_correct"] == needed
        assert POSITIVE_ARMS[f"{CONTRAST}__targeted"] == needed
        assert needed == properties.WRONG_LEARNER_REPLICATES

    def test_the_positive_arms_follow_the_calibration_pilot(self) -> None:
        """The ``both_correct`` and ``outcome_correct`` arms and the claimed rungs, at the pilot."""
        rows = self._calibration_rows("one_cause")
        for cell_label, name in properties.COEFFICIENTS["one_cause"]:
            arm = rows.loc[rows["estimand"] == name]
            error = (arm["estimate"] - arm["truth"]).to_numpy(dtype=float)
            standardized = abs(error.mean()) / error.std(ddof=1) + 1.0 / np.sqrt(error.size)
            for cell, replicates in POSITIVE_ARMS.items():
                if cell.startswith(f"{cell_label}__") and cell.endswith(
                    ("__both_correct", "__outcome_correct", "__n_2000", "__n_8000")
                ):
                    probability = sizing.bias_equivalent_probability(
                        MARGINS, standardized, replicates
                    )
                    assert probability >= POSITIVE_PASS, (cell, probability)

    def test_the_primary_replicates_follow_the_shared_rule(self) -> None:
        """:func:`~tests.studies.repeated_sizing.primary_replicates` on the calibration pilots."""
        covered: dict[str, int] = {}
        groups = []
        for scenario in study.SCENARIOS:
            rows = self._calibration_rows(scenario)
            assert int(rows["n"].iloc[0]) == study.PRIMARY_N
            names = list(study.ESTIMANDS[scenario])
            hits = rows.pivot(index="replicate", columns="estimand", values="covered")[names]
            covered.update({f"{scenario}/{name}": int(hits[name].sum()) for name in names})
            error = rows.assign(error=rows["estimate"] - rows["truth"]).pivot(
                index="replicate", columns="estimand", values="error"
            )[names]
            standardized = error / error.std(ddof=1)
            groups.append(
                sizing.PilotGroup(
                    standardized.mean().to_numpy(), standardized.cov().to_numpy() / len(hits)
                )
            )
        share = sizing.PRIMARY_CLAUSE_SHARE
        assert (
            sizing.primary_replicates(
                MARGINS,
                covered,
                sizing.CALIBRATION_PILOT_DRAWS,
                implementations=1,
                coverage_pass=share,
                bias_groups=groups,
                bias_pass=share,
            )
            == study.PRIMARY_REPLICATES
        )


# ---------------------------------------------------------------------------- the IC gap


def _ic_gaps(fit_label: str) -> dict[str, tuple[list[float], float]]:
    rows = _pilot("ic_gap")
    if rows.empty:
        pytest.skip("the ic_gap pilot has not been committed")
    small_n, large_n = properties.IC_GAP_SIZES
    rows = rows.loc[rows["fit_label"] == fit_label]
    out: dict[str, tuple[list[float], float]] = {}
    for label, _ in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == label]
        small = arm.loc[arm["n"] == small_n, "estimate"].to_numpy(dtype=float)
        large = arm.loc[arm["n"] == large_n, "estimate"].to_numpy(dtype=float)
        assert small.size == properties.IC_GAP_DRAWS and large.size == 1, (fit_label, label)
        out[label.rsplit("_", 1)[1]] = (list(small), float(large[0]))
    return out


class TestTheInfluenceCurveGap:
    @pytest.mark.parametrize("fit_label", study.CALIBRATION_FITS)
    def test_a_fit_meets_the_declared_bound(self, fit_label: str) -> None:
        small_n, large_n = properties.IC_GAP_SIZES
        for arm, (small, large) in _ic_gaps(fit_label).items():
            median = float(np.median(small))
            assert median <= properties.IC_GAP_BOUND[small_n], (arm, small)
            assert large <= properties.IC_GAP_BOUND[large_n], (arm, large)
            assert large <= properties.IC_GAP_FALL * median, (arm, median, large)

    @pytest.mark.parametrize("fit_label", study.CALIBRATION_FITS)
    def test_a_large_fit_reports_the_bound(self, fit_label: str) -> None:
        """At n = 40,000 each fit's influence-curve SD is within 5% of the bound."""
        rows = _pilot("ic_gap")
        if rows.empty:
            pytest.skip("the ic_gap pilot has not been committed")
        rows = rows.loc[(rows["fit_label"] == fit_label) & (rows["n"] == pilot.BOUND_N)]
        assert len(rows) == len(properties.COEFFICIENTS[fit_label])
        for label, ratio in zip(rows["estimand"], rows["estimate"], strict=True):
            assert ratio == pytest.approx(1.0, abs=0.05), label


# ---------------------------------------------------------------------------- pilot readings


def _paired_ratios(
    rows: pd.DataFrame, fit_label: str, column: str = "iterated_estimate"
) -> dict[str, float]:
    ratios = {}
    for label, name in properties.COEFFICIENTS[fit_label]:
        arm = rows.loc[rows["estimand"] == name]
        difference = arm["estimate"] - arm[column]
        ratios[label] = float(difference.std(ddof=1)) / _scale(label, pilot.PILOT_N)
    return ratios


@pytest.mark.parametrize("fit_label", study.DIFFERENCE_FITS)
def test_the_agreement_pilot_is_reported(fit_label: str) -> None:
    """The ``iterated_mean_difference`` readings, reported with no floor."""
    rows = _pilot(f"fits_differ_{fit_label}")
    if rows.empty:
        pytest.skip("the pilot has not been committed")
    assert rows["refusal"].fillna("").eq("").all()
    assert rows["replicate"].nunique() == pilot.FITS_DIFFER_DRAWS
    decreases = rows.loc[rows["estimand"] == "largest_decrease", "estimate"]
    ratios = _paired_ratios(rows, fit_label)
    ratios |= {
        f"{label} (isotonic)": value
        for label, value in _paired_ratios(rows, fit_label, "isotonic_estimate").items()
    }
    reading = properties.fits_differ_reading(ratios, int((decreases > 0.0).sum()), len(decreases))
    assert all(np.isfinite(value) and value > 0.0 for value in reading["ratios"].values())
    print(reading["text"])


def test_the_agreement_reading_carries_no_floor() -> None:
    reading = properties.fits_differ_reading({"one_cause_contrast": 0.03}, 0, 200)
    assert not {key for key in reading if key.endswith("passes")}
    assert reading["share_interval"] == (0.0, pytest.approx(0.0261, abs=1e-4))
    assert "floor" in reading["text"] and "0.05" not in reading["text"]


# ---------------------------------------------------------------------------- support rows


def _fit_key(fit_label: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The support key of one fit label's fit, read where the estimator classifies it."""
    keys: list[Any] = []
    original = estimator_module.classify_longitudinal_fit

    def spy(estimator: Any, data: Any, **settings: Any) -> Any:
        names = (
            "outcome_kind",
            "cross_fitted",
            "known_factors",
            "has_msm",
            "regimen_kinds",
            "tuned_cap",
            "separable_first",
        )
        key, _ = _support.longitudinal_key(
            estimator, data, **{name: settings[name] for name in names if name in settings}
        )
        keys.append(key)
        return original(estimator, data, **settings)

    monkeypatch.setattr(estimator_module, "classify_longitudinal_fit", spy)
    result = _quiet(study.fit, fit_label, study.law_of(fit_label).draw(600, 1))
    return keys[-1], result


class TestTheSupportRows:
    @pytest.mark.parametrize("row", study.declared_support_rows(), ids=lambda row: row.id)
    def test_each_declared_row_covers_its_fit(
        self, row: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        key, result = _fit_key(study.G63_FITS[row.id], monkeypatch)
        assert key.is_covered_by(row.key), key.differing_axes(row.key)
        assert row.evidence == (study.STUDY.slug,)
        assert set(row.cells) <= set(properties.ESTIMAND)
        labels = {estimate.support.label for estimate in result.estimates.values()}
        assert labels == {"experimental"}

    def test_the_rows_are_not_yet_support_rows(self) -> None:
        ids = {row.id for row in LONGITUDINAL_ROWS}
        assert not ids & set(study.G63_FITS)

    def test_the_key_needs_an_axis_for_covariates_after_baseline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why registration adds a key axis: today a registered row covers these keys.

        ``G31`` and ``G32`` license held hazard fits without covariates after baseline.  Their
        keys cover the in-sample, competing and cross-fitted fits here, which only the X70
        condition keeps ``experimental``; a ``G63.held`` row with the same key would collide.
        """
        covering = {}
        for row_id in ("G63.held.in_sample", "G63.held.competing", "G63.held.cross_fitted"):
            key, _ = _fit_key(study.G63_FITS[row_id], monkeypatch)
            covering[row_id] = {row.id for row in LONGITUDINAL_ROWS if key.is_covered_by(row.key)}
        assert covering == {
            "G63.held.in_sample": {"G31.survival"},
            "G63.held.competing": {"G31.competing"},
            "G63.held.cross_fitted": {"G32.survival"},
        }


# ---------------------------------------------------------------------------- cells and seeds


class TestTheCells:
    def test_the_record_lists_every_study_specific_module(self) -> None:
        from tests.unit.test_study_provenance import _required_from

        required = _required_from(study.STUDY.runner_module, study.STUDY.properties_module)
        assert required <= set(study.STUDY.modules)

    def test_the_record_declares_every_cell_the_module_writes(self) -> None:
        declared = {
            f"{family}/{cell}"
            for family, cells in study.STUDY.property_cells.items()
            for cell in cells
        }
        written = {f"{cell.property}/{cell.cell}" for cell in properties.declared_cells()}
        filled = {
            name
            for key in properties._payloads(properties.FIT_SETS, 1)
            for name in properties.cells_of(key)
        }
        rate = {name for name in declared if name.startswith("root_n_rate/")}
        assert written | rate == declared
        assert filled | rate == declared

    def test_the_study_is_not_yet_registered(self) -> None:
        # Registration comes with the artifacts, the evidence page and the grid row.
        assert study.STUDY.slug not in {record.slug for record in registered()}

    def test_the_seeds_are_distinct(self) -> None:
        from tests.studies import canonical_hazard_msm, canonical_hazard_mtp

        records = {record.slug: record for record in registered()}
        for sibling in (canonical_hazard_msm.STUDY, canonical_hazard_mtp.STUDY):
            records.setdefault(sibling.slug, sibling)
        others = [record for slug, record in records.items() if slug != study.STUDY.slug]
        assert study.SEED not in {record.seed for record in others}
        assert study.RESAMPLING_SEED not in {record.resampling_seed for record in others}
        seeds = [
            replicate_seed(study.STUDY, scenario, r)
            for scenario in study.SCENARIOS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "property_sample", family, stream, r)
            for family, stream, *_ in properties.FIT_SETS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        assert len(set(seeds)) == len(seeds)
        foreign = {
            replicate_seed(record, scenario, r)
            for record in others
            for scenario in record.scenarios
            for r in range(50)
        }
        assert not foreign & set(seeds)
