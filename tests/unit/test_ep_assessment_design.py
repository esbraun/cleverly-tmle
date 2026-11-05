"""The ``ep-assessment`` declaration: the readings, the control identity, the verdicts and the
pass probabilities.

The study's calibration reading is checked against
:meth:`cleverly.effect_curves.CurveDiagnostics.calibration` on a small fit with cheap learners,
the stretched control against its exact ``1/1.5`` identity, the share bounds against
Clopper-Pearson, and the verdict code against synthetic rows in which each claim and the
control fail when the rows say they should.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from tests.studies import ep_assessment as study
from tests.studies import ep_assessment_properties as properties
from tests.studies import ep_cate
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.seeds import stream_seed


@pytest.fixture(scope="module")
def small_curve() -> object:
    frame, _ = study.draw_from_seed("strong_2k", 300, 11)
    effect = ep_cate.effect_of(study.STRONG, frame)
    from cleverly.bases import Sieve

    sieve = Sieve.default(ep_cate.modifiers(study.STRONG), max_size=3, size=2)
    return effect.estimate(
        "ep",
        sieve=sieve,
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(),
        outcome_by_arm=True,
        n_folds=2,
        q_bounds=ep_cate.q_bounds(study.STRONG),
        random_state=0,
    )


class TestTheReadings:
    def test_the_slope_is_the_shipped_calibrations(self, small_curve: object) -> None:
        diagnostics = small_curve.diagnostics  # type: ignore[attr-defined]
        reading = study.calibration_reading(diagnostics, 3)
        shipped = diagnostics.calibration(n_folds=study.N_FOLDS, random_state=3)
        assert reading["slope"] == pytest.approx(shipped.slope, abs=1e-12)

    def test_the_stretched_slope_is_exactly_the_slope_over_the_stretch(
        self, small_curve: object
    ) -> None:
        reading = study.calibration_reading(small_curve.diagnostics, 3)  # type: ignore[attr-defined]
        assert reading["stretched_slope"] == pytest.approx(
            reading["slope"] / study.STRETCH, rel=1e-10
        )
        # So the control fails the band whenever the claim's upper bound is below 1.275.
        assert study.CALIBRATION_BAND[0] * study.STRETCH == pytest.approx(1.275)

    def test_the_exact_difference_reads_the_true_cate(self, small_curve: object) -> None:
        frame, _ = study.draw_from_seed("strong_2k", 300, 11)
        tau = ep_cate.cate(study.STRONG, frame.loc[:, list(study.STRONG.columns)].to_numpy())
        reading = {"predictions": tau, "constant": np.full_like(tau, tau.mean())}
        # A curve at the truth beats the constant by the CATE's variance.
        assert study.exact_difference(study.STRONG, frame, reading) == pytest.approx(
            -float(np.var(tau))
        )

    def test_the_weak_law_has_the_constant_cate(self) -> None:
        w = np.array([[0.3, -0.5, 0.8], [-0.9, 0.1, 0.2]])
        assert study.WEAK.cate == "constant"
        assert ep_cate.cate(study.WEAK, w) == pytest.approx([ep_cate.WEAK_CATE] * 2)
        assert ep_cate.baseline(study.WEAK, w) == pytest.approx(ep_cate.baseline(study.STRONG, w))
        assert ep_cate.propensity(study.WEAK, w) == pytest.approx(
            ep_cate.propensity(study.STRONG, w)
        )

    def test_the_share_bound_is_clopper_pearson(self) -> None:
        assert study.share_lower(90, 100) == pytest.approx(
            float(stats.beta.ppf(0.01, 90, 11)), rel=1e-12
        )
        assert study.share_lower(0, 100) == 0.0

    def test_the_streams_and_seeds_are_disjoint(self) -> None:
        seeds = {
            stream_seed(study.STUDY, stream, group.label, r)
            for stream in ("property_sample", "assessment_pilot")
            for group in study.GROUPS
            for r in range(3)
        }
        assert len(seeds) == 2 * len(study.GROUPS) * 3
        assert study.SEED not in {ep_cate.SEED, ep_cate.RESAMPLING_SEED}


class TestTheDesignNumbers:
    @pytest.fixture(scope="class")
    def numbers(self) -> dict:
        return study.design_numbers()

    @pytest.mark.parametrize("claim", ["calibration", "control", "stable", "sign"])
    def test_the_claim_reaches_the_target(self, numbers: dict, claim: str) -> None:
        assert numbers[claim] >= study.TARGET_PASS, (claim, numbers)

    def test_the_weak_law_stability_claim_cannot_pass_and_blocks_the_run(
        self, numbers: dict
    ) -> None:
        """The pilot's finding: split-seed refits barely move a curve of noise.

        The weak law's ``relative_sd`` sits near 0.09, so the share above 0.5 is about 0 and
        the plan's claim fails at any R.  The claim is not moved; the run is refused until the
        owner decides (``ep_assessment.BLOCKED``).
        """
        pilot = study.pilot_rows()[("weak_2k", "relative_sd")]
        assert pilot["mean"] + 5 * pilot["sd"] < study.STABILITY_THRESHOLD
        assert numbers["unstable"] < 1e-6
        assert study.BLOCKED


def synthetic_rows(
    *, slope: float = 1.0, stable: float = 1.0, unstable: float = 1.0, sign: float = 1.0
) -> pd.DataFrame:
    """Rows of every declared cell; the indicator cells are 1 with the given probabilities."""
    rng = np.random.default_rng(12)
    rows = []
    for family, suffix, role, group, requested in study.CELLS:
        cell = study.cell_name(group, suffix)
        for r in range(requested):
            if suffix == "slope":
                value = slope + 0.1 * rng.standard_normal()
            elif suffix == "stretched_slope":
                value = slope / study.STRETCH + 0.07 * rng.standard_normal()
            elif suffix in ("stable", "unstable", "sign_agrees"):
                share = {"stable": stable, "unstable": unstable, "sign_agrees": sign}[suffix]
                value = float(rng.uniform() < share)
            else:
                value = rng.standard_normal()
            rows.append(
                {
                    "property": family,
                    "cell": cell,
                    "role": property_verdicts.DIAGNOSTIC_ROLE if role == study.DIAGNOSTIC else role,
                    "replicate": r,
                    "n": study.N,
                    "requested_replicates": requested,
                    "failed_replicates": 0,
                    "truth": 0.0,
                    "estimate": value,
                    "std_error": 1.0,
                    "covered": 1,
                    "rejected": 0,
                }
            )
    return pd.DataFrame(rows)


class TestTheVerdicts:
    def failed(self, **kwargs: float) -> set[str]:
        summary = properties.summarize_properties(synthetic_rows(**kwargs))
        return set(summary.loc[~summary["passed"].astype(bool), "cell"])

    def test_good_rows_pass(self) -> None:
        assert self.failed() == set()

    def test_a_shrunken_curve_fails_and_its_stretch_detects_nothing(self) -> None:
        # A slope of 1.4: the claim fails, and the stretched slope (0.93) no longer lies below
        # the band, so the control fails too.
        assert self.failed(slope=1.4) == {"strong_2k__slope", "strong_2k__stretched_slope"}

    def test_a_flat_curve_fails_the_claim_while_the_control_detects(self) -> None:
        assert self.failed(slope=0.7) == {"strong_2k__slope"}

    @pytest.mark.parametrize(
        ("setting", "cell"),
        [
            ({"stable": 0.7}, "strong_2k__stable"),
            ({"unstable": 0.7}, "weak_2k__unstable"),
            ({"sign": 0.7}, "strong_2k__sign_agrees"),
        ],
    )
    def test_a_low_share_fails(self, setting: dict, cell: str) -> None:
        assert self.failed(**setting) == {cell}
