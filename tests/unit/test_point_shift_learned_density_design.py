"""The declared constants of ``point-shift-learned-density`` (X121 part b2), recomputed.

Each number here was fixed before any declared run. The closed forms are recomputed and checked
against an independent Monte Carlo; the sizing figures are computed, never copied. The checks
that read the sizing pilot (``tests/studies/point_shift_learned_density_pilot.csv``) skip until
the pilot has run. The declared seeds are only computed, never drawn from.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from cleverly._support import density_learner_family, point_key, registered_rows
from cleverly.data.causal_data import CausalData
from cleverly.learners.density import bin_edges
from tests.studies import _point_shift_learned_density_law as law
from tests.studies import mtp_study_tools as tools
from tests.studies import point_shift_learned_density as study
from tests.studies import point_shift_learned_density_pilot as pilot
from tests.studies import point_shift_learned_density_properties as properties
from tests.studies.canonical_shift_policies import OracleShiftDensity
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("point_shift_learned_density_design")

#: The page probe's seeds (``sweep_dose.py 9000 120``), which the study must not reuse.
PROBE_SEEDS = set(range(9_000, 9_120))
#: The record seeds a smoke drew from before they were rotated (the study docstring).
SMOKED_SEEDS = {20261211, 20261212}


def _monte_carlo(n: int = 2_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(n, 3))
    dose = np.asarray(law.DGP.dose_mean(latent)) + rng.normal(size=n)
    noise = law.DGP.noise_scale * rng.normal(size=n)
    return latent, dose, noise


# ----------------------------------------------------------------------------- the law


def test_the_truths_match_the_closed_form_of_the_uncapped_shifts() -> None:
    # E[0.5 d + 0.25 ((A + d)^2 - A^2)] with E[A] = 2.
    for tag, delta in (("uncapped", 0.5), ("one", 1.0)):
        assert law.TRUTH[tag] == pytest.approx(
            0.5 * delta + 0.25 * (4 * delta + delta**2), abs=1e-6
        )
    latent, dose, _ = _monte_carlo()
    shifted = np.where(dose + 0.5 > 5.0, dose, dose + 0.5)
    effect = law.DGP.outcome_mean(latent, shifted) - law.DGP.outcome_mean(latent, dose)
    assert law.TRUTH["capped"] == pytest.approx(float(np.mean(effect)), abs=2e-3)
    # The means: E[1 + 0.5 a + 0.25 a^2] at A + d, with E[A] = 2 and Var A = 1.58.
    for tag, delta in (("current", 0.0), ("uncapped", 0.5), ("one", 1.0)):
        mean = 2.0 + delta
        expected = 1.0 + 0.5 * mean + 0.25 * (mean**2 + 1.58)
        assert law.TRUTH[f"ey_{tag}"] == pytest.approx(expected, abs=1e-5)
    for tag in law.CONTRASTS:
        difference = law.TRUTH[f"ey_{tag}"] - law.TRUTH["ey_current"]
        assert law.TRUTH[tag] == pytest.approx(difference, abs=1e-12)


def test_the_outcome_range_is_the_law_tail_quantiles_rounded_outward() -> None:
    low, high = law.outcome_tail_quantiles()
    assert low == pytest.approx(-7.8878, abs=1e-4) and high == pytest.approx(35.9063, abs=1e-4)
    assert (np.floor(low * 100) / 100, np.ceil(high * 100) / 100) == law.Q_BOUNDS
    # The one-integral CDF agrees with a direct draw at a quantile Monte Carlo can see.
    latent, dose, noise = _monte_carlo()
    outcome = law.DGP.outcome_mean(latent, dose) + noise
    for q in (1e-3, 1 - 1e-3):
        y = float(np.quantile(outcome, q))
        assert law.outcome_cdf(y) == pytest.approx(q, rel=0.05, abs=1e-4)


def test_the_efficiency_bounds_and_ratio_sds() -> None:
    bounds = law.efficiency_bounds()
    ratio_sd = law.true_ratio_sd()
    for tag in law.CONTRASTS:
        assert bounds[tag] == pytest.approx(study.TAG_BOUNDS[tag], abs=2e-6)
        assert ratio_sd[tag] == pytest.approx(study.TRUE_RATIO_SD[tag], abs=2e-6)
    # An independent draw of the efficient influence curve itself.
    latent, dose, noise = _monte_carlo()
    natural = law.DGP.outcome_mean(latent, dose)
    for tag in law.CONTRASTS:
        delta, cap = law.SHIFT_OF[tag]
        shifted = law.DGP.outcome_mean(latent, law.DGP.shifted(dose, delta, cap))
        curve = shifted - natural + (law.true_ratio(tag, dose, latent) - 1.0) * noise
        assert float(np.std(curve)) == pytest.approx(bounds[tag], rel=0.01)


def test_the_true_ratio_integrates_to_one() -> None:
    latent, dose, _ = _monte_carlo()
    for tag in law.CONTRASTS:
        assert float(np.mean(law.true_ratio(tag, dose, latent))) == pytest.approx(1.0, abs=5e-3)


# ---------------------------------------------------------------------------- the fits


def test_the_fits_are_the_declared_ones() -> None:
    frame = law.draw(200, 1)
    learned = law.estimator("in_sample_learned", frame, 5)
    crossed = law.estimator("crossfit_learned", frame, 5)
    oracle = law.estimator("in_sample_oracle", frame, 5)
    for estimator in (learned, crossed, oracle):
        assert estimator.density_bins == law.DENSITY_BINS and estimator.ratio == "density"
        assert [p.name for p in estimator.policies] == [name for *_, name in law.POLICIES]
    assert not learned.cross_fit and not oracle.cross_fit
    assert crossed.cross_fit and crossed.n_folds == 3 and crossed.q_bounds == law.Q_BOUNDS
    booster = learned.treatment_learner
    assert booster is not None
    params = booster.get_params()
    assert params["random_state"] == 5
    assert {k: params[k] for k in law.BOOSTER} == law.BOOSTER
    assert isinstance(oracle.treatment_learner, OracleShiftDensity)
    assert density_learner_family(learned.treatment_learner) == "adaptive"
    assert density_learner_family(oracle.treatment_learner) == "known"


def _key(estimator: Any, frame: pd.DataFrame) -> tuple[Any, Any]:
    data = CausalData.from_frame(
        frame,
        outcome="Y",
        treatment="A",
        covariates=list(law.DGP.covariate_names),
        treatment_kind="continuous",
    )
    return point_key(estimator, data, ["ate_policy", "ey_policy"])


def test_the_declared_row_matches_the_crossfit_arm_alone() -> None:
    frame = law.draw(400, 3)
    row = study.support_row()
    assert row.matches(*_key(law.estimator("crossfit_learned", frame, 7), frame))
    for arm in ("in_sample_learned", "in_sample_oracle"):
        assert not row.matches(*_key(law.estimator(arm, frame, 7), frame)), arm
    edges = tuple(float(v) for v in bin_edges(frame["A"].to_numpy(dtype=float), 40))
    for learner in (None, OracleShiftDensity(law.DGP, edges)):
        other = law.estimator("crossfit_learned", frame, 7, treatment_learner=learner)
        assert not row.matches(*_key(other, frame)), learner


def test_the_cells() -> None:
    assert study.STUDY.scenarios == {study.SCENARIO: tuple(study.LABELS.values())}
    assert len(study.LABELS) == 21
    assert set(study.PROPERTY_CELLS) == {"interval_calibration"}
    assert len(study.PROPERTY_CELLS["interval_calibration"]) == 9
    assert set(study.EFFICIENCY_BOUNDS) == set(study.CALIBRATION_LABELS)
    assert study.STUDY.publication_policy == "reporting"
    assert not study.STUDY.truth_varies_by_replicate


# --------------------------------------------------------------------------- the sizes


def test_the_primary_count() -> None:
    assert study.PRIMARY_REPLICATES == 1_200
    assert study.PRIMARY_REPLICATES % tools.R_STEP == 0


def test_the_calibration_count_at_an_ideal_pilot() -> None:
    sized = tools.per_arm_replicates([1.0, 1.0, 1.0], pilot.PILOT_DRAWS)
    assert sized["R"] == 4_800 <= properties.CALIBRATION_REPLICATES
    # A known ratio of 1 (an infinite pilot) needs 3,200: the floor any pilot can reach.
    assert tools.smallest_replicates(1.0, 10**9, 3, tau=0.0)[0] == 3_200


def test_the_calibration_cap() -> None:
    assert properties.calibration_cap(tools.C_FIT) == properties.CALIBRATION_REPLICATES_CAP


def test_an_unsizable_heavy_tailed_pilot_runs_at_the_ceiling() -> None:
    sized = properties.calibration_count([1.0, 1.0, 1.06], [1.5, 1.5, 1.5])
    assert sized["declared"] == properties.CALIBRATION_REPLICATES_CEILING


def test_heavier_tails_raise_the_count() -> None:
    sized = properties.calibration_count([1.0, 1.0, 1.0], [1.25, 1.0, 1.0])
    assert sized["c"] == 1.25 and sized["rule"] > 4_800
    assert sized["declared"] == min(sized["rule"], sized["cap"])


def test_the_outcome_range_holds_over_the_budget() -> None:
    fits = study.PRIMARY_REPLICATES + pilot.PILOT_DRAWS + properties.CALIBRATION_REPLICATES_CAP
    assert 2 * law.N * fits * law.TAIL_PROBABILITY < 0.015


def test_the_controls_have_power() -> None:
    low, _ = study.STUDY.margins.calibration_se_ratio
    # The shrunken control moves a calibrated ratio to 0.70.
    assert low - 0.15 > properties.SHRUNKEN_SE_FACTOR
    # The noise control adds one efficient SD; the arm's own SD is at least that, so the
    # empirical SD of a calibrated cell grows by at least sqrt(1 + 1 / 2) when its SD is at most
    # sqrt(2) efficient SDs, and the ratio falls below 0.82.
    assert 1 / np.sqrt(1.5) < low - 0.1


# --------------------------------------------------------------------------- the pilot


def _pilot() -> pd.DataFrame:
    if not pilot.PILOT_CSV.exists():
        pytest.skip("the sizing pilot has not been run")
    return pd.read_csv(pilot.PILOT_CSV).set_index("contrast")


def _contrasts() -> pd.DataFrame:
    return _pilot().loc[list(law.CONTRASTS)]


def test_a_resumed_pilot_skips_a_cut_line(tmp_path: Any) -> None:
    path = tmp_path / "pilot_0_9.jsonl"
    path.write_text(json.dumps({"replicate": 0}) + '\n{"replicate": 1, "est')
    assert set(pilot.read_raw(tmp_path)) == {0}


def test_the_pilot_covers_every_parameter() -> None:
    frame = _pilot()
    assert set(frame.index) == set(law.PARAMETERS)
    assert (frame["draws"] == pilot.PILOT_DRAWS).all()


def test_the_calibration_count_meets_the_pilot_rule() -> None:
    """The declared count is the rule at ``max(C_FIT, pilot c)``, capped; the run proceeds at it."""
    frame = _contrasts()
    sized = properties.calibration_count(frame["se_ratio"].tolist(), frame["implied_c"].tolist())
    assert sized["declared"] == properties.CALIBRATION_REPLICATES, (
        f"set CALIBRATION_REPLICATES to {sized['declared']} before the declared run ({sized})"
    )


def test_the_noise_control_has_power_at_the_pilot_spread() -> None:
    """At the arm's measured SD, in efficient units ``k``, the noisy SE ratio is about
    ``k / sqrt(k^2 + 1)`` times the cell's own pilot ratio; its 99% upper bound clears 0.93 with
    probability at least 0.99 at the declared count and the pilot's spread constant."""
    frame = _contrasts()
    low, _ = study.STUDY.margins.calibration_se_ratio
    c = max(tools.C_FIT, *frame["implied_c"])
    sigma = c / np.sqrt(2.0 * properties.CALIBRATION_REPLICATES)
    for tag, row in frame.iterrows():
        k = float(row["efficiency_k"])
        noisy = float(row["se_ratio"]) * k / np.sqrt(k**2 + 1.0)
        power = float(norm.cdf((low - tools.Z * sigma - noisy) / sigma))
        assert power >= 0.99, (tag, k, power)


def test_the_bias_legs_are_powered_at_the_pilot_bias() -> None:
    frame = _pilot()
    for tag, row in frame.iterrows():
        power = tools.equivalence_power(
            abs(float(row["standardized_bias"])), study.PRIMARY_REPLICATES
        )
        assert power >= 0.95, (tag, power)


# --------------------------------------------------------------------------- the seeds


def _others() -> list[Any]:
    return [record for record in registered() if record.slug != study.STUDY.slug]


def test_the_record_seeds_are_unused() -> None:
    others = {record.seed for record in _others()} | {
        record.resampling_seed for record in _others() if record.resampling_seed is not None
    }
    declared = {study.SEED, study.RESAMPLING_SEED}
    assert not declared & (others | PROBE_SEEDS | SMOKED_SEEDS)
    assert pilot.PILOT_SEED not in others | declared | PROBE_SEEDS
    assert pilot.PILOT_SEED + 1 not in others | declared | PROBE_SEEDS


def test_no_drawn_seed_is_a_probe_seed_or_repeats() -> None:
    primary = {
        replicate_seed(study.STUDY, study.SCENARIO, r) for r in range(study.PRIMARY_REPLICATES)
    }
    calibration = {
        stream_seed(study.STUDY, "property_sample", "interval_calibration", properties.LABEL, r)
        for r in range(properties.CALIBRATION_REPLICATES_CAP)
    }
    piloted = {pilot.sample_seed(r) for r in range(pilot.PILOT_DRAWS)}
    primary_fits = {
        stream_seed(study.STUDY, "fit_seed", "primary", study.SCENARIO, r)
        for r in range(study.PRIMARY_REPLICATES)
    }
    calibration_fits = {
        stream_seed(study.STUDY, "fit_seed", "interval_calibration", properties.LABEL, r)
        for r in range(properties.CALIBRATION_REPLICATES_CAP)
    }
    pilot_fits = {pilot.fit_seed(r) for r in range(pilot.PILOT_DRAWS)}
    # Sample seeds draw data and fit seeds seed the booster: distinct roles, so each role's
    # seeds must not repeat, and no seed of either role may be a probe seed.
    samples = (primary, calibration, piloted)
    seeded = (primary_fits, calibration_fits, pilot_fits)
    for group in (samples, seeded):
        assert sum(len(s) for s in group) == len(set().union(*group))
    union = set().union(*samples, *seeded)
    assert not union & PROBE_SEEDS


def test_the_named_streams_are_distinct_from_every_registered_study() -> None:
    labels = [("interval_calibration", "correctly_specified"), ("efficiency_noise", "x")]
    records = [*_others(), study.STUDY]
    seeds = [stream_seed(record, *label) for record in records for label in labels]
    assert len(set(seeds)) == len(seeds)


def test_the_study_is_registered_after_its_declared_run() -> None:
    if not (study.STUDY.artifacts / "manifest.json").exists():
        pytest.skip("the declared run has not been made")
    assert study.STUDY.slug in {record.slug for record in registered()}


def test_the_registered_row_is_the_declared_row() -> None:
    """The passing reading added :func:`support_row` verbatim to the shipped rows."""
    rows = {row.id: row for row in registered_rows()}
    assert rows["T-point-shift-learned-density"] == study.support_row()


def test_the_declared_reading_holds() -> None:
    """The committed verdicts meet the reading that licenses the row (the study docstring)."""
    performance = pd.read_csv(study.STUDY.artifacts / "performance-tests.csv")
    gated = performance[performance["estimand"].str.startswith(f"{study.GATED_ARM}: ")]
    assert len(gated) == len(law.PARAMETERS)
    assert gated["passed"].all()
    properties_table = pd.read_csv(study.STUDY.artifacts / "properties.csv")
    assert set(properties_table["cell"]) == set(study.PROPERTY_CELLS["interval_calibration"])
    assert properties_table["passed"].all()


def test_an_unmeasured_density_setting_or_bin_count_stays_out_of_the_row() -> None:
    """The probe's cross-fitted default booster, and the measured booster on other bins, miss."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    frame = law.draw(400, 3)
    row = study.support_row()
    others = {
        "default booster": {"treatment_learner": HistGradientBoostingClassifier(random_state=7)},
        "deeper booster": {
            "treatment_learner": HistGradientBoostingClassifier(
                **{**law.BOOSTER, "max_depth": 3}, random_state=7
            )
        },
        "other bins": {"density_bins": 20},
    }
    for name, override in others.items():
        other = law.estimator("crossfit_learned", frame, 7, **override)
        assert not row.matches(*_key(other, frame)), name
    seeded = law.estimator("crossfit_learned", frame, 123)
    assert row.matches(*_key(seeded, frame)), "the seed is not a setting"
