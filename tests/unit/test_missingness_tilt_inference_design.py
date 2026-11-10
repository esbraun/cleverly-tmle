"""The ``missingness-tilt-inference`` declaration: truths, bounds, sizes, seeds and rules.

The truths are recomputed by adaptive quadrature and by simulating the full-data law the tilt
describes; the efficient spread against a Monte Carlo of its closed form; the mechanism bounds on
the central 99.9% of the covariate; the replicate counts against the committed pilot; and the
instrument rule of each scenario against the fit it declares.  Nothing here reads a study run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.special import expit, logit
from scipy.stats import norm

from cleverly.validation import _interpretation
from tests.studies import missingness_tilt_inference as study
from tests.studies import missingness_tilt_law as law
from tests.studies import missingness_tilt_pilot as pilot
from tests.studies import missingness_tilt_properties as properties
from tests.studies.evidence.registry import registered
from tests.studies.evidence.seeds import replicate_seed, stream_seed

pytestmark = pytest.mark.xdist_group("missingness_tilt_design")

GAMMAS = (-1.0, 0.0, 1.0, 2.0)


def _quad(function: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """:math:`E f(W)` by adaptive quadrature in :math:`W_1` against the normal density."""
    total = 0.0
    for w2 in (0.0, 1.0):

        def integrand(w1: float, w2: float = w2) -> float:
            value = function(np.array([w1]), np.array([w2]))
            return float(value[0]) * float(norm.pdf(w1))

        total += 0.5 * integrate.quad(integrand, -12.0, 12.0, epsabs=1e-13, epsrel=1e-12)[0]
    return total


class TestTheTruths:
    @pytest.mark.parametrize("gamma", GAMMAS)
    def test_every_truth_is_a_second_quadratures(self, gamma: float) -> None:
        def arm(a: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
            def m(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
                q = law.outcome(a, w1, w2)
                pi = law.response(a, w1, w2)
                return pi * q + (1 - pi) * expit(logit(q) + gamma)

            return m

        m1, m0 = arm(1), arm(0)
        ey1 = _quad(m1)
        ey0 = _quad(m0)
        treated = _quad(law.propensity)
        att = _quad(lambda w1, w2: law.propensity(w1, w2) * (m1(w1, w2) - m0(w1, w2))) / treated
        ey_obs = _quad(
            lambda w1, w2: (
                law.propensity(w1, w2) * m1(w1, w2) + (1 - law.propensity(w1, w2)) * m0(w1, w2)
            )
        )
        expected = {
            "ey1": ey1,
            "ey0": ey0,
            "ate": ey1 - ey0,
            "att": att,
            "rr": ey1 / ey0,
            "ey_obs": ey_obs,
            "par": ey_obs - ey0,
        }
        truths = law.truths(gamma)
        for name, value in expected.items():
            assert truths[name] == pytest.approx(value, abs=1e-10), name

    def test_the_truths_are_the_full_data_means_the_tilt_describes(self) -> None:
        """Simulate the full-data law: an unobserved outcome is Bernoulli(expit(logit Q + gamma))."""
        rng = np.random.default_rng(stream_seed(study.STUDY, "design", "full_data"))
        n = 2_000_000
        w1 = rng.standard_normal(n)
        w2 = rng.binomial(1, 0.5, n).astype(float)
        a = rng.binomial(1, law.propensity(w1, w2))
        gamma = 2.0
        means = {}
        for arm in (0, 1):
            q = law.outcome(arm, w1, w2)
            observed = rng.binomial(1, law.response(arm, w1, w2)) == 1
            p = np.where(observed, q, expit(logit(q) + gamma))
            means[arm] = rng.binomial(1, p)
        truths = law.truths(gamma)
        tolerance = 5.0 / np.sqrt(n)
        assert np.mean(means[1]) == pytest.approx(truths["ey1"], abs=tolerance)
        assert np.mean(means[0]) == pytest.approx(truths["ey0"], abs=tolerance)
        treated = a == 1
        att = np.mean(means[1][treated] - means[0][treated])
        assert att == pytest.approx(truths["att"], abs=2 * tolerance / np.sqrt(np.mean(treated)))
        observed_arm = np.where(treated, means[1], means[0])
        assert np.mean(observed_arm) == pytest.approx(truths["ey_obs"], abs=tolerance)

    def test_the_tilt_moves_every_target(self) -> None:
        """A nonzero witness: the gamma = 2 truths sit away from the MAR ones."""
        mar, tilted = law.truths(0.0), law.truths(2.0)
        for name in law.TARGETS:
            assert abs(tilted[name] - mar[name]) > 0.01, name

    def test_the_study_truths_are_the_laws(self) -> None:
        for scenario, spec in study.SCENARIOS.items():
            assert set(study.TRUTH[scenario]) == set(study.ESTIMANDS[scenario])
            for gamma in study.GAMMAS:
                for target in spec.targets:
                    name = study.estimand_name(target, gamma)
                    assert study.TRUTH[scenario][name] == law.truths(gamma)[target]

    def test_the_weights_are_independent_of_the_data(self) -> None:
        frame = law.sample(200_000, stream_seed(study.STUDY, "design", "weights"), weighted=True)
        low, high = law.WEIGHT_RANGE
        assert frame["wt"].between(low, high).all()
        for column in ("W1", "A", "Delta"):
            assert abs(np.corrcoef(frame["wt"], frame[column])[0, 1]) < 0.01


class TestTheMechanisms:
    def test_the_propensity_and_response_bounds(self) -> None:
        low, high = (float(x) for x in norm.ppf([0.0005, 0.9995]))
        w1 = np.linspace(low, high, 2001)
        g = np.concatenate([law.propensity(w1, np.full_like(w1, w2)) for w2 in (0.0, 1.0)])
        pi = np.concatenate(
            [law.response(a, w1, np.full_like(w1, w2)) for a in (0, 1) for w2 in (0.0, 1.0)]
        )
        assert 0.12 <= g.min() < g.max() <= 0.86
        assert 0.49 <= pi.min() < pi.max() <= 0.95

    def test_the_sample_reads_its_mechanisms(self) -> None:
        frame = law.sample(400_000, stream_seed(study.STUDY, "design", "sample"))
        w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
        a = frame["A"].to_numpy()
        assert frame["A"].mean() == pytest.approx(np.mean(law.propensity(w1, w2)), abs=0.004)
        assert frame["Delta"].mean() == pytest.approx(np.mean(law.response(a, w1, w2)), abs=0.004)
        observed = frame["Delta"] == 1
        assert frame.loc[~observed, "Y"].isna().all()
        assert frame.loc[observed, "Y"].mean() == pytest.approx(
            np.mean(law.outcome(a, w1, w2)[observed.to_numpy()]), abs=0.004
        )


class TestTheEfficientSpread:
    def test_the_spread_is_a_monte_carlo_of_the_closed_form(self) -> None:
        gamma = 2.0
        frame = law.sample(2_000_000, stream_seed(study.STUDY, "design", "efficiency"))
        w1, w2 = frame["W1"].to_numpy(), frame["W2"].to_numpy()
        a = frame["A"].to_numpy()
        delta = frame["Delta"].to_numpy(dtype=float)
        y = np.nan_to_num(frame["Y"].to_numpy(dtype=float))
        curve = law.tilted_arm(1, gamma, w1, w2) - law.tilted_arm(0, gamma, w1, w2)
        curve = curve - law.truths(gamma)["ate"]
        for arm, sign in ((1, 1.0), (0, -1.0)):
            q = law.outcome(arm, w1, w2)
            pi = law.response(arm, w1, w2)
            h = expit(logit(q) + gamma)
            kappa = pi + (1 - pi) * h * (1 - h) / (q * (1 - q))
            g = law.propensity(w1, w2) if arm == 1 else 1 - law.propensity(w1, w2)
            block = delta * kappa * (y - q) / pi + (q - h) * (delta - pi)
            curve = curve + sign * (a == arm) / g * block
        assert abs(np.mean(curve)) < 5 * np.std(curve) / np.sqrt(len(curve))
        assert np.std(curve) == pytest.approx(law.efficiency_sd(gamma), rel=0.005)

    def test_the_mar_spread_falls_below_the_tilted_one(self) -> None:
        """The A1-era control separates only if the MAR SD sits well below the tilted SD."""
        assert law.efficiency_sd(0.0) / law.efficiency_sd(2.0) < 0.85


class TestTheSizes:
    def test_the_property_sizes_are_the_pilots(self) -> None:
        committed = pd.read_csv(pilot.OUTPUT)
        assert pilot.calibration_size(committed) == properties.CALIBRATION_REPLICATES
        assert pilot.necessity_size(committed) == properties.NECESSITY_REPLICATES

    def test_the_primary_size_is_the_pilots(self) -> None:
        committed = pd.read_csv(pilot.OUTPUT)
        full = [s for s in study.SCENARIOS if (committed["label"] == s).any()]
        if not full:
            pytest.skip(
                "draft declaration: only the local pilot is committed, whose 200 draws read "
                "no budget; the runner's full pilot sets PRIMARY_REPLICATES"
            )
        assert pilot.primary_size(committed) == study.PRIMARY_REPLICATES

    def test_the_pilot_lost_no_fit(self) -> None:
        committed = pd.read_csv(pilot.OUTPUT)
        assert committed["refusal"].isna().all() or (committed["refusal"].fillna("") == "").all()

    def test_the_local_pilot_is_what_the_module_declares(self) -> None:
        committed = pd.read_csv(pilot.OUTPUT)
        for label in pilot.LOCAL_LABELS:
            rows = committed.loc[committed["label"] == label]
            assert rows["replicate"].nunique() == pilot.PILOTS[label][2], label


class TestTheRecord:
    def test_the_seeds_are_unused_by_any_other_study(self) -> None:
        others = [record for record in registered() if record.slug != study.STUDY.slug]
        used = {record.seed for record in others} | {
            record.resampling_seed for record in others if record.resampling_seed is not None
        }
        assert study.SEED not in used
        assert study.RESAMPLING_SEED not in used
        assert study.SEED != study.RESAMPLING_SEED

    def test_the_sample_seeds_are_distinct(self) -> None:
        seeds = [
            replicate_seed(study.STUDY, scenario, r)
            for scenario in study.SCENARIOS
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "property_sample", family, properties.LABEL, r)
            for family in (properties.CALIBRATION, properties.NECESSITY)
            for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "pilot", label, r) for label in pilot.PILOTS for r in range(50)
        ]
        seeds += [
            stream_seed(study.STUDY, "design", label)
            for label in ("full_data", "weights", "sample", "efficiency")
        ]
        assert len(set(seeds)) == len(seeds)

    def test_the_record_declares_the_scenario_sizes(self) -> None:
        assert study.STUDY.n_of(study.LADDER) == study.LADDER_N
        assert all(
            study.STUDY.n_of(s) == study.PRIMARY_N for s in study.SCENARIOS if s != study.LADDER
        )
        assert study.PRIMARY_N >= 1_000

    def test_every_property_cell_is_declared(self) -> None:
        declared = {(family, cell) for family, cell, _, _ in properties.declared_truths()}
        published = {
            (family, cell) for family, cells in study.PROPERTY_CELLS.items() for cell in cells
        }
        assert declared == published


#: The instrument rule each scenario's fit computes, by target kind (see the study docstring).
RULES = {
    study.IN_SAMPLE: {
        "ey1": "level",
        "ey0": "level",
        "ate": "contrast",
        "att": "conditional",
        "rr": "ratio",
    },
    study.CROSS_FIT: {
        "ey1": "level+cross_fit",
        "ate": "contrast+cross_fit",
        "att": "conditional+cross_fit",
        "rr": "ratio+cross_fit",
    },
    study.CV_EVALUATION: {
        "ey1": "level+cv_evaluation",
        "ate": "contrast+cv_evaluation",
        "att": "conditional+cv_evaluation",
    },
    study.WEIGHTED: {
        "ey1": "level+weighted",
        "ate": "contrast+weighted",
        "att": "conditional+weighted",
        "rr": "ratio+weighted",
    },
    study.NATURAL_COURSE: {"ey_obs": "natural_course", "ey0": "level", "par": "par"},
    study.FLEXIBLE_Q: {"ate": "contrast+cross_fit", "rr": "ratio+cross_fit"},
}


@pytest.mark.parametrize("scenario", list(RULES))
def test_each_scenario_reads_the_rule_it_licenses(scenario: str) -> None:
    """The rule ``_tilt_reading`` hands the instrument for each target of the scenario's fit."""
    frame, _ = study.draw_from_seed(
        scenario, 400, stream_seed(study.STUDY, "design", "rules", scenario)
    )
    result = study.fit(frame, scenario)
    seen: list[tuple[str, str]] = []
    original = _interpretation.resolve

    def capture(instrument: str, rule: str, fit: Any) -> Any:
        seen.append((instrument, rule))
        return original(instrument, rule, fit)

    targets = study.SCENARIOS[scenario].targets
    with mock.patch.object(_interpretation, "resolve", capture):
        tilt = study.tilt(result, targets, (1.0,))
    assert [name for name, _ in tilt.readings] == list(targets)
    assert {instrument for instrument, _ in seen} == {"missingness_tilt_interval"}
    rules = dict(zip(targets, (rule for _, rule in seen), strict=True))
    for target, rule in RULES[scenario].items():
        assert rules[target] == rule, target
