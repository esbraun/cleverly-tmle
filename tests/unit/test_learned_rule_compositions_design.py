"""The declared constants of the ``learned-rule-compositions`` study, recomputed by quadrature.

Each number here was fixed before any registered run. The checks call no sampler: they read
the population projection of the declared learner and integrate on the RM30 grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.studies import _learned_rule_compositions_law as law
from tests.studies import learned_rule_compositions as study
from tests.studies import learned_rule_compositions_properties as properties
from tests.studies import mtp_study_tools as tools


@pytest.fixture(scope="module")
def rules() -> dict[str, dict[float, np.ndarray]]:
    return {scenario: law.population_rule(scenario) for scenario in law.PRIMARY_SCENARIOS}


def test_the_efficiency_bounds(rules: dict[str, dict[float, np.ndarray]]) -> None:
    for scenario, rule in rules.items():
        _, variance = law.rule_value(rule, scenario)
        if scenario in law.CLUSTER_SIZES:
            size = law.CLUSTER_SIZES[scenario]
            variance += (size - 1) * law.cluster_covariance(rule, scenario)
        assert np.sqrt(variance) == pytest.approx(study.EFFICIENCY_BOUNDS[scenario], abs=2e-6)


def test_the_cluster_correlation_is_declared(rules: dict[str, dict[float, np.ndarray]]) -> None:
    """The intra-cluster correlation of the row curve, on the curve scale."""
    _, variance = law.rule_value(rules["clustered"], "clustered")
    covariance = law.cluster_covariance(rules["clustered"], "clustered")
    assert covariance / variance == pytest.approx(0.144, abs=5e-4)


def test_a_cluster_effect_of_zero_has_no_covariance(
    rules: dict[str, dict[float, np.ndarray]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The witness: the covariance comes from the shared effect, and vanishes without it."""
    monkeypatch.setattr(law, "SIGMA_U", 0.0)
    assert law.cluster_covariance(rules["clustered"], "clustered") == pytest.approx(0.0, abs=1e-15)


def test_the_weighted_limit_rule_differs_from_the_unweighted_one(
    rules: dict[str, dict[float, np.ndarray]], monkeypatch: pytest.MonkeyPatch
) -> None:
    weighted = rules["weighted"]
    monkeypatch.setattr(law, "weight", lambda w1: np.ones_like(np.asarray(w1, dtype=float)))
    unweighted = law.population_rule("weighted")
    share = np.mean([np.mean(weighted[w2] != unweighted[w2]) for w2 in weighted])
    assert share == pytest.approx(0.0169, abs=1e-3)


#: The calibration pilot: 4,000 draws of each calibration positive at its declared size, from
#: ``SeedSequence([20263099, cell, draw])``, a label no declared stream reads. Each value is the
#: mean reported SE over the SD of the error ``estimate - truth``, in the order
#: ``fold_targeted``, ``weighted``, ``clustered``, ``clustered_few``. No draw failed and no
#: solver warned. The pilot script is ``X11-A-study-design.md`` in the plans repository.
CALIBRATION_PILOT = (0.9953, 0.9836, 1.0123, 0.9785)
CALIBRATION_PILOT_DRAWS = 4_000


def _per_arm_pass(replicates: int) -> tuple[float, ...]:
    """Each positive's pass probability at ``replicates``, predictive over its own pilot error."""
    sigma = tools.pilot_sigma(CALIBRATION_PILOT_DRAWS)
    return tuple(
        float(
            np.sum(
                tools._GH_W * tools.cell_pass(ratio + sigma * tools._GH_X, replicates, tools.C_FIT)
            )
        )
        for ratio in CALIBRATION_PILOT
    )


def test_the_calibration_replicates_follow_the_per_arm_rule() -> None:
    """R is the smallest multiple of 400 whose family-wise pass probability reaches 0.95.

    The four positives are four settings, not exchangeable arms of one estimator, so each is
    predicted from its own pilot ratio. The arm-pooled rule of ``mtp_study_tools`` reads their
    spread as between-arm noise and accepts no R up to 40,000 (family-wise 0.936 there).
    """
    family = {r: float(np.prod(_per_arm_pass(r))) for r in range(400, 20_001, 400)}
    first = next(r for r, value in family.items() if value >= tools.FAMILY_TARGET)
    assert first == properties.CALIBRATION_REPLICATES == properties.FEW_CLUSTER_REPLICATES == 8_400
    assert family[8_000] < tools.FAMILY_TARGET
    assert min(_per_arm_pass(8_400)) >= tools.CELL_TARGET
    pooled = tools.replicates_from_pilot(CALIBRATION_PILOT, CALIBRATION_PILOT_DRAWS)
    assert pooled["R"] is None
