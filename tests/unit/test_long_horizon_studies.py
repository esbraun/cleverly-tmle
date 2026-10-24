"""The three long-horizon study records and their primary drivers, without a regeneration.

Each record declares its comparator as pending, a size per scenario and the truths of the oracle.
Each driver returns the published-row contract at a smoke size, and each declared reference runner
is a stub that stops.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests import discrete_law_longitudinal_horizon as H
from tests.studies import (
    canonical_long_horizon_categorical_ltmle,
    canonical_long_horizon_ltmle,
    canonical_long_horizon_sdr,
)
from tests.studies.evidence.registry import ROOT, registered
from tests.studies.evidence.schema import INFERENCE_SCALES, REPLICATE_COLUMNS
from tests.studies.long_horizon_primary import contrast_name, mean_name, truths

STUDIES = (
    canonical_long_horizon_ltmle,
    canonical_long_horizon_categorical_ltmle,
    canonical_long_horizon_sdr,
)
PENDING = {
    "long-horizon-ltmle": "ltmle",
    "long-horizon-categorical-ltmle": "lmtp",
    "long-horizon-sdr": "lmtp",
}


@pytest.mark.parametrize("study", STUDIES, ids=lambda module: module.STUDY.slug)
def test_the_record_declares_a_pending_comparator_and_its_sizes(study: object) -> None:
    record = study.STUDY  # type: ignore[attr-defined]
    assert record.comparator_state == "pending"
    assert record.reference is None
    assert record.pending_reference == PENDING[record.slug]
    assert "X59" in record.pending_reason
    assert record.publication_policy == "reporting"
    sizes = record.sizes(record.n)
    assert sizes == {s.name: s.n for s in study.SCENARIOS}  # type: ignore[attr-defined]
    assert all(size % 4_000 == 0 for size in sizes.values())
    assert study.PRIMARY_N == record.n == min(sizes.values())  # type: ignore[attr-defined]
    assert record.slug in {other.slug for other in registered()}


@pytest.mark.parametrize("study", STUDIES, ids=lambda module: module.STUDY.slug)
def test_every_truth_is_the_oracles(study: object) -> None:
    for scenario in study.SCENARIOS:  # type: ignore[attr-defined]
        values = truths(scenario)
        assert set(values) == set(study.STUDY.scenarios[scenario.name])  # type: ignore[attr-defined]
        if scenario.msm:
            continue
        for plan in scenario.plans:
            assert values[mean_name(plan)] == H.truth(scenario.K, scenario.m, plan)
        for first, second in scenario.paired:
            expected = H.truth(scenario.K, scenario.m, first) - H.truth(
                scenario.K, scenario.m, second
            )
            assert values[contrast_name(first, second)] == pytest.approx(expected, abs=1e-15)


def test_a_cross_fitted_scenario_draws_its_owners_sample() -> None:
    for study in STUDIES[1:]:
        record = study.STUDY
        for follower, owner in record.scenario_seed_owners.items():
            assert record.n_of(follower) == record.n_of(owner)
            assert {s.name: s.n_folds for s in study.SCENARIOS}[follower] == 5


@pytest.mark.parametrize("study", [STUDIES[0], STUDIES[2]], ids=["ltmle", "sdr"])
def test_the_driver_returns_the_replication_contract(study: object) -> None:
    """One replication of every scenario at the smoke size n = 8,000 (an override)."""
    result = study.draw_and_fit(replicates=1, n=8_000, n_jobs=1)  # type: ignore[attr-defined]
    rows = result.rows
    record = study.STUDY  # type: ignore[attr-defined]
    assert result.samples is None and result.truths is None
    assert tuple(rows.columns) == REPLICATE_COLUMNS
    assert set(rows["implementation"]) == {record.implementation}
    assert set(rows["n"]) == {8_000}
    for scenario, estimands in record.scenarios.items():
        subset = rows.loc[rows["scenario"] == scenario]
        assert list(subset["estimand"]) == list(estimands)
    assert np.isfinite(rows[["estimate", "std_error", "initial_estimate"]].to_numpy()).all()
    assert (rows["std_error"] > 0).all()
    recomputed = (rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])
    assert (recomputed.astype(int) == rows["covered"]).all()
    assert set(rows["inference_scale"]) <= INFERENCE_SCALES


_RUNNER = re.compile(r"""runner\s*=\s*["'](?P<runner>[^"']+\.R)["']""")


@pytest.mark.parametrize("study", STUDIES, ids=lambda module: module.STUDY.slug)
def test_the_declared_runner_is_a_stub_that_stops(study: object) -> None:
    record = study.STUDY  # type: ignore[attr-defined]
    regenerate = (record.artifacts / "regenerate.py").read_text(encoding="utf-8")
    runner = _RUNNER.search(regenerate)
    assert runner is not None
    path = ROOT / "tests" / "canonical" / runner.group("runner")
    assert path.parent == record.artifacts
    source = path.read_bytes()
    assert b"\r\n" not in source
    text = source.decode("utf-8")
    assert "X59" in text
    assert re.search(r"^stop\(", text, re.M)


@pytest.mark.parametrize("study", STUDIES[1:], ids=lambda module: module.STUDY.slug)
def test_the_cross_fitted_scenario_declares_its_comparison_kind_with_the_runs(
    study: object,
) -> None:
    """The comparison kind stays one default while the comparator is pending.

    ``test_a_study_with_a_reference_declares_its_comparison_kind`` forbids an override on a
    record with no compared reference.  X59 declares ``"<crossfit scenario>:*"`` as a
    ``"different estimator"`` override together with the R runs.
    """
    record = study.STUDY  # type: ignore[attr-defined]
    assert record.reference is None
    assert not record.comparison_overrides
    crossfit = next(s for s in study.SCENARIOS if s.name.endswith("k5_crossfit"))  # type: ignore[attr-defined]
    for scenario in study.SCENARIOS:  # type: ignore[attr-defined]
        for estimand in record.scenarios[scenario.name]:
            assert record.comparison_kind(scenario.name, estimand) == "matching implementation"
    assert crossfit.n_folds == 5
