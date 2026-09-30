"""Repeated-sampling seeds cannot duplicate a draw under separate replication labels."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cleverly.inference import make_estimate
from cleverly.validation import CoverageStudy
from cleverly.validation._seeds import sample_seed_streams
from tests.studies.evidence import properties
from tests.studies.evidence.properties import PropertyCell, run_cells


def test_the_expanded_rung_replaces_real_duplicate_seeds_only() -> None:
    candidates = np.random.SeedSequence(24_000).generate_state(73_000)
    assert candidates[6_205] == candidates[34_785]
    repaired = sample_seed_streams({24_000: 73_000})[24_000]
    assert repaired[6_205] != repaired[34_785]
    assert len(set(repaired)) == 73_000
    assert repaired[:600] == tuple(map(int, candidates[:600]))
    # All first occurrences retain their sample.  Only repeated candidates change.
    seen: set[int] = set()
    for candidate, seed in zip(candidates, repaired, strict=True):
        if int(candidate) not in seen:
            assert seed == int(candidate)
        seen.add(int(candidate))


def test_distinct_roots_do_not_share_the_real_outer_rung_collisions() -> None:
    roots = (24_000, 24_200)
    candidates = {
        root: set(map(int, np.random.SeedSequence(root).generate_state(73_000))) for root in roots
    }
    assert len(candidates[roots[0]] & candidates[roots[1]]) == 10
    repaired = sample_seed_streams(dict.fromkeys(roots, 73_000))
    assert not set(repaired[roots[0]]) & set(repaired[roots[1]])
    assert all(len(set(repaired[root])) == 73_000 for root in roots)


def test_a_budget_extension_preserves_old_prefixes_and_unequal_short_streams() -> None:
    short = sample_seed_streams({24_000: 600, 24_200: 600, 24_700: 600})
    full = sample_seed_streams({24_000: 73_000, 24_200: 73_000, 24_700: 600})
    assert all(full[root][:600] == short[root] for root in short)
    assert full == sample_seed_streams({24_700: 600, 24_200: 73_000, 24_000: 73_000})


def test_uncollided_streams_keep_their_exact_sample_seeds() -> None:
    budgets = {7: 100, 17: 100}
    streams = sample_seed_streams(budgets)
    for root, count in budgets.items():
        assert streams[root] == tuple(map(int, np.random.SeedSequence(root).generate_state(count)))


def test_extending_one_stream_can_change_a_longer_stream_only_after_the_old_common_prefix() -> None:
    shorter = sample_seed_streams({24_000: 600, 24_200: 73_000})
    extended = sample_seed_streams({24_000: 73_000, 24_200: 73_000})
    assert shorter[24_200][:600] == extended[24_200][:600]
    assert shorter[24_200] != extended[24_200]


def test_a_plan_refuses_more_distinct_draws_than_the_seed_space() -> None:
    with pytest.raises(ValueError, match="seed space"):
        sample_seed_streams({0: 2**32, 1: 1})


class _Estimator:
    def fit(self, frame: SimpleNamespace, **kwargs: object) -> _Estimator:
        self.estimates = {
            "ate": make_estimate("ate", frame.seed / 2**32, np.array([-1.0, 1.0]), n=2)
        }
        return self

    def __getitem__(self, name: str):  # type: ignore[no-untyped-def]
        return self.estimates[name]


def _draw(n: int, seed: int) -> tuple[SimpleNamespace, dict[str, float]]:
    # The seed remains acceptable to legacy uint32 consumers as well as default_rng.
    np.random.RandomState(seed)
    return SimpleNamespace(seed=seed), {"ate": 0.0}


def test_coverage_study_records_and_uses_the_explicit_sample_seed_vector() -> None:
    seeds = [7, 11, 13]
    study = CoverageStudy(_draw, _Estimator, n=2, n_replicates=3, sample_seeds=seeds)
    seeds[0] = 19
    result = study.run()
    assert [record.seed for record in result.replications] == [7, 11, 13]
    assert [record.estimate for record in result.replications] == [
        7 / 2**32,
        11 / 2**32,
        13 / 2**32,
    ]


def test_coverage_study_uses_distinct_seeds_on_its_default_path() -> None:
    candidates = np.random.SeedSequence(11_100).generate_state(2_400)
    assert len(set(candidates)) == 2_399
    result = CoverageStudy(_draw, _Estimator, n=2, n_replicates=2_400, seed=11_100).run()
    assert len({record.seed for record in result.replications}) == 2_400


@pytest.mark.parametrize(
    ("seeds", "message"),
    [
        ([1, 2], "entries"),
        ([1, 2, 2], "distinct"),
        ([1, 2, True], "uint32"),
        ([1, 2, 3.0], "uint32"),
        ([1, 2, -1], "uint32"),
        ([1, 2, 2**32], "uint32"),
    ],
)
def test_explicit_seed_refusals_precede_the_first_draw(seeds, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CoverageStudy(pytest.fail, pytest.fail, n_replicates=3, sample_seeds=seeds)


def test_a_changed_explicit_vector_is_validated_again_before_drawing() -> None:
    study = CoverageStudy(pytest.fail, pytest.fail, n_replicates=3, sample_seeds=(1, 2, 3))
    study.sample_seeds = (1, 2, 2)
    with pytest.raises(ValueError, match="distinct"):
        study.run()


def test_shared_property_roots_retain_paired_draws_at_unequal_budgets() -> None:
    cell = PropertyCell("witness", "short", _draw, _Estimator, _Estimator, 2, 3, 7)
    rows = run_cells(
        (cell, replace(cell, cell="long", replicates=5)), lambda cell: _Estimator, n_jobs=1
    )
    short = rows.loc[rows["cell"] == "short", "estimate"].to_numpy()
    long = rows.loc[rows["cell"] == "long", "estimate"].to_numpy()
    np.testing.assert_array_equal(short, long[:3])


def test_property_studies_pass_the_joint_plan_to_every_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[int, tuple[int, ...]] = {}

    class RecordedStudy:
        def __init__(self, **kwargs: object) -> None:
            called[int(kwargs["seed"])] = tuple(kwargs["sample_seeds"])

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(replications=(), n_failed=0)

    monkeypatch.setattr(properties, "CoverageStudy", RecordedStudy)
    cell = PropertyCell("witness", "first", _draw, _Estimator, _Estimator, 2, 73_000, 24_000)
    run_cells((cell, replace(cell, cell="second", seed=24_200)), lambda cell: _Estimator, n_jobs=1)
    assert called == sample_seed_streams({24_000: 73_000, 24_200: 73_000})
    assert not set(called[24_000]) & set(called[24_200])
