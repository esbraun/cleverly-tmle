"""The ``longitudinal-repeated-crossfit`` drivers, checked without running the study.

One primary replication of every scenario at n = 2,000 checks the row contract the shared
driver publishes.  Two property fits under a checkpoint root check that a part run in chunks
resumes: a second call reuses every replicate the first one finished and returns the same rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.canonical.longitudinal_repeated_crossfit import part
from tests.canonical.regenerate import digest, run_key, runs_root
from tests.studies import longitudinal_repeated_crossfit as study
from tests.studies import longitudinal_repeated_crossfit_properties as properties
from tests.studies.evidence import replication
from tests.studies.evidence.schema import REPLICATE_COLUMNS

pytestmark = pytest.mark.xdist_group("longitudinal_repeated_crossfit_study")


def test_the_primary_driver_returns_the_replication_contract() -> None:
    result = study.draw_and_fit(replicates=1, n=2_000, n_jobs=1)
    assert isinstance(result, replication.PhaseResult)
    rows = result.rows
    assert tuple(rows.columns) == REPLICATE_COLUMNS
    assert set(rows["implementation"]) == {study.STUDY.implementation}
    assert set(rows["replicate"]) == {0} and set(rows["n"]) == {2_000}
    for scenario, names in study.SCENARIOS.items():
        subset = rows.loc[rows["scenario"] == scenario]
        assert list(subset["estimand"]) == list(names)
    assert np.isfinite(rows[["estimate", "std_error"]].to_numpy()).all()
    recomputed = (rows["ci_lower"] <= rows["truth"]) & (rows["truth"] <= rows["ci_upper"])
    assert (recomputed.astype(int) == rows["covered"]).all()
    ratio = rows.loc[rows["estimand"] == study.D_RATIO]
    assert set(ratio["inference_scale"]) == {"log"}


def test_a_part_resumes_from_its_checkpoints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A chunk of one replicate, then a chunk of two: the second fits only the new one."""
    with replication.checkpoints(tmp_path):
        first = properties.fit_part("rate_2000", n_jobs=1, first=1)
    capsys.readouterr()
    with replication.checkpoints(tmp_path):
        second = properties.fit_part("rate_2000", n_jobs=1, first=2)
    assert "reusing 1 checkpointed replicates" in capsys.readouterr().out
    head = pd.DataFrame(second)
    head = head.loc[head["replicate"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(head, pd.DataFrame(first))
    assert set(pd.DataFrame(second)["requested_replicates"]) == {properties.RATE_REPLICATES}


def test_every_part_is_a_declared_fit_set() -> None:
    assert ("primary", *properties.FIT_SETS, "stability_d", "stability_i") == part.PARTS
    with pytest.raises(ValueError, match="at least 1"):
        properties.fit_part("rate_2000", n_jobs=1, first=0)


def test_a_part_writes_into_the_declared_run_scratch() -> None:
    """The driver keys a declared run by the record at the declared count and size."""
    record = study.STUDY
    expected = runs_root() / record.slug / digest(run_key(record))[:12]
    assert part.scratch()[0] == expected
