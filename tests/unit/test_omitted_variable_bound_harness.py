"""The row harness of ``omitted-variable-bound-se`` under the shipped interval rule.

The bound's limits read the estimate's own interval rule about each end, the skew-widened t on
this continuous outcome.  A standard error read back off the limit with a normal quantile is
then not the end's, so each row reports the influence standard error of the end's own curve,
which must reproduce the reported limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cleverly.inference.cluster import influence_variance
from cleverly.sensitivity import omitted_variable as ov
from tests.studies import omitted_variable_bound as study

N = 300
SEED = 5


@pytest.fixture(scope="module")
def drawn() -> tuple[Any, dict[str, float], Any]:
    frame, truth = study.draw_from_seed("linear", N, SEED)
    return frame, truth, study.fit(frame)


def _own_and_backed_out(result: Any, name: str) -> dict[str, tuple[float, float]]:
    bounds = ov.omitted_variable_bounds(
        result, name, cf_y=study.CF_Y, cf_d=study.CF_D, rho=study.RHO
    )
    elements = ov.sensitivity_elements(result, name)
    curve = np.asarray(result[name].influence_curve, dtype=float)
    bias = bounds.confounding_strength * np.asarray(elements.psi_max_bias, dtype=float)
    z = stats.norm.ppf(0.95)
    return {
        "lower": (
            float(np.sqrt(influence_variance(curve - bias, result.data.cluster))),
            (bounds.lower - bounds.ci_lower) / z,
        ),
        "upper": (
            float(np.sqrt(influence_variance(curve + bias, result.data.cluster))),
            (bounds.ci_upper - bounds.upper) / z,
        ),
    }


def test_one_draw_gives_rows_at_each_end_s_own_standard_error(drawn: Any) -> None:
    frame, truth, result = drawn
    rows = study.cleverly_rows(frame, truth, "linear", 0)
    assert len(rows) == len(study.BOUNDS)
    for row in rows:
        name, end = row["estimand"].rsplit("_", 1)
        own, _ = _own_and_backed_out(result, name)[end]
        assert row["std_error"] == pytest.approx(own, rel=1e-13)
        half = study.TWO_SIDED * row["std_error"]
        assert (row["ci_lower"], row["ci_upper"]) == pytest.approx(
            (row["estimate"] - half, row["estimate"] + half), rel=1e-15
        )


def test_the_read_back_error_is_not_the_end_s(drawn: Any) -> None:
    """Nonzero witness: under the skew-widened t the back-out misses the end's error."""
    _, _, result = drawn
    for name in study.FITTED:
        assert result[name].interval_scale == "skew_t"
        for own, backed_out in _own_and_backed_out(result, name).values():
            assert abs(own - backed_out) > 1e-6 * own, name


def test_a_moved_standard_error_fails_the_limit_check(
    drawn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberate mutation: a standard error 1e-6 off no longer reproduces the limit."""
    _, _, result = drawn
    original = study.influence_variance
    monkeypatch.setattr(
        study,
        "influence_variance",
        lambda curve, cluster: (1.0 + 1e-6) ** 2 * original(curve, cluster),
    )
    with pytest.raises(RuntimeError, match="standard error"):
        study.bound_standard_errors(result, "ate")


def test_replicate_zero_reproduces_the_committed_rows() -> None:
    """The harness is result-neutral for the committed artifact: replicate 0 refits to the
    committed estimate, standard error and interval (``tests/canonical/provenance-revisions.md``,
    "Study module edits after a run")."""
    from tests.studies.evidence.manifest import read_rows

    frame, truth = study.draw_scenario("linear", study.PRIMARY_N, 0)
    rows = pd.DataFrame(study.cleverly_rows(frame, truth, "linear", 0))
    committed = read_rows(study.STUDY.artifacts / "replicates.parquet")
    committed = committed.loc[
        (committed["implementation"] == study.STUDY.implementation) & (committed["replicate"] == 0)
    ]
    merged = rows.merge(committed, on=["scenario", "estimand", "replicate"], suffixes=("", "_c"))
    assert len(merged) == len(rows) == len(study.BOUNDS)
    # The refit matched to about 1e-16 when written; 1e-6 leaves room for a library's rounding
    # and still fails a standard error read back off the limit, which moves it by about 1e-2.
    for column in ("estimate", "std_error", "ci_lower", "ci_upper"):
        np.testing.assert_allclose(merged[column], merged[f"{column}_c"], rtol=1e-6, atol=0)
    assert (merged["covered"] == merged["covered_c"]).all()
