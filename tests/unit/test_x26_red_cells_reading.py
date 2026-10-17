r"""The diagnostics of the two red cells of ``omitted-variable-bound-plugin-se`` (X26).

``tests/diagnostics/x26_red_cells/agreement.py`` is rule 2 of the red-cell policy.  On 20 draws of
``multinomial_logistic_crossfit`` it compares the package curve of the plug-in
:math:`\hat\nu^2` with an infinitesimal jackknife (unit witness W5) and a numerical sandwich
(``TestTheNumericalSandwich``).  ``rerun.py`` is rule 3.  These tests hold what the committed
agreement rows say, the wiring of the rerun, and the rounded numbers that the documents quote from
``rerun-reading.csv``.

A comparison that passes does not show that it could fail, so a deliberate mutation follows the
agreement: with the mechanism term of the curve dropped, both comparators leave their tolerance.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import cleverly.sensitivity._mechanism_stack as ms
import cleverly.sensitivity.omitted_variable as ov
from cleverly.inference.cluster import influence_variance
from tests.diagnostics.x26_red_cells import agreement, rerun
from tests.studies import omitted_variable_bound_plugin as study
from tests.studies.evidence.seeds import replicate_seed

CANONICAL = study.STUDY.artifacts
RED_CELLS = {
    token: f"multinomial_logistic_crossfit_{token}_lower__correctly_specified"
    for token in agreement.TOKENS
}


def _rows() -> pd.DataFrame:
    return pd.read_csv(agreement.HERE / "agreement-rows.csv", float_precision="round_trip")


def test_the_rows_are_the_declared_design() -> None:
    rows = _rows()
    assert list(rows.columns) == list(agreement.ROW_COLUMNS)
    assert len(rows) == agreement.REPLICATES * len(agreement.TOKENS)
    assert sorted(rows["replicate"].unique()) == list(range(agreement.REPLICATES))
    assert set(rows["parameter"]) == set(agreement.TOKENS)
    assert set(rows["n"]) == {study.PRIMARY_N}
    assert int(rows["jackknife_rows"].min()) >= study.PRIMARY_N // agreement.ROW_STRIDE


def test_the_draws_are_the_ones_the_red_cells_fitted() -> None:
    """The refit reproduces the standard error that the declared run committed, per replicate."""
    committed = pd.read_csv(
        CANONICAL / "property-replicates.csv.gz",
        usecols=["cell", "replicate", "std_error"],
        float_precision="round_trip",
    )
    rows = _rows()
    for token, cell in RED_CELLS.items():
        held = committed.loc[committed["cell"] == cell].set_index("replicate")["std_error"]
        ours = rows.loc[rows["parameter"] == token].set_index("replicate")["lower_se_study"]
        assert set(ours.index) <= set(held.index)
        np.testing.assert_allclose(ours.to_numpy(), held.loc[ours.index].to_numpy(), rtol=1e-9)


def test_the_reading_is_rebuilt_from_the_rows() -> None:
    committed = pd.read_csv(agreement.HERE / "agreement-reading.csv")
    rebuilt = agreement.reading(_rows())
    pd.testing.assert_frame_equal(
        rebuilt.drop(columns="agrees"),
        committed.drop(columns="agrees"),
        check_exact=False,
        rtol=1e-5,
        atol=1e-12,
    )
    assert list(committed["agrees"]) == list(rebuilt["agrees"])


def test_the_package_curve_agrees_with_both_comparators() -> None:
    """The rule 2 finding: no mismatch, so the cells are not a defect of the curve."""
    reading = agreement.reading(_rows())
    assert reading["agrees"].all()
    assert reading["jackknife_deviation_tight"].max() < agreement.RTOL
    assert reading["sandwich_gap"].max() < agreement.RTOL
    assert (reading["replicates"] == agreement.REPLICATES).all()


def test_the_declared_fit_moves_the_reported_error_by_less_than_a_part_in_a_thousand() -> None:
    """The solver of the declared fit does not explain a 5% excess of the standard error."""
    reading = agreement.reading(_rows())
    assert reading["mean_lower_se_shift"].abs().max() < 1e-3


def test_dropping_the_mechanism_term_fails_both_comparators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberate mutation: the comparators must see a curve without its mechanism term."""
    frame = agreement.draw(0)
    result = agreement.fit_tight(frame)
    original = ms.plugin_mechanism_term

    def dropped(*arguments: object) -> object:
        out = original(*arguments)  # type: ignore[arg-type]
        if isinstance(out, ms.MechanismRefusal):
            return out
        return replace(out, values=np.zeros_like(out.values))

    monkeypatch.setattr(ov, "plugin_mechanism_term", dropped)
    token = "att_2v0"
    name = agreement.oracle_name(token)
    elements = ov.sensitivity_elements(
        result, study.PARAMETERS[agreement.SCENARIO][token][0], nu2_estimator="plugin"
    )
    assert elements.psi_nu2 is not None
    curve = np.asarray(elements.psi_nu2, dtype=float)
    x = np.asarray(result.data.covariates, dtype=float)
    a = np.asarray(result.data.treatment, dtype=float)
    folds = agreement.folds_of(result)
    bounds = agreement.bounds_of(result, token)
    rows = np.arange(0, study.PRIMARY_N, 100)
    expected = agreement.jackknife(x, a, folds, rows, {token: name}, {token: bounds})[token]
    deviation = float(np.max(np.abs(curve[rows] - expected)) / np.max(np.abs(expected)))
    assert deviation > 1e3 * agreement.RTOL
    _, variance = agreement.sandwich(x, a, folds, name, bounds)
    n = study.PRIMARY_N
    curve_variance = float(influence_variance(curve, None)) * (n - 1) / n
    assert abs(variance - curve_variance) / variance > 1e3 * agreement.RTOL


# --- the rerun of rule 3 ----------------------------------------------------------------------


def test_the_rerun_design_is_the_declared_one() -> None:
    assert (rerun.N, rerun.REPLICATES) == (4_000, 5_000)
    assert rerun.SCENARIO == "multinomial_logistic_crossfit"
    assert rerun.TOKENS == agreement.TOKENS


def test_the_rerun_seed_stream_is_disjoint_from_the_declared_run() -> None:
    ours = {rerun.sample_seed(rerun.N, k) for k in range(200)}
    assert len(ours) == 200
    primary = {
        replicate_seed(study.STUDY, name, k) for name in study.STUDY.scenarios for k in range(200)
    }
    properties = {agreement.sample_seed(k) for k in range(200)}
    assert not ours & primary
    assert not ours & properties
    assert rerun.sample_seed(4_000, 0) != rerun.sample_seed(1_000, 0)


def _checkpoint(directory: Path, replicate: int, *, refuse: str | None = None) -> None:
    held: dict[str, object] = {}
    for token in rerun.TOKENS:
        if token == "att_2v0" and refuse is not None:
            held[token] = {"refusal": refuse}
            continue
        held[token] = {
            row: {"truth": 1.0, "estimate": 1.0 + 0.01 * replicate, "std_error": 0.1}
            for row in study.CLAIM_ROWS
        }
    rerun.checkpoint_path(directory, replicate).write_text(
        json.dumps({"replicate": replicate, "n": rerun.N, "parameters": held}), encoding="utf-8"
    )


def test_the_rerun_stamps_a_refused_replicate_on_its_own_cells(tmp_path: Path) -> None:
    for replicate in range(4):
        _checkpoint(tmp_path, replicate, refuse="truncation_active" if replicate == 2 else None)
    rows = rerun.collect(tmp_path, 4, rerun.N)
    counts = rows.groupby("cell")["replicate"].nunique()
    stamps = rows.groupby("cell")["failed_replicates"].first()
    for cell in counts.index:
        lost = 1 if "att_2v0" in cell else 0
        assert counts[cell] == 4 - lost
        assert stamps[cell] == lost
    assert len(counts) == len(rerun.TOKENS) * len(study.CLAIM_ROWS)
    assert set(rows["n"]) == {rerun.N}
    assert set(rows["requested_replicates"]) == {4}
    table = rerun.refusals(tmp_path, 4)
    assert table.to_dict("records") == [
        {"parameter": "att_2v0", "key": "truncation_active", "lost": 1, "rate": 0.25}
    ]


def test_the_rerun_needs_every_checkpoint(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 0)
    with pytest.raises(SystemExit, match="no checkpoint"):
        rerun.collect(tmp_path, 2, rerun.N)


def test_a_finished_replicate_is_not_fitted_again(tmp_path: Path) -> None:
    """A checkpoint on disk is final: the worker returns without fitting the replicate."""
    _checkpoint(tmp_path, 0)
    before = rerun.checkpoint_path(tmp_path, 0).read_bytes()
    rerun.run_replicate(rerun.N, 0, tmp_path)
    assert rerun.checkpoint_path(tmp_path, 0).read_bytes() == before


def test_the_rerun_refuses_checkpoints_of_another_configuration(tmp_path: Path) -> None:
    rerun.prepare(tmp_path, rerun.N, fresh=False)
    rerun.prepare(tmp_path, rerun.N, fresh=False)
    with pytest.raises(SystemExit, match="--fresh"):
        rerun.prepare(tmp_path, 1_000, fresh=False)
    rerun.prepare(tmp_path, 1_000, fresh=True)


#: The numbers that the roadmap owner rows, the red-cell page and the study page quote from
#: ``rerun-reading.csv``, at the rounding they print: the SE ratio and its 99% interval to three
#: decimals, the coverage to four.
QUOTED_RERUN = {
    "ate_2v0_lower": (0.998, 0.973, 1.025, 0.9506),
    "ate_2v0_upper": (1.002, 0.976, 1.028, 0.9520),
    "att_2v0_lower": (1.002, 0.977, 1.027, 0.9534),
    "att_2v0_upper": (1.007, 0.982, 1.034, 0.9528),
}


def test_the_documents_quote_the_committed_rerun_reading() -> None:
    reading = pd.read_csv(rerun.HERE / "rerun-reading.csv")
    assert len(reading) == len(QUOTED_RERUN)
    for token, (ratio, lower, upper, coverage) in QUOTED_RERUN.items():
        row = reading.loc[reading["cell"].str.contains(f"crossfit_{token}__")]
        assert len(row) == 1, token
        held = row.iloc[0]
        assert (held["n"], held["replicates"], held["failed_replicates"]) == (4_000, 5_000, 0)
        assert round(held["se_ratio"], 3) == ratio
        assert round(held["se_ratio_ci_lower"], 3) == lower
        assert round(held["se_ratio_ci_upper"], 3) == upper
        assert round(held["coverage"], 4) == coverage


def test_the_rerun_reads_both_red_cells_as_contracting() -> None:
    """The reading rule 8 gives: the 99% SE-ratio interval at n = 4000 contains 1."""
    reading = pd.read_csv(rerun.HERE / "rerun-reading.csv")
    for token in ("ate_2v0", "att_2v0"):
        row = reading.loc[reading["cell"].str.contains(f"crossfit_{token}_lower__")].iloc[0]
        assert row["se_ratio_ci_lower"] < 1.0 < row["se_ratio_ci_upper"]
        assert row["coverage_ci_lower"] < 0.95 < row["coverage_ci_upper"]
        assert bool(row["passed"])
