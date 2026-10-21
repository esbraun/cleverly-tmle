"""The lmtp pairing of ``longitudinal-repeated-sdr`` (X32b PR C), checked without R.

R is not installed in the cloud container, so the lmtp rows arrive from the owner's queue run.
These tests fix everything that can be checked before then: the columns the R runner reads
exist in what the driver writes, the runner and the driver agree on the design, the gate
functions separate a correct pairing from the N8 mutations, and the queued run's Python rows
must equal PR B's.  The parity test on committed rows skips, with its reason, while the
committed rows hold no lmtp rows.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.canonical.sdr_repeated import gates, regenerate
from tests.studies import canonical_sdr_repeated as study
from tests.studies.canonical_ltmle_repeated import COLUMNS, RULE_LABEL
from tests.studies.evidence.registry import ROOT

pytestmark = pytest.mark.filterwarnings("ignore")

CANONICAL = ROOT / "tests" / "canonical"
RUNNER = CANONICAL / "sdr_repeated" / "run_study.R"
ADAPTER = CANONICAL / "lmtp_sdr_repeated_adapter.R"
SMALL_N = 1_000
_SOURCED = re.compile(r"""source\(["']/fixture/(?P<path>[^"']+)["']\)""")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _closure(runner: Path) -> set[str]:
    """Every file the runner reads into R, by the ``source`` chain, relative to the fixture."""
    seen: set[str] = set()
    pending = [runner]
    while pending:
        for match in _SOURCED.finditer(_read(pending.pop())):
            name = match.group("path")
            if name not in seen:
                seen.add(name)
                pending.append(CANONICAL / name)
    return seen


# ---------------------------------------------------------------------------- the R sources


def test_the_driver_declares_every_file_the_runner_sources_transitively() -> None:
    reference = regenerate.REFERENCE
    assert reference.runner == RUNNER.relative_to(CANONICAL).as_posix()
    assert reference.runner_root == CANONICAL
    assert reference.mount_runner
    assert _closure(RUNNER) == set(reference.extra_files)


def test_the_new_adapter_sources_the_shared_adapter_and_never_edits_its_bindings() -> None:
    source = _read(ADAPTER)
    assert 'source("/fixture/lmtp_sdr_adapter.R")' in source
    # The shared file is gated for longitudinal-sdr, and this file only wraps run_ensemble.
    assert source.count('assign("run_ensemble"') == 1
    assert "SL.cellmeans.outcomes" in source
    assert '"^Y[0-9]+$"' in source and '"^A[0-9]+$"' in source
    # The unseen cell stays NA through the class the shared adapter already registers.
    assert 'class = "lmtp_cellmeans"' in source


def test_the_runner_restates_the_study_design() -> None:
    source = _read(RUNNER)
    # The time-varying blocks and the baseline, read back from the R text.
    blocks = re.search(r"blocks <- list\((?P<body>.*)\)\n", source)
    assert blocks is not None
    body = blocks.group("body")
    assert body == 'NULL, c("Y1", "L2"), c("Y2", "L3")'
    assert COLUMNS["time_varying"] == [[], ["L2"], ["L3"]]
    assert COLUMNS["baseline"] == ["W"]
    assert 'baseline = "W"' in source
    assert 'outcome_type = "binomial"' in source
    assert 'learners_outcome = "SL.cellmeans.outcomes"' in source
    assert 'trt = sprintf("A%d", nodes)' in source
    assert 'cens = sprintf("C%d", nodes)' in source
    assert 'outcome = sprintf("Y%d", horizon)' in source
    assert "horizons <- 3L" in source
    assert COLUMNS["treatment"] == ["A1", "A2", "A3"]
    assert COLUMNS["censoring"] == ["C1", "C2", "C3"]
    assert len(study.HORIZONS) == 3
    # k = Inf and .trim = 1 are the adapter defaults, which this runner does not override.
    assert "control =" not in source
    assert "k = Inf" in _read(CANONICAL / "lmtp_sdr_adapter.R")
    assert ".trim = 1" in _read(CANONICAL / "lmtp_sdr_adapter.R")


def test_the_runner_streams_several_replicates_per_chunk_and_promotes_warnings() -> None:
    source = _read(RUNNER)
    chunk = int(re.search(r"chunk_lines = (\d+)L", source).group(1))  # type: ignore[union-attr]
    assert chunk // (len(study.SCENARIOS) * study.PRIMARY_N) >= 8
    assert "study_stage" in _read(CANONICAL / "lmtp_sdr_adapter.R")
    # The harness contract (tests/unit/test_harness_failures.py): a promoted warning raises the
    # fatal class, so a stage never records it as a fitting failure.
    handler = re.search(
        r"warning = function\(condition\) \{\s+study_fatal_error\(sprintf\(", source
    )
    assert handler is not None
    assert "withCallingHandlers(" in source
    # The unseen-cell check runs before the fit and outside any stage, so a refusal is fatal.
    assert 'study_stage("unseen cells"' not in source
    assert source.index(
        "check_cells(frame, label, horizon)\n  withCallingHandlers("
    ) < source.index("lmtp_sdr_with_folds(\n")


def _r_slug(label: str) -> str:
    """The runner's ``slug``: ``gsub("^_+|_+$", "", gsub("[^a-z0-9]+", "_", tolower(label)))``."""
    return re.sub(r"^_+|_+$", "", re.sub(r"[^a-z0-9]+", "_", label.lower()))


def _r_parse(name: str) -> tuple[str, str | None, int]:
    """The runner's ``parse_estimand``, restated from its text."""
    inner = re.sub(r"^[a-z_]+\[(.*)\]$", r"\1", name)
    horizon = int(re.sub(r".* @ t=", "", inner))
    inner = re.sub(r" @ t=[0-9]+$", "", inner)
    if name.startswith("ate_regimen["):
        left, right = inner.split(" vs ")
        return left, right, horizon
    return inner, None, horizon


def test_the_runner_reads_columns_the_driver_writes() -> None:
    """Every column the runner reads is in the sample that ``pairing_sample`` builds."""
    frame, _ = study.draw_scenario("repeated_folds_5", SMALL_N, 0)
    result = study.fit_cleverly(frame, folds=5)
    sample = study.pairing_sample(frame, result, "repeated_folds_5", 0)
    read = {"scenario", "replicate", "row", "fold", "W"}
    needed_plans: dict[str, set[int]] = {}
    for name in study.ESTIMANDS:
        left, right, horizon = _r_parse(name)
        for label in filter(None, (left, right)):
            needed_plans.setdefault(label, set()).add(horizon)
    # Every plan is one the driver wrote columns for.
    written = {study.slug(plan.label) for plan in study.plans_of(result)}
    assert {_r_slug(label) for label in needed_plans} <= written
    assert _r_slug(RULE_LABEL) in written
    for label, horizons in needed_plans.items():
        for horizon in horizons:
            for node in range(1, horizon + 1):
                read |= {
                    f"A{node}",
                    f"C{node}",
                    f"ratio__{_r_slug(label)}__{node}",
                    f"shift__{_r_slug(label)}__{node}__1",
                }
            read.add(f"Y{horizon}")
            read |= {"Y1", "L2", "Y2", "L3"} & {
                column for block in [["Y1", "L2"], ["Y2", "L3"]][: horizon - 1] for column in block
            }
    assert read <= set(sample.columns), sorted(read - set(sample.columns))
    # The truth table holds the names the runner parses, and the horizon is a prefix.
    assert all(1 <= _r_parse(name)[2] <= 3 for name in study.ESTIMANDS)


def test_the_runner_and_the_gates_agree_on_the_influence_curve_files() -> None:
    assert 'sprintf("ic-%s-%d.csv.gz", scenario, as.integer(replicate))' in _read(RUNNER)
    assert 'f"ic-{scenario}-{int(replicate)}.csv.gz"' in _read(Path(gates.__file__))
    assert 'study_env("CLEVERLY_IC_DIR")' in _read(RUNNER)
    assert "CLEVERLY_IC_DIR=/work/ic" in _read(Path(gates.__file__))


def test_the_primary_row_the_runner_publishes_is_the_schema() -> None:
    from tests.studies.evidence.schema import REPLICATE_COLUMNS
    from tests.unit.test_canonical_runner_parity import published_keys

    assert published_keys(_read(RUNNER))[: len(REPLICATE_COLUMNS)] == REPLICATE_COLUMNS


# ---------------------------------------------------------------------------- the gate functions


def _synthetic(*, cells: int = 4, replicates: int = 6) -> pd.DataFrame:
    """Paired rows whose lmtp correction is about 0.03 and whose two sides agree exactly."""
    rng = np.random.default_rng(7)
    rows = []
    for scenario in ("a", "b"):
        for cell in range(cells):
            for replicate in range(replicates):
                initial = rng.normal(0.4, 0.02)
                estimate = initial + rng.normal(0.03, 0.01)
                for implementation in ("cleverly", "lmtp"):
                    rows.append(
                        {
                            "implementation": implementation,
                            "scenario": scenario,
                            "replicate": replicate,
                            "estimand": f"e{cell}",
                            "estimate": estimate,
                            "initial_estimate": initial,
                        }
                    )
    return pd.DataFrame(rows)


def test_the_parity_ratio_passes_equal_rows_and_reads_every_cell() -> None:
    rows = _synthetic()
    table = gates.parity_table(rows)
    assert len(table) == 8
    assert table["passed"].all()
    assert (table["max_gap"] == 0.0).all()
    assert (table["rms_correction"] > 0.01).all()


@pytest.mark.parametrize("mutation", [gates.mutate_correction, gates.mutate_plug_in])
def test_n8_fails_the_parity_ratio_in_every_cell(mutation) -> None:  # type: ignore[no-untyped-def]
    """N8: a plug-in, or a correction shortened by 10%, is a gap of at least 10% of the RMS."""
    table = gates.parity_table(mutation(_synthetic()))
    assert not table["passed"].any()
    assert (table["ratio"] >= 0.1).all()


def test_the_parity_ratio_edge_sits_at_one_percent() -> None:
    rows = _synthetic()
    mine = rows["implementation"] == "cleverly"
    base = gates.parity_table(rows)
    rms = base["rms_correction"].min()
    shifted = rows.copy()
    shifted.loc[mine, "estimate"] += 0.5 * gates.PARITY_SHARE * rms
    assert gates.parity_table(shifted)["passed"].all()
    shifted = rows.copy()
    shifted.loc[mine, "estimate"] += 5.0 * gates.PARITY_SHARE * base["rms_correction"].max()
    assert not gates.parity_table(shifted)["passed"].any()


def test_a_cell_with_no_lmtp_correction_is_not_passed() -> None:
    rows = _synthetic()
    rows["initial_estimate"] = rows["estimate"]
    assert not gates.parity_table(rows)["passed"].any()


def test_the_parity_ratio_refuses_rows_without_both_sides() -> None:
    rows = _synthetic()
    with pytest.raises(ValueError, match="no \\['lmtp'\\] rows"):
        gates.parity_table(rows.loc[rows["implementation"] == "cleverly"])


def _curves(rows: pd.DataFrame, shift: float = 0.0) -> dict:
    out = {}
    rng = np.random.default_rng(3)
    for scenario, replicate, estimand in rows[gates.PAIR].drop_duplicates().itertuples(index=False):
        curve = rng.normal(size=50)
        theirs = curve.copy()
        theirs[17] += shift
        out[(scenario, int(replicate), estimand)] = (curve, theirs)
    return out


def test_g1r_passes_agreeing_sides_and_fails_a_curve_or_an_estimate_gap() -> None:
    rows = _synthetic()
    mine = rows.loc[rows["implementation"] == "cleverly"]
    theirs = rows.loc[rows["implementation"] == "lmtp"]
    table = gates.g1r_table(mine, theirs, _curves(rows))
    assert table["passes"].all()
    assert gates.g1r_cells(table)["passes"].all()
    # A curve gap of 1e-9 at one unit fails its cell and no other.
    table = gates.g1r_table(mine, theirs, _curves(rows, shift=1e-9))
    assert not table["passes"].any()
    assert (table["ic_gap"] > gates.G1R_GATE).all()
    # An estimate gap fails although the curves agree.
    moved = theirs.copy()
    moved["estimate"] += 1e-9
    table = gates.g1r_table(mine, moved, _curves(rows))
    assert not table["passes"].any()
    assert (table["ic_gap"] == 0.0).all()


def test_g1r_centres_the_curves_because_lmtp_eif_has_mean_psi() -> None:
    """lmtp's eif has mean equal to the estimate; cleverly's curve has mean zero."""
    rows = _synthetic()
    mine = rows.loc[rows["implementation"] == "cleverly"]
    theirs = rows.loc[rows["implementation"] == "lmtp"]
    psi = dict(zip(map(tuple, mine[gates.PAIR].to_numpy()), mine["estimate"], strict=True))
    curves = {}
    for key, (curve, same) in _curves(rows).items():
        curve = curve - curve.mean()
        curves[key] = (curve, same - same.mean() + psi[key])
    table = gates.g1r_table(mine, theirs, curves)
    assert table["passes"].all()
    assert (table["ic_gap"] < 1e-12).all()


def test_g1r_fails_a_real_per_unit_difference_after_centring() -> None:
    rows = _synthetic()
    mine = rows.loc[rows["implementation"] == "cleverly"]
    theirs = rows.loc[rows["implementation"] == "lmtp"]
    curves = {}
    for key, (curve, same) in _curves(rows).items():
        different = same.copy()
        different[3] += 1e-3  # one unit differs, so centring cannot remove it
        curves[key] = (curve - curve.mean(), different + 0.4)
    table = gates.g1r_table(mine, theirs, curves)
    assert not table["passes"].any()
    assert (table["ic_gap"] > 5e-4).all()


def test_g1r_refuses_unpaired_rows() -> None:
    rows = _synthetic()
    mine = rows.loc[rows["implementation"] == "cleverly"]
    theirs = rows.loc[rows["implementation"] == "lmtp"].iloc[1:]
    with pytest.raises(AssertionError, match="G1r pairs"):
        gates.g1r_table(mine, theirs, _curves(rows))


# ---------------------------------------------------------------------------- PR B's rows


def _committed() -> pd.DataFrame:
    return pd.read_csv(study.STUDY.artifacts / "replicates.csv.gz", float_precision="round_trip")


def test_the_baseline_is_the_file_pr_b_committed() -> None:
    rows = gates.pr_b_rows()
    assert set(rows["implementation"]) == {study.STUDY.implementation}
    assert len(rows) == study.PRIMARY_REPLICATES * len(study.SCENARIOS) * len(study.ESTIMANDS)


def test_the_python_rows_of_this_run_equal_the_rows_pr_b_committed() -> None:
    """The queued run draws the same seeds, so its Python rows are PR B's to 1e-10 (plan 10)."""
    differences = gates.pr_b_differences(_committed(), gates.pr_b_rows())
    assert differences.empty, differences.head()


def test_the_pr_b_comparison_catches_a_moved_or_missing_row() -> None:
    baseline = gates.pr_b_rows()
    moved = baseline.copy()
    moved.loc[moved.index[5], "estimate"] += 1e-9
    found = gates.pr_b_differences(moved, baseline)
    assert list(found["column"]) == ["estimate"]
    assert gates.pr_b_differences(baseline.iloc[1:], baseline)["column"].tolist() == ["(row)"]
    nan = baseline.copy()
    nan.loc[nan.index[7], "std_error"] = np.nan
    assert list(gates.pr_b_differences(nan, baseline)["column"]) == ["std_error"]
    # A change below the gate is not a difference.
    near = baseline.copy()
    near.loc[near.index[5], "estimate"] += 1e-12
    assert gates.pr_b_differences(near, baseline).empty


def test_the_pr_b_comparison_ignores_the_lmtp_rows_of_the_new_run() -> None:
    baseline = gates.pr_b_rows()
    extra = baseline.iloc[:3].copy()
    extra["implementation"] = "lmtp"
    assert gates.pr_b_differences(pd.concat([baseline, extra]), baseline).empty


# ---------------------------------------------------------------------------- the committed rows


def test_the_parity_ratio_on_the_committed_rows() -> None:
    """WP: once the record compares against lmtp, committed rows without lmtp fail this test."""
    rows = _committed()
    assert study.STUDY.reference == "lmtp"
    assert str(study.STUDY.reference) in set(rows["implementation"]), (
        "longitudinal-repeated-sdr declares reference=lmtp but the committed rows hold no lmtp "
        "rows; the owner's Study queue run (plan X32b PR C, item 8) writes them"
    )
    table = gates.parity_table(rows)
    assert len(table) == len(study.SCENARIOS) * len(study.ESTIMANDS)
    assert table["passed"].all(), table.loc[~table["passed"]]
    for mutation in (gates.mutate_correction, gates.mutate_plug_in):
        assert not gates.parity_table(mutation(rows))["passed"].any()


def test_the_g1r_verdict_recomputes_from_the_committed_gates_file() -> None:
    """Precedent: ``test_longitudinal_sdr_design.py``; skips while the file is absent."""
    path = study.STUDY.artifacts / "gates.csv"
    if not path.exists():
        pytest.skip("gates.csv is committed by the queue after G1r (plan X32b PR C, step 2)")
    table = pd.read_csv(path, float_precision="round_trip")
    assert list(table.columns) == [*gates.PAIR, "estimate_gap", "ic_gap", "passes"]
    expected = len(study.SCENARIOS) * len(study.ESTIMANDS) * gates.G1R_REPLICATES
    assert len(table) == expected
    assert set(table["scenario"]) == set(study.SCENARIOS)
    assert set(table["estimand"]) == set(study.ESTIMANDS)
    assert set(table["replicate"]) == set(range(gates.G1R_REPLICATES))
    recomputed = (table["estimate_gap"] <= gates.G1R_GATE) & (table["ic_gap"] <= gates.G1R_GATE)
    assert (recomputed == table["passes"]).all()
    assert recomputed.all(), table.loc[~recomputed]
