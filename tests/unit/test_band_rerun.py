"""The band-rerun plan, its replay rule, its composite manifest, its wave and its sidecars.

``tests/studies/evidence/band_rerun.py`` refits only the band keys of a study after the default
band moved (band-finite-sample plan sections 4.4, 8 and 9).  These tests hold the committed plan
to the tree without fitting, hold the replay rule, the splice, the summary rule and the
composite gate to synthetic rows, each with a mutation that the gate must refuse, and run the
post-fit path of five studies with extra columns on their committed rows, which must publish
the committed bytes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.studies.evidence import band_rerun
from tests.studies.evidence.manifest import read_rows
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.registry import ROOT

BAND = "simultaneous_coverage/a__simultaneous_band"
POINT = "point/a"
#: The planned studies whose committed rows carry columns beyond ``REPLICATE_COLUMNS``.
EXTRA_COLUMN_STUDIES = (
    "adaptive-iptw",
    "composite-missing-drtmle",
    "ctmle-oat-per-arm-parametric-extensions",
    "long-horizon-categorical-ltmle",
    "long-horizon-ltmle",
)
#: The studies whose own summary of the full table takes minutes.  Their case shows that the
#: spliced rows are the committed rows, every column, and hands the committed summary on;
#: ``test_method_evidence``'s ``test_paper_property_verdicts_are_recomputed_from_the_replication_rows``
#: already recomputes that summary from those rows in the fast suite.
SLICE = 400
LONG_SUMMARIES = (
    "ctmle-oat-per-arm-parametric-extensions",
    "long-horizon-categorical-ltmle",
    "long-horizon-ltmle",
)


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    return json.loads(band_rerun.PLAN_PATH.read_bytes())


def test_the_committed_plan_holds_against_the_tree(plan: dict[str, Any]) -> None:
    assert band_rerun.validate_plan(plan) == []


def test_the_plan_excludes_the_studies_the_default_does_not_reach(plan: dict[str, Any]) -> None:
    """The curve band keeps its options, and the SDR band cells name theirs (plan 4.3, 4.4)."""
    assert set(plan["excluded"]) == {"longitudinal-sdr"}
    assert "targeting-curves" not in plan["studies"]
    assert any(
        band_rerun.EXCLUSION_WITNESS["longitudinal-sdr"] in pin
        for pin in plan["excluded"]["longitudinal-sdr"]["pins"]
    )
    for entry in plan["studies"].values():
        assert entry["band_fits"] > 0 and entry["moved_cells"], entry["artifact_directory"]


def test_the_plan_records_every_moved_cell_and_the_true_draw_pins(plan: dict[str, Any]) -> None:
    """Controls and diagnostics are moved cells; only a literal draw count is a draw pin."""
    extensions = {
        c["cell"]: c for c in plan["studies"]["ctmle-oat-per-arm-parametric-extensions"]["cells"]
    }
    for cell in ("binary_population", "three_arm"):
        control = extensions[f"{cell}__shipped_curve_band_control"]
        assert control["role"] == "control"
    categorical = plan["studies"]["long-horizon-categorical-ltmle"]
    assert categorical["moved_cells"] == ["simultaneous_coverage/k5__ic_band"]
    assert "simultaneous_coverage/k5__simultaneous_band" in categorical["not_moved"]
    assert plan["totals"]["pinning_draws"] == ["composite-missing-drtmle", "multi-arm-mar-drtmle"]


def test_the_planned_studies_install_as_one_unit(plan: dict[str, Any]) -> None:
    """Every planned study is at its baseline, or every one is a composite, or every one has a
    sidecar beside its baseline: a wave never installs in part."""
    states = set(band_rerun.installed_state(plan).values())
    assert len(states) == 1 and states <= {"baseline", "composite", "sidecar"}, states


PLANNED = tuple(json.loads(band_rerun.PLAN_PATH.read_bytes())["studies"])


@pytest.mark.parametrize("slug", PLANNED)
def test_every_planned_study_smokes_on_the_smoke_stream(slug: str) -> None:
    """The first band key of each call carries a salted seed or draws one (``--smoke``).

    A seed that the driver builds into a key, a draw that takes its own and a key that carries
    a declared seed are each read; the smoke itself refuses any key this check would fail.
    """
    assert band_rerun.salt_check(slug) == []


def test_the_salt_does_not_outlive_its_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """A module imported inside the salted block binds the salted seed through the swapped
    ``seeds`` module; the block's exit rebinds it, or a later summary reads smoke seeds."""
    import sys
    import types

    from tests.studies.evidence import seeds

    probe = types.ModuleType("tests._salt_probe")
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    with band_rerun.salted():
        exec("from tests.studies.evidence.seeds import stream_seed", probe.__dict__)
        assert probe.stream_seed is not band_rerun._ORIGINAL_SEEDS["stream_seed"]
    assert probe.stream_seed is band_rerun._ORIGINAL_SEEDS["stream_seed"]
    assert seeds.stream_seed is band_rerun._ORIGINAL_SEEDS["stream_seed"]


def test_a_key_with_a_declared_seed_or_no_salt_is_refused() -> None:
    declared, salted_seed = 2**40 + 1, 2**40 + 2
    band_rerun.DECLARED_SEEDS.add(declared)
    band_rerun.SALTED_SEEDS.add(salted_seed)
    assert band_rerun.salt_findings([("a", 3, salted_seed)], [0]) == []
    assert band_rerun.salt_findings([("a", 3)], [1]) == []
    assert band_rerun.salt_findings([("a", 3)], [0]) == ["('a', 3) drew no salted seed"]
    found = band_rerun.salt_findings([("a", declared, salted_seed)], [1])
    assert found and "declared seed" in found[0]


def test_every_blocked_sidecar_matches_its_record(plan: dict[str, Any]) -> None:
    for record in sorted((ROOT / "tests/canonical").glob(f"*/{band_rerun.SIDECAR}/record.json")):
        assert band_rerun.validate_sidecar(record.parent.parent, plan) == [], record


def test_no_support_or_grid_source_reads_a_sidecar() -> None:
    """A blocked run is diagnostic: its verdicts reach no support row and no grid row (4.4)."""
    sources = [*(ROOT / "src/cleverly").rglob("*.py"), *(ROOT / "tests/studies").rglob("*.py")]
    readers = [
        path.relative_to(ROOT).as_posix()
        for path in sources
        if path.name != "band_rerun.py"
        and any(
            text in path.read_text(encoding="utf-8")
            for text in ("band_rerun", *band_rerun.SIDECAR_FILES)
        )
    ]
    assert readers == []


# ---------------------------------------------------------------------------- the post-fit path


@pytest.mark.parametrize("slug", EXTRA_COLUMN_STUDIES)
def test_the_post_fit_path_publishes_the_committed_bytes(
    slug: str, plan: dict[str, Any], tmp_path: Path
) -> None:
    """With the committed worker values handed back, every output file is the committed file.

    The witness covers the columns outside ``REPLICATE_COLUMNS`` that these studies' rows
    carry: a splice, a carried digest or a replay that dropped them would change the bytes or
    stop the study's own summary.  Two studies run their own summary here; the three of
    :data:`LONG_SUMMARIES` hand on the committed one once their spliced rows are shown to be the
    committed rows, value for value, and replay the first :data:`SLICE` band keys of
    each call.
    """
    entry = plan["studies"][slug]
    record = band_rerun.record_for(slug)
    committed = read_rows(record.artifacts / "property-replicates.parquet")
    assert set(committed.columns) - set(REPLICATE_COLUMNS)
    index = band_rerun.committed_index(committed)
    calls = band_rerun.capture(record, entry["moved_cells"], index)
    assert band_rerun.selection(calls, entry["moved_cells"]) == entry["selection"]
    if slug in LONG_SUMMARIES:
        # A slice of each call's band keys: the rows outside it are the committed rows anyway.
        for each in calls:
            each.selected = each.selected[:SLICE]
    values = {
        call.label: [band_rerun.committed_value(call, call.keys[i], index) for i in call.selected]
        for call in calls
        if call.selected
    }
    summarize = None
    if slug in LONG_SUMMARIES:

        def summarize(rows: pd.DataFrame) -> pd.DataFrame:
            pd.testing.assert_frame_equal(rows, committed, check_exact=True)
            return pd.read_csv(record.artifacts / "properties.csv")

    composite = band_rerun.post_fit(
        record, entry, committed, calls, values, jobs=1, summarize=summarize, index=index
    )
    assert composite.summary_record["changed_rows"] == []
    hashes = band_rerun.write_outputs(composite, record, tmp_path)
    assert hashes == entry["artifact_sha256"]


# ---------------------------------------------------------------------------- synthetic rows


def row(cell: str, replicate: int, estimate: float, **fields: Any) -> dict[str, Any]:
    prop, name = cell.split("/", 1)
    out = {
        "property": prop,
        "cell": name,
        "role": "positive",
        "replicate": replicate,
        "n": 500,
        "requested_replicates": 2,
        "failed_replicates": 0,
        "truth": 0.25,
        "estimate": estimate,
        "std_error": 1.0,
        "covered": 1,
        "rejected": 0,
    }
    out.update(fields)
    columns = [*REPLICATE_COLUMNS, "rounds"]
    return {column: out.get(column, np.nan) for column in columns}


def committed() -> pd.DataFrame:
    return pd.DataFrame(
        [row(POINT, r, 0.3 + r, rounds=3.0) for r in range(2)]
        + [row(BAND, r, 1.5) for r in range(2)]
    )


def call() -> band_rerun.Call:
    return band_rerun.Call(
        label="properties",
        keys=[0, 1],
        selected=[0, 1],
        options={"cells_of": lambda key: [POINT, BAND], "replicate_of": lambda key: key},
    )


def refitted(point_shift: float = 0.0, rounds: float = 3.0) -> list[Any]:
    return [
        [
            row(POINT, r, 0.3 + r + point_shift, rounds=rounds),
            row(BAND, r, 2.0, covered=0, rejected=1, std_error=1.5),
        ]
        for r in range(2)
    ]


def test_the_replay_rule_admits_rounding_and_refuses_a_real_change() -> None:
    before = row(POINT, 0, 0.3, rounds=3.0)
    assert band_rerun.compare(before, row(POINT, 0, 0.3 + 1e-12, rounds=3.0))["estimate"] < 1e-9
    with pytest.raises(band_rerun.ReplayMiss, match="estimate"):
        band_rerun.compare(before, row(POINT, 0, 0.3 + 1e-6, rounds=3.0))
    with pytest.raises(band_rerun.ReplayMiss, match="covered"):
        band_rerun.compare(before, row(POINT, 0, 0.3, covered=0, rounds=3.0))


def test_a_refit_replaces_only_band_value_fields() -> None:
    rows = committed()
    refit = band_rerun.replay(
        [call()], {"properties": refitted()}, band_rerun.committed_index(rows), [BAND]
    )
    assert (refit.compared, refit.exact, len(refit.band_rows)) == (2, 2, 2)
    composite = band_rerun.splice(rows, refit.band_rows)
    assert list(composite.columns) == list(rows.columns)
    assert band_rerun.carried_digest(composite, [BAND]) == band_rerun.carried_digest(rows, [BAND])
    mask = band_rerun.band_mask(composite, [BAND])
    assert composite.loc[mask, "covered"].tolist() == [0, 0]
    assert composite.loc[~mask].equals(rows.loc[~mask])


def test_a_moved_extra_column_is_a_replay_miss() -> None:
    """Mutation control: a column outside ``REPLICATE_COLUMNS`` is held to the rule too."""
    rows = committed()
    with pytest.raises(band_rerun.ReplayMiss, match="rounds"):
        band_rerun.replay(
            [call()], {"properties": refitted(rounds=4.0)}, band_rerun.committed_index(rows), [BAND]
        )


def test_a_moved_non_band_row_is_a_replay_miss() -> None:
    """Mutation control: a refit whose point estimate moves takes a complete rerun."""
    rows = committed()
    with pytest.raises(band_rerun.ReplayMiss):
        band_rerun.replay(
            [call()], {"properties": refitted(1e-4)}, band_rerun.committed_index(rows), [BAND]
        )


def failure(inside: bool) -> band_rerun.KeyFailure:
    return band_rerun.KeyFailure(inside, "ValueError: boom", ("m:f",))


def test_a_failure_inside_the_band_is_a_reading_and_outside_is_drift() -> None:
    rows = committed()
    index = band_rerun.committed_index(rows)
    values = [failure(True), refitted()[1]]
    refit = band_rerun.replay([call()], {"properties": values}, index, [BAND])
    assert refit.band_failures == {BAND: 1}
    assert [int(r["covered"]) for r in refit.band_rows] == [0, 0]
    with pytest.raises(band_rerun.ReplayMiss, match="outside the band"):
        band_rerun.replay([call()], {"properties": [failure(False), values[1]]}, index, [BAND])


def raised(function: Any, *args: Any) -> BaseException:
    with pytest.raises(Exception) as caught:
        function(*args)
    return caught.value


def test_a_raise_inside_the_band_classifies_inside() -> None:
    from cleverly.inference.multiplier import simultaneous_bands

    assert band_rerun.classify(raised(simultaneous_bands, {})).inside_band


def test_a_learner_failure_under_a_band_fit_wrapper_classifies_outside() -> None:
    """``dose_response_properties.band_fit`` is a fit wrapper, not a band entry point."""
    from tests.studies import dose_response_properties

    frame = pd.DataFrame({"W": [0.0, 1.0], "A": [0.5, 1.5], "Y": [0.0, 1.0]})
    failure = band_rerun.classify(raised(dose_response_properties.band_fit, frame))
    assert any(frame_name.endswith(":band_fit") for frame_name in failure.frames)
    assert not failure.inside_band


def test_a_bootstrap_band_failure_classifies_outside() -> None:
    from cleverly.inference.bootstrap import bootstrap_band

    failure = band_rerun.classify(
        raised(lambda: bootstrap_band(None, {"a": 1.0}, {"a": 0.1}, alpha=0.05))
    )
    assert any(frame_name.endswith(":bootstrap_band") for frame_name in failure.frames)
    assert not failure.inside_band


def test_refusals_count_per_row() -> None:
    rows = committed()
    refused = row(BAND, 0, 2.5, covered=0, rejected=1, std_error=1.5)
    refused["estimate"] = refused["std_error"] + 1.0
    values = [[row(POINT, 0, 0.3, rounds=3.0), refused], refitted()[1]]
    refit = band_rerun.replay(
        [call()], {"properties": values}, band_rerun.committed_index(rows), [BAND]
    )
    assert refit.refused == {BAND: 1}


def test_the_splice_refuses_a_new_band_row() -> None:
    with pytest.raises(ValueError, match="new or repeated"):
        band_rerun.splice(committed(), [row(BAND, 7, 2.5)])


# ---------------------------------------------------------------------------- the summary


SUMMARY = (
    "property,cell,role,coverage,passed,property_passed\n"
    "point,a,positive,0.95,True,True\n"
    "simultaneous_coverage,a__simultaneous_band,positive,0.95,True,True\n"
)


def test_an_unmoved_summary_row_keeps_its_bytes_and_a_moved_one_changes() -> None:
    new = SUMMARY.replace("0.95,True,True\n", "0.9500000000000001,True,True\n", 1)
    new = new.replace("a__simultaneous_band,positive,0.95", "a__simultaneous_band,positive,0.91")
    text, record = band_rerun.compose_summary(SUMMARY, new, [BAND])
    assert text.splitlines()[1] == SUMMARY.splitlines()[1]
    assert text.splitlines()[2].endswith("0.91,True,True")
    assert record["changed_rows"] == [BAND]
    assert 0 < record["maximum_carried_difference"] <= band_rerun.SUMMARY_TOLERANCE


@pytest.mark.parametrize(
    "mutation", [("0.95,True,True\n", "0.951,True,True\n"), (",True,True\n", ",False,True\n")]
)
def test_a_moved_unmoved_summary_row_stops_the_study(mutation: tuple[str, str]) -> None:
    with pytest.raises(band_rerun.SummaryMiss):
        band_rerun.compose_summary(SUMMARY, SUMMARY.replace(*mutation, 1), [BAND])


# ---------------------------------------------------------------------------- the decision


def test_the_mcnemar_p_value_is_the_exact_binomial_tail() -> None:
    assert band_rerun.mcnemar_less(0, 0) == 1.0
    assert band_rerun.mcnemar_less(3, 0) == pytest.approx(0.125)
    assert band_rerun.mcnemar_less(10, 2) == pytest.approx(79 / 4096)


def test_discordant_pairs_are_read_by_replicate() -> None:
    before = committed()
    after = band_rerun.splice(before, [row(BAND, 1, 2.5, covered=0, rejected=1)])
    assert band_rerun.discordant(before, after, BAND) == (1, 0, 2)


def interval(lower: float, upper: float, coverage: float = 0.95) -> dict[str, float]:
    return {"coverage_ci_lower": lower, "coverage_ci_upper": upper, "coverage": coverage}


def test_the_paired_test_reads_the_direction_of_the_breach() -> None:
    assert band_rerun.breach_direction("positive", interval(0.91, 0.94), 0.95) == "less"
    assert band_rerun.breach_direction("positive", interval(0.96, 0.99), 0.95) == "more"
    assert band_rerun.breach_direction("control", interval(0.93, 0.96, 0.95), 0.90) == "more"
    assert band_rerun.breach_direction("control", interval(0.80, 0.86, 0.83), 0.90) == "less"


def test_the_two_rules_block_as_declared() -> None:
    assert band_rerun.blocks("strict", True, 0.5)
    assert not band_rerun.blocks("paired", True, 0.5)
    assert band_rerun.blocks("paired", True, 0.01)
    assert not band_rerun.blocks("strict", False, 0.0)
    with pytest.raises(ValueError):
        band_rerun.blocks("other", True, 0.0)


def test_no_wave_runs_before_the_owner_binds_the_rule(
    plan: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan commits ``decision_rule: null`` until the owner picks the blocking clause."""
    assert "decision_rule" in plan and plan["decision_rule"] in (None, *band_rerun.RULES)
    if plan["decision_rule"] is not None:
        return
    monkeypatch.setattr(band_rerun, "validate_plan", lambda plan: [])
    with pytest.raises(SystemExit, match="decision_rule"):
        band_rerun.preflight(plan, "canonical-ltmle", scientific=True)
    with pytest.raises(ValueError, match="decision_rule"):
        band_rerun.decide(tmp_path)
    with pytest.raises(SystemExit):
        band_rerun.main(["--decide", str(tmp_path)])


def test_the_closed_interval_test() -> None:
    reading = {"coverage_ci_lower": 0.92, "coverage_ci_upper": 0.98}
    assert band_rerun.interval_red(reading) is False
    assert band_rerun.interval_red({**reading, "coverage_ci_lower": 0.9199}) is True
    assert band_rerun.interval_red({**reading, "coverage_ci_upper": 0.9801}) is True


def test_a_stamped_cache_refuses_another_run(tmp_path: Path) -> None:
    directory = band_rerun.cache_directory(tmp_path, "s", "a" * 40, "b" * 64)
    assert directory == tmp_path / "s" / ("a" * 40) / ("b" * 64)
    (directory / "stamp.json").write_text('{"slug": "other"}')
    with pytest.raises(SystemExit, match="stamped"):
        band_rerun.cache_directory(tmp_path, "s", "a" * 40, "b" * 64)


# ---------------------------------------------------------------------------- the composite


def operation() -> dict[str, Any]:
    subject = {"cleverly_commit": "a" * 40, "cleverly_worktree_clean": True, "numpy": "9.0"}
    out: dict[str, Any] = {
        "subject": subject,
        "source_sha256": {"src/cleverly/__init__.py": "0" * 64},
        "distribution_versions": [["numpy", "9.0"]],
        "distribution_versions_sha256": "1" * 64,
    }
    out["producer_sha256"] = band_rerun.digest(band_rerun.json_bytes(out))
    return out


def composite_fixture() -> tuple[dict[str, Any], bytes]:
    rows = committed()
    baseline = {
        "slug": "synthetic",
        "study": "synthetic",
        "schema_version": 1,
        "generated_with": {"subject": {"numpy": "8.0"}},
        "configuration": {},
        "sha256": {
            "property-replicates.parquet": "2" * 64,
            "properties.csv": "3" * 64,
            "replicates.parquet": "4" * 64,
        },
    }
    baseline_raw = band_rerun.json_bytes(baseline)
    selection = [{"label": "properties", "band_keys": 2, "cells": {BAND: 2}}]
    entry = {
        "manifest_sha256": band_rerun.digest(baseline_raw),
        "selection": selection,
        "band_fits": 2,
        "moved_cells": [BAND],
        "baseline_subject": {"numpy": "8.0"},
        "carried_rows_sha256": band_rerun.carried_digest(rows, [BAND]),
        "cells": [{"cell": "a__simultaneous_band", "replicates": 2}],
    }
    plan_raw = band_rerun.json_bytes(
        {"baseline_commit": band_rerun.BASELINE_COMMIT, "studies": {"synthetic": entry}}
    )
    refit = band_rerun.replay(
        [call()], {"properties": refitted()}, band_rerun.committed_index(rows), [BAND]
    )
    composite = band_rerun.Composite(
        rows=band_rerun.splice(rows, refit.band_rows),
        summary="",
        refit=refit,
        readings=[{"cell": "a__simultaneous_band"}],
        summary_record={
            "summary_tolerance": band_rerun.SUMMARY_TOLERANCE,
            "maximum_carried_difference": 0.0,
            "changed_rows": [BAND],
        },
    )
    hashes = dict(baseline["sha256"], **{"property-replicates.parquet": "5" * 64})
    manifest = band_rerun.composite_manifest(
        baseline_raw,
        plan_raw=plan_raw,
        slug="synthetic",
        operation=operation(),
        artifact_hashes=hashes,
        composite=composite,
        command=["python", "-m", "tests.studies.evidence.band_rerun"],
    )
    return manifest, plan_raw


def test_a_composite_manifest_passes_its_gate() -> None:
    manifest, plan_raw = composite_fixture()
    band_rerun.validate_composite_manifest(manifest, plan_raw)
    assert manifest["composite"]["replacement"]["declared_stack"] == {
        "numpy": {"baseline": "8.0", "ran": "9.0"}
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "baseline_artifact",
        "restamped_producer",
        "carried_receipt",
        "replay_rule",
        "selection",
        "dirty_tree",
        "summary_drift",
    ],
)
def test_the_composite_gate_refuses_each_mutation(mutation: str) -> None:
    manifest, plan_raw = composite_fixture()
    altered = copy.deepcopy(manifest)
    section = altered["composite"]
    if mutation == "baseline_artifact":
        altered["sha256"]["replicates.parquet"] = "6" * 64
    elif mutation == "restamped_producer":
        altered["generated_with"] = {"subject": {"numpy": "9.0"}}
    elif mutation == "carried_receipt":
        section["carried_property_rows_sha256"] = "7" * 64
    elif mutation == "replay_rule":
        section["replacement"]["replay"]["tolerance"] = 1e-6
    elif mutation == "selection":
        section["replacement"]["band_fits"] = 3
    elif mutation == "summary_drift":
        section["analysis"]["maximum_carried_difference"] = 1e-6
    else:
        section["analysis"]["subject"]["cleverly_worktree_clean"] = False
    with pytest.raises(ValueError):
        band_rerun.validate_composite_manifest(altered, plan_raw)


def test_a_sidecar_with_an_altered_file_is_refused(tmp_path: Path, plan: dict[str, Any]) -> None:
    """Mutation control on the blocked sidecar: its files are held to their recorded hashes."""
    slug, entry = next(iter(plan["studies"].items()))
    directory = ROOT / entry["artifact_directory"]
    side = tmp_path / band_rerun.SIDECAR
    side.mkdir()
    for name in band_rerun.SIDECAR_FILES:
        (side / name).write_bytes(b"rows\n")
    record = {
        "mode": band_rerun.MODE,
        "blocked": True,
        "slug": slug,
        "baseline_commit": band_rerun.BASELINE_COMMIT,
        "plan_sha256": "8" * 64,
        "sha256": {name: band_rerun.digest(b"rows\n") for name in band_rerun.SIDECAR_FILES},
    }
    (side / "record.json").write_text(json.dumps(record))
    (tmp_path / "manifest.json").write_bytes((directory / "manifest.json").read_bytes())
    findings = band_rerun.validate_sidecar(tmp_path, plan)
    assert findings == [f"{slug}: the sidecar sits in another study's directory"]
    (side / band_rerun.SIDECAR_FILES[0]).write_bytes(b"other rows\n")
    findings = band_rerun.validate_sidecar(tmp_path, plan)
    assert f"{slug}: {band_rerun.SIDECAR_FILES[0]} differs from its recorded hash" in findings


# ---------------------------------------------------------------------------- a complete rerun


SLUG = "synthetic"
STUDY = "tests/canonical/synthetic"
WAVE_COMMIT = "c" * 40
REPLICATES = 20
#: The study sources a regeneration never writes.
SOURCES = ("README.md", "__init__.py", "regenerate.py")


def study_rows(covered: list[int], point_covered: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        [row(POINT, r, 0.3, rounds=3.0, covered=point_covered) for r in range(REPLICATES)]
        + [row(BAND, r, 1.5, covered=flag, rejected=1 - flag) for r, flag in enumerate(covered)]
    )


def study_summary(
    coverage: float, lower: float, upper: float, *, band_passed: bool, point_passed: bool = True
) -> pd.DataFrame:
    base = {"role": "positive", "n": 500, "replicates": REPLICATES}
    return pd.DataFrame(
        [
            {
                **base,
                "property": "point",
                "cell": "a",
                "passed": point_passed,
                "coverage": 0.95,
                "coverage_ci_lower": 0.93,
                "coverage_ci_upper": 0.97,
                "property_passed": point_passed,
            },
            {
                **base,
                "property": "simultaneous_coverage",
                "cell": "a__simultaneous_band",
                "passed": band_passed,
                "coverage": coverage,
                "coverage_ci_lower": lower,
                "coverage_ci_upper": upper,
                "property_passed": band_passed,
            },
        ]
    )


def write_regeneration(
    directory: Path,
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    commit: str = WAVE_COMMIT,
    clean: bool = True,
    slug: str = SLUG,
    extra: tuple[str, ...] = (),
    numpy: str = "2.0",
) -> bytes:
    """What ``tests/canonical/regenerate.py:main`` leaves: the artifacts and their manifest."""
    from tests.studies.evidence.manifest import write_csv, write_rows

    directory.mkdir(parents=True, exist_ok=True)
    write_rows(rows, directory / "property-replicates.parquet")
    write_csv(summary, directory / "properties.csv")
    for name in (*band_rerun.REGENERATION_ARTIFACTS, *extra):
        if name.endswith(".csv") and name != "properties.csv":
            (directory / name).write_text(f"check,passed\n{name},{commit[0]}\n")
        elif name.endswith(".parquet") and name != "property-replicates.parquet":
            write_rows(
                pd.DataFrame({"replicate": [0, 1], "source": [name, commit]}), directory / name
            )
    names = (*band_rerun.REGENERATION_ARTIFACTS, *extra)
    manifest = {
        "schema_version": 2,
        "study": "Synthetic",
        "slug": slug,
        "generated_with": {
            "subject": {
                "cleverly_commit": commit,
                "cleverly_worktree_clean": clean,
                "numpy": numpy,
            },
            "reference": None,
        },
        "sha256": {name: band_rerun.digest((directory / name).read_bytes()) for name in names},
    }
    raw = band_rerun.json_bytes(manifest)
    (directory / "manifest.json").write_bytes(raw)
    return raw


BASELINE_COVERED = [1] * (REPLICATES - 1) + [0]


class Wave:
    """A fabricated wave of one planned study whose targeted run missed the replay rule."""

    def __init__(self, root: Path, wave: Path) -> None:
        self.study = root / STUDY
        self.wave = wave
        self.output = wave / SLUG
        baseline_rows = study_rows(BASELINE_COVERED)
        summary = study_summary(0.95, 0.75, 0.999, band_passed=True)
        # The committed baseline: a regeneration from another commit, with one optional file.
        raw = write_regeneration(
            self.study,
            baseline_rows,
            summary,
            commit="b" * 40,
            clean=False,
            extra=("failures.csv",),
        )
        for name in SOURCES:
            (self.study / name).write_text(f"{name}\n")
        entry = {
            "artifact_directory": STUDY,
            "manifest_sha256": band_rerun.digest(raw),
            "moved_cells": [BAND],
            "cells": band_rerun.moved_readings(summary, [BAND]),
        }
        self.plan: dict[str, Any] = {
            "decision_rule": "paired",
            "baseline_commit": band_rerun.BASELINE_COMMIT,
            "studies": {SLUG: entry},
        }
        self.plan_raw = band_rerun.json_bytes(self.plan)
        self.plan_path = root / "band-rerun-plan.json"
        self.plan_path.write_bytes(self.plan_raw)
        self.output.mkdir(parents=True)
        (self.output / band_rerun.MISS).write_bytes(
            band_rerun.json_bytes({"miss": "point/a/r3: estimate", "started": "t", "study": SLUG})
        )

    def complete(self, covered: list[int], summary: pd.DataFrame, **options: Any) -> Path:
        directory = self.output / band_rerun.COMPLETE
        rows = study_rows(covered, options.pop("point", 1))
        drop = options.pop("drop", None)
        if drop is not None:
            rows = rows.loc[~(band_rerun.band_mask(rows, [BAND]) & (rows["replicate"] == drop))]
        write_regeneration(directory, rows.reset_index(drop=True), summary, **options)
        return directory

    def decide(self) -> dict[str, Any]:
        return band_rerun.decide(self.wave, plan_raw=self.plan_raw)


@pytest.fixture
def wave(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Wave:
    import types

    fabricated = Wave(tmp_path / "root", tmp_path / "wave")
    monkeypatch.setattr(band_rerun, "ROOT", tmp_path / "root")
    monkeypatch.setattr(band_rerun, "PLAN_PATH", fabricated.plan_path)
    monkeypatch.setattr(
        band_rerun,
        "record_for",
        lambda slug: types.SimpleNamespace(slug=slug, artifacts=fabricated.study),
    )
    monkeypatch.setattr(band_rerun, "descends_from_flip", lambda commit: commit == WAVE_COMMIT)
    return fabricated


def green_rerun(wave: Wave) -> Path:
    """A complete rerun whose band cell trades one replicate each way and stays green, and
    whose non-band cell turns red."""
    covered = [0] + [1] * (REPLICATES - 2) + [1]
    return wave.complete(
        covered, study_summary(0.95, 0.75, 0.999, band_passed=True, point_passed=False)
    )


def red_rerun(wave: Wave) -> Path:
    """A complete rerun whose band cell loses twelve replicates: red, and paired p <= 0.01."""
    covered = [0] * 12 + [1] * (REPLICATES - 13) + [0]
    return wave.complete(covered, study_summary(0.35, 0.15, 0.59, band_passed=False))


def test_a_complete_rerun_is_read_and_decided(wave: Wave) -> None:
    complete = green_rerun(wave)
    decision = wave.decide()
    assert decision["decision"] == "flip" and decision["commit"] == WAVE_COMMIT
    reading = decision["readings"][SLUG]
    assert reading["route"] == band_rerun.COMPLETE_ROUTE
    (cell,) = reading["cells"]
    assert (cell["b_covered_before_only"], cell["c_covered_now_only"]) == (1, 1)
    assert cell["mcnemar_p_covers_less"] == pytest.approx(0.75)
    assert cell["band_failures"] == 0 and cell["lost"] is False
    assert reading["manifest_sha256"] == band_rerun.digest(
        (complete / "manifest.json").read_bytes()
    )
    assert reading["complete_rerun_changes"] == {
        "cells": [{"cell": POINT, "baseline_passed": "True", "passed": "False"}],
        "properties": [
            {
                "property": "point",
                "baseline_property_passed": ["True"],
                "property_passed": ["False"],
            }
        ],
    }
    assert decision["would_block"] == [] and reading["failure_files"] == []
    assert band_rerun.json_bytes(wave.decide()) == band_rerun.json_bytes(decision)


def test_a_red_complete_rerun_blocks_under_the_paired_rule(wave: Wave) -> None:
    red_rerun(wave)
    decision = wave.decide()
    (cell,) = decision["readings"][SLUG]["cells"]
    assert (cell["b_covered_before_only"], cell["c_covered_now_only"]) == (12, 0)
    assert cell["breach"] == "less" and cell["interval_red"] is True
    assert decision["decision"] == "blocked" and len(decision["would_block"]) == 1


def test_a_flipped_band_replicate_of_a_complete_rerun_moves_b_or_c(wave: Wave) -> None:
    """Mutation control: the reading is by replicate on the band rows, not on the summary."""
    summary = study_summary(0.95, 0.75, 0.999, band_passed=True)

    def counts(covered: list[int], point: int = 1) -> tuple[int, int]:
        wave.complete(covered, summary, point=point)
        (cell,) = wave.decide()["readings"][SLUG]["cells"]
        return cell["b_covered_before_only"], cell["c_covered_now_only"]

    assert counts(BASELINE_COVERED) == (0, 0)
    assert counts(BASELINE_COVERED, point=0) == (0, 0)
    assert counts([0, *BASELINE_COVERED[1:]]) == (1, 0)
    assert counts([*BASELINE_COVERED[:-1], 1]) == (0, 1)


def test_a_complete_rerun_without_a_replay_miss_is_refused(wave: Wave) -> None:
    green_rerun(wave)
    (wave.output / band_rerun.MISS).unlink()
    with pytest.raises(ValueError, match="replay miss"):
        wave.decide()


def test_a_replay_miss_without_a_complete_rerun_is_refused(wave: Wave) -> None:
    with pytest.raises(ValueError, match="no complete rerun"):
        wave.decide()


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("commit", "does not descend"),
        ("hash", "summary.csv differs"),
        ("dirty", "unclean tree"),
        ("slug", "another study"),
        ("miss", "miss names another study"),
        ("composite", "not a normal regeneration"),
    ],
)
def test_the_complete_route_refuses_each_mutation(wave: Wave, mutation: str, message: str) -> None:
    summary = study_summary(0.95, 0.75, 0.999, band_passed=True)
    mutations: dict[str, dict[str, Any]] = {
        "commit": {"commit": "d" * 40},
        "dirty": {"clean": False},
        "slug": {"slug": "other"},
    }
    options = mutations.get(mutation, {})
    complete = wave.complete(BASELINE_COVERED, summary, **options)
    if mutation == "hash":
        with (complete / "summary.csv").open("a") as handle:
            handle.write("tampered,True\n")
    elif mutation == "miss":
        (wave.output / band_rerun.MISS).write_text('{"study": "other"}')
    elif mutation == "composite":
        manifest = json.loads((complete / "manifest.json").read_bytes())
        manifest["generation_mode"] = band_rerun.MODE
        (complete / "manifest.json").write_bytes(band_rerun.json_bytes(manifest))
    with pytest.raises(ValueError, match=message):
        wave.decide()


def test_a_complete_rerun_from_another_commit_than_the_wave_is_refused(
    wave: Wave, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One commit for the whole wave: a targeted study at the wave's commit and a complete
    rerun at another descendant of the flip do not decide together."""
    other = "e" * 40
    wave.complete(
        BASELINE_COVERED, study_summary(0.95, 0.75, 0.999, band_passed=True), commit=other
    )
    wave.plan["studies"]["targeted"] = wave.plan["studies"][SLUG]
    wave.plan_raw = band_rerun.json_bytes(wave.plan)
    (wave.wave / "targeted").mkdir()
    (wave.wave / "targeted" / "manifest.json").write_text("{}")
    monkeypatch.setattr(
        band_rerun, "validate_output", lambda slug, output, plan_raw: {"source_commit": WAVE_COMMIT}
    )
    monkeypatch.setattr(band_rerun, "descends_from_flip", lambda commit: True)
    with pytest.raises(ValueError, match="2 commits"):
        wave.decide()


def test_a_study_with_both_layouts_is_refused(wave: Wave) -> None:
    green_rerun(wave)
    (wave.output / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="both a composite and a complete rerun"):
        wave.decide()
    with pytest.raises(ValueError, match="both a composite and a complete rerun"):
        band_rerun.validate_output(SLUG, wave.output, wave.plan_raw)


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_a_flip_installs_the_complete_rerun_as_its_regeneration_wrote_it(wave: Wave) -> None:
    complete = green_rerun(wave)
    (wave.wave / band_rerun.DECISION).write_bytes(band_rerun.json_bytes(wave.decide()))
    sources = {name: (wave.study / name).read_bytes() for name in SOURCES}
    assert band_rerun.install_wave(wave.wave) == "flip"
    installed = snapshot(wave.study)
    record = installed.pop(f"{band_rerun.SIDECAR}/{band_rerun.COMPLETE_RECORD}")
    assert installed == {**snapshot(complete), **sources}
    assert not (wave.study / "failures.csv").exists()
    assert (
        json.loads(record)["baseline"]["manifest_sha256"]
        == wave.plan["studies"][SLUG]["manifest_sha256"]
    )
    assert band_rerun.installed_state(wave.plan) == {SLUG: "composite"}
    raw = band_rerun.baseline_manifest_raw(band_rerun.record_for(SLUG))
    assert band_rerun.digest(raw) == wave.plan["studies"][SLUG]["manifest_sha256"]


def test_install_re_derives_the_decision_of_a_complete_rerun(wave: Wave) -> None:
    complete = green_rerun(wave)
    (wave.wave / band_rerun.DECISION).write_bytes(band_rerun.json_bytes(wave.decide()))
    before = snapshot(wave.study)
    red_rerun(wave)
    assert complete.exists()
    with pytest.raises(SystemExit, match="no longer gives its recorded decision"):
        band_rerun.install_wave(wave.wave)
    assert snapshot(wave.study) == before


def test_a_blocked_wave_puts_the_complete_rerun_in_the_sidecar(wave: Wave) -> None:
    complete = red_rerun(wave)
    (wave.wave / band_rerun.DECISION).write_bytes(band_rerun.json_bytes(wave.decide()))
    before = snapshot(wave.study)
    assert band_rerun.install_wave(wave.wave) == "blocked"
    after = snapshot(wave.study)
    side = {name: raw for name, raw in after.items() if name.startswith(f"{band_rerun.SIDECAR}/")}
    assert {name: raw for name, raw in after.items() if name not in side} == before
    assert band_rerun.installed_state(wave.plan) == {SLUG: "sidecar"}
    assert band_rerun.validate_sidecar(wave.study, wave.plan) == []
    record = json.loads(side[f"{band_rerun.SIDECAR}/record.json"])
    assert record["route"] == band_rerun.COMPLETE_ROUTE
    manifest = json.loads((complete / "manifest.json").read_bytes())
    assert set(record["sha256"]) == set(band_rerun.sidecar_files(record["route"], manifest))
    assert "complete-replicates.parquet" in record["sha256"]
    assert "complete-properties.csv" not in record["sha256"]
    for name in record["sha256"]:
        if name.endswith(".csv") and name != band_rerun.SIDECAR_FILES[2]:
            header = (wave.study / band_rerun.SIDECAR / name).read_text().splitlines()[0]
            assert "diagnostic_passed" in header.split(",") and "passed" not in header.split(","), (
                name
            )
    (wave.study / band_rerun.SIDECAR / "complete-summary.csv").write_text("check\n")
    assert band_rerun.validate_sidecar(wave.study, wave.plan) == [
        f"{SLUG}: complete-summary.csv differs from its recorded hash"
    ]


def test_a_replicate_the_complete_rerun_lost_is_not_covered_there(wave: Wave) -> None:
    """A covered baseline replicate the rerun lacks adds to b, and the pairs span the union."""
    summary = study_summary(0.95, 0.75, 0.999, band_passed=True)
    wave.complete(BASELINE_COVERED, summary, drop=0)
    (cell,) = wave.decide()["readings"][SLUG]["cells"]
    assert (cell["b_covered_before_only"], cell["c_covered_now_only"]) == (1, 0)
    assert cell["replicates"] == REPLICATES
    assert (cell["replicates_lost_now"], cell["replicates_lost_before"]) == (1, 0)
    # The baseline's uncovered replicate lost by the rerun moves neither count.
    wave.complete(BASELINE_COVERED, summary, drop=REPLICATES - 1)
    (cell,) = wave.decide()["readings"][SLUG]["cells"]
    assert (cell["b_covered_before_only"], cell["c_covered_now_only"]) == (0, 0)
    assert cell["replicates_lost_now"] == 1


def test_the_targeted_route_still_requires_the_same_replicates() -> None:
    before = committed()
    after = before.loc[~(band_rerun.band_mask(before, [BAND]) & (before["replicate"] == 0))]
    with pytest.raises(ValueError, match="replicates differ"):
        band_rerun.discordant(before, after, BAND)
    assert band_rerun.discordant_union(before, after, BAND) == (1, 0, 2, 1, 0)


def test_a_complete_rerun_records_its_stack_against_the_baseline(wave: Wave) -> None:
    summary = study_summary(0.95, 0.75, 0.999, band_passed=True)
    wave.complete(BASELINE_COVERED, summary)
    assert wave.decide()["readings"][SLUG]["stack_differences"] == {}
    wave.complete(BASELINE_COVERED, summary, numpy="2.1")
    assert wave.decide()["readings"][SLUG]["stack_differences"] == {
        "numpy": {"baseline": "2.0", "ran": "2.1"}
    }
