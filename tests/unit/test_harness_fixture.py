"""The fast-tier gate over the RM39 harness smoke's committed evidence (n5).

CI installs no R, so the harness's behavior is tested by ``python -m
tests.canonical.harness_fixture.smoke``, which runs it in Docker and commits what it saw.  This
module gates that evidence, not the behavior:

* the manifest names the current sha256 of the harness, both fixture runners, the smoke and the
  ``tmle3`` Dockerfile, so an edit to any of them fails until the smoke runs again;
* each committed output still has its recorded sha256;
* each recorded fact is the one its case asserts: a killed group is retried and the table is
  the no-kill table, a resumed run writes the uninterrupted run's bytes, and so on.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.canonical.harness_fixture import smoke
from tests.studies.evidence.registry import ROOT


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, Any]:
    assert smoke.MANIFEST.exists(), (
        "no fixture-manifest.json: run python -m tests.canonical.harness_fixture.smoke"
    )
    return json.loads(smoke.MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def facts() -> dict[str, Any]:
    _manifest()
    return json.loads((smoke.OUTPUTS / "results.json").read_text(encoding="utf-8"))


def test_the_evidence_describes_the_current_inputs() -> None:
    recorded = _manifest()["inputs"]
    current = {path.relative_to(ROOT).as_posix(): _sha(path) for path in smoke.INPUTS}
    stale = sorted(name for name in current if recorded.get(name) != current[name])
    assert not stale, (
        f"{stale} changed after the harness smoke ran. Run "
        f"python -m tests.canonical.harness_fixture.smoke and commit its outputs"
    )
    assert set(recorded) == set(current)


def test_each_committed_output_has_its_recorded_sha256() -> None:
    outputs = _manifest()["outputs"]
    assert "smoke/results.json" in outputs
    for name, digest in outputs.items():
        assert _sha(smoke.HERE / name) == digest, name


def test_the_manifest_records_the_image_and_the_date() -> None:
    manifest = _manifest()
    assert manifest["image"].startswith("sha256:")
    assert manifest["date"]


def test_every_fit_forks(facts: dict[str, Any]) -> None:
    """B7: no fit runs in the parent, at one worker, at several, for one chunk, for a solo
    second attempt, and under a runner-owned one-core ``mclapply``."""
    for case in (
        "plain",
        "forks_one_worker",
        "forks_several_workers",
        "forks_one_chunk",
        "forks_onecore_mclapply",
        "kill",
    ):
        assert facts[case]["forked"], case
    assert facts["kill"]["solo_second_attempt_forked"]


def test_a_killed_group_is_retried_and_the_table_is_unchanged(facts: dict[str, Any]) -> None:
    plain = facts["plain"]["table"]
    for case in (
        "kill",
        "preschedule_kill",
        "killed_in_loop",
        "concurrency_halves",
        "stream_retry",
        "forks_one_chunk",
        "forks_onecore_mclapply",
    ):
        assert facts[case]["exit"] == 0, case
        assert facts[case]["table"] == plain, case
    assert facts["kill"]["attempts"] == 2
    # Under preschedule, the kill loses only the group in flight; the slice's unstarted
    # groups run in the loop.
    assert int(facts["preschedule_kill"]["passes"][0].split()[1]) > 1
    # A group first started in the loop and killed there is retried alone.
    assert facts["killed_in_loop"]["attempts_g7"] == 2


def test_the_pass_after_a_kill_halves_the_workers(facts: dict[str, Any]) -> None:
    passes = facts["concurrency_halves"]["passes"]
    assert any("1 killed" in line and "workers 2," in line for line in passes)
    stream = facts["stream_retry"]["passes"]
    killed = next(index for index, line in enumerate(stream) if "1 killed" in line)
    assert "workers 2," in stream[killed]
    # The halved count carries into the next chunk.
    assert all("workers 2," in line for line in stream[killed + 1 :])
    assert len(stream) > killed + 1


def test_a_second_kill_and_a_pass_without_progress_stop_with_exit_3(
    facts: dict[str, Any],
) -> None:
    assert facts["killed_twice"]["exit"] == 3
    assert facts["killed_twice"]["names_id"] and facts["killed_twice"]["memory_line"]
    assert facts["no_progress"]["exit"] == 3 and facts["no_progress"]["message"]


def test_no_retry_makes_the_first_kill_fatal(facts: dict[str, Any]) -> None:
    """n20: the RM19 wrapper's switch."""
    assert facts["no_retry"]["exit"] not in (0, 3)
    assert facts["no_retry"]["message"]


def test_an_error_is_fatal_and_a_resume_clears_it(facts: dict[str, Any]) -> None:
    error = facts["error_fatal"]
    assert error["exit"] == 1 and error["message"]
    assert error["checkpoints_kept"] > 0 and error["fatal_written"]
    resumed = facts["error_fatal_resume"]
    assert resumed["exit"] == 0 and resumed["cleared_fatal"]
    assert resumed["table"] == facts["plain"]["table"]


def test_a_resumed_run_writes_the_uninterrupted_bytes(facts: dict[str, Any]) -> None:
    """Q5: the fixture draws a random number in each fit, so an unseeded draw fails here."""
    full = facts["random_full"]["table"]
    assert facts["random_full"]["exit"] == 0
    assert facts["random_one_core"]["table"] == full
    for case in ("resume_fitter", "resume_stream"):
        assert facts[case]["stopped_exit"] == 137, case
        assert facts[case]["exit"] == 0 and facts[case]["reused"], case
        assert facts[case]["table"] == full, case
        # Stale start markers are cleared without counting as kills.
        assert facts[case]["stale_markers"] > 0, case
        assert facts[case]["cleared_markers"] == facts[case]["stale_markers"], case
        assert facts[case]["kills_counted"] == 0, case


def test_a_torn_checkpoint_is_deleted_and_refitted(facts: dict[str, Any]) -> None:
    torn = facts["torn"]
    assert torn["first_exit"] == 0 and torn["exit"] == 0 and torn["deleted"]
    assert torn["table"] == facts["plain"]["table"]


def test_the_old_and_new_harness_write_the_same_bytes(facts: dict[str, Any]) -> None:
    old_new = facts["old_new"]
    assert old_new["old_fitter"] == facts["plain"]["table"]
    assert old_new["old_stream"] == old_new["new_stream"] == facts["plain"]["table"]


def test_a_whole_container_kill_is_run_again_once(facts: dict[str, Any]) -> None:
    """Q-h: the driver re-invokes with a worker cap, and a second kill stops."""
    once = facts["container_kill_1"]
    assert once["complete"] and once["killed_logged"] and once["cap_passed"]
    assert once["table"] == facts["plain"]["table"]
    twice = facts["container_kill_2"]
    assert twice["stopped_twice"] and not twice["complete"]


def test_calibration_sees_the_copy_on_write_cost(facts: dict[str, Any]) -> None:
    """B5: a bigger parent heap raises the per-worker cost and lowers the worker count."""
    rows = sorted(facts["memory_cow"], key=lambda row: row["heap_mb"])
    assert all(row["exit"] == 0 for row in rows)
    private = [row["private_after_gc"] for row in rows]
    per_worker = [row["per_worker"] for row in rows]
    workers = [row["workers_at_6g"] for row in rows]
    assert private == sorted(private) and private[-1] > private[0] + 1000
    assert per_worker == sorted(per_worker)
    assert workers == sorted(workers, reverse=True) and workers[-1] < workers[0]


def test_calibration_sees_a_transient_peak(facts: dict[str, Any]) -> None:
    """B6: a 400 MB allocation freed before the fit returns still sets the per-worker cost.

    A metric that read only private memory after collection would miss it.
    """
    transient = facts["memory_transient"]
    assert transient["exit"] == 0
    groups = {group["id"]: group for group in transient["groups"]}
    assert len(groups) == 2  # the first group of each scenario
    heaviest = max(groups.values(), key=lambda group: group["per_worker"])
    lightest = min(groups.values(), key=lambda group: group["per_worker"])
    assert heaviest["peak_growth"] > 400
    assert heaviest["private_after_gc"] < 400
    for group in groups.values():
        assert group["per_worker"] == pytest.approx(
            group["peak_growth"] + group["private_after_gc"], abs=0.01
        )
    assert lightest["peak_growth"] < 100


def test_the_worker_plan_arithmetic(facts: dict[str, Any]) -> None:
    plan = facts["memory_plan"]
    limited = plan["limited"]
    expected = max(1, min(8, int(0.85 * limited["available"] // (1.25 * 500))))
    assert limited["workers"] == expected
    assert limited["available"] <= 2048
    # Under a 2 GB limit, 500 MB a worker binds before 8 cores do.
    assert expected < 8 and limited["reason"] == "memory"
    assert plan["unlimited_no_cgroup_limit"]
    assert plan["capped"]["workers"] == 3 and plan["capped"]["reason"] == "worker cap"
    override = plan["override"]
    assert override["factor"] == 2.0 and override["share"] == 0.5
    assert plan["override_reason_logged"]


def test_the_run_log_carries_every_field(facts: dict[str, Any]) -> None:
    missing = [label for label, present in facts["run_log"].items() if not present]
    assert not missing


def test_the_fork_cost_is_measured(facts: dict[str, Any]) -> None:
    cost = facts["fork_cost"]
    assert cost["0"]["forked"] and cost["1500"]["forked"]
    assert cost["0"]["seconds_per_fit"] >= 0 and cost["1500"]["seconds_per_fit"] >= 0


# ---- RM39a2: the comparator stage and its failure records (plan 7.1, rev 2 M8) ----

#: The record columns the harness writes; the driver adds ``implementation`` in front.
RECORD_COLUMNS = [
    "group_id",
    "scenarios",
    "replicates",
    "attempt",
    "class",
    "stage",
    "call",
    "message",
    "seed",
]


def _one_record(facts: dict[str, Any], case: str) -> dict[str, str]:
    records = facts[case]["records"]
    assert len(records) == 1, (case, records)
    return records[0]


def test_an_error_inside_the_stage_is_recorded(facts: dict[str, Any]) -> None:
    case = facts["stage_recorded"]
    assert case["exit"] == 0 and case["log_names_group"]
    assert case["columns"] == RECORD_COLUMNS
    record = _one_record(facts, "stage_recorded")
    assert record["group_id"] == "g000000005"
    assert record["scenarios"] == "a;b" and record["replicates"] == "4"
    assert record["attempt"] == "1" and record["class"] == "fitting_failure"
    assert record["stage"] == "fixture fit"
    assert record["message"] == "fixture stage error in g000000005"
    # The replicate group spans both scenarios, so it takes the replicate's `*` seed.
    assert record["seed"] == str(1000 + 7 * 4 + 2)
    assert 4 not in case["replicates"] and len(case["replicates"]) == smoke.REPLICATES - 1


def test_a_re_signalled_or_argument_error_inside_the_stage_is_recorded(
    facts: dict[str, Any],
) -> None:
    """A ``future::value`` re-signal and a package argument error (m2's limit) are recorded."""
    assert facts["future_installed"]
    for case in ("stage_future", "stage_argument_error"):
        assert facts[case]["exit"] == 0, case
        record = _one_record(facts, case)
        assert record["class"] == "fitting_failure" and record["group_id"] == "g000000005"
        assert facts[case]["table"] == facts["stage_recorded"]["table"], case
    assert "fixture stage error" in _one_record(facts, "stage_future")["message"]
    assert "nonsense" in _one_record(facts, "stage_argument_error")["message"]


def test_errors_outside_the_stage_and_the_fatal_class_stay_fatal(facts: dict[str, Any]) -> None:
    """A re-signal around the stage, a runner stop, ``study_fatal_error``, a warning promoted
    by a calling handler (the ltmle shape) and C stack exhaustion stop the run."""
    for case in (
        "stage_outside_resignal",
        "stage_fatal_class",
        "stage_ltmle_shape",
        "stage_cstack",
        "stage_runner_stop",
    ):
        assert facts[case]["exit"] == 1, case
        assert facts[case]["message"], case
        assert facts[case]["table"] is None, case
    for case in (
        "stage_outside_resignal",
        "stage_fatal_class",
        "stage_ltmle_shape",
        "stage_cstack",
    ):
        assert facts[case]["fatal_written"], case


def test_a_frame_without_a_scenario_column_counts_by_group(facts: dict[str, Any]) -> None:
    """B3: the record names ``*`` and the group's two keys are counted."""
    case = facts["stage_noscenario"]
    assert case["exit"] == 0
    record = _one_record(facts, "stage_noscenario")
    assert record["scenarios"] == "*" and record["replicates"] == "2"
    assert 2 not in case["replicates"]


def test_every_dispatch_path_records(facts: dict[str, Any]) -> None:
    """Prescheduled ``mclapply``, a one-core ``mclapply``, the solo second attempt, the
    in-place tail and the stream all write the record and the same table."""
    recorded = facts["stage_recorded"]["table"]
    for case in ("stage_preschedule", "stage_onecore", "stage_second_attempt", "stage_inplace"):
        assert facts[case]["exit"] == 0, case
        assert facts[case]["table"] == recorded, case
        assert _one_record(facts, case)["group_id"] == "g000000005", case
    assert _one_record(facts, "stage_second_attempt")["attempt"] == "2"
    assert _one_record(facts, "stage_preschedule")["attempt"] == "1"
    assert facts["inplace_clean"]["exit"] == 0 and not facts["inplace_clean"]["records"]
    assert facts["inplace_clean"]["table"] == facts["plain"]["table"]
    stream = facts["stage_stream"]
    assert stream["exit"] == 0 and _one_record(facts, "stage_stream")["group_id"] == "r000000002"
    assert 2 not in stream["replicates"]


def test_a_resume_reuses_a_failure_checkpoint(facts: dict[str, Any]) -> None:
    resumed = facts["resume_with_failure"]
    assert resumed["first_exit"] == 0 and resumed["exit"] == 0
    assert resumed["attempts_g5"] == 1 and resumed["reused_all"]
    assert resumed["table"] == facts["stage_recorded"]["table"]
    assert len(resumed["records"]) == 1
    stream = facts["stage_stream_resume"]
    assert stream["stopped_exit"] == 137 and stream["exit"] == 0 and stream["reused"]
    assert stream["attempts_r2"] == 1 and len(stream["records"]) == 1
    assert stream["table"] == facts["stage_stream"]["table"]


def test_each_row_rule_gives_an_invalid_result_record(facts: dict[str, Any]) -> None:
    case = facts["invalid_rows"]
    assert case["exit"] == 0
    assert [record["class"] for record in case["records"]] == ["invalid_result"] * 5
    assert {record["stage"] for record in case["records"]} == {"row check"}
    messages = case["messages"]
    assert messages[0].startswith("non-finite inference_estimate")
    assert messages[1].startswith("non-positive std_error")
    assert messages[2].startswith("ci_lower above ci_upper")
    assert messages[3].startswith("ci_lower above ci_upper")  # an NA endpoint breaks the order
    assert messages[4].startswith("covered disagrees")
    assert case["replicates"] == [0, 2, 4, 6, 8, 10, 11]


def test_twenty_failed_groups_stop_the_run_as_systematic(facts: dict[str, Any]) -> None:
    for case in ("systematic", "systematic_stream", "calibration_systematic"):
        assert facts[case]["exit"] == 1 and facts[case]["message"], case
    assert facts["every_group_failed"]["exit"] == 1 and facts["every_group_failed"]["message"]


def test_calibration_takes_the_next_group_of_a_failed_scenario(facts: dict[str, Any]) -> None:
    """Decision Q3."""
    case = facts["calibration_failure"]
    assert case["exit"] == 0 and case["logged"]
    assert case["groups"] == ["g000000002"]


def test_memory_exhaustion_is_a_kill(facts: dict[str, Any]) -> None:
    """Decision Q2: no record, a retry at half the workers, and the no-failure table."""
    case = facts["memory_kill"]
    assert case["exit"] == 0 and case["killed_logged"] and not case["records"]
    assert case["attempts"] == 2
    assert any("1 killed" in line for line in case["passes"])
    assert case["table"] == facts["plain"]["table"]


def test_without_a_failure_nothing_new_is_written(facts: dict[str, Any]) -> None:
    """A stale failures file is deleted, and the a2 harness writes the a1 harness's bytes."""
    assert facts["stale_failures"]["deleted"] and facts["stale_failures"]["exit"] == 0
    same = facts["no_failure_bytes"]
    assert same["no_failures_file"]
    assert same["a1_fitter"] == same["plain"] == facts["plain"]["table"]
    assert same["a1_stream"] == same["new_stream"]
