"""The option fixtures: the pending state, the parity comparison and the pending label.

Four R scripts in ``tests/canonical/option_fixtures`` produce the R side of the comparisons for
the targeting options: the ``tmle3`` updater, ``drtmle(stratify=)``, ``ctmleDiscrete(patience=)`` and
``ctmleDiscrete(Wg=)``.  R is not in the cloud session, so every fixture is ``pending``: it has
no JSON file, a reason, and the checksum of its script.  The local Study queue runs the script
(``tests/canonical/option_fixtures/README.md``) and a follow-up change adds the JSON and sets the
status to ``present``.

* The state test asserts the registry, the script bytes and the README, and the mutation controls
  break each part in a temporary copy.
* A pending fixture never reads as a pass.  Its parity test reports XFAIL with the reason, so the
  summary line (``-ra``) lists it.  A present fixture is compared at the plan's tolerance.
* The parity comparison is proven on stand-in payloads that Python writes in the R schema.  The
  stand-in of the ``tmle3`` updater is a transcription of ``tmle3_Update``, and the comparison
  must pass on it and fail on a copy with one value changed.
* The pending label of the reader-facing rows is tied to the registry, and a control that sets one
  entry to ``present`` makes its labelled row fail.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.canonical.option_fixtures import parity, stand_ins
from tests.canonical.option_fixtures.registry import (
    DOCS_FILES,
    HERE,
    PENDING_LABEL,
    REGISTRY,
    ROOT,
    FixtureEntry,
    build_command,
    check_entry,
    dockerfile_facts,
    label_findings,
    load,
    run_command,
    script_sha256,
)

BY_NAME = {entry.name: entry for entry in REGISTRY}
README = (HERE / "README.md").read_text(encoding="utf-8")
SHARED_BLOCK = re.compile(
    r"^# ---- option fixtures shared: begin ----\n.*?^# ---- option fixtures shared: end ----\n",
    re.DOTALL | re.MULTILINE,
)


def ids(entry: FixtureEntry) -> str:
    return entry.name


# ------------------------------------------------------------------------------ the registry


class TestTheRegistry:
    def test_it_names_the_four_fixtures_and_the_parts_they_check(self) -> None:
        assert [entry.name for entry in REGISTRY] == [
            "tmle3_updater",
            "drtmle_stratify",
            "ctmle_patience",
            "ctmle_wg",
        ]
        assert sorted(part for entry in REGISTRY for part in entry.parts) == [
            *"abcfgh",
        ]
        assert {entry.script for entry in REGISTRY} == {f"{entry.name}.R" for entry in REGISTRY}
        assert set(parity.COMPARATORS) == set(BY_NAME)

    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_each_entry_is_in_a_coherent_state(self, entry: FixtureEntry) -> None:
        assert check_entry(entry) == []

    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_the_script_is_lf_bytes_with_the_registered_checksum(self, entry: FixtureEntry) -> None:
        data = (HERE / entry.script).read_bytes()
        assert b"\r" not in data
        assert script_sha256(HERE / entry.script) == entry.script_sha256

    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_each_script_calls_the_comparator_it_registers(self, entry: FixtureEntry) -> None:
        text = (HERE / entry.script).read_text(encoding="utf-8")
        calls = {
            "tmle3_updater": ("tmle3_Update$new(", "tmle_TSM_all()", "fit_tmle3("),
            "drtmle_stratify": ("drtmle::drtmle(", "stratify = stratify", "guard = NULL"),
            "ctmle_patience": ("ctmle::ctmleDiscrete(", "patience = patience", "preOrder = TRUE"),
            "ctmle_wg": ("ctmle::ctmleDiscrete(", "Wg = wg", "preOrder = TRUE"),
        }
        for call in calls[entry.name]:
            assert call in text, f"{entry.script} does not contain {call}"
        assert entry.script_sha256 not in text

    def test_the_shared_block_is_identical_in_the_four_scripts(self) -> None:
        blocks = []
        for entry in REGISTRY:
            found = SHARED_BLOCK.findall((HERE / entry.script).read_text(encoding="utf-8"))
            assert len(found) == 1, f"{entry.script} holds {len(found)} shared blocks"
            blocks.append(found[0])
        assert len(set(blocks)) == 1

    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_the_image_tag_names_the_dockerfile_commit(self, entry: FixtureEntry) -> None:
        facts = dockerfile_facts(entry)
        assert facts["base"].startswith("rocker/r-ver:4.5.2@sha256:")
        assert re.fullmatch(r"[0-9a-f]{40}", facts["commit"])
        assert facts["commit"].startswith(entry.image.rsplit(":", 1)[1])

    def test_the_images_are_the_ones_the_registered_studies_use(self) -> None:
        regenerators = {
            "tests/canonical/tmle3/regenerate.py": "cleverly-tmle3-reference:ed72f8a",
            "tests/canonical/drtmle/regenerate.py": "cleverly-drtmle-reference:538a3a2",
            "tests/canonical/ctmle_logistic_plugin/regenerate.py": "cleverly-ctmle-reference:18de559",
        }
        for path, image in regenerators.items():
            assert image in (ROOT / path).read_text(encoding="utf-8")
        assert {entry.image for entry in REGISTRY} == set(regenerators.values())


class TestTheStateIsAsserted:
    """Each control breaks one part of a copy of the directory and requires a finding."""

    @staticmethod
    def copy_of(tmp_path: Path, entry: FixtureEntry) -> Path:
        (tmp_path / entry.script).write_bytes((HERE / entry.script).read_bytes())
        return tmp_path

    @staticmethod
    def write_json(root: Path, entry: FixtureEntry, **provenance: Any) -> None:
        payload = {
            "fixture": entry.name,
            "provenance": {
                "script_sha256": entry.script_sha256,
                "image_id": "sha256:" + "a" * 64,
                **provenance,
            },
        }
        (root / entry.output).write_text(json.dumps(payload), encoding="utf-8")

    def present(self, entry: FixtureEntry) -> FixtureEntry:
        return dataclasses.replace(entry, status="present", reason="")

    def test_a_copy_of_each_pending_entry_is_coherent(self, tmp_path: Path) -> None:
        for entry in REGISTRY:
            assert check_entry(entry, self.copy_of(tmp_path, entry)) == []

    def test_a_pending_fixture_with_its_json_is_caught(self, tmp_path: Path) -> None:
        entry = REGISTRY[0]
        root = self.copy_of(tmp_path, entry)
        self.write_json(root, entry)
        assert any("pending, but" in text for text in check_entry(entry, root))

    def test_a_pending_fixture_without_a_reason_is_caught(self, tmp_path: Path) -> None:
        entry = dataclasses.replace(REGISTRY[0], reason="  ")
        found = check_entry(entry, self.copy_of(tmp_path, entry))
        assert any("needs a reason" in text for text in found)

    def test_a_present_fixture_without_its_json_is_caught(self, tmp_path: Path) -> None:
        entry = self.present(REGISTRY[1])
        found = check_entry(entry, self.copy_of(tmp_path, entry))
        assert any("does not exist" in text for text in found)

    def test_a_present_fixture_with_its_json_is_coherent(self, tmp_path: Path) -> None:
        entry = self.present(REGISTRY[1])
        root = self.copy_of(tmp_path, entry)
        self.write_json(root, entry)
        assert check_entry(entry, root) == []

    def test_a_json_from_another_script_version_is_caught(self, tmp_path: Path) -> None:
        entry = self.present(REGISTRY[1])
        root = self.copy_of(tmp_path, entry)
        self.write_json(root, entry, script_sha256="0" * 64)
        assert any("another version" in text for text in check_entry(entry, root))

    def test_a_json_without_an_image_id_is_caught(self, tmp_path: Path) -> None:
        entry = self.present(REGISTRY[1])
        root = self.copy_of(tmp_path, entry)
        self.write_json(root, entry, image_id=None)
        assert any("no image id" in text for text in check_entry(entry, root))

    def test_a_present_fixture_that_keeps_its_pending_reason_is_caught(
        self, tmp_path: Path
    ) -> None:
        entry = dataclasses.replace(REGISTRY[1], status="present")
        root = self.copy_of(tmp_path, entry)
        self.write_json(root, entry)
        assert any("carries no pending reason" in text for text in check_entry(entry, root))

    def test_an_edited_script_is_caught_by_its_checksum(self, tmp_path: Path) -> None:
        entry = REGISTRY[2]
        root = self.copy_of(tmp_path, entry)
        with (root / entry.script).open("ab") as handle:
            handle.write(b"# edited\n")
        assert any("registered checksum" in text for text in check_entry(entry, root))

    def test_a_crlf_script_is_caught(self, tmp_path: Path) -> None:
        entry = REGISTRY[3]
        root = self.copy_of(tmp_path, entry)
        (root / entry.script).write_bytes(
            (HERE / entry.script).read_bytes().replace(b"\n", b"\r\n")
        )
        assert any("CR bytes" in text for text in check_entry(entry, root))

    def test_an_unknown_status_is_caught(self) -> None:
        entry = dataclasses.replace(REGISTRY[0], status="done")
        assert any("neither" in text for text in check_entry(entry))


class TestTheReadme:
    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_it_names_the_image_the_digest_and_the_exact_commands(
        self, entry: FixtureEntry
    ) -> None:
        facts = dockerfile_facts(entry)
        assert entry.image in README
        assert facts["base"] in README
        assert facts["commit"] in README
        assert build_command(entry) in README
        assert run_command(entry) in README
        assert entry.output in README

    def test_the_run_command_sets_the_entry_point_and_the_provenance(self) -> None:
        for entry in REGISTRY:
            command = run_command(entry)
            assert "--entrypoint Rscript" in command
            assert "CLEVERLY_IMAGE_ID" in command
            assert "CLEVERLY_SCRIPT_SHA256" in command
            assert command.endswith(f"/fixture/{entry.script} /fixture/{entry.output}")

    def test_it_states_the_pending_label_and_the_tolerances(self) -> None:
        assert PENDING_LABEL in README
        for text in ("1e-6", "1e-8"):
            assert text in README


# ------------------------------------------------------- pending is reported, never a pass


@pytest.mark.parametrize("entry", REGISTRY, ids=ids)
def test_the_parity_of_a_fixture(entry: FixtureEntry) -> None:
    """A present fixture is compared.  A pending one is reported as XFAIL with its reason."""
    if entry.pending:
        pytest.xfail(f"{entry.name}: comparator run pending. {entry.reason}")
    payload = load(entry)
    assert payload is not None
    assert parity.COMPARATORS[entry.name](payload) == []


def test_a_pending_fixture_has_no_payload_to_compare() -> None:
    """Pending means no file, so the parity test above cannot pass by reading nothing."""
    pending = [entry for entry in REGISTRY if entry.pending]
    assert pending, "no fixture is pending; set the follow-up's status flips in the registry"
    for entry in pending:
        assert load(entry) is None


# --------------------------------------------------------------- the parity comparison itself


def roundtrip(payload: dict[str, Any], tmp_path: Path, name: str) -> dict[str, Any]:
    """Write the payload as a JSON file and read it back, as the real fixture is read."""
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def mutated(payload: dict[str, Any], change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    copied = copy.deepcopy(payload)
    change(copied)
    return copied


def updater(payload: dict[str, Any], name: str) -> dict[str, Any]:
    (found,) = (item for item in payload["updaters"] if item["name"] == name)
    return found


def _container(payload: dict[str, Any], name: str, path: tuple[str | int, ...]) -> Any:
    target: Any = updater(payload, name)
    for key in path:
        target = target[key]
    return target


def bump(
    name: str, path: tuple[str | int, ...], index: int, amount: float
) -> Callable[[dict[str, Any]], None]:
    """A change that adds ``amount`` to one number of one updater."""

    def change(payload: dict[str, Any]) -> None:
        container = _container(payload, name, path)
        container[index] = container[index] + amount

    return change


def scale(
    name: str, path: tuple[str | int, ...], index: int, factor: float
) -> Callable[[dict[str, Any]], None]:
    """A change that multiplies one number of one updater by ``factor``."""

    def change(payload: dict[str, Any]) -> None:
        container = _container(payload, name, path)
        container[index] = container[index] * factor

    return change


def put(
    name: str, path: tuple[str | int, ...], key: str, value: Any
) -> Callable[[dict[str, Any]], None]:
    """A change that sets one field of one updater."""

    def change(payload: dict[str, Any]) -> None:
        _container(payload, name, path)[key] = value

    return change


def drop_last_step(name: str) -> Callable[[dict[str, Any]], None]:
    """A change that removes the last step of an updater and lowers its count."""

    def change(payload: dict[str, Any]) -> None:
        item = updater(payload, name)
        item["epsilons"].pop()
        item["eds"].pop()
        item["steps"] -= 1

    return change


def set_best_to_last(name: str) -> Callable[[dict[str, Any]], None]:
    """A change that reports the last step as the best one."""

    def change(payload: dict[str, Any]) -> None:
        item = updater(payload, name)
        item["best_step"] = item["steps"]

    return change


def bump_run(
    run: str, key: str, index: int | None, amount: float, *, factor: bool = False
) -> Callable[[dict[str, Any]], None]:
    """A change to one number of one ``ctmleDiscrete`` run: added, or multiplied by ``1 + amount``."""

    def change(payload: dict[str, Any]) -> None:
        target = payload["runs"][run]
        if index is None:
            target[key] = target[key] * (1.0 + amount) if factor else target[key] + amount
        else:
            current = target[key][index]
            target[key][index] = current * (1.0 + amount) if factor else current + amount

    return change


def put_run(run: str, key: str, value: Any) -> Callable[[dict[str, Any]], None]:
    """A change that sets one field of one ``ctmleDiscrete`` run."""

    def change(payload: dict[str, Any]) -> None:
        payload["runs"][run][key] = value

    return change


@pytest.fixture(scope="module")
def tmle3_walks(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    """Three stand-ins: converging walks, an oscillating fixed walk, and a met initial rule."""
    root = tmp_path_factory.mktemp("tmle3")
    settings = {"converging": (11, 0.004), "oscillating": (13, 0.06), "met": (12, 0.002)}
    return {
        label: roundtrip(stand_ins.tmle3_updater_stand_in(seed=seed, delta=delta), root, label)
        for label, (seed, delta) in settings.items()
    }


class TestTheTmle3Comparison:
    def test_it_passes_on_a_transcription_of_tmle3_update(
        self, tmle3_walks: dict[str, dict[str, Any]]
    ) -> None:
        for label, payload in tmle3_walks.items():
            assert parity.compare_tmle3_updater(payload) == [], label

    def test_the_stand_ins_exercise_what_the_comparison_reads(
        self, tmle3_walks: dict[str, dict[str, Any]]
    ) -> None:
        converging = tmle3_walks["converging"]
        assert updater(converging, "constrained_optim_last")["steps"] >= 4
        assert updater(converging, "constrained_fixed_last")["steps"] >= 4
        assert updater(converging, "constrained_optim_scaled_var_last")["steps"] >= 3
        assert all(
            item["steps"] == 1
            for item in converging["updaters"]
            if "constrained" not in item["name"]
        )
        oscillating = updater(tmle3_walks["oscillating"], "constrained_fixed_best")
        assert oscillating["steps"] == 200
        assert oscillating["best_step"] != oscillating["steps"]
        # The initial fit already meets the rule: tmle3 still takes its first step.
        met = tmle3_walks["met"]
        assert updater(met, "constrained_fixed_last")["steps"] == 1
        assert updater(met, "constrained_optim_last")["steps"] == 1
        # The two rules give different thresholds on the same fit.
        sample = updater(converging, "constrained_optim_last")["final"]["thresholds"]
        scaled = updater(converging, "constrained_optim_scaled_var_last")["final"]["thresholds"]
        assert abs(sample[0] - scaled[0]) > 1e-4

    def test_the_initial_outcome_regression_is_in_range_so_the_bound_never_binds(
        self, tmle3_walks: dict[str, dict[str, Any]]
    ) -> None:
        for payload in tmle3_walks.values():
            for item in payload["updaters"]:
                assert item["replay"]["bound_binds"] is False
                assert item["replay"]["max_abs_difference"] < 1e-12

    def test_a_perturbation_below_the_tolerance_passes(
        self, tmle3_walks: dict[str, dict[str, Any]]
    ) -> None:
        def nudge(payload: dict[str, Any]) -> None:
            item = updater(payload, "constrained_optim_last")
            item["epsilons"][0] = [item["epsilons"][0][0] + 1e-9]

        assert parity.compare_tmle3_updater(mutated(tmle3_walks["converging"], nudge)) == []

    @pytest.mark.parametrize(
        ("label", "walk", "change", "expected"),
        [
            (
                "a step epsilon",
                "converging",
                bump("constrained_optim_last", ("epsilons", 0), 0, 1e-4),
                "epsilon",
            ),
            (
                "an unconstrained epsilon",
                "converging",
                bump("sample_size_last", ("epsilons", 0), 0, 1e-4),
                "epsilon",
            ),
            (
                "the step count",
                "converging",
                drop_last_step("constrained_fixed_last"),
                "cleverly walked",
            ),
            (
                "the reported best step",
                "oscillating",
                set_best_to_last("constrained_fixed_best"),
                "best_step",
            ),
            (
                "a threshold",
                "converging",
                scale("constrained_optim_scaled_var_last", ("final", "thresholds"), 0, 1.001),
                "thresholds",
            ),
            (
                "the final ED",
                "converging",
                bump("constrained_optim_last", ("final", "ed"), 0, 1e-6),
                "score at the final fit",
            ),
            (
                "a final estimate",
                "converging",
                bump("constrained_optim_last", ("final", "psi"), 1, 1e-5),
                "psi",
            ),
            (
                "the bound flag",
                "converging",
                put("sample_size_last", ("replay",), "bound_binds", True),
                "bound(., 0.005)",
            ),
            (
                "the replay",
                "converging",
                put("sample_size_last", ("replay",), "max_abs_difference", 1e-3),
                "replay",
            ),
            (
                "the initial ED",
                "converging",
                lambda p: p["initial"]["ed"].__setitem__(0, p["initial"]["ed"][0] + 1e-5),
                "initial ED",
            ),
            (
                "an R error",
                "converging",
                lambda p: p["updaters"].__setitem__(0, {"name": "x", "error": "boom"}),
                "R raised an error",
            ),
            (
                "the first step of a met initial rule",
                "met",
                scale("constrained_fixed_last", ("epsilons", 0), 0, 0.5),
                "epsilon",
            ),
        ],
        ids=lambda value: value if isinstance(value, str) and " " in value else "",
    )
    def test_a_changed_value_is_caught(
        self,
        tmle3_walks: dict[str, dict[str, Any]],
        label: str,
        walk: str,
        change: Callable[[dict[str, Any]], None],
        expected: str,
    ) -> None:
        findings = parity.compare_tmle3_updater(mutated(tmle3_walks[walk], change))
        assert findings, f"changing {label} was not caught"
        assert any(expected in text for text in findings), findings

    def test_the_epsilon_direction_convention_is_checked(
        self, tmle3_walks: dict[str, dict[str, Any]]
    ) -> None:
        """A walk along ED/||ED|| with the wrong sign moves the cumulative epsilon."""

        def flip(payload: dict[str, Any]) -> None:
            item = updater(payload, "constrained_fixed_last")
            item["epsilons"] = [[-e[0]] for e in item["epsilons"]]

        assert any(
            "epsilon" in text
            for text in parity.compare_tmle3_updater(mutated(tmle3_walks["converging"], flip))
        )


class TestTheDrtmleComparison:
    @pytest.fixture(scope="class")
    def payload(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        return roundtrip(
            stand_ins.drtmle_stratify_stand_in(), tmp_path_factory.mktemp("drtmle"), "drtmle"
        )

    def setting(self, payload: dict[str, Any], stratify: bool) -> dict[str, Any]:
        (found,) = (s for s in payload["settings"] if s["stratify"] is stratify)
        return found

    def test_it_passes_on_independent_fits(self, payload: dict[str, Any]) -> None:
        assert parity.compare_drtmle_stratify(payload) == []

    def test_the_stratified_and_pooled_fits_differ(self, payload: dict[str, Any]) -> None:
        gap = np.max(
            np.abs(
                np.asarray(self.setting(payload, True)["qn_arm1"])
                - np.asarray(self.setting(payload, False)["qn_arm1"])
            )
        )
        assert gap > 1e-2

    def test_a_perturbation_below_the_tolerance_passes(self, payload: dict[str, Any]) -> None:
        def nudge(p: dict[str, Any]) -> None:
            self.setting(p, True)["qn_arm1"][3] += 1e-10
            self.setting(p, False)["tmle"][0] += 1e-9

        assert parity.compare_drtmle_stratify(mutated(payload, nudge)) == []

    @pytest.mark.parametrize(
        ("stratify", "key", "index", "amount", "expected"),
        [
            (True, "qn_arm1", 5, 1e-6, "qn_arm1"),
            (False, "qn_arm0", 7, 1e-6, "qn_arm0"),
            (True, "gn_arm0", 2, 1e-6, "gn_arm0"),
            (False, "gn_arm1", 4, 1e-6, "gn_arm1"),
            (True, "tmle", 1, 1e-5, "TMLE estimates"),
            (False, "tmle", 0, 1e-5, "TMLE estimates"),
        ],
    )
    def test_a_changed_value_is_caught(
        self,
        payload: dict[str, Any],
        stratify: bool,
        key: str,
        index: int,
        amount: float,
        expected: str,
    ) -> None:
        def change(p: dict[str, Any]) -> None:
            self.setting(p, stratify)[key][index] += amount

        findings = parity.compare_drtmle_stratify(mutated(payload, change))
        assert any(expected in text for text in findings), findings

    def test_swapped_settings_are_caught(self, payload: dict[str, Any]) -> None:
        """The stratified R fit read as the pooled one fails the outcome regression."""

        def swap(p: dict[str, Any]) -> None:
            for setting in p["settings"]:
                setting["stratify"] = not setting["stratify"]

        findings = parity.compare_drtmle_stratify(mutated(payload, swap))
        assert any("qn_arm" in text for text in findings)

    def test_a_missing_setting_is_caught(self, payload: dict[str, Any]) -> None:
        findings = parity.compare_drtmle_stratify(mutated(payload, lambda p: p["settings"].pop()))
        assert any("one stratified and one pooled" in text for text in findings)

    def test_equal_settings_are_caught(self, payload: dict[str, Any]) -> None:
        def equal(p: dict[str, Any]) -> None:
            pooled = self.setting(p, False)
            stratified = self.setting(p, True)
            for key in ("qn_arm0", "qn_arm1", "tmle"):
                stratified[key] = list(pooled[key])

        findings = parity.compare_drtmle_stratify(mutated(payload, equal))
        assert any("does not separate" in text for text in findings)


class TestTheCtmleComparisons:
    @pytest.fixture(scope="class")
    def patience(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        return roundtrip(
            stand_ins.ctmle_patience_stand_in(), tmp_path_factory.mktemp("patience"), "patience"
        )

    @pytest.fixture(scope="class")
    def wg(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        return roundtrip(stand_ins.ctmle_wg_stand_in(), tmp_path_factory.mktemp("wg"), "wg")

    def test_the_patience_comparison_passes_and_the_stand_in_exercises_the_counter(
        self, patience: dict[str, Any]
    ) -> None:
        assert parity.compare_ctmle_patience(patience) == []
        whole, patient = patience["runs"]["whole_path"], patience["runs"]["patient"]
        assert patient["stopped_early"] and not whole["stopped_early"]
        assert patient["scored"] < patient["candidates"]
        assert patient["cv_score"][-1] is None
        # A global minimum past the stop: patience changes the selection.
        assert patient["best_k"] != whole["best_k"]
        assert patient["est"] != pytest.approx(whole["est"], abs=1e-4)

    def test_the_wg_comparison_passes_and_the_restriction_has_an_effect(
        self, wg: dict[str, Any]
    ) -> None:
        assert parity.compare_ctmle_wg(wg) == []
        assert abs(wg["runs"]["restricted"]["est"] - wg["runs"]["whole_pool"]["est"]) > 1e-2

    def test_a_perturbation_below_the_tolerance_passes(
        self, patience: dict[str, Any], wg: dict[str, Any]
    ) -> None:
        nudged = mutated(patience, bump_run("whole_path", "est", None, 1e-8))
        bump_run("whole_path", "cv_score", 2, 1e-8, factor=True)(nudged)
        assert parity.compare_ctmle_patience(nudged) == []
        assert parity.compare_ctmle_wg(mutated(wg, bump_run("restricted", "est", None, 1e-8))) == []

    @pytest.mark.parametrize(
        ("label", "change", "expected"),
        [
            ("the selected index", bump_run("patient", "best_k", None, 1), "selects k="),
            (
                "a score",
                bump_run("whole_path", "cv_score", 1, 1e-5, factor=True),
                "cross-validated scores",
            ),
            ("a patient score", bump_run("patient", "cv_score", 0, 1e-3), "cross-validated scores"),
            ("the estimate", bump_run("patient", "est", None, 1e-5), "estimate"),
            ("the early stop", put_run("patient", "stopped_early", False), "did not stop early"),
            (
                "the scored prefix",
                lambda p: p["runs"]["patient"]["cv_score"].__setitem__(-1, 400.0),
                "scored",
            ),
            (
                "an unscored hole",
                lambda p: p["runs"]["patient"]["cv_score"].__setitem__(1, None),
                "suffix",
            ),
            (
                "the initial outcome regression",
                lambda p: p["q_arm1"].__setitem__(0, p["q_arm1"][0] + 1e-6),
                "q_arm1",
            ),
        ],
    )
    def test_a_changed_patience_value_is_caught(
        self,
        patience: dict[str, Any],
        label: str,
        change: Callable[[dict[str, Any]], None],
        expected: str,
    ) -> None:
        findings = parity.compare_ctmle_patience(mutated(patience, change))
        assert findings, f"changing {label} was not caught"
        assert any(expected in text for text in findings), findings

    @pytest.mark.parametrize(
        ("label", "change", "expected"),
        [
            ("the selected index", put_run("whole_pool", "best_k", 1), "selects k="),
            (
                "a restricted score",
                bump_run("restricted", "cv_score", 2, 1e-2),
                "cross-validated scores",
            ),
            ("the restricted estimate", bump_run("restricted", "est", None, 1e-5), "estimate"),
            ("the restricted pool", put_run("restricted", "pool", ["W1", "W2"]), "restricted pool"),
            (
                "the terms leaving the pool",
                put_run("restricted", "terms", ["W1", "W2"]),
                "leave the pool",
            ),
            (
                "an estimate equal to the restricted one",
                lambda p: p["runs"]["whole_pool"].__setitem__(
                    "est", p["runs"]["restricted"]["est"]
                ),
                "no effect to compare",
            ),
        ],
    )
    def test_a_changed_wg_value_is_caught(
        self,
        wg: dict[str, Any],
        label: str,
        change: Callable[[dict[str, Any]], None],
        expected: str,
    ) -> None:
        findings = parity.compare_ctmle_wg(mutated(wg, change))
        assert findings, f"changing {label} was not caught"
        assert any(expected in text for text in findings), findings


# ------------------------------------------------------------------------ the pending label


class TestThePendingLabel:
    @staticmethod
    def rows(label: str | None, *, skip: str | None = None) -> str:
        lines = ["| option | comparator | evidence |", "| --- | --- | --- |"]
        for entry in REGISTRY:
            if entry.name == skip:
                continue
            text = f"`{entry.output}`" + (f"; {label}" if label else "")
            lines.append(f"| {entry.parts[0]} | R | {text} |")
        return "\n".join(lines) + "\n"

    def test_the_published_pages_carry_the_label_exactly_for_the_pending_fixtures(self) -> None:
        texts = {name: (ROOT / name).read_text(encoding="utf-8") for name in DOCS_FILES}
        assert label_findings(REGISTRY, texts) == []

    def test_labelled_rows_for_pending_fixtures_pass(self) -> None:
        assert label_findings(REGISTRY, {"page": self.rows(PENDING_LABEL)}) == []

    def test_an_unlabelled_row_for_a_pending_fixture_fails(self) -> None:
        found = label_findings(REGISTRY, {"page": self.rows(None)})
        assert len(found) == len(REGISTRY)
        assert all("lacks the label" in text for text in found)

    def test_a_present_fixture_with_its_labelled_row_fails(self) -> None:
        """The control of the plan: one entry set to present in a copy fails its row."""
        copied = list(REGISTRY)
        copied[1] = dataclasses.replace(copied[1], status="present", reason="")
        found = label_findings(copied, {"page": self.rows(PENDING_LABEL)})
        assert found == [
            f"page: the row for present {copied[1].output} still carries the pending label"
        ]

    def test_a_present_fixture_without_the_label_passes(self) -> None:
        copied = list(REGISTRY)
        copied[1] = dataclasses.replace(copied[1], status="present", reason="")
        lines = self.rows(PENDING_LABEL).splitlines()
        lines = [
            line.replace(f"; {PENDING_LABEL}", "") if copied[1].output in line else line
            for line in lines
        ]
        assert label_findings(copied, {"page": "\n".join(lines) + "\n"}) == []

    def test_a_page_that_names_only_some_fixtures_fails(self) -> None:
        found = label_findings(REGISTRY, {"page": self.rows(PENDING_LABEL, skip="ctmle_wg")})
        assert found == ["page: no row names ctmle_wg.json"]

    def test_a_page_that_names_no_fixture_is_not_read(self) -> None:
        assert label_findings(REGISTRY, {"page": "| a | b |\n| --- | --- |\n| 1 | 2 |\n"}) == []

    def test_text_outside_a_table_row_is_not_a_row(self) -> None:
        prose = f"The file `{REGISTRY[0].output}` is pending.\n"
        assert label_findings(REGISTRY, {"page": prose}) == []
