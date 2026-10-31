"""The X33 fixtures: their state, the parity comparison and its mutation controls.

Three R scripts in ``tests/canonical/longitudinal_option_fixtures`` print the R side of the
comparisons for the longitudinal fitting options: ``lmtp``'s ratio trim (``ratio_trim=``),
``lmtp``'s history windows (``markov_order=``) and ``survtmle``'s pooled censoring
(``hazard_pooling="pooled"``).  The pinned images build on the local Study queue, so each
fixture is ``pending`` until that run adds its JSON file.

* The state test asserts the registry and the script bytes.  The scripts share the option
  fixtures' JSON block byte for byte.
* A pending fixture never reads as a pass: its parity test reports XFAIL with the reason.
* The comparisons are proven on stand-ins: the same scripts' output from a cloud image with the
  pinned base and CRAN snapshot (``stand_ins/``).  Each comparison passes on its stand-in and
  fails on a copy with one value changed.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.canonical.longitudinal_option_fixtures import parity
from tests.canonical.longitudinal_option_fixtures.registry import (
    HERE,
    REGISTRY,
    build_command,
    check_entry,
    run_command,
    script_sha256,
)
from tests.canonical.option_fixtures.registry import HERE as OPTION_FIXTURES
from tests.canonical.option_fixtures.registry import FixtureEntry, load

pytestmark = pytest.mark.xdist_group("longitudinal_option_fixtures")

BY_NAME = {entry.name: entry for entry in REGISTRY}
STAND_INS = HERE / "stand_ins"
SHARED_BLOCK = re.compile(
    r"^# ---- option fixtures shared: begin ----\n.*?^# ---- option fixtures shared: end ----\n",
    re.DOTALL | re.MULTILINE,
)


def ids(entry: FixtureEntry) -> str:
    return entry.name


def stand_in(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((STAND_INS / f"{name}.json").read_text(encoding="utf-8"))
    return loaded


def mutated(payload: dict[str, Any], change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    copied = copy.deepcopy(payload)
    change(copied)
    return copied


class TestTheRegistry:
    def test_it_names_the_three_fixtures_and_their_parts(self) -> None:
        assert [(entry.name, entry.parts) for entry in REGISTRY] == [
            ("lmtp_trim", ("d",)),
            ("lmtp_history", ("e",)),
            ("survtmle_pooled", ("j",)),
        ]
        assert set(parity.COMPARATORS) == set(BY_NAME)

    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_each_entry_is_in_a_coherent_state(self, entry: FixtureEntry) -> None:
        assert check_entry(entry, HERE) == []

    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_the_script_is_lf_bytes_with_the_registered_checksum(self, entry: FixtureEntry) -> None:
        assert b"\r" not in (HERE / entry.script).read_bytes()
        assert script_sha256(HERE / entry.script) == entry.script_sha256

    def test_the_scripts_carry_the_option_fixtures_shared_block(self) -> None:
        (reference,) = SHARED_BLOCK.findall(
            (OPTION_FIXTURES / "drtmle_stratify.R").read_text(encoding="utf-8")
        )
        for entry in REGISTRY:
            found = SHARED_BLOCK.findall((HERE / entry.script).read_text(encoding="utf-8"))
            assert found == [reference], entry.script

    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_the_image_is_one_a_registered_study_builds(self, entry: FixtureEntry) -> None:
        dockerfile = (HERE.parents[2] / entry.context / "Dockerfile").read_text(encoding="utf-8")
        assert re.search(r"^FROM rocker/r-ver:4\.5\.2@sha256:[0-9a-f]{64}$", dockerfile, re.M)
        assert build_command(entry) == f"docker build -t {entry.image} {entry.context}"
        command = run_command(entry)
        assert f"/fixture/{entry.script} /fixture/{entry.output}" in command
        assert "CLEVERLY_IMAGE_ID" in command and "CLEVERLY_SCRIPT_SHA256" in command

    @pytest.mark.parametrize("entry", REGISTRY, ids=ids)
    def test_the_stand_in_comes_from_this_script(self, entry: FixtureEntry) -> None:
        payload = stand_in(entry.name)
        assert payload["fixture"] == entry.name
        assert payload["provenance"]["script_sha256"] == entry.script_sha256
        assert payload["provenance"]["image_id"] is None
        assert "not the pinned image" in payload["provenance"]["stand_in"]


@pytest.mark.parametrize("entry", REGISTRY, ids=ids)
def test_the_parity_of_a_fixture(entry: FixtureEntry) -> None:
    """A present fixture is compared.  A pending one is reported as XFAIL with its reason."""
    if entry.pending:
        pytest.xfail(f"{entry.name}: comparator run pending. {entry.reason}")
    payload = load(entry, HERE)
    assert payload is not None
    assert parity.COMPARATORS[entry.name](payload) == []


def test_a_pending_fixture_has_no_payload(tmp_path: Path) -> None:
    for entry in REGISTRY:
        if entry.pending:
            assert load(entry, HERE) is None


# --------------------------------------------------------------- the comparisons on stand-ins


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_the_comparison_passes_on_its_stand_in(name: str) -> None:
    assert parity.COMPARATORS[name](stand_in(name)) == []


def _node(payload: dict[str, Any], order: str, node: int) -> dict[str, Any]:
    (item,) = (entry for entry in payload["orders"] if entry["order"] == order)
    found: dict[str, Any] = item["nodes"][node - 1]
    return found


HISTORY_MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    # The window one node too short or too long, on each side.
    "mechanism drops its oldest arm": lambda p: _node(p, "1", 3)["mechanism"].remove("A1"),
    "mechanism adds an older covariate": lambda p: _node(p, "0", 3)["mechanism"].append("L2"),
    "outcome adds an older covariate": lambda p: _node(p, "1", 4)["outcome"].append("L2"),
    "outcome drops a window arm": lambda p: _node(p, "2", 4)["outcome"].remove("A2"),
}


@pytest.mark.parametrize("mutation", sorted(HISTORY_MUTATIONS))
def test_a_changed_history_window_fails(mutation: str) -> None:
    payload = mutated(stand_in("lmtp_history"), HISTORY_MUTATIONS[mutation])
    assert parity.compare_lmtp_history(payload), mutation


def test_the_rule_arm_lag_is_what_the_comparison_reads() -> None:
    """With no lag the rule designs of finite orders fail, so the lag is measured, not assumed."""
    payload = stand_in("lmtp_history")
    original = parity.RULE_ARM_LAG
    try:
        parity.RULE_ARM_LAG = 10_000
        findings = parity.compare_lmtp_history(payload)
    finally:
        parity.RULE_ARM_LAG = original
    assert findings and all("rule outcome" in item for item in findings)


def _bump_trimmed(payload: dict[str, Any]) -> None:
    payload["fit"]["trimmed"][1][7] = payload["fit"]["trimmed"][1][7] + 1e-12


def _threshold(payload: dict[str, Any]) -> None:
    payload["fixed"]["cases"][2]["threshold"] = payload["fixed"]["cases"][2]["threshold"] * (
        1 + 1e-12
    )


def _zero(payload: dict[str, Any]) -> None:
    column = payload["fit"]["untrimmed"][0]
    index = int(np.argmax(np.asarray(column, dtype=float)))
    column[index] = 0.0


TRIM_MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "a trimmed entry moved": _bump_trimmed,
    "a fixed threshold moved": _threshold,
    "a positive ratio set to zero": _zero,
}


@pytest.mark.parametrize("mutation", sorted(TRIM_MUTATIONS))
def test_a_changed_trim_fails(mutation: str) -> None:
    payload = mutated(stand_in("lmtp_trim"), TRIM_MUTATIONS[mutation])
    assert parity.compare_lmtp_trim(payload), mutation


def _swap_settings(payload: dict[str, Any]) -> None:
    node, pooled = payload["settings"]
    node["hazard_pooling"], pooled["hazard_pooling"] = "pooled", "node"


def _shift_hazard(payload: dict[str, Any]) -> None:
    (pooled,) = (item for item in payload["settings"] if item["hazard_pooling"] == "pooled")
    pooled["censoring_hazard"][1] = [value + 1e-5 for value in pooled["censoring_hazard"][1]]


def _event_at_t0(payload: dict[str, Any]) -> None:
    events = payload["data"]["event"]
    times = payload["data"]["time"]
    index = next(i for i, (t, e) in enumerate(zip(times, events, strict=True)) if t == 4 and e == 0)
    events[index] = 1.0


POOLED_MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "the settings swapped": _swap_settings,
    "a pooled hazard moved by 1e-5": _shift_hazard,
    "an event at t0": _event_at_t0,
}


@pytest.mark.parametrize("mutation", sorted(POOLED_MUTATIONS))
def test_a_changed_pooled_fit_fails(mutation: str) -> None:
    payload = mutated(stand_in("survtmle_pooled"), POOLED_MUTATIONS[mutation])
    assert parity.compare_survtmle_pooled(payload), mutation
