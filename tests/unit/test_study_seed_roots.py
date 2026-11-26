"""No two studies draw from one seed root unless this file says why.

Studies use distinct named seed streams, for the reason :func:`tests.studies.evidence.seeds.stream_seed`
gives for hashing stream labels.  Two studies whose roots are
equal draw the same per-replication seeds wherever they spawn the same keys, so the Monte Carlo
errors of rows a reader compares are correlated by construction, and a pilot that shares a root
with the run it sizes reads the run's own draws.

A *root* here is every registered :class:`StudyRecord`'s ``seed`` and ``resampling_seed``, and
every module-level integer constant whose name contains ``SEED`` under ``tests/studies/`` and
``tests/canonical/``.  A constant in a study's runner or properties module belongs to that study,
so a record and the constant it is built from are one owner, not a collision.  Constants are read
statically with :mod:`ast`; no module beyond the registry is imported.

:data:`SHARED_ROOTS` lists the roots that more than one owner held when this check was added.
None of them is re-seeded here: each owner already ran, and moving a seed under committed rows
would leave a manifest or a pilot summary describing draws that never ran (``CLAUDE.md``, "Tests
and tooling"; the fast tier refits committed replications from their seeds).  An entry is one of:

* a **pairing**: the owners share samples on purpose, which :func:`test_paired_studies_share_their_samples`
  checks;
* an **auxiliary** root: one owner's constant seeds a Sobol scramble, a Monte Carlo constant, a
  perturbation map or a fold split, not a study's sample draws;
* a **legacy** root: two owners drew samples from it before this check existed.  ``reseed``
  names the owner that takes a fresh seed when it next draws, normally the one declared later,
  and pins the committed output of its run.  A rerun that changes the output without a fresh seed
  fails :func:`test_a_rerun_draws_from_a_fresh_root`.

Every entry is a ratchet: an entry whose owners no longer share its root fails until removed.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.studies.evidence.registry import ROOT, registered

#: The directories whose module-level ``*SEED*`` constants are roots.
SCANNED = ("tests/studies", "tests/canonical")


@dataclass(frozen=True)
class SharedRoot:
    """One seed root that several owners hold, and why that is accepted.

    Parameters
    ----------
    owners : frozenset of str
        Registered study slugs, or dotted module names for constants outside a study's runner and
        properties modules.
    reason : str
        Why the owners share the root.
    paired : frozenset of str
        Owners that share samples on purpose.  Their first replication must be one frame.
    reseed : Mapping
        Owner -> ``{path: digest}`` of the committed output its run drew from this root.  The
        owner takes a fresh seed at its next run.
    """

    owners: frozenset[str]
    reason: str
    paired: frozenset[str] = frozenset()
    reseed: Mapping[str, Mapping[str, str]] = field(default_factory=dict)


def _owners(*names: str) -> frozenset[str]:
    return frozenset(names)


PILOTS = "tests.studies."  # prefix of a pilot module owner
WITNESS = "tests.canonical.ctmle_derivative_witness.regenerate"

SHARED_ROOTS: dict[int, SharedRoot] = {
    0: SharedRoot(
        _owners("structural-nested-mean", "tests.studies.long_horizon_reference"),
        "auxiliary: both constants are fold-split seeds equal to the random_state=0 their fits "
        "pass, not sample draws",
    ),
    20240822: SharedRoot(
        _owners("canonical-ctmle-selector", "fold-targeted-cvtmle"),
        "legacy: fold-targeted-cvtmle (2026-08-31) took canonical-ctmle-selector's seed "
        "(2026-08-21)",
        reseed={"fold-targeted-cvtmle": {"tests/canonical/zepid_cvtmle": "051c415df0f33f13"}},
    ),
    20260824: SharedRoot(
        _owners("canonical-drtmle", "drtmle-foldwise", "canonical-ltmle-crossfit"),
        "drtmle-foldwise refits canonical-drtmle's registered draws on purpose; "
        "canonical-ltmle-crossfit (2026-08-24) is legacy, and it re-seeds rather than the pair",
        paired=_owners("canonical-drtmle", "drtmle-foldwise"),
        reseed={"canonical-ltmle-crossfit": {"tests/canonical/lmtp_ltmle": "46036b26eefe1f38"}},
    ),
    20260826: SharedRoot(
        _owners("canonical-drtmle", "canonical-ltmle-competing"),
        "legacy: canonical-drtmle's resampling root is canonical-ltmle-competing's seed, both "
        "declared 2026-08-25",
        reseed={
            "canonical-ltmle-competing": {
                "tests/canonical/lmtp_ltmle_competing": "31a2c08aa5fa000b"
            }
        },
    ),
    20260827: SharedRoot(
        _owners(
            "canonical-ltmle-competing-crossfit",
            "canonical-categorical-ltmle",
            "canonical-multi-arm-tmle",
        ),
        "legacy: the categorical and multi-arm studies (2026-08-27) took the competing cross-fit "
        "study's seed (2026-08-25)",
        reseed={
            "canonical-categorical-ltmle": {
                "tests/canonical/categorical_ltmle": "33109f6d62932162"
            },
            "canonical-multi-arm-tmle": {"tests/canonical/multi_arm_tmle3": "ab65594917a62978"},
        },
    ),
    20260828: SharedRoot(
        _owners(
            "canonical-ltmle-competing",
            "point-msm",
            "canonical-categorical-ltmle-crossfit",
            "canonical-multi-arm-ctmle-oat",
        ),
        "legacy: three seeds declared 2026-08-26 and 2026-08-27 equal canonical-ltmle-competing's "
        "resampling root (2026-08-25)",
        reseed={
            "point-msm": {"tests/canonical/tmle3_msm": "c4c9ae9427002390"},
            "canonical-categorical-ltmle-crossfit": {
                "tests/canonical/categorical_ltmle_crossfit": "b9e0f63630473cd7"
            },
            "canonical-multi-arm-ctmle-oat": {
                "tests/canonical/multi_arm_ctmle3_oat": "384af1a0d1b27741"
            },
        },
    ),
    20260829: SharedRoot(
        _owners("longitudinal-msm", "canonical-multi-arm-ctmle-selector"),
        "legacy: canonical-multi-arm-ctmle-selector (2026-08-27) took longitudinal-msm's seed "
        "(2026-08-26)",
        reseed={
            "canonical-multi-arm-ctmle-selector": {
                "tests/canonical/multi_arm_ctmle_selector": "716b4687d2f698f1"
            }
        },
    ),
    20260830: SharedRoot(
        _owners("canonical-multi-arm-drtmle", "multi-arm-drtmle-foldwise", "point-msm"),
        "multi-arm-drtmle-foldwise refits canonical-multi-arm-drtmle's registered draws on "
        "purpose; point-msm's resampling root is legacy, and it re-seeds rather than the pair",
        paired=_owners("canonical-multi-arm-drtmle", "multi-arm-drtmle-foldwise"),
        reseed={"point-msm": {"tests/canonical/tmle3_msm": "c4c9ae9427002390"}},
    ),
    20260924: SharedRoot(
        _owners("mar-drtmle", "repeated-crossfit-tmle"),
        "legacy: repeated-crossfit-tmle's seed (2026-08-28) is mar-drtmle's resampling root "
        "(2026-08-27)",
        reseed={
            "repeated-crossfit-tmle": {"tests/canonical/repeated_crossfit": "a1c165a5b9f5aef5"}
        },
    ),
    20261001: SharedRoot(
        _owners("canonical-multi-arm-tmle", "weighted-tmle"),
        "legacy: weighted-tmle's seed (2026-08-29) is canonical-multi-arm-tmle's resampling root "
        "(2026-08-27)",
        reseed={"weighted-tmle": {"tests/canonical/tmle_weighted": "6edf9bccc2afac90"}},
    ),
    20261002: SharedRoot(
        _owners("canonical-multi-arm-ctmle-oat", "default-simultaneous-bands"),
        "auxiliary: default-simultaneous-bands' DESIGN_SEED draws the normal vectors of its design computation",
    ),
    20261014: SharedRoot(
        _owners(
            "tests.canonical.sequential_baselines.make_data",
            "tests.studies.evidence.property_failures",
        ),
        "auxiliary: AFFINE_SEED seeds the affine perturbation maps of the property-failure "
        "controls",
    ),
    20261015: SharedRoot(
        _owners("tests.canonical.stremr.make_data", "tests.studies.evidence.property_failures"),
        "auxiliary: MIXED_SEED seeds the mixed perturbation draws of the property-failure controls",
    ),
    20261109: SharedRoot(
        _owners("stacked-mar-natural-course-cvtmle", "targeting-curves"),
        "auxiliary: targeting-curves' SOBOL_SEED scrambles the Sobol set of cell S's truth",
    ),
    20261110: SharedRoot(
        _owners("stacked-mar-natural-course-cvtmle", "targeting-curves"),
        "auxiliary: targeting-curves' DESIGN_SOBOL_SEED scrambles the design's Sobol set",
    ),
    20261112: SharedRoot(
        _owners(
            "band-t-reference",
            f"{PILOTS}option_stopping_pilot",
            f"{PILOTS}targeting_curves_pilot",
        ),
        "auxiliary: band-t-reference's SIZING_SEED draws its sizing simulation and "
        "targeting_curves_pilot's SOBOL_CHECK_SEED scrambles a Sobol check set; only the "
        "stopping pilot draws samples",
    ),
    20261113: SharedRoot(
        _owners(
            "tests.studies.evidence.simultaneous", f"{PILOTS}option_treatment_adjustment_pilot"
        ),
        "auxiliary: MVT_SEED draws the multivariate t diagnostic's Monte Carlo vectors",
    ),
    20261114: SharedRoot(
        _owners("tests.studies.mtp_density_tails_fits", f"{PILOTS}option_outcome_by_arm_pilot"),
        "auxiliary: BOUND_SEED draws the Monte Carlo integral of a variant case's efficiency bound",
    ),
    20261115: SharedRoot(
        _owners(
            "hazard-multinode",
            f"{PILOTS}option_markov_order_pilot",
            f"{PILOTS}option_patience_pilot",
        ),
        "legacy: both Fix-8 pilots (2026-11-13) used one root, which hazard-multinode then "
        "declared as its seed (2026-11-14); the registered study keeps it",
        reseed={
            f"{PILOTS}option_markov_order_pilot": {
                "tests/studies/option_markov_order_pilot.csv": "c01ba1f4f9e972a2"
            },
            f"{PILOTS}option_patience_pilot": {
                "tests/studies/option_patience_pilot.csv": "f8dcca37f6ac8f40"
            },
        },
    ),
    20261126: SharedRoot(
        _owners("ep-assessment", "tests.studies._point_shift_learned_density_law"),
        "auxiliary: CONSTANT_SEED scrambles the Sobol points of a truth constant",
    ),
    20261131: SharedRoot(
        _owners(f"{PILOTS}learned_rule_ep_pilot", f"{PILOTS}isotonic_binding_pilot"),
        "legacy: isotonic_binding_pilot (2026-11-13) took learned_rule_ep_pilot's root "
        "(2026-11-05)",
        reseed={
            f"{PILOTS}isotonic_binding_pilot": {
                "tests/studies/isotonic_binding_pilot.csv": "f8370d9646d00734",
                "tests/studies/isotonic_binding_tolerance.csv": "dbb8a6aaab2908b4",
            }
        },
    ),
    20262321: SharedRoot(
        _owners("marginal-bound-limit-coverage", "ovb-limit-coverage"),
        "legacy: ovb-limit-coverage moved to fresh seeds on 2026-10-22 and landed on the seed "
        "marginal-bound-limit-coverage had declared 20 minutes earlier",
        reseed={"ovb-limit-coverage": {"tests/canonical/ovb_limit_coverage": "b718b61405578417"}},
    ),
    20263001: SharedRoot(
        _owners("learned-rule-cvtmle", "ctmle-derivative"),
        "legacy: ctmle-derivative (2026-10-12) took learned-rule-cvtmle's seed (2026-09-28)",
        reseed={"ctmle-derivative": {"tests/canonical/ctmle_derivative": "73fb174fdde0105d"}},
    ),
    20263002: SharedRoot(
        _owners("learned-rule-cvtmle", WITNESS),
        "legacy: the ctmle-derivative witness data (2026-10-12) draw from learned-rule-cvtmle's "
        "resampling root (2026-09-28)",
        reseed={WITNESS: {"tests/canonical/ctmle_derivative_witness": "d29d38119d885840"}},
    ),
    20268821: SharedRoot(
        _owners("canonical-stratified-drtmle", "stratified-drtmle-foldwise"),
        "stratified-drtmle-foldwise refits canonical-stratified-drtmle's registered draws on "
        "purpose",
        paired=_owners("canonical-stratified-drtmle", "stratified-drtmle-foldwise"),
    ),
    2026111201: SharedRoot(
        _owners("drtmle-foldwise-n1500", "small-sample-wald"),
        "legacy: small-sample-wald (2026-11-12 23:55 UTC) declared drtmle-foldwise-n1500's seed "
        "(21:10 UTC the same day)",
        reseed={"small-sample-wald": {"tests/canonical/small_sample_wald": "9b5038e30b6286ff"}},
    ),
}


def _int_constants(path: Path) -> Iterable[tuple[str, int]]:
    """Module-level ``NAME = <int literal>`` assignments whose name contains ``SEED``."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name) or "SEED" not in target.id:
            continue
        if isinstance(value, ast.Constant) and type(value.value) is int:
            yield target.id, value.value


def seed_roots(
    records: Iterable[object], root: Path, scanned: Iterable[str]
) -> dict[int, set[str]]:
    """Root value -> the owners that hold it."""
    records = tuple(records)
    module_owner: dict[str, str] = {}
    for record in records:
        module_owner[record.runner_module] = record.slug
        module_owner[record.properties_module] = record.slug
    owners: dict[int, set[str]] = defaultdict(set)
    for record in records:
        owners[record.seed].add(record.slug)
        if record.resampling_seed is not None:
            owners[record.resampling_seed].add(record.slug)
    for directory in scanned:
        for path in sorted((root / directory).rglob("*.py")):
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            for _, value in _int_constants(path):
                owners[value].add(module_owner.get(module, module))
    return dict(owners)


def shared(owners: Mapping[int, set[str]]) -> dict[int, frozenset[str]]:
    return {value: frozenset(held) for value, held in owners.items() if len(held) > 1}


@pytest.fixture(scope="module")
def roots() -> dict[int, frozenset[str]]:
    return shared(seed_roots(registered(), ROOT, SCANNED))


def test_no_two_owners_share_a_root_unless_listed(roots: dict[int, frozenset[str]]) -> None:
    unlisted = {
        value: sorted(owners)
        for value, owners in roots.items()
        if value not in SHARED_ROOTS or not owners <= SHARED_ROOTS[value].owners
    }
    assert not unlisted, (
        "these seed roots are held by more than one study, pilot or helper; give the newer "
        f"owner a fresh seed that no other owner uses: {unlisted}"
    )


def test_every_listed_root_is_still_shared(roots: dict[int, frozenset[str]]) -> None:
    stale = {
        value: sorted(entry.owners - roots.get(value, frozenset()))
        for value, entry in SHARED_ROOTS.items()
        if roots.get(value) != entry.owners
    }
    assert not stale, f"remove or narrow these SHARED_ROOTS entries; owners gone: {stale}"


def test_every_listed_root_names_its_reason() -> None:
    for value, entry in SHARED_ROOTS.items():
        assert entry.paired <= entry.owners and set(entry.reseed) <= entry.owners, value
        if entry.reseed:
            assert "legacy" in entry.reason, value
        elif entry.paired:
            assert entry.paired == entry.owners, value
        else:
            assert entry.reason.startswith("auxiliary: "), value


def _digest(path: Path) -> str:
    """What a run writes: a manifest's artifact hashes, or a committed summary's bytes."""
    if path.is_dir():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        payload = json.dumps(manifest["sha256"], sort_keys=True).encode()
    else:
        payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("value", "owner", "path", "pin"),
    [
        (value, owner, path, pin)
        for value, entry in SHARED_ROOTS.items()
        for owner, pins in entry.reseed.items()
        for path, pin in pins.items()
    ],
)
def test_a_rerun_draws_from_a_fresh_root(value: int, owner: str, path: str, pin: str) -> None:
    assert _digest(ROOT / path).startswith(pin), (
        f"{owner} re-ran, and it still draws from root {value}, which "
        f"{sorted(SHARED_ROOTS[value].owners - {owner})} also hold.  Give it a fresh seed and "
        "drop it from this SHARED_ROOTS entry.  Update the pin instead only if the rerun drew "
        "nothing from this root."
    )


@pytest.mark.parametrize(
    "value", [value for value, entry in SHARED_ROOTS.items() if entry.paired], ids=str
)
def test_paired_studies_share_their_samples(value: int) -> None:
    studies = [record for record in registered() if record.slug in SHARED_ROOTS[value].paired]
    common = [s for s in studies[0].scenarios if all(s in r.scenarios for r in studies)]
    assert common
    first, *others = (record.runner().draw_scenario(common[0], 200, 0)[0] for record in studies)
    for frame in others:
        pd.testing.assert_frame_equal(frame, first)


def _record(slug: str, seed: int, resampling_seed: int | None = None) -> SimpleNamespace:
    module = f"tests.studies.{slug}"
    return SimpleNamespace(
        slug=slug,
        seed=seed,
        resampling_seed=resampling_seed,
        runner_module=module,
        properties_module=f"{module}_properties",
    )


def test_the_scan_finds_a_planted_collision(tmp_path: Path) -> None:
    """Witness: a pilot constant on another study's seed, and one study's constant, are found."""
    studies = tmp_path / "tests" / "studies"
    studies.mkdir(parents=True)
    (studies / "alpha.py").write_text("SEED = 11\nRESAMPLING_SEED: int = 12\nN = 11\n")
    (studies / "alpha_pilot.py").write_text('PILOT_SEED = 11\nNAME = "SEED"\nOTHER_SEED = 1.0\n')
    (studies / "beta_properties.py").write_text("DESIGN_SEED = 12\n")
    records = [_record("alpha", 11, 12), _record("beta", 13)]
    found = shared(seed_roots(records, tmp_path, ["tests/studies"]))
    assert found == {
        11: frozenset({"alpha", "tests.studies.alpha_pilot"}),
        12: frozenset({"alpha", "beta"}),
    }
    # A study and the constants of its own runner and properties modules are one owner.
    (studies / "alpha_pilot.py").unlink()
    (studies / "beta_properties.py").write_text("DESIGN_SEED = 13\n")
    assert not shared(seed_roots(records, tmp_path, ["tests/studies"]))
