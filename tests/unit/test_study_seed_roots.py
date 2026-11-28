"""No two studies draw from one seed root unless this file says why.

Studies use distinct named seed streams, for the reason
:func:`tests.studies.evidence.seeds.stream_seed` gives for hashing stream labels.  Studies and
pilots spawn per-replication seeds as ``SeedSequence(entropy=root, spawn_key=...)``, so two owners
whose roots are equal draw identical seeds wherever their spawn keys agree.  The Monte Carlo errors
of rows a reader compares are then correlated by construction, and a pilot that shares a root with
the run it sizes reads that run's own draws.

A *holder* is one place a root is written down:

* a registered :class:`StudyRecord`'s ``seed`` or ``resampling_seed``;
* a module-level constant whose name ends in ``SEED``, written as an integer, as a tuple or list
  of integers, or as another same-module constant plus or minus an integer;
* an integer of at least :data:`INLINE_FLOOR` passed straight to ``default_rng``,
  ``SeedSequence`` or ``RandomState``, or as a ``seed=``, ``resampling_seed=`` or ``entropy=``
  keyword.  Smaller inline integers are the ``random_state=0``-style seeds of fixtures, which no
  study publishes as a root.

The scan reads every module under ``tests/`` except ``tests/unit`` and ``tests/e2e``, statically,
with :mod:`ast`.  A holder in a registered study's runner or properties module belongs to that
study; any other holder belongs to its module.  A root that another root's ``range`` sweeps, such
as a witness's rejected-seed search, is not read.

:data:`SHARED_ROOTS` lists the roots that more than one owner held when this check was added, by
holder, so an exemption covers only the constants it describes.  None is re-seeded here: each owner
already ran, and moving a seed under committed rows would leave a manifest or a pilot summary
describing draws that never ran (``CLAUDE.md``, "Tests and tooling"; the fast tier refits
committed replications from their seeds).  An entry is one of:

* a **pairing**: the owners share samples on purpose, which
  :func:`test_paired_studies_share_their_samples` checks;
* an **auxiliary** root: one holder seeds a Sobol scramble, a Monte Carlo constant, a perturbation
  map or a fold split, not an owner's sample draws;
* a **legacy** root: two owners drew samples from it before this check existed.  ``reseed`` names
  the owner that takes a fresh seed when it next draws, normally the one declared later, and pins
  its committed run, which :func:`test_a_rerun_draws_from_a_fresh_root` checks.

Every entry is a ratchet: an entry whose holders no longer share its root fails until it is
removed or narrowed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import pandas as pd
import pytest

from tests.studies.evidence.registry import ROOT, registered

#: Top-level directories of ``tests/`` that hold no study, pilot or helper roots.
SKIPPED = frozenset({"unit", "e2e"})
#: The smallest inline integer read as a root.  Every root a study publishes is date-like.
INLINE_FLOOR = 1_000_000
#: A constant whose name matches this is a root; ``N_SEEDS`` and ``*_SEED_TRIALS`` are counts.
CONSTANT = re.compile(r"SEED$")
SEEDERS = frozenset({"default_rng", "SeedSequence", "RandomState"})
SEED_KEYWORDS = frozenset({"seed", "resampling_seed", "entropy"})

#: ``(owner, holder)``: a study slug or a dotted module name, and the constant or field there.
Holder = tuple[str, str]


class Record(Protocol):
    """What the scan reads from a :class:`StudyRecord`."""

    slug: str
    seed: int
    resampling_seed: int | None
    runner_module: str
    properties_module: str


@dataclass(frozen=True)
class RunPin:
    """A registered study's committed run, named by the commit its manifest records.

    Parameters
    ----------
    artifacts : str
        The study's artifact directory, relative to the repository.
    commit : str
        A prefix of the manifest's ``generated_with.subject.cleverly_commit``, which changes only
        when the study is regenerated.
    """

    artifacts: str
    commit: str


@dataclass(frozen=True)
class FilePin:
    """A pilot's or witness's committed output, and the constant it was drawn from.

    Parameters
    ----------
    module : str
        The module, relative to the repository, that holds the constant.
    constant : str
        The constant's name.
    files : Mapping
        Committed file -> a prefix of the sha256 of its bytes.
    """

    module: str
    constant: str
    files: Mapping[str, str]


@dataclass(frozen=True)
class SharedRoot:
    """One seed root that several owners hold, and why that is accepted.

    Parameters
    ----------
    holders : frozenset of Holder
        Every ``(owner, holder)`` that writes the root down.
    reason : str
        Why the owners share the root.
    paired : frozenset of str
        Owners that share samples on purpose.  Their first replications must be the same frames.
    reseed : Mapping
        Owner -> the pin of the committed run it drew from this root.  The owner takes a fresh
        seed at its next run.
    """

    holders: frozenset[Holder]
    reason: str
    paired: frozenset[str] = frozenset()
    reseed: Mapping[str, RunPin | FilePin] = field(default_factory=dict)

    @property
    def owners(self) -> frozenset[str]:
        return frozenset(owner for owner, _ in self.holders)


def _study(slug: str, *names: str) -> set[Holder]:
    """A registered study's record field (``seed``, ``resampling_seed``) and named constants."""
    fields = {"seed": "record.seed", "resampling_seed": "record.resampling_seed"}
    return {(slug, fields.get(name, name)) for name in names}


def _module(module: str, *names: str) -> set[Holder]:
    return {(module, name) for name in names}


def _holders(*groups: set[Holder]) -> frozenset[Holder]:
    return frozenset().union(*groups)


def _run(artifacts: str, commit: str) -> RunPin:
    return RunPin(f"tests/canonical/{artifacts}", commit)


#: The constant pair a registered study's root is usually written as.
SEED = ("seed", "SEED")
RESAMPLING = ("resampling_seed", "RESAMPLING_SEED")
#: A registered study whose resampling root is written inline in its ``StudyRecord`` call.
INLINE_RESAMPLING = ("resampling_seed",)
PILOT = "tests.studies."
WITNESS = "tests.canonical.ctmle_derivative_witness.regenerate"

SHARED_ROOTS: dict[int, SharedRoot] = {
    0: SharedRoot(
        _holders(
            _study("structural-nested-mean", "COMPARATOR_FOLD_SEED"),
            _module("tests.studies.long_horizon_reference", "FOLD_SEED"),
        ),
        "auxiliary: both are fold-split seeds equal to the random_state=0 their fits pass, not "
        "sample draws",
    ),
    20240822: SharedRoot(
        _holders(_study("canonical-ctmle-selector", *SEED), _study("fold-targeted-cvtmle", *SEED)),
        "legacy: fold-targeted-cvtmle (2026-08-31) took canonical-ctmle-selector's seed "
        "(2026-08-21)",
        reseed={"fold-targeted-cvtmle": _run("zepid_cvtmle", "d546049f76b8")},
    ),
    20260824: SharedRoot(
        _holders(
            _study("canonical-ltmle-crossfit", *SEED),
            _study("canonical-drtmle", *SEED),
            _study("drtmle-foldwise", "seed"),
        ),
        "legacy: canonical-drtmle (2026-08-25) took canonical-ltmle-crossfit's seed "
        "(2026-08-24); drtmle-foldwise refits canonical-drtmle's registered draws on purpose, so "
        "the pair keeps the root and canonical-ltmle-crossfit re-seeds",
        paired=frozenset({"canonical-drtmle", "drtmle-foldwise"}),
        reseed={"canonical-ltmle-crossfit": _run("lmtp_ltmle", "fbf0c5c40eb0")},
    ),
    20260826: SharedRoot(
        _holders(
            _study("canonical-drtmle", *INLINE_RESAMPLING),
            _study("canonical-ltmle-competing", *SEED),
        ),
        "legacy: canonical-drtmle's resampling root is canonical-ltmle-competing's seed, both "
        "declared 2026-08-25",
        reseed={"canonical-ltmle-competing": _run("lmtp_ltmle_competing", "7e559f970774")},
    ),
    20260827: SharedRoot(
        _holders(
            _study("canonical-ltmle-competing-crossfit", *SEED),
            _study("canonical-categorical-ltmle", *SEED),
            _study("canonical-multi-arm-tmle", *SEED),
        ),
        "legacy: the categorical and multi-arm studies (2026-08-27) took the competing "
        "cross-fit study's seed (2026-08-25)",
        reseed={
            "canonical-categorical-ltmle": _run("categorical_ltmle", "50b808f4c9bf"),
            "canonical-multi-arm-tmle": _run("multi_arm_tmle3", "a5e160288f4c"),
        },
    ),
    20260828: SharedRoot(
        _holders(
            _study("canonical-ltmle-competing", *INLINE_RESAMPLING),
            _study("point-msm", *SEED),
            _study("canonical-categorical-ltmle-crossfit", *SEED),
            _study("canonical-multi-arm-ctmle-oat", *SEED),
        ),
        "legacy: three seeds declared 2026-08-26 and 2026-08-27 equal canonical-ltmle-competing's "
        "resampling root (2026-08-25)",
        reseed={
            "point-msm": _run("tmle3_msm", "c355a275dc6e"),
            "canonical-categorical-ltmle-crossfit": _run(
                "categorical_ltmle_crossfit", "fbf0c5c40eb0"
            ),
            "canonical-multi-arm-ctmle-oat": _run("multi_arm_ctmle3_oat", "a01177b6ce7f"),
        },
    ),
    20260829: SharedRoot(
        _holders(
            _study("longitudinal-msm", *SEED), _study("canonical-multi-arm-ctmle-selector", *SEED)
        ),
        "legacy: canonical-multi-arm-ctmle-selector (2026-08-27) took longitudinal-msm's seed "
        "(2026-08-26)",
        reseed={
            "canonical-multi-arm-ctmle-selector": _run("multi_arm_ctmle_selector", "474cda363b12")
        },
    ),
    20260830: SharedRoot(
        _holders(
            _study("point-msm", *INLINE_RESAMPLING),
            _study("canonical-multi-arm-drtmle", *SEED),
            _study("multi-arm-drtmle-foldwise", "seed"),
        ),
        "legacy: canonical-multi-arm-drtmle's seed (2026-08-27) is point-msm's resampling root "
        "(2026-08-26); multi-arm-drtmle-foldwise refits canonical-multi-arm-drtmle's registered "
        "draws on purpose, so the pair keeps the root and point-msm re-seeds",
        paired=frozenset({"canonical-multi-arm-drtmle", "multi-arm-drtmle-foldwise"}),
        reseed={"point-msm": _run("tmle3_msm", "c355a275dc6e")},
    ),
    20260924: SharedRoot(
        _holders(_study("mar-drtmle", *INLINE_RESAMPLING), _study("repeated-crossfit-tmle", *SEED)),
        "legacy: repeated-crossfit-tmle's seed (2026-08-28) is mar-drtmle's resampling root "
        "(2026-08-27)",
        reseed={"repeated-crossfit-tmle": _run("repeated_crossfit", "b545a579945a")},
    ),
    20261001: SharedRoot(
        _holders(
            _study("canonical-multi-arm-tmle", *INLINE_RESAMPLING), _study("weighted-tmle", *SEED)
        ),
        "legacy: weighted-tmle's seed (2026-08-29) is canonical-multi-arm-tmle's resampling root "
        "(2026-08-27)",
        reseed={"weighted-tmle": _run("tmle_weighted", "a10d1a4a24c1")},
    ),
    20261002: SharedRoot(
        _holders(
            _study("canonical-multi-arm-ctmle-oat", *INLINE_RESAMPLING),
            _study("default-simultaneous-bands", "DESIGN_SEED"),
        ),
        "auxiliary: DESIGN_SEED draws the normal vectors of default-simultaneous-bands' design "
        "computation",
    ),
    20261014: SharedRoot(
        _holders(
            _module("tests.canonical.sequential_baselines.make_data", "SEED"),
            _module("tests.studies.evidence.property_failures", "AFFINE_SEED"),
        ),
        "auxiliary: AFFINE_SEED seeds the affine perturbation maps of the property-failure "
        "controls",
    ),
    20261015: SharedRoot(
        _holders(
            _module("tests.canonical.stremr.make_data", "SEED"),
            _module("tests.studies.evidence.property_failures", "MIXED_SEED"),
        ),
        "auxiliary: MIXED_SEED seeds the mixed perturbation draws of the property-failure controls",
    ),
    20261016: SharedRoot(
        _holders(
            _module("tests.canonical.property_differential", "SYNTHETIC_SEED"),
            _module("tests.studies.oat_multi_arm_quadrature", "default_rng() in critical_value"),
        ),
        "auxiliary: critical_value draws the Monte Carlo vectors of an exact-law band's critical "
        "value, and SYNTHETIC_SEED seeds synthetic differential-test rows",
    ),
    20261109: SharedRoot(
        _holders(
            _study("stacked-mar-natural-course-cvtmle", *SEED),
            _study("targeting-curves", "SOBOL_SEED"),
        ),
        "auxiliary: targeting-curves' SOBOL_SEED scrambles the Sobol set of cell S's truth",
    ),
    20261110: SharedRoot(
        _holders(
            _study("stacked-mar-natural-course-cvtmle", *INLINE_RESAMPLING),
            _study("targeting-curves", "DESIGN_SOBOL_SEED"),
        ),
        "auxiliary: targeting-curves' DESIGN_SOBOL_SEED scrambles the design's Sobol set",
    ),
    20261112: SharedRoot(
        _holders(
            _study("band-t-reference", "SIZING_SEED"),
            _module(f"{PILOT}option_stopping_pilot", "PILOT_SEED"),
            _module(f"{PILOT}targeting_curves_pilot", "SOBOL_CHECK_SEED"),
        ),
        "auxiliary: band-t-reference's SIZING_SEED draws its sizing simulation and "
        "SOBOL_CHECK_SEED scrambles a Sobol check set; only the stopping pilot draws samples",
    ),
    20261113: SharedRoot(
        _holders(
            _module("tests.studies.evidence.simultaneous", "MVT_SEED"),
            _module(f"{PILOT}option_treatment_adjustment_pilot", "PILOT_SEED"),
        ),
        "auxiliary: MVT_SEED draws the multivariate t diagnostic's Monte Carlo vectors",
    ),
    20261114: SharedRoot(
        _holders(
            _module("tests.studies.mtp_density_tails_fits", "BOUND_SEED"),
            _module(f"{PILOT}option_outcome_by_arm_pilot", "PILOT_SEED"),
        ),
        "auxiliary: BOUND_SEED draws the Monte Carlo integral of a variant case's efficiency bound",
    ),
    20261115: SharedRoot(
        _holders(
            _study("hazard-multinode", *SEED),
            _module(f"{PILOT}option_markov_order_pilot", "PILOT_SEED"),
            _module(f"{PILOT}option_patience_pilot", "PILOT_SEED"),
        ),
        "legacy: both Fix-8 pilots (2026-11-13) used one root, which hazard-multinode then "
        "declared as its seed (2026-11-14); the registered study keeps it",
        reseed={
            f"{PILOT}option_markov_order_pilot": FilePin(
                "tests/studies/option_markov_order_pilot.py",
                "PILOT_SEED",
                {"tests/studies/option_markov_order_pilot.csv": "c01ba1f4f9e972a2"},
            ),
            f"{PILOT}option_patience_pilot": FilePin(
                "tests/studies/option_patience_pilot.py",
                "PILOT_SEED",
                {"tests/studies/option_patience_pilot.csv": "f8dcca37f6ac8f40"},
            ),
        },
    ),
    20261126: SharedRoot(
        _holders(
            _study("ep-assessment", *RESAMPLING),
            _module("tests.studies._point_shift_learned_density_law", "CONSTANT_SEED"),
        ),
        "auxiliary: CONSTANT_SEED scrambles the Sobol points of a truth constant",
    ),
    20261131: SharedRoot(
        _holders(
            _module(f"{PILOT}learned_rule_ep_pilot", "PILOT_SEED"),
            _module(f"{PILOT}isotonic_binding_pilot", "PILOT_SEED"),
        ),
        "legacy: isotonic_binding_pilot (2026-11-13) took learned_rule_ep_pilot's root "
        "(2026-11-05)",
        reseed={
            f"{PILOT}isotonic_binding_pilot": FilePin(
                "tests/studies/isotonic_binding_pilot.py",
                "PILOT_SEED",
                {
                    "tests/studies/isotonic_binding_pilot.csv": "f8370d9646d00734",
                    "tests/studies/isotonic_binding_tolerance.csv": "dbb8a6aaab2908b4",
                },
            )
        },
    ),
    20262321: SharedRoot(
        _holders(
            _study("marginal-bound-limit-coverage", *SEED), _study("ovb-limit-coverage", *SEED)
        ),
        "legacy: ovb-limit-coverage moved to fresh seeds on 2026-10-22 and landed on the seed "
        "marginal-bound-limit-coverage had declared 20 minutes earlier",
        reseed={"ovb-limit-coverage": _run("ovb_limit_coverage", "9baf8d16a7b2")},
    ),
    20263001: SharedRoot(
        _holders(_study("learned-rule-cvtmle", *SEED), _study("ctmle-derivative", *SEED)),
        "legacy: ctmle-derivative (2026-10-12) took learned-rule-cvtmle's seed (2026-09-28)",
        reseed={"ctmle-derivative": _run("ctmle_derivative", "553f01077b29")},
    ),
    20263002: SharedRoot(
        _holders(_study("learned-rule-cvtmle", *RESAMPLING), _module(WITNESS, "SEED")),
        "legacy: the ctmle-derivative witness data (2026-10-12) draw from learned-rule-cvtmle's "
        "resampling root (2026-09-28)",
        reseed={
            WITNESS: FilePin(
                "tests/canonical/ctmle_derivative_witness/regenerate.py",
                "SEED",
                {"tests/canonical/ctmle_derivative_witness/inputs.csv": "deae86c9925bd646"},
            )
        },
    ),
    20263003: SharedRoot(
        _holders(
            _study("learned-rule-cvtmle-boundary", *SEED),
            _module(f"{PILOT}ctmle_derivative_dr_pilot", "default_rng() in projection_kl"),
        ),
        "auxiliary: projection_kl draws the covariates of a Monte Carlo KL gap",
    ),
    20263004: SharedRoot(
        _holders(
            _study("learned-rule-cvtmle-boundary", *RESAMPLING),
            _module(f"{PILOT}ctmle_derivative_dr_pilot", "default_rng() in truncation_share"),
        ),
        "auxiliary: truncation_share draws the covariates of a Monte Carlo truncation share",
    ),
    20268821: SharedRoot(
        _holders(
            _study("canonical-stratified-drtmle", *SEED),
            _study("stratified-drtmle-foldwise", "seed"),
        ),
        "stratified-drtmle-foldwise refits canonical-stratified-drtmle's registered draws on "
        "purpose",
        paired=frozenset({"canonical-stratified-drtmle", "stratified-drtmle-foldwise"}),
    ),
    2026111201: SharedRoot(
        _holders(_study("drtmle-foldwise-n1500", *SEED), _study("small-sample-wald", *SEED)),
        "legacy: small-sample-wald (2026-11-12 23:55 UTC) declared drtmle-foldwise-n1500's seed "
        "(21:10 UTC the same day)",
        reseed={"small-sample-wald": _run("small_sample_wald", "34ed36672fb2")},
    ),
}


def _integer(node: ast.expr, constants: Mapping[str, int]) -> int | None:
    """An integer literal, or a same-module integer constant plus or minus one."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add | ast.Sub)
        and isinstance(node.left, ast.Name)
        and node.left.id in constants
        and isinstance(node.right, ast.Constant)
        and type(node.right.value) is int
    ):
        offset = node.right.value if isinstance(node.op, ast.Add) else -node.right.value
        return constants[node.left.id] + offset
    return None


def _assignments(tree: ast.Module) -> Iterator[tuple[str, ast.expr]]:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            yield node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            yield node.target.id, node.value


class _Calls(ast.NodeVisitor):
    """Inline roots: integers handed to a seeder or a seed keyword, named by enclosing scope."""

    def __init__(self, constants: Mapping[str, int], skip_records: bool) -> None:
        self.constants = constants
        self.skip_records = skip_records
        self.scope = ["<module>"]
        self.found: list[tuple[str, int]] = []

    def _enter(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _enter

    def _inline(self, node: ast.expr) -> int | None:
        value = _integer(node, self.constants)
        return value if value is not None and value >= INLINE_FLOOR else None

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        where = self.scope[-1]
        if name in SEEDERS and node.args and (value := self._inline(node.args[0])) is not None:
            self.found.append((f"{name}() in {where}", value))
        if not (self.skip_records and name == "StudyRecord"):
            for keyword in node.keywords:
                if keyword.arg in SEED_KEYWORDS and (value := self._inline(keyword.value)):
                    self.found.append((f"{name}({keyword.arg}=) in {where}", value))
        self.generic_visit(node)


def module_holders(path: Path, *, registered_module: bool = False) -> list[tuple[str, int]]:
    """Every ``(holder, root)`` the module writes down.

    A registered study's own ``StudyRecord`` call is skipped: its roots are read off the record.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants: dict[str, int] = {}
    found: list[tuple[str, int]] = []
    for name, node in _assignments(tree):
        value = _integer(node, constants)
        if value is not None:
            constants[name] = value
        if not CONSTANT.search(name):
            continue
        if value is not None:
            found.append((name, value))
        elif isinstance(node, ast.Tuple | ast.List):
            for index, element in enumerate(node.elts):
                if (item := _integer(element, constants)) is not None:
                    found.append((f"{name}[{index}]", item))
    calls = _Calls(constants, skip_records=registered_module)
    calls.visit(tree)
    return found + calls.found


def seed_roots(records: Iterable[Record], tests: Path) -> dict[int, set[Holder]]:
    """Root value -> every ``(owner, holder)`` that writes it down, under ``tests``."""
    records = tuple(records)
    module_owner: dict[str, str] = {}
    for record in records:
        module_owner[record.runner_module] = record.slug
        module_owner[record.properties_module] = record.slug
    holders: dict[int, set[Holder]] = defaultdict(set)
    for record in records:
        holders[record.seed].add((record.slug, "record.seed"))
        if record.resampling_seed is not None:
            holders[record.resampling_seed].add((record.slug, "record.resampling_seed"))
    for path in sorted(tests.rglob("*.py")):
        relative = path.relative_to(tests.parent)
        if relative.parts[1] in SKIPPED:
            continue
        module = ".".join(relative.with_suffix("").parts)
        owner = module_owner.get(module, module)
        for holder, value in module_holders(path, registered_module=module in module_owner):
            holders[value].add((owner, holder))
    return dict(holders)


def shared(holders: Mapping[int, set[Holder]]) -> dict[int, frozenset[Holder]]:
    """The roots that more than one owner writes down."""
    return {
        value: frozenset(held)
        for value, held in holders.items()
        if len({owner for owner, _ in held}) > 1
    }


@pytest.fixture(scope="module")
def roots() -> dict[int, frozenset[Holder]]:
    """The shared roots of the repository as it stands."""
    return shared(seed_roots(registered(), ROOT / "tests"))


def test_no_two_owners_share_a_root_unless_listed(roots: dict[int, frozenset[Holder]]) -> None:
    unlisted = {
        value: sorted(held)
        for value, held in roots.items()
        if value not in SHARED_ROOTS or not held <= SHARED_ROOTS[value].holders
    }
    assert not unlisted, (
        "these seed roots are written down by more than one study, pilot or helper; give the "
        f"newer holder a fresh seed that nothing else uses: {unlisted}"
    )


def test_every_listed_root_is_still_shared(roots: dict[int, frozenset[Holder]]) -> None:
    stale = {
        value: sorted(entry.holders - roots.get(value, frozenset()))
        for value, entry in SHARED_ROOTS.items()
        if roots.get(value) != entry.holders
    }
    assert not stale, (
        "remove or narrow these SHARED_ROOTS entries; holders gone: "
        f"{stale}.  A seed that moved needs a rerun that drew from the new root, which "
        "test_a_rerun_draws_from_a_fresh_root checks while the entry stands."
    )


def test_every_listed_root_names_its_reason() -> None:
    for value, entry in SHARED_ROOTS.items():
        assert entry.paired <= entry.owners and set(entry.reseed) <= entry.owners, value
        if entry.reseed:
            assert "legacy" in entry.reason, value
        elif entry.paired:
            assert entry.paired == entry.owners, value
        else:
            assert entry.reason.startswith("auxiliary: "), value


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _constant(module: str, name: str) -> int | None:
    holders = dict(module_holders(ROOT / module))
    return holders.get(name)


@pytest.mark.parametrize(
    ("value", "owner"),
    [(value, owner) for value, entry in SHARED_ROOTS.items() for owner in entry.reseed],
    ids=str,
)
def test_a_rerun_draws_from_a_fresh_root(value: int, owner: str) -> None:
    pin = SHARED_ROOTS[value].reseed[owner]
    others = sorted(SHARED_ROOTS[value].owners - {owner})
    if isinstance(pin, RunPin):
        # A registered study's seed cannot move without a rerun: the manifest-hash test of
        # test_method_evidence fails on a configuration seed the record no longer declares.
        manifest = json.loads((ROOT / pin.artifacts / "manifest.json").read_text("utf-8"))
        commit = manifest["generated_with"]["subject"]["cleverly_commit"]
        assert commit.startswith(pin.commit), (
            f"{owner} was regenerated and still draws from root {value}, which {others} also "
            "hold.  Give it a fresh seed and drop it from this SHARED_ROOTS entry."
        )
        return
    unchanged = {path: _sha256(path).startswith(digest) for path, digest in pin.files.items()}
    if _constant(pin.module, pin.constant) == value:
        assert all(unchanged.values()), (
            f"{owner} re-ran and still draws from root {value}, which {others} also hold.  Give "
            f"{pin.constant} a fresh value and drop it from this SHARED_ROOTS entry."
        )
    else:
        assert not any(unchanged.values()), (
            f"{owner}'s {pin.constant} moved off root {value}, but {sorted(unchanged)} are the "
            "committed draws of the old root.  Rerun it on the new root, or restore the seed."
        )


@pytest.mark.parametrize(
    "value", [value for value, entry in SHARED_ROOTS.items() if entry.paired], ids=str
)
def test_paired_studies_share_their_samples(value: int) -> None:
    studies = [record for record in registered() if record.slug in SHARED_ROOTS[value].paired]
    common = [s for s in studies[0].scenarios if all(s in r.scenarios for r in studies)]
    assert common
    for replicate in (0, 1):
        first, *others = (
            record.runner().draw_scenario(common[0], 200, replicate)[0] for record in studies
        )
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


def test_the_scan_finds_planted_collisions(tmp_path: Path) -> None:
    """Witness: each way of writing a root down is found, and a study's own roots are one owner."""
    tests = tmp_path / "tests"
    studies = tests / "studies"
    studies.mkdir(parents=True)
    (tests / "unit").mkdir()
    (studies / "alpha.py").write_text(
        "SEED = 11\nRESAMPLING_SEED: int = 12\nN = 11\n"
        "STUDY = StudyRecord(seed=SEED, resampling_seed=2_000_012)\n"
    )
    (studies / "alpha_pilot.py").write_text(
        'PILOT_SEED = 11\nNAME = "SEED"\nOTHER_SEED = 1.0\nN_SEEDS = 12\n'
        "LIMIT_SEEDS = 3\nBASE_SEED = 1_999_999\nNEXT_SEED = BASE_SEED + 1\n"
    )
    (studies / "beta_properties.py").write_text(
        "DESIGN_SEED = 12\nGRID_SEED = (5, 3_000_000)\n"
        "def check():\n    return np.random.default_rng(3_000_000), f(seed=4_000_000, n=0)\n"
    )
    (tests / "gamma_pilot.py").write_text(
        "def draw():\n    return SeedSequence(entropy=4_000_000), default_rng(13)\n"
    )
    (tests / "unit" / "test_x.py").write_text("SEED = 11\n")
    records = [_record("alpha", 11, 2_000_000), _record("beta", 13)]
    found = shared(seed_roots(records, tests))
    assert found == {
        11: frozenset(
            {("alpha", "record.seed"), ("alpha", "SEED"), (f"{PILOT}alpha_pilot", "PILOT_SEED")}
        ),
        12: frozenset({("alpha", "RESAMPLING_SEED"), ("beta", "DESIGN_SEED")}),
        2_000_000: frozenset(
            {("alpha", "record.resampling_seed"), (f"{PILOT}alpha_pilot", "NEXT_SEED")}
        ),
        4_000_000: frozenset(
            {("beta", "f(seed=) in check"), ("tests.gamma_pilot", "SeedSequence(entropy=) in draw")}
        ),
    }
    # Within one study the tuple element and the inline seed of the same value are one owner.
    assert 3_000_000 not in found and 13 not in found
