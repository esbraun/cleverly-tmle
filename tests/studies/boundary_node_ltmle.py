r"""Registered evidence for the boundary-node supplement of ``LTMLE`` (X54 part (i)).

An ``LTMLE`` node regression whose fitted rows hold one value drops the node's term from the
influence curve. Inside the supplement's reach the estimate adds the node's variance at the
exact one-sided binomial bound of its hazard (:mod:`cleverly.inference.node_variance`). This
study measures that construction on the laws of :mod:`tests.studies.boundary_node_laws`, where
the last node is flagged on about half the draws at :math:`n = 200`.

**Primary.** In-sample iterated-mean ``LTMLE`` on law A at :math:`n = 200`, 1,000 replications:
the risks of ``always`` and ``never`` at horizon 4 and their contrast. No comparator can be
paired: R ``survtmle`` reads such a node as zero and has no supplement, so it would reproduce
only the ``plugin`` arm. The record declares no reference and publishes an empty equivalence
artifact.

**Property ``boundary_node``.** Eight cells, each fitted on 4,000 draws
(:data:`CELLS`): law A in sample at :math:`n = 100, 200, 400`, cross-fitted at :math:`n = 200`
with five folds, clustered at 40 clusters of 10 rows, and its zero variant at :math:`n = 200`,
on the iterated-mean path; law B in sample and its zero variant at :math:`n = 200` on the hazard
path, which targets a held baseline treatment only. Each draw fits once, and every target
(``always``, ``never``, their contrast at horizon 4) publishes three arms read off that one fit:

==================  ==========================================================================
arm                 interval
==================  ==========================================================================
``supplemented``    the shipped interval, the curve's variance plus the supplement (positive)
``plugin``          the same estimate at the curve's variance alone: the supplement removed
``oracle_node``     the curve's variance plus the flagged nodes' variance at the true row
                    hazards, in the same Minkowski form (diagnostic)
==================  ==========================================================================

Each row also records whether node 4 was flagged for the target's regimen (``firing``: under
``always`` for ``always`` and the contrast, under ``never`` for ``never``), whether another
node was (``multi_flag``), whether the estimate added the supplement at any node
(``supplemented``), whether the estimate withheld its interval (``withheld``, then a miss in
every arm, with ``std_error`` NaN), and the half-width of the arm's interval. The rules are those of
:mod:`tests.studies.boundary_node_properties`.

Learners: GLM main terms for the node regressions (the study's quasi-binomial GLM) and an
unpenalized logistic regression for the hazards, with the known treatment mechanisms. The policy
is ``reporting``, declared before any run.
"""

from __future__ import annotations

import dataclasses
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.longitudinal import sequential as sequential_module
from tests.parallel import STUDY_JOBS
from tests.studies import boundary_node_laws as laws
from tests.studies.canonical_ltmle import regimen_rows
from tests.studies.canonical_ltmle_survival import horizon_initials
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.fractional_glm import QuasiBinomialGLM

PRIMARY_REPLICATES: Final[int] = 1_000
PRIMARY_N: Final[int] = 200
PROPERTY_REPLICATES: Final[int] = 4_000
#: The redeclaration's seeds (plan 5.5 rev 3.3), used by no other stream. The first declared
#: run (:data:`FIRST_DECLARED_RUN`) drew from 20264541 and 20264542, so the rerun reads clause
#: (ii') on draws that did not inform its design.
SEED: Final[int] = 20264751
RESAMPLING_SEED: Final[int] = 20264752
SCENARIO: Final[str] = "interior_n200"
FAMILY: Final[str] = "boundary_node"
HORIZON: Final[int] = laws.HORIZON

#: The first declared run, kept in history: the declaration it ran, the commit of its
#: artifacts, and its seeds. Its paired clause (ii) compared the plug-in's firing coverage with
#: its coverage on non-firing draws, which under-cover for the rare-event reason, and went red
#: although the supplement gained 304 firing draws and lost none. This declaration replaces that
#: clause with clause (ii') and is informed by the run.
FIRST_DECLARED_RUN: Final[Mapping[str, Any]] = {
    "declaration": "91cb39ae0469937a54aef7cf9cd8031faccb349d",
    "artifacts": "cbbd634c9b9c6eaf2c20f4c5d8e2e30ad1f97039",
    "seed": 20264541,
    "resampling_seed": 20264542,
}

TARGETS: Final[dict[str, str]] = {
    "always": f"risk_regimen[always @ t={HORIZON}]",
    "never": f"risk_regimen[never @ t={HORIZON}]",
    "contrast": f"ate_regimen[always vs never @ t={HORIZON}]",
}
#: The regimen whose node 4 a target's ``firing`` reads.
FIRING_REGIMEN: Final[dict[str, str]] = {"always": "always", "never": "never", "contrast": "always"}
ESTIMANDS: Final[tuple[str, ...]] = tuple(TARGETS.values())
#: The legs of a primary truth row, as ``performance-tests.csv`` publishes them.
PRIMARY_LEGS: Final[tuple[str, ...]] = ("bias_equivalent", "coverage_valid", "se_calibrated")


@dataclasses.dataclass(frozen=True)
class RedReading:
    """A primary row's declared red reading: its roadmap owner and the legs the owner may hold.

    Parameters
    ----------
    owner : str
        The ``id`` of the row's owner in the roadmap's "Red-cell owners" table.
    legs : tuple of str
        The legs of :data:`PRIMARY_LEGS` whose failure the reading explains. A failure of any
        other leg is not covered by the owner and is a defect suspect.
    reason : str
        Why the leg can fail with no defect.
    """

    owner: str
    legs: tuple[str, ...]
    reason: str


#: The readings declared before the run for the primary rows (``regenerate.py``). The owner of
#: a red row claims the whole row in the red-cell ledger, so :func:`unowned_primary_failures`
#: keeps a bias or coverage failure from hiding under a width owner.
PRIMARY_RED_READINGS: Final[dict[str, RedReading]] = {
    TARGETS["always"]: RedReading(
        "X54-supplement-width",
        ("se_calibrated",),
        "the supplement adds a flagged node's variance at the one-sided bound of its hazard, so "
        "the SE ratio is predicted above 1.20",
    ),
    TARGETS["contrast"]: RedReading(
        "X54-supplement-width",
        ("se_calibrated",),
        "the contrast reads the supplemented risk under always, so its SE ratio is predicted "
        "near or above 1.20",
    ),
    TARGETS["never"]: RedReading(
        "X54-rare-event-coverage",
        ("coverage_valid",),
        "node 4 is flagged under never on one draw in ten, so most draws show the ordinary "
        "interval at a node with one or two events: F31's rare-event limit, the reading the "
        "property family gives a quiet-draw shortfall (rare_event_limit)",
    ),
}


def unowned_primary_failures(performance: pd.DataFrame) -> list[str]:
    """Every failed primary leg that no declared reading covers, one sentence each.

    Parameters
    ----------
    performance : DataFrame
        The study's ``performance-tests.csv``: one row per estimand, with the boolean columns of
        :data:`PRIMARY_LEGS`.

    Returns
    -------
    list of str
        Empty when each failed leg is one its row's reading names.
    """
    out = []
    for row in performance.itertuples(index=False):
        reading = PRIMARY_RED_READINGS.get(str(row.estimand))
        allowed = () if reading is None else reading.legs
        for leg in PRIMARY_LEGS:
            if not bool(getattr(row, leg)) and leg not in allowed:
                out.append(f"{row.estimand}: {leg} failed, and no declared reading covers it")
    return out


ARMS: Final[tuple[str, ...]] = ("supplemented", "plugin", "oracle_node")


@dataclasses.dataclass(frozen=True)
class Cell:
    """One property cell: a law variant, a size, folds, a cluster size and a path.

    Parameters
    ----------
    name : str
        The cell's published prefix.
    variant : str
        The law variant of :func:`tests.studies.boundary_node_laws.law_of`.
    n : int
        Rows per draw.
    n_folds : int
        Cross-fitting folds; 1 in sample.
    cluster_size : int or None
        Rows per cluster, ``None`` on independent rows.
    hazard : bool
        Whether the cell fits the hazard path (law B) rather than the iterated-mean path.
    """

    name: str
    variant: str
    n: int
    n_folds: int = 1
    cluster_size: int | None = None
    hazard: bool = False

    @property
    def zero(self) -> bool:
        """Whether the cell reads a zero variant."""
        return self.variant in {"zero", "held_zero"}


CELLS: Final[tuple[Cell, ...]] = (
    Cell("interior_n100", "interior", 100),
    Cell("interior_n200", "interior", 200),
    Cell("interior_n400", "interior", 400),
    Cell("interior_n200_crossfit", "interior", 200, n_folds=5),
    Cell("clustered_j40", "clustered", 400, cluster_size=10),
    Cell("zero_n200", "zero", 200),
    Cell("held_n200", "held", 200, hazard=True),
    Cell("held_zero_n200", "held_zero", 200, hazard=True),
)
CELL_BY_NAME: Final[dict[str, Cell]] = {cell.name: cell for cell in CELLS}


def cell_name(cell: str, target: str, arm: str) -> str:
    """The published name of one property cell."""
    return f"{cell}__{target}__{arm}"


def arm_role(cell: str, target: str, arm: str) -> str:
    """``positive`` for the shipped interval, ``control`` for the paired control's plug-in arm."""
    if arm == "supplemented":
        return "positive"
    if arm == "plugin" and (cell, target) == PAIRED_CONTROL:
        return "control"
    return "diagnostic"


#: The cell and target of the ``paired_attributable`` control (plan 5.5, B1).
PAIRED_CONTROL: Final[tuple[str, str]] = ("interior_n200", "always")

PUBLISHED_CELLS: Final[tuple[str, ...]] = tuple(
    cell_name(cell.name, target, arm) for cell in CELLS for target in TARGETS for arm in ARMS
)

STUDY = StudyRecord(
    name="the boundary-node supplement of LTMLE",
    slug="boundary-node-ltmle",
    artifacts=ROOT / "tests" / "canonical" / "boundary_node_ltmle",
    document="docs/technical-reference/method-evidence/boundary-node-ltmle.md",
    anchor="the-boundary-node-supplement-of-ltmle",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    failure_bound={"cleverly": 0.05},
    modules=(
        "tests/studies/boundary_node_ltmle.py",
        "tests/studies/boundary_node_properties.py",
        "tests/studies/boundary_node_laws.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_ltmle_survival.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/inference/node_variance.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.boundary_node_ltmle",
    properties_module="tests.studies.boundary_node_properties",
    property_cells={FAMILY: PUBLISHED_CELLS},
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "the curve's variance plus the boundary-node supplement at the exact "
    "one-sided binomial bound of a flagged node's hazard",
    "law_a": "three known treatment decisions before event nodes 1, 3 and 4; iterated-mean path",
    "law_b": "one held baseline decision; hazard path",
    "law_a_hazards": [float(x) for x in laws.law_of("interior").alpha],
    "law_b_hazards": [float(x) for x in laws.law_of("held").alpha],
    "outcome_learner": "quasi-binomial GLM main terms (iterated mean); unpenalized logistic "
    "regression on W, A per node (hazard)",
    "treatment_mechanism": "known",
    "cells": [dataclasses.asdict(cell) for cell in CELLS],
    "simultaneous_intervals": False,
}


# ------------------------------------------------------------------------------------ fits


def fit(frame: pd.DataFrame, cell: Cell) -> Any:
    """The cell's ``LTMLE`` fit of one draw."""
    if cell.hazard:
        data = LongitudinalData.from_time_to_event(
            frame,
            time="time",
            event="event",
            treatment="A",
            baseline=["W"],
            grid=list(range(1, laws.NODES + 1)),
            treatment_probabilities={0: "p0", 1: "p1"},
        )
        settings: dict[str, Any] = {
            "survival_model": "hazard",
            "outcome_learner": LogisticRegression(penalty=None, max_iter=1000),
            "censoring_learner": LogisticRegression(penalty=None, max_iter=1000),
        }
    else:
        data = LongitudinalData.from_frame(
            frame,
            outcome=[f"Y{t}" for t in range(1, laws.NODES + 1)],
            treatment=[f"A{t}" for t in range(1, laws.NODES + 1)],
            baseline=["W"],
            treatment_probabilities=frame[[f"p{t}" for t in range(1, laws.NODES + 1)]].to_numpy(),
            id="cluster" if cell.cluster_size is not None else None,
        )
        # A pseudo-outcome carried back from a separated later fit is binary to within 1e-5, and
        # the default learner raises on it; the flag keeps the last iterate there and changes no
        # fit that converges (:class:`~tests.studies.fractional_glm.QuasiBinomialGLM`).
        settings = {
            "outcome_learner": QuasiBinomialGLM(numerical_separation=True),
            "pseudo_learner": QuasiBinomialGLM(numerical_separation=True),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator = LTMLE(
            laws.REGIMENS,
            reference="never",
            n_folds=cell.n_folds,
            simultaneous=False,
            random_state=0,
            g_bounds=(1e-8, 1.0),
            **settings,
        )
        return estimator.fit(data)


def _fit_key(level: str) -> str:
    """``risk_regimen[always @ t=4]`` -> ``always @ t=4``, the key of ``result.fits``."""
    match = re.search(r"\[(.*)\]", level)
    if match is None:  # pragma: no cover - a level name always carries its key
        raise ValueError(level)
    return match.group(1)


def flagged_nodes(result: Any, regimen: str) -> set[int]:
    """The flagged nodes of the regimen's horizon-4 fit."""
    fit_ = result.fits[f"{regimen} @ t={HORIZON}"]
    return {int(cell.time) for cell in sequential_module.boundary_cells(fit_)}


def oracle_node_variance(result: Any, name: str, law: laws.Law, frame: pd.DataFrame) -> float:
    """The flagged nodes' variance of ``name`` at the true row hazards.

    :math:`n^{-2}\\sum_u(\\sum_t\\sum_{i \\in u} |\\lambda_r|\\,k\\,|C_i|\\,\\sigma_i)^2` when
    clustered and :math:`n^{-2}\\sum_u\\sum_t(\\cdot)^2` otherwise, the supplement's form with
    :math:`\\sigma_i^2 = h(1-h)` from the law in place of the bound.
    """
    estimate = result[name]
    if estimate.node is None:
        return 0.0
    loadings = estimate.node.loadings
    weights = np.abs(np.asarray(estimate.node.gradient, dtype=float))
    units = (
        np.arange(len(frame))
        if result.data.cluster is None
        else np.unique(np.asarray(result.data.cluster), return_inverse=True)[1]
    )
    obs = np.asarray(result.data.weights, dtype=float)
    per_node: dict[int, np.ndarray] = {}
    for record in loadings.nodes:
        lam = weights[loadings.levels.index(record.level)]
        key = _fit_key(record.level)
        fit_ = result.fits[key]
        regimen = laws.REGIMENS[key.split(" @ ")[0]]
        coefficient = np.abs(sequential_module.boundary_coefficient(fit_, record.node))
        spread = laws.oracle_spread(law, frame, record.node, regimen)
        for cell in sequential_module.boundary_cells(fit_):
            if cell.time != record.node:
                continue
            rows = np.asarray(cell.rows, dtype=bool) & (obs > 0.0)
            term = lam * cell.terms * coefficient * spread * rows
            per_node[record.node] = per_node.get(record.node, 0.0) + np.bincount(
                units, weights=term, minlength=units.max() + 1
            )
    if not per_node:
        return 0.0
    stacked = np.array(list(per_node.values()))
    total = np.sum(stacked.sum(axis=0) ** 2) if loadings.clustered else np.sum(stacked**2)
    return float(total) / float(loadings.n) ** 2


def _moved(estimate: Any, variance: float) -> Any:
    """The same estimate at another variance and without the supplement."""
    return dataclasses.replace(estimate, node=None, variance=variance)


def draw_arms(frame: pd.DataFrame, cell: Cell, truth: Mapping[str, float]) -> list[dict[str, Any]]:
    """One draw's targets, each with its three arms, off one fit.

    Each entry holds the target, its truth, the point, the flags, and per arm the interval,
    the standard error, ``covered`` and ``rejected``. When the estimate withholds its interval
    every arm reads a withheld miss: ``covered`` and ``rejected`` 0, and the standard error and
    interval NaN. The plug-in arm of an estimate at 0 or 1 has a curve variance of exactly 0,
    which the row check would reject and lose the whole draw with it (the first declared run
    lost 8 keys so); NaN on a withheld row is the one value the row check accepts there.
    """
    law = laws.law_of(cell.variant)
    result = fit(frame, cell)
    flags = {regimen: flagged_nodes(result, regimen) for regimen in laws.REGIMENS}
    out: list[dict[str, Any]] = []
    for target, name in TARGETS.items():
        estimate = result[name]
        shown = bool(estimate.supplies_inference)
        curve = (
            float(estimate.node.influence_variance)
            if estimate.node is not None
            else float(estimate.variance)
        )
        oracle = oracle_node_variance(result, name, law, frame)
        versions = {
            "supplemented": estimate,
            "plugin": _moved(estimate, curve),
            "oracle_node": _moved(estimate, curve + oracle),
        }
        regimen = FIRING_REGIMEN[target]
        other = {node for nodes in flags.values() for node in nodes if node != laws.NODES}
        entry: dict[str, Any] = {
            "target": target,
            "truth": float(truth[target]),
            "psi": float(estimate.psi),
            "withheld": "" if shown else str(estimate.inference),
            "firing": laws.NODES in flags[regimen],
            "multi_flag": laws.NODES in flags[regimen] and bool(other),
            "supplemented_here": estimate.node is not None,
        }
        for arm, version in versions.items():
            low, high = version.plugin_interval
            se = float(version.plugin_std_error)
            if not shown:
                low = high = se = float("nan")
            entry[arm] = {
                "std_error": se,
                "low": float(low),
                "high": float(high),
                "covered": int(shown and low <= truth[target] <= high),
                "rejected": int(shown and not low <= 0.0 <= high),
            }
        out.append(entry)
    return out


# --------------------------------------------------------------------------------- primary


def law_truth() -> dict[str, float]:
    """The primary law's truth under each reported name."""
    truth = laws.law_of("interior").truths()
    return {name: truth[target] for target, name in TARGETS.items()}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One primary draw of law A."""
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return laws.law_of("interior").draw(n, np.random.default_rng(seed)), law_truth()


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """The primary rows of one draw: the shipped estimate and interval of each target."""
    result = fit(frame, CELL_BY_NAME[SCENARIO])
    return regimen_rows(
        STUDY,
        result,
        truth,
        horizon_initials(result, ESTIMANDS[:2], ESTIMANDS[2:]),
        ESTIMANDS,
        scenario,
        replicate,
        n=len(frame),
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _reference_input(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """The replicate's sample and truth rows, which need no fit, and no subject rows."""
    scenario, replicate, n = key
    frame, truth = drawn
    sample = frame.copy()
    sample.insert(0, "nominal_n", n)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truth_rows, []


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    sample, truth_rows, _ = _reference_input(key, drawn)
    return sample, truth_rows, cleverly_rows(frame, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit the primary; the samples are kept as every study keeps them."""
    outcomes = replication.fit_primary(
        [(SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        reference_input=_reference_input,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )


def expected_firing(cell: Cell, target: str, replicates: int = PROPERTY_REPLICATES) -> float:
    """The exact expected count of firing draws of one target at ``replicates`` draws.

    Node 4 of the target's firing regimen flagged, from the law
    (:meth:`~tests.studies.boundary_node_laws.Law.flag_probability`, given the cluster effect
    on the clustered cell).
    """
    law = laws.law_of(cell.variant)
    regimen = laws.REGIMENS[FIRING_REGIMEN[target]]
    return replicates * law.flag_probability(laws.NODES, regimen, cell.n, cell.cluster_size)


#: The smallest expected firing count at which a cell's ``firing_floor`` is gated (plan 5.5).
FIRING_GATE: Final[int] = 600


def firing_gated(cell: Cell, target: str) -> bool:
    """Whether ``firing_floor`` gates this cell and target: 600 or more expected firing draws."""
    return expected_firing(cell, target) >= FIRING_GATE


__all__: Sequence[str] = (
    "CELLS",
    "STUDY",
    "TARGETS",
    "draw_and_fit",
    "draw_arms",
    "expected_firing",
    "fit",
)
