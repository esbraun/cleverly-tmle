"""Property families of ``mtp-boundaries``.

Every cell reads one contrast, ``ate_regimen[<plan> vs natural]``, at one fold on the density
route with the oracle hazard (:mod:`tests.studies.mtp_boundaries_common`).

==========================  =====================================================================
family                      cells, replications and size
==========================  =====================================================================
``interval_calibration``    seven positives with the correctly specified outcome model, with the
                            two derived controls of each: ``LTMLE`` under ``p1`` to ``p4`` on
                            law B and ``p5`` on law B', and ``SDR`` under ``p4`` and ``p5``
                            (``p4_sdr``, ``p5_sdr``).  n = 2,000, except ``p4`` and ``p4_sdr``
                            at n = 4,000 (:data:`CALIBRATION_SIZES`).  The gating positives
                            have :data:`POOLED_REPLICATES` replications, set from the pilot by
                            the arm-pooled rule; ``p4_sdr`` reports
                            (:data:`REPORTING_LABELS`)
``jacobian_necessity``      ``p3``, dose-blind outcome: the declared fit against the same draw
                            with every piece's inverse slope replaced by 1; 1,000 at n = 2,000
``branch_necessity``        ``p4``, wrong outcome: Equation (3) summed over the first branch
                            only; 1,000 at n = 2,000
``atom_necessity``          ``p5``, wrong outcome: the atom's ratio without the mass of the
                            interval sent to it (ratio 1 at the atom); 1,000 at n = 2,000
``ratio_necessity``         ``p2``, wrong outcome: the declared ratio against a ratio of one
                            (the classifier route with the prior classifier); 1,000 at n = 2,000.
                            With ratio one on both arms the ``Y`` terms of the curve cancel, so
                            the control's reported SE is near zero; its ``covered`` and
                            ``rejected`` columns are no coverage reading (design review n1)
==========================  =====================================================================

**The wrong outcome** (:class:`~tests.studies.mtp_boundaries_common.WrongOutcome`) omits ``L2``
and ``A2^2`` at node 2, and the first node reads least squares on ``[W1, W2, A1]``.  The
targeting step then carries the correction, so each defect in the ratio moves the estimate.
The Jacobian pair reads the dose-blind model instead
(:class:`~tests.studies.mtp_boundaries_common.DoseBlindOutcome`, without ``A2`` and ``A2^2``).
Its defect is a factor monotone in the dose, and targeting absorbs the factor's constant part.
On 100 paired draws at n = 2,000 (design review B1), under the wrong outcome, whose error is
U-shaped in the dose, the control's bias was 0.07 of its own SD and the displacement 0.05 of
the positive's.  Under the dose-blind outcome they were 1.69 and 0.71 (MC SE 0.07), with the
positive at -0.01.

**The defects patch** :mod:`cleverly.interventions.policy` **and**
:mod:`cleverly.learners.density` **module-wide.**  A patch is safe only because each property
fit runs alone in its process, serially: :func:`~cleverly.utils.parallel.map_parallel` uses
process workers, and each fit sets ``n_jobs=1``.  :func:`ratio_defect` refuses to start while
another patch is active.  ``_check_monotone`` checks each declared inverse against its map,
so a control cannot be written as a mis-declared policy; it is a patch of the ratio instead.

**Sizes, declared before any pilot.**

1. The calibration pilot (:func:`calibration_pilot`) fits :data:`PILOT_DRAWS` draws per
   calibration positive on the ``diagnostic`` stream, disjoint from the study's.  Each arm's
   pilot SE ratio is screened (:func:`~tests.studies.mtp_study_tools.screen_half_width`,
   0.046 at 3,000 draws).  ``R`` is then the smallest multiple of 400 with family-wise pass
   probability at least 0.95 and per-cell at least 0.90 under the arm-pooled rule
   (:func:`~tests.studies.mtp_study_tools.replicates_from_pilot`).  Expected: 6,000 at a pilot
   mean of 1.000, 6,400 at 1.005, 8,000 at 1.010.
2. If the pilot mean is off 1 by 0.015 or more, or no ``R`` up to 40,000 meets the rule, the
   family is a defect suspect and is diagnosed before declaration.  A defect is fixed.  A
   diagnosed finite-sample SE bias at n = 2,000 moves the cell to n = 4,000, re-piloted, and
   if that is still infeasible the cell is declared ``reporting`` with an owner and the pilot
   published.  No ``R`` is bought to pass a biased cell, and no margin moves.
3. Each necessity pair is read from :func:`necessity_pilot`: :data:`NECESSITY_PILOT_DRAWS`
   paired draws at n = 2,000 on the ``diagnostic`` stream, each arm standardized by its own
   empirical SD, the scale the verdicts read.  Each pair must give power at least 0.90 for
   ``bias_discriminated`` (control) and ``bias_equivalent`` (positive) at 1,000 replications
   (:func:`~tests.studies.mtp_study_tools.pair_reading`) and a displacement of at least
   :data:`NECESSITY_DISPLACEMENT`.  A pair that cannot is redesigned before declaration.

The study's property loop runs through
:func:`~tests.studies.evidence.replication.fit_properties` under :data:`FAILURE_BOUND`.  It
draws each sample outside the fit and checkpoints each key
(:func:`~tests.studies.mtp_study_tools.property_checkpoint`), so a run stopped by a container
restart resumes.  The pilots keep :func:`~tests.studies.mtp_study_tools.run_in_parts`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier

from cleverly.interventions import policy as policy_module
from cleverly.learners import density as density_module
from tests.parallel import STUDY_JOBS, memory_capped_workers
from tests.studies import canonical_mtp_boundaries as study
from tests.studies import mtp_boundaries_common as common
from tests.studies import mtp_study_tools as tools
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    Necessity,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.oracle_density_bins import fit_bytes as one_fit_bytes

STUDY = study.STUDY
CALIBRATION_N = 2_000
NECESSITY_REPLICATES = 1_000
NECESSITY_N = 2_000
SHRUNKEN_SE_FACTOR = 0.70
#: The declared minimum displacement of every necessity family.
NECESSITY_DISPLACEMENT = 0.10
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: Each calibration label: ``(plan, estimator)``.
CELLS: dict[str, tuple[str, str]] = {
    "p1": ("p1", "ltmle"),
    "p2": ("p2", "ltmle"),
    "p3": ("p3", "ltmle"),
    "p4": ("p4", "ltmle"),
    "p5": ("p5", "ltmle"),
    "p4_sdr": ("p4", "sdr"),
    "p5_sdr": ("p5", "sdr"),
}
#: The pilot's draws per calibration positive.
PILOT_DRAWS = 3_000
#: Each calibration positive's sample size.  ``p4`` (LTMLE) moved to n = 4,000 under sizing
#: rule 2: its n = 2,000 pilot read 0.942, outside the screen, and the diagnosis found
#: right-skewed standard errors from rare large weights and no defect
#: (``X40-pilot-diagnosis.md`` E6).  At n = 4,000 it reads 0.971, inside the screen.
#: ``p4_sdr`` moved with it (diagnosis E4) and stays outside, so it reports.
CALIBRATION_SIZES: dict[str, int] = dict.fromkeys(CELLS, CALIBRATION_N) | {
    "p4": 4_000,
    "p4_sdr": 4_000,
}
#: The calibration positives declared ``reporting`` with an owner before the run (sizing rule
#: 2).  ``p4_sdr`` read 0.919 at n = 2,000 and 0.926 at n = 4,000 with the projected
#: pseudo learner, both outside the screen.  Owner: the SDR caveat in
#: ``docs/technical-reference/longitudinal-tmle.md``, which states both readings and the
#: unprojected 0.573.  Such an arm leaves the pooled rule and the gate, and it runs at
#: :data:`PILOT_DRAWS` replications.
REPORTING_LABELS: frozenset[str] = frozenset({"p4_sdr"})
#: Each arm's pilot SE ratio at its declared size, from the published pilots
#: (``reviews/x40/pilots`` in the plans repository): ``p1`` to ``p5`` and ``p5_sdr`` from
#: ``s1-calibration-pilot.json`` (``p5_sdr`` from the projected re-pilot), ``p4`` from the
#: n = 4,000 re-pilot.  The reporting arm is not pooled.
PILOT_SE_RATIOS: dict[str, float] = {
    "p1": 1.0182,
    "p2": 0.9797,
    "p3": 0.9938,
    "p4": 0.9707,
    "p5": 0.9957,
    "p5_sdr": 0.9988,
}
#: The pooled rule's ``R`` from :data:`PILOT_SE_RATIOS`
#: (:func:`~tests.studies.mtp_study_tools.replicates_from_pilot`: mean 0.9928, tau 0.0196,
#: per-cell 0.991, family-wise 0.950).
POOLED_REPLICATES = 26_400
#: The replications of each calibration positive.
CALIBRATION_REPLICATES: dict[str, int] = {
    label: PILOT_DRAWS if label in REPORTING_LABELS else POOLED_REPLICATES for label in CELLS
}
#: Each necessity family: ``(plan, (positive arm, control arm), defect)``.
NECESSITY: dict[str, tuple[str, tuple[str, str], str]] = {
    family: (label, (positive, control), family.removesuffix("_necessity"))
    for family, (label, positive, control) in study.NECESSITY_CELLS.items()
}
NECESSITY_PILOT_DRAWS = tools.PAIR_PILOT_DRAWS
#: Each necessity family's wrong outcome model.  The Jacobian pair reads the dose-blind model,
#: the only one whose error varies with the dose the defect reweights (design review B1).
NECESSITY_OUTCOME: dict[str, str] = {
    family: "dose_blind" if family == "jacobian_necessity" else "wrong" for family in NECESSITY
}
#: The cells that gate the merge though the record publishes under ``reporting`` (design
#: review M7): every calibration positive with its two derived controls, and both arms of every
#: necessity pair (the control's row carries the displacement).  Every Python truth row of the
#: four primary scenarios gates too (:data:`GATING_TRUTH`).  Only the margin-read atom pairing
#: rows and the five-fold rows report.  ``tests/unit/test_mtp_boundaries_design.py`` reads the
#: committed tables through :func:`~tests.studies.mtp_study_tools.gating_findings`.
GATING_CELLS: tuple[str, ...] = tuple(
    f"{family}/{cell}"
    for family, cells in study.PROPERTY_CELLS.items()
    for cell in cells
    if not (family == "interval_calibration" and cell.split("__")[0] in REPORTING_LABELS)
)
GATING_TRUTH: tuple[str, ...] = (study.STUDY.implementation,)
#: The efficiency bound of each calibration label, which sizes the noise control only
#: (:func:`~tests.studies.mtp_boundaries_common.contrast_sd`, pinned).
EFFICIENCY_SD: dict[str, float] = {
    label: common.BOUNDS.get(plan, 1.0) for label, (plan, _) in CELLS.items()
}


def estimand(plan: str) -> str:
    return f"ate_regimen[{plan} vs {study.REFERENCE}]"


# ---------------------------------------------------------------------------- the controls

_PIECE_RATIO = policy_module._PieceBranch.ratio
_MASS_BETWEEN = density_module.MixedDensity.mass_between


def _unit_slope(piece: Any) -> Any:
    if piece.map is None or piece.to_atom is not None:
        return piece
    return policy_module.Piece(
        piece.lower,
        piece.upper,
        piece.map,
        piece.inverse,
        lambda b, h: np.ones_like(np.asarray(b, dtype=float)),
        piece.closed,
    )


@contextmanager
def ratio_defect(defect: str | None) -> Iterator[None]:
    """The density-route ratio with one deliberate defect, or unchanged for ``None``.

    * ``"jacobian"`` reads Equation (3) with every inverse slope replaced by 1.
    * ``"branch"`` reads Equation (3) over the first moving piece only: the rows the second
      branch sends to a value lose its preimage term.
    * ``"atom"`` drops the mass of every interval sent to a declared atom, so the ratio at
      the atom reads only the atoms that map there (1 under ``p5``).
    * ``"ratio"`` patches nothing: its control is a fit on the classifier route with the
      prior classifier, whose ratio is 1 (:func:`necessity_fit`).
    """
    if defect in (None, "ratio"):
        yield
        return
    if (
        policy_module._PieceBranch.ratio is not _PIECE_RATIO
        or density_module.MixedDensity.mass_between is not _MASS_BETWEEN
    ):
        raise RuntimeError("a ratio defect is already active")
    if defect == "jacobian":

        def unit(self: Any, values: Any, frame: Any, density: Any) -> Any:
            pieces = tuple(_unit_slope(piece) for piece in self.pieces)
            return _PIECE_RATIO(
                policy_module._PieceBranch(self.name, pieces), values, frame, density
            )

        policy_module._PieceBranch.ratio = unit  # type: ignore[method-assign]
    elif defect == "branch":

        def first(self: Any, values: Any, frame: Any, density: Any) -> Any:
            moving = [k for k, piece in enumerate(self.pieces) if piece.map is not None]
            pieces = tuple(p for k, p in enumerate(self.pieces) if k not in moving[1:])
            return _PIECE_RATIO(
                policy_module._PieceBranch(self.name, pieces), values, frame, density
            )

        policy_module._PieceBranch.ratio = first  # type: ignore[method-assign]
    elif defect == "atom":

        def no_mass(self: Any, lower: Any, upper: Any) -> Any:
            return np.zeros(self.n)

        density_module.MixedDensity.mass_between = no_mass  # type: ignore[method-assign]
    else:
        raise ValueError(f"unknown defect {defect!r}")
    try:
        yield
    finally:
        policy_module._PieceBranch.ratio = _PIECE_RATIO  # type: ignore[method-assign]
        density_module.MixedDensity.mass_between = _MASS_BETWEEN  # type: ignore[method-assign]


# ---------------------------------------------------------------------------- fits


def calibration_fit(label: str, frame: pd.DataFrame) -> Any:
    plan, estimator = CELLS[label]
    law = common.LAW_OF[plan]
    # SDR regresses an unbounded transform; its earlier node reads the [0, 1] projection of
    # the same basis (X40-pilot-diagnosis.md).  LTMLE's targets already lie in [0, 1].
    pseudo = common.UnitRangePseudo() if estimator == "sdr" else None
    return study.fit(
        frame,
        law,
        (study.REFERENCE, plan),
        estimator=estimator,
        outcome="correct",
        pseudo_learner=pseudo,
    )


def necessity_fit(family: str, frame: pd.DataFrame, *, control: bool) -> Any:
    """One arm of a necessity pair: the declared fit, or the same draw with the defect."""
    plan, _, defect = NECESSITY[family]
    law = common.LAW_OF[plan]
    labels = (study.REFERENCE, plan)
    outcome: Any = NECESSITY_OUTCOME[family]
    if control and defect == "ratio":
        return study.fit(
            frame,
            law,
            labels,
            outcome=outcome,
            ratio="classifier",
            treatment_learner=DummyClassifier(strategy="prior"),
        )
    with ratio_defect(defect if control else None):
        return study.fit(frame, law, labels, outcome=outcome)


# ---------------------------------------------------------------------------- the grid


class DeclaredLaw:
    """The law a declared cell reads, with the exact truth of its contrast.

    Parameters
    ----------
    plan : str
        The plan contrasted with ``natural``.
    """

    def __init__(self, plan: str) -> None:
        self.plan = plan
        self.name = common.LAW_OF[plan]

    def truth(self) -> dict[str, float]:
        return {estimand(self.plan): common.contrast(self.plan)}


#: Every fit set: ``(family, stream label, n, replicates)``.
FIT_SETS: tuple[tuple[str, str, int, int], ...] = (
    *(
        ("interval_calibration", label, CALIBRATION_SIZES[label], CALIBRATION_REPLICATES[label])
        for label in CELLS
    ),
    *((family, "paired", NECESSITY_N, NECESSITY_REPLICATES) for family in NECESSITY),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def _pilot_seed(family: str, label: str, replicate: int, n: int = CALIBRATION_N) -> int:
    # A re-pilot at another size (plan rule 2) draws fresh seeds, disjoint from the first.
    if n == CALIBRATION_N:
        return stream_seed(STUDY, "diagnostic", "pilot", family, label, replicate)
    return stream_seed(STUDY, "diagnostic", "pilot", family, label, n, replicate)


def _plan_of(family: str, label: str) -> str:
    return CELLS[label][0] if family == "interval_calibration" else NECESSITY[family][0]


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(family: str, cell: str, role: str, plan: str, n: int, reps: int, stream: str) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(plan),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(family, stream, 0),
                role=role,
                estimand=estimand(plan),
            )
        )

    for label, (plan, _) in CELLS.items():
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                plan,
                CALIBRATION_SIZES[label],
                CALIBRATION_REPLICATES[label],
                label,
            )
    for family, (plan, arms, _) in NECESSITY.items():
        for arm, role in zip(arms, ("positive", "control"), strict=True):
            add(family, f"{plan}__{arm}", role, plan, NECESSITY_N, NECESSITY_REPLICATES, "paired")
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _draw_frame(family: str, label: str, n: int, seed: int) -> pd.DataFrame:
    """The sample of a fit set at ``seed``, from the law of its plan."""
    return common.SAMPLERS[common.LAW_OF[_plan_of(family, label)]](n, seed)


def _fit_rows(
    family: str, label: str, replicate: int, n: int, requested: int, frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """The rows of one replication of one fit set, fitted on its drawn ``frame``."""
    plan = _plan_of(family, label)
    name = estimand(plan)
    value = common.contrast(plan)
    shared: dict[str, Any] = {"replicate": replicate, "n": n, "requested": requested}
    if family == "interval_calibration":
        result = calibration_fit(label, frame)
        return [
            replicate_row(
                property_name=family,
                cell=f"{label}__correctly_specified",
                role="positive",
                truth=value,
                estimate=result[name],
                alpha=STUDY.margins.alpha,
                **shared,
            )
        ]
    _, (positive_arm, control_arm), _ = NECESSITY[family]
    declared = necessity_fit(family, frame, control=False)
    defective = necessity_fit(family, frame, control=True)
    return [
        replicate_row(
            property_name=family,
            cell=f"{plan}__{positive_arm}",
            role="positive",
            truth=value,
            estimate=declared[name],
            alpha=STUDY.margins.alpha,
            **shared,
        ),
        control_row(
            property_name=family,
            cell=f"{plan}__{control_arm}",
            truth=value,
            estimate=float(defective[name].psi),
            standard_error=float(defective[name].std_error),
            critical=CRITICAL,
            **shared,
        ),
    ]


def fit_set_rows(
    family: str, label: str, replicate: int, n: int, requested: int, seed: int | None = None
) -> list[dict[str, Any]]:
    """The rows of one replication of one fit set, from its declared stream or ``seed``."""
    frame = _draw_frame(family, label, n, _seed(family, label, replicate) if seed is None else seed)
    return _fit_rows(family, label, replicate, n, requested, frame)


#: A property key: ``(family, stream label, replicate, n, requested)``.
_Key = tuple[str, str, int, int, int]

#: Each calibration positive's derived controls, which the keys name and :func:`calibration_controls`
#: writes after the loop.
DERIVED_CELLS: dict[str, tuple[str, ...]] = calibration_derived(tuple(CELLS))


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    family, label, replicate, n, _ = key
    return _draw_frame(family, label, n, _seed(family, label, replicate))


def _fit(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _fit_rows(*key, frame)


def _cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the key's rows and derived controls fill."""
    family, label, *_ = key
    if family == "interval_calibration":
        positive = f"{family}/{label}__correctly_specified"
        return [positive, *DERIVED_CELLS[positive]]
    plan, arms, _ = NECESSITY[family]
    return [f"{family}/{plan}__{arm}" for arm in arms]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def _seed_of(key: _Key) -> int:
    family, label, replicate, *_ = key
    return _seed(family, label, replicate)


def fit_bytes(n: int) -> int:
    """A generous peak of one property worker at ``n`` rows: two oracle fits, at least 1 GiB.

    The one-replicate smoke measured 0.61 GiB of peak resident set for one process that ran
    the four primary fits at n = 2,000, the interpreter and the quadrature included.
    """
    return max(2 * one_fit_bytes(n), 2**30)


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6).  Up to ``floor(0.01 R)`` lost
#: replicates are judged by the worst case of each clause; one more is ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, parts: Path | None = None
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check.

    The keys of every fit set run in one :func:`~tests.studies.evidence.replication.
    fit_properties` loop.  A finished key is kept under
    :func:`~tests.studies.mtp_study_tools.property_checkpoint` (the driver's checkpoint
    scratch, else ``parts``), so a stopped run resumes.  A ``budget`` run keeps nothing,
    because a key's row carries ``requested``.
    """
    jobs = memory_capped_workers(n_jobs, fit_bytes(max(n for _, _, n, _ in FIT_SETS)))
    keys: list[_Key] = []
    for family, label, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        keys += [(family, label, r, n, requested) for r in range(requested)]
    outcomes = fit_properties(
        keys,
        _fit,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
        seed_of=_seed_of,
        checkpoint=tools.property_checkpoint(parts) if budget is None else None,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    # The noise control's scale reads each positive's own n, so each size is one call.
    controls = [
        calibration_controls(
            rows,
            STUDY,
            labels=tuple(label for label in CELLS if CALIBRATION_SIZES[label] == size),
            efficiency_bounds=EFFICIENCY_SD,
            calibration_n=size,
            shrunken_se_factor=SHRUNKEN_SE_FACTOR,
            critical=CRITICAL,
        )
        for size in sorted(set(CALIBRATION_SIZES.values()))
    ]
    rows = pd.concat([rows, *controls], ignore_index=True)
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


#: Each necessity family's declaration, read by the verdict and by its rule.
NECESSITIES: tuple[Necessity, ...] = tuple(
    Necessity(
        family=family,
        labels=(plan,),
        arms=arms,
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    )
    for family, (plan, arms, _) in NECESSITY.items()
)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("necessity_displacement",), rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    for necessity in NECESSITIES:
        necessity_verdicts(summary, rows, necessity)
    return finish(summary, rates)


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes no
#: efficiency band, because the study's noise control answers to the SE-ratio rule.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(*NECESSITIES),
)


# ---------------------------------------------------------------------------- pilots


def _pilot_rows(
    family: str, label: str, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    rows = fit_set_rows(
        family, label, replicate, n, requested, seed=_pilot_seed(family, label, replicate, n)
    )
    seconds = time.perf_counter() - started
    for row in rows:
        row["seconds"] = seconds / len(rows)
    return rows


def calibration_pilot(
    parts: Path,
    *,
    draws: int = PILOT_DRAWS,
    n_jobs: int = 1,
    labels: tuple[str, ...] = (),
    n: int = CALIBRATION_N,
) -> dict[str, Any]:
    """The calibration pilot: ``draws`` per positive on the ``diagnostic`` stream, in parts.

    Returns each arm's pilot SE ratio, standardized bias and mean seconds a fit, and the
    declared rule's reading (:func:`~tests.studies.mtp_study_tools.replicates_from_pilot`).
    """
    chosen = labels or tuple(CELLS)
    out: dict[str, Any] = {"arms": {}}
    for label in chosen:
        keys = [("interval_calibration", label, r, n, draws) for r in range(draws)]
        part = f"pilot__{label}" if n == CALIBRATION_N else f"pilot__{label}__n{n}"
        rows = tools.run_in_parts(_pilot_rows, keys, label=part, root=parts, n_jobs=n_jobs)
        error = rows["estimate"] - rows["truth"]
        spread = float(rows["estimate"].std(ddof=1))
        out["arms"][label] = {
            "draws": len(rows),
            "n": n,
            "se_ratio": float(rows["std_error"].mean() / spread),
            "standardized_bias": float(error.mean() / spread),
            "coverage": float(rows["covered"].mean()),
            "seconds": float(rows["seconds"].mean()),
        }
    if len(chosen) == len(CELLS) and n == CALIBRATION_N:
        ratios = [out["arms"][label]["se_ratio"] for label in CELLS]
        out["rule"] = tools.replicates_from_pilot(ratios, draws)
    return out


def necessity_pilot(
    parts: Path, *, draws: int = NECESSITY_PILOT_DRAWS, n: int = NECESSITY_N, n_jobs: int = 1
) -> dict[str, Any]:
    """Each necessity pair on :data:`NECESSITY_PILOT_DRAWS` paired draws at n = 2,000.

    Read on each arm's own SD (:func:`~tests.studies.mtp_study_tools.pair_reading`): the
    control's power for ``bias_discriminated``, the positive's for ``bias_equivalent``, and
    the paired displacement.  A pair below power 0.90 on either arm is redesigned before
    declaration.
    """
    out: dict[str, Any] = {}
    for family, (plan, (positive, control), _) in NECESSITY.items():
        keys = [(family, "paired", r, n, draws) for r in range(draws)]
        rows = tools.run_in_parts(
            _pilot_rows, keys, label=f"necessity__{family}__{n}", root=parts, n_jobs=n_jobs
        )
        declared = rows.loc[rows["cell"] == f"{plan}__{positive}"].set_index("replicate")
        broken = rows.loc[rows["cell"] == f"{plan}__{control}"].set_index("replicate")
        broken = broken.loc[declared.index]
        out[family] = {
            **tools.pair_reading(
                declared["estimate"],
                broken["estimate"],
                declared["truth"],
                NECESSITY_REPLICATES,
                NECESSITY_DISPLACEMENT,
            ),
            "seconds": float(rows["seconds"].sum() / len(declared)),
        }
    return out


#: Per-fit seconds and peak memory from the one-replicate smoke (warm process, one core,
#: n = 2,000, 2026-10-15), and the CPU-hours they imply.  The calibration pilot replaces the
#: calibration figure with its own mean seconds a fit.
_CALIBRATION_SECONDS = {
    "p1": 0.83,
    "p2": 0.79,
    "p3": 0.80,
    "p4": 3.72,  # n = 4,000, the re-pilot's mean
    "p5": 0.80,
    "p4_sdr": 2.56,  # n = 4,000, the re-pilot's mean
    "p5_sdr": 0.77,
}
_PAIR_SECONDS = {
    "jacobian_necessity": 1.58,
    "branch_necessity": 1.60,
    "atom_necessity": 1.52,
    "ratio_necessity": 0.87,
}
_PRIMARY_SECONDS = 3.41  # the four scenarios of one replicate
_HOUR = 3600.0


def _calibration_hours(replicates: int | dict[str, int]) -> float:
    counts = replicates if isinstance(replicates, dict) else dict.fromkeys(CELLS, replicates)
    return sum(_CALIBRATION_SECONDS[label] * counts[label] for label in CELLS) / _HOUR


BUDGET: dict[str, Any] = {
    "seconds": {
        "primary_replicate": _PRIMARY_SECONDS,
        "calibration_fit": _CALIBRATION_SECONDS,
        "necessity_pair": _PAIR_SECONDS,
    },
    "peak_gib": 0.57,
    "cpu_hours": {
        "calibration_pilot": _calibration_hours(PILOT_DRAWS),
        "necessity_pilot": NECESSITY_PILOT_DRAWS * sum(_PAIR_SECONDS.values()) / _HOUR,
        "calibration": _calibration_hours(CALIBRATION_REPLICATES),
        "necessity": NECESSITY_REPLICATES * sum(_PAIR_SECONDS.values()) / _HOUR,
        "primary": study.PRIMARY_REPLICATES * _PRIMARY_SECONDS / _HOUR,
    },
}
BUDGET["cpu_hours"]["python_total"] = (
    BUDGET["cpu_hours"]["calibration_pilot"]
    + BUDGET["cpu_hours"]["necessity_pilot"]
    + BUDGET["cpu_hours"]["calibration"]
    + BUDGET["cpu_hours"]["necessity"]
    + BUDGET["cpu_hours"]["primary"]
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pilot", choices=("calibration", "necessity"))
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--labels", nargs="*", default=())
    parser.add_argument("--n", type=int, default=CALIBRATION_N)
    arguments = parser.parse_args(argv)
    if arguments.pilot == "calibration":
        result = calibration_pilot(
            arguments.parts,
            draws=arguments.draws or PILOT_DRAWS,
            n_jobs=arguments.jobs,
            labels=tuple(arguments.labels),
            n=arguments.n,
        )
    else:
        result = necessity_pilot(
            arguments.parts, draws=arguments.draws or NECESSITY_PILOT_DRAWS, n_jobs=arguments.jobs
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    (arguments.parts / f"{arguments.pilot}-pilot.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
