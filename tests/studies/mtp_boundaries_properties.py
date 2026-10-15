"""Property families of ``mtp-boundaries``.

Every cell reads one contrast, ``ate_regimen[<plan> vs natural]``, at one fold on the density
route with the oracle hazard (:mod:`tests.studies.mtp_boundaries_common`).

==========================  =====================================================================
family                      cells, replications and size
==========================  =====================================================================
``interval_calibration``    seven positives with the correctly specified outcome model, at
                            n = 2,000, with the two derived controls of each: ``LTMLE`` under
                            ``p1`` to ``p4`` on law B and ``p5`` on law B', and ``SDR`` under
                            ``p4`` and ``p5`` (``p4_sdr``, ``p5_sdr``).  Each positive has its
                            own replications, :data:`CALIBRATION_REPLICATES`, set from the
                            calibration pilot by the arm-pooled rule
``jacobian_necessity``      ``p3``, wrong outcome: the declared fit against the same draw with
                            every piece's inverse slope replaced by 1; 1,000 at n = 2,000
``branch_necessity``        ``p4``, wrong outcome: Equation (3) summed over the first branch
                            only; 1,000 at n = 2,000
``atom_necessity``          ``p5``, wrong outcome: the atom's ratio without the mass of the
                            interval sent to it (ratio 1 at the atom); 1,000 at n = 2,000
``ratio_necessity``         ``p2``, wrong outcome: the declared ratio against a ratio of one
                            (the classifier route with the prior classifier); 1,000 at n = 2,000
==========================  =====================================================================

**The wrong outcome** (:class:`~tests.studies.mtp_boundaries_common.WrongOutcome`) omits ``L2``
and ``A2^2`` at node 2, and the first node reads least squares on ``[W1, W2, A1]``.  The
targeting step then carries the correction, so each defect in the ratio moves the estimate.

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
3. Each necessity control's standardized bias at n = 2,000 comes from
   :func:`necessity_pilot`: :data:`NECESSITY_PILOT_DRAWS` paired draws at n = 8,000 on the
   ``diagnostic`` stream, with its Monte Carlo SE.  Each pair must give power at least 0.90
   for ``bias_discriminated`` at 1,000 replications
   (:func:`~tests.studies.mtp_study_tools.necessity_power`) and a displacement of at least
   :data:`NECESSITY_DISPLACEMENT`.  A pair that cannot is redesigned before declaration.

The failure rule is ``vector-mtp``'s: the property phase records no failure, so a raise stops
the run.  Every fit set is fitted in parts (:func:`~tests.studies.mtp_study_tools.
run_in_parts`), so a run stopped by a container restart resumes.
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
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
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
#: The replications of each calibration positive.  Provisional: the expected ``R`` at a pilot
#: mean of 1.005.  The declaration commit replaces it with the ``R`` that
#: :func:`~tests.studies.mtp_study_tools.replicates_from_pilot` reads off the pilot.
CALIBRATION_REPLICATES: dict[str, int] = dict.fromkeys(CELLS, 6_400)
#: Each necessity family: ``(plan, (positive arm, control arm), defect)``.
NECESSITY: dict[str, tuple[str, tuple[str, str], str]] = {
    family: (label, (positive, control), family.removesuffix("_necessity"))
    for family, (label, positive, control) in study.NECESSITY_CELLS.items()
}
NECESSITY_PILOT_DRAWS = 8
NECESSITY_PILOT_N = 8_000
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
    return study.fit(frame, law, (study.REFERENCE, plan), estimator=estimator, outcome="correct")


def necessity_fit(family: str, frame: pd.DataFrame, *, control: bool) -> Any:
    """One arm of a necessity pair: the declared fit, or the same draw with the defect."""
    plan, _, defect = NECESSITY[family]
    law = common.LAW_OF[plan]
    labels = (study.REFERENCE, plan)
    if control and defect == "ratio":
        return study.fit(
            frame,
            law,
            labels,
            outcome="wrong",
            ratio="classifier",
            treatment_learner=DummyClassifier(strategy="prior"),
        )
    with ratio_defect(defect if control else None):
        return study.fit(frame, law, labels, outcome="wrong")


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
        ("interval_calibration", label, CALIBRATION_N, CALIBRATION_REPLICATES[label])
        for label in CELLS
    ),
    *((family, "paired", NECESSITY_N, NECESSITY_REPLICATES) for family in NECESSITY),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def _pilot_seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "diagnostic", "pilot", family, label, replicate)


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
                CALIBRATION_N,
                CALIBRATION_REPLICATES[label],
                label,
            )
    for family, (plan, arms, _) in NECESSITY.items():
        for arm, role in zip(arms, ("positive", "control"), strict=True):
            add(family, f"{plan}__{arm}", role, plan, NECESSITY_N, NECESSITY_REPLICATES, "paired")
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def fit_set_rows(
    family: str, label: str, replicate: int, n: int, requested: int, seed: int | None = None
) -> list[dict[str, Any]]:
    """The rows of one replication of one fit set, from its declared stream or ``seed``."""
    plan = _plan_of(family, label)
    law = common.LAW_OF[plan]
    frame = common.SAMPLERS[law](n, _seed(family, label, replicate) if seed is None else seed)
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


def fit_bytes(n: int) -> int:
    """A generous peak of one property worker at ``n`` rows: two oracle fits, at least 1 GiB.

    The one-replicate smoke measured 0.61 GiB of peak resident set for one process that ran
    the four primary fits at n = 2,000, the interpreter and the quadrature included.
    """
    return max(2 * one_fit_bytes(n), 2**30)


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, parts: Path | None = None
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check.

    Each fit set is fitted in parts under :func:`~tests.studies.mtp_study_tools.parts_root`
    (the driver's checkpoint scratch, else ``parts``), so a stopped run resumes.
    """
    jobs = memory_capped_workers(n_jobs, fit_bytes(max(n for _, _, n, _ in FIT_SETS)))
    root = tools.parts_root(parts)
    frames = []
    for family, label, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        keys = [(family, label, r, n, requested) for r in range(requested)]
        frames.append(
            tools.run_in_parts(
                fit_set_rows,
                keys,
                label=f"{family}__{label}__{requested}",
                root=root,
                n_jobs=jobs,
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=tuple(CELLS),
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=("necessity_displacement",), rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    for family, (plan, arms, _) in NECESSITY.items():
        necessity_verdicts(
            summary,
            rows,
            family=family,
            labels=(plan,),
            arms=arms,
            column="necessity_displacement",
            threshold=NECESSITY_DISPLACEMENT,
        )
    return finish(summary, rates)


# ---------------------------------------------------------------------------- pilots


def _pilot_rows(
    family: str, label: str, replicate: int, n: int, requested: int
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    rows = fit_set_rows(
        family, label, replicate, n, requested, seed=_pilot_seed(family, label, replicate)
    )
    seconds = time.perf_counter() - started
    for row in rows:
        row["seconds"] = seconds / len(rows)
    return rows


def calibration_pilot(
    parts: Path, *, draws: int = PILOT_DRAWS, n_jobs: int = 1, labels: tuple[str, ...] = ()
) -> dict[str, Any]:
    """The calibration pilot: ``draws`` per positive on the ``diagnostic`` stream, in parts.

    Returns each arm's pilot SE ratio, standardized bias and mean seconds a fit, and the
    declared rule's reading (:func:`~tests.studies.mtp_study_tools.replicates_from_pilot`).
    """
    chosen = labels or tuple(CELLS)
    out: dict[str, Any] = {"arms": {}}
    for label in chosen:
        keys = [("interval_calibration", label, r, CALIBRATION_N, draws) for r in range(draws)]
        rows = tools.run_in_parts(
            _pilot_rows, keys, label=f"pilot__{label}", root=parts, n_jobs=n_jobs
        )
        error = rows["estimate"] - rows["truth"]
        spread = float(rows["estimate"].std(ddof=1))
        out["arms"][label] = {
            "draws": len(rows),
            "se_ratio": float(rows["std_error"].mean() / spread),
            "standardized_bias": float(error.mean() / spread),
            "coverage": float(rows["covered"].mean()),
            "seconds": float(rows["seconds"].mean()),
        }
    if len(chosen) == len(CELLS):
        ratios = [out["arms"][label]["se_ratio"] for label in CELLS]
        out["rule"] = tools.replicates_from_pilot(ratios, draws)
    return out


def necessity_pilot(
    parts: Path, *, draws: int = NECESSITY_PILOT_DRAWS, n: int = NECESSITY_PILOT_N, n_jobs: int = 1
) -> dict[str, Any]:
    """Each necessity pair at ``n`` on the ``diagnostic`` stream: bias, power, displacement.

    The control's bias is read at ``n``, where the finite-sample terms are smaller, and
    standardized by the declared arm's spread at n = 2,000, which is its reported SE at ``n``
    times ``sqrt(n / 2000)``.
    """
    out: dict[str, Any] = {}
    for family, (plan, (positive, control), _) in NECESSITY.items():
        keys = [(family, "paired", r, n, draws) for r in range(draws)]
        rows = tools.run_in_parts(
            _pilot_rows, keys, label=f"necessity__{family}__{n}", root=parts, n_jobs=n_jobs
        )
        declared = rows.loc[rows["cell"] == f"{plan}__{positive}"].set_index("replicate")
        broken = rows.loc[rows["cell"] == f"{plan}__{control}"].set_index("replicate")
        sd = float(declared["std_error"].mean() * np.sqrt(n / NECESSITY_N))
        bias = broken["estimate"] - broken["truth"]
        moved = broken["estimate"] - declared["estimate"]
        delta = float(bias.mean() / sd)
        out[family] = {
            "standardized_bias": delta,
            "standardized_bias_mc_se": float(bias.std(ddof=1) / np.sqrt(len(bias)) / sd),
            "declared_standardized_bias": float(
                (declared["estimate"] - declared["truth"]).mean() / sd
            ),
            "displacement": float(abs(moved.mean()) / sd),
            "power": tools.necessity_power(delta, NECESSITY_REPLICATES),
            "seconds": float(rows["seconds"].mean()),
        }
    return out


#: Per-fit seconds and peak memory from the one-replicate smoke (warm process, one core,
#: n = 2,000, 2026-10-15), and the CPU-hours they imply.  The calibration pilot replaces the
#: calibration figure with its own mean seconds a fit.
_CALIBRATION_SECONDS = {
    "p1": 0.83,
    "p2": 0.79,
    "p3": 0.80,
    "p4": 0.86,
    "p5": 0.80,
    "p4_sdr": 0.84,
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


def _calibration_hours(replicates: int) -> float:
    return sum(_CALIBRATION_SECONDS.values()) * replicates / _HOUR


BUDGET: dict[str, Any] = {
    "seconds": {
        "primary_replicate": _PRIMARY_SECONDS,
        "calibration_fit": _CALIBRATION_SECONDS,
        "necessity_pair": _PAIR_SECONDS,
    },
    "peak_gib": 0.57,
    "cpu_hours": {
        "calibration_pilot": _calibration_hours(PILOT_DRAWS),
        "necessity_pilot": NECESSITY_PILOT_DRAWS
        * sum(_PAIR_SECONDS.values())
        * (NECESSITY_PILOT_N / NECESSITY_N)
        / _HOUR,
        "calibration": {r: _calibration_hours(r) for r in (6_000, 6_400, 8_000)},
        "necessity": NECESSITY_REPLICATES * sum(_PAIR_SECONDS.values()) / _HOUR,
        "primary": study.PRIMARY_REPLICATES * _PRIMARY_SECONDS / _HOUR,
    },
}
BUDGET["cpu_hours"]["python_total"] = (
    BUDGET["cpu_hours"]["calibration_pilot"]
    + BUDGET["cpu_hours"]["necessity_pilot"]
    + BUDGET["cpu_hours"]["calibration"][6_400]
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
    arguments = parser.parse_args(argv)
    if arguments.pilot == "calibration":
        result = calibration_pilot(
            arguments.parts,
            draws=arguments.draws or PILOT_DRAWS,
            n_jobs=arguments.jobs,
            labels=tuple(arguments.labels),
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
