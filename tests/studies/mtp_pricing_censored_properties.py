"""Property families of ``mtp-pricing-censored``.

Every cell reads the mean purchase under ``price_up``, ``ey_regimen[price_up]``, on law P-C
(:mod:`tests.studies.mtp_pricing_censored_common`) at three nodes, with five folds, the
screened classifier of ``mtp-pricing`` at three nodes
(:data:`~tests.studies.mtp_pricing_properties.SCREENED_CLASSIFIER`, ``QuadraticLogistic``), and
the declared censoring learner (:func:`~tests.studies.mtp_pricing_censored_common.censoring_learner`).

==========================  =====================================================================
family                      cells, replications and size
==========================  =====================================================================
``interval_calibration``    ``censored_crossfit_{ltmle,sdr}_tau3``: the declared outcome learners
                            (``FinalOutcome``, ``SplinePseudo``), each positive with the derived
                            ``noise_control`` and ``shrunken_se_control``;
                            :data:`CALIBRATION_REPLICATES` at :data:`CALIBRATION_N`
``censoring_necessity``     per engine, ``up_<engine>_tau3__learned_censoring`` (positive)
                            against ``up_<engine>_tau3__prior_censoring_control``: the wrong
                            outcome GLM (``WrongOutcome``) at every node and the learned
                            classifier; the control's censoring factor is the per-node marginal
                            retention (``DummyClassifier(strategy="prior")``), what a fit that
                            assumes non-informative dropout uses.  With a wrong outcome model only
                            the weights correct for the informative dropout.
                            :data:`PAIR_REPLICATES` at :data:`PAIR_N`
==========================  =====================================================================

Each calibration cell changes one thing against its complete-data sibling of ``mtp-pricing``
(``crossfit_<engine>_learned_classifier_tau3``): the dropout.  The necessity positives are also
double-robustness evidence under dropout (wrong outcome, correct ratio and censoring), and their
bias clause gates.

**Sizes, declared before the pilots** (X77 plan rev 2, section 5).

* Calibration: sized from the pilots by the per-arm rule (:func:`tools.per_arm_replicates`;
  the two engines are distinct settings).  The plan declared R = 2,800 by the guide's fallback
  for 400-draw pilots, with an arm whose plug-in pass probability fell below 0.90 to have its
  pilot extended.  At 400 draws the arms read SE ratios 1.066 (``LTMLE``) and 1.059 (``SDR``),
  at 1,200 draws 1.047 and 1.016, and at 2,800 draws 1.019 and 1.020
  (:data:`PILOT_SE_RATIOS`), coverage 0.949 and 0.951.  The ``LTMLE`` excess at 1,200 draws
  was the same with the true retention as the censoring learner (1.044 against 1.047 on the
  same draws), so it did not come from learning the censoring model, and its 99% bootstrap
  interval, [1.00, 1.10], held 1.  At 2,800 draws the plug-in pass at R = 2,800 is 0.83 per
  arm, so the per-arm rule over the 2,800-draw pilots sets R = 10,800: pass 0.976 and 0.975,
  family 0.952 (:func:`calibration_replicates`).  ``mtp-pricing``'s siblings ran 7,200.
* Necessity pairs: ``mtp-pricing``'s pair-pilot rule (:func:`pair_pilot`), at least
  :data:`PAIR_PILOT_DRAWS` paired draws per engine at the declared n, each positive's
  ``equivalence_power`` and each control's ``necessity_power`` at least 0.90 at
  :data:`PAIR_REPLICATES`.  Exit declared before the pilot: a larger n, or ``reporting`` with an
  owner, never a weaker law.  The 200-draw pilots at n = 2,400 (:data:`PAIR_PILOT_READINGS`)
  read positive biases 0.002 and 0.042 SD and controls displaced by 0.47 and 0.61 SD; every
  power is above 0.9999, so R = 1,000 and the retention intercept 2.8 stand.

The red-cell reading is in :mod:`tests.studies.canonical_mtp_pricing_censored`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import norm
from sklearn.dummy import DummyClassifier

from tests.parallel import STUDY_JOBS, memory_capped_workers
from tests.studies import canonical_mtp_pricing_censored as study
from tests.studies import mtp_pricing_censored_common as law
from tests.studies import mtp_pricing_common as common
from tests.studies import mtp_pricing_properties as pricing_properties
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

STUDY = study.STUDY
TAU = law.TAU
PLAN = study.PLAN
CALIBRATION_N = study.PRIMARY_N
PAIR_N = study.PRIMARY_N
PAIR_REPLICATES = 1_000
CROSSFIT_FOLDS = 5
SHRUNKEN_SE_FACTOR = 0.70
NECESSITY_DISPLACEMENT = 0.10
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
CLASSIFIER = pricing_properties.SCREENED_CLASSIFIER[TAU]
ESTIMAND = f"ey_regimen[{PLAN}]"

#: Each calibration label: its engine.
CELLS: dict[str, str] = {f"censored_crossfit_{e}_tau{TAU}": e for e in ("ltmle", "sdr")}
assert tuple(CELLS) == study.CALIBRATION_LABELS

#: The calibration pilot's draws per arm as first declared, and as extended and read.
PILOT_DRAWS = 400
PILOT_DRAWS_READ = 2_800
PAIR_PILOT_DRAWS = 200
#: The plan's fallback size, at which an arm's plug-in pass probability is read.
TARGET_REPLICATES = 2_800
#: Each arm's SE ratio on its 2,800-draw pilot (``diagnostic`` stream).
PILOT_SE_RATIOS: dict[str, float] = {
    "censored_crossfit_ltmle_tau3": 1.0194685493760738,
    "censored_crossfit_sdr_tau3": 1.0197899214098862,
}
#: Each arm's replications: :func:`calibration_replicates`.
SIZED_REPLICATES = 10_800
CALIBRATION_REPLICATES: dict[str, int] = dict.fromkeys(CELLS, SIZED_REPLICATES)
#: The noise of each ``noise_control`` is ``NOISE_FACTOR`` times the arm's own SD (its
#: calibration pilot's empirical SD times sqrt(n)), as in ``mtp-pricing``.
NOISE_FACTOR = 1.0
#: Each arm's ``noise_sd`` from its 2,800-draw calibration pilot.
NOISE_SD: dict[str, float] = {
    "censored_crossfit_ltmle_tau3": 1.038,
    "censored_crossfit_sdr_tau3": 1.053,
}
assert set(NOISE_SD) == set(CELLS)

#: The pair pilots' readings (``tools.pair_reading``, 200 paired draws, ``diagnostic`` stream).
PAIR_PILOT_READINGS: dict[str, dict[str, float]] = {
    "necessity_ltmle_tau3": {
        "positive_standardized_bias": 0.002075644680509305,
        "control_standardized_bias": 0.5061164454214552,
        "equivalence_power": 0.9999998954438447,
        "necessity_power": 0.9999999833639209,
        "displacement": 0.46869860202749836,
    },
    "necessity_sdr_tau3": {
        "positive_standardized_bias": 0.04224099396042749,
        "control_standardized_bias": 0.7407867744721034,
        "equivalence_power": 0.9999675280243919,
        "necessity_power": 1.0,
        "displacement": 0.6120209562029517,
    },
}

#: Each pair set: its engine.
PAIR_SETS: dict[str, str] = {f"necessity_{e}_tau{TAU}": e for e in ("ltmle", "sdr")}
PAIR_ARMS: tuple[str, str] = ("learned_censoring", "prior_censoring_control")

#: The cells that gate the merge though the record publishes under ``reporting``.
GATING_CELLS: tuple[str, ...] = (
    *(
        f"interval_calibration/{label}__{cell}"
        for label in CELLS
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    *(
        f"{family}/{cell}"
        for family, pairs in study.PAIRED_CELLS.items()
        for pair in pairs
        for cell in pair
    ),
)

_LEARNED_LIMIT = (
    "a learned ratio: the fit's own curve reads the classifier's limit, which has no closed "
    "form, and the efficient influence function's tail size alone is a lower bound of its own"
)
#: The calibration positives whose n is below twice their tail size.
BELOW_TAIL_SIZE: frozenset[str] = frozenset()
#: The calibration positives whose n is at least twice their tail size.
WALD_GATED_CELLS: frozenset[str] = frozenset()
#: The calibration positives with no computed tail size, each with the reason.
TAIL_SIZE_NOT_COMPUTED: dict[str, str] = {
    f"interval_calibration/{label}__correctly_specified": _LEARNED_LIMIT for label in CELLS
}


def truth() -> float:
    return common.truth(TAU, f"ey[{PLAN}]")


def calibration_replicates() -> dict[str, Any]:
    """The per-arm rule over the 2,800-draw pilots: R, each arm's pass and the family's."""
    return tools.per_arm_replicates(
        [PILOT_SE_RATIOS[label] for label in CELLS], PILOT_DRAWS_READ, tools.C_FIT
    )


def target_replicates() -> dict[str, Any]:
    """The smallest multiple of 400 passing a calibrated arm at 0.975 and two at 0.95."""
    for replicates in range(tools.R_STEP, tools.R_MAX + 1, tools.R_STEP):
        cell = float(tools.cell_pass(1.0, replicates, tools.C_FIT))
        if cell**2 >= tools.FAMILY_TARGET:
            return {"R": replicates, "per_cell": cell, "family_wise": cell**2}
    raise AssertionError("no R reaches the target")


# ---------------------------------------------------------------------------- fits


def classifier() -> Any:
    return pricing_properties.classifier(CLASSIFIER)


def calibration_fit(label: str, frame: pd.DataFrame) -> Any:
    return study.fit(
        frame,
        estimator=CELLS[label],
        n_folds=CROSSFIT_FOLDS,
        ratio="classifier",
        treatment_learner=classifier(),
        **pricing_properties.declared_learners(),
    )


def pair_fits(name: str, frame: pd.DataFrame) -> tuple[Any, Any]:
    """The positive and the control of one necessity pair, on one draw."""
    settings: dict[str, Any] = {
        "estimator": PAIR_SETS[name],
        "n_folds": CROSSFIT_FOLDS,
        "ratio": "classifier",
        "treatment_learner": classifier(),
        **pricing_properties.wrong_learners(),
    }
    declared = study.fit(frame, **settings)
    control = study.fit(frame, censoring_learner=DummyClassifier(strategy="prior"), **settings)
    return declared, control


# ---------------------------------------------------------------------------- the grid


class DeclaredLaw:
    """Law P-C at three nodes, with the truth of the cells' estimand."""

    name = "P_C_tau3"

    def truth(self) -> dict[str, float]:
        return {ESTIMAND: truth()}


#: Every fit set: ``(family, stream label, n, replicates)``.
FIT_SETS: tuple[tuple[str, str, int, int], ...] = (
    *(
        ("interval_calibration", label, CALIBRATION_N, CALIBRATION_REPLICATES[label])
        for label in CELLS
    ),
    *(("pairs", name, PAIR_N, PAIR_REPLICATES) for name in PAIR_SETS),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def _pilot_seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "diagnostic", "pilot", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(family: str, cell: str, role: str, n: int, reps: int, stream: str, kind: str) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(kind, stream, 0),
                role=role,
                estimand=ESTIMAND,
            )
        )

    for label in CELLS:
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                CALIBRATION_N,
                CALIBRATION_REPLICATES[label],
                label,
                "interval_calibration",
            )
    for name, engine in PAIR_SETS.items():
        for arm, role in zip(PAIR_ARMS, ("positive", "control"), strict=True):
            add(
                "censoring_necessity",
                f"up_{engine}_tau{TAU}__{arm}",
                role,
                PAIR_N,
                PAIR_REPLICATES,
                name,
                "pairs",
            )
    return tuple(cells)


# ---------------------------------------------------------------------------- rows


def _fit_rows(
    family: str, label: str, replicate: int, n: int, requested: int, frame: pd.DataFrame
) -> list[dict[str, Any]]:
    """The rows of one replication of one fit set, fitted on its drawn ``frame``."""
    shared: dict[str, Any] = {"replicate": replicate, "n": n, "requested": requested}
    value = truth()
    if family == "interval_calibration":
        result = calibration_fit(label, frame)
        return [
            replicate_row(
                property_name=family,
                cell=f"{label}__correctly_specified",
                role="positive",
                truth=value,
                estimate=result[ESTIMAND],
                alpha=STUDY.margins.alpha,
                **shared,
            )
        ]
    engine = PAIR_SETS[label]
    declared, control = pair_fits(label, frame)
    positive, negative = PAIR_ARMS
    return [
        replicate_row(
            property_name="censoring_necessity",
            cell=f"up_{engine}_tau{TAU}__{positive}",
            role="positive",
            truth=value,
            estimate=declared[ESTIMAND],
            alpha=STUDY.margins.alpha,
            **shared,
        ),
        control_row(
            property_name="censoring_necessity",
            cell=f"up_{engine}_tau{TAU}__{negative}",
            truth=value,
            estimate=float(control[ESTIMAND].psi),
            standard_error=float(control[ESTIMAND].std_error),
            critical=CRITICAL,
            **shared,
        ),
    ]


def fit_set_rows(
    family: str, label: str, replicate: int, n: int, requested: int, seed: int | None = None
) -> list[dict[str, Any]]:
    """The rows of one replication of one fit set, from its declared stream or ``seed``."""
    frame = law.sample(n, _seed(family, label, replicate) if seed is None else seed)
    return _fit_rows(family, label, replicate, n, requested, frame)


#: A property key: ``(family, stream label, replicate, n, requested)``.
_Key = tuple[str, str, int, int, int]

#: Each calibration positive's derived controls.
DERIVED_CELLS: dict[str, tuple[str, ...]] = calibration_derived(tuple(CELLS))


def _draw(key: _Key) -> pd.DataFrame:
    family, label, replicate, n, _ = key
    return law.sample(n, _seed(family, label, replicate))


def _fit(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _fit_rows(*key, frame)


def _cells_of(key: _Key) -> list[str]:
    family, label, *_ = key
    if family == "interval_calibration":
        positive = f"{family}/{label}__correctly_specified"
        return [positive, *DERIVED_CELLS[positive]]
    engine = PAIR_SETS[label]
    return [f"censoring_necessity/up_{engine}_tau{TAU}__{arm}" for arm in PAIR_ARMS]


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def _seed_of(key: _Key) -> int:
    family, label, replicate, *_ = key
    return _seed(family, label, replicate)


#: A generous peak of one property worker at n = 2,000.
FIT_BYTES = 2**30
#: The share of a cell's requested replications the accounting lets a run lose.
FAILURE_BOUND = 0.01
#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, parts: Path | None = None
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set."""
    jobs = memory_capped_workers(n_jobs, FIT_BYTES)
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
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=tuple(CELLS),
                efficiency_bounds=NOISE_SD,
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


#: The necessity family's declaration, read by the verdict and by its rule.
NECESSITIES: tuple[Necessity, ...] = (
    Necessity(
        family="censoring_necessity",
        labels=tuple(f"up_{engine}_tau{TAU}" for engine in PAIR_SETS.values()),
        arms=PAIR_ARMS,
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    ),
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


#: The failure accounting's declared rules, as ``mtp-pricing``'s.
PROPERTY_RULES = (
    property_verdicts.calibration_rule(efficiency_band=None),
    *property_verdicts.necessity_rules(*NECESSITIES),
)


# ---------------------------------------------------------------------------- pilots


def _pilot_rows(family: str, label: str, replicate: int, n: int, requested: int) -> list[dict]:
    started = time.perf_counter()
    rows = fit_set_rows(
        family, label, replicate, n, requested, seed=_pilot_seed(family, label, replicate)
    )
    seconds = time.perf_counter() - started
    for row in rows:
        row["seconds"] = seconds / len(rows)
    return rows


def arm_summary(rows: pd.DataFrame) -> dict[str, Any]:
    """An arm's pilot SE ratio, standardized bias, coverage, noise scale and pass probability."""
    reading = pricing_properties.arm_summary(rows)
    reading["noise_sd"] = float(NOISE_FACTOR * reading["noise_sd"])
    reading["plug_in_pass"] = float(
        tools.cell_pass(reading["se_ratio"], TARGET_REPLICATES, tools.C_FIT)
    )
    return reading


def calibration_pilot(parts: Path, *, draws: int = PILOT_DRAWS, n_jobs: int = 1) -> dict[str, Any]:
    """Each calibration arm on the ``diagnostic`` stream, before the declaration."""
    out: dict[str, Any] = {"target": target_replicates(), "arms": {}}
    for label in CELLS:
        keys = [("interval_calibration", label, r, CALIBRATION_N, draws) for r in range(draws)]
        rows = tools.run_in_parts(
            _pilot_rows, keys, label=f"pilot__{label}", root=parts, n_jobs=n_jobs
        )
        out["arms"][label] = arm_summary(rows)
    return out


def pair_pilot(parts: Path, *, draws: int = PAIR_PILOT_DRAWS, n_jobs: int = 1) -> dict[str, Any]:
    """Both arms of each necessity pair on paired draws, before the declaration."""
    out: dict[str, Any] = {}
    for name, engine in PAIR_SETS.items():
        keys = [("pairs", name, r, PAIR_N, draws) for r in range(draws)]
        rows = tools.run_in_parts(
            _pilot_rows, keys, label=f"pair__{name}", root=parts, n_jobs=n_jobs
        )
        positive, control = (f"up_{engine}_tau{TAU}__{arm}" for arm in PAIR_ARMS)
        declared = rows.loc[rows["cell"] == positive].set_index("replicate")
        broken = rows.loc[rows["cell"] == control].set_index("replicate").loc[declared.index]
        reading = tools.pair_reading(
            declared["estimate"],
            broken["estimate"],
            declared["truth"],
            PAIR_REPLICATES,
            NECESSITY_DISPLACEMENT,
        )
        reading["seconds"] = float(rows["seconds"].sum() / draws)
        out[name] = reading
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phase", choices=("calibration-pilot", "pair-pilot"))
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--draws", type=int, default=None)
    args = parser.parse_args(argv)
    if args.phase == "calibration-pilot":
        reading = calibration_pilot(args.parts, draws=args.draws or PILOT_DRAWS, n_jobs=args.jobs)
    else:
        reading = pair_pilot(args.parts, draws=args.draws or PAIR_PILOT_DRAWS, n_jobs=args.jobs)
    print(json.dumps(reading, indent=2, default=float))


if __name__ == "__main__":
    main()
