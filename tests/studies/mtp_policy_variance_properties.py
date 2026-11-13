r"""Property families of the ``mtp-policy-variance`` study.

The driver and the record are :mod:`tests.studies.canonical_mtp_policy_variance`.  Each fit set
below draws its own samples on the ``property_sample`` stream, fits once, and writes the rows of
both variances: the option (``variance="policy_weighted"``) and the default Wald interval.  The
controls are derived from a positive's own rows and cost no fit.

==========================  =================================================================
family                      cells
==========================  =================================================================
``policy_variance``         ``pricing_{ltmle,sdr}_tau{5,2}`` (law P, ``price_down``, n = 5,000),
                            ``boundaries_{ltmle,sdr}_p4`` (law B, primary learners, n = 5,000):
                            ``<set>__<q>__option`` gated on bias, the coverage floor and the
                            SE-ratio clause, with ``__coverage_floor_control`` (option SE times
                            :data:`COVERAGE_FLOOR_FACTOR`).  ``tau5`` adds the Wald control
                            ``__wald`` (coverage upper bound below 0.95) and
                            ``__inflated_se_control`` (option SE times 1.40).  The other Wald
                            rows report, ``boundaries_*`` with the RMS-ratio reading.
                            ``pricing_learned_{ltmle,sdr}_tau5`` (the screened classifier) report.
``policy_calibration``      ``p4_{ltmle,sdr}`` (law B ``p4``, correct outcome model, oracle
                            hazard, n = 4,000): ``__option`` gated as above, with
                            ``__shrunken_se_control``, ``__inflated_se_control``,
                            ``__noise_control`` and ``__coverage_floor_control``; ``__wald``
                            reports.
``interval_calibration``    the Wald calibration twins ``p4_twin``, ``p4_sdr_twin`` (104,400),
                            ``p5_twin``, ``p5_sdr_twin`` (6,400), ``up_oracle_tau5_twin``
                            (4,000), each with the shrunken-SE and noise controls, read by the
                            source studies' calibration rule.
``reporting_calibration``   ``up_learned_classifier_tau5_twin`` (4,000): its tail size is a lower
                            bound only, so it reports.
``sdr_default_pseudo``      ``p4_sdr__superlearner`` (the pseudo learner ``SDR`` builds by
                            default), both variances, reporting with the follow-up trigger.
``tail_option``             the option on the large coverage twins, reporting.
==========================  =================================================================

**The SE-ratio clause**, on every gated option row, reads the 99% bootstrap interval
``[lo, hi]`` of ``mean(SE) / SD(estimate)``: ``lo < 0.93`` is ``red: too narrow``, ``lo > 1.25``
is ``red: too wide``; otherwise the row is ``conservative`` when ``hi > 1.07`` and
``calibrated`` when not, provided its coverage floor and bias hold (:func:`reading`).

**The coverage-floor factor** :data:`COVERAGE_FLOOR_FACTOR` is the largest of
:data:`FLOOR_FACTORS` whose control fails as required with probability at least 0.95 on every
gated option row, read from the committed probe-5 draws of
``tests/diagnostics/fix5_heavy_ratios`` (:func:`floor_control_power`), fixed before any gated
run.

Replication counts, pilots and pass probabilities are checked by
``tests/unit/test_mtp_policy_variance_design.py`` from the committed pilot
(:data:`PILOT_ROWS`).
"""

from __future__ import annotations

import argparse
import functools
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, binom, norm

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_mtp_policy_variance as study
from tests.studies import mtp_boundaries_common as law_b
from tests.studies import mtp_pricing_properties as pricing_properties
from tests.studies import mtp_study_tools as tools
from tests.studies.canonical_mtp_policy_variance import (
    B_REF,
    DEFAULT,
    DOWN,
    OPTION,
    UP,
    Config,
    ate,
    ey,
    quiet,
    readings,
)
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    ratio_intervals,
    summary_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    indexed_noise,
)
from tests.studies.evidence.registry import ROOT
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
CRITICAL = study.CRITICAL
FAILURE_BOUND = 0.01
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS

#: The SE-ratio clause: ``lo`` below the floor is red-narrow, above the ceiling red-wide, and
#: ``hi`` above :data:`CALIBRATED_CEILING` is conservative.
CLAUSE_FLOOR = 0.93
CLAUSE_CEILING = 1.25
CALIBRATED_CEILING = 1.07
#: The Wald control's coverage ceiling: its 99% upper bound must fall below nominal.
WALD_CONTROL_CEILING = 0.95
#: The coverage-floor control's ceiling on its 99% upper bound.
FLOOR_CONTROL_CEILING = study.CONTROL_COVERAGE_CEILING
#: The declared factors of the coverage-floor control on option rows, largest first.
FLOOR_FACTORS: tuple[float, ...] = (0.70, 0.65, 0.60, 0.55, 0.50)
#: The declared factor: the largest of :data:`FLOOR_FACTORS` whose control fails as required
#: with probability at least 0.95 on every gated option row (:func:`chosen_floor_factor`).
COVERAGE_FLOOR_FACTOR = 0.55
SHRUNKEN_SE_FACTOR = 0.70
INFLATED_SE_FACTOR = 1.40
#: The power every control must reach, and the family-wise target of the gated rows.
CONTROL_POWER = 0.95
FAMILY_TARGET = 0.95

PILOT_DRAWS = 400
PILOT_ROWS = ROOT / "tests" / "studies" / "mtp_policy_variance_pilot.csv.gz"
PROBE5_ROWS = study.DIAGNOSTIC / "probe5_rows.csv.gz"

QUANTITIES = ("ey", "ate")


# ---------------------------------------------------------------------------- the fit sets

#: Each option fit set of ``policy_variance``: its configuration.
OPTION_SETS: dict[str, Config] = {
    **{
        f"pricing_{est}_tau{tau}": Config("pricing", est, tau, DOWN)
        for tau in (5, 2)
        for est in ("ltmle", "sdr")
    },
    **{
        f"boundaries_{est}_p4": Config("boundaries", est, 2, "p4", outcome="primary")
        for est in ("ltmle", "sdr")
    },
    **{
        f"pricing_learned_{est}_tau5": Config("pricing_learned", est, 5, DOWN)
        for est in ("ltmle", "sdr")
    },
}
#: The ``policy_variance`` sets that report (outside the condition of the option's guarantee).
REPORTING_SETS: frozenset[str] = frozenset(
    {"pricing_learned_ltmle_tau5", "pricing_learned_sdr_tau5"}
)
#: The sets whose Wald rows are the below-tail control, with the inflated-SE control.
WALD_CONTROL_SETS: frozenset[str] = frozenset({"pricing_ltmle_tau5", "pricing_sdr_tau5"})
#: The sets whose Wald rows carry the RMS-ratio reading.
RMS_SETS: frozenset[str] = frozenset({"boundaries_ltmle_p4", "boundaries_sdr_p4"})
#: The option calibration cells: law B ``p4``, correct outcome model, oracle hazard.
CALIBRATION_SETS: dict[str, Config] = {
    "p4_ltmle": Config("boundaries", "ltmle", 2, "p4", outcome="correct"),
    "p4_sdr": Config("boundaries", "sdr", 2, "p4", outcome="correct", pseudo="unit_range"),
}
CALIBRATION_N = 4_000
#: The Wald calibration twins and the reporting one: configuration and n.
TWIN_SETS: dict[str, tuple[Config, int]] = {
    "p4_twin": (
        Config("boundaries", "ltmle", 2, "p4", outcome="correct", ratio="classifier"),
        study.twin_n("p4_twin"),
    ),
    "p4_sdr_twin": (
        Config(
            "boundaries",
            "sdr",
            2,
            "p4",
            outcome="correct",
            ratio="classifier",
            pseudo="unit_range",
        ),
        study.twin_n("p4_sdr_twin"),
    ),
    "p5_twin": (Config("boundaries", "ltmle", 2, "p5", outcome="correct"), study.twin_n("p5_twin")),
    "p5_sdr_twin": (
        Config("boundaries", "sdr", 2, "p5", outcome="correct", pseudo="unit_range"),
        study.twin_n("p5_sdr_twin"),
    ),
    "up_oracle_tau5_twin": (
        Config("pricing_property", "ltmle", 5, UP, label="up_oracle_tau5"),
        study.twin_n("up_oracle_tau5_twin"),
    ),
    "up_learned_classifier_tau5_twin": (
        Config("pricing_property", "ltmle", 5, UP, label="up_learned_classifier_tau5"),
        study.twin_n("up_learned_classifier_tau5_twin"),
    ),
}
REPORTING_TWINS: frozenset[str] = frozenset({"up_learned_classifier_tau5_twin"})
GATED_TWINS: tuple[str, ...] = tuple(t for t in TWIN_SETS if t not in REPORTING_TWINS)
#: The SD that scales each calibration twin's noise control: law B's efficiency bound, law P's
#: pilot SD times sqrt(2,000) (``mtp_pricing_properties.NOISE_SD``).
TWIN_NOISE_SD: dict[str, float] = {
    "p4_twin": law_b.BOUNDS["p4"],
    "p4_sdr_twin": law_b.BOUNDS["p4"],
    "p5_twin": law_b.BOUNDS["p5"],
    "p5_sdr_twin": law_b.BOUNDS["p5"],
    "up_oracle_tau5_twin": pricing_properties.NOISE_SD["up_oracle_tau5"],
}
#: The default-pseudo-learner cell and its trigger: the default Wald's coverage upper bound
#: below 0.95, or its SE-ratio interval wholly below 0.93.
SUPERLEARNER = Config("boundaries", "sdr", 2, "p4", outcome="correct", pseudo="default")
SUPERLEARNER_CELL = "p4_sdr__superlearner"

#: The replications of each fit set (the design test checks each against the pilot).
REPLICATES: dict[str, int] = {
    "pricing_ltmle_tau5": 2_600,
    "pricing_sdr_tau5": 2_600,
    "pricing_ltmle_tau2": 4_400,
    "pricing_sdr_tau2": 4_400,
    "boundaries_ltmle_p4": 1_000,
    "boundaries_sdr_p4": 1_000,
    "pricing_learned_ltmle_tau5": 1_000,
    "pricing_learned_sdr_tau5": 1_000,
    "p4_ltmle": 2_800,
    "p4_sdr": 2_800,
    **dict.fromkeys(TWIN_SETS, 3_600),
    SUPERLEARNER_CELL: 600,
    **dict.fromkeys(study.LARGE_TWINS, study.TAIL_OPTION_REPLICATES),
}

#: Every fit set: ``(family, label, n)``.
FIT_SETS: tuple[tuple[str, str, int], ...] = (
    *(("policy_variance", label, 5_000) for label in OPTION_SETS),
    *(("policy_calibration", label, CALIBRATION_N) for label in CALIBRATION_SETS),
    *(
        (
            "reporting_calibration" if label in REPORTING_TWINS else "interval_calibration",
            label,
            n,
        )
        for label, (_, n) in TWIN_SETS.items()
    ),
    ("sdr_default_pseudo", SUPERLEARNER_CELL, CALIBRATION_N),
    *(("tail_option", twin, study.SCENARIO_N[twin]) for twin in study.LARGE_TWINS),
)


def config_of(family: str, label: str) -> Config:
    if family == "policy_variance":
        return OPTION_SETS[label]
    if family == "policy_calibration":
        return CALIBRATION_SETS[label]
    if family in ("interval_calibration", "reporting_calibration"):
        return TWIN_SETS[label][0]
    if family == "sdr_default_pseudo":
        return SUPERLEARNER
    if family == "tail_option":
        return study.TWINS[label][0]
    raise ValueError(family)


def _quantity_name(config: Config, quantity: str) -> str:
    ey_name, ate_name = config.names
    return ey_name if quantity == "ey" else ate_name


# ---------------------------------------------------------------------------- cells


def option_cells(label: str) -> dict[str, str]:
    """A ``policy_variance`` set's own cells and their roles."""
    reporting = label in REPORTING_SETS
    out = {}
    for q in QUANTITIES:
        out[f"{label}__{q}__option"] = DIAGNOSTIC_ROLE if reporting else "positive"
        out[f"{label}__{q}__wald"] = "control" if label in WALD_CONTROL_SETS else DIAGNOSTIC_ROLE
    return out


def option_derived(label: str) -> dict[str, tuple[str, ...]]:
    """A gated ``policy_variance`` set's derived controls, per option cell."""
    if label in REPORTING_SETS:
        return {}
    kinds = ["coverage_floor_control"]
    if label in WALD_CONTROL_SETS:
        kinds.append("inflated_se_control")
    return {
        f"policy_variance/{label}__{q}__option": tuple(
            f"policy_variance/{label}__{q}__{kind}" for kind in kinds
        )
        for q in QUANTITIES
    }


CALIBRATION_CONTROLS = (
    "shrunken_se_control",
    "inflated_se_control",
    "noise_control",
    "coverage_floor_control",
)


def own_cells(family: str, label: str) -> dict[str, str]:
    """The cells a fit set's rows name, with their roles (derived controls excluded)."""
    if family == "policy_variance":
        return option_cells(label)
    if family == "policy_calibration":
        return {f"{label}__option": "positive", f"{label}__wald": DIAGNOSTIC_ROLE}
    if family == "interval_calibration":
        return {f"{label}__correctly_specified": "positive"}
    if family == "reporting_calibration":
        return {f"{label}__wald": DIAGNOSTIC_ROLE}
    if family == "sdr_default_pseudo":
        return {f"{label}__wald": DIAGNOSTIC_ROLE, f"{label}__option": DIAGNOSTIC_ROLE}
    if family == "tail_option":
        return {
            f"{label}__{q}__{kind}": DIAGNOSTIC_ROLE
            for q in QUANTITIES
            for kind in ("option", "wald")
        }
    raise ValueError(family)


#: Each positive cell's derived controls, which the keys name and :func:`derived_rows` writes.
DERIVED_CELLS: dict[str, tuple[str, ...]] = {
    **{k: v for label in OPTION_SETS for k, v in option_derived(label).items()},
    **{
        f"policy_calibration/{label}__option": tuple(
            f"policy_calibration/{label}__{kind}" for kind in CALIBRATION_CONTROLS
        )
        for label in CALIBRATION_SETS
    },
    **calibration_derived(GATED_TWINS),
}


def declared_cells() -> dict[str, set[str]]:
    """Every cell the fit sets and their derived controls write, by family."""
    out: dict[str, set[str]] = {}
    for family, label, _ in FIT_SETS:
        for cell in own_cells(family, label):
            out.setdefault(family, set()).add(cell)
            out[family] |= {d.split("/", 1)[1] for d in DERIVED_CELLS.get(f"{family}/{cell}", ())}
    return out


# ---------------------------------------------------------------------------- rows


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def pilot_seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "diagnostic", "pilot", family, label, replicate)


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    psi: float,
    se: float,
) -> dict[str, Any]:
    half = CRITICAL * se
    return {
        "property": family,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": psi,
        "std_error": se,
        "covered": int(psi - half <= truth <= psi + half),
        "rejected": int(abs(psi / se) > CRITICAL) if se > 0 else 0,
    }


def fit_rows(
    family: str,
    label: str,
    replicate: int,
    n: int,
    requested: int,
    frame: pd.DataFrame,
    *,
    wald_only: bool = False,
) -> list[dict[str, Any]]:
    """The own rows of one replication of one fit set, fitted on its drawn ``frame``.

    ``wald_only`` fits the default variance and writes the Wald rows alone (the twins'
    pilots).
    """
    config = config_of(family, label)
    truth = config.truth()
    roles = own_cells(family, label)
    option = family not in ("interval_calibration", "reporting_calibration") and not wald_only
    with quiet():
        result = config.fit(frame, OPTION if option else DEFAULT)
    values = readings(result, config.names)

    def row(cell: str, name: str, kind: str) -> dict[str, Any]:
        se = values[name]["se"] if kind == "option" else values[name]["se_ic"]
        return _row(
            family, cell, roles[cell], replicate, n, requested, truth[name], values[name]["psi"], se
        )

    out = []
    for cell in roles:
        kind = cell.rsplit("__", 1)[-1]
        if wald_only and kind == "option":
            continue
        if family in ("policy_variance", "tail_option"):
            name = _quantity_name(config, cell.split("__")[1])
        else:
            name = config.names[1]
        out.append(row(cell, name, "wald" if kind == "correctly_specified" else kind))
    return out


def draw_frame(family: str, label: str, n: int, seed: int) -> pd.DataFrame:
    return config_of(family, label).sample(n, seed)


#: A property key: ``(family, label, replicate, n, requested)``.
_Key = tuple[str, str, int, int, int]


def _draw(key: _Key) -> pd.DataFrame:
    family, label, replicate, n, _ = key
    return draw_frame(family, label, n, _seed(family, label, replicate))


def _fit(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_rows(*key, frame)


def _cells_of(key: _Key) -> list[str]:
    family, label, *_ = key
    out = []
    for cell in own_cells(family, label):
        name = f"{family}/{cell}"
        out += [name, *DERIVED_CELLS.get(name, ())]
    return out


def _cell_of(key: _Key) -> str:
    return _cells_of(key)[0]


def _replicate_of(key: _Key) -> int:
    return key[2]


def _seed_of(key: _Key) -> int:
    family, label, replicate, *_ = key
    return _seed(family, label, replicate)


def _recompute(rows: pd.DataFrame) -> pd.DataFrame:
    half = CRITICAL * rows["std_error"]
    rows["covered"] = (
        (rows["estimate"] - half <= rows["truth"]) & (rows["truth"] <= rows["estimate"] + half)
    ).astype(int)
    rows["rejected"] = (np.abs(rows["estimate"] / rows["std_error"]) > CRITICAL).astype(int)
    return rows


def _scaled(base: pd.DataFrame, cell: str, factor: float) -> pd.DataFrame:
    out = base.copy()
    out["cell"] = cell
    out["role"] = "control"
    out["std_error"] = out["std_error"] * factor
    return _recompute(out)


def derived_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Every derived control, from its positive's own rows: no fit, no sample.

    The SE controls scale the reported SE by their factor.  The option calibration cells'
    noise control adds one efficiency-bound unit of noise (law B's ``p4`` bound over
    ``sqrt(n)``), indexed by replicate as
    :func:`~tests.studies.evidence.property_verdicts.calibration_controls` draws it.  The
    calibration twins' controls are :func:`calibration_controls`'s.
    """
    factors = {
        "coverage_floor_control": COVERAGE_FLOOR_FACTOR,
        "inflated_se_control": INFLATED_SE_FACTOR,
        "shrunken_se_control": SHRUNKEN_SE_FACTOR,
    }
    out = []
    for base_name, controls in DERIVED_CELLS.items():
        family, cell = base_name.split("/", 1)
        if family == "interval_calibration":
            continue
        base = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
        if base.empty:
            continue
        for control in controls:
            target = control.split("/", 1)[1]
            kind = target.rsplit("__", 1)[-1]
            if kind in factors:
                out.append(_scaled(base, target, factors[kind]))
                continue
            noisy = base.copy()
            label = cell.removesuffix("__option")
            rng = np.random.default_rng(stream_seed(STUDY, "efficiency_noise", family, label))
            scale = law_b.BOUNDS["p4"] / np.sqrt(CALIBRATION_N)
            noisy["cell"] = target
            noisy["role"] = "control"
            noisy["estimate"] = noisy["estimate"] + indexed_noise(rng, base, label, scale=scale)
            out.append(_recompute(noisy))
    twins = [
        t for t in GATED_TWINS if not rows.loc[rows["cell"] == f"{t}__correctly_specified"].empty
    ]
    for size in sorted({TWIN_SETS[t][1] for t in twins}):
        out.append(
            calibration_controls(
                rows,
                STUDY,
                labels=tuple(t for t in twins if TWIN_SETS[t][1] == size),
                efficiency_bounds=TWIN_NOISE_SD,
                calibration_n=size,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            )
        )
    return pd.concat(out, ignore_index=True) if out else rows.iloc[:0]


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, parts: Path | None = None
) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    keys: list[_Key] = []
    for family, label, n in FIT_SETS:
        requested = REPLICATES[label] if budget is None else budget
        keys += [(family, label, r, n, requested) for r in range(requested)]
    outcomes = fit_properties(
        keys,
        _fit,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
        seed_of=_seed_of,
        checkpoint=tools.property_checkpoint(parts) if budget is None else None,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat([rows, derived_rows(rows)], ignore_index=True)
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------- verdicts

READING_COLUMNS = (
    "reading",
    "rms_ratio",
    "rms_ratio_ci_lower",
    "rms_ratio_ci_upper",
    "follow_up_trigger",
)
OPTION_FAMILIES = frozenset({"policy_variance", "policy_calibration"})


def kind_of(cell: str) -> str:
    return cell.rsplit("__", 1)[-1]


def reading(lo: float, hi: float, floor_met: bool, bias_met: bool) -> str:
    """The SE-ratio clause's reading of a gated option row."""
    if lo < CLAUSE_FLOOR:
        return "red: too narrow"
    if lo > CLAUSE_CEILING:
        return "red: too wide"
    if not floor_met:
        return "red: coverage"
    if not bias_met:
        return "red: bias"
    return "conservative" if hi > CALIBRATED_CEILING else "calibrated"


def rms_interval(group: pd.DataFrame, seed: int, replicates: int) -> tuple[float, float, float]:
    """``sqrt(mean(SE^2)) / RMSE`` and its 99% percentile bootstrap interval."""
    error = (group["estimate"] - group["truth"]).to_numpy(dtype=float)
    se2 = group["std_error"].to_numpy(dtype=float) ** 2
    point = float(np.sqrt(se2.mean() / np.mean(error**2)))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for block in range(0, replicates, 500):
        index = rng.integers(0, len(error), size=(min(500, replicates - block), len(error)))
        draws[block : block + len(index)] = np.sqrt(
            se2[index].mean(axis=1) / (error[index] ** 2).mean(axis=1)
        )
    low, high = np.percentile(draws, [0.5, 99.5])
    return point, float(low), float(high)


def _se_interval(rows: pd.DataFrame, family: str, cell: str) -> Interval:
    group = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
    return ratio_intervals(
        group,
        replicates=STUDY.margins.bootstrap_replicates,
        confidence_level=STUDY.margins.confidence_level,
        seed=stream_seed(STUDY, family, cell),
    )["se_ratio"]


def policy_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The option families' and the reporting families' verdicts and readings."""
    margins = STUDY.margins
    summary["reading"] = pd.Series([None] * len(summary), dtype=object, index=summary.index)
    summary["follow_up_trigger"] = summary["follow_up_trigger"].astype(object)
    reported = summary["property"].isin(
        OPTION_FAMILIES | {"reporting_calibration", "sdr_default_pseudo", "tail_option"}
    )
    for index in summary.index[reported]:
        family, cell = str(summary.at[index, "property"]), str(summary.at[index, "cell"])
        role, kind = str(summary.at[index, "role"]), kind_of(cell)
        ratio = _se_interval(rows, family, cell)
        summary.at[index, "se_ratio_ci_lower"] = ratio.low
        summary.at[index, "se_ratio_ci_upper"] = ratio.high
        coverage = summary_interval(summary, index, "coverage")
        if role == DIAGNOSTIC_ROLE:
            passed = True
            if family == "policy_variance" and kind == "wald" and cell.split("__")[0] in RMS_SETS:
                group = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
                point, low, high = rms_interval(
                    group, stream_seed(STUDY, "rms_ratio", cell), margins.bootstrap_replicates
                )
                summary.at[index, "rms_ratio"] = point
                summary.at[index, "rms_ratio_ci_lower"] = low
                summary.at[index, "rms_ratio_ci_upper"] = high
            if family == "sdr_default_pseudo" and kind == "wald":
                summary.at[index, "follow_up_trigger"] = bool(
                    coverage.high < WALD_CONTROL_CEILING or ratio.high < CLAUSE_FLOOR
                )
        elif kind == "option":
            floor = coverage.low >= margins.coverage_floor
            bias = bool(summary.at[index, "bias_equivalent"])
            summary.at[index, "reading"] = reading(ratio.low, ratio.high, floor, bias)
            passed = bias and floor and CLAUSE_FLOOR <= ratio.low <= CLAUSE_CEILING
        elif kind == "wald":
            passed = coverage.high < WALD_CONTROL_CEILING
        elif kind == "coverage_floor_control":
            passed = coverage.high < FLOOR_CONTROL_CEILING
        elif kind == "inflated_se_control":
            passed = ratio.low > CLAUSE_CEILING
        elif kind in ("shrunken_se_control", "noise_control"):
            passed = ratio.high < CLAUSE_FLOOR
        else:
            raise ValueError(f"unknown cell kind {kind!r} in {family}/{cell}")
        summary.at[index, "passed"] = bool(passed)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts, the calibration twins' rule and the option families' clause."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, extra_columns=READING_COLUMNS, rate_labels=()
    )
    calibration_verdicts(summary, margins=STUDY.margins)
    policy_verdicts(summary, rows)
    return finish(summary, rates)


def _cell_seed(ctx: pf.Context, row: pd.Series) -> int:
    return stream_seed(ctx.record, str(row["property"]), str(row["cell"]))


def _kind(*names: str, role: str | None = None) -> pf.Applies:
    def applies(ctx: pf.Context, row: pd.Series) -> bool:
        return kind_of(str(row["cell"])) in names and (role is None or str(row["role"]) == role)

    return applies


def _clause_judge(ctx: pf.Context, row: pd.Series, iv: Interval) -> pf.Judged:
    slack = min(iv.low - CLAUSE_FLOOR, CLAUSE_CEILING - iv.low)
    return pf.Judged(slack, bool(CLAUSE_FLOOR <= iv.low <= CLAUSE_CEILING))


def policy_rule() -> pf.FamilyRule:
    """The accounting rule of :func:`policy_verdicts` on the two option families."""
    option = _kind("option", role="positive")
    return pf.FamilyRule(
        families=OPTION_FAMILIES,
        shared="policy_verdicts",
        clauses=(
            pf.bias_clause("bias equivalent", kind="equivalent", applies=option),
            pf.proportion_clause(
                "coverage floor",
                statistic="S2",
                judge=pf.floor_judge(lambda ctx: ctx.margins.coverage_floor),
                applies=option,
            ),
            pf.se_interval_clause(
                "se ratio clause",
                which="se_ratio",
                judge=_clause_judge,
                seed=_cell_seed,
                applies=option,
            ),
            pf.proportion_clause(
                "wald coverage below nominal",
                statistic="S2",
                judge=pf.below_judge(lambda ctx: WALD_CONTROL_CEILING),
                applies=_kind("wald", role="control"),
            ),
            pf.proportion_clause(
                "coverage floor control below ceiling",
                statistic="S2",
                judge=pf.below_judge(lambda ctx: FLOOR_CONTROL_CEILING),
                applies=_kind("coverage_floor_control"),
            ),
            pf.se_interval_clause(
                "se ratio above clause",
                which="se_ratio",
                judge=pf.above_judge(lambda ctx: CLAUSE_CEILING),
                seed=_cell_seed,
                applies=_kind("inflated_se_control"),
            ),
            pf.se_interval_clause(
                "se ratio below clause",
                which="se_ratio",
                judge=pf.below_judge(lambda ctx: CLAUSE_FLOOR),
                seed=_cell_seed,
                applies=_kind("shrunken_se_control", "noise_control"),
            ),
        ),
    )


PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=None), policy_rule())


# ---------------------------------------------------------------------------- the factor


@functools.cache
def probe5() -> pd.DataFrame:
    return pd.read_csv(PROBE5_ROWS, float_precision="round_trip")


#: Each gated option row's probe-5 draws: ``(probe config, estimator, estimand) -> fit set``.
PROBE5_ROWS_OF: dict[tuple[str, str, str], str] = {
    **{
        ("pricing_exact_d5", est, name): f"pricing_{est}_tau5"
        for est in ("ltmle", "sdr")
        for name in (ey(DOWN), ate(DOWN, study.P_REF))
    },
    **{("lawb_correct_d3", est, ate("p4", B_REF)): f"p4_{est}" for est in ("ltmle", "sdr")},
    **{
        ("lawb_primary_d3", est, name): f"boundaries_{est}_p4"
        for est in ("ltmle", "sdr")
        for name in (ey("p4"), ate("p4", B_REF))
    },
}


def max_covered(replicates: int, ceiling: float, level: float = 0.99) -> int:
    """The most covered replicates whose Clopper-Pearson upper bound falls below ``ceiling``."""
    covered = np.arange(replicates)
    upper = beta.ppf(1 - (1 - level) / 2, covered + 1, replicates - covered)
    below = covered[upper < ceiling]
    return int(below.max()) if below.size else -1


def min_covered(replicates: int, floor: float, level: float = 0.99) -> int:
    """The fewest covered replicates whose Clopper-Pearson lower bound reaches ``floor``."""
    covered = np.arange(1, replicates + 1)
    lower = beta.ppf((1 - level) / 2, covered, replicates - covered + 1)
    above = covered[lower >= floor]
    return int(above.min()) if above.size else replicates + 1


_JEFFREYS = 20_000


def jeffreys_below(covered: int, draws: int, replicates: int, ceiling: float) -> float:
    """P(the CP99 upper bound falls below ``ceiling`` at ``replicates``), Jeffreys predictive."""
    p = beta.rvs(covered + 0.5, draws - covered + 0.5, size=_JEFFREYS, random_state=1)
    return float(binom.cdf(max_covered(replicates, ceiling), replicates, p).mean())


def jeffreys_floor(covered: int, draws: int, replicates: int, floor: float) -> float:
    """P(the CP99 lower bound reaches ``floor`` at ``replicates``), Jeffreys predictive."""
    p = beta.rvs(covered + 0.5, draws - covered + 0.5, size=_JEFFREYS, random_state=1)
    return float(binom.sf(min_covered(replicates, floor) - 1, replicates, p).mean())


def floor_control_power(factor: float, replicates: Mapping[str, int] = REPLICATES) -> pd.DataFrame:
    """Each probed gated option row's coverage-floor control power at ``factor``."""
    rows = probe5()
    out = []
    for (config, est, name), label in PROBE5_ROWS_OF.items():
        group = rows.loc[(rows.config == config) & (rows.est == est) & (rows.estimand == name)]
        error = (group["psi"] - group["truth"]).abs()
        covered = int((error <= CRITICAL * factor * group["se"]).sum())
        out.append(
            {
                "set": label,
                "estimand": name,
                "draws": len(group),
                "covered": covered,
                "power": jeffreys_below(
                    covered, len(group), replicates[label], FLOOR_CONTROL_CEILING
                ),
            }
        )
    return pd.DataFrame(out)


def chosen_floor_factor() -> float | None:
    """The largest declared factor whose control reaches :data:`CONTROL_POWER` on every row."""
    for factor in FLOOR_FACTORS:
        if floor_control_power(factor)["power"].min() >= CONTROL_POWER:
            return factor
    return None


# ---------------------------------------------------------------------------- pilots

#: Every pilot set: ``(family, label, n, wald_only)``.  The twins' pilots fit Wald alone.
PILOT_SETS: tuple[tuple[str, str, int, bool], ...] = (
    *(
        ("policy_variance", label, 5_000, False)
        for label in OPTION_SETS
        if label not in REPORTING_SETS
    ),
    *(("policy_calibration", label, CALIBRATION_N, False) for label in CALIBRATION_SETS),
    *(("interval_calibration", label, TWIN_SETS[label][1], True) for label in GATED_TWINS),
    *(("primary", twin, study.SCENARIO_N[twin], True) for twin in study.TWINS),
)


def pilot_rows(
    family: str, label: str, replicate: int, n: int, wald_only: bool
) -> list[dict[str, Any]]:
    """One pilot draw on the ``diagnostic`` stream, with its seconds."""
    started = time.perf_counter()
    seed = pilot_seed(family, label, replicate)
    if family == "primary":
        config, names = study.TWINS[label]
        frame = config.sample(n, seed)
        truth = config.truth()
        with quiet():
            values = readings(config.fit(frame, DEFAULT), config.names)
        rows = [
            _row(
                "primary",
                f"{label}__{name}",
                "positive",
                replicate,
                n,
                PILOT_DRAWS,
                truth[name],
                values[name]["psi"],
                values[name]["se_ic"],
            )
            for name in names
        ]
    else:
        frame = draw_frame(family, label, n, seed)
        rows = fit_rows(family, label, replicate, n, PILOT_DRAWS, frame, wald_only=wald_only)
    seconds = time.perf_counter() - started
    for row in rows:
        row["seconds"] = seconds
        row["set"] = label
    return rows


def run_pilots(parts: Path, *, draws: int = PILOT_DRAWS, n_jobs: int = 1) -> pd.DataFrame:
    """Every pilot set's ``draws`` draws, resumable in parts."""
    frames = []
    for family, label, n, wald_only in PILOT_SETS:
        keys = [(family, label, r, n, wald_only) for r in range(draws)]
        frames.append(
            tools.run_in_parts(
                pilot_rows,
                keys,
                label=f"pilot__{family}__{label}",
                root=parts,
                chunk=20,
                n_jobs=n_jobs,
            )
        )
    return pd.concat(frames, ignore_index=True)


@functools.cache
def committed_pilot() -> pd.DataFrame:
    return pd.read_csv(PILOT_ROWS, float_precision="round_trip")


# ---------------------------------------------------------------------------- sizing


def _pilot_ratio(group: pd.DataFrame, seed: int) -> tuple[float, float]:
    """A pilot cell's SE ratio and the bootstrap SD of that ratio."""
    error = (group["estimate"] - group["truth"]).to_numpy(dtype=float)
    se = group["std_error"].to_numpy(dtype=float)
    point = float(se.mean() / error.std(ddof=1))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(error), size=(2_000, len(error)))
    draws = se[index].mean(axis=1) / error[index].std(axis=1, ddof=1)
    return point, float(draws.std(ddof=1))


def ratio_probability(
    point: float, sd: float, draws: int, replicates: int, low: float, high: float, *, side: str
) -> float:
    """P(the 99% SE-ratio interval at ``replicates`` meets a rule), predictive over the pilot.

    The ratio at ``replicates`` is normal about the pilot's with variance ``sd^2 (1 + P / R)``
    and its interval half-width ``2.576 sd sqrt(P / R)``.  ``side`` names the rule:
    ``"clause"`` (the lower end inside ``[low, high]``), ``"below"`` (the upper end below
    ``low``), ``"above"`` (the lower end above ``high``) or ``"band"`` (inside ``[low, high]``).
    """
    s_r = sd * np.sqrt(draws / replicates)
    spread = float(np.sqrt(sd**2 + s_r**2))
    half = 2.576 * s_r
    if side == "clause":
        return float(
            norm.cdf((high + half - point) / spread) - norm.cdf((low + half - point) / spread)
        )
    if side == "below":
        return float(norm.cdf((low - half - point) / spread))
    if side == "above":
        return float(norm.sf((high + half - point) / spread))
    if side == "band":
        return float(
            max(
                norm.cdf((high - half - point) / spread) - norm.cdf((low + half - point) / spread),
                0.0,
            )
        )
    raise ValueError(side)


def bias_probability(standardized: float, draws: int, replicates: int) -> float:
    """P(the bias interval lies within +-0.25 SD at ``replicates``), predictive over the pilot."""
    half = 2.576 / np.sqrt(replicates)
    spread = float(np.sqrt(1 / draws + 1 / replicates))
    margin = STUDY.margins.standardized_bias
    if half >= margin:
        return 0.0
    return float(
        norm.cdf((margin - half - standardized) / spread)
        - norm.cdf((-margin + half - standardized) / spread)
    )


def pilot_cells(pilot: pd.DataFrame) -> pd.DataFrame:
    """Each pilot cell's ratio, ratio SD, standardized bias, coverage and seconds."""
    out = []
    for (family, cell), group in pilot.groupby(["property", "cell"], sort=True):
        error = group["estimate"] - group["truth"]
        point, sd = _pilot_ratio(group, stream_seed(STUDY, "diagnostic", "pilot_ratio", cell))
        out.append(
            {
                "property": family,
                "cell": cell,
                "set": str(group["set"].iloc[0]),
                "role": str(group["role"].iloc[0]),
                "draws": len(group),
                "se_ratio": point,
                "se_ratio_sd": sd,
                "standardized_bias": float(error.mean() / error.std(ddof=1)),
                "covered": int(group["covered"].sum()),
                "seconds": float(group["seconds"].mean()),
            }
        )
    return pd.DataFrame(out)


def _covered_at(group: pd.DataFrame, factor: float) -> int:
    error = (group["estimate"] - group["truth"]).abs()
    return int((error <= CRITICAL * factor * group["std_error"]).sum())


def pass_table(
    pilot: pd.DataFrame, replicates: Mapping[str, int], primary_replicates: int
) -> pd.DataFrame:
    """Every gated row's and control's pass probability at the given replication counts."""
    cells = pilot_cells(pilot)
    margins = STUDY.margins
    out = []
    for row in cells.itertuples():
        group = pilot.loc[(pilot["property"] == row.property) & (pilot["cell"] == row.cell)]
        draws = int(row.draws)
        kind = kind_of(str(row.cell))
        family = str(row.property)
        label = str(row.set)
        r = primary_replicates if family == "primary" else replicates[label]

        def add(name: str, probability: float, role: str, family: str = family, r: int = r) -> None:
            out.append(
                {
                    "property": family,
                    "cell": name,
                    "role": role,
                    "replicates": r,
                    "probability": probability,
                }
            )

        bias = bias_probability(float(row.standardized_bias), draws, r)
        if family == "primary":
            floor = jeffreys_floor(int(row.covered), draws, r, margins.coverage_floor)
            add(str(row.cell), floor * bias, "positive")
            control = jeffreys_below(
                _covered_at(group, study.CONTROL_SE_FACTOR), draws, r, FLOOR_CONTROL_CEILING
            )
            add(f"{row.cell}__coverage_floor_control", control, "control")
        elif family == "interval_calibration":
            ratio = ratio_probability(
                float(row.se_ratio),
                float(row.se_ratio_sd),
                draws,
                r,
                *margins.calibration_se_ratio,
                side="band",
            )
            low, high = margins.calibration_coverage
            p = beta.rvs(
                row.covered + 0.5, draws - row.covered + 0.5, size=_JEFFREYS, random_state=1
            )
            coverage = float(
                (
                    binom.sf(min_covered(r, low) - 1, r, p) - binom.sf(max_covered(r, high), r, p)
                ).mean()
            )
            add(str(row.cell), ratio * coverage, "positive")
            for control, factor in (("shrunken_se_control", SHRUNKEN_SE_FACTOR),):
                add(
                    str(row.cell).replace("correctly_specified", control),
                    ratio_probability(
                        factor * float(row.se_ratio),
                        factor * float(row.se_ratio_sd),
                        draws,
                        r,
                        margins.calibration_se_ratio[0],
                        0.0,
                        side="below",
                    ),
                    "control",
                )
            noise = TWIN_NOISE_SD[label] / np.sqrt(int(group["n"].iloc[0]))
            spread = float((group["estimate"] - group["truth"]).std(ddof=1))
            shrink = spread / np.sqrt(spread**2 + noise**2)
            add(
                str(row.cell).replace("correctly_specified", "noise_control"),
                ratio_probability(
                    shrink * float(row.se_ratio),
                    shrink * float(row.se_ratio_sd),
                    draws,
                    r,
                    margins.calibration_se_ratio[0],
                    0.0,
                    side="below",
                ),
                "control",
            )
        elif kind == "option":
            clause = ratio_probability(
                float(row.se_ratio),
                float(row.se_ratio_sd),
                draws,
                r,
                CLAUSE_FLOOR,
                CLAUSE_CEILING,
                side="clause",
            )
            floor = jeffreys_floor(int(row.covered), draws, r, margins.coverage_floor)
            add(str(row.cell), clause * floor * bias, "positive")
            add(
                str(row.cell).replace("option", "coverage_floor_control"),
                jeffreys_below(
                    _covered_at(group, COVERAGE_FLOOR_FACTOR), draws, r, FLOOR_CONTROL_CEILING
                ),
                "control",
            )
            controls: list[tuple[str, float, str]] = []
            if label in WALD_CONTROL_SETS or family == "policy_calibration":
                controls.append(("inflated_se_control", INFLATED_SE_FACTOR, "above"))
            if family == "policy_calibration":
                controls.append(("shrunken_se_control", SHRUNKEN_SE_FACTOR, "below"))
            for control, factor, side in controls:
                add(
                    str(row.cell).replace("option", control),
                    ratio_probability(
                        factor * float(row.se_ratio),
                        factor * float(row.se_ratio_sd),
                        draws,
                        r,
                        CLAUSE_FLOOR,
                        CLAUSE_CEILING,
                        side=side,
                    ),
                    "control",
                )
            if family == "policy_calibration":
                noise = law_b.BOUNDS["p4"] / np.sqrt(CALIBRATION_N)
                spread = float((group["estimate"] - group["truth"]).std(ddof=1))
                shrink = spread / np.sqrt(spread**2 + noise**2)
                add(
                    str(row.cell).replace("option", "noise_control"),
                    ratio_probability(
                        shrink * float(row.se_ratio),
                        shrink * float(row.se_ratio_sd),
                        draws,
                        r,
                        CLAUSE_FLOOR,
                        CLAUSE_CEILING,
                        side="below",
                    ),
                    "control",
                )
        elif kind == "wald" and str(row.role) == "control":
            committed = wald_control_draws(label, str(row.cell))
            covered = int(row.covered) + committed[0]
            add(
                str(row.cell),
                jeffreys_below(covered, draws + committed[1], r, WALD_CONTROL_CEILING),
                "control",
            )
    return pd.DataFrame(out)


def wald_control_draws(label: str, cell: str) -> tuple[int, int]:
    """The committed probe-5 draws of a Wald control row: ``(covered, draws)``.

    The 1,000 draws of ``pricing_exact_d3`` are the committed ``mtp-pricing`` seeds; the Wald
    interval does not read the variance learner.
    """
    rows = probe5()
    est = "ltmle" if "ltmle" in label else "sdr"
    name = ey(DOWN) if "__ey__" in cell else ate(DOWN, study.P_REF)
    group = rows.loc[
        (rows.config == "pricing_exact_d3") & (rows.est == est) & (rows.estimand == name)
    ]
    covered = int(((group["psi"] - group["truth"]).abs() <= CRITICAL * group["se_ic"]).sum())
    return covered, len(group)


def family_wise(table: pd.DataFrame) -> float:
    """The product of the gated positive rows' pass probabilities."""
    return float(np.prod(table.loc[table["role"] == "positive", "probability"]))


# ---------------------------------------------------------------------------- budget


def budget_hours(cells: pd.DataFrame, seconds: Mapping[str, float]) -> dict[str, float]:
    """CPU-hours of each part of the run from per-fit seconds (``seconds[label]``)."""
    out = {label: REPLICATES[label] * seconds[label] / 3600.0 for label in REPLICATES}
    out["primary"] = study.PRIMARY_REPLICATES * sum(seconds[t] for t in study.TWINS) / 3600.0
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("what", choices=("pilots", "screen"))
    parser.add_argument("--parts", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.what == "pilots":
        rows = run_pilots(args.parts, n_jobs=args.jobs)
    else:
        rows = study.learner_screen(args.parts, n_jobs=args.jobs)
    rows.to_csv(args.output, index=False, compression="gzip", float_format="%.17g")
    print(f"wrote {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------- the sizing rule

#: The planned replication counts the sizing rule starts from.
PLANNED_REPLICATES: dict[str, int] = {
    **dict(REPLICATES),
    "pricing_ltmle_tau5": 2_600,
    "pricing_sdr_tau5": 2_600,
    "pricing_ltmle_tau2": 4_400,
    "pricing_sdr_tau2": 4_400,
    "boundaries_ltmle_p4": 1_000,
    "boundaries_sdr_p4": 1_000,
    "p4_ltmle": 2_800,
    "p4_sdr": 2_800,
    **dict.fromkeys(GATED_TWINS, 3_600),
}
PLANNED_PRIMARY = 1_600
R_STEP = 400
R_CAP = 40_000


def sized_replicates(pilot: pd.DataFrame) -> tuple[dict[str, int], int, pd.DataFrame]:
    """The declared sizing rule on the pilot: raise the weakest gated row by 400 until the
    family-wise pass probability reaches :data:`FAMILY_TARGET`.

    Returns the fit sets' counts, the primary count and the final pass table.  A row at
    :data:`R_CAP` stops the rule; the table then shows the shortfall.
    """
    replicates = dict(PLANNED_REPLICATES)
    primary = PLANNED_PRIMARY
    while True:
        table = pass_table(pilot, replicates, primary)
        if family_wise(table) >= FAMILY_TARGET:
            return replicates, primary, table
        positives = table.loc[table["role"] == "positive"].sort_values("probability")
        weakest = positives.iloc[0]
        if weakest["property"] == "primary":
            if primary >= R_CAP:
                return replicates, primary, table
            primary += R_STEP
        else:
            label = str(weakest["cell"]).split("__")[0]
            if replicates[label] >= R_CAP:
                return replicates, primary, table
            replicates[label] += R_STEP
