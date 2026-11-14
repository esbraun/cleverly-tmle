r"""The declared readings of ``continuous-survival-variance``: candidates, ship rule, floors.

Every reading is computed by code from the committed primary rows, which carry, for each
target of each fit, the reported (switched) standard error and the influence-curve (IC),
predictable-variation (PV), no-square PV and oracle standard errors with the IC's
``variance_rel_sd`` (:mod:`tests.studies.continuous_survival_variance`).  Each candidate's
interval is the shipped one at its standard error: range-logit on (0, 1) for a risk, identity for
a difference (:func:`shipped_interval`).  All rules and controls were declared before any run (X6
plan rev 5, section 5.1).

**Candidates** (:data:`CANDIDATES`).  ``ic``; ``s``, the default; reported only: ``max``
(max(IC, PV) everywhere), ``pv``, ``pv_switched`` (PV where the relative SD exceeds 0.125),
``oracle`` and ``pv_nosq`` (T-a2's mutation); for each PV candidate the width-matched
``ic_k_<candidate>``, the IC standard error times :math:`k_{\mathrm{row}}`, that candidate's mean
standard error over the IC's in the row.  Controls: ``s_pv115`` and ``s_pv130`` (S with the PV
variance times 1.15 and 1.3), ``s_never`` (a switch that never fires: the IC), ``s_shrunk`` (S's
standard error times 0.85).

**Ship rule** (:func:`ship_rule`).  S ships iff (1) *harm*: at no row of any cell does S's 99%
SE-ratio interval lie wholly above 1.07, or its coverage interval wholly above 0.98, while the
IC's does not; and (2) *benefit*: at one first-quartile row at least, the paired 99% bootstrap
interval of :math:`|\mathrm{cov}_{IC} - 0.95| - |\mathrm{cov}_S - 0.95|` lies above 0.  Stated
power at 3,600 per cell: clause 1 passes with 0.997 or more when every true S row is inside the
band, fires with 0.04 at a true SE ratio of 1.08, 0.40 at 1.10 and 0.89 at 1.12; clause 2 passes
with 1.000.  Controls on the same fits: ``s_pv115`` fires clause 1 with 0.908 in one base cell
(reported), ``s_pv130`` fires it with 1.000 and ``s_never`` fails clause 2 with certainty (each an
instrument check: otherwise the run is void), and ``s_shrunk`` fails the calibration rule at every
gated row (power 1.000).

**Floors** (:func:`floors`, :func:`floor_values`).  Read from the default's rows: S's if the rule
ships S, the IC's otherwise.  ``r_prim`` is the largest median ``variance_rel_sd`` rung over the
gated rows of all five cells such that every primary row at or below it passes
(:func:`rel_sd_floor`); the share floor falls from 0.40 to 0.25 only if every first-quartile row
passes the calibration rule, and then ``r_cal`` is read the same way from those rows; ``MEASURED_N``
falls to 500 only if every primary row of ``v3_half_n`` passes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

from tests.studies import continuous_survival_variance as study
from tests.studies import continuous_survival_variance_law as laws
from tests.studies.evidence.inference import (
    bootstrap,
    clopper_pearson,
    percentile_interval,
    standardized_bias_verdict,
)

MARGINS = study.MARGINS
Z = float(norm.ppf(1.0 - MARGINS.alpha / 2.0))
THRESHOLD = 0.125
#: The PV candidates whose width-matched IC control is read.
PV_CANDIDATES = ("s", "max", "pv", "pv_switched")
CONTROLS = ("s_pv115", "s_pv130", "s_never", "s_shrunk")
CANDIDATES = (
    "ic",
    "s",
    "max",
    "pv",
    "pv_switched",
    "oracle",
    "pv_nosq",
    *(f"ic_k_{name}" for name in PV_CANDIDATES),
    *CONTROLS,
)
SHRUNKEN_SE_FACTOR = 0.85
#: The bootstrap draws of every interval: the record's own count.
BOOTSTRAP = MARGINS.bootstrap_replicates
#: Rows per bootstrap batch: 20 resampled columns of 7,200 draws stay near 300 MB.
BATCH = 250


def shipped_interval(psi: np.ndarray, se: np.ndarray, level: np.ndarray) -> np.ndarray:
    """``(n, 2)`` the shipped Wald interval: range-logit on (0, 1) for a level, else identity.

    A level on or outside (0, 1) keeps the identity interval, as
    :class:`~cleverly.inference.ParameterEstimate` does.
    """
    psi = np.asarray(psi, dtype=float)
    se = np.asarray(se, dtype=float)
    out = np.column_stack([psi - Z * se, psi + Z * se])
    inside = np.asarray(level, dtype=bool) & (psi > 0.0) & (psi < 1.0)
    if inside.any():
        p = psi[inside]
        half = Z * se[inside] / (p * (1.0 - p))
        out[inside, 0] = expit(logit(p) - half)
        out[inside, 1] = expit(logit(p) + half)
    return out


def long_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per cell, replicate and target, the first-quartile targets included."""
    fields = ["estimate", "std_error", *study.VARIANCE_FIELDS]
    keep = ["scenario", "replicate", "estimand", "truth", *fields]
    gated = rows.loc[:, keep].assign(first_quartile=False)
    fits = rows.drop_duplicates(["scenario", "replicate"])
    parts = [gated]
    for scenario, group in fits.groupby("scenario", sort=True):
        law = study.CELLS[str(scenario)][0]
        truth = laws.truth(law)
        for index, name in enumerate(laws.first_quartile(law)):
            frame = pd.DataFrame(
                {
                    "scenario": scenario,
                    "replicate": group["replicate"].to_numpy(),
                    "estimand": name,
                    "truth": truth[name],
                    **{
                        field: group[f"early{index}_{field}"].to_numpy(dtype=float)
                        for field in fields
                    },
                }
            )
            parts.append(frame.assign(first_quartile=True))
    out = pd.concat(parts, ignore_index=True)
    out["level"] = out["estimand"].str.startswith("cif_regimen")
    return out.sort_values(["scenario", "estimand", "replicate"], ignore_index=True)


def candidate_errors(group: pd.DataFrame) -> dict[str, np.ndarray]:
    """Each candidate's standard error on one cell and target's fits."""
    ic = group["ic_std_error"].to_numpy(dtype=float)
    pv = group["pv_std_error"].to_numpy(dtype=float)
    sparse = group["variance_rel_sd"].to_numpy(dtype=float) > THRESHOLD
    s = group["std_error"].to_numpy(dtype=float)
    out = {
        "ic": ic,
        "s": s,
        "max": np.maximum(ic, pv),
        "pv": pv,
        "pv_switched": np.where(sparse, pv, ic),
        "oracle": group["oracle_std_error"].to_numpy(dtype=float),
        "pv_nosq": group["pv_nosq_std_error"].to_numpy(dtype=float),
        "s_pv115": np.where(sparse, np.maximum(ic, np.sqrt(1.15) * pv), ic),
        "s_pv130": np.where(sparse, np.maximum(ic, np.sqrt(1.3) * pv), ic),
        "s_never": ic,
        "s_shrunk": SHRUNKEN_SE_FACTOR * s,
    }
    for name in PV_CANDIDATES:
        out[f"ic_k_{name}"] = ic * (out[name].mean() / ic.mean())
    return out


def _seed(scenario: str, estimand: str) -> int:
    cell = list(study.CELLS).index(scenario)
    law = study.CELLS[scenario][0]
    names = list(laws.truth(law))
    sequence = np.random.SeedSequence([study.RESAMPLING_SEED, cell, names.index(estimand)])
    return int(sequence.generate_state(1)[0])


def candidate_readings(long: pd.DataFrame, *, replicates: int = BOOTSTRAP) -> pd.DataFrame:
    """Per cell, target and candidate: bias, SE ratio, coverage and the paired differences.

    ``long`` is :func:`long_rows` of the primary rows, the committed ``variance-rows.csv.gz``.
    """
    records: list[dict[str, Any]] = []
    for (scenario, estimand), group in long.groupby(["scenario", "estimand"]):
        estimate = group["estimate"].to_numpy(dtype=float)
        truth = float(group["truth"].iloc[0])
        level = group["level"].to_numpy(dtype=bool)
        errors = candidate_errors(group)
        covered = {}
        for name, se in errors.items():
            interval = shipped_interval(estimate, se, level)
            covered[name] = ((interval[:, 0] <= truth) & (truth <= interval[:, 1])).astype(float)
        arrays = {"estimate": estimate, **{f"se_{k}": v for k, v in errors.items()}}
        arrays.update({f"cov_{k}": v for k, v in covered.items()})
        statistics: dict[str, Any] = {}
        for name in CANDIDATES:
            statistics[f"ratio_{name}"] = lambda d, k=name: (
                d[f"se_{k}"].mean(axis=1) / d["estimate"].std(axis=1, ddof=1)
            )
            statistics[f"benefit_{name}"] = lambda d, k=name: (
                np.abs(d["cov_ic"].mean(axis=1) - 0.95) - np.abs(d[f"cov_{k}"].mean(axis=1) - 0.95)
            )
        for name in PV_CANDIDATES:
            statistics[f"matched_{name}"] = lambda d, k=name: (
                d[f"cov_{k}"].mean(axis=1) - d[f"cov_ic_k_{k}"].mean(axis=1)
            )
        draws = bootstrap(
            arrays,
            statistics,
            replicates=replicates,
            seed=_seed(str(scenario), str(estimand)),
            batch_size=BATCH,
        )
        spread = float(np.std(estimate, ddof=1))
        bias = standardized_bias_verdict(
            estimate - truth,
            margin=MARGINS.standardized_bias,
            confidence_level=MARGINS.confidence_level,
        )
        level_ = MARGINS.confidence_level
        for name in CANDIDATES:
            hits = int(covered[name].sum())
            coverage = clopper_pearson(hits, len(estimate), confidence_level=level_)
            ratio = percentile_interval(draws[f"ratio_{name}"], confidence_level=level_)
            benefit = percentile_interval(draws[f"benefit_{name}"], confidence_level=level_)
            record = {
                "scenario": scenario,
                "estimand": estimand,
                "first_quartile": bool(group["first_quartile"].iloc[0]),
                "candidate": name,
                "fits": len(estimate),
                "truth": truth,
                "bias": bias.bias,
                "standardized_bias": bias.standardized,
                "bias_equivalent": bool(bias.equivalent),
                "empirical_se": spread,
                "mean_std_error": float(errors[name].mean()),
                "se_ratio": float(errors[name].mean() / spread),
                "se_ratio_low": ratio.low,
                "se_ratio_high": ratio.high,
                "coverage": hits / len(estimate),
                "coverage_low": coverage.low,
                "coverage_high": coverage.high,
                "k_row": float(errors[name].mean() / errors["ic"].mean()),
                "coverage_gain": float(covered[name].mean() - covered["ic"].mean()),
                "benefit": float(
                    abs(covered["ic"].mean() - 0.95) - abs(covered[name].mean() - 0.95)
                ),
                "benefit_low": benefit.low,
                "benefit_high": benefit.high,
                "median_rel_sd": float(np.median(group["variance_rel_sd"])),
                "switched_share": float(np.mean(group["variance_rel_sd"] > THRESHOLD)),
            }
            if name in PV_CANDIDATES:
                matched = percentile_interval(draws[f"matched_{name}"], confidence_level=level_)
                record.update(
                    matched_gain=float(covered[name].mean() - covered[f"ic_k_{name}"].mean()),
                    matched_low=matched.low,
                    matched_high=matched.high,
                )
            records.append(record)
    return pd.DataFrame.from_records(records)


# ----------------------------------------------------------------------- row rules


def primary_pass(frame: pd.DataFrame) -> pd.Series:
    """The registered study's primary rule on each candidate row."""
    low, high = MARGINS.se_ratio_sanity
    return (
        (frame["coverage_low"] > MARGINS.coverage_floor)
        & (frame["coverage_high"] < MARGINS.over_coverage_ceiling)
        & (frame["se_ratio_low"] >= low)
        & (frame["se_ratio_high"] <= high)
        & frame["bias_equivalent"].astype(bool)
    )


def calibration_pass(frame: pd.DataFrame) -> pd.Series:
    """The calibration rule: SE ratio's interval in (0.93, 1.07), coverage's in (0.92, 0.98)."""
    low, high = MARGINS.calibration_se_ratio
    return (
        (frame["se_ratio_low"] >= low)
        & (frame["se_ratio_high"] <= high)
        & (frame["coverage_low"] >= 0.92)
        & (frame["coverage_high"] <= 0.98)
    )


def _above_band(frame: pd.DataFrame) -> pd.Series:
    """The harm clause's reading of one row: demonstrably above the band."""
    return (frame["se_ratio_low"] > MARGINS.calibration_se_ratio[1]) | (
        frame["coverage_low"] > 0.98
    )


def ship_rule(candidates: pd.DataFrame) -> pd.DataFrame:
    """The two clauses for S and each control, with the declared expectation of each."""
    keys = ["scenario", "estimand"]
    ic = candidates.loc[candidates["candidate"] == "ic"].set_index(keys)
    records = []
    expected = {
        "s": "declared rule: ships iff both clauses pass",
        "s_pv115": "reported: clause 1 fires with 0.908 in one base cell",
        "s_pv130": "instrument: clause 1 must fire",
        "s_never": "instrument: clause 2 must fail",
    }
    for name, expectation in expected.items():
        rows = candidates.loc[candidates["candidate"] == name].set_index(keys)
        fires = _above_band(rows) & ~_above_band(ic.loc[rows.index])
        early = rows.loc[rows["first_quartile"].astype(bool)]
        benefit = early.index[early["benefit_low"] > 0.0]
        harm_pass = not bool(fires.any())
        benefit_pass = len(benefit) > 0
        if name == "s_pv130":
            instrument = not harm_pass
        elif name == "s_never":
            instrument = not benefit_pass
        else:
            instrument = True
        records.append(
            {
                "candidate": name,
                "expectation": expectation,
                "harm_rows": int(fires.sum()),
                "harm_where": ";".join(f"{c}/{e}" for c, e in fires.index[fires.to_numpy()]),
                "harm_pass": harm_pass,
                "benefit_rows": len(benefit),
                "benefit_where": ";".join(f"{c}/{e}" for c, e in benefit),
                "benefit_pass": benefit_pass,
                "ships": harm_pass and benefit_pass,
                "instrument_ok": instrument,
            }
        )
    shrunk = candidates.loc[
        (candidates["candidate"] == "s_shrunk") & ~candidates["first_quartile"].astype(bool)
    ]
    red = ~calibration_pass(shrunk)
    records.append(
        {
            "candidate": "s_shrunk",
            "expectation": "instrument: every gated row fails the calibration rule",
            "harm_rows": int(red.sum()),
            "harm_where": "",
            "harm_pass": False,
            "benefit_rows": 0,
            "benefit_where": "",
            "benefit_pass": False,
            "ships": False,
            "instrument_ok": bool(red.all()),
        }
    )
    return pd.DataFrame.from_records(records)


def ships(rule: pd.DataFrame) -> bool:
    """Whether the committed ship rule ships S."""
    return bool(rule.set_index("candidate").loc["s", "ships"])


# ----------------------------------------------------------------------------- floors


def rel_sd_floor(rungs: Sequence[float], passed: Sequence[bool]) -> float:
    """The largest rung such that every row at or below it passes; ``0.0`` if the lowest fails.

    The cap is monotone: a red row caps the floor below its rung even when a higher rung passes,
    so no rung above a red one reads supported.
    """
    order = np.argsort(np.asarray(rungs, dtype=float), kind="stable")
    floor = 0.0
    for index in order:
        if not passed[index]:
            break
        floor = float(rungs[index])
    return floor


def floors(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per gated and first-quartile row of the default: its rung, its rule and whether it passes."""
    default = "s" if ships(ship_rule(candidates)) else "ic"
    frame = candidates.loc[candidates["candidate"] == default].copy()
    early = frame["first_quartile"].astype(bool)
    frame["rule"] = np.where(early, "calibration", "primary")
    frame["rule_passed"] = np.where(early, calibration_pass(frame), primary_pass(frame))
    frame["default"] = default
    return frame.loc[
        :, ["scenario", "estimand", "rule", "median_rel_sd", "rule_passed", "default"]
    ].reset_index(drop=True)


def floor_values(table: pd.DataFrame) -> Mapping[str, Any]:
    """The three floors from ``floors.csv``: ``r_prim``, the share floor, ``r_cal``, ``n``."""
    primary = table.loc[table["rule"] == "primary"]
    early = table.loc[table["rule"] == "calibration"]
    share = 0.25 if bool(early["rule_passed"].all()) else 0.40
    v3 = primary.loc[primary["scenario"] == "v3_half_n"]
    return {
        "default": str(table["default"].iloc[0]),
        "r_prim": rel_sd_floor(
            primary["median_rel_sd"].tolist(), primary["rule_passed"].astype(bool).tolist()
        ),
        "share_floor": share,
        "r_cal": rel_sd_floor(
            early["median_rel_sd"].tolist(), early["rule_passed"].astype(bool).tolist()
        )
        if share == 0.25
        else None,
        "measured_n": 500 if bool(v3["rule_passed"].all()) else 1_000,
    }
