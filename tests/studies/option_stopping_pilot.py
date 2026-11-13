"""The pre-read of the two stopping-rule studies (Fix-8 plan section 5.0), and its summary.

Before the declaration, every gated calibration cell of ``option-stopping-point`` and
``option-stopping-ltmle`` is read on :data:`DRAWS` pilot draws at the declared n and learners,
with the necessity cells and the residual witnesses beside them.  The draws come from
:data:`PILOT_SEED`, a stream no study draws from.  Each draw is fitted by
:mod:`tests.studies._option_stopping_laws` exactly as the property phase fits it.

The committed summary :data:`SUMMARY` holds one row per pilot cell: the draws, the coverage of
the widened interval (``ci``) and of the plain interval (the same estimate with
``rule_widened=False``), the SD of the estimates, the SE ratio (mean reported SE over that SD),
the standardized bias, the mean absolute residual in standard errors, the share of draws whose
residual is at least :data:`WITNESS_FLOOR` standard errors, and the share of fits that met their
rule.  The design
tests read it (``tests/unit/test_option_stopping_point_design.py``,
``tests/unit/test_option_stopping_ltmle_design.py``).

Run ``python -m tests.studies.option_stopping_pilot run --start S --draws D --out FILE`` once
per block of draws, and ``python -m tests.studies.option_stopping_pilot summarize FILE...`` to
write the summary.  The per-draw rows are not committed.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.studies import _option_stopping_laws as laws

#: The pilot draws per cell (plan 5.0: at least 2,000).
DRAWS = 2_000
#: The pilot's own seed stream.
PILOT_SEED = 20261112
#: The residual a draw must reach to count for the witness share, in standard errors.
WITNESS_FLOOR = 0.01
CRITICAL = float(norm.ppf(0.975))
SUMMARY = Path(__file__).with_name("option_stopping_pilot.csv")

#: The point configurations fitted on each L1 draw: ``label -> (stopping, targeting)``.
POINT_CONFIGURATIONS: dict[str, tuple[str, str]] = {
    "rel_newton": ("relative", "iterative"),
    **{
        f"{laws.RULE_LABELS[rule]}_{laws.TARGETING_LABELS[targeting]}": (rule, targeting)
        for rule in laws.RULES
        for targeting in laws.TARGETINGS
    },
}


def pilot_seed(law: str, draw: int) -> int:
    """The seed of one pilot draw of one law."""
    key = sum(ord(character) * 31**index for index, character in enumerate(law)) % 2**31
    sequence = np.random.SeedSequence(entropy=PILOT_SEED, spawn_key=(key, draw))
    return int(sequence.generate_state(1)[0])


def _record(
    study: str,
    law: str,
    cell: str,
    parameter: str,
    draw: int,
    estimate: Any,
    truth: float,
    rule_met: bool,
) -> dict[str, Any]:
    low, high = estimate.ci
    plain_low, plain_high = replace(estimate, rule_widened=False).ci
    se = float(estimate.std_error)
    return {
        "study": study,
        "law": law,
        "cell": cell,
        "parameter": parameter,
        "draw": draw,
        "truth": float(truth),
        "estimate": float(estimate.psi),
        "std_error": se,
        "residual": float(estimate.score) / se,
        "covered": int(low <= truth <= high),
        "plain_covered": int(plain_low <= truth <= plain_high),
        "rule_met": int(rule_met),
    }


def point_draw(draw: int) -> list[dict[str, Any]]:
    """Every point pilot record of one draw index."""
    rows: list[dict[str, Any]] = []
    for outcome in ("binary", "continuous"):
        frame, truth = laws.draw_l1(outcome, laws.N, pilot_seed(outcome, draw))
        for label, (rule, targeting) in POINT_CONFIGURATIONS.items():
            result = laws.fit_point(outcome, frame, stopping=rule, targeting=targeting)
            met = laws.point_rule_met(result)
            for parameter in ("ate", "ey1"):
                rows.append(
                    _record(
                        "point",
                        outcome,
                        label,
                        parameter,
                        draw,
                        result[parameter],
                        truth[parameter],
                        met,
                    )
                )
    frame, truth = laws.draw_ln(laws.N, pilot_seed("LN", draw))
    for scale in (1.0, laws.NECESSITY_SCALE):
        result = laws.fit_point(
            "LN",
            frame,
            stopping="scaled_var",
            targeting="one_step",
            estimands=("ey1",),
            scale=scale,
        )
        rows.append(
            _record(
                "point",
                "LN",
                f"sv_one_step_K{scale:g}",
                "ey1",
                draw,
                result["ey1"],
                truth["ey1"],
                laws.point_rule_met(result),
            )
        )
    return rows


def longitudinal_draw(draw: int) -> list[dict[str, Any]]:
    """Every longitudinal pilot record of one draw index."""
    rows: list[dict[str, Any]] = []
    frame, truth = laws.draw_l3(laws.N, pilot_seed("L3", draw))
    for rule in ("relative", *laws.RULES):
        result = laws.fit_l3(frame, stopping=rule)
        rows.append(
            _record(
                "ltmle",
                "L3",
                f"{laws.RULE_LABELS[rule]}_newton",
                laws.L3_CONTRAST,
                draw,
                result[laws.L3_CONTRAST],
                truth[laws.L3_CONTRAST],
                laws.longitudinal_rule_met(result),
            )
        )
    frame, truth = laws.draw_l4(laws.N, pilot_seed("L4", draw))
    for rule in ("relative", *laws.RULES):
        result = laws.fit_l4(frame, stopping=rule)
        met = laws.longitudinal_rule_met(result)
        for parameter in (laws.L4_CONTRAST, laws.L4_RISK):
            rows.append(
                _record(
                    "ltmle",
                    "L4",
                    f"{laws.RULE_LABELS[rule]}_one_step",
                    parameter,
                    draw,
                    result[parameter],
                    truth[parameter],
                    met,
                )
            )
    for scale in (1.0, laws.NECESSITY_SCALE):
        result = laws.fit_l4(
            frame, stopping="scaled_var", configuration="mechanism_correct", scale=scale
        )
        rows.append(
            _record(
                "ltmle",
                "L4",
                f"mechanism_correct_sv_one_step_K{scale:g}",
                laws.L4_RISK,
                draw,
                result[laws.L4_RISK],
                truth[laws.L4_RISK],
                laws.longitudinal_rule_met(result),
            )
        )
    return rows


def run(start: int, draws: int, out: Path) -> None:
    """Fit pilot draws ``start`` to ``start + draws - 1`` and write their records to ``out``."""
    rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for draw in range(start, start + draws):
            rows += point_draw(draw)
            rows += longitudinal_draw(draw)
            if (draw - start) % 100 == 99:
                pd.DataFrame(rows).to_parquet(out)
    pd.DataFrame(rows).to_parquet(out)


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    """One summary row per pilot cell and parameter."""
    out = []
    keys = ["study", "law", "cell", "parameter"]
    for key, block in records.groupby(keys, sort=True):
        sd = float(block["estimate"].std(ddof=1))
        residual = block["residual"].abs()
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "draws": len(block),
                "coverage": float(block["covered"].mean()),
                "plain_coverage": float(block["plain_covered"].mean()),
                "empirical_sd": sd,
                "se_ratio": float(block["std_error"].mean()) / sd,
                "standardized_bias": float((block["estimate"] - block["truth"]).mean()) / sd,
                "mean_abs_residual": float(residual.mean()),
                "mean_residual": float(block["residual"].mean()),
                "residual_share": float((residual >= WITNESS_FLOOR).mean()),
                "rule_met_share": float(block["rule_met"].mean()),
            }
        )
    return pd.DataFrame(out)


def read_summary() -> pd.DataFrame:
    """The committed summary."""
    return pd.read_csv(SUMMARY, float_precision="round_trip")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--start", type=int, required=True)
    runner.add_argument("--draws", type=int, required=True)
    runner.add_argument("--out", type=Path, required=True)
    summary = commands.add_parser("summarize")
    summary.add_argument("files", type=Path, nargs="+")
    arguments = parser.parse_args(argv)
    if arguments.command == "run":
        run(arguments.start, arguments.draws, arguments.out)
        return
    records = pd.concat([pd.read_parquet(path) for path in arguments.files], ignore_index=True)
    records = records.drop_duplicates(["study", "law", "cell", "parameter", "draw"])
    summarize(records).to_csv(SUMMARY, index=False)


if __name__ == "__main__":
    main()
