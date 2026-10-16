"""The parity check of ``vector-mtp``'s unsupplied pair, run before the declared run.

``write`` draws three samples of the ordered law at n = 2,000 and fits the study's primary fit
of ``longitudinal_vector_ordered_classifier``: one ``LTMLE`` fit per plan of
:data:`~tests.studies.canonical_vector_mtp.CLASSIFIER_PLANS`, each over ``(natural course,
plan)`` with :class:`~tests.studies.vector_mtp_common.OracleLogOdds` on that plan's log ratio.
It writes the natural columns, the shifted components, and the node ratio of every regimen of
every fit (``ratio__<fit>__<regimen>``).  It also writes :func:`grid`, the points at which the R
transcription of :func:`~tests.studies.vector_mtp_common.ordered_ratio` is checked.

``tests/canonical/vector_mtp_parity.R`` fits ``lmtp``'s own ratios on the same rows through the
vector adapter, with ``mtp = TRUE`` and ``SL.logratio.<plan>``, and evaluates the R ratio on the
grid.  ``compare`` reads both and applies three checks.

1. **The R transcription.**  Every grid ratio agrees with Python at ``rtol = 1e-12`` (absolute
   floor :data:`RATIO_FLOOR_ATOL`), and every log-ratio feature at ``1e-12``.
2. **The nonzero witness.**  On each moved plan the fraction of rows with a positive ``lmtp``
   ratio equals the package's.  Under ``mtp = TRUE`` both fractions are 1: the package's
   floored log-ratio classifier never predicts 0, and ``lmtp`` bounds its predictions away from
   0.  The fraction is therefore not the share of rows with a positive true ratio.  With
   ``mtp = FALSE`` ``lmtp`` multiplies its ratio by the indicator that no component moved, so
   its fraction falls to the share of unmoved rows and the witness fails.
3. **The ratios.**  The reading takes one of three outcomes, declared before the run:

   ===================================================  ======================================
   largest gap                                          outcome
   ===================================================  ======================================
   every absolute gap below ``SMOKE_GATE`` (``1e-6``)   ``exact``: the scenario becomes an
                                                        exactness witness by a declaration
                                                        amendment
   every relative gap below :data:`PARITY_TOLERANCE`,   ``margin-read`` only after the cause is
   and a diagnosed, named numerical cause               diagnosed, named and recorded in the
                                                        amendment
   anything else, or no diagnosed cause                 ``blocked``: the declaration waits
                                                        until the cause is found
   ===================================================  ======================================

``compare`` prints the outcome the numbers allow and exits nonzero unless it is ``exact``.  It
cannot diagnose a cause, so it never prints ``margin-read`` as settled.

Usage::

    python -m tests.canonical.vector_mtp_parity write DIR
    docker run ... Rscript /fixture/vector_mtp_parity.R DIR
    python -m tests.canonical.vector_mtp_parity compare DIR
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tests.studies import canonical_vector_mtp as study
from tests.studies import vector_mtp_common as common
from tests.studies.evidence.seeds import stream_seed

DRAWS = 3
N = 2_000
#: The relative ratio gap below which a diagnosed, named numerical cause may make the pair
#: margin-read instead of exact.
PARITY_TOLERANCE = 1e-4
#: The relative tolerance of the R transcription of the ratio and its absolute floor.
TRANSCRIPTION_RTOL = 1e-12
RATIO_FLOOR_ATOL = 1e-15
#: Each grid axis holds the values on both sides of every break of the ordered plans: the
#: knee 3 and its image 3.75, the scale's hold 4.4 and cap 5.5, and the shift's hold 5.0.
GRID_D1 = (
    *(0.0, 0.5, 2.39, 2.4, 2.99, 3.0, 3.01, 3.74, 3.75, 3.76),
    *(4.39, 4.4, 4.41, 5.49, 5.5, 5.51, 6.0),
)
GRID_D2 = (0.0, 0.49, 0.5, 0.51, 2.0, 4.99, 5.0, 5.01, 5.49, 5.5, 5.51, 6.0)
GRID_W1 = (-1.0, -0.3, 0.7)


def grid() -> pd.DataFrame:
    """Every ``(plan, F, D1, D2, W1, W2)`` grid point with Python's ratio and feature."""
    rows = list(itertools.product(study.CLASSIFIER_PLANS, (0.0, 1.0), GRID_D1, GRID_D2))
    points = [
        (plan, f, d1, d2, w1, w2)
        for (plan, f, d1, d2), w1, w2 in itertools.product(rows, GRID_W1, (0.0, 1.0))
    ]
    frame = pd.DataFrame(points, columns=["plan", "F", "D1", "D2", "W1", "W2"])
    ratio = np.empty(len(frame))
    for plan in study.CLASSIFIER_PLANS:
        mask = (frame["plan"] == plan).to_numpy()
        part = frame.loc[mask]
        ratio[mask] = common.ordered_ratio(
            plan, *(part[c].to_numpy(dtype=float) for c in ("F", "D1", "D2", "W1", "W2"))
        )
    frame["ratio"] = ratio
    frame["log_ratio"] = np.log(np.clip(ratio, common.RATIO_FLOOR, None))
    return frame


def write(directory: Path) -> None:
    scenario = study.ORDERED_CLASSIFIER
    frames = []
    for draw in range(DRAWS):
        frame = common.sample_ordered(N, stream_seed(study.STUDY, "parity", draw))
        result = study.fit(frame, scenario)
        sample = frame.copy()
        for name, values in study.pairing_columns(result, scenario).items():
            if name.startswith("shift__"):
                sample[name] = values
        for plan, fitted in result.by_plan.items():
            for label in (study.REFERENCE, plan):
                ratio = fitted.fits[label].node_ratio[:, 0]
                sample[f"ratio__{study.slug(plan)}__{study.slug(label)}"] = ratio
        sample.insert(0, "row", np.arange(N))
        sample.insert(0, "draw", draw)
        frames.append(sample)
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(directory / "draws.csv", index=False)
    grid().to_csv(directory / "grid.csv", index=False)


def _transcription(directory: Path) -> bool:
    python = pd.read_csv(directory / "grid.csv", float_precision="round_trip")
    r = pd.read_csv(directory / "grid_lmtp.csv", float_precision="round_trip")
    if len(python) != len(r):
        print(f"the R grid has {len(r)} rows, the Python grid {len(python)}")
        return False
    ratio = np.isclose(r["ratio"], python["ratio"], rtol=TRANSCRIPTION_RTOL, atol=RATIO_FLOOR_ATOL)
    feature = np.isclose(r["log_ratio"], python["log_ratio"], rtol=0.0, atol=TRANSCRIPTION_RTOL)
    gap = float(np.max(np.abs(r["ratio"] - python["ratio"])))
    print(
        f"R transcription: {int((~ratio).sum())} ratio and {int((~feature).sum())} feature "
        f"mismatches over {len(python)} grid points (largest ratio gap {gap:.3g})"
    )
    return bool(ratio.all() and feature.all())


def compare(directory: Path) -> str:
    """The outcome the numbers allow: ``exact`` or ``blocked``."""
    draws = pd.read_csv(directory / "draws.csv", float_precision="round_trip")
    lmtp = pd.read_csv(directory / "lmtp.csv", float_precision="round_trip")
    transcribed = _transcription(directory)
    witnessed = True
    worst_absolute = worst_relative = 0.0
    for plan in study.CLASSIFIER_PLANS:
        for label in (study.REFERENCE, plan):
            column = f"ratio__{study.slug(plan)}__{study.slug(label)}"
            theirs = lmtp.loc[
                (lmtp["fit"] == study.slug(plan)) & (lmtp["plan"] == study.slug(label)),
                ["draw", "row", "lmtp_ratio"],
            ]
            joined = draws[["draw", "row", column]].merge(
                theirs, on=["draw", "row"], validate="one_to_one"
            )
            mine, other = joined[column].to_numpy(), joined["lmtp_ratio"].to_numpy()
            absolute = float(np.max(np.abs(mine - other)))
            relative = float(np.max(np.abs(mine - other) / np.abs(mine)))
            print(
                f"fit {plan}, regimen {label}: largest gap {absolute:.3g} (relative {relative:.3g})"
            )
            worst_absolute = max(worst_absolute, absolute)
            worst_relative = max(worst_relative, relative)
            if label != study.REFERENCE:
                positive = (float(np.mean(mine > 0.0)), float(np.mean(other > 0.0)))
                print(
                    f"  positive-ratio fraction: package {positive[0]:.4f}, lmtp {positive[1]:.4f}"
                )
                witnessed &= positive[0] == positive[1]
    if not (transcribed and witnessed):
        outcome = "blocked"
        print(
            "blocked: the R transcription or the nonzero witness failed; diagnose before declaring"
        )
    elif worst_absolute < study.SMOKE_GATE:
        outcome = "exact"
        print(f"exact: every gap is below {study.SMOKE_GATE:g}; amend the declaration")
    elif worst_relative < PARITY_TOLERANCE:
        outcome = "blocked"
        print(
            f"blocked: gaps are below {PARITY_TOLERANCE:g} relative but above {study.SMOKE_GATE:g}. "
            "The pair is margin-read only after a diagnosed, named numerical cause is recorded "
            "in the declaration amendment"
        )
    else:
        outcome = "blocked"
        print("blocked: a gap exceeds the declared relative tolerance; diagnose before declaring")
    return outcome


if __name__ == "__main__":
    command, where = sys.argv[1], Path(sys.argv[2])
    if command == "write":
        write(where)
    elif command == "compare":
        raise SystemExit(0 if compare(where) == "exact" else 1)
    else:
        raise SystemExit("usage: python -m tests.canonical.vector_mtp_parity write|compare DIR")
