r"""The tail sizes of the censored pricing law's own curves (``mtp-pricing-censored``).

``law_pc_moments.csv``
    For each fit of
    :data:`~tests.studies.mtp_pricing_censored_own_curve.LAW_PC_FITS` and each mean and
    contrast the study reads, the SD, skewness and kurtosis of the curve on :math:`10^8` fresh
    draws of law P-C with the limit nuisances of a :math:`10^7`-row fit, the 1% and 99%
    chunk-bootstrap quantiles of the kurtosis, and the tail size declared at the 99% bound
    (:func:`~tests.studies.mtp_pricing_censored_own_curve.law_pc_moments`).  About two
    CPU-hours.  ``--checkpoint`` keeps each fit and each chunk, so a stopped run resumes, and
    ``--worker K/N`` computes every N-th chunk from the K-th into the checkpoint, so N processes
    share the draws once the fits are kept.

``tests/unit/test_mtp_pricing_censored_design.py`` reads the file and derives the study's
tail-size classification from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tests.studies import mtp_pricing_censored_own_curve as own

HERE = Path(__file__).resolve().parent


def law_pc(checkpoint: Path | None) -> pd.DataFrame:
    rows = []
    for (learners, estimator, quantity), reading in own.law_pc_moments(
        checkpoint=checkpoint
    ).items():
        rows.append(
            {
                "learners": learners,
                "estimator": estimator,
                "quantity": quantity,
                "draws": reading.draws,
                "sd": reading.sd,
                "skewness": reading.skewness,
                "kappa": reading.kappa,
                "kappa_lower": reading.kappa_lower,
                "kappa_upper": reading.kappa_upper,
                "n_kappa": reading.n_kappa,
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--fits-only", action="store_true")
    parser.add_argument("--worker", default=None, help="K/N: every N-th chunk from the K-th")
    args = parser.parse_args(argv)
    if args.checkpoint is not None:
        args.checkpoint.mkdir(parents=True, exist_ok=True)
    curves = own.law_pc_curves(checkpoint=args.checkpoint)
    if args.fits_only:
        return
    if args.worker is not None:
        k, n = (int(part) for part in args.worker.split("/"))
        for chunk in range(k, own.LAW_PC_DRAWS // own.LAW_PC_CHUNK, n):
            own.chunk_sums(curves, chunk, checkpoint=args.checkpoint)
        return
    law_pc(args.checkpoint).to_csv(HERE / "law_pc_moments.csv", index=False)


if __name__ == "__main__":
    main()
