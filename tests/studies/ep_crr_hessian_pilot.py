"""The oracle-Hessian pilot of the ``ep-crr`` study (design review S1 of PR-b).

A pilot, not a result.  The oracle :func:`tests.studies.ep_crr.oracle_fit` reads only the
draw, the true nuisances and the schedule's sieve design, so its Hessian check needs no
nuisance fit and runs in milliseconds.  This pilot runs :data:`REPLICATES` checks for every
law at each gated ``n`` and records how many end at a Hessian that is not positive definite.
:func:`tests.studies.ep_crr.hessian_rate` reads the committed summary,
``ep_crr_hessian_pilot.csv`` beside this module.  The misspecified learner's oracle is the same
oracle on the same design, so it is not checked twice.

Every draw comes from ``stream_seed(STUDY, "crr_hessian_pilot", label, n, r)``, disjoint from
the run's ``property_sample`` stream and the other pilots::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python -m tests.studies.ep_crr_hessian_pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cleverly.bases import Sieve, resolve
from tests.studies import ep_cate
from tests.studies import ep_crr as study
from tests.studies.evidence.seeds import stream_seed

STREAM = "crr_hessian_pilot"

#: Checks per law and ``n``: 24,000 in all.
REPLICATES = 3_000

PILOT_CSV = Path(__file__).with_name("ep_crr_hessian_pilot.csv")


def design_of(law: study.Law, frame: pd.DataFrame, n: int) -> np.ndarray:
    """The schedule's sieve design at ``n``, as the EP fit resolves it."""
    modifiers = study.modifiers(law)
    family = Sieve.default(modifiers, max_size=ep_cate.SIEVE_MAX).basis
    member = resolve(family.truncate(ep_cate.schedule(n)), modifiers)
    return np.asarray(member.evaluate(frame.loc[:, list(law.columns)], modifiers), dtype=float)


def check(law: study.Law, n: int, replicate: int) -> tuple[bool, float]:
    """Whether one replicate's oracle Hessian is positive definite, and its negative share."""
    rng = np.random.default_rng(stream_seed(study.STUDY, STREAM, law.label, n, replicate))
    frame = study.sample(law, n, rng)
    design = design_of(law, frame, n)
    w = frame.loc[:, list(law.columns)].to_numpy()
    start = np.zeros(design.shape[1])
    plug_in, _ = study.oracle_fit(design, study.treated_mean(law, w), study.baseline(w), start)
    d_t, d_r = study.oracle_pseudo(law, frame)
    _, positive = study.oracle_fit(design, d_t, d_r, plug_in)
    return positive, float(np.mean(d_t + d_r < 0.0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    arguments = parser.parse_args(argv)
    rows = []
    for law in study.LAWS:
        for n in study.gated_n(law):
            results = [check(law, n, r) for r in range(arguments.replicates)]
            rows.append(
                {
                    "law": law.label,
                    "n": n,
                    "checks": len(results),
                    "non_pd": sum(not positive for positive, _ in results),
                    "negative_share": float(np.mean([share for _, share in results])),
                }
            )
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(PILOT_CSV, index=False, float_format="%.6g")
    return 0


__all__ = ["PILOT_CSV", "REPLICATES", "STREAM"]

if __name__ == "__main__":
    raise SystemExit(main())
