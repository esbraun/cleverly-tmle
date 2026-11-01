"""Benchmark every covariate of the twins fit, and find the tipping strengths by bisection.

Run from the repository root, with ``TWINS_DATA`` set to a directory that holds the pinned CEVAE
files or left unset to download them::

    python reviews/notebook-review/probes/twins-birth-weight/sensitivity_all.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import common

start = time.time()
result = common.fit()
att = result["att"]
print(f"fit {time.time() - start:.1f} s, att {att.psi:.4f}, se {att.std_error:.4f}", flush=True)
sens = result.sensitivity
rv = sens.robustness_value(estimand="att")
print(f"rv {rv['rv']:.4f} rva {rv['rva']:.4f}", flush=True)
records = {}
for name in common.ADJUSTMENT:
    start = time.time()
    record = sens.benchmark(covariates=(name,), estimand="att", random_state=common.SEED)
    records[name] = record
    print(
        f"{name:28s} cf_y {record.cf_y:.5f} gain_y {record.gain_y:.5f} cf_d {record.cf_d:.5f} "
        f"rho {record.rho:+.4f} move {record.psi_short - record.psi_long:+.2f} "
        f"({time.time() - start:.0f} s)",
        flush=True,
    )


def bisect(upper_at, low, high):
    """The strength in [low, high] at which the 95% upper limit reaches zero."""
    for _ in range(40):
        middle = (low + high) / 2
        if upper_at(middle) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


max_cf_d = max(record.cf_d for record in records.values())
max_cf_y = max(record.cf_y for record in records.values())
print(f"largest cf_d {max_cf_d:.5f}; largest cf_y {max_cf_y:.5f}")
tip_d = bisect(lambda d: sens.omitted_confounding(estimand="att", cf_y=1.0, cf_d=d).ci_upper, 0.0, 0.99)
print(f"cf_d tipping the 95% upper limit at cf_y = 1: {tip_d:.4f} ({tip_d / max_cf_d:.1f}x largest)")
tip_y = bisect(
    lambda y: sens.omitted_confounding(estimand="att", cf_y=y, cf_d=max_cf_d).ci_upper, 0.0, 1.0
)
print(f"cf_y tipping at the largest cf_d: {tip_y:.4f} ({tip_y / max_cf_y:.1f}x largest cf_y)")
