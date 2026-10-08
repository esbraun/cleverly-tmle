"""Reported SE over the bound for the declared categorical fit at large n.

The post-run review of run 3 wrote and ran this probe, 20 draws per size.  ``large_n.log`` is
its output.

    python tests/diagnostics/longitudinal_mtp_categorical_efficiency/large_n.py 8000,32000,128000
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3].as_posix()
sys.path[:0] = [ROOT + "/src", ROOT]

import numpy as np  # noqa: E402

import cleverly  # noqa: E402

assert cleverly.__file__.replace("\\", "/").startswith(ROOT), cleverly.__file__
from tests.studies import canonical_longitudinal_mtp as study  # noqa: E402
from tests.studies import longitudinal_mtp_common as common  # noqa: E402
from tests.studies import longitudinal_mtp_properties as P  # noqa: E402

bound = P.EFFICIENCY_SD["categorical_mtp"]
truth = float(P.TRUTH[P.CATEGORICAL][P.CATEGORICAL_NAME])
draws = int(sys.argv[2]) if len(sys.argv) > 2 else 20
for n in [int(v) for v in sys.argv[1].split(",")]:
    ratios, errors = [], []
    for r in range(draws):
        frame = common.CATEGORICAL_LAW.sample(n, 990_000_000 + 1000 * r + n % 997)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = study.fit(frame, study.CATEGORICAL)[P.CATEGORICAL_NAME]
        ratios.append(float(e.std_error) * np.sqrt(n) / bound)
        errors.append((float(e.psi) - truth) * np.sqrt(n) / bound)
    ratios = np.array(ratios)
    print(
        f"n={n} draws={draws} reported/bound mean={ratios.mean():.4f} "
        f"sd={ratios.std(ddof=1):.4f} min={ratios.min():.4f} max={ratios.max():.4f} "
        f"rms_std_error_of_estimate={np.sqrt(np.mean(np.square(errors))):.3f}",
        flush=True,
    )
