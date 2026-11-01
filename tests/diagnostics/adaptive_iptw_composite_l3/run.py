"""Is ``bias/ey_low__composite_l3`` of ``adaptive-iptw`` a bias or a draw?

The declared run reads a standardized bias of -0.153 on the 400 draws of the scene ``c_l3``, with a
99% interval of about [-0.282, -0.023] in standard deviations.  The predicted bias ``b`` of the
cell is near zero, so the declared bias reading does not apply.  This diagnostic runs the same
cell on ``PROBE_DRAWS`` fresh draws:

* the study's own scene ``c_l3``: the law, ``n``, the fit (cross-fitted, ten folds, the cubic
  reduced basis, the oracle factor learners declared as estimated), the truth and the row writer;
* seeds from the labels in :data:`SEED_LABELS`, which the study never uses (it uses
  ``property_sample`` and ``property_fit``), so the probe draws share no seed with the study;
* the study's definitions: the error is estimate minus truth, the spread is the sample standard
  deviation, and the interval is the Student interval of the mean error at the study's confidence
  level, divided by the spread.

The rows go to ``rows.csv.gz`` and the reading to ``summary.txt``.
``tests/unit/test_adaptive_iptw_composite_l3_reading.py`` rebuilds the reading from the rows.

    python -m tests.diagnostics.adaptive_iptw_composite_l3.run --jobs 4
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from tests.studies import adaptive_iptw_properties as properties
from tests.studies import canonical_adaptive_iptw as study
from tests.studies.evidence.inference import student_interval
from tests.studies.evidence.seeds import stream_seed

HERE = Path(__file__).parent
SCENE = "c_l3"
KEY = "ey[low]"
PROBE_DRAWS = 4_000
SEED_LABELS = ("confirm_probe_sample", "confirm_probe_fit")


def seeds(replicate: int) -> tuple[int, int]:
    """The sample seed and the fold seed of probe draw ``replicate``."""
    return tuple(stream_seed(study.STUDY, label, SCENE, replicate) for label in SEED_LABELS)  # type: ignore[return-value]


def fit_draw(replicate: int) -> dict[str, float]:
    """One probe draw: the study's fit of the scene, and the row it writes for the cell."""
    scene = study.SCENES[SCENE]
    sample_seed, fit_seed = seeds(replicate)
    frame = study.draw_frame(scene.law, scene.n, sample_seed)
    truth = study.truths(scene.law)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.FITS[scene.fits[0]](frame, scene.n, fit_seed)
    reader = next(
        item
        for item in properties.READERS
        if (item.scene, item.fit, item.key, item.family) == (SCENE, scene.fits[0], KEY, "bias")
    )
    row = properties._estimate_row(
        reader, result, truth, replicate=replicate, n=scene.n, requested=0
    )
    return {
        "replicate": replicate,
        "estimate": float(row["estimate"]),
        "truth": float(row["truth"]),
        **properties.convergence(result),
    }


def reading(rows: pd.DataFrame) -> dict[str, float]:
    """The standardized bias and its interval in spread units, with the study's definitions."""
    errors = (rows["estimate"] - rows["truth"]).to_numpy(dtype=float)
    spread = float(np.std(errors, ddof=1))
    interval = student_interval(errors, confidence_level=study.STUDY.margins.confidence_level)
    return {
        "draws": float(len(errors)),
        "mean_error": float(np.mean(errors)),
        "spread": spread,
        "standardized_bias": float(np.mean(errors)) / spread,
        "interval_lower": interval.low / spread,
        "interval_upper": interval.high / spread,
        "nonconverged": float((rows["converged"] == 0).sum()),
        "max_rounds": float(rows["rounds"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4)
    arguments = parser.parse_args()
    with ProcessPoolExecutor(arguments.jobs) as pool:
        rows = pd.DataFrame(list(pool.map(fit_draw, range(PROBE_DRAWS), chunksize=8)))
    rows.to_csv(HERE / "rows.csv.gz", index=False)
    values = reading(rows)
    lines = [
        f"draws {int(values['draws'])}; seed labels {SEED_LABELS}",
        f"mean error {values['mean_error']:.6f}; spread {values['spread']:.6f}",
        f"standardized bias {values['standardized_bias']:.4f}; "
        f"99% interval [{values['interval_lower']:.4f}, {values['interval_upper']:.4f}]",
        f"nonconverged fits {int(values['nonconverged'])}; max rounds {int(values['max_rounds'])}",
    ]
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
