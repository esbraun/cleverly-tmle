"""The post-run diagnosis of the red cells of ``small-sample-wald`` (Fix-3).

The declared run published five red gated cells.  This diagnostic reads each on fresh draws,
fitted with the study's own samplers, fits and rule columns.  Every draw uses seed
``2**40 + offset + r``, above every 32-bit ``stream_seed`` of the declared run, so the draws are
disjoint from it.  It is not a declared run and no verdict reads it.

=================  ============================================================================
probe              what it fits
=================  ============================================================================
``h3``             ``fit_msm`` on held-out law H3: 3,000 draws at n = 2,000 and 1,500 at 8,000
                   (``h3_fits.csv.gz``), and one fit at n = 400,000 whose influence curve is
                   averaged per support point (``h3_big.csv.gz``), against H3's exact efficient
                   influence function
``confirm``        3,000 draws each of H1 (``ho_sparse_treat``), H2 (``ho_hurdle``) and the
                   lognormal law of skewness 2.0, through ``fit_group`` and ``rule_columns``
                   (``confirm.csv.gz``); a lost fit is a row with an ``error``
``h2_leverage``    the H2 draws of ``confirm`` again, with a hand-built stratum-2 curve beside the
                   fit and the same curve with each residual divided by ``sqrt(1 - 1/n_cell)``,
                   the cell mean's leverage (``h2_leverage.csv.gz``); a diagnostic of the
                   mechanism, not a proposed interval
=================  ============================================================================

``tests/unit/test_fix3_postrun_reading.py`` rebuilds every number the evidence page and the
roadmap quote from these files.

    python -m tests.diagnostics.fix3_postrun.run <probe> --jobs 4
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from cleverly.utils.parallel import map_parallel
from tests import discrete_law
from tests.studies import small_sample_laws as laws
from tests.studies import small_sample_wald as study
from tests.studies import stratified_alternating_law as alternating

HERE = Path(__file__).resolve().parent
BASE = 2**40
#: Each H3 run: its seed offset, size and number of draws.
H3_DESIGN = {"n2000": (0, 2_000, 3_000), "n8000": (1_000_000, 8_000, 1_500)}
H3_BIG_N = 400_000
H3_BIG_SEED = BASE + 9_000_000
#: Each confirmed group's seed offset; every group runs 3,000 draws.
CONFIRM_OFFSET = {"ho_sparse_treat": 2_000_000, "ho_hurdle": 3_000_000, "lognormal06": 4_000_000}
CONFIRM_DRAWS = 3_000
H2_NAMES = ("ey[1][V=2]", "ey[0][V=2]", "ate[V=2]")


def h3_efficient_curve() -> dict[str, np.ndarray]:
    """H3's exact efficient influence function of each coefficient, per support point."""
    probabilities = laws.h3_probs()
    support = [(index,) for index in range(len(alternating.SUPPORT))]
    return {
        name: np.real(
            np.asarray(
                discrete_law.contamination_eif(
                    alternating.functional(name, "logit"), probabilities, support
                )
            )
        )
        for name in laws.H3_NAMES
    }


def _h3_fit(n: int, replicate: int, offset: int) -> list[dict[str, Any]]:
    frame = laws.h3_sample(n, BASE + offset + replicate)
    fit = laws.h3_fit(frame)
    return [
        {
            "n": n,
            "r": replicate,
            "name": name,
            "psi": float(fit[name].psi),
            "se": float(fit[name].plugin_std_error),
        }
        for name in laws.H3_NAMES
    ]


def _h3_big() -> pd.DataFrame:
    frame = laws.h3_sample(H3_BIG_N, H3_BIG_SEED)
    fit = laws.h3_fit(frame)
    cell = np.array(
        [
            alternating.SUPPORT.index(tuple(x))
            for x in frame[["V", "W", "A", "Y"]].to_numpy().tolist()
        ]
    )
    rows = []
    for name in laws.H3_NAMES:
        estimate = fit[name]
        curve = pd.Series(np.asarray(estimate.influence_curve, dtype=float)).groupby(cell)
        for support, mean in curve.mean().items():
            rows.append(
                {
                    "name": name,
                    "cell": int(support),
                    "fitted_mean": float(mean),
                    "count": int(curve.size()[support]),
                    "psi": float(estimate.psi),
                    "se": float(estimate.plugin_std_error),
                }
            )
    return pd.DataFrame(rows)


def _confirm(label: str, replicate: int) -> list[dict[str, Any]]:
    group = study.GROUP_BY_LABEL[label]
    frame = study.sample(group.law, group.n, BASE + CONFIRM_OFFSET[label] + replicate)
    truth = study.truths(group.law)
    try:
        fits = study.fit_group(group, frame)
    except Exception as exc:  # a lost fit is a recorded row, as in the study
        return [{"label": label, "r": replicate, "error": repr(exc)[:100]}]
    names = {cell.estimand for cell in group.cells}
    if group.law == "h1":
        names |= {"ey[1]", "ey[0]", "ate"}
    rows = []
    for name in sorted(names):
        estimate = fits["main"][name]
        columns = study.rule_columns(frame, name, estimate, group.family)
        rows.append(
            {
                "label": label,
                "r": replicate,
                "estimand": name,
                "truth": float(truth[name]),
                "psi": float(estimate.psi),
                "se": float(estimate.plugin_std_error),
                **{
                    key: columns[key]
                    for key in (
                        "nu",
                        "kish",
                        "bound_low",
                        "bound_high",
                        "rule_lower",
                        "rule_upper",
                        "withheld",
                    )
                },
            }
        )
    return rows


def _h2_leverage(replicate: int) -> list[dict[str, Any]]:
    frame = laws.h2_sample(2_000, BASE + CONFIRM_OFFSET["ho_hurdle"] + replicate)
    try:
        fit = laws.h2_fit(frame)
    except Exception:  # the empty-cell draws, lost in the study too
        return []
    g = (
        LogisticRegression(C=1e6, max_iter=2000)
        .fit(frame[["W", "V"]], frame.A)
        .predict_proba(frame[["W", "V"]])[:, 1]
    )
    stratum = frame.V.to_numpy() == 2
    y, a, w = (frame[column].to_numpy()[stratum] for column in ("Y", "A", "W"))
    g = g[stratum]
    q: dict[tuple[int, int], float] = {}
    count: dict[tuple[int, int], int] = {}
    for arm in (0, 1):
        for level in (0, 1, 2):
            mask = (a == arm) & (w == level)
            q[arm, level] = y[mask].mean()
            count[arm, level] = int(mask.sum())
    q1 = np.array([q[1, x] for x in w])
    q0 = np.array([q[0, x] for x in w])
    residual = y - np.where(a == 1, q1, q0)
    leverage = np.array([1.0 / count[arm, level] for arm, level in zip(a, w, strict=True)])
    rows = []
    for name in H2_NAMES:
        if name.startswith("ey[1]"):
            weight, plug = a / g, q1
        elif name.startswith("ey[0]"):
            weight, plug = (1 - a) / (1 - g), q0
        else:
            weight, plug = a / g - (1 - a) / (1 - g), q1 - q0
        psi = plug.mean()
        curve = weight * residual + plug - psi
        adjusted = weight * residual / np.sqrt(1 - np.minimum(leverage, 0.999)) + plug - psi
        size = stratum.sum()
        se_hand = np.sqrt(np.sum(curve**2)) / size
        estimate = fit[name]
        rows.append(
            {
                "r": replicate,
                "name": name,
                "truth": laws.h2_truths()[name],
                "psi": float(estimate.psi),
                "psi_hand": psi,
                "se": float(estimate.plugin_std_error),
                "se_hand": se_hand,
                "se_adj_ratio": (np.sqrt(np.sum(adjusted**2)) / size) / se_hand,
                "nu": float(estimate.small_sample_df),
            }
        )
    return rows


def _quiet(func: Any, *args: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args)


def _write(rows: pd.DataFrame, name: str) -> None:
    with open(HERE / name, "wb") as handle:
        rows.to_csv(
            handle, index=False, compression={"method": "gzip", "mtime": 0}, lineterminator="\n"
        )


def run(probe: str, jobs: int, draws: int | None = None) -> dict[str, pd.DataFrame]:
    """Fit one probe.  ``draws`` caps each design's draw count, for a reproduction check."""
    out: dict[str, pd.DataFrame] = {}
    if probe == "h3":
        payloads = [
            (_h3_fit, n, r, offset)
            for offset, n, count in H3_DESIGN.values()
            for r in range(count if draws is None else min(count, draws))
        ]
        rows = map_parallel(_quiet, payloads, n_jobs=jobs)
        out["h3_fits.csv.gz"] = pd.DataFrame([row for result in rows for row in result])
        if draws is None:
            out["h3_big.csv.gz"] = _quiet(_h3_big)
    elif probe == "confirm":
        count = CONFIRM_DRAWS if draws is None else draws
        payloads = [(_confirm, label, r) for label in CONFIRM_OFFSET for r in range(count)]
        rows = map_parallel(_quiet, payloads, n_jobs=jobs)
        out["confirm.csv.gz"] = pd.DataFrame([row for result in rows for row in result])
    elif probe == "h2_leverage":
        count = CONFIRM_DRAWS if draws is None else draws
        rows = map_parallel(_quiet, [(_h2_leverage, r) for r in range(count)], n_jobs=jobs)
        out["h2_leverage.csv.gz"] = pd.DataFrame([row for result in rows for row in result])
    else:
        raise KeyError(probe)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", choices=("h3", "confirm", "h2_leverage"))
    parser.add_argument("--jobs", type=int, default=1)
    arguments = parser.parse_args()
    for name, rows in run(arguments.probe, arguments.jobs).items():
        _write(rows, name)


if __name__ == "__main__":
    main()
