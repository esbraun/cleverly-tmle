"""The probes the ``projection-methods`` redesign rests on (P1 PR 3 plan rev 3, R3.1 to R3.8).

The first declared run (:data:`~tests.canonical.projection_methods.first_run.FIRST_RUN`) failed
two gated cells.  The redesign sized the new cells from declared-stream probes, which this module
regenerates through the study's own :func:`~tests.studies.projection_methods.draw_from_seed` and
:func:`~tests.studies.projection_methods.fit_group`, with the seeds of
:func:`~tests.studies.projection_methods_properties.sample_seed` and ``fold_seed`` on the probe
streams.  No group of the declared run reads those streams.  The design test reads the two
artifacts and never refits, except one replicate of each probe against its committed row.

:data:`PROBES` (``projection_methods_probes.parquet``) holds one row per replicate and term:

- ``glm4k_probe``: ``dr_glm_4k`` at n = 4,000, 1,000 replicates, the finite-n SE ratio that sets
  the planning ratio 1.105;
- ``ladder_probe``: the slow mechanism at n = 64,000, 100 replicates, the planning bias of the
  new top rung;
- ``slow_1k_x<c>``: the slow mechanism at n = 1,000 with its offset times ``c`` in 0 to 3, 400
  replicates each, the reason the 1,000-row rung went (the first run's red rung).

:data:`LIMIT` (``projection_methods_limit.csv``) holds, in long form:

- ``limit``: fits at n = 200,000 on ``limit_probe`` of ``dr_known_4k`` and ``dr_glm_4k`` under the
  declared guard ``("Q", "g")`` (replicates 0 and 1), the guard ``("Q",)`` (0 and 1) and the
  empty guard (0): the reported curve's SD times ``sqrt(n)``, the "true" SD (the curve with its
  projection on the mechanism GLM's scores removed, ``LawP.conservative_ratio``'s construction)
  and their ratio.  Under ``("Q",)`` the fit is the (8)+(9) estimator the quadrature models;
- ``glm4k_probe``, ``ladder_probe`` and ``slow_1k_x<c>``: each probe's summary per term;
- ``first_run``: rows copied from the first run's ``properties.csv`` and the bootstrap spread
  constant of its ``dr_glm`` intercept rows, with the commit named.

Run from the repository root with ``OMP_NUM_THREADS=1``::

    python -m tests.studies.projection_methods_limit --jobs 4
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tests.canonical.projection_methods import first_run
from tests.studies import projection_methods as study
from tests.studies import projection_methods_properties as properties
from tests.studies.evidence.seeds import stream_seed

LIMIT = Path(__file__).with_name("projection_methods_limit.csv")
PROBES = Path(__file__).with_name("projection_methods_probes.parquet")

#: Replicates of each probe (plan rev 3, R3.1 and R3.2).
GLM4K_REPLICATES = 1_000
LADDER_REPLICATES = 100
SLOW_1K_REPLICATES = 400
#: The slow offset's scales at n = 1,000: none, the declared one, and two larger.
SLOW_1K_SCALES = (0, 1, 2, 3)
#: The size and fits of the limit probe: ``(guard, replicate)``.
LIMIT_N = 200_000
LIMIT_FITS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("Q", "g"), 0),
    (("Q", "g"), 1),
    (("Q",), 0),
    (("Q",), 1),
    ((), 0),
)
LIMIT_LABELS = ("dr_known_4k", "dr_glm_4k")
#: Draws of the bootstrap spread constant.
BOOTSTRAP_DRAWS = 10_000
#: The first run's cells this module copies, and the columns copied.
FIRST_RUN_CELLS = (
    "dr_glm__intercept__conservative",
    "dr_glm__W1__valid",
    *(
        f"dr_slow_n{n}__{t}__first_order"
        for n in (1_000, 4_000, 16_000)
        for t in ("intercept", "W1")
    ),
    "dr_known_4k_intercept__d_star_reading",
    "dr_known_4k_W1__d_star_reading",
)
FIRST_RUN_COLUMNS = (
    "replicates",
    "bias",
    "standardized_bias",
    "empirical_se",
    "mean_std_error",
    "se_ratio",
    "se_ratio_ci_lower",
    "se_ratio_ci_upper",
    "coverage",
    "coverage_ci_lower",
    "coverage_ci_upper",
    "passed",
)
LIMIT_COLUMNS = ("block", "cell", "guard", "replicate", "term", "quantity", "value", "commit")
PROBE_COLUMNS = (
    "mode",
    "replicate",
    "term",
    "n",
    "sample_seed",
    "fold_seed",
    "estimate",
    "truth",
    "plugin_std_error",
)


def guard_name(guard: tuple[str, ...]) -> str:
    """The guard as the artifact writes it: ``Q,g``, ``Q`` or ``none``."""
    return ",".join(guard) or "none"


# ----------------------------------------------------------------------------- the probes


def probe_group(mode: str) -> tuple[study.Group, str, float]:
    """The group, stream and offset scale of a probe ``mode``."""
    if mode == "glm4k_probe":
        return study.GROUP_BY_LABEL["dr_glm_4k"], "glm4k_probe", 1.0
    if mode == "ladder_probe":
        return study.GROUP_BY_LABEL["dr_slow_64k"], "ladder_probe", 1.0
    scale = float(mode.removeprefix("slow_1k_x"))
    group = dataclasses.replace(study.GROUP_BY_LABEL["dr_slow_4k"], label="dr_slow_1k", n=1_000)
    return group, "limit_probe", scale


def probe_rows(mode: str, replicate: int) -> list[dict[str, Any]]:
    """One replicate of a probe, through the study's draw and fit."""
    group, stream, scale = probe_group(mode)
    seed = properties.sample_seed(group.label, replicate, stream)
    fold = properties.fold_seed(group.label, replicate, stream)
    frame, truths = study.draw_from_seed(
        "dr_slow_4k" if mode.startswith("slow") else group.label, group.n, seed
    )
    probe = {} if scale == 1.0 else {"offset_scale": scale}
    estimates = study.fit_group(group, frame, random_state=fold, **probe)["estimates"]
    return [
        {
            "mode": mode,
            "replicate": replicate,
            "term": term,
            "n": group.n,
            "sample_seed": seed,
            "fold_seed": fold,
            "estimate": float(estimates[term].psi),
            "truth": float(truths[group.name(term)]),
            "plugin_std_error": float(estimates[term].plugin_std_error),
        }
        for term in group.terms
    ]


def probe_modes() -> dict[str, int]:
    """Every probe and its replicates."""
    return {
        "glm4k_probe": GLM4K_REPLICATES,
        "ladder_probe": LADDER_REPLICATES,
        **{f"slow_1k_x{c}": SLOW_1K_REPLICATES for c in SLOW_1K_SCALES},
    }


def probe_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Each probe's summary per term, in the long form of :data:`LIMIT`."""
    out = []
    for (mode, term), g in rows.groupby(["mode", "term"], sort=True):
        error = (g["estimate"] - g["truth"]).to_numpy()
        m = len(error)
        sd = float(error.std(ddof=1))
        bias = float(error.mean())
        standardized = bias / sd
        values = {
            "replicates": float(m),
            "bias": bias,
            "bias_se": sd / math.sqrt(m),
            "sd": sd,
            "mean_se": float(g["plugin_std_error"].mean()),
            "ratio": float(g["plugin_std_error"].mean()) / sd,
            "standardized_bias": standardized,
            # The delta-method SE of the standardized bias, bias and SD errors together.
            "standardized_bias_se": math.sqrt((1.0 + standardized**2 / 2.0) / m),
        }
        out += [
            {"block": mode, "cell": mode, "term": term, "quantity": q, "value": v}
            for q, v in values.items()
        ]
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------ the limit


def limit_rows(label: str, guard: tuple[str, ...], replicate: int) -> list[dict[str, Any]]:
    """One fit at :data:`LIMIT_N` under ``guard``: the reported and true SDs and their ratio."""
    group = dataclasses.replace(study.GROUP_BY_LABEL[label], n=LIMIT_N)
    seed = properties.sample_seed("limit", replicate, "limit_probe")
    fold = properties.fold_seed("limit", replicate, "limit_probe")
    frame, _ = study.draw_from_seed(label, LIMIT_N, seed)
    result = study.fit(group, frame, fold, guard=guard)
    mechanism = result.fluctuations["effect_modification"].mechanism
    treated = frame["A"].to_numpy(dtype=float)
    if mechanism is not None:
        g1 = np.asarray(mechanism.propensity, dtype=float)
    elif group.mechanism == "known":
        g1 = frame["p1"].to_numpy(dtype=float)
    else:
        g1 = None
    out = []
    for term in group.terms:
        curve = np.asarray(result.estimates[group.name(term)].influence_curve, dtype=float)
        reported = float(np.std(curve))
        true = math.nan
        if g1 is not None:
            scores = (
                np.column_stack([np.ones(LIMIT_N), frame["W1"].to_numpy(), frame["W2"].to_numpy()])
                * (treated - g1)[:, None]
            )
            coefficients = np.linalg.lstsq(scores, curve, rcond=None)[0]
            true = float(np.std(curve - scores @ coefficients))
        for quantity, value in (
            ("reported_sd_root_n", reported),
            ("true_sd_root_n", true),
            ("ratio", reported / true),
        ):
            out.append(
                {
                    "block": "limit",
                    "cell": label,
                    "guard": guard_name(guard),
                    "replicate": replicate,
                    "term": term,
                    "quantity": quantity,
                    "value": value,
                }
            )
    return out


# -------------------------------------------------------------------------- the first run


def spread_constant(error: np.ndarray, std_error: np.ndarray, seed: int) -> float:
    """``k``: the bootstrap SD of the SE ratio times ``sqrt(R)`` over the ratio (R3.1)."""
    rng = np.random.default_rng(seed)
    count = len(error)
    index = rng.integers(0, count, size=(BOOTSTRAP_DRAWS, count))
    ratios = std_error[index].mean(axis=1) / error[index].std(axis=1, ddof=1)
    ratio = std_error.mean() / error.std(ddof=1)
    return float(ratios.std(ddof=1) * math.sqrt(count) / ratio)


def first_run_rows(read: Callable[[str], pd.DataFrame] = first_run.read) -> pd.DataFrame:
    """The first run's rows the redesign reads, copied from its committed artifacts."""
    summary = read("properties.csv").set_index("cell")
    out = [
        {
            "block": "first_run",
            "cell": cell,
            "quantity": column,
            "value": float(summary.loc[cell, column]),
        }
        for cell in FIRST_RUN_CELLS
        for column in FIRST_RUN_COLUMNS
    ]
    rows = read("property-replicates.parquet")
    cell = "dr_glm__intercept__conservative"
    chosen = rows.loc[rows["cell"] == cell].sort_values("replicate")
    error = (chosen["estimate"] - chosen["truth"]).to_numpy(dtype=float)
    k = spread_constant(error, chosen["std_error"].to_numpy(dtype=float), study.SEED)
    out.append({"block": "first_run", "cell": cell, "quantity": "spread_constant", "value": k})
    frame = pd.DataFrame(out)
    frame["commit"] = first_run.FIRST_RUN
    return frame


def ratio_spread_seed() -> int:
    """The seed of the design test's bootstrap of the probe's spread constant."""
    return stream_seed(study.STUDY, "ratio_spread")


# --------------------------------------------------------------------------------- writing


def _long(frame: pd.DataFrame) -> pd.DataFrame:
    for column in LIMIT_COLUMNS:
        if column not in frame:
            frame[column] = None
    return frame.loc[:, list(LIMIT_COLUMNS)]


def main(jobs: int, parts: tuple[str, ...]) -> None:
    """Regenerate ``parts`` (``probes``, ``limit``) and rewrite :data:`LIMIT`."""
    blocks = pd.read_csv(LIMIT) if LIMIT.exists() else pd.DataFrame(columns=list(LIMIT_COLUMNS))
    if "probes" in parts:
        keys = [(mode, r) for mode, count in probe_modes().items() for r in range(count)]
        keys.sort(key=lambda key: -probe_group(key[0])[0].n)
        done = Parallel(n_jobs=jobs, batch_size=1)(delayed(probe_rows)(*key) for key in keys)
        rows = pd.DataFrame([row for part in done for row in part]).loc[:, list(PROBE_COLUMNS)]
        rows = rows.sort_values(["mode", "replicate", "term"], ignore_index=True)
        rows.to_parquet(PROBES, index=False)
    probes = pd.read_parquet(PROBES)
    parts_out = [_long(probe_summary(probes)), _long(first_run_rows())]
    if "limit" in parts:
        keys = [(label, guard, r) for guard, r in LIMIT_FITS for label in LIMIT_LABELS]
        done = Parallel(n_jobs=jobs, batch_size=1)(delayed(limit_rows)(*key) for key in keys)
        parts_out.insert(0, _long(pd.DataFrame([row for part in done for row in part])))
    else:
        parts_out.insert(0, blocks.loc[blocks["block"] == "limit"])
    pd.concat(parts_out, ignore_index=True).to_csv(LIMIT, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--parts", nargs="+", default=["probes", "limit"])
    arguments = parser.parse_args()
    main(arguments.jobs, tuple(arguments.parts))
