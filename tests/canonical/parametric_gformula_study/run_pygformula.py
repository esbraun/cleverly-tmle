"""Run ``pygformula`` 1.2.0 on the fixture panels of ``parametric-gformula``.

The driver (``compare_pygformula.py``) writes the long panels and a spec, then runs this file in
the comparator image with the three paths of every mounted runner::

    python run_pygformula.py <samples.csv.gz> <spec.json> <output.csv>

For each fixture the runner fits ``pygformula`` once with ``model_fits=True`` and writes the
coefficient of every term and the residual SD of each Gaussian covariate.  It then runs the
simulation for each seed in the spec at ``n_simul = n`` and ``sim_trunc=False``, and writes the
risk or mean under ``treat_all`` and ``treat_none``.  The output is one CSV with LF endings.  The
columns are ``fixture, kind, model, term, estimand, horizon, seed, value``.

**What is declared.**  ``pygformula`` fits one pooled-over-time model per covariate and parses
bare column names.  So each fixture is a law where the pooled model is the node-specific model
(``tests/studies/parametric_gformula_fixtures.py``).  The survival statements are cell-means
models in time, ``0 + C(t) + C(t):W + C(t):A + C(t):lag1_A + C(t):lag2_A``.  A coefficient that
is a structural zero (a lag before time zero) has no counterpart on the ``cleverly`` side and the
runner does not write it.  The terms are renamed to the design columns of the ``cleverly`` side
(``lag1_Z`` at node 2 is ``Z1``).

**Packaging.**  The 1.2.0 wheel imports ``pygformula.version``, a module the wheel does not ship,
and ``pygformula.plot`` calls ``matplotlib.use("Qt5Agg")``, which fails without a display.  The
runner registers a version module and a plotting module that does nothing, before it imports
``pygformula``.  No number depends on either.

**Failures.**  The panels are fixed and each fit is deterministic, so a ``pygformula`` error is a
defect to fix and not a replicate to drop.  Every error stops the run.
"""

from __future__ import annotations

import json
import re
import sys
import types
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _register_shims() -> None:
    """Register the two modules the 1.2.0 wheel lacks or cannot import headless."""
    version = types.ModuleType("pygformula.version")
    version.__version__ = "1.2.0"  # type: ignore[attr-defined]
    sys.modules["pygformula.version"] = version
    plot = types.ModuleType("pygformula.plot")
    plot.plot_natural_course = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    plot.plot_interventions = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["pygformula.plot"] = plot


COLUMNS = ("fixture", "kind", "model", "term", "estimand", "horizon", "seed", "value")
INTERCEPT = "(intercept)"
SIGMA = "(sigma)"
HISTORY = r"C\(t\)\[(?P<node>\d+)\](?::(?P<term>\w+))?"

#: The statements of each fixture.  ``covariates`` lists name, type and model statement; the
#: treatment model only serves the natural-course pass that ``pygformula`` always runs.
FIXTURES: dict[str, dict[str, Any]] = {
    "cov_gaussian": {
        "outcome_type": "continuous_eof",
        "ymodel": "Y ~ W + lag1_Z + lag1_B + lag1_A + Z + B + A",
        "covariates": [
            ("Z", "normal", "Z ~ W + lag1_Z + lag1_B + lag1_A"),
            ("B", "binary", "B ~ W + lag1_Z + lag1_B + lag1_A + Z"),
            ("A", "binary", "A ~ W + lag1_Z + lag1_B + lag1_A + Z + B"),
        ],
        "n_times": 2,
    },
    "cov_binary": {
        "outcome_type": "binary_eof",
        "ymodel": "Y ~ W + lag1_Z + lag1_B + lag1_A + Z + B + A",
        "covariates": [
            ("Z", "normal", "Z ~ W + lag1_Z + lag1_B + lag1_A"),
            ("B", "binary", "B ~ W + lag1_Z + lag1_B + lag1_A + Z"),
            ("A", "binary", "A ~ W + lag1_Z + lag1_B + lag1_A + Z + B"),
        ],
        "n_times": 2,
    },
    "survival": {
        "outcome_type": "survival",
        "ymodel": "Y ~ 0 + C(t) + C(t):W + C(t):A + C(t):lag1_A + C(t):lag2_A",
        "covariates": [("A", "binary", "A ~ W + lag1_A")],
        "n_times": 3,
        "outcome_label": "E",
    },
    "competing": {
        "outcome_type": "survival",
        "ymodel": "Y ~ 0 + C(t) + C(t):W + C(t):A + C(t):lag1_A",
        "covariates": [("A", "binary", "A ~ W + lag1_A")],
        "n_times": 2,
        "outcome_label": "Y",
        "compevent_name": "D",
        "compevent_model": "D ~ W + A + lag1_A",
        # The competing event acts at the second node only, so the pooled model has that
        # node's rows alone and equals the node-specific model.
        "compevent_restrictions": [[{"t": lambda x: x == 1}, 0.0]],
    },
}


def _design_name(term: str, node: int, covariates: set[str]) -> str:
    """The ``cleverly`` design column of a ``pygformula`` term fitted at ``node`` (counted from 1).

    ``lag<i>_X`` is the value ``i`` nodes back, and a bare covariate name is the value at
    ``node``.  A baseline column keeps its name.
    """
    if term in ("Intercept", ""):
        return INTERCEPT
    lagged = re.fullmatch(r"lag(\d+)_(\w+)", term)
    if lagged:
        return f"{lagged.group(2)}{node - int(lagged.group(1))}"
    return f"{term}{node}" if term in covariates else term


def _structurally_zero(term: str, node: int) -> bool:
    """Whether a lag reaches before the first node, so the column is zero by construction."""
    lagged = re.fullmatch(r"lag(\d+)_\w+", term)
    return lagged is not None and int(lagged.group(1)) >= node


def coefficient_rows(name: str, params: pd.Series, covariates: set[str]) -> list[dict[str, Any]]:
    """The coefficient rows of one fitted model, renamed to ``cleverly``'s labels.

    A model with ``C(t)`` terms is a cell-means model in time and splits into one model per time.
    """
    rows = []
    for raw, value in params.items():
        match = re.fullmatch(HISTORY, str(raw))
        if match:
            node = int(match.group("node")) + 1
            term = match.group("term") or "Intercept"
            if _structurally_zero(term, node):
                continue
            label, node_term = f"{name}@{node}", _design_name(term, node, covariates)
        else:
            node = 2
            label, node_term = f"{name}@{node}", _design_name(str(raw), node, covariates)
        rows.append({"model": label, "term": node_term, "value": float(value)})
    return rows


def load_pygformula() -> tuple[Any, Any]:
    """Register the shims, then import ``ParametricGformula`` and the ``static`` intervention."""
    _register_shims()
    from pygformula import ParametricGformula
    from pygformula.interventions import static

    return ParametricGformula, static


def run_fixture(name: str, long: pd.DataFrame, seeds: list[int]) -> list[dict[str, Any]]:
    """Every output row of one fixture."""
    ParametricGformula, static = load_pygformula()
    config = FIXTURES[name]
    n_times = config["n_times"]
    frame = long.loc[:, ["id", "t", "W", "A", "Y"]].copy()
    names = [covariate[0] for covariate in config["covariates"]]
    if "Z" in names:
        frame["Z"] = long["Z"]
        frame["B"] = long["B"]
    if "compevent_name" in config:
        frame["D"] = long["D"]
    frame = frame.sort_values(["id", "t"], ignore_index=True)
    arguments: dict[str, Any] = {
        "obs_data": frame,
        "id": "id",
        "time_name": "t",
        "outcome_name": "Y",
        "outcome_type": config["outcome_type"],
        "ymodel": config["ymodel"],
        "covnames": names,
        "covtypes": [covariate[1] for covariate in config["covariates"]],
        "covmodels": [covariate[2] for covariate in config["covariates"]],
        "basecovs": ["W"],
        "int_descript": ["treat_all", "treat_none"],
        "sim_trunc": False,
        "model_fits": True,
        "nsamples": 0,
    }
    if config["outcome_type"] == "survival":
        arguments["time_points"] = n_times
    if "compevent_name" in config:
        arguments["compevent_name"] = config["compevent_name"]
        arguments["compevent_model"] = config["compevent_model"]
        arguments["compevent_restrictions"] = config["compevent_restrictions"]
    rows: list[dict[str, Any]] = []
    covariates = set(names)
    for seed in seeds:
        # ``pygformula`` inserts the treatment name into each intervention list in place, so
        # every fit gets fresh lists.
        plans = {
            "Intervention1_A": [static, np.ones(n_times)],
            "Intervention2_A": [static, np.zeros(n_times)],
        }
        fit = ParametricGformula(seed=seed, **arguments, **plans)
        fit.fit()
        if seed == seeds[0]:  # the fits do not depend on the seed
            summary = fit.summary_dict
            for model_name, params in summary["model_coeffs"].items():
                if model_name == "A":
                    continue  # the natural-course treatment model: no static plan reads it
                label = config.get("outcome_label", "Y") if model_name == "Y" else model_name
                for item in coefficient_rows(label, params, covariates):
                    rows.append(
                        {
                            "fixture": name,
                            "kind": "coefficient",
                            "estimand": "",
                            "horizon": "",
                            "seed": "",
                            **item,
                        }
                    )
            for model_name, rmse in (summary["rmses"] or {}).items():
                if {c[0]: c[1] for c in config["covariates"]}.get(model_name) == "normal":
                    rows.append(
                        {
                            "fixture": name,
                            "kind": "coefficient",
                            "model": f"{model_name}@2",
                            "term": SIGMA,
                            "estimand": "",
                            "horizon": "",
                            "seed": "",
                            "value": float(rmse),
                        }
                    )
        for label, result in zip(("treat_all", "treat_none"), fit.g_results[1:], strict=True):
            risks = [result] if np.isscalar(result) else list(result)
            horizons = [n_times] if np.isscalar(result) else list(range(1, len(risks) + 1))
            for horizon, value in zip(horizons, risks, strict=True):
                rows.append(
                    {
                        "fixture": name,
                        "kind": "estimate",
                        "model": "",
                        "term": "",
                        "estimand": label,
                        "horizon": horizon,
                        "seed": seed,
                        "value": float(value),  # type: ignore[arg-type]
                    }
                )
    return rows


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        sys.stderr.write("usage: run_pygformula.py <samples.csv.gz> <spec.json> <output.csv>\n")
        return 2
    samples, spec_path, output = Path(argv[1]), Path(argv[2]), Path(argv[3])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    panels = pd.read_csv(samples)
    rows: list[dict[str, Any]] = []
    for name in spec["fixtures"]:
        long = panels.loc[panels["fixture"] == name].reset_index(drop=True)
        sys.stdout.write(f"{name}: {long['id'].nunique()} units, {len(spec['seeds'])} seeds\n")
        sys.stdout.flush()
        rows += run_fixture(name, long, [int(seed) for seed in spec["seeds"]])
    table = pd.DataFrame(rows, columns=list(COLUMNS))
    table.to_csv(output, index=False, lineterminator="\n")
    versions = {
        package: metadata.version(package)
        for package in ("pygformula", "numpy", "pandas", "scipy", "statsmodels", "patsy")
    }
    environment = output.with_name(f"{output.stem}.environment.json")
    environment.write_text(
        json.dumps({"python": sys.version.split()[0], **versions}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
