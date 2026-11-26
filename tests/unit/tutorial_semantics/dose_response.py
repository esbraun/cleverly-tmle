"""The reviewed semantic callback for ``docs/examples/dose-response``.

``tests/unit/test_documentation_runtime.py`` runs the notebook at its documented size and passes
the namespace to :func:`check`.  Each assertion is a claim the prose makes about this draw, not a
statistical guarantee.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tests.documents import ROOT
from tests.unit.tutorial_semantics import assert_protocol_recorded, figure_title

NOTEBOOK = ROOT / "docs" / "examples" / "dose-response.ipynb"
_PROBE = Path("reviews") / "notebook-review" / "probes" / "dose-response"

#: Decimals the prose quotes from the page's seed sweep and its cause probes, by file
#: (reviews/notebook-review/probes/dose-response). :func:`_check_probes` asserts each one.
UNPRINTED_DECIMALS = {
    "0.950": f"pointwise coverage at dose 20, bandwidth 2.5, redrawn engine: "
    f"{_PROBE}/summary-jitter1.log",
    "0.940": f"pointwise coverage at dose 20, bandwidth 5, redrawn engine: "
    f"{_PROBE}/summary-jitter1.log",
    "0.895": f"pointwise coverage at dose 5, bandwidth 5, redrawn engine: "
    f"{_PROBE}/summary-jitter1.log",
    "0.8": "the earlier engine's jitter SD, a setting of the probe (sweep.py JITTER default)",
    "1.0": "the page's jitter SD, printed in the data cell's code, not its output",
    "0.820": f"pointwise coverage at dose 5, bandwidth 2.5, earlier engine: {_PROBE}/summary.log",
    "0.760": f"pointwise coverage at dose 5, bandwidth 5, earlier engine: {_PROBE}/summary.log",
    "0.835": f"band covers all 12 pairs, earlier engine: {_PROBE}/summary.log",
    "0.69": f"SE ratio at dose 5, bandwidth 2.5, earlier engine: {_PROBE}/summary.log",
    "0.60": f"SE ratio at dose 5, bandwidth 5, earlier engine: {_PROBE}/summary.log",
    "0.881": f"coverage, flexible discount model, dose 5, bandwidth 2.5: {_PROBE}/cause-summary.log",
    "0.800": f"coverage, flexible discount model, dose 5, bandwidth 5: {_PROBE}/cause-summary.log",
    "0.934": f"coverage with the true models, dose 5, bandwidth 2.5: {_PROBE}/oracle_onestep.log",
    "0.932": f"coverage with the true models, dose 5, bandwidth 5: {_PROBE}/oracle_onestep.log",
}


def _per_pair(log: str, column: str) -> dict[tuple[float, float], float]:
    """Read one column of a summarize.py "per pair" table into {(bandwidth, dose): value}."""
    lines = log.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("per pair:"))
    header = lines[start + 1].split()
    values: dict[tuple[float, float], float] = {}
    bandwidth = None
    for line in lines[start + 3 : start + 15]:
        fields = line.split()
        if len(fields) == len(header) + 2:
            bandwidth = float(fields.pop(0))
        values[(float(bandwidth), float(fields[0]))] = float(fields[1 + header.index(column)])
    return values


def _check_probes() -> None:
    """Each probe decimal the prose quotes equals the committed log that holds it."""
    redrawn = (ROOT / _PROBE / "summary-jitter1.log").read_text(encoding="utf-8")
    earlier = (ROOT / _PROBE / "summary.log").read_text(encoding="utf-8")
    cover = _per_pair(redrawn, "cover")
    assert f"{cover[(2.5, 20.0)]:.3f}" == "0.950" and f"{cover[(5.0, 20.0)]:.3f}" == "0.940"
    assert f"{cover[(5.0, 5.0)]:.3f}" == "0.895"
    assert "draws: 200 " in redrawn and "draws: 200 " in earlier
    old_cover, old_ratio = _per_pair(earlier, "cover"), _per_pair(earlier, "se_ratio")
    assert f"{old_cover[(2.5, 5.0)]:.3f}" == "0.820" and f"{old_cover[(5.0, 5.0)]:.3f}" == "0.760"
    assert f"{old_ratio[(2.5, 5.0)]:.2f}" == "0.69" and f"{old_ratio[(5.0, 5.0)]:.2f}" == "0.60"
    assert "band covers all 12 pairs: 0.835 " in earlier
    # "on other draws of this engine the 30% pairs fall below the floor about one time in four"
    below = _per_pair(redrawn, "below_floor")
    assert 0.2 <= below[(2.5, 30.0)] <= 0.3 and 0.2 <= below[(5.0, 30.0)] <= 0.3
    arms = (ROOT / _PROBE / "cause-summary.log").read_text(encoding="utf-8")
    flexible = arms.split("arm: flexible_density")[1].splitlines()
    rows = [line.split() for line in flexible if line.strip().endswith(tuple("0123456789"))]
    assert rows[0][:2] == ["2.5", "5.0"] and rows[0][-1] == "0.881"
    assert rows[6][:2] == ["5.0", "5.0"] and rows[6][-1] == "0.800"
    oracle = (ROOT / _PROBE / "oracle_onestep.log").read_text(encoding="utf-8")
    assert "h=2.5 dose=5: coverage 0.934," in oracle and "h=5 dose=5: coverage 0.932," in oracle


#: The refit cells the rule derives from this notebook today, in notebook order.  A change here is
#: a change in what the fast gate skips, so it shows in the diff of the pull request that causes it.
REFIT_CELLS: tuple[str, ...] = ()

#: Cells read under the looser freshness rule of :mod:`tests.notebook_freshness`.
_BOOSTED = (
    "reads the boosted outcome fit: HistGradientBoostingClassifier sums its histograms in float32, "
    "which round with the CPU's vector instructions, and one CI runner moved these numbers by up "
    "to 0.2% (tests.notebook_freshness)"
)
HARDWARE_SENSITIVE_CELLS = {
    "estimate": _BOOSTED,
    "compare": _BOOSTED,
    "assessment": _BOOSTED,
    "support": _BOOSTED,
}


def _rows(namespace: dict[str, Any], h: float) -> Any:
    table = namespace["table"]
    return table[table["bandwidth"] == h].set_index("dose")


def _check_data(namespace: dict[str, Any]) -> None:
    """The true curve rises steeply, then flattens; the simple comparison falls instead."""
    true_curve = namespace["true_curve"]
    early = true_curve(15) - true_curve(0)
    late = true_curve(30) - true_curve(15)
    assert early > 2 * late > 0, "most of the gain comes from the first 15 points"
    naive = namespace["naive"]
    shares = naive["share_renewed"].to_numpy()
    assert shares[-1] == shares.min() and shares[-1] < shares[0], "deep discounts renew least"
    assert np.all(np.diff(naive["true share at band middle"]) > 0), "the true curve rises"
    assert np.all(np.diff(naive["mean_usage"]) < 0), "bigger discounts go to less active users"


def _check_declaration(namespace: dict[str, Any]) -> None:
    """Twelve pairs, and a range inside the observed discounts with a margin."""
    curve, frame = namespace["curve"], namespace["frame"]
    assert len(curve.pairs) == 12
    lower, upper = curve.support
    assert frame["discount"].min() < lower and upper < frame["discount"].max()
    assert curve.nodes_per_piece == 8, "eight per piece by default"
    assert_protocol_recorded(NOTEBOOK, "study", namespace["protocol"], namespace["result"])


def _check_estimate(namespace: dict[str, Any]) -> None:
    """Every band covers; every interval but the 20% pairs covers; the curve rises to 20%."""
    table = namespace["table"]
    assert ((table["band_lower"] <= table["true"]) & (table["true"] <= table["band_upper"])).all()
    at_20 = table["dose"] == 20.0
    off = table[~at_20]
    assert ((off["lower"] <= off["true"]) & (off["true"] <= off["upper"])).all()
    # At 20% both estimates sit above the truth and each lower end is level with it.
    near = table[at_20]
    assert (near["estimate"] > near["true"]).all()
    assert ((near["lower"] - near["true"]).abs() < 0.002).all()
    wide = _rows(namespace, 5.0)
    assert wide.loc[5.0, "estimate"] < wide.loc[15.0, "estimate"] < wide.loc[20.0, "estimate"]
    assert 0.17 < wide.loc[20.0, "estimate"] - wide.loc[5.0, "estimate"] < 0.19, "about 18 points"
    true_rise = wide.loc[20.0, "true"] - wide.loc[5.0, "true"]
    assert 0.14 < true_rise < 0.16, "the true rise is about 15"
    narrow = _rows(namespace, 2.5)
    for rows in (wide, narrow):
        assert np.all(np.diff(rows.loc[5.0:20.0, "estimate"]) > 0), "both series rise to 20%"
        assert (rows.loc[25.0:30.0, "lower"] < rows.loc[20.0, "estimate"]).all(), "within bars"
    assert not namespace["caught"], "the fit raised no warning"


def _check_band(namespace: dict[str, Any]) -> None:
    """The band separates 5% from 20% at both bandwidths; the wider window is narrower to 25%."""
    result = namespace["result"]
    assert result.band.critical_value > result.band.pointwise_critical_value
    wide, narrow = _rows(namespace, 5.0), _rows(namespace, 2.5)
    assert wide.loc[5.0, "band_upper"] < wide.loc[20.0, "band_lower"]
    assert narrow.loc[5.0, "band_upper"] < narrow.loc[20.0, "band_lower"]
    width = namespace["width"]
    assert (width.loc[5.0:25.0, 5.0] < width.loc[5.0:25.0, 2.5]).all(), "narrower up to 25%"


def _check_assessment(namespace: dict[str, Any]) -> None:
    """The quick checks pass or describe, and the outcome model's slope is near 1."""
    report = namespace["result"].validate()
    statuses = {item.name: item.status.value for item in report.items}
    assert statuses == {
        "score_equations": "passed",
        "support": "completed",
        "nuisance_models": "completed",
    }
    model = namespace["outcome_model"]
    assert abs(model["calibration_slope"] - 1.0) < 2 * model["calibration_slope_se"]


def _check_support(namespace: dict[str, Any]) -> None:
    """No pair is below the floor; the closest is 30% at bandwidth 5; every label is supported."""
    support, result, curve = namespace["support"], namespace["result"], namespace["curve"]
    assert not support["low_support"].any()
    assert set(support["status"]) == {"influence_curve"}
    closest = support.loc[support["kish"].idxmin()]
    assert (closest["bandwidth"], closest["dose"]) == (5.0, 30.0)
    assert curve.support_floor == 50.0 < closest["kish"] < 75.0, "68.1 effective rows"
    assert result.quadrature_shift < 0.05, "below the level at which the fit warns"
    assert set(support["label"]) == {"supported"}
    assert result.band.support.label == "supported"
    assert figure_title(namespace["figure"]) == "Smoothed dose-response curve"


def _check_weights(namespace: dict[str, Any]) -> None:
    """The weight is non-negative in the middle and signed at the end of the range."""
    curve, doses = namespace["curve"], namespace["doses"]
    assert curve.weight(5.0, 15.0, doses).min() >= 0.0
    assert -0.01 < curve.weight(5.0, 30.0, doses).min() < 0.0
    end = curve.weight(5.0, 32.0, doses)
    middle = curve.weight(5.0, 15.0, doses)
    assert abs(end.min() + 0.92 / 5.0) < 0.005, "the reference's -0.92/h at the end"
    ratio = np.trapezoid(end**2, doses) / np.trapezoid(middle**2, doses)
    assert 7.3 < ratio < 7.7
    # The 5% window at bandwidth 5 is cut at the range's lower end.
    low = curve.weight(5.0, 5.0, doses)
    assert doses[low != 0.0].min() == curve.support[0]


def check(namespace: dict[str, Any]) -> None:
    """The dose-response tutorial's claims hold at the displayed sample size."""
    _check_data(namespace)
    _check_declaration(namespace)
    _check_estimate(namespace)
    _check_band(namespace)
    _check_assessment(namespace)
    _check_support(namespace)
    _check_weights(namespace)
    _check_probes()
