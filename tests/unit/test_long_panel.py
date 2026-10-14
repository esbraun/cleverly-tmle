"""The long-panel adapter: ``LongitudinalData.from_long_panel`` and its inverse.

A long panel holds one row per unit and decision time.  The adapter reshapes it into the
wide trajectory representation and hands that to ``from_frame``, so the tests here pin
three things: the reshape round-trips, every refusal that only a long panel can express
fires, and a fit on the adapted panel is the fit on the equivalent wide frame.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import CausalStudy, LongitudinalTreatment, LongPanel, RegimeContrast
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import DataError
from cleverly.interventions import Stochastic
from cleverly.longitudinal import (
    LTMLE,
    SDR,
    DynamicRegimen,
    LongitudinalData,
    Regimen,
    Repeated,
    each_node,
)

ROLES: dict[str, Any] = {
    "unit": "customer",
    "time": "week",
    "treatment": "offer",
    "baseline": ["age"],
    "time_varying": ["eng"],
    "censoring": "active",
    "outcome": "bought",
}


def long_panel(n: int = 300, *, n_times: int = 3, seed: int = 0) -> pd.DataFrame:
    """A long panel with confounded offers, irregular censoring and string labels.

    Weeks are labelled 10, 20, 30 so that a node index and a decision-time label differ.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n):
        age = rng.normal()
        eng = rng.normal() + 0.3 * age
        offers = 0
        for t in range(1, n_times + 1):
            offer = "offer" if rng.random() < 1 / (1 + np.exp(-(0.4 * eng - 0.2))) else "none"
            offers += offer == "offer"
            stays = rng.random() < 0.9 or t == n_times
            rows.append(
                {
                    "customer": f"c{unit:04d}",
                    "week": 10 * t,
                    "age": age,
                    "eng": eng,
                    "offer": offer,
                    "active": float(stays),
                }
            )
            if not stays:
                break
            eng = 0.6 * eng + 0.5 * (offer == "offer") + rng.normal(scale=0.7)
        else:
            p = 1 / (1 + np.exp(-(0.5 * eng + 0.4 * offers - 0.6 + 0.3 * age)))
            bought = float(rng.random() < p)
            for row in rows[-n_times:]:
                row["bought"] = bought
            continue
    frame = pd.DataFrame(rows)
    if "bought" not in frame:
        frame["bought"] = np.nan
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def wide_frame(panel: pd.DataFrame, n_times: int = 3) -> pd.DataFrame:
    """The equivalent wide frame, built by hand rather than by the adapter."""
    units = sorted(panel["customer"].unique())
    out: dict[str, Any] = {"customer": units}
    indexed = panel.set_index(["customer", "week"])
    first = panel.sort_values("week").groupby("customer").first()
    out["age"] = first.loc[units, "age"].to_numpy()
    out["bought"] = panel.groupby("customer")["bought"].first().loc[units].to_numpy()
    for t in range(1, n_times + 1):
        for column in ("eng", "offer", "active"):
            values = [
                indexed.loc[(unit, 10 * t), column] if (unit, 10 * t) in indexed.index else None
                for unit in units
            ]
            out[f"{column}@{t}"] = np.asarray(values, dtype=object if column == "offer" else float)
    return pd.DataFrame(out)


WIDE_ROLES: dict[str, Any] = {
    "outcome": "bought",
    "treatment": ["offer@1", "offer@2", "offer@3"],
    "baseline": ["age"],
    "time_varying": [["eng@1"], ["eng@2"], ["eng@3"]],
    "censoring": ["active@1", "active@2", "active@3"],
}


def build(frame: Any, **overrides: Any) -> LongitudinalData:
    return LongitudinalData.from_long_panel(frame, **{**ROLES, **overrides})


def canonical(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["customer", "week"]).reset_index(drop=True)


def learners() -> dict[str, Any]:
    return {
        "outcome_learner": LogisticRegression(max_iter=1000),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "n_folds": 2,
        "random_state": 0,
    }


def engaged(history: Any, t: int) -> Any:
    """Offer at node ``t`` when that node's engagement is positive."""
    return np.where(np.asarray(history[f"eng@{t}"]) > 0, "offer", "none")


# ------------------------------------------------------------------ round trip


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_round_trip_with_irregular_censoring_and_unsorted_rows(backend: str) -> None:
    panel = long_panel()
    assert panel["week"].is_monotonic_increasing is False
    left = panel.groupby("customer")["week"].max()
    assert {10, 20, 30} <= set(left)  # units leave at every node
    source = panel if backend == "pandas" else pl.from_pandas(panel)
    data = build(source)
    assert data.n == panel["customer"].nunique() and data.n_times == 3
    assert data.treatment_levels == (("none", "offer"),) * 3
    assert data.panel is not None and data.panel.decision_times == (10, 20, 30)
    back = data.to_long_panel()
    if backend == "polars":
        assert isinstance(back, pl.DataFrame)
        back = back.to_pandas()
    expected = canonical(panel)[list(back.columns)]
    pd.testing.assert_frame_equal(back, expected, check_dtype=False)


def test_unsorted_input_gives_the_sorted_container() -> None:
    panel = long_panel(n=120)
    shuffled, ordered = build(panel), build(canonical(panel))
    np.testing.assert_array_equal(shuffled.unit_ids, ordered.unit_ids)
    np.testing.assert_array_equal(shuffled.treatment, ordered.treatment)
    np.testing.assert_array_equal(shuffled.uncensored, ordered.uncensored)
    for left, right in zip(shuffled.time_varying, ordered.time_varying, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(shuffled.outcome, ordered.outcome)
    assert list(shuffled.unit_ids) == sorted(panel["customer"].unique())


def test_a_vector_node_round_trips_and_names_its_components() -> None:
    panel = long_panel(n=150)
    rng = np.random.default_rng(3)
    panel["channel"] = rng.choice(["email", "app"], size=len(panel))
    data = build(panel, treatment=["offer", "channel"])
    assert data.treatment_names[1] == "offer@2+channel@2"
    assert data.policy_names(3)[-2:] == ("offer@1+channel@1", "offer@2+channel@2")
    back = data.to_long_panel()
    expected = canonical(panel)[list(back.columns)]
    pd.testing.assert_frame_equal(back, expected, check_dtype=False)


def test_a_survival_event_round_trips() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for unit in range(200):
        w = rng.normal()
        for t in (1, 2, 3):
            event = float(rng.random() < 0.15)
            stays = 1.0 if event or rng.random() < 0.9 or t == 3 else 0.0
            rows.append(
                {
                    "id": unit,
                    "t": t,
                    "w": w,
                    "a": int(rng.random() < 0.5),
                    "c": stays,
                    "y": np.nan if stays == 0.0 else event,
                    "l": rng.normal(),
                }
            )
            if event or stays == 0.0:
                break
    panel = pd.DataFrame(rows)
    data = LongitudinalData.from_long_panel(
        panel,
        unit="id",
        time="t",
        treatment="a",
        baseline=["w"],
        time_varying=["l"],
        censoring="c",
        event="y",
    )
    assert data.event is not None and data.event[:, -1].any()
    back = data.to_long_panel()
    pd.testing.assert_frame_equal(
        back,
        panel.sort_values(["id", "t"]).reset_index(drop=True)[list(back.columns)],
        check_dtype=False,
    )


def test_cluster_labels_come_back_and_follow_a_subset() -> None:
    panel = long_panel(n=120)
    panel["store"] = panel["customer"].str[-1].map(lambda d: f"s{int(d) % 4}")
    data = build(panel, cluster="store")
    back = data.to_long_panel()
    pd.testing.assert_series_equal(
        back["store"], canonical(panel)["store"], check_dtype=False, check_names=False
    )
    keep = np.flatnonzero(np.isin(data.cluster, [1, 3]))
    sub = data.subset(keep)
    np.testing.assert_array_equal(sub.unit_ids, np.asarray(data.unit_ids)[keep])
    assert sub.panel is not None and sub.panel.cluster_labels == ("s1", "s3")
    labels = sub.to_long_panel()["store"]
    assert set(labels) == {"s1", "s3"}


# ------------------------------------------------------------------ refusals


def _refused(panel: pd.DataFrame, match: str, **overrides: Any) -> None:
    with pytest.raises(DataError, match=match):
        build(panel, **overrides)


def test_refuses_a_missing_unit_or_time() -> None:
    panel = long_panel(n=60)
    for column in ("customer", "week"):
        broken = panel.astype({column: object})
        broken.loc[0, column] = None
        _refused(broken, rf"column '{column}' is missing on 1 row")


def test_refuses_unsortable_identifiers() -> None:
    panel = long_panel(n=60).astype({"customer": object})
    panel.loc[panel["customer"] == "c0001", "customer"] = 1
    _refused(panel, "do not sort")


def test_refuses_a_duplicated_row() -> None:
    panel = long_panel(n=60)
    _refused(pd.concat([panel, panel.iloc[[0]]]), r"appear on more than one row")


def test_refuses_a_time_outside_the_declared_grid() -> None:
    _refused(long_panel(n=60), r"times= does not list", times=[10, 20])


def test_refuses_late_entry() -> None:
    panel = long_panel(n=60)
    unit = canonical(panel).groupby("customer").size().idxmax()
    broken = panel[~((panel["customer"] == unit) & (panel["week"] == 10))]
    _refused(broken, r"no row at the first decision time")


def test_refuses_a_gap_in_decision_time() -> None:
    panel = long_panel(n=60)
    unit = canonical(panel).groupby("customer").size().idxmax()
    broken = panel[~((panel["customer"] == unit) & (panel["week"] == 20))]
    _refused(broken, r"skip a decision time.*no row at 20")


def test_refuses_rows_that_stop_without_leaving() -> None:
    panel = long_panel(n=60)
    unit = canonical(panel).groupby("customer").size().idxmax()
    broken = panel[~((panel["customer"] == unit) & (panel["week"] == 30))]
    _refused(broken, r"neither left nor had the event")
    complete = panel.groupby("customer").filter(lambda rows: len(rows) == 3)
    shortened = complete[~((complete["customer"] == unit) & (complete["week"] == 30))]
    _refused(shortened.drop(columns="active"), r"Declare censoring=", censoring=None)


def test_refuses_rows_after_censoring() -> None:
    panel = long_panel(n=60)
    unit = canonical(panel).groupby("customer").size().idxmax()
    broken = panel.copy()
    broken.loc[(broken["customer"] == unit) & (broken["week"] == 10), "active"] = 0.0
    _refused(broken, r"rows after they left the study")


def test_refuses_an_event_on_a_censored_row() -> None:
    panel = pd.DataFrame(
        {
            "id": np.repeat(np.arange(40), 2),
            "t": np.tile([1, 2], 40),
            "w": np.repeat(np.linspace(-1, 1, 40), 2),
            "a": np.tile([0, 1], 40),
            "c": 1.0,
            "y": 0.0,
        }
    )
    panel.loc[1, "y"] = 1.0
    panel.loc[2, ["c", "y"]] = [0.0, 1.0]
    panel = panel.drop(index=3)
    with pytest.raises(DataError, match=r"censoring 'c' is 0.*unit 1 at 1"):
        LongitudinalData.from_long_panel(
            panel, unit="id", time="t", treatment="a", baseline=["w"], censoring="c", event="y"
        )


def test_the_wide_path_refuses_an_event_on_a_censored_row() -> None:
    """The same order, ``A_t, C_t, Y_t``, on the wide path, which used to drop the event."""
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame(
        {
            "W": rng.normal(size=n),
            "A1": rng.integers(0, 2, n).astype(float),
            "C1": 1.0,
            "Y1": 0.0,
            "A2": rng.integers(0, 2, n).astype(float),
            "C2": 1.0,
            "Y2": 0.0,
        }
    )
    frame.loc[0, ["C1", "Y1"]] = [0.0, 1.0]
    frame.loc[0, ["A2", "C2", "Y2"]] = np.nan
    columns = {
        "outcome": ["Y1", "Y2"],
        "treatment": ["A1", "A2"],
        "baseline": ["W"],
        "censoring": ["C1", "C2"],
    }
    with pytest.raises(DataError, match=r"'Y1' holds a value for 1 unit.*A_t, C_t, Y_t"):
        LongitudinalData.from_frame(frame, **columns)
    frame.loc[0, "Y1"] = np.nan
    LongitudinalData.from_frame(frame, **columns)


@pytest.mark.parametrize(
    ("column", "role", "match"),
    [
        ("age", {}, r"'age' holds more than one value.*time-varying"),
        ("store", {"cluster": "store"}, r"'store' holds more than one value"),
        ("w", {"weights": "w"}, r"'w' holds more than one value"),
        ("bought", {}, r"'bought' holds more than one value.*event=.*Repeated"),
    ],
)
def test_refuses_a_unit_level_column_that_varies(
    column: str, role: dict[str, str], match: str
) -> None:
    panel = long_panel(n=60)
    panel["store"] = "s" + panel["customer"].str[-1]
    panel["w"] = 1.0 + panel["age"].abs()
    unit = canonical(panel).groupby("customer").size().idxmax()
    rows = (panel["customer"] == unit) & (panel["week"] == 30)
    panel.loc[rows, column] = "other" if column == "store" else 99.0
    _refused(panel, match, **role)


def test_refuses_a_repeated_outcome() -> None:
    """A long panel takes one end-of-study outcome column; a repeated outcome is wide only."""
    _refused(long_panel(n=60), r"repeated outcome.*wide frame", outcome=Repeated(["bought"]))


def test_refuses_a_baseline_recorded_after_the_first_row() -> None:
    panel = long_panel(n=60)
    unit = canonical(panel).groupby("customer").size().idxmax()
    panel.loc[(panel["customer"] == unit) & (panel["week"] == 10), "age"] = np.nan
    _refused(panel, r"'age' is missing on the first row.*first recorded at 20")


@pytest.mark.parametrize("column", ["customer", "week", "bought"])
def test_refuses_a_column_in_two_roles(column: str) -> None:
    _refused(long_panel(n=60), rf"'{column}' is declared as .* and as ", time_varying=[column])


def test_refuses_the_outcome_under_another_name() -> None:
    panel = long_panel(n=120)
    panel["converted"] = panel["bought"]
    _refused(panel, r"'converted' equals the outcome", time_varying=["eng", "converted"])
    _refused(
        panel.fillna({"converted": 0.0}),
        r"'converted' equals the outcome",
        baseline=["age", "converted"],
    )


def test_refuses_the_event_under_another_name() -> None:
    rng = np.random.default_rng(1)
    rows = []
    for unit in range(60):
        for t in (1, 2):
            event = float(rng.random() < 0.3)
            rows.append(
                {
                    "id": unit,
                    "t": t,
                    "w": unit / 60,
                    "a": int(rng.random() < 0.5),
                    "y": event,
                    "l": rng.normal(),
                }
            )
            if event:
                break
    panel = pd.DataFrame(rows)
    panel["copy"] = panel["y"]
    with pytest.raises(DataError, match=r"'copy' equals the event 'y'"):
        LongitudinalData.from_long_panel(
            panel,
            unit="id",
            time="t",
            treatment="a",
            baseline=["w"],
            time_varying=["l", "copy"],
            event="y",
        )


def test_a_leading_lag_starts_late_and_a_later_outage_is_refused() -> None:
    panel = long_panel(n=120)
    panel["previous"] = np.where(panel["week"] == 10, np.nan, panel["eng"] - 0.1)
    data = build(panel, time_varying=["eng", "previous"])
    assert data.history_names(1) == ("age", "eng@1")
    assert data.history_names(2) == ("age", "eng@1", "eng@2", "previous@2")
    assert data.panel is not None and data.panel.omitted == (("previous", 1),)
    assert "left out: previous@1" in data.panel.describe()
    outage = panel.copy()
    outage.loc[outage["week"] == 30, "previous"] = np.nan
    _refused(
        outage, r"'previous' has no value at decision time 30", time_varying=["eng", "previous"]
    )


def test_refuses_a_treatment_held_within_every_unit() -> None:
    panel = long_panel(n=60)
    first = panel.sort_values("week").groupby("customer")["offer"].first()
    panel["offer"] = panel["customer"].map(first)
    _refused(panel, r"never changes within a unit.*held design")


def test_refuses_a_generated_name_that_collides() -> None:
    panel = long_panel(n=60)
    panel["eng@2"] = 0.0
    _refused(panel, r"'eng@2' is also the generated name of 'eng'")


# ------------------------------------------------------------------ policy history


def test_a_rule_reads_its_own_node_and_never_a_later_one() -> None:
    data = build(long_panel(n=120))
    assert data.history_names(2) == ("age", "eng@1", "eng@2")
    assert data.policy_names(2) == ("age", "eng@1", "eng@2", "offer@1")
    written = DynamicRegimen("engaged", each_node(engaged, 3), rule_kind="known")
    by_hand = DynamicRegimen(
        "by hand",
        tuple(
            (lambda h, t=t: np.where(np.asarray(h[f"eng@{t}"]) > 0, "offer", "none"))
            for t in (1, 2, 3)
        ),
        rule_kind="known",
    )
    lagged = DynamicRegimen(
        "lagged",
        (written.plan[0], *each_node(lambda h, t: engaged(h, t - 1), 3)[1:]),
        rule_kind="known",
    )
    np.testing.assert_array_equal(written.assignment(data), by_hand.assignment(data))
    # The mutation that reads the previous node assigns different arms.
    assert not np.array_equal(written.assignment(data), lagged.assignment(data))
    future = DynamicRegimen("future", (lambda h: engaged(h, 2), "none", "none"), rule_kind="known")
    with pytest.raises(DataError, match=r"rule at time 1 .* raised KeyError"):
        future.assignment(data)


def test_a_screened_out_node_raises_rather_than_read_an_earlier_node() -> None:
    panel = long_panel(n=120)
    panel["visits"] = np.random.default_rng(7).poisson(2.0, len(panel)).astype(float)
    panel.loc[panel["week"] == 20, "eng"] = 1.0
    with pytest.warns(Warning, match="eng@2"):
        data = build(panel, time_varying=["eng", "visits"])
    assert "eng@2" not in data.history_names(2)
    regimen = DynamicRegimen("engaged", each_node(engaged, 3), rule_kind="known")
    with pytest.raises(DataError, match=r"rule at time 2 .* raised KeyError: 'eng@2'"):
        regimen.assignment(data)


def test_each_node_names_and_refusals() -> None:
    rules = each_node(engaged, 2)
    assert [rule.__name__ for rule in rules] == ["engaged@1", "engaged@2"]
    assert each_node(lambda h, t: h, 1)[0].__name__ == "d@1"
    with pytest.raises(DataError, match="positive number of nodes"):
        each_node(engaged, 0)


# ------------------------------------------------------------------ fits


def _fit_both(estimator: Any, regimens: Any) -> tuple[Any, Any]:
    panel = long_panel(n=400, seed=2)
    adapted = estimator(regimens, **learners()).fit(build(panel))
    wide = estimator(regimens, **learners()).fit(
        LongitudinalData.from_frame(wide_frame(panel), **WIDE_ROLES)
    )
    return adapted, wide


def test_ltmle_on_the_adapted_panel_equals_the_wide_fit() -> None:
    regimens = [
        Regimen("always", ("offer",) * 3),
        DynamicRegimen("engaged", each_node(engaged, 3), rule_kind="known"),
    ]
    adapted, wide = _fit_both(LTMLE, regimens)
    # A nonzero targeting update, so the comparison is not between two plug-ins.
    assert any(
        np.any(np.abs(step.fluctuation.epsilon) > 1e-6)
        for fit in adapted.fits.values()
        for step in fit.steps
    )
    left, right = adapted.to_frame(), wide.to_frame()
    pd.testing.assert_frame_equal(left, right, rtol=1e-12)
    for name, curve in adapted.influence_curves.items():
        np.testing.assert_allclose(curve, wide.influence_curves[name], rtol=1e-12, atol=1e-14)
    assert "long panel: unit column 'customer', decision times 10, 20, 30" in str(adapted.summary())


def test_sdr_on_the_adapted_panel_equals_the_wide_fit() -> None:
    adapted, wide = _fit_both(SDR, [Regimen("always", ("offer",) * 3)])
    pd.testing.assert_frame_equal(adapted.to_frame(), wide.to_frame(), rtol=1e-12)


def test_the_study_path_matches_the_direct_fit() -> None:
    panel = long_panel(n=400, seed=2)
    design = LongPanel(**ROLES)
    estimand = RegimeContrast({"never": ("none",) * 3, "always": ("offer",) * 3}, reference="never")
    study = CausalStudy(panel, design=design)
    via_study = study.estimate(estimand, **learners())
    direct = LTMLE(
        [Regimen("never", ("none",) * 3), Regimen("always", ("offer",) * 3)], **learners()
    ).fit(build(panel))
    difference = direct["ey_regimen[always]"].psi - direct["ey_regimen[never]"].psi
    assert via_study["ate_regimen[always vs never]"].psi == pytest.approx(difference, rel=1e-10)
    prepared = build(panel)
    assert CausalStudy(prepared, design=design).data is prepared
    with pytest.raises(DataError, match=r"built with time_varying=\('eng',\)"):
        CausalStudy(prepared, design=LongPanel(**{**ROLES, "time_varying": []}))
    wide = LongitudinalData.from_frame(wide_frame(panel), **WIDE_ROLES)
    with pytest.raises(DataError, match="built from a wide frame"):
        CausalStudy(wide, design=design)
    with pytest.raises(DataError, match="exactly one of outcome="):
        LongPanel(**{**ROLES, "event": "bought"})
    assert isinstance(LongitudinalTreatment(**WIDE_ROLES), LongitudinalTreatment)


# ------------------------------------------------------------------ persistence


def test_a_callable_shared_by_two_regimens_stays_shared() -> None:
    def rule(history: Any) -> Any:
        return np.where(np.asarray(history["eng@1"]) > 0, "offer", "none")

    regimens = [
        DynamicRegimen("one", (rule, "none", "none"), rule_kind="known"),
        DynamicRegimen("two", (rule, "offer", "offer"), rule_kind="known"),
    ]
    result = LTMLE(regimens, **learners()).fit(build(long_panel(n=200)))
    restored = loads(dumps(result))
    first, second = restored.config.regimens
    assert first.plan[0] is second.plan[0]


_SAVE = """
import json, sys
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE, DynamicRegimen, Regimen, each_node
from tests.unit.test_long_panel import build, long_panel
if sys.argv[2] == "joblib":
    import joblib
    import cleverly.estimators.serialize as serialize
    serialize.cloudpickle.dump = joblib.dump
threshold = 0.2
def mixing(strength):
    def density(frame):
        p = 1 / (1 + np.exp(-strength * np.asarray(frame["eng@1"])))
        return np.column_stack([1 - p, p])
    return density
regimens = [
    Regimen("never", ("none",) * 3),
    DynamicRegimen(
        "inline",
        (lambda h: np.where(np.asarray(h["eng@1"]) > threshold, "offer", "none"), "none", "none"),
        rule_kind="known",
    ),
    DynamicRegimen(
        "mixed", (Stochastic(mixing(1.5), "q", density_kind="known"), "none", "none")
    ),
    DynamicRegimen(
        "engaged",
        each_node(lambda h, t: np.where(np.asarray(h[f"eng@{t}"]) > 0, "offer", "none"), 3),
        rule_kind="known",
    ),
]
result = LTMLE(
    regimens,
    outcome_learner=LogisticRegression(max_iter=1000),
    pseudo_learner=LinearRegression(),
    treatment_learner=LogisticRegression(max_iter=1000),
    censoring_learner=LogisticRegression(max_iter=1000),
    n_folds=2,
    random_state=0,
).fit(build(long_panel(n=200)))
result.save(sys.argv[1])
np.savez(
    sys.argv[1] + ".npz",
    **{
        regimen.label: np.asarray(regimen.assignment(result.data), dtype=str)
        for regimen in result.config.regimens
        if isinstance(regimen, DynamicRegimen)
    },
    density=regimens[2].policy_density(result.data, 1),
)
print(json.dumps(str(result.summary()).splitlines()))
"""

_LOAD = """
import sys
import numpy as np
import cleverly
from cleverly.longitudinal import DynamicRegimen
result = cleverly.load(sys.argv[1])
saved = np.load(sys.argv[1] + ".npz")
for regimen in result.config.regimens:
    if isinstance(regimen, DynamicRegimen):
        np.testing.assert_array_equal(
            np.asarray(regimen.assignment(result.data), dtype=str), saved[regimen.label]
        )
np.testing.assert_array_equal(
    result.config.regimens[2].policy_density(result.data, 1), saved["density"]
)
back = result.data.to_long_panel()
assert back.shape[0] > 0 and list(back.columns)[:2] == ["customer", "week"]
print("restored")
"""


def _run(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
        timeout=300,
    )


def test_policies_defined_in_main_survive_a_save_in_a_fresh_interpreter(tmp_path: Path) -> None:
    """Fit and save in one interpreter, load and re-evaluate in another.

    The rules live in ``__main__`` of the saving process, so pickling them by reference
    would fail to load anywhere else.  The control runs the same script on the old
    ``joblib.dump`` path, which cannot save the inline lambda.
    """
    root = Path(__file__).resolve().parents[2]
    artifact = str(tmp_path / "panel.joblib")
    saved = _run(_SAVE, artifact, "cloudpickle", cwd=root)
    assert saved.returncode == 0, saved.stderr
    assert "long panel: unit column 'customer'" in saved.stdout
    loaded = _run(_LOAD, artifact, cwd=root)
    assert loaded.returncode == 0, loaded.stderr
    assert loaded.stdout.strip().endswith("restored")
    control = _run(_SAVE, str(tmp_path / "control.joblib"), "joblib", cwd=root)
    assert control.returncode != 0
    assert "not serializable" in control.stderr


def test_a_stochastic_closure_round_trips_in_process() -> None:
    def mixing(strength: float) -> Any:
        def density(frame: Any) -> Any:
            p = 1 / (1 + np.exp(-strength * np.asarray(frame["eng@1"])))
            return np.column_stack([1 - p, p])

        return density

    regimen = DynamicRegimen(
        "mixed", (Stochastic(mixing(2.0), "q", density_kind="known"), "none", "none")
    )
    result = LTMLE([regimen], **learners()).fit(build(long_panel(n=200)))
    restored = loads(dumps(result))
    np.testing.assert_array_equal(
        restored.config.regimens[0].policy_density(restored.data, 1),
        regimen.policy_density(result.data, 1),
    )


# ------------------------------------------------------------------ time order


def _relabelled(labels: dict[int, str]) -> pd.DataFrame:
    panel = long_panel(n=120)
    panel["week"] = panel["week"].map(labels)
    return panel


def test_a_declared_grid_is_taken_in_the_order_given() -> None:
    """Labels whose sort order is not their time order keep the declared order.

    ``"jan" < "feb"`` is false alphabetically, so a grid that sorted the labels would put
    February's engagement into January's history.
    """
    labels = {10: "jan", 20: "feb", 30: "mar"}
    panel = _relabelled(labels)
    data = build(panel, times=["jan", "feb", "mar"])
    assert data.panel is not None and data.panel.decision_times == ("jan", "feb", "mar")
    reference = build(long_panel(n=120))
    for left, right in zip(data.time_varying, reference.time_varying, strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(data.treatment, reference.treatment)
    back = data.to_long_panel()
    assert list(back.loc[back["customer"] == back["customer"].iloc[0], "week"])[:2] == [
        "jan",
        "feb",
    ]
    design = LongPanel(**{**ROLES, "times": ["jan", "feb", "mar"]})
    assert CausalStudy(data, design=design).data is data


def test_labels_that_do_not_sort_in_time_need_a_declared_grid() -> None:
    panel = _relabelled({10: "w1", 20: "w2", 30: "w10"})
    _refused(panel, r"not numbers, dates or durations.*Pass times=")
    data = build(panel, times=["w1", "w2", "w10"])
    assert data.panel is not None and data.panel.decision_times == ("w1", "w2", "w10")


def test_dates_sort_in_time() -> None:
    panel = long_panel(n=60)
    panel["week"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(panel["week"], unit="D")
    data = build(panel)
    assert data.panel is not None
    assert [stamp.day for stamp in data.panel.decision_times] == [11, 21, 31]


def test_refuses_an_outcome_copy_recorded_on_one_row_only() -> None:
    panel = long_panel(n=120)
    last = panel.groupby("customer")["week"].transform("max") == panel["week"]
    panel["final"] = np.where(last, panel["bought"], np.nan)
    _refused(panel, r"'final' equals the outcome", time_varying=["eng", "final"])
    first = panel["week"] == 10
    panel["early"] = np.where(first, panel["bought"], np.nan)
    _refused(panel, r"'early' equals the outcome", baseline=["age", "early"])


def test_durations_sort_in_time() -> None:
    panel = long_panel(n=60)
    panel["week"] = pd.to_timedelta(panel["week"], unit="D")
    data = build(panel)
    assert data.panel is not None
    assert [stamp.days for stamp in data.panel.decision_times] == [10, 20, 30]


def test_refuses_a_declared_grid_that_is_not_increasing() -> None:
    _refused(long_panel(n=60), r"not increasing", times=[30, 20, 10])


def test_refuses_a_status_column_that_equals_the_outcome_at_one_node() -> None:
    panel = long_panel(n=120)
    last = panel.groupby("customer")["week"].transform("max") == panel["week"]
    panel["status"] = np.where(last & panel["bought"].notna(), panel["bought"], 0.0)
    _refused(
        panel,
        r"'status' equals the outcome 'bought' at decision time 30",
        time_varying=["eng", "status"],
    )


def test_a_prepared_panel_must_declare_the_same_known_probabilities() -> None:
    panel = long_panel(n=120)
    panel["p_offer"] = 0.5
    panel["p_none"] = 0.5
    known = {"offer": "p_offer", "none": "p_none"}
    data = build(panel, treatment_probabilities=known)
    reordered = LongPanel(**ROLES, treatment_probabilities={"none": "p_none", "offer": "p_offer"})
    assert CausalStudy(data, design=reordered).data is data
    with pytest.raises(DataError, match="built with treatment_probabilities="):
        CausalStudy(data, design=LongPanel(**ROLES))
    with pytest.raises(DataError, match="built with treatment_probabilities="):
        CausalStudy(build(panel), design=LongPanel(**ROLES, treatment_probabilities=known))
