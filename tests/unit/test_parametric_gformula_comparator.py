"""The ``pygformula`` comparator of ``parametric-gformula``: fixtures, ``cleverly`` side, verdicts.

The container run needs Docker and is not part of the fast tier.  These tests pin everything
around it: the panels, the long form the runner reads, the names the two sides share, the
test-only hazard learner, and the verdicts, each with a deliberate-mutation control.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from tests.canonical.parametric_gformula_study import compare_pygformula as driver
from tests.canonical.parametric_gformula_study import run_pygformula as runner
from tests.studies import parametric_gformula_fixtures as fx

STUDY = Path(driver.__file__).resolve().parent


@pytest.fixture(scope="module")
def coefficients() -> dict[str, pd.DataFrame]:
    return {name: fx.cleverly_coefficients(name, fx.sample(name)) for name in fx.FIXTURES}


# ---------------------------------------------------------------------------------- panels


def test_a_panel_is_deterministic_and_has_the_declared_shape() -> None:
    for name, fixture in fx.FIXTURES.items():
        first, second = fx.sample(name), fx.sample(name)
        pd.testing.assert_frame_equal(first, second)
        assert len(first) == fx.N
        assert first["W"].notna().all()
        assert fixture.n_times in (2, 3)


def test_the_inputs_are_written_byte_for_byte_the_same_twice(tmp_path: Path) -> None:
    first = fx.write_inputs(tmp_path / "a")
    second = fx.write_inputs(tmp_path / "b")
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()
    assert b"\r" not in first[1].read_bytes()


def test_a_survival_panel_ends_a_unit_at_its_event() -> None:
    wide = fx.sample("survival")
    long = fx.to_long("survival", wide)
    last = long.groupby("id")["t"].max() + 1
    event = wide[["E1", "E2", "E3"]].to_numpy()
    expected = np.where((event == 1).any(axis=1), (event == 1).argmax(axis=1) + 1, 3)
    np.testing.assert_array_equal(last.to_numpy(), expected)
    assert long["Y"].notna().all()


def test_a_covariate_panel_puts_the_outcome_on_the_last_row_only() -> None:
    long = fx.to_long("cov_gaussian", fx.sample("cov_gaussian"))
    assert long.groupby("id").size().eq(2).all()
    assert long.loc[long["t"] == 0, "Y"].isna().all()
    assert long.loc[long["t"] == 1, "Y"].notna().all()


def test_the_competing_event_acts_at_node_two_only_and_masks_the_event() -> None:
    wide = fx.sample("competing")
    long = fx.to_long("competing", wide)
    assert (long.loc[long["t"] == 0, "D"] == 0).all()
    second = long.loc[long["t"] == 1]
    assert second["D"].sum() > 0, "the competing event must occur, or the fixture tests nothing"
    assert second.loc[second["D"] == 1, "Y"].isna().all()
    assert second.loc[second["D"] == 0, "Y"].notna().all()


# ------------------------------------------------------------------------- the cleverly side


EXPECTED_MODELS = {
    "cov_gaussian": {"Z@2": 6, "B@2": 6, "Y@2": 8},
    "cov_binary": {"Z@2": 6, "B@2": 6, "Y@2": 8},
    "survival": {"E@1": 3, "E@2": 4, "E@3": 5},
    "competing": {"Y@1": 3, "Y@2": 4, "D@2": 4},
}


@pytest.mark.parametrize("name", list(fx.FIXTURES))
def test_the_cleverly_side_names_the_models_the_runner_names(
    name: str, coefficients: dict[str, pd.DataFrame]
) -> None:
    table = coefficients[name]
    assert table.groupby("model").size().to_dict() == EXPECTED_MODELS[name]
    assert table["value"].notna().all()


def test_the_coefficients_recover_a_nonzero_effect_of_the_law() -> None:
    """The outcome coefficient of ``A2`` is 0.6 in the law: a zeroed or shifted column fails."""
    table = fx.cleverly_coefficients("cov_gaussian", fx.sample("cov_gaussian"))
    outcome = table.loc[table["model"] == "Y@2"].set_index("term")["value"]
    assert outcome["A2"] == pytest.approx(0.6, abs=0.15)
    assert outcome["Z2"] == pytest.approx(0.5, abs=0.15)
    sigma = table.loc[(table["model"] == "Z@2") & (table["term"] == fx.SIGMA), "value"].iloc[0]
    assert sigma == pytest.approx(0.8, abs=0.05)


def _hazard(table: pd.DataFrame, model: str, design: dict[str, np.ndarray]) -> np.ndarray:
    rows = table.loc[table["model"] == model].set_index("term")["value"]
    eta = np.full(len(next(iter(design.values()))), rows[fx.INTERCEPT])
    for term, column in design.items():
        eta = eta + rows[term] * column
    return np.asarray(expit(eta))


def test_the_survival_estimate_is_the_hazard_product_of_the_coefficient_table(
    coefficients: dict[str, pd.DataFrame],
) -> None:
    """An independent evaluation of the risk from the table the comparison reads."""
    wide = fx.sample("survival")
    table = coefficients["survival"]
    w = wide["W"].to_numpy()
    estimates = fx.cleverly_estimates("survival", wide, seed=3).set_index(["estimand", "horizon"])
    for label, value in fx.REGIMENS.items():
        a = [np.full(len(w), value)] * 3
        survival, risk = np.ones(len(w)), np.zeros(len(w))
        for node in (1, 2, 3):
            design = {"W": w, **{f"A{k}": a[k - 1] for k in range(1, node + 1)}}
            hazard = _hazard(table, f"E@{node}", design)
            risk = risk + survival * hazard
            survival = survival * (1.0 - hazard)
            reported = estimates.loc[(label, node), "value"]
            assert float(risk.mean()) == pytest.approx(reported, abs=1e-9)


def test_the_competing_estimate_is_the_sequential_form_of_the_coefficient_table(
    coefficients: dict[str, pd.DataFrame],
) -> None:
    """``P(Y at t) = S_{t-1} (1 - q_t) p_t`` with ``q`` fitted first: not the multinomial form."""
    wide = fx.sample("competing")
    table = coefficients["competing"]
    w = wide["W"].to_numpy()
    estimates = fx.cleverly_estimates("competing", wide, seed=3).set_index(["estimand", "horizon"])
    for label, value in fx.REGIMENS.items():
        a1 = a2 = np.full(len(w), value)
        p1 = _hazard(table, "Y@1", {"W": w, "A1": a1})
        design = {"W": w, "A1": a1, "A2": a2}
        p2 = _hazard(table, "Y@2", design)
        q2 = _hazard(table, "D@2", design)
        survival = 1.0 - p1
        cif2 = p1 + survival * (1.0 - q2) * p2
        assert float(p1.mean()) == pytest.approx(estimates.loc[(label, 1), "value"], abs=1e-9)
        assert float(cif2.mean()) == pytest.approx(estimates.loc[(label, 2), "value"], abs=1e-9)


def test_the_sequential_learner_returns_the_product_form_and_sums_to_one() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 2))
    died = rng.random(400) < expit(-1.0 + x[:, 0])
    event = (~died) & (rng.random(400) < expit(-0.5 + x[:, 1]))
    labels = np.where(died, 2.0, np.where(event, 1.0, 0.0))
    learner = fx.SequentialHazard().fit(x, labels)
    probabilities = learner.predict_proba(x)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert not isinstance(learner.competing_, float) and not isinstance(learner.interest_, float)
    q, p = learner.competing_.predict(x), learner.interest_.predict(x)
    np.testing.assert_allclose(probabilities[:, 2], q)
    np.testing.assert_allclose(probabilities[:, 1], (1.0 - q) * p)
    # The multinomial law would give a different split: the learner is not it.
    assert abs(probabilities[:, 1] - p).max() > 0.01


def test_the_sequential_learner_handles_a_node_with_no_competing_event() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 1))
    labels = (rng.random(200) < expit(x[:, 0])).astype(float)
    probabilities = fx.SequentialHazard().fit(x, labels).predict_proba(x)
    assert (probabilities[:, 2] == 0.0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


# ------------------------------------------------------------------------------ the verdicts


def _estimates(seeds: int, shift: float = 0.0, spread: float = 0.01) -> tuple[pd.DataFrame, ...]:
    rng = np.random.default_rng(5)
    frames = []
    for offset in (0.0, shift):
        frames.append(
            pd.DataFrame(
                {
                    "fixture": "cov_gaussian",
                    "estimand": "treat_all",
                    "horizon": 2,
                    "seed": range(seeds),
                    "value": 1.0 + offset + spread * rng.standard_normal(seeds),
                }
            )
        )
        frames[-1] = pd.concat(
            [
                frames[-1],
                frames[-1].assign(estimand="treat_none", value=lambda f: f["value"] - 1.0),
            ],
            ignore_index=True,
        )
    return frames[0], frames[1]


def test_equal_estimates_pass_and_a_shifted_series_fails() -> None:
    same = fx.estimate_rows(*_estimates(24))
    assert same["passed"].all()
    assert set(same["estimand"]) == {"treat_all", "treat_none", fx.CONTRAST}
    shifted = fx.estimate_rows(*_estimates(24, shift=0.05))
    assert not shifted.loc[shifted["estimand"] == "treat_all", "passed"].any()
    assert (shifted["ratio"] > fx.GATE_SE).any()


def test_a_gate_with_fewer_than_twenty_seeds_does_not_pass() -> None:
    assert not fx.estimate_rows(*_estimates(19))["passed"].any()


def test_a_deterministic_estimand_must_agree_to_rounding() -> None:
    cleverly = pd.DataFrame(
        {
            "fixture": "survival",
            "estimand": "treat_all",
            "horizon": 1,
            "seed": range(24),
            "value": 0.2,
        }
    )
    other = cleverly.assign(estimand="treat_none", value=0.3)
    base = pd.concat([cleverly, other], ignore_index=True)
    assert fx.estimate_rows(base, base)["deterministic"].all()
    assert fx.estimate_rows(base, base)["passed"].all()
    moved = base.assign(value=lambda f: f["value"] + 1e-6)
    verdicts = fx.estimate_rows(base, moved).set_index("estimand")["passed"]
    assert not verdicts[["treat_all", "treat_none"]].any()
    assert verdicts[fx.CONTRAST], "a common shift cancels in the contrast"


def _coefficients() -> tuple[pd.DataFrame, pd.DataFrame]:
    left = pd.DataFrame(
        {
            "fixture": "f",
            "model": ["Y@2", "Y@2", "Z@2"],
            "term": [fx.INTERCEPT, "W", fx.SIGMA],
            "value": [1.0, 0.5, 0.8],
        }
    )
    return left, left.copy()


def test_equal_coefficients_pass_and_each_kind_of_difference_fails() -> None:
    left, right = _coefficients()
    assert fx.coefficient_rows(left, right)["passed"].all()
    for change in (
        right.assign(value=lambda f: f["value"] * (1.0 + 1e-5)),  # beyond rtol
        right.iloc[:-1],  # a term the other side lacks
        pd.concat([right, right.iloc[[0]].assign(term="extra")], ignore_index=True),
    ):
        assert not fx.coefficient_rows(left, change)["passed"].all()
    near = right.assign(value=lambda f: f["value"] * (1.0 + 1e-8))
    assert fx.coefficient_rows(left, near)["passed"].all()


def test_findings_name_every_failed_row() -> None:
    left, right = _coefficients()
    bad = right.assign(value=lambda f: f["value"] + 1.0)
    estimates = fx.estimate_rows(*_estimates(24, shift=0.05))
    found = fx.findings(fx.coefficient_rows(left, bad), estimates)
    assert any(line.startswith("coefficient f/Y@2/W") for line in found)
    assert any(line.startswith("estimate cov_gaussian/treat_all@2") for line in found)
    assert fx.findings(fx.coefficient_rows(left, right), fx.estimate_rows(*_estimates(24))) == []


# ------------------------------------------------------------------------------ the runner


def test_the_runner_and_the_fixtures_declare_the_same_fixtures() -> None:
    assert set(runner.FIXTURES) == set(fx.FIXTURES)
    for name, fixture in fx.FIXTURES.items():
        assert runner.FIXTURES[name]["n_times"] == fixture.n_times


def test_the_runner_renames_terms_to_the_design_columns_of_the_other_side() -> None:
    covariates = {"A"}
    params = pd.Series(
        {
            "C(t)[0]": -1.5,
            "C(t)[0]:W": 0.4,
            "C(t)[0]:A": -0.5,
            "C(t)[0]:lag1_A": 0.0,
            "C(t)[1]": -1.4,
            "C(t)[1]:lag1_A": 0.2,
            "C(t)[2]:lag2_A": 0.3,
            "C(t)[2]:A": -0.6,
        }
    )
    rows = runner.coefficient_rows("E", params, covariates)
    named = {(row["model"], row["term"]): row["value"] for row in rows}
    assert ("E@1", "A1") in named and ("E@1", "W") in named
    assert ("E@1", "A0") not in named, "a lag before the first node is a structural zero"
    assert named[("E@2", "A1")] == 0.2
    assert named[("E@3", "A1")] == 0.3
    assert named[("E@3", "A3")] == -0.6
    pooled = runner.coefficient_rows(
        "Z", pd.Series({"Intercept": 0.1, "W": 0.4, "lag1_Z": 0.5, "Z": 0.9}), {"Z"}
    )
    assert {row["term"] for row in pooled} == {fx.INTERCEPT, "W", "Z1", "Z2"}


def test_the_runner_registers_the_shims_a_wheel_without_them_needs() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "pygformula.version" in source and "pygformula.plot" in source
    assert "sim_trunc" in source and '"sim_trunc": False' in source
    assert "nsamples" in source and '"nsamples": 0' in source


# ------------------------------------------------------------------------------- the sources


def test_the_declared_sources_pin_the_comparator() -> None:
    pinned = (STUDY / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "pygformula==1.2.0" in pinned
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][0-9A-Za-z.\-+!]*", line) for line in pinned)
    assert "pandas==2.2.3" in pinned, "pygformula 1.2.0 needs groupby.apply(include_groups=)"
    assert driver.REFERENCE.runner == "run_pygformula.py"
    assert driver.REFERENCE.mount_runner and driver.REFERENCE.interpreter == "python"
    assert {path.name for path in driver.REFERENCE.files(STUDY)} == {
        "Dockerfile",
        "run_pygformula.py",
        "requirements.txt",
    }


def test_the_gate_matches_the_declared_design() -> None:
    assert len(fx.SEEDS) >= 20
    assert fx.GATE_SE == 4.0
    assert fx.COEFFICIENT_RTOL == 1e-6
    assert dataclasses.is_dataclass(fx.Fixture)


# ---------------------------------------------------------------------- the committed comparison


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_committed_run_record_matches_the_committed_files() -> None:
    record = json.loads((STUDY / driver.RUN_RECORD).read_text(encoding="utf-8"))
    assert record["pygformula"] == driver.PYGFORMULA
    assert record["fixtures"] == list(fx.FIXTURES)
    assert record["seeds"] == list(fx.SEEDS)
    assert record["gate_se"] == fx.GATE_SE
    assert record["image_id"], "the committed comparison ran in the container"
    assert record["results_sha256"] == _sha256(STUDY / driver.RESULTS)
    assert record["sources_sha256"] == {
        path.name: _sha256(path) for path in driver.REFERENCE.files(STUDY)
    }


def test_the_committed_verdicts_recompute_from_the_committed_rows() -> None:
    pyg_coefficients, pyg_estimates = fx.read_pygformula(STUDY / driver.RESULTS)
    committed = pd.read_csv(STUDY / driver.COEFFICIENTS, keep_default_na=False, na_values=[""])
    cleverly_coefficients = committed.loc[:, ["fixture", "model", "term", "cleverly"]].rename(
        columns={"cleverly": "value"}
    )
    coefficients = fx.coefficient_rows(cleverly_coefficients, pyg_coefficients)
    assert coefficients["passed"].tolist() == committed["passed"].tolist()
    np.testing.assert_allclose(coefficients["pygformula"], committed["pygformula"], rtol=1e-11)

    seeds = pd.read_csv(STUDY / driver.SEED_TABLE)
    by_side = {
        name: frame.drop(columns="implementation").reset_index(drop=True)
        for name, frame in seeds.groupby("implementation")
    }
    pyg = by_side["pygformula"].sort_values(["fixture", "estimand", "horizon", "seed"])
    expected = pyg_estimates.sort_values(["fixture", "estimand", "horizon", "seed"])
    np.testing.assert_allclose(pyg["value"].to_numpy(), expected["value"].to_numpy(), rtol=1e-11)
    estimates = fx.estimate_rows(by_side["cleverly"], by_side["pygformula"])
    published = pd.read_csv(STUDY / driver.ESTIMATES)
    assert estimates["passed"].tolist() == published["passed"].tolist()
    assert estimates["deterministic"].tolist() == published["deterministic"].tolist()
    np.testing.assert_allclose(estimates["ratio"], published["ratio"], rtol=1e-6, atol=1e-12)
    assert fx.findings(coefficients, estimates) == []
