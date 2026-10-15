"""The replay gates of ``canonical-ctmle-sequences`` (X16 plan, rev 2.2, section 3.4.2).

The first rows check the gate rule on small tables.  The middle rows run one fabricated R row
through :func:`~tests.studies.ctmle_sequence_replay.replay_one`, with R's outputs computed by
the package on the same arrays, so every gap is 0 and a perturbed output is caught.  The last
row recomputes the committed verdict from ``replay.csv``, once the run has written it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cleverly.fluctuation.iterative import InitialFit
from tests.canonical.ctmle_sequence_witness import harness
from tests.studies import canonical_ctmle_sequences as study
from tests.studies import ctmle_sequence_replay as replay
from tests.studies.evidence import red_cells


def _compared(scenario: str, count: int, **values: Any) -> pd.DataFrame:
    """``count`` clean replay rows of one scenario, with ``values`` set on every row."""
    rows = []
    for replicate in range(count):
        row: dict[str, Any] = dict.fromkeys(replay.REPLAY_COLUMNS, math.nan)
        row.update(
            scenario=scenario,
            replicate=replicate,
            near_tie=0,
            clip=0,
            path_margin=1e-3,
            option_margin=1e-3,
            accepted_r="3;21",
            accepted_package="3;21",
            accepted_match=1,
            est_gap=1e-9,
            loss_gap=1e-10,
            final_starts=1 if scenario == "binary_lasso" else 0,
            final_index_match=1,
            final_est_gap=1e-9,
            final_loss_gap=1e-10,
        )
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows, columns=replay.REPLAY_COLUMNS)


def _verdict(lasso: pd.DataFrame, learners: pd.DataFrame | None = None) -> pd.Series:
    if learners is None:
        learners = _compared("binary_learners", replay.MIN_UNFLAGGED)
    summary = replay.replay_summary(pd.concat([lasso, learners], ignore_index=True))
    return summary.set_index("scenario")["passed"]


class TestTheGateRule:
    def test_clean_rows_pass(self) -> None:
        assert _verdict(_compared("binary_lasso", replay.MIN_UNFLAGGED)).all()

    @pytest.mark.parametrize(
        "column, value",
        [
            ("accepted_match", 0),
            ("est_gap", 2e-6),
            ("loss_gap", 2e-6),
            ("final_index_match", 0),
            ("final_est_gap", 2e-6),
            ("final_loss_gap", 2e-6),
        ],
    )
    def test_one_unflagged_miss_fails_the_scenario(self, column: str, value: float) -> None:
        lasso = _compared("binary_lasso", replay.MIN_UNFLAGGED + 1)
        lasso.loc[0, column] = value
        verdict = _verdict(lasso)
        assert not verdict["binary_lasso"] and verdict["binary_learners"]

    def test_a_red_gate_needs_its_owner_on_the_roadmap(self) -> None:
        """S1 of design review 2: a red replay gate is routed, a green one needs no owner."""
        owner = study.RED_CELL_OWNERS["replay"]
        lasso = _compared("binary_lasso", replay.MIN_UNFLAGGED + 1)
        learners = _compared("binary_learners", replay.MIN_UNFLAGGED)
        green = replay.replay_summary(pd.concat([lasso, learners], ignore_index=True))
        lasso.loc[0, "est_gap"] = 2e-6
        red = replay.replay_summary(pd.concat([lasso, learners], ignore_index=True))
        assert replay.unowned_red_gates(green, []) == []
        assert replay.unowned_red_gates(red, ["F18"]) == ["binary_lasso"]
        assert replay.unowned_red_gates(red, ["F18", owner]) == []

    @pytest.mark.parametrize("flag", ["near_tie", "clip"])
    def test_a_flagged_miss_is_left_out(self, flag: str) -> None:
        lasso = _compared("binary_lasso", replay.MIN_UNFLAGGED + 1)
        lasso.loc[0, ["accepted_match", flag]] = [0, 1]
        assert _verdict(lasso)["binary_lasso"]

    def test_a_flagged_gap_is_published_and_not_gated(self) -> None:
        lasso = _compared("binary_lasso", replay.MIN_UNFLAGGED + 1)
        lasso.loc[0, ["near_tie", "est_gap", "final_loss_gap"]] = [1, 0.3, 0.2]
        learners = _compared("binary_learners", replay.MIN_UNFLAGGED)
        summary = replay.replay_summary(pd.concat([lasso, learners], ignore_index=True))
        row = summary.set_index("scenario").loc["binary_lasso"]
        assert row["passed"] and row["max_est_gap"] == 1e-9
        assert row["flagged_max_est_gap"] == 0.3 and row["flagged_max_final_loss_gap"] == 0.2
        assert np.isnan(summary.set_index("scenario").loc["binary_learners", "flagged_max_est_gap"])

    def test_too_few_unflagged_replicates_fail(self) -> None:
        lasso = _compared("binary_lasso", replay.MIN_UNFLAGGED)
        lasso.loc[0, "near_tie"] = 1
        assert not _verdict(lasso)["binary_lasso"]

    def test_too_few_final_step_comparisons_fail(self) -> None:
        lasso = _compared("binary_lasso", replay.MIN_UNFLAGGED)
        lasso.loc[replay.MIN_FINAL["binary_lasso"] - 1 :, "final_starts"] = 0
        assert not _verdict(lasso)["binary_lasso"]
        # The learners gate needs no final step.
        assert _verdict(_compared("binary_lasso", replay.MIN_UNFLAGGED))["binary_learners"]


# --------------------------------------------------------------- one fabricated R row


def _text(values: Any) -> str:
    return ";".join(f"{value:.17g}" for value in np.ravel(np.asarray(values, dtype=float)))


def _fabricated(scenario: str, start: int | None) -> dict[str, Any]:
    """An R row whose outputs the package computes on the same arrays.

    The coefficients are a main-terms fit shrunk toward its intercept, one step per
    candidate, so the path holds distinct mechanisms.
    """
    frame, _ = study.draw_scenario(scenario, study.PRIMARY_N, 0)
    frame = frame.reset_index(drop=True)
    x = frame.loc[:, list(study.COVARIATES)].to_numpy(dtype=float)
    a = frame["A"].to_numpy()
    outcome = LogisticRegression(C=1e8, max_iter=5000, tol=1e-12)
    outcome.fit(np.column_stack([a, x]), frame["Y"])
    q_coef = np.concatenate([outcome.intercept_, outcome.coef_[0]])
    propensity = LogisticRegression(C=1e8, max_iter=5000, tol=1e-12).fit(x, a)
    full = np.concatenate([propensity.intercept_, propensity.coef_[0]])
    count = study.PATH_LENGTHS[scenario]
    g_coef = np.array([[full[0], *(full[1:] * t)] for t in np.linspace(0.0, 1.0, count)])
    columns = np.column_stack([harness.linear_logit(row, x) for row in g_coef])
    chooser = harness.selector(
        frame,
        covariates=study.COVARIATES,
        outcome_learner=harness.FixedLogit(tuple(q_coef)),
        split=np.arange(len(frame)) % study.SELECTION_FOLDS,
    )
    path = harness.indexed_path(chooser, columns)
    row: dict[str, Any] = dict.fromkeys(study.REFERENCE_COLUMNS, math.nan)
    row.update(
        scenario=scenario,
        replicate=0,
        n=study.PRIMARY_N,
        q_coef=_text(q_coef),
        g_coef=_text(g_coef),
        r_accepted=";".join(str(index + 1) for index in path.accepted),
        cand_est=_text(path.estimates),
        cand_loss=_text(path.likelihoods(frame["Y"].to_numpy(dtype=float))),
    )
    if start is not None:
        index, candidate, options = harness.final_options(
            chooser, path, columns, columns[:, start:], start - 1
        )
        row.update(
            final_pkg_start=start,
            final_pkg_coef=_text(g_coef[start:]),
            final_pkg_loss=_text([option.loss for option in options]),
            final_pkg_index=index + 1,
            final_pkg_est=harness.ate(candidate.targeted),
        )
    return row


@pytest.fixture(scope="module")
def lasso_row() -> dict[str, Any]:
    return _fabricated("binary_lasso", start=5)


@pytest.fixture(scope="module")
def learners_row() -> dict[str, Any]:
    return _fabricated("binary_learners", start=None)


def test_the_lasso_replay_reproduces_its_own_outputs(lasso_row: dict[str, Any]) -> None:
    out = replay.replay_one(lasso_row)
    assert out["accepted_match"] == 1 and out["final_index_match"] == 1
    assert out["final_starts"] == 1 and out["final_pkg_start"] == 5
    assert out["est_gap"] == 0.0 and out["loss_gap"] == 0.0
    assert out["final_est_gap"] == 0.0 and out["final_loss_gap"] == 0.0
    assert out["final_pkg_index_r"] == out["final_pkg_index_package"]


def test_the_public_learners_fit_reproduces_the_indexed_path(
    learners_row: dict[str, Any],
) -> None:
    """The learners replay reads the public fit; its outputs came from the bare path."""
    out = replay.replay_one(learners_row)
    assert out["accepted_match"] == 1 and out["final_starts"] == 0
    assert out["est_gap"] <= 1e-12 and out["loss_gap"] <= 1e-12


@pytest.mark.parametrize(
    "column, gap",
    [("cand_est", "est_gap"), ("cand_loss", "loss_gap"), ("final_pkg_est", "final_est_gap")],
)
def test_a_perturbed_r_output_is_caught(lasso_row: dict[str, Any], column: str, gap: str) -> None:
    """A deliberate mutation: R's output moved by 1e-4 relative opens a gap above 1e-6."""
    row = dict(lasso_row)
    values = replay.vector(row[column])
    assert values is not None
    values = values.copy()
    values[-1] *= 1.0 + 1e-4
    row[column] = _text(values) if values.size > 1 else float(values[0])
    assert replay.replay_one(row)[gap] > replay.GLM_TOLERANCE


def test_a_wrong_accepted_index_is_caught(lasso_row: dict[str, Any]) -> None:
    row = dict(lasso_row)
    accepted = [int(value) for value in str(row["r_accepted"]).split(";")]
    row["r_accepted"] = ";".join(str(value + 1) for value in accepted)
    assert replay.replay_one(row)["accepted_match"] == 0


def _targeted(observed: float, arm: float) -> SimpleNamespace:
    n = 4
    fit = InitialFit(np.full(n, 0.5), {0.0: np.full(n, 0.5), 1.0: np.full(n, 0.5)})
    fit.observed[0] = observed
    fit.arms[1.0][0] = arm
    return SimpleNamespace(targeted=fit)


def _one_candidate(observed: float, arm: float) -> SimpleNamespace:
    candidate = _targeted(observed, arm)
    return SimpleNamespace(candidates=[candidate], tries=(candidate,))


@pytest.mark.parametrize("value", [1.0 - harness.ALPHA, harness.ALPHA])
def test_the_clip_flag_reads_a_value_at_a_bound(value: float) -> None:
    """B1 of design review 2: a value the ``alpha`` bound clipped lies exactly at the bound.

    ``1 - 0.995`` is above ``0.005`` in floating point, so an interior read against the
    literal ``0.005`` passed it.  The two scenarios read different rows.
    """
    assert harness.interior(_one_candidate(0.5, 0.5))
    arm = _one_candidate(0.5, value)
    assert not harness.interior(arm)
    assert harness.interior(arm, arms=False)
    assert not harness.interior(_one_candidate(value, 0.5), arms=False)


@pytest.mark.parametrize("value", [1.0 - harness.ALPHA, harness.ALPHA])
def test_the_clip_flag_reads_an_off_path_try(value: float) -> None:
    """A clipped try that the path discarded still flags the replicate.

    A round-1 try that an acceptance discards, or a try that loses its round, scores a
    deciding gap, so a clip there can move the path without touching a kept candidate.
    """
    kept = _targeted(0.5, 0.5)
    path = SimpleNamespace(candidates=[kept], tries=(kept, _targeted(value, 0.5)))
    assert not harness.interior(path, arms=False)
    assert harness.interior(SimpleNamespace(candidates=[kept], tries=(kept,)), arms=False)


def test_a_vector_column_reads_a_single_value_and_a_missing_one() -> None:
    np.testing.assert_array_equal(replay.vector(3), [3.0])
    np.testing.assert_array_equal(replay.vector("1;2.5"), [1.0, 2.5])
    assert replay.vector(math.nan) is None


# --------------------------------------------------------------- the committed verdict


def test_the_committed_replay_verdict_is_the_rule_s_output() -> None:
    compared_path = study.ARTIFACTS / study.REPLAY_FILE
    summary_path = study.ARTIFACTS / study.REPLAY_SUMMARY_FILE
    if study.STUDY.comparator_state == "pending":
        # The replay reads R's arrays, so a pending comparator publishes none of it.
        assert not compared_path.exists() and not summary_path.exists()
        return
    if not (compared_path.exists() and summary_path.exists()):
        pytest.skip("replay.csv is written by the study run")
    compared = pd.read_csv(compared_path, float_precision="round_trip")
    committed = pd.read_csv(summary_path, float_precision="round_trip")
    pd.testing.assert_frame_equal(
        replay.replay_summary(compared), committed, check_dtype=False, check_exact=False
    )


def test_a_red_committed_replay_gate_reaches_its_owner() -> None:
    """A red gate in the committed summary fails here until the roadmap lists its owner."""
    summary_path = study.ARTIFACTS / study.REPLAY_SUMMARY_FILE
    if study.STUDY.comparator_state == "pending":
        assert not summary_path.exists(), "a pending comparator publishes no replay gate"
        return
    if not summary_path.exists():
        pytest.skip("replay-summary.csv is written by the study run")
    committed = pd.read_csv(summary_path, float_precision="round_trip")
    assert replay.unowned_red_gates(committed, red_cells.owner_ids()) == []
