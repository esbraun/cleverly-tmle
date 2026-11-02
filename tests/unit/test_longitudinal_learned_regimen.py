"""The learned two-node regimen of ``LTMLE`` (X11 part (b), PR K).

``LTMLE(regimens=LearnedRegimen())`` learns one rule pair per outer fold by backward plug-in
Q-learning on that fold's training rows, runs each fold's untargeted recursion under its own
pair, and targets the stitched predictions with one pooled logistic fluctuation per node.  The
target is ``sum_v (n_v / n) Psi_{d_v}(P_0)``.

=====  =======================================================================================
claim  what fails it
=====  =======================================================================================
L4     each fold's rules and blips are a longhand Q-learning refit (sklearn, fold training
       rows), column by column: the second-node blip under each first-node arm, and the
       first-node blip under the learned second-node rule
W-b1   rules stubbed to a known rule pair give the ``DynamicRegimen(rule_kind="known")`` fit
       bit for bit: estimate, curve, every node array and the standard error
W-b2   perturbing fold v's validation outcomes changes no rule of fold v (premise: another
       fold's rules move); learning the second-node rule on every row (mutation) moves them
W-b3   each node's pooled score is solved, with the clever covariate rebuilt from the
       mechanism at each row's own fold rule; a clever covariate read at another fold's rule
       (mutation) leaves a declared score
W-b4   the first-node rule under the learned second-node rule differs from the one under
       treat-all at the second node on a declared share of rows (the injected carried value
       is not a vanishing term)
W-b5   the variance is ``n^-2 sum_i (Z_i - psi)^2`` with ``Z_i`` rebuilt from the node curves
       and centred at the pooled estimate; the fold-centred form differs (premise)
record the fold weights, treated shares, gap quantiles, disagreement and summary line
refuse  every row of the plan's refusal table, before any learner, with its remedy
admit   censoring, node bounds, ``ratio_trim`` and a known mechanism run and read experimental
store   ``save``/``load`` and ``dumps``/``loads`` keep the record and the estimate
=====  =======================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import cleverly
from cleverly._support_rows import LONGITUDINAL_ROWS
from cleverly.datasets import make_longitudinal_competing, make_longitudinal_survival
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.interventions import Stochastic
from cleverly.learners import SuperLearner
from cleverly.longitudinal import (
    LTMLE,
    SDR,
    DynamicRegimen,
    LearnedRegimen,
    LongitudinalData,
    LongitudinalResult,
    Repeated,
    SequentialGComputation,
    estimator,
    learned,
)
from cleverly.longitudinal.estimator import longitudinal_truncation_curve
from cleverly.longitudinal.learned import LearnedRules, learned_plan
from cleverly.longitudinal.sequential import fit_mechanism, untargeted_recursion

LABEL = "ey_regimen[learned regimen]"
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
N, FOLDS = 1000, 3


def logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


def settings(**overrides: Any) -> dict[str, Any]:
    return {
        "n_folds": FOLDS,
        "random_state": 0,
        "outcome_learner": logistic(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": logistic(),
        "censoring_learner": logistic(),
        **overrides,
    }


def panel(n: int = N, seed: int = 1, *, three: bool = False) -> pd.DataFrame:
    """Two nodes, a binary treatment at each, censoring after each, an end-of-study outcome.

    The second-node effect has the sign of ``1.5 L2 - 0.6 A1`` and the first-node effect
    moves with ``W1``, so both rules treat a share well inside (0, 1), and the first-node
    effect depends on the second-node rule through ``L2``.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    a1 = rng.binomial(1, expit(0.3 * w1 - 0.2 * w2)).astype(float)
    c1 = rng.binomial(1, expit(2.2 + 0.3 * w1)).astype(float)
    l2 = 0.5 * w1 - 0.6 * a1 + rng.normal(size=n)
    a2 = rng.binomial(1, expit(0.2 * l2 + 0.3 * a1 - 0.1)).astype(float)
    c2 = rng.binomial(1, expit(2.2 - 0.2 * l2)).astype(float)
    logit = -0.2 + 0.4 * w1 + 0.3 * l2 + a2 * (1.5 * l2 - 0.6 * a1) + a1 * (1.2 * w1 + 0.4)
    y = rng.binomial(1, expit(logit)).astype(float)
    frame = pd.DataFrame(
        {"W1": w1, "W2": w2, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y}
    )
    frame["g1"] = expit(0.3 * w1 - 0.2 * w2)
    frame["g2"] = expit(0.2 * l2 + 0.3 * a1 - 0.1)
    if three:
        frame["A3"] = rng.binomial(1, 0.5, n).astype(float)
        frame["C3"] = 1.0
        frame.loc[(c1 == 0) | (c2 == 0), ["A3", "C3"]] = np.nan
    frame.loc[c1 == 0, ["L2", "A2", "C2", "Y", "g2"]] = np.nan
    frame.loc[(c1 == 1) & (c2 == 0), ["Y"]] = np.nan
    return frame


def fit(frame: pd.DataFrame | None = None, regimens: Any = None, **overrides: Any) -> Any:
    frame = panel() if frame is None else frame
    regimens = LearnedRegimen() if regimens is None else regimens
    return LTMLE(regimens, **settings(**overrides)).fit(frame, **COLUMNS)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return panel()


@pytest.fixture(scope="module")
def result(frame: pd.DataFrame) -> LongitudinalResult:
    return fit(frame)


def _training(folds: Any, fold: int, n: int) -> np.ndarray:
    rows = np.zeros(n, dtype=bool)
    rows[next(train for k, (train, _) in enumerate(folds) if k == fold)] = True
    return rows


# --------------------------------------------------------------------------- L4


def _longhand(frame: pd.DataFrame, train: np.ndarray) -> tuple[np.ndarray, ...]:
    """Backward plug-in Q-learning on ``train`` with sklearn alone.

    Returns the second-node blip under each first-node arm (``nan`` where censored after the
    first node), the first-node blip under the learned second-node rule, and both rules.
    """
    a1 = frame["A1"].to_numpy()
    a2 = frame["A2"].to_numpy()
    observed = (frame["C1"] == 1).to_numpy()
    complete = observed & (frame["C2"] == 1).to_numpy()
    x2 = np.nan_to_num(frame[["W1", "W2", "L2"]].to_numpy())
    x1 = frame[["W1", "W2"]].to_numpy()
    y = frame["Y"].to_numpy()
    second = np.full((len(frame), 2), np.nan)
    first = np.zeros((len(frame), 2))
    for arm in (0, 1):
        by_arm = []
        for later in (0, 1):
            rows = train & complete & (a1 == arm) & (a2 == later)
            model = LogisticRegression(max_iter=1000).fit(x2[rows], y[rows])
            by_arm.append(model.predict_proba(x2)[:, 1])
        blip = np.where(observed, by_arm[1] - by_arm[0], np.nan)
        second[:, arm] = blip
        rule = np.where(np.nan_to_num(blip) > 0, 1, 0)
        target = np.where(rule == 1, by_arm[1], by_arm[0])
        rows = train & observed & (a1 == arm)
        model = LinearRegression().fit(x1[rows], target[rows])
        first[:, arm] = np.clip(model.predict(x1), 0.0, 1.0)
    first_blip = first[:, 1] - first[:, 0]
    first_rule = np.where(first_blip > 0, 1, 0)
    second_rule = np.where(np.nan_to_num(second[np.arange(len(frame)), first_rule]) > 0, 1, 0)
    return second, first_blip, first_rule, second_rule


def test_l4_each_fold_is_a_longhand_q_learning_refit(
    frame: pd.DataFrame, result: LongitudinalResult
) -> None:
    record = result.learned_regimen
    assert record is not None
    rules = _learned_rules(result)
    for fold in range(FOLDS):
        train = _training(result.folds, fold, len(frame))
        second, first_blip, first_rule, second_rule = _longhand(frame, train)
        # Each column on its own: the second-node blip under a1 = 0 and under a1 = 1.
        for arm in (0, 1):
            np.testing.assert_allclose(
                rules.second_node_blips[fold][:, arm], second[:, arm], atol=1e-7, equal_nan=True
            )
        np.testing.assert_allclose(record.fold_blips[fold, :, 0], first_blip, atol=1e-7)
        assert np.array_equal(record.fold_rules[fold, :, 0], first_rule)
        assert np.array_equal(record.fold_rules[fold, :, 1], second_rule)
    # Nonzero premises: both rules treat a share inside (0, 1) at every fold, and the two
    # first-node branches of the second-node rule differ.
    for node in record.nodes:
        assert all(0.15 < share < 0.85 for share in node.treated_shares), node
    differ = np.nanmean(
        np.sign(rules.second_node_blips[0][:, 0]) != np.sign(rules.second_node_blips[0][:, 1])
    )
    assert differ > 0.05


def _learned_rules(result: LongitudinalResult) -> LearnedRules:
    """The rules of the result's one draw, learned again with the fit's own learners."""
    return learned.learn_fold_rules(
        result.data,
        learned_plan(LearnedRegimen(), result.data),
        folds=result.folds,
        scaler=result.scaler,
        outcome_learner=logistic(),
        pseudo_learner=LinearRegression(),
    )


def test_the_record_holds_the_rules_the_fit_learned(result: LongitudinalResult) -> None:
    record = result.learned_regimen
    assert record is not None
    rules = _learned_rules(result)
    assert np.array_equal(record.fold_rules, np.stack(rules.assignments).astype(np.int8))
    (fit_,) = result.fits.values()
    assert np.array_equal(fit_.assignment, rules.stitched())
    # Each row follows the rule of the fold that held it out.
    owner = np.asarray(result.folds.assignment)
    rows = np.arange(result.data.n)
    assert np.array_equal(fit_.assignment, record.fold_rules[owner, rows].astype(float))


# --------------------------------------------------------------------------- W-b1


def _known_rules(history: Any) -> Any:
    return (history["W1"] > 0.2).astype(float)


def _known_second(history: Any) -> Any:
    return (history["L2"] > -0.1).astype(float)


def test_wb1_stubbed_rules_give_the_known_rule_fit_bit_for_bit(
    frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    known = fit(
        frame,
        DynamicRegimen("learned regimen", (_known_rules, _known_second), rule_kind="known"),
    )
    (known_fit,) = known.fits.values()
    seen: list[int] = []

    def stub(data: Any, plan: Any, *, folds: Any, **_: Any) -> LearnedRules:
        seen.append(folds.n_folds)
        copies = tuple(np.array(known_fit.assignment, copy=True) for _ in range(folds.n_folds))
        zeros = tuple(np.zeros((data.n, 2)) for _ in range(folds.n_folds))
        return LearnedRules(copies, zeros, zeros, folds)

    monkeypatch.setattr(estimator, "learn_fold_rules", stub)
    stubbed = fit(frame)
    assert seen == [FOLDS]
    (stubbed_fit,) = stubbed.fits.values()
    assert stubbed_fit.psi_scaled == known_fit.psi_scaled
    assert np.array_equal(stubbed_fit.influence_curve_scaled, known_fit.influence_curve_scaled)
    assert np.array_equal(stubbed_fit.assignment, known_fit.assignment)
    assert np.array_equal(stubbed_fit.cumulative, known_fit.cumulative)
    for step, twin in zip(stubbed_fit.steps, known_fit.steps, strict=True):
        for name in ("at_risk", "trained_on", "initial", "targeted", "clever"):
            assert np.array_equal(getattr(step, name), getattr(twin, name)), name
        assert step.fluctuation.epsilon == twin.fluctuation.epsilon
    assert stubbed[LABEL].psi == known[LABEL].psi
    assert stubbed[LABEL].std_error == known[LABEL].std_error
    # The rules are not a static plan: both arms are assigned at both nodes.
    for column in (0, 1):
        share = float(np.mean(known_fit.assignment[:, column]))
        assert 0.2 < share < 0.8


def test_wb1_each_fold_is_evaluated_under_its_own_rule_pair(result: LongitudinalResult) -> None:
    """The stitched initial of each held-out row is its own fold's recursion under its rules."""
    data = result.data
    record = result.learned_regimen
    assert record is not None
    (fit_,) = result.fits.values()
    plan = learned_plan(LearnedRegimen(), data)
    owner = np.asarray(result.folds.assignment)
    for fold in range(FOLDS):
        own = plan.with_assignment(record.fold_rules[fold].astype(float))
        nodes = untargeted_recursion(
            data,
            own,
            masks=own.masks(data),
            horizon=2,
            scaler=result.scaler,
            cause=None,
            outcome_learner=logistic(),
            pseudo_learner=LinearRegression(),
            fit_rows=_training(result.folds, fold, data.n),
            outer_fold=fold,
        )
        held = owner == fold
        for node in nodes:
            step = fit_.steps[node.time - 1]
            assert np.array_equal(step.initial[held], node.initial[held]), (fold, node.time)
    # Premise: the folds' rules differ on held-out rows, so another fold's recursion would not
    # give these predictions.
    assert np.mean(record.fold_rules[0] != record.fold_rules[1]) > 0.05


# --------------------------------------------------------------------------- W-b2


def _perturbed(frame: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    changed = frame.copy()
    observed = rows & changed["Y"].notna().to_numpy()
    changed.loc[observed, "Y"] = 1.0 - changed.loc[observed, "Y"]
    return changed


def _rules_on(frame: pd.DataFrame, folds: Any) -> LearnedRules:
    data = LongitudinalData.from_frame(frame, **COLUMNS)
    return learned.learn_fold_rules(
        data,
        learned_plan(LearnedRegimen(), data),
        folds=folds,
        scaler=_scaler(data),
        outcome_learner=logistic(),
        pseudo_learner=LinearRegression(),
    )


def _scaler(data: LongitudinalData) -> Any:
    return LTMLE(LearnedRegimen(), **settings())._scaler(data)


def test_wb2_a_validation_outcome_moves_no_rule_of_its_fold(
    frame: pd.DataFrame, result: LongitudinalResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    folds = result.folds
    held = np.asarray(folds.assignment) == 0
    before = _rules_on(frame, folds)
    after = _rules_on(_perturbed(frame, held), folds)
    assert np.array_equal(before.assignments[0], after.assignments[0])
    assert np.array_equal(before.blips[0], after.blips[0], equal_nan=True)
    # Premise: the perturbation has power, because it moves the folds that train on it.
    moved = [float(np.mean(before.assignments[k] != after.assignments[k])) for k in range(1, FOLDS)]
    assert min(moved) > 0.02, moved

    # Mutation: the second-node rule learned on every row of its stratum, as a full-data
    # d_A(1) would be.  Fold 0's rules then read its own validation outcomes.
    original = learned._single_arm_predictions

    def every_row(*args: Any, fit_rows: np.ndarray, **kwargs: Any) -> Any:
        data = args[0]
        codes = np.asarray(data.node_codes(1), dtype=float)
        arm = codes[fit_rows][0]
        return original(*args, fit_rows=codes == arm, **kwargs)

    monkeypatch.setattr(learned, "_single_arm_predictions", every_row)
    mutated_before = _rules_on(frame, folds)
    mutated_after = _rules_on(_perturbed(frame, held), folds)
    changed = float(np.mean(mutated_before.assignments[0] != mutated_after.assignments[0]))
    assert changed > 0.02, changed


# --------------------------------------------------------------------------- W-b3


def _scores(fit_: Any) -> list[float]:
    return [
        float(
            np.sum(
                np.where(step.trained_on, step.clever * (step.pseudo_outcome - step.targeted), 0)
            )
        )
        / fit_.influence_curve_scaled.size
        for step in fit_.steps
    ]


def test_wb3_each_nodes_pooled_score_is_solved_at_each_rows_rule(
    result: LongitudinalResult,
) -> None:
    (fit_,) = result.fits.values()
    data = result.data
    plan = learned_plan(LearnedRegimen(), data).with_assignment(fit_.assignment)
    masks = plan.masks(data)
    mechanism = result.mechanism
    _, cumulative = mechanism.cumulative_with_unbounded(data, plan, result.config.g_bounds)
    # The mechanism is the one fitted at each row's own rule, not at the placeholder plan.
    refitted = fit_mechanism(
        data,
        (plan,),
        treatment_learner=logistic(),
        censoring_learner=logistic(),
        folds=result.folds,
    )
    np.testing.assert_array_equal(
        refitted.cumulative(data, plan, result.config.g_bounds), fit_.cumulative
    )
    for time, step in enumerate(fit_.steps, start=1):
        rebuilt = np.where(masks.following(time), 1.0 / cumulative[:, time - 1], 0.0)
        np.testing.assert_allclose(step.clever, rebuilt, rtol=1e-12)
    assert all(abs(score) < 1e-8 for score in _scores(fit_))
    # Mutation: every row reads fold 0's rule in its clever covariate.
    record = result.learned_regimen
    assert record is not None
    other = plan.with_assignment(record.fold_rules[0].astype(float))
    other_mechanism = fit_mechanism(
        data,
        (other,),
        treatment_learner=logistic(),
        censoring_learner=logistic(),
        folds=result.folds,
    )
    _, other_cumulative = other_mechanism.cumulative_with_unbounded(
        data, other, result.config.g_bounds
    )
    other_masks = other.masks(data)
    mutated = []
    for time, step in enumerate(fit_.steps, start=1):
        clever = np.where(other_masks.following(time), 1.0 / other_cumulative[:, time - 1], 0.0)
        mutated.append(float(np.sum(clever * (step.pseudo_outcome - step.targeted))) / data.n)
    assert max(abs(score) for score in mutated) > 1e-3, mutated


# --------------------------------------------------------------------------- W-b4


def test_wb4_the_first_rule_reads_the_learned_second_rule(result: LongitudinalResult) -> None:
    data = result.data
    plan = learned_plan(LearnedRegimen(), data)
    for fold in range(FOLDS):
        common = {
            "scaler": result.scaler,
            "outcome_learner": logistic(),
            "pseudo_learner": LinearRegression(),
            "train": _training(result.folds, fold, data.n),
            "fold": fold,
            "bounds": None,
        }
        own, _, _ = learned._learn_in_fold(data, plan, **common)
        treat_all, _, _ = learned._learn_in_fold(data, plan, second_rule=np.ones(data.n), **common)
        share = float(np.mean(own[:, 0] != treat_all[:, 0]))
        assert share > 0.15, (fold, share)


# --------------------------------------------------------------------------- W-b5


def test_wb5_the_variance_is_the_stacked_curve_centred_at_the_pooled_estimate(
    result: LongitudinalResult,
) -> None:
    (fit_,) = result.fits.values()
    n = result.data.n
    first = fit_.steps[0]
    z = first.targeted.copy()
    for step in fit_.steps:
        z = z + np.where(step.trained_on, step.clever * (step.pseudo_outcome - step.targeted), 0)
    psi = float(np.mean(first.targeted))
    assert psi == pytest.approx(fit_.psi_scaled, abs=1e-14)
    np.testing.assert_allclose(z - psi, fit_.influence_curve_scaled, atol=1e-12)
    variance = float(np.sum((z - psi) ** 2)) / n**2
    # The stored rule is the centred one, n / (n - 1) times the plan's form, as for the
    # stacked point-treatment report (the pooled curve's mean is 0 to solver tolerance).
    assert abs(float(np.mean(z - psi))) < 1e-9
    assert result[LABEL].std_error ** 2 == pytest.approx(variance * n / (n - 1), rel=1e-10)
    # The fold-centred form of van der Laan and Luedtke (2015), Appendix B, differs.
    owner = np.asarray(result.folds.assignment)
    centred = sum(float(np.sum((z[owner == k] - z[owner == k].mean()) ** 2)) for k in range(FOLDS))
    assert abs(centred / n**2 - variance) / variance > 1e-4


# --------------------------------------------------------------------------- record


def test_the_record_reports_weights_shares_and_gaps(result: LongitudinalResult) -> None:
    record = result.learned_regimen
    assert record is not None
    owner = np.asarray(result.folds.assignment)
    sizes = tuple(int(np.sum(owner == k)) for k in range(FOLDS))
    assert record.fold_sizes == sizes
    assert record.fold_weights == tuple(size / sum(sizes) for size in sizes)
    (fit_,) = result.fits.values()
    targeted = fit_.steps[0].targeted
    fold_values = [float(np.mean(targeted[owner == k])) for k in range(FOLDS)]
    assert fit_.psi_scaled == pytest.approx(
        sum(w * v for w, v in zip(record.fold_weights, fold_values, strict=True)), abs=1e-12
    )
    data = result.data
    first_arm = np.asarray(data.node_codes(1), dtype=float)
    uncensored = (np.asarray(data.uncensored)[:, 0]).astype(bool)
    for fold in range(FOLDS):
        held = owner == fold
        rules = record.fold_rules[fold]
        assert record.nodes[0].treated_shares[fold] == pytest.approx(rules[held, 0].mean())
        at_risk = held & uncensored & (first_arm == rules[:, 0])
        assert record.nodes[1].treated_shares[fold] == pytest.approx(rules[at_risk, 1].mean())
        gaps = np.quantile(np.abs(record.fold_blips[fold, held, 0]), record.quantile_levels)
        np.testing.assert_allclose(record.nodes[0].gap_quantiles[fold], gaps)
    disagree = np.mean(np.any(record.fold_rules[:, :, 0] != record.fold_rules[:1, :, 0], axis=0))
    assert record.nodes[0].rule_disagreement == pytest.approx(float(disagree))
    assert record.nodes[0].arms == ("0.0", "1.0") or record.nodes[0].arms == ("0", "1")
    text = record.describe()
    assert "3 rule pairs" in text and "Node 1 (A1)" in text and "Node 2 (A2)" in text
    assert record.target.startswith("fold-size-weighted")
    assert " ".join(text.split()) in " ".join(str(result.summary(detail="full")).split())


def test_the_fit_reads_experimental_under_the_learned_kind(result: LongitudinalResult) -> None:
    estimate = result[LABEL]
    assert estimate.support.label == "experimental"
    key = result.config.support_key
    assert key is not None and "learned" in key.family
    assert not any("learned" in row.key.family for row in LONGITUDINAL_ROWS)
    # One parameter, so no simultaneous band is built, and no contrast is reported.
    assert list(result.estimates) == [LABEL]
    assert result.simultaneous is None


# --------------------------------------------------------------------------- refusals


def _refused(call: Any, *needles: str, roadmap: str = "X11") -> CapabilityError:
    with pytest.raises(CapabilityError) as caught:
        call()
    error = caught.value
    assert error.refusal is not None
    assert error.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
    assert error.refusal.roadmap == roadmap
    for needle in needles:
        assert needle in str(error), (needle, str(error))
    return error


@pytest.fixture
def no_learner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any fold regression or mechanism fit: each refusal comes before any learner."""

    def boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("a learner ran before the refusal")

    monkeypatch.setattr(estimator, "learn_fold_rules", boom)
    monkeypatch.setattr(estimator, "fit_mechanism", boom)


@pytest.mark.usefixtures("no_learner")
def test_repeats_are_refused_for_a_learned_regimen(frame: pd.DataFrame) -> None:
    """The repeat witness of ``learned_regimen`` (test_longitudinal_repeats.RESULT_SURFACE)."""
    _refused(lambda: fit(frame, repeats=3), "median", "X11 (b2)", "Set repeats=1")


@pytest.mark.usefixtures("no_learner")
class TestSettingRefusals:
    def test_another_estimator(self, frame: pd.DataFrame) -> None:
        for kind, folds in ((SDR, FOLDS), (SequentialGComputation, 1)):
            base = settings(n_folds=folds)
            base.pop("pseudo_learner")
            _refused(
                lambda kind=kind, base=base: kind(LearnedRegimen(), **base).fit(frame, **COLUMNS),
                "does not fit a LearnedRegimen",
                "X11 (b2)",
            )

    def test_msm(self, frame: pd.DataFrame) -> None:
        from cleverly.msm import MSM

        def design(label: Any, horizon: int, baseline: Any) -> np.ndarray:
            return np.ones((len(baseline), 1))

        msm = MSM(design=design, terms=("(intercept)",), design_kind="known")
        _refused(lambda: fit(frame, msm=msm), "msm=", "X11 (b2)", "Omit msm=")

    def test_hazard_and_one_step_targeting(self, frame: pd.DataFrame) -> None:
        _refused(
            lambda: fit(frame, survival_model="hazard", targeting="one_step", pseudo_learner=None),
            "one-step",
            "X11 (b2)",
            "targeting='iterative'",
        )

    def test_one_fold(self, frame: pd.DataFrame) -> None:
        _refused(lambda: fit(frame, n_folds=1), "n_folds=1", "Theorem 5", "X11 (b2)")

    def test_bootstrap(self, frame: pd.DataFrame) -> None:
        _refused(lambda: fit(frame, n_bootstrap=20), "relearn", "n_bootstrap=0", roadmap="F27")

    @pytest.mark.parametrize(
        "learner",
        [
            SuperLearner([LinearRegression()], task="regression"),
            LassoCV(),
            Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV())]),
        ],
        ids=["super_learner", "lasso_cv", "pipeline_ridge_cv"],
    )
    def test_a_selector_as_the_first_nodes_regression(
        self, frame: pd.DataFrame, learner: Any
    ) -> None:
        _refused(
            lambda: fit(frame, pseudo_learner=learner),
            "cross-validated risk",
            "X11 (b2)",
            "pseudo_learner=",
        )

    def test_the_default_first_node_learner_is_a_selector(self, frame: pd.DataFrame) -> None:
        _refused(lambda: fit(frame, pseudo_learner=None, outcome_learner=None), "Super Learner")

    @pytest.mark.parametrize("beside", ["static", "stochastic", "rule"])
    def test_a_regimen_beside_it(self, frame: pd.DataFrame, beside: str) -> None:
        other: Any = {
            "static": 1,
            "stochastic": DynamicRegimen(
                "draw",
                (Stochastic(lambda h: np.full((len(h), 2), 0.5), "half", density_kind="known"),)
                * 2,
            ),
            "rule": DynamicRegimen("rule", (_known_rules, _known_second), rule_kind="known"),
        }[beside]
        _refused(
            lambda: fit(frame, {"learned": LearnedRegimen(), "other": other}),
            "fitted alone",
            "simultaneous band",
            "X11 (b2)",
        )
        _refused(
            lambda: fit(
                frame,
                [LearnedRegimen(), DynamicRegimen("rule", (1, _known_second), rule_kind="known")],
            ),
            "fitted alone",
        )


@pytest.mark.usefixtures("no_learner")
class TestDataRefusals:
    def test_survival_and_competing_outcomes(self) -> None:
        survival, _ = make_longitudinal_survival(n=300, seed=4)
        shared = {k: v for k, v in COLUMNS.items() if k != "outcome"}
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(
                survival, outcome=("Y1", "Y2"), **shared
            ),
            "a survival outcome",
            "X11 (b2)",
        )
        competing, _ = make_longitudinal_competing(n=300, seed=5)
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(
                competing, outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")}, **shared
            ),
            "a competing-risk outcome",
        )

    def test_a_repeated_outcome(self, frame: pd.DataFrame) -> None:
        first = np.where(frame["C1"] == 1, (frame["W1"] > 0).astype(float), np.nan)
        repeated = frame.assign(Y1=first, Y2=frame["Y"])
        shared = {k: v for k, v in COLUMNS.items() if k != "outcome"}
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(
                repeated, outcome=Repeated(("Y1", "Y2")), **shared
            ),
            "a repeated outcome",
        )

    def test_one_node_and_three_nodes(self, frame: pd.DataFrame) -> None:
        one = {
            "outcome": "Y",
            "treatment": ["A1"],
            "baseline": ["W1", "W2"],
            "time_varying": [[]],
            "censoring": ["C1"],
        }
        observed = frame.assign(Y=np.where(frame["C1"] == 1, frame["Y"].fillna(0.0), np.nan))
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(observed, **one),
            "this design has 1",
            "TMLE(learned_rule=LearnedRule())",
        )
        three = panel(three=True)
        columns = {
            **COLUMNS,
            "treatment": ["A1", "A2", "A3"],
            "time_varying": [[], ["L2"], []],
            "censoring": ["C1", "C2", "C3"],
        }
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(three, **columns),
            "this design has 3",
            "X11 (b2)",
        )

    def test_a_node_that_is_not_one_binary_treatment(self, frame: pd.DataFrame) -> None:
        many = frame.assign(A1=frame["A1"] + (frame["W1"] > 1.0))
        _refused(lambda: fit(many), "node 1 ('A1') is not one binary treatment", "X11 (b2)")
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(
                frame, continuous_treatment=["A2"], **COLUMNS
            ),
            "node 2 ('A2') is not one binary treatment",
        )

    def test_weights_and_clusters(self, frame: pd.DataFrame) -> None:
        weighted = frame.assign(w=1.0 + (frame["W2"] > 0))
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(weighted, weights="w", **COLUMNS),
            "weights=",
            "Omit weights=",
        )
        clustered = frame.assign(cluster=np.arange(len(frame)) // 2)
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(clustered, id="cluster", **COLUMNS),
            "id=",
            "Omit id=",
        )

    def test_a_markov_order(self, frame: pd.DataFrame) -> None:
        data = LongitudinalData.from_frame(frame, **COLUMNS)
        panel_ = LongitudinalData.from_frame(
            frame.assign(L1=frame["W1"]),
            **{**COLUMNS, "time_varying": [["L1"], ["L2"]]},
        )
        del data
        windowed = panel_.with_markov_order(0)
        _refused(
            lambda: LTMLE(LearnedRegimen(), **settings()).fit(windowed),
            "Markov order",
            "X11 (b2)",
        )


def test_an_estimated_dynamic_regimen_names_the_learned_regimen() -> None:
    with pytest.raises(CapabilityError) as caught:
        DynamicRegimen("d", (1, _known_second), rule_kind="estimated")
    assert "LearnedRegimen" in str(caught.value)


class TestRefitSurfaces:
    def test_the_truncation_curve(self, result: LongitudinalResult) -> None:
        _refused(
            lambda: longitudinal_truncation_curve(result, [0.05]),
            "the truncation curve",
            roadmap="F27",
        )

    def test_a_refit(self, result: LongitudinalResult) -> None:
        assert result.estimator is not None
        _refused(lambda: result.estimator.refit(result.data), "a refit", roadmap="F27")

    def test_a_refutation(self, result: LongitudinalResult) -> None:
        from cleverly.validation import refute

        with pytest.raises(CapabilityError, match="relearns the rules"):
            refute(result, estimand=LABEL, tests=("subset",))


def test_a_label_must_be_a_string() -> None:
    with pytest.raises(cleverly.exceptions.DataError):
        LearnedRegimen("")
    assert learned.learned_regimen_in({"mine": LearnedRegimen()}) == LearnedRegimen("mine")
    assert learned.learned_regimen_in({"always": 1}) is None


# --------------------------------------------------------------------------- admitted


@pytest.mark.parametrize(
    "overrides",
    [
        {"ratio_trim": 0.99},
        {"node_bounds": {1: (0.02, 0.98)}},
        {"stopping": "sample_size"},
    ],
    ids=["ratio_trim", "node_bounds", "stopping"],
)
def test_admitted_compositions_run_and_read_experimental(
    frame: pd.DataFrame, overrides: dict[str, Any]
) -> None:
    result = fit(frame, **overrides)
    estimate = result[LABEL]
    assert np.isfinite(estimate.psi)
    assert estimate.support.label == "experimental"
    assert result.learned_regimen is not None


def test_a_known_mechanism_runs(frame: pd.DataFrame) -> None:
    declared = frame[["g1", "g2"]].to_numpy(dtype=float)
    result = LTMLE(LearnedRegimen(), **settings()).fit(
        frame, treatment_probabilities=declared, **COLUMNS
    )
    estimate = result[LABEL]
    assert np.isfinite(estimate.psi) and estimate.support.label == "experimental"
    (fit_,) = result.fits.values()
    rows = np.arange(len(frame))
    first = np.column_stack([1.0 - frame["g1"], frame["g1"]])
    arms = fit_.assignment[:, 0].astype(int)
    np.testing.assert_allclose(
        result.mechanism.treatment[0]["learned regimen"], first[rows, arms], rtol=1e-12
    )


def test_censoring_is_part_of_each_node(result: LongitudinalResult) -> None:
    assert result.data.has_censoring
    assert "censoring" in result.config.support_key.missing  # type: ignore[union-attr]


# --------------------------------------------------------------------------- persistence


def _same(left: LongitudinalResult, right: LongitudinalResult) -> None:
    assert left[LABEL].psi == right[LABEL].psi
    assert left[LABEL].std_error == right[LABEL].std_error
    a, b = left.learned_regimen, right.learned_regimen
    assert a is not None and b is not None
    assert np.array_equal(a.fold_rules, b.fold_rules)
    assert np.array_equal(a.fold_blips, b.fold_blips, equal_nan=True)
    assert a.nodes == b.nodes and a.fold_sizes == b.fold_sizes
    assert a.describe() == b.describe()


def test_save_load_and_dumps_loads_keep_the_record(
    result: LongitudinalResult, tmp_path: Path
) -> None:
    path = result.save(tmp_path / "learned.joblib")
    _same(result, cleverly.load(path))
    _same(result, loads(dumps(result)))
