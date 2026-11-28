"""X82 part F4: the frozen-rule replay of a learned-rule fit.

A refit of a learned-rule fit relearns the rule inside each training fold, so a perturbed
draw would estimate the value of other rules.  The replay keeps the fit's own folds and the
realized rule of every row.  Each cell then evaluates the value of these fixed rules
fold-wise, with the ``1/V`` weights of the learned-rule record.  These tests pin the
reproduction at zero strength, the frozen regimes, the fold weights, and two mutation
controls that relearn the rule or redraw the folds.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from functools import cache
from typing import Any

import numpy as np
import pytest

from cleverly import CausalStudy, LearnedRuleContrast, LearnedRuleValue, PointTreatment
from cleverly._summary.vocabulary import NOTE_TEXTS
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import NaturalCourse, Static, Stochastic
from cleverly.interventions.learned import FrozenLearnedRule, fold_rules, freeze_learned_rule
from cleverly.sensitivity import ConfounderStrengthGrid, simulated_confounding
from cleverly.sensitivity import _simulated_confounding_request as request_module
from tests.unit import _learned_rule_support as support
from tests.unit._confounding_support import replacement as _replacement

_SEED = 3
_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.2), outcome=(0.0, 0.2))
#: 403 rows in 5 folds give test folds of 81, 81, 81, 80 and 80 rows, so the ``1/V``
#: weights of the record differ from row weights.
_N = 403


@cache
def _fit() -> Any:
    study = CausalStudy(
        support.law_frame(n=_N),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    return study.identify(LearnedRuleValue()).estimate(
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        cv_evaluation=True,
        simultaneous=False,
        random_state=0,
    )


@cache
def _surface() -> Any:
    return simulated_confounding(_fit(), estimand=support.NAME, grid=_GRID, random_state=_SEED)


def _frozen_refit(result: Any, data: Any) -> Any:
    """The manual replay: the fit's folds and the realized rule of every row."""
    replay = copy.copy(result.estimator)
    replay.learned_rule = FrozenLearnedRule(result.estimator.learned_rule.name, _treat(result))
    return replay.refit(data, random_state=_SEED, frozen_folds=[result.nuisance.folds])


def _treat(result: Any) -> np.ndarray:
    """The ``int8`` arm code of each row's realized rule, by hand from the one-hot density."""
    return np.argmax(np.asarray(result.nuisance.regimes.values[:, :, 0]), axis=1).astype(np.int8)


def _check_cells_equal_frozen_refits(result: Any, surface: Any) -> None:
    for cell in surface.cells[1:]:
        data = _replacement(result, surface, cell.treatment_strength, cell.outcome_strength)
        manual = _frozen_refit(result, data)
        assert cell.estimate == pytest.approx(manual[support.NAME].psi, abs=1e-12)


def test_zero_strength_reproduces_the_fit_exactly() -> None:
    result, surface = _fit(), _surface()
    assert result.sensitivity.capability("simulated_confounding").available
    assert surface.complete
    assert surface.cells[0].estimate == result[support.NAME].psi
    assert surface.cells[0].displacement == 0.0
    reproduced = _frozen_refit(result, result.data)
    assert reproduced[support.NAME].psi == result[support.NAME].psi


def test_every_cell_is_the_frozen_rule_refit_of_its_perturbed_table() -> None:
    result, surface = _fit(), _surface()
    _check_cells_equal_frozen_refits(result, surface)
    assert max(abs(cell.displacement) for cell in surface.cells) > 5e-3


def test_a_perturbed_refit_keeps_the_realized_rules_and_the_folds() -> None:
    result, surface = _fit(), _surface()
    cell = surface.cells[-1]
    data = _replacement(result, surface, cell.treatment_strength, cell.outcome_strength)
    frozen = _frozen_refit(result, data)
    assert np.array_equal(_treat(frozen), _treat(result))
    assert [test.tolist() for _, test in frozen.nuisance.folds] == [
        test.tolist() for _, test in result.nuisance.folds
    ]
    # Witness: a relearning refit on the same folds and table changes the rule of some rows.
    relearned = result.estimator.refit(
        data, random_state=_SEED, frozen_folds=[result.nuisance.folds]
    )
    assert np.any(_treat(relearned) != _treat(result))


def test_a_cell_averages_the_fold_values_with_equal_weights() -> None:
    result, surface = _fit(), _surface()
    cell = surface.cells[-1]
    data = _replacement(result, surface, cell.treatment_strength, cell.outcome_strength)
    (split,) = _frozen_refit(result, data).extra["learned_rule"].splits
    assert split.fold_sizes == (81, 81, 81, 80, 80)
    folds = np.asarray(split.fold_estimates)
    sizes = np.asarray(split.fold_sizes, dtype=float)
    assert cell.estimate == pytest.approx(folds.mean(), abs=1e-12)
    # Row weights give another number, so the check separates the two averages.
    assert abs(float(np.average(folds, weights=sizes)) - folds.mean()) > 1e-6


@pytest.mark.parametrize("mutation", ["relearns_the_rule", "redraws_the_folds"])
def test_a_replay_without_the_frozen_rule_or_folds_fails_the_cell_check(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Mutation control: each replay that drops one frozen component moves the cells."""
    result = _fit()
    validate = request_module.validate_learned_rule_replay

    def mutated(fit: Any, estimand: str, key: Any) -> Any:
        replay, folds, kind = validate(fit, estimand, key)
        if mutation == "relearns_the_rule":
            return fit.estimator, folds, kind
        return replay, None, kind

    monkeypatch.setattr(request_module, "validate_learned_rule_replay", mutated)
    surface = simulated_confounding(result, estimand=support.NAME, grid=_GRID, random_state=_SEED)
    assert surface.complete
    with pytest.raises(AssertionError):
        _check_cells_equal_frozen_refits(result, surface)


def test_the_summary_states_the_fixed_rule_reading() -> None:
    summary = _surface().summary()
    assert summary.note("learned_rule_reading").text == NOTE_TEXTS["learned_rule_reading"]
    settings = _surface().summary(detail="full").section("Settings").lines
    assert any("not of the learned-rule procedure" in line for line in settings)


def test_inconsistent_fold_metadata_is_refused() -> None:
    result = _fit()
    original = result.extra["learned_rule"]
    split = replace(original.splits[0], fold_sizes=(80, 81, 81, 81, 80))
    record = replace(original, splits=(split,))
    tampered = replace(result, extra={**result.extra, "learned_rule": record})
    with pytest.raises(CapabilityError, match="inconsistent learned-rule parameter metadata"):
        simulated_confounding(tampered, estimand=support.NAME, grid=_GRID)


def test_the_frozen_rule_checks_its_rows() -> None:
    result = _fit()
    rule = freeze_learned_rule(result)
    assert rule.name == result.estimator.learned_rule.name
    subset = rule.rows(np.array([2, 0, 0]))
    assert subset.treat.tolist() == [rule.treat[2], rule.treat[0], rule.treat[0]]
    with pytest.raises(DataError, match="one int8 arm code per row"):
        FrozenLearnedRule("learned rule", np.zeros(3))
    with pytest.raises(DataError, match="one int8 arm code per row"):
        FrozenLearnedRule("learned rule", np.zeros(3, dtype=bool))
    replay = copy.copy(result.estimator)
    replay.learned_rule = subset
    with pytest.raises(DataError, match="holds 3 rows"):
        replay.refit(result.data, random_state=_SEED, frozen_folds=[result.nuisance.folds])


# ------------------------------------------------------------- the frozen replay at a level
#
# X11 (j4a).  Each cell holds the treatment, flips the intermediate in the latent tail and
# moves the outcome, and refits on the fit's folds and repeat seeds with the frozen rules of
# the fit's level; the refit holds the intermediate at that level.

_LEVEL_GRID = ConfounderStrengthGrid(treatment=(0.0,), intermediate=(0.0, 0.2), outcome=(0.0, 0.2))
#: W-j4f3: an intermediate-only cell moves the level-0 value by 0.030 and the level-1 value by
#: 0.058, and its induced intermediate association is -0.174 against -0.081 at the anchor;
#: each margin is about half the smaller observed value.
W_J4F3_MOVEMENT = 0.015
W_J4F3_ASSOCIATION = 0.045
#: W-j4f4: over both levels and every nonzero cell, a relearning refit moves the cell by 0.0065
#: to 0.089 and changes 6.7% to 27% of the fold-rule entries, and the other level's frozen
#: rules move it by 0.0052 to 0.087; the margin is half the smallest movement.
W_J4F4_GAP = 2.5e-3


def _level_study() -> CausalStudy:
    from tests.unit.test_learned_rule_intermediate import cde_frame

    # 403 rows give test folds of 81, 81, 81, 80 and 80 rows (W-j4f8).
    return CausalStudy(
        cde_frame(_N, 0),
        design=PointTreatment(
            outcome="Y", treatment="A", adjustment=("W1", "W2"), intermediate="Z"
        ),
    )


def _level_options(**overrides: Any) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    from tests.unit.test_learned_rule_intermediate import outcome_learner

    options: dict[str, Any] = {
        "outcome_learner": outcome_learner(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "intermediate_learner": LogisticRegression(max_iter=1000),
        "n_folds": 5,
        "cv_evaluation": True,
        "simultaneous": False,
        "random_state": 0,
    }
    options.update(overrides)
    return options


@cache
def _level_fit(z: float) -> Any:
    return _level_study().identify(LearnedRuleValue(intermediate=z)).estimate(**_level_options())


@cache
def _level_surface(z: float) -> Any:
    return simulated_confounding(_level_fit(z), estimand=support.NAME, grid=_LEVEL_GRID)


def level_table(result: Any, surface: Any, cell: Any) -> Any:
    """The cell's table by hand: the treatment held, Z and Y flipped in the latent tail."""
    from statistics import NormalDist

    data = result.data
    latent = np.random.default_rng(surface.latent_seed).normal(size=data.n)
    assert cell.treatment_strength == 0.0

    def flip(values: np.ndarray, strength: float) -> np.ndarray:
        if strength == 0.0:
            return values.copy()
        moved = latent >= NormalDist().inv_cdf(1.0 - strength)
        return np.where(moved, 1.0 - values, values)

    table = data.with_outcome(
        flip(data.outcome, cell.outcome_strength),
        family=data.family,
        name="simulated-confounding outcome",
    )
    return replace(table, intermediate=flip(data.intermediate, cell.intermediate_strength))


def _level_refit(result: Any, data: Any, treat: np.ndarray | None = None) -> Any:
    """The manual frozen replay at the fit's level, on the fit's folds and seed."""
    z = result.intermediate_value
    replay = copy.copy(result.estimator)
    replay.learned_rule = FrozenLearnedRule(
        result.estimator.learned_rule.name, _treat(result) if treat is None else treat, (), z
    )
    return replay.refit(
        data, intermediate_value=z, random_state=0, frozen_folds=[result.nuisance.folds]
    )


@pytest.mark.parametrize("z", [0.0, 1.0])
class TestTheFrozenReplayAtALevel:
    """W-j4f1 to W-j4f8 of the plan."""

    def test_the_anchor_and_the_manual_refit_reproduce_the_fit(self, z: float) -> None:
        """W-j4f1."""
        result, surface = _level_fit(z), _level_surface(z)
        assert result.sensitivity.capability("simulated_confounding").available
        assert surface.complete and surface.rule_replay == "frozen"
        assert surface.intermediate_value == z
        assert surface.cells[0].estimate == result[support.NAME].psi
        assert _level_refit(result, result.data)[support.NAME].psi == result[support.NAME].psi

    def test_every_cell_is_the_frozen_refit_of_its_table(self, z: float) -> None:
        """W-j4f2: Z flipped by the published flip, Y moved, A identical."""
        result, surface = _level_fit(z), _level_surface(z)
        for cell in surface.cells[1:]:
            table = level_table(result, surface, cell)
            assert np.array_equal(table.treatment, result.data.treatment)
            manual = _level_refit(result, table)
            assert cell.estimate == pytest.approx(manual[support.NAME].psi, abs=1e-12)

    def test_an_intermediate_cell_moves_with_the_latent(self, z: float) -> None:
        """W-j4f3, nonzero."""
        surface = _level_surface(z)
        cell = next(c for c in surface.cells if c.intermediate_strength and not c.outcome_strength)
        anchor = surface.cells[0].induced_intermediate_association
        assert abs(cell.displacement) > W_J4F3_MOVEMENT
        assert abs(cell.induced_intermediate_association - anchor) > W_J4F3_ASSOCIATION

    def test_the_refit_keeps_the_level_rules(self, z: float) -> None:
        """W-j4f4: the rules are the level's; relearned or other-level rules miss the cell."""
        from tests.unit.test_learned_rule_intermediate import LEVEL_RULE_SHARE

        result, surface = _level_fit(z), _level_surface(z)
        other = _treat(_level_fit(1.0 - z))
        assert float(np.mean(other != _treat(result))) > LEVEL_RULE_SHARE
        for cell in surface.cells[1:]:
            table = level_table(result, surface, cell)
            frozen = _level_refit(result, table)
            assert np.array_equal(_treat(frozen), _treat(result))
            assert [t.tolist() for _, t in frozen.nuisance.folds] == [
                t.tolist() for _, t in result.nuisance.folds
            ]
            relearned = result.estimator.refit(
                table, intermediate_value=z, random_state=0, frozen_folds=[result.nuisance.folds]
            )
            changed = fold_rules(relearned.nuisance) != fold_rules(result.nuisance)
            assert float(np.mean(changed)) > 0.02
            assert abs(relearned[support.NAME].psi - cell.estimate) > W_J4F4_GAP
            wrong_level = _level_refit(result, table, other)
            assert abs(wrong_level[support.NAME].psi - cell.estimate) > W_J4F4_GAP

    def test_a_record_of_another_level_is_refused_before_any_draw(
        self, z: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W-j4f5."""
        from tests.unit._confounding_support import forbid_draw_and_refit

        result = _level_fit(z)
        record = replace(result.extra["learned_rule"], intermediate_level=1.0 - z)
        tampered = replace(result, extra={**result.extra, "learned_rule": record})
        forbid_draw_and_refit(monkeypatch, tampered.estimator)
        with pytest.raises(CapabilityError) as caught:
            simulated_confounding(tampered, estimand=support.NAME, grid=_LEVEL_GRID)
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert "inconsistent learned-rule parameter metadata" in str(caught.value)

    def test_a_treatment_strength_is_refused_before_any_draw(
        self, z: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W-j4f6: R1, the F15 refusal of the controlled-direct-effect surface."""
        from tests.unit._confounding_support import forbid_draw_and_refit

        result = _level_fit(z)
        forbid_draw_and_refit(monkeypatch, result.estimator)
        with pytest.raises(CapabilityError) as caught:
            simulated_confounding(result, estimand=support.NAME, grid=_GRID)
        assert caught.value.refusal.roadmap == "F15"

    def test_a_cell_averages_the_fold_values_with_equal_weights(self, z: float) -> None:
        """W-j4f8: folds of 81 and 80 rows."""
        result, surface = _level_fit(z), _level_surface(z)
        cell = surface.cells[-1]
        (split,) = (
            _level_refit(result, level_table(result, surface, cell)).extra["learned_rule"].splits
        )
        assert split.fold_sizes == (81, 81, 81, 80, 80)
        folds = np.asarray(split.fold_estimates)
        assert cell.estimate == pytest.approx(folds.mean(), abs=1e-12)
        sizes = np.asarray(split.fold_sizes, dtype=float)
        assert abs(float(np.average(folds, weights=sizes)) - folds.mean()) > 1e-6

    def test_the_reading_names_the_level(self, z: float) -> None:
        surface = _level_surface(z)
        assert "learned_rule_reading" in [n.code for n in surface.summary().notes]
        assert "optimal_rule_reading" not in [n.code for n in surface.summary().notes]
        assert any(
            line.startswith("learned-rule reading")
            and line.endswith(f"every rule holds the intermediate at Z = {z:g}")
            for line in surface.population_lines()
        )


def test_an_unseeded_frozen_level_fit_replays_with_its_own_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W-j4f7: the frozen replay keeps the fit's repeat seeds (X11 (f4), the seed change).

    The refit of a cell, run on the fit's own table, reproduces the stored estimate exactly.
    The mutation, the same refit without the seeds, draws other trees and moves it.
    """
    from sklearn.ensemble import RandomForestClassifier

    result = (
        _level_study()
        .identify(LearnedRuleValue(intermediate=1.0))
        .estimate(
            **_level_options(
                outcome_learner=RandomForestClassifier(n_estimators=20, min_samples_leaf=20),
                random_state=None,
            )
        )
    )
    assert result.estimator.random_state is None
    seeds = [repeat.seed for repeat in result.repeats]
    refit = type(result.estimator).refit
    calls: list[tuple[Any, dict[str, Any]]] = []

    def spy(estimator: Any, data: Any, **kwargs: Any) -> Any:
        calls.append((estimator, kwargs))
        return refit(estimator, data, **kwargs)

    monkeypatch.setattr(type(result.estimator), "refit", spy)
    grid = ConfounderStrengthGrid(treatment=(0.0,), intermediate=(0.0, 0.2), outcome=(0.0,))
    surface = simulated_confounding(result, estimand=support.NAME, grid=grid)
    monkeypatch.undo()
    assert surface.complete and len(calls) == 1
    replay, keywords = calls[0]
    assert keywords["frozen_seeds"] == seeds
    assert refit(replay, result.data, **keywords)[support.NAME].psi == result[support.NAME].psi
    stale = refit(replay, result.data, **{k: v for k, v in keywords.items() if k != "frozen_seeds"})
    assert stale[support.NAME].psi != result[support.NAME].psi


# ----------------------------------------------- the frozen replay beside comparators
#
# X11 (f4c).  The learned rules are frozen as FrozenLearnedRule and every known comparator at
# the densities the fit stored; the natural course carries no array.


def _offer(w: Any) -> np.ndarray:
    p = 0.5 * (np.asarray(w["W1"]) > 0)
    return np.column_stack([1.0 - p, p])


_CONTRASTS: dict[str, Any] = {
    "known": lambda: LearnedRuleContrast(
        against=(Static(0), Stochastic(_offer, name="offer", density_kind="known"))
    ),
    "natural": lambda: LearnedRuleContrast(against=(NaturalCourse(),)),
}

#: W-f4c: over the four cells the rule cell and each comparator cell of the "known" fit differ
#: by 0.043 to 0.138 (observed); the margin is half the smallest.
W_F4C_COLUMN_GAP = 0.02


@cache
def _contrast_fit(name: str, repeats: int = 1) -> Any:
    study = CausalStudy(
        support.law_frame(n=_N),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    return study.identify(_CONTRASTS[name]()).estimate(
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        cv_evaluation=True,
        simultaneous=False,
        random_state=0,
        repeats=repeats,
    )


@cache
def _contrast_surfaces(name: str) -> dict[str, Any]:
    result = _contrast_fit(name)
    return {
        alias: simulated_confounding(result, estimand=alias, grid=_GRID, random_state=_SEED)
        for alias in result.estimates
        if not alias.startswith("ey_natural_course[")
    }


@pytest.mark.parametrize("name", list(_CONTRASTS))
class TestTheFrozenReplayBesideComparators:
    """The anchor, the manual refit, the contrast, the freeze, R2 and R4 (plan 4.5)."""

    def test_every_cell_of_every_alias_is_the_frozen_refit(self, name: str) -> None:
        result, surfaces = _contrast_fit(name), _contrast_surfaces(name)
        assert result.sensitivity.capability("simulated_confounding").available
        reproduced = _frozen_refit(result, result.data)
        first = next(iter(surfaces.values()))
        for alias, surface in surfaces.items():
            assert surface.complete
            assert surface.cells[0].estimate == result[alias].psi
            assert reproduced[alias].psi == result[alias].psi
        for index, cell in enumerate(first.cells[1:], start=1):
            data = _replacement(result, first, cell.treatment_strength, cell.outcome_strength)
            # The manual refit evaluates the declared comparators again.
            manual = _frozen_refit(result, data)
            for alias, surface in surfaces.items():
                assert surface.cells[index].estimate == pytest.approx(
                    manual[alias].psi, abs=1e-12
                ), alias

    def test_each_contrast_is_the_rule_cell_minus_the_comparator_cell(self, name: str) -> None:
        result, surfaces = _contrast_fit(name), _contrast_surfaces(name)
        rule = surfaces[support.NAME]
        folds = result.nuisance.folds
        for alias, surface in surfaces.items():
            if not alias.startswith("ate_learned_rule["):
                continue
            comparator = alias.split(" vs ")[1].rstrip("]")
            for index, cell in enumerate(surface.cells):
                if comparator == "natural course":
                    if index == 0:
                        outcome = result.data.outcome
                    else:
                        outcome = _replacement(
                            result, surface, cell.treatment_strength, cell.outcome_strength
                        ).outcome
                    other = float(np.mean([outcome[test].mean() for _, test in folds]))
                else:
                    other = surfaces[f"ey_known_regime[{comparator}]"].cells[index].estimate
                    assert abs(rule.cells[index].estimate - other) > W_F4C_COLUMN_GAP
                assert cell.estimate == pytest.approx(rule.cells[index].estimate - other, abs=1e-12)

    def test_every_alias_carries_the_frozen_reading(self, name: str) -> None:
        """W-f4c-read."""
        for surface in _contrast_surfaces(name).values():
            codes = [n.code for n in surface.summary().notes]
            assert surface.rule_replay == "frozen"
            assert "learned_rule_reading" in codes and "optimal_rule_reading" not in codes

    def test_a_cell_averages_the_fold_values_with_equal_weights(self, name: str) -> None:
        """W-f4c-fold: folds of 81 and 80 rows, for the rule and the first contrast."""
        result, surfaces = _contrast_fit(name), _contrast_surfaces(name)
        cell = surfaces[support.NAME].cells[-1]
        data = _replacement(
            result, surfaces[support.NAME], cell.treatment_strength, cell.outcome_strength
        )
        refit = _frozen_refit(result, data)
        (split,) = refit.extra["learned_rule"].splits
        assert split.fold_sizes == (81, 81, 81, 80, 80)
        contrast = next(a for a in surfaces if a.startswith("ate_learned_rule["))
        assert np.allclose(
            _fold_values(result, refit, support.NAME), split.fold_estimates, atol=1e-12
        )
        for alias in (support.NAME, contrast):
            folds = _fold_values(result, refit, alias)
            sizes = np.asarray(split.fold_sizes, dtype=float)
            assert surfaces[alias].cells[-1].estimate == pytest.approx(folds.mean(), abs=1e-12)
            assert abs(float(np.average(folds, weights=sizes)) - folds.mean()) > 1e-6

    def test_the_natural_course_value_and_a_repeated_fit_are_refused(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R2 on the natural-course value, and R4 at repeats=2, before any draw."""
        from tests.unit._confounding_support import forbid_draw_and_refit

        result = _contrast_fit(name)
        if name == "natural":
            alias = "ey_natural_course[natural course]"
            forbid_draw_and_refit(monkeypatch, result.estimator)
            with pytest.raises(CapabilityError, match="it is E\\[Y\\]") as caught:
                simulated_confounding(result, estimand=alias, grid=_GRID)
            assert "ate_learned_rule[learned rule vs natural course]" in str(
                caught.value.refusal.alternative
            )
            monkeypatch.undo()
        repeated = _contrast_fit(name, 2)
        forbid_draw_and_refit(monkeypatch, repeated.estimator)
        with pytest.raises(CapabilityError) as caught:
            simulated_confounding(repeated, estimand=support.NAME, grid=_GRID)
        assert caught.value.refusal.reason is RefusalReason.DIFFERENT_ESTIMAND


def _fold_values(fit: Any, result: Any, alias: str) -> np.ndarray:
    """The per-fold values of an alias of ``fit`` in its refit ``result``.

    Each fold's mean of the targeted value of the alias's column.

    The rule's column reproduces the record's ``fold_estimates``; a known regime reads its
    own column, the natural course the fold's outcome mean, and a contrast the difference.
    """
    targeted = result.fluctuations["regime"].targeted
    values = np.asarray(result.nuisance.regimes.values, dtype=float)
    predictions = np.column_stack([targeted.arms[level] for level in targeted.levels])
    names = result.nuisance.regimes.names
    folds = result.nuisance.folds

    def column(name: str) -> np.ndarray:
        if name == "natural course":
            return np.asarray([result.data.outcome[test].mean() for _, test in folds])
        mixture = np.einsum("ij,ij->i", values[:, :, names.index(name)], predictions)
        return np.asarray([mixture[test].mean() for _, test in folds])

    key = fit.parameter_keys[alias]
    if key.reference is None:
        return column(key.value)
    return column(key.value) - column(key.reference)


def test_the_comparators_are_frozen_and_a_changed_callback_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W-f4r4 on the frozen replay: the replay's columns are the stored ones (R5)."""
    from tests.unit._confounding_support import forbid_draw_and_refit

    result = _contrast_fit("known")
    alias = "ate_learned_rule[learned rule vs offer]"
    replay, _, kind = request_module.validate_learned_rule_replay(
        result, alias, result.parameter_keys[alias]
    )
    assert kind == "frozen" and isinstance(replay.learned_rule, FrozenLearnedRule)
    stored = np.asarray(result.nuisance.regimes.values)
    for index, frozen in enumerate(replay.interventions, start=1):
        assert np.array_equal(frozen.density(result.data), stored[:, :, index])
    state = {"moved": False}

    def stateful(w: Any) -> np.ndarray:
        p = (0.6 if state["moved"] else 0.5) * (np.asarray(w["W1"]) > 0)
        return np.column_stack([1.0 - p, p])

    plan = Stochastic(stateful, name="offer", density_kind="known")
    study = CausalStudy(
        support.law_frame(n=_N),
        design=PointTreatment(outcome="Y", treatment="A", adjustment=support.COVARIATES),
    )
    fit = study.identify(LearnedRuleContrast(against=(plan,))).estimate(
        outcome_learner=support.outcome_learner(),
        treatment_learner=support.treatment_learner(),
        n_folds=5,
        cv_evaluation=True,
        simultaneous=False,
        random_state=0,
    )
    state["moved"] = True
    forbid_draw_and_refit(monkeypatch, fit.estimator)
    with pytest.raises(CapabilityError, match="disagree with a stored cross-fitting draw"):
        simulated_confounding(fit, estimand=alias, grid=_GRID)


def test_the_frozen_replay_at_a_level_beside_a_known_regime() -> None:
    """X11 (j4a) beside Static(1): every alias is the frozen refit at the level."""
    result = (
        _level_study()
        .identify(LearnedRuleContrast(against=(Static(1),), intermediate=1.0))
        .estimate(**_level_options())
    )
    surfaces = {
        alias: simulated_confounding(result, estimand=alias, grid=_LEVEL_GRID)
        for alias in result.estimates
    }
    first = next(iter(surfaces.values()))
    for index, cell in enumerate(first.cells[1:], start=1):
        manual = _level_refit(result, level_table(result, first, cell))
        for alias, surface in surfaces.items():
            assert surface.rule_replay == "frozen"
            assert surface.cells[index].estimate == pytest.approx(manual[alias].psi, abs=1e-12)
