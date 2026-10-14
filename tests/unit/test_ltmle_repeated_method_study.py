"""The design of the repeated-outcome longitudinal study, checked before any run.

Each number the study declares before its run is derived here from the exact law in
:mod:`tests.studies.ltmle_repeated_law`: the truths, the control limits, the replication
budgets, and the power of every control.  The tests assert the properties of the law
(section "the law"), the limit functional and its agreement with cleverly on a sample, the
runner and the R adapter, and the budgets.  The fast tier reads no committed property
artifact here.  The framework tests in ``tests/unit/test_method_evidence.py`` recompute the
committed verdicts.
"""

from __future__ import annotations

import argparse
import inspect
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, binom, norm

from cleverly.datasets import REPEATED_RULE_LABEL, make_longitudinal_repeated, repeated_truth
from cleverly.datasets.longitudinal import (
    _REPEATED_A,
    _REPEATED_A1,
    _REPEATED_C,
    _REPEATED_C1,
    _REPEATED_L,
    _REPEATED_Y,
)
from cleverly.longitudinal import LTMLE, LongitudinalData
from tests.studies import canonical_ltmle_repeated as study
from tests.studies import ltmle_repeated_law as law
from tests.studies import ltmle_repeated_properties as properties
from tests.studies.evidence import replication
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.registry import registered
from tests.studies.fractional_glm import QuasiBinomialSeparationWarning

PLANS = {"never": "never", "always": "always", REPEATED_RULE_LABEL: law.RULE}
CONFIDENCE = study.STUDY.margins.confidence_level
Z = float(norm.ppf(0.5 + CONFIDENCE / 2.0))
MARGIN = study.STUDY.margins.standardized_bias


def _parse(name: str) -> tuple[str, int, str | None]:
    """``(plan, horizon, reference)`` of a fit's estimand name, in the law's plan names."""
    if name.startswith("ate_regimen["):
        comparison, horizon = name[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
        left, right = comparison.split(" vs ")
        return PLANS[left], int(horizon), PLANS[right]
    plan, horizon = name[len("ey_regimen[") : -1].rsplit(" @ t=", 1)
    return PLANS[plan], int(horizon), None


def _sd(label: str, family: str) -> float:
    """The efficiency bound's standard deviation of a property label at its family's ``n``."""
    plan, horizon, reference = _parse(study.label_estimand(label))
    scale = law.BINARY.eif_sd(plan, horizon, reference=reference)
    return scale / np.sqrt(properties.family_n(family))


def _discriminated(gap: float, scale: float, replicates: int) -> float:
    """P(the control's 99% bias interval clears ``MARGIN`` spreads), at standardized ``gap``."""
    shift = abs(gap) / scale
    return float(norm.cdf(np.sqrt(replicates) * (shift - MARGIN) - Z))


def _equivalent(bias: float, replicates: int) -> float:
    """P(an arm's 99% bias interval lies inside ``MARGIN`` spreads), at standardized ``bias``."""
    root = np.sqrt(replicates)
    return float(norm.cdf(root * (MARGIN - bias) - Z) - norm.cdf(-root * (MARGIN + bias) + Z))


# ---------------------------------------------------------------------------- the record


def test_the_record_is_registered_with_its_own_seeds() -> None:
    record = study.STUDY
    assert record in registered()
    others = [other for other in registered() if other.slug != record.slug]
    taken = {other.seed for other in others} | {other.resampling_seed for other in others}
    assert record.seed not in taken
    assert record.resampling_seed not in taken
    assert record.seed != record.resampling_seed
    assert (record.replicates, record.n) == (2_400, 3_000)
    assert record.publication_policy == "gated"
    assert record.artifacts.name == "ltmle_repeated"


def test_the_primary_study_reports_every_unique_parameter_once() -> None:
    assert len(set(study.ESTIMANDS)) == len(study.ESTIMANDS) == 13
    assert set(study.TRUTH) == set(study.ESTIMANDS)
    assert not set(study.DUPLICATES) & set(study.ESTIMANDS)
    gated = {study.label_estimand(label) for label in study.GATED_LABELS}
    earlier = {study.label_estimand(label) for label in study.EARLIER_LABELS}
    assert gated | earlier == set(study.ESTIMANDS)
    assert not gated & earlier


# ---------------------------------------------------------------------------- the law


@pytest.mark.parametrize("plan", ["never", "always", REPEATED_RULE_LABEL])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_the_truth_agrees_three_ways(plan: str, horizon: int) -> None:
    """The generator's enumeration, the forward composition and the observed g-formula."""
    enumerated = repeated_truth(plan, horizon)
    assert law.BINARY.counterfactual(PLANS[plan], horizon) == pytest.approx(enumerated, abs=1e-12)
    assert law.BINARY.g_formula(PLANS[plan], horizon) == pytest.approx(enumerated, abs=1e-12)


def test_the_reversed_rule_is_identified_and_far_from_the_rule() -> None:
    for horizon in (2, 3):
        reversed_value = law.BINARY.counterfactual("reversed", horizon)
        assert law.BINARY.g_formula("reversed", horizon) == pytest.approx(reversed_value, abs=1e-12)
        gap = reversed_value - repeated_truth(REPEATED_RULE_LABEL, horizon)
        assert gap < -0.05, horizon


def test_the_continuous_truths_are_identified() -> None:
    for plan in ("never", "always", law.RULE):
        assert law.CONTINUOUS.g_formula(plan, 3) == pytest.approx(
            law.CONTINUOUS.counterfactual(plan, 3), abs=1e-12
        )
    assert properties.CONTINUOUS_TRUTH["always_vs_never_t3"] < -0.05


def test_the_outcome_reads_two_nodes_back() -> None:
    """``E[Y_3 | full history]`` moves with ``Y_1`` and with ``L_2``: the second-order terms are live."""
    paths = law.BINARY.paths
    observed = paths.loc[paths["C3"] == 1.0]
    keys = ["W", "Y1", "L2", "Y2", "L3", "A3"]
    mass = observed.groupby(keys)["p"].sum()
    mean = (observed["p"] * observed["Y3"]).groupby([observed[k] for k in keys]).sum() / mass
    table = pd.DataFrame({"mean": mean, "mass": mass}).reset_index()

    def effect(column: str) -> tuple[float, float]:
        """The mass-weighted mean and the minimum of the difference ``column = 1`` minus ``0``."""
        others = [k for k in keys if k != column]
        wide = table.pivot_table(index=others, columns=column, values="mean")
        weight = table.groupby(others)["mass"].sum().reindex(wide.index)
        difference = wide[1.0] - wide[0.0]
        return float(np.average(difference, weights=weight)), float(difference.min())

    y1_mean, y1_min = effect("Y1")
    l2_mean, l2_min = effect("L2")
    assert y1_mean >= 0.2
    assert l2_mean >= 0.02
    # The sign never reverses, so no cell masks the effect in the average.
    assert y1_min > 0.1
    assert l2_min > 0.005


def test_a_markov_reading_misses_the_truth() -> None:
    """The first-order history is wrong at ``t = 3`` and exact at ``t <= 2``."""
    n = properties.FULL_HISTORY_N
    for label in study.GATED_LABELS:
        plan, horizon, reference = _parse(study.label_estimand(label))
        truth = _law_functional(plan, horizon, reference, history="full")
        coarse = _law_functional(plan, horizon, reference, history="markov")
        sd = law.BINARY.eif_sd(plan, horizon, reference=reference) / np.sqrt(n)
        assert abs(coarse - truth) / sd >= 0.5, label
    for label in (*study.EARLIER_LABELS, "never_t1"):
        plan, horizon, reference = _parse(study.label_estimand(label))
        truth = _law_functional(plan, horizon, reference, history="full")
        coarse = _law_functional(plan, horizon, reference, history="markov")
        assert coarse == pytest.approx(truth, abs=1e-12), label


def _law_functional(plan: str, horizon: int, reference: str | None, **options: Any) -> float:
    """The g-formula of a plan, or of its contrast with ``reference``."""
    value = law.BINARY.g_formula(plan, horizon, **options)
    if reference is not None:
        value -= law.BINARY.g_formula(reference, horizon, **options)
    return value


def test_node_one_is_unsaturated() -> None:
    """Three baseline levels against a one-column design: the first fluctuation can move."""
    assert len(set(law.BINARY.paths["W"])) == 3
    frame, _ = make_longitudinal_repeated(n=300, seed=14, backend="pandas")
    design = _data(frame).history_design(1)
    parameters = design.shape[1] + 1  # the intercept
    assert design.shape[1] == 1
    assert parameters < len(np.unique(design[:, 0]))


def test_censoring_reads_the_history_and_never_the_current_outcome() -> None:
    """``C_t`` reads ``L_t``, ``Y_{t-1}``, ``Y_{t-2}`` and ``A_t`` only, and each term is nonzero."""
    assert set(_REPEATED_C1) == {"intercept", "w", "a"}
    assert set(_REPEATED_C) == {"intercept", "l", "y", "a", "y2"}
    assert set(_REPEATED_A) == {"intercept", "w", "l", "y", "a", "y2"}
    assert set(_REPEATED_L) == {"intercept", "w", "a", "y", "l1"}
    assert set(_REPEATED_Y) == {"intercept", "w", "a", "l", "y", "ay", "y2", "l1"}
    assert all(
        value != 0.0
        for table in (
            _REPEATED_A1,
            _REPEATED_A,
            _REPEATED_C1,
            _REPEATED_C,
            _REPEATED_L,
            _REPEATED_Y,
        )
        for value in table.values()
    )
    # On the paths, staying at a node is independent of that node's outcome given H_t, A_t.
    paths = law.BINARY.paths
    for time in (2, 3):
        keys = [*law.BINARY.history(time), f"A{time}", f"Y{time}"]
        weights = paths["p"]
        stayed = weights * (paths[f"C{time}"] == 1.0)
        rate = (
            stayed.groupby([paths[k] for k in keys]).sum()
            / weights.groupby([paths[k] for k in keys]).sum()
        )
        by_outcome = rate.unstack(f"Y{time}")
        np.testing.assert_allclose(by_outcome[0.0], by_outcome[1.0], atol=1e-12)


def test_every_plan_keeps_positivity_through_the_last_node() -> None:
    """The minimum cumulative probability and the follower share, with the mass near zero."""
    weights_of = law.BINARY.paths["p"].to_numpy()
    for plan in ("never", "always", law.RULE, "reversed"):
        followed = law.BINARY.followed(plan, 3)
        cumulative = law.BINARY.cumulative(plan, 3)[followed]
        assert float(np.min(cumulative)) > 0.02, plan
        assert float(np.sum(weights_of[followed])) > 0.10, plan
        small = float(np.sum(weights_of[followed][cumulative < 0.05]))
        assert small / float(np.sum(weights_of[followed])) < 0.05, plan


def test_every_contrast_is_far_from_zero() -> None:
    """A sign error in a contrast would show: the truth sits several EIF spreads from zero."""
    for name in study.ESTIMANDS:
        if not name.startswith("ate_regimen["):
            continue
        plan, horizon, reference = _parse(name)
        sd = law.BINARY.eif_sd(plan, horizon, reference=reference) / np.sqrt(study.PRIMARY_N)
        floor = 5.0 if horizon == 1 else 2.5
        assert abs(study.TRUTH[name]) / sd >= floor, name


def _data(frame: pd.DataFrame) -> LongitudinalData:
    return LongitudinalData.from_frame(frame, **study.COLUMNS)


def test_the_known_mechanisms_read_the_declared_history_design() -> None:
    frame, _ = make_longitudinal_repeated(n=800, seed=11, backend="pandas")
    data = _data(frame)
    treatment = study.KnownRepeatedMechanism("treatment").fit(None, None)
    censoring = study.KnownRepeatedMechanism("censoring").fit(None, None)
    filled = frame.fillna(0.0)
    w = filled["W"].to_numpy()
    zeros = np.zeros(len(frame))
    for time in (1, 2, 3):
        y_prev = filled[f"Y{time - 1}"].to_numpy() if time > 1 else zeros
        y_prev2 = filled[f"Y{time - 2}"].to_numpy() if time > 2 else zeros
        a_prev = filled[f"A{time - 1}"].to_numpy() if time > 1 else zeros
        l_t = filled[f"L{time}"].to_numpy() if time > 1 else zeros
        a_t = filled[f"A{time}"].to_numpy()
        history = data.history_design(time)
        assert history.shape[1] == {1: 1, 2: 4, 3: 7}[time]
        expected_g = np.asarray(law.BINARY.treatment(time, w, l_t, y_prev, a_prev, y_prev2))
        current = data.history_design(time, include_current=True)
        assert current.shape[1] == {1: 2, 2: 5, 3: 8}[time]
        expected_c = np.asarray(law.BINARY.stay(time, w, l_t, y_prev, a_t, y_prev2))
        at_risk = frame[f"A{time}"].notna().to_numpy()
        np.testing.assert_allclose(
            treatment.predict_proba(history)[at_risk, 1], expected_g[at_risk], atol=1e-12
        )
        np.testing.assert_allclose(
            censoring.predict_proba(current)[at_risk, 1], expected_c[at_risk], atol=1e-12
        )


def test_a_column_swap_moves_the_known_mechanisms() -> None:
    """The deliberate mutation: the width check alone cannot see a permuted design."""
    frame, _ = make_longitudinal_repeated(n=800, seed=12, backend="pandas")
    data = _data(frame)
    # The censoring probabilities sit near 0.97 and read ``L_t`` and ``Y_{t-1}`` with small
    # weights, so a swap moves them by 0.012 at least.  The floor sits below that move.
    for kind, include, floor in (("treatment", False, 0.1), ("censoring", True, 0.005)):
        model = study.KnownRepeatedMechanism(kind).fit(None, None)
        for time in (2, 3):
            design = data.history_design(time, include_current=include)
            # ``Y_{t-1}`` sits just before ``L_t`` in the covariate history.
            outcome = 2 * (time - 1) - 1
            swapped = design.copy()
            swapped[:, [outcome, outcome + 1]] = swapped[:, [outcome + 1, outcome]]
            moved = np.abs(model.predict_proba(design)[:, 1] - model.predict_proba(swapped)[:, 1])
            assert float(np.max(moved)) > floor, (kind, time)


def test_swapping_the_two_outcomes_moves_the_node_three_mechanisms() -> None:
    """A nonzero witness for the second-order term: ``Y1`` and ``Y2`` carry different weights."""
    frame, _ = make_longitudinal_repeated(n=800, seed=15, backend="pandas")
    data = _data(frame)
    for kind, include, floor in (("treatment", False, 0.1), ("censoring", True, 0.02)):
        model = study.KnownRepeatedMechanism(kind).fit(None, None)
        design = data.history_design(3, include_current=include)
        swapped = design.copy()
        swapped[:, [1, 3]] = swapped[:, [3, 1]]  # Y1 <-> Y2
        moved = np.abs(model.predict_proba(design)[:, 1] - model.predict_proba(swapped)[:, 1])
        assert float(np.max(moved)) > floor, kind
        # At node 2 the second-order term reads the zero node, so the term is absent there.
        assert data.history_design(2, include_current=include).shape[1] < design.shape[1]


ADAPTER = Path(__file__).parents[1] / "canonical" / "ltmle_repeated_adapter.R"

#: The R variable each term of a ``plogis`` line multiplies, and the generator term it restates.
#: The generator's one arm term ``a`` splits in two.  The treatment tables read the prior arm
#: ``abar[, t-1]`` (``a_prev``), and the censoring tables read the current arm ``abar[, t]``
#: (``a_now``), so an exchange of the two columns changes a table.
_ADAPTER_TERMS = {
    None: "intercept",
    "w": "w",
    "l": "l",
    "y": "y",
    "y2": "y2",
    "abar_prev": "a_prev",
    "abar_now": "a_now",
}
PLOGIS_LINE = r"^\s*p_[ac] <- plogis\((.*)\)\s*$"
_ADAPTER_ASSIGNMENTS = (
    'l <- fill(sprintf("L%d", t))',
    'y <- fill(sprintf("Y%d", t - 1))',
    'y2 <- if (t >= 3) fill(sprintf("Y%d", t - 2)) else 0',
)


def _adapter_tables(text: str) -> list[dict[str, float]]:
    """The coefficient table of each ``plogis`` line of the R adapter, in file order."""
    expressions = re.findall(PLOGIS_LINE, text, flags=re.MULTILINE)
    tables = []
    for expression in expressions:
        expression = re.sub(r"abar\[,\s*t\s*-\s*1\s*\]", "abar_prev", expression)
        expression = re.sub(r"abar\[,\s*(?:t|1)\s*\]", "abar_now", expression)
        table: dict[str, float] = {}
        for sign, number, name in re.findall(
            r"([+-]?)\s*(\d+(?:\.\d+)?)(?:\s*\*\s*(\w+))?", expression
        ):
            key = _ADAPTER_TERMS[name or None]
            assert key not in table, (key, expression)
            table[key] = float(f"{sign}{number}")
        tables.append(table)
    return tables


def _named_arm(table: dict[str, float], name: str) -> dict[str, float]:
    """The generator table with its arm term ``a`` renamed to the adapter's ``name``."""
    return {(name if key == "a" else key): value for key, value in table.items()}


def _adapter_restates_the_generator(text: str) -> bool:
    """Whether the four ``plogis`` lines and the three history reads match the generator."""
    declared = [
        _REPEATED_A1,
        _named_arm(_REPEATED_C1, "a_now"),
        _named_arm(_REPEATED_A, "a_prev"),
        _named_arm(_REPEATED_C, "a_now"),
    ]
    return _adapter_tables(text) == declared and all(line in text for line in _ADAPTER_ASSIGNMENTS)


def test_the_r_adapter_restates_the_generator_coefficients() -> None:
    """The R copy of the mechanisms is checked before any run, and a changed literal fails it."""
    text = ADAPTER.read_text()
    assert len(_adapter_tables(text)) == 4
    assert _adapter_restates_the_generator(text)


def test_a_changed_adapter_literal_fails_the_transcription_guard() -> None:
    """The deliberate mutation: each literal of each ``plogis`` line, changed alone, is caught."""
    text = ADAPTER.read_text()
    spans = []
    for line in re.finditer(PLOGIS_LINE, text, flags=re.MULTILINE):
        for match in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?!\s*\])", line.group(1)):
            spans.append((line.start(1) + match.start(), line.start(1) + match.end()))
    # A1: 2, C1: 3, A: 6 and C: 5 literals.
    assert len(spans) == 16
    for start, end in spans:
        mutated = text[:start] + f"{float(text[start:end]) + 0.1:.1f}" + text[end:]
        assert not _adapter_restates_the_generator(mutated), text[start - 12 : end + 6]
    # A dropped term and a wrong history column are caught as well.
    assert not _adapter_restates_the_generator(text.replace(" + 1.6 * y2", ""))
    assert not _adapter_restates_the_generator(text.replace('Y%d", t - 2', 'Y%d", t - 1'))


def test_exchanging_the_prior_and_current_arm_fails_the_transcription_guard() -> None:
    """The treatment tables read ``abar[, t - 1]`` and the censoring tables read ``abar[, t]``."""
    text = ADAPTER.read_text()
    prior = re.compile(r"abar\[,\s*t\s*-\s*1\s*\]")
    current = re.compile(r"abar\[,\s*t\s*\]")
    lines = text.splitlines()
    treatment = next(
        line for line in lines if re.match(r"^\s*p_a <- ", line) and prior.search(line)
    )
    censoring = next(
        line for line in lines if re.match(r"^\s*p_c <- ", line) and current.search(line)
    )
    assert prior.search(treatment) and current.search(censoring)
    # Both columns exchanged, then each line alone.
    both = prior.sub("ARM_NOW", text)
    both = current.sub("abar[, t - 1]", both).replace("ARM_NOW", "abar[, t]")
    assert not _adapter_restates_the_generator(both)
    only_treatment = text.replace(treatment, prior.sub("abar[, t]", treatment))
    only_censoring = text.replace(censoring, current.sub("abar[, t - 1]", censoring))
    assert not _adapter_restates_the_generator(only_treatment)
    assert not _adapter_restates_the_generator(only_censoring)
    # The first-node censoring line reads the current arm, abar[, 1], in the same key.
    assert _adapter_tables(text)[1]["a_now"] == _REPEATED_C1["a"]


def test_the_subject_publishes_no_plug_in() -> None:
    """A single-fold fit exposes no untargeted plug-in, so the column is ``nan`` for every cell."""
    frame, truth = make_longitudinal_repeated(n=study.PRIMARY_N, seed=16, backend="pandas")
    rows = pd.DataFrame(study.cleverly_rows(frame, truth, study.SCENARIO, 0))
    assert set(rows["estimand"]) == set(study.ESTIMANDS)
    assert rows["initial_estimate"].isna().all()
    # The estimates themselves are finite, so the column is absent by design and not by failure.
    assert np.isfinite(rows["estimate"].to_numpy(dtype=float)).all()


def test_the_rule_duplicates_always_at_the_first_node() -> None:
    assert repeated_truth(REPEATED_RULE_LABEL, 1) == repeated_truth("always", 1)
    frame, _ = make_longitudinal_repeated(n=1_000, seed=13, backend="pandas")
    result = study.fit_cleverly(frame)
    assert set(result.estimates) == set(study.ESTIMANDS) | set(study.DUPLICATES)
    left, right = study.DUPLICATES
    assert float(result[left].psi) == pytest.approx(
        float(result["ey_regimen[always @ t=1]"].psi), abs=1e-12
    )
    assert float(result[right].psi) == pytest.approx(
        float(result["ate_regimen[always vs never @ t=1]"].psi), abs=1e-12
    )


# ---------------------------------------------------------------------------- the controls


_SAMPLE_SEED = 5_001
_SAMPLE_N = 5_000
_PLAN_OF = {"never": "never", "always": "always", "rule": law.RULE}


def _plan_of(label: str) -> str:
    return _PLAN_OF[label.rsplit("_t", 1)[0]]


@pytest.mark.parametrize(
    ("configuration", "kind", "tolerance"),
    [("both_wrong", "baseline", 1e-10), ("markov_history", "markov", 1e-4)],
)
def test_the_limit_functional_reproduces_cleverly_on_a_sample(
    configuration: str, kind: str, tolerance: float
) -> None:
    """The functional on a sample's empirical measure is the value the fit returns.

    The single-fold fit with cell-mean regressions and intercept-only mechanisms is the
    functional of the sample.  The baseline regressions agree to rounding.  The Markov
    regressions have cells whose mean is exactly 0 or 1, and the fit bounds its submodel's
    probabilities at ``alpha = 0.9995``, which moves the estimate by about 1e-5.  The mutation
    is the same functional with a constant clever covariate: it reads no fluctuation, and it
    must miss the fit for the rule.
    """
    frame = properties.sample(configuration, _SAMPLE_SEED, _SAMPLE_N)
    result = properties.fit(frame, configuration)
    for label in ("never_t3", "always_t3", "rule_t3"):
        plan = _plan_of(label)
        value = result[study.label_estimand(label)].psi
        limit = law.empirical_tmle_limit(frame, plan, 3, q_history=kind)
        assert value == pytest.approx(limit, abs=tolerance), label
    filled = law.with_latent(frame.fillna(0.0).reset_index(drop=True))
    weights = np.full(len(filled), 1.0 / len(filled))
    unfluctuated = law.tmle_limit_functional(
        filled,
        weights,
        law.RULE,
        3,
        q_history=kind,
        cumulative=lambda time: np.ones(len(filled)),
    )
    if kind == "markov":
        fitted = result[study.label_estimand("rule_t3")].psi
        assert abs(fitted - unfluctuated) > 1e-3


def _baseline_plug_in(frame: pd.DataFrame, plan: str, horizon: int) -> float:
    """The iterated cell means of ``W`` among the followers, from the law module's helpers."""
    filled = law.with_latent(frame.fillna(0.0).reset_index(drop=True))
    weights = np.full(len(filled), 1.0 / len(filled))
    carried = filled[f"Y{horizon}"].to_numpy().astype(float)
    for time in range(horizon, 0, -1):
        followed = law.followed_mask(filled, plan, time)
        carried = law.conditional_mean(filled, weights, carried, followed, ["W"])
    return float(np.mean(carried))


def test_the_untargeted_arm_is_the_w_only_plug_in() -> None:
    """``untargeted`` is the cell-mean chain, and the targeted fit differs from it."""
    frame = properties.sample("targeted", 77, 2_000)
    targeted = properties.fit(frame, "mechanism_correct")
    for label in study.GATED_LABELS:
        plan, horizon, reference = _parse(study.label_estimand(label))
        expected = _baseline_plug_in(frame, plan, horizon)
        if reference is not None:
            expected -= _baseline_plug_in(frame, reference, horizon)
        assert properties.untargeted(frame, label) == pytest.approx(expected, abs=1e-12)
        # A nonzero witness: the fluctuation moves every label on this sample.
        moved = targeted[study.label_estimand(label)].psi - expected
        assert abs(moved) > 1e-3, label


def test_the_both_wrong_fluctuation_is_exactly_zero_for_static_plans() -> None:
    """Intercept mechanisms make the static clever covariate constant, so the fit is the plug-in."""
    frame = properties.sample("both_wrong", 77, 2_000)
    result = properties.fit(frame, "both_wrong")
    for label in ("never_t3", "always_t3", "always_vs_never_t3"):
        assert result[study.label_estimand(label)].psi == pytest.approx(
            properties.untargeted(frame, label), abs=1e-12
        ), label
    # The rule's clever covariate reads Y_{t-1}, so its fit leaves the plug-in.
    rule = study.label_estimand("rule_t3")
    assert abs(result[rule].psi - properties.untargeted(frame, "rule_t3")) > 1e-3


class _Recorder:
    """A learner that records the design columns it is fitted on."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> _Recorder:
        self.seen = np.asarray(X)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


def test_the_markov_wrapper_reads_the_declared_columns() -> None:
    """Widths 1, 3 and 5 keep ``[0]``, ``[0, 1, 2]`` and ``[0, 3, 4]``."""
    columns = properties.CellMeansOn.columns
    assert [columns("markov", width) for width in (1, 3, 5)] == [[0], [0, 1, 2], [0, 3, 4]]
    assert [columns("baseline", width) for width in (1, 3, 5)] == [[0]] * 3
    assert columns("full", 5) == [0, 1, 2, 3, 4]
    design = np.arange(30.0).reshape(6, 5)
    wrapper = properties.CellMeansOn("markov", _Recorder()).fit(design, np.zeros(6))
    np.testing.assert_array_equal(wrapper.fitted_.seen, design[:, [0, 3, 4]])
    with pytest.raises(ValueError):
        columns("second", 3)
    assert not hasattr(properties.CellMeansOn("markov", _Recorder()), "predict_proba")


def test_the_markov_control_differs_from_the_full_history_arm_on_a_sample() -> None:
    """Mutation: a wrapper that kept every column would make the two arms equal."""
    frame = properties.sample("markov_history", _SAMPLE_SEED, _SAMPLE_N)
    positive = properties.fit(frame, "full_history")
    control = properties.fit(frame, "markov_history")
    for label in ("never_t3", "always_t3"):
        name = study.label_estimand(label)
        assert abs(positive[name].psi - control[name].psi) > 1e-3, label
    # The mutation: the same learner on every column reproduces the full-history arm.
    mutated = properties.LTMLE(
        properties.regimens(),
        reference="never",
        outcome_learner=properties.CellMeansOn("full", properties.CellMeansBackoff()),
        pseudo_learner=properties.CellMeansOn("full", properties.CellMeansBackoff()),
        treatment_learner=properties.DummyClassifier(strategy="prior"),
        censoring_learner=properties.DummyClassifier(strategy="prior"),
        n_folds=1,
        g_bounds=properties.G_BOUNDS,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, **properties.COLUMNS)
    name = study.label_estimand("never_t3")
    assert mutated[name].psi == pytest.approx(positive[name].psi, abs=1e-12)


def _control_limits() -> list[tuple[str, str, float, float]]:
    """``(family, label, limit, truth)`` for every control that reads an exact limit."""
    truth = properties.TRUTH
    designs = []
    for label in properties.ABSORBING_LABELS:
        designs.append(
            (
                "repeated_outcome_necessity",
                label,
                properties.ABSORBING_LIMIT[label],
                truth[study.label_estimand(label)],
            )
        )
    for label in properties.HISTORY_LABELS:
        designs.append(
            (
                "outcome_history_necessity",
                label,
                properties.HISTORY_FREE_LIMIT[label],
                truth[study.label_estimand(label)],
            )
        )
    for label in study.GATED_LABELS:
        value = truth[study.label_estimand(label)]
        designs.append(("double_robustness", label, properties.BOTH_WRONG_LIMIT[label], value))
        designs.append(("full_history_necessity", label, properties.MARKOV_LIMIT[label], value))
        designs.append(("targeting_necessity", label, properties.PLUG_IN_LIMIT[label], value))
    for label in properties.RULE_LABELS:
        designs.append(
            (
                "rule_necessity",
                label,
                properties.REVERSED_TRUTH[label],
                truth[study.label_estimand(label)],
            )
        )
    return designs


def test_each_control_limit_sits_away_from_the_truth() -> None:
    """Every control's limit is at least half an efficiency-bound spread from the truth."""
    for family, label, limit, truth in _control_limits():
        gap = abs(limit - truth)
        assert gap >= 0.5 * _sd(label, family), (family, label)


def test_the_history_free_reading_is_refused_for_a_rule() -> None:
    with pytest.raises(ValueError, match="rule"):
        law.BINARY.g_formula(law.RULE, 3, history="free")


# ---------------------------------------------------------------------------- the limits


def test_the_tmle_limit_is_inert_exactly_where_theory_says() -> None:
    """Intercept mechanisms make a static plan's clever covariate constant, so nothing moves."""
    for kind in ("baseline", "markov"):
        for plan in ("never", "always"):
            for horizon in (1, 2, 3):
                limit = law.BINARY.tmle_limit(plan, horizon, q_history=kind)
                assert limit == pytest.approx(
                    law.BINARY.g_formula(plan, horizon, history=kind), abs=1e-12
                ), (kind, plan, horizon)
    # Nonzero witnesses: the rule's clever covariate reads ``Y_{t-1}``, which a coarse
    # regression does not, so the fluctuation moves.  Where the coarse history already holds
    # ``Y_{t-1}`` and the regression is the truth (markov at t = 2), it stays inert.
    moved = {
        (kind, horizon): abs(
            law.BINARY.tmle_limit(law.RULE, horizon, q_history=kind)
            - law.BINARY.g_formula(law.RULE, horizon, history=kind)
        )
        for kind in ("baseline", "markov")
        for horizon in (2, 3)
    }
    assert moved["markov", 3] > 0.005
    assert moved["baseline", 2] > 0.005
    assert moved["baseline", 3] > 0.001
    assert moved["markov", 2] < 1e-12


@pytest.mark.parametrize("plan", ["never", "always", law.RULE, "reversed"])
@pytest.mark.parametrize("kind", ["baseline", "markov", "full"])
def test_the_tmle_limit_with_the_true_mechanisms_is_the_truth(plan: str, kind: str) -> None:
    """Double robustness, an independent check of the functional.

    With the generating mechanisms the one-step limit equals the truth whatever the outcome
    regression reads, and the coarse plug-in does not.
    """
    for horizon in (2, 3):
        truth = law.BINARY.counterfactual(plan, horizon)
        limit = law.BINARY.tmle_limit(plan, horizon, q_history=kind, mechanisms="true")
        assert limit == pytest.approx(truth, abs=1e-10)
    if kind != "full":
        plug_in = law.BINARY.g_formula(plan, 3, history=kind)
        assert abs(plug_in - law.BINARY.counterfactual(plan, 3)) > 0.01


def _replications() -> dict[str, int]:
    """The replications each family draws, from the declared fit sets."""
    return {family: replicates for family, _, replicates in properties.FIT_SETS}


def _control_designs() -> list[tuple[str, str, float, float, int]]:
    """``(family, label, gap, scale, replicates)`` for every control the study gates."""
    return [
        (
            family,
            label,
            limit - truth,
            _sd(label, family),
            _replications()[family],
        )
        for family, label, limit, truth in _control_limits()
    ]


@pytest.mark.parametrize(
    ("family", "label", "gap", "scale", "replicates"),
    _control_designs(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_control_is_powered(
    family: str, label: str, gap: float, scale: float, replicates: int
) -> None:
    """The control's spread is approximated by the target plan's efficiency bound."""
    assert _discriminated(gap, scale, replicates) >= 0.99
    # The paired displacement, in the positive arm's spread, clears the declared threshold.
    assert abs(gap) / scale >= 2 * properties.NECESSITY_DISPLACEMENT
    # The positive arm, and a control that converges to its own limit, read as equivalent.
    assert _equivalent(0.0, replicates) >= 0.99


def test_the_smallest_control_gap_is_the_one_the_design_reports() -> None:
    shifts = {
        (family, label): abs(gap) / scale for family, label, gap, scale, _ in _control_designs()
    }
    weakest = min(shifts, key=shifts.__getitem__)
    assert weakest == ("outcome_history_necessity", "always_t3")
    assert shifts[weakest] == pytest.approx(0.73, abs=0.02)
    # The weakest control of the new families, for the record.
    assert shifts["double_robustness", "rule_t3"] == pytest.approx(0.74, abs=0.02)
    assert shifts["rule_necessity", "rule_t3"] > 2.0


# ---------------------------------------------------------------------------- the budgets


def test_the_calibration_budget_is_the_declared_rule_applied_to_the_pilot() -> None:
    """The family-wise product over 28 cells reaches 0.95 at the budget and not 400 below.

    Pilot P2 ran 2,000 draws per arm at ``n = 8,000``.  Its SE ratios have a standard
    deviation of 0.014 across the 28 cells, the sampling error of one ratio at that size, so
    the rule evaluates each cell 0.02 farther from one and the product binds at 25,600.
    """
    ratios = properties.CALIBRATION_PILOT_RATIOS
    pilot = properties.CALIBRATION_PILOT_REPLICATES
    assert properties.CALIBRATION_N == 8_000
    assert len(ratios) == 28
    assert set(ratios) == {
        f"{label}__{arm}"
        for arm in properties.CALIBRATION_ARMS
        for label in (
            (*study.GATED_LABELS, *study.EARLIER_LABELS)
            if arm == "estimated_mechanisms"
            else study.GATED_LABELS
        )
    }
    assert properties.calibration_replicates(ratios, pilot) == properties.CALIBRATION_REPLICATES
    evaluated = [properties.evaluated_ratio(value, pilot) for value in ratios.values()]

    def product(budget: int) -> float:
        return float(
            np.prod([properties.calibration_pass_probability(budget, ratio=r) for r in evaluated])
        )

    budget = properties.CALIBRATION_REPLICATES
    assert product(budget) >= 0.95
    assert product(budget - properties.BUDGET_STEP) < 0.95
    # The evaluated ratio lies on the far side of one for every cell, never nearer.
    assert all(
        abs(e - 1.0) > abs(value - 1.0) for e, value in zip(evaluated, ratios.values(), strict=True)
    )
    # The pilot point estimates alone would pass at 9,600, which the declared rule does not
    # use: it prices the pilot's own sampling error, as the primary rule does.
    at_point = np.prod(
        [properties.calibration_pass_probability(9_600, ratio=r) for r in ratios.values()]
    )
    assert at_point >= 0.95
    # A ratio of exactly one is evaluated on the upper side, and the pass probability falls
    # off both sides of one.
    assert properties.evaluated_ratio(1.0, pilot) > 1.0
    assert properties.calibration_pass_probability(
        budget, ratio=1.0
    ) > properties.calibration_pass_probability(budget, ratio=1.04)


def test_the_study_fit_clips_probabilities_where_ltmle_does() -> None:
    """``alpha`` is the upper clip, so 0.9999 matches R's ``[0.0001, 0.9999]`` and not 0.9995."""
    assert pytest.approx(1.0 - 1e-4) == study.PROBABILITY_BOUND
    # The library default is not the study's choice, and the study does not change it.
    assert inspect.signature(LTMLE).parameters["alpha"].default == 0.9995
    seen: dict[str, Any] = {}
    real = study.LTMLE

    def recorder(*args: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return real(*args, **kwargs)

    frame, _ = make_longitudinal_repeated(n=1_000, seed=13, backend="pandas")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "LTMLE", recorder)
        study.fit_cleverly(frame)
    assert seen["alpha"] == study.PROBABILITY_BOUND


def test_the_probability_bound_moves_an_estimate_on_a_separated_draw() -> None:
    """Nonzero witness: replicate 157 separates its fit, so the two bounds differ a little."""
    frame, _ = study.draw_scenario(study.SCENARIO, study.PRIMARY_N, 157)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuasiBinomialSeparationWarning)
        wide = study.fit_cleverly(frame)
        narrow = LTMLE(
            study.REGIMENS,
            reference=study.REFERENCE,
            outcome_learner=study.QuasiBinomialGLM(),
            pseudo_learner=study.QuasiBinomialGLM(),
            treatment_learner=study.KnownRepeatedMechanism("treatment"),
            censoring_learner=study.KnownRepeatedMechanism("censoring"),
            n_folds=1,
            g_bounds=study.G_BOUNDS,
            simultaneous=False,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        ).fit(frame, **study.COLUMNS)
    differences = [
        abs(float(wide[name].psi) - float(narrow[name].psi)) / float(wide[name].std_error)
        for name in study.ESTIMANDS
    ]
    assert 1e-6 < max(differences) < 1e-2


def test_the_small_sample_arms_are_reported_not_gated() -> None:
    """The pilot's ``n = 2,000`` in-sample shortfall publishes with no verdict."""
    assert properties.N == 2_000
    assert properties.family_n("small_sample_calibration") == properties.N
    assert properties.family_n("interval_calibration") == properties.CALIBRATION_N
    assert "small_sample_calibration" in properties.REPORTED_CALIBRATION


def test_the_absorbing_control_draws_at_its_own_n() -> None:
    """The run at 5728c3ce left N1 short of its limit at n = 2,000; it now draws 8,000."""
    assert properties.ABSORBING_N == 8_000
    assert properties.family_n("repeated_outcome_necessity") == properties.ABSORBING_N
    assert properties.family_n("outcome_history_necessity") == properties.N
    assert properties.FULL_HISTORY_N == 8_000
    assert properties.family_n("full_history_necessity") == properties.FULL_HISTORY_N


def test_the_backoff_learner_reads_a_seen_cell_and_extrapolates_an_unseen_one() -> None:
    """A seen row returns its cell mean; an unseen row returns the GLM, not the training mean."""
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([0.1, 0.3, 0.5, 0.7, 0.4, 0.6])
    learner = properties.CellMeansBackoff().fit(X, y)
    seen = learner.predict(np.array([[1.0, 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(seen, [0.6, 0.5])
    unseen = learner.predict(np.array([[1.0, 1.0]]))
    # The training mean is 0.433; the main-terms GLM puts the cell with both effects above 0.6.
    assert unseen[0] > 0.6
    np.testing.assert_allclose(unseen, learner.backoff_.predict(np.array([[1.0, 1.0]])))


def test_the_primary_budget_is_the_declared_rule_applied_to_the_pilot() -> None:
    """``primary_replicates`` returns the recorded budget, and each clause is minimal or met.

    The coverage-floor clause binds: 2,000 replications pass a cell at coverage 0.93 with
    probability 0.986, below 0.99, and 2,400 pass with 0.996.  The family-wise clause reads
    the pilot's one-sided 80% lower bounds, so it prices the pilot's sampling error.
    """
    covered, pilot = properties.PRIMARY_PILOT_COVERED, properties.PRIMARY_PILOT_REPLICATES
    assert (study.PRIMARY_N, study.PRIMARY_REPLICATES) == (3_000, 2_400)
    assert properties.primary_replicates(covered, pilot) == study.PRIMARY_REPLICATES
    assert min(covered.values()) / pilot >= 0.935
    replicates = study.PRIMARY_REPLICATES
    assert properties.bias_equivalent_probability(0.15, replicates) >= 0.90
    assert properties.coverage_floor_pass_probability(replicates, 0.93) >= 0.99
    assert properties.coverage_floor_pass_probability(replicates - 400, 0.93) < 0.99
    lower = {
        cell: float(beta.ppf(0.2, count, pilot - count + 1)) for cell, count in covered.items()
    }

    def family_wise(budget: int) -> float:
        return float(
            np.prod([properties.coverage_floor_pass_probability(budget, v) for v in lower.values()])
            ** 2
        )

    assert family_wise(replicates) >= 0.95
    # The deliberate mutation: one cell whose pilot coverage is 0.925 makes the family-wise
    # clause bind and raises the budget.
    worse = {**covered, "ey_regimen[treat then treat after an outcome @ t=3]": 1850}
    assert properties.primary_replicates(worse, pilot) > replicates
    # A cell at coverage 0.93 is bias-equivalent with probability 0.99 at 0.15 and fails
    # the bias clause at 0.30, which the 0.15 margin of the paired comparison guards.
    assert properties.bias_equivalent_probability(0.30, replicates) < 0.90


def test_the_shrunken_error_control_leaves_the_band() -> None:
    low, _ = study.STUDY.margins.calibration_se_ratio
    spread = properties.SHRUNKEN_SE_FACTOR / np.sqrt(2.0 * (properties.CALIBRATION_REPLICATES - 1))
    upper = properties.SHRUNKEN_SE_FACTOR + Z * spread
    assert float(norm.cdf((low - upper) / spread)) >= 0.99


def _joint_draws() -> np.ndarray:
    names = sorted(set(study.ESTIMANDS) | set(study.DUPLICATES))
    curves = []
    for name in names:
        plan, horizon, reference = _parse(name)
        curves.append(law.BINARY.eif(plan, horizon, reference=reference))
    matrix = np.array(curves)
    weights = law.BINARY.paths["p"].to_numpy()
    covariance = (matrix * weights) @ matrix.T
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    rng = np.random.default_rng(2026320199)
    return rng.multivariate_normal(np.zeros(len(names)), correlation, size=200_000, method="eigh")


def test_the_simultaneous_band_and_its_control_are_powered() -> None:
    draws = _joint_draws()
    critical = float(norm.ppf(1.0 - study.STUDY.margins.alpha / 2.0))
    pointwise = float(np.mean(np.all(np.abs(draws) <= critical, axis=1)))
    assert pointwise < 0.85
    replicates = properties.CALIBRATION_REPLICATES
    counts = np.arange(replicates + 1)

    def passing(rule: Any, coverage: float) -> float:
        inside = np.array(
            [
                rule(clopper_pearson(int(count), replicates, confidence_level=CONFIDENCE))
                for count in counts
            ]
        )
        return float(np.sum(binom.pmf(counts[inside], replicates, coverage)))

    low, high = study.STUDY.margins.calibration_coverage
    band = passing(lambda interval: interval.low >= low and interval.high <= high, 0.95)
    control = passing(lambda interval: interval.high < 1.0 - study.STUDY.margins.alpha, pointwise)
    assert band >= 0.99
    assert control >= 0.99


def test_one_replication_of_each_fit_set_emits_exactly_the_declared_cells() -> None:
    fit_sets = tuple((family, configuration, 1) for family, configuration, _ in properties.FIT_SETS)
    rows = pd.DataFrame(
        [
            row
            for (payload,) in properties._payloads(fit_sets)
            for row in properties._replication(payload)
        ]
    )
    controls = properties.calibration_controls(
        rows,
        study.STUDY,
        labels=study.GATED_LABELS,
        efficiency_bounds=properties.EFFICIENCY_SD,
        calibration_n=properties.CALIBRATION_N,
        shrunken_se_factor=properties.SHRUNKEN_SE_FACTOR,
        critical=properties.CRITICAL,
        positive_suffix="estimated_mechanisms",
    )
    rows = pd.concat([rows, controls], ignore_index=True)
    emitted = {
        (str(family), str(cell))
        for family, cell in rows[["property", "cell"]].itertuples(index=False)
    }
    declared = {
        (family, cell) for family, cells in study.STUDY.property_cells.items() for cell in cells
    }
    assert emitted == declared
    estimates = rows["estimate"].to_numpy(dtype=float)
    assert np.isfinite(estimates).all()
    # Every point estimate is a mean or a contrast of means of a variable in [0, 1].  The
    # noise control adds noise, and a joint-coverage row records an indicator.
    pointwise = ~rows["cell"].str.endswith("noise_control") & (
        rows["property"] != "simultaneous_coverage"
    )
    assert (np.abs(estimates[pointwise.to_numpy()]) <= 1.0).all()


def test_the_declared_truths_are_finite_and_cover_every_label() -> None:
    truths = properties.declared_truths()
    assert set(truths["continuous"]) == set(study.GATED_LABELS)
    assert set(truths["absorbing"]) == set(properties.ABSORBING_LABELS)
    assert set(truths["history_free"]) == set(properties.HISTORY_LABELS)
    for kind in ("both_wrong", "plug_in", "markov"):
        assert set(truths[kind]) == set(study.GATED_LABELS), kind
    assert set(truths["reversed"]) == set(properties.RULE_LABELS)
    for table in truths.values():
        assert all(np.isfinite(value) for value in table.values())


def test_the_rule_reads_only_the_earlier_outcome() -> None:
    history = pd.DataFrame({"Y1": [0.0, 1.0, 0.0], "L2": [1.0, 0.0, 0.0]})
    np.testing.assert_array_equal(study.treat_after("Y1")(history), [0, 1, 0])


def test_the_primary_phase_returns_the_driver_contract() -> None:
    """The regeneration driver refuses anything but a ``PhaseResult`` (Study queue item 11)."""
    result = study.draw_and_fit(replicates=1, n=study.PRIMARY_N, n_jobs=1)
    assert isinstance(result, replication.PhaseResult)
    assert result.failures is None
    assert set(result.rows["estimand"]) == set(study.ESTIMANDS)
    assert result.samples is not None and result.truths is not None


# ---------------------------------------------------------------------------- parity

#: The largest ratio of the root-mean-square difference between the two implementations to
#: the root-mean-square size of R's own targeting move, in any primary cell.  Declared before
#: the run.  The paired margins cannot see a missing targeting step, because the untargeted
#: sequential regression is nearly unbiased on this law.  The ratio can: a skipped step has
#: ratio 1 and a step 10% short has ratio 0.1.  The bound sits ten times below the second and
#: far above the disagreement of two solvers that stop at a relative score of 1e-8 to 1e-10.
PARITY_RMS_BOUND = 0.01
#: The largest single difference between the two implementations, as a fraction of R's mean
#: reported standard error in the cell.  Declared before the run.
PARITY_MAX_SE_FRACTION = 0.01
#: R's targeting step must exist in every cell: its root-mean-square size is at least this.
PARITY_MIN_MOVE = 1e-7


def _rms(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def _parity_findings(paired: pd.DataFrame) -> list[str]:
    """The cells of aligned paired rows that break the parity rule, as readable findings.

    ``paired`` has one row per replication and estimand with ``r_initial`` (the ``gcomp``
    call), ``r_estimate`` (the targeted R fit), ``cl_estimate`` and ``r_std_error``.  The
    denominator is R's own targeting move, because cleverly publishes no plug-in.
    """
    findings = []
    for estimand, cell in paired.groupby("estimand"):
        move = cell["r_estimate"] - cell["r_initial"]
        difference = cell["cl_estimate"] - cell["r_estimate"]
        if _rms(move) <= PARITY_MIN_MOVE:
            findings.append(f"{estimand}: R's targeting step is absent (rms {_rms(move):.2e})")
        elif _rms(difference) > PARITY_RMS_BOUND * _rms(move):
            findings.append(
                f"{estimand}: rms difference is {_rms(difference) / _rms(move):.3g} of R's "
                f"targeting move, above {PARITY_RMS_BOUND}"
            )
        worst = float(difference.abs().max())
        if worst > PARITY_MAX_SE_FRACTION * float(cell["r_std_error"].mean()):
            findings.append(
                f"{estimand}: largest difference {worst:.2e} exceeds {PARITY_MAX_SE_FRACTION} "
                "of R's mean standard error"
            )
    return findings


def _paired_rows(path: Path, replicates: int | None = None) -> pd.DataFrame:
    """The primary rows of both implementations in ``path``, aligned on replication and estimand.

    Fails, rather than skips, on rows that do not belong to the current declaration.  With
    ``replicates`` the file must hold exactly that many replications for every cell, as the
    declared run does.  Without it, the two implementations are paired on the rows they
    share, as a smoke run with ``--allow-failures`` needs.
    """
    rows = pd.read_csv(path)
    key = ["scenario", "replicate", "estimand"]
    subject = rows[rows["implementation"] == study.STUDY.implementation].set_index(key)
    reference = rows[rows["implementation"] == study.STUDY.reference].set_index(key)
    stale = []
    if set(rows["n"]) != {study.PRIMARY_N}:
        stale.append(f"n is {sorted(set(rows['n']))}, declared {study.PRIMARY_N}")
    if replicates is not None and len(subject) != len(study.ESTIMANDS) * replicates:
        stale.append(f"{len(subject)} cleverly rows, declared {len(study.ESTIMANDS) * replicates}")
    if subject["initial_estimate"].notna().any():
        stale.append("cleverly publishes an initial estimate, which the declaration removes")
    assert not stale, (
        "the rows in "
        + str(path)
        + " predate the fourth declaration ("
        + "; ".join(stale)
        + "); run the declared study to refresh tests/canonical/ltmle_repeated"
    )
    if replicates is not None:
        assert len(subject) == len(reference), "the two implementations have different row counts"
    shared = subject.index.intersection(reference.index).sort_values()
    subject = subject.loc[shared]
    reference = reference.loc[shared]
    return pd.DataFrame(
        {
            "estimand": subject.index.get_level_values("estimand"),
            "r_initial": reference["initial_estimate"].to_numpy(),
            "r_estimate": reference["estimate"].to_numpy(),
            "cl_estimate": subject["estimate"].to_numpy(),
            "r_std_error": reference["std_error"].to_numpy(),
        }
    )


def _committed_paired_rows() -> pd.DataFrame:
    """The committed primary rows of both implementations, aligned on replication and estimand."""
    return _paired_rows(study.STUDY.artifact("replicates.csv.gz"), study.PRIMARY_REPLICATES)


def main(argv: list[str] | None = None) -> int:
    """Check the parity rule on the ``replicates.csv.gz`` of any run, such as a smoke run.

    ``python -m tests.unit.test_ltmle_repeated_method_study PATH`` prints one line for each
    breach and returns 1 when there is one.  The rule is the one the fast tier applies to
    the committed rows.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("path", type=Path, help="the replicates.csv.gz of a run")
    paired = _paired_rows(parser.parse_args(argv).path)
    findings = _parity_findings(paired)
    cells = paired.groupby("estimand").size()
    print(
        f"{len(paired)} paired rows, {cells.min()} to {cells.max()} in each of {len(cells)} cells"
    )
    for finding in findings:
        print(finding)
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())


def test_cleverly_reproduces_r_targeting_move_in_every_cell() -> None:
    """The parity gate on the committed paired rows, in all 13 primary cells."""
    paired = _committed_paired_rows()
    assert set(paired["estimand"]) == set(study.ESTIMANDS)
    assert _parity_findings(paired) == []


def _synthetic_pairs(replicates: int = 200) -> pd.DataFrame:
    """Aligned rows with a nonzero targeting move and a rounding-level disagreement."""
    rng = np.random.default_rng(2026320411)
    records = []
    for estimand in study.ESTIMANDS:
        initial = rng.normal(0.3, 0.02, replicates)
        move = rng.normal(0.002, 0.003, replicates)
        records.append(
            pd.DataFrame(
                {
                    "estimand": estimand,
                    "r_initial": initial,
                    "r_estimate": initial + move,
                    "cl_estimate": initial + move + rng.normal(0.0, 1e-9, replicates),
                    "r_std_error": 0.02,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def test_a_skipped_or_short_targeting_step_fails_the_parity_gate() -> None:
    """Deliberate mutations: cleverly with no targeting, and with a step 10% short."""
    paired = _synthetic_pairs()
    assert _parity_findings(paired) == []
    move = paired["r_estimate"] - paired["r_initial"]
    skipped = paired.assign(cl_estimate=paired["r_initial"])
    short = paired.assign(cl_estimate=paired["r_initial"] + 0.9 * move)
    for name, mutated in (("skipped", skipped), ("ten percent short", short)):
        flagged = {line.split(":")[0] for line in _parity_findings(mutated)}
        assert flagged == set(study.ESTIMANDS), name
    # A single replicate off by 2% of the standard error trips the maximum clause alone.
    off = paired.copy()
    off.loc[0, "cl_estimate"] += 0.02 * 0.02
    assert [line.split(":")[0] for line in _parity_findings(off)] == [study.ESTIMANDS[0]]
    # R's step absent in a cell is a finding of its own.
    flat = paired.assign(r_estimate=paired["r_initial"], cl_estimate=paired["r_initial"])
    assert all("targeting step is absent" in line for line in _parity_findings(flat))


def _run_file(paired: pd.DataFrame, path: Path) -> Path:
    """Write aligned pairs in the layout of ``replicates.csv.gz``, one frame for both sides."""
    base = {"scenario": study.SCENARIO, "n": study.PRIMARY_N}
    ordinal = paired.groupby("estimand").cumcount()
    subject = pd.DataFrame(
        {
            **base,
            "replicate": ordinal,
            "estimand": paired["estimand"],
            "implementation": study.STUDY.implementation,
            "estimate": paired["cl_estimate"],
            "std_error": paired["r_std_error"],
            "initial_estimate": np.nan,
        }
    )
    reference = subject.assign(
        implementation=study.STUDY.reference,
        estimate=paired["r_estimate"],
        initial_estimate=paired["r_initial"],
    )
    pd.concat([subject, reference], ignore_index=True).to_csv(path, index=False)
    return path


def test_the_parity_check_reads_the_rows_of_any_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The smoke run's file goes through the same rule as the committed one."""
    paired = _synthetic_pairs()
    good = _run_file(paired, tmp_path / "good.csv.gz")
    assert main([str(good)]) == 0
    assert f"{len(study.ESTIMANDS) * 200} paired rows" in capsys.readouterr().out
    # A skipped targeting step in cleverly is a breach on the file, as on the frame.
    skipped = _run_file(paired.assign(cl_estimate=paired["r_initial"]), tmp_path / "bad.csv.gz")
    assert main([str(skipped)]) == 1
    assert "rms difference" in capsys.readouterr().out
    # A row dropped on one side pairs on the rest instead of failing the alignment.
    frame = pd.read_csv(good)
    dropped = frame.drop(frame.index[0])
    dropped.to_csv(tmp_path / "dropped.csv.gz", index=False)
    assert len(_paired_rows(tmp_path / "dropped.csv.gz")) == len(frame) // 2 - 1
    # The declared count is strict: the same file fails it.
    with pytest.raises(AssertionError, match="cleverly rows"):
        _paired_rows(good, study.PRIMARY_REPLICATES)
    # A file at another sample size is refused.
    wrong = frame.assign(n=study.PRIMARY_N + 1)
    wrong.to_csv(tmp_path / "wrong.csv.gz", index=False)
    with pytest.raises(AssertionError, match="predate"):
        _paired_rows(tmp_path / "wrong.csv.gz")
