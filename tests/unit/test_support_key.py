"""The support key, ``classify``, ``combine_support`` and the phase-1 record switch.

The key is read from an estimator and its data before any learner runs. Each test names the
axis it moves and a witness that differs from the baseline, so a key reader that ignores the
axis fails. The record switch changes no output: a fit with the switch on returns the numbers
of a fit with it off.
"""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import _support
from cleverly._refusal import Refusal, RefusalReason
from cleverly._support import (
    DEFAULT_BAND_OPTION,
    RECORD_ENVIRONMENT_VARIABLE,
    UNMEASURED_BAND_OPTION,
    Classification,
    Support,
    SupportKey,
    SupportRow,
    band_support,
    classify,
    classify_key,
    combine_support,
    longitudinal_key,
    point_key,
    recording_directory,
    support_for,
)
from cleverly._support_rows import POINT_ROWS
from cleverly.data.causal_data import CausalData
from cleverly.datasets import (
    make_binary_outcome,
    make_clustered,
    make_linear_ate,
    make_missing_outcome_binary,
)
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.exceptions import CapabilityError
from cleverly.interventions import Incremental, Shift, Static, Stochastic
from cleverly.interventions.learned import LearnedRule
from cleverly.longitudinal import LTMLE, LongitudinalData
from tests.conftest import linear_in_sample
from tests.studies.support_cases import CASES, Observed
from tests.unit._inference_status_support import LONGITUDINAL_N, longitudinal_learners
from tests.unit.test_longitudinal_markov_order import _panel as markov_panel
from tests.unit.test_sequential_design import COLUMNS, multivalue_panel

SUPPORTED_A = Support("supported", "A study covers it.", ("study-a",), row="R1", limits=("l1",))
SUPPORTED_B = Support("supported", "A study covers it.", ("study-b",), row="R2")
EXPERIMENTAL_X = Support("experimental", "No study covers X.", ("X1",))
EXPERIMENTAL_Y = Support("experimental", "No study covers Y.", ("X1", "Y2"), limits=("l2",))


def prepared(frame: Any, **roles: Any) -> CausalData:
    """The prepared data a fit on ``frame`` reads."""
    return CausalData.from_frame(frame, outcome="Y", treatment="A", **roles)


def key_of(estimator: TMLE, frame: Any, estimands: tuple[str, ...] = ("ate",), **roles: Any):
    data = prepared(frame, **roles)
    return point_key(estimator, data, estimands)


@pytest.fixture(scope="module")
def frame() -> Any:
    return make_linear_ate(n=200, seed=3)[0]


def clustered_frame(k: int) -> Any:
    return make_clustered(n=4 * k, cluster_size=4, seed=5)[0]


class TestCombineSupport:
    def test_equal_records_stay(self) -> None:
        assert combine_support([SUPPORTED_A, SUPPORTED_A, SUPPORTED_A]) is SUPPORTED_A

    def test_one_record_is_itself(self) -> None:
        assert combine_support([EXPERIMENTAL_X]) is EXPERIMENTAL_X

    def test_an_experimental_input_makes_the_result_experimental(self) -> None:
        """The witness: a supported record beside an experimental one must not stay supported."""
        combined = combine_support([SUPPORTED_A, EXPERIMENTAL_X])
        assert combined.label == "experimental"
        assert combined.reason == EXPERIMENTAL_X.reason
        assert combined.evidence == ("X1",)
        assert combined.row is None

    def test_reasons_and_evidence_join_in_first_seen_order(self) -> None:
        combined = combine_support([EXPERIMENTAL_X, SUPPORTED_A, EXPERIMENTAL_Y, EXPERIMENTAL_X])
        assert combined.reason == f"{EXPERIMENTAL_X.reason} {EXPERIMENTAL_Y.reason}"
        assert combined.evidence == ("X1", "Y2")
        assert combined.limits == ("l1", "l2")

    def test_two_supported_rows_are_experimental(self) -> None:
        combined = combine_support([SUPPORTED_A, SUPPORTED_B])
        assert combined.label == "experimental"
        assert combined.reason == "This estimate combines estimates of two support rows."

    def test_the_same_row_with_other_evidence_is_two_records(self) -> None:
        other = Support("supported", "A study covers it.", ("study-c",), row="R1", limits=("l1",))
        assert combine_support([SUPPORTED_A, other]).label == "experimental"

    def test_an_empty_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            combine_support([])


class TestPointKey:
    def test_a_default_tmle_key(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        assert key.kind == "point"
        assert key.engine == ("tmle",)
        assert key.family == frozenset({"P1"})
        assert key.outcome == ("gaussian", "unbounded")
        assert key.missing == ("complete",)
        assert key.treatment == ("binary",)
        assert key.weighting == "none"
        assert key.splitting == ("in_sample",)
        assert key.dependence == "iid"
        assert key.qualifiers == frozenset()
        assert dict(key.options) == {
            "fluctuation": "logistic",
            "targeting": "iterative",
            "target_weights": False,
            "stopping": "relative",
            "step_bound": "none",
            "iterate": "last",
            "outcome_by_arm": False,
            "patience": False,
            "treatment_adjustment": False,
        }
        assert learners["outcome"] is LinearRegression
        assert learners["treatment"] is LogisticRegression
        assert learners["missingness"] is None

    def test_the_estimand_names_map_to_their_families(self, frame: Any) -> None:
        estimator = TMLE(**linear_in_sample())
        names = {
            "P1": ("ey1", "ate", "ey0"),
            "P2": ("rr",),
            "P3": ("att", "atc"),
            "P4": ("ey_obs", "par", "paf"),
        }
        for family, estimands in names.items():
            key, _ = key_of(estimator, frame, estimands)
            assert key.family == frozenset({family}), estimands
        key, _ = key_of(estimator, frame, ("ey1", "rr", "att", "par"))
        assert key.family == frozenset({"P1", "P2", "P3", "P4"})

    def test_a_regime_is_p5_and_a_known_stochastic_regime_is_p6(self, frame: Any) -> None:
        deterministic = TMLE(
            **linear_in_sample(interventions=(Static(0.0, name="never"), Static(1.0, name="all")))
        )
        stochastic = TMLE(
            **linear_in_sample(
                interventions=(
                    Static(0.0, name="never"),
                    Stochastic(
                        lambda data: np.full(len(data), 0.5), name="tilt", density_kind="known"
                    ),
                )
            )
        )
        assert key_of(deterministic, frame, ("ey_regime",))[0].family == frozenset({"P5"})
        assert key_of(stochastic, frame, ("ey_regime",))[0].family == frozenset({"P6"})

    def test_the_other_axes_map_to_one_family_each(self, frame: Any) -> None:
        for kwargs, family in (
            ({"incremental": [Incremental(2.0)]}, "P9"),
            ({"policies": (Shift(0.5, cap=None),)}, "P7"),
            ({"learned_rule": LearnedRule()}, "P11"),
        ):
            key, _ = key_of(TMLE(**linear_in_sample(**kwargs)), frame, ("ey_ipsi",))
            assert key.family == frozenset({family}), kwargs

    def test_a_binary_outcome_ignores_the_bounds_declaration(self) -> None:
        data = make_binary_outcome(n=200, seed=1)[0]
        for bounds in (None, (0.0, 1.0)):
            key, _ = key_of(TMLE(**linear_in_sample(q_bounds=bounds)), data)
            assert key.outcome == ("binomial",)

    @pytest.mark.parametrize(
        ("overrides", "splitting"),
        [
            ({"cross_fit": False}, ("in_sample",)),
            ({"cross_fit": True}, ("stacked",)),
            ({"cross_fit": True, "cv_evaluation": True}, ("fold_evaluated",)),
            ({"cross_fit": True, "targeting_scheme": "fold"}, ("fold_targeted",)),
            ({"cross_fit": True, "repeats": 2}, ("stacked", "repeated")),
        ],
    )
    def test_splitting(self, frame: Any, overrides: dict[str, Any], splitting: tuple) -> None:
        key, _ = key_of(TMLE(**linear_in_sample(**overrides)), frame)
        assert key.splitting == splitting

    def test_options_move_the_key(self, frame: Any) -> None:
        base, _ = key_of(TMLE(**linear_in_sample()), frame)
        for overrides in (
            {"fluctuation": "linear"},
            {"targeting": "one_step"},
            {"target_weights": True},
        ):
            moved, _ = key_of(TMLE(**linear_in_sample(**overrides)), frame)
            assert moved != base, overrides
            assert len(base.differing_axes(moved)) == 1
            assert base.differing_axes(moved) == ("options",)

    @pytest.mark.parametrize("kind", ["mammen", "normal"])
    def test_the_band_multiplier_does_not_move_the_key(self, frame: Any, kind: str) -> None:
        """Pointwise inference never reads the multiplier, so the fit keeps its row."""
        base, learners = key_of(TMLE(**linear_in_sample()), frame)
        moved, _ = key_of(TMLE(**linear_in_sample(multiplier_kind=kind)), frame)
        assert moved == base
        assert classify_key(moved, learners).row == classify_key(base, learners).row

    def test_the_outcome_bounds_axis(self, frame: Any) -> None:
        bounded = TMLE(**linear_in_sample(q_bounds=(-100.0, 100.0)))
        key, _ = key_of(bounded, frame)
        assert key.outcome == ("gaussian", "bounded")

    def test_a_missing_outcome_is_on_the_missing_axis(self) -> None:
        data = make_missing_outcome_binary(n=200, seed=1)[0]
        key, _ = key_of(TMLE(**linear_in_sample()), data, delta="Delta")
        assert key.missing == ("outcome",)

    def test_estimated_weights_differ_from_fixed_and_from_none(self, frame: Any) -> None:
        weighted = frame.assign(w=1.0 + 0.37 * (frame["W1"] > 0))
        estimator = TMLE(**linear_in_sample())
        none, _ = key_of(estimator, frame)
        fixed, _ = key_of(estimator, weighted, weights="w")
        estimated, _ = key_of(estimator, weighted, weights="w", weights_estimated=True)
        assert (none.weighting, fixed.weighting, estimated.weighting) == (
            "none",
            "fixed",
            "estimated",
        )

    def test_strata_are_a_qualifier(self, frame: Any) -> None:
        stratified = frame.assign(S=(frame["W1"] > 0).astype(int))
        key, _ = key_of(TMLE(**linear_in_sample()), stratified, strata=["S"])
        assert key.qualifiers == frozenset({"strata"})

    def test_a_named_reference_does_not_change_the_key(self, frame: Any) -> None:
        plain, _ = key_of(TMLE(**linear_in_sample()), frame)
        named, _ = key_of(TMLE(**linear_in_sample(reference=0)), frame)
        assert plain == named

    def test_an_explicit_learner_class_is_read(self, frame: Any) -> None:
        class MyOutcome(LinearRegression):
            pass

        _, learners = key_of(TMLE(**linear_in_sample(outcome_learner=MyOutcome())), frame)
        assert learners["outcome"] is MyOutcome
        assert learners["outcome"] is not LinearRegression

    def test_drtmle_options_and_engine(self, frame: Any) -> None:
        key, _ = key_of(DRTMLE(**linear_in_sample()), frame)
        assert key.engine == ("drtmle",)
        options = dict(key.options)
        assert {"guard", "reduction", "reduced_crossfit", "update_order", "randomized"} <= set(
            options
        )
        assert options["evaluation"] is False
        plain, _ = key_of(TMLE(**linear_in_sample()), frame)
        assert "guard" not in dict(plain.options)

    def test_ctmle_engine_holds_strategy_and_design(self, frame: Any) -> None:
        key, _ = key_of(CTMLE(**linear_in_sample(strategy="oat", oat_design="per_arm")), frame)
        assert key.engine == ("collaborative_tmle", "oat", "per_arm")

    @pytest.mark.parametrize(
        ("clusters", "tier"),
        [(5, "below_floor"), (9, "below_floor"), (10, "t"), (39, "t"), (40, "clustered")],
    )
    def test_the_dependence_tiers_at_their_boundaries(self, clusters: int, tier: str) -> None:
        """The floors are 10 and 40: each boundary count lands on the named side."""
        estimator = TMLE(**linear_in_sample())
        key, _ = key_of(estimator, clustered_frame(clusters), id="cluster")
        assert key.dependence == tier

    def test_an_iid_fit_is_iid(self, frame: Any) -> None:
        key, _ = key_of(TMLE(**linear_in_sample()), frame)
        assert key.dependence == "iid"

    def test_the_longitudinal_floor_is_twenty(self) -> None:
        assert _support._dependence(19, 20) == "below_floor"
        assert _support._dependence(20, 20) == "t"
        assert _support._dependence(40, 20) == "clustered"
        assert _support._dependence(19, 10) == "t"


#: The point options of X76 and their values in a fit that declares none of them.
OPTION_DEFAULTS: dict[str, Any] = {
    "stopping": "relative",
    "step_bound": "none",
    "iterate": "last",
    "outcome_by_arm": False,
    "patience": False,
    "treatment_adjustment": False,
}
RULES = ("relative", "scaled_var", "sample_size")
#: Every valid step setting: the full Newton step, a bound with a line search, the whole bound.
STEPS = ((None, True, "none"), (0.05, True, "search"), (0.05, False, "fixed"))


def t01_binary() -> tuple[Any, CausalData, tuple[str, ...]]:
    """The estimator, data and estimands of the case that stands for ``T01.binary``."""
    built = next(case for case in CASES if case.row == "T01.binary").build()
    assert not isinstance(built, Observed)
    return built


def with_settings(estimator: Any, **settings: Any) -> Any:
    changed = copy.copy(estimator)
    for name, value in settings.items():
        setattr(changed, name, value)
    return changed


def long_key(estimator: Any, data: Any) -> SupportKey:
    return longitudinal_key(
        estimator,
        data,
        outcome_kind="end_of_study",
        cross_fitted=False,
        known_factors=(),
        has_msm=False,
        regimen_kinds={"static"},
    )[0]


class TestOptionKeys:
    """The targeting and fitting options of X76 enter the keys (Fix-8 PR 0).

    A key that drops an option puts a fit that declares it on the row of the default fit,
    which reads supported, so each test asserts the row id or the exact key value.
    """

    @pytest.mark.parametrize(
        ("estimator", "roles", "name", "value"),
        [
            *((TMLE, {"stopping": rule}, "stopping", rule) for rule in RULES),
            (TMLE, {"outcome_by_arm": True}, "outcome_by_arm", True),
            (TMLE, {"outcome_by_arm": False}, "outcome_by_arm", False),
            (CTMLE, {"patience": 2}, "patience", True),
            (CTMLE, {"patience": None}, "patience", False),
        ],
    )
    def test_each_point_option_lands_in_the_key_and_moves_only_itself(
        self, frame: Any, estimator: type, roles: dict[str, Any], name: str, value: Any
    ) -> None:
        base, _ = key_of(estimator(**linear_in_sample()), frame)
        moved, _ = key_of(estimator(**linear_in_sample(**roles)), frame)
        assert {key: dict(base.options)[key] for key in OPTION_DEFAULTS} == OPTION_DEFAULTS
        assert dict(moved.options) == dict(base.options) | {name: value}

    @pytest.mark.parametrize(
        ("subset", "declared"),
        [(None, False), (["W1"], True), ([], True), (["W1", "W2", "W3", "W4"], False)],
    )
    def test_a_treatment_adjustment_subset_is_keyed_when_it_restricts(
        self, frame: Any, subset: list[str] | None, declared: bool
    ) -> None:
        # The data read the whole adjustment set as no restriction, so that fit is the
        # default one; an empty subset is a restriction (a mechanism that reads nothing).
        base, _ = key_of(TMLE(**linear_in_sample()), frame)
        roles = {} if subset is None else {"treatment_covariates": subset}
        moved, _ = key_of(TMLE(**linear_in_sample()), frame, **roles)
        assert dict(moved.options) == dict(base.options) | {"treatment_adjustment": declared}

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize(("bound", "search", "step"), STEPS)
    @pytest.mark.parametrize("iterate", ["last", "best"])
    def test_step_bound_and_iterate_are_keyed_under_a_declared_rule_only(
        self, frame: Any, rule: str, bound: float | None, search: bool, step: str, iterate: str
    ) -> None:
        settings = {"stopping": rule, "step_bound": bound, "step_search": search}
        key, _ = key_of(TMLE(**linear_in_sample(**settings, iterate=iterate)), frame)
        expected = (
            {"stopping": rule, "step_bound": "none", "iterate": "last"}
            if rule == "relative"
            else {"stopping": rule, "step_bound": step, "iterate": iterate}
        )
        assert {name: dict(key.options)[name] for name in expected} == expected

    def test_a_relative_step_bound_keeps_t01_and_a_rule_step_bound_leaves_it(self) -> None:
        estimator, data, estimands = t01_binary()
        relative = classify(with_settings(estimator, step_bound=0.05), data, estimands)
        assert relative.row is not None and relative.row.id == "T01.binary"
        assert dict(relative.key.options)["step_bound"] == "none"
        ruled = classify(
            with_settings(estimator, stopping="sample_size", step_bound=0.05), data, estimands
        )
        assert dict(ruled.key.options)["step_bound"] == "search"
        assert ruled.row is None and ruled.nearest == ("T01.binary", ("options",))

    def test_control_a_step_bound_keyed_under_relative_leaves_t01(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mutation control: a key that records the step bound whatever the rule moves the
        # relative step-bound fit off T01, so the test above would fail.
        def everywhere(estimator: Any) -> dict[str, str]:
            bound = "none" if estimator.step_bound is None else "search"
            return {"stopping": estimator.stopping, "step_bound": bound, "iterate": "last"}

        monkeypatch.setattr(_support, "_stopping_options", everywhere)
        estimator, data, estimands = t01_binary()
        assert classify(with_settings(estimator, step_bound=0.05), data, estimands).row is None

    @pytest.mark.parametrize(
        "settings",
        [
            *({"stopping": rule} for rule in RULES[1:]),
            *({"stopping": rule, "iterate": "best"} for rule in RULES[1:]),
            *({"stopping": rule, "step_bound": 0.05} for rule in RULES[1:]),
            {"outcome_by_arm": True},
        ],
    )
    def test_a_declared_option_moves_the_fit_off_its_row(self, settings: dict[str, Any]) -> None:
        estimator, data, estimands = t01_binary()
        moved = classify(with_settings(estimator, **settings), data, estimands)
        assert moved.row is None
        assert moved.nearest == ("T01.binary", ("options",))
        # Control: the same key with the option at its default is the T01 fit, so a key
        # that drops the option reads supported.
        options = tuple(
            (key, OPTION_DEFAULTS.get(key, value) if key in settings else value)
            for key, value in moved.key.options
        )
        reset = classify_key(replace(moved.key, options=options), moved.learners)
        assert reset.row is not None and reset.row.id == "T01.binary"

    def test_a_declared_treatment_adjustment_moves_the_fit_off_its_row(self) -> None:
        estimator, data, estimands = t01_binary()
        assert classify(estimator, data, estimands).row.id == "T01.binary"  # type: ignore[union-attr]
        restricted = replace(data, treatment_covariate_names=data.covariate_names[:1])
        moved = classify(estimator, restricted, estimands)
        assert dict(moved.key.options)["treatment_adjustment"] is True
        assert moved.row is None and moved.nearest == ("T01.binary", ("options",))

    def test_control_without_the_stopping_option_a_rule_fit_reads_t01(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Mutation control: keys and rows without the stopping options put a rule fit on
        # T01, which would fail the row-id assertions above.
        monkeypatch.setattr(_support, "_stopping_options", lambda estimator: {})
        estimator, data, estimands = t01_binary()
        key, learners = point_key(with_settings(estimator, stopping="scaled_var"), data, estimands)
        stopping = {"stopping", "step_bound", "iterate"}
        rows = [
            replace(
                row,
                key=replace(
                    row.key,
                    options=tuple(item for item in row.key.options if item[0] not in stopping),
                ),
            )
            for row in POINT_ROWS
        ]
        ruled = classify_key(key, learners, rows=rows)
        assert ruled.row is not None and ruled.row.id == "T01.binary"

    @pytest.mark.parametrize(
        ("settings", "named"),
        [
            ({"stopping": "scaled_var"}, "options (stopping='scaled_var')"),
            (
                {"stopping": "sample_size", "step_bound": 0.05, "step_search": False},
                "options (step_bound='fixed' and stopping='sample_size')",
            ),
            ({"outcome_by_arm": True}, "options (outcome_by_arm=True)"),
        ],
    )
    def test_the_nearest_row_reason_names_each_differing_option(
        self, settings: dict[str, Any], named: str
    ) -> None:
        estimator, data, estimands = t01_binary()
        moved = classify(with_settings(estimator, **settings), data, estimands)
        reason = support_for(moved, "influence_curve").reason
        assert reason == (
            "No registered study covers this configuration. Nearest row: T01.binary "
            f"(canonical-tmle), differing on {named}."
        )

    def test_an_option_value_with_punctuation_or_no_value_reads_its_bare_name(self) -> None:
        # A float, a dotted name, a tuple and an option the fit lacks would put a full stop, a
        # comma or a parenthesis in the reason, which the plain reader splits on.
        row_key = next(row for row in POINT_ROWS if row.id == "T01.binary").key
        options = {name: value for name, value in row_key.options if name != "patience"}
        options |= {"target_weights": 0.5, "fluctuation": "a.b", "guard": ("Q", "g")}
        options |= {"tuned_cap": True}
        fit = replace(row_key, options=tuple(sorted(options.items())))
        assert _support._named_axis("options", fit, row_key) == (
            "options (fluctuation and guard and patience and target_weights and tuned_cap=True)"
        )
        assert _support._named_axis("splitting", fit, row_key) == "splitting"

    @pytest.mark.parametrize("rule", RULES)
    @pytest.mark.parametrize(("bound", "search", "step"), STEPS)
    @pytest.mark.parametrize("iterate", ["last", "best"])
    def test_the_longitudinal_key_reads_the_rule_with_the_point_convention(
        self, rule: str, bound: float | None, search: bool, step: str, iterate: str
    ) -> None:
        data = LongitudinalData.from_frame(multivalue_panel(n=120, seed=43), **COLUMNS)
        estimator = LTMLE(
            {"never": 0, "always": 1},
            reference="never",
            stopping=rule,
            step_bound=bound,
            step_search=search,
            iterate=iterate,
        )
        options = dict(long_key(estimator, data).options)
        expected = (
            {"stopping": rule, "step_bound": "none", "iterate": "last"}
            if rule == "relative"
            else {"stopping": rule, "step_bound": step, "iterate": iterate}
        )
        assert {name: options[name] for name in expected} == expected

    @pytest.mark.parametrize(
        ("order", "declared"), [(None, False), (0, True), (1, True), (2, True)]
    )
    def test_the_longitudinal_key_records_a_declared_markov_order_as_a_flag(
        self, order: int | None, declared: bool
    ) -> None:
        # Order 0 is a declared restriction too, so a key that reads the order's truth value
        # fails here. Four nodes, so orders 1 and 2 restrict the history.
        data = LongitudinalData.from_frame(
            markov_panel(),
            outcome="Y",
            treatment=[f"A{t}" for t in range(1, 5)],
            baseline=["W"],
            time_varying=[[f"L{t}"] for t in range(1, 5)],
        )
        estimator = LTMLE({"never": 0, "always": 1}, reference="never")
        options = dict(long_key(estimator, data.with_markov_order(order)).options)
        assert options["markov_order"] is declared


class TestKeyAxes:
    def test_differing_axes_name_each_moved_axis_once(self, frame: Any) -> None:
        base, _ = key_of(TMLE(**linear_in_sample()), frame)
        other, _ = key_of(TMLE(**linear_in_sample(cross_fit=True)), frame, ("ate", "rr"))
        assert other.differing_axes(base) == ("family", "splitting")
        assert base.differing_axes(other) == ("splitting",)
        assert base.differing_axes(base) == ()

    def test_to_json_round_trips_through_json(self, frame: Any) -> None:
        key, _ = key_of(TMLE(**linear_in_sample()), frame)
        again = json.loads(json.dumps(key.to_json()))
        assert again["engine"] == ["tmle"]
        assert again["qualifiers"] == []


def make_row(row_id: str, key: SupportKey, **kwargs: Any) -> SupportRow:
    return SupportRow(row_id, key, frozenset({"point", "wald"}), ("study",), **kwargs)


class TestClassify:
    def test_no_rows_is_unmatched_with_no_nearest(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        classification = classify_key(key, learners, rows=[])
        assert isinstance(classification, Classification)
        assert classification.row is None
        assert classification.nearest is None
        assert classification.label == "experimental"

    def test_a_matching_row_is_supported(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        row = make_row("R1", key)
        classification = classify_key(key, learners, rows=[row])
        assert classification.row is row
        assert classification.label == "supported"

    def test_a_condition_makes_a_matched_fit_experimental(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        condition = Refusal(RefusalReason.IMPLEMENTATION_GAP, "A condition.", "Instead.", "X60")
        classification = classify_key(key, learners, [condition], rows=[make_row("R1", key)])
        assert classification.row is not None
        assert classification.label == "experimental"

    def test_an_unmatched_key_names_the_nearest_row_and_its_axes(self, frame: Any) -> None:
        base, learners = key_of(TMLE(**linear_in_sample()), frame)
        far, _ = key_of(
            TMLE(**linear_in_sample(cross_fit=True, repeats=2, fluctuation="linear")), frame
        )
        near, _ = key_of(TMLE(**linear_in_sample(cross_fit=True)), frame)
        rows = [make_row("FAR", far), make_row("NEAR", near)]
        classification = classify_key(base, learners, rows=rows)
        assert classification.row is None
        assert classification.nearest == ("NEAR", ("splitting",))

    def test_a_learner_the_row_names_must_be_the_fits(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        row = make_row("R1", key, learners={"outcome": LogisticRegression})
        classification = classify_key(key, learners, rows=[row])
        assert classification.row is None
        assert classification.nearest == ("R1", ("learners",))
        exact = make_row("R2", key, learners={"outcome": LinearRegression})
        assert classify_key(key, learners, rows=[exact]).row is exact

    def test_two_matching_rows_are_a_bug(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        with pytest.raises(RuntimeError, match="all match one fit"):
            classify_key(key, learners, rows=[make_row("A", key), make_row("B", key)])


def band_row(key: SupportKey, **kwargs: Any) -> SupportRow:
    kinds = frozenset({"point", "wald", "band", "bootstrap_band"})
    return SupportRow("RB", key, kinds, ("study-b",), **kwargs)


class TestBandSupport:
    """A band option the row does not list turns the band experimental, never the fit."""

    @pytest.fixture
    def classification(self, frame: Any) -> Classification:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        return classify_key(key, learners, rows=[band_row(key)])

    @staticmethod
    def supported() -> Support:
        return Support("supported", "Row RB covers this configuration.", ("study-b",), row="RB")

    def test_the_default_option_is_supported(self, classification: Classification) -> None:
        """Every row's band cells measured the default band since band-finite-sample."""
        band = band_support(classification, "influence_curve", [self.supported()])
        assert band == band_support(
            classification, "influence_curve", [self.supported()], options=DEFAULT_BAND_OPTION
        )
        assert DEFAULT_BAND_OPTION == ("mammen", "logit", "normal")
        assert band.label == "supported"

    @pytest.mark.parametrize(
        "option",
        [
            ("rademacher", "identity", "normal"),
            ("normal", "identity", "normal"),
            ("mammen", "identity", "normal"),
        ],
    )
    def test_an_option_no_study_measures_names_why(
        self, classification: Classification, option: tuple[str, str, str]
    ) -> None:
        band = band_support(classification, "influence_curve", [self.supported()], options=option)
        assert band.label == "experimental"
        assert UNMEASURED_BAND_OPTION == "no registered study after band-finite-sample"
        assert UNMEASURED_BAND_OPTION in band.reason

    @pytest.mark.parametrize(
        "option", [("mammen", "identity", "normal"), ("rademacher", "logit", "normal")]
    )
    def test_an_unlisted_option_makes_the_band_experimental(
        self, classification: Classification, option: tuple[str, str, str]
    ) -> None:
        band = band_support(classification, "influence_curve", [self.supported()], options=option)
        assert band.label == "experimental"
        assert f"multiplier_kind={option[0]!r} with band_scale={option[1]!r}" in band.reason
        # The fit itself keeps its row and its label.
        assert classification.label == "supported"

    def test_a_listed_option_is_supported(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        row = band_row(
            key, band_options=(DEFAULT_BAND_OPTION, ("rademacher", "identity", "normal"))
        )
        classification = classify_key(key, learners, rows=[row])
        band = band_support(
            classification,
            "influence_curve",
            [self.supported()],
            options=("rademacher", "identity", "normal"),
        )
        assert band.label == "supported"

    def test_an_option_shortfall_reads_for_that_option_alone(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        option = ("rademacher", "identity", "normal")
        sentence = "Row RB is red for the Rademacher band: joint coverage 0.91 to 0.94."
        row = band_row(
            key,
            band_options=(DEFAULT_BAND_OPTION, option),
            band_option_shortfalls=((option, sentence),),
        )
        classification = classify_key(key, learners, rows=[row])
        red = band_support(classification, "influence_curve", [self.supported()], options=option)
        assert red.label == "experimental"
        assert sentence in red.reason
        default = band_support(classification, "influence_curve", [self.supported()])
        assert default.label == "supported"

    def test_the_bootstrap_band_reads_no_multiplier(self, classification: Classification) -> None:
        band = band_support(
            classification,
            "influence_curve",
            [self.supported()],
            kind="bootstrap_band",
            options=("normal", "identity", "normal"),
        )
        assert band.label == "supported"

    def test_every_band_row_lists_the_default_option_at_its_measured_draws(self) -> None:
        from cleverly._support_rows import (
            BAND_CELLS,
            BAND_OPTION_DRAWS,
            DEFAULT_BAND_DRAWS,
            LONGITUDINAL_ROWS,
            POINT_ROWS,
        )

        rows = {row.id: row for row in (*POINT_ROWS, *LONGITUDINAL_ROWS)}
        banded = {row_id: row for row_id, row in rows.items() if "band" in row.kinds}
        assert set(banded) == set(BAND_CELLS)
        assert {row.band_options for row in banded.values()} == {(DEFAULT_BAND_OPTION,)}
        assert dict(BAND_OPTION_DRAWS) == {"D09": 1_000}
        for row_id, row in banded.items():
            measured = BAND_OPTION_DRAWS.get(row_id, DEFAULT_BAND_DRAWS)
            assert row.band_option_draws == ((DEFAULT_BAND_OPTION, measured),)
        row = rows["G23.cross_fitted"]
        classification = Classification(row.key, {}, row)
        support = _support.support_for(classification, "influence_curve")
        mammen = band_support(classification, "influence_curve", [support], draws=10_000)
        normal = band_support(
            classification, "influence_curve", [support], options=("normal", "identity", "normal")
        )
        assert mammen.label == "supported"
        assert normal.label == "experimental"
        # The cell measured 10,000 draws; fewer, or an unstated count, is experimental.
        for draws in (2_000, None):
            fewer = band_support(classification, "influence_curve", [support], draws=draws)
            assert fewer.label == "experimental"
            assert "10,000 multiplier draws" in str(fewer.reason)
        # D09's cell pins 1,000 draws, so 1,000 is enough there.
        row = rows["D09"]
        classification = Classification(row.key, {}, row)
        support = _support.support_for(classification, "influence_curve")
        assert band_support(classification, "influence_curve", [support], draws=1_000).label == (
            "supported"
        )
        assert band_support(classification, "influence_curve", [support], draws=999).label == (
            "experimental"
        )

    def test_a_row_without_the_band_gives_one_reason(self, frame: Any) -> None:
        key, learners = key_of(TMLE(**linear_in_sample()), frame)
        classification = classify_key(key, learners, rows=[make_row("R1", key)])
        band = band_support(
            classification, "influence_curve", [], options=("normal", "identity", "normal")
        )
        assert band.reason == "Row R1 licenses no simultaneous band for this estimate."


class TestRecordSwitch:
    def test_the_directory_is_none_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(RECORD_ENVIRONMENT_VARIABLE, raising=False)
        assert recording_directory() is None
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, "")
        assert recording_directory() is None

    def test_with_the_switch_off_the_classifier_never_runs(
        self, monkeypatch: pytest.MonkeyPatch, frame: Any
    ) -> None:
        monkeypatch.delenv(RECORD_ENVIRONMENT_VARIABLE, raising=False)

        def fail(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("the classifier ran with the switch off")

        monkeypatch.setattr(_support, "classify", fail)
        monkeypatch.setattr(_support, "longitudinal_key", fail)
        TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A")

    def test_a_point_fit_records_and_changes_no_number(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any
    ) -> None:
        monkeypatch.delenv(RECORD_ENVIRONMENT_VARIABLE, raising=False)
        off = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path / "fast"))
        on = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        assert on.estimates["ate"].psi == off.estimates["ate"].psi
        assert on.estimates["ate"].std_error == off.estimates["ate"].std_error

        records = read_records(tmp_path / "fast")
        assert [r["record"] for r in records] == ["classification", "fit_outcome"]
        start, end = records
        assert start["source"] == "TMLE._fit_single"
        assert start["key"]["engine"] == ["tmle"]
        assert start["label"] == "supported"
        assert start["row"] == "T01.continuous"
        assert start["subdirectory"] == "fast"
        assert "test_support_key.py" in start["pytest"]
        assert end["statuses"] == ["influence_curve"]
        assert end["supplies_inference"] == [True]
        assert end["band_built_today"] is False
        assert end["n_estimates"] == len(on.estimates)

    def test_a_bootstrap_replicate_does_not_classify(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        estimator = TMLE(**linear_in_sample(n_bootstrap=3))
        result = estimator.fit(frame, outcome="Y", treatment="A").single()
        assert result.bootstrap is not None
        records = read_records(tmp_path)
        assert [r["record"] for r in records] == ["classification", "fit_outcome"]
        assert records[-1]["bootstrap_attached"] is True

    def test_retarget_classifies_its_estimand_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any
    ) -> None:
        result = TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        result.estimator.retarget(result.data, result.nuisance, estimands=("ey1",))
        records = read_records(tmp_path)
        assert [r["source"] for r in records] == ["TMLE.retarget", "TMLE.retarget"]
        assert records[0]["key"]["family"] == ["P1"]

    def test_a_converted_site_is_recorded_as_a_condition_of_the_fit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        data = make_missing_outcome_binary(n=300, seed=1)[0]
        estimator = TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            fluctuation="linear",
            simultaneous=False,
            n_folds=2,
            estimands=("ate",),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimator.fit(data, outcome="Y", treatment="A", delta="Delta").single()
        records = read_records(tmp_path)
        assert [record["record"] for record in records] == ["classification", "fit_outcome"]
        assert records[0]["conditions"] == ["X60"]
        assert records[0]["label"] == "experimental"
        assert result.estimates["ate"].support.label == "experimental"

    def test_a_flagged_site_is_marked_flagged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        data = make_missing_outcome_binary(n=300, seed=1)[0]
        estimator = TMLE(
            outcome_learner=LogisticRegression(),
            treatment_learner=LogisticRegression(),
            targeting_scheme="fold",
            simultaneous=False,
            n_folds=2,
            estimands=("ate",),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(CapabilityError):
                estimator.fit(data, outcome="Y", treatment="A", delta="Delta")
        (record,) = read_records(tmp_path)
        assert (record["site"], record["flagged"], record["roadmap"]) == (
            "arm.fold_targeting",
            True,
            "X53",
        )

    def test_a_classifier_failure_is_recorded_and_the_fit_runs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, frame: Any
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))

        def fail(*args: Any, **kwargs: Any) -> None:
            raise KeyError("boom")

        monkeypatch.setattr(_support, "classify", fail)
        TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A")
        (record,) = read_records(tmp_path)
        assert record["record"] == "classification_error"
        assert "boom" in record["error"]


class TestLongitudinalRecord:
    def fit(self, **overrides: Any) -> Any:
        frame = multivalue_panel(n=LONGITUDINAL_N, seed=43)
        return LTMLE(
            {"never": 0, "always": 1}, reference="never", **longitudinal_learners(**overrides)
        ).fit(frame, **COLUMNS)

    def test_a_fit_records_once_and_changes_no_number(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(RECORD_ENVIRONMENT_VARIABLE, raising=False)
        off = self.fit()
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        on = self.fit()
        for name, estimate in off.estimates.items():
            assert on.estimates[name].psi == estimate.psi
        records = read_records(tmp_path)
        assert [r["record"] for r in records] == ["classification", "fit_outcome"]
        key = records[0]["key"]
        assert key["kind"] == "longitudinal"
        assert key["engine"] == ["tmle"]
        assert key["family"] == ["end_of_study", "regimens", "static"]
        assert key["splitting"] == ["in_sample"]
        assert key["treatment"] == ["3", "2"]
        assert key["missing"] == ["censoring"]

    def test_a_bootstrap_replicate_does_not_classify(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        result = self.fit(n_bootstrap=2)
        assert result.bootstrap is not None
        records = read_records(tmp_path)
        assert [r["record"] for r in records] == ["classification", "fit_outcome"]


def read_records(directory: Path) -> list[dict[str, Any]]:
    """Every record under ``directory``, in file and line order."""
    lines: list[str] = []
    for path in sorted(directory.glob("*.jsonl")):
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    return [json.loads(line) for line in lines]


class TestRefusalSiteRecord:
    def test_a_fit_site_records_the_refusal_kind(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(RECORD_ENVIRONMENT_VARIABLE, str(tmp_path))
        _support.record_refusal_site("x", RefusalReason.IMPLEMENTATION_GAP, "X1")
        assert [r["record"] for r in read_records(tmp_path)] == ["refusal_site"]
