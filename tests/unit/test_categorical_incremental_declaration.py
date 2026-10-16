r"""What ``Incremental`` accepts, and what it refuses before any learner fits.

* C9: the declaration.  A scalar names two arms.  A mapping names levels by label and every
  unnamed level keeps the factor ``1``.  A scalar on other than two levels (R1), a key that is
  not a level (R2), a non-positive or non-finite factor, and a continuous treatment are
  refused before the first learner call, with the text a reader needs to repair the call.
* The names, the hash and the equality of a declaration.
* The ``intermediate=`` redirection reads right for a treatment with more than two levels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.exceptions import CapabilityError, DataError, RefusalReason
from cleverly.interventions import Incremental, IPSISet
from tests import discrete_law_multi as base
from tests.unit import _categorical_incremental_support as sup
from tests.unit._natural_course_support import Counting, CountingLogistic
from tests.unit.test_refusals_before_the_nuisance_fit import make_cde

LEVELS = base.SORTED_LABELS


def _fit(declared: list[Incremental], **roles: Any) -> Any:
    Counting.calls = 0
    return TMLE(
        incremental=declared,
        outcome_learner=CountingLogistic(max_iter=1000),
        treatment_learner=CountingLogistic(max_iter=1000),
        cross_fit=False,
        simultaneous=False,
    ).fit(sup.frame(), outcome="Y", treatment="A", covariates=["W"], **roles)


class TestTheDeclaration:
    def test_a_scalar_is_the_odds_multiplier_of_the_second_level(self) -> None:
        np.testing.assert_array_equal(Incremental(2.5).factors((0.0, 1.0)), [1.0, 2.5])
        assert Incremental(2.5).delta == 2.5

    def test_a_mapping_is_stored_as_label_pairs_in_the_declared_order(self) -> None:
        tilt = Incremental({"mid": 2.0, "low": 0.5})
        assert tilt.delta == (("mid", 2.0), ("low", 0.5))

    def test_unnamed_levels_keep_the_factor_one_and_labels_not_positions_choose_the_slot(
        self,
    ) -> None:
        tilt = Incremental({"mid": 2.0, "low": 0.5})
        np.testing.assert_array_equal(tilt.factors(LEVELS), [1.0, 0.5, 2.0])
        np.testing.assert_array_equal(tilt.factors(("mid", "high", "low")), [2.0, 1.0, 0.5])

    def test_the_factors_are_floats_whatever_the_caller_wrote(self) -> None:
        assert Incremental({"low": 2}).delta == (("low", 2.0),)
        assert all(isinstance(v, float) for _, v in Incremental({"low": 2}).delta)

    @pytest.mark.parametrize(
        ("tilt", "name"),
        [
            (Incremental(2.0), "odds x2"),
            (Incremental(0.5), "odds x0.5"),
            (Incremental(1.0), "natural course"),
            (Incremental({"high": 2.0, "low": 0.5}), "odds high x2, low x0.5"),
            (Incremental({"high": 1.0, "low": 1.0}), "natural course"),
            (Incremental({"high": 2.0}, name="up"), "up"),
        ],
    )
    def test_the_default_name(self, tilt: Incremental, name: str) -> None:
        assert tilt.name == name

    def test_a_declaration_is_hashable_and_equal_by_value(self) -> None:
        one, two = Incremental({"low": 2.0}), Incremental((("low", 2.0),))
        assert one == two
        assert hash(one) == hash(two)
        assert len({one, two, Incremental({"low": 3.0})}) == 2

    def test_a_declaration_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            Incremental({"low": 2.0}).delta = 3.0  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_factor_must_be_positive_and_finite(self, bad: float) -> None:
        with pytest.raises(DataError, match=r"strictly positive|finite"):
            Incremental(bad)
        with pytest.raises(DataError, match=r"strictly positive|finite"):
            Incremental({"low": bad})

    def test_an_empty_mapping_is_refused(self) -> None:
        with pytest.raises(DataError, match="at least one"):
            Incremental({})

    def test_a_level_named_twice_is_refused(self) -> None:
        with pytest.raises(DataError, match="names a level twice"):
            Incremental((("low", 2.0), ("low", 3.0)))


class TestR1AScalarNamesTwoArms:
    def test_the_text_names_the_levels_and_shows_a_repair(self) -> None:
        with pytest.raises(DataError) as raised:
            Incremental(2.0).factors(LEVELS, subject="'A'")
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert raised.value.refusal.condition == (
            "a scalar odds multiplier names two arms; 'A' has 3 levels "
            "(['high', 'low', 'mid']). Declare one factor per level, for example "
            "Incremental({'high': 2.0}), which multiplies the odds of 'high' against every "
            "unnamed level."
        )

    def test_a_fit_is_refused_before_any_learner(self) -> None:
        with pytest.raises(DataError, match="a scalar odds multiplier names two arms") as raised:
            _fit([Incremental(2.0)])
        assert Counting.calls == 0
        assert "'A' has 3 levels" in str(raised.value)

    def test_one_bad_declaration_among_good_ones_still_refuses_first(self) -> None:
        with pytest.raises(DataError, match="a scalar odds multiplier"):
            _fit([Incremental({"high": 2.0}), Incremental(2.0)])
        assert Counting.calls == 0


class TestR2AKeyThatIsNotALevel:
    def test_the_text_names_the_key_and_the_levels(self) -> None:
        with pytest.raises(DataError) as raised:
            Incremental({"hi": 2.0}).factors(LEVELS, subject="'A'")
        assert raised.value.refusal.reason is RefusalReason.INVALID_DECLARATION
        assert raised.value.refusal.condition == (
            "Incremental names 'hi', which is not a level of 'A'; its levels are "
            "['high', 'low', 'mid']."
        )

    def test_a_fit_is_refused_before_any_learner(self) -> None:
        with pytest.raises(DataError, match="'hi', which is not a level"):
            _fit([Incremental({"hi": 2.0})])
        assert Counting.calls == 0

    def test_a_position_is_not_a_label(self) -> None:
        """Keys are the labels the data reports, so ``0`` does not mean the first level."""
        with pytest.raises(DataError, match="not a level"):
            _fit([Incremental({0: 2.0})])
        assert Counting.calls == 0

    def test_a_learner_that_raises_is_never_reached(self) -> None:
        class Explodes(CountingLogistic):
            def fit(self, *args: Any, **kwargs: Any) -> Any:
                raise AssertionError("a learner was fitted before the declaration was checked")

        with pytest.raises(DataError, match="not a level"):
            TMLE(
                incremental=[Incremental({"nope": 2.0})],
                outcome_learner=Explodes(),
                treatment_learner=Explodes(),
                cross_fit=False,
                simultaneous=False,
            ).fit(sup.frame(), outcome="Y", treatment="A", covariates=["W"])


class TestRefusalsThatDoNotDependOnTheLevels:
    @staticmethod
    def _continuous() -> CausalData:
        rng = np.random.default_rng(0)
        n = 300
        frame = pd.DataFrame(
            {
                "Y": rng.binomial(1, 0.5, n).astype(float),
                "A": rng.normal(size=n),
                "W": rng.normal(size=n),
            }
        )
        return CausalData.from_frame(
            frame, outcome="Y", treatment="A", covariates=["W"], treatment_kind="continuous"
        )

    def test_a_continuous_treatment_is_refused_by_a_fit_before_any_learner(self) -> None:
        Counting.calls = 0
        with pytest.raises(DataError, match="was declared continuous") as raised:
            TMLE(
                incremental=[Incremental({"high": 2.0})],
                outcome_learner=CountingLogistic(),
                treatment_learner=CountingLogistic(),
                cross_fit=False,
                simultaneous=False,
            ).fit(self._continuous())
        assert raised.value.refusal.reason is RefusalReason.DIFFERENT_ESTIMAND
        assert Counting.calls == 0

    def test_the_evaluator_refuses_a_continuous_treatment_on_its_own(self) -> None:
        data = self._continuous()
        g = np.full((data.n, 2), 0.5)
        with pytest.raises(DataError, match="is continuous and has no levels") as raised:
            IPSISet.evaluate((Incremental(2.0),), data, g)
        assert raised.value.refusal.reason is RefusalReason.DIFFERENT_ESTIMAND

    def test_the_intermediate_redirection_names_the_mapping_form(self) -> None:
        frame, _ = make_cde(n=300, seed=3)
        frame = frame.assign(A=np.where(np.arange(len(frame)) % 3 == 0, 2.0, frame["A"]))
        Counting.calls = 0
        with pytest.raises(CapabilityError) as raised:
            TMLE(
                incremental=[Incremental({2.0: 2.0})],
                outcome_learner=CountingLogistic(),
                treatment_learner=CountingLogistic(),
                cross_fit=False,
                simultaneous=False,
            ).fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                intermediate="Z",
            )
        assert Counting.calls == 0
        assert "incremental= and intermediate= are not combined" in str(raised.value)
        assert "Incremental({level: factor})" in (raised.value.refusal.alternative or "")
