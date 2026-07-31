"""The longitudinal process, and the quadrature every accuracy claim about it rests on.

``make_longitudinal`` is not in :data:`cleverly.datasets.synthetic.GENERATORS` -- its
truth is keyed by regimen rather than by ``ate``/``ey1``/``ey0``, so the parametrised
gates in ``test_datasets.py`` cannot read it -- which left it with none of the structural
checks the other generators get, and left :func:`longitudinal_truth` with none at all.

That second gap is the one that matters. Every accuracy assertion in
``tests/e2e/test_ltmle.py`` compares an estimate against a number this Gauss--Hermite
rule produced, and nothing compared *that* number against anything. A wrong quadrature
would make the truth-recovery test and the confounding test wrong in the same direction
and leave both passing.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest
from scipy.special import expit

from cleverly.datasets import (
    RULE_LABEL,
    longitudinal_rule_truth,
    longitudinal_truth,
    make_longitudinal,
    make_longitudinal_survival,
    rule_arm_at_node_two,
    survival_truth,
)
from cleverly.datasets.longitudinal import _L2, _hazard_one, _hazard_two
from cleverly.longitudinal import LongitudinalData

#: The four static plans over two nodes.  Written out rather than read off the generator,
#: so a change there shows up here.
PLANS = ((1.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


class TestTheQuadratureIsRight:
    @pytest.mark.parametrize("plan", PLANS)
    def test_it_agrees_with_plain_monte_carlo(self, plan: tuple[float, float]) -> None:
        """The independent check: sample the intervened process and average.

        Under the intervention the mechanism is gone, so the counterfactual mean is an
        ordinary expectation that can be simulated directly -- no estimator, no
        identification, nothing the library supplies.  Half a million draws puts the
        Monte Carlo error near 3e-4, so 5e-3 is loose enough not to be flaky and far
        tighter than any error a wrong integral would make.
        """
        a1, a2 = plan
        rng = np.random.default_rng(0)
        n = 500_000
        w1 = rng.standard_normal(n)
        w2 = rng.standard_normal(n)
        # L2 under the intervention: A1 is set to a1 rather than drawn.
        l2 = 0.6 * w1 + 0.9 * a1 + rng.standard_normal(n)
        index = -0.4 + 0.5 * a1 + 0.8 * a2 + 0.4 * l2 + 0.3 * w1 - 0.2 * w2 + 0.5 * np.tanh(l2)
        assert longitudinal_truth(a1, a2) == pytest.approx(float(np.mean(expit(index))), abs=5e-3)

    @pytest.mark.parametrize("plan", PLANS)
    def test_it_has_converged_in_the_number_of_nodes(self, plan: tuple[float, float]) -> None:
        """A product rule that has converged does not move when it is refined.

        Cheaper than the Monte Carlo check above and far more sensitive: if 48 nodes were
        not enough for the ``tanh`` kink, 64 would give a different answer.  The docstring
        claims "well under 1e-10", so that is what is asserted.
        """
        coarse = longitudinal_truth(*plan, nodes=48)
        fine = longitudinal_truth(*plan, nodes=64)
        assert coarse == pytest.approx(fine, abs=1e-10)

    def test_the_truth_is_deterministic(self) -> None:
        assert longitudinal_truth(1.0, 1.0) == longitudinal_truth(1.0, 1.0)

    def test_the_rule_agrees_with_plain_monte_carlo(self) -> None:
        """The same independent check for the dynamic regimen ``make_longitudinal`` ships.

        ``A2`` is a function of the *intervened* ``L2``, which is where a quadrature that
        built ``L2`` from the observed treatment instead would come apart.  Reimplemented
        longhand, sharing nothing with the generator but the threshold -- and in
        particular not sharing the split-panel rule, which is the thing under test.
        """
        rng = np.random.default_rng(0)
        n = 500_000
        a1 = 1.0
        w1 = rng.standard_normal(n)
        w2 = rng.standard_normal(n)
        l2 = 0.6 * w1 + 0.9 * a1 + rng.standard_normal(n)
        a2 = (l2 > 0.0).astype(float)
        index = -0.4 + 0.5 * a1 + 0.8 * a2 + 0.4 * l2 + 0.3 * w1 - 0.2 * w2 + 0.5 * np.tanh(l2)
        _, truth = make_longitudinal(n=50, seed=0)
        assert truth[f"ey_regimen[{RULE_LABEL}]"] == pytest.approx(
            float(np.mean(expit(index))), abs=5e-3
        )

    def test_the_rule_has_converged_in_both_rules_it_is_built_from(self) -> None:
        """The check that caught this being wrong, kept because it caught it.

        A rule puts a step function into the integrand, and a Gauss--Hermite rule
        converges *algebraically* rather than spectrally on one.  Substituting an
        indicator into the plain three-dimensional rule -- the obvious implementation --
        moved the answer by ``1.7e-3`` between 48 and 64 nodes: worse than the Monte Carlo
        above, and useless as a truth for a coverage study whose standard errors are near
        ``0.02``.  Splitting the ``L2`` axis at the jump fixed it, and this refines *both*
        the outer Gauss--Hermite count and the panel count, since only refining the one
        the bug was not in would have looked converged either way.
        """
        base = longitudinal_rule_truth(1.0, rule_arm_at_node_two, nodes=48, panel=160)
        for nodes, panel in ((64, 160), (48, 240), (64, 320)):
            refined = longitudinal_rule_truth(1.0, rule_arm_at_node_two, nodes=nodes, panel=panel)
            assert base == pytest.approx(refined, abs=1e-10), (nodes, panel)

    @pytest.mark.parametrize(
        ("rule", "why"),
        [
            (lambda l2: (np.asarray(l2) > 1.0).astype(float), "jump away from split"),
            (lambda l2: (np.abs(np.asarray(l2)) < 2.0).astype(float), "two jumps"),
            (lambda l2: np.full(np.asarray(l2).shape, 0.5), "not an arm"),
        ],
    )
    def test_a_rule_off_the_contract_is_refused_rather_than_integrated(
        self, rule: Any, why: str
    ) -> None:
        """The failure mode this routine has that a plain quadrature does not.

        The arm is read once per panel, so a rule whose jump is not at ``split`` is
        integrated as though it were: the answer does not blow up or fail to converge, it
        comes back a plausible number for a *different* regimen. Since every accuracy
        claim in the longitudinal section is checked against this function, that number
        would move the reference rather than the estimate, and both sides would agree.
        """
        with pytest.raises(ValueError, match=r"split|binary treatment"):
            longitudinal_rule_truth(1.0, rule)

    def test_the_check_does_not_move_the_answer(self) -> None:
        """The contract case is what it was before the check existed."""
        assert longitudinal_rule_truth(1.0, rule_arm_at_node_two) == pytest.approx(
            0.7400375306197754, abs=1e-13
        )

    def test_a_rule_jumping_elsewhere_is_usable_once_split_says_so(self) -> None:
        """The refusal names a fix, so the fix has to work.

        Otherwise ``split=`` reads as a knob for the caller to guess at rather than the
        statement about the integrand that it is.
        """
        rule = lambda l2: (np.asarray(l2) > 1.0).astype(float)  # noqa: E731
        value = longitudinal_rule_truth(1.0, rule, split=1.0)
        # Between the two constants it interpolates, and not equal to either of them.
        low, high = longitudinal_truth(1.0, 0.0), longitudinal_truth(1.0, 1.0)
        assert low < value < high

    def test_the_rule_is_a_parameter_of_its_own(self) -> None:
        """Distinct from every static plan, and from the filler the recursion uses.

        Both coincidences are ways a broken fit passes: ``0.5`` is ``_FILLER``, so a
        prediction leaking from a censored row would land there, and a rule whose mean
        equalled a static regimen's would leave the dynamic path unfalsifiable against the
        constant plan beside it.  The **first node's arm** was chosen to avoid both -- the
        first draft, ``d_1 = 0`` with the same ``d_2``, came to *exactly* ``0.5``, and
        ``longitudinal_rule_truth(0.0, rule_arm_at_node_two)`` still does -- so a change
        that reintroduces either should fail here rather than quietly weaken the tier.
        """
        _, truth = make_longitudinal(n=50, seed=0)
        rule = truth[f"ey_regimen[{RULE_LABEL}]"]
        assert abs(rule - 0.5) > 0.05
        for label in ("always", "never", "early", "late"):
            assert abs(rule - truth[f"ey_regimen[{label}]"]) > 0.01, label

    def test_the_reported_contrast_is_the_difference_of_the_reported_means(self) -> None:
        _, truth = make_longitudinal(n=50, seed=0)
        assert truth["ate_regimen[always vs never]"] == pytest.approx(
            truth["ey_regimen[always]"] - truth["ey_regimen[never]"], abs=0
        )

    def test_treating_at_both_nodes_beats_treating_at_neither(self) -> None:
        """The sign of the effect, which the outcome regression's coefficients fix."""
        _, truth = make_longitudinal(n=50, seed=0)
        assert truth["ey_regimen[never]"] < truth["ey_regimen[early]"]
        assert truth["ey_regimen[early]"] < truth["ey_regimen[always]"]


class TestTheProcess:
    """The structural gates ``test_datasets.py`` applies to every other generator."""

    def test_the_frame_is_usable_and_time_ordered(self) -> None:
        frame, truth = make_longitudinal(n=200, seed=1)
        assert list(frame.columns) == ["W1", "W2", "A1", "C1", "L2", "A2", "C2", "Y"]
        assert len(frame) == 200
        assert truth

    def test_the_same_seed_gives_the_same_data(self) -> None:
        left, _ = make_longitudinal(n=100, seed=3)
        right, _ = make_longitudinal(n=100, seed=3)
        np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())

    def test_different_seeds_give_different_data(self) -> None:
        left, _ = make_longitudinal(n=100, seed=3)
        right, _ = make_longitudinal(n=100, seed=4)
        assert not np.array_equal(np.nan_to_num(left.to_numpy()), np.nan_to_num(right.to_numpy()))

    def test_a_censored_units_later_nodes_are_missing(self) -> None:
        """The convention ``LongitudinalData`` enforces, produced rather than assumed."""
        frame, _ = make_longitudinal(n=2000, seed=2)
        left = frame["C1"] == 0
        assert left.any()
        for column in ("L2", "A2", "C2", "Y"):
            assert frame.loc[left, column].isna().all()
        stayed = (frame["C1"] == 1) & (frame["C2"] == 1)
        assert frame.loc[stayed, "Y"].notna().all()

    def test_declaring_no_censoring_drops_the_columns(self) -> None:
        frame, _ = make_longitudinal(n=200, seed=2, censoring=False)
        assert "C1" not in frame.columns
        assert frame["Y"].notna().all()

    @pytest.mark.parametrize("backend,expected", [("pandas", "DataFrame"), ("polars", "DataFrame")])
    def test_the_requested_backend_is_produced(self, backend: str, expected: str) -> None:
        pytest.importorskip(backend)
        frame, _ = make_longitudinal(n=100, seed=0, backend=backend)
        assert type(frame).__name__ == expected
        assert backend in type(frame).__module__

    def test_the_time_varying_confounder_is_moved_by_the_first_treatment(self) -> None:
        """Without this the process would not need a longitudinal estimator at all."""
        frame, _ = make_longitudinal(n=5000, seed=6)
        under: Any = frame[frame["C1"] == 1]
        treated = under.loc[under["A1"] == 1, "L2"].mean()
        untreated = under.loc[under["A1"] == 0, "L2"].mean()
        assert treated - untreated > 0.5


class TestTheSurvivalProcess:
    """``make_longitudinal_survival``: the quadrature, and the missingness it produces."""

    PLANS: ClassVar = [(1.0, 1.0), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]

    @pytest.mark.parametrize("a1,a2", PLANS)
    @pytest.mark.parametrize("horizon", [1, 2])
    def test_the_quadrature_agrees_with_plain_monte_carlo(
        self, a1: float, a2: float, horizon: int
    ) -> None:
        """Two independent routes to the same number, as the end-of-study truth has.

        The Monte Carlo draws the *intervened* process directly -- no mechanism, no
        censoring, both hazards evaluated at the regimen's arms -- so it shares nothing
        with the Gauss--Hermite rule but the two hazard functions.
        """
        rng = np.random.default_rng(11)
        m = 400_000
        w1, w2, noise = (rng.standard_normal(m) for _ in range(3))
        hazard1 = _hazard_one(w1, w2, a1)
        y1 = rng.binomial(1, hazard1)
        l2 = _L2["w1"] * w1 + _L2["a1"] * a1 + noise
        y2 = rng.binomial(1, _hazard_two(w1, w2, l2, a1, a2))
        observed = y1.mean() if horizon == 1 else np.where(y1 == 1, 1, y2).mean()
        assert survival_truth(a1, a2, horizon) == pytest.approx(float(observed), abs=5e-3)

    def test_the_quadrature_has_converged(self) -> None:
        """Refining the rule must not move the answer, or the tolerance above is fiction."""
        for a1, a2 in self.PLANS:
            for horizon in (1, 2):
                coarse = survival_truth(a1, a2, horizon, 48)
                fine = survival_truth(a1, a2, horizon, 64)
                assert coarse == pytest.approx(fine, abs=1e-12)

    def test_the_risk_is_monotone_in_the_horizon(self) -> None:
        _, truth = make_longitudinal_survival(n=50, seed=0)
        for label in ("always", "never", "early", "late"):
            assert truth[f"risk_regimen[{label} @ t=1]"] <= truth[f"risk_regimen[{label} @ t=2]"]

    def test_no_truth_sits_on_the_filler(self) -> None:
        """``sequential._FILLER`` is a half; a parameter sitting there proves nothing.

        The end-of-study rule's docstring records falling into this trap once. With two
        horizons per regimen there are twice as many ways to.
        """
        _, truth = make_longitudinal_survival(n=50, seed=0)
        for name, value in truth.items():
            assert abs(value - 0.5) > 1e-2, name

    def test_treatment_lowers_the_risk_at_both_horizons(self) -> None:
        _, truth = make_longitudinal_survival(n=50, seed=0)
        for horizon in (1, 2):
            assert truth[f"ate_regimen[always vs never @ t={horizon}]"] < 0

    def test_the_frame_is_usable_and_time_ordered(self) -> None:
        frame, truth = make_longitudinal_survival(n=200, seed=1)
        assert list(frame.columns) == ["W1", "W2", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2"]
        assert len(frame) == 200
        assert truth

    def test_the_same_seed_gives_the_same_data(self) -> None:
        left, _ = make_longitudinal_survival(n=100, seed=3)
        right, _ = make_longitudinal_survival(n=100, seed=3)
        np.testing.assert_array_equal(left.to_numpy(), right.to_numpy())

    def test_the_event_is_absorbing_in_the_frame(self) -> None:
        """A unit that has the event has no later nodes, and carries its ``1`` forward.

        Two causes of missingness rather than one, which is what a survival frame adds:
        ``LongitudinalData`` refuses a node recorded after *either* exit, so producing
        this correctly is a precondition of the generator being usable at all.
        """
        frame, _ = make_longitudinal_survival(n=3000, seed=2)
        failed = frame["Y1"] == 1
        assert failed.any()
        for column in ("L2", "A2", "C2"):
            assert frame.loc[failed, column].isna().all()
        assert (frame.loc[failed, "Y2"] == 1).all()
        censored = frame["C1"] == 0
        assert censored.any()
        for column in ("Y1", "L2", "A2", "C2", "Y2"):
            assert frame.loc[censored, column].isna().all()

    def test_the_container_accepts_what_the_generator_produces(self) -> None:
        """The generator's convention and the container's are the same one."""
        frame, _ = make_longitudinal_survival(n=500, seed=5)
        data = LongitudinalData.from_frame(
            frame,
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W1", "W2"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
        assert data.is_survival
        assert data.n_times == 2

    def test_declaring_no_censoring_drops_the_columns(self) -> None:
        frame, _ = make_longitudinal_survival(n=200, seed=2, censoring=False)
        assert "C1" not in frame.columns
        assert frame["Y1"].notna().all()

    @pytest.mark.parametrize("backend,expected", [("pandas", "DataFrame"), ("polars", "DataFrame")])
    def test_the_requested_backend_is_produced(self, backend: str, expected: str) -> None:
        pytest.importorskip(backend)
        frame, _ = make_longitudinal_survival(n=100, seed=0, backend=backend)
        assert type(frame).__name__ == expected
        assert backend in type(frame).__module__
