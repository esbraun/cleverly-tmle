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

from typing import Any

import numpy as np
import pytest
from scipy.special import expit

from cleverly.datasets import longitudinal_truth, make_longitudinal

#: The four static plans over two nodes, which is every regimen this process has a truth
#: for.  Written out rather than read off the generator, so a change there shows up here.
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
