"""F5's applied stress cell is the stress setting it declares.

``benchmarks/drtmle_stress.py`` is what runs in place of F8 clause 3's paper reproduction, and
it discharges gate 2 clause 4 -- *the advantage persists in at least one applied stress
setting*.  Its whole content is a claim about the law: **both** primary nuisances are beyond a
GLM.  A cell where that quietly stopped being true would still run, still report coverage, and
would no longer be a stress setting, and nothing in a coverage table would say so.

So the misspecification is **asserted here rather than described in the module docstring**, and
it is asserted on the statistic the theorem's rate conditions are stated in -- the excess risk
against the law's own function -- rather than on a held-out risk against the observed target.
The difference is not pedantic: measured on a Brier score the mechanism's ratio reads ``1.06x``,
diluted by the irreducible Bernoulli variance every candidate shares, and measured as excess
risk the same fit reads ``3.32x``.  A floor placed on the first would have been a floor on
mostly noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks import drtmle_remainder, drtmle_stress


class TestTheLawIsAdmissibleToTheCompanion:
    """The remainder machinery must accept this law, or the cell has no remainder column."""

    def test_the_companion_does_not_refuse_it(self) -> None:
        # `_refuse_unsupported` names four laws a deterministic companion cannot be built for.
        # This one is none of them, and if it ever became one the failure would otherwise
        # surface as a mid-cohort exception hours into a confirmation run.
        drtmle_remainder._refuse_unsupported(drtmle_stress.base_law())

    def test_every_latent_is_observed(self) -> None:
        law = drtmle_stress.base_law()
        assert law.n_latent == len(law.covariate_names)

    def test_the_outcome_is_gaussian(self) -> None:
        # Setting Y to its conditional mean is a valid quadrature only where the mean is a value
        # the outcome can take, which a binomial draw at Q-bar is not.
        assert drtmle_stress.base_law().family == "gaussian"


class TestTheTruthIsWhatTheCompanionIntegrates:
    """``DGP.truth`` and ``truth_at`` must be one quantity, not two that agree by luck."""

    def test_the_quadrature_truth_matches_the_laws_own(self) -> None:
        law = drtmle_stress.base_law()
        declared = law.truth()
        integrated = drtmle_remainder.truth_at(law, 4_096)
        for name in ("ate", "ey1", "ey0"):
            assert integrated[name] == pytest.approx(declared[name], abs=5e-3), name

    def test_the_ate_is_not_degenerate(self) -> None:
        # A stress cell whose effect were near zero would make every coverage comparison a
        # comparison of intervals around the same number.
        assert abs(drtmle_stress.base_law().truth()["ate"]) > 0.5


class TestTheDeclaredMisspecificationHolds:
    """The load-bearing claim: a GLM is materially worse on **both** nuisances.

    Run at a reduced size and fold count relative to
    :func:`~benchmarks.drtmle_stress.misspecification_reading`'s defaults, because the fast tier
    is meant to stay in the low minutes and the measured ratios are ``3x``--``10x``: a claim
    that large does not need 4,000 rows to separate from ``1.05``.  The reduced reading is what
    is asserted; the module's default reading is what the study records.
    """

    @pytest.fixture(scope="class")
    def reading(self) -> list[dict[str, object]]:
        return drtmle_stress.misspecification_reading(n=1_200, seed=20260204, folds=3)

    def test_every_nuisance_clears_the_floor(self, reading: list[dict[str, object]]) -> None:
        failed = [row["nuisance"] for row in reading if not row["clears_floor"]]
        assert not failed, (
            f"{failed} did not clear {drtmle_stress.MISSPECIFICATION_FLOOR:g}x -- this cell is "
            "declared as a setting where neither nuisance is well specified, and a law a GLM "
            "handles is not that setting however nonlinear its formula looks"
        )

    def test_the_mechanism_is_misspecified_too(self, reading: list[dict[str, object]]) -> None:
        # Named separately from the loop above: `g` is the one the Brier score understates, and
        # it is the nuisance the whole cell would silently lose first.
        mechanism = next(row for row in reading if row["nuisance"] == "g")
        assert float(mechanism["ratio"]) > drtmle_stress.MISSPECIFICATION_FLOOR

    def test_the_excess_risk_is_not_the_plain_risk(self, reading: list[dict[str, object]]) -> None:
        """The two statistics must differ, or the oracle column is not doing anything.

        This is the pin on the reading's *instrument* rather than on the law: if `excess` ever
        equalled `risk`, the oracle would have stopped being subtracted and the floor would
        again be sitting on irreducible variance.
        """
        for row in reading:
            assert float(row["glm_excess"]) < float(row["glm_risk"])


class TestTheReadingDoesNotPerturbTheStudy:
    """It runs on its own size and its own seed, so measuring the property cannot move it."""

    def test_the_settings_are_not_size_dependent(self) -> None:
        # Nothing in this cell carries a bandwidth sequence -- that is what makes it the applied
        # cell rather than a third drift regime -- so the settings must not vary with n.
        assert drtmle_stress.settings("nonlinear", 600) == drtmle_stress.settings(
            "nonlinear", 2_400
        )

    def test_the_reduced_learners_are_not_set_here(self) -> None:
        # Each F5 arm supplies its own pair; a default left in this dictionary would silently
        # win the argument the study is about.
        settings = drtmle_stress.settings("nonlinear", 600)
        assert "reduced_outcome_learner" not in settings
        assert "reduced_treatment_learner" not in settings

    def test_an_unknown_cell_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cell must be one of"):
            drtmle_stress.settings("q-drift", 600)

    def test_the_primary_library_is_flexible(self) -> None:
        settings = drtmle_stress.settings("nonlinear", 600)
        assert settings["outcome_learner"] == drtmle_stress.LEARNER
        assert settings["treatment_learner"] == drtmle_stress.LEARNER
        assert drtmle_stress.LEARNER != "glm"


class TestTheSizesAreDeclared:
    def test_two_sizes_not_three(self) -> None:
        # The middle size exists in the drift cells for clause 4's trend, and this cell's
        # remainder is declared not to be read against that trend.
        assert drtmle_stress.SIZES == (600, 2_400)

    def test_the_cell_name_is_a_tuple(self) -> None:
        # Same surface as `drtmle_tier2` / `drtmle_injection`, so the harness holds one call
        # site rather than a branch on which module supplied the cell.
        assert isinstance(drtmle_stress.CELLS, tuple)
        assert drtmle_stress.CELLS == ("nonlinear",)


def test_the_rejected_alternative_really_is_bound_active() -> None:
    """``weak_overlap_dgp`` was rejected on scope, and the reason is measurable.

    The module records that its propensities crowd against 0 and 1 so that nearly every fit
    would exit bound-active and outside section 7's scope.  That is a claim about a law, so it
    is checked rather than asserted in prose -- if it were false, the rejection would need
    rewriting rather than keeping.
    """
    from cleverly.datasets import weak_overlap_dgp

    law = weak_overlap_dgp()
    latent = law.quadrature(4_096)
    scores = np.asarray(law.propensity(latent), dtype=float)
    crowded = float(np.mean((scores < 0.025) | (scores > 0.975)))

    stress = drtmle_stress.base_law()
    ours = np.asarray(stress.propensity(stress.quadrature(4_096)), dtype=float)
    ours_crowded = float(np.mean((ours < 0.025) | (ours > 0.975)))

    assert crowded > 0.20, crowded
    assert ours_crowded < 0.02, ours_crowded
