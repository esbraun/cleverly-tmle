"""Witnesses of a binding isotonic projection (X76 part (g), Fix-8 plan sections 2.7 and 4).

``isotonic=True`` projects each regimen's two-horizon risk curve by unit-weight isotonic
regression, so with :math:`D = \\hat\\theta_2 - \\hat\\theta_1` each level moves by
:math:`D^-/2` and keeps its standard error.  Horizon :math:`k`'s level is the mean of its
targeted node-1 regression, and the node-1 fluctuation solves its score, so

.. math::

    D = P_n[H_1\\, dY] + P_n[(1 - H_1)\\, dQ^*],

with :math:`dY = (1 - Y_1) Q^*_{22} \\ge 0` on the followers and
:math:`dQ^* = Q^*_{21} - Q^*_{11}` (:func:`tests.studies._isotonic_binding_law.score_identity`).

* On a fixed sample of the study law ``LG2`` (continuous ``W``, logistic outcome learner,
  linear pseudo learner) the identity holds at ``1e-8``, its second term is negative and larger
  than the first, so ``D < 0`` and the never regimen's horizon-1 level moves by more than 0.1 SE
  at the logistic tolerances ``1e-13`` and ``1e-10`` alike.  The moved level's interval is
  centred at the moved level with the unprojected SE.
* On a fixed sample of the discrete survival law ``LG`` fitted with saturated cell means, the
  identity holds, its second term is 0, ``D >= 0`` and nothing moves.

The mutation controls drop the second term, replace :math:`H_1` by the follower indicator, and
centre the interval at the unprojected level; each fails a check.  The pool-adjacent-violators
checks of ``tests/unit/test_isotonic_projection.py`` (E9, M10, M13) stay as they are.
"""

from __future__ import annotations

import functools
import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from tests import discrete_law_survival as mnode
from tests.studies import _isotonic_binding_law as law
from tests.studies.canonical_ltmle import G_BOUNDS, declared_regimens
from tests.studies.ltmle_survival_properties import sample as discrete_sample

estimator_module = importlib.import_module("cleverly.longitudinal.estimator")

#: The fixed ``LG2`` sample: seed 100 moves the never horizon-1 level by about 0.24 SE.
LG2_SEED = 100
LG2_N = 2_000
#: The fixed ``LG`` sample.
LG_SEED = 7_000
LG_N = 2_000
IDENTITY = 1e-8
EXACT = 1e-15
#: Law ``LG``: the always path's node-2 hazard and node-2 retention (rev 2.1 plan, 5.8).
LG_HAZARD = 0.02
LG_RETENTION = 0.25


@functools.cache
def lg2_frame() -> pd.DataFrame:
    return law.draw(LG2_N, LG2_SEED)


@functools.cache
def lg2_fit(tol: float = law.LOGISTIC_TOL, isotonic: bool = True) -> Any:
    return law.fit(lg2_frame(), tol, isotonic=isotonic)


def lg_probabilities() -> np.ndarray:
    """The cell probabilities of ``LG``: the discrete law with the always path's node 2 changed."""
    hazard = mnode.H2.copy()
    hazard[:, 1, :, 1] = LG_HAZARD
    retention = mnode.C2.copy()
    retention[:, 1, :, 1] = LG_RETENTION
    masses = []
    for point in mnode.SUPPORT:
        w, a1, c1, y1, l2, a2, c2, y2 = point
        mass = mnode.P_W[w] * (mnode.G1[w] if a1 == 1 else 1.0 - mnode.G1[w])
        if c1 == 0:
            masses.append(mass * (1.0 - mnode.C1[w, a1]))
            continue
        mass *= mnode.C1[w, a1] * (mnode.H1[w, a1] if y1 == 1 else 1.0 - mnode.H1[w, a1])
        if y1 == 1:
            masses.append(mass)
            continue
        mass *= mnode.P_L2[w, a1] if l2 == 1 else 1.0 - mnode.P_L2[w, a1]
        mass *= mnode.G2[w, a1, l2] if a2 == 1 else 1.0 - mnode.G2[w, a1, l2]
        keep = retention[w, a1, l2, a2]
        if c2 == 0:
            masses.append(mass * (1.0 - keep))
            continue
        risk = hazard[w, a1, l2, a2]
        masses.append(mass * keep * (risk if y2 == 1 else 1.0 - risk))
    probabilities = np.array(masses)
    return probabilities / probabilities.sum()


@functools.cache
def lg_fit() -> Any:
    from cleverly.longitudinal import LTMLE

    frame = discrete_sample(lg_probabilities(), LG_N, LG_SEED)
    return LTMLE(
        declared_regimens(mnode.REGIMEN_SPEC),
        reference=mnode.REGIMEN_REFERENCE,
        outcome_learner=mnode.CellMeans(),
        pseudo_learner=mnode.CellMeans(),
        treatment_learner=mnode.CellMeans(),
        censoring_learner=mnode.CellMeans(),
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        isotonic=True,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def identity_gap(terms: law.ScoreIdentity, *, drop_second: bool = False) -> float:
    explained = terms.followers + terms.residual + (0.0 if drop_second else terms.others)
    return abs(explained - terms.difference)


def centred(result: Any, regimen: str = "never", horizon: int = 1) -> bool:
    """The interval sits at the reported level with the plug-in SE, and the level moved."""
    estimate = result.estimates[law.name(regimen, horizon)]
    level = law.levels(result)[(regimen, horizon)]
    low, high = estimate.ci
    critical = float(norm.ppf(0.975))
    return bool(
        abs((low + high) / 2.0 - estimate.psi) < 1e-12
        and abs((high - low) / 2.0 - critical * level.std_error) < 1e-12
        and abs(estimate.psi - level.unprojected) > 1e-12
    )


class TestTheProjection:
    def test_the_two_horizon_projection_moves_each_level_by_half_the_violation(self) -> None:
        projected, plain = lg2_fit(), lg2_fit(isotonic=False)
        for regimen in law.REGIMENS:
            first, second = (float(plain.estimates[law.name(regimen, h)].psi) for h in law.HORIZONS)
            violation = max(0.0, first - second)
            after = [float(projected.estimates[law.name(regimen, h)].psi) for h in law.HORIZONS]
            assert after[0] == pytest.approx(first - violation / 2.0, abs=EXACT)
            assert after[1] == pytest.approx(second + violation / 2.0, abs=EXACT)

    def test_the_standard_errors_do_not_move(self) -> None:
        projected, plain = lg2_fit(), lg2_fit(isotonic=False)
        for name in plain.estimates:
            assert projected.estimates[name].std_error == plain.estimates[name].std_error

    def test_the_levels_read_back_the_unprojected_fit(self) -> None:
        plain = lg2_fit(isotonic=False)
        for (regimen, horizon), level in law.levels(lg2_fit()).items():
            assert level.unprojected == pytest.approx(
                float(plain.estimates[law.name(regimen, horizon)].psi), abs=EXACT
            )


class TestTheScoreIdentity:
    def test_the_identity_holds_on_lg2(self) -> None:
        for regimen in law.REGIMENS:
            terms = law.score_identity(lg2_fit(), regimen)
            assert identity_gap(terms) < IDENTITY
            assert abs(terms.residual) < IDENTITY

    def test_the_identity_reads_the_same_terms_off_the_unprojected_fit(self) -> None:
        projected = law.score_identity(lg2_fit())
        plain = law.score_identity(lg2_fit(isotonic=False))
        assert projected.difference == pytest.approx(plain.difference, abs=EXACT)
        assert projected.others == pytest.approx(plain.others, abs=EXACT)

    def test_the_first_term_is_nonnegative(self) -> None:
        for regimen in law.REGIMENS:
            assert law.score_identity(lg2_fit(), regimen).followers >= 0.0

    def test_lg2_binds_through_the_second_term(self) -> None:
        terms = law.score_identity(lg2_fit())
        assert terms.others < 0.0
        assert abs(terms.others) > terms.followers
        assert terms.difference < 0.0

    def test_cell_means_on_lg_never_bind(self) -> None:
        result = lg_fit()
        assert result.isotonic_moves == ()
        for regimen in ("never", "always"):
            terms = law.score_identity(result, regimen)
            assert identity_gap(terms) < IDENTITY
            assert abs(terms.others) < 1e-12
            assert terms.difference >= 0.0


class TestTheMove:
    def test_the_move_is_material_and_the_same_at_both_tolerances(self) -> None:
        moves = [
            law.levels(lg2_fit(tol))[("never", 1)].move for tol in (law.LOGISTIC_TOL, law.REFIT_TOL)
        ]
        assert min(moves) >= 0.1
        assert moves[1] == pytest.approx(moves[0], rel=1e-6)

    def test_the_moved_interval_is_centred_at_the_moved_level(self) -> None:
        assert centred(lg2_fit())


class TestTheMutations:
    def test_mutation_dropping_the_second_term_fails_the_identity(self) -> None:
        assert identity_gap(law.score_identity(lg2_fit()), drop_second=True) > 1e-3

    def test_mutation_replacing_the_clever_covariate_by_the_follower_indicator_fails(
        self,
    ) -> None:
        assert identity_gap(law.score_identity(lg2_fit(), indicator=True)) > 1e-3

    def test_mutation_centring_the_interval_at_the_unprojected_level_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = estimator_module._project_fits

        def unmoved(fits: Any) -> Any:
            projected, moves = original(fits)
            return {
                key: replace(fit, psi_scaled=fits[key].psi_scaled) for key, fit in projected.items()
            }, moves

        monkeypatch.setattr(estimator_module, "_project_fits", unmoved)
        assert not centred(law.fit(lg2_frame()))
