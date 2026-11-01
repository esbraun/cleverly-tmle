r"""X11 (e): the stacked report of the learned-rule value (``cv_evaluation=False``).

One pooled :math:`\varepsilon` fluctuates every validation row, and the report is the pooled
plug-in :math:`\hat\psi = \sum_v (W_v/W)\psi^*_{nv}`, with :math:`W_v` fold :math:`v`'s weight
mass. Its target is :math:`\tilde\psi^{s}_{0n} = \sum_v (W_v/W)\Psi_{d_{nv}}(P_0)`, and its curve
is centred at the pooled estimate, :math:`\phi_i = w_i(Z_i - \hat\psi)` with
:math:`Z_i = H_i(Y_i - \bar Q^*_i) + \bar Q^*_i(d_i)`. The stored variance is the centred rule
of the stacked regime report, :math:`\{n(n-1)\}^{-1}\sum_i (\phi_i - \bar\phi)^2`, and
:math:`\bar\phi = 0` to solver tolerance because the pooled score is solved.

* W-e1 (identity): on an exact discrete law whose three unequal folds learn different rules,
  :math:`Z_i` is rebuilt longhand from the oracle mechanism and each fold's targeted regression,
  and

  .. math::

      \sum_i (Z_i-\hat\psi)^2 = \sum_i D_i^2 + 2\sum_v n_v(\psi_v-\hat\psi)s_v
      + \sum_v n_v(\psi_v-\hat\psi)^2,

  with :math:`D_i = Z_i - \psi_{v(i)}` and :math:`s_v` the fold mean of :math:`D`, matches the
  stored variance. The cross term is negative and larger than the between-fold term, so the
  stacked variance is not conservative at finite n; a variance without it misses the stored
  value. Under fold targeting each :math:`s_v` is 0 and so is the cross term. On the continuous law at equal folds
  the fold-evaluated fit at the same split stores :math:`\sum_i D_i^2 / n^2`, and the identity
  holds across the two fits.
* W-e2 (weighting, mutation): an unequal cluster ``split_plan``, unweighted and weighted, gives a
  point of :math:`\sum_v (W_v/W)\psi^*_{nv}` with each fold value rebuilt from the targeted fit.
  The ``1/V`` weighting differs by more than :data:`ONE_OVER_V_GAP`, and under weights the
  ``n_v/n`` weighting differs by more than :data:`ROW_SHARE_GAP`. The weighted variance is
  :math:`W^{-2}\sum_i w_i^2 (Z_i-\hat\psi_w)^2` up to the factor :math:`n/(n-1)`.
* W-e3 (aggregation): ``repeats=4`` reports the mean of four single-split fits at the same split
  seeds, and its variance is the centred rule applied to the mean curve. The median differs from
  the mean by more than :data:`MEAN_MEDIAN_GAP` on the witness draw.
* W-e4 (strata): each stratum estimate is :math:`\sum_v (W_{v,s}/W_s)\psi^*_{s,v}`, and the
  shipped stratified fluctuation solves each stratum's pooled score under that weighting, while
  each fold's own stratum score stays unsolved.
* The record is built for the stacked report, ``learned_rule_truth`` reads its fold weights, the
  summary names the report, the fit reads its registered row T42 under its own ``stacked`` key
  while fold targeting reads experimental, and the omitted-variable bound refuses as an
  implementation gap.
* The stacked record survives ``save``, ``dumps`` and ``retarget``. A blip rule learner builds it,
  and under ``id=`` the stored variance is the cluster-total rule, rebuilt by hand.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import SplitPlan, load
from cleverly.estimators import TMLE
from cleverly.estimators.serialize import dumps, loads
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.inference import influence_variance
from cleverly.interventions import LearnedRule
from cleverly.interventions.learned import LearnedRuleSplit, fold_rules
from cleverly.learners import random_partition
from cleverly.sensitivity import omitted_variable_bounds
from tests.conftest import OracleTreatment
from tests.unit import _learned_rule_support as support
from tests.unit import test_learned_rule_learners as learners
from tests.unit import test_learned_rule_strata as exact

pytestmark = pytest.mark.xdist_group("learned_rule_stacked")

NAME = support.NAME

#: Solver tolerance on the exact law, whose fits run at ``tol=1e-12``.
EXACT = 1e-9

#: The declared floor of ``|1/V average - stacked point|`` on the unequal cluster plan. The
#: unweighted draw gives about 1.5e-4 and the weighted one about 5.9e-4.
ONE_OVER_V_GAP = 1e-4

#: The declared floor of ``|n_v/n average - stacked point|`` under weights. The draw gives
#: about 1.0e-3.
ROW_SHARE_GAP = 5e-4

#: The declared floor of ``|mean - median|`` of the four stacked split estimates on
#: ``law_frame(400, 0)`` at ``random_state=0``. The draw gives about 0.0027, roughly 0.09 of the
#: reported standard error.
MEAN_MEDIAN_GAP = 0.002


# ----------------------------------------------------------------------------- exact law


@cache
def _exact(*, weighted: bool = False, strata: bool = False, fold: bool = False) -> Any:
    """A stacked fit of the exact law of ``test_learned_rule_strata.py``: 1,000 rows, 3 folds."""
    estimator = TMLE(
        learned_rule=LearnedRule(),
        outcome_learner=exact.CellMeans(),
        treatment_learner=OracleTreatment(exact._Law(exact.PROBS)),
        cross_fit=True,
        cv_evaluation=False,
        targeting_scheme="fold" if fold else "pooled",
        n_folds=exact.N_FOLDS,
        simultaneous=False,
        random_state=exact.SEED,
        tol=1e-12,
        max_iter=200,
    )
    columns: dict[str, Any] = {"weights": "weight"} if weighted else {}
    if strata:
        columns["strata"] = ["S"]
    return estimator.fit(
        exact._frame(weighted), outcome="Y", treatment="A", covariates=["W", "S"], **columns
    ).single()


def _longhand(fit: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Z_i``, each fold's weighted plug-in ``psi_v`` and each row's fold, rebuilt by hand.

    ``H_i = 1{A_i = d_v(W_i)} / g(A_i | W_i)`` reads the oracle mechanism and the rule that
    ``exact._fold_rule`` relearns from the training rows, and ``Qbar*`` is the fold's targeted
    regression.
    """
    rows = exact._rows(fit)
    z = np.empty(fit.data.n)
    fold_of = np.empty(fit.data.n, dtype=int)
    plug_ins = []
    for fold, (train, test) in enumerate(exact._folds(fit)):
        density = exact._density(exact._fold_rule(fit, train))
        q_star = exact._q_star(fit, test)
        w, a = rows["w"][test], rows["a"][test].astype(int)
        g = np.where(a == 1, exact.G_EXACT[w], 1.0 - exact.G_EXACT[w])
        plug_in = (density * q_star).sum(axis=1)[w]
        z[test] = density[w, a] / g * (rows["y"][test] - q_star[w, a]) + plug_in
        plug_ins.append(float(np.average(plug_in, weights=rows["weight"][test])))
        fold_of[test] = fold
    return z, np.asarray(plug_ins), fold_of


def _terms(z: np.ndarray, plug_ins: np.ndarray, fold_of: np.ndarray, psi: float) -> dict[str, Any]:
    """The three terms of the identity, and the per-fold residual means ``s_v``."""
    d = z - plug_ins[fold_of]
    sizes = np.bincount(fold_of)
    s = np.array([float(np.mean(d[fold_of == fold])) for fold in range(sizes.size)])
    return {
        "within": float(np.sum(d**2)),
        "cross": float(2.0 * np.sum(sizes * (plug_ins - psi) * s)),
        "between": float(np.sum(sizes * (plug_ins - psi) ** 2)),
        "total": float(np.sum((z - psi) ** 2)),
        "s": s,
    }


class TestTheVarianceIdentity:
    """W-e1."""

    def test_premise_the_folds_are_unequal_and_learn_different_rules(self) -> None:
        fit = _exact()
        rules = fold_rules(fit.nuisance)
        assert np.any(rules.min(axis=1) != rules.max(axis=1))
        sizes = [test.size for _, test in exact._folds(fit)]
        assert len(set(sizes)) > 1, sizes

    def test_the_stored_curve_is_z_minus_the_pooled_estimate(self) -> None:
        fit = _exact()
        z, plug_ins, _ = _longhand(fit)
        estimate = fit.estimates[NAME]
        np.testing.assert_allclose(estimate.influence_curve, z - estimate.psi, rtol=0, atol=EXACT)
        # The pooled score is solved, so the curve has mean zero to solver tolerance.
        assert abs(float(np.mean(estimate.influence_curve))) < EXACT
        np.testing.assert_allclose(
            fit.extra["learned_rule"].splits[0].fold_estimates, plug_ins, rtol=0, atol=EXACT
        )

    def test_the_identity_rebuilds_the_stored_variance_with_the_cross_term(self) -> None:
        fit = _exact()
        estimate, n = fit.estimates[NAME], fit.data.n
        terms = _terms(*_longhand(fit), estimate.psi)
        assert terms["total"] == pytest.approx(
            terms["within"] + terms["cross"] + terms["between"], rel=1e-12
        )
        stored = estimate.variance * n * (n - 1)
        assert stored == pytest.approx(terms["total"], rel=1e-9)
        # Nonzero: pooled targeting leaves every fold's score unsolved, and the cross term is
        # negative and larger than the between-fold term, so the stacked variance is below the
        # fold second moment on this draw.
        assert np.min(np.abs(terms["s"])) > 1e-2, terms["s"]
        assert terms["cross"] < -terms["between"] < 0.0, terms
        assert stored < terms["within"]

    def test_a_variance_without_the_cross_term_misses_the_stored_value(self) -> None:
        fit = _exact()
        estimate, n = fit.estimates[NAME], fit.data.n
        terms = _terms(*_longhand(fit), estimate.psi)
        mutated = terms["within"] + terms["between"]
        assert estimate.variance * n * (n - 1) != pytest.approx(mutated, rel=1e-4)

    def test_under_fold_targeting_the_cross_term_vanishes(self) -> None:
        fit = _exact(fold=True)
        estimate, n = fit.estimates[NAME], fit.data.n
        terms = _terms(*_longhand(fit), estimate.psi)
        assert np.max(np.abs(terms["s"])) < EXACT, terms["s"]
        assert abs(terms["cross"]) < EXACT
        assert estimate.variance * n * (n - 1) == pytest.approx(
            terms["within"] + terms["between"], rel=1e-9
        )
        # The target keeps the fold weights W_v / W.
        (split,) = fit.extra["learned_rule"].splits
        assert estimate.psi == pytest.approx(
            float(np.dot(split.fold_weights, split.fold_estimates)), abs=1e-12
        )

    def test_at_equal_folds_the_fold_evaluated_fit_stores_the_within_term(self) -> None:
        """Across two fits: the stacked variance is the fold-evaluated one plus both terms."""
        stacked = support.fit(cv_evaluation=False)
        evaluated = support.fit(cv_evaluation=True)
        assert {test.size for _, test in stacked.nuisance.folds} == {80}
        np.testing.assert_allclose(
            stacked.fluctuations["regime"].epsilon,
            evaluated.fluctuations["regime"].epsilon,
            rtol=0,
            atol=1e-12,
        )
        estimate, n = stacked.estimates[NAME], stacked.data.n
        (split,) = stacked.extra["learned_rule"].splits
        plug_ins = np.asarray(split.fold_estimates)
        fold_of = np.asarray(stacked.nuisance.folds.assignment, dtype=int)
        z = estimate.influence_curve + estimate.psi
        terms = _terms(z, plug_ins, fold_of, estimate.psi)
        np.testing.assert_allclose(
            evaluated.estimates[NAME].influence_curve,
            z - plug_ins[fold_of],
            rtol=0,
            atol=1e-12,
        )
        within = evaluated.estimates[NAME].variance * n**2
        assert within == pytest.approx(terms["within"], rel=1e-10)
        assert estimate.variance * n * (n - 1) == pytest.approx(
            within + terms["cross"] + terms["between"], rel=1e-9
        )
        assert terms["cross"] < -terms["between"] < 0.0, terms


# ----------------------------------------------------------------------- weighting


def _clustered(seed: int = 0) -> pd.DataFrame:
    """60 clusters of 2 to 14 rows on the support law, with a known weight of 1 or 2.5."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 15, 60)
    frame = support.law_frame(int(sizes.sum()), seed)
    return frame.assign(c=np.repeat(np.arange(sizes.size), sizes), w=1.0 + 1.5 * (frame["W1"] > 0))


@cache
def _unequal(weighted: bool) -> Any:
    frame = _clustered()
    draw = random_partition(len(frame), 5, cluster=frame["c"].to_numpy(), seed=3)
    estimator = support.learned_tmle(cv_evaluation=False, split_plan=SplitPlan.from_folds([draw]))
    keywords: dict[str, Any] = {"weights": "w"} if weighted else {}
    return estimator.fit(
        frame, outcome="Y", treatment="A", covariates=support.COVARIATES, id="c", **keywords
    ).single()


def _fold_values(fit: Any) -> np.ndarray:
    """Each fold's weighted mean of ``Qbar*(d_i, W_i)``, from the targeted fit by hand."""
    targeted = fit.fluctuations["regime"].targeted
    rule = np.asarray(fit.nuisance.regimes.values[:, 1, 0], dtype=float)
    value = rule * np.asarray(targeted.arms[1.0]) + (1.0 - rule) * np.asarray(targeted.arms[0.0])
    weights = np.asarray(fit.data.weights, dtype=float)
    return np.array(
        [float(np.average(value[test], weights=weights[test])) for _, test in fit.nuisance.folds]
    )


class TestTheFoldWeights:
    """W-e2."""

    def test_premise_the_plan_has_unequal_folds(self) -> None:
        (split,) = _unequal(False).extra["learned_rule"].splits
        assert max(split.fold_sizes) - min(split.fold_sizes) >= 20, split.fold_sizes

    @pytest.mark.parametrize("weighted", [False, True], ids=["rows", "weighted"])
    def test_the_point_weights_each_fold_by_its_mass(self, weighted: bool) -> None:
        fit = _unequal(weighted)
        (split,) = fit.extra["learned_rule"].splits
        weights = np.asarray(fit.data.weights, dtype=float)
        masses = np.array([np.sum(weights[test]) for _, test in fit.nuisance.folds])
        np.testing.assert_allclose(split.fold_masses, masses, rtol=0, atol=1e-12)
        np.testing.assert_allclose(split.fold_weights, masses / masses.sum(), rtol=0, atol=1e-15)
        values = _fold_values(fit)
        np.testing.assert_allclose(split.fold_estimates, values, rtol=0, atol=1e-12)
        point = fit.estimates[NAME].psi
        assert point == pytest.approx(float(np.dot(masses / masses.sum(), values)), abs=1e-12)
        assert split.estimate == point
        # The 1/V mutation: the fold-evaluated weighting moves the point by a declared amount.
        assert abs(float(np.mean(values)) - point) > ONE_OVER_V_GAP

    def test_under_weights_the_row_share_is_a_mutation_too(self) -> None:
        fit = _unequal(True)
        (split,) = fit.extra["learned_rule"].splits
        sizes = np.asarray(split.fold_sizes, dtype=float)
        rows = float(np.dot(sizes / sizes.sum(), _fold_values(fit)))
        assert abs(rows - fit.estimates[NAME].psi) > ROW_SHARE_GAP

    def test_the_weighted_variance_is_the_weighted_second_moment(self) -> None:
        """``W^{-2} sum_i w_i^2 (Z_i - psi_w)^2``, times ``n / (n - 1)``, on mean-one weights."""
        fit = _unequal(True)
        estimate = fit.estimates[NAME]
        weights = np.asarray(fit.data.weights, dtype=float)
        n, total = fit.data.n, float(np.sum(weights))
        assert total == pytest.approx(n, rel=1e-12)
        z = estimate.influence_curve / weights + estimate.psi
        assert abs(float(np.average(z - estimate.psi, weights=weights))) < 1e-8
        weighted = float(np.sum(weights**2 * (z - estimate.psi) ** 2) / total**2)
        # The fit is clustered, so its stored variance is the cluster rule; the row rule of
        # the same curve is the weighted second moment.
        assert influence_variance(estimate.influence_curve) == pytest.approx(
            weighted * n / (n - 1), rel=1e-6
        )


# ------------------------------------------------------------------- mean over splits


@cache
def _repeated() -> Any:
    return support.fit(cv_evaluation=False, repeats=4)


@cache
def _singles() -> tuple[Any, ...]:
    return tuple(
        support.fit(cv_evaluation=False, random_state=repeat.seed) for repeat in _repeated().repeats
    )


class TestTheMeanOverSplits:
    """W-e3."""

    def test_the_point_is_the_mean_of_single_split_fits(self) -> None:
        reported = _repeated().estimates[NAME]
        singles = [fit.estimates[NAME].psi for fit in _singles()]
        assert reported.psi == float(np.mean(singles))
        record = _repeated().extra["learned_rule"]
        assert [split.estimate for split in record.splits] == singles
        assert {split.scheme for split in record.splits} == {"stacked"}
        # Nonzero: the median is not the mean on this draw.
        assert abs(float(np.median(singles)) - reported.psi) > MEAN_MEDIAN_GAP

    def test_the_variance_is_the_centred_rule_of_the_mean_curve(self) -> None:
        reported = _repeated().estimates[NAME]
        curves = [fit.estimates[NAME].influence_curve for fit in _singles()]
        phibar = np.mean(curves, axis=0)
        np.testing.assert_array_equal(reported.influence_curve, phibar)
        assert reported.variance == influence_variance(phibar)
        # Each split's curve is centred at its own pooled estimate.
        assert max(abs(float(np.mean(curve))) for curve in curves) < 1e-8
        n = _repeated().data.n
        assert reported.variance == pytest.approx(np.sum(phibar**2) / (n * (n - 1)), rel=1e-6)


# ----------------------------------------------------------------------------- strata


class TestTheStrata:
    """W-e4."""

    @pytest.mark.parametrize("weighted", [False, True], ids=["rows", "weighted"])
    def test_each_stratum_targets_the_stratum_mass_weighting(self, weighted: bool) -> None:
        fit = _exact(weighted=weighted, strata=True)
        rows = exact._rows(fit)
        folds = exact._folds(fit)
        for code in exact.STRATA:
            masses = np.array(
                [np.sum(rows["weight"][test][rows["s"][test] == code]) for _, test in folds]
            )
            plug_ins = np.array(
                [
                    exact._plug_in(fit, code, test, exact._fold_rule(fit, train))
                    for train, test in folds
                ]
            )
            reported = fit.estimates[exact._stratum(code)].psi
            assert reported == pytest.approx(
                float(np.dot(masses / masses.sum(), plug_ins)), abs=EXACT
            )
            if code == 0:
                # Stratum 0 holds the cell where the fold rules disagree, so its fold
                # plug-ins differ and the 1/V weighting is a different number.
                assert abs(float(np.mean(plug_ins)) - reported) > 1e-6

    @pytest.mark.parametrize("weighted", [False, True], ids=["rows", "weighted"])
    def test_each_stratum_pooled_score_is_solved(self, weighted: bool) -> None:
        fit = _exact(weighted=weighted, strata=True)
        rows = exact._rows(fit)
        for code in exact.STRATA:
            per_fold, masses = [], []
            for train, test in exact._folds(fit):
                per_fold.append(exact._score(fit, code, test, exact._fold_rule(fit, train)))
                masses.append(float(np.sum(rows["weight"][test][rows["s"][test] == code])))
            pooled = float(np.dot(masses, per_fold) / np.sum(masses))
            assert abs(pooled) < EXACT, per_fold
            # Nonzero witness: each fold's own stratum score is not solved.
            assert max(abs(value) for value in per_fold) > 1e-3, per_fold


# ----------------------------------------------------------------- record and surfaces


class TestTheRecordAndTheSurfaces:
    """The record, the summary, the support key and the refusal of the bound."""

    def test_the_record_names_the_stacked_report(self) -> None:
        record = support.fit(cv_evaluation=False).extra["learned_rule"]
        assert record.scheme == "stacked"
        assert record.target == "fold-mass-weighted data-adaptive value"
        line = record.describe()
        assert line.startswith(
            "Target: the average of the values of the 5 rules learned on the training folds, "
            "weighted by each fold's share of the rows (fold weights "
        ), line
        assert "(the stacked report)" in line and "weights=" not in line

    def test_a_weighted_record_names_the_total_weight(self) -> None:
        line = _unequal(True).extra["learned_rule"].describe()
        assert "weighted by each fold's share of the total weight (fold weights " in line, line
        assert "share of the rows" not in _unequal(True).extra["learned_rule"].describe()
        assert "share of the rows" in _unequal(False).extra["learned_rule"].describe()

    def test_the_fold_evaluated_record_keeps_one_over_v(self) -> None:
        (split,) = _unequal_fold_evaluated().extra["learned_rule"].splits
        assert split.scheme == "fold_evaluated"
        assert split.fold_weights == (0.2,) * 5
        assert len(set(split.fold_sizes)) > 1

    def test_an_unknown_scheme_is_a_defect(self) -> None:
        with pytest.raises(ValueError, match="scheme"):
            LearnedRuleSplit(
                seed=0,
                scheme="pooled",
                estimate=0.0,
                fold_sizes=(1,),
                fold_masses=(1.0,),
                fold_estimates=(0.0,),
                arm_shares=((1.0, 0.0),),
                gap_quantiles=((0.0,),),
                score_quantiles=(),
                inner_seeds=(),
            )

    def test_the_fit_reads_its_registered_row(self) -> None:
        """``learned-rule-stacked`` registers T42; a composition it did not measure stays off."""
        result = support.fit(cv_evaluation=False)
        assert result.config.support_key.splitting == ("stacked",)
        assert result.estimates[NAME].support.label == "supported"
        assert result.estimates[NAME].support.row == "T42.rows"
        # The mutation: fold targeting under the stacked report keys ``fold_targeted``, which
        # no row of the study declares.
        folded = support.fit(cv_evaluation=False, targeting_scheme="fold")
        assert folded.config.support_key.splitting == ("fold_targeted",)
        assert folded.estimates[NAME].support.label == "experimental"
        clustered = _unequal(False)
        assert "unequal_clusters" in clustered.config.support_key.qualifiers

    def test_the_omitted_variable_bound_refuses_as_a_gap(self) -> None:
        result = support.fit(cv_evaluation=False)
        with pytest.raises(CapabilityError) as caught:
            omitted_variable_bounds(result, NAME, cf_y=0.05, cf_d=0.05)
        assert caught.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP
        message = str(caught.value)
        assert "X11 (n2)" in message and "W_v / W" in message
        # The remedy, the fold-evaluated report, runs the bound.
        assert omitted_variable_bounds(support.fit(), NAME, cf_y=0.05, cf_d=0.05) is not None


class TestThePersistenceAndTheCompositions:
    """The stacked record survives a round trip and ``retarget``, and the admitted rule
    learner and cluster compositions build it."""

    @pytest.mark.parametrize("route", ["save", "dumps"])
    def test_a_round_trip_keeps_the_stacked_record(self, route: str, tmp_path: Any) -> None:
        result = _unequal(True)
        restored = (
            load(result.save(tmp_path / "stacked.joblib"))
            if route == "save"
            else loads(dumps(result))
        )
        record, saved = result.extra["learned_rule"], restored.extra["learned_rule"]
        assert saved == record
        assert saved.scheme == "stacked"
        (split,), (saved_split,) = record.splits, saved.splits
        assert saved_split.fold_masses == split.fold_masses
        assert saved_split.fold_weights == split.fold_weights
        # The witness is not the 1/V weighting, which a lost field would fall back to.
        assert saved_split.fold_weights != (0.2,) * 5
        estimate, kept = result.estimates[NAME], restored.estimates[NAME]
        assert kept.psi == estimate.psi and kept.variance == estimate.variance

    def test_retarget_reproduces_the_stacked_fit(self) -> None:
        result = _unequal(True)
        estimates, _ = result.estimator.retarget(
            result.data, result.nuisance, estimands=("ey_learned_rule",)
        )
        estimate = result.estimates[NAME]
        assert estimates[NAME].psi == pytest.approx(estimate.psi, abs=1e-12)
        np.testing.assert_allclose(
            estimates[NAME].influence_curve, estimate.influence_curve, rtol=0, atol=1e-12
        )
        assert estimates[NAME].variance == pytest.approx(estimate.variance, rel=1e-10)

    def test_a_stacked_blip_fit_builds_its_record(self) -> None:
        result = support.fit(cv_evaluation=False, learned_rule=learners.blip_rule())
        (split,) = result.extra["learned_rule"].splits
        assert split.scheme == "stacked"
        assert len(split.score_quantiles) == len(split.fold_sizes)
        assert all(len(levels) > 0 for levels in split.score_quantiles)
        assert float(np.dot(split.fold_weights, split.fold_estimates)) == pytest.approx(
            result.estimates[NAME].psi, abs=1e-12
        )

    def test_the_clustered_variance_is_the_cluster_total_rule(self) -> None:
        """``J / (J - 1)`` times the sum of squared centred cluster totals, over ``n^2``."""
        fit = _unequal(False)
        curve = fit.estimates[NAME].influence_curve
        totals = pd.Series(curve).groupby(_clustered()["c"].to_numpy()).sum().to_numpy()
        clusters, n = totals.size, fit.data.n
        assert clusters == 60
        by_hand = clusters / (clusters - 1) * float(np.sum((totals - totals.mean()) ** 2)) / n**2
        assert fit.estimates[NAME].variance == pytest.approx(by_hand, rel=1e-12)
        # The row rule of the same curve differs, so the check reads the cluster rule.
        assert abs(influence_variance(curve) / by_hand - 1.0) > 0.05


@cache
def _unequal_fold_evaluated() -> Any:
    frame = _clustered()
    draw = random_partition(len(frame), 5, cluster=frame["c"].to_numpy(), seed=3)
    estimator = support.learned_tmle(split_plan=SplitPlan.from_folds([draw]))
    return estimator.fit(
        frame, outcome="Y", treatment="A", covariates=support.COVARIATES, id="c"
    ).single()
