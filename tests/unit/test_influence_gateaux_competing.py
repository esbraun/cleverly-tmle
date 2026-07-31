r"""The competing-risks law, checked against itself before any estimator reads it.

:mod:`tests.discrete_law_competing` states the cause-specific cumulative incidence
longhand and differentiates it by a complex step.  Everything in this module is a
property of that law alone -- no ``LTMLE``, no clever covariate, no cumulative product --
and every one of them would hold of a correct law whether or not the library could fit it.

That is the point of running them first.  The evidence this package offers that an
influence curve is right is agreement between two independently written things, and an
oracle that agrees with a buggy estimator because both share a mistake is worth nothing.
So the law is made to answer for itself here: an efficient influence function has mean
zero under the law it is taken at, the causes and the survival probability exhaust the
mass, and collapsing the two causes into one has to reproduce the single-event recursion
the survival law already proves.  None of those can be arranged by agreeing with an
estimator, because no estimator is involved.

One mutation was run against this module and seen to fail it, which is the only thing that
makes the claims above evidence rather than assertion: writing the survival factor in
:func:`~tests.discrete_law_competing.functional` as ``1 - hazard1`` -- the cause's *own*
survival -- in place of the all-cause ``survived / reached``.  It takes **4 of the 82**
tests with it, and the informative part is *which* four: the two that ask whether the
causes still separate, the sum-to-one identity, and the merge against the single-event
recursion.  The sum comes back ``1.0625``, an incidence total greater than one.

What does **not** catch it is every test in :class:`TestTheInfluenceCurves`, and that is
worth stating rather than discovering later.  The mutated functional is still perfectly
pathwise differentiable, so its curve still has mean zero and still varies across the
support; it is simply the influence function of a different parameter.  Mean-zero
validates the machinery, never the estimand -- which is why the identities in
:class:`TestTheParameterIsACumulativeIncidence` are here and are not decoration.

A second mutation was run against the **library**, and it is the one all of this is aimed
at: ``fit_regimen`` composing the pseudo-outcome with the cause's own survival,
``failed + (1 - failed) * carried``, in place of the all-cause
``failed + (1 - event_by(time)) * carried``.  It takes **21 of the 130** tests, and again
the pattern is the informative part -- every failure is at ``t = 2``.  At the first horizon
there is no survival factor to get wrong, so the mutation is invisible there and the ten
horizon-1 parameters stay green.

Unlike the survival module's mechanism mutation, this one is *not* silent in the point
estimate: it changes what the earlier regression is fitted to, so the plug-in moves and the
``psi`` tests fail beside the Gateaux ones.  That is worth knowing in the other direction --
it means a competing-risks fit that is wrong this way cannot be mistaken for one that is
merely inefficient.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cleverly.longitudinal import LTMLE

from .. import discrete_law_competing as law


def _cif(label: str, cause: str, horizon: int) -> float:
    return law.TRUTH[f"cif_regimen[{label}, {cause} @ t={horizon}]"]


def _ate(label: str, cause: str, horizon: int) -> float:
    return law.TRUTH[f"ate_regimen[{label} vs {law.REGIMEN_REFERENCE}, {cause} @ t={horizon}]"]


class TestTheSampleRealisesTheLaw:
    """The claims that make every other assertion here exact rather than statistical."""

    def test_the_cells_are_multiples_of_one_over_n(self) -> None:
        """Import-time guard, restated as a test so it reads as a claim and not a fluke."""
        assert law.COUNTS.sum() == law.N
        np.testing.assert_array_equal(law.PROBS, law.COUNTS / law.N)
        assert abs(float(law.PROBS.sum()) - 1.0) < 1e-15

    def test_no_support_point_is_empty(self) -> None:
        """An empty block would compare the influence curve at the wrong row.

        :func:`~tests.discrete_law_competing.first_row_of` is a prefix sum of the counts,
        so a block of size zero makes its entry point at the *next* block's first row.
        The Gateaux comparison would then read a curve belonging to a different history
        and could still pass, which is the worst of the available failures -- hence a
        cause-specific hazard is never allowed to be zero.
        """
        assert int(law.COUNTS.min()) > 0
        starts = law.first_row_of()
        assert len(starts) == len(law.SUPPORT)
        assert len(np.unique(starts)) == len(starts)

    def test_the_frame_carries_the_absorbing_structure(self) -> None:
        """Exclusivity and absorption, read off the frame a fit is actually handed.

        The support tuple is three-valued, so exclusivity is structural *there*; the frame
        expands it into an indicator per cause, and this is where that expansion is
        checked. A unit at risk has an answer for both causes and fires at most one; a
        unit that fired has nothing at any later node.
        """
        frame = law.frame()
        assert frame.shape[0] == law.N
        assert list(frame.columns) == ["W", "A1", "C1", "R1", "D1", "L2", "A2", "C2", "R2", "D2"]

        at_risk = frame["C1"] == 1
        assert frame.loc[at_risk, ["R1", "D1"]].notna().all().all()
        assert not ((frame["R1"] == 1) & (frame["D1"] == 1)).any()
        assert not ((frame["R2"] == 1) & (frame["D2"] == 1)).any()

        had_event = (frame["R1"] == 1) | (frame["D1"] == 1)
        assert had_event.any(), "no unit had an event at the first node"
        assert frame.loc[had_event, ["L2", "A2", "C2", "R2", "D2"]].isna().all().all()

    def test_outcome_columns_matches_the_frame(self) -> None:
        """The declaration and the frame are built from one table, and must agree."""
        columns = law.outcome_columns()
        assert list(columns) == list(law.CAUSES)
        frame = law.frame()
        for cause, names in columns.items():
            assert len(names) == len(law.HORIZONS), cause
            for name in names:
                assert name in frame.columns


class TestTheLawIsWorthCheckingAgainst:
    """Non-degeneracy: a parameter no bug can move is not evidence of anything."""

    @pytest.mark.parametrize("name", law.NAMES)
    def test_the_truth_is_finite_and_on_scale(self, name: str) -> None:
        value = law.TRUTH[name]
        assert np.isfinite(value)
        if name.startswith("cif_regimen["):
            assert 0.0 < value < 1.0

    def test_no_incidence_sits_on_the_filler(self) -> None:
        """``sequential._FILLER`` is ``0.5``, the value a row nothing reads is given.

        A cumulative incidence that happened to equal it would let a filled row leak into
        the estimate without moving the number, so the constants are chosen to keep every
        level away from it.  A *contrast* is a difference and carries no such reading,
        which is why only the levels are checked.
        """
        for label, cause, horizon in itertools.product(law.REGIMEN_ARMS, law.CAUSES, law.HORIZONS):
            assert abs(_cif(label, cause, horizon) - 0.5) > 1e-3

    @pytest.mark.parametrize("cause", law.CAUSES)
    @pytest.mark.parametrize("horizon", law.HORIZONS)
    def test_the_regimens_separate(self, cause: str, horizon: int) -> None:
        """Every pair of regimens differs -- except the one pair that must not.

        ``continue_if_l2`` treats at the first node, so at ``t = 1`` it **is** ``always``.
        That equality is a true statement about the law rather than a degeneracy, and the
        estimator-side module asserts it bit for bit; here it is exempted rather than
        silently tolerated.
        """
        for left, right in itertools.combinations(law.REGIMEN_ARMS, 2):
            gap = abs(_cif(left, cause, horizon) - _cif(right, cause, horizon))
            if horizon == 1 and {left, right} == {"always", "continue_if_l2"}:
                assert gap == 0.0
            else:
                assert gap > 0.02, f"{left} and {right} coincide at t={horizon}"

    @pytest.mark.parametrize("cause", law.CAUSES)
    @pytest.mark.parametrize("horizon", law.HORIZONS)
    def test_the_causes_separate(self, cause: str, horizon: int) -> None:
        """The two causes' incidences differ under every regimen.

        With equal incidences a fit that reported one cause's curve under both names
        would pass everything.
        """
        del cause  # the assertion is about the pair, and runs once per parametrisation
        for label in law.REGIMEN_ARMS:
            gap = abs(_cif(label, law.CAUSES[0], horizon) - _cif(label, law.CAUSES[1], horizon))
            assert gap > 0.02, f"the causes coincide for {label} at t={horizon}"

    def test_no_contrast_is_zero_and_the_causes_are_not_mirrored(self) -> None:
        """Each contrast is a parameter a bug can move, and the two are not symmetric.

        Treatment raises the incidence of relapse and lowers that of death, so the two
        contrasts differ in sign.  They are also kept different in *size*: were they
        mirror images, a fit that swapped the causes would cancel rather than show.
        """
        for label in law.REGIMEN_ARMS:
            if label == law.REGIMEN_REFERENCE:
                continue
            for horizon in law.HORIZONS:
                first = _ate(label, law.CAUSES[0], horizon)
                second = _ate(label, law.CAUSES[1], horizon)
                assert abs(first) > 0.02 and abs(second) > 0.02
                assert first * second < 0.0, "the causes should move in opposite directions"
                assert abs(abs(first) - abs(second)) > 0.01, "the contrasts are mirror images"

    def test_the_rule_is_a_parameter_no_static_plan_reaches(self) -> None:
        """``continue_if_l2`` must part company with ``always`` once it has acted.

        It idles nowhere at the first node, so the two agree at ``t = 1`` exactly; if they
        also agreed at ``t = 2`` the dynamic path would be untested by this law.
        """
        for cause in law.CAUSES:
            assert _cif("continue_if_l2", cause, 1) == _cif("always", cause, 1)
            assert abs(_cif("continue_if_l2", cause, 2) - _cif("always", cause, 2)) > 0.02


class TestTheParameterIsACumulativeIncidence:
    """The identities that say these numbers are incidences and not something else."""

    @pytest.mark.parametrize("cause", law.CAUSES)
    def test_each_incidence_is_monotone_in_the_horizon(self, cause: str) -> None:
        """A cumulative incidence cannot fall: the event is absorbing."""
        for label in law.REGIMEN_ARMS:
            assert _cif(label, cause, 2) >= _cif(label, cause, 1)

    def test_the_causes_and_survival_exhaust_the_mass(self) -> None:
        r"""``sum_j F_j(k) + S(k) == 1``, against a separately written ``S``.

        :func:`~tests.discrete_law_competing.survival_functional` states the all-cause
        event-free probability longhand rather than as one minus the numbers being
        checked, so this is a real identity and not an algebraic tautology.  It is exact
        here because it is one functional of one law -- which is precisely the contrast
        with the *estimator*, where each cause is separately regressed and separately
        fluctuated and the sum is a diagnostic rather than a guarantee.
        """
        for label, horizon in itertools.product(law.REGIMEN_ARMS, law.HORIZONS):
            total = sum(_cif(label, cause, horizon) for cause in law.CAUSES)
            survival = float(law.survival_functional(law.PROBS, label, horizon))
            assert total + survival == pytest.approx(1.0, abs=1e-12)

    def test_merging_the_causes_reproduces_the_single_event_recursion(self) -> None:
        r"""The all-cause incidence must equal what one absorbing event would give.

        Collapse the two causes and the parameter is the cumulative risk
        :mod:`tests.discrete_law_survival` already proves the derivation of:

        .. math::

            F(2) = \sum_w P(W = w)\bigl[h_1 + (1 - h_1) \sum_l P(L_2 = l) h_2\bigr]

        with :math:`h` the *all-cause* hazard.  Summing the cause-specific incidences
        telescopes to that expression only because each of them carries the all-cause
        survival factor.  Under the mistake this law exists to catch -- a cause's own
        survival, :math:`1 - h_{1j}`, in place of :math:`1 - \sum_{j'} h_{1j'}` -- the sum
        picks up a cross term and this identity is the first thing to break.

        Written out here against the law's own masses rather than imported from the
        survival module, whose constants are different.
        """
        probs = law.PROBS
        for label in law.REGIMEN_ARMS:
            node1, node2 = law.REGIMEN_ARMS[label]
            for horizon in law.HORIZONS:
                merged = 0.0
                total_mass = law._mass(probs)
                for w in (0, 1):
                    a1 = law._arm(node1, w)
                    share = law._mass(probs, w=w) / total_mass
                    reached = law._mass(probs, w=w, a1=a1, c1=1)
                    survived = law._mass(probs, w=w, a1=a1, c1=1, j1=0)
                    hazard = 1.0 - survived / reached
                    if horizon == 1:
                        merged += share * hazard
                        continue
                    later = 0.0
                    for l2 in (0, 1):
                        a2 = law._arm(node2, w, l2)
                        density = law._mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2) / survived
                        at_risk = law._mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1)
                        free = law._mass(probs, w=w, a1=a1, c1=1, j1=0, l2=l2, a2=a2, c2=1, j2=0)
                        later += density * (1.0 - free / at_risk)
                    merged += share * (hazard + (1.0 - hazard) * later)

                per_cause = sum(_cif(label, cause, horizon) for cause in law.CAUSES)
                assert per_cause == pytest.approx(merged, abs=1e-12), (
                    f"the causes do not sum to the all-cause risk for {label} at t={horizon}"
                )


class TestTheInfluenceCurves:
    """Properties of the complex-step derivative, checked without an estimator."""

    @pytest.mark.parametrize("name", law.NAMES)
    def test_the_influence_curve_has_mean_zero(self, name: str) -> None:
        """An efficient influence function integrates to zero under its own law.

        This validates :func:`~tests.discrete_law_competing.functional` and the complex
        step together and by themselves: a functional that was not pathwise differentiable
        at ``PROBS``, or a step that came back real because a comparison touched the
        probabilities, would not produce a mean-zero curve by accident.
        """
        curve = law.eif(name)
        assert curve.shape == (len(law.SUPPORT),)
        assert np.all(np.isfinite(curve))
        assert float(np.dot(law.PROBS, curve)) == pytest.approx(0.0, abs=1e-13)

    @pytest.mark.parametrize("name", law.NAMES)
    def test_the_complex_step_did_not_come_back_real(self, name: str) -> None:
        """A constant curve is what a lost derivative looks like.

        If an arm were selected by comparing *probabilities* rather than by cell index,
        the perturbation would never reach the selection and the derivative would be
        silently wrong rather than absent.  A curve that does not vary across the support
        is the signature.
        """
        curve = law.eif(name)
        assert float(np.ptp(curve)) > 1e-6

    def test_the_contrast_curve_is_the_difference_of_the_two(self) -> None:
        """``IC(ate) == IC(treated) - IC(reference)`` at the same cause and horizon.

        Cheap, exact, and it fails on the mistake a third index invites: pairing a
        contrast with the reference regimen at the wrong cause or the wrong horizon.
        """
        for label in law.REGIMEN_ARMS:
            if label == law.REGIMEN_REFERENCE:
                continue
            for cause, horizon in itertools.product(law.CAUSES, law.HORIZONS):
                left = law.eif(f"cif_regimen[{label}, {cause} @ t={horizon}]")
                right = law.eif(f"cif_regimen[{law.REGIMEN_REFERENCE}, {cause} @ t={horizon}]")
                both = law.eif(
                    f"ate_regimen[{label} vs {law.REGIMEN_REFERENCE}, {cause} @ t={horizon}]"
                )
                np.testing.assert_allclose(both, left - right, atol=1e-13, rtol=0)


class TestTheNameTable:
    """The registry gate's law-side half; the estimator-side half arrives with the fit."""

    def test_every_name_has_a_truth_and_functional_refuses_the_rest(self) -> None:
        """``functional`` raising on an unknown name is what makes the gate real."""
        assert set(law.TRUTH) == set(law.NAMES)
        assert len(law.NAMES) == len(law.REGIMEN_ARMS) * len(law.CAUSES) * len(law.HORIZONS) + (
            (len(law.REGIMEN_ARMS) - 1) * len(law.CAUSES) * len(law.HORIZONS)
        )
        for name in law.NAMES:
            assert law.functional(law.PROBS, name) == pytest.approx(law.TRUTH[name], abs=0)
        with pytest.raises(ValueError, match="unknown estimand"):
            law.functional(law.PROBS, "risk_regimen[always @ t=1]")

    def test_the_index_is_split_from_the_outside_in(self) -> None:
        """Horizon, then cause, then the contrast -- and the order is not arbitrary.

        A contrast's index is ``"always vs never, death @ t=2"``.  Splitting on ``" vs "``
        first would leave ``"never, death @ t=2"`` as the reference label; splitting the
        cause off before the horizon would leave ``"death @ t=2"`` as the cause.  Both
        produce a lookup failure rather than a wrong number, but only because the tables
        happen not to contain the mangled key -- so the order is pinned here.
        """
        assert law._split_horizon("always vs never, death @ t=2") == ("always vs never, death", 2)
        assert law._split_cause("always vs never, death") == ("always vs never", "death")
        assert law._split_cause("always, relapse") == ("always", "relapse")


# --------------------------------------------------------------------------- the fit

#: Truncation wide enough never to bind: the law's conditionals all lie in [0.25, 0.75].
NO_TRUNCATION = (1e-8, 1.0 - 1e-8)

COLUMNS = {
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}


def _oracle_fit(frame: object, **overrides: object) -> object:
    """A fit of ``frame`` with the saturated learner at every node."""
    settings: dict[str, object] = {
        "reference": law.REGIMEN_REFERENCE,
        "outcome_learner": law.CellMeans(),
        "pseudo_learner": law.CellMeans(),
        "treatment_learner": law.CellMeans(),
        "censoring_learner": law.CellMeans(),
        "n_folds": 1,
        "g_bounds": NO_TRUNCATION,
        # Nothing here reads the bands, and three regimens over two causes and two
        # horizons make twenty parameters -- a multiplier bootstrap over a matrix nobody
        # looks at.
        "simultaneous": False,
    }
    columns = dict(COLUMNS)
    outcome = overrides.pop("outcome", law.outcome_columns())
    for key in ("censoring",):
        if key in overrides:
            columns[key] = overrides.pop(key)  # type: ignore[assignment]
    settings.update(overrides)
    regimens = settings.pop("regimens", law.REGIMEN_SPEC)
    return LTMLE(regimens, **settings).fit(frame, outcome=outcome, **columns)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def fit() -> object:
    """One fit of the exact law, shared by every test below."""
    return _oracle_fit(law.frame())


class TestTheFitAnswersTheLaw:
    """The central claim: the estimator and the oracle agree, to machine precision."""

    def test_every_reported_parameter_has_an_oracle_and_no_more(self, fit: object) -> None:
        """The bidirectional gate, now over three indexes rather than two.

        A competing-risks fit reports a parameter per regimen per cause per horizon, so
        the count moves with three things and the reverse direction earns its keep:
        adding a cause or a horizon to what ``_estimates`` reports fails here until a
        longhand functional exists for it.
        """
        assert set(fit) == set(law.NAMES)

    @pytest.mark.parametrize("name", law.NAMES)
    def test_point_estimate_is_the_g_formula(self, fit: object, name: str) -> None:
        """With exact nuisances the estimate is the truth, to the last bit."""
        assert fit.psi(name) == pytest.approx(law.TRUTH[name], abs=1e-12)

    @pytest.mark.parametrize("name", law.NAMES)
    def test_influence_curve_is_the_gateaux_derivative(self, fit: object, name: str) -> None:
        """The reported curve is the complex-step derivative of the longhand functional.

        ``rtol=0`` as in every sibling module: these curves reach order 20, so a default
        relative tolerance would quietly loosen this to ~1e-6 -- six orders short of what
        the comparison actually holds to, on the module's central claim.
        """
        reported = fit.influence_curves[name][law.first_row_of()]
        np.testing.assert_allclose(reported, law.eif(name), atol=1e-14, rtol=0)

    def test_targeting_had_nothing_to_do(self, fit: object) -> None:
        """An exact initial fit already solves every score equation, so ``epsilon`` is zero.

        There are ``R * J * T(T+1)/2`` of them here -- eighteen -- because every cause is
        its own backward pass over every horizon, sharing the mechanism and nothing else.
        """
        steps = [step for regimen_fit in fit.fits.values() for step in regimen_fit.steps]
        assert len(steps) == len(law.REGIMEN_ARMS) * len(law.CAUSES) * 3
        for step in steps:
            assert step.fluctuation.converged
            assert abs(float(step.fluctuation.epsilon[0])) < 1e-8
            np.testing.assert_allclose(step.targeted, step.initial, atol=1e-9, rtol=0)

    def test_the_curve_is_a_function_of_the_support_point_alone(self, fit: object) -> None:
        """Two rows with the same history carry the same influence-curve value."""
        starts = law.first_row_of()
        for name in law.NAMES[:4]:
            curve = fit.influence_curves[name]
            for start, count in zip(starts, law.COUNTS, strict=True):
                block = curve[start : start + count]
                np.testing.assert_allclose(block, block[0], atol=1e-12, rtol=0)

    def test_the_rule_matches_the_constant_it_equals_at_the_first_horizon(
        self, fit: object
    ) -> None:
        """``continue_if_l2`` treats at the first node, so at ``t = 1`` it *is* ``always``.

        Bit for bit, and at every cause: the horizon-1 pass sees the same arms, so the
        same masks, the same design and the same fluctuation.  It fails if the cause or
        the horizon ever leaks into a mask or a design it has no business in.
        """
        for cause in law.CAUSES:
            rule = fit.fits[f"continue_if_l2, {cause} @ t=1"]
            constant = fit.fits[f"always, {cause} @ t=1"]
            assert rule.psi_scaled == constant.psi_scaled
            np.testing.assert_array_equal(
                rule.influence_curve_scaled, constant.influence_curve_scaled
            )

    def test_the_report_carries_the_cause_as_a_column(self, fit: object) -> None:
        """``curve()`` reads the composed index rather than splitting the name."""
        curve = fit.curve()
        assert "cause" in curve.columns
        assert set(curve["cause"]) == set(law.CAUSES)
        # The ``regimen`` column carries a regimen for a level and the contrast string for
        # a difference, exactly as it does on a survival fit; what is pinned here is that
        # a regimen label comes back whole rather than cut at the cause's separator.
        assert set(law.REGIMEN_ARMS) <= set(curve["regimen"])
        assert {"always vs never", "continue_if_l2 vs never"} <= set(curve["regimen"])
        assert "cause" in fit.diagnostics().columns

    def test_the_incidences_sum_to_the_truth_here(self, fit: object) -> None:
        """With exact nuisances the causes *do* exhaust the mass, so the excess is zero.

        The identity is not enforced anywhere, which is why this is worth asserting: it
        comes out right because every cause's regression is right, and
        ``incidence_total()`` is there for the fits where they are not.
        """
        total = fit.incidence_total()
        assert list(total["excess"]) == [0.0] * len(total["excess"])
        for label, horizon, value in zip(
            total["regimen"], total["time"], total["total"], strict=True
        ):
            expected = sum(_cif(label, cause, int(horizon)) for cause in law.CAUSES)
            assert float(value) == pytest.approx(expected, abs=1e-12)


class TestTheControlsBite:
    """Ways of getting competing risks wrong, each shown to move the answer.

    Both leave the fit convergent with every score at machine zero, so neither is caught
    by anything else here.  They are checked at four orders of magnitude past the window
    the real assertions use.
    """

    def test_censoring_at_the_competing_event_would_be_wrong(self) -> None:
        """Recoding death as censoring answers the *other* competing-risks question.

        Censoring a unit at its competing event is the estimator for the incidence of
        relapse in a world where death has been eliminated -- a controlled direct effect
        with its own identification, needing no-unmeasured-confounding and positivity for
        the competing event itself.  It is a different parameter, and it is refused by
        name rather than offered as a setting.  Here it is shown to be a different
        *number*, which is what makes the refusal worth having: the two would otherwise
        differ only in what the docstring claimed.
        """
        frame = law.frame()
        recoded = frame.copy()
        # Leaving at the competing event, rather than passing through it.
        recoded["C1"] = np.where(frame["D1"] == 1.0, 0.0, frame["C1"])
        for column in ("L2", "A2", "C2", "R2", "D2"):
            recoded[column] = np.where(frame["D1"] == 1.0, np.nan, frame[column])

        eliminated = _oracle_fit(
            recoded,
            outcome=["R1", "R2"],
            regimens={"always": 1},
            reference="always",
        )
        for horizon in law.HORIZONS:
            reported = eliminated.psi(f"risk_regimen[always @ t={horizon}]")
            truth = _cif("always", "relapse", horizon)
            assert abs(reported - truth) > 1e-2, (
                f"censoring at the competing event reproduced the incidence at t={horizon}, "
                "so this law cannot tell the two estimands apart"
            )

    def test_dropping_the_censoring_factor_would_be_wrong(self) -> None:
        """Treating the uncensored as the whole sample misses the truth.

        The law's censoring depends on the history, so the complete cases are not a
        random subsample.  Censoring now interleaves with two event nodes rather than
        one, which is why it is checked here as well as in the survival module.
        """
        frame = law.frame()
        complete = frame[
            (frame["C1"] == 1.0)
            & ((frame["R1"] == 1.0) | (frame["D1"] == 1.0) | (frame["C2"] == 1.0))
        ]
        naive = _oracle_fit(
            complete.reset_index(drop=True),
            censoring=None,
            regimens={"always": 1},
            reference="always",
        )
        gaps = [
            abs(
                naive.psi(f"cif_regimen[always, {cause} @ t={horizon}]")
                - _cif("always", cause, horizon)
            )
            for cause in law.CAUSES
            for horizon in law.HORIZONS
        ]
        assert max(gaps) > 1e-2, "dropping the censoring factors left every incidence intact"
