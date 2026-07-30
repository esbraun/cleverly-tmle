"""The target registry: its invariants, and the evidence it demands.

The registry makes adding an estimand cheap.  These tests make sure it does not also
make it *careless*.  The one that matters most is
:func:`test_every_target_has_an_oracle`: this package's evidence that an influence
curve is correct is that it agrees, to ~1e-12, with one obtained by complex-step
differentiation of an independently written functional on an exactly representable
discrete law.  A target with no branch in ``tests.discrete_law.functional`` has no
such evidence, and would ship on the strength of its author's arithmetic alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleverly.fluctuation.submodel import (
    SUBMODEL_BUILDERS,
    Submodel,
    register_submodel,
)
from cleverly.inference.influence import ParameterEstimate
from cleverly.targets import (
    TARGETS,
    Identification,
    Target,
    all_names,
    default_names,
    groups_for,
    register,
    resolve_estimands,
    targets_for,
)
from tests import discrete_law as law
from tests import discrete_law_shift as shift_law

#: Every law the coverage gate answers to, and together they must cover the registry
#: exactly.  A tuple rather than one law because no single discrete law can express
#: every estimand: the arm- and regime-indexed ones need a treatment with *arms*, and a
#: shift needs one with *ordered doses* it can be moved along.  Widening ``discrete_law``
#: to four doses would change every truth it already pins, which is the thing an oracle
#: exists not to do.
LAWS = (law, shift_law)

#: Read off the submodel registry rather than written down, so a fluctuation added there
#: does not need this list edited too -- which is the whole point of the registry.
VALID_GROUPS = set(SUBMODEL_BUILDERS)
VALID_SCALES = {"level", "difference", "ratio"}


def oracle_for(reported: str):  # type: ignore[no-untyped-def]
    """The law whose ``functional`` has a branch for this parameter name, or ``None``."""
    for candidate in LAWS:
        try:
            candidate.functional(candidate.PROBS, reported)
        except (ValueError, KeyError):
            continue
        return candidate
    return None


def reported_names(target: str) -> tuple[str, ...]:
    """Every parameter name any law says ``target`` reports."""
    return tuple(name for candidate in LAWS for name in candidate.oracle_names(target))


def truth_for(reported: str) -> float:
    """The population value, from whichever law owns this parameter."""
    for candidate in LAWS:
        if reported in candidate.TRUTH:
            return float(candidate.TRUTH[reported])
    raise KeyError(f"no oracle law declares a truth for {reported!r}")


class TestOracleCoverage:
    def test_every_target_has_an_oracle(self) -> None:
        """A registered estimand must be checkable against one of the discrete laws.

        Walks the *parameter* names each target reports rather than the target names,
        because a target reporting one number per arm reports several -- and each of them
        needs its own independently written functional, not just the first.
        """
        missing = []
        for name in TARGETS:
            names = reported_names(name)
            if not names:
                # No law claims this target at all, which is the failure this gate is
                # for: the name it reports is not even known to an oracle.
                missing.append(name)
            missing.extend(reported for reported in names if oracle_for(reported) is None)
        assert not missing, (
            f"targets {missing} are registered but have no branch in the `functional` of "
            "any oracle law (tests.discrete_law for arms and regimes, "
            "tests.discrete_law_shift for shifts), so their influence curve is not "
            "checked against a numerically differentiated one. Add the functional "
            "longhand there (sharing no code with src/) before registering the target."
        )

    def test_the_oracle_covers_no_more_than_the_registry(self) -> None:
        """The reverse direction: an oracle branch with no target is dead code.

        Unioned over the laws, so registering ``ey_shift`` without ``ate_shift`` fails
        here even though the forward direction passes -- the shift law declares a truth
        for both.
        """
        reported = {name for target in TARGETS for name in reported_names(target)}
        declared = {name for candidate in LAWS for name in candidate.TRUTH}
        assert declared == reported

    def test_no_two_laws_claim_the_same_parameter(self) -> None:
        """The union is only well defined if the laws partition the estimands.

        Two laws answering to one name would let ``truth_for`` pick either, and a test
        comparing an estimate against "the" truth would silently depend on law order.
        """
        seen: dict[str, str] = {}
        for candidate in LAWS:
            for name in candidate.TRUTH:
                assert name not in seen, f"{name!r} is claimed by both {seen[name]} and {candidate}"
                seen[name] = str(candidate)


class TestRegistryInvariants:
    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_declares_a_known_group_and_scale(self, name: str) -> None:
        target = TARGETS[name]
        assert target.name == name, "registry key must match the target's own name"
        assert target.group in VALID_GROUPS
        assert target.scale in VALID_SCALES

    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_ratios_require_a_binary_outcome(self, name: str) -> None:
        """A ratio of two means only makes sense when the means are probabilities."""
        target = TARGETS[name]
        if target.scale == "ratio":
            assert target.requires_family == "binomial"

    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_identification_is_declared(self, name: str) -> None:
        ident = TARGETS[name].identification
        assert isinstance(ident, Identification)
        assert ident.assumptions, "a target must say what it rests on"
        assert ident.required_nuisances
        assert ident.dr_condition

    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_bounded_targets_say_when_they_are_undefined(self, name: str) -> None:
        """Anything with a restricted range must explain the restriction.

        ``undefined_when`` is what licenses a fold to drop the estimand instead of
        aborting, so a bounded target without it would abort the whole fold.
        """
        target = TARGETS[name]
        if target.parameter_bounds is not None:
            assert target.undefined_when


class TestResolution:
    def test_default_report_matches_the_family(self) -> None:
        assert set(default_names("gaussian")) == {"ate", "att", "atc", "ey1", "ey0"}
        # A binary outcome adds every ratio it supports. `ey` is the one registered
        # target left out: it reports a mean per arm, which on two arms is `ey1` and
        # `ey0` under clumsier names, so it joins the default report only when there
        # are more arms than those two can name.
        assert set(default_names("binomial")) == set(all_names("binomial")) - {"ey"}

    def test_the_default_report_follows_the_arm_count(self) -> None:
        """Three arms drop what names two of them, and gain the per-arm mean."""
        two = set(default_names("gaussian", 2))
        three = set(default_names("gaussian", 3))
        assert {"att", "atc", "ey1", "ey0"} <= two
        assert three.isdisjoint({"att", "atc", "ey1", "ey0"})
        assert "ey" in three and "ey" not in two
        assert "ate" in two and "ate" in three

    def test_a_binary_only_estimand_is_refused_on_a_multi_arm_fit(self) -> None:
        with pytest.raises(ValueError, match="binary treatment only"):
            resolve_estimands(["ate", "att"], "gaussian", 3)

    def test_ratios_are_refused_for_a_continuous_outcome(self) -> None:
        with pytest.raises(ValueError, match="binomial"):
            resolve_estimands(["ate", "rr"], "gaussian")

    def test_unknown_estimand_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown estimand"):
            resolve_estimands(["nope"], "binomial")

    def test_report_order_is_registry_order_not_request_order(self) -> None:
        assert resolve_estimands(["ey0", "ate", "att"], "gaussian") == ("ate", "att", "ey0")

    def test_all_expands_by_family(self) -> None:
        assert "rr" in resolve_estimands("all", "binomial")
        assert "rr" not in resolve_estimands("all", "gaussian")

    def test_empty_request_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no estimands"):
            resolve_estimands([], "binomial")


class TestGroups:
    def test_one_fluctuation_serves_the_whole_mean_family(self) -> None:
        """The five mean-group estimands must not fit five fluctuations."""
        assert groups_for(["ate", "ey1", "ey0", "rr", "or"]) == ["mean"]

    def test_conditional_estimands_need_their_own_fluctuation(self) -> None:
        assert groups_for(["ate", "att", "atc"]) == ["mean", "att", "atc"]

    def test_targets_for_filters_by_group_and_request(self) -> None:
        names = [t.name for t in targets_for("mean", ["ate", "ey1", "att"])]
        assert names == ["ate", "ey1"]


class TestRegistration:
    def test_duplicate_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register(TARGETS["ate"])

    def test_a_new_target_needs_no_estimator_changes(self) -> None:
        """The point of the registry: one object, no edits elsewhere.

        Registers a target, checks it flows through resolution and grouping, then
        removes it again so the rest of the suite sees the stock registry.
        """
        built: dict[str, ParameterEstimate] = {}

        def build(ctx):  # type: ignore[no-untyped-def]
            one, zero = ctx.means[1.0], ctx.means[0.0]
            # Half the ATE, purely to exercise the plumbing.
            diff = one.psi - zero.psi
            ic = one.influence_curve - zero.influence_curve
            est = ctx.finish("half_ate", 0.5 * diff, 0.5 * ic, "difference")
            built["half_ate"] = est
            return [est]

        target = Target(
            name="half_ate",
            group="mean",
            scale="difference",
            build=build,
            identification=Identification(
                assumptions=("whatever the ATE assumes",),
                required_nuisances=("outcome_regression", "treatment_mechanism"),
                dr_condition="as the ATE",
            ),
        )
        register(target)
        try:
            assert "half_ate" in resolve_estimands(["ate", "half_ate"], "gaussian")
            assert groups_for(["half_ate"]) == ["mean"]
            assert [t.name for t in targets_for("mean", ["half_ate"])] == ["half_ate"]
        finally:
            del TARGETS["half_ate"]
        assert "half_ate" not in TARGETS


class TestGroupRegistry:
    """A target's group has to name a fluctuation that exists.

    Before the submodel registry, ``TargetGroup`` was a ``Literal`` and this was a static
    check: mypy refused an unknown group and the runtime never had to. A registry the
    caller can extend cannot have an exhaustive ``Literal``, so the check moved to
    registration time -- earlier than the fit, which is where the old code would have
    discovered it.
    """

    def test_every_registered_target_names_a_registered_fluctuation(self) -> None:
        for name, target in TARGETS.items():
            assert target.group in SUBMODEL_BUILDERS, (
                f"target {name!r} declares group {target.group!r}, which no submodel "
                "builder provides, so its score equation cannot be solved"
            )

    def test_registering_a_target_with_an_unknown_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no submodel builder"):
            register(
                Target(
                    name="groupless",
                    group="no_such_fluctuation",
                    scale="difference",
                    build=lambda ctx: None,  # type: ignore[arg-type,return-value]
                    identification=Identification(
                        assumptions=("none",),
                        required_nuisances=("outcome_regression",),
                        dr_condition="none",
                    ),
                )
            )
        assert "groupless" not in TARGETS

    def test_a_duplicate_submodel_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_submodel("mean", SUBMODEL_BUILDERS["mean"])

    def test_replace_is_honoured_when_asked_for_explicitly(self) -> None:
        original = SUBMODEL_BUILDERS["atc"]
        try:
            register_submodel("atc", original, replace=True)
            assert SUBMODEL_BUILDERS["atc"] is original
        finally:
            SUBMODEL_BUILDERS["atc"] = original

    def test_a_builder_must_label_its_submodel_with_its_own_group(self) -> None:
        """A mismatch would silently route the wrong influence curve at the wrong estimand."""
        from cleverly.fluctuation.submodel import mean_submodel, submodel_for

        register_submodel("mislabelled", mean_submodel)
        try:
            with pytest.raises(ValueError, match="the two must agree"):
                submodel_for("mislabelled", np.array([0.0, 1.0]), np.array([0.4, 0.6]))
        finally:
            del SUBMODEL_BUILDERS["mislabelled"]


class TestACustomFluctuation:
    """Registering a new score equation, end to end, without touching the estimator.

    The built-in ``mean`` fluctuation fits two columns whose supports are disjoint, one
    per arm, so its Hessian is diagonal and the coefficient on the treated column is the
    same one a single-column fluctuation targeting only ``E[Y(1)]`` would find.  That is
    what makes this a real check rather than a smoke test: the custom group must
    reproduce the built-in ``ey1`` to solver precision, and it can only do so if the
    registry, the clever covariate, the influence curve and the plug-in all lined up.
    """

    GROUP = "treated_only"
    NAME = "ey1_solo"

    @staticmethod
    def _builder(
        treatment,
        propensity,
        *,
        arms=(0.0, 1.0),
        treated_fraction=None,
        missingness=None,
        intermediate_density=None,
        selection=None,
        regimes=None,
        shifts=None,
        msm=None,
    ):  # type: ignore[no-untyped-def]
        """One column: ``1{A = 1} / g_1(W)``, the Riesz representer of ``E[Y(1)]``."""
        a = np.asarray(treatment, dtype=float).reshape(-1)
        g = np.asarray(propensity, dtype=float)
        g1 = g.reshape(-1) if g.ndim == 1 else g[:, arms.index(1.0)]
        n = a.shape[0]
        inverse = (1.0 / g1).reshape(-1, 1)
        return Submodel(
            (a.reshape(-1, 1) * inverse),
            {1.0: inverse, 0.0: np.zeros((n, 1))},
            ("h1",),
            TestACustomFluctuation.GROUP,
            {1.0: 0},
        )

    @staticmethod
    def _build(ctx):  # type: ignore[no-untyped-def]
        residual = np.where(ctx.observed, ctx.scaled - ctx.targeted.observed, 0.0)
        psi = float(np.average(ctx.targeted.arms[1.0], weights=ctx.weights))
        curve = ctx.weights * (
            ctx.submodel.column_for(1.0) * residual + ctx.targeted.arms[1.0] - psi
        )
        return [ctx.finish(TestACustomFluctuation.NAME, psi, curve, "level")]

    @pytest.fixture
    def registered(self):  # type: ignore[no-untyped-def]
        register_submodel(self.GROUP, self._builder)
        register(
            Target(
                name=self.NAME,
                group=self.GROUP,
                scale="level",
                build=self._build,
                identification=Identification(
                    assumptions=("consistency", "no unmeasured confounding given W", "positivity"),
                    required_nuisances=("outcome_regression", "treatment_mechanism"),
                    dr_condition="consistent if either Qbar or g is consistent",
                ),
            )
        )
        try:
            yield
        finally:
            # Popped so the oracle-coverage test above, which iterates the whole
            # registry and demands a discrete-law branch per target, still holds.
            del TARGETS[self.NAME]
            del SUBMODEL_BUILDERS[self.GROUP]

    def test_it_reaches_the_estimator_and_agrees_with_the_built_in(self, registered) -> None:
        from cleverly.datasets import make_linear_ate
        from tests.conftest import fast_tmle

        frame, _ = make_linear_ate(n=400, seed=3)
        result = (
            fast_tmle(estimands=["ey1", self.NAME]).fit(frame, outcome="Y", treatment="A").single()
        )
        assert self.GROUP in result.fluctuations
        assert result.fluctuations[self.GROUP].converged
        assert result.psi(self.NAME) == pytest.approx(result.psi("ey1"), abs=1e-8)
        assert result[self.NAME].std_error == pytest.approx(result["ey1"].std_error, rel=1e-6)

    def test_the_new_group_is_grouped_and_resolved_like_any_other(self, registered) -> None:
        assert self.GROUP in groups_for([self.NAME])
        assert groups_for(["ate", self.NAME]) == ["mean", self.GROUP]
        assert [t.name for t in targets_for(self.GROUP, [self.NAME])] == [self.NAME]


class TestAgainstTheOracle:
    """The registry must not have changed what the estimands mean."""

    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_truth_is_finite_and_matches_the_declared_scale(self, name: str) -> None:
        target = TARGETS[name]
        for reported in reported_names(name):
            self._check(truth_for(reported), target)

    @staticmethod
    def _check(value: float, target) -> None:  # type: ignore[no-untyped-def]
        assert np.isfinite(value)
        if target.scale == "level" and target.parameter_axis != "msm":
            # A counterfactual mean of a binary outcome is a probability. The exception is
            # not a loosening: `msm` reports the coefficients of a working model, and
            # `scale="level"` there says only that inference is a Wald interval on the
            # coefficient itself with no log transform. A slope is not a mean of anything
            # and has no reason to sit in [0, 1] -- it happens to on this law, which is
            # exactly why the assertion would have passed while meaning nothing.
            assert 0.0 <= value <= 1.0
        if target.parameter_bounds is not None:
            # Ratios are held on the log scale by the oracle, so exponentiate first.
            lower, upper = target.parameter_bounds
            assert lower <= float(np.exp(value)) <= upper


class TestTheScalingContract:
    """``finish`` unscales linearly, which is exact only for a linear functional.

    Discovered while writing the README's custom-target example: a number-needed-to-
    treat target computed from the scaled means is right on a binary outcome and wrong
    on a scaled continuous one, because ``1 / (range * x)`` is not ``range * (1 / x)``.
    ``requires_family="binomial"`` is what makes such a target safe, and this pins that
    it actually does so.
    """

    @staticmethod
    def _nnt(ctx):  # type: ignore[no-untyped-def]
        one, zero = ctx.means[1.0], ctx.means[0.0]
        difference = one.psi - zero.psi
        ic = one.influence_curve - zero.influence_curve
        return [ctx.finish("nnt", 1.0 / difference, -ic / difference**2, "difference")]

    def _register(self, **kwargs):  # type: ignore[no-untyped-def]
        return register(
            Target(
                name="nnt",
                group="mean",
                scale="difference",
                build=self._nnt,
                identification=Identification(
                    assumptions=("consistency",),
                    required_nuisances=("outcome_regression", "treatment_mechanism"),
                    dr_condition="as the ATE",
                ),
                **kwargs,
            )
        )

    def test_a_binary_only_target_is_exact(self) -> None:
        from cleverly import TMLE
        from cleverly.datasets import GENERATORS

        self._register(requires_family="binomial")
        try:
            frame, _ = GENERATORS["binary_outcome"](n=300, seed=1)
            covariates = [c for c in frame.columns if c.startswith("W")]
            result = (
                TMLE(
                    outcome_learner="glm",
                    treatment_learner="glm",
                    n_folds=4,
                    random_state=7,
                    estimands=["ate", "nnt"],
                )
                .fit(frame, outcome="Y", treatment="A", covariates=covariates)
                .single()
            )
            assert result.nuisance.scaler.is_identity
            assert result["nnt"].psi == pytest.approx(1.0 / result["ate"].psi, rel=1e-12)
        finally:
            del TARGETS["nnt"]

    def test_the_same_target_on_a_scaled_outcome_would_be_wrong(self) -> None:
        """The failure the ``requires_family`` guard exists to prevent."""
        from cleverly import TMLE
        from cleverly.datasets import GENERATORS

        self._register()  # no requires_family: allowed onto a continuous outcome
        try:
            frame, _ = GENERATORS["linear_ate"](n=300, seed=1)
            covariates = [c for c in frame.columns if c.startswith("W")]
            result = (
                TMLE(
                    outcome_learner="glm",
                    treatment_learner="glm",
                    n_folds=4,
                    random_state=7,
                    estimands=["ate", "nnt"],
                )
                .fit(frame, outcome="Y", treatment="A", covariates=covariates)
                .single()
            )
            assert not result.nuisance.scaler.is_identity
            # Off by range**2, because the reciprocal does not commute with unscaling.
            assert result["nnt"].psi != pytest.approx(1.0 / result["ate"].psi, rel=1e-3)
        finally:
            del TARGETS["nnt"]
