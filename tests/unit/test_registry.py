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

VALID_GROUPS = {"mean", "att", "atc"}
VALID_SCALES = {"level", "difference", "ratio"}


class TestOracleCoverage:
    def test_every_target_has_an_oracle(self) -> None:
        """A registered estimand must be checkable against the discrete law."""
        missing = []
        for name in TARGETS:
            try:
                law.functional(law.PROBS, name)
            except ValueError:
                missing.append(name)
        assert not missing, (
            f"targets {missing} are registered but have no branch in "
            "tests.discrete_law.functional, so their influence curve is not checked "
            "against a numerically differentiated one. Add the functional longhand "
            "there (sharing no code with src/) before registering the target."
        )

    def test_the_oracle_covers_no_more_than_the_registry(self) -> None:
        """The reverse direction: an oracle branch with no target is dead code."""
        assert set(law.TRUTH) == set(TARGETS)


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
        assert set(default_names("binomial")) == set(all_names("binomial"))

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
            psi_one, ic_one, psi_zero, ic_zero = ctx.means
            # Number needed to treat is 1 / ATE; use the reciprocal on the difference
            # scale purely to exercise the plumbing.
            diff = psi_one - psi_zero
            est = ctx.finish("half_ate", 0.5 * diff, 0.5 * (ic_one - ic_zero), "difference")
            built["half_ate"] = est
            return est

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


class TestAgainstTheOracle:
    """The registry must not have changed what the estimands mean."""

    @pytest.mark.parametrize("name", sorted(TARGETS))
    def test_truth_is_finite_and_matches_the_declared_scale(self, name: str) -> None:
        value = law.TRUTH[name]
        assert np.isfinite(value)
        target = TARGETS[name]
        if target.scale == "level":
            # A counterfactual mean of a binary outcome is a probability.
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
        psi_one, ic_one, psi_zero, ic_zero = ctx.means
        difference = psi_one - psi_zero
        return ctx.finish(
            "nnt", 1.0 / difference, -(ic_one - ic_zero) / difference**2, "difference"
        )

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
            result = TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=4,
                random_state=7,
                estimands=["ate", "nnt"],
            ).fit(frame, outcome="Y", treatment="A", covariates=covariates)
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
            result = TMLE(
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=4,
                random_state=7,
                estimands=["ate", "nnt"],
            ).fit(frame, outcome="Y", treatment="A", covariates=covariates)
            assert not result.nuisance.scaler.is_identity
            # Off by range**2, because the reciprocal does not commute with unscaling.
            assert result["nnt"].psi != pytest.approx(1.0 / result["ate"].psi, rel=1e-3)
        finally:
            del TARGETS["nnt"]
