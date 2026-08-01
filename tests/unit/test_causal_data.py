"""The ``CausalData`` container: validation, encoding, and backend round-trips."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cleverly.data import CausalData
from cleverly.data.validate import encode_binary, encode_treatment
from cleverly.exceptions import DataError, DataWarning, WeightingWarning


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "W1": rng.normal(size=n),
            "W2": rng.integers(0, 5, n).astype(float),
            "A": rng.binomial(1, 0.5, n).astype(float),
            "Y": rng.normal(size=n),
        }
    )


class TestConstruction:
    def test_unclaimed_columns_become_covariates(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        assert data.covariate_names == ("W1", "W2")

    def test_explicit_covariates_are_respected(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A", covariates=["W1"])
        assert data.covariate_names == ("W1",)

    def test_from_arrays_names_covariates(self) -> None:
        frame = _frame()
        data = CausalData.from_arrays(
            frame["Y"].to_numpy(), frame["A"].to_numpy(), frame[["W1", "W2"]].to_numpy()
        )
        assert data.covariate_names == ("W1", "W2")
        assert data.n == len(frame)

    def test_rejects_a_role_used_as_a_covariate(self) -> None:
        with pytest.raises(DataError, match="both as a role and a covariate"):
            CausalData.from_frame(_frame(), outcome="Y", treatment="A", covariates=["W1", "A"])

    def test_rejects_missing_columns(self) -> None:
        with pytest.raises(DataError, match="columns not found"):
            CausalData.from_frame(_frame(), outcome="Y", treatment="nope")

    def test_rejects_numpy_input(self) -> None:
        with pytest.raises(DataError, match="from_arrays"):
            CausalData.from_frame(np.zeros((20, 3)), outcome="Y", treatment="A")

    def test_rejects_too_few_observations(self) -> None:
        with pytest.raises(DataError, match="at least 10 observations"):
            CausalData.from_frame(_frame(n=5), outcome="Y", treatment="A")


class TestTreatmentEncoding:
    def test_two_level_strings_are_encoded_with_a_recorded_order(self) -> None:
        frame = _frame()
        frame["A"] = np.where(frame["A"] == 1.0, "treated", "control")
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        # Sorted levels: "control" -> 0, "treated" -> 1. Recorded, not guessed.
        assert data.treatment_levels == ("control", "treated")
        assert set(np.unique(data.treatment)) == {0.0, 1.0}

    def test_two_level_numeric_maps_the_larger_value_to_one(self) -> None:
        frame = _frame()
        frame["A"] = np.where(frame["A"] == 1.0, 5.0, 2.0)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        assert data.treatment_levels == (2.0, 5.0)

    def test_a_multi_valued_treatment_records_every_level(self) -> None:
        frame = _frame()
        frame["A"] = np.repeat([0.0, 1.0, 2.0, 3.0], len(frame) // 4)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        assert data.treatment_levels == (0.0, 1.0, 2.0, 3.0)
        assert data.n_arms == 4
        assert data.arm_codes == (0.0, 1.0, 2.0, 3.0)
        assert not data.is_binary_treatment

    def test_multi_valued_labels_survive_encoding(self) -> None:
        frame = _frame()
        frame["A"] = np.resize(["low", "med", "high"], len(frame))
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        # Levels sort in their natural order, which for strings is alphabetical -- the
        # codes are an implementation detail and the labels are what gets reported.
        assert data.treatment_levels == ("high", "low", "med")
        assert [data.arm_label(c) for c in data.arm_codes] == ["high", "low", "med"]

    def test_rejects_a_single_armed_treatment(self) -> None:
        frame = _frame()
        frame["A"] = 1.0
        with pytest.raises(DataError, match="takes only one value"):
            CausalData.from_frame(frame, outcome="Y", treatment="A")

    def test_rejects_more_arms_than_the_limit(self) -> None:
        frame = _frame(n=200)
        frame["A"] = np.repeat(np.arange(25.0), len(frame) // 25)
        with pytest.raises(DataError, match="above the limit of 20"):
            CausalData.from_frame(frame, outcome="Y", treatment="A")

    def test_encode_treatment_reproduces_encode_binary_exactly(self) -> None:
        """The two-arm path must be byte-identical, not merely equivalent.

        Every regression fixture in the suite depends on a binary fit producing the
        numbers it always did, and the encoder is the first place that could drift.
        """
        for values in (
            np.array([0.0, 1.0] * 10),
            np.array([0, 1] * 10),
            np.array([2.0, 5.0] * 10),
            np.array([-1.0, 1.0] * 10),
            np.array(["ctl", "trt"] * 10),
        ):
            binary_codes, binary_levels = encode_binary(values, "A")
            multi_codes, multi_levels = encode_treatment(values, "A")
            assert binary_codes.tobytes() == multi_codes.tobytes()
            assert tuple(binary_levels) == tuple(multi_levels)

    def test_a_thin_arm_is_refused_when_a_minimum_is_asked_for(self) -> None:
        values = np.array([0.0] * 20 + [1.0] * 20 + [2.0])
        with pytest.raises(DataError, match="too few observations"):
            encode_treatment(values, "A", min_per_arm=5)


class TestCovariateHandling:
    def test_categorical_columns_are_one_hot_with_a_dropped_level(self) -> None:
        frame = _frame()
        frame["G"] = np.resize(["a", "b", "c"], len(frame))
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        assert data.covariate_names == ("W1", "W2", "G__b", "G__c")
        encoding = data.encodings[0]
        # Dropping one level keeps a linear design matrix full rank.
        assert encoding.dropped_level == "a"
        assert encoding.generated == ("G__b", "G__c")

    def test_boolean_columns_pass_through_as_indicators(self) -> None:
        frame = _frame()
        frame["flag"] = frame["W1"] > 0
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        assert "flag" in data.covariate_names

    def test_constant_covariates_are_dropped_with_a_warning(self) -> None:
        frame = _frame()
        frame["const"] = 3.0
        with pytest.warns(UserWarning, match="constant or duplicated"):
            data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        assert "const" not in data.covariate_names
        assert data.dropped_covariates == ("const",)

    def test_duplicated_covariates_are_dropped(self) -> None:
        frame = _frame()
        frame["copy"] = frame["W1"]
        with pytest.warns(UserWarning, match="constant or duplicated"):
            data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        assert "copy" not in data.covariate_names

    def test_missing_covariates_are_an_error_not_an_imputation(self) -> None:
        frame = _frame()
        frame.loc[frame.index[:5], "W1"] = np.nan
        with pytest.raises(DataError, match="missing or non-finite"):
            CausalData.from_frame(frame, outcome="Y", treatment="A")

    def test_high_cardinality_categoricals_are_refused(self) -> None:
        frame = _frame(n=200)
        frame["id_like"] = [f"v{i}" for i in range(len(frame))]
        with pytest.raises(DataError, match="levels; encode it yourself"):
            CausalData.from_frame(frame, outcome="Y", treatment="A")

    def test_requires_at_least_one_covariate(self) -> None:
        frame = _frame()[["A", "Y"]]
        with pytest.raises(DataError, match="no covariate columns"):
            CausalData.from_frame(frame, outcome="Y", treatment="A")


class TestOutcomeAndFamily:
    def test_binary_outcome_is_detected(self) -> None:
        frame = _frame()
        frame["Y"] = (frame["Y"] > 0).astype(float)
        assert CausalData.from_frame(frame, outcome="Y", treatment="A").family == "binomial"

    def test_continuous_outcome_is_detected(self) -> None:
        assert CausalData.from_frame(_frame(), outcome="Y", treatment="A").family == "gaussian"

    def test_declared_binomial_family_is_checked(self) -> None:
        with pytest.raises(DataError, match="requires a 0/1 outcome"):
            CausalData.from_frame(_frame(), outcome="Y", treatment="A", family="binomial")

    def test_missing_outcomes_without_delta_are_refused_with_guidance(self) -> None:
        frame = _frame()
        frame.loc[frame.index[:5], "Y"] = np.nan
        with pytest.raises(DataError, match="Pass delta="):
            CausalData.from_frame(frame, outcome="Y", treatment="A")

    def test_delta_permits_missing_outcomes(self) -> None:
        frame = _frame()
        observed = np.ones(len(frame))
        observed[:20] = 0.0
        frame["Y"] = np.where(observed == 1.0, frame["Y"], np.nan)
        frame["D"] = observed
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", delta="D")
        assert data.has_missing_outcome
        assert int(data.observed.sum()) == len(frame) - 20
        # Unobserved outcomes are neutralised so they cannot propagate NaN.
        assert np.all(np.isfinite(data.outcome))

    def test_delta_must_agree_with_the_outcome(self) -> None:
        frame = _frame()
        frame.loc[frame.index[:5], "Y"] = np.nan
        frame["D"] = 1.0
        with pytest.raises(DataError, match="flagged as observed"):
            CausalData.from_frame(frame, outcome="Y", treatment="A", delta="D")


class TestWeightsAndClusters:
    def test_weights_are_normalised_to_mean_one(self) -> None:
        frame = _frame()
        frame["w"] = np.linspace(0.5, 2.0, len(frame))
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", weights="w")
        assert float(data.weights.mean()) == pytest.approx(1.0)
        assert data.is_weighted

    def test_negative_weights_are_refused(self) -> None:
        frame = _frame()
        frame["w"] = -1.0
        with pytest.raises(DataError, match="negative values"):
            CausalData.from_frame(frame, outcome="Y", treatment="A", weights="w")

    def test_the_weight_spec_records_how_to_read_the_column(self) -> None:
        frame = _frame()
        frame["w"] = np.linspace(0.5, 2.0, len(frame))
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", weights="w", weights_estimated=True
        )
        assert data.weight_spec.kind == "probability"
        assert data.weight_spec.estimated
        assert data.weight_spec.name == "w"
        # The supplied weights are recoverable from the normalised ones and the scale.
        assert data.weight_spec.scale == pytest.approx(float(frame["w"].mean()))
        np.testing.assert_allclose(data.weights * data.weight_spec.scale, frame["w"])

    def test_unweighted_data_still_has_a_spec(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        assert data.weight_spec.name is None
        assert data.effective_n == pytest.approx(data.n)
        assert not data.weight_report().is_weighted

    @pytest.mark.parametrize("alias", ["frequency", "count", "fweight"])
    def test_frequency_weights_are_refused_by_any_name(self, alias: str) -> None:
        frame = _frame()
        frame["w"] = 2.0
        with pytest.raises(DataError, match="not supported"):
            CausalData.from_frame(
                frame, outcome="Y", treatment="A", weights="w", weights_type=alias
            )

    @pytest.mark.parametrize("alias", ["probability", "sampling", "survey", "design", "pweight"])
    def test_probability_synonyms_are_accepted(self, alias: str) -> None:
        frame = _frame()
        frame["w"] = np.linspace(0.5, 2.0, len(frame))
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", weights="w", weights_type=alias
        )
        assert data.weight_spec.kind == "probability"

    def test_an_unknown_weights_type_is_refused(self) -> None:
        with pytest.raises(DataError, match="unknown weights_type"):
            CausalData.from_frame(_frame(), outcome="Y", treatment="A", weights_type="aweight")

    def test_count_looking_weights_warn_but_are_accepted(self) -> None:
        frame = _frame()
        frame["w"] = np.repeat([1.0, 2.0, 3.0, 4.0], len(frame) // 4)
        with pytest.warns(WeightingWarning, match="counts"):
            data = CausalData.from_frame(frame, outcome="Y", treatment="A", weights="w")
        assert data.is_weighted

    def test_non_integer_weights_do_not_warn(self) -> None:
        frame = _frame()
        frame["w"] = np.linspace(0.5, 3.0, len(frame))
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightingWarning)
            CausalData.from_frame(frame, outcome="Y", treatment="A", weights="w")

    def test_clusters_are_recoded_contiguously(self) -> None:
        frame = _frame(n=100)
        frame["cl"] = np.repeat([10, 20, 30, 40], 25)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", id="cl")
        assert data.n_clusters == 4
        assert set(np.unique(data.cluster)) == {0, 1, 2, 3}

    def test_a_single_cluster_is_refused(self) -> None:
        frame = _frame()
        frame["cl"] = 1
        with pytest.raises(DataError, match="single cluster"):
            CausalData.from_frame(frame, outcome="Y", treatment="A", id="cl")

    def test_n_clusters_defaults_to_n(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        assert data.n_clusters == data.n


class TestReshaping:
    def test_subset_renormalises_weights_and_recodes_clusters(self) -> None:
        frame = _frame(n=100)
        frame["cl"] = np.repeat(np.arange(10), 10)
        frame["w"] = np.linspace(0.5, 1.5, 100)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", id="cl", weights="w")
        subset = data.subset(np.arange(40))
        assert subset.n == 40
        assert float(subset.weights.mean()) == pytest.approx(1.0)
        assert subset.n_clusters == 4
        # Renormalising rescales; the spec keeps enough to recover the supplied weights.
        np.testing.assert_allclose(
            subset.weights * subset.weight_spec.scale, frame["w"].to_numpy()[:40]
        )

    def test_subset_accepts_a_boolean_mask(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        mask = np.zeros(data.n, dtype=bool)
        mask[:50] = True
        assert data.subset(mask).n == 50

    def test_with_treatment_replaces_the_arm(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        rng = np.random.default_rng(0)
        permuted = rng.permutation(data.treatment)
        replaced = data.with_treatment(permuted)
        assert np.array_equal(replaced.treatment, permuted)
        assert np.array_equal(replaced.outcome, data.outcome)

    def test_with_extra_covariate_appends_a_column(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        extended = data.with_extra_covariate(np.arange(data.n, dtype=float), "noise")
        assert extended.covariate_names[-1] == "noise"
        assert extended.n_covariates == data.n_covariates + 1

    def test_with_extra_covariate_rejects_a_duplicate_name(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        with pytest.raises(DataError, match="already exists"):
            data.with_extra_covariate(np.zeros(data.n), "W1")

    def test_without_covariates_drops_by_name(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        assert data.without_covariates(["W1"]).covariate_names == ("W2",)

    def test_without_covariates_refuses_to_empty_the_set(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        with pytest.raises(DataError, match="cannot drop every covariate"):
            data.without_covariates(["W1", "W2"])


class TestDesignMatrices:
    def test_treatment_design_puts_treatment_first(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        design = data.treatment_design()
        assert design.shape == (data.n, data.n_covariates + 1)
        assert np.array_equal(design[:, 0], data.treatment)

    def test_counterfactual_design_pins_the_treatment(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        assert np.all(data.counterfactual_design(1.0)[:, 0] == 1.0)
        assert np.all(data.counterfactual_design(0.0)[:, 0] == 0.0)

    def test_intermediate_is_appended_when_requested(self) -> None:
        frame = _frame()
        frame["Z"] = (frame["W1"] > 0).astype(float)
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", intermediate="Z", covariates=["W1", "W2"]
        )
        design = data.treatment_design(include_intermediate=True)
        assert design.shape[1] == 4
        assert np.array_equal(design[:, -1], data.intermediate)

    def test_intermediate_design_requires_an_intermediate(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        with pytest.raises(DataError, match="no intermediate"):
            data.treatment_design(include_intermediate=True)

    def test_a_binary_design_is_byte_identical_to_the_single_column_form(self) -> None:
        """The K-1 indicator block must not perturb the two-arm design at all.

        With two arms the block is one column holding the 0/1 code, so this is the
        assertion that lets every binary regression fixture stand unchanged.
        """
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        legacy = np.hstack([data.treatment.reshape(-1, 1), data.covariates])
        assert data.treatment_design().tobytes() == legacy.tobytes()
        for arm in (0.0, 1.0):
            legacy_arm = np.hstack([np.full((data.n, 1), arm), data.covariates])
            assert data.counterfactual_design(arm).tobytes() == legacy_arm.tobytes()

    def test_a_three_arm_design_uses_drop_first_indicators(self) -> None:
        frame = _frame(n=300)
        frame["A"] = np.resize([0.0, 1.0, 2.0], len(frame))
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        design = data.treatment_design()
        assert design.shape == (data.n, data.n_covariates + 2)
        # The dropped first arm is the all-zero row of the indicator block, and each
        # counterfactual design pins exactly one pattern.
        assert np.array_equal(data.counterfactual_design(0.0)[:, :2], np.zeros((data.n, 2)))
        assert np.array_equal(data.counterfactual_design(1.0)[:, 0], np.ones(data.n))
        assert np.array_equal(data.counterfactual_design(1.0)[:, 1], np.zeros(data.n))
        assert np.array_equal(data.counterfactual_design(2.0)[:, 1], np.ones(data.n))
        # An indicator design leaves the arms unconstrained: no two counterfactual
        # designs coincide, which a single numeric column could not guarantee.
        blocks = {data.counterfactual_design(a)[:, :2].tobytes() for a in data.arm_codes}
        assert len(blocks) == data.n_arms

    def test_counterfactual_design_rejects_an_arm_outside_the_support(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        with pytest.raises(DataError, match="is not an arm of A"):
            data.counterfactual_design(2.0)


class TestBackends:
    def test_polars_and_pandas_produce_identical_arrays(self) -> None:
        frame = _frame()
        from_pandas = CausalData.from_frame(frame, outcome="Y", treatment="A")
        from_polars = CausalData.from_frame(pl.from_pandas(frame), outcome="Y", treatment="A")
        assert np.array_equal(from_pandas.covariates, from_polars.covariates)
        assert np.array_equal(from_pandas.outcome, from_polars.outcome)
        assert from_pandas.covariate_names == from_polars.covariate_names

    def test_results_come_back_in_the_input_backend(self) -> None:
        frame = _frame()
        assert isinstance(
            CausalData.from_frame(frame, outcome="Y", treatment="A").to_frame(),
            pd.DataFrame,
        )
        assert isinstance(
            CausalData.from_frame(pl.from_pandas(frame), outcome="Y", treatment="A").to_frame(),
            pl.DataFrame,
        )

    def test_numpy_input_defaults_to_pandas_output(self) -> None:
        frame = _frame()
        data = CausalData.from_arrays(
            frame["Y"].to_numpy(), frame["A"].to_numpy(), frame[["W1"]].to_numpy()
        )
        assert data.backend is None
        assert isinstance(data.to_frame(), pd.DataFrame)

    def test_round_trip_preserves_every_role(self) -> None:
        frame = _frame(n=100)
        frame["cl"] = np.repeat(np.arange(10), 10)
        frame["w"] = 1.5
        data = CausalData.from_frame(
            frame, outcome="Y", treatment="A", id="cl", weights="w", covariates=["W1"]
        )
        out = data.to_frame()
        assert {"Y", "A", "W1", "w", "cl"} <= set(out.columns)


def _continuous_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "W1": w1,
            "W2": rng.normal(size=n),
            "A": w1 * 0.7 + rng.normal(size=n),
            "Y": rng.normal(size=n),
        }
    )


class TestContinuousTreatment:
    """A treatment declared continuous has no arms, and says so rather than pretending.

    The point of ``n_arms == 0`` is that every arm loop becomes *empty* rather than
    *wrong*; the accessors that name an arm raise instead of answering, so a caller that
    reaches for one gets an error rather than a silently degenerate fit.
    """

    def _data(self, **kwargs: object) -> CausalData:
        return CausalData.from_frame(
            _continuous_frame(), outcome="Y", treatment="A", treatment_kind="continuous", **kwargs
        )

    def test_a_continuous_treatment_has_no_arms(self) -> None:
        data = self._data()
        assert data.is_continuous_treatment
        assert data.n_arms == 0
        assert data.arm_codes == ()
        assert data.treatment_levels == ()
        assert not data.is_binary_treatment

    def test_the_treatment_keeps_its_own_values(self) -> None:
        frame = _continuous_frame()
        data = CausalData.from_frame(frame, outcome="Y", treatment="A", treatment_kind="continuous")
        # Not codes: the numbers a shift moves along have to survive encoding.
        np.testing.assert_array_equal(data.treatment, frame["A"].to_numpy())

    def test_the_design_carries_the_dose_as_one_column(self) -> None:
        data = self._data()
        block = data.treatment_block(data.treatment)
        assert block.shape == (data.n, 1)
        np.testing.assert_array_equal(block[:, 0], data.treatment)

    def test_the_counterfactual_design_takes_a_value_per_row(self) -> None:
        data = self._data()
        shifted = data.treatment + 0.5
        design = data.counterfactual_design(shifted)
        assert design.shape == data.treatment_design().shape
        np.testing.assert_allclose(design[:, 0], shifted)
        # A scalar still broadcasts, so the constant-dose question is still askable.
        np.testing.assert_allclose(data.counterfactual_design(2.0)[:, 0], 2.0)

    def test_a_mis_sized_counterfactual_vector_is_refused(self) -> None:
        data = self._data()
        with pytest.raises(DataError, match="one value for everybody or one per row"):
            data.counterfactual_design(np.zeros(7))

    @pytest.mark.parametrize(
        ("accessor", "match"),
        [
            (lambda d: d.arm_label(0.0), "has no arms"),
            (lambda d: d.treated_fraction, "names no quantity"),
        ],
    )
    def test_the_arm_accessors_refuse_rather_than_answer(self, accessor, match: str) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(DataError, match=match):
            accessor(self._data())

    def test_a_permutation_replaces_the_treatment(self) -> None:
        data = self._data()
        rng = np.random.default_rng(3)
        replaced = data.with_treatment(rng.permutation(data.treatment))
        assert replaced.treatment_kind == "continuous"
        np.testing.assert_allclose(np.sort(replaced.treatment), np.sort(data.treatment))

    def test_the_kind_survives_a_subset(self) -> None:
        data = self._data()
        assert data.subset(np.arange(50)).treatment_kind == "continuous"

    def test_a_non_numeric_column_cannot_be_continuous(self) -> None:
        frame = _continuous_frame()
        frame["A"] = np.where(frame["A"] > 0, "high", "low")
        with pytest.raises(DataError, match="cannot be treated as continuous"):
            CausalData.from_frame(frame, outcome="Y", treatment="A", treatment_kind="continuous")

    def test_too_few_distinct_values_warns_rather_than_refusing(self) -> None:
        # A coarse support is estimable -- the density is a probability mass function and a
        # shift moves along the ordered values, which is exactly what the discrete oracle
        # law relies on. It is usually a mistake, so it warns; it is not an error.
        frame = _continuous_frame()
        frame["A"] = np.round(frame["A"] * 0.4)
        with pytest.warns(DataWarning, match="declared continuous but takes only"):
            data = CausalData.from_frame(
                frame, outcome="Y", treatment="A", treatment_kind="continuous"
            )
        assert data.is_continuous_treatment

    @pytest.mark.parametrize("role", ["delta", "weights", "intermediate"])
    def test_the_roles_a_dose_used_to_refuse_are_accepted(self, role: str) -> None:
        """All three were refused together, on a reason that was wrong for all three.

        ``P(Delta = 1 | A, W)`` and ``P(Z = z | A, W)`` are conditional *probabilities* of
        binary events -- an ordinary classifier with the dose as a numeric feature -- and
        do not become densities because ``A`` is continuous; a weight tilts the population
        and is not in the clever covariate at all.  What was genuinely missing was the
        mechanisms evaluated at the *shifted* dose, and an oracle law to check the
        composition against; ``docs/roadmap.md``'s *Refusals worth lifting* item 5 records
        both.
        """
        frame = _continuous_frame()
        # Both levels present, and a positive weight: the refusal used to fire before any
        # of these columns was validated, so a constant 1.0 passed for all three roles.
        frame["extra"] = np.tile([0.0, 1.0], len(frame) // 2) if role != "weights" else 1.0
        data = CausalData.from_frame(
            frame,
            outcome="Y",
            treatment="A",
            treatment_kind="continuous",
            covariates=["W1", "W2"],
            **{role: "extra"},
        )
        assert data.is_continuous_treatment

    def test_an_unknown_kind_is_refused(self) -> None:
        with pytest.raises(DataError, match="treatment_kind must be"):
            CausalData.from_frame(
                _continuous_frame(),
                outcome="Y",
                treatment="A",
                treatment_kind="ordinal",  # type: ignore[arg-type]
            )


class TestTheDiscretePathDidNotMove:
    """``treatment_kind`` defaults to the arm-coded path, byte for byte.

    CLAUDE.md calls the arm path a regression surface.  The continuous branch is new code
    on the same methods, so these assert the old answers directly rather than trusting
    that a default argument left them alone.
    """

    def test_the_default_is_the_arm_coded_path(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        assert data.treatment_kind == "discrete"
        assert not data.is_continuous_treatment
        assert data.arm_codes == (0.0, 1.0)

    def test_a_binary_design_is_still_the_code_itself(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        np.testing.assert_array_equal(data.treatment_block(data.treatment)[:, 0], data.treatment)

    def test_a_counterfactual_arm_is_still_validated(self) -> None:
        data = CausalData.from_frame(_frame(), outcome="Y", treatment="A")
        with pytest.raises(DataError, match="is not an arm of"):
            data.counterfactual_design(2.0)

    def test_a_multi_arm_design_is_still_indicators(self) -> None:
        frame = _frame()
        rng = np.random.default_rng(1)
        frame["A"] = rng.integers(0, 3, len(frame)).astype(float)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        assert data.treatment_block(data.treatment).shape == (data.n, 2)
