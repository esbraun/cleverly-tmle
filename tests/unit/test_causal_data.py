"""The ``CausalData`` container: validation, encoding, and backend round-trips."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest

from cleverly.data import CausalData
from cleverly.exceptions import DataError, WeightingWarning


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

    def test_rejects_a_multi_valued_treatment(self) -> None:
        frame = _frame()
        frame["A"] = np.repeat([0.0, 1.0, 2.0, 3.0], len(frame) // 4)
        with pytest.raises(DataError, match="must be binary"):
            CausalData.from_frame(frame, outcome="Y", treatment="A")

    def test_rejects_a_single_armed_treatment(self) -> None:
        frame = _frame()
        frame["A"] = 1.0
        with pytest.raises(DataError, match="both levels must be present"):
            CausalData.from_frame(frame, outcome="Y", treatment="A")


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
