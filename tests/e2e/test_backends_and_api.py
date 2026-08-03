"""Dataframe-backend parity and the public result API.

The promise the library makes is "pandas in, pandas out; polars in, polars out", with
*identical numbers* either way.  Identical, not merely close: the two paths differ only
in how the data is unwrapped, so any numerical discrepancy would signal that one of them
is quietly reordering or recasting something.

The same standard applies to arrow-backed pandas, which is a third *dtype* backend rather
than a third library.  It used to be tested nowhere -- ``pyarrow`` sat in the ``dev``
extra and was imported by nothing -- and it worked only because narwhals happens to map
arrow numerics onto ``float64``.  The ingestion path now casts inside narwhals instead, so
what these tests pin is that the cast is exact and that a null reaches the same refusal
whichever dtype carried it.
"""

from __future__ import annotations

import re
from pathlib import Path

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from cleverly import TMLE
from cleverly.datasets import make_binary_outcome, make_linear_ate, make_missing_outcome
from tests.conftest import fast_tmle

ESTIMANDS = ("ate", "att", "atc", "ey1", "ey0")


def arrow_backed(frame: pd.DataFrame) -> pd.DataFrame:
    """The same pandas frame with every column on an ``ArrowDtype``.

    A helper rather than a generator option: ``backend=`` names the dataframe *library*,
    and arrow-backed pandas is still pandas.  Which is the point -- the fit below has to
    come out the same as the numpy-backed one, not merely close to it.
    """
    return frame.convert_dtypes(dtype_backend="pyarrow")


@pytest.fixture(scope="module")
def paired_fits() -> tuple[object, object, object]:
    pandas_frame, _ = make_linear_ate(n=900, seed=91, backend="pandas")
    polars_frame, _ = make_linear_ate(n=900, seed=91, backend="polars")
    columns = {"outcome": "Y", "treatment": "A"}
    return (
        fast_tmle(estimands=ESTIMANDS).fit(pandas_frame, **columns).single(),
        fast_tmle(estimands=ESTIMANDS).fit(polars_frame, **columns).single(),
        fast_tmle(estimands=ESTIMANDS).fit(arrow_backed(pandas_frame), **columns).single(),
    )


class TestBackendParity:
    def test_the_generators_agree_across_backends(self) -> None:
        pandas_frame, _ = make_linear_ate(n=200, seed=92, backend="pandas")
        polars_frame, _ = make_linear_ate(n=200, seed=92, backend="polars")
        assert np.allclose(pandas_frame["Y"].to_numpy(), polars_frame["Y"].to_numpy())

    @pytest.mark.parametrize("estimand", ESTIMANDS)
    def test_estimates_are_bit_identical(self, paired_fits, estimand: str) -> None:
        from_pandas, from_polars, _ = paired_fits
        assert from_pandas.psi(estimand) == from_polars.psi(estimand)
        assert from_pandas[estimand].std_error == from_polars[estimand].std_error
        assert from_pandas[estimand].ci == from_polars[estimand].ci
        assert from_pandas[estimand].pvalue == from_polars[estimand].pvalue

    def test_influence_curves_are_identical(self, paired_fits) -> None:
        from_pandas, from_polars, _ = paired_fits
        for estimand in ESTIMANDS:
            assert np.array_equal(
                from_pandas[estimand].influence_curve,
                from_polars[estimand].influence_curve,
            )

    def test_results_come_back_in_the_input_backend(self, paired_fits) -> None:
        from_pandas, from_polars, _ = paired_fits
        assert isinstance(from_pandas.to_frame(), pd.DataFrame)
        assert isinstance(from_polars.to_frame(), pl.DataFrame)
        assert isinstance(from_pandas.influence_frame(), pd.DataFrame)
        assert isinstance(from_polars.influence_frame(), pl.DataFrame)

    def test_diagnostic_frames_follow_the_backend_too(self, paired_fits) -> None:
        from_pandas, from_polars, _ = paired_fits
        assert isinstance(
            from_pandas.sensitivity.truncation_curve([0.01], estimands=["ate"]), pd.DataFrame
        )
        assert isinstance(
            from_polars.sensitivity.truncation_curve([0.01], estimands=["ate"]), pl.DataFrame
        )

    def test_the_summaries_are_identical_text(self, paired_fits) -> None:
        """Same text, character for character -- apart from when each fit ran.

        The summary ends with a provenance line carrying ``created_utc`` at one-second
        resolution (:func:`cleverly.provenance.record`).  The two fits in the fixture take
        about a second each, so they straddle a second boundary often enough for a bare
        string comparison to be a coin flip rather than a test -- which is what it was.
        The backend claim is about the numbers and the layout, not about the clock, so the
        timestamp is normalised out and asserted separately to be present.
        """
        from_pandas, from_polars, _ = paired_fits
        stamp = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}")
        assert stamp.search(from_pandas.summary()), "the provenance line should carry a time"
        assert stamp.sub("<t>", from_pandas.summary()) == stamp.sub("<t>", from_polars.summary())

    @pytest.mark.parametrize("estimand", ESTIMANDS)
    def test_an_arrow_backed_fit_is_bit_identical(self, paired_fits, estimand: str) -> None:
        """``dtype_backend="pyarrow"`` must be the same fit, not a nearby one.

        Bit-for-bit is the right bar and not a strict one: the arrow column holds the same
        float64 values, so the only way a digit could move is if the ingestion path took a
        different route through them -- which is exactly what this is here to catch.
        """
        from_pandas, _, from_arrow = paired_fits
        assert from_pandas.psi(estimand) == from_arrow.psi(estimand)
        assert from_pandas[estimand].std_error == from_arrow[estimand].std_error
        assert from_pandas[estimand].ci == from_arrow[estimand].ci
        assert np.array_equal(
            from_pandas[estimand].influence_curve, from_arrow[estimand].influence_curve
        )

    def test_an_arrow_backed_fit_reports_pandas_and_says_so(self, paired_fits) -> None:
        """Arrow in, *numpy-backed* pandas out -- the documented limit of the promise.

        The library is preserved; the dtype backend is not, because results are built from
        numpy through ``nw.from_dict``, which has no ``dtype_backend`` knob.  Pinned rather
        than left implicit so that changing it is a decision instead of a surprise.
        """
        _, _, from_arrow = paired_fits
        frame = from_arrow.to_frame()
        assert isinstance(frame, pd.DataFrame)
        assert from_arrow.data.backend == "pandas"
        assert not isinstance(frame["psi"].dtype, pd.ArrowDtype)

    def test_a_pyarrow_table_is_a_declared_backend(self) -> None:
        """``narwhals`` accepts one, so the choice was declaring it or half-supporting it."""
        pa = pytest.importorskip("pyarrow")
        pandas_frame, _ = make_linear_ate(n=300, seed=97, backend="pandas")
        result = (
            fast_tmle(estimands=("ate",))
            .fit(pa.Table.from_pandas(pandas_frame), outcome="Y", treatment="A")
            .single()
        )
        assert result.data.backend == "pyarrow"
        assert isinstance(result.to_frame(), pa.Table)

    def test_a_polars_fit_with_every_role(self) -> None:
        frame, _ = make_missing_outcome(n=900, seed=93, backend="polars")
        frame = frame.with_columns(
            pl.Series("w", np.linspace(0.5, 1.5, len(frame))),
            pl.Series("cl", np.repeat(np.arange(len(frame) // 10), 10).astype(float)),
        )
        result = (
            fast_tmle(estimands=("ate",))
            .fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2", "W3"],
                delta="Delta",
                weights="w",
                id="cl",
            )
            .single()
        )
        assert isinstance(result.to_frame(), pl.DataFrame)
        assert result.data.n_clusters == 90
        assert result.validation.score_check().passed


class TestEveryReportFollowsTheBackend:
    """Not just the estimates: the diagnostics too, with nothing threaded in by hand.

    Each of these ``to_frame()`` methods takes an optional container and used to fall back
    to the *default* backend without one -- and nothing inside the package ever passed one,
    so a polars fit's diagnostics all came back as pandas.  The reports carry the backend
    name themselves now.  Called bare here on purpose: passing ``data=`` would test the
    override rather than the default, which is the thing that was broken.
    """

    @pytest.fixture(scope="class")
    def polars_fit(self):  # type: ignore[no-untyped-def]
        frame, _ = make_linear_ate(n=400, seed=98, backend="polars")
        return (
            fast_tmle(estimands=("ate",), targeting_scheme="fold")
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )

    def test_the_validation_reports(self, polars_fit) -> None:
        assert isinstance(polars_fit.validation.score_check().to_frame(), pl.DataFrame)
        assert isinstance(polars_fit.validation.nuisance().to_frame(), pl.DataFrame)
        refutation = polars_fit.validation.refute(tests=["placebo"], n_replicates=2)
        assert isinstance(refutation.to_frame(), pl.DataFrame)

    def test_the_positivity_report(self, polars_fit) -> None:
        assert isinstance(polars_fit.sensitivity.positivity().to_frame(), pl.DataFrame)

    def test_the_fold_targeting_report(self, polars_fit) -> None:
        """And it is a *frame*: this one alone used to hand back a bare ``dict``."""
        frame = polars_fit.cv_targeting.to_frame()
        assert isinstance(frame, pl.DataFrame)
        assert "estimand" in frame.columns

    def test_the_regime_support_report(self) -> None:
        """``RegimeSupport.to_frame`` documented this behaviour before it had it."""
        from cleverly.interventions import Static

        frame, _ = make_linear_ate(n=400, seed=99, backend="polars")
        result = (
            fast_tmle(estimands=("ey_regime",), interventions=(Static(0), Static(1)))
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert isinstance(result.sensitivity.support().to_frame(), pl.DataFrame)

    def test_a_saved_fit_remembers_its_backend(self, polars_fit, tmp_path: Path) -> None:
        """The backend is a *name* now, so it survives the round trip -- it could not
        while the container held the input frame itself."""
        import cleverly

        path = tmp_path / "polars-fit.npz"
        polars_fit.save(path)
        reloaded = cleverly.load(path)
        assert reloaded.data.backend == "polars"
        assert isinstance(reloaded.to_frame(), pl.DataFrame)
        assert isinstance(reloaded.validation.score_check().to_frame(), pl.DataFrame)


class TestResultApi:
    def test_mapping_style_access(self, paired_fits) -> None:
        result, _, _ = paired_fits
        assert "ate" in result
        assert set(result) == set(ESTIMANDS)
        assert result["ate"] is result.ate
        assert result.psi("ate") == result.ate.psi

    def test_an_unrequested_estimand_gives_a_helpful_error(self, paired_fits) -> None:
        result, _, _ = paired_fits
        with pytest.raises(KeyError, match="was not requested"):
            result["rr"]

    def test_the_tidy_frame_has_one_row_per_estimand(self, paired_fits) -> None:
        result, _, _ = paired_fits
        frame = nw.from_native(result.to_frame(), eager_only=True)
        assert len(frame) == len(ESTIMANDS)
        assert frame["estimand"].to_list() == list(ESTIMANDS)
        assert {"psi", "std_err", "ci_lower", "ci_upper", "p_value"} <= set(frame.columns)

    def test_the_influence_frame_has_one_column_per_estimand(self, paired_fits) -> None:
        result, _, _ = paired_fits
        frame = nw.from_native(result.influence_frame(), eager_only=True)
        assert set(frame.columns) == set(ESTIMANDS)
        assert len(frame) == result.n

    def test_the_summary_reports_the_configuration_actually_used(self, paired_fits) -> None:
        result, _, _ = paired_fits
        text = result.summary()
        assert "Targeted maximum likelihood estimation" in text
        assert "cross-fitted over 5 folds" in text
        assert "propensity truncated to" in text
        # The resolved outcome bounds are reported, not just the request.
        assert "outcome scaled from" in text
        for estimand in ESTIMANDS:
            assert estimand in text

    def test_the_config_records_resolved_values(self, paired_fits) -> None:
        result, _, _ = paired_fits
        config = result.config
        assert config.family == "gaussian"
        assert config.estimands == ESTIMANDS
        assert config.n_folds == 5
        assert 0.0 < config.g_bounds[0] < 0.5
        assert config.q_bounds is not None

    def test_the_score_property_is_the_mean_influence_curve(self, paired_fits) -> None:
        result, _, _ = paired_fits
        for estimand in ESTIMANDS:
            estimate = result[estimand]
            assert estimate.score == pytest.approx(float(estimate.influence_curve.mean()))

    def test_diagnostics_are_cached_across_calls(self, paired_fits) -> None:
        result, _, _ = paired_fits
        assert result.sensitivity is result.sensitivity
        assert result.validation is result.validation

    def test_estimates_expose_a_dict_form(self, paired_fits) -> None:
        result, _, _ = paired_fits
        row = result["ate"].to_dict()
        assert row["estimand"] == "ate"
        assert row["scale"] == "difference"

    def test_a_ratio_estimate_reports_its_log_scale(self) -> None:
        frame, _ = make_binary_outcome(n=900, seed=94)
        result = fast_tmle(estimands=("rr", "or")).fit(frame, outcome="Y", treatment="A").single()
        for name in ("rr", "or"):
            estimate = result[name]
            assert estimate.scale == "ratio"
            assert estimate.log_psi is not None
            assert estimate.psi == pytest.approx(np.exp(estimate.log_psi))
            assert "log_psi" in estimate.to_dict()

    def test_alpha_sig_widens_the_reported_interval(self) -> None:
        frame, _ = make_linear_ate(n=700, seed=95)
        narrow = (
            fast_tmle(estimands=("ate",), alpha_sig=0.05)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        wide = (
            fast_tmle(estimands=("ate",), alpha_sig=0.01)
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
        assert (wide["ate"].ci[1] - wide["ate"].ci[0]) > (narrow["ate"].ci[1] - narrow["ate"].ci[0])
        assert "99% CI" in wide.summary()

    def test_causal_data_can_be_passed_directly(self) -> None:
        """Passing a built container must be the same fit as passing the frame.

        Asserted as an equality against the frame-based fit rather than as a coverage
        check on the truth.  The claim here is about *plumbing* -- that ``fit`` accepts a
        ``CausalData`` and routes it to the same place -- and a single-fit interval covers
        the truth only 95% of the time by construction, so the old assertion was a coin
        flip standing in for something exactly checkable.
        """
        from cleverly import CausalData

        frame, _ = make_linear_ate(n=800, seed=96)
        data = CausalData.from_frame(frame, outcome="Y", treatment="A")
        via_container = fast_tmle(estimands=("ate",)).fit(data).single()
        via_frame = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A").single()
        assert via_container["ate"].psi == via_frame["ate"].psi
        assert via_container["ate"].std_error == via_frame["ate"].std_error
        np.testing.assert_array_equal(
            via_container["ate"].influence_curve, via_frame["ate"].influence_curve
        )

    def test_the_nuisance_fits_are_retained_for_reuse(self, paired_fits) -> None:
        result, _, _ = paired_fits
        nuisance = result.nuisance
        # One column per arm, even for a binary treatment, where column 0 is 1 - g1.
        assert nuisance.propensity.values.shape == (result.n, 2)
        assert nuisance.propensity.arms == (0.0, 1.0)
        assert nuisance.outcome.observed.shape == (result.n,)
        # The raw, untruncated propensity is what makes the truncation curve cheap.
        assert nuisance.propensity.arm(1.0).min() < result.config.g_bounds[1]
        bounded = nuisance.bounded_propensity((0.2, 0.8))
        assert bounded.shape == (result.n, 2)
        assert bounded.min() >= 0.2
        assert bounded.max() <= 0.8
        # The two arms still sum to one exactly: with two arms the bound is applied to
        # g1 and arm 0 taken as its complement, rather than clipped independently.
        np.testing.assert_array_equal(bounded.sum(axis=1), np.ones(result.n))


class TestPackageSurface:
    def test_the_top_level_namespace_is_importable(self) -> None:
        import cleverly

        for name in cleverly.__all__:
            assert hasattr(cleverly, name), name

    def test_the_version_is_exposed(self) -> None:
        import cleverly

        assert cleverly.__version__.count(".") >= 2

    def test_submodules_export_what_they_advertise(self) -> None:
        import importlib

        for module_name in (
            "cleverly.data",
            "cleverly.datasets",
            "cleverly.estimators",
            "cleverly.fluctuation",
            "cleverly.inference",
            "cleverly.learners",
            "cleverly.longitudinal",
            "cleverly.sensitivity",
            "cleverly.utils",
            "cleverly.validation",
        ):
            module = importlib.import_module(module_name)
            for name in module.__all__:
                assert hasattr(module, name), f"{module_name}.{name}"

    def test_the_readme_quickstart_runs(self) -> None:
        """The quickstart in the README, executed as written.

        Cheaper learners and a smaller ``n`` than the README's, which is the fast tier's
        rule; what is pinned is that every call in it exists and returns what it claims.
        """
        from cleverly.datasets import make_nonlinear_ate

        frame, truth = make_nonlinear_ate(n=800, seed=0, backend="polars")
        est = TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            n_folds=4,
            estimands=("ate", "att", "atc", "ey1", "ey0"),
            random_state=0,
        )
        res = est.fit(
            frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3", "W4"]
        ).single()
        assert isinstance(res.summary(), str)
        assert isinstance(res.to_frame(), pl.DataFrame)
        assert isinstance(res.estimates["ate"].psi, float)
        assert len(res.estimates["ate"].ci) == 2
        assert res.estimates["ate"].influence_curve.shape == (800,)
        assert "ate" in truth

    def test_the_readme_end_to_end_example_runs(self, tmp_path: Path) -> None:
        """The README's end-to-end fit: every diagnostic it shows, off one fitted result.

        The point of that section is that nothing after ``fit`` refits, so the assertions
        are that each call exists and returns its own report -- not what the numbers are,
        which the README states for the default learner library rather than for ``glm``.
        """
        import cleverly
        from cleverly.datasets import make_nonlinear_ate

        frame, _ = make_nonlinear_ate(n=800, seed=0)
        res = (
            TMLE(
                estimands=("ate", "ey1", "ey0"),
                outcome_learner="glm",
                treatment_learner="glm",
                n_folds=4,
                learner_folds=3,
                random_state=0,
            )
            .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2", "W3", "W4"])
            .single()
        )
        assert isinstance(res.summary(), str)

        assert isinstance(res.sensitivity.positivity().summary(), str)
        assert set(res.sensitivity.robustness_value()) >= {"rv", "rva"}
        assert res.sensitivity.benchmark(["W1", "W2"]) is not None

        assert isinstance(res.validation.score_check().summary(), str)
        assert isinstance(res.validation.nuisance().summary(), str)

        # The risk ratio was not among the requested estimands: it comes from the joint
        # influence curve by the delta method, with no refit.
        ratio = res.contrast(lambda psi: psi[0] / psi[1], ["ey1", "ey0"])
        assert ratio.psi == pytest.approx(res.estimates["ey1"].psi / res.estimates["ey0"].psi)

        path = tmp_path / "fit.npz"
        res.save(path)
        assert cleverly.load(path).estimates["ate"].psi == res.estimates["ate"].psi


class TestTheResultSet:
    """``fit`` returns a mapping of results, always.

    It used to return ``TMLEResult | TMLEResultSet`` -- one result ordinarily, one per
    level of the intermediate for a controlled direct effect.  A union return puts an
    ``isinstance`` check in every caller that cannot know in advance which it will get,
    and the library carried one of its own in ``CoverageStudy``.  Now there is one type,
    and the ordinary fit is the single-entry case keyed ``None`` -- the same sentinel the
    estimator already uses internally for "no intermediate variable".
    """

    @pytest.fixture(scope="class")
    def plain(self):  # type: ignore[no-untyped-def]
        frame, _ = make_linear_ate(n=300, seed=0)
        return fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A")

    @pytest.fixture(scope="class")
    def two_levels(self):  # type: ignore[no-untyped-def]
        from cleverly.datasets import GENERATORS

        frame, _ = GENERATORS["cde"](n=300, seed=1)
        covariates = [c for c in frame.columns if c.startswith("W")]
        estimator = TMLE(
            outcome_learner="glm",
            treatment_learner="glm",
            intermediate_learner="glm",
            estimands=("ate",),
            n_folds=4,
            random_state=0,
        )
        return estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=covariates,
            intermediate="Z",
        )

    def test_an_ordinary_fit_is_a_single_entry_set_keyed_none(self, plain) -> None:
        from cleverly.estimators.base import TMLEResult, TMLEResultSet

        assert isinstance(plain, TMLEResultSet)
        assert list(plain) == [None]
        assert len(plain) == 1
        assert isinstance(plain.single(), TMLEResult)
        assert plain[None] is plain.single()

    def test_the_single_entry_summary_has_no_level_header(self, plain) -> None:
        """Otherwise every ordinary fit would print the internal ``None`` sentinel."""
        assert plain.summary() == plain.single().summary()
        assert "None" not in plain.summary().splitlines()[0]

    def test_the_single_entry_frame_is_the_result_frame(self, plain) -> None:
        expected = plain.single().to_frame()
        assert list(plain.to_frame().columns) == list(expected.columns)
        assert len(plain.to_frame()) == len(expected)

    def test_a_missing_key_names_what_is_available(self, plain) -> None:
        with pytest.raises(KeyError, match="available"):
            plain[1.0]

    def test_a_controlled_direct_effect_holds_one_result_per_level(self, two_levels) -> None:
        assert list(two_levels) == [0.0, 1.0]
        assert two_levels[0.0].intermediate_value == 0.0
        assert two_levels[1.0].intermediate_value == 1.0
        # Two parameters, so two answers: they agree only without an A-by-Z interaction.
        assert two_levels[0.0].psi("ate") != two_levels[1.0].psi("ate")

    def test_single_refuses_to_pick_between_two_parameters(self, two_levels) -> None:
        """The whole reason ``fit`` cannot just return one result for a CDE."""
        with pytest.raises(KeyError, match="no single one to return"):
            two_levels.single()

    def test_a_two_level_summary_is_headed_by_level(self, two_levels) -> None:
        text = two_levels.summary()
        assert "--- Z = 0 ---" in text
        assert "--- Z = 1 ---" in text

    def test_a_two_level_frame_stacks_the_levels(self, two_levels) -> None:
        assert len(two_levels.to_frame()) == sum(
            len(two_levels[level].to_frame()) for level in two_levels
        )

    def test_the_array_entry_point_returns_a_set_too(self) -> None:
        from cleverly import tmle

        rng = np.random.default_rng(0)
        w = rng.normal(size=(200, 2))
        a = rng.binomial(1, 0.5, 200).astype(float)
        y = a + w[:, 0] + rng.normal(size=200)
        fitted = tmle(y, a, w, outcome_learner="glm", treatment_learner="glm", n_folds=4)
        assert list(fitted) == [None]
        assert fitted.single().psi("ate") == pytest.approx(1.0, abs=0.4)

    def test_indexing_by_estimand_says_so_instead_of_failing_on_float(self) -> None:
        """``res["ate"]`` is the likeliest mistake, so it gets a message about itself.

        A result is indexed by estimand and a result *set* by level.  Left to ``float()``,
        the mix-up surfaced as "could not convert string to float: 'ate'".
        """
        frame, _ = make_linear_ate(n=200, seed=7)
        fitted = fast_tmle(estimands=("ate",)).fit(frame, outcome="Y", treatment="A")
        with pytest.raises(KeyError, match="indexed by intermediate level, not by estimand"):
            fitted["ate"]  # type: ignore[index]
