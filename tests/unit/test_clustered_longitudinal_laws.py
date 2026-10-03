"""The clustered longitudinal study laws keep the unclustered truths and carry the clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleverly.datasets import make_longitudinal, make_longitudinal_survival
from cleverly.datasets.longitudinal import _outcome_probability
from tests.studies import clustered_longitudinal_laws as law

SEED = 20261101


def _intracluster_correlation(residual: np.ndarray, ids: np.ndarray) -> float:
    """The one-way ANOVA intraclass correlation of ``residual`` within ``ids``."""
    groups = [residual[ids == code] for code in np.unique(ids)]
    k, total = len(groups), len(residual)
    sizes = np.array([len(group) for group in groups])
    grand = residual.mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups) / (total - k)
    n0 = (total - (sizes**2).sum() / total) / (k - 1)
    return float((between - within) / (between + (n0 - 1) * within))


class TestTheLatentKeepsTheMean:
    @pytest.mark.parametrize("p", [0.0, 1.0, 0.03, 0.5, 0.97, 0.31])
    def test_the_two_signs_average_to_p_exactly(self, p: float) -> None:
        for magnitude in (1.0, 0.5, 0.73):
            up = law.perturbed_probability(np.array([p]), np.array([magnitude]))
            down = law.perturbed_probability(np.array([p]), np.array([-magnitude]))
            assert (up + down)[0] / 2.0 == pytest.approx(p, abs=1e-15)
            assert 0.0 <= down[0] <= up[0] <= 1.0

    def test_the_scaled_latent_has_mean_zero_and_the_declared_support(self) -> None:
        u = law.cluster_latents("scaled", 200_000, SEED)
        assert np.all((np.abs(u) >= 0.5) & (np.abs(u) <= 1.0))
        assert abs(u.mean()) < 0.005
        assert u.var() == pytest.approx(7.0 / 12.0, abs=0.005)

    def test_the_rademacher_latent_is_a_sign(self) -> None:
        u = law.cluster_latents("rademacher", 1000, SEED)
        assert set(np.unique(u).tolist()) == {-1.0, 1.0}


class TestTheEndOfStudyLaw:
    @pytest.mark.parametrize("sizes", ["equal40", "unequal40"])
    def test_every_column_but_the_outcome_is_the_generators(self, sizes: str) -> None:
        frame, truth = law.draw_end_of_study(30, sizes, SEED)
        base, base_truth = make_longitudinal(
            n=len(frame), seed=SEED, censoring=True, backend="pandas"
        )
        pd.testing.assert_frame_equal(frame.drop(columns=["Y", "id"]), base.drop(columns=["Y"]))
        assert truth == base_truth
        assert frame["Y"].isna().equals(base["Y"].isna())

    def test_the_unequal_sizes_come_from_the_declared_support(self) -> None:
        sizes = law.cluster_sizes("unequal40", 5000, SEED)
        assert sizes.min() == 10 and sizes.max() == 70
        assert sizes.mean() == pytest.approx(40.0, abs=0.5)

    def test_the_three_streams_are_distinct(self) -> None:
        draws = [law.child_stream(SEED, index).random(4).tolist() for index in range(3)]
        assert draws[0] != draws[1] and draws[1] != draws[2] and draws[0] != draws[2]
        # Each draw reads the latents and the uniforms from their own streams, whatever the
        # size stream drew: the outcome is rebuilt here from the two streams alone.
        for sizes in ("equal40", "unequal40"):
            frame, _ = law.draw_end_of_study(30, sizes, SEED, latent="scaled")
            ids = frame["id"].to_numpy().astype(int)
            a1 = frame["A1"].to_numpy().astype(int)
            p = _outcome_probability(
                frame["W1"].to_numpy(),
                frame["W2"].to_numpy(),
                np.nan_to_num(frame["L2"].to_numpy()),
                frame["A1"].to_numpy(),
                np.nan_to_num(frame["A2"].to_numpy()),
            )
            rng = law.child_stream(SEED, law.LATENT_STREAM)
            u = rng.choice([-1.0, 1.0], size=(30, 2)) * rng.uniform(0.5, 1.0, size=(30, 2))
            uniforms = law.child_stream(SEED, law.OUTCOME_STREAM).random(len(frame))
            y = (uniforms < law.perturbed_probability(p, u[ids, a1])).astype(float)
            observed = frame["Y"].notna().to_numpy()
            assert np.array_equal(frame["Y"].to_numpy()[observed], y[observed])
        equal, _ = law.draw_end_of_study(30, "equal40", SEED)
        unequal, _ = law.draw_end_of_study(30, "unequal40", SEED)
        assert len(equal) == 1200 and len(unequal) != 1200

    @pytest.mark.parametrize("latent", law.LATENTS)
    def test_the_outcome_is_correlated_within_a_cluster(self, latent: str) -> None:
        frame, _ = law.draw_end_of_study(100, "equal40", SEED, latent=latent)
        kept = frame[frame["Y"].notna() & (frame["A1"] == 1.0)]
        p = _outcome_probability(
            kept["W1"].to_numpy(),
            kept["W2"].to_numpy(),
            kept["L2"].to_numpy(),
            kept["A1"].to_numpy(),
            kept["A2"].to_numpy(),
        )
        icc = _intracluster_correlation(kept["Y"].to_numpy() - p, kept["id"].to_numpy().astype(int))
        assert icc > law.INTRACLUSTER_WITNESS[latent]

    def test_censored_rows_keep_a_missing_outcome(self) -> None:
        frame, _ = law.draw_end_of_study(30, "equal40", SEED)
        censored = (frame["C1"] == 0.0) | (frame["C2"] == 0.0)
        assert censored.any()
        assert frame.loc[censored, "Y"].isna().all()


class TestTheSurvivalLaw:
    def test_only_the_at_risk_second_event_is_redrawn(self) -> None:
        frame, truth = law.draw_survival(30, "equal40", SEED)
        base, base_truth = make_longitudinal_survival(
            n=len(frame), seed=SEED, censoring=True, backend="pandas"
        )
        assert truth == base_truth
        pd.testing.assert_frame_equal(frame.drop(columns=["Y2", "id"]), base.drop(columns=["Y2"]))
        at_risk = (frame["Y1"] == 0.0) & frame["Y2"].notna()
        untouched = ~at_risk
        pd.testing.assert_series_equal(frame.loc[untouched, "Y2"], base.loc[untouched, "Y2"])
        assert (frame.loc[frame["Y1"] == 1.0, "Y2"] == 1.0).all()
        assert frame["Y2"].isna().equals(base["Y2"].isna())
        # The redraw moved some at-risk rows, so the component is present.
        assert (frame.loc[at_risk, "Y2"] != base.loc[at_risk, "Y2"]).any()
