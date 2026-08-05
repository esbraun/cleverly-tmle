r"""What the replication harness reports, and the two things in it that are not arithmetic.

``benchmarks/drtmle_companion_grid.py`` is a characterisation and asserts nothing, so almost
nothing in it is under test here.  What is under test is **the part a reading rests on**,
which is the same rule ``tests/unit/test_drtmle_coverage.py`` is written under:

* **a draw is one fit.**  The whole design -- every replicate of both rules, every rung, read
  off one fit -- rests on that, and the identity itself is pinned in
  ``tests/unit/test_drtmle_remainder_study.py`` against a companion fitted alone.  What is
  pinned here is that this module *uses* it: one fit whatever the ladder's length and the
  replicate counts, with the whole weight vector passed and a ``Window`` doing the slicing.
* **the decomposition is conditional, and the two rules estimate one common term.**  This is
  E1b's whole correction: the rule's error comes from replication at a fixed fit rather than
  from differencing two marginal variances or from refining a grid.  Both halves are checked
  on constructed rows whose answer is known by hand.
* **every share carries an interval**, and the interval widens as draws are removed.  A share
  printed without one is what E1 printed.
* **``delta`` is paired within the draw *and* within the scramble.**  Across scrambles it
  would carry the rule's own error, which is the column it is supposed to be free of.
* **the table's rows are the width of its headers**, the structural pin against the one
  mistake that produces a complete, plausible, wrong table rather than a failure.
* **a failed replicate is a row and not a gap**, since a ladder with a rung missing looks
  like a shorter ladder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the benchmarks package is not installed, only checked out
    sys.path.insert(0, str(ROOT))

from benchmarks import drtmle_companion_grid as grid  # noqa: E402
from benchmarks import drtmle_remainder as remainder  # noqa: E402

#: The smallest fit that runs the whole path, since what is checked here is wiring.
N = 300
POINTS = (64, 128)


def row(**overrides) -> grid.GridRow:
    """A hand-built replicate, so every table below can be tested without fitting anything."""
    defaults: dict[str, object] = {
        "cell": "q-drift",
        "n": 600,
        "data_seed": 1,
        "fold_seed": 2,
        "rule": "sobol",
        "replicate": 500,
        "points": 1_024,
        "rows": 2_048,
        "p0_curve": -0.11,
        "remaining": 0.017,
        "root_n_remaining": 0.42,
        "companion_se": 1.19,
        "branch_q": -0.001,
        "branch_g": 0.019,
        "branch_error": 0.002,
        "seconds": 4.7,
    }
    return grid.GridRow(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestADrawIsOneFit:
    """One fit per draw whatever the ladder's length or the replicate counts.

    E1's version of this counted two -- one per rule -- and E1b's counts one, because both
    rules now live in one stacked companion.  A refit per replicate would show up here as
    ``scrambles + draw_replicates``, and would also be the thing that makes a spread across
    replicates uninterpretable: it would carry whatever two fits differ by as well.
    """

    def test_every_replicate_and_every_rung_come_off_one_draw(self, monkeypatch) -> None:
        fits: list[int] = []
        original = grid._fit

        def counted(payload, evaluation):
            fits.append(1)
            return original(payload, evaluation)

        monkeypatch.setattr(grid, "_fit", counted)
        payload = grid.Payload("q-drift", N, 11, 3, POINTS, 2, 128, 2)

        rows = grid.one_draw(payload)

        assert len(fits) == 1
        sobol = [r for r in rows if r.rule == "sobol"]
        drawn = [r for r in rows if r.rule == "draw"]
        # Two scrambles times two rungs, and two i.i.d. companions with no ladder on them.
        assert sorted(r.points for r in sobol) == sorted(POINTS * 2)
        assert len({r.replicate for r in sobol}) == 2
        assert len(drawn) == 2 and {r.points for r in drawn} == {0}

    def test_a_coarse_rung_is_read_at_the_full_weight_vector(self) -> None:
        """The integrators check their weights against the companion's row count.

        Slicing the vector before handing it over is exactly the stale-weights mistake that
        check exists to catch, so a coarse rung has to pass the whole vector and let the
        ``Window`` do the work.  This is the negative control for that: a pre-sliced vector
        raises rather than silently integrating against the wrong measure.
        """
        law = grid.injection.base_law()
        payload = grid.Payload("q-drift", N, 11, 3, POINTS, 2, 0, 0)
        rows = grid.one_draw(payload)
        assert all(np.isfinite(r.p0_curve) for r in rows), [r.error for r in rows]

        stack = remainder.stacked_companion(law, points=128, scrambles=(1, 2))
        fit = grid._fit(payload, stack.frame)
        with pytest.raises(ValueError, match="weight"):
            remainder.corrected_remainder(
                fit, law, stack.weights[:256], remainder.Window.prefix(256)
            )


class TestTheDecompositionIsConditional:
    """E1b's arithmetic, on rows whose answer is known by hand.

    Nothing here fits.  What these check is that the rule's error is the **within-fit**
    spread and the estimator's is what is left of the between-fit one -- rather than one
    marginal variance divided by another, which is what E1 reported and which is identified
    only if both rules' errors are mean-zero given the fit.
    """

    def test_the_rule_variance_is_the_within_fit_spread(self) -> None:
        """Two fits far apart, each with the same tight replicate spread.

        A marginal variance here is dominated by the gap between the fits; the conditional
        one sees only the replicates, which is the whole point.
        """
        values = [[10.0, 11.0, 12.0], [50.0, 51.0, 52.0]]

        decomposed = grid.decompose(values)

        assert decomposed.rule_variance == pytest.approx(1.0)
        assert decomposed.estimator_variance == pytest.approx(800.0 - 1.0 / 3)
        assert decomposed.share < 0.01

    def test_the_estimator_variance_is_corrected_for_the_replicate_count(self) -> None:
        """The between-fit spread carries ``Var(e)/R`` and it is subtracted rather than left.

        Left in, the estimator's variance would be overstated by exactly the amount the
        averaging already removed -- and the share understated to match, which is the
        direction that would flatter the instrument.
        """
        values = [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]

        decomposed = grid.decompose(values)

        # Every fit has mean 1, so nothing is left between fits once Var(e)/R is taken out.
        assert decomposed.rule_variance == pytest.approx(2.0)
        assert decomposed.estimator_variance == pytest.approx(-1.0)
        assert decomposed.replicates == 2

    def test_a_single_replicate_leaves_the_rule_variance_unmeasured(self) -> None:
        """``nan`` rather than zero: one replicate has no spread, which is not no error."""
        decomposed = grid.decompose([[1.0], [2.0], [3.0]])

        assert np.isnan(decomposed.rule_variance)
        assert not np.isnan(decomposed.estimator_variance)

    def test_the_share_is_one_when_the_fits_agree(self) -> None:
        """All of the spread is the rule when the estimator contributes none of it."""
        rng = np.random.default_rng(0)
        values = [list(rng.normal(size=8)) for _ in range(40)]

        assert grid.decompose(values).share > 0.9


class TestEveryShareCarriesAnInterval:
    """A share printed without one is what E1 printed, and the reason it was over-read."""

    def test_the_interval_brackets_the_point_estimate(self) -> None:
        rng = np.random.default_rng(1)
        values = [list(30.0 * rng.normal() + rng.normal(size=6)) for _ in range(30)]

        low, high = grid.bootstrap_share(values)

        assert low <= grid.decompose(values).share <= high

    def test_fewer_draws_widen_it(self) -> None:
        """The draw is the independent unit, so the interval has to answer to how many there are.

        Resampling *replicates* instead would report the conditional error as the whole and
        would barely move when draws are removed -- which is the mistake this pins against.
        """
        rng = np.random.default_rng(2)
        values = [list(3.0 * rng.normal() + rng.normal(size=6)) for _ in range(40)]

        wide = grid.bootstrap_share(values[:6])
        narrow = grid.bootstrap_share(values)

        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


class TestTheTableSaysWhatWasMeasured:
    """Arithmetic on hand-built rows; nothing here fits."""

    def test_the_rows_are_the_width_of_the_headers(self) -> None:
        records = [
            row(data_seed=seed, replicate=rep, points=points, rows=2 * points)
            for seed in range(3)
            for rep in (500, 501)
            for points in (256, 512)
        ] + [
            row(data_seed=seed, replicate=rep, rule="draw", points=0, rows=2_000)
            for seed in range(3)
            for rep in (900, 901)
        ]

        for builder, headers in (
            (grid.grid_rows, grid.GRID_HEADERS),
            (grid.cost_rows, grid.COST_HEADERS),
        ):
            built = builder(records)
            assert built
            assert all(len(r) == len(headers) for r in built), headers

    def test_the_coarsest_rung_has_no_delta_and_the_others_do(self) -> None:
        records = [
            row(
                data_seed=seed,
                replicate=rep,
                points=points,
                rows=2 * points,
                root_n_remaining=1.0 + 0.1 * step,
            )
            for seed in range(3)
            for rep in (500, 501)
            for step, points in enumerate((256, 512))
        ]

        cells = [dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)]

        assert cells[0]["delta"] == "-"
        assert cells[1]["delta"] == f"{0.1:.5f}"

    def test_the_delta_is_paired_within_the_scramble_and_not_only_the_draw(self) -> None:
        """Paired across scrambles it would report the rule's own error as refinement.

        Two scrambles of one draw whose columns are far apart and whose *movement* between
        rungs is small.  Pairing on the draw alone would match a rung of one scramble against
        the other's and read the gap between them; pairing on both reads the movement, which
        is the only one of the two that is about refinement.
        """
        records = [
            row(data_seed=0, replicate=500, points=256, rows=512, root_n_remaining=+5.0),
            row(data_seed=0, replicate=500, points=512, rows=1_024, root_n_remaining=+5.02),
            row(data_seed=0, replicate=501, points=256, rows=512, root_n_remaining=-5.0),
            row(data_seed=0, replicate=501, points=512, rows=1_024, root_n_remaining=-4.98),
        ]

        cells = [dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)]

        assert cells[1]["delta"] == f"{0.02:.5f}"

    def test_the_share_column_is_the_conditional_one(self) -> None:
        """What E5 sizes against, and it is two estimated variances rather than a difference.

        The draw rows here carry a large within-fit spread and the Sobol rows a small one, on
        the same three fits -- so the shares must come out far apart even though neither is
        computed by referring to the other.  E1's column was ``1 - s²/s²_control`` and could
        not be computed for a rule without a control at all.
        """
        rng = np.random.default_rng(3)
        records = []
        for seed in range(12):
            level = 4.0 * rng.normal()
            for rep in range(6):
                records.append(
                    row(
                        data_seed=seed,
                        replicate=500 + rep,
                        points=256,
                        rows=512,
                        root_n_remaining=level + 0.05 * rng.normal(),
                    )
                )
                records.append(
                    row(
                        data_seed=seed,
                        replicate=900 + rep,
                        rule="draw",
                        points=0,
                        rows=2_000,
                        root_n_remaining=level + 8.0 * rng.normal(),
                    )
                )

        cells = {
            r[2]: dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)
        }

        assert float(cells["sobol"]["share"]) < 0.05
        assert float(cells["draw"]["share"]) > 0.7
        assert cells["draw"]["share 90%"].startswith("[")

    def test_a_failed_replicate_is_a_row_rather_than_a_gap(self, monkeypatch) -> None:
        """A ladder with a rung missing looks like a shorter ladder, which is a different claim.

        Driven through the real ``except`` branch rather than a hand-built row: a test that
        only renders a failure it constructed itself pins the table and says nothing about
        whether anything ever produces one.  ``_row`` is made to raise on the coarse rung,
        which is what a binned limit hitting an empty grid would do.
        """
        original = grid._row

        def refuse(payload, fit, stack, block, **kwargs):
            if kwargs["points"] == POINTS[0]:
                raise ValueError("no")
            return original(payload, fit, stack, block, **kwargs)

        monkeypatch.setattr(grid, "_row", refuse)
        records = grid.one_draw(grid.Payload("q-drift", N, 11, 3, POINTS, 1, 0, 0))

        assert [r.error for r in records] == ["ValueError", ""]
        cells = [dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)]
        assert [c["points"] for c in cells] == [f"{p:,}" for p in POINTS]
        assert cells[0]["draws"] == "0"
        assert cells[0]["sqrt(n) R_rem"] == "-"


class TestDeltaIsNotAnErrorBound:
    """The measurement E1b was written on, kept as a test so the claim cannot come back.

    E1 read the movement between two rungs as bounding the rule's error.  On a
    piecewise-smooth integrand -- the kind Tier 2's kernel cutoff produces -- it is not a
    bound in either direction, and this reproduces that at the ladder's own geometry rather
    than asserting it.  ``d = 4`` is ``linear_dgp``'s ``n_latent``; the reference is an
    average over independent scrambles at a far finer grid, whose own error is checked to be
    negligible against the gap being demonstrated.

    Cheap on purpose: it integrates a closure, fits nothing, and runs in well under a second.
    """

    RUNGS = (256, 512, 1_024, 2_048)
    REFERENCE_POINTS = 2**15

    @staticmethod
    def _integrand(latent: np.ndarray) -> np.ndarray:
        """Smooth, plus a jump across a curved surface -- a kernel cutoff's shape."""
        smooth = np.exp(-0.25 * latent[:, 0] ** 2) * (0.5 + 0.3 * latent[:, 1])
        edge = 0.6 * latent[:, 0] - 0.5 * latent[:, 1] + 0.3 * latent[:, 2] * latent[:, 3]
        return smooth + 0.25 * (np.abs(edge) < 0.7)

    def _at(self, dgp, points: int, scramble: int) -> float:
        return float(self._integrand(dgp.quadrature(points, scramble=scramble)).mean())

    def test_the_finest_delta_is_smaller_than_the_error_it_was_read_as_bounding(self) -> None:
        from cleverly.datasets.synthetic import linear_dgp

        dgp = linear_dgp()
        reference = np.array(
            [self._at(dgp, self.REFERENCE_POINTS, 900_000 + s) for s in range(8)], dtype=float
        )
        truth = float(reference.mean())
        # The reference is itself an estimate, so its own error is checked rather than
        # assumed -- otherwise this test could pass on a badly measured truth.
        assert reference.std(ddof=1) / np.sqrt(reference.size) < 1e-4

        fixed = [self._at(dgp, points, 20240101) for points in self.RUNGS]
        delta = abs(fixed[-1] - fixed[-2])
        error = abs(fixed[-1] - truth)

        assert delta < error, (delta, error)

    def test_the_spread_across_scrambles_does_not_understate_it(self) -> None:
        """What replaces the ladder, and the property that makes it the right statistic.

        Independent randomisations of one unbiased rule, so the spread across them is a
        standard deviation of the rule's own error rather than a statement about how a
        sequence moved.  It has to be *at least* the size of a typical fixed-scramble error,
        which is what the ladder was not.
        """
        from cleverly.datasets.synthetic import linear_dgp

        dgp = linear_dgp()
        finest = self.RUNGS[-1]
        reference = float(
            np.mean([self._at(dgp, self.REFERENCE_POINTS, 900_000 + s) for s in range(8)])
        )
        replicates = np.array([self._at(dgp, finest, 1_000 + s) for s in range(16)], dtype=float)

        assert replicates.std(ddof=1) > abs(self._at(dgp, finest, 20240101) - reference)
