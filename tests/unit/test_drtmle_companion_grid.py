r"""What the ladder harness reports, and the one thing in it that is not arithmetic.

``benchmarks/drtmle_companion_grid.py`` is a characterisation and asserts nothing, so almost
nothing in it is under test here.  What is under test is **the part a reading rests on**,
which is the same rule ``tests/unit/test_drtmle_coverage.py`` is written under:

* **a rung is a prefix and not a refit.**  The whole design -- one fit, every grid read off it
  -- rests on that, and the identity itself is pinned in
  ``tests/unit/test_drtmle_remainder_study.py`` against an actual refit.  What is pinned here
  is that this module *uses* it: two fits per draw whatever the ladder's length, and the whole
  weight vector passed with ``limit`` doing the slicing rather than a pre-sliced one, which
  the integrators' own row-count check would reject.
* **``delta`` is paired within the draw.**  Unpaired it would carry the between-draw spread
  the pairing exists to remove, and would then read as a grid error that is mostly sampling
  noise -- which is the confusion the whole module was written to end.
* **the table's rows are the width of its headers**, the structural pin against the one
  mistake that produces a complete, plausible, wrong table rather than a failure.
* **a failed rung is a row and not a gap**, since a ladder with a rung missing looks like a
  shorter ladder.
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
    """A hand-built rung, so every table below can be tested without fitting anything."""
    defaults: dict[str, object] = {
        "cell": "q-drift",
        "n": 600,
        "data_seed": 1,
        "fold_seed": 2,
        "rule": "sobol",
        "points": 1_024,
        "rows": 2_048,
        "p0_curve": -0.11,
        "remaining": 0.017,
        "root_n_remaining": 0.42,
        "companion_se": 1.19,
        "companion_halving": 0.008,
        "branch_q": -0.001,
        "branch_g": 0.019,
        "branch_error": 0.002,
        "seconds": 4.7,
    }
    return grid.GridRow(**{**defaults, **overrides})  # type: ignore[arg-type]


class TestTheLadderIsOneFit:
    """Two fits per draw whatever the ladder's length, which is the module's whole design."""

    def test_every_rung_and_the_control_come_off_one_draw(self, monkeypatch) -> None:
        """Counted rather than argued: a refit per rung would show up here as four."""
        fits: list[int] = []
        original = grid._fit

        def counted(payload, evaluation):
            fits.append(1)
            return original(payload, evaluation)

        monkeypatch.setattr(grid, "_fit", counted)
        payload = grid.Payload("q-drift", N, 11, 3, POINTS, 256)

        rows = grid.one_draw(payload)

        assert len(fits) == 2
        assert [r.points for r in rows if r.rule == "sobol"] == sorted(POINTS)
        assert [r.rows for r in rows if r.rule == "sobol"] == [2 * p for p in sorted(POINTS)]
        assert [r.rule for r in rows][-1] == "draw"

    def test_a_coarse_rung_is_read_at_the_full_weight_vector(self) -> None:
        """The integrators check their weights against the companion's row count.

        Slicing the vector before handing it over is exactly the stale-weights mistake that
        check exists to catch, so a coarse rung has to pass the whole vector and let
        ``limit`` do the work.  This is the negative control for that: a pre-sliced vector
        raises rather than silently integrating against the wrong measure.
        """
        law = grid.injection.base_law()
        payload = grid.Payload("q-drift", N, 11, 3, (64, 128), 0)
        rows = grid.one_draw(payload)
        assert all(np.isfinite(r.p0_curve) for r in rows), [r.error for r in rows]

        frame, weights = remainder.quadrature_frame(law, 128)
        fit = grid._fit(payload, frame)
        with pytest.raises(ValueError, match="weight"):
            remainder.corrected_remainder(fit, law, weights[:128], 128)


class TestTheTableSaysWhatTheLadderMeasured:
    """Arithmetic on hand-built rungs; nothing here fits."""

    def test_the_rows_are_the_width_of_the_headers(self) -> None:
        records = [
            row(data_seed=seed, points=points, rows=2 * points)
            for seed in range(3)
            for points in (256, 512)
        ] + [row(data_seed=seed, rule="draw", points=0, rows=2_000) for seed in range(3)]

        for builder, headers in (
            (grid.grid_rows, grid.GRID_HEADERS),
            (grid.cost_rows, grid.COST_HEADERS),
        ):
            built = builder(records)
            assert built
            assert all(len(r) == len(headers) for r in built), headers

    def test_the_coarsest_rung_has_no_delta_and_the_others_do(self) -> None:
        records = [
            row(data_seed=seed, points=points, rows=2 * points, root_n_remaining=1.0 + 0.1 * step)
            for seed in range(3)
            for step, points in enumerate((256, 512))
        ]

        cells = [dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)]

        assert cells[0]["delta"] == "-"
        assert cells[1]["delta"] == f"{0.1:.5f}"

    def test_the_delta_is_paired_within_the_draw(self) -> None:
        """Unpaired it would report the between-draw spread as the grid's error.

        Two draws whose columns are far apart and whose *movement* between rungs is small:
        an unpaired difference of means would read near zero here by cancellation, and an
        unpaired mean of absolute differences would read the spread.  The paired one reads
        the movement, which is the only one of the three that is the quadrature.
        """
        records = [
            row(data_seed=0, points=256, rows=512, root_n_remaining=+5.0),
            row(data_seed=0, points=512, rows=1_024, root_n_remaining=+5.02),
            row(data_seed=1, points=256, rows=512, root_n_remaining=-5.0),
            row(data_seed=1, points=512, rows=1_024, root_n_remaining=-4.98),
        ]

        cells = [dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)]

        assert cells[1]["delta"] == f"{0.02:.5f}"

    def test_the_variance_removed_is_measured_against_the_control_and_not_the_witness(
        self,
    ) -> None:
        """What E5 sizes against, and it is two measured spreads rather than a modelled one.

        The witness-based version of this column read **above one** on every control row of
        the first sweep, which is not a share of anything: halving a noise-dominated rule
        doubles a variance, so the halving witness reads about ``1.4x`` the standard error it
        stands in for.  A difference of two spreads on the same draws needs no such model.
        """
        records = [
            row(data_seed=seed, root_n_remaining=value)
            for seed, value in enumerate((0.0, 2.0, 4.0))
        ] + [
            row(data_seed=seed, rule="draw", points=0, rows=2_000, root_n_remaining=value)
            for seed, value in enumerate((0.0, 4.0, 8.0))
        ]

        cells = [dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)]

        # sd 2 against the control's sd 4, so three quarters of the variance is the rule's.
        assert cells[0]["spread"] == "2.0000"
        assert cells[0]["var removed"] == "0.750"
        assert cells[1]["var removed"] == "-"

    def test_the_witness_is_not_read_as_a_share(self) -> None:
        """The control row prints its witness and claims nothing about it.

        A ``rule err`` above ``spread`` is a real reading of an inflated witness rather than a
        rule accounting for more than the whole variance, and the table has to be able to
        print the first without implying the second.
        """
        records = [
            row(data_seed=seed, rule="draw", points=0, rows=2_000, root_n_remaining=value)
            for seed, value in enumerate((0.0, 1.0, 2.0))
        ]

        (cell,) = [dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)]

        assert float(cell["rule err"]) > 0.0
        assert cell["var removed"] == "-"

    def test_a_failed_rung_is_a_row_rather_than_a_gap(self, monkeypatch) -> None:
        """A ladder with a rung missing looks like a shorter ladder, which is a different claim.

        Driven through the real ``except`` branch rather than a hand-built row: a test that
        only renders a failure it constructed itself pins the table and says nothing about
        whether anything ever produces one.  ``_row`` is made to raise on the coarse rung,
        which is what a binned limit hitting an empty grid would do.
        """
        original = grid._row

        def refuse(payload, fit, **kwargs):
            if kwargs["points"] == POINTS[0]:
                raise ValueError("no")
            return original(payload, fit, **kwargs)

        monkeypatch.setattr(grid, "_row", refuse)
        records = grid.one_draw(grid.Payload("q-drift", N, 11, 3, POINTS, 0))

        assert [r.error for r in records] == ["ValueError", ""]
        cells = [dict(zip(grid.GRID_HEADERS, r, strict=True)) for r in grid.grid_rows(records)]
        assert [c["points"] for c in cells] == [f"{p:,}" for p in POINTS]
        assert cells[0]["draws"] == "0"
        assert cells[0]["sqrt(n) R_rem"] == "-"
