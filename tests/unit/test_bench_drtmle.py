"""The sweep instrument's comparison table, at the one column a frozen rule depends on.

``benchmarks/bench_drtmle.py`` is a characterisation and asserts nothing, so nothing in it
is under test here except the arithmetic one *rule* reads.  The update-order rule frozen in
``docs/drtmle/validation-plan.md`` §4 has four clauses, and the fourth is that **no fit in
either arm** fails its score check or its state identity.  Three of the four are answered by
columns that were already there; the fourth was not, because :func:`curve_rows` -- the table
carrying the identity -- filters to the base arm, and :func:`comparison_rows` reported the
variant's ``check fails`` and not its identity.

Every :class:`~benchmarks.bench_drtmle.Exit` already carries a populated ``curve``:
:func:`one_fit` computes it whatever the arm, so the number existed and only the table
dropped it.  What is pinned here is that it no longer does, and that the cell's verdict is
its **worst** fit rather than its typical one -- an identity's right value is zero, so a
median would let one broken fit hide behind eleven sound ones, which is the shape of failure
this column exists to make visible.

There is no other test module for this script and this one does not become one.  The rest of
its tables are read by a human out of a job log; only this column has a predeclared rule
hanging off it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # the benchmarks package is not installed, only checked out
    sys.path.insert(0, str(ROOT))

from benchmarks.bench_drtmle import Curve, Exit, comparison_rows, route_rows  # noqa: E402

#: Where ``worst identity`` sits in a row of :func:`comparison_rows`, counting from zero:
#: ``process, n, pairs, med |dpsi|/se, max |dpsi|/se, med se ratio, se ratio range,
#: check fails, worst identity, med rounds``.
IDENTITY = 8

#: Where the ``<arm> > reseed`` count sits in a row of :func:`route_rows`, counting from
#: zero: ``process, n, pairs, med arm |dpsi|/se, med reseed |dpsi|/se, arm > reseed,
#: mean arm +/- se``.
COUNT = 5


def _fit(variant: str, seed: int, *, identity: float, psi: float = 0.5) -> Exit:
    """One record, with every field a comparison does not read left at a neutral value."""
    return Exit(
        process="nonlinear",
        n=600,
        data_seed=seed,
        fold_seed=seed,
        variant=variant,
        seconds=1.0,
        exit_reason="tolerance",
        rounds=8,
        converged=True,
        failure="",
        ill_conditioned=0,
        closing=1,
        closing_capped=False,
        loop_reduced=1e-11,
        loop_mechanism=1e-11,
        closing_reduced=1e-11,
        scale_reduced=1e-3,
        epsilon_max=1e-6,
        score_ok=True,
        worst_share=1e-7,
        psi=psi,
        se=0.1,
        curve=Curve(identity=identity),
    )


class TestTheComparisonReportsTheVariantArmsIdentity:
    """Clause 4 asks for the identity in *either* arm, so the variant's has to be on a table.

    Delete the ``worst identity`` column from :func:`comparison_rows` and the first two
    assertions below go red, both reading ``8`` -- ``med rounds``, which slides into the
    position.  That was watched rather than assumed, and it is why the constant above spells
    the whole header out: a column inserted anywhere but where the printed header says would
    make this module agree with a table a reader cannot.
    """

    def test_a_broken_variant_fit_is_reported_rather_than_silent(self) -> None:
        """A paper-order fit whose identity fails says so, on the table its arm appears on.

        The base fit is sound and the variant is not, which is exactly the case
        :func:`curve_rows` cannot see: it filters to ``base``, so before this column the
        cell read clean.
        """
        results = [
            _fit("base", 0, identity=2e-19),
            _fit("paper", 0, identity=7e-08),
        ]

        rows = comparison_rows(results, "paper")

        assert len(rows) == 1
        assert float(rows[0][IDENTITY]) == 7e-08

    def test_the_cell_reports_its_worst_fit_and_not_its_typical_one(self) -> None:
        """A max rather than a median, for the reason ``curve_rows`` takes one.

        The identity's right value is zero, so a cell with one failure and two clean fits
        has failed.  A median would report ``2e-19`` here and read as a passing cell.
        """
        results = [_fit("base", seed, identity=2e-19) for seed in (0, 1, 2)] + [
            _fit("paper", 0, identity=2e-19),
            _fit("paper", 1, identity=7e-08),
            _fit("paper", 2, identity=3e-19),
        ]

        rows = comparison_rows(results, "paper")

        assert float(rows[0][IDENTITY]) == 7e-08

    def test_an_arm_with_no_partner_contributes_no_row(self) -> None:
        """Unchanged behaviour, kept here because the column must not create a row.

        A variant fit whose base partner raised has nothing to be a difference from, and
        the table skips the cell rather than reporting an identity against no comparison.
        """
        results = [_fit("paper", 0, identity=7e-08)]

        assert comparison_rows(results, "paper") == []


class TestTheReseedYardstickReadsWhicheverArmItIsGiven:
    """:func:`route_rows` serves two frozen rules, so the arm it reads is an argument.

    §4's update-order rule and §7's cross-fitting rule ask the same *arithmetic* question --
    is this arm's ``|dpsi|/se`` larger than a redrawn split's -- against the same ``reseed``
    control, and differ only in which answer supports which conclusion.  One function with
    an arm argument is what keeps the two rules reading one instrument; hard-coding
    ``"paper"`` inside it, as it was, would have meant a second copy for §7 and two places
    for the count to drift.
    """

    def test_the_count_column_answers_for_the_arm_it_was_asked_about(self) -> None:
        """The `nested` arm moves psi further than the reseed does; `paper` does not.

        Both arms are on the same results, so a builder that ignored its argument would
        report one of these numbers twice -- which is the mutation: hard-code ``"paper"``
        again and the ``nested`` assertion reads ``0/2``.
        """
        results = [_fit("base", seed, identity=0.0, psi=0.5) for seed in (0, 1)]
        results += [_fit("reseed", seed, identity=0.0, psi=0.55) for seed in (0, 1)]
        results += [_fit("paper", seed, identity=0.0, psi=0.51) for seed in (0, 1)]
        results += [_fit("nested", seed, identity=0.0, psi=0.9) for seed in (0, 1)]

        assert route_rows(results, "paper")[0][COUNT] == "0/2"
        assert route_rows(results, "nested")[0][COUNT] == "2/2"

    def test_the_default_arm_is_the_update_orders(self) -> None:
        """§4's call site passes it explicitly now, but the default must not have moved."""
        results = [_fit("base", 0, identity=0.0, psi=0.5)]
        results += [_fit("reseed", 0, identity=0.0, psi=0.55)]
        results += [_fit("paper", 0, identity=0.0, psi=0.51)]

        assert route_rows(results) == route_rows(results, "paper")
