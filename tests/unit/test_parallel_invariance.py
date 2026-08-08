r"""``n_jobs`` schedules work; it must not change the answer.

Every parallel surface in the package takes an ``n_jobs`` and defaults it to one --
:class:`~cleverly.TMLE`, :class:`~cleverly.LTMLE`,
:class:`~cleverly.validation.CoverageStudy`, the targeted bootstrap -- and each of them
hands the same closure to :func:`cleverly.utils.parallel.map_parallel`, which runs a list
comprehension at ``n_jobs=1`` and a joblib ``Parallel`` above it.  Two code paths, one
claimed result.

**Nothing checked that claim until this module.**  ``n_jobs`` appears in the suite only as
``n_jobs=2`` throughout :mod:`tests.e2e.test_coverage_slow`, where it is a runtime setting
on a study whose output is a coverage rate with a Monte Carlo error around it -- so a
scheduling bug that perturbed a fit would have to be large to show there, and a small one
never would.  That is the gap: the setting is used where it cannot be audited and audited
nowhere else.

It matters now because the ``docs`` tier is about to *depend* on it.  Forty minutes of that
tier is fold-parallel work running one fold at a time, and the fix is to raise ``n_jobs``
for the run -- which is only legitimate if this holds.  If it fails, that is not a reason to
loosen this test; it is a bug in fold scheduling and the injection does not happen.

**Bit for bit, not to a tolerance.**  The folds are seeded and each fold's nuisance fit sees
only its own training rows, so the parallel and serial paths do arithmetic in the same order
on the same arrays.  Anything short of exact equality here would mean the fold assignment or
the accumulation order moved with the worker count, and a tolerance would be a way of not
noticing.

**Verified by mutation, and the result was not what it was written to be** -- which is worth
recording, because it says where the safety actually lives.  Reversing the payload order
inside ``map_parallel``'s parallel branch turns
:meth:`TestMapParallel.test_both_branches_return_the_same_list_in_order` red and leaves
**every** fit-level test in :class:`TestOneFit` green.

The fit absorbs it, by construction rather than by luck.
:func:`cleverly.estimators._nuisance.cross_fit_predictions` scatters each fold's
predictions with ``out[name][test] = values`` -- by the fold's own row indices, not by
position -- and re-sorts the companion slabs through ``argsort(order)``, with a comment
saying why: "a slab indexed by position would silently answer for the wrong fold if it ever
stopped doing so".  So fold *ordering* is already not a way to get a wrong answer here, and
a test claiming to catch it would be claiming credit for that code.

What :class:`TestOneFit` is for is therefore the other half, and it is the half no
construction rules out: a fit that comes back *different* because it ran in a worker --
state that does not survive pickling, an RNG re-seeded per process, a learner reading
something global.  Nothing in the reassembly protects against that, and it is exactly what
would go unnoticed, since the result would still be a plausible fit.  The two classes fail
on disjoint mistakes, which is the argument for keeping both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cleverly import TMLE
from cleverly.datasets import make_nonlinear_ate
from cleverly.utils.parallel import map_parallel, resolve_n_jobs

#: Two is enough and four is not better.  What is being checked is that the parallel branch
#: of ``map_parallel`` is taken at all and agrees with the serial one; the branch does not
#: know how many workers it has, so a wider sweep would buy a longer runtime and no further
#: coverage.  ``n_folds`` is above the worker count so there is real work to distribute.
PARALLEL_JOBS = 2


def _fit(n_jobs: int) -> Any:
    return (
        TMLE(
            estimands=("ate", "att", "ey1", "ey0"),
            outcome_learner="glm",
            treatment_learner="glm",
            n_folds=5,
            learner_folds=3,
            random_state=0,
            simultaneous=False,
            n_jobs=n_jobs,
        )
        .fit(FRAME, outcome="Y", treatment="A")
        .single()
    )


FRAME, _TRUTH = make_nonlinear_ate(n=600, seed=0)


class TestOneFit:
    """The same fit, scheduled two ways."""

    @pytest.fixture(scope="class")
    def pair(self) -> tuple[Any, Any]:
        # Module-scoped so the two fits are paid for once across every test below; they are
        # the whole cost of this module and the fast tier's budget is the reason it is
        # ``glm`` at n=600 rather than anything the guide would show.
        return _fit(1), _fit(PARALLEL_JOBS)

    def test_the_point_estimates_are_identical(self, pair: tuple[Any, Any]) -> None:
        serial, parallel = pair
        assert set(serial.estimates) == set(parallel.estimates)
        for name, estimate in serial.estimates.items():
            other = parallel.estimates[name]
            assert estimate.psi == other.psi, name
            assert estimate.variance == other.variance, name

    def test_the_influence_curves_are_identical(self, pair: tuple[Any, Any]) -> None:
        """The array everything downstream reads, so the one that has to match exactly.

        ``rtol=0`` and ``atol=0``: the bands, the delta method, the cluster-robust variance
        and the score diagnostic all read the curve rather than the variance, so a curve
        that moved with the worker count would move all four while each stayed internally
        consistent -- the shape of drift that is hardest to notice.
        """
        serial, parallel = pair
        for name, curve in serial.influence_curves.items():
            np.testing.assert_allclose(curve, parallel.influence_curves[name], rtol=0, atol=0)

    def test_the_folds_are_the_same_assignment(self, pair: tuple[Any, Any]) -> None:
        """Necessary and *not* sufficient, which is why it is not the only test here.

        The digest is of the realised fold assignment.  Scheduling those folds across two
        workers instead of one does not change the assignment, so this passing says the
        split was reproducible and says nothing about whether the results came back in the
        right order.  The two tests above are what say that.
        """
        serial, parallel = pair
        assert serial.provenance.fold_fingerprint == parallel.provenance.fold_fingerprint
        assert serial.provenance.data_fingerprint == parallel.provenance.data_fingerprint

    def test_the_nuisance_predictions_are_identical(self, pair: tuple[Any, Any]) -> None:
        """One level below the estimate, where a fold-ordering bug would first appear."""
        serial, parallel = pair
        np.testing.assert_allclose(
            serial.nuisance.propensity.values,
            parallel.nuisance.propensity.values,
            rtol=0,
            atol=0,
        )
        for arm, values in serial.nuisance.outcome.arms.items():
            np.testing.assert_allclose(values, parallel.nuisance.outcome.arms[arm], rtol=0, atol=0)


class TestMapParallel:
    """The helper itself, without paying for a fit.

    Cheap enough to state the properties directly, and worth stating: every caller in the
    package relies on them and none of them is obvious from the call site.
    """

    def test_both_branches_return_the_same_list_in_order(self) -> None:
        payloads = list(range(20))
        serial = map_parallel(lambda value: value * value, payloads, n_jobs=1)
        parallel = map_parallel(lambda value: value * value, payloads, n_jobs=PARALLEL_JOBS)
        assert serial == parallel == [value * value for value in payloads]

    def test_a_single_payload_never_starts_a_pool(self) -> None:
        """The short circuit at ``len(items) <= 1``, which is load-bearing for cost.

        A one-fold fit -- ``n_folds=1``, the in-sample path -- would otherwise pay joblib's
        process start-up to run one closure.
        """
        assert map_parallel(lambda value: value + 1, [41], n_jobs=8) == [42]

    @pytest.mark.parametrize("requested,expected", [(None, 1), (1, 1), (4, 4), (-1, -1)])
    def test_resolve_normalises_the_sklearn_convention(
        self, requested: int | None, expected: int
    ) -> None:
        assert resolve_n_jobs(requested) == expected

    def test_zero_is_refused_rather_than_read_as_serial(self) -> None:
        """``0`` is the value that would otherwise mean whatever joblib decided that day."""
        with pytest.raises(ValueError, match="positive integer"):
            resolve_n_jobs(0)
