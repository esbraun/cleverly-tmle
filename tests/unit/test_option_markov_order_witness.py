"""Exact-law witnesses of ``option-markov-order`` (X76 part (f), Fix-8 plan section 4).

``LongitudinalData.with_markov_order(k)`` makes every nuisance at node ``t`` read ``W``, the
covariates ``L_{t-k}, ..., L_t`` and the arms ``A_{t-k-1}, ..., A_{t-1}``.  The witnesses fit
``LTMLE`` with probability weights on the quadrature law of
:func:`tests.studies._option_markov_law.quadrature_frame` (Gauss-Legendre nodes in ``W`` times
every path of the binary nodes), so the weighted empirical law is the law and every fit returns
its population limit.  Under law ``LF1`` the treatment reads ``L_{t-1}`` and the outcome reads
``L_3``, so order 1 is the smallest correct window:

* the order-1, order-2 and full-history fits return the truth to ``1e-8``;
* the order-0 fit misses the ATE by more than ``1e-3``: history beyond the order matters, and
  the window drops it;
* under law ``LF2`` the treatment reads only ``A_{t-1}``, which the order-0 arm block keeps, so
  the order-0 mechanisms are correct while its outcome window is not, and the fit returns the
  truth: double robustness, with the window applied to the outcome side alone.

The mutation controls shorten the covariate window by one node, and start the arm block at the
window's first covariate node instead of one node before it.  The first fails the order-1
witness and the second fails the double-robustness witness.

The support tests read the key: a windowed fit records ``markov_order=True`` and matches no
shipped row until ``G57`` registers.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import textwrap
import warnings
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from cleverly.longitudinal import LongitudinalData, SequentialGComputation
from tests.studies import _option_markov_law as law

data_module = importlib.import_module("cleverly.longitudinal.data")

EXACT = 1e-8
FLOOR = 1e-3


@functools.cache
def cached_limit(name: str) -> dict[str, law.ExactLimit]:
    return law.exact_limit(name)


def estimates(name: str, *, cache: bool = True) -> dict[str, float]:
    limits = cached_limit(name) if cache else law.exact_limit(name)
    return {estimand: limit.estimate for estimand, limit in limits.items()}


def check_order_one_witness() -> None:
    truth = law.truth()
    for estimand, value in estimates("lf1_order1", cache=False).items():
        assert value == pytest.approx(truth[estimand], abs=EXACT), estimand


def check_double_robustness_witness() -> None:
    truth = law.truth()
    for estimand, value in estimates("lf2_order0", cache=False).items():
        assert value == pytest.approx(truth[estimand], abs=EXACT), estimand


def mutated(owner: type, name: str, old: str, new: str) -> Any:
    """``owner.name`` recompiled with ``old`` replaced by ``new`` once in its source."""
    source = textwrap.dedent(inspect.getsource(getattr(owner, name)))
    assert source.count(old) == 1, f"{name} no longer holds {old!r}"
    namespace: dict[str, Any] = {}
    exec(
        compile(source.replace(old, new), f"<mutated {name}>", "exec"), vars(data_module), namespace
    )
    return namespace[name]


class TestTheQuadratureLaw:
    def test_the_weights_are_a_probability_law(self) -> None:
        for name in law.LAWS:
            frame = law.quadrature_frame(name)
            assert frame["weight"].sum() == pytest.approx(1.0, abs=1e-12)
            assert (frame["weight"] > 0).all()

    def test_the_two_laws_differ_only_in_the_lagged_covariate_term(self) -> None:
        lf1, lf2 = law.TREATMENT["LF1"], law.TREATMENT["LF2"]
        assert lf1[:4] == lf2[:4]
        assert lf1[4] != 0.0
        assert lf2[4] == 0.0


class TestTheWindowWitness:
    @pytest.mark.parametrize("name", ["lf1_order1", "lf1_order2", "lf1_full"])
    def test_a_correct_window_returns_the_truth(self, name: str) -> None:
        truth = law.truth()
        for estimand, value in estimates(name).items():
            assert value == pytest.approx(truth[estimand], abs=EXACT), estimand

    def test_the_order_zero_window_misses_the_truth(self) -> None:
        truth = law.truth()
        values = estimates("lf1_order0")
        assert abs(values[law.ATE] - truth[law.ATE]) > FLOOR
        assert abs(values[law.EY_ALWAYS] - truth[law.EY_ALWAYS]) > FLOOR

    def test_the_order_zero_window_is_rescued_by_correct_mechanisms(self) -> None:
        truth = law.truth()
        for estimand, value in estimates("lf2_order0").items():
            assert value == pytest.approx(truth[estimand], abs=EXACT), estimand

    def test_the_order_zero_window_misses_the_initial_outcome_under_lf2(self) -> None:
        """The rescue is the mechanisms': the order-0 outcome regressions alone are wrong.

        The sequential regression estimator reads the same windowed outcome designs without the
        mechanisms; on the full history it returns the truth, so the miss is the window's.
        """
        frame = law.quadrature_frame("LF2")
        truth = law.truth()
        for order, wrong in ((0, True), (None, False)):
            estimator = SequentialGComputation(
                dict(law.REGIMENS),
                reference=law.REFERENCE,
                outcome_learner=LinearRegression(),
                pseudo_learner=LinearRegression(),
                n_folds=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = estimator.fit(law.longitudinal_data(frame, order, weights="weight"))
            for estimand in law.ESTIMANDS:
                error = abs(result.estimates[estimand].psi - truth[estimand])
                assert (error > FLOOR) if wrong else (error < EXACT), (order, estimand)

    def test_mutation_shortening_the_window_fails_the_order_one_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        check_order_one_witness()
        short = mutated(
            LongitudinalData,
            "window_start",
            "time - self.markov_order",
            "time - self.markov_order + 1",
        )
        monkeypatch.setattr(LongitudinalData, "window_start", short)
        with pytest.raises(AssertionError):
            check_order_one_witness()

    def test_mutation_dropping_the_preceding_arm_fails_the_robustness_witness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        check_double_robustness_witness()
        late = mutated(
            LongitudinalData,
            "history_design",
            "max(1, self.window_start(time) - 1)",
            "self.window_start(time)",
        )
        monkeypatch.setattr(LongitudinalData, "history_design", late)
        with pytest.raises(AssertionError):
            check_double_robustness_witness()


class TestTheWindowDesign:
    def test_the_mechanism_design_holds_the_window(self) -> None:
        frame = law.draw("LF1", 200, 0)
        full = law.longitudinal_data(frame, None)
        for order in (0, 1, 2):
            data = law.longitudinal_data(frame, order)
            for time in range(1, law.T + 1):
                start = max(1, time - order)
                expected = np.hstack(
                    [
                        full.baseline,
                        *full.time_varying[start - 1 : time],
                        *(
                            np.nan_to_num(full.treatment[:, [node - 1]], nan=0.0)
                            for node in range(max(1, start - 1), time)
                        ),
                    ]
                )
                design = data.history_design(time)
                assert design.shape == expected.shape, (order, time)
                assert np.allclose(np.nan_to_num(design, nan=0.0), np.nan_to_num(expected)), (
                    order,
                    time,
                )


class TestTheSupportKey:
    def test_a_windowed_fit_is_keyed_off_every_shipped_row(self) -> None:
        """At registration this asserts ``G57``."""
        frame = law.draw("LF1", 400, 0)
        windowed = law.fit(frame, "lf1_order1").estimates[law.ATE].support
        assert windowed.label == "experimental"
        assert windowed.row is None
        assert "markov_order=True" in windowed.reason
        full = law.fit(frame, "lf1_full").estimates[law.ATE].support
        assert "markov_order" not in (full.reason or "")
