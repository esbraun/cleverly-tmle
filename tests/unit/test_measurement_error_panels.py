"""``bootstrap_measurement_error`` on panels and on strata variables (X82 part E5).

A panel perturbs each declared column at every node where it appears, and a point fit holds
its strata variables at their recorded values. Each rule has a witness here that fails when
the rule is dropped.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.data import CausalData
from cleverly.exceptions import CapabilityError
from cleverly.longitudinal.data import LongitudinalData
from cleverly.validation import refute
from cleverly.validation.refute import (
    BootstrapMeasurementError,
    RelativeGaussianNoise,
    _perturb_measurement_error,
)
from tests.unit.test_longitudinal_generated_outcomes import CONTRAST, _fit, _panel


def _long_panel(n: int = 400, seed: int = 5) -> LongitudinalData:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n):
        w = rng.normal()
        for time in (1, 2, 3):
            rows.append(
                {
                    "id": unit,
                    "t": time,
                    "w": w,
                    "a": int(rng.random() < 0.5),
                    "c": 1.0,
                    # The node scale grows with t, so a single pooled scale is detectable.
                    "l": float(time) * rng.normal(),
                    "y": rng.normal() if time == 3 else np.nan,
                }
            )
    return LongitudinalData.from_long_panel(
        pd.DataFrame(rows),
        unit="id",
        time="t",
        treatment="a",
        baseline=["w"],
        time_varying=["l"],
        censoring="c",
        outcome="y",
    )


@pytest.fixture(scope="module")
def fitted() -> Any:
    return _fit(_panel())


def _run(result: Any, variables: tuple[str, ...], draws: int = 40, estimand: str = CONTRAST) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(
            result,
            estimand=estimand,
            tests=("bootstrap_measurement_error",),
            bootstrap_measurement_error=BootstrapMeasurementError(variables=variables),
            n_replicates=draws,
            random_state=1,
        )["bootstrap_measurement_error"]


class TestPanelPerturbation:
    def test_a_long_panel_column_moves_at_every_node(self) -> None:
        data = _long_panel()
        declaration = BootstrapMeasurementError(
            variables=("l",), numeric_noise=RelativeGaussianNoise(0.5)
        )
        moved = _perturb_measurement_error(data, declaration, seed=3)
        np.testing.assert_array_equal(moved.baseline, data.baseline)
        for time in range(data.n_times):
            before = data.time_varying[time][:, 0]
            change = moved.time_varying[time][:, 0] - before
            # Mutation: perturbing the first appearance only leaves the later nodes still.
            assert np.all(change != 0.0)
            # Each node takes its own scale: 0.5 times that node's standard deviation.
            ratio = float(np.std(change) / (0.5 * np.std(before)))
            assert ratio == pytest.approx(1.0, abs=0.12)

    def test_a_missing_value_stays_missing(self, fitted: Any) -> None:
        data = fitted.data
        declaration = BootstrapMeasurementError(variables=("L2",))
        moved = _perturb_measurement_error(data, declaration, seed=3)
        column = data.time_varying_names[1].index("L2")
        before = data.time_varying[1][:, column]
        after = moved.time_varying[1][:, column]
        missing = ~np.isfinite(before)
        assert np.any(missing) and np.all(np.isnan(after[missing]))
        assert np.all(after[~missing] != before[~missing])
        # The other node covariate and the baseline stay as recorded.
        other = data.time_varying_names[1].index("Y1")
        np.testing.assert_array_equal(
            moved.time_varying[1][:, other], data.time_varying[1][:, other]
        )
        np.testing.assert_array_equal(moved.baseline, data.baseline)

    def test_a_panel_refit_runs_and_names_its_nodes(self, fitted: Any) -> None:
        test = _run(fitted, ("W", "L2"))
        assert len(test.values) == 40 and test.passed is None and test.rung == "L0"
        assert "W at baseline; L2 at node 2" in test.scope_sentence
        assert np.std(test.values) > 0.0


class TestPanelRefusals:
    @pytest.mark.parametrize(
        ("variables", "match"),
        [
            (("A1",), r"unknown covariate 'A1'; choose from the baseline and node covariates"),
            (("Q",), r"unknown covariate 'Q'"),
        ],
    )
    def test_names_outside_the_covariates(
        self, fitted: Any, variables: tuple[str, ...], match: str
    ) -> None:
        with pytest.raises(CapabilityError, match=match):
            _run(fitted, variables)

    def test_a_repeated_outcome(self) -> None:
        repeated = _fit(_panel(), repeated=True)
        with pytest.raises(CapabilityError, match="it is an outcome of this panel"):
            _run(repeated, ("Y1",), estimand="ate_regimen[never vs always @ t=2]")

    def test_a_binary_or_constant_column(self) -> None:
        frame = _panel().assign(B=lambda f: (f["W"] > 0).astype(float), K=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tests.unit.test_longitudinal_generated_outcomes import _ltmle

            result = _ltmle().fit(
                frame,
                outcome="Y2",
                treatment=["A1", "A2"],
                baseline=["W", "B", "K"],
                time_varying=[[], ["Y1", "L2"]],
                censoring=["C1", "C2"],
            )
        if "K" in result.data.baseline_names:
            with pytest.raises(CapabilityError, match="constant"):
                _run(result, ("K",))
        with pytest.raises(CapabilityError, match="all 0 or 1"):
            _run(result, ("B",))


class TestStrata:
    def _data(self) -> CausalData:
        n = 60
        frame = pd.DataFrame(
            {
                "Y": np.linspace(-1.0, 1.0, n),
                "A": np.tile([0, 1], n // 2),
                "numeric": np.arange(n, dtype=float),
                "level": np.tile([1.0, 2.0, 3.0], n // 3),
            }
        )
        return CausalData.from_frame(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("numeric", "level"),
            strata=("level",),
        )

    def test_the_strata_keep_their_recorded_values(self) -> None:
        data = self._data()
        declaration = BootstrapMeasurementError(variables=("numeric", "level"))
        moved = _perturb_measurement_error(data, declaration, seed=2)
        level = data.covariate_names.index("level")
        numeric = data.covariate_names.index("numeric")
        # Mutation: perturbing the stratum column would move each row's stratum value.
        np.testing.assert_array_equal(moved.covariates[:, level], data.covariates[:, level])
        np.testing.assert_array_equal(moved.strata, data.strata)
        assert np.all(moved.covariates[:, numeric] != data.covariates[:, numeric])
