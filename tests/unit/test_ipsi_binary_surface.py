r"""Does the categorical machinery leave the two-level longitudinal tilt where it was?

T16: the first five replicates of the committed longitudinal incremental studies, refitted
here, equal the committed ``cleverly`` rows.  The rows were written by the code before the
categorical change, so a two-level node must still run Kennedy's arithmetic bit for bit.
Where the committed rows were computed (``EXACT``) the comparison is ``==``.  CI runs on
another platform whose last bits can differ, so it compares at a relative ``1e-12``.
``TestTheBinaryBranchRunsTheOperationsOfTheBinaryCode`` pins the elementwise operations of the
binary branch bit for bit on every platform.

* ``longitudinal-incremental``: the cross-fitted IPSI estimator without dropout.
* ``longitudinal-incremental-dropout``: the same with dropout, and the survival curve.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly.longitudinal import regimen as regimen_module
from tests.studies import canonical_longitudinal_incremental as plain
from tests.studies import canonical_longitudinal_incremental_dropout as dropout

pytestmark = pytest.mark.xdist_group("ipsi_binary_surface")

REPLICATES = 5
#: Bit-exact identities hold on the platform that computed them.
EXACT = os.environ.get("CI") is None
#: An absolute floor for a value that may be zero.
FLOOR = 1e-12
ROOT = "tests/canonical/"
COMMITTED = {
    plain.STUDY.implementation: ROOT + "npcausal_longitudinal_incremental/replicates.csv.gz",
    dropout.STUDY.implementation: ROOT + "longitudinal_incremental_dropout/replicates.csv.gz",
}


def _committed(implementation: str, scenario: str, replicate: int) -> pd.DataFrame:
    rows = pd.read_csv(COMMITTED[implementation], float_precision="round_trip")
    keep = (
        (rows["implementation"] == implementation)
        & (rows["scenario"] == scenario)
        & (rows["replicate"] == replicate)
    )
    assert keep.any(), "the committed artifact has no rows for this replicate"
    return rows[keep].set_index("estimand")


def _agree(result: Any, committed: pd.DataFrame) -> None:
    for name, row in committed.iterrows():
        estimate = result[str(name)]
        if EXACT:
            assert estimate.psi == row["estimate"], name
            assert estimate.std_error == row["std_error"], name
        else:
            assert estimate.psi == pytest.approx(row["estimate"], rel=1e-12, abs=FLOOR), name
            assert estimate.std_error == pytest.approx(row["std_error"], rel=1e-12, abs=FLOOR), name


@pytest.mark.parametrize("replicate", range(REPLICATES))
def test_the_cross_fitted_study_rows_come_back(replicate: int) -> None:
    scenario = "ipsi_folds_5"
    frame, _ = plain.draw_scenario(scenario, plain.PRIMARY_N, replicate)
    _agree(plain.fit(frame, scenario), _committed(plain.STUDY.implementation, scenario, replicate))


@pytest.mark.parametrize("scenario", ["dropout_folds_5", "survival_folds_1"])
@pytest.mark.parametrize("replicate", range(REPLICATES))
def test_the_dropout_study_rows_come_back(scenario: str, replicate: int) -> None:
    frame, _ = dropout.draw_scenario(scenario, dropout.PRIMARY_N, replicate)
    _agree(
        dropout.fit(frame, scenario),
        _committed(dropout.STUDY.implementation, scenario, replicate),
    )


# ----------------------------------------------------------------------------- bitwise witness

DELTAS = [1.7, 0.43, 6.0]


def _two_level_inputs(delta: float) -> tuple[Any, ...]:
    """A running tilt product, arm codes with missing rows, and a two-level mechanism."""
    rng = np.random.default_rng(20261016)
    n = 4000
    arm = rng.integers(0, 2, n).astype(float)
    arm[::37] = np.nan
    pi = rng.uniform(0.05, 0.95, n)
    mechanism = np.column_stack([1.0 - pi, pi])
    running = rng.uniform(0.3, 3.0, n)
    by_arm = rng.normal(size=(n, 2))
    marginal = rng.normal(size=n)
    data = SimpleNamespace(node_codes=lambda time: arm)
    plan = SimpleNamespace(tilt_factors=lambda time: np.array([1.0, delta]))
    return data, plan, running, np.nan_to_num(arm, nan=0.0), pi, mechanism, by_arm, marginal


class TestTheBinaryBranchRunsTheOperationsOfTheBinaryCode:
    """Each assertion is ``np.array_equal`` with the expression the binary code held.

    Elementwise IEEE operations give the same bits on every platform, so these hold in CI.
    ``(a * b) / c`` and ``a * (b / c)`` differ in the last place in a share of entries, which
    the second assertion of the weight test measures; a reordered branch fails the first.
    """

    @pytest.mark.parametrize("delta", DELTAS)
    def test_the_running_tilt_product_keeps_the_order_of_the_binary_code(
        self, delta: float
    ) -> None:
        from cleverly.longitudinal.ipsi import _extend_tilt

        data, plan, running, arm, pi, mechanism, _, _ = _two_level_inputs(delta)
        numerator = delta * arm + 1.0 - arm
        denominator = delta * pi + 1.0 - pi
        main = running * numerator / denominator
        assert np.array_equal(_extend_tilt(running, data, plan, mechanism, 1), main)
        assert np.count_nonzero(running * (numerator / denominator) != main) > 100

    @pytest.mark.parametrize("delta", DELTAS)
    def test_the_mechanism_term_keeps_the_order_of_the_binary_code(self, delta: float) -> None:
        from cleverly.longitudinal.ipsi import _mechanism_term

        data, plan, _, arm, pi, mechanism, by_arm, marginal = _two_level_inputs(delta)
        blip = by_arm[:, 1] - by_arm[:, 0]
        main = delta * blip * (arm - pi) / (delta * pi + 1.0 - pi) ** 2
        assert np.array_equal(_mechanism_term(data, plan, mechanism, by_arm, marginal, 1), main)

    @pytest.mark.parametrize("delta", DELTAS)
    def test_the_tilted_density_keeps_the_order_of_the_binary_code(self, delta: float) -> None:
        _, _, _, _, pi, mechanism, _, _ = _two_level_inputs(delta)
        plan = regimen_module.Plan(
            regimen=None,  # type: ignore[arg-type]
            values=np.zeros((pi.shape[0], 1)),
            node_factors=((1.0, delta),),
        )
        tilted = delta * pi / (delta * pi + 1.0 - pi)
        main = np.column_stack([1.0 - tilted, tilted])
        attached = plan.with_tilts({1: mechanism}).policy[0]
        assert np.array_equal(attached, main)

    def test_a_mapping_that_names_the_first_level_is_the_scalar_multiplier(self) -> None:
        """``{0: 0.5, 1: 3.0}`` reads the odds multiplier ``6.0``, in the binary arithmetic."""
        from cleverly.longitudinal.ipsi import _extend_tilt

        data, _, running, _, _, mechanism, _, _ = _two_level_inputs(6.0)
        mapped = SimpleNamespace(tilt_factors=lambda time: np.array([0.5, 3.0]))
        scalar = SimpleNamespace(tilt_factors=lambda time: np.array([1.0, 6.0]))
        assert np.array_equal(
            _extend_tilt(running, data, mapped, mechanism, 1),
            _extend_tilt(running, data, scalar, mechanism, 1),
        )
