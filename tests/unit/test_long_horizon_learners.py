"""The test learners of the long-horizon studies read the right node and the right columns.

:class:`~tests.studies.long_horizon_common.KnownHorizonMechanism` must return the law's arm
probabilities from the engine's treatment design, and its two-arm mutation must change exactly
the column it names.  :class:`~tests.studies.long_horizon_common.NodePattern` must route each
node's fit to its own learner by the design width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests import discrete_law_longitudinal_horizon as H
from tests.studies import long_horizon_common as C


def _treatment_design(frame: pd.DataFrame, t: int, m: int) -> np.ndarray:
    """``[W, L_2..L_t]`` then one drop-first block per earlier arm, on the rows at node ``t``."""
    columns = [frame["W"].to_numpy(dtype=float)]
    columns += [frame[f"L{s}"].to_numpy(dtype=float) for s in range(2, t + 1)]
    for s in range(1, t):
        codes = frame[f"A{s}"].to_numpy(dtype=float)
        columns += [(codes == k).astype(float) for k in range(1, m)]
    return np.column_stack(columns)


@pytest.mark.parametrize(("K", "m"), [(3, 2), (3, 3), (5, 2), (5, 3)])
def test_the_known_mechanism_is_the_law_at_every_node(K: int, m: int) -> None:
    """At every node and both arm counts, the learner returns ``conditionals().g`` row by row."""
    frame = H.weighted_frame(K, m)
    support = H.support(K, m)
    mechanism = H.conditionals(K, m, H.L_R)
    learner = C.KnownHorizonMechanism(K, m, H.L_R).fit(None, None)
    np.testing.assert_array_equal(learner.classes_, np.arange(m, dtype=float))
    for t in range(1, K + 1):
        reached = frame[f"A{t}"].notna().to_numpy()
        got = learner.predict_proba(_treatment_design(frame.loc[reached], t, m))
        want = mechanism.g[t][support.path[reached, t - 1]]
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-15)
        np.testing.assert_allclose(got.sum(axis=1), 1.0, rtol=0, atol=1e-12)
        # The witness: a swap of two arm columns is a different mechanism on these rows.
        assert float(np.max(np.abs(got[:, 0] - got[:, -1]))) > 1e-2


def test_the_mechanism_refuses_a_width_past_the_horizon() -> None:
    learner = C.KnownHorizonMechanism(3, 3, H.L_R).fit(None, None)
    with pytest.raises(ValueError, match="past node 3"):
        learner.predict_proba(np.zeros((2, 1 + 3 * 3)))


def test_the_complement_changes_only_the_middle_arm_at_its_nodes() -> None:
    K, m = 4, 3
    frame = H.weighted_frame(K, m)
    known = C.KnownHorizonMechanism(K, m, H.L_R).fit(None, None)
    mutated = C.HorizonComplementMechanism(K, m, H.L_R, nodes=(2,)).fit(None, None)
    for t in range(1, K + 1):
        reached = frame[f"A{t}"].notna().to_numpy()
        design = _treatment_design(frame.loc[reached], t, m)
        left, right = known.predict_proba(design), mutated.predict_proba(design)
        if t != 2:
            np.testing.assert_array_equal(left, right)
            continue
        np.testing.assert_array_equal(left[:, [0, 2]], right[:, [0, 2]])
        np.testing.assert_allclose(right[:, 1], 1.0 - left[:, 2], rtol=0, atol=1e-15)
        assert float(np.max(np.abs(right.sum(axis=1) - 1.0))) > 1e-2


def test_node_pattern_routes_each_width_to_its_node() -> None:
    """Node 2 of a ``mix`` design gets its own learner; every other node gets the default."""
    K, m = 3, 3
    policy = (1, 2, 3)
    pattern = C.NodePattern({2: C.InterceptOnly()}, None, m, K, policy)
    rng = np.random.default_rng(0)
    for t in range(1, K + 1):
        width = t + (m - 1) * t
        X = rng.integers(0, 2, size=(60, width)).astype(float)
        y = rng.uniform(size=60)
        fitted = pattern.fit(X, y).fitted_
        assert isinstance(fitted, C.InterceptOnly) == (t == 2)
    with pytest.raises(ValueError, match="outcome width"):
        pattern.fit(np.zeros((4, 4)), np.zeros(4))
