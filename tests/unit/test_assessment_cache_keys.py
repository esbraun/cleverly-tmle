"""Different argument objects give different assessment cache keys (X89 plan 6.7).

``_assessment_cache`` keys an operation's arguments by ``repr`` unless the object names its
content through ``_cache_token``. A fitted result's repr is a rounded one-line summary, so two
different results can share it; their keys must still differ.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

from cleverly._assessment_cache import _cache_key
from tests.unit.summary_fixtures.point_tmle import canonical, single_ate


def _key(value: Any) -> str:
    return _cache_key("operation", (value,), {})


def _nudged(result: Any) -> Any:
    """The same result with one estimate moved below the printed precision."""
    estimates = dict(result.estimates)
    estimates["ate"] = dataclasses.replace(estimates["ate"], psi=estimates["ate"].psi + 1e-9)
    return dataclasses.replace(result, estimates=estimates)


def test_two_results_with_one_repr_key_apart() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = single_ate()
    other = _nudged(result)
    assert repr(other) == repr(result)
    assert _key(other) != _key(result)
    assert _key(result) == _key(result)


def test_two_estimates_with_one_repr_key_apart() -> None:
    estimate = canonical().estimates["ate"]
    other = dataclasses.replace(estimate, influence_curve=estimate.influence_curve + 1e-12)
    assert _key(other) != _key(estimate)
    assert _key(dataclasses.replace(estimate)) == _key(estimate)


def test_a_settings_object_still_keys_by_its_repr() -> None:
    config = canonical().config
    assert _key(config) == _key(dataclasses.replace(config))
    assert _key(config) != _key(dataclasses.replace(config, n_folds=config.n_folds + 1))
