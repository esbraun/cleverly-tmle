"""What saved results can redo, whose summaries the contract test checks (X89 T3).

``replayability(result)`` of a compact point fit and of a longitudinal fit, plus records built
by hand so that every part a result can lack prints, and a record that can redo nothing.
"""

from __future__ import annotations

import warnings
from typing import Any

from cleverly._summary import vocabulary
from cleverly.assessment import (
    POINT_REPLAY_DECLARATION,
    POINT_REPLAY_REFIT_CONFIGURATION,
    Replayability,
    replayability,
)

from . import Case
from . import longitudinal as longitudinal_fits
from ._check_fidelity import header_values, text_tables
from .battery import _fit

__all__ = ["FIXTURES", "read_replay"]


def read_replay(replay: Any, text: str) -> None:
    """Assert that the count and every row printed in ``text`` match ``replay``.

    Parameters
    ----------
    replay : Replayability
        The record.
    text : str
        Its short printed summary.
    """
    slots = {
        "show the stored checks": replay.summarize_existing_artifacts,
        "re-estimate from the stored models without refitting them": (
            replay.retarget_cached_nuisances
        ),
        "evaluate other quantities from the stored weighting function": (
            replay.evaluate_stored_representer
        ),
        "refit the models": replay.refit_nuisances,
        "score new data": replay.evaluate_new_data,
        "rerun the same settings on changed data": replay.rerun_configuration,
    }
    possible = sum(bool(value) for value in slots.values())
    assert header_values(text)[vocabulary.REPLAY_POSSIBLE] == f"{possible} of {len(slots)}"
    [(_, labels, rows)] = text_tables(text)
    assert labels == ["action", "possible"]
    assert [cells[0] for cells in rows] == list(slots)
    for cells in rows:
        assert cells[1] == ("yes" if slots[cells[0]] else "no"), cells


def point() -> Replayability:
    """A point fit through a study."""
    return replayability(_fit())


def longitudinal() -> Replayability:
    """A longitudinal fit, whose models cannot re-estimate without a refit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return replayability(longitudinal_fits.means())


def every_part() -> Replayability:
    """Witness: a record that lacks every part a result can lack."""
    return Replayability(
        True,
        True,
        False,
        False,
        False,
        (
            "estimator configuration",
            POINT_REPLAY_DECLARATION,
            POINT_REPLAY_REFIT_CONFIGURATION,
            "longitudinal_replay_learner_unclonable",
            "longitudinal_replay_random_state_unseeded",
            "longitudinal_replay_random_state_non_integer",
            "longitudinal_replay_fitted_bound_mismatch",
            "longitudinal_replay_regimen_declaration",
            "longitudinal_replay_msm_declaration",
        ),
    )


def everything() -> Replayability:
    """Witness: a record that can redo every action."""
    return Replayability(True, True, True, True, True, rerun_configuration=True)


FIXTURES: dict[str, Case] = {
    f"replay_{name}": Case(build, fidelity=read_replay)
    for name, build in {
        "point": point,
        "longitudinal": longitudinal,
        "every_part": every_part,
        "everything": everything,
    }.items()
}
