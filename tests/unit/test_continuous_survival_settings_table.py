"""The settings table on ``continuous-time-survival.md`` is the engine's signature.

``## Settings`` lists every argument of :class:`~cleverly.survival.ContinuousSurvivalTMLE`
once, with its default, where a ``CausalStudy`` route takes it from, and whether a value other
whether the support label reads it as configuration. The first two columns are read from the signature,
the route column from :func:`cleverly.survival.data.continuous_kwargs` and the keywords
``CausalStudy`` passes from the estimand, and the label column from
:data:`cleverly.survival.result.MEASURED_DEFAULTS` and the other inputs of
:func:`cleverly.survival.result.continuous_survival_support`.
"""

from __future__ import annotations

import ast
import inspect
import re
import textwrap
from pathlib import Path
from typing import Any

from cleverly import TMLEMethod
from cleverly.methods import Inference, Targeting
from cleverly.survival import ContinuousSurvivalTMLE
from cleverly.survival.data import continuous_kwargs
from cleverly.survival.result import MEASURED_DEFAULTS, continuous_survival_support

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "technical-reference" / "continuous-time-survival.md"
HEADING = "## Settings"

#: The arguments that reach the label through an input of ``continuous_survival_support`` other
#: than ``settings``: the default nuisances, ``cross_fitted``, the restricted horizons, and the
#: event share at each target time.  ``causes`` only labels the observed codes, every one of
#: which must be named, so the cause count the label reads does not move with it.
OTHER_LABEL_INPUTS = frozenset(
    {
        "hazard_library",
        "censoring_library",
        "treatment_learner",
        "cross_fit",
        "restricted_horizons",
        "horizons",
    }
)


def _default(value: Any) -> str:
    if value is inspect.Parameter.empty:
        return "required"
    if isinstance(value, str):
        return f'`"{value}"`'
    return f"`{value!r}`"


def _estimand_keywords() -> set[str]:
    """The keywords ``_continuous_engine`` passes from the estimand, read from its source."""
    import cleverly.study as study

    source = textwrap.dedent(inspect.getsource(study))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_continuous_engine":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == (
                    "ContinuousSurvivalTMLE"
                ):
                    named = {kw.arg for kw in call.keywords if kw.arg is not None}
                    return named | {"horizons"}
    raise AssertionError("CausalStudy._continuous_engine builds no ContinuousSurvivalTMLE")


#: The inputs of ``continuous_survival_support``.  A new input may read a new argument, so it
#: fails here until the table and :data:`OTHER_LABEL_INPUTS` account for it.
SUPPORT_INPUTS = frozenset(
    {
        "scale",
        "event_share",
        "n",
        "causes",
        "cross_fitted",
        "default_nuisances",
        "settings",
        "restricted",
        "restricted_horizons",
        "variance_rel_sd",
        "past_last_time",
        "jump_bound_pairs",
    }
)


def _optional_keywords() -> set[str]:
    """The keywords ``continuous_kwargs`` passes only when the method sets them, from its source."""
    from cleverly.survival import data

    source = inspect.getsource(data.continuous_kwargs)
    match = re.search(r"optional = \{.*?\n    \}", source, re.S)
    assert match is not None, "continuous_kwargs builds no optional mapping"
    return set(re.findall(r'\("(\w+)", \w+\.\w+\)', match.group(0)))


def _method_keywords() -> set[str]:
    """The keywords ``continuous_kwargs`` reads from a method, ``norm`` and ``variance`` included.

    Those two are passed only when the method sets them, so the method here sets both.
    """
    method = TMLEMethod(
        targeting=Targeting(norm="variance"), inference=Inference(variance="influence_curve")
    )
    keywords = set(continuous_kwargs(method))
    assert _optional_keywords() <= keywords, "a new optional keyword needs a value set here"
    return keywords


def expected_rows() -> list[list[str]]:
    """``[setting, default, study route, moves the label]`` per argument, in signature order."""
    method = _method_keywords()
    estimand = _estimand_keywords()
    label = set(MEASURED_DEFAULTS) | OTHER_LABEL_INPUTS
    rows = []
    for name, parameter in inspect.signature(ContinuousSurvivalTMLE.__init__).parameters.items():
        if name == "self":
            continue
        route = "the method" if name in method else "the estimand" if name in estimand else "none"
        rows.append(
            [f"`{name}`", _default(parameter.default), route, "yes" if name in label else "no"]
        )
    return rows


def _page_rows() -> list[list[str]]:
    text = PAGE.read_text(encoding="utf-8")
    section = text.split(HEADING + "\n", 1)[1].split("\n#", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("| `")]
    return [[cell.strip() for cell in line.strip().split("|")[1:-1]][:4] for line in lines]


def test_the_page_table_is_the_signature() -> None:
    assert _page_rows() == expected_rows()


def test_each_measured_default_moves_the_label() -> None:
    """A nonzero witness: changing any one measured setting makes a supported fit experimental."""
    base = {
        "scale": "level",
        "event_share": 0.6,
        "n": 1_000,
        "causes": 2,
        "cross_fitted": True,
        "default_nuisances": True,
    }
    assert continuous_survival_support(settings=MEASURED_DEFAULTS, **base).label == "supported"
    for key in MEASURED_DEFAULTS:
        moved = dict(MEASURED_DEFAULTS, **{key: object()})
        assert continuous_survival_support(settings=moved, **base).label == "experimental", key
    assert (
        continuous_survival_support(
            settings=MEASURED_DEFAULTS, **dict(base, default_nuisances=False)
        ).label
        == "experimental"
    )


def test_the_label_inputs_are_known() -> None:
    assert set(inspect.signature(continuous_survival_support).parameters) == SUPPORT_INPUTS


def test_the_route_column_names_real_keywords() -> None:
    signature = set(inspect.signature(ContinuousSurvivalTMLE.__init__).parameters)
    assert _method_keywords() <= signature
    assert _estimand_keywords() <= signature
