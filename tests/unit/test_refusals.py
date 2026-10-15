r"""The refusals, their reason codes, and the fact that the source and the ledger agree.

Every refusal carries a reason code, a :class:`~cleverly.exceptions.RefusalReason`, because the
code tells a reader **where the problem is** and therefore what to do about it: ask for the
feature, re-ask the question, or reconsider the declaration. The message and the code are one
record, ``error.refusal``. This module holds a ledger of *which refusal has which code*, reads
the message out of the live source table, and checks the code on the error a user receives.

So this module does not restate the messages, because restating them is how a copy drifts. It
checks three things:

* the **code** on the raised error is the one the ledger records;
* every **kind** the ledger uses is a value of :class:`~cleverly.exceptions.RefusalReason`;
* the **set** of refusals and the set of rows are the same set.

**The bidirectional half is exact**, and it is exact because ``src`` carries a
machine-readable registry to check against:
:data:`cleverly.longitudinal.estimator._REFUSED` is the one place a refusal is a *row* and
not a ``raise`` site.  Every key there needs a row here and every row naming a keyword needs
a live key, so lifting a refusal without reclassifying it fails, and classifying one that no
longer exists fails too.  Refusals raised inline elsewhere are covered by the 200-odd
``pytest.raises`` tests already in this suite.

**What no mutation here catches is a row filed under the wrong code**: moving ``cross_fit``
from a redirection to a different estimand leaves the source-string checks green, since both
are things a row is allowed to be. The code follows
``docs/technical-reference/scope-and-refusals.md``, and which code a refusal has is a judgement
about where the problem lies. This module makes sure the raised error carries the code the
ledger states, and that the ledger and the source hold the same keywords.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from cleverly.datasets import make_longitudinal
from cleverly.exceptions import CapabilityError, RefusalReason
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.estimator import _REFUSED
from tests.unit.test_pooled_longitudinal_targeting import _linear_settings
from tests.unit.test_sequential_design import multivalue_panel

#: The code of a keyword that is accepted and rejected only because it is declared somewhere
#: *else*: ``weights=`` is a column of the data, a survival outcome is the shape of
#: ``outcome=``. It is not a missing feature but a redirection, and the message has to name the
#: route.
REDIRECTION = RefusalReason.API_REDIRECTION.value


@dataclasses.dataclass(frozen=True)
class Row:
    """One refusal, and which of the kinds it is."""

    #: The keyword, as the user writes it.  For a longitudinal row this is a key of
    #: ``_REFUSED``, which is what makes the coverage check exact.
    keyword: str
    #: The expected :class:`~cleverly.exceptions.RefusalReason` value of the raised error.
    kind: str
    #: What triggers it.  Kept as a callable rather than as a settings dict so a row for a
    #: refusal that is not a keyword of ``LTMLE`` can join the ledger unchanged.
    call: Any
    #: The exception the user sees.
    error: type[BaseException] = CapabilityError
    #: For a redirection, the thing that *does* work -- checked to be mentioned in the
    #: message, so a redirection cannot point nowhere.
    instead: str | None = None


def _ltmle(**settings: Any) -> Any:
    """Trigger a refusal at construction, where ``refuse_unsupported`` sees the settings."""
    return LTMLE({"always": 1, "never": 0}, **settings)


def _ltmle_fit(**columns: Any) -> Any:
    """Trigger one at ``fit`` instead.

    ``msm=`` is the row that needs this and the reason is worth stating: it is a *real*
    constructor parameter of ``LTMLE`` -- a working model over regimens is supported -- so
    the constructor's type check fires first and the refusal is never reached there.  What
    ``_REFUSED["msm"]`` answers is passing it to ``fit`` beside the columns, which is where
    someone coming from the point-treatment path would put it.  A ledger that triggered
    every row the same way would record that refusal as unreachable and be wrong.
    """
    frame, _ = make_longitudinal(n=50, seed=0)
    return LTMLE({"always": 1, "never": 0}).fit(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        **columns,
    )


#: Every refusal ``LTMLE`` states by name, and what kind each is.  The reasons are *not*
#: copied here -- they are read from ``_REFUSED`` at check time.
LEDGER: tuple[Row, ...] = (
    Row("weights", REDIRECTION, lambda: _ltmle(weights=1), instead="fit"),
    Row("msm", REDIRECTION, lambda: _ltmle_fit(msm=1), instead="MSM"),
    Row("event", REDIRECTION, lambda: _ltmle(event=1), instead="outcome"),
    Row("competing", REDIRECTION, lambda: _ltmle(competing=1), instead="outcome"),
    Row(
        "intermediate",
        RefusalReason.DIFFERENT_ESTIMAND.value,
        lambda: _ltmle(intermediate=1),
        instead="TMLE",
    ),
    Row(
        "interventions",
        REDIRECTION,
        lambda: _ltmle(interventions=1),
        instead="regimens=",
    ),
    Row("policies", REDIRECTION, lambda: _ltmle(policies=1), instead="regimens="),
    Row("incremental", REDIRECTION, lambda: _ltmle(incremental=1), instead="IPSI"),
    Row("delta", REDIRECTION, lambda: _ltmle(delta=1), instead="censoring"),
    Row(
        "eliminate",
        RefusalReason.DIFFERENT_ESTIMAND.value,
        lambda: _ltmle(eliminate=1),
        instead="cause-specific",
    ),
    Row("cross_fit", REDIRECTION, lambda: _ltmle(cross_fit=1), instead="n_folds"),
)


def test_the_ledger_is_not_empty() -> None:
    assert len(LEDGER) >= len(_REFUSED) >= 5


def test_every_refused_keyword_has_a_row() -> None:
    """Forward: a keyword refused in ``src`` and unclassified here is one nobody filed."""
    rowed = {row.keyword for row in LEDGER}
    missing = sorted(set(_REFUSED) - rowed)
    assert missing == [], (
        f"keywords in _REFUSED with no row here: {missing}. A refusal a reader cannot place "
        f"in the taxonomy is one they cannot act on -- ask for it, re-ask the question, or "
        f"reconsider the analysis are three different responses"
    )


def test_every_row_names_a_live_keyword() -> None:
    """Reverse: a row for a keyword nothing refuses is a taxonomy of a fiction."""
    stale = sorted({row.keyword for row in LEDGER} - set(_REFUSED))
    assert stale == [], (
        f"rows classifying keywords that _REFUSED no longer carries: {stale}. If the "
        f"refusal was lifted, the row goes with it -- and scope-and-refusals.md's table "
        f"needs the same edit"
    )


def test_every_kind_is_a_refusal_reason() -> None:
    allowed = {reason.value for reason in RefusalReason}
    wrong = sorted({row.kind for row in LEDGER} - allowed)
    assert wrong == [], (
        f"rows filed under kinds that are not a RefusalReason value: {wrong}; "
        f"the values are {sorted(allowed)}"
    )


@pytest.mark.parametrize("row", LEDGER, ids=lambda row: row.keyword)
def test_every_row_raises_the_reason_the_source_states(row: Row) -> None:
    """The message a user reads is the one ``_REFUSED`` holds, not a copy of it.

    This is the check that catches a reason reworded in ``src`` while the ledger, the
    documentation table and the reader's mental model all go on describing the old one.
    """
    with pytest.raises(row.error) as raised:
        row.call()
    message = str(raised.value)
    refusal = _REFUSED[row.keyword]
    assert refusal.condition[:40] in message, (
        f"{row.keyword}= raises a message that is not the reason _REFUSED states:\n"
        f"  raised: {message}\n  table:  {refusal.condition}"
    )
    assert row.keyword in message, f"{row.keyword}= refuses without naming itself: {message}"
    assert isinstance(raised.value, CapabilityError), type(raised.value)
    assert raised.value.refusal.reason.value == row.kind, (
        f"{row.keyword}= raises code {raised.value.refusal.reason.value!r}, "
        f"and the ledger files it under {row.kind!r}"
    )
    assert refusal.reason.value == row.kind, (
        f"_REFUSED[{row.keyword!r}] holds code {refusal.reason.value!r}, and the ledger files "
        f"it under {row.kind!r}"
    )


@pytest.mark.parametrize(
    "row", [row for row in LEDGER if row.kind == REDIRECTION], ids=lambda row: row.keyword
)
def test_a_redirection_says_where_to_go_instead(row: Row) -> None:
    """A redirection that does not name the route is a refusal wearing the wrong label."""
    assert row.instead is not None, f"{row.keyword} is a redirection with nowhere to go"
    with pytest.raises(row.error) as raised:
        row.call()
    assert row.instead in str(raised.value), (
        f"{row.keyword}= is filed as {REDIRECTION!r} but its message does not mention "
        f"{row.instead!r}, so a reader is told the feature is missing rather than moved: "
        f"{raised.value}"
    )


@pytest.mark.parametrize(
    "row", [row for row in LEDGER if row.kind != REDIRECTION], ids=lambda row: row.keyword
)
def test_a_real_refusal_says_what_the_derivation_would_need(row: Row) -> None:
    """The length floor, which is the cheapest proxy for "it gives a reason".

    A useful refusal says what the derivation would need instead of quietly reporting the
    wrong thing.  A one-line refusal satisfies the
    letter of that and not the point of it.
    """
    # The floor is 60. A complete reason names the mismatch and the thing that would have
    # to change, and every reason in the table runs well past the floor. A floor near
    # their length would ask for words rather than for content, which is the failure
    # mode of a length check.
    text = _REFUSED[row.keyword].message()
    assert len(text) > 60, (
        f"{row.keyword}= is refused in {len(text)} characters. Filed as {row.kind!r}, so "
        f"the message has to leave a reader able to tell that from the other codes"
    )


def test_the_policies_redirection_names_the_plan_node_form() -> None:
    """``policies=`` on ``LTMLE`` points to the plan-node form and the continuous nodes.

    A longitudinal modified treatment policy is a node of a regimen, so the keyword the
    point-treatment fit takes is a redirection here, not a missing feature.
    """
    with pytest.raises(CapabilityError) as raised:
        LTMLE({"always": 1, "never": 0}, policies=1)
    message = str(raised.value)
    for needed in ("node of a regimen", "DynamicRegimen", "continuous_treatment="):
        assert needed in message, (needed, message)
    assert "not written yet" not in message, message


def test_ltmle_fits_the_third_label_of_a_categorical_node() -> None:
    """The fit the ``policies=`` reason names: a regimen that assigns a node's third label."""
    settings = _linear_settings(reference="never")
    assert settings["n_folds"] == 3
    frame = multivalue_panel(n=600, seed=8)
    result = LTMLE({"third": (2, 0), "never": 0}, **settings).fit(
        frame,
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1",),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    )
    diagnostics = result.diagnostics.support().to_frame()
    keys = zip(diagnostics["regimen"], diagnostics["time"], strict=True)
    shares = dict(zip(keys, diagnostics["assigned_shares"], strict=True))
    entries = [entry.split("=") for entry in shares[("third", 1)].split(", ")]
    assert len(entries) == 3, shares[("third", 1)]
    assert entries[2][1] == "1", shares[("third", 1)]
    assert np.isfinite(result["ey_regimen[third]"].psi)
