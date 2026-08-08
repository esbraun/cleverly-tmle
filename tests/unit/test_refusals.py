r"""The refusals, their taxonomy, and the fact that all three still agree.

``docs/methodology.md``'s *How to read a refusal* is one of the most load-bearing pages in
the documentation set, because it tells a reader **where the problem is** -- in this
package, in the question, or in the method -- and therefore what to do about it.  It is
also, until this module, three tables of prose that nothing checked.  A refusal can be
reworded, lifted, or quietly start working, and those tables would go on saying otherwise
in a register that reads as authoritative.

Three artefacts have to agree and each can move without the others:

* the **message** raised in ``src``, which is what a user actually reads;
* the **kind** it is filed under, which is what tells them whether to ask for the feature,
  re-ask their question, or reconsider their analysis;
* the **section** in ``docs/methodology.md`` that defines that kind.

So this module does not restate the messages -- restating them is how a copy drifts.  It
holds a ledger of *which refusal is which kind*, reads the message out of the live source
table, and checks the kinds against the headings the documentation really defines.

**The bidirectional half is exact**, and it is exact because ``src`` carries a
machine-readable registry to check against:
:data:`cleverly.longitudinal.estimator._REFUSED` is the one place a refusal is a *row* and
not a ``raise`` site.  Every key there needs a row here and every row naming a keyword needs
a live key, so lifting a refusal without reclassifying it fails, and classifying one that no
longer exists fails too.  Refusals raised inline elsewhere are covered by the 200-odd
``pytest.raises`` tests already in this suite; what none of those pin is the taxonomy, which
is what this is for.

**Verified by mutation**: deleting the ``eliminate`` row turned the forward direction red;
adding a row for ``n_folds``, which ``LTMLE`` accepts, turned the reverse red *and* the
source-string check with it; and renaming the *Wrong by construction* heading in
``docs/methodology.md`` turned :func:`test_every_kind_is_one_the_documentation_defines`
red.

**What no mutation here catches is a row filed under the wrong kind** -- moving
``cross_fit`` from a redirection to *wrong by construction* leaves every test green, since
both are things a row is allowed to be.  That is not an oversight to fix later: which of
the three a refusal *is* is a judgement about where the problem lies, and there is nothing
in the code to check it against.  What this module can do, and does, is make sure the kind
is one the documentation defines, that the message the user reads is the one ``src`` holds,
and that the set of refusals and the set of rows are the same set.  The judgement itself
stays a review question, and saying so is better than a check that appears to make it one.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import pytest

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.estimator import _REFUSED

ROOT = Path(__file__).resolve().parents[2]
METHODOLOGY = ROOT / "docs" / "methodology.md"

#: The kind for a keyword that is accepted and rejected only because it is declared
#: somewhere *else* -- ``weights=`` is a column of the data, a survival outcome is the shape
#: of ``outcome=``.  ``docs/methodology.md`` names three kinds and this is not one of them,
#: deliberately: it is not a refusal at all but a redirection, and filing it under one of the
#: three would tell a reader the feature is missing when it is one argument away.  It is
#: named here rather than added to the page because the page is about what a refusal *means*,
#: and a redirection means nothing except "you are holding it wrong".
REDIRECTION = "declared elsewhere"


def documented_kinds() -> set[str]:
    """The kinds ``docs/methodology.md`` defines, read off its taxonomy table.

    Read rather than written down for the usual reason: a heading renamed in the document
    and copied here by hand would leave the two agreeing about a word and disagreeing about
    which section it points at.  The taxonomy table's first column is a link per kind, so
    the link text *is* the kind and the anchor is checkable.
    """
    text = METHODOLOGY.read_text(encoding="utf-8")
    section = text.split("## How to read a refusal", 1)
    assert len(section) == 2, "docs/methodology.md has no 'How to read a refusal' section"
    kinds = {
        label.strip().lower(): anchor
        for label, anchor in re.findall(r"\|\s*\[([^\]]+)\]\(#([\w-]+)\)", section[1][:2000])
    }
    assert kinds, "the taxonomy table names no kinds"
    for label, anchor in kinds.items():
        assert f"\n### {label.title().replace(' A ', ' a ')}" in text or any(
            heading.strip().lower() == label for heading in re.findall(r"\n### (.+)", text)
        ), f"the taxonomy links to #{anchor} but no '### {label}' heading exists"
    return set(kinds)


@dataclasses.dataclass(frozen=True)
class Row:
    """One refusal, and which of the kinds it is."""

    #: The keyword, as the user writes it.  For a longitudinal row this is a key of
    #: ``_REFUSED``, which is what makes the coverage check exact.
    keyword: str
    kind: str
    #: What triggers it.  Kept as a callable rather than as a settings dict so a row for a
    #: refusal that is not a keyword of ``LTMLE`` can join the ledger unchanged.
    call: Any
    #: The exception the user sees.
    error: type[BaseException] = TypeError
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
    Row("intermediate", "a different question", lambda: _ltmle(intermediate=1)),
    Row("interventions", "a different question", lambda: _ltmle(interventions=1)),
    Row("shifts", "a different question", lambda: _ltmle(shifts=1)),
    Row("incremental", "a different question", lambda: _ltmle(incremental=1)),
    Row("delta", "a different question", lambda: _ltmle(delta=1)),
    Row("eliminate", "a different question", lambda: _ltmle(eliminate=1)),
    Row("n_bootstrap", "not written yet", lambda: _ltmle(n_bootstrap=1)),
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
        f"refusal was lifted, the row goes with it -- and docs/methodology.md's table "
        f"needs the same edit"
    )


def test_every_kind_is_one_the_documentation_defines() -> None:
    documented = documented_kinds()
    allowed = documented | {REDIRECTION}
    wrong = sorted({row.kind for row in LEDGER} - allowed)
    assert wrong == [], (
        f"rows filed under kinds docs/methodology.md does not define: {wrong}; it defines "
        f"{sorted(documented)}, and this module adds {REDIRECTION!r} for a keyword that is "
        f"not refused at all but declared elsewhere"
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
    reason = _REFUSED[row.keyword]
    assert reason[:40] in message, (
        f"{row.keyword}= raises a message that is not the reason _REFUSED states:\n"
        f"  raised: {message}\n  table:  {reason}"
    )
    assert row.keyword in message, f"{row.keyword}= refuses without naming itself: {message}"


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

    ``CLAUDE.md``'s rule is to prefer refusing "with a message that says what the derivation
    would need" over quietly reporting the wrong thing.  A one-line refusal satisfies the
    letter of that and not the point of it.
    """
    # The floor is 60 rather than a rounder number because ``shifts=`` sets it: "a shift
    # moves a continuous dose, and a longitudinal fit takes a binary treatment at every
    # node" is 94 characters and is a complete reason -- it names the mismatch and the
    # thing that would have to change. A floor above that would be asking for words rather
    # than for content, which is the failure mode of a length check.
    reason = _REFUSED[row.keyword]
    assert len(reason) > 60, (
        f"{row.keyword}= is refused in {len(reason)} characters. Filed as {row.kind!r}, so "
        f"the message has to leave a reader able to tell that from the other two kinds"
    )
