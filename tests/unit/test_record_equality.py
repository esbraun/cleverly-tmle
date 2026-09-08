"""A report row's ``nan`` is a sentinel, so two rows carrying it are the same row.

These tests are about :func:`cleverly.utils.records.sentinel_equality` and deliberately not
about a fit.  The defect it was written for surfaced as a serialisation failure -- a
reloaded ``DRTMLE`` fit's ``correction_check`` rows stopped equalling the ones it was saved
from -- but the round trip was never the subject: it was one way of getting two rows whose
``nan`` fields were separately constructed, and it only reached that state on Python 3.13,
where the generated ``__eq__`` changed from a tuple comparison (identity shortcut, so two
copies of the *default* ``nan`` compared equal) to a chain of per-field ``==`` (no
shortcut, so ``nan != nan`` decides it).

So the tests below construct the two ``nan``\\ s themselves.  That is what makes them fail
on every interpreter when the decorator is removed rather than only on the one whose
codegen changed -- and a test that can only fail on an interpreter the developer is not
running pins nothing.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

import cleverly
from cleverly.utils.records import _DefaultingUnpickle, sentinel_equality
from cleverly.validation import CorrectionRow, ScoreCheckRow


@sentinel_equality
@dataclass(frozen=True)
class Row:
    """A stand-in with the shape the real rows have: a key, a value, a sentinel column."""

    name: str
    value: float
    optional: float = float("nan")
    hidden: float = field(default=0.0, compare=False)


class TestTheSentinelCompares:
    def test_two_separately_built_sentinels_are_equal(self) -> None:
        """The case the generated ``__eq__`` never got right, on any interpreter.

        Both rows carry ``nan`` in ``optional``, and the two are distinct objects -- which
        is what a row built at a call site rather than from the class default looks like.
        Up to 3.12 the tuple comparison's identity shortcut rescued the *default* case and
        this one alike went unnoticed; it is the general statement of the same claim.
        """
        left, right = Row("a", 1.0, float("nan")), Row("a", 1.0, float("nan"))

        assert left.optional is not right.optional
        assert left == right

    def test_a_sentinel_is_not_equal_to_a_number(self) -> None:
        """Not applicable, and measured-and-it-was-zero, are different rows."""
        assert Row("a", 1.0, float("nan")) != Row("a", 1.0, 0.0)

    def test_an_ordinary_field_still_decides(self) -> None:
        """Canonicalising the sentinel must not make everything else compare equal too."""
        assert Row("a", 1.0) != Row("b", 1.0)
        assert Row("a", 1.0) != Row("a", 2.0)

    def test_a_nan_in_a_measured_column_is_a_sentinel_too(self) -> None:
        """There is no per-field opt-in, and that is the intended reading.

        A ``nan`` anywhere in one of these rows means the column did not apply to it --
        every quantity they carry is either finite or absent, since a genuinely
        indeterminate score would be a defect the check exists to report rather than a
        value to compare.
        """
        assert Row("a", float("nan")) == Row("a", float("nan"))

    def test_a_different_class_is_not_equal_rather_than_an_error(self) -> None:
        """``NotImplemented`` rather than a raise, so ``!=`` against anything works."""
        assert Row("a", 1.0) != "a"
        assert Row("a", 1.0) != CorrectionRow(0, 1.0, "1", "D*_Q", 0.0, 0.0, 0, True)

    def test_a_field_marked_uncompared_stays_uncompared(self) -> None:
        """The decorator reads ``field.compare``, as the generated ``__eq__`` does."""
        assert Row("a", 1.0, hidden=1.0) == Row("a", 1.0, hidden=2.0)


class TestTheRowsStayHashable:
    """``ScoreCheck`` and ``CorrectionCheck`` are frozen dataclasses holding these rows.

    Assigning ``__eq__`` unsets a class's ``__hash__``, so a decorator that only replaced
    equality would make every container holding these rows unhashable -- which is the kind
    of breakage that shows up three subsystems away.
    """

    def test_equal_rows_hash_alike(self) -> None:
        assert hash(Row("a", 1.0, float("nan"))) == hash(Row("a", 1.0, float("nan")))

    def test_a_row_can_go_in_a_set(self) -> None:
        assert len({Row("a", 1.0, float("nan")), Row("a", 1.0, float("nan"))}) == 1

    @pytest.mark.parametrize(
        "row",
        [
            CorrectionRow(0, 1.0, "1", "D*_Q", 0.0, 0.0, 0, True),
            ScoreCheckRow("ate", "score", 0.0, 1.0, 1.0, True, True, 1, "newton"),
        ],
        ids=["correction", "score"],
    )
    def test_the_real_rows_hash(self, row: object) -> None:
        assert isinstance(hash(row), int)


class TestTheRealRowsUseIt:
    """Applied where the sentinels are, and the two classes are checked by name.

    Both carry ``nan`` defaults on their optional columns and both are compared with ``==``
    by a test -- ``CorrectionRow`` through ``after.rows == before.rows`` across a save/load,
    ``ScoreCheckRow`` through ``fit.score_verdict.rows == fit.diagnostics.score_equations().rows``
    -- so a class that grew a sentinel column and did not get the decorator would be found
    by a failing round trip rather than here.  This is the cheaper place to notice.
    """

    def test_a_correction_row_with_two_sentinels_is_equal_to_its_twin(self) -> None:
        left = CorrectionRow(0, 1.0, "1", "D*_Q", 1e-14, 1e-14, 0, True, float("nan"), 0.5)
        right = CorrectionRow(0, 1.0, "1", "D*_Q", 1e-14, 1e-14, 0, True, float("nan"), 0.5)

        assert left.clip_bias is not right.clip_bias
        assert left == right

    def test_a_score_row_with_two_sentinels_is_equal_to_its_twin(self) -> None:
        args = ("ate", "score", 1e-14, 1e-3, 0.1, True, True, 3, "newton")
        left = ScoreCheckRow(*args, score_initial=float("nan"))
        right = ScoreCheckRow(*args, score_initial=float("nan"))

        assert left.score_initial is not right.score_initial
        assert left == right

    def test_a_correction_row_still_separates_two_different_fits(self) -> None:
        """The sentinel must not swallow the number the row exists to report."""
        left = CorrectionRow(0, 1.0, "1", "D*_Q", 1e-14, 1e-14, 0, True)
        right = CorrectionRow(0, 1.0, "1", "D*_Q", 1e-14, 2e-14, 0, True)

        assert left != right


# --------------------------------------------------------------------------------------
# The pickle backfill, which is a second list of field names
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Restored(_DefaultingUnpickle):
    """A record with one optional field a stored value has to differ on."""

    name: str
    reason: str | None = None

    _PICKLE_BACKFILL: ClassVar[dict[str, Any]] = {"reason": "the record predates the field"}


@dataclass(frozen=True)
class _Renamed(_DefaultingUnpickle):
    """The same record after the field was renamed and the entry was not."""

    name: str
    omission: str | None = None

    _PICKLE_BACKFILL: ClassVar[dict[str, Any]] = {"reason": "the record predates the field"}


@dataclass(frozen=True)
class _Required(_DefaultingUnpickle):
    """A record naming a field that has no default, so no pickle can be filled with one."""

    name: str
    reason: str

    _PICKLE_BACKFILL: ClassVar[dict[str, Any]] = {"reason": "the record predates the field"}


def _restore(record_class: type, state: dict[str, Any]) -> Any:
    """Run ``__setstate__`` the way ``pickle`` does, on an uninitialised instance."""
    record = object.__new__(record_class)
    record.__setstate__(state)
    return record


class TestThePickleBackfill:
    """What a stale ``_PICKLE_BACKFILL`` does, and what it is made to do instead.

    The map is a second list of field names beside the dataclass's own, so it is the half
    that goes stale, and every way it can go stale is quiet by default.  A key that names
    nothing is skipped and the field takes its own ``None``, which restores a report saying
    it has no score load and no reason for one.  A key that names a field with no default
    used to be filled from the map before anything asked whether the field had a default,
    which would call an unreadable pickle restored.  Both are refused now.
    """

    def test_the_stored_value_replaces_the_constructor_default(self) -> None:
        assert _restore(_Restored, {"name": "row"}).reason == "the record predates the field"

    def test_a_present_field_is_not_backfilled_over(self) -> None:
        assert _restore(_Restored, {"name": "row", "reason": "kept"}).reason == "kept"

    def test_a_key_that_names_no_field_is_refused(self) -> None:
        with pytest.raises(TypeError, match="names no such field: reason"):
            _restore(_Renamed, {"name": "row"})

    def test_a_key_that_names_a_required_field_is_refused(self) -> None:
        with pytest.raises(TypeError, match="names a required field: reason"):
            _restore(_Required, {"name": "row"})

    def test_every_shipped_record_declares_a_backfill_its_fields_still_have(self) -> None:
        """The sweep, so a rename fails here rather than on somebody's stored result.

        The refusals above run at restore time, which is the moment nobody is watching.
        Walking the package makes the same check run in the fast tier, over the classes
        that actually ship, without either list naming the other.
        """
        modules = [cleverly]
        for found in pkgutil.walk_packages(cleverly.__path__, "cleverly."):
            modules.append(importlib.import_module(found.name))
        shipped = {
            obj
            for module in modules
            for obj in vars(module).values()
            if inspect.isclass(obj)
            and issubclass(obj, _DefaultingUnpickle)
            and obj is not _DefaultingUnpickle
            and obj.__module__.startswith("cleverly")
        }

        assert shipped, "no _DefaultingUnpickle subclasses found; check the walk"
        assert any(obj._PICKLE_BACKFILL for obj in shipped), (
            "no shipped record declares a backfill; this sweep would pass vacuously"
        )
        for obj in shipped:
            obj.check_pickle_backfill()
