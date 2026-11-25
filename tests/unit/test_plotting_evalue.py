"""The E-value figure: the level sets of the bounding factor at the stored risk ratios.

Each test computes the bounding factor ``B(x, y) = x y / (x + y - 1)`` itself and compares it
with the drawn vertices. The square root of a controlled direct effect, the reciprocal below
the null and the choice of limit can each vanish at the truth, so each has a witness that
draws and a mutation of the fold that the consistency guard refuses.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import matplotlib
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cleverly import ATE, CausalStudy, PointTreatment
from cleverly._refusal import RefusalReason
from cleverly.datasets import make_linear_ate
from cleverly.exceptions import CapabilityError
from cleverly.plotting import _evalue as module
from cleverly.plotting import confounding, notes
from cleverly.plotting._core import key_labels, subtitle_text
from cleverly.sensitivity import EValue, evalue_from_rr
from tests.unit._direct_effect_support import binary_cde_frame, fit_cde
from tests.unit._plotting_checks import assert_style, assert_text_inside
from tests.unit.test_evalue_part_g import SKEWED_LAW

pytestmark = pytest.mark.xdist_group("plotting_evalue")


def factor(x: Any, y: Any) -> Any:
    """Ding and VanderWeele's bounding factor, written here apart from the module's copy."""
    return x * y / (x + y - 1.0)


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


def _fit(negate: bool) -> Any:
    frame, _ = make_linear_ate(n=300, seed=1)
    if negate:
        frame["Y"] = -frame["Y"]
    design = PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3", "W4"))
    return (
        CausalStudy(frame, design=design)
        .identify(ATE())
        .estimate(
            outcome_learner=LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            cross_fit=False,
            random_state=0,
        )
    )


@pytest.fixture(scope="module")
def above() -> EValue:
    return _fit(negate=False).sensitivity.evalue()


@pytest.fixture(scope="module")
def below() -> EValue:
    return _fit(negate=True).sensitivity.evalue()


@pytest.fixture(scope="module")
def per_arm() -> EValue:
    fits = fit_cde(binary_cde_frame(), ("ate", "rr", "or", "ey1", "ey0"))
    return fits[0.0].sensitivity.evalue("rr")


def _curve(fig: Any, gid: str) -> np.ndarray[Any, Any]:
    (line,) = [line for line in fig.axes[0].lines if line.get_gid() == gid]
    return np.asarray(line.get_xydata())


def _marks(fig: Any) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for line in fig.axes[0].lines
        if line.get_gid() == module.GID_EVALUE_MARK
        for x, y in line.get_xydata()
    ]


def _has(fig: Any, gid: str) -> bool:
    return any(line.get_gid() == gid for line in fig.axes[0].lines)


class TestTheCurves:
    def test_each_vertex_reaches_the_stored_ratio(self, above: EValue) -> None:
        fig = confounding(above)
        ratio, limit = above.risk_ratio, above.risk_ratio_ci[0]
        estimate = _curve(fig, module.GID_EVALUE_ESTIMATE)
        bound = _curve(fig, module.GID_EVALUE_LIMIT)
        np.testing.assert_allclose(factor(estimate[:, 0], estimate[:, 1]), ratio, rtol=1e-9)
        np.testing.assert_allclose(factor(bound[:, 0], bound[:, 1]), limit, rtol=1e-9)
        assert _marks(fig) == [(above.point, above.point), (above.limit, above.limit)]
        assert factor(above.point, above.point) == pytest.approx(ratio, rel=1e-12)

    def test_every_vertex_lies_inside_the_axes(self, above: EValue) -> None:
        fig = confounding(above, max_strength=10)
        assert fig.axes[0].get_xlim() == (1.0, 10.0) == fig.axes[0].get_ylim()
        for gid in (module.GID_EVALUE_ESTIMATE, module.GID_EVALUE_LIMIT):
            vertices = _curve(fig, gid)
            assert vertices.min() > 1.0 and vertices.max() <= 10.0 * (1 + 1e-12)

    def test_a_ratio_of_two_trades_three_against_four(self, above: EValue) -> None:
        """Witness from the reference: associations 3 and 4 give a bounding factor of 2."""
        two = replace(
            above,
            risk_ratio=2.0,
            risk_ratio_ci=(1.5, 2.5),
            point=evalue_from_rr(2.0),
            limit=evalue_from_rr(1.5),
        )
        vertices = _curve(confounding(two), module.GID_EVALUE_ESTIMATE)
        for x, y in ((3.0, 4.0), (4.0, 3.0)):
            assert np.interp(x, vertices[:, 0], vertices[:, 1]) == pytest.approx(y, abs=1e-3)
        assert two.point == pytest.approx(3.4142, abs=1e-4)

    def test_a_ratio_below_the_null_uses_its_reciprocal(self, below: EValue) -> None:
        assert below.risk_ratio < 1.0
        fig = confounding(below)
        estimate = _curve(fig, module.GID_EVALUE_ESTIMATE)
        bound = _curve(fig, module.GID_EVALUE_LIMIT)
        upper = below.risk_ratio_ci[1]
        np.testing.assert_allclose(
            factor(estimate[:, 0], estimate[:, 1]), 1.0 / below.risk_ratio, rtol=1e-9
        )
        np.testing.assert_allclose(factor(bound[:, 0], bound[:, 1]), 1.0 / upper, rtol=1e-9)
        assert module.BELOW_NULL_NOTE in notes(fig)

    def test_a_controlled_direct_effect_takes_the_square_root(self, per_arm: EValue) -> None:
        assert per_arm.bias_factor == "per_arm_product"
        fig = confounding(per_arm)
        estimate = _curve(fig, module.GID_EVALUE_ESTIMATE)
        np.testing.assert_allclose(
            factor(estimate[:, 0], estimate[:, 1]) ** 2, per_arm.risk_ratio, rtol=1e-9
        )
        assert (per_arm.point, per_arm.point) in _marks(fig)
        assert fig.axes[0].get_xlabel() == "Each arm's treatment association (risk ratio)"

    def test_a_law_that_explains_its_ratio_away_lies_beyond_the_per_arm_curve(
        self, per_arm: EValue
    ) -> None:
        """SKEWED_LAW explains its ratio away with every association at most 9.

        Its pair (9, 9) must lie on or beyond the per-arm curve; a curve without the square
        root would put it inside, saying the law cannot explain its own ratio away.
        """
        bias = SKEWED_LAW.bias
        law = replace(
            per_arm,
            risk_ratio=bias,
            risk_ratio_ci=(bias / 2.0, bias * 2.0),
            point=evalue_from_rr(math.sqrt(bias)),
            limit=evalue_from_rr(math.sqrt(bias / 2.0)),
        )
        vertices = _curve(confounding(law), module.GID_EVALUE_ESTIMATE)
        assert np.interp(9.0, vertices[:, 0], vertices[:, 1]) <= 9.0
        assert factor(9.0, 9.0) >= math.sqrt(bias)
        assert factor(9.0, 9.0) < bias


class TestMutations:
    """A fold without its square root, reciprocal or side is refused by the separate guard."""

    def _patched(self, monkeypatch: pytest.MonkeyPatch, change: Any) -> None:
        real = module._thresholds
        monkeypatch.setattr(module, "_thresholds", lambda data: change(data, *real(data)))

    def test_dropping_the_square_root_is_refused(
        self, per_arm: EValue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patched(monkeypatch, lambda data, t, tl, below: (t * t, tl * tl, below))
        with pytest.raises(ValueError, match="after the per-arm square root"):
            confounding(per_arm)

    def test_dropping_the_reciprocal_is_refused(
        self, below: EValue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patched(monkeypatch, lambda data, t, tl, flag: (1.0 / t, 1.0 / tl, flag))
        with pytest.raises(ValueError, match="this E-value's point value"):
            confounding(below)

    def test_reading_the_far_limit_is_refused(
        self, below: EValue, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        far = 1.0 / below.risk_ratio_ci[0]
        self._patched(monkeypatch, lambda data, t, tl, flag: (t, far, flag))
        with pytest.raises(ValueError, match="limit value"):
            confounding(below)

    @pytest.mark.parametrize("field", ["point", "limit"])
    def test_an_edited_evalue_is_refused(self, above: EValue, field: str) -> None:
        edited = replace(above, **{field: getattr(above, field) + 0.1})
        with pytest.raises(ValueError, match=f"this E-value's {field} value"):
            confounding(edited)

    def test_a_missing_limit_is_refused(self, above: EValue) -> None:
        with pytest.raises(ValueError, match="limit value"):
            confounding(replace(above, limit=float("nan")))


class TestTheLimitAndTheRange:
    def test_a_limit_at_the_null_draws_no_limit_curve(self, above: EValue) -> None:
        reached = replace(above, risk_ratio_ci=(0.9, above.risk_ratio_ci[1]), limit=1.0)
        fig = confounding(reached)
        assert not _has(fig, module.GID_EVALUE_LIMIT)
        assert module.LIMIT_AT_NULL_NOTE.format(level="95%") in notes(fig)
        # Nonzero witness: the unedited E-value draws its limit curve and no such note.
        drawn = confounding(above)
        assert _has(drawn, module.GID_EVALUE_LIMIT)
        assert module.LIMIT_AT_NULL_NOTE.format(level="95%") not in notes(drawn)

    @pytest.mark.parametrize("point", [1.0, float("nan")])
    def test_an_estimate_at_the_null_is_refused(self, above: EValue, point: float) -> None:
        null = replace(above, risk_ratio=1.0, risk_ratio_ci=(0.8, 1.2), point=point, limit=1.0)
        with pytest.raises(CapabilityError, match="there is no curve to draw") as caught:
            confounding(null)
        assert caught.value.refusal.reason is RefusalReason.INVALID_DECLARATION

    @pytest.mark.parametrize("strength", ["below", math.nan, math.inf])
    def test_a_range_that_misses_the_evalue_is_refused(self, above: EValue, strength: Any) -> None:
        value = above.point - 0.01 if strength == "below" else strength
        with pytest.raises(ValueError, match="must be a finite number above the E-value"):
            confounding(above, max_strength=value)

    def test_the_default_range_doubles_the_evalue(self, above: EValue) -> None:
        assert confounding(above).axes[0].get_xlim() == (1.0, math.ceil(2 * above.point))


class TestNotes:
    def test_the_notes_define_the_curve_and_repeat_the_summary(self, above: EValue) -> None:
        recorded = notes(confounding(above))
        assert recorded[0] == module.CURVE_NOTE
        assert all(note.text in recorded for note in above.summary().notes)
        assert not any("experimental" in line.lower() for line in recorded)
        assert above.limit_descriptive is False
        assert above.limit_reading is None or above.limit_reading.sentence not in recorded

    def test_a_per_arm_evalue_says_what_it_reads(self, per_arm: EValue) -> None:
        recorded = notes(confounding(per_arm))
        assert recorded[0] == module.PER_ARM_CURVE_NOTE
        assert "cell ratio" in recorded[0]
        codes = {note.code for note in per_arm.summary().notes}
        assert {"experimental", "limits_descriptive", "per_arm_conditions"} <= codes
        assert all(note.text in recorded for note in per_arm.summary().notes)
        assert per_arm.limit_reading is not None
        assert per_arm.limit_reading.sentence in recorded

    def test_an_experimental_estimate_says_so(self, above: EValue, per_arm: EValue) -> None:
        """Mutation control: the support of an experimental fit brings its note."""
        flagged = replace(above, support=per_arm.support)
        text = next(n.text for n in flagged.summary().notes if n.code == "experimental")
        assert text in notes(confounding(flagged))


class TestConventions:
    def test_title_subtitle_axes_and_key(self, above: EValue) -> None:
        fig = confounding(above)
        ax = fig.axes[0]
        assert ax.get_title(loc="left") == module.TITLE
        assert subtitle_text(ax) is not None
        assert "E-values 3.07 for the estimate and 2.67 for the 95% limit" in " ".join(
            (subtitle_text(ax) or "").split()
        )
        assert ax.get_xlabel() == "Association with treatment (risk ratio)"
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == [
            "Could explain the estimate away",
            "Could move the 95% limit to the null",
            "Equal strengths",
        ]

    def test_style_and_layout(self, above: EValue, per_arm: EValue) -> None:
        for value in (above, per_arm):
            fig = confounding(value)
            assert_style(fig)
            assert_text_inside(fig)

    def test_dispatch_and_plot_draw_the_same(self, above: EValue) -> None:
        first, second = confounding(above), above.plot()
        for gid in (module.GID_EVALUE_ESTIMATE, module.GID_EVALUE_LIMIT):
            np.testing.assert_array_equal(_curve(first, gid), _curve(second, gid))

    def test_the_formatting_keywords_apply(self, above: EValue) -> None:
        fig = above.plot(title="Mine", xlabel="Link to the offer", ylabel="Link to the score")
        ax = fig.axes[0]
        assert ax.get_title(loc="left") == "Mine"
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Link to the offer", "Link to the score")
        right = above.plot(legend="right")
        assert "Could explain the estimate away" in key_labels(right)
        assert_text_inside(right)

    def test_a_supplied_axes_is_drawn_on(self, above: EValue) -> None:
        fig, ax = plt.subplots()
        assert confounding(above, ax=ax) is fig
        assert _has(fig, module.GID_EVALUE_ESTIMATE)

    def test_png_render(self, above: EValue, tmp_path: Any) -> None:
        path = tmp_path / "evalue.png"
        confounding(above).savefig(path, dpi=60)
        assert path.stat().st_size > 1000

    def test_a_contour_keyword_is_not_taken(self, above: EValue) -> None:
        with pytest.raises(TypeError):
            confounding(above, robustness=None)
