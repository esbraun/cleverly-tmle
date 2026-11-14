# Plotting

Use these functions after estimation to look at a result. Each draws numbers the object
already holds and returns the matplotlib `Figure`, so you can save it, restyle it, or add to it.
A plot never draws an interval or band the fit did not compute.

The figure itself carries no notes. `cleverly.plotting.notes(fig)` returns what a reader of that
fit may need, such as why an interval is missing or which quantities are descriptive. It also
says whether the estimate or its band is
[experimental](../user-guide/reading-output.md#labels-warnings-and-notes), so that its interval
has no checked coverage. The figure does not mark an experimental result, so read the notes, or
the result's `summary()`, before you publish a figure. Each function's documentation says what
its notes cover.

Every plot takes the same formatting keywords. `title=` and `subtitle=` replace the short default
title and the subtitle under it, which names the estimate and its headline number; `False`
removes either. `xlabel=` and `ylabel=` rename the axes: a string renames every panel that shares
the first panel's label, and a list gives each labelled panel its own. `legend=` puts the key
above the axes (`"top"`, the default), to their right (`"right"`), under the figure
(`"bottom"`), or nowhere (`False`). `figsize=` sets the size in inches of a figure the plot
creates, and `ax=` draws a one-panel plot on your own axes.

```python
from cleverly.plotting import forest

fig = forest(result, title="Effect of the offer", xlabel="Change in score", legend="right")
fig.savefig("effect.png", bbox_inches="tight")
```

On simulated data, pass the known true values as `truth=` to `forest` or `curve`. Each is drawn
as a cross beside its estimate, or as a line along a conditional effect curve, and the key names
it "True value (simulated data only)".

matplotlib is optional. Install it with `pip install "cleverly[plot]"`.

The [plot results](../examples/plots.ipynb) tutorial shows the main figures on simulated data.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.plotting.forest
   cleverly.plotting.curve
   cleverly.plotting.sensitivity
   cleverly.plotting.confounding
   cleverly.plotting.overlap
   cleverly.plotting.targeting
   cleverly.plotting.weights
   cleverly.plotting.balance
   cleverly.plotting.zipper
   cleverly.plotting.calibration
   cleverly.plotting.learners
   cleverly.plotting.repeats
   cleverly.plotting.bootstrap
   cleverly.plotting.influence
   cleverly.plotting.notes
   cleverly.plotting.style
```

The results that have one natural figure also draw it with `plot()`:
`TMLEResult.plot()` and `TMLEResultSet.plot()` call `forest`, and `TruncationCurve.plot()`,
`MarginalSensitivityBounds.plot()` and `MissingnessTilt.plot()` call `sensitivity`, and
`SensitivityContour.plot()` and `SimulatedConfoundingResult.plot()` call `confounding`.
`TargetingCurve.plot()` calls `targeting`.
`CovariateBalance.plot()` calls `balance`, and `StudyResult.plot()` calls `zipper`.
`CurveCalibration.plot()` and `NuisanceModelReport.plot()` call `calibration`,
`SuperLearnerDiagnostics.plot()` and `CTMLESelection.plot()` call `learners`, and
`BootstrapResult.plot()` calls `bootstrap`.

`LongitudinalResult.plot()` draws its curve over time with `curve`, or the estimates with
`forest` when the fit has no curve, and `EffectCurveResult.plot()` draws the conditional effect
over a modifier with `curve`.

## Fits over time and with external data

`overlap`, `learners` and `calibration` also draw fits whose treatment is set at more than one
time point, or whose rows come from a trial and an external source.

- A survival fit of a treatment given once at baseline draws, in `overlap`, each person's
  predicted chance of each arm. The fit's probability limits apply to the product of that chance
  and the chance of staying in follow-up, so `overlap(result, censoring=True)` adds that product
  for each plan, with the limit drawn. A collaborative fit draws each arm's selected treatment
  model, and `learners` draws each arm's path of added terms with the chosen one marked.
- A fit with a treatment at several time points draws one panel per time point. `overlap`
  shows the chance of the plan's treatment among the units still following the plan, split by
  whether they then took it; `censoring=True` adds each time point's chance of staying in the
  study. `learners` stacks each model's Super Learner weights by time point, and `calibration`
  draws each probability model at each time point. `regimen=`, `nodes=`, `horizon=` and
  `cause=` choose what is drawn. An incremental or modified treatment policy has no single
  chance to draw, so `overlap` refuses it and `weights` draws its weights.
- A structural nested mean fit draws, in `overlap`, each time point's chance of each treatment
  level, with the margin from 0 and 1 that its positivity summary counts.
- An adaptive fit with external data draws, in `overlap`, each trial patient's chance of
  treatment and each row's chance of being a trial row, with the floor its positivity warning
  checks. Its nuisance report keeps learner names only, so `learners` and `calibration` refuse.

A continuous-time survival fit does not yet draw `curve`, `overlap`, `weights` or `learners`,
and covariate balance over time is not drawn for any fit over time. Each refusal names the
[roadmap](../roadmap.md) item that tracks it.
