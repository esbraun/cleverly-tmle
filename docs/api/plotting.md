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
`SensitivityContour.plot()`, `SimulatedConfoundingResult.plot()` and `EValue.plot()` call
`confounding`. For an E-value it draws the pairs of confounder strengths, on the risk-ratio scale,
that could explain the estimate away. A second curve holds the pairs that could move the interval
limit nearest the null to the null, and each E-value sits where the two strengths are equal. On a
time-to-event surface its outcome axis reads the strength as a log-odds shift of each time point's
event chance on a survival panel, and as a log-hazard shift of each event type on a fit with exact
times, with one benchmark mark for each event type.
`TargetingCurve.plot()` calls `targeting`.
`CovariateBalance.plot()` and `BalanceOverTime.plot()` call `balance`, and `StudyResult.plot()`
calls `zipper`.
`CurveCalibration.plot()` and `NuisanceModelReport.plot()` call `calibration`,
`SuperLearnerDiagnostics.plot()` and `CTMLESelection.plot()` call `learners`, and
`BootstrapResult.plot()` calls `bootstrap`.

`LongitudinalResult.plot()` draws its curve over time with `curve`, or the estimates with
`forest` when the fit has no curve, `ContinuousSurvivalResult.plot()` draws its risks at the
target times with `curve`, and `EffectCurveResult.plot()` draws the conditional effect over a
modifier with `curve`.

## Large data

Four figures can draw one mark per row, per replication or per covariate. With large data that
crowds the marks together and makes a saved vector file (SVG or PDF) large. Each has a keyword
that limits what is drawn and nothing else: the estimates, intervals, shares and counts the
figure states are computed from all the data either way, and `cleverly.plotting.notes(fig)`
says how many marks were drawn out of how many.

| keyword | what it limits | what stays exact |
| --- | --- | --- |
| `influence(..., max_points=)` | the points the line passes through, spread evenly along it, both ends and the top-1% point kept | the ranking, the top share and the Kish count (how many equal rows would carry the same spread) |
| `curve(..., max_rug_points=)` | the ticks along the bottom of a conditional effect curve, evenly spaced by rank; `0` draws none | the curve, its interval and band |
| `zipper(..., max_intervals=)` | the intervals drawn per panel, at evenly spaced places in the sorted order, both ends kept | the coverage and the counts of misses |
| `balance(..., page=)` | which covariates are drawn: page 1 holds the `max_covariates` ranked first, covariates above the threshold after weighting first | every difference and the subtitle |

Without these keywords each figure draws what it always has. Without `max_rug_points`, a panel with
more than 20,000 rows draws every k-th row's tick in row order, and the notes say so. Without
`page`, `balance` draws every covariate above the threshold after weighting even past
`max_covariates`; with `page`, each page holds exactly `max_covariates` covariates, so every
covariate can be reached.

## Fits over time and with external data

`overlap`, `learners` and `calibration` also draw fits whose treatment is set at more than one
time point, or whose rows come from a trial and an external source.

- A survival fit of a treatment given once at baseline draws, in `overlap`, each person's
  predicted chance of each arm. The fit's probability limits apply to the product of that chance
  and the chance of staying in follow-up, so `overlap(result, censoring=True)` adds that product
  for each plan, with the limit drawn. A collaborative fit draws each arm's selected treatment
  model, and `learners` draws each arm's path of added terms with the chosen one marked.
- A fit with a treatment at several time points draws one panel per time point. `overlap` shows the
  chance of the plan's treatment among the units still following the plan, split by whether they
  then took it. It draws one block of panels per plan, up to three, or the plan that `regimen=`
  names. `censoring=True` adds each time point's chance of staying in the study. An incremental or
  modified treatment policy has no single chance to draw, so `overlap` refuses it and `weights`
  draws its weights.
- On the same fit, `learners` stacks each model's Super Learner weights by time point, and
  `calibration` draws each probability model at each time point. Both draw the outcome models of one
  plan, the first unless `regimen=` names one, and the models of one cause, the first unless
  `cause=` names one. `nodes=` and `horizon=` choose further.
- A structural nested mean fit draws, in `overlap`, each time point's chance of each treatment
  level, with the margin from 0 and 1 that its positivity summary counts.
- An adaptive fit with external data draws, in `overlap`, each trial patient's chance of
  treatment and each row's chance of being a trial row, with the floor its positivity warning
  checks. Its nuisance report keeps learner names only, so `learners` and `calibration` refuse.
- A continuous-time survival fit draws, in `overlap`, each person's predicted chance of the
  treated arm. Its floor bounds the product of the chance of the arm and the chance of staying in
  follow-up just before a target time, so `overlap(result, censoring=True)` adds that product for
  each arm, with the floor drawn. `weights` draws, at each target time, the weight that inverts
  that product for each person under the arm they received, or under the other arm with
  `counterfactual=True`. `learners` draws each hazard's and the treatment model's candidates with
  the chosen one marked; a cross-fitted fit keeps the first fold's choices.
- On the same fit, `curve` draws each arm's risk, or its cumulative incidence of each cause in one
  column per cause, and their difference below. Each estimate is a point at a time in
  `horizons=`, with its interval and, when the fit holds one, its simultaneous band. Nothing is
  drawn between those times, because the fit estimates no risk there.
  `curve(result, measure="rmst")` or `"rmtl"` draws the restricted means at the times in
  `restricted_horizons=`. `result.plot()` calls `curve`.

- `balance` draws a longitudinal, time-to-event or continuous-time survival fit's covariate
  balance over time, which `covariate_balance` computes. At each treatment time it compares the
  covariates recorded so far across the treatment, among units with the same earlier treatments,
  before and after weighting. The default view has one panel per treatment time, each marker a
  covariate's difference averaged over those groups. `view="panels"` draws each group's difference
  as a dot, and `diagnostic=` picks the check: `"confounding"`, `"feedback"` (later covariates
  across an earlier treatment), `"censoring"`, or a continuous-time fit's `"joint"`. A survival
  fit has no feedback check, because its later covariates exist only for units still free of the
  event, and a continuous-time fit has no groups of earlier treatments, so it draws the default
  view only.

A continuous-time fit's truncation curve carries the point-treatment columns, so `sensitivity`
draws it.
