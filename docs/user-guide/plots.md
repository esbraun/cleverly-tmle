# Which plot answers which question

Each function in `cleverly.plotting` answers one question about a result you already have, such as
the `result` that `effect.estimate(...)` returns in the [quickstart](../getting-started/quickstart.md).
It draws numbers the result holds and returns a matplotlib figure. It never draws an interval, or a
band (an interval that covers a whole curve at once), that the fit did not compute. Install
matplotlib with `pip install "cleverly[plot]"`.

```python
from cleverly.plotting import forest, notes

fig = forest(result)  # each estimate with its confidence interval
print(notes(fig))  # what a reader of this figure needs to know
fig.savefig("effect.png", bbox_inches="tight")
```

## Pick the plot by question

The last column names the method that draws the same figure from the object itself.

| question | function | what you pass | shortcut |
| --- | --- | --- | --- |
| How large is each estimate, and how sure is it? | `forest` | a fitted result, or several estimates | `result.plot()` on a point-treatment result |
| How does a result change over time, dose, policy or a modifier? | `curve` | a result over time, a dose-response curve, a conditional effect curve, a fit with exact event times, or a fit over a family of policies or treatment levels | `result.plot()` on a result over time, a fit with exact event times or a conditional effect curve |
| Did every kind of unit have a chance of each treatment? | `overlap` | a fitted result with a treatment model | none |
| Did weighting make treated and untreated units alike on each covariate? | `balance` | a balance table (`covariate_balance`), or a fitted result to compute it from | `.plot()` on the balance table |
| At which time point do the weights grow too large? | `weights` | a result over time or a fit with exact event times | none |
| Do predicted probabilities or effects match what was measured? | `calibration` | a fitted result, its model report (`NuisanceModelReport`), or a conditional effect curve's calibration check (`CurveCalibration`) | `.plot()` on the model report or the calibration check |
| Which learners carry the fit, and how sharp was the choice? | `learners` | a fitted result, or the record of an ensemble's or a collaborative fit's choice (`SuperLearnerDiagnostics`, `CTMLESelection`) | `.plot()` on those records |
| Does the estimate depend on how the rows were split to fit and check the models? | `repeats` | a fit that repeated the random splits at least twice (`repeats=`) | none |
| Is the bootstrap distribution close to normal? | `bootstrap` | a bootstrap result (`BootstrapResult`), or a fit that holds one | `.plot()` on the bootstrap result |
| How few rows or clusters carry the uncertainty? | `influence` | an estimate, or a fit and the name of one estimate | none |
| Does the estimate depend on a sensitivity setting, such as a weight cap? | `sensitivity` | a truncation curve, sensitivity bounds or a missingness tilt (`TruncationCurve`, `MarginalSensitivityBounds`, `MissingnessTilt`) | `.plot()` on each |
| How strong would hidden confounding need to be to explain the effect away? | `confounding` | a sensitivity contour, a simulated-confounding result or an E-value | `.plot()` on each |
| How much does treating by a score first gain over treating at random? | `targeting` | a curve that ranks units by a score (`TargetingCurve`) | `.plot()` on that curve |
| Do a simulation study's intervals cover the true value, and how do they miss? | `zipper` | a finished coverage study, which refits on simulated data with a known truth (`StudyResult`) | `study.plot()` |

The [plotting reference](../api/plotting.md) lists which result types each shortcut belongs to, and
how the plots draw fits over time, fits with outside data and fits with exact event times. The
[plot results](../examples/plots.ipynb) tutorial draws the main figures on simulated data.

## Read the notes before you publish

The figure shows no caveats; `notes(fig)` returns them. They say what a reader of that fit needs,
such as why an interval is missing or which numbers only describe the data. They also say when the
estimate or its band is *experimental*: no simulation study covers a fit like it, or a condition
such as too few clusters applies, so its interval has no checked coverage. The figure itself does
not mark this, so read `notes(fig)` or the result's `summary()` before you share a figure, as
[read the support label](reading-output.md#read-the-support-label) explains.

## Large data

When there is a lot to draw, some figures thin the marks so the figure stays readable and fast.
`max_points` on `influence`, `max_rug_points` on `curve` and `max_intervals` on `zipper` set how
many marks are drawn. `page` on `balance` picks which page of covariates to draw when there are
more than `max_covariates`. Each keeps the numbers the figure reports exact and only draws fewer
marks, as [large data](../api/plotting.md#large-data) in the plotting reference explains.

## Titles, labels and style

Every plot takes the same formatting keywords: `title=`, `subtitle=`, `xlabel=`, `ylabel=`,
`legend=` and `figsize=`, and one-panel plots also take `ax=` to draw on axes of your own. On
simulated data, `truth=` on `forest` or `curve` marks the known true values. The
[plotting reference](../api/plotting.md) describes each.

To draw a figure of your own in the same style, draw it inside `with cleverly.plotting.style():`.
The three series colours are `cleverly.plotting.PALETTE`, a tuple of hex codes, so you can match
them in other tools.
