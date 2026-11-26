# Which plot answers which question

Each function in `cleverly.plotting` answers one question about a result you already have. It
draws numbers the result holds and returns a matplotlib figure. It never draws an interval or band
the fit did not compute. Install matplotlib with `pip install "cleverly[plot]"`.

```python
from cleverly.plotting import forest, notes

fig = forest(result)  # each estimate with its 95% interval
print(notes(fig))  # what a reader of this figure needs to know
fig.savefig("effect.png", bbox_inches="tight")
```

## Pick the plot by question

The questions come from the `cleverly.plotting` module, which owns the list. The last column names
the method that draws the same figure from the result itself.

| question | function | what you pass | shortcut |
| --- | --- | --- | --- |
| How large is each estimate, and how sure is it? | `forest` | a fitted result, or several estimates | `result.plot()` on a point-treatment result |
| How does a result change over time, dose, policy or a modifier? | `curve` | a result over time, a dose-response curve, a conditional effect curve or a fit with exact event times | `result.plot()` on a result over time, a fit with exact event times or a conditional effect curve |
| Did every kind of unit have a chance of each treatment? | `overlap` | a fitted result with a treatment model | none |
| Did weighting make treated and untreated units alike on each covariate? | `balance` | a covariate balance table, or a fitted result to compute it from | `balance.plot()` |
| At which time point do the weights grow too large? | `weights` | a result over time or a fit with exact event times | none |
| Do predicted probabilities or effects match what was measured? | `calibration` | a fitted result or its model report | `.plot()` on the model report |
| Which learners carry the fit, and how sharp was the choice? | `learners` | a fitted result, or a Super Learner's or collaborative fit's record | `.plot()` on those records |
| Does the estimate depend on how the data were split into folds? | `repeats` | a fit with at least two repeats of the folds | none |
| Is the bootstrap distribution close to normal? | `bootstrap` | a bootstrap result, or a fit that holds one | `bootstrap.plot()` |
| How few rows or clusters carry the uncertainty? | `influence` | an estimate, or a fit and the name of one estimate | none |
| Does the estimate depend on a sensitivity setting, such as a weight cap? | `sensitivity` | a truncation curve, sensitivity bounds or a missingness tilt | `.plot()` on each |
| How strong would hidden confounding need to be to explain the effect away? | `confounding` | a sensitivity contour, a simulated-confounding result or an E-value | `.plot()` on each |
| How much does treating by a score first gain over treating at random? | `targeting` | a targeting curve | `.plot()` on the targeting curve |
| Do a simulation study's intervals cover the true value, and how do they miss? | `zipper` | a finished coverage study | `study.plot()` |

The [plotting reference](../api/plotting.md) lists which result types each shortcut belongs to, and
how `overlap`, `learners` and `calibration` draw fits over time, fits with outside data and fits
with exact event times. The [plot results](../examples/plots.ipynb) tutorial draws the main figures
on simulated data.

## Read the notes before you publish

A figure carries no notes. `notes(fig)` returns what a reader of that fit needs, such as why an
interval is missing or which numbers only describe the data. It also says when the estimate or its
band is *experimental*: no simulation study covers a fit like it, so its interval has no checked
coverage. The figure itself does not mark this, so read `notes(fig)` or the result's `summary()`
before you share a figure, as [read the support label](results-assessment.md#read-the-support-label)
explains.

## Large data

On a large fit some figures thin what they draw so the figure stays readable and fast. Four
keywords set how much: `max_points` on `influence`, `max_rug_points` on `curve`, `max_intervals`
on `zipper`, and `page` on `balance`, which splits many covariates over several figures. Each keeps
the numbers the figure reports exact and only draws fewer marks, as
[large data](../api/plotting.md#large-data) in the plotting reference explains.

## Titles, labels and style

Every plot takes the same formatting keywords: `title=`, `subtitle=`, `xlabel=`, `ylabel=`,
`legend=`, `figsize=` and `ax=` to draw on axes of your own. On simulated data, `truth=` on `forest`
or `curve` marks the known true values. The [plotting reference](../api/plotting.md) describes each.

To draw a figure of your own in the same style, draw it inside `with cleverly.plotting.style():`.
The three series colours are `cleverly.plotting.PALETTE`, a tuple of hex codes, so you can match
them in other tools.
