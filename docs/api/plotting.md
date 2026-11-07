# Plotting

Use these functions after estimation to look at a result. Each draws numbers the object
already holds and returns the matplotlib `Figure`, so you can save it, restyle it, or add to it.
The title of each figure states the question it answers, and the note under it says what an
interval is, why one is missing, or that a quantity is descriptive. A plot never draws an
interval or band the fit did not compute. On simulated data, pass the known true values as
`truth=` to `forest` or `curve`. Each is drawn as a cross beside its estimate, or as a line
along a conditional effect curve, and the key names it "True value (simulated data only)".

matplotlib is optional. Install it with `pip install "cleverly[plot]"`.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.plotting.forest
   cleverly.plotting.curve
   cleverly.plotting.sensitivity
   cleverly.plotting.confounding
   cleverly.plotting.overlap
   cleverly.plotting.weights
   cleverly.plotting.balance
   cleverly.plotting.zipper
   cleverly.plotting.calibration
   cleverly.plotting.learners
   cleverly.plotting.repeats
   cleverly.plotting.bootstrap
   cleverly.plotting.influence
   cleverly.plotting.style
```

The results that have one natural figure also draw it with `plot()`:
`TMLEResult.plot()` and `TMLEResultSet.plot()` call `forest`, and `TruncationCurve.plot()`,
`MarginalSensitivityBounds.plot()` and `MissingnessTilt.plot()` call `sensitivity`, and
`SensitivityContour.plot()` and `SimulatedConfoundingResult.plot()` call `confounding`.
`CovariateBalance.plot()` calls `balance`, and `StudyResult.plot()` calls `zipper`.
`CurveCalibration.plot()` and `NuisanceModelReport.plot()` call `calibration`,
`SuperLearnerDiagnostics.plot()` and `CTMLESelection.plot()` call `learners`, and
`BootstrapResult.plot()` calls `bootstrap`.

`LongitudinalResult.plot()` draws its curve over time with `curve`, or the estimates with
`forest` when the fit has no curve, and `EffectCurveResult.plot()` draws the conditional effect
over a modifier with `curve`.
