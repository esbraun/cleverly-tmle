# Exceptions and warnings

Catch `CleverlyError` subclasses when an application can recover from invalid data, configuration,
or lifecycle use. Treat warnings as requests to inspect convergence, overlap, or weighting.

```{eval-rst}
.. autosummary::
   :nosignatures:

   cleverly.CleverlyError
   cleverly.CapabilityError
   cleverly.DataError
   cleverly.MethodConfigurationError
   cleverly.NotFittedError
   cleverly.ConvergenceWarning
   cleverly.PositivityWarning
   cleverly.WeightingWarning
```

## Package version

The installed version is available as `cleverly.__version__`.
