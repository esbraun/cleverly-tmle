"""Canonical labels for study constructions compared across evidence records."""

POOLED_LONGITUDINAL_CROSS_FIT = "pooled_cross_fit"
TRAINING_FOLD_FLUCTUATION = "training_fold_fluctuation"
#: The shipped SDR cross-fit: each fold's recursion reads its own nuisances at its training rows.
SDR_TRAINING_FOLD_RECURSION = "sdr_training_fold_recursion"
#: The printed Step 2 of Díaz et al. (2023, Section 5.3), a diagnostic cell only.
SDR_STITCHED_CROSS_FIT = "sdr_stitched_cross_fit"
