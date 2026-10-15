# Placeholder runner of `longitudinal-repeated-sdr` (plan X32b, PR B).
#
# The study declares its lmtp 1.5.4 comparator as pending, so the regeneration driver runs no
# reference process.  The manifest records the bytes of this file as the declared source.
# PR C replaces this file with the per-horizon runner (plan K12): one `lmtp_sdr_with_folds`
# call per plan and horizon, on the panel truncated after that horizon.
stop(
  "longitudinal-repeated-sdr: the lmtp runner is not part of PR B. ",
  "PR C supplies the per-horizon lmtp runner (plan K12)."
)
