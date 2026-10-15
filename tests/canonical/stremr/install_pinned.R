# Install stremr and its GitHub dependencies from pinned commits, in dependency order.
pinned <- list(
  c("osofr/condensier", "6bcc973291cb311f586d3035a43a2468eae541a9"),
  c("osofr/gridisl", "617b1b7f156d99621c05bbd145a4ae2c9a592b9e"),
  c("jeremyrcoyle/sl3", "0e8f2365bcbe54010b8120c04a7a2dcfc8119227"),
  c("osofr/stremr", "e075f719d5978c343555d3615e409b30d381e2b9")
)
for (entry in pinned) {
  repo <- entry[[1]]
  commit <- entry[[2]]
  target <- file.path(tempdir(), basename(repo))
  stopifnot(system2("git", c("clone", "--quiet", paste0("https://github.com/", repo, ".git"), target)) == 0)
  stopifnot(system2("git", c("-C", target, "checkout", "--quiet", commit)) == 0)
  head <- system2("git", c("-C", target, "rev-parse", "HEAD"), stdout = TRUE)
  stopifnot(identical(head, commit))
  stopifnot(system2("R", c("CMD", "INSTALL", "--no-test-load", target)) == 0)
}
library(stremr)
writeLines(paste("stremr", as.character(packageVersion("stremr"))))
