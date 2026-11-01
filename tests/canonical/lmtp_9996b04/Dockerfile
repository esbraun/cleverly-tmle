# lmtp at the development commit 9996b04 (packageVersion 1.5.5), for the X59 long-horizon
# comparators (X59-plan.md section 2.2).
#
# The image is built FROM the shipped lmtp 1.5.4 image (tests/canonical/lmtp_crossfit), so every
# package the two images share is the same build: G0's cross-image identity compares lmtp alone,
# and G0 fails on any other drift (x59_smoke.py, package_drift).  On top it adds what 9996b04
# needs and 1.5.4 does not: its NAMESPACE imports origami::make_folds, which its DESCRIPTION does
# not list, so origami is installed from the CRAN archive first (X39-plan-review-rev2.md, minor
# 4); collapse comes from the shipped image.  The driver builds the shipped image first
# (Reference.base in tests/canonical/regenerate.py).
#
# TODO-at-first-build: every value marked TODO below is a placeholder, not a hash.  The first
# build fails at its sha256sum -c line by design.  Fetch each archive once, record its sha256
# here, put the full 40-character lmtp commit in the archive URL, and commit that change before
# any run.  The same commit refreshes the pending manifests' declared digests with
# `python -m tests.canonical.x59_smoke declared-sources ltmle categorical sdr`, because this file
# is a declared source of the two lmtp studies.  Do not run with a placeholder in place.  The
# origami version is a working choice: confirm it against the 9996b04 NAMESPACE imports at the
# same first build.
FROM cleverly-lmtp-crossfit:1.5.4

RUN install2.r --error --skipinstalled Rcpp future.apply assertthat abind \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/Archive/origami/origami_1.0.7.tar.gz', '/tmp/origami.tar.gz', mode = 'wb')" \
    && echo "b44034541ac358e0686682684c40e9a1de8d78c7913e56e4d3dbe41a2a55c62c  /tmp/origami.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/origami.tar.gz \
    && Rscript -e "download.file('https://github.com/nt-williams/lmtp/archive/9996b04dcbb3ae0b1ef8862097c36d95e9f2fcf9.tar.gz', '/tmp/lmtp.tar.gz', mode = 'wb')" \
    && echo "b3c5896272cf6c07127764c1880758fdb5020506a271b453e907a7b69bebb2ab  /tmp/lmtp.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/lmtp.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('ife')) == '0.2.3', as.character(packageVersion('lmtp')) == '1.5.5', as.character(packageVersion('origami')) == '1.0.7')" \
    && mkdir -p /opt/cleverly \
    && Rscript -e "p <- installed.packages()[, c('Package', 'Version')]; write.table(p[order(p[, 'Package']), ], '/opt/cleverly/r-packages.tsv', sep = '\t', quote = FALSE, row.names = FALSE)" \
    && rm /tmp/origami.tar.gz /tmp/lmtp.tar.gz

WORKDIR /work
ENTRYPOINT ["Rscript"]
