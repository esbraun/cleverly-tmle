# lmtp at the development commit 9996b04 (packageVersion 1.5.5), for the X59 long-horizon
# comparators (X59-plan.md section 2.2).  At 9996b04 lmtp's NAMESPACE imports origami::make_folds
# and collapse::fmean and fsum, which its DESCRIPTION does not list, so both are installed from
# the CRAN archive first (X39-plan-review-rev2.md, minor 4).
#
# TODO-at-first-build: every value marked TODO below is a placeholder, not a hash.  The first
# build fails at its sha256sum -c line by design.  Fetch each archive once, record its sha256
# here, put the full 40-character lmtp commit in the archive URL, and commit that change before
# any run; the manifest then hashes the pinned file.  Do not run with a placeholder in place.
# The origami and collapse versions are a working choice: confirm each against the 9996b04
# NAMESPACE imports at the same first build.
FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

RUN install2.r --error --skipinstalled \
      nnls cli R6 generics future progressr data.table checkmate SuperLearner isotone lifecycle \
      S7 Rcpp future.apply assertthat abind \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/Archive/ife/ife_0.2.3.tar.gz', '/tmp/ife.tar.gz', mode = 'wb')" \
    && echo "b6be1e9ba514db95118e425d2f78deabb2c9f745f44f35a301ff9b5f266d5ed2  /tmp/ife.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/ife.tar.gz \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/Archive/collapse/collapse_2.0.19.tar.gz', '/tmp/collapse.tar.gz', mode = 'wb')" \
    && echo "TODO-at-first-build-collapse-2.0.19-tarball-sha256  /tmp/collapse.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/collapse.tar.gz \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/Archive/origami/origami_1.0.7.tar.gz', '/tmp/origami.tar.gz', mode = 'wb')" \
    && echo "TODO-at-first-build-origami-1.0.7-tarball-sha256  /tmp/origami.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/origami.tar.gz \
    && Rscript -e "download.file('https://github.com/nt-williams/lmtp/archive/9996b04.tar.gz', '/tmp/lmtp.tar.gz', mode = 'wb')" \
    && echo "TODO-at-first-build-lmtp-9996b04-tarball-sha256  /tmp/lmtp.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/lmtp.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('ife')) == '0.2.3', as.character(packageVersion('lmtp')) == '1.5.5', as.character(packageVersion('origami')) == '1.0.7', as.character(packageVersion('collapse')) == '2.0.19')" \
    && mkdir -p /opt/cleverly \
    && Rscript -e "p <- installed.packages()[, c('Package', 'Version')]; write.table(p[order(p[, 'Package']), ], '/opt/cleverly/r-packages.tsv', sep = '\t', quote = FALSE, row.names = FALSE)" \
    && rm /tmp/ife.tar.gz /tmp/collapse.tar.gz /tmp/origami.tar.gz /tmp/lmtp.tar.gz

WORKDIR /work
ENTRYPOINT ["Rscript"]
