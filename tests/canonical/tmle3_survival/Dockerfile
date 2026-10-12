FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG TMLE3_COMMIT=ed72f8a20e64c914ab25ffe015d865f7a9963d27
ARG SL3_COMMIT=0e8f2365bcbe54010b8120c04a7a2dcfc8119227
# tmle3's survival updater calls glmnet::glmnet, and neither tmle3 nor sl3 imports it.  The
# sha256 is the one conda-forge's r-glmnet recipe records for the CRAN tarball of 5.0.
ARG GLMNET_VERSION=5.0
ARG GLMNET_SHA256=f0b23da6383a45d502cd138b64da6300422d9b8d7b2c4f14627d412cde79d2a0

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglpk40 gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && install2.r --error --skipinstalled remotes foreach shape Rcpp RcppEigen \
    && Rscript -e "u <- sprintf('https://cran.r-project.org/src/contrib/%s/glmnet_%s.tar.gz', c('.', 'Archive/glmnet'), '${GLMNET_VERSION}'); ok <- FALSE; for (x in u) if (!ok) ok <- tryCatch({download.file(x, '/tmp/glmnet.tar.gz', mode = 'wb'); TRUE}, error = function(e) FALSE); stopifnot(ok)" \
    && echo "${GLMNET_SHA256}  /tmp/glmnet.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/glmnet.tar.gz \
    && rm /tmp/glmnet.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('glmnet')) == '${GLMNET_VERSION}')" \
    && Rscript -e "remotes::install_github('tlverse/sl3@${SL3_COMMIT}', upgrade = 'never')" \
    && Rscript -e "remotes::install_github('tlverse/tmle3@${TMLE3_COMMIT}', upgrade = 'never')" \
    && Rscript -e "stopifnot(as.character(packageVersion('tmle3')) == '0.2.0')"

ENV TMLE3_COMMIT=${TMLE3_COMMIT}
ENV SL3_COMMIT=${SL3_COMMIT}
ENV GLMNET_VERSION=${GLMNET_VERSION}

WORKDIR /work
ENTRYPOINT ["Rscript"]
