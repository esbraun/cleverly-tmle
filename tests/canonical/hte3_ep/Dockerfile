# The hte3 comparison of the EP curve (tests/studies/ep_hte3.py).  The base digest, SL3_COMMIT
# and TMLE3_COMMIT are tests/canonical/tmle3/Dockerfile's.  CRAN packages come from the Posit
# snapshot the base image names (CRAN=https://p3m.dev/cran/__linux__/noble/2026-03-10), and each
# package hte3 calls is checked against that snapshot's version.
FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG TMLE3_COMMIT=ed72f8a20e64c914ab25ffe015d865f7a9963d27
ARG SL3_COMMIT=0e8f2365bcbe54010b8120c04a7a2dcfc8119227
ARG HTE3_COMMIT=eac64673f6fdf75f130275646657f08d1cb198e9

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglpk40 \
    && rm -rf /var/lib/apt/lists/* \
    && install2.r --error --skipinstalled remotes Sieve glmnet data.table uuid R6 combinat \
    && Rscript -e "remotes::install_github('tlverse/sl3@${SL3_COMMIT}', upgrade = 'never')" \
    && Rscript -e "remotes::install_github('tlverse/tmle3@${TMLE3_COMMIT}', upgrade = 'never')" \
    && Rscript -e "remotes::install_github('Larsvanderlaan/hte3@${HTE3_COMMIT}', upgrade = 'never', dependencies = FALSE)" \
    && Rscript -e "stopifnot(as.character(packageVersion('tmle3')) == '0.2.0', as.character(packageVersion('hte3')) == '0.1.1', as.character(packageVersion('Sieve')) == '2.1', as.character(packageVersion('glmnet')) == '4.1.10', as.character(packageVersion('data.table')) == '1.18.2.1', as.character(packageVersion('uuid')) == '1.2.2', as.character(packageVersion('R6')) == '2.6.1', as.character(packageVersion('combinat')) == '0.0.8')"

ENV TMLE3_COMMIT=${TMLE3_COMMIT}
ENV SL3_COMMIT=${SL3_COMMIT}
ENV HTE3_COMMIT=${HTE3_COMMIT}

WORKDIR /work
ENTRYPOINT ["Rscript"]
