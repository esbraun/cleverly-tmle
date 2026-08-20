FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG TMLE3_COMMIT=ed72f8a20e64c914ab25ffe015d865f7a9963d27
ARG SL3_COMMIT=0e8f2365bcbe54010b8120c04a7a2dcfc8119227

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglpk40 \
    && rm -rf /var/lib/apt/lists/* \
    && install2.r --error --skipinstalled remotes \
    && Rscript -e "remotes::install_github('tlverse/sl3@${SL3_COMMIT}', upgrade = 'never')" \
    && Rscript -e "remotes::install_github('tlverse/tmle3@${TMLE3_COMMIT}', upgrade = 'never')" \
    && Rscript -e "stopifnot(as.character(packageVersion('tmle3')) == '0.2.0')"

ENV TMLE3_COMMIT=${TMLE3_COMMIT}
ENV SL3_COMMIT=${SL3_COMMIT}

COPY run_tmle3_cvtmle.R /opt/run_tmle3_cvtmle.R
WORKDIR /work
ENTRYPOINT ["Rscript", "/opt/run_tmle3_cvtmle.R"]
