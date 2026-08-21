FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG CTMLE3_COMMIT=a4ea77b07747dfee9b2eecb9cbca88262e0559ea
ARG TMLE3_COMMIT=3a61005
ARG SL3_COMMIT=821ca89

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglpk40 libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && install2.r --error --skipinstalled remotes \
    && Rscript -e "remotes::install_github('tlverse/sl3@${SL3_COMMIT}', upgrade='never')" \
    && Rscript -e "remotes::install_github('tlverse/tmle3@${TMLE3_COMMIT}', upgrade='never')" \
    && Rscript -e "remotes::install_github('tlverse/ctmle3@${CTMLE3_COMMIT}', upgrade='never')" \
    && Rscript -e "stopifnot(as.character(packageVersion('ctmle3')) == '0.1.0')"

ENV CTMLE3_COMMIT=${CTMLE3_COMMIT}
ENV TMLE3_COMMIT=${TMLE3_COMMIT}
ENV SL3_COMMIT=${SL3_COMMIT}
COPY run_ctmle3_oat.R /opt/run_ctmle3_oat.R
WORKDIR /work
ENTRYPOINT ["Rscript", "/opt/run_ctmle3_oat.R"]
