FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG CTMLE_COMMIT=18de559f47dc1286617350a0668391e80e1dbf7c

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglpk40 libcurl4-openssl-dev libssl-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/* \
    && install2.r --error --skipinstalled remotes \
    && Rscript -e "remotes::install_github('jucheng1992/ctmle@${CTMLE_COMMIT}', upgrade='never')" \
    && Rscript -e "stopifnot(as.character(packageVersion('ctmle')) == '0.1.2')"

ENV CTMLE_COMMIT=${CTMLE_COMMIT}
COPY run_ctmle.R /opt/run_ctmle.R
WORKDIR /work
ENTRYPOINT ["Rscript", "/opt/run_ctmle.R"]
