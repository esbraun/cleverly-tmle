FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG DRTMLE_COMMIT=538a3a264c1ca984b6d88978ca7f96165f43152c

RUN install2.r --error --skipinstalled remotes \
    && Rscript -e "remotes::install_github('benkeser/drtmle@${DRTMLE_COMMIT}', upgrade = 'never', dependencies = NA)" \
    && Rscript -e "stopifnot(as.character(packageVersion('drtmle')) == '1.1.2')"

ENV DRTMLE_COMMIT=${DRTMLE_COMMIT}
COPY run_drtmle.R /opt/run_drtmle.R
WORKDIR /work
ENTRYPOINT ["Rscript", "/opt/run_drtmle.R"]
