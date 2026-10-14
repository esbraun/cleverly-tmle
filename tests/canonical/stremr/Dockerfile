# R `stremr` at osofr/stremr e075f719 (2026-02-17), with its GitHub dependencies pinned to the
# last commit at or before that date.  Each source is cloned and checked out at its full commit
# hash, which pins its content.  CRAN packages come from the snapshot of the base image.
FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libxml2-dev libcurl4-openssl-dev libssl-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# The h2o binary is large, and R's default 60 s download timeout cuts it off.
RUN echo 'options(timeout = 3600)' >> "${R_HOME}/etc/Rprofile.site"

RUN install2.r --error --skipinstalled \
        assertthat data.table dplyr magrittr R6 Rcpp speedglm stringr purrr tibble tidyr zoo \
        pander rmarkdown SuperLearner xgboost h2o ggplot2 uuid BBmisc delayed digest Rdpack \
        caret ROCR origami

COPY install_pinned.R /tmp/install_pinned.R
RUN Rscript /tmp/install_pinned.R && rm /tmp/install_pinned.R

WORKDIR /work
ENTRYPOINT ["Rscript"]
