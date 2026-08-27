FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG TMLE_VERSION=2.1.1
ARG TMLE_SHA256=5e1fccaea7bf923456b8197d3eca5314db074dcbec8ca0510a15cb837883b133

RUN install2.r --error --skipinstalled glmnet SuperLearner \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/tmle_${TMLE_VERSION}.tar.gz', '/tmp/tmle.tar.gz', mode = 'wb')" \
    && echo "${TMLE_SHA256}  /tmp/tmle.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/tmle.tar.gz \
    && rm /tmp/tmle.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('tmle')) == '${TMLE_VERSION}')"

ENV TMLE_VERSION=${TMLE_VERSION}
ENV TMLE_SHA256=${TMLE_SHA256}

WORKDIR /work
ENTRYPOINT ["Rscript"]
