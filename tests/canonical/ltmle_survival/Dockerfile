FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

RUN install2.r --error --skipinstalled SuperLearner matrixStats \
    && Rscript -e "download.file('https://cloud.r-project.org/src/contrib/ltmle_1.3-0.tar.gz', '/tmp/ltmle.tar.gz', mode = 'wb')" \
    && echo "fb31d0dd6ab81687b81f3279b414c21e91c655e10aac12f73fc6723efd848aad  /tmp/ltmle.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/ltmle.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('ltmle')) == '1.3.0')" \
    && rm /tmp/ltmle.tar.gz

WORKDIR /work
ENTRYPOINT ["Rscript"]
