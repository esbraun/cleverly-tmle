FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

RUN install2.r --error --skipinstalled SuperLearner matrixStats \
    && Rscript -e "install.packages('https://cloud.r-project.org/src/contrib/ltmle_1.3-0.tar.gz', repos = NULL, type = 'source'); stopifnot(as.character(packageVersion('ltmle')) == '1.3.0')"

WORKDIR /work
ENTRYPOINT ["Rscript"]
