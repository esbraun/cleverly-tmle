FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

RUN install2.r --error --skipinstalled \
      nnls cli R6 generics future progressr data.table checkmate SuperLearner origami Rcpp \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/Archive/simul/simul_0.1.1.tar.gz', '/tmp/simul.tar.gz', mode = 'wb')" \
    && echo "4150d4ef250dfe32e96dcd4de4869fdc44e4d41892c175de5868a4c525be88aa  /tmp/simul.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/simul.tar.gz \
    && Rscript -e "download.file('https://github.com/mtpverse/imtp/archive/d4b5204f9505147a54ac415180e84b86f005c8b2.tar.gz', '/tmp/imtp.tar.gz', mode = 'wb')" \
    && echo "1d76583b27095208ebb3daf87ccbabee12e0aa0c6ea8c332e476f3364ba69660  /tmp/imtp.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/imtp.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('imtp')) == '0.1.0')" \
    && rm /tmp/simul.tar.gz /tmp/imtp.tar.gz

WORKDIR /work
ENTRYPOINT ["Rscript"]
