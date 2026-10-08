FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

RUN install2.r --error --skipinstalled \
      Matrix speedglm SuperLearner plyr dplyr tidyr stringr ggplot2 ggsci \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/Archive/survtmle/survtmle_1.1.1.tar.gz', '/tmp/survtmle.tar.gz', mode = 'wb')" \
    && echo "d286799265f35f2ae7631f2f1714936ff2b1866cdd95a2848c252482f9ce23d0  /tmp/survtmle.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/survtmle.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('survtmle')) == '1.1.1')" \
    && rm /tmp/survtmle.tar.gz

WORKDIR /work
ENTRYPOINT ["Rscript"]
