FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

RUN install2.r --error --skipinstalled \
      nnls cli R6 generics future progressr data.table checkmate SuperLearner isotone lifecycle \
      collapse S7 \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/Archive/ife/ife_0.2.3.tar.gz', '/tmp/ife.tar.gz', mode = 'wb')" \
    && echo "b6be1e9ba514db95118e425d2f78deabb2c9f745f44f35a301ff9b5f266d5ed2  /tmp/ife.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/ife.tar.gz \
    && Rscript -e "download.file('https://github.com/nt-williams/lmtp/archive/f04a2b47f46debc515ce4ae778e05ebfde922c44.tar.gz', '/tmp/lmtp.tar.gz', mode = 'wb')" \
    && echo "fd49d9f291d4ddabb78c36d152b25aaa234a7204b645b9921f998c152e3d2ba5  /tmp/lmtp.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/lmtp.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('ife')) == '0.2.3', as.character(packageVersion('lmtp')) == '1.5.4')" \
    && rm /tmp/ife.tar.gz /tmp/lmtp.tar.gz

WORKDIR /work
ENTRYPOINT ["Rscript"]
