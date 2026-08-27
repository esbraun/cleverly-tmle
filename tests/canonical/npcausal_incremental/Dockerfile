FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

RUN install2.r --error --skipinstalled \
      nnls SuperLearner earth gam ranger rpart \
    && Rscript -e "download.file('https://github.com/ehkennedy/npcausal/archive/56a5ac117a29258b67b94874be662a171b5131f7.tar.gz', '/tmp/npcausal.tar.gz', mode = 'wb')" \
    && echo "9b0f72d057224365953a65e2247b996c01ff1e02ebbbeeb13a244c7ea49f1795  /tmp/npcausal.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/npcausal.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('npcausal')) == '0.1.0')" \
    && rm /tmp/npcausal.tar.gz

WORKDIR /work
ENTRYPOINT ["Rscript"]
