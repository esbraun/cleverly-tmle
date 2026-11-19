# R concrete 1.0.5, the comparator of the continuous-time-survival concrete phase (roadmap X6
# part f).  CRAN archived the package on 2024-04-02, and no commit of imbroglio-dc/concrete
# matches the CRAN tarball byte for byte (nearest: 94a86a4 and ef91f89), so the pin is the
# archive tarball itself, checked against its sha256 before R CMD INSTALL.  concrete compiles
# against RcppArmadillo and links R's LAPACK, BLAS and Fortran libraries, hence gfortran.
# SuperLearner, glmnet and the other packages come from the CRAN snapshot of the base image.
FROM rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c

ARG CONCRETE_VERSION=1.0.5
ARG CONCRETE_SHA256=4b7a67357bb0852d70322180f7be9e79a11c99321f73c01f34e720d8661c1416

RUN apt-get update \
    && apt-get install -y --no-install-recommends gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && install2.r --error --skipinstalled \
      data.table zoo origami SuperLearner nnls glmnet nleqslv Rcpp RcppArmadillo \
    && Rscript -e "download.file('https://cran.r-project.org/src/contrib/Archive/concrete/concrete_${CONCRETE_VERSION}.tar.gz', '/tmp/concrete.tar.gz', mode = 'wb')" \
    && echo "${CONCRETE_SHA256}  /tmp/concrete.tar.gz" | sha256sum -c - \
    && R CMD INSTALL /tmp/concrete.tar.gz \
    && Rscript -e "stopifnot(as.character(packageVersion('concrete')) == '${CONCRETE_VERSION}')" \
    && rm /tmp/concrete.tar.gz

ENV CONCRETE_VERSION=${CONCRETE_VERSION}
ENV CONCRETE_SHA256=${CONCRETE_SHA256}

WORKDIR /work
ENTRYPOINT ["Rscript"]
