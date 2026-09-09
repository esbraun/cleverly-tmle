FROM python:3.10-slim-bookworm@sha256:673f009e3763f8d03953b525c89b03a9ee8ca315bf6b8006979b5c4a4e2e4d68

ARG ZEPID_COMMIT=16a0f96f8b2c65df8715085801f21757d1478e1e

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir \
        pip==24.0 setuptools==69.0.3 wheel==0.42.0 \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import zepid; assert zepid.__version__ == '0.9.1'"

ENV ZEPID_COMMIT=${ZEPID_COMMIT}
ENV MPLBACKEND=Agg

WORKDIR /work
ENTRYPOINT ["python"]
