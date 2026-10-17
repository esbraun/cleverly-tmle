FROM python:3.10-slim-bookworm@sha256:673f009e3763f8d03953b525c89b03a9ee8ca315bf6b8006979b5c4a4e2e4d68

ARG PYGFORMULA=1.2.0

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir \
        pip==24.0 setuptools==69.0.3 wheel==0.42.0 \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import importlib.metadata as m; assert m.version('pygformula') == '1.2.0'"

ENV PYGFORMULA=${PYGFORMULA}
ENV MPLBACKEND=Agg

WORKDIR /work
ENTRYPOINT ["python"]
