FROM python:3.13-slim-bookworm@sha256:a1165e272e578941b84abc79e4ab38a0305cd12803a5c4247979ac7655f4d641

COPY requirements.txt /tmp/requirements.txt
# The optional build secret ``pip_cert`` is a CA bundle for a build host whose egress re-terminates
# TLS.  pip reads it only when the secret is present.  No layer holds it, and every installed
# file is fixed by the hashes of requirements.txt, so the image content does not depend on it.
RUN --mount=type=secret,id=pip_cert \
    if [ -s /run/secrets/pip_cert ]; then export PIP_CERT=/run/secrets/pip_cert; fi \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import econml; assert econml.__version__ == '0.17.0'"

WORKDIR /work
ENTRYPOINT ["python"]
