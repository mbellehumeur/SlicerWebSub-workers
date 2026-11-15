# Shared imaging conversion tools (CPU). Used as a sidecar by Hub workers
# for DICOM ↔ NIfTI and NIfTI labelmap → DICOM SEG.
#
#   docker build -t websub/tools:cpu \
#     -f tools/docker/Dockerfile tools/docker

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        curl \
        dcm2niix \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m venv /opt/venv

RUN pip install --upgrade pip \
    && pip install \
        "numpy>=1.24" \
        "nibabel>=5.0" \
        "pydicom>=2.4" \
        "highdicom>=0.22" \
        "SimpleITK>=2.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

RUN mkdir -p /data/input /data/output

ENTRYPOINT ["python", "/app/entrypoint.py"]
