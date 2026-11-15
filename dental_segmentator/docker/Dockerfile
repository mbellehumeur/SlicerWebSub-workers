# Dental Segmentator inference — Blackwell-capable (PyTorch cu128).
# Weights are NOT baked in; the entrypoint downloads Dataset111 into /models
# on first run (mount a host cache volume).
#
#   docker build -t websub/dental_segmentator:cu128 \
#     -f dental_segmentator/docker/Dockerfile dental_segmentator/docker

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    DENTAL_SEG_MODELS=/models \
    DENTAL_SEG_DEVICE=cuda \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        curl \
        unzip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m venv /opt/venv

# cu128 wheels for RTX 50 / sm_120 (install into venv, not system site-packages)
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 torch \
    && pip install nnunetv2 SimpleITK nibabel

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

RUN mkdir -p /models /data/input /data/output

ENTRYPOINT ["python", "/app/entrypoint.py"]
