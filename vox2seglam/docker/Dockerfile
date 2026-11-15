# Vox2SegLAM inference — Blackwell-capable (PyTorch cu128).
# Weights are NOT baked in; vox2seglam downloads into VOX2SEGLAM_ASSETS
# (/models) on first use — mount a host cache volume.
#
#   docker build -t websub/vox2seglam:cu128 \
#     -f vox2seglam/docker/Dockerfile vox2seglam/docker
#
# Needs Python >=3.11 (Vox2SegLAM requires it); Ubuntu 24.04 provides 3.12.

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    VOX2SEGLAM_ASSETS=/models \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ca-certificates \
        curl \
        git \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m venv /opt/venv

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 torch \
    && pip install "git+https://github.com/Neuro-iX/Vox2SegLAM.git"

WORKDIR /app
RUN mkdir -p /models /data/input /data/output

ENTRYPOINT ["vox2seglam"]
