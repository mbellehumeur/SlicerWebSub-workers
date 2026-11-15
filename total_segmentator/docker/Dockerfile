# TotalSegmentator inference — Blackwell-capable (PyTorch cu128).
# Weights are NOT baked in; TotalSegmentator downloads into TOTALSEG_HOME_DIR
# (/models) on first use — mount a host cache volume.
#
#   docker build -t websub/totalsegmentator:cu128 \
#     -f total_segmentator/docker/Dockerfile total_segmentator/docker

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOTALSEG_HOME_DIR=/models \
    PATH=/opt/venv/bin:$PATH

# build-essential + python3-dev: TotalSegmentator pulls dipy, which may compile
# from source when no matching wheel exists for this platform/Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        pkg-config \
        ca-certificates \
        curl \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m venv /opt/venv

# torch + torchvision must come from the same cu128 index. TotalSegmentator
# (via timm/nnunet) otherwise pulls a mismatched PyPI torchvision and fails with
# "operator torchvision::nms does not exist".
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision \
    && pip install TotalSegmentator highdicom \
    && pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall torchvision

WORKDIR /app
RUN mkdir -p /models /data/input /data/output

ENTRYPOINT ["TotalSegmentator"]
